"""HNN simulation, cached data, and waveform plotting for both tutorials.

HNN and NEURON are imported only when generating new simulations.
Parameters use milliseconds and microsiemens; dipoles use nAm.
"""

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

TIMES = np.arange(10.0, 76.0)
NOISE_SD = 2e-6
PARAMETER_KEYS = ("mu", "weight_pyr")
PARAMETER_LABELS = ["Input time (ms)", "Pyramidal weight (µS)"]
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hnn-evoked.npz"


def set_drive(params, net):
    """Configure one proximal input on the fresh HNN network copy."""
    weights = {
        "L2_basket": 1e-4,
        "L5_basket": 1e-4,
        "L2_pyramidal": float(params["weight_pyr"]),
        "L5_pyramidal": float(params["weight_pyr"]),
    }
    delays = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    net.add_evoked_drive(
        "evprox",
        mu=float(params["mu"]),
        sigma=3.0,
        numspikes=1,
        location="proximal",
        weights_ampa=weights,
        synaptic_delays=delays,
        event_seed=int(params["event_seed"]),
    )


def summarize_waveforms(results):
    """Reduce HNN results to baseline-corrected, smoothed dipole vectors."""
    rows = []
    for result in results:
        dipole = result["dpl"][0].copy().smooth(window_len=5)
        baseline = dipole.data["agg"][
            (dipole.times >= 5) & (dipole.times < 15)
        ].mean()
        waveform = (
            np.interp(TIMES, dipole.times, dipole.data["agg"]) - baseline
        )
        rows.append(
            {
                "mu": [result["param_values"]["mu"]],
                "weight_pyr": [result["param_values"]["weight_pyr"]],
                "dipole": waveform,
            }
        )
    return rows


def generate_dataset(n_samples, seed=42, n_jobs=4):
    """Simulate independent prior draws and add measurement noise."""
    # Imported only when regenerating; cached offline training needs no HNN.
    from importlib.metadata import version
    from platform import python_version

    from hnn_core import neymotin_2020_model
    from hnn_core.batch_simulate import BatchSimulate

    rng = np.random.default_rng(seed)
    grid = {
        "mu": rng.uniform(25, 45, n_samples),
        "weight_pyr": np.exp(
            rng.uniform(np.log(1e-4), np.log(8e-4), n_samples)
        ),
        "event_seed": rng.choice(10_000_000, n_samples, replace=False),
    }
    batch = BatchSimulate(
        net=neymotin_2020_model(mesh_shape=(3, 3)),
        set_params=set_drive,
        summary_func=summarize_waveforms,
        tstop=80.0,
        dt=0.025,
        n_trials=1,
        batch_size=64,
        clear_cache=True,
    )
    result = batch.run(grid, combinations=False, n_jobs=n_jobs, verbose=False)
    rows = [row for chunk in result["summary_statistics"] for row in chunk]
    data = {
        key: np.asarray([row[key] for row in rows], dtype=np.float32)
        for key in ("mu", "weight_pyr", "dipole")
    }
    data["dipole_clean"] = data["dipole"].copy()
    data["dipole"] += rng.normal(0, NOISE_SD, data["dipole"].shape).astype(
        np.float32
    )
    data["event_seed"] = grid["event_seed"]
    data["times"] = TIMES
    data["metadata"] = json.dumps(
        {
            "hnn_core": version("hnn-core"),
            "neuron": version("neuron"),
            "python": python_version(),
            "numpy": np.__version__,
            "seed": seed,
            "model": "neymotin_2020_model",
            "weight_basket_uS": 1e-4,
            "drive_sigma_ms": 3.0,
            "mesh_shape": [3, 3],
            "tstop_ms": 80,
            "dt_ms": 0.025,
            "smooth_ms": 5,
            "noise_sd_nAm": NOISE_SD,
            "n_trials": 1,
            "n_samples": n_samples,
            "mu_prior_ms": [25, 45],
            "weight_pyr_prior_uS": [1e-4, 8e-4],
        }
    )
    return data


def load_dataset(
    path=DATA_PATH,
    *,
    regenerate=False,
    n_train=1024,
    n_validation=128,
    n_test=128,
    seed=42,
    n_jobs=4,
):
    """Load HNN data and return independent train/validation/test splits."""
    path = Path(path)
    n_samples = n_train + n_validation + n_test
    if regenerate or not path.exists():
        start = perf_counter()
        generated = generate_dataset(n_samples, seed=seed, n_jobs=n_jobs)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **generated)
        print(f"Generated HNN data in {perf_counter() - start:.1f} s")

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in (*PARAMETER_KEYS, "dipole")}
        metadata = json.loads(str(archive["metadata"]))
        np.testing.assert_array_equal(archive["times"], TIMES)
        seeds = archive["event_seed"]
        if len(np.unique(seeds)) != n_samples:
            raise ValueError("Expected a distinct event seed per simulation.")
    if data["dipole"].shape != (n_samples, len(TIMES)):
        raise ValueError("Cached dataset does not match the requested splits.")
    if any(data[key].shape != (n_samples, 1) for key in PARAMETER_KEYS):
        raise ValueError("Each HNN parameter must have one column.")
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError("Cached HNN data contains nonfinite values.")
    if metadata["noise_sd_nAm"] != NOISE_SD:
        raise ValueError("Cached data uses a different measurement-noise SD.")
    edges = (0, n_train, n_train + n_validation, n_samples)
    return tuple(
        {key: values[start:stop] for key, values in data.items()}
        for start, stop in zip(edges[:-1], edges[1:], strict=True)
    )


def plot_waveforms(data, n_examples=40):
    """Plot the same waveforms colored by input timing and pyramidal weight."""
    if n_examples == 1:
        fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
        ax.plot(TIMES, data["dipole"][0])
        ax.set(xlabel="Time (ms)", ylabel="Current dipole (nAm)")
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    labels = ["Input time (ms)", "Pyramidal weight (nS)"]
    for ax, key, label in zip(axes, PARAMETER_KEYS, labels, strict=True):
        values = data[key][:n_examples, 0] * (
            1000 if key == "weight_pyr" else 1
        )
        norm = plt.Normalize(values.min(), values.max())
        for waveform, value in zip(
            data["dipole"][:n_examples], values, strict=True
        ):
            ax.plot(
                TIMES, waveform, color=plt.cm.viridis(norm(value)), alpha=0.65
            )
        ax.set(xlabel="Time (ms)", ylabel="Current dipole (nAm)")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=ax,
            label=label,
        )
    return fig
