"""Simulation and plotting helpers for a three-segment planar robot arm."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches

from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from sklearn.cluster import MeanShift
from sklearn.neighbors import KernelDensity


class InverseKinematics:
    """Map height and three relative joint angles to Cartesian endpoints.

    Parameter batches have shape ``(n_samples, 4)`` and contain height,
    angle 1, angle 2, and angle 3. Endpoints use ``(horizontal, vertical)``
    coordinates; the notebook's observables use the reverse order.
    """

    n_parameters = 4
    n_observations = 2
    name = "inverse-kinematics"

    def __init__(
        self,
        lens=(0.5, 0.5, 1.0),
        sigmas=(0.25, 0.5, 0.5, 0.5),
        linecolors=("gray", "gray", "gray"),
    ):
        self.lens = np.array(lens, dtype=float)
        self.sigmas = np.array(sigmas, dtype=float)
        if self.lens.shape != (3,) or not np.all(
            np.isfinite(self.lens) & (self.lens > 0)
        ):
            raise ValueError("lens must contain three finite positive values.")
        if self.sigmas.shape != (4,) or not np.all(
            np.isfinite(self.sigmas) & (self.sigmas > 0)
        ):
            raise ValueError(
                "sigmas must contain four finite positive values."
            )
        if len(linecolors) != 3:
            raise ValueError("linecolors must contain three colors.")

        self.linecolors = tuple(linecolors)
        self.rangex = (-0.35, 2.25)
        self.rangey = (-1.3, 1.3)
        self.colors = [cm.tab20c(16 + index) for index in range(3)]
        self.prior_reference = {
            f"theta_{index}": norm(0, scale)
            for index, scale in enumerate(self.sigmas, start=1)
        }

    def sample_prior(self, n_samples):
        """Draw Gaussian parameters, returning shape ``(n_samples, 4)``."""
        return np.random.normal(size=(n_samples, 4)) * self.sigmas

    @staticmethod
    def segment_points(start, length, angle):
        """Return segment starts and ends without mutating the input arrays."""
        start = np.asarray(start, dtype=float)
        offsets = length * np.column_stack((np.cos(angle), np.sin(angle)))
        return start, start + offsets

    def _joint_positions(self, parameters):
        """Compute the base and all joints once for simulation and plotting."""
        parameters = np.asarray(parameters, dtype=float)
        if parameters.ndim != 2 or parameters.shape[1] != self.n_parameters:
            raise ValueError("parameters must have shape (n_samples, 4).")

        base = np.zeros((len(parameters), 1, 2))
        base[:, 0, 1] = parameters[:, 0]
        angles = np.cumsum(parameters[:, 1:], axis=1)
        offsets = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        offsets *= self.lens[:, None]
        joints = base + np.cumsum(offsets, axis=1)
        return np.concatenate((base, joints), axis=1)

    def forward_process(self, x):
        """Return Cartesian arm endpoints with shape ``(n_samples, 2)``."""
        return self._joint_positions(x)[:, -1]

    @staticmethod
    def find_map(samples):
        """Return the sample index nearest the densest mean-shift center.

        This is a heuristic representative configuration, not an exact MAP
        estimate. Kernel density estimation ranks the cluster centers.
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or len(samples) == 0:
            raise ValueError("samples must be a nonempty 2D array.")
        centers = MeanShift().fit(samples).cluster_centers_
        kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(samples)
        best_center = centers[np.argmax(kde.score_samples(centers))]
        return np.argmin(np.sum((samples - best_center) ** 2, axis=1))

    # Preserve the original spelling for existing workshop code.
    find_MAP = find_map

    @staticmethod
    def arcarrow(start, target, dist=0.3, open_angle=150, kw=None, ax=None):
        """Draw an angle indicator on the supplied axes (or current axes)."""
        start = np.asarray(start, dtype=float)
        direction = np.asarray(target) - start
        angle = np.arctan2(direction[1], direction[0])
        angles = angle + np.radians(open_angle / 2) * np.array([-1, 1])
        endpoints = start + dist * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        if kw is None:
            kw = {
                "arrowstyle": "<->, head_width=1, head_length=2",
                "ec": "black",
                "lw": 0.5,
            }
        ax = plt.gca() if ax is None else ax
        ax.add_patch(
            patches.FancyArrowPatch(
                *endpoints, connectionstyle="arc3, rad=.6", **kw
            )
        )

    def draw_isolines(self, samples, color, filter_width, ax=None):
        """Draw a smoothed endpoint contour enclosing about 97% of grid mass.

        Mass is measured within the plotting ranges. A nonpositive smoothing
        width disables the contour.
        """
        if filter_width <= 0:
            return
        endpoints = self.forward_process(samples)
        hist, x_edges, y_edges = np.histogram2d(
            *endpoints.T, bins=600, range=[self.rangex, self.rangey]
        )
        hist = gaussian_filter(hist, filter_width)
        if hist.sum() == 0:
            return

        densities = np.sort(hist.ravel())
        cutoff = np.searchsorted(np.cumsum(densities), 0.03 * hist.sum())
        level = densities[min(cutoff, len(densities) - 1)]
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        ax = plt.gca() if ax is None else ax
        contour_style = dict(colors=color, linewidths=0.7, zorder=3)
        ax.contour(x_centers, y_centers, hist.T, [level], **contour_style)

    @staticmethod
    def init_plot():
        """Create an empty square figure for arm visualizations."""
        return plt.figure(figsize=(8, 8))

    def update_plot_ax(
        self,
        ax,
        x,
        y_target,
        exemplar=None,
        arrows=False,
        target_label=False,
        vline_color="black",
        exemplar_color="#e6e7eb",
        cross_color="maroon",
    ):
        """Overlay sampled configurations and a representative arm on axes."""
        joints = self._joint_positions(x)
        if exemplar is None:
            exemplar = self.find_map(x)
        arm = joints[exemplar]
        ax.axvline(0, color=vline_color, linewidth=1, alpha=0.8)

        if not arrows:
            target = np.asarray(y_target)
            cross_style = dict(
                color=cross_color, linewidth=0.8, rasterized=True
            )
            for offset in (np.array([0.6, 0]), np.array([0, 0.6])):
                cross = np.stack((target - offset, target + offset))
                ax.plot(*cross.T, **cross_style)
            if target_label:
                label_style = dict(ha="left", va="bottom", color="magenta")
                ax.text(
                    *(target + 0.15), target_label, fontsize=10, **label_style
                )

        quiver_options = {
            "alpha": 0.10,
            "scale": 1,
            "angles": "xy",
            "scale_units": "xy",
            "headlength": 0,
            "headaxislength": 0,
            "linewidth": 1.0,
            "rasterized": True,
        }
        for index, color in enumerate(self.linecolors):
            start = joints[:, index]
            offset = joints[:, index + 1] - start
            ax.quiver(*start.T, *offset.T, color=color, **quiver_options)
        endpoint_style = dict(color=self.linecolors[0], s=1, alpha=0.20)
        ax.scatter(*joints[:, -1].T, rasterized=True, **endpoint_style)
        arm_style = dict(color=exemplar_color, linewidth=2, zorder=4)
        ax.plot(*arm[:3].T, rasterized=True, **arm_style)

        if arrows:
            arrow_style = {
                "arrowstyle": "<->, head_width=.1, head_length=.2",
                "ec": "black",
                "lw": 0.5,
            }
            ax.annotate(
                "",
                xy=(-0.125, -0.5),
                xytext=(-0.125, 0.5),
                zorder=2,
                arrowprops=arrow_style,
            )
            for start, end in zip(arm[:-1], arm[1:], strict=True):
                self.arcarrow(start, end, ax=ax)
            labels = (
                (-0.09, -0.60, r"$x_1$"),
                (0.13, -0.38, r"$x_2$"),
                (0.60, -0.40, r"$x_3$"),
                (1.10, -0.44, r"$x_4$"),
                (1.97, -0.27, r"$\mathbf{y}$"),
            )
            label_style = dict(ha="center", va="center", fontsize=10)
            for horizontal, vertical, label in labels:
                ax.text(horizontal, vertical, label, **label_style)

        arrow_head = dict(head_width=0.05, head_length=0.04, overhang=0.1)
        ax.arrow(
            *arm[2],
            *(arm[3] - arm[2]),
            length_includes_head=True,
            **arm_style,
            **arrow_head,
        )
        joint_options = {
            "linewidth": 1,
            "edgecolors": "black",
            "facecolors": exemplar_color,
            "alpha": 0.8,
            "rasterized": True,
        }
        ax.scatter(*arm[0], s=40, marker="s", zorder=3, **joint_options)
        ax.scatter(*arm[:3].T, s=20, zorder=5, **joint_options)
        ax.set(xlim=(-0.01, 1.8), ylim=self.rangey, aspect="equal")
        ax.set(xticks=[], yticks=[])
        return ax
