# AI for Science Summer School BayesFlow Materials

This repository hosts hands-on tutorials for simulation-based inference (SBI) with [BayesFlow](https://github.com/bayesflow-org/bayesflow). Learn to build Bayesian simulators, train neural estimators, and put them to the test.

## Install with uv

The workshop pins **[BayesFlow 2.0.14](https://pypi.org/project/bayesflow/2.0.14/)** and uses **JAX** as the Keras backend. BayesFlow 2.0.14 requires Python **3.12 or 3.13**; `.python-version` selects 3.12. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/bayesflow-org/ai4science.git
cd ai4science
uv sync
```

You can also download the repository as a ZIP, extract it, and run the last two commands from the extracted folder. uv downloads Python if needed, creates `.venv`, and installs the dependencies declared in `pyproject.toml`. `uv.lock` records the resolved versions for reproducible workshop environments. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/) for details.

For VS Code, select `.venv` as the notebook's Python environment. I recommend installing the UV extension too.

## Repository contents

```text
ai4science/
├── pyproject.toml                 # Dependencies, packaging, and code checks
├── uv.lock                        # Resolved dependency versions
├── tutorials/
│   ├── diffusion-models.ipynb      # Diffusion-based posterior inference
│   └── helpers/
│       └── kinematics.py          # Robot-arm simulation and plotting
└── tests/                         # Helper regression tests
```

The tutorial infers the height and three angles of a planar robot arm from its endpoint. Several configurations can produce the same endpoint, giving an example of a multimodal posterior. It covers simulation, offline training, posterior sampling, coverage checks, and inference-time guidance. Training time depends on hardware and compilation overhead.

## What is simulation-based inference?

A simulator generates data $x$ from parameters $\theta$, even when its likelihood $p(x | \theta)$ is difficult or impossible to evaluate. Bayesian inference combines that model with a prior to obtain $p(\theta | x) \propto p(x | \theta) p(\theta)$.

In amortized SBI, we draw parameters from the prior, simulate matching data, and train a neural network on these pairs. The trained estimator can then be reused on new datasets from the same model and training domain. The simulation and training costs are paid up front; subsequent inference can be much faster. BayesFlow provides simulators, data adapters, neural networks, training workflows, and diagnostics for this process. See the [BayesFlow workflow paper](https://arxiv.org/abs/2306.16015).

### NPE, NLE, and NRE

These methods differ in what they learn and how that result is used.

| Method | Learned quantity | Inference for observed data | BayesFlow components |
| --- | --- | --- | --- |
| Neural posterior estimation (NPE) | $$q(\theta \mid x)$$ | Draw approximate posterior samples directly. | `BasicWorkflow` or `ContinuousApproximator` with parameters as inference variables and data as conditions. |
| Neural likelihood estimation (NLE) | $$q(x \mid \theta)$$ | Combine the learned likelihood with a prior; use a sampler such as MCMC to infer parameters. | `ContinuousApproximator` with data as inference variables and parameters as conditions. |
| Neural ratio estimation (NRE) | $$r(\theta, x) \approx \frac{p(x \mid \theta)}{p(x)}$$ | Multiply the ratio by the prior to obtain an unnormalized posterior; use MCMC or importance sampling. | `RatioApproximator`, which implements contrastive NRE-C. |

NPE is convenient when many datasets need posterior inference under a fixed prior. NLE and NRE let you combine the learned model with different priors, provided the new prior stays within the parameter region represented during training. In NRE, the evidence $p(x)$ is constant with respect to parameters for a fixed observation, so it cancels in posterior normalization. See the [BayesFlow approximator guide](https://bayesflow.org/v2.0.14/user_guide/approximators.html).

The diffusion tutorial uses **NPE**. Its central configuration is:

```python
workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=bf.networks.DiffusionModel(
        subnet_kwargs={"widths": (128,) * 3}
    ),
    inference_variables="parameters",
    inference_conditions="observables",
    standardize="all"
)
```

Here `simulator` produces dictionaries containing `parameters` and `observables`, as shown in the notebook. Train with `workflow.fit_offline(...)` and draw samples with `workflow.sample(conditions=..., num_samples=...)`. For stochastic models with a continuous observation density, reversing the inferred and conditioning variables defines an NLE task; its samples would be simulated data conditional on parameters. This tutorial's simulator is deterministic, so density-based NLE would require an explicit observation-noise model.

**Offline training** uses a stored simulation dataset, as in this notebook. **Online training** generates fresh simulations during training. **Sequential SBI** concentrates later simulations around a particular observation; NPE then needs to account for the changed parameter proposal to target the original prior's posterior.

### Scoring rules and posterior summaries

Scoring rules turn predictions and realized outcomes into losses. A proper scoring rule is minimized in expectation by the true predictive distribution; strict propriety makes that optimum unique. Examples include the negative logarithmic score, the continuous ranked probability score (CRPS) for scalar outcomes, and the energy score for multivariate outcomes. Sample-based scores can also train generative posterior estimators without evaluating their density; see the [scoring-rule minimization paper](https://arxiv.org/abs/2205.15784).

BayesFlow's `ScoringRuleNetwork` and `ScoringRuleApproximator` support amortized Bayes risk minimization: squared-error loss estimates posterior means, absolute-error loss estimates medians, and quantile loss estimates chosen posterior quantiles. Distributional scoring heads can fit parametric posterior approximations; sampling and density evaluation depend on the configured head. A mean or a few quantiles alone do not describe a full multimodal posterior. See the [scoring-rule API](https://bayesflow.org/v2.0.14/api/bayesflow.approximators.ScoringRuleApproximator.html).

The **score function** used by diffusion models, `∇theta log p(theta | x)`, is a gradient of log density. It is a different concept from a **scoring rule**, which evaluates predictions.

### More approaches available in BayesFlow

- **Normalizing flows**, such as `CouplingFlow`, transform a simple base distribution using invertible mappings and train with negative log density.
- **Flow matching** learns a time-dependent velocity field that transports noise to posterior samples through an ordinary differential equation.
- **Diffusion models** learn to reverse a noise process using a score or equivalent parameterization, then generate samples through numerical integration. The notebook illustrates guidance during sampling; guided samples target a modified distribution and need separate validation.
- **Consistency models** learn mappings from noisy states to clean samples, enabling sampling with few network evaluations.
- **Summary networks** encode structured observations before inference, for example using `DeepSet` for exchangeable observations or a transformer.
- **Model comparison**, **ensembles**, and **compositional inference** extend the workflow to discrete model probabilities, combinations of estimators, and combinations of conditional distributions under suitable assumptions.

See the [inference network guide](https://bayesflow.org/v2.0.14/user_guide/inference_networks.html) and [workflow guide](https://bayesflow.org/v2.0.14/user_guide/workflows.html) for architectures and examples.

### Validate the inference

A low training loss alone does not establish accurate posterior inference. Use independent simulations to check parameter recovery, credible-interval coverage, and simulation-based calibration (SBC) ranks. Posterior predictive checks simulate data from inferred parameters and compare them with observed data. For the robot arm, compare simulated endpoints with the target and inspect whether all plausible configurations are represented.

Calibration checks assess behavior over simulated datasets; they do not guarantee accuracy for every observation. Also check whether real observations and parameter values are represented by the training distribution. The notebook demonstrates `bf.diagnostics.plots.coverage`; see the [BayesFlow diagnostics reference](https://bayesflow.org/v2.0.14/api/bayesflow.diagnostics.html) for additional checks.
