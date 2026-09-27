# When Does Physics Help? Exploring Success and Failure Modes of Physics-Informed Neural Networks

## Team

- [Team member 1]
- [Team member 2]
- [Team member 3]

## Topic

**Physics-informed learning — Physics-Informed Neural Networks (PINNs)**

Our lab will investigate how Physics-Informed Neural Networks use differential equations as an inductive bias, why this can be useful when observations are sparse, and why the same physics-informed objective can become difficult to optimize as the underlying physical regime becomes more challenging.

Rather than presenting PINNs only as a method for solving PDEs, we want the lab to expose both their strengths and their failure modes through a sequence of interactive experiments.

---

## Central question

Our central question is:

> **How does increasing the difficulty of a physical system affect a PINN's ability to recover the correct solution, and can a simple change in the training strategy mitigate the resulting failure?**

We plan to study this through three connected questions:

1. **When does adding a physics constraint help compared with fitting sparse observations alone?**
2. **How can a PINN fail even when the governing physical equation is known?**
3. **Can a controlled change in the optimization strategy recover the correct solution in regimes where standard training fails?**

A secondary question that will appear throughout the lab is:

> **Does a small physics residual necessarily imply that the learned solution is accurate?**

This distinction between satisfying the training objective and recovering the correct physical solution will be one of the main ideas we want another group to discover through exploration.

---

## Papers

Our investigation will build primarily on the following papers.

### Physics-Informed Neural Networks

Maziar Raissi, Paris Perdikaris, and George Em Karniadakis.  
**Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.**  
arXiv:1711.10561, 2017.

https://arxiv.org/abs/1711.10561

This paper introduces the core PINN formulation that we will use in the toy experiment: a neural representation is trained not only from observations or boundary conditions, but also by penalizing violations of the governing differential equation at collocation points.

### PINN failure modes

Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, and Michael W. Mahoney.  
**Characterizing Possible Failure Modes in Physics-Informed Neural Networks.**  
NeurIPS 2021.

https://arxiv.org/abs/2109.01050

This paper will motivate the second and third stages of our investigation. In particular, it shows that standard PINN training can fail on relatively simple differential equations as the physical regime becomes harder, and studies strategies such as curriculum training for mitigating these failures.

We may also use the following course readings for broader context around physics-informed learning and equation discovery:

- Raissi et al., **Hidden Fluid Mechanics**  
  https://arxiv.org/abs/1808.04327
- Brunton et al., **Discovering governing equations from data by sparse identification of nonlinear dynamical systems**  
  https://arxiv.org/abs/1509.03580
- Champion et al., **Data-driven discovery of coordinates and governing equations**  
  https://arxiv.org/abs/1904.02107

However, our current plan is to keep the lab focused on the PINN side of physics-informed learning rather than attempting to cover both PDE solving and equation discovery in a single investigation.

---

## Why is this question interesting?

Physics-informed learning is attractive because it allows prior scientific knowledge to become part of the learning objective.

For a standard supervised neural network, knowledge about the solution generally comes from observations. A PINN adds another source of information: the governing differential equation itself. The network is encouraged to produce a function whose derivatives satisfy the known physical law across the domain.

This makes PINNs particularly appealing when observations are limited.

However, knowing the correct physical equation does not automatically make the learning problem easy. The resulting optimization problem can still have difficult landscapes, competing loss terms, and solutions that obtain low training losses without accurately representing the desired physical solution.

We find this tension particularly interesting:

> **Physics can strongly constrain what the correct answer should look like, while simultaneously making the optimization problem difficult.**

The goal of our lab is therefore not simply to demonstrate that PINNs can solve differential equations. Instead, we want users to experimentally determine:

- when the physics constraint provides useful information;
- what PINN failure looks like;
- whether common diagnostics make failure obvious;
- and whether a simple intervention can improve training.

This also gives the lab a natural experimental progression: **mechanism → failure → intervention**.

---

# Tentative investigation

The final lab will consist of three connected notebooks.

## Notebook 1 — Toy case: how does the physics loss constrain a neural solution?

The first notebook will isolate the basic PINN mechanism in the smallest setting possible.

We currently plan to use a one-dimensional differential equation with a known analytical solution, for example:

\[
u''(x) = -\pi^2 \sin(\pi x),
\]

with boundary conditions

\[
u(0)=u(1)=0,
\]

whose solution is

\[
u(x)=\sin(\pi x).
\]

We will compare two neural representations with the same architecture:

### Data-only model

The first model will only fit a small set of observed values:

\[
\mathcal{L}_{data}
=
\frac{1}{N}
\sum_i
|u_\theta(x_i)-u_i|^2.
\]

### Physics-informed model

The second model will additionally minimize the differential-equation residual:

\[
\mathcal{L}
=
\mathcal{L}_{data}
+
\lambda_{\mathrm{physics}}
\mathcal{L}_{\mathrm{PDE}}.
\]

Automatic differentiation will be used to compute the required derivatives.

The goal is not to demonstrate a difficult PDE solver, but to make the role of the physics term visually obvious.

### Questions for the user

Users should be able to investigate questions such as:

- How many observations does a data-only network need?
- Can a PINN recover the solution from fewer observations?
- What happens if the physics term is weighted too weakly?
- What happens if it is weighted too strongly?
- How does the number or location of collocation points affect the result?
- Can the network fit the observed points while behaving incorrectly between them?

---

## Notebook 2 — Paper result: when does a PINN start to fail?

The second notebook will move from a deliberately easy example to a regime where PINN optimization becomes challenging.

Our current plan is to reproduce a qualitative result from Krishnapriyan et al. using a one-dimensional convection equation such as

\[
u_t + \beta u_x = 0.
\]

The coefficient \(\beta\) provides a natural experimental control.

Rather than treating the PDE as a single benchmark, we will interpret \(\beta\) as a **difficulty parameter** and investigate how the behavior of the trained PINN changes as it increases.

We expect to pretrain models for several representative values, for example

\[
\beta \in \{1,5,10,20,30,40\},
\]

with the exact grid potentially adjusted after initial experiments.

For every regime, we plan to store:

- the exact solution;
- the PINN prediction;
- the pointwise absolute error;
- the PDE residual over the space-time domain;
- relative \(L_2\) error;
- training loss curves;
- physics and boundary-condition losses separately;
- selected checkpoints during training when useful.

The objective will be to reproduce the qualitative observation that a standard PINN can work well in an easier regime and deteriorate sharply as the physical problem becomes harder.

---

## Notebook 3 — Extension: can we recover from the failure?

The third notebook will make one controlled modification.

Our current extension is **curriculum training**.

Instead of training the target system directly at a difficult parameter value such as

\[
\beta=30,
\]

we will initialize training using an easier system and gradually increase the difficulty, for example

\[
\beta=1
\rightarrow
5
\rightarrow
10
\rightarrow
20
\rightarrow
30.
\]

We will compare:

- **direct training** on the final target problem;
- **curriculum training** ending at the same target problem.

The architecture, target PDE, final parameter value, evaluation grid, and main training budget will be kept as controlled as possible.

This should let us investigate whether the failure originates purely from model capacity, or whether the path taken during optimization has an important effect.

A possible additional extension, if time permits, will be to compare a standard MLP representation with a representation better suited to higher-frequency functions, such as Fourier features. This would connect the investigation to neural representations studied elsewhere in the course. We currently view this as optional rather than part of the core lab.

---

# Interactive visualization and exploration design

Interactive exploration will be a central component of the lab rather than an interface added after the experiments are complete.

Our goal is for another group to be able to **form hypotheses, manipulate meaningful quantities, visually compare outcomes, and reach the main conclusions themselves** within the class period.

The interaction will therefore be organized around scientific questions rather than around implementation parameters.

---

## 1. A consistent visual language across all three notebooks

Where possible, the notebooks will use the same visual conventions throughout the investigation.

For example, users will repeatedly encounter:

- the exact or reference solution;
- the neural prediction;
- the prediction error;
- the physics residual;
- quantitative error metrics.

This should make the progression from the toy setting to the harder PDE easy to follow.

For two-dimensional space-time solutions, we plan to use aligned heatmaps such as:

| Ground truth | PINN prediction |
|---|---|
| Pointwise error | PDE residual |

The plots will use common axes and normalization wherever appropriate so that differences can be visually compared rather than inferred from unrelated figures.

---

## 2. Interactive Toy Case: manipulate the sources of supervision

The first notebook will contain lightweight interactions that can run live.

Potential controls include:

- **number of observed data points**;
- **number of collocation points**;
- **physics-loss weight \(\lambda_{\mathrm{physics}}\)**;
- **data-loss weight**;
- optionally the spatial distribution of collocation points.

A possible interface would expose controls such as:

```text
Observed points       [------|------] 5
Collocation points    [----------|--] 50
Physics weight λ      [-----|-------] 1.0

Model:
(o) Data only
( ) Physics informed
```

The corresponding plot would update to show:

- analytical solution;
- current neural prediction;
- observed training points;
- collocation locations;
- optionally the PDE residual along the domain.

One particularly useful interaction will allow users to compare the **same sparse observations** under data-only and physics-informed training.

This should make the inductive bias visible directly instead of only reporting a final numerical error.

### Lightweight live training

At least part of this notebook will support actual live computation.

Because the toy network and differential equation will be deliberately small, users should be able to modify one parameter and perform a small number of additional optimization steps without retraining a large model.

For example, users could change \(\lambda_{\mathrm{physics}}\) and run another 50–200 optimization steps.

We will explicitly display whether an output comes from:

- **live computation**, or
- a **cached pretrained result**.

The expected runtime will be shown next to each live action.

---

## 3. Interactive difficulty sweep

The core interaction of the second notebook will treat physical difficulty as something users can directly vary.

We envision a slider similar to:

```text
Convection coefficient β

1 -------- 5 -------- 10 -------- 20 -------- 30 -------- 40
                                             ^
```

Changing the parameter will immediately load the corresponding cached solution and update all linked visualizations.

This will allow a user to move continuously through the experimental story:

```text
easy regime
     ↓
moderate regime
     ↓
failure regime
```

Instead of showing only one successful and one failed example, users will be encouraged to identify where the qualitative transition occurs.

We may also expose a plot of

\[
\beta
\quad \text{vs.} \quad
\text{relative }L_2\text{ error},
\]

with the currently selected experiment highlighted.

This gives both a global and local view of the experiment.

---

## 4. Linked solution, error, and residual views

A major design goal will be to avoid making the prediction itself the only diagnostic.

For a selected model, users will simultaneously inspect:

### Solution

\[
u_\theta(x,t)
\]

### Ground truth

\[
u^*(x,t)
\]

### Pointwise prediction error

\[
|u_\theta(x,t)-u^*(x,t)|
\]

### Physics residual

\[
|u_t+\beta u_x|.
\]

This allows users to ask whether regions with a small physics residual also correspond to regions with a small solution error.

Where practical, hovering over a location in one heatmap will expose the corresponding coordinates and values in the others.

This is intended to make the distinction between **optimizing the PINN objective** and **recovering the correct physical solution** concrete.

---

## 5. Loss curves as an interactive diagnostic

We plan to expose training histories rather than showing only final checkpoints.

Possible curves include:

\[
\mathcal{L}_{total},
\qquad
\mathcal{L}_{physics},
\qquad
\mathcal{L}_{BC},
\qquad
\text{relative solution error}.
\]

A training-step slider may allow users to inspect intermediate checkpoints:

```text
Training step

0 ----------------------●---------------------- 20,000
```

As the user changes the checkpoint, the predicted solution and error map can update.

This creates an opportunity to investigate questions such as:

> Does the physics loss decrease before the correct solution is learned?

or:

> Can optimization appear to be progressing according to the loss while the actual prediction remains poor?

We expect this to be particularly useful for interpreting failure cases.

---

## 6. Direct comparison mode

Whenever possible, important comparisons will be presented **side-by-side rather than sequentially**.

For example:

```text
β = 30

STANDARD PINN             CURRICULUM PINN

Prediction                Prediction

Error                     Error

Residual                   Residual

Rel. L2: ...              Rel. L2: ...
```

Users should therefore be able to compare methods without remembering what an earlier plot looked like.

The same principle will be used in the first notebook for data-only vs physics-informed learning.

---

## 7. Extension interaction: direct training vs curriculum

The final notebook will expose the extension through a method selector:

```text
Training strategy

(o) Direct
( ) Curriculum
```

Users will be able to select the same difficult target regime under both strategies.

We will show the two final solutions, their errors, and their training trajectories.

If runtime allows, we may additionally let users control the curriculum itself, for example by choosing between:

```text
Direct:
30

Short curriculum:
5 → 15 → 30

Gradual curriculum:
1 → 5 → 10 → 20 → 30
```

These variants would use cached results.

The purpose would not be to optimize the best possible curriculum, but to make the role of the optimization path visible.

---

## 8. Hypothesis-first exploration

We want users to make predictions before seeing some results.

For example, before revealing a high-\(\beta\) experiment, the notebook may ask:

> The network architecture, number of collocation points, and training procedure will remain unchanged, but \(\beta\) will increase from 1 to 30. What do you expect to happen?

Similarly, before comparing curriculum and direct training:

> If the network architecture is unchanged, should changing only the path through parameter space affect the final solution?

The notebook will then allow the user to reveal and inspect the experiment.

These short prompts are intended to turn the notebook from a sequence of figures into an investigation.

---

## 9. Meaningful rather than arbitrary controls

We want to avoid interactions that simply expose every hyperparameter.

The main controls will correspond to quantities with a clear scientific interpretation:

- amount of observed information;
- strength of the physics constraint;
- number/location of physics constraints;
- physical difficulty parameter;
- training strategy;
- training stage.

Parameters such as hidden-layer width, learning rate, or optimizer settings may be available in optional sections, but they will not form the main exploration path unless they become scientifically relevant to the central question.

---

## 10. Fast default exploration path

The default class-time path should not require expensive retraining.

We plan to precompute and cache representative models and arrays for the parameter sweep.

Cached artifacts may include:

- model predictions on fixed evaluation grids;
- analytical/reference solutions;
- PDE residual grids;
- pointwise error grids;
- scalar error metrics;
- training curves;
- selected intermediate checkpoints.

Changing a major interactive control such as \(\beta\) or training strategy should therefore update immediately by loading cached arrays.

Expensive model training will be clearly separated from the exploration path.

The default path through all three notebooks should fit comfortably within the approximately 55–60 minute exploration period.

---

## 11. Clear distinction between cached and live experiments

Every interactive element will be labelled as either:

**LIVE**  
Runs a lightweight computation when the user interacts with it.

or

**CACHED**  
Displays results generated in advance from a full training run.

For example:

```text
Physics weight experiment
LIVE — expected runtime: < 10 s

β sweep
CACHED — updates immediately

Full PINN retraining
OPTIONAL — not required for the class-time path
```

The final runtime numbers will be measured on the target environment rather than estimated.

---

## 12. Lightweight live modification

The assignment requires at least one meaningful live modification without full retraining.

Our current plan is for the toy notebook to satisfy this requirement.

Users will be able to change a scientifically meaningful parameter such as:

- physics-loss weight;
- number of observed points;
- or number of collocation points;

and run a short optimization update.

The resulting change in the predicted function and PDE residual will be displayed immediately.

If feasible, we may also support a second live experiment in the harder PDE notebook, for example taking a cached checkpoint and applying a small number of additional optimization steps under a modified loss weight.

---

## 13. Failure-safe design

Because another group must be able to complete the investigation during one class, the lab should remain usable even if the compute environment does not behave as expected.

For every essential experiment, we will provide cached fallback outputs.

If a widget or training step fails, the notebook will still include:

- static reference figures;
- cached predictions;
- cached error maps;
- stored scalar metrics;
- explanatory captions.

The scientific argument should therefore remain accessible without depending on successful GPU execution or long training jobs.

---

# Intended exploration path

A possible path through the lab will be:

### Stage 1 — Discover the mechanism

Start with only a few observations.

Compare data-only and physics-informed learning.

Vary the amount and strength of physical supervision.

**Tentative conclusion:**  
The differential equation provides useful information between sparse observations.

### Stage 2 — Increase the physical difficulty

Move to the convection experiment.

Start from an easy regime and gradually increase \(\beta\).

Inspect predictions, residuals, errors, and training histories.

**Tentative conclusion:**  
Providing the correct physical law does not guarantee that the optimization procedure will recover the correct solution.

### Stage 3 — Diagnose the failure

Compare physics residual and actual solution error.

Inspect training trajectories.

**Tentative conclusion:**  
A low or decreasing PINN training objective is not by itself sufficient evidence that the correct physical solution has been recovered.

### Stage 4 — Modify the optimization path

Compare direct and curriculum training for the same difficult target problem.

**Tentative conclusion:**  
The optimization path can have a substantial effect even when the architecture and final physical problem remain unchanged.

---

# What we hope another group will conclude

After exploring the three notebooks, we hope the reviewing group will be able to articulate something close to the following:

> Physics-informed losses provide a powerful inductive bias and can compensate for sparse observations, but imposing the correct physical equation does not automatically make neural optimization reliable. As the physical regime becomes more challenging, standard PINN training can fail even on relatively simple systems. Examining both the physical residual and the actual solution error is important, and changing the training strategy can substantially affect whether the correct solution is recovered.

Importantly, we want this conclusion to emerge from the users' own manipulations and comparisons rather than being stated only in explanatory text.

---

# Tentative technical plan

We currently expect to implement the models in PyTorch and use automatic differentiation for the physical derivatives.

Interactive components will likely use lightweight Jupyter-compatible tools such as `ipywidgets` and Matplotlib.

The class-time version will prioritize CPU-compatible cached exploration. Full training scripts will be included for reproducibility but will not be required to follow the main interactive path.

All important cached outputs will be small enough to store in the repository when possible. We will avoid committing large checkpoints, environments, private data, or unnecessary generated artifacts.

---

# Current scope

The exact PDE parameters, architecture, training budgets, and interactive controls may change as we experiment and receive feedback.

However, we intend to preserve the core structure:

> **Show why physics helps → expose a PINN failure → make the failure explorable → test a controlled intervention.**

This structure will connect the three notebooks into a single investigation rather than three independent demonstrations.
