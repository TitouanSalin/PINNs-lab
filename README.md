# When Does Physics Help? Success and Failure Modes of Physics-Informed Neural Networks

## Team

- [Team member 1]
- [Team member 2]
- [Team member 3]

## Topic and central question

**Topic:** Physics-informed learning, with a focus on Physics-Informed Neural Networks (PINNs).

PINNs train a neural network not only from observations, but also by penalizing violations of a known physical law. For a PDE written abstractly as

$$\mathcal{N}[u](x,t)=0,$$

the training objective typically combines a data or boundary-condition loss with a physics residual:

$$
\mathcal{L}
=
\mathcal{L}_{\text{data}}
+
\lambda_{\text{physics}}\mathcal{L}_{\text{physics}}.
$$

Our central question is:

> **How does increasing the difficulty of a physical system affect a PINN's ability to recover the correct solution, and can a simple change in the training strategy mitigate the resulting failure?**

We also want to investigate whether a small physics loss necessarily implies that the learned solution is accurate.

---

## Papers

Our investigation will primarily build on:

1. **Raissi, Perdikaris, and Karniadakis — Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations**  
   https://arxiv.org/abs/1711.10561

   This paper provides the basic PINN formulation and motivates the use of PDE residuals as an additional source of supervision.

2. **Krishnapriyan et al. — Characterizing Possible Failure Modes in Physics-Informed Neural Networks**  
   https://arxiv.org/abs/2109.01050

   This paper studies cases where standard PINN training fails as the underlying physical problem becomes more difficult, and explores strategies such as curriculum training.

We may also use the other physics-informed learning papers from the course for additional context, but the lab will remain focused on PINNs rather than equation discovery.

---

## Why is this question interesting?

PINNs are appealing because they allow known physical laws to constrain a neural model even when only a small amount of observed data is available.

However, knowing the correct governing equation does not guarantee that the neural network will successfully learn the correct solution. As the physical regime becomes harder, the PINN optimization problem can become difficult and the model may converge to an inaccurate solution.

This creates an interesting contrast:

> **Physics can provide a strong and useful inductive bias, while still producing a difficult optimization problem.**

We want the lab to make both sides of this behavior visible: first why the physics constraint helps, and then how and why standard PINN training can fail.

---

## Tentative plan

### 1. Toy case — Why does the physics loss help?

We will start with a simple one-dimensional differential equation with a known analytical solution, for example:

$$
u''(x)=-\pi^2\sin(\pi x),
$$

with

$$
u(0)=u(1)=0.
$$

The exact solution is

$$
u(x)=\sin(\pi x).
$$

We will compare:

- a neural network trained only from a few observed points;
- a PINN trained from the same observations plus the physics residual.

The goal is to isolate the main PINN mechanism and show how physical constraints can improve learning when observations are sparse.

---

### 2. Paper result — When does a PINN fail?

We plan to reproduce a qualitative result from Krishnapriyan et al. using a one-dimensional convection equation:

$$
u_t+\beta u_x=0.
$$

The parameter $\beta$ will act as a controllable measure of problem difficulty.

We will train or cache PINN results for several values of $\beta$ and investigate how prediction accuracy changes as $\beta$ increases.

We will compare:

- the exact solution;
- the PINN prediction;
- the pointwise error;
- the PDE residual;
- the relative solution error;
- relevant training curves.

The goal is to reproduce the observation that a PINN can perform well in an easier regime but fail as the physical problem becomes harder.

---

### 3. Extension — Can a different training strategy reduce the failure?

Our planned extension is **curriculum training**.

Instead of training directly on a difficult target such as $\beta=30$, we will gradually increase the difficulty, for example:

$$
1 \rightarrow 5 \rightarrow 10 \rightarrow 20 \rightarrow 30.
$$

We will compare direct training against curriculum training while keeping the network architecture and final physical problem fixed.

This will let us study whether the optimization path itself has an important effect on the final solution.

---

## Interactive visualization and exploration

The interactive component will be designed around meaningful scientific variations rather than generic hyperparameter tuning.

In the toy notebook, users should be able to vary quantities such as:

- the number of observed data points;
- the number of collocation points;
- the physics-loss weight $\lambda_{\text{physics}}$.

They will be able to compare the data-only model and the PINN and immediately observe how these choices affect the learned function and the PDE residual.

For the convection experiment, we plan to provide an interactive control for $\beta$. Changing $\beta$ will load cached results and update several linked views, including:

- ground-truth solution;
- PINN prediction;
- pointwise error;
- PDE residual;
- quantitative relative error.

We also plan to provide a direct comparison between **standard training** and **curriculum training** for the same difficult regime.

The main class-time path will use cached models and arrays so that interactions update quickly without requiring full retraining. At least one lightweight experiment in the toy notebook will be computed live, for example by changing the physics-loss weight or number of collocation points and running a small number of additional optimization steps.

Fallback plots and cached outputs will be included so that the investigation remains usable even if live computation fails.

The intended exploration should help another group answer the following questions:

- When does physics-informed supervision help compared with data-only fitting?
- How does PINN performance change as the physical regime becomes harder?
- Does a small PDE residual necessarily imply an accurate solution?
- Can changing only the training strategy improve the result?
