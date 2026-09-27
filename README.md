# When Does Physics Help? Success and Failure Modes of Physics-Informed Neural Networks

## Team

Team Name: the best team ever

- Saanya Manoj
- Dorsa Molaverdikhani
- Titouan Salin

## Topic and central question

**Topic:** Physics-informed learning, with a focus on Physics-Informed Neural Networks (PINNs).

PINNs train a neural network not only from observations, but also by penalizing violations of a known physical law. For a PDE written abstractly as

```math
\mathcal{N}\left[u\right](x,t) = 0,
```

the training objective typically combines a data or boundary-condition loss with a physics residual:

```math
\mathcal{L}
=
\mathcal{L}_{\text{data}}
+
\lambda_{\text{physics}} \mathcal{L}_{\text{physics}}.
```

Our central question is:

> **How does increasing the difficulty of a physical system affect a PINN's ability to recover the correct solution, and can a simple change in the training strategy mitigate the resulting failure?**

---

## Papers

Our investigation will primarily build on:

1. **Raissi, Perdikaris, and Karniadakis — Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations**  
   https://arxiv.org/abs/1711.10561

   This paper provides the basic PINN formulation and motivates the use of PDE residuals as an additional source of supervision.

2. **Krishnapriyan et al. — Characterizing Possible Failure Modes in Physics-Informed Neural Networks**  
   https://arxiv.org/abs/2109.01050

   This paper studies cases where standard PINN training fails as the underlying physical problem becomes more difficult, and explores strategies such as curriculum training.

We will also probably use other physics-informed learning papers for the last notebook on possible extensions on PINNs failure.

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

### 3. Extension — Can we mitigate PINN failure?

The third notebook will investigate whether a controlled modification of the standard PINN training procedure can mitigate the failure observed in the previous experiment.

One possibility is **curriculum training**, following the idea explored by Krishnapriyan et al. Instead of training directly on a difficult target such as $\beta = 30$, we would progressively increase the difficulty:

```math
1 \rightarrow 5 \rightarrow 10 \rightarrow 20 \rightarrow 30.
```

We could then compare direct training and curriculum training while keeping the network architecture and final physical problem fixed. This would help us study whether the optimization path itself has an important effect on the final solution.

We also plan to explore other approaches proposed in the PINN literature for improving training and robustness. Possible directions include:

- **Adaptive balancing of the loss terms and gradients**, motivated by *Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks* (Wang et al.), which studies imbalanced gradients between the different components of the PINN loss and proposes adaptive weighting strategies.  
  https://arxiv.org/abs/2001.04536

- **Adaptive weighting based on training dynamics**, motivated by *When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective* (Wang et al.), which analyzes differences in the convergence rates of the different PINN loss components using the Neural Tangent Kernel.  
  https://arxiv.org/abs/2007.14527

- **Self-adaptive PINNs**, following *Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism* (McClenny and Braga-Neto), where trainable weights allow the model to focus more strongly on regions of the domain that are difficult to learn.  
  https://arxiv.org/abs/2009.04544

- **Domain decomposition with XPINNs**, following Jagtap and Karniadakis, where the space-time domain is divided into subdomains handled by separate neural networks. This provides another possible way of handling more difficult PDE regimes.  
  https://github.com/AmeyaJagtap/XPINNs/blob/master/XPINNs_Paper.pdf

- We might also take a look at other papers.

For the final lab, we will select one of these modifications and compare it against the standard PINN under the same difficult regime.

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

In the last notebook, we have not planned any visualization yet, as we need to study further the possible extensions.

### AI Usage Acknowledgement

We used generative AI tools to help organize, clarify, and format our initial ideas for this README.
