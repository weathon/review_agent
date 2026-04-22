# CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 2

## Abstract
Neural operator surrogates for time-dependent partial differential equations (PDEs) conventionally employ autoregressive (AR) prediction schemes, which accumulate error over long rollouts and require uniform temporal discretization.
We introduce the Continuous Flow Operator (CFO), a framework that learns continuous-time PDE dynamics without the computational burden of standard continuous approaches, e.g., neural ODE. The key insight is repurposing flow matching to directly learn the right-hand side of PDEs without backpropagating through ODE solvers. CFO fits temporal splines to trajectory data, using finite-difference estimates of time derivatives at knots to construct probability paths whose velocities closely approximate the true PDE dynamics. A neural operator is then trained via flow matching to predict these analytic velocity fields. This approach is inherently time-resolution invariant: training accepts trajectories sampled on trajectory-specific, non-uniform time grids while inference queries solutions at any temporal resolution through ODE integration. 
Across four benchmarks (Lorenz, 1D Burgers, 2D diffusion-reaction, 2D shallow water), CFO demonstrates superior long-horizon stability and remarkable data efficiency. 
CFO trained on only 25% of irregularly subsampled time points outperforms autoregressive baselines trained on complete data, with relative error reductions up to 87%. 
Despite requiring numerical integration at inference, CFO achieves competitive efficiency, exceeding AR accuracy using only half the AR rollout step budget, while uniquely enabling reverse-time inference and arbitrary temporal querying. Code is available at https://github.com/shannon-hou/CFO_official

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces the ​​Continuous Flow Operator (CFO)​​, a novel framework for learning continuous-time PDE dynamics by repurposing flow matching to directly approximate the right-hand side of PDEs. This approach avoids the computational burden of backpropagating through ODE solvers (as in Neural ODEs) while enabling training on irregular time grids and inference at arbitrary temporal resolutions. Evaluated on four benchmarks (Lorenz, 1D Burgers, 2D diffusion-reaction, 2D shallow water), CFO demonstrates exceptional data efficiency and long-horizon stability.

### Strengths
- Novel Continuous-Time Framework​​: CFO bridges flow matching (typically used for generative modeling) with PDE dynamics learning.

- Architecture Agnosticism​​: CFO is compatible with various neural operators (e.g., U-Net, FNO, DiT). Experiments show consistent performance across backbones, allowing users to choose architectures based on spatial inductive biases or hardware constraints.

### Weaknesses
- Spline Order Sensitivity​​: While quintic splines generally outperform linear ones, the optimal spline order may depend on the PDE complexity and data density. The paper does not provide guidelines for selecting spline orders adaptively.
- High-Dimensional Scaling​​: Experiments are limited to 2D spatial domains. The computational cost of spline fitting and ODE integration in higher dimensions is not discussed.
- Comparison to Modern Baselines​​: While CFO outperforms autoregressive models and Neural ODEs, comparisons to recent continuous dynamics modeling methods (such as CORAL[1], MARBLE[2]) are limited. These methods integrate Neural ODEs and neural fields, achieves better long-horizon prediction performance.  A broader baseline comparison would better contextualize its advantages.

Reference

[1] Serrano, Louis, et al. "Operator learning with neural fields: Tackling pdes on general geometries.". NeurIPS 2023. 

[2] Wang, Honghui, Shiji Song, and Gao Huang. "GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling." ICLR 2025.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Continuous Flow Operator (CFO), a method that learns continuous-time PDE dynamics from data. Its key innovation is using flow matching and spline interpolation to directly learn the underlying dynamics, avoiding the computational cost of methods like Neural ODEs. CFO is uniquely time-resolution invariant, meaning it can be trained on data with irregular, non-uniform time steps and make predictions at any time resolution.

### Strengths
1. This paper is well-organized and highly readable.

2. The adoption of the flow matching concept can significantly reduce the training complexity of neural networks, and may even enhance the learning accuracy in certain scenarios.

3. The author has conducted experimental analyses across multiple datasets.

### Weaknesses
1. The work lacks originality, as the idea of applying flow matching to train Neural ODEs has been previously explored in several earlier studies.

2. The experimental section lacks validation on real-world datasets, which significantly limits the persuasiveness of the claims.

3. It lacks comparison with existing SOTA baseline methods.

### Questions
1. When using finite difference approximations for gradients, estimation errors are present. Particularly in cases involving non-uniform time steps and large time intervals, direct flow matching may suffer from insufficient accuracy.

2. During inference, the authors employed the RK4 method, whose computational efficiency remains relatively low.

3. It should be noted that the neural operators referenced by the CFO are distinct from established architectures such as DeepONet or FNO, which may cause potential misunderstandings.

4. See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the Continuous Flow Operator (CFO), a framework that learns continuous-time PDE dynamics by flow-matching the analytic time-derivative of a spline interpolant fitted to each training trajectory. Concretely, the authors estimate temporal derivatives on possibly irregular grids, build a piecewise quintic Hermite spline $s(t; u)$ that matches values and derivatives at knots, and then train a neural operator $N_\theta(t, u)$ to regress $\partial_t s(t)$ without back propagating through any ODE solver during training. At inference, the learned vector field $du/dt=N_\theta(t, u)$ is integrated forward or backward for arbitrary time resolutions. This yields time-resolution invariance in training and querying, competitive accuracy-efficiency trade-offs via solver/NFE sweeps, and notable data efficiency: training with only 25 % irregular time points still beats strong autoregressive baselines trained at full resolution across Lorenz, 1D Burgers, 2D diffusion-reaction, and 2D shallow water.

### Strengths
From an originality perspective, the idea of repurposing flow matching, which is originally developed for CNFs/diffusion, to directly learn RHS dynamics is elegant and avoids the heavy simulation-through-solvers that slows Neural ODE/SDE training. The positioning is credible relative to Flow Matching in generative modeling and stochastic interpolants, as CFO leverages fixed probability paths defined by the spline while keeping the operator-learning focus. I appreciate that the authors discuss flow matching lineage and deliver a practical instance specialized to temporal PDEs.

The time-resolution invariance protocol with per-trajectory random time grids is convincing, and the table shows that quintic-CFO trained on 25% of points outperforms AR trained on 100%, with large relative error reductions. The Lorenz downsampling study that compares linear vs. quintic spline velocities and observes first- vs. second-order endpoint behavior is technically sound and matches finite-difference theory. The NFE sweeps across Euler/Heun provide a clear view of accuracy vs. compute. 

CFO serves as a bridge between discrete AR operators and fully continuous Neural ODE training. It inherits irregular-time training, continuous-time querying, and even reverse-time rollouts without the pain of adjoint backprop; in my opinion this is a practical step forward for scientific ML systems that see irregular telemetry or experiment logs. The method also appears architecture-agnostic (FNO / DiT backbones), which increases adoptability in operator-learning pipelines.

### Weaknesses
The reverse-time inference claim is attractive, but dissipative PDEs often make backward integration ill-posed. I would prefer a short quantitative check showing how error grows with backward horizon and perturbation size to avoid over-selling.

Baseline coverage against continuous-time sequence models could be expanded. Since CFO’s training avoids solver backprop similarly to newer flow-matching approaches for time series, a comparison to Trajectory Flow Matching (TFM) would situate the contribution more precisely; likewise Latent Neural CDEs are classic irregular-time references. 

Finally, efficiency comparisons mix fixed-step AR with variable NFE CFO. The NFE sweeps are good, but I suggest harmonizing by wall-clock on the same hardware and showing error vs. seconds to eliminate any suspicion of budget mismatch.

### Questions
1. How far does reverse-time hold before numerical blow-up on dissipative systems like Burgers/SWE, and what step control or regularization is needed in practice? A short curve of backward-horizon vs. $L^2$ error would be enough.

2. For fairness, would the authors include an AR baseline with the same spatial backbone and time embedding that CFO uses, trained on the full grid, plus Latent ODE and Neural CDE implementations adapted to the PDE setting (or at least reported on Lorenz)? This would help isolate the gain due to the training objective rather than the architecture.

3. It would be very valuable to add one harder PDE. Two natural choices are 2D Navier–Stokes (Kolmogorov flow) and Kuramoto–Sivashinsky, both available in community suites such as PDEBench. A single appendix experiment on either would strengthen generality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method to train spatial neural PDE models without relying on explicit integration-based optimization, as is common in neural ODE frameworks. Instead, temporal derivatives are supervised through stochastically interpolated trajectories derived from discrete training snapshots. The study further evaluates the method’s in-domain predictive performance within the same temporal horizon, under varying temporal sampling densities.

### Strengths
The main strengths of the paper are as follows:
- It proposes an approach that significantly reduces the computational and memory costs associated with applying directly neural ODE frameworks to PDE prediction tasks.
- It bridges flow learning and stochastic modeling by introducing a strategy to estimate temporal derivatives at unobserved time points through stochastic interpolation of training snapshots.

### Weaknesses
- Limited baselines: The paper only compares against a discrete-time autoregressive model. While neural ODE-based approaches can indeed be computationally demanding, this alone is not a sufficient reason to omit them from quantitative comparison. Including related continuous-time baselines such as the mentioned continuous-time methods (e.g., Chen et al., 2018 for ODE/PDEs; Yin et al., 2023 for PDEs) or other comparable approaches would strengthen the experimental validation.  
- Unclear training details for autoregressive models: The paper does not specify how the autoregressive baselines are trained. Please provide more details about their training procedures (e.g., whether the model is rolled out for several time steps and supervised over the full trajectory).  
- Different training strategies in practice similar to the proposed method: Common strategies such as teacher forcing (supervising one-step-ahead predictions) or temporal curriculum scheduling (progressively extending prediction horizons) can simplify training for both autoregressive and neural ODE models while reducing computational cost. These strategies can be viewed as alternative ways to achieve interpolation. A discussion or comparison of how these techniques relate to the proposed approach would provide valuable insight.  
- Limited evaluation scope: The experiments focus solely on in-domain prediction. To better assess the model’s ability to learn underlying dynamics, the authors should evaluate extrapolation beyond the training horizon (e.g., training on data within [0, T/2) and testing on both [0, T/2) and [T/2, T)).  

References  
- Chen et al., 2018. Neural Ordinary Differential Equations. *NeurIPS 2018.*  
- Yin et al., 2023. Continuous PDE Dynamics Forecasting with Implicit Neural Representations. *ICLR 2023.*

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
