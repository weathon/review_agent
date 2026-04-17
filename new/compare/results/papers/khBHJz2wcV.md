# Physics-Constrained Fine-Tuning Of Flow- Matching Models For Generation And Inverse Problems

Jan Tauberschmidt1,2, Sophie Fellenz2, Sebastian J. Vollmer1,2**, Andrew B. Duncan**3 1DSA, German Research Center for Artificial Intelligence (DFKI)
2Department of Computer Science, University of Kaiserslautern–Landau (RPTU)
3Department of Mathematics, Imperial College London
{jan.tauberschmidt, sebastian.vollmer}@dfki.de, fellenz@cs.uni-kl.de, a.duncan@imperial.ac.uk

## Abstract

We present a framework for fine-tuning flow-matching generative models to enforce physical constraints and solve inverse problems in scientific systems. Starting from a model trained on low-fidelity or observational data, we apply a differentiable post-training procedure that minimizes weak-form residuals of governing partial differential equations (PDEs), promoting physical consistency and adherence to boundary conditions without distorting the underlying learned distribution. To infer unknown physical inputs, such as source terms, material parameters, or boundary data, we augment the generative process with a learnable latent parameter predictor and propose a joint optimization strategy. The resulting model produces physically valid field solutions alongside plausible estimates of hidden parameters, effectively addressing ill-posed inverse problems in a datadriven yet physics-aware manner. We validate our method on canonical PDE problems, demonstrating improved satisfaction of physical constraints and accurate recovery of latent coefficients. Further, we confirm cross-domain utility through fine-tuning of natural-image models. Our approach bridges generative modeling and scientific inference, opening new avenues for simulation-augmented discovery and data-efficient modeling of physical systems.

## 1 Introduction

Physical systems with rich spatio-temporal structure can be effectively represented by deep generative models, including diffusion and flow-matching methods (Kerrigan et al., 2024; Erichson et al., 2025; Baldan et al., 2025; Price et al., 2023). Although their dynamics can be highly complex, these systems are often governed by fundamental principles, such as conservation laws, symmetries, and boundary conditions, that constrain the space of admissible solutions. Incorporating such physical structure into generative modeling can improve both sample fidelity and out-of-distribution generalization. In many scientific domains, including atmospheric and oceanographic modeling, seismic inversion, and medical imaging, we often observe system states without access to the underlying physical parameters that govern them. Crucially, PDE-based constraints are typically parameter-dependent, with residuals that vary according to material properties, source terms, or other latent variables. Prior work has largely focused on simple or global constraints—such as fixed boundaries or symmetries, that apply uniformly across the data distribution. Handling parameter-dependent constraints naively would require training over the joint distribution of solutions and parameters, which is often infeasible because parametric labels are missing, expensive to obtain, or high-dimensional. Addressing this limitation is critical for scientific discovery. Many inverse problems in the natural sciences and engineering require reasoning about unobserved parameters or exploring hypothetical scenarios inaccessible to direct experimentation. A generative model that can enforce parameter-dependent PDE constraints using only observational data would provide a powerful tool for data-efficient simulation, hypothesis testing, and the discovery of new physical phenomena, helping to bridge the gap between raw observations and mechanistic understanding. This work proposes a framework for fine-tuning flow-matching generative models to enforce parameter-dependent PDE constraints without requiring joint parameter–solution training data. This work aligns with a growing trend of simulation-augmented machine learning (Karniadakis et al., 2021), where generative models accelerate scientific discovery by efficiently exploring physically plausible solution spaces. Our approach reformulates fine-tuning as a stochastic optimal control problem via Adjoint Matching (Domingo-Enrich et al., 2025), guided by weak-form PDE residuals. By augmenting the model with a latent parameter evolution, we enable joint generation of physically consistent solution–parameter pairs, addressing ill-posed inverse problems. We evaluate our proposed fine-tuning framework on four representative PDE families spanning elliptic diffusion, elasticity, wave propagation, and incompressible flow and show an application to natural images. We demonstrate denoising and conditional generation capabilities, including robustness to noisy data and the ability to infer latent parameters from sparse observations. Visual and quantitative results, including strong reductions in residuals across tasks and robustness to model misspecification, highlight the flexibility of our method for integrating physical constraints into generative modeling. To sum up, our contributions are as follows:
- POST-TRAINING ENFORCEMENT OF PHYSICAL CONSTRAINTS: We introduce a finetuning strategy that tilts the generative distribution toward PDE-consistent samples using weak-form residuals, improving physical validity while preserving diversity.

- ADJOINT-MATCHING FINE-TUNING WITH THEORETICAL GROUNDING: Leveraging the adjoint-matching framework, we recast reward-based fine-tuning as a stochastic control problem, extending flow-matching models to generate latent parameters alongside states, enabling inverse problem inference without paired training data.

- BRIDGING GENERATIVE MODELING AND PHYSICS-INFORMED LEARNING: Our approach connects preference-aligned generation with physics-based inference, enabling simulation-augmented models to generate solutions that respect complex physical laws.

An implementation of our method is available at https://github.com/ jantauberschmidt/PCFT.

## 2 Related Work

Physics-Constrained Generative Models Integrating physical constraints—such as boundary conditions, symmetry invariances, and partial differential equation (PDE) constraints—into machine learning models improves both accuracy and out-of-distribution generalization. Classical approaches, such as Physics-Informed Neural Networks (PINNs, Raissi et al. (2019)), directly regress solutions that satisfy governing equations. While effective for forward or inverse problems, PINNs do not capture distributions over solutions, making them unsuitable for generative tasks that require sampling diverse plausible outcomes. In the generative setting, the main challenge is ensuring that the physically constrained samples retain the variability of the underlying generative model, avoiding pathological issues such as mode collapse. Bastek et al. (2024) proposes a unified framework for introducing physical constraints into Denoising Diffusion Probabilistic Models (DDPMs, Ho et al. (2020)) at pre-training time, by adding a first-principles physics-residual loss to the diffusion training objective. This loss penalizes violations of governing PDEs (e.g. fluid dynamics equations) so that generated samples inherently satisfy physical laws. The method was empirically shown to reduce residual errors for individual samples significantly, while simultaneously acting as a regularizer against overfitting, thereby improving generalization. To evaluate the physics-residual loss, one needs to compute the expected PDE residual of the final denoised sample conditioned on the current noisy state in the DDPM process. Accurately estimating this expectation requires generating multiple reverse-diffusion trajectories from the same noisy sample, which makes pre-training significantly more expensive. A common alternative is to use Tweedie's formula to approximate the conditional expectation in a single pass, but this shortcut introduces bias, particularly in the final denoising steps.

![2_image_0.png](2_image_0.png)

Zhang & Zou (2025) proposes enforcing constraints through a post-hoc distillation stage, where a deterministic student model is trained from a vanilla diffusion model to generate samples in onestep, regularized by a PDE residual loss. In Wang et al. (2025) the authors introduce PhyDA, diffusion-based data assimilation framework that ensures reconstructions obey PDE-based dynamics, specifically for atmospheric science. An autoencoder is used to encode sparse observations into a structured latent prior for the diffusion model, which is trained with an additional physical residual loss.

Inference- and Post-Training Constraint Enforcement Various works have proposed approaches to enforce PDE constraints at inference time, often in combination with observational constraints, drawing connections to conditional diffusion models (Dhariwal & Nichol, 2021; Ho & Salimans, 2021). Huang et al. (2024) introduce guidance terms within the denoising update of a score-based diffusion model to steer the denoising process towards solutions which are both consistent with data and underlying PDEs. A related approach was considered by Xu et al. (2025), further introducing an adaptive constraint to mitigate instabilities in early diffusion steps. In Christopher et al. (2024), the authors recast the inference-time sampling of a diffusion process as a constrained optimization problem, each diffusion step is projected to satisfy user-defined constraints or physical principles. This allows strict enforcement of hard constraints (including convex and non-convex constraints, as well as ODE-based physical laws) on the generated data. Lu & Xu (2024) consider the setting where the base diffusion model is trained on cheap, low-fidelity simulations, leveraging a similar approach to generate down-scaled samples via projection. Flow-Matching Models for Simulation and Inverse Problems Flow-matching (FM, Lipman et al. (2023)) has emerged as a flexible generative modeling paradigm for complex physical systems across science, including molecular systems (Hassan et al., 2024), weather (Price et al., 2023) and geology (Zhang et al., 2025) . In the context of physics-constrained generative models Utkarsh et al. (2025) introduces a zero-shot inference framework to enforce hard physical constraints in pre-trained flow models, by repeatedly projecting the generative flow at sampling time. Similarly, Cheng et al. (2024) proposed the ECI algorithm, to adapt a pre-trained flow-matching model so that it exactly satisfies constraints without using analytical gradients. In each iteration of flow sampling, ECI performs: an Extrapolation step (advancing along the learned flow), a Correction step (applying a constraint-enforcement operation), and an Interpolation step (adjusting back towards the model's trajectory). While projection approaches are a compelling strategy for hard constraints, they can be challenging particularly for local constraints such as boundary conditions, as direct enforcement can introduce discontinuities. The above approach mitigates this by interleaving projections with flow steps, however this relies on the flow's ability to rapidly correct such non-physical artifacts. Baldan et al. (2025) propose Physics-Based Flow Matching (PBFM), which embeds constraints (PDE or symmetries) directly into the FM loss during training. The approach leverages temporal unrolling to refine noise-free final state predictions and jointly minimizes generative and physicsbased losses without manual hyperparameter tuning of their tradeoff. To mitigate conflicts between physical constraints and the data loss, they employ the ConFIG (Liu et al., 2024), which combines the gradients of both losses in a way that ensures that gradient updates always minimize both losses simultaneously. Related to our approach are the works on generative models for Bayesian inverse problems (Stuart, 2010), where the goal is to infer distributions over latent PDE parameters given partial or noisy observations. Conditional diffusion and flow-matching models can be used to generate samples from conditional distributions and posterior distributions, supporting amortized inference and uncertainty quantification (Song et al., 2021; Utkarsh et al., 2025; Zhang et al., 2023). Conditioning is typically achieved either through explicit parameter inputs or guidance mechanisms during sampling, as in classifier-guided diffusion. While effective when large volumes of paired training data is available, these approaches are less relevant to observational settings where parameters are unobserved. In contrast, our approach connects the observed data to the latent parameters only during post-training, requiring substantially smaller volumes of data.

## 3 Method

FM models are trained to learn and sample from a given distribution of data pdata. They approximate this distribution by constructing a Markovian transformation from noise to data, such that the time marginals of this transformation match those of a *reference flow* Xt = βtX1 + γtX0. Specifically FM models learn a vector field vt(x) that transports noise to data, via the ODE dXt = vt(Xt) dt.

We can optionally inject a noise schedule σ(t) along the trajectory to define an equivalent SDE that preserves the same time marginals (Maoutsa et al., 2020),

$$dX_{t}=\left(v_{t}(X_{t})+\frac{\sigma(t)^{2}}{2\eta_{t}}\left(v_{t}(X_{t})-\frac{\dot{\beta}_{t}}{\beta_{t}}X_{t}\right)\right)\,dt+\sigma(t)\,dB_{t}\ =:\,b_{t}(X_{t})\,dt+\sigma(t)\,dB_{t},\tag{1}$$

where we combine coefficients βt and γt into ηt = γtβ˙t β

$$i\gamma_{t}-\dot{\gamma}_{t}).$$

Assuming we have access to a FM model which generates samples according to distribution p(x), we seek to adjust this model so as to generate samples from the tilted distribution pr(x) ∝ e λ r(x)p(x),
where r is a reward function and λ characterizes the degree of distribution shift induced by finetuning. To achieve this, we leverage the adjoint-matching framework of Domingo-Enrich et al. (2025). This work reformulates reward fine-tuning for flow-based generative models as a control problem in which the base generative process given by v base tis steered toward high-reward samples via modifying the learned vector field, which we denote as v ft t with corresponding drift term b ft t. Our approach is conceptually related to reward- or preference-based fine-tuning of generative models (Christiano et al., 2017; Sun et al., 2024), where a learned or computed reward steers generation toward desired properties. Here, the reward is defined via PDE residuals, encoding knowledge about underlying dynamics and physical constraints to the solutions space as deviations to differential operators or boundary conditions. Notably, we assume that the distribution generated by the base model p(x) only captures an observed quantity, but does not provide us with corresponding parameters or coefficient fields often needed to evaluate the respective differential operator. In the following, we will present a strategy of jointly recovering unknown parameters and fine-tuning the generation process.

## 3.1 Reward

A generative model can reproduce the visual characteristics of empirical data while ignoring the physics that governs it, thereby rendering the samples unusable for downstream scientific tasks. To bridge this gap we impose the known governing equations as *soft constraints*, expressed through differential operators Lαx = 0 with parameters α. Throughout, a generated sample x is interpreted as a discretization of a continuous field x(ξ) on a domain Ω. The *strong* PDE residual is defined as

$${\mathcal{R}}_{\mathrm{strong}}(x,\alpha)=\left\|{\mathcal{L}}_{\alpha}x\right\|_{L^{2}(\Omega)}^{2}.$$

In practice, strong residuals involve high-order derivatives that make the optimization landscape unstable. We therefore adopt *weak-form residuals* of the form ⟨Lα*x, ψ*⟩L2(Ω) for suitably chosen test functions ψ ∈ Ψ, which are numerically more stable under noisy or misspecified data. Repeated applications of integration-by-parts can transfer derivatives from x to ψ. The set Ψ is composed of compactly supported local polynomial kernels. For each evaluation we draw Ntest such functions; their centers and length-scales are sampled at random. A mollifier enforces ψ|∂Ω = 0, justifying the integration by parts. The resulting residual is

$${\mathcal{R}}_{\mathrm{weak}}(x,\alpha)={\frac{1}{N_{\mathrm{test}}}}\sum_{i=1}^{N_{\mathrm{test}}}\bigl\vert\langle{\mathcal{L}}_{\alpha}x,\psi^{(i)}\rangle_{L^{2}(\Omega)}\bigr\vert^{2}.$$

These randomly sampled local test functions act as stochastic probes of PDE violations, providing a low-variance, data-efficient learning signal. A more detailed description of the test functions used can be found in Appendix D.3. Note that the residual might be augmented by adding soft constraints for boundary conditions.

## 3.2 Joint Evolution

Fine-tuning is nontrivial in our setting because we must infer latent physical parameters jointly with the generated solutions. On fully denoised samples, we can train an inverse predictor, i.e.,
φ(x1) = α1, such that the weak PDE residual is minimized. As a na¨ıve approach, this already induces a joint distribution over (x1, α1) via the push-forward through φ. However, we advocate a
more principled formulation that evolves *both* x and α along vector fields, enabling joint sampling of parameters and solutions, as well as a controlled regularization of fine-tuning through the Adjoint Matching framework as outlined below. In the fine-tuning model, this can be achieved by directly learning the vector field v
ft
t,α jointly with v
ft
t,x by augmenting the neural architecture. Since no
ground-truth flow of α for the base model is available, at each state (xt, αt) we define a *surrogate*
base flow using the inverse predictor φ. Specifically, we consider the one-step estimates
$${\hat{x}}_{1}\ =\ x_{t}+(1-t)\,v_{t}^{\mathrm{base}}(x_{t}),\qquad{\hat{\alpha}}_{1}\ =\ \varphi({\hat{x}}_{1}).$$

## The Direction From The Current State Αt To The Predicted Final Parameter Αˆ1 Serves As A Base Vector
Field Which We Use To Evolve Α, I.E. V
Base
T, Α (Αt) = (ˆΑ1 −Αt)/(1−T) Inducing Corresponding Drift B
Base
T,Α .
This *Surrogate Base Flow*, Starting At A Noise Sample Α
Base
0 ∼ N (0, I), Emulates A Denoising Process
Of The Recovered Parameter. We Denote By Α
Base The Parameter Aligned With The Base Trajectory
X
Base. While The Evolution Of Α
Base Does Not Influence The Trajectory Of X
Base, The Inferred Vector
Field Can Be Used To Effectively Regularize The Generation Of The Fine-Tuned Model. Similarly, To Regularize Towards The Parameter Recovered Under The Base Model, We Introduce An Additional Field V
Reg
T, Α(Α
Ft
T) = (ˆΑ
Base
1 − Α
Ft
T)/(1 − T). This Vector Field Points From The Current Parameter Estimate Of
The Fine-Tuned Trajectory Α
Ft
Tto The Recovered Parameter Under The Base Model Αˆ
Base
1. The Field Is Used
To Pull The Fine-Tuned Dynamics Towards Final Samples Associated With Parameters Similar To Those Of The Base Trajectory. The Introduced Vector Fields Are Visualized In Fig. 1. 3.3 Adjoint Matching

Considering an augmented state variable of the joint evolution X˜t = (XT
t, αT
t)
T, we cast fine-tuning as a stochastic optimal control problem:

$$\min_{\tilde{u}}\mathbb{E}\left[\int_{0}^{1}\left(\frac{1}{2}\left\|\tilde{u}_{t}(\tilde{X}_{t})\right\|^{2}+f(\tilde{X}_{t})\right)dt+g(\tilde{X}_{1})\right]\tag{2}$$  s.t. $d\tilde{X}_{t}=\left(\tilde{b}_{t}^{\text{base}}(\tilde{X}_{t})+\sigma(t)\,\tilde{u}_{t}(\tilde{X}_{t})\right)dt+\sigma(t)\,d\tilde{B}_{t}$

with control u˜t(X˜t), running state cost f(X˜t), and terminal cost g(X˜1). In this formulation, finetuning amounts to a point-wise modification of the base drift through application of control u˜, i.e.

$$\tilde{b}_{t}^{\mathrm{ft}}(\tilde{X}_{t})=\tilde{b}_{t}^{\mathrm{base}}(\tilde{X}_{t})+\sigma(t)\,\tilde{u}_{t}(\tilde{X}_{t}).$$
$\pi$. 
In Domingo-Enrich et al. (2025), Adjoint Matching is introduced as a technique with lower variance and computational cost than standard adjoint methods. The method is based on a *Lean Adjoint* state, which is initialized as

$$\tilde{a}_{1}^{T}=\tilde{\lambda}\nabla_{\tilde{x}}\,g(\tilde{X}_{1})=\left(\lambda_{x}\nabla_{x}\,g(X_{1},\alpha_{1}),\,\lambda_{\alpha}\nabla_{\alpha}\,g(X_{1},\alpha_{1})\right)$$

and evolves backward in time according to

$$\frac{d}{dt}\hat{a}_{t}=-\left(\nabla_{\hat{x}}\hat{b}_{t}^{\text{base}}(\tilde{X}_{t})^{T}\,\hat{a}_{t}+\nabla_{\hat{x}}f(\tilde{X}_{t})^{T}\right)=-\left(\begin{matrix}J_{xx}^{T}&J_{xx}^{T}\\ J_{xx}^{T}&J_{xx}^{T}\end{matrix}\right)\left(\begin{matrix}a_{t,x}\\ a_{t,\alpha}\end{matrix}\right)-\left(\begin{matrix}\nabla_{x}f(X_{t},\alpha_{t})^{T}\\ \nabla_{\alpha}f(X_{t},\alpha_{t})^{T}\end{matrix}\right)\tag{3}$$  where the block-Jacobian is evaluated along the base drift for $X$ and $\alpha$, which means that 
Jij = ∇j b base t,i (Xt, αt) for i, j ∈ {*x, α*}. The hyperparameters λx and λα can be used to regulate the extent to which the fine-tuned distribution departs from the base distribution. The Adjoint Matching objective can then be formulated as a consistency loss:

$$\mathcal{L}(\tilde{u};\tilde{X})=\tfrac{1}{2}\!\int_{0}^{1}\left\|\tilde{u}_{t}(\tilde{X}_{t})+\sigma(t)\,\tilde{u}_{t}\right\|^{2}dt$$ $$=\tfrac{1}{2}\!\int_{0}^{1}\left(\left\|u_{,x}(X_{t},\alpha_{t})+\sigma(t)\,a_{t,x}\right\|^{2}+\left\|u_{t,\alpha}(X_{t},\alpha_{t})+\sigma(t)\,a_{t,\alpha}\right\|^{2}\right)dt.$$
$$\mathbf{\Sigma}(4)$$

It can be shown (Domingo-Enrich et al., 2025) that with f = 0, this objective is consistent with the tilted target distribution for reward r = −g, if optimized with a *memoryless* noise schedule. This
schedule ensures sufficient mixing during generation such that the final sample X1 is independent of X0. To stabilize fine-tuning we introduce a scaled variant of the memoryless noise schedule. Instead
of using the canonical choice σ
2(t*) = 2*ηt identified by Domingo-Enrich et al. (2025), we adopt
$$\sigma^{2}(t)=\left(1-\kappa\right)2\eta_{t},\qquad0\leq\kappa<1,$$
which retains the theoretical memoryless property (see Lemma 1 in Appendix D.4) while attenuating the magnitude of the noise variance. The introduction of the scaling factor 0 ≤ κ < 1 constitutes
a simple but novel extension of the adjoint-matching framework. Whereas prior work highlighted a unique schedule, our analysis shows that a family of scaled schedules remains consistent with the memoryless condition. This additional degree of freedom acts as a *numerical stabilisation knob*, mitigating blow-ups near t → 0 without losing theoretical consistency. Further, it offers a control– fidelity trade-off by regulating the amount of exploration. In practice, this flexibility allows practitioners to adapt fine-tuning to the conditioning of the PDE residuals and the stability of the solver, a feature not available in the original formulation. Equation 2 is optimized by iteratively sampling trajectories with the fine-tuned model while following a memoryless noise schedule, numerically computing the lean adjoint states by solving the ODE in Equation 3, and taking a gradient descent step to minimize the loss in Equation 4. Note that gradients are only computed through the control u˜t and not through the adjoint, reducing the optimization
target to a simple regression loss. We state the full training algorithm and implementation details in Appendix D.5. Adjoint Matching steers the generator toward the reward-tilted distribution, thereby reshaping the entire output distribution rather than correcting individual trajectories. However, when fine-tuning observational data or under system misspecification, we might be interested in retaining samplespecific detail. Empirically we find that this can be effectively encoded by imposing similarity of the inferred coefficients between base and fine-tuned model. Therefore, we add a running state cost
$$f(\alpha)=\lambda_{f}\,\left\|v_{t,\,\alpha}^{\mathrm{ft}}(\alpha)-v_{t,\,\alpha}^{\mathrm{reg}}(\alpha)\right\|^{2}$$
which penalizes deviations of the fine-tuned α-drift from the direction pointing toward the base estimate αˆ
base 1. The hyper-parameter λf controls a smooth trade-off: λf = 0 recovers pure Adjoint Matching, while larger λf progressively anchors the final parameters α1 obtained under the finetuned model to their base-model counterparts, thus retaining trajectory-level detail.

## 4 Experiments

We evaluate across five settings: four PDE systems (including boundary and system misspecification, and observational noise) and a natural-image model. Unlike latent-space fine-tuning for images, our PDE models operate directly in pixel space. High-variance noise during sampling can drive off-manifold trajectories and perturb PDE residuals, motivating κ > 0 for these models. For base Flow Matching backbones, we use U-FNO (Wen et al., 2022) for PDEs and the DiT-based latent FM of Dao et al. (2023) for images. In all experiments we first sample from the base generator and pre-train the inverse predictor φ to recover α by minimizing the (PDE) residual, then finetune. Following Domingo-Enrich et al. (2025), fine-tuning is initialized from the base weights. We augment capacity to condition v ft t,x on αt and add a separate head for v ft t,α. Fine-tuning uses a memoryless noise schedule, while all reported results are generated without injected noise (σ(t) = 0). Implementation details appear in App. D.2. Comparisons, ablations, and metrics. Our proposed method converts a single-variable flow into a joint generative model (Sec. 3.2). We compare against: (i) a *Base AM* variant (vanilla Adjoint Matching) where φ is frozen and used only to compute residuals, (ii) a *Base AM+*φ variant where φ continues to train but the flow over α is not modeled jointly, and (iii) *PBFM* (Baldan et al., 2025), augmented with our pre-trained φ to enable residual evaluation. Details on the comparison methods can be found in App. E.2. All evaluations use 256 samples, generated from shared seeds across methods. We report weak and strong residuals, Rweak and Rstrong, scaled by the mean residual of a fixed reference set. The reference set Dref is a synthetic, clean dataset generated under the target PDE specification assumed during fine-tuning (no noise, modified BCs, lossless Helmholtz, or unforced Stokes respectively). We also report Maximum Mean Discrepancy (MMD) based distributional similarities for states and parameters (MMDx*, MMD*α) computed against this dataset (details in App. E.1). While the main text shows representative results, the complete set of experimental evaluations is provided in App. F.

## 4.1 Darcy Flow

Consider a square domain Ω = [0, 1]2 where a permeability α(ξ) and forcing f(ξ) induce a pressure field x(ξ) governed by *−∇ ·* (α(ξ)∇x(ξ)) − f(ξ) = 0 with zero Dirichlet boundary conditions and constant f. We draw α from a discretized Gaussian process and corrupt pressures with observation noise before training the base FM. Dataset details are in App. B. Figure 2 compares three Darcy samples generated from the *same* noise seed x0: the base draw, fine-tuning with our regularization (here λf = 1.0), and fine-tuning without regularization. The base pressure x base is visibly contaminated by high-frequency noise, and the inverse predictor φ correspondingly yields a scattered, artifact-ridden permeability map α base. With regularization enabled, fine-tuning attenuates noise in the pressure x ft while remaining close to α base. Because α base is itself fragmented, some artifacts persist. In contrast, disabling regularization produces a fully denoised pressure and a markedly more coherent α ft, but at the expected expense of erasing sample-specific details present in the base realization.

![6_image_0.png](6_image_0.png)

We quantify the controllable trade-offs in Fig. 3. Panel (a) increases λx = λα at λf = 0, which reduces the PDE residual while also reducing diversity in the inferred permeabilities (measured via the complement of the mean pairwise SSIM; see App. E.1). Panel (b) fixes λx = λα = 20K and varies λf , reporting MMDx between the fine-tuned samples and the base dataset. As expected, stronger regularization preserves distributional fidelity (lower MMD) but yields higher residuals.

![7_image_0.png](7_image_0.png)

These ablations illustrate how practitioners can target residual reduction or distributional fidelity by tuning (λx, λα, λf ).

Computationally, adaptation is lightweight: fine-tuning on noisy Darcy requires only 20 gradient steps (hyperparameters in App. E.3) and completes in under 15 minutes on a single NVIDIA L40S,
after which sampling proceeds at base-model cost with no inference-time adjustments.

## 4.2 Guidance On Sparse Observations

In many realistic settings dense observations of a state variable are available for pre-training a generative model, whereas only a few measurements of the latent parameter can be collected. To sample from the posterior of parameter–state pairs that respect such sparse evidence we steer the generative process through *guidance*. Huang et al. (2024) demonstrate guided sampling towards sparse observations from a model that was pre-trained on the joint parameter-state distribution. Our approach applies a similar guidance mechanism, however, to a model that was pre-trained on noisy state observations alone. We state details on the guiding mechanism in E.4. Figure 4 shows that the guided sampler adheres to sparse measurements while preserving realistic variability in the generated samples. Additional results for different amounts of conditioning observations are given in Appendix F.3.5.

![7_image_1.png](7_image_1.png)

## 4.3 Linear Elasticity

We consider plane–strain linear elasticity on Ω = [0, 1]2 with spatially varying Young's modulus α(ξ) and fixed Poisson ratio. Boundaries are Dirichlet: left/right clamped, top/bottom receive inward sinusoidal *normal* displacements with zero tangential slip. During fine-tuning, we impose a modified lower-boundary amplitude to induce controlled misspecification (see App. B) and include an MSE boundary penalty in the weak residual. We report quantitative BC results as the MSE at the boundary in Table 4.3 and a qualitative comparison is provided in App. F. Our method attains low weak/strong residuals while keeping distributional shift modest; PBFM and FM+ECI drift distributionally or present high residuals (full details and non-curated samples in App. E.5, F.3.2).

| Model   | BC error (MSE) ↓     | Rweak (rel) ↓       | Rstrong (rel) ↓     | MMDx ↓   | MMDα ↓   |
|---------|----------------------|---------------------|---------------------|----------|----------|
| FM      | 6.98 × 10−5 (± 0.53) | 1.59 × 101 (± 0.37) | 1.83 × 101 (± 0.66) | 0.24     | 0.05     |
| PBFM    | 2.32 × 10−5 (± 0.87) | 6.32 × 100 (± 0.82) | 4.22 × 100 (± 0.26) | 0.92     | 0.54     |
| FM+ECI  | 0.0                  | 1.01 × 103 (± 0.13) | 2.49 × 102 (± 0.32) | 1.16     | 0.36     |
| Ours    | 1.71 × 10−6 (± 0.50) | 6.15 × 100 (± 0.77) | 3.79 × 100 (± 0.87) | 0.15     | 0.12     |

Table 1: Linear elasticity under BC misspecification. Quantitative boundary-condition (BC) error, relative weak/strong residuals, and distributional metrics. Our method achieves low residuals with limited distributional shift.

## 4.4 Helmholtz

We consider time–harmonic wave propagation governed by the heterogeneous Helmholtz equation
−∆u − (1 − itan δ) κ(x)
2u = s on Ω = [0, 1]2 with Robin boundary conditions. Training data use a small damping term (tan δ > 0) producing complex attenuated fields, while fine-tuning assumes the idealized lossless model (tan δ = 0), inducing a controlled model mismatch.

| Model       | Criterion   | Rweak (rel) ↓       | Rstrong (rel) ↓     | MMDx ↓   | MMDα ↓   |
|-------------|-------------|---------------------|---------------------|----------|----------|
| FM          | -           | 1.5 × 101 (± 0.59)  | 2.55 × 101 (± 0.55) | 0.18     | 0.03     |
| PBFM        | -           | 8.33 × 100 (± 3.04) | 1.22 × 101 (± 0.33) | 0.09     | 0.03     |
| Base AM     | Rweak       | 4.9 × 100 (± 1.85)  | 1.34 × 101 (± 0.32) | 0.15     | 0.04     |
| Base AM     | MMDx        | 5.64 × 100 (± 2.09) | 1.59 × 101 (± 0.33) | 0.13     | 0.04     |
| Base AM + φ | Rweak       | 4.99 × 100 (± 2.12) | 1.16 × 101 (± 0.33) | 0.13     | 0.05     |
| Base AM + φ | MMDx        | 5.46 × 100 (± 1.94) | 1.59 × 101 (± 0.33) | 0.12     | 0.04     |
| AM          | Rweak       | 4.3 × 100 (± 1.29)  | 1.14 × 101 (± 0.29) | 0.07     | 0.04     |
| AM          | MMDx        | 4.32 × 100 (± 1.43) | 1.05 × 101 (± 0.30) | 0.06     | 0.04     |

Table 2: Helmholtz: residuals and distribution (representative configs). Normalized weak/strong residuals and MMD metrics. We include AM variants and a PBFM-style baseline. Table 2 reports representative configurations for each method, selected as either the setting with the lowest weak residual (Rweak) or the lowest MMDx. Full results are provided in App. F. The base FM model shows the largest weak and strong residuals due to the damped–vs.–lossless mismatch.

PBFM substantially reduces both residuals relative to FM and, notably, also lowers MMDx and preserves MMDα. The AM ablations (Base AM and Base AM+φ) further reduce the weak residuals into the range 4.9× 100–5.6× 100, with strong residuals similar to those of PBFM, but they incur a moderate increase in MMDx and MMDα compared to PBFM. Our full joint AM model achieves the lowest residuals overall (weak residuals down to 4.3×100and strong residuals near 1.05×101) while simultaneously attaining the lowest MMDx among all methods and maintaining MMDα comparable to the ablations. This indicates that the joint flow most effectively resolves the misspecification while preserving distributional fidelity.

## 4.5 Stokes Lid-Driven Cavity

We consider steady incompressible flow in the Stokes regime (linear, low–Reynolds-number proxy)
governed by −∇· (ν(x)∇u) + ∇p = f, ∇· u = 0 with no-slip walls and a smooth moving lid. The dataset uses nonzero Kolmogorov forcing f ̸= 0, while fine-tuning assumes f ≡ 0, creating a systematic model mismatch. Figure 5 reports the residual–distribution trade-offs for the Stokes lid-driven cavity. We show only the Base AM variants and our joint model: the base FM model exhibits extremely large residuals (3.05 × 102 ± 3.16) and is omitted for clarity, while PBFM fails to converge to meaningful velocity–pressure fields (strong residuals 1.15 × 101 ± 0.05; see App. F).

In contrast to Darcy and Helmholtz, the attainable weak residuals of all remaining variants are similar (Rweak ≈4–15). However, the joint model reaches *substantially lower* parameter-distribution discrepancies, achieving MMDα ≈ 0.07–0.13, whereas both ablations remain around 0.22–0.28.

Overall, although residual levels are similar across AM variants, only the joint model can enter the low–MMD regime—particularly for MMDα. This highlights the joint flow's greater flexibility in achieving high-fidelity parameter distributions.

![9_image_0.png](9_image_0.png)

## 4.6 Natural Images: Parametric Color Transformation

To demonstrate cross-domain utility, we apply our method to natural images by introducing a parametric recoloring pathway: analogous to the hidden PDE parameter, α here specifies a polynomial color transform that operates outside the latent space, enabling exploration of image appearances not well supported by the base distribution. We use a class-conditional Latent Flow Matching (LFM) model (Dao et al., 2023) pre-trained on ImageNet-1k (Deng et al., 2009) and optimize PickScore
(Kirstain et al., 2023) with a globally fixed prompt. As a concrete example, we fine-tune on the class macaw with the prompt "close-up Pop Art of a macaw parrot," yielding the samples in Fig. 6. Joint fine-tuning with recoloring produces markedly more vibrant palettes and, crucially, *joint* adjustments (e.g., background textures that the recoloring exploits). Details about the recoloring parametrization are given in Appendix E.7 and further non-curated samples are provided in Appendix F.3.6.

![9_image_1.png](9_image_1.png)

## 5 Conclusion

We have introduced a framework for post-training fine-tuning of flow-matching generative models to enforce physical constraints and jointly infer latent physical parameters informing the constraints. Through a novel architecture, combined with the combination of weak-form PDE residuals with an adjoint-matching scheme our method can produce samples that adhere to complex constraints without significantly affecting the sample diversity. Experiments across PDE problems demonstrate the potential of this method to reduce residuals and enable joint solution–parameter generation, supporting its promise for physics-aware generative modeling. Future steps include adaptive approaches to optimizing trade-off between constraint enforcement and generative diversity, and extending the framework to more complex and multi-physics systems, including coupled PDEs and stochastic or chaotic dynamics. We would also explore how this methodology can be leveraged for uncertainty quantification and propagation, and downstream tasks such as optimal sensor placement and scientific discovery workflows.

## Reproducibility Statement

We report datasets, model backbones, training schedules, loss definitions, evaluation metrics, and the key hyperparameters required to reproduce our results in the main text and appendix. Remaining implementation choices are documented in the released configuration files. We fixed random seeds where applicable and specify hardware/software versions.

## Acknowledgements

This work was supported by funding from the German Federal Ministry for Education and Research (Bundesministerium fur Bildung und Forschung, BMBF) under grants 16IS24071A / 16IS24071B ¨ and 01IW23005. SF additionally acknowledges support by the DFG through FOR 5359 (ID 459419731), TRR 375 (ID 511263698), and SPP 2331 (441958259, 553345933, 466468799), and by the Carl-Zeiss Foundation through the initiatives AI-Care and Process Engineering 4.0. We thank the anonymous reviewers for their comments.

## References

Martin S. Alnaes, Anders Logg, Kristian B. Ølgaard, Marie E. Rognes, and Garth N. Wells. Unified form language: A domain-specific language for weak formulations of partial differential equations. *ACM Transactions on Mathematical Software*, 40, 2014. doi: 10.1145/2566630.

Giacomo Baldan, Qiang Liu, Alberto Guardone, and Nils Thuerey. Flow matching meets pdes: A
unified framework for physics-constrained generation. *arXiv preprint arXiv:2506.08604*, 2025.

Igor A. Baratta, Joseph P. Dean, Jørgen S. Dokken, Michal Habera, Jack S. Hale, Chris N. Richardson, Marie E. Rognes, Matthew W. Scroggs, Nathan Sime, and Garth N. Wells. DOLFINx: the next generation FEniCS problem solving environment. preprint, 2023.

Jan-Hendrik Bastek, WaiChing Sun, and Dennis Kochmann. Physics-informed diffusion models. In The Thirteenth International Conference on Learning Representations, 2024.

Yoav Chai, Raja Giryes, and Lior Wolf. Supervised and unsupervised learning of parameterized color enhancement. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 992–1000, 2020.

Chaoran Cheng, Boran Han, Danielle C Maddix, Abdul Fatir Ansari, Andrew Stuart, Michael W
Mahoney, and Yuyang Wang. Gradient-free generation for hard-constrained systems. arXiv preprint arXiv:2412.01786, 2024.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In *Proceedings of the IEEE/CVF conference on computer* vision and pattern recognition, pp. 2818–2829, 2023.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Jacob K Christopher, Stephen Baek, and Nando Fioretto. Constrained synthesis with projected diffusion models. *Advances in Neural Information Processing Systems*, 37:89307–89333, 2024.

Fabio Crameri, Grace Shephard, and Philip Heron. The misuse of colour in science communication.

Nature Communications, 11, 2020. doi: 10.1038/s41467-020-19160-7.

Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran. Flow matching in latent space. arXiv preprint arXiv:2307.08698, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint matching:
Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Representation Learning, 2025.

N Benjamin Erichson, Vinicius Mikuni, Dongwei Lyu, Yang Gao, Omri Azencot, Soon Hoe Lim, and Michael W Mahoney. Flex: A backbone for diffusion-based modeling of spatio-temporal physical systems. *arXiv preprint arXiv:2505.17351*, 2025.

Damien Garreau, Wittawat Jitkrittum, and Motonobu Kanagawa. Large sample analysis of the median heuristic. *arXiv preprint arXiv:1707.07269*, 2017.

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. ¨
A kernel two-sample test. *The journal of machine learning research*, 13(1):723–773, 2012.

Majdi Hassan, Nikhil Shenoy, Jungyoon Lee, Hannes Stark, Stephan Thaler, and Dominique Beaini. ¨
Et-flow: Equivariant flow-matching for molecular conformer generation. Advances in Neural Information Processing Systems, 37:128798–128824, 2024.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In *NeurIPS 2021 Workshop on* Deep Generative Models and Downstream Applications, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park. Diffusionpde: Generative pdesolving under partial observation. *Advances in Neural Information Processing Systems*, 37: 130291–130323, 2024.

George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang.

Physics-informed machine learning. *Nature Reviews Physics*, 3(6):422–440, 2021.

Gavin Kerrigan, Giosue Migliorini, and Padhraic Smyth. Functional flow matching. In *International* Conference on Artificial Intelligence and Statistics, pp. 3934–3942. PMLR, 2024.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. *Advances in neural* information processing systems, 36:36652–36663, 2023.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. *arXiv preprint arXiv:2010.08895*, 2020.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In 11th International Conference on Learning Representations, ICLR
2023, 2023.

Qiang Liu, Mengyu Chu, and Nils Thuerey. Config: Towards conflict-free training of physics informed neural networks. *arXiv preprint arXiv:2408.11104*, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Yulong Lu and Wuzhe Xu. Generative downscaling of pde solvers with physics-guided diffusion models. *Journal of scientific computing*, 101(3):71, 2024.

Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–
planck equations through gradient–log–density estimation. *Entropy*, 22(8):802, 2020.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. *Advances in neural information processing systems*, 32, 2019.