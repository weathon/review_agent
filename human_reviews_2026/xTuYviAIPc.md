# Guided Diffusion by Optimized Loss Functions on Relaxed Parameters for Inverse Material Design

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Inverse design problems are common in engineering and materials science. The forward direction, i.e., computing output quantities from design parameters, typically requires running a numerical simulation, such as a FEM, as an intermediate step, which is often an optimization problem by itself. In many scenarios, several design parameters can lead to the same or similar output values. For such cases, multi-modal probabilistic approaches are advantageous to obtain diverse solutions. Additional difficulties arise if the design problem is constrained. We propose a novel inverse design method based on diffusion models. The model learns a prior over possible approximate designs in a relaxed parameter space. Parameters are sampled using guided diffusion for which we leverage implicit differentiation of the simulation to evaluate the loss function. A design sample is obtained by backprojecting the sampled parameters. We develop our approach for a composite material design problem where the forward process is modeled as a linear FEM problem. We evaluate with the objective of finding designs that match a specified bulk modulus. We demonstrate that our method can propose diverse designs within 1% relative error margin from medium to high target bulk moduli in 2D and 3D settings. We also demonstrate that the material density of generated samples can be minimized simultaneously by using a multi-objective loss function.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a loss-guided diffusion framework for inverse material design where the loss requires solving an inner optimization problem, such as a FEM simulation. The approach relaxes the discrete design space into a continuous grid, trains an unconditional diffusion model as a prior over plausible microstructures, and steers denoising using gradients from implicit differentiation of the FEM equations.  After sampling, the “relaxed” parameters are back-projected into discrete material designs.  The method is demonstrated on synthetic 2-D and 3-D composite-material problems targeting prescribed bulk moduli.

### Strengths
Interesting integration of implicit differentiation through physics-based simulations with diffusion-based generative modeling.
The idea of using a diffusion prior to regularize exploration of a high-dimensional, discrete design space is conceptually appealing.
The experiments are carefully described, with detailed dataset generation, FEM setup, and hyperparameters.
Visual results show that the method can generate multiple microstructures achieving similar effective moduli.

### Weaknesses
1.	Ill-posedness motivation. The paper treats non-uniqueness in inverse design as a “difficulty,” whereas multiple feasible designs are natural and desirable. The actual problem is the non-differentiability and discreteness of the design space, not ill-posedness itself. This conceptual confusion weakens the motivation.
	2.	Equation (3). The “optimized loss function” derivation is standard total-derivative or adjoint sensitivity analysis used in PDE constrained optimization for decades. There is no new mathematical contribution, and the presentation of (∂c/∂u)^{-1} instead of an adjoint formulation may be computationally unrealistic.
	3.	Overlap with Optimal Experimental Design (OED): The proposed method is essentially OED or gradient-based design optimization wrapped in a diffusion sampler: sensitivities from a differentiable simulator guide exploration of parameter space toward a target observation. This strong conceptual link to classical and Bayesian OED is not acknowledged, leaving the work not well contextualized.
	4.	Unclear advantage of loss-guided diffusion: Since FEM gradients are available, one could directly perform deterministic or probabilistic gradient-based optimization. The paper provides no evidence that diffusion-based sampling yields better accuracy, stability, or diversity than standard OED or adjoint methods.
	5.	Missing baselines. No comparison is made to:
	•	gradient-based design optimization using the same sensitivities,
	•	Bayesian or stochastic OED samplers, or
	•	existing diffusion-based design methods
Without these, performance gains cannot be assessed.
	6.	Notation and clarity issues. The paper inconsistently switches between t and i as diffusion indices, making it unclear how gradients \nabla_{x_t}\ell_y are applied in practice. Terms such as “optimized loss” and “relaxed parameter space” are used ambiguously.

### Questions
1.	What is the concrete advantage of loss-guided diffusion over direct gradient-based or Bayesian OED methods using the same FEM sensitivities?
	2.	Could the authors clarify whether the approach can handle nonlinear or time-dependent PDEs, or is it limited to linear FEMs?
	3.	Would results change if the same loss were optimized directly via gradient descent rather than diffusion sampling?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a novel guided diffusion method for inverse material design. Its key innovation is using implicit differentiation of a physics-based simulation (FEM) to guide the denoising process, which eliminates the need for surrogate models. The model operates on a continuous, relaxed representation of material microstructures and is trained to generate designs that meet target bulk moduli. The approach successfully produces diverse 2D and 3D designs with low error for mid-to-high-range target values.

### Strengths
The paper introduces an innovative, training-free guidance mechanism that uses implicit differentiation of FEM simulations, avoiding the need for surrogate models and increasing flexibility.

It effectively uses a continuous parameter space to enable differentiability, while a learned prior on the microstructures ensures physical plausibility.

The approach is validated with comprehensive experiments in both 2D and 3D, using diverse metrics such as relative error and material coverage.

The method demonstrates practical utility by using real material properties and employing a task-agnostic diffusion model that can be reused for different objectives.

The submission includes detailed ablation studies, architectural specifications, and hyperparameters that support reproducibility and clarify design choices.

### Weaknesses
The evaluation is confined to linear elastic FEM problems and shows significantly degraded performance for targets at the extremes of the property distribution.

The evaluation is missing comparisons to established inverse design baselines like genetic algorithms or topology optimization, making it difficult to contextualize the method's performance.

The paper lacks a detailed analysis of computational cost, scalability with increased resolution, and the computational bottleneck of the guidance step.

The method relies on manual tuning of gradient scaling factors, which suggests sensitivity to parameter scales and raises questions about its generalizability.

The evaluation relies on a potentially arbitrary coverage metric and lacks statistical robustness, with most results based on single training runs.

### Questions
- Could you elaborate on the specific challenges of extending this method to non-linear FEM and what approximations might be required?

- To better contextualize your contribution, could you compare your method against at least one established baseline (e.g., GA, TO)? There is a rich literature on guided-diffusion models for design (see [here](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html) and [here](https://ojs.aaai.org/index.php/AAAI/article/view/26093) for example).

- How did you determine the manual gradient scaling factors? Can you offer a more principled approach for tuning these on new problems?

- How does the computational cost scale with simulation resolution? What is the main bottleneck: the FEM solver, implicit differentiation, or the denoising process?

- How could your framework incorporate hard design constraints, such as minimum wall thickness or material density limits?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a loss-guided diffusion framework for inverse material design. The method combines an unconditional diffusion prior trained over relaxed material representations (continuous per-element E,ν,ρ) with physics-based guidance derived from implicit differentiation through a FEM simulator. During generation, the reverse diffusion process is guided by loss gradients computed from the FEM, encouraging samples that achieve a target property (bulk modulus K). Generated relaxed samples are then “backprojected” to discrete material designs using a 2-component Gaussian Mixture Model (GMM) and skeletonization to identify particle/matrix assignments, particle radius, and volume fraction. The paper demonstrates the approach on 2D and 3D composite materials with spherical inclusions, targeting specific bulk moduli. Experiments report that several generated samples fall within 1 to 5% relative error of the target, showing diversity across feasible material designs.

### Strengths
The combination of an unconditional diffusion prior with implicit FEM-based loss guidance removes the need for surrogate property predictors and allows direct physics-based gradients to guide the diffusion process.

Results show the framework working across different scales with simple ablations (unguided vs guided, varying diffusion steps) and an analysis of runtime.

The use of bounds to validate generated samples demonstrates awareness of physical plausibility.

### Weaknesses
Weaknesses

1.Unclear positioning: The paper’s core mechanism of loss-guided diffusion is already well established through prior frameworks such as classifier-free guidance. A lot of work in engineering applications has also been done to solve inverse problems using diffusion models. The claimed novelty claimed here is the integration of FEM-based implicit gradients, but there is no direct comparison against simpler or existing alternatives (e.g., conditional or classifier-free guided diffusion, or regressors). Without such baselines, it is unclear what practical benefit the proposed method offers beyond using FEM gradients instead of learned surrogates.

2. Weak treatment of constraints and backprojection; The method relies on relaxing discrete design variables into continuous fields, then heuristically mapping them back to discrete material configurations via GMM fitting and skeletonization. This post-hoc step lacks guarantees, and there’s no theoretical or empirical analysis of its robustness. It will be helpful if the authors can discuss whether the approach ensures design feasibility during diffusion or not or it simply hopes the prior keeps samples plausible. 

3. All results are restricted to isotropic, linear composites with spherical particles. There’s no evidence the approach scales to nonlinear materials, multiple objectives, or more complex geometries, such as those seen in topology optimization. Moreover, there are no comparisons with standard baselines such as diffusion-based topology optimization, adjoint-based inverse design, or surrogate-conditioned diffusion. The metrics could also be expanded by reporting diversity, uniformity, quality,etc.

### Questions
Can you provide results against strong baselines such as conditional diffusion models, a regressor-guided sampler, or a classical optimization to show concrete benefits?

How often does the backprojection fail or deviate significantly from the guided relaxed solution? Can you quantify 

How sensitive are your results to the number of denoising steps, or gradient scaling? Do these affect sample diversity or physical validity? 

Can this framework handle nonlinear or anisotropic materials, or multiple design targets? 

How does the method work for extrapolation beyond the original space?

Reproducibility: Since the MatWeb data cannot be shared, can you release a synthetic dataset and a pretrained model for verification?

### Soundness
2

### Presentation
3

### Contribution
2
