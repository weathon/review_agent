# Efficient Diffusion Models under Nonconvex Equality and Inequality constraints via Landing

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The generative modeling of data in constrained sets is central to scientific and engineering applications with physical, geometric, or safety constraints (e.g., molecular generation, robotics). This article constructs constrained diffusion models on a generic nonconvex feasible sets $\Sigma$, by introducing a unified framework that simultaneously enforces both equality and inequality constraints throughout the diffusion process. Our theory and implementations encompass both overdamped and underdamped dynamics for the forward and backward sampling. The key algorithmic ingredient is a computationally efficient landing mechanism that replaces costly and not-always-well-defined projections onto $\Sigma$, maintaining feasibility without Newton solves and avoiding projection failures. Leveraging underdamped dynamics whose faster mixing reduces the steps needed to reach the prior distribution, the commonly-believed unavoidable heavy forward simulation cost in the constrained diffusion is alleviated. Empirically, this reduces function evaluations, enabling more efficient inference and training while preserving sample quality and substantially lowering memory usage. On equality-only and mixed (equality and inequality) benchmarks, our method shows reasonable sample quality, while substantially reducing computational cost and function evaluations. These results indicate that landing-based enforcement combined with underdamped dynamics provides a practical and scalable recipe for constrained diffusion on nonconvex feasible sets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a formulation to enforce constraints on diffusion models using the landing method. Diffusion models are viewed as a stochastic differential equation (SDE) and the "trajectory" of the SDE is refined at each time step to guide the generative processes to respect both equality and inequality constraints. While many of the previous works "project" the dynamic trajectory onto the constraint manifold, the paper argues that the projections are costly and not-always-well-defined. Instead, they introduce the landing approach, which arguably is more computationally efficient, to replace projections. The key mathematical idea behind this approach is to introduce an additional term (landing term) that enforces exponential decay of constraint violation in the SDE. Assuming that the gradients from these constraints are linearly independent, the landing term can be explicitly computed via simple matrix operations. This idea is tested on multiple benchmark examples, including both equality-constrained and mixed (equality + inequality-constrained) scenarios.

### Strengths
I find the landing term very interesting and novel, although I have to admit that I'm not fully familiar with the relevant literature. I do know some projection-based methods, so compared to those, as the authors argue, this landing approach sounds like an interesting, original idea. I also appreciate the fact that the landing term, given an assumption on the gradients of the constraint functions, can be further developed into an explicit solution. Overall, it is an interesting and potentially impactful idea for the research community.

### Weaknesses
Below are some potential weaknesses that the authors may want to help me understand better:
- I observe that the LICQ assumption (linearly independent gradients of the constraint functions) is a bit too strong but I'm not confident about this assessment (see my questions below). I would appreciate some further clarification on why this assumption is typically valid in many real-world problems. Also, some discussion about the cases where gradients are (almost) dependent (aka when the assumption breaks).
- The computation of the Jacobian and the (inverse of) Gram matrix sounds scary computationally. Would this be scalable to the cases where the dimension "d" is very large (e.g., generation of high-resolution micrographs of some material microstructures under some physical constraints)?
- The fact that the proposed method does not fully outperform projection-based methods may not be the critical weakness, assuming the mathematical idea is indeed new (to which I'll rely on other reviewers' assessment). However, it would still be valuable scientific knowledge to understand why that is the case. Is it because the additional term in the SDE adds more complexity in training optimization? Or that the constraint violation is penalized harder than the projection-based method? More analysis on the behavior of the proposed method, in comparison to other projection-based approaches, would make the paper a lot more interesting to read.

### Questions
- I'm just curious: is the LICQ assumption a valid assumption in the context of generative modeling? Using the generation of physics data as a hypothetical example, wouldn't there be a situation where gradients from different constraints may end up being collinear (linearly dependent) due to some sort of coupled effect? I can't think of a specific counterexample, though. Obviously, the authors have thought this through, so I would like their thoughts on this.
- Also, along the same line of thought, if the stacked Jacobian does not have a full rank or the matrix is ill-conditioned due to some "almost dependent" gradient vectors, how would the performance of the algorithm change?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces landing-based constrained diffusion, an approach for generating from a constrained distributions. The method is proposed for general nonconvex constraint sets, employing a "landing" method which guides the sampling towards the feasible set. As opposed to prior approaches like projection-based sampling, the proposed approach relies on a Lagrangian form, characterizing the entire sampling process. The paper argues that such an approach is more computationally efficient, reporting better runtimes than Riemannian Denoising Diffusion Probabilistic Models.

### Strengths
- **Mathematical Framework:** The formal presentation of the method is interesting, especially the incorporation of the constraint term inside the forward/reverse SDE. The provided justifications for the approach are well presented, with sufficient theoretical support to justify the method.

- **Efficiency Improvement:** Constrained sampling approaches often suffer from much longer runtimes. While that is often permissible when constraint satisfaction is strictly required, the development of efficient constrained sampling procedures is likely of value is time-sensitive settings.

### Weaknesses
- **Limited Novelty:** The inclusion of Lagrangian updates within the diffusion sampling process is not a new concept. For example, [1-3] propose augmented-Lagrangian sampling scheme for the reverse diffusion. While the inclusion of Lagrangian updates in the forward SDE is of potential interest, and differentiates the work from these predecessors, it is difficult to evaluate the impact of this without comparison to existing methods (or even any discussion from the authors). 

- **Experimental Evaluation:** The evaluation is conducted with a very limited set of baselines (*only Riemannian models*) , making it difficult to assess the actual performance of the method. Even compared to this single baseline, the approach is unable to provide a decisive edge, besides training and sampling. Additionally, the experimental settings really constitute toy examples, making it difficult to assess how this method would perform in the real-world.


---

[1] Liang, Jinhao, et al. "Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models." arXiv preprint arXiv:2502.03607 (2025).

[2] Lee, Seungjun, and Shinjae Yoo. "Efficient Physics-Constrained Diffusion Models for Solving Inverse Problems."

[3] Blanke, Matthieu, et al. "Strictly Constrained Generative Modeling via Split Augmented Langevin Sampling." arXiv preprint arXiv:2505.18017 (2025).

### Questions
- How does this approach compare to simply incorporating a guidance term into the training / sampling process?

- Why is the sampling time omitted for RFM (Table 1)?

- Has any ablation been conducted on the effect of hyperparameters (e.g., landing rate)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a unified framework for constrained diffusion models that operate under nonconvex equality and inequality constraints. The key innovation is the landing mechanism, which replaces explicit projection or reflection with a continuous “landing drift” that exponentially drives samples toward the feasible manifold. The authors derive both overdamped and underdamped Langevin variants (OLLA and ULLA) and develop a Conditional Wasserstein Path Matching (CWPM) objective to train diffusion models stably under such constraints.

### Strengths
- The paper proposes a novel, physically motivated mechanism (landing) to enforce constraints in diffusion processes. It avoids costly projection steps and unifies equality/inequality constraints in a single stochastic framework.

- The derivations are sound, connecting constrained Langevin dynamics with generative diffusion models. The introduction of CWPM as a Wasserstein-based objective is interesting and potentially useful beyond this context.

- The paper is generally well written and clearly structured, with good motivation and illustrative figures showing the landing behavior on different manifolds.

- The approach is computationally efficient and practically relevant for geometry- or physics-constrained generation problems such as molecular modeling and robotic trajectory generation. The proposed method achieves significant efficiency gains (up to 47× speedup) over prior constrained diffusion models.

### Weaknesses
- The landing drift term involves a large coefficient $( \alpha\sigma(t)^2 )$, but the paper does not explain how the landing rate $( \alpha )$ is chosen or how the integration step $( \Delta t )$ should depend on $( \alpha )$. The stability and sensitivity to these parameters are not well studied in experiments.
- The theoretical analysis relies on strong assumptions, such as the Linear Independence Constraint Qualification (LICQ) and the log-Sobolev inequality (LSI), which may not hold for nonconvex or high-dimensional constraint sets.
- The experiments mainly focus on **low-dimensional toy tasks**; scalability to high-dimensional or complex constraint geometries might be a potential issue.

### Questions
-  Does the landing mechanism act as a strong (projection-like) constraint or a weak (penalty-like) correction? How significant is the residual constraint violation under finite step size?
- Is it possible to extend the method to more general or nonconvex feasible sets? If not, what are the main theoretical or numerical challenges that prevent such an extension?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This article proposes a landing-based scheme for constraint-based diffusion modelling, which is comparatively softer (compared to projection-based schemes). A discretization scheme and score-matching loss function are derived, and experiments are also performed.

### Strengths
The method provides a principled means of translating landing-based schemes to the diffusion-model setting. 

The method provides a means of training the score, and this seems to accelerate the training time compared to other methods in this setting.

### Weaknesses
The convergence theory for this method is understandably a bit lacking, given its complexity, but it still may have been nice to have some more. At least, it would be helpful to see how the score-estimation error plays into any error guarantees.

Such a complicated scheme (involving multiplicative diffusion coefficients) may indeed provide an efficient mechanism for this problem, but the experimental guarantees do not seem to be outstanding (although the training-time speeds up).

### Questions
I have reservations about Remark 1. Generally, we do not expect the underdamped variant of DDPMs to provide much speedup compared to their overdamped variants. This is because, in contrast to LMC where the exponential integrator means that only the position error appears, for SGMs, one has both position and momentum errors in the score estimation term.

Miscellaneous typos:

- 137 embed -> embeds

- 189 both, the -> both the

- 192 produce -> produces

- 248 for forward and backward -> for the forward and backward processes

- 249 parametrized backward -> the parametrized backward process

- 300 involves relationship -> involves the relationship

- 390 computatioanl -> computational

- 392 reduce -> reduces

- 400 the similar generated distribution -> similar generated distributions

### Soundness
3

### Presentation
2

### Contribution
2
