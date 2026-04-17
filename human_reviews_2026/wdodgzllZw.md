# DistillKac: Few-Step Image Generation via Damped Wave Equations

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
We present DistillKac, a fast image generator that uses the damped wave equation and its stochastic Kac representation to move probability mass at finite speed. In contrast to diffusion models whose reverse time velocities can become stiff and implicitly allow unbounded propagation speed, Kac dynamics enforce finite speed transport and yield globally bounded kinetic energy. Building on this structure, we introduce classifier free guidance in velocity space that preserves square integrability under mild conditions. We then propose endpoint only distillation that trains a student to match a frozen teacher over long intervals. We prove a stability result that promotes supervision at the endpoints to closeness along the entire path. Experiments demonstrate DistillKac delivers high quality samples with very few function evaluations while retaining the numerical stability benefits of finite speed probability flows.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose DistillKac a strategy for distillation for generative models whose densities evolve with the damped wave equation. This parameterization should remove the singularities at the end points which is common in diffusion/flow models. This achieved by fixing the propagation speed to some bounded value. The authors then propose a distillation strategy for this existing family of models.

### Strengths
I liked the detailed maths and found the Kac representation to be interesting.

* Theorem 8 and Corollary 9 seem interesting tools for analysis in the literature on (consistency) distillation.
* Theorem 4 seems novel within the context of CFG
* Addressing the problems of singularities in diffusion/flow models is important

### Weaknesses
## Primary concerns
### Theory
* How does this work clearly differentiate from [1]? It is unclear where [1] ends and where the contributions of this work begin. My rough understanding is that this work is [1] plus Theorems 4, 8 and 9 with distillation?
* I'm surprised that learning 1-D independent coordinate processes yields anything useful. It seems strange that this works as there is often interactions between the different coordinates. This seems to be an **incredibly strong** assumption to make.
* How do you initially train the teacher? Flow matching and diffusion models are nice because of simulation-free training, it is not clear to me that the Kac representation yields this for the teacher.
* Also since in Section 2.2 the process $X(t)$ is defined as a jump process it doesn't seem right to use standard numerical schemes for ODEs (like Euler or Heun) to integrate teacher model as suggested in line 203.
* Also since the Kac process is stochastic why are the flow maps governed by an ODE, shouldn't the flow maps be random variables *a la* [3]?

### Empirical
* Both CIFAR-10 and CelebA-64 are quite small and generally considered toy datasets, in particular since one of the contributions is improved CFG I feel like more compelling experiments for guidance should be included than just CIFAR-10 (32 x 32).
* The empirical results don't look great for Kac Flows. The traditional diffusion models seem to still outperform them despite the singularity issues, so what is the advantage? This seems especially true when compared to consistency models.
* Overall, the empirical results are wholly **uncompelling** in convincing the reader of the advantages of Kac flows over standard flow/diffusion models.

## Minor comments / suggestions
* Table 1 should get reworked it is unclear what each row corresponds to. Some labels would be nice, e.g., PDE, SDE, papers.
* Since there is a mixture of prior results in the theoretical sections, maybe highlight your contributions with a colored box?
* FID is a notoriously fickle and unwieldy metric to work with. Recent papers looking at text-guided generation [2] have included CLIP-score  [4] and Image Reward [5].

## References
[1] Richard Duong, Jannis Chemseddine, Peter K Friz, and Gabriele Steidl. Telegrapher’s generative
model via kac flows. arXiv preprint arXiv:2506.20641v3, 2025.

[2] Skreta et al., 2025, Feynman-Kac Correctors in Diffusion:
Annealing, Guidance, and Product of Experts, ICML, https://arxiv.org/pdf/2503.02819

[3] Kunita, Hiroshi. Stochastic flows and jump-diffusions. Springer Singapore, 2019.

[4] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PmLR, 2021.

[5] Xu, Jiazheng, et al. "Imagereward: Learning and evaluating human preferences for text-to-image generation." Advances in Neural Information Processing Systems 36 (2023): 15903-15935.

### Questions
1. Why do you refer to the random paths and evolution of the densities as the *Kac process* and *Kac flow* (resp.)? On line 117 you say when $d \geq 2$ such processes are called *random flights*. Aren't you working with $d \geq 2$ objects? The terminology is a bit confusing
2. How does this work relate to [1]?
3. Do Theorem 8 and Corollary 9 apply to any flow maps or only those admitted by the *Kac process*?
4. Likewise, how does Theorem 8 compare to [2, Theorem 4.1] outside of W1 vs W2?
5. How do you justify the coordinate-wise independence?
6. Please address any questions raised in the weaknesses sections.


[1] Skreta et al., 2025, Feynman-Kac Correctors in Diffusion:
Annealing, Guidance, and Product of Experts, ICML, https://arxiv.org/pdf/2503.02819

[2] Dou et al., 2024, Provable Statistical Rates for Consistency Diffusion Models, https://arxiv.org/pdf/2406.16213

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
4

### Summary
DistillKac presents an image generative model grounded in the damped wave equation, a hyperbolic PDE represented stochastically through the Kac process. Unlike diffusion-based ODEs or SDEs, whose velocity norms can diverge near the terminal time, Kac dynamics maintain globally bounded kinetic energy and Lipschitz continuity in Wasserstein space. The main contributions include adapting classifier-free guidance directly in velocity space while preserving bounded-energy guarantees, and introducing an endpoint-only distillation scheme with a theoretical stability bound linking endpoint consistency to full-trajectory alignment. Empirically, these features enable stable few-step sampling with competitive image quality, and theoretically, the work clarifies why endpoint matching is sufficient under finite-speed Kac dynamics.

### Strengths
- The paper is clearly written and presents solid theoretical foundations with rigorous stability proofs. 
- Adopting a hyperbolic PDE instead of the conventional diffusion formulation offers a novel structural perspective in generative modeling, characterized by finite propagation speed and bounded velocity. 
- The introduction of Kac flow–based generative modeling provides an innovative alternative to diffusion approaches, and the inclusion of stability guarantees and asymptotic links to diffusion broadens the methodological toolkit for finite-speed generative flows.

### Weaknesses
- While this paper provides solid theoretical formulations and formal stability proofs, it lacks empirical analysis demonstrating how Kac flow concretely improves stability in practice. Theoretical claims of bounded energy and finite propagation speed are not supported by detailed quantitative studies.
- Although Table 1 attempts to ensure fairness by matching the number of function evaluations (NFEs) across methods, the comparison would be more convincing if it also included training time, FLOPs, or wall-clock cost, since these reflect actual computational efficiency.
- The paper does not show whether the claimed stability advantages translate into observable robustness improvements, such as reduced divergence, fewer NaNs, or more consistent quality under large classifier-free guidance scales or few-step sampling.

### Questions
Overall, the work would benefit from comprehensive runtime and stability analyses to substantiate that the proposed Kac flow offers practical, measurable improvements beyond theoretical guarantees.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a fast image generator, DistillKac,  that uses the damped wave equation and its stochastic Kac representation to move probability mass at finite speed. Based on that, it introduces classifier-free guidance in velocity space that preserves square integrability and proposes endpoint-only distillation that trains a student to match a frozen teacher over long intervals.

### Strengths
1. Explore a new equation form (damped wave equation) for the diffusion process.
2. Provide a solid and detailed proof of the formulation, including the error bounds.

### Weaknesses
1. If I’m getting this right, the comparison in Figure 2 is made against normal sampling, specifically the non-distilled version of the model at the same number of steps. But doesn’t that seem a bit unfair? Plus, similar distillation methods can also achieve just a slight increase in FID. How does that show that your distillation method is better? It might be helpful to include comparisons with other distillation methods.

2. In the related work section, you only mention a few papers that focus on CIFAR-10 and CelebA-64, but there’s actually a lot of literature on distillation in latent space that has algorithms similar to yours. For instance, the skipping steps in LCM[1] have the same meaning as N in your Algorithm 1. It would be great to see some mention and analysis of those similarities.

3. While the proof goes into detail about the error and stability bounds, the experiments don’t seem to show better error and stability compared to other distillation methods. This leaves us unsure about the advantages of your proposed distillation method in practice.

[1]Luo, Simian, et al. "Latent consistency models: Synthesizing high-resolution images with few-step inference." arXiv preprint arXiv:2310.04378 (2023).

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DistillKac, a new framework for fast image generation based on damped wave equations and their stochastic Kac representation. Unlike diffusion models that rely on parabolic Fokker–Planck dynamics, DistillKac models generative flows as finite-speed hyperbolic PDEs, ensuring bounded kinetic energy and improved numerical stability. The authors extend this formulation with (1) velocity-space classifier-free guidance, which allows controlled generation without destabilizing the dynamics, and (2) endpoint distillation, a novel few-step training strategy where a student model learns from the teacher’s terminal distribution rather than stepwise trajectories.

### Strengths
1. The theoretical formulation is elegant and original, bridging kinetic transport theory (via the Kac process) with modern generative modeling. Modeling probability flow through damped wave equations offers a compelling alternative to diffusion-based parabolic PDEs.
2. The proposed endpoint distillation is conceptually simple yet powerful, backed by a provable endpoint-to-trajectory stability guarantee. This provides a rare theoretical justification for few-step model distillation.
3. The velocity-space classifier-free guidance is well-motivated and resolves instability issues common in diffusion guidance, maintaining bounded energy across time.

### Weaknesses
1. While mathematically grounded, the physical intuition behind using wave dynamics for generative flow could be developed further — especially how finite-speed propagation impacts sample diversity or convergence in practice.
2. The evaluation scope is modest, focusing mainly on CIFAR-10 and CelebA-64. Demonstrations on higher-resolution or more complex datasets (e.g., LSUN, ImageNet) would strengthen the empirical case.
3. The architecture choice (UNet) and the fixed set of integration schemes limit understanding of how well DistillKac generalizes to larger backbones (e.g., DiT) or different ODE solvers.

### Questions
1. Can the authors provide more empirical evidence for the endpoint-to-trajectory stability theorem, e.g., quantitative correlation between endpoint and full-trajectory discrepancies?
2. How does finite propagation speed affect perceptual smoothness or texture diversity compared to diffusion’s infinite-speed propagation?

### Soundness
3

### Presentation
3

### Contribution
3
