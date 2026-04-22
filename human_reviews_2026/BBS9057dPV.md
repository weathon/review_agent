# SteinDiff: Resolving the Contractivity Trap via Reference-free Stein Regularization

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 8, 2

## Abstract
A fundamental tension arises when accelerating diffusion-based generative models via their deterministic probability flow ordinary differential equation (PF-ODE)  paths, which we formally identify as the *contractivity trap*: efficient inference requires large step sizes, but stable convergence demands strong contractivity that limits model expressiveness. This results in error accumulation in inference as contractivity weakens. In this work, we propose a principled inference approach, called *SteinDiff*, that relaxes the contractivity constraints through reference-free Stein regularization. Specifically, drawing on Krasnosel'skiĭ-Mann theory, we reformulate the discretized ODE update operator to interpolate between predictions and current states. Importantly, we contribute efficient closed-form regularization estimators via Stein's identity,  which is grounded in the continuous SDE theory of diffusion models. Our step-wise  analytical approach eliminates the need for ground truth data to adapt to the local geometry of the data distribution while preserving the expressiveness of the vanilla model. Theoretically, our approach not only relaxes the strict contractivity requirements for robust convergence but also reveals a principle behind the stability of state-of-the-art (SOTA) pre-conditioned parameterizations. 
Practically, we offer a reference-free solution that reduces the risk of mode collapse in large-step inference. Extensive experiments validate our theoretical framework and demonstrate significant gains in generative inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses a fundamental limitation in diffusion model inference known as the contractivity trap. This trap arises from the tension between efficiency (large integration steps), expressiveness (complex score networks), and stability (the need for contractive updates in ODE-based sampling). While contractivity is a sufficient condition for convergence, this assumption fails for expressive modern diffusion models, leading to instability and degraded sample quality at low step counts. To overcome this, they propose SteinDiff, a new inference algorithm that enforces convergence guarantees leveraging Krasnosel’skiĭ–Mann fixed-point theory. Specifically, the update rule is re-expressed as a regularization of the standard ODE step with parameters being computed in closed form by leveraging the Stein Identity.  Experiments on standard benchmarks (CIFAR-10, ImageNet 64×64) show consistent improvements in FID and FD-DINOv2 scores under low-NFE conditions, with modest computational overhead.

### Strengths
(1) Replacing the Banch theorem with Krasnosel’skiĭ–Mann to enable larger step sizes and improved convergence speed is novel and well motivated
(2) Deriving a closed-form adaptive regularization parameters by leveraging the Stein Identity is really nice
(3) The experiments show meaningful gains (e.g., FID improvements) in low-NFE (number of function evaluations) regimes on standard benchmarks such as CIFAR-10 and ImageNet.

### Weaknesses
(1) Motivation and Illustration: To strengthen the motivation, consider adding plots that illustrate how a large Lipschitz constant constrains the step size on Cifar10 or ImageNet. Such visualizations would clarify the “contractivity trap” and emphasize why small steps are necessary for stability.
(2) Experimental Scope: Most experiments are limited to CIFAR-10 and ImageNet-64×64. These datasets are relatively small; extending evaluations to higher-resolution or larger-scale settings would significantly reinforce the paper’s claims about generality and scalability.
(3) Scalability and Clarity: The closed-form expression for γ (Eq. 12) depends on the trace approximation of \nabla u_k, estimated via the Hutchinson method (Algorithm 1). This approach may face scalability issues in high dimensions since only a limited number of probe vectors 𝑣 are used. Moreover, several symbols in Algorithm 1 appear without prior definition in the text—these should be clearly introduced for readability.
(4) Formatting: conclusion should show on page 9

### Questions
(1) Could the authors provide plots illustrating that current diffusion models exhibit large Lipschitz constants restricting the allowable step size and leading to instability?

(2) Given the limited scalability of the \gamma solution due to the trace approximation, wouldn't this approach be more suited for latent diffusuion models? Can you show results on large scale datasets? When does the approach break because of this approximation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper attributes inference errors in diffusion models to the contractive assumption (Lipschitz constant < 1) imposed on the discretization operator. This assumption arises from constructing the operator via the Banach–Picard theorem. To address this limitation, the authors propose a regularized discretizer based on Krasnosel’skii–Mann theory, which removes the need for contraction. However, the regularized formulation introduces the challenge of selecting an optimal interpolation weight. By casting this as an optimization problem, the authors show that the optimal weight admits a closed-form solution when applying Stein’s identity. The resulting algorithm is termed SteinDiff.

### Strengths
- The paper presents a rigorous and well-structured critique of the so-called contractivity trap in DM inference.
 - The adoption of the Krasnosel’skii–Mann (KM) framework provides a theoretically sound justification for why the approach by Karras et al. (2022) performs well.
 - The paper offers comprehensive error analysis, convergence results, and compelling empirical evidence supporting the proposed method.

Overall, I found the paper technically solid and enjoyable to read.

### Weaknesses
It is not clear that relaxing the operator from contractive ($L<1$) (from Banach–Picard theorem) to nonexpansive ($L \leq 1$) in Theorem 4.4 is sufficient to accommodate the Lipschitz constant of the data predictive function. Further clarification or justification of this transition would strengthen the argument.

#### Minor Comments

- Several symbols are undefined, including $\alpha$ and $\sigma_t$ (and by extension $\sigma_k$ and $\sigma_s$). A notation section in the appendix would fix this if space is a problem.
- The term non-expansive mapping is not formally defined.
- The acronym DM is first introduced in Section 2 but used earlier in Section 1.
- Figures 2 and 3 require more descriptive captions, as they are difficult to interpret based on the current text.
- Line 215 is missing a "to" as in "we shift to the design" and a "the" before contractive trap.

### Questions
* Why does the relaxation from $L < 1$ to $L \leq 1$ suffice to guarantee convergence of DM inference under the update of Eq. (7) with $γ^*$ from Eq. (12)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Stable convergence for ODE flow methods during inference/generation for diffusion models typically requires taking small enough step sizes so that the maps are contractive, which increases computation time. The authors refer to this as the "contractivity trap". Can one speed up inference without in spite of this? The authors propose SteinDiff, which allows fast inference without inheriting the problems of non-contractive maps, by interpolating with the previous iterate (as suggested by Krasnosel’ski˘ı-Mann theory). Practically, the authors provide a way to compute the optimal parameters for this regularization from data using Stein's identity. Experiments on show that it improves image generation in the low NFE (Number of Function Evaluations) regime on CIFAR-10 and ImageNet.

### Strengths
Speeding up diffusion models while preserving generation quality is an important question. The algorithm is easy to implement, and is a purely inference-time modification. It shows improvements on experiments in a low NFE regime.

### Weaknesses
In general, the conceptual explanations/discussions are loose and don't seem supported by theory. In particular, it is not clear to me how the mathematical theory presented (Theorem 4.4 based on KM theory) justifies the actual algorithm, which calls the method into question. The connection between the theory and the actual algorithm is missing. See my key question below.

### Questions
The update operator $T_\theta$ implicitly depends on the start and end time $s$ and $t$ (which is unfortunately suppressed in the notation), so during inference, a sequence of these operators which are *different* are composed with each other. However, the theorems about convergence to a fixed point relies on a *fixed* map $T$, so it is unclear to me how they apply. This is a major confusion for me. Please detail carefully in which parts you are considering different $T$'s, and which part you are considering the same $T$, because it is not valid to apply fixed-point theory when the map is changing.

It is unclear what the authors mean by "expressiveness", and especially by the claim that higher expressiveness requires large Lipschitz constant in the T map. This appears to me to be a misunderstanding resulting from conflating a step of discretization (where we want the added update to have Lipschitz constant <1) with the entire map itself (which can have large Lipschitz constant even though all updates are small). Please clarify.

Minor: 
In Algorithm 1, the $u_k$ are not defined.
Appendix A: "Detailed" -> "Details"

### Soundness
1

### Presentation
2

### Contribution
2
