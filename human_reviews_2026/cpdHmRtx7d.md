# One step further with Monte-Carlo sampler to guide diffusion better

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 4

## Abstract
Stochastic differential equation (SDE)-based generative models have achieved
substantial progress in conditional generation via training-free differentiable
loss-guided approaches. However, existing methodologies utilizing posterior sam-
pling typically confront a substantial estimation error, which results in inaccurate
gradients for guidance and leading to inconsistent generation results. To mitigate
this issue, we propose that performing an additional backward denoising step and
Monte-Carlo sampling (ABMS) can achieve better guided diffusion, which is a
plug-and-play adjustment strategy. To verify the effectiveness of our method, we
provide theoretical analysis and propose the adoption of a dual-evaluation frame-
work, which further serves to highlight the critical problem of cross-condition
interference prevalent in existing approaches. We conduct experiments across var-
ious task settings and data types, mainly including conditional online handwritten
trajectory generation, image inverse problems (inpainting, super resolution and
gaussian deblurring), and molecular inverse design. Experimental results demon-
strate that our approach consistently improves the quality of generation samples
across all the different scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes ABMS for solving inverse problems using diffusion models.
It identifies and addresses errors arising from the imprecise gradient guidance in a prominent baseline method, DPS.
To resolve this, ABMS computes the gradient guidance by drawing multiple samples at each denoising step and using the diffusion model's predictions for those samples.
The proposed method shows enhanced performance on a variety of tasks, such as character generation, image restoration, and monocular property prediction.

### Strengths
- It highlights that many existing training-free inverse problem methodologies including DPS rely on numerous assumptions and approximations, which often leads to suboptimal results.

- It makes a significant contribution by demonstrating the trade-off inherent in using gradient guidance through a dual-focus evaluation.

### Weaknesses
- There are concerns regarding the practical applicability of the proposed method. ABMS is computationally heavy as it requires M diffusion model operations at each step, and 1000 sampling steps.

- Most of the demonstrated tasks have limited practicality; for instance, the inpainting task only uses very small masks instead of large ones.

- The results are shown on pixel-space diffusion models. However, state-of-the-art diffusion models like Stable Diffusion or Flux-dev operate in latent space. Including results on these models would broaden the paper's scope.

### Questions
Is the proposed methodology applicable to recent state-of-the-art diffusion models such as Flux-dev?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a training-free guidance method to improve the inverse problem in diffusion models. The authors argue that the existing method, DPS, suffers from a biased gradient. To mitigate this, ABMS leverages the gradient of the averaged multiple backward predictions of diffusion models. With theoretical justification, ABMS shows improved results compared to existing baselines.

### Strengths
- **Well-Motivated and Simple Solution**: The proposed ABMS method is intuitive, well-motivated by the law of total expectation, and directly targets the identified source of bias (single-point estimation). It seems to be easily adapted to the existing codebase due to its simplicity and can be widely utilized as it does not require any additional conditions.

- **Comprehensive Experiments**: The method's effectiveness is demonstrated across three different domains. The consistent improvements across all tasks support the claims.

- **Theoretical Analysis**: The estimation error analysis showing that ABMS's error bound under the given assumptions avoids the $\delta_{f}(x_t)$, which plagues the DPS bound, provides a theoretical justification for the effectiveness of the ABMS.

### Weaknesses
- **Computational Overhead**: The most significant weakness is the increased computational cost. ABMS requires $M$ denoising network evaluations in addition to the original denoising steps. While it can be parallelized, the memory consumption can grow rapidly as it also requires additional gradient calculations. For a more comprehensive analysis, the additional computational time and memory consumption for ABMS should be reported.

- **Novelty in Context**: The idea of using Monte Carlo sampling to get better estimates in diffusion guidance is not entirely new (e.g., LGD-MC). The paper's novelty lies in the one-step-back formulation. The paper could be stronger if it more directly compared against a simpler MC-DPS baseline (i.e., averaging $M$ estimates from $x_t$, not $x_{t-1}$) to isolate the benefit of the backward step from the benefit of MC sampling. Furthermore, while the authors argue that LGD-MC incurs high computational costs, the proposed ABMS is a more computationally intensive method as it requires multiple diffusion model calls with gradient calculation, while LGD-MC requires only one diffusion model call.

- **Limited Scope of Samplers**: While the authors acknowledge this limitation, it is unclear how ABMS, which relies on the SDE-based one-step transition, would be adapted to faster ODE-based or higher-order solvers (like DPM-Solver++, etc.) that are now state-of-the-art for fast sampling. This potentially limits the method's practical application.

### Questions
- **Clarification on Computational Cost**: Please refer to Weakness 1.

- **Ablation Study Without DSG**: ABMS opts for the DSG for the stability of the algorithm. It could be helpful if the base performance of ABMS without DSG is analyzed compared to DPS.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets bias in training-free diffusion posterior sampling (DPS) for conditional generation and inverse problems. DPS approximates the posterior p(x0∣y)∝p(x0)exp⁡[−L(x0;y)]p(x_0|y) through two linearizations: (1) moving the expectation inside the loss, and (2) replacing the conditional mean with the denoiser output via Tweedie’s formula. The authors argue these lead to biased gradients and propose ABMS (Additional Backward step with Monte-Carlo Sampling): before estimating guidance at step ttt, it samples xt−1∼ p(xt−1∣xt) multiple times, evaluates the loss on each denoised x^0(xt−1), and averages the results to reduce bias. A dual-focus evaluation (alignment vs. global quality) is introduced. Experiments on classifier-guided digit synthesis, image inverse problems (SR, inpainting, deblurring), and molecular property conditioning show modest but consistent gains.

### Strengths
The problem—bias in loss-guided diffusion—is relevant and well-motivated.

The proposed method (MC sampling one step earlier) is simple, plug-and-play, and compatible with existing samplers. Also parallelization of MC sampler justifies the added computations. 

Evaluation across multiple domains, with clear quantitative metrics.

### Weaknesses
The “unbiased” claim is overstated: ABMS still produces a lower-bias approximation but not a provably unbiased gradient of the tilted posterior. No formal unbiasedness proof or convergence result is provided.

The theoretical bound (Sec. 4.2) only compares error upper bounds under strong assumptions (Lipschitz fff, monotone denoiser accuracy). Variance of the stochastic gradient is unaddressed.

Scope limited to DPS. Extensions to other plug-and-play or variational samplers (e.g., RED-Diff, flow-matching, ODE solvers) are not discussed experimentally.

The empirical improvements, though consistent, are incremental; figures often lack statistical significance or ablations isolating the MC vs. scaling effects.

### Questions
Can the authors clarify whether the proposed MC sampling yields an unbiased estimator of gradients? If not, what assumptions make the bias negligible?

Does the ABMS correction still hold under deterministic DDIM/flow samplers?

How does the method compare to re-scoring or re-noise-based variance-reduction schemes such as RED-Diff or Score-DPO?

Are there ablations showing the effect of the hypersphere projection alone versus MC averaging alone?

For molecular tasks, how sensitive is performance to the sampling count MMM?

### Soundness
2

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
4

### Summary
This paper addresses a critical and widely recognized problem in training-free guided diffusion models: the degradation of sample quality (e.g., FID) under strong conditional guidance. The authors convincingly argue that this issue stems from a systematic estimation error and bias in the guidance gradient, which is typically derived from a single, noisy denoising step. To mitigate this, they propose ABMS, a plug-and-play Monte-Carlo sampling strategy. At each reverse diffusion step, instead of relying on a single estimation path, ABMS explores multiple (M) potential predecessor states, denoises each one to predict a clean output, and then averages the guidance function evaluations across these M predictions. This Monte-Carlo averaging yields a more stable and accurate estimate of the true guidance gradient. The paper provides a theoretical justification (Proposition 1) showing that ABMS achieves a lower estimation error bound compared to the standard DPS approach. The effectiveness of ABMS is demonstrated empirically across a diverse set of tasks, including stylized handwriting generation, standard image inverse problems (inpainting, super-resolution, deblurring) on ImageNet, and molecular property prediction. The results consistently show that ABMS allows for stronger guidance without the typical collapse in sample quality, outperforming existing methods.

### Strengths
1. The paper tackles a fundamental challenge in guided diffusion. The "dual-focus evaluation" paradigm, which explicitly calls for balancing task-specific performance (e.g., reconstruction error) with general sample quality (e.g., FID), is an excellent and necessary framing for this problem area. The authors clearly articulate why strong guidance often leads to poor results, providing strong motivation for their work.
2. The proposed ABMS method is simple, well-motivated, and elegant. The core idea of using Monte-Carlo averaging to obtain a more robust estimate of an expectation is a classic statistical principle applied very effectively here. Its "plug-and-play" nature makes it broadly applicable to various diffusion frameworks and tasks without requiring model retraining, which is a significant practical advantage.
2.1 While not a deep theoretical paper, the inclusion of the estimation error analysis and Proposition 1 provides a solid theoretical grounding for why ABMS should be expected to work better than single-path estimators. It connects the intuitive idea of averaging to the mathematical problem of reducing the bias introduced by Jensen's inequality.

3. The experimental section is a major strength of this paper.
  - Diversity of Tasks: Testing the method on stylized text, natural images, and molecular data convincingly demonstrates its generality.
  - Rigorous Evaluation: The use of performance curves (Figure 3) that plot task-specific distance against FID is particularly effective. These plots provide a clear and powerful visualization of the core contribution, showing that ABMS dominates other methods by achieving a better trade-off frontier.

### Weaknesses
1. The primary drawback of ABMS is the increased computational cost, which scales with the number of Monte-Carlo samples, `M`. The paper demonstrates the effectiveness of `M=3` and `M=5` but never explicitly analyzes or reports the trade-off between performance and inference time/FLOPs. For a sampling method, this performance-cost analysis is crucial for researchers and practitioners to assess its viability. While Figure 3 implicitly shows the performance gain for different `M`, the associated cost is not quantified.
2. While the use of ImageNet 256x256 is a standard and respectable benchmark, the field of generative models is rapidly moving towards much higher resolutions (512x512, 1024x1024) and significantly larger models (e.g., Stable Diffusion). The paper does not demonstrate whether the benefits of ABMS hold or are even more critical at this larger scale, where guidance is often essential. Demonstrating scalability to at least one high-resolution setting would substantially increase the paper's impact.
3. The paper would benefit from a more detailed discussion of how ABMS relates to other methods that also aim to improve guided sampling by investing more computation. A key missing comparison is with "Restart Sampling"[1]. Both methods address quality degradation under strong guidance but seem to operate on different principles. A discussion clarifying these differences would better situate the paper's contribution within the current literature.

Refrence:
[1] Xu, Yilun, et al. "Restart sampling for improving generative processes." Advances in Neural Information Processing Systems 36 (2023): 76806-76838.

### Questions
See the weakness.
While the Weakness.3 is the most important, which affects the innovation of this paper, with an excellent explanation, I will increase the score, while a bad explanation, I will decrease the score.

### Soundness
2

### Presentation
3

### Contribution
2
