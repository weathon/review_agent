# BézierFlow: Learning Bézier Stochastic Interpolant Schedulers for Few-Step Generation

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 4, 8

## Abstract
We introduce BézierFlow, a lightweight training approach for few-step generation with pretrained diffusion and flow models. BézierFlow achieves a 2–3× performance improvement for sampling with $\leq$ 10 NFEs while requiring only 15 minutes of training. Recent lightweight training approaches have shown promise by learning optimal timesteps, but their scope remains restricted to ODE discretizations. To broaden this scope, we propose learning the optimal transformation of the sampling trajectory by parameterizing stochastic interpolant (SI) schedulers. The main challenge lies in designing a parameterization that satisfies critical desiderata, including boundary conditions, differentiability, and monotonicity of the SNR. To effectively meet these requirements, we represent scheduler functions as Bézier functions, where control points naturally enforce these properties. This reduces the problem to learning an ordered set of points in the time range, while the interpretation of the points changes from ODE timesteps to Bézier control points. Across a range of pretrained diffusion and flow models, BézierFlow consistently outperforms prior timestep-learning methods, demonstrating the effectiveness of expanding the search space from discrete timesteps to Bézier-based trajectory transformations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes BézierFlow (BF), a lightweight, training-efficient way to learn a scheduler for Stochastic-Interpolant (SI) models so that few-step ODE sampling matches a strong many-step teacher. Schedulers $\left(\bar{\alpha},\bar{\sigma}\right)$ are parameterized as low-dimensional Bézier curves and optimized via learnable control points against the teacher trajectory. BézierFlow demonstrate high-quality few-step sampling ($\approx$3–8 NFEs) with minimal tuning and no model retraining.

### Strengths
– Writting is clear and easy to follow. 

– Using Bézier control points to model SI schedulers is simple and effective. 

– Competitive results vs. baselines at very low NFE.

### Weaknesses
– Although the target problem differs, VDM [1] and Multi-marginal SI [2] also optimize schedules to improve performance. A brief discussion contrasting these with BézierFlow would clarify the advantages of using Bézier curves versus alternative parameterizations (and the trade-offs without Bézier).

– I’m curious about the method’s limits. Recent distillation works [3,4] reduce sampling to a few or even one step while maintaining quality. I understand BF is lightweight and not the same setting, but probing the few-to-one NFE regime would better reveal the method’s capability.

[1] Kingma et al., “Variational Diffusion Models”, NeurIPS 2021

[2] Albergo et al., “Multimarginal generative modeling with stochastic interpolants”, NeurIPS 2024

[3] Zhou et al., “Inductive moment matching”, ICML 2025

[4] Kim et al., “Consistency trajectory models: Learning probability flow ode trajectory of diffusion”, ICLR 2024

### Questions
– How far can BF push NFE down (e.g., $\le2$) before quality collapses?

– Does a scheduler learned on dataset A transfer to B (or to different guidance scales/resolutions) without re-tuning?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes BézierFlow, a lightweight method to improve few-step generation for pretrained diffusion and flow models by learning the sampling trajectory—formulated as a stochastic interpolant (SI) scheduler—instead of only optimizing discrete ODE timesteps. The scheduler’s coefficient functions are parameterized as 1D Bézier curves whose control points enforce boundary conditions, differentiability, and (claimed) monotonic SNR. Training is a teacher-forcing alignment to a high-NFE teacher using a perceptual loss, and takes ~minutes.

### Strengths
1. This paper moves beyond learned timesteps to continuous path learning via SI schedulers, which conceptually unifies diffusion and flow settings and widens the search space versus LD3.  Besides, it also provides analyses for endpoint-marginal preservation and schedule-invariance of the SI training objective. 

2. This method only needs a few parameters and minutes of training, no finetuning of the base model, and it supports plug-and-play at inference.

### Weaknesses
1. Could the authors include more qualitative comparisons and broader evaluation metrics on SD3 models? Besides, for large-scale t2i model like SD3, except the FID values, other metrics, like CLIP are also important to evaluate the performance.

2. Notice that training relies solely on LPIPS. Could this induce instability or mode collapse by using only this loss across most models training? 

3. The authors analyze briefly in Sec 4.4 the difference between BézierFlow and other schedulers, but this anlysis is not convincing to clarify its advantages over other schedulers. Could the authors provide more solid theoretical analysis?

4. While “15 minutes” training time is appealing, results are only on CIFAR-10. Include runtime and memory analysis on larger datasets and models (e.g., ImageNet, SD3) to support the scalability claim.

### Questions
see Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a new method for efficient sampling with generative models in the stochastic interpolants framework [Albergo et al. 2023], such as diffusion, score-based, and flow-based models. They optimize stochastic trajectories (i.e., the path from the latent space to the data distribution), parameterized by continuous functions $\alpha(s)$ and $\sigma(s)$, so that a scheduler requiring a small number of NFEs can be distilled from a more complex one in a teacher–student manner. The student scheduler parameters $\bar{\alpha}(s)$ and $\bar{\sigma}(s)$ are modeled using degree-$n$ 1-D Bézier curves with learnable control points. Bézier polynomials are chosen because they satisfy several constraints required by scheduler functions. The authors demonstrate effectiveness on multiple diffusion and flow-based models and datasets, achieving results that are better or comparable to the state of the art for low-NFE samplers.

### Strengths
* The paper is well-written and easy to follow.
* The use of Bézier polynomials to optimize schedulers for low-NFE samplers is interesting and well-motivated, and the authors conduct extensive experiments on multiple models within the stochastic interpolant framework.
* In low-NFE regimes, the method achieves quality that is often superior to existing approaches while requiring minimal training cost.

### Weaknesses
* In the experiments section, both the “Generalizability to Unseen NFEs” and “Training Efficiency” subsections would benefit from more detailed explanations. In particular, I found the first one somewhat unclear, while the comparison in the latter subsection is a bit confusing. I suggest providing additional training-time comparisons at equal NFE settings, including against non-distillation-based methods.
* All experiments are done on image-based models, so the choice of the LPIPS loss is justified, but it limits the scope of the evaluation.
* Minor: typo at line 299.

### Questions
Beyond the additional details requested above, could the authors comment on potential alternatives to LPIPS when moving beyond vision tasks? Would their method retain the same advantages under different distance metrics?

### Soundness
3

### Presentation
3

### Contribution
3
