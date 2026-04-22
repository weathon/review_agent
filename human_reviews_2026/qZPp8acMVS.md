# Curvature-Aware Residual Prediction for Stable and Faithful Diffusion Transformer Acceleration Under Large Sampling Intervals

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Diffusion Transformers have achieved remarkable performance in generative tasks, yet their large model size and multi-step sampling requirement lead to prohibitively expensive inference. Conventional caching methods reuse features across timesteps to reduce computation, but introduce approximation errors that accumulate during denoising—a problem exacerbated under large sampling intervals where significant feature variations amplify errors. Recent prediction-based approaches (e.g., TaylorSeers) improve efficiency but remain limited by sensitivity to feature variations across distant timesteps and the inherent truncation errors of Taylor expansions.
To address these issues, we propose a novel **C**urvature-**A**ware **R**esidual **P**rediction (CARP) framework, which shifts the prediction target from raw features to residual updates within Diffusion Transformer blocks.  We observe that residuals exhibit more stable and predictable dynamics over time compared to raw features, making them better suited for extrapolation. Our approach employs a rational function-based predictor, whose theoretical superiority over polynomial approximations is rigorously established: the numerator performs linear extrapolation using adjacent features, while the denominator incorporates discrete curvature to adaptively modulate the strength and behavior of the prediction. This design effectively captures the alternation between gradual refinements and abrupt transitions in diffusion denoising trajectories. Additionally, we introduce a curvature-guided gating mechanism that regulates the use of predicted values, enhancing robustness under large sampling steps. Extensive experiments on FLUX, DiT-XL/2, and Wan2.1 demonstrate our method's effectiveness. For instance, at 20 denoising steps, we achieve up to 2.88× speedup on FLUX, 1.46× on DiT-XL/2, and 1.72× on Wan2.1, while maintaining high quality across FID, CLIP, Aesthetic, and VBench metrics, significantly outperforming existing feature caching methods. In user studies on FLUX, CARP receives nearly 25\% more preference than the second-best method. These results underscore the advantages of residual-targeted prediction combined with a rational function-based extrapolator for efficient, training-free acceleration of diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes the CARP: a training-free, curvature-aware residual prediction scheme that ensures theoretical superiority over current prediction-based methods and reuse-based acceleration methods by shifting the prediction target from raw features to residual updates within the Diffusion Transformer Blocks. The author introduces a rational-function-based prediction method that involves a numerator performing linear extrapolation using adjacent features and a denominator that incorporates discrete curvature to ensure optimal skipping in steps. Experiment results show strong speedups with preserved qualities of generation in both image and video generation tasks.

### Strengths
1. Practical deployability. CARP is training-free and plug-and-play with no architecture changes. There is also minimal overhead introduced in both memory and inference, given the method uses a tiny, fixed history (3 residuals) and a simple gate, so it's easy to integrate and adds little overhead while yielding meaningful speedups.

2. Stability under large strides. The curvature-gated fallback with rational predictor adaptively controls the error introduced in acceleration, avoiding over-/undershoot and reducing error accumulation in comparison with evaluated baselines.

3. Methodology novelty. While many existing studies have studied the step-wise sparsity in multi-step generation, the method in this paper provides a new solution with mathematical justification.

### Weaknesses
1. Limited Ablation on Samplers. It appears that the DDIM is the only sampler evaluated; please provide a solver sweep (DPM-Solver / DPM-Solver++, Euler, and a flow-matching ODE case) under identical prompts, seeds, and step counts (e.g., 20/50), reporting FID and LPIPS, latency, and speedups to verify solver-agnostic robustness.

2. Missing Baselines. There are many similar step-wise sparsity-aware open-sourced baselines that have been proposed before this paper but not yet evaluated: DeepCache (CVPR 24'), AdaptiveDiffusion (NeurIPS 24'), SADA (ICML 25'), AB-Cache (ArXiv, optional as may not be open-sourced yet, but good to mention results in paper). Please evaluate these baselines on one of the DDIM/DPM Solver/Euler samplers. Please also provide all hyperparameters used in the comparison of these open-source baselines.

3. Missing Metric. The author should also report the LPIPS (also reported in TaylorSeer but missing in CARP) as a metric to justify the perceptual distance shifting from the unaccelerated model. Please compare your method with the TaylorSeer and baselines mentioned above on FID, PSNR, SSIM, and LPIPS with speedups.



Refs:

[1] Narayanan, Arvind, et al. "Deepcache: A deep learning based framework for content caching." Proceedings of the 2018 Workshop on Network Meets AI & ML. 2018.

[2] Ye, Hancheng, et al. "Training-free adaptive diffusion with bounded difference approximation strategy." Advances in Neural Information Processing Systems 37 (2024): 306-332.

[3] Jiang, Ting, et al. "Sada: Stability-guided adaptive diffusion acceleration." arXiv preprint arXiv:2507.17135 (2025).

[4] Yu, Zichao, et al. "AB-Cache: Training-Free Acceleration of Diffusion Models via Adams-Bashforth Cached Feature Reuse." arXiv preprint arXiv:2504.10540 (2025).

### Questions
1. The author states, "selected the optimal hyperparameters to ensure a fair comparison" in the experiment section. Can the author list all hyperparameters for each baseline they have tested?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a training-free acceleration framework for Diffusion Transformers. Instead of reducing the number of denoising steps, CARP predicts the residual updates between steps to avoid redundant Transformer computations. It introduces a rational-function predictor that extrapolates residuals using recent history. A curvature-based gating mechanism adaptively decides whether to use the predicted residual or perform a real forward pass, ensuring both stability and fidelity across smooth and complex denoising regimes. CARP achieves up to 2.9× speedup on FLUX and 1.5–1.7× on DiT-XL/2 and Wan 2.1 with minimal quality degradation.

### Strengths
1. The method offers a training-free, plug-and-play acceleration solution that can be integrated into any Diffusion Transformer without retraining or architecture modification.

2. The paper provides strong empirical and theoretical validation. Experiments on multiple high-end Diffusion Transformer backbones—FLUX, DiT-XL/2, and Wan2.1—demonstrate consistent acceleration (up to 2.9×) with minimal perceptual or quantitative degradation.

### Weaknesses
1. Limited evaluation scope and lack of scaling/generalization evidence.

The evaluation focuses exclusively on three Diffusion Transformer architectures (FLUX, DiT-XL/2, and Wan 2.1), each tested only under a 20-step denoising schedule. However, the curvature-aware predictor’s stability and error behavior could change under longer sampling horizons (e.g., 50 steps or 100 steps), where residual trajectories evolve more gradually but accumulate error differently. Assessing CARP’s performance across multiple step settings would provide stronger evidence of its robustness.

Moreover, although CARP claims to be architecture-agnostic, all experiments are conducted on Transformer-based diffusion models. It remains unclear how well the curvature-aware residual prediction generalizes to UNet-based diffusion models such as Stable Diffusion XL (SDXL) or EDM, where spatial convolutional dependencies differ substantially from DiT’s attention-driven dynamics. Including such baselines would better support the general-applicability claim.

2. Clarification on the Definition and Scope of Single, Dual, and Full DiTBlocks in Table 6.

I find the terminology in Table 6 — Single, Dual, and Full DiTBlocks — somewhat unclear. Does “Dual DiTBlock” refer to the double-stream blocks used in FLUX? What exactly does “Full DiTBlock” mean? Please clarify. Also, am I correct in understanding that Table 6 studies the performance difference between partial-block skipping and skipping the entire model stack at once?

3. Limited exploration of curvature threshold sensitivity.

The gating threshold 𝑁=1.4 is empirically selected but not deeply analyzed. Understanding how this hyperparameter trades off between stability and acceleration—and whether adaptive thresholds (learned or schedule-based) could outperform a fixed value—would make CARP more robust across models.

### Questions
I am curious about this training-free method: does it always skip the same block for different text inputs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
CARP is a training-free, model-agnostic acceleration for Diffusion Transformers that predicts residual updates (instead of raw features) using a curvature-aware rational predictor with gating, reducing accumulated errors—especially at large sampling steps. It delivers up to 2.88×/1.46×/1.72× speedups on FLUX/DiT-XL/2/Wan2.1 at 20 steps while preserving quality (FID/CLIP/Aesthetic/VBench) and ~25% higher user preference on FLUX, outperforming cache- and Taylor-based baselines.

### Strengths
1. The proposed method achieves SOTA performance.
2. It is a training-free method that does not consume many resources.

### Weaknesses
1. The writing is very poor.
2. The reason for employing the residual instead of the output itself is not clear.
3. The meaning of $\widetilde\kappa_t$ is not well justified.
4. The reason for using the first-order residual when $\widetilde\kappa_t$ is large is also not clear.
5. I am curious about the memory consumption of this method, as it requires storing lots of residuals.
6. The performance on SOTA video generation methods, e.g., Hunyuan and Wan2.1 on high-resolution generation, e.g., 720p and beyond, is missing. The acceleration of more powerful video generation models towards higher resolution should be more challenging and practical.
7. The author should include the experiments on few-step diffusion models.
8. I am curious why the authors do not compare their method with TaylorSeer at a higher order compensation, which can achieve better performance.

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CARP, a training-free inference acceleration scheme for Diffusion Transformers. Instead of extrapolating raw hidden states across timesteps, CARP predicts end-to-end residual updates from a short history and uses a rational extrapolator whose denominator is modulated by a curvature signal; a curvature-guided gate decides whether to trust the prediction or fall back to a full forward pass. Experiments on FLUX (text-to-image), DiT-XL/2 (ImageNet), and Wan2.1 (text-to-video) report speedups up to 2.88× with limited degradation on standard metrics, plus a user preference boost on FLUX. The history window is fixed to 3 residuals and the method uses curvature thresholds for both the rational term and the hard gate. A theoretical note links the sign of discrete curvature to the direction of linear-extrapolation error, motivating the denominator’s form.

### Strengths
1. The overall idea (predict residuals, not features) and the rational predictor/gating are easy to follow. 

2. CARP does not modify architectures and targets the low-step regime most relevant for latency. 

3. Reported speedups at 20 steps on FLUX/DiT-XL/2/Wan2.1 with competitive quality (e.g., FLUX up to 2.88×) and a positive user study.

### Weaknesses
1. Heavy reliance on hand-crafted thresholds / hyperparameters. CARP’s gating and denominator strength hinge on manually chosen thresholds (e.g., 𝑁). While there is an ablation, the paper does not convincingly show that these can be set once and generalize across models, datasets, and samplers without per-scenario tuning. 

2. Fixed and narrow temporal context. The method fixes the history window to 3, which may limit robustness when trajectories are noisier or when step schedules differ. There’s no exploration of adaptive or larger windows. 

3. Decision signals feel manual/heuristic. The normalized curvature, thresholds, and hard gate are designed features rather than learned criteria; it’s unclear how stable they are under distribution shift (e.g., different prompt mixes, seeds, schedulers). 

4. Missing comparisons to few-step/step-distillation baselines. Since the paper targets aggressive low-step regimes, it should compare against step-distilled or solver-distilled models under matched wall-clock/quality budgets, not only cache/prediction baselines (ToCa, TeaCache, Δ-DiT, TaylorSeer). The current suite does not settle the “best way” to achieve low-latency sampling. 

5. Limited discussion of overheads. Computing curvature, gating, and rational inference adds control-flow and tensor ops. The paper would benefit from a breakdown of where the speedups come from (skipped blocks vs. cache reuse vs. prediction) and their sensitivity to GPU/TPU kernels.

### Questions
1. Window size/generalization. How does CARP perform with other window sizes or an adaptive window (e.g., expand when curvature is stable, shrink when volatile)? Please report sensitivity curves and costs.
2. Can you provide cross-model, cross-dataset results showing that a single set of hyper-parameter works well without retuning? Any automatic calibration procedure?
3. Please include a kernel-level latency breakdown (prediction/gating vs. DiT forward) to show that wins aren’t hardware-specific and to guide future optimizations.

### Soundness
3

### Presentation
3

### Contribution
3
