# ETC: training-free diffusion models acceleration with Error-aware Trend Consistency

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Diffusion models have achieved remarkable generative quality but remain bottlenecked by costly iterative sampling. Recent training-free methods accelerate diffusion process by reusing model outputs. However, these methods ignore denoising trends and lack error control for model-specific tolerance, leading to trajectory deviations under multi-step reuse and exacerbating inconsistencies in the generated results. To address these issues, we introduce Error-aware Trend Consistency (ETC), a framework that (1) introduces a consistent trend predictor that leverages the smooth continuity of diffusion trajectories, projecting historical denoising patterns into stable future directions and progressively distributing them across multiple approximation steps to achieve acceleration without deviating; (2) proposes a model-specific error tolerance search mechanism that derives corrective thresholds by identifying transition points from volatile semantic planning to stable quality refinement. Experiments show that ETC achieves a 2.65× acceleration over FLUX with negligible (-0.074 SSIM score) degradation of consistency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a diffusion model acceleration framework called Error-aware Trend Consistency (ETC), which accelerates diffusion sampling without training while maintaining generation fidelity. ETC consists of a consistent trend predictor and a model-specific error tolerance search mechanism: the consistent trend predictor estimates a stable future denoising direction through historical model outputs, and the model-specific error tolerance search mechanism determines a safely reusable threshold based on denoising dynamics. Experiments show that ETC achieves a maximum speedup of 2.65× in image, video, and audio tasks, with negligible quality degradation.

### Strengths
1. The experiments are relatively comprehensive, covering mainstream quantitative comparison results and qualitative visualization results. Additionally, ablation experiments are conducted to verify the effectiveness of error control, statistical robustness, and the role of key parameters $n$ and $\alpha$.
2. The qualitative visualization results show that ETC maintains better generation consistency and avoids the structural distortion and detail loss issues present in the baseline methods.
3.  This paper features detailed derivations, which reduces the difficulty of understanding the proposed method. The provided illustrations and algorithm pseudocode further facilitate the comprehension of this paper.

### Weaknesses
1. It has not been compared with other training-free acceleration methods, such as methods based on intra-layer feature reuse and methods based on solvers.
2. The description of the search for model-specific error tolerance is relatively vague, which is only illustrated through Algorithm 2. Specific workflows are not provided for models like the Trend Inflection Point Analysis Model.
3. The important hyperparameters $\alpha$ and $n$ depend on being specified manually.
4. Some parameter names are incorrect. For example, in the title of Table 4, $p$ should be $n$.

### Questions
1. What is the difference in computational overhead between storing all historical trends and the recursive historical trend weighting method proposed in this paper?
2. Is the ETC proposed in this paper compatible with DDIM[1]? And is it compatible with other high-order samplers such as iPNDM[2] and DPM-Solver++[3]?
3. Does an adaptive optimization strategy for $\alpha$ exist?

[1] Song J, Meng C, Ermon S. Denoising diffusion implicit models[J]. arXiv preprint arXiv:2010.02502, 2020.

[2] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on
 manifolds. arXiv preprint arXiv:2202.09778, 2022.

[3] Lu C, Zhou Y, Bao F, et al. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models[J]. Machine Intelligence Research, 2025: 1-22.

### Soundness
2

### Presentation
2

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
The proposed ETC (Error-aware Trend Consistency) is a training-free method that speeds up diffusion-model sampling by (i) recursively smoothing all past noise predictions to form a Consistent Trend Predictor, then evenly applying this forecast across the next k timesteps to skip costly forward passes, and (ii) searching a model-specific error threshold by locating the breakpoint between the semantic-planning and quality-refinement phases of denoising, which adaptively controls the skip window. Tested on SDXL and Flux, Open-Sora 1.2 and Wan 2.1, and TangoFlux, ETC delivers up to 2.65× faster inference with virtually no perceptual degradation.

### Strengths
1.	The paper is clearly written.

2.	The approach requires no retraining and no architectural changes, yet shows consistent gains across diverse generation task.

### Weaknesses
1.	The error threshold is obtained through offline search. Its robustness to new resolutions, schedulers, or unseen models is unclear. An online or learned adaptation mechanism may strengthen the claim of generality. 

2.	The proposed trend-reuse idea is conceptually close to TeaCache and SADA, and its two-phase treatment of the denoising trajectory ( semantic-planning vs. quality-refinement ) resembles the block-specific reuse strategy in BlockDance. However, the paper gives little theoretical justification or quantitative ablation that clarifies how ETC departs from— or improves upon— these predecessors, leaving the boundary of novelty and the precise source of its gains insufficiently articulated. 

3.	This paper lacks comparison with training-based accelerators (e.g., progressive/step distillation), making it hard to judge how much of ETC’s benefit stems from being training-free. Moreover, all experiments cap the speed-up at ~2–2.6 ×; there is no analysis of more aggressive settings (3–5 ×), so it remains unclear whether ETC experiences significant quality collapse under higher acceleration.

### Questions
N/A

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
3

### Summary
This paper proposes ETC (Error-aware Trend Consistency), a training-free method to accelerate diffusion/flow model sampling by skipping some denoiser calls while preserving output consistency. ETC (i) aggregates historical inter-step model output changes with exponential smoothing to form a trend predictor and progressively distributes the predicted delta across a short window of skipped steps, and (ii) uses a model-specific error/tolerance search to adaptively decide when and how far to glide (skip) before re-evaluating the denoiser. Experiments on images (e.g., SDXL/FLUX), video (e.g., Open-Sora/Wan), and audio (e.g., TangoFlux) show ~2×–2.6× speedups with small drops in performance, and favorable comparisons against training-free reuse based baselines (TeaCache, SADA, MagCache, AdaptiveDiffusion).

### Strengths
- Practical, training-free modification; no retraining or distillation required.
- Conceptual advance over step-wise reuse: uses all-history, smoothed trend prediction rather than just the latest pair; progressive distribution mitigates drift during multi-step reuse.
- Adaptive skip length via a model-specific tolerance, more robust than fixed global thresholds.
- Cross-modality and cross-backbone evaluation (image/video/audio) with consistent speed vs. quality improvements; strong empirical appeal.
- Sensible ablations (tolerance sweeps, stability/error accumulation, sensitivity) and clear intuition for early/mid vs. late-step behavior.

### Weaknesses
- It seems that the main results of the paper rely on the assumption that $\epsilon_\theta(x,t,c)$ changes smoothly in $x$ and $t$, but this isn’t formally proven here. That’s fine for an empirical paper, yet it means there’s no hard guarantee, but only evidence from experiments.
- Algorithm/equation mismatch: There is a mismatch on the usage on the usage of delta in text and in the pseudo algorithm. It is not clear which one is implemented and it seems that it would make a difference in results depending on the option.
- Minor polish: Notation/indexing inconsistencies for $d$

### Questions
1) If you pick $\sigma$ on one small prompt set, does it transfer to out-of-distribution prompts (e.g., dense text or multi-object scenes) without retuning?

2) At equal perceptual quality (e.g. same FID score), which is faster: ETC or simply reducing steps with for example DDIM solver?

3) Table  4 shows SSIM sensitivity to $n$ but the latency impact of different $n$ is not reported. Could you add (or at least state) the corresponding wall‑clock changes for different values and give a simple "rule‑of‑thumb" for picking $n$?

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
3

### Summary
This submission focuses on the problem of generation time in diffusion models with multiple denoising steps. A new approach is presented to improve the sampling speed without significantly reducing the quality of samples, where instead of using only the most recent model output, ETC looks at all past outputs to estimate the denoising trend and create a smoothed weighted projection of historical changes.

### Strengths
- The idea of leveraging historical denoising trajectories to stabilize trend prediction and adaptively control approximation frequency is well-motivated, addressing instability in multi-step reuse methods.
- The evaluation performed with several different diffusion models with different domains is extensive and impressive, showing the scalability of the proposed approach

### Weaknesses
- The contribution is incremental in nature, in this submission an acceleration method is proposed that combines several observations to reduce the number of inference steps necessary to denoise the output. At the same time, evaluation and comparison to other methods does not include more complex approaches as for example second order denoising methods (DPM solver).
- In the presented comparison it seems that the proposed approach slightly outperforms evaluated approaches, but often providing higher speed-up in expense for the lower quality of generations
- I am puzzled, what is the motivation by enforcing the same trajectory between the original and accelerated method? I fail to see a proper explanation in the submission. I guess that maybe trajectory deviation might be the reason for reduced diversity of generations, but only if different trajectories converge to the same one. I do not see any experiments evaluating that. Could authors elaborate on that?
(Presentation) The introduction is filled with novel terms introduced by the authors without proper explanation, which makes it harder to understand the main contribution of the paper.
- “Despite their differing formulations, these approaches [Different formulations of diffusion process] all ensure the smooth temporal evolution of data features.” - this is not obvious for me. To the best of my knowledge there are no implicit mechanisms enforcing smoothness between different timesteps in any of the diffusion process formulations. This might be an emergent effect, but there are numerous works showing that the process varies significantly between different timesteps rather than opposite.
- The recursive trend estimation, dynamic window, and tolerance search introduce extra computations. The paper doesn’t quantify this overhead, so the “training-free” claim may still imply nontrivial offline cost.
- It is unclear for me what is the memory footprint of the proposed solution.
- Y axis in Fig 1a is confusing. I guess this is the error between ground truth model prediction and accelerated version. SImilarly X axis for Fig 3a is not explained before reference. Also “As shown in Figure 3a, we subtract a fixed value from the latent to represent the approximation error” - where is it shown?

### Questions
See weaknesses

### Soundness
3

### Presentation
1

### Contribution
2
