# Absorbing Quantization Error by Deformable Noise Scheduler for Diffusion Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Diffusion models deliver state-of-the-art image quality but are expensive to deploy. Post-training quantization (PTQ) can shrink models and speed up inference, yet residual quantization errors distort the diffusion distribution (the timestep-wise marginal over $\vx_t$), degrading sample quality. We propose a distribution-preserving framework that absorbs quantization error into the generative process without changing architecture or adding steps.
(1) Distribution-Calibrated Noise Compensation (DCNC) corrects the non-Gaussian kurtosis of quantization noise via a calibrated uniform component, yielding a closer Gaussian approximation for robust denoising.
(2) Deformable Noise Scheduler (DNS) reinterprets quantization as a principled timestep shift, mapping the quantized prediction distribution $\vx_t$ back onto the original diffusion distribution so that the target marginal is preserved.
Unlike trajectory-preserving or noise-injection methods limited to stochastic samplers, our approach preserves the distribution under both stochastic and deterministic samplers and extends to flow-matching with Gaussian conditional paths. It is plug-and-play and complements existing PTQ schemes. On DiT-XL (W4A8), our method reduces FID from 9.83 to 8.51, surpassing the FP16 baseline (9.81), demonstrating substantial quality gains without sacrificing the efficiency benefits of quantization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a post-training method to absorb quantization error in diffusion models with two main technologies: (1) Distribution-Calibrated Noise Compensation (DCNC). The authors point out that an assumption in previous work -- quantization error is Gaussian -- does not hold. Instead, they use DCNC to statistically correct the mismatch. (2) Deformable Noise Scheduler (DNS). The authors theoretically interpret the quantization noise as a timestep shift in diffusion process and use DNS to schedule the timesteps and compensate for the timestep shift.

### Strengths
1. The flow of the paper is smooth, from problem observation to the methods that precisely target the problems. 
2. The approaches are inspiring. I especially like the part reinterpreting the quantization noise with diffusion timestep shift. 
3. The mitigation of quantization noise kurtosis in Figure 5 is impressive.

### Weaknesses
1. The evaluations need more clarification and improvement. 
2. Lacking comprehensive discussion compared to other outlier-aware methods. 
3. The quality of generated images is not substantially improved.

### Questions
1. This paper tackles the quantization error problem -- however, it is anti-intuitive that the results are better than using 16 bits. This important to justify. Is it because quantization error is large even in 16 bits? What if using the plug-and-play approach on FP16 or BF16? 
2. Are the evaluation metrics reported averaged over sufficient number of generated images? Error bars should be included for more statistical insights. 
3. Mitigating kurtosis of quantization noise seems overlap with outliers handling, as both target distribution tails. Could you discuss and compare the two?
4. Which timestep is Figure 4 at?

### Soundness
4

### Presentation
3

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
The author proposed a plug-and-play approach to improve the quantization
performance of diffusion models through introducing the distribution-Calibrated Noise Compensation (DCNC) that manages to calibrate the distribution to Gaussian, and Deformable Noise Scheduler (DNS) that is aware of the distribution shifts from the quantization. Together, the author claims that the approach preserves the distribution under ubiquitous samplers that fit most situations of multi-step diffusion models. This approach overcomes the limitations of current approaches (reshaping weight distributions, injecting approximated Gaussian noise to absorb residual errors) by ensuring the Gaussian assumption and distribution-awareness. The evaluation results indicate some improvements on strong baselines.

### Strengths
1. training-free, plugin-and-play

2. Broadly applicable: works across UNet/DiT/FLUX (incl. deterministic / flow-matching setups) with light calibration and no architecture changes; tangible gains in low-bit regimes (e.g., W4A8).

### Weaknesses
1. The improvement seems incremental. The teasing image in Fig. 1 seems to give a rare indication of distributional preservation. In Fig. 2, the FID score of the method is higher than the baseline (22.64 v.s. 22.78).

2. Missing metrics: for distributional matching tasks, LPIPS tells the perceptual distance and is also an important metric; the author should also report this metric in their paper. This metric has been exhaustively reported in SVDQuant, yet is missing in this article. The author should report this to strengthen their contribution.

3. Limited robustness diagnostics: most component/variance-estimator ablations are on a small LDM-4 setup; sensitivity to $W_u$, variance estimator choice, and DNS mapping under different schedulers/backbones (DiT-XL, FLUX) isn't thoroughly characterized.

### Questions
1. Is the timestep shift global or per-sample/timestep? Are there any failure cases where DNS degrades diversity or introduces artifacts?

2. For DCNC, any theoretical/experimental justification for the Gaussian claim? Please report the distribution-bias measurement metrics, such as kernel inception distance (KID), across all backbones.

### Soundness
3

### Presentation
3

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
This work proposed a distribution-preserving approach that absorbs quantization-induced shifts directly into the sampling process of diffusion/flow models. It proceeds in two stages: 
(i) Distribution-Calibrated Noise Compensation, which adds a uniform component to the quantization residual to correct its heavy-tailed deviation; 
and (ii) Deformable Noise Scheduler reinterprets quantization as a fractional timestep shift, rewriting the noise schedule so the quantized conditional matches the full-precision marginal. 

The method is plug-and-play and works with stochastic and deterministic samplers as well as flow-matching models.

### Strengths
1. This work is sufficiently innovative to me. The paper reframes PTQ for diffusion/flow models through a distribution-preserving lens, combining kurtosis-corrected residuals via a calibrated uniform component with treating quantization as a principled timestep shift, extending error absorption beyond stochastic sampling to deterministic samplers and flow-matching.  

2. The results of this article are solid, proving the effectiveness of the proposed scheme. Almost all have achieved SOTA in the field of quantization without extra sampling steps, e.g. DiT-XL W4A8 FID 9.83→8.51, even surpassing FP16. 

3. The intuition of work seems reasonable to me.

### Weaknesses
1. The connection and comparison with previous work is not clear enough.

2. The method of correction does not have a strong theoretical foundation.

3. No quantitative analysis of runtime was performed.

### Questions
1. Please position more explicitly what is new beyond: (i) replacing Gaussian residuals with DCNC’s uniform-calibrated residual, and (ii) mapping to a fractional timestep and redesigning the schedule—ideally with a side-by-side derivation comparing your Eqs. (15–20) to PTQD’s update (Eqs. 9–10) and a conceptual table of assumptions/scope (stochastic vs. deterministic/flow).

2. Uniform correction is heuristic; explore richer residual models. DCNC chooses a uniform component to cancel excess kurtosis (Eq. 13/Appendix A.1), which controls a single moment but ignores skew and higher-order structure.

3. The paper states “no extra steps” and “no additional memory,” but there is no quantitative runtime profile for per-step cost or calibration cost. I'm expecting an end-to-end latency and memory for FP16/PTQ/PTQD/Ours on the same GPU, plus the calibration time once vs. reused across runs; include breakdowns to reassure deployability.

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
- This paper addresses the challenge of post-training quantization (PTQ) for large diffusion models. Quantization reduces model size and speeds up inference but introduces errors that are particularly damaging to diffusion models, as these errors accumulate over the iterative denoising steps, leading to significant quality degradation.  
- The authors identify a key limitation in prior work (like PTQD), which attempts to absorb quantization error as Gaussian noise but is restricted to stochastic samplers and fails for deterministic or modern flow-matching models (e.g., FLUX).  
- The paper introduces a novel, plug-and-play framework with two core components:  
  - Distribution-Calibrated Noise Compensation (DCNC): This component corrects for the fact that quantization error is not perfectly Gaussian but is often "heavy-tailed." It analytically derives and adds a calibrated amount of uniform noise to the quantization residual, making its statistical properties (specifically, its kurtosis) much closer to a true Gaussian distribution.  
  - Deformable Noise Scheduler (DNS): The paper shows that the effect of quantization on the model's output distribution can be interpreted as a timestep shift in the original diffusion process. Instead of adding extra noise, DNS dynamically adjusts the noise schedule (the $\alpha$ values) for the quantized model so that its output distribution at each step aligns with the distribution of the original, full-precision model at a different, "shifted" timestep. This preserves the original generative distribution.  
- The method is plug-and-play enhancement approach that works with existing PTQ techniques (e.g., Q-Diffusion, PTQ4DiT, SVDQuant) across various model architectures (LDM, DiT, FLUX.1). It consistently improves quantitative metrics without adding inference latency or memory overhead. Crucially, it works for both stochastic (DDPM) and deterministic (DDIM, Flow Matching) samplers.

### Strengths
- The paper is well written.  
- The core idea of reinterpreting quantization error as a timestep shift is innovative and provides a principled, unified framework for the problem. It moves beyond heuristic corrections to a distribution-preserving solution.  
- A major strength is its compatibility with various sampler types (stochastic, deterministic) and model frameworks (diffusion, flow-matching).  
- The method requires no retraining (PTQ), adds no inference overhead, and can be seamlessly integrated with existing quantization methods.  
- The paper provides extensive experiments on multiple model backbones (LDM-4, DiT-XL, FLUX.1), bit-precisions (W8A8, W4A8, W4A4), and under different sampling settings (steps, guidance scales).  
- The evaluation of the approach shows promising results, outperforming other state-of-the-art works.

### Weaknesses
- The method requires a one-time calibration step involving a dataset (3,000 images in the paper) to estimate the noise statistics.  
- The method introduces a tunable weight \$W_u\$ for the uniform correction. While the paper selects a value of 0.2, its optimal value might be dataset- or model-dependent, requiring minor hyperparameter tuning for best performance.  
- Missing comparison to [1]-[5].  
- Minor typo at line 262 ("is" is written twice).  
  
[1] TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models (https://arxiv.org/pdf/2311.16503).   
[2] Post-training Quantization on Diffusion Models (https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf).  
[3] PQD: Post-training Quantization for Efficient Diffusion Models (https://arxiv.org/abs/2501.00124).  
[4] Towards Accurate Post-training Quantization for Diffusion Models (https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Towards_Accurate_Post-training_Quantization_for_Diffusion_Models_CVPR_2024_paper.pdf).  
[5] BiDM: Pushing the Limit of Quantization for Diffusion Models (https://proceedings.neurips.cc/paper_files/paper/2024/file/44b61c5c0ba06d55ab5a1cfb9cfff763-Paper-Conference.pdf).

### Questions
How does this method perform when compared to the references mentioned in the weaknesses?

### Soundness
3

### Presentation
4

### Contribution
2
