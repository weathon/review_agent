# ScalingCache: Extreme Acceleration of DiTs through Difference Scaling and Dynamic Interval Caching

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Diffusion Transformers (DiTs) have emerged as powerful generative models, but their iterative denoising structure and deep transformer blocks incur substantial computational overhead, limiting the accessibility and practical deployment of high-quality video generation. To address this bottleneck, we propose ScalingCache, a training-free acceleration framework specifically designed for DiTs. ScalingCache exploits the inherent redundancy in model representations by performing lightweight offline analysis on a small number of samples and dynamically reusing previously computed activations during inference, thereby avoiding full computation at certain denoising steps. Experimental results demonstrate that ScalingCache achieves significant acceleration in both image and video generation tasks while maintaining near-lossless generation quality. On widely used video generation models including Wan2.1 and HunyuanVideo, it achieves approximately 2.5$\times$ acceleration with only 0.5$\%$ drop in VBench scores; on FLUX, it achieves 3.1$\times$ near-lossless acceleration, with human preference tests showing comparable quality to original outputs. Moreover, under similar acceleration ratios, ScalingCache outperforms prior state-of-the-art caching strategies, achieving a 45$\%$ reduction in LPIPS for text-to-image generation and 20$-$30$\%$ reduction for text-to-video generation, highlighting its superior fidelity preservation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ScalingCache, a training-free acceleration framework for Diffusion Transformers (DiTs), focused on visual generative models for both image and video generation. The approach introduces an adaptive dynamic caching mechanism, leveraging block-wise differential scaling coefficients (precomputed offline) and runtime error-adaptive cache interval selection to reduce redundant computation during denoising inference. Experimental results on multiple state-of-the-art image and video DiTs show substantial speedup (up to 3.1×) with minimal loss in visual quality, outperforming previously established caching baselines in both fidelity and efficiency on several standard benchmarks.

### Strengths
1. The paper addresses a pressing bottleneck in generative modeling—accelerating the slow inference of DiTs—through an approach requiring no retraining.
2. The method section provides a systematic derivation of the prediction formula (Eq. for $\hat{\boldsymbol{y}}{t}^{l}$, Page 4), the estimation of scaling coefficients (Eq. for $\alpha{t}^{l}$), and the adaptive error-based update rule. Full derivations and algorithmic steps are offered in the main text and appendices.
3. Results across major models (Wan2.1, HunyuanVideo, FLUX) and diverse settings show consistently strong speedups and lower LPIPS/SSIM drops versus all baselines.

### Weaknesses
1. Several directly related and competitive caching acceleration methods—TokenCache (Lou et al., 2024) [1], Gradient-Optimized Cache (Qiu et al., 2025) [2], FastCache (Liu et al., 2025) [3], SpeCa (Liu et al., 2024) [4], DiTFastAttn (Kim et al., 2024) [5], and Dynamic Diffusion Transformer (Wang et al., 2024) [6]—are not cited, discussed, or included as baselines. While the paper does compare to prominent recent caching strategies, this omission leaves a gap in situating ScalingCache's novelty and advancement versus the current best practices.
2. The derivation of the dynamic error threshold $\delta_s$ in Algorithm 1 (Page 6) could benefit from further theoretical and empirical justification—currently, it is set based on an empirical mean of prior errors, and may be sensitive to outliers or sample diversity. The implications for worst-case quality loss (e.g., video flicker) as a function of $\delta_s$ are left unexplored.
3. While near-lossless acceleration is highlighted, the methodology is not extensively challenged at higher speedup factors, nor is there a rigorous exploration of when the trade-off between speed and quality breaks down (e.g., in especially long sequences, rare prompts, or highly dynamic scenes).
4. In some equations, notation such as $\Delta \boldsymbol{y}_{\tau}^{l}$ is used before being defined, and formatting is at times inconsistent (e.g., parameter lists in Algorithm 1). In Section 3.2 (Page 4), certain variables and indices are introduced abruptly, which may cause confusion for readers less familiar with blockwise DiT architectures.

[1] Lou J, Luo W, Liu Y, et al. Token caching for diffusion transformer acceleration[J]. arXiv preprint arXiv:2409.18523, 2024.

[2] Qiu J, Liu L, Wang S, et al. Accelerating diffusion transformer via gradient-optimized cache[J]. arXiv preprint arXiv:2503.05156, 2025.

[3] Liu D, Yu Y, Zhang J, et al. Fastcache: Fast caching for diffusion transformer through learnable linear approximation[J]. arXiv preprint arXiv:2505.20353, 2025.

[4] Liu J, Zou C, Lyu Y, et al. Speca: Accelerating diffusion transformers with speculative feature caching[J]. arXiv preprint arXiv:2509.11628, 2025.

[5] Yuan Z, Zhang H, Pu L, et al. Ditfastattn: Attention compression for diffusion transformer models[J]. Advances in Neural Information Processing Systems, 2024, 37: 1196-1219.

[6] Zhao W, Han Y, Tang J, et al. Dynamic diffusion transformer[J]. arXiv preprint arXiv:2410.03456, 2024.

### Questions
1. Can the authors provide a more systematic analysis of scaling/failure scenarios? For instance, under what prompt or video conditions does the method’s error substantially increase, and what diagnostic measures could be advised in practice?
2. How sensitive is the method to the quality and diversity of prompts used for offline estimation of $\alpha$ coefficients? Figure 6 shows convergence, but quantitative analysis across tasks would clarify real-world robustness.
3. Given the omission of several related works (see above), how does ScalingCache’s performance and computational overhead compare to FastCache, TokenCache, and GOC, both qualitatively and in speedup/fidelity metrics?

### Soundness
3

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
5

### Summary
The paper introduces ScalingCache, a training-free acceleration framework tailored for DiTs. By synergizing differential-scaling-based prediction with runtime-adaptive caching intervals, ScalingCache delivers significant speed-ups on both image and video generation while retaining near-lossless quality. Extensive experiments on Wan2.1, HunyuanVideo and FLUX show 2.3–3.1× acceleration with only 0.3–0.5 % VBench drop, and outperform prior state-of-the-art caching methods in LPIPS and other fidelity metrics, demonstrating superior fidelity-efficiency trade-offs.

### Strengths
1.	The proposed algorithm is clearly described, with a well-defined formulation and solid explanation.
2.	The manuscript is clearly structured and well-articulated, making it easy for readers to follow.

### Weaknesses
1.	The related work section overlooks the discussion of cache acceleration methods for UNet-based models, even though the cache acceleration technique for DiT-based models is an extension of and inspired by the earlier approaches developed for UNet-based models.
2.	It would be great if the proposed method could further improve the sampling speed of the distilled models.
3.	It's better to provide a user study to verify, through human evaluation, whether the generative performance of the method is close to the baseline.
4.	Is using 50 prompts sufficient to determine the appropriate scale?

### Questions
The authors are encouraged to further explore the applicability of the proposed approach to few-step distilled models.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents ScalingCache, a training-free inference acceleration framework specifically designed for Diffusion Transformers (DiTs), targeting image and video generation tasks. The core idea leverages the temporal redundancy in the hidden states during the denoising process in DiT. It conducts lightweight offline analysis on a small number of samples to precompute differential scaling factors and dynamically reuses previously computed activations during inference to bypass certain computation steps. This method achieves significant speedup while maintaining near-lossless generation quality, outperforming existing caching strategies. It particularly demonstrates better robustness in complex video generation scenarios.

### Strengths
1. Important and Practical Problem: The high computational cost of DiT significantly limits its deployment in real-world scenarios like video generation. The proposed training-free acceleration approach holds clear practical value.

2. Sophisticated Technical Design:
The introduction of the differential scaling factor α effectively combines zero-order and first-order predictions, addressing the issue of large prediction errors in certain layers seen with methods like Taylorseer.

3. Comprehensive Experiments with Outstanding Results:
Covers multiple SOTA models (Wan2.1-1.3B/14B, HunyuanVideo, FLUX);
Evaluates both image and video tasks with metrics including PSNR/SSIM/LPIPS/VBench/CLIP Score and human preference.

4. Engineering-Friendly: Only requires tuning one hyperparameter (Sf, i.e., number of initial full computation steps), without the need for training, fine-tuning, or complex scheduling logic, making it easy to integrate.

### Weaknesses
1. Generalization of α: The α coefficients need to be estimated offline using a small number of prompts (~50 prompts). While the paper claims convergence and low variance (Figure 6), it doesn’t verify its generalization to out-of-distribution prompts (e.g., extreme styles or rare objects). If α is sensitive to prompts, frequent re-estimation may be necessary for deployment.

2. (Section 3.3) The authors acknowledge that their strategy may fail for "static-to-dynamic" videos (e.g., a scene suddenly transitioning from stillness to high-speed movement). Such scenarios are not uncommon in real-world videos. Furthermore, as the denoising process can theoretically access tokens from all frames at every step, this raises the question of why such scenarios would significantly impact this strategy and whether a reasonable threshold can still be estimated.

3. While the authors claim "no additional inference overhead," the calculation of dynamic errors (Equation 7) and all-reduce operations (Appendix F) under sequence parallelism still involve communication and computational overhead.
The authors fail to report the storage cost of caching (storing y and Δy per module) and do not discuss the potential impact on devices with limited memory.

4. Lack of Comparisons: Why didn’t the authors compare their method with other acceleration approaches, such as Sparse VideoGen?

### Questions
1. In Table 1, why does MixCache achieve the highest score on the 14B WAN2.1 model? Could the authors explain this anomaly?

2. Concerns remain regarding the generalization of α. How does α perform in out-of-distribution prompts? If the selected prompts are not diverse enough, could this lead to suboptimal results?

3. Regarding the human evaluation experiments, were the participants professionals or anonymous general users? Could this introduce bias?

4. In the real-world deployment of 14B models, how much does the cache increase memory consumption?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents ScalingCache, a training-free method to accelerate Diffusion Transformers (DiTs). It improves upon standard feature caching by introducing a pre-computed scaling factor (alpha) for more accurate feature prediction and a dynamic caching strategy to adaptively skip computation steps. The method achieves significant speedups (2.5-3.1x) with minimal quality loss on major text-to-video and text-to-image models.

### Strengths
1. Effective: The differential scaling with alpha is a lightweight approach to improve feature prediction accuracy.
2. Strong Results: The method delivers impressive speedups while preserving high visual fidelity, outperforming prior methods in key metrics.
3. Practical: As a training-free solution, it is easy to apply to existing models without expensive retraining.

### Weaknesses
1. Robustness of Alpha: Is calculating the alpha coefficient from only 50 prompts sufficient for generalization across diverse inputs? The paper should discuss the method's robustness and show potential failure cases.
2. Analysis of Dynamic Caching: The ablation study confirms the dynamic caching strategy is useful, but lacks a deeper analysis. How does it adaptively change intervals for different content (e.g., static vs. dynamic scenes)?
3. VBench Score Breakdown: The analysis of VBench scores is too general. A breakdown by dimension (e.g., image quality, temporal consistency) is needed to clarify where the method truly excels. An explanation for why it doesn't achieve top scores on all models would also be helpful.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
