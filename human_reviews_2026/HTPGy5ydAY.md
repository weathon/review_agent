# Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Diffusion-based image generative models produce high-fidelity images through iterative denoising but remain vulnerable to memorization, where they unintentionally reproduce exact copies or parts of training images. Recent memorization detection methods are primarily based on the norm of score difference as indicators of memorization. We prove that such norm-based metrics are mainly effective under the assumption of isotropic log-probability distributions, which generally holds at high or medium noise levels. In contrast, analyzing the anisotropic regime reveals that memorized samples exhibit strong angular alignment between the guidance vector and unconditional scores in the low-noise setting. Through these insights, we develop a memorization detection metric by integrating isotropic norm and anisotropic alignment. Our detection metric can be computed directly on pure noise inputs via two conditional and unconditional forward passes, eliminating the need for costly denoising steps. Detection experiments on Stable Diffusion v1.4 and v2 show that our metric outperforms existing denoising-free detection methods while being at least approximately 5x faster than the previous best approach. Finally, we demonstrate the effectiveness of our approach by utilizing a mitigation strategy that adapts memorized prompts based on our developed metric. The code is available at https://github.com/rohanasthana/memorization-anisotropy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a memorization detection (and mitigation) method for Stable Diffusion models that prioritizes angular alignment of scores over their norms, especially in the low-noise (late) phase of generation. It is positioned as a generalization of prior work (e.g., Wen 2024; Jeon 2025): earlier metrics largely emphasize the magnitude of a (conditional - unconditional) score difference, whereas this paper argues that direction (cosine similarity) carries the decisive signal under anisotropic noise. Empirically, the paper reports detection gains and mitigation impacts, but several theoretical claims and some experimental details need strengthening.

### Strengths
1. Conceptually extends prior works (Wen 2024; Jeon 2025) to anisotropic noise settings in diffusion models. 
2. Provides initial empirical evidence suggesting angular features may correlate better with memorization risk.

### Weaknesses
1. The theoretical argument for “angular alignment is important” (Remark 1, Sec. 4.1) lacks rigor; Appendix A.1 remains heuristic without formal derivation or proof.
2. Confuses the norm of the score with the norm of the score difference; prior methods are not clearly distinguished. 
3. The key computation $s(X_T, t\approx 0, c)$ is conceptually inconsistent, evaluating the final-time score on the initial noise requires justification or on-manifold correction.
4. Experimental details (Figure 2 setup, time-step sampling, model version, datasets) are missing, making reproduction difficult. 
5. Comparison with Jeon 2025 mitigation results is absent, weakening empirical completeness. 
6. Overall, the method’s novelty is somewhat limited without stronger theoretical grounding or broader validation.

### Questions
See weaknesses

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
The paper studies memorization in diffusion models through the lens of anisotropy in the log-probability landscape. It argues that prior norm-based metrics are only effective under isotropy and introduces a denoising-free detection metric that combines (i) cosine alignment between conditional and unconditional scores in the anisotropic regime and (ii) score-norm magnitude in the isotropic regime. The method requires only two forward passes—conditional and unconditional—and is shown to outperform existing denoising-free metrics on Stable Diffusion v1.4 and v2.0 while being significantly faster.

### Strengths
•	The identification of anisotropy as a key factor in memorization detection is conceptually strong and empirically supported (variance of Hessian eigenvalues increasing toward low noise).

•	The method is denoising-free, computationally efficient, and integrates both magnitude and directional cues.

•	Experiments follow prior evaluation standards and report competitive or superior results. Using new designated bencharks such as MemBench is also a bonus.

### Weaknesses
•	The section titled “Failure of Norm-Based Methods in Anisotropy”	 starts with “We now prove…” However it does not constitute a formal proof. Rephrase to “demonstrate” or “show analytically” or similar.



•	In Table 1, please report standard deviations or confidence intervals. With 500 prompts, small improvements (such as the +0.001 AUC over Jeon et al. at n = 4, SD v1.4) may fall within statistical noise.



•	Fig. 3 (a), (b) should be key visualizations supporting the main hypothesis - however they are insufficiently explained.



•	Qualitative examples of generations with mitigated memorization are absent. Please provide such a comparison with the competitors. In addition, it is preferable to add an exhaustive set of such generations to the appendix.



•	Please refer to the concurrent [1], which employs a different criterion that similarly emphasizes direction over magnitude. The numerical computation of its curvature criterion sums angles (cosines) between the conditioning score and surface normals. Clarifying how this differs from your approach, and whether it fits within your anisotropic framework, would aid contextualization for future reference.



•	Please acknowledge [2] in the related work. The claim that “this phenomenon was firstly identified by Somepalli et al. (2023a)” overlooks [2], which concurrently with Somepalli et. al, demonstrated training data extraction (memorized samples) from diffusion models. 



•	The inference-time mitigation description lacks clarity - reducing reproducibility. Please specify how γ₁ and γ₂ were selected and how the score at ($t \approx 0$) is obtained without full denoising (a clarification in the Appendix would suffice).



-----



[1] Brokman et al. (2025). Tracking Memorization Geometry throughout the Diffusion Model Generative Process. NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations.



Link: https://openreview.net/forum?id=4XSVk26sHj



[2] Carlini et al. (2023). Extracting Training Data from Diffusion Models. USENIX Security Symposium.



Link: https://www.usenix.org/conference/usenixsecurity23/presentation/carlini

### Questions
Please see weaknesses above.

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
The paper investigates memorization in text-to-image diffusion models and argues that existing norm-based detection methods only work under isotropic log-probability assumptions, which is the case for high or medium noise levels. It formalizes how anisotropy arises in the low-noise regime and proposes a new denoising-free metric that is defined as the cosine similarity between conditional and unconditional score estimates. This new metric is combined with the known “isotropic term” - norm of the guidance vector. The method requires only two denoising steps (only noise estimation, not real denoising). The method improves AUC and TPR (at 1% FPR) on Stable Diffusion v1.4/v2.0. It also integrates the metric into an inference-time prompt-optimization scheme for mitigation, evaluated on MemBench.

### Strengths
1.	Sound theoretical framing. The analysis clearly connects memorization signatures to the isotropy and anisotropy regimes of the log-probability, filling a conceptual gap in prior isotropic norm-based methods.
2.	Efficient and simple metric. The proposed anisotropy-aware score is computationally cheap (two model steps, no actual denoising) yet achieves higher detection accuracy.
3.	The method is not limited to detection, it also demonstrates practical mitigation by optimizing prompt embeddings, producing non-memorized images while maintaining text alignment and aesthetic quality.

### Weaknesses
1. The reported “speed-up” is only shown relative to one denoising-free baseline, while simpler existing methods remain faster with no meaningful loss in performance, limiting the practical efficiency benefit of the proposed approach.
2. Incremental improvement weakness: AUC/TPR gains are marginal because prior methods already near saturation, making the contribution incremental rather than substantively advancing the state of the art, even if the method itself is technically sound.
3. No code to allow future comparison, and no indication that it will be provided in the future (Wen et al., which the authors compare to, do provide code). I urge the authors to add code so that their work will be maximally helpful to others. 
4. The imporovements are small. For mitigation the improvement looks more consistent, and for detection, Wen et al. are (~3x) faster and slighty worse, while Jeon et al. are on par and much slower (~6x). 
5. Minor - Equation 8 typo: the unconditional score in the cosine similarity term should not have “c” in it.

### Questions
1.	Given that prior methods already achieve near-saturated AUC/TPR, how do you justify the significance of the contribution beyond small numerical gains (with x2-3 in latency)?
2.	Since your claimed speed-up is only relative to a single denoising-free baseline, and simpler non-denoising-free methods remain faster with comparable performance (Wen et al.), can you clarify in what scenarios your method is actually the preferred choice in practice?
3.	Regarding the t=0 probe using high-noise latents - the model is inputted a high-noise latent but set with t=0, meaning the model “believes” it is operating at the end of the denoising process, which the authors showed is the low-noise regime. However the actual noise latent is far from the data manifold. Can you explain why a cosine similarity computed under this deliberate mismatch, based on what the model thinks rather than the actual noise level, should remain reliable and theoretically meaningful for detecting memorization (as empirically demonstrated)?
4.	Ablation - table 3 (in the Appendix) shows that gamma_1 differs is 2 and 0.1in  SD v1.4 and SD v2.0, suggesting that the anisotropy term is unstable and requires model-specific tuning.  Can you explain why such tuning is necessary, and can you provide an ablation where we can understand the individual contribution and necessity of each component (gamma1 =0  and gamma2=0 in separate settings)?

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
4

### Summary
This paper addresses memorization in text-to-image diffusion models, a phenomenon where models reproduce training samples. The authors argue that existing norm-based memorization detection methods, which rely on the magnitude of score estimates, implicitly assume isotropic log-probability distributions, an assumption that holds mainly in high- or mid-noise regimes. The paper identifies that in low-noise (anisotropic) regimes, memorization manifests instead through strong angular alignment between conditional and unconditional score functions. Based on this insight, the authors propose a new denoising-free memorization detection metric that combines (1) the cosine similarity between guidance and unconditional score vectors in the anisotropic regime and (2) the score-norm difference in the isotropic regime. This metric requires only two forward passes (conditional and unconditional), making it substantially faster than prior methods. Experiments on Stable Diffusion v1.4 and v2.0 show improved AUC and TPR@1%FPR compared to previous denoising-free baselines, with up to 5× speedup. The method is also used for inference-time mitigation via prompt augmentation, demonstrating improved trade-offs between similarity, CLIP, and aesthetic scores on MemBench.

### Strengths
1. The paper is well-organized, making it easy to follow. The visualizations clearly illustrate the messages and insights it wishes to deliver.
2. This paper contributes to privacy-preserving text-to-image diffusion models, which are practically significant for preventing copyright risks.
3. The proposed metric’s efficiency (denoising-free and fast) makes it appealing for large-scale model auditing. 
4. The paper introduces a new theoretical and empirical perspective on diffusion model memorization by analyzing the anisotropy of log-probability. The conceptual shift from norm magnitude to angular alignment in the low-noise regime is novel and well-motivated. The derived connection between isotropy, anisotropy, and curvature sharpness is mathematically supported and empirically verified.
5. Experimental results are comprehensive, covering multiple Stable Diffusion versions, standard benchmarks, and ablations. Comparisons include strong baselines and standard metrics.

### Weaknesses
1. The paper could benefit from additional discussion on whether the proposed metric generalizes across architectures beyond SD v1.4/v2.0.
2. Although Appendix A.4 shows some robustness, the weights are empirically tuned per model, suggesting potential calibration issues when scaling to new settings.
3. The proposed cosine-similarity measure is intuitive but may be sensitive to normalization choices. A sensitivity analysis on score normalization or noise schedule parameters would strengthen the robustness claims.
4. Despite being well-organized, there are several equations in the paper that remain unlabelled on page 5 (between equations 5 and 6).
5. The discussion on Figure 2’s overlap could benefit from quantitative metrics (e.g., KL divergence between distributions).

### Questions
See the above strengths and weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
