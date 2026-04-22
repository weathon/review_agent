# Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Classifier-free guidance (CFG) has become an essential component of modern conditional diffusion models. Although highly effective in practice, the underlying mechanisms by which CFG enhances quality, detail, and prompt alignment are not fully understood. This paper presents a novel perspective on CFG by analyzing its effects in the frequency domain, showing that low and high frequencies have distinct impacts on generation quality. Specifically, low-frequency guidance governs global structure and condition alignment, while high-frequency guidance mainly enhances visual fidelity. However, applying a uniform scale across all frequencies---as is done in standard CFG---leads to oversaturation and reduced diversity at high scales and degraded visual quality at low scales. Based on these insights, we propose frequency-decoupled guidance (FDG), an effective approach that decomposes CFG into low- and high-frequency components and applies separate guidance strengths to each component. FDG improves image quality at low guidance scales and avoids the drawbacks of high CFG scales by design, i.e., retaining the benefits of CFG over unguided generation without its common failures. Through extensive experiments across multiple datasets and models, we demonstrate that FDG consistently enhances sample fidelity while preserving diversity, leading to improved FID and recall compared to CFG, establishing our method as a plug-and-play alternative to standard classifier-free guidance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper analyzes the role of Classifier-Free Guidance (CFG) in conditional diffusion models, specifically examining its differential impact on low- and high-frequency components during image generation. The authors posit that an excessively high CFG scale applied to the low-frequency domain results in color oversaturation and reduced sample diversity.

To address this, the paper introduces Frequency Decoupled Guidance (FDG), a novel method that utilizes the invertible and linear Laplacian pyramid transformation to decompose the image into distinct frequency bands. FDG then applies different guidance scales to these components—notably, a lower scale for low-frequencies and a higher scale for high-frequencies. The efficacy of FDG is evaluated against standard CFG across various class-conditional and text-to-image generation models.

### Strengths
1. This work provides an initial analysis that distinguishes the functional roles of the low- and high-frequency components within the CFG mechanism.

2. The proposed FDG method is training-free. It can be readily integrated into any pre-trained conditional diffusion model with minimal implementation overhead.

### Weaknesses
1. Insufficient Evidence: The central claim regarding the distinct roles of CFG's low- and high-frequency components is insufficiently substantiated. It appears to rest primarily on a few qualitative samples presented in Figure 2, which is not robust enough to support such a definitive conclusion.

2. Lack of Clarity: The paper suffers from a significant lack of clarity regarding critical experimental settings.
- The resolution of the ImageNet models used (e.g., for EDM) is not specified, though it is presumed to be $512 \times 512$.
- For a paper centered on guidance, the omission of specific guidance scales in the main text is a critical flaw. Key information is missing: The FDG scales ($w_{low}$, $w_{high}$) used for the visualization in Figure 1.The corresponding $w_{high}$ value used in Figures 3 and 4.
- The guidance scales used to generate the quantitative results in Tables 1 and 2. While this information is reportedly located in the supplementary material, its absence from the main body severely hinders readability and prevents a clear assessment of the paper's claims.

3. Unfair Comparison: A valid comparison between CFG and FDG necessitates evaluating both methods at their respective optimal guidance scales. However, the scales selected for the CFG baselines in Tables 1 (EDM) and 2 (Stable Diffusion) do not appear to align with those from official repositories or standard evaluation practices. The qualitative examples also seem to use suboptimal scales for CFG, potentially skewing the comparison.

4. Inconsistent Performance: The reported baseline performance metrics are inconsistent with established benchmarks. For instance, the official EDM2-S and EDM2-XXL models achieve FIDs of 2.23 and 1.81 (scales 1.40/1.20) and FIDs of 2.56 and 1.91 (even without CFG), respectively. This paper, however, reports much poorer CFG baseline FIDs of 9.77 and 8.65. Any deviations in the evaluation protocol (e.g., the number of images used for FID calculation) that might explain this significant discrepancy are not described.

5. Potential Cherry-Picking of Parameters: The use of different guidance scales for the Stable Diffusion model when reporting metrics in Table 1 versus Table 2 is highly suspect. This suggests that the scales may have been selected via parameter search to find specific, narrow settings where FDG artifactually outperforms CFG on a given metric, rather than demonstrating robust and general superiority.

6. Omission of Critical Metrics: The paper omits the Inception Score (IS). IS is generally more sensitive to sample fidelity, whereas FID is more sensitive to diversity. It is a known phenomenon that FID can often be improved at the expense of IS. The provided precision (lower) and recall (higher) metrics already suggest such a trade-off (sacrificing fidelity for diversity). Without the IS metric, it is impossible to verify if FDG genuinely improves generation quality or simply trades fidelity for a better FID score.

7. Missing Comparison and Overclaiming: The paper fails to compare against relevant prior art, most notably CFG++. Consequently, the claim that this work is the first approach to improve image quality at low CFG scales is an overstatement, as CFG++ addresses a very similar problem.

### Questions
1. Could you provide a more rigorous validation of the distinct roles of low/high-frequency components beyond the few samples in Figure 2? Furthermore, to improve clarity, could you please state the exact guidance scales (e.g., $w_{low}$, $w_{high}$, and baseline $w$) used for all figures and tables directly in the main paper?

2. The reported baseline FID scores for EDM models (e.g., 9.77) are significantly worse than those in the official repository (e.g., 2.23). Can you explain this discrepancy? How did you determine the "optimal" guidance scale for the CFG baseline, as the one chosen does not seem to reflect its best performance?

3. Why were different guidance scales used for the Stable Diffusion model in Table 1 versus Table 2? This raises concerns about parameter cherry-picking to favor FDG on specific metrics.

4. The precision/recall scores suggest FDG may be trading fidelity for diversity. Can you provide the Inception Score (IS) results for your experiments to confirm that fidelity is not degraded?

5. How does FDG conceptually and empirically compare to CFG++? Why was this relevant prior work, which also addresses quality at low guidance scales, omitted from your comparisons?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a well-known and critical trade-off in conditional diffusion models that utilizes the classifier-free guidance (CFG). The authors present a novel analysis of this problem by decomposing the CFG update signal in the frequency domain. The key insight is that low-frequency (LF) and high-frequency (HF) components of the guidance signal have distinct and opposing effects. The authors find that LF-CFG governs global structure and condition alignment. However, LF-CFG is the main cause of diversity loss and color oversaturation. On the other hand, HF-CFG primarily enhances visual fidelity and fine details. Based on this, the authors propose frequency-decoupled guidance (FDG) that decomposes the CFG signal into LF and HF parts and amplifies the HF-CFG signal to generate high-fidelity samples while maintaining the diversity. Experiments on diverse diffusion models prove the effectiveness and generalizability of the FDG.

### Strengths
- **Simplicity**: FDG is a plug-and-play and training-free method that only introduces a few additional lines of code with negligible computational cost. This makes FDG a versatile method for general conditional diffusion models. 

- **Comprehensive Evaluation**: The authors evaluate FDG's compatibility with various base diffusion models (EDM, SD, etc.), various samplers (DDIM, DPM++, etc.), distilled models (SDXL-Lightning), and other guidance-improvement techniques (CADS, APG, FreeU), showing the robustness of the method.

- **Principled Analysis on Guidance**: The authors analyze CFG from the perspective of the frequency domain. In addition, the authors provide a theoretical analysis of why auto guidance (AG) has shown superior performance compared to CFG. This paves the way to analyze the effectiveness of the guidance, which previously heavily relied on the output sample quality.

### Weaknesses
- **Discrepancy in FID Reporting**: There appears to be a mismatch between the reported baseline CFG FIDs for models in Table 1 and the FIDs commonly reported in their respective original papers, like EDM2-S (9.77 vs. 2.23), EDM2-XXL (8.65 vs. 1.81), and DiT-XL/2 (9.31 vs. 2.27). While I'm not comparing all the other models, it seems inconsistency of FIDs may stem from the arbitrary guidance scale (e.g., DiT-XL utilizes cfg=1.5 for the best FID, but the paper reports FID with cfg=2.0 according to Table 12) and incomprehensive evaluation metric (i.e., the paper did not report IS which is common metric to measure the sample quality in c2i image synthesis). 

- **Details of Evaluation**: In addition to the above weakness, the evaluation setup is not principled. Specifically, for Table 1, baseline CFG is set to $w_{high}$ of FDG, while for Table 2, baseline CFG is set to $w_{low}$. I suspect that this inconsistency may stem from the fact that Table 1 usually reports the metrics regarding the diversity, where low CFG is beneficial, while Table 2 usually reports the metrics regarding the sample quality, where high CFG is beneficial. This suggests the baselines are not being evaluated fairly or in their optimal settings. This inconsistent methodology severely damages the faithfulness of the paper's comparisons.

- **Limited Applicability to Diffusion Models**: The Recent visual generative modeling community actively researches autoregressive models (e.g., LlamaGen, VAR) and masked generative models (MaskGIT, MaskGIL) for compatibility and seamless integration with LLMs.
which often utilize discrete visual tokens and perform CFG in a logit space.

### Questions
- It seems in Figure 2, FDG also shows color saturation in the dog in the bottom right. 
- The method's robustness to the frequency decomposition process itself is not adequately addressed. FDG relies on separating HF and LF signals, a process that is highly sensitive to the specifics of the low-pass filter (e.g., the Gaussian kernel's size and $\sigma$). The paper lacks an ablation study on how different kernel parameters would affect FDG's performance. 
- $w_{low}$ and $w_{high}$ are key hyperparameters to balance the frequency components. Is there a principled approach or a systematic heuristic for tuning these weights to best navigate the sample quality vs. diversity trade-off?

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
This paper empirically observes that applying strong classifier-free guidance (CFG) to low-frequency components often leads to over-saturation and reduced sample quality. To address this, the authors propose decomposing each predicted image into low- and high-frequency components at every inference step and applying a stronger CFG scale to the high-frequency components in the frequency domain. This approach mitigates the inherent trade-off in CFG between diversity and quality, where low CFG scales yield high diversity but low quality, and high CFG scales produce low diversity but high quality, while also alleviating the saturation problem observed at high guidance scales. The method achieves this improvement without increasing the number of model predictions required per sampling step, using only a simple Laplacian transform for frequency separation.

### Strengths
- This paper tackles one of the key issues in CFG, saturation at high guidance scales, and provides a principled way to mitigate it.
- Unlike other guidance methods that require multiple model predictions (often 2×2 = 4 predictions) to combine different forms of guidance, the proposed approach enhances sample quality without additional model calls, maintaining computational efficiency.
- The use of a lightweight frequency-domain decomposition (via Laplacian transform) makes the approach conceptually clean and easy to integrate with existing diffusion frameworks.

### Weaknesses
- Lack of proper citation. In the Introduction (L084-L098), the discussion lacks references, even though prior works have already analyzed diffusion processes in the frequency domain, such as FreeU (Si et al., 2024) and [a]. These should be cited appropriately to situate this work in the existing literature.
- Ambiguity in frequency computation. It is unclear how the frequency components are calculated. Since intermediate predictions are inherently noisy, it should be explicitly stated whether the decomposition is applied directly to the noisy intermediate prediction or to the denoised $\hat{x}_0$. This detail appears only in the pseudocode or Supplementary Material D, which makes it difficult for readers to follow. It would improve clarity to include this explanation in the main method section.
    
    Moreover, as pointed out by [a], high-frequency components tend to be destroyed in high-noise regions during the diffusion process. The paper should therefore discuss how meaningful frequency separation is achieved when the predicted $\hat{x}_0$. may already lose high-frequency information in these regions.
    
- Missing comparison with self-attention guidance. The paper should discuss and cite Self-Attention Guidance [b]. In [b], the authors blur high-frequency regions in the final image to create degraded samples for guidance, improving quality, conceptually, the opposite of enhancing high-frequency guidance as proposed here. Interpreting [b] from the frequency-domain perspective of this work could yield valuable insights, and such a discussion would enrich the paper.
- Experimental details are overly deferred to the Supplementary Material. Important information, such as experimental settings, resolution for ImageNet evaluation, and key hyperparameters (e.g., CFG scale used for baselines, FID10K vs FID50K), should appear in the main paper. While visual results are helpful, some could be moved to the supplement to make room for these critical quantitative details.

[a] Dieleman, Sander, Diffusion is spectral autoregression, 2024

[b] Hong, Susung, et al., Improving sample quality of diffusion models using self-attention guidance. ICCV 2023

### Questions
I believe the overall presentation could be significantly improved. The number of qualitative result figures could be reduced, and the details discussed in the weaknesses section should be more thoroughly incorporated into the main text to make the paper more solid and convincing.

### Soundness
3

### Presentation
2

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
The authors analyze classifier-free guidance in diffusion models from the perspective of frequency domain. They argue that by decoupling guidance for higher-freqencies from lower-frequencies by using different weights in each, they can achieve the best performance. An extensive set of experiments is provided.

### Strengths
1. I like the presentation of the paper. The motivation, figures, and explanations are clear.
2. The empirical results are suggesting that the proposed method significantly improves the standard classifier-free guidance
3. I appreciate the authors take on explaining the performance of autoguidance and guidance in limited time intervals. Seems convincing to me.

### Weaknesses
The biggest weakness of the paper, in my opinion, is around the question of whether the comparison with CFG is fair. The authors say (line 310): "hyperparameters used for each experiment are given in Appendix D", and they are provided in Table 12. However, the procedure for selecting these hyperparameters: "the guidance scales are selected in the same way practitioners typically choose CFG values for a model, i.e., generating a few samples and visually inspecting them" makes me doubt that the comparison is fair. The reported improvements for multiple (almost all) metrics in tables 1, and 2 are extremely large, and much larger than what I would expect based on Figure 5. The paper does not convince me that the hyperparameters were properly tuned for CFG to produce the results in Tables 1, and 2. I have no reason to believe that the performance gaps would be much smaller if I tuned the parameters for CFG. An experiment that would convince me would be: perform a sweep over CFG weights, and compare the best performing ones with FDG.

I think that the proposed idea is very promising, and I am prepared to raise my score if the authors present convincing evidence that the reported improvements are not an artifact of poorly tuned CFG weights.

### Questions
1. Can the authors perform a quantitative comparison (as in Tables 1, 2) but find the best parameters for CFG? Then, in qualitative comparisons (e.g. Figures 3 and 4), use the CFG parameter that actually performed best?
2. It looks like FDG hurts precision, which aims to measure "how realistic" the samples are. Figure 5 shows that the proposed approach is consistently worse than the standard CFG. Similarly, figure 11. Can the authors comment on this? It feels like an important metric that shouldn't be disregarded.
3. I don't think I understand Figure 6. To me, it looks like the blue curve (standard CFG) achieves the best performance, and the green curve corresponds to the proposed approach (stronger guidance for high frequencies than low). What choice of parameters does the dotted line correspond to? Either way, the figure clearly suggests that the same performance can be obtained with the standard CFG.
4. The authors say that the method "does not incur any noticeable computational overhead". Can this statement be made more precisely? Is the overhead 10% or 0.01%? Just to be clear, I am not asking for any additional experiments here, just curious.
5. Usually, increasing the CFG weight makes FID worse, but improves the inception score (IS). People usually evaluate multiple CFG weights and look at the "pareto curves", i.e. FID on one axis and IS on the other. Can the authors produce such a curve for regular CFG, and show where on the FID/IS graph the proposed approach lands?

### Soundness
2

### Presentation
3

### Contribution
3
