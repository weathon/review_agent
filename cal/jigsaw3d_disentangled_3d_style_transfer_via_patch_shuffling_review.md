=== CALIBRATION EXAMPLE 46 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: a jigsaw-based approach for disentangled 3D style transfer. The abstract clearly states the problem, limitations of existing methods, and the proposed solution (jigsaw operation, reference attention, multi-view diffusion, style baking). Claims of high style fidelity, lower latency, and generalization are supported in the paper. The abstract is concise and matches the paper’s content.

### Introduction & Motivation
The introduction effectively motivates the problem of 3D stylization, highlighting the lack of paired style–texture data and the shortcomings of current methods (entanglement, per-asset optimization). The insight that local patches can carry style and that shuffling/masking can disentangle style from content is well-grounded in prior work (Wang et al., 2023) and the authors’ own analysis (Fig. 3). Contributions are clearly listed and align with the paper’s content.

### Methods
- **3.1 Style‑Texture Pairs Creation**: The jigsaw operation (shuffling + masking) is a clever way to synthesize training data from existing 3D assets. The quantitative analysis in Fig. 3 and the proof in Appendix A.2 support the claim that shuffling preserves style statistics while destroying semantics. However, the choice of mask ratio (0–0.25) and patch size is justified empirically but lacks a deeper theoretical justification. The impact of masking on higher‑order style statistics (beyond mean/variance) is not discussed.
- **3.2 Multi‑View Style Generation**: The architecture integrates geometry injection (normal/position maps via a T2I‑Adapter‑like encoder) and style injection via reference attention. Several important details are missing or unclear:
    - How exactly is the geometry condition injected via “additive feature modulation”? The description is too high‑level for reproducibility.
    - The “row‑wise self‑attention” for multi‑view attention is not explained; a citation (Li et al., 2024) is given, but the adaptation should be described briefly.
    - The reference attention uses features \(f_{\text{ref}}\) extracted from a pre‑trained U‑Net. Which layers are used? Are they pooled or concatenated? This affects style representation.
    - Text conditioning: What captions are used during training (e.g., from Objaverse)? How does text dropout interact with style conditioning?
    - The log‑SNR offset adjustment by \(\log(n)\) (where \(n=6\)) is mentioned without justification; its role in multi‑view consistency should be explained.
- **3.2.1 3D Style Baking**: The description is high‑level but sufficient for understanding the pipeline. More details on visibility‑aware reprojection and seam‑aware blending are needed for full reproducibility (though some are in Appendix A.9).

Overall, the method is novel and well‑motivated, but key implementation details are omitted or under‑specified. This could hinder reproduction and fair comparison.

### Experiments & Results
- **Baselines and Metrics**: The choice of baselines (StyleTex, MV‑Adapter, 3D‑style‑LRM) is appropriate. Style metrics (Gram, AdaIN) and CLIP score for disentanglement are standard. However, the evaluation lacks:
    - A metric for multi‑view consistency (e.g., LPIPS between views or texture consistency score).
    - A metric for content/geometry preservation (e.g., comparing rendered normals before/after stylization).
    - Hardware specifications for runtime comparisons (Table 1). Stating GPU type is essential for fair comparison.
- **Quantitative Results**: Table 1 shows superior style fidelity (lower Gram/AdaIN) and competitive CLIP scores. The runtime advantage over optimization‑based methods is clear. However, the CLIP score interpretation is nuanced: StyleTex uses style text descriptions, giving it an unfair advantage. The authors note this, but the comparison should be framed accordingly.
- **Qualitative Results**: Figures 4, 8, 9 demonstrate strong style transfer and geometry preservation. The multi‑object scene results (Fig. 6) are convincing. Appendix results (partial stylization, tileable textures) support the generalization claims.
- **Ablation Study**: Figure 5 qualitatively shows the necessity of jigsaw in both training and inference. However, quantitative ablations (e.g., metrics for different jigsaw settings, mask ratios, patch sizes) are missing from the main paper; they appear only in the appendix (Figs. 13, 14). This weakens the main paper’s evidence.
- **Limitations**: The discussion of failure cases (text/symbols) is honest and attributed to the SDXL backbone. However, other limitations (e.g., handling of global style patterns, dependence on the quality of UV mapping) are not addressed.

### Writing & Clarity
The paper is generally well‑written, with clear figures and logical flow. However, the methods section suffers from ambiguous descriptions (as noted above). Some technical terms (e.g., “row‑wise self‑attention”) are not explained. The appendix is extensive but should be referenced more explicitly in the main text for critical details.

### Limitations & Broader Impact
The authors acknowledge the inability to transfer fine‑grained patterns (text/symbols) and attribute it to SDXL. They also provide an ethical statement about potential misuse (e.g., copyright infringement). However, broader limitations—such as the assumption that style is purely local (shuffling may break global style patterns), sensitivity to UV parameterization, and potential biases in the training data—are not discussed. The societal impact statement is adequate.

## Overall Assessment
Jigsaw3D presents a novel and effective approach for 3D style transfer, with a clever jigsaw operation to disentangle style from content and a well‑designed multi‑view diffusion pipeline. The paper demonstrates state‑of‑the‑art style fidelity, significant speed improvements over optimization‑based methods, and promising generalization to various applications. However, the paper has notable weaknesses: (1) insufficient methodological details for reproducibility, (2) incomplete evaluation (missing multi‑view consistency and content‑preservation metrics), and (3) under‑developed ablation studies in the main text. Addressing these concerns—especially by providing clear architectural details and more comprehensive evaluation—would strengthen the paper considerably. As it stands, the core idea is strong and the results are compelling, making it a solid contribution that could be accepted to ICLR after revisions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces JIGSAW3D, a framework for 3D style transfer that disentangles style from content via a novel jigsaw operation (spatial patch shuffling and masking) applied to 2D reference images. This operation suppresses semantic content while preserving stylistic statistics, enabling the creation of pseudo-paired training data from existing textured 3D assets. The method trains a multi-view diffusion model with a dedicated reference-attention module to inject style features, generating view-consistent stylized renderings that are baked into seamless textures. The approach avoids per-asset optimization, supports partial and multi-object stylization, and demonstrates strong generalization.

### Strengths
1. **Innovative disentanglement strategy**: The jigsaw operation is a clever, well-motivated technique to isolate style from content. The paper provides both quantitative analysis (Figure 3) and a theoretical proof (Appendix A.2) showing that shuffling preserves first- and second-order style statistics while destroying semantic structure.
2. **Comprehensive empirical validation**: Experiments are thorough, comparing against multiple state-of-the-art baselines (StyleTex, MV-Adapter, 3D-style-LRM) on collected and WikiArt datasets using style fidelity (Gram, AdaIN) and disentanglement (CLIP) metrics. Results show superior or competitive performance with significantly lower inference time (~40s vs. 15min for optimization-based methods).
3. **Versatility and practical applications**: The method demonstrates compelling extensions beyond single-object stylization, including partial reference stylization, multi-object scene consistency, and tileable texture generation (Figures 6, 15, 16), highlighting its potential for real-world content creation.
4. **Clear ablation studies**: Ablations on the jigsaw module (Figure 5), mask ratio, and patch size (Appendix A.5) validate design choices and provide insights into the method's behavior.

### Weaknesses
1. **Limitation in preserving fine-grained patterns**: The paper acknowledges that the method struggles with text, symbols, or highly structured patterns (Figure 7), attributing this to the Stable Diffusion XL backbone. This is a notable drawback for practical applications where such details are important.
2. **Marginal quantitative gains in some metrics**: While style fidelity metrics (Gram, AdaIN) show clear improvements, the CLIP scores (measuring content disentanglement) are only competitive, not best (Table 1). The absolute differences in some metrics are modest, raising questions about the practical significance of the improvements.
3. **Insufficient exploration of style diversity**: The evaluation primarily uses artistic styles; the method's performance on more diverse, non-artistic styles (e.g., photorealistic textures, material finishes) is not thoroughly tested. The reliance on SDXL may also bias results toward its training distribution.
4. **Light theoretical grounding for masking**: The role of random masking in the jigsaw operation is motivated empirically but lacks a deep theoretical justification compared to shuffling. The ablation on mask ratio (Figure 13) shows sensitivity, but a more principled analysis is needed.

### Novelty & Significance
**Novelty**: The core idea of using patch shuffling and masking (jigsaw operation) for style-content disentanglement in 3D stylization is novel. While patch shuffling has been explored in 2D style transfer, its application to 3D, combined with masking and integration into a multi-view diffusion pipeline with reference attention, constitutes a clear original contribution. The method also innovates by generating pseudo-paired training data from existing 3D assets, circumventing the need for curated style-texture datasets.

**Significance**: The work addresses key challenges in 3D stylization—disentanglement, multi-view consistency, and scalability—and demonstrates a fast, feed-forward approach that avoids test-time optimization. This could significantly impact areas like game development, VR/AR, and digital content creation where efficient, controllable stylization is valuable. The code release and reproducibility efforts further enhance its potential impact.

### Suggestions for Improvement
1. **Strengthen evaluation with user studies and broader style types**: Include a perceptual user study to assess visual quality and style fidelity beyond automated metrics. Also, test on a wider range of reference styles, including non-artistic textures (e.g., fabrics, metals) to better demonstrate generalization.
2. **Deeper analysis of the jigsaw operation**: Provide a more rigorous theoretical or empirical analysis of why masking (beyond shuffling) is beneficial, perhaps linking it to masked autoencoder principles or style statistics at different patch granularities.
3. **Address fine-grained pattern failure**: Propose and evaluate potential mitigations for the text/symbol limitation, such as incorporating dedicated token-based controls, using a more capable backbone, or post-processing refinements. This would greatly improve practical utility.
4. **Clarify comparisons and limitations**: More explicitly discuss the trade-offs between the proposed method and baselines (e.g., quality vs. speed, flexibility vs. detail preservation). Also, discuss computational requirements (GPU memory, training time) for better reproducibility assessment.
5. **Enhance visualization and clarity**: Some figures (e.g., Figure 2) are dense and could be simplified. Ensure all references to figures/tables in the text are correct (noting potential parser artifacts). The appendix is extensive but could be better integrated into the main narrative for flow.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No comparison with Style3D and other recent attention-based methods.** The paper mentions Style3D but does not include it in quantitative or qualitative comparisons. Without this, the claim of state-of-the-art performance against training-free attention-based methods is not fully substantiated.
2. **Missing ablation on the multi-view attention component.** The contribution of the multi-view attention mechanism (borrowed from prior work) to cross-view consistency is not isolated. An ablation study removing it is necessary to validate its importance relative to the novel jigsaw and reference-attention modules.
3. **No evaluation of multi-view consistency with standard metrics.** Metrics like LPIPS between views or dedicated consistency scores (e.g., MVDream's consistency metric) are absent. The claim of "multi-view consistency" relies only on qualitative visuals, which is insufficient for ICLR.
4. **No runtime breakdown or comparison of generation vs. baking stages.** The claimed low latency (~40s) is not analyzed. A breakdown of time spent in multi-view generation versus UV baking, compared to baselines, is needed to properly assess efficiency gains.

### Deeper Analysis Needed (top 3-5 only)
1. **Lack of analysis on the jigsaw operation's disentanglement efficacy across diverse semantic content.** The analysis in Figure 3 uses a single metric (classification score) on limited data. A rigorous analysis is needed to show that jigsawing consistently destroys semantics (e.g., using CLIP similarity to original content) while preserving style statistics for references with strong, structured objects (e.g., faces, text).
2. **No quantitative evaluation of partial reference stylization.** The claim of supporting partial references is only shown qualitatively. A quantitative study measuring style fidelity and consistency when using cropped vs. full references is required to demonstrate robust inference from partial cues.
3. **Insufficient analysis of style-content entanglement in baselines vs. our method.** The CLIP score is used as a proxy for disentanglement, but a deeper analysis (e.g., visualizing attention maps or feature spaces) is missing to show *how* the jigsaw operation prevents semantic leakage compared to baselines.

### Visualizations & Case Studies
1. **Side-by-side visualization of UV texture maps and baking artifacts.** The paper shows final renders but not the baked texture maps. Displaying UV maps for our method and baselines would reveal seam handling, blurriness, and inconsistencies critical for assessing practical utility.
2. **Systematic visualization of failure cases beyond text/symbols.** The admitted failure on text/symbols is too narrow. Case studies should show failures on complex geometric details, high-frequency patterns, or stylization of thin structures to properly delineate method limitations.

### Obvious Next Steps
1. **Ablation study on jigsaw parameters during inference.** The paper ablates mask ratio and patch size only during training (Appendix). A systematic study varying these parameters *at inference time* is essential to understand sensitivity and provide guidance for users.
2. **Ablation on the choice of pre-trained diffusion backbone.** The use of SDXL is not justified. An experiment comparing with other backbones (e.g., SD 1.5, Stable Diffusion 3) would show how dependent the performance is on the specific model choice.
3. **User study to validate perceptual quality.** The metrics used (Gram, AdaIN, CLIP) are proxy measures. A user study rating style fidelity, consistency, and overall quality is necessary to align claimed advantages with human judgment, a standard in graphics/vision conferences.

# Final Consolidated Review
## Summary
This paper introduces JIGSAW3D, a framework for 3D style transfer that disentangles style from content via a novel jigsaw operation—spatial patch shuffling and masking of reference images. This enables the creation of pseudo-paired training data from existing 3D assets, which is used to train a multi-view diffusion model with a reference-attention module for fast, view-consistent stylization without per-asset optimization.

## Strengths
- **Innovative disentanglement strategy via jigsaw operation:** The paper provides quantitative analysis (Figure 3) and a theoretical proof (Appendix A.2) demonstrating that shuffling preserves style statistics (mean, variance) while destroying semantic content, enabling effective training data synthesis and style isolation.
- **Strong empirical performance and efficiency:** Experiments show superior style fidelity (lower Gram and AdaIN metrics) and competitive disentanglement (CLIP scores) compared to baselines like StyleTex and MV-Adapter, with significantly faster inference (~40 seconds vs. 15 minutes for optimization-based methods).
- **Versatility in applications:** The method generalizes to partial reference stylization, multi-object scene styling, and tileable texture generation (Figures 6, 15, 16), demonstrating practical utility beyond single-object transfer.

## Weaknesses
- **Limitation in preserving fine-grained patterns:** As acknowledged in the paper (Figure 7), the method struggles with text, symbols, or highly structured patterns due to limitations of the SDXL backbone, which reduces its applicability in scenarios requiring detailed stylistic elements.
- **Insufficient quantitative evaluation of multi-view consistency:** The paper lacks standard metrics (e.g., LPIPS between views or dedicated consistency scores) to validate cross-view consistency, relying only on qualitative visuals; this omission undermines the claim of robust view consistency.
- **Under-specified methodological details:** Key implementation aspects, such as the exact mechanism for geometry injection via "additive feature modulation" and the extraction process for reference features \(f_{\text{ref}}\) in the attention module, are described at a high level, which could hinder reproducibility.

## Nice-to-Haves
- A user study to assess perceptual quality and style fidelity beyond automated metrics.
- Ablation study on the multi-view attention component to isolate its contribution to cross-view consistency.
- Evaluation on a broader range of non-artistic styles (e.g., photorealistic textures) to demonstrate generalization.

## Novel Insights
The core novelty lies in adapting patch shuffling and masking—previously explored in 2D style transfer—to 3D stylization for explicit style-content disentanglement, enabling scalable training from unpaired 3D assets. The integration of this jigsaw operation with a multi-view diffusion pipeline and reference-attention mechanism offers a feed-forward alternative to optimization-based methods, balancing speed and quality.

## Suggestions
- Incorporate multi-view consistency metrics (e.g., LPIPS) into the evaluation to quantitatively substantiate view-consistent claims.
- Provide more detailed architectural descriptions in the main text or appendix for critical components like geometry injection and reference feature extraction to enhance reproducibility.
- Conduct a quantitative analysis of partial reference stylization (e.g., measuring style fidelity with cropped vs. full references) to better support the generalization claim.

# Actual Human Scores
Individual reviewer scores: [8.0, 2.0, 4.0, 4.0]
Average score: 4.5
Binary outcome: Reject
