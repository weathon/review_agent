=== CALIBRATION EXAMPLE 52 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: a 3D style transfer method using patch shuffling and masking (jigsaw) for disentanglement. The abstract clearly states the problem (entanglement, per-scene optimization), the key idea (jigsaw operation to suppress semantics and isolate style), the integration into a multi-view diffusion pipeline, and the claimed benefits (fast, view-consistent, versatile). The abstract’s claims are supported by the content, but the claim of "substantially lower latency" is relative; Table 1 shows ~40s vs. 15min for StyleTex, but other feed-forward baselines are similarly fast. This should be nuanced.

### Introduction & Motivation
The introduction effectively motivates the problem: lack of paired 3D style data, limitations of existing methods (entanglement, optimization cost). The proposed solution—using a jigsaw transform to create semantics-agnostic style references—is well-motivated. Contributions are clearly listed. However, the introduction could better highlight the novelty of applying jigsaw to 3D stylization and the reference-attention mechanism, as these are key differences from prior work like StyleAdapter (Wang et al., 2023) which uses shuffling in 2D.

### Methods
#### 3.1 Style-Texture Pairs Creation
The jigsaw operation (shuffling + masking) is the core innovation for disentanglement. The motivation is plausible: destroying global structure preserves local style statistics. Figure 3 provides quantitative support, but the description is confusing: the caption says Gram similarity increases for N≤8, but the plot shows a peak at N=8 then a drop. This must be clarified. The choice of N=8 and mask ratio p (0–0.25) is justified empirically, but a deeper analysis of why these values work would strengthen the section. The training procedure (using different views of the same object for reference and target) is clever for creating pseudo-pairs, but it assumes that the style is consistent across views of the same object, which may not hold if the object’s texture varies. This assumption should be discussed.

#### 3.2 Multi-View Style Generation
The architecture integrates geometric cues (normal/position maps via an adapter) and style cues (via reference attention). The reference attention module is clearly defined (Eq. 3). However, several details are missing, hindering reproducibility:
- The "reference U-Net" is pre-trained (likely SDXL), but which layers are used for \( f_{\text{ref}} \)? The text says "intermediate hidden state features from the self-attention layers," but specific layers are not specified.
- Why is timestep \( t=0 \) used for the reference U-Net? This is non-standard; diffusion models typically operate on noisy latents. Justification is needed.
- The style U-Net is built on SDXL, but it's unclear if it is fine-tuned from SDXL weights or trained from scratch. The training details mention "conditioning dropout," but how the image condition is dropped (null embedding?) is not explained.
- The multi-view attention uses "row-wise self-attention" (Li et al., 2024), but a brief explanation would help readers unfamiliar with this design.

#### Proof of Style Statistics Preservation (Appendix A.2)
The proof shows that shuffling preserves pixel-wise mean and variance. However, style transfer relies on deep feature statistics (e.g., Gram matrices), not pixel statistics. The proof is therefore misleading and does not support the claim that style is preserved under shuffling. The empirical results (Fig. 3) are more relevant, but the proof should either be removed or reframed as a simple observation about low-level statistics, with a note that deep feature preservation is empirically validated.

### Experiments & Results
#### 4.1 Qualitative and Quantitative Comparisons
Qualitative results (Fig. 4) show strong style fidelity and geometric consistency compared to baselines. However, some baseline adaptations are unclear: e.g., how is MV-Adapter, a text-conditioned model, adapted for image-guided style transfer? The authors should specify the conditioning method used for each baseline to ensure fairness.

Quantitative metrics (Gram, AdaIN, CLIP) are standard. The results in Table 1 show superior style fidelity (lower Gram/AdaIN) and competitive CLIP scores. The CLIP score is used to measure style-content disentanglement (lower is better), but it is computed between generated views and the reference image. This may not fully capture disentanglement; a better metric might compare semantic similarity to the original 3D object’s content. The authors note that StyleTex uses style text descriptions, giving it an advantage in CLIP. This is a reasonable explanation.

Computational time: The claim of "substantially lower latency" is supported relative to optimization-based StyleTex (15min vs. 40s), but other feed-forward baselines (MV-Adapter, 3D-style-LRM) have similar inference times (~35–40s). The advantage is thus not in speed over all baselines, but in quality.

#### 4.2 Ablation Study
The ablation on the jigsaw module (Fig. 5) is qualitative only. Quantitative results on a validation set would strengthen the analysis. The appendix contains additional ablations on patch size and mask ratio, which are important but relegated to the appendix. Key findings (e.g., training patch size 64×64, inference size 128×128, mask ratio 0–0.25) should be summarized in the main text.

#### 4.3 More Applications and Limitations
Applications (multi-object scenes, partial stylization, tileable textures) demonstrate versatility and are impressive. Limitations are honestly discussed: difficulty with fine-grained patterns (text/symbols) due to SDXL, and potential UV baking issues (addressed in Appendix A.9). The limitation regarding text is acceptable given the backbone, but the authors could discuss whether this is a fundamental limitation of the jigsaw approach or the diffusion model.

### Writing & Clarity
The paper is generally well-written, but some sections lack clarity:
- Figure 2 is cluttered and hard to parse; a cleaner diagram would help.
- The method description jumps between training and inference; a more structured flow would improve readability.
- Terminology: "style U-Net" vs. "reference U-Net" can be confusing; consistent naming is needed.
- The appendix is extensive but some figures (e.g., Fig. 11-12) are too small and may be illegible in print.

### Limitations & Broader Impact
The limitations section acknowledges failure cases (fine patterns, UV baking) and attributes them appropriately. The ethical statement covers potential misuse (IP infringement, misinformation) and emphasizes responsible use. The reproducibility statement promises code and pre-trained weights, which is good. However, the societal impact of enabling easy 3D stylization could be expanded, e.g., implications for digital art, game development, and potential for deceptive content.

### Overall Assessment
This paper presents a novel and effective method for 3D style transfer. The core idea—using a jigsaw operation to disentangle style from content—is simple yet powerful, and the integration into a multi-view diffusion pipeline yields high-quality, consistent results. The method outperforms existing approaches in style fidelity and is fast (feed-forward). However, the paper has weaknesses: the proof of style preservation is misaligned with actual style metrics, some methodological details are unclear, and the evaluation could be more rigorous (e.g., user study, better disentanglement metrics). Despite these issues, the contribution is significant and likely of interest to the ICLR community. With revisions addressing the concerns above, the paper would be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Jigsaw3D, a method for 3D style transfer that aims to disentangle style from semantic content in a reference image via a "jigsaw" operation (patch shuffling and masking). The processed reference is used to condition a multi-view diffusion model, trained on pseudo-paired data synthesized from textured 3D assets, to generate view-consistent stylized renderings. These are then baked into a texture map. The approach is fast (feed-forward) and demonstrates applications in partial stylization, multi-object scenes, and tileable texture generation.

### Strengths
1. **Novel and Well-Motivated Core Mechanism**: The proposed jigsaw operation is a simple yet effective and well-analyzed technique to suppress semantic content while preserving style statistics (e.g., color, texture). The paper provides quantitative analysis (Fig. 3) showing the trade-off between content suppression (classification score drop) and style preservation (Gram matrix similarity), solidly motivating the design choice (N=8 divisions). This addresses a key challenge in style transfer.
2. **Strong Empirical Results and Efficiency**: The method achieves state-of-the-art or competitive results on standard style fidelity metrics (Gram, AdaIN) compared to recent baselines (Table 1). Qualitatively, it shows improved style consistency and reduced content leakage (Fig. 4). Crucially, it is significantly faster (~40 seconds) than per-asset optimization methods like StyleTex (15 minutes), highlighting its practical advantage.
3. **Demonstrated Versatility**: The paper goes beyond single-object stylization to show compelling results on multiple downstream tasks without retraining: partial reference stylization (Fig. 15), consistent styling of multi-object scenes (Fig. 6), and tileable texture generation (Fig. 16). This demonstrates the generality and robustness of the learned style representation.

### Weaknesses
1. **Limited and Potentially Flawed Evaluation of 3D Consistency**: While the paper uses metrics like Gram and AdaIN computed per-view, it lacks a direct, quantitative measure for the *multi-view 3D consistency* of the generated texture—a core claim and challenge for 3D stylization. Qualitative multi-view renderings are provided, but a metric like texture reprojection error across views or a user study would strengthen the evaluation. The CLIP score is used for disentanglement but is an indirect measure.
2. **Insufficient Detail on Training Data and Potential Data Leakage**: The method is trained on pseudo-pairs from Objaverse. The paper states evaluation uses meshes "distinct from those used during training," but does not specify the train/test split criteria (e.g., by category, instance, or hash). Given the diversity of Objaverse, more details are needed to ensure a fair evaluation and rule out near-duplicate content between splits. The process of selecting the 20 test objects is unclear.
3. **Acknowledged but Under-Explored Limitations**: The limitation regarding failure on fine-grained patterns like text/symbols (Fig. 7) is correctly attributed to the SDXL backbone. However, this points to a broader issue: the method's performance is inherently bounded by the pre-trained 2D diffusion model's capabilities (e.g., its bias towards natural image statistics). The paper does not explore how this foundational choice affects stylization of more abstract or non-naturalistic artistic styles beyond the shown examples.

### Novelty & Significance
**Novelty** is **moderate-to-high**. The specific application of patch shuffling/masking (the "jigsaw" operation) to construct style references for **3D** stylization is novel and well-justified. While patch shuffling for style-content separation has been explored in 2D (e.g., StyleAdapter), its integration into a full 3D stylization pipeline with multi-view diffusion and geometry conditioning represents a non-trivial extension. The reference-attention mechanism within a multi-view U-Net is a solid engineering contribution.
**Significance** is **potentially high** for the field. If the claims hold robustly, the method offers a compelling solution to the speed and content-leakage problems of prior work. The ability to perform fast, feed-forward 3D stylization with user control (partial reference) could have practical impact in content creation. The core idea of using data degradation to create supervision for disentanglement could inspire follow-up work.

### Suggestions for Improvement
1. **Strengthen the 3D Consistency Evaluation**: Introduce a quantitative metric for view consistency. For example, compute the LPIPS or MSE between a rendered novel view and the corresponding view "projected" from the generated textured mesh. Alternatively, conduct a small-scale user study comparing the perceptual 3D consistency of outputs from different methods.
2. **Clarify Dataset Splits and Training Details**: Provide clear information on how the Objaverse training set and the 20-test-object set were constructed. Include statistics (e.g., category distribution) to demonstrate non-overlap. In the appendix, detail the number of assets used for training and the rendering parameters for creating pseudo-pairs.
3. **Expand Ablation Studies and Analysis**:
    *   Ablate the contribution of the **reference-attention module** versus simply using the jigsaw features as an additional condition via cross-attention. How crucial is the specific proposed architecture?
    *   Analyze failure modes beyond text. For example, how does the method perform with style references that have strong, structured geometric patterns (e.g., grids, spirals) that might conflict with the target object's geometry?
    *   Discuss the impact of the **choice of pre-trained diffusion backbone** (SDXL). Would a more controllable or symbol-aware base model (if available) mitigate the cited limitation?

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Quantitative evaluation of multi-view consistency.** The paper claims multi-view consistency but provides no metric (e.g., LPIPS between views, or a dedicated consistency score). Without this, the core claim of view-consistent generation is unsubstantiated.
2. **Ablation study on the reference attention module.** The novel reference attention is a key component, but its individual contribution is not isolated. An ablation comparing it to simpler style injection (e.g., AdaIN) is needed to verify its necessity.
3. **Comparison with state-of-the-art 3DGS-based stylization (e.g., StyleGaussian) in the main paper.** The appendix includes a brief comparison, but for ICLR, a direct, quantitative comparison in the main experiments is required to position the method against efficient, contemporary baselines.
4. **User study for qualitative evaluation.** Style transfer is highly subjective. Relying only on automated metrics and visual examples is insufficient; a user study is standard practice to validate perceptual quality claims.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of content preservation versus style transfer.** The paper measures style fidelity and disentanglement but not how well the original object geometry/content is preserved. Metrics like SSIM between rendered views of the original and stylized mesh (with identical geometry) are needed to ensure the method doesn't distort shape.
2. **Systematic analysis of the jigsaw operation's limitations.** The analysis shows jigsaw preserves statistics, but it fails to discuss or analyze cases where style is inherently global (e.g., a compositional style like "The Scream"). A discussion of when jigsaw is expected to fail is critical for understanding the method's scope.
3. **Analysis of the impact of jigsaw parameters (patch size, mask ratio) across different style categories.** The ablation is limited to a few examples. A per-style-category analysis (e.g., strokes vs. color fields) would show whether the chosen parameters generalize or need to be style-adaptive.

### Visualizations & Case Studies
1. **Visualization of attention maps from the reference attention module.** Showing which parts of the jigsawed reference are attended to for different object regions would demonstrate whether the module performs meaningful, dynamic style recombination.
2. **Visualization of the baked UV texture maps.** The paper shows only rendered views. Displaying the generated UV map is necessary to evaluate seamlessness and baking quality, which are key to the final 3D asset.
3. **Detailed case studies on failure modes.** The paper mentions text/symbols as a limitation, but more diverse failure cases (e.g., styles with strong semantic layouts) should be shown and diagnosed to clarify the method's boundaries.

### Obvious Next Steps
1. **Add a quantitative multi-view consistency metric and report it in the main results.** This is a fundamental requirement for any 3D-aware generation paper.
2. **Conduct and report a user study.** This is standard in style transfer to back up qualitative claims.
3. **Integrate a comparison with 3DGS-based stylization methods (e.g., StyleGaussian) into the main quantitative table and discussion.**
4. **Release full code and pre-trained models.** ICLR emphasizes reproducibility; providing these is essential for the community to verify and build upon the work.

# Final Consolidated Review
## Summary
JIGSAW3D introduces a 3D style transfer method that uses a jigsaw operation—patch shuffling and masking—to disentangle style from content in reference images. This enables fast, feed-forward generation of view-consistent stylized renderings via a multi-view diffusion model trained on pseudo-paired data from textured 3D assets, with applications to partial stylization, multi-object scenes, and tileable textures.

## Strengths
- **Effective style-content disentanglement via jigsaw operation:** The method is grounded in a simple yet powerful mechanism, quantitatively analyzed in Figure 3, which shows that shuffling patches (e.g., at N=8 divisions) suppresses semantic content while preserving style statistics like color and texture.
- **Strong empirical performance and efficiency:** Achieves state-of-the-art style fidelity on standard metrics (Gram matrix similarity and AdaIN distance) as shown in Table 1, and qualitatively outperforms baselines in visual consistency and geometric preservation (Figure 4). It is significantly faster than optimization-based methods (~40 seconds vs. 15 minutes).
- **Demonstrated versatility without retraining:** The method generalizes to challenging downstream tasks such as partial reference stylization, consistent styling of multi-object scenes, and tileable texture generation (Figures 6, 15, 16), highlighting its robustness and practical utility.

## Weaknesses
- **Lacks quantitative evaluation of multi-view consistency:** The paper claims view-consistent generation as a core contribution but provides no quantitative metric (e.g., texture reprojection error or LPIPS between views) to substantiate this. Reliance solely on qualitative renderings undermines the assessment of a key challenge in 3D stylization.
- **Insufficient detail on training data split and potential data leakage:** While the paper states that test meshes are distinct from training, it does not specify the split criteria (e.g., by category or instance) or provide statistics on the training set. This omission raises concerns about fair evaluation and reproducibility.
- **Key methodological details are omitted or unclear:** For reproducibility, specifics such as the exact layers used for extracting \( f_{\text{ref}} \) from the reference U-Net, the rationale for using timestep \( t=0 \) in that U-Net, and the architecture of the multi-view attention module are not adequately described, hindering replication and understanding.

## Nice-to-Haves
- A user study to validate perceptual quality and consistency, as style transfer is inherently subjective.
- Ablation study on the reference attention module to isolate its contribution beyond the jigsaw operation.
- Inclusion of a quantitative comparison with 3DGS-based stylization methods (e.g., StyleGaussian) in the main results, rather than only in the appendix.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Proof of style statistics preservation in Appendix A.2:** While the proof focuses on pixel-level mean and variance preservation rather than deep feature statistics, this does not invalidate the method because the paper empirically validates style preservation using perceptual metrics (Gram, AdaIN) derived from VGG features.
- **Minor critique of the abstract's "substantially lower latency" claim:** The claim is supported by the comparison to optimization-based StyleTex (40s vs. 15min), and nuances relative to other feed-forward baselines do not constitute a substantive weakness.
- **Formatting nitpicks about figure clutter or writing style:** These are not central to the technical contribution and should not affect the evaluation.

## Novel Insights
The paper's core insight is that patch-level operations—shuffling and masking—can effectively isolate style statistics for 3D stylization by destroying global semantics while preserving local style cues. This enables the creation of pseudo-paired training data from existing 3D assets, bypassing the need for curated style-texture pairs. The integration of this idea into a multi-view diffusion pipeline with reference attention represents a novel extension of 2D disentanglement strategies to 3D.

## Suggestions
- Introduce a quantitative metric for multi-view consistency, such as the average LPIPS difference between generated views or a novel view synthesis error computed from the baked texture.
- Provide a detailed description of the Objaverse train-test split, including criteria (e.g., hash-based or category-based separation) and category distributions, in the method or appendix.
- Clarify the architectural choices in the method, potentially through a table or expanded diagram specifying the layers used for \( f_{\text{ref}} \), the conditioning dropout implementation, and the multi-view attention design.

# Actual Human Scores
Individual reviewer scores: [8.0, 2.0, 4.0, 4.0]
Average score: 4.5
Binary outcome: Reject
