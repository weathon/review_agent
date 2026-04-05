=== CALIBRATION EXAMPLE 45 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the core pipeline and the central "jigsaw" operation. 
- The abstract clearly states the problem (per-asset optimization, style-content entanglement), the proposed method (patch shuffling/masking + multi-view diffusion + style baking), and key results (speed, consistency, generalization).
- The claim that the jigsaw operation "decouples style from content" and enables "explicit content–style disentanglement" is slightly overstated. As detailed in Sec. A.2 and Fig. 3, the operation preserves first- and second-order statistics, which are necessary but insufficient for full perceptual style disentanglement. The abstraction would benefit from tempering "explicit disentanglement" to "statistical style isolation" or clarifying how higher-order style semantics are captured downstream.

### Introduction & Motivation
- The motivation is well-grounded: the scarcity of paired 3D style–texture datasets and the high computational cost of SDS-based test-time optimization are genuine bottlenecks in the field.
- The contributions are clearly listed and align with the proposed pipeline.
- The introduction slightly downplays prior 2D work on patch shuffling for style isolation (e.g., StyleAdapter, Gu et al.) by framing jigsaw as a novel conceptual leap rather than a principled 2D-to-3D adaptation. More precise positioning would strengthen the claimed novelty. Additionally, the statement "By construction, the jigsawed reference suppresses content leakage while retaining style statistics" implies that the operation alone guarantees disentanglement; in practice, local patches still carry strong semantic cues (e.g., organic vs. architectural textures), which the introduction should acknowledge.

### Method / Approach
- **Logical Gap in Training Data Construction (Sec. 3.1):** This is the most critical concern. You state: *"For each 3D object, we render K orthogonal views as texture targets, along with several additional random views as reference images... the original texture targets serve as ground truth supervision."* If the reference and target are rendered from the *same* 3D asset, the model is trained to reconstruct the asset's own original texture conditioned on a jigsawed version of itself (plus geometry). This is effectively a **view-consistent texture inpainting/reconstruction** objective, not a style transfer objective. It is unclear how this training regime teaches the model to map *novel, out-of-distribution styles* (e.g., WikiArt) to arbitrary geometries. If the intent is cross-asset style transfer, the training pairs should mix reference styles from one asset with target views from a different asset. Please clarify the training pair sampling strategy and justify how style generalization is achieved without cross-object/style pairing.
- **Reference Feature Extraction & Dimension Alignment (Sec. 3.2):** The method extracts intermediate hidden states `f_ref` from a pretrained diffusion U-Net at `t=0` on the jigsawed reference. Which layers are extracted? Are spatial dimensions pooled, flattened, or padded to match the multi-view sequence length for the cross-attention in Eq. 3? The standard cross-attention formula assumes aligned sequence dimensions, but a single-view reference and K-view queries require explicit spatial/sequence alignment mechanisms (e.g., adaptive pooling, spatial broadcasting, or learned projections) that are omitted.
- **Assumptions:** The method assumes that normal and position maps (`G ∈ R^{B×2K×H×W}`) provide sufficient geometric conditioning for stylization. For objects with concavities, self-occlusions, or complex topology, 2D projections of normals/positions suffer from ambiguity. A brief discussion on how this conditioning handles geometric ambiguities or view-dependent effects would strengthen the reproducibility and theoretical grounding.
- **Reference Attention:** Eq. 3 is standard cross-attention. The novelty claim rests on the *input representations* (`f_ref`) and training regime, not the attention formulation itself. This should be clarified to avoid over-attribution.

### Experiments & Results
- **Metrics:** The quantitative evaluation relies on Gram Matrix Similarity, AdaIN Distance, and CLIP Score. While standard in early stylization literature, these are widely considered insufficient for modern diffusion-based generation, which operates in high-frequency, perceptual, and semantic spaces. The absence of modern perceptual metrics (e.g., LPIPS, DreamSim), distribution-level metrics (FID/KID per view), or a user study on style fidelity vs. geometry preservation is a significant gap for ICLR.
- **Baselines & Comparisons:** The selected baselines are relevant, but StyleTex (SDS-based) is inherently slower, making the latency comparison expected. MV-Adapter and 3D-style-LRM are fair training-free comparisons. However, Table 1 shows marginal quantitative gains (e.g., Gram 4.81 vs 4.82 vs 4.91). The claim of "significantly superior performance" is not fully supported by these tight margins, especially without variance/error bars.
- **Test Set Scale & Variability:** Evaluation uses 20 test objects and 70 reference images. This is quite small for a diffusion-based method claiming strong generalization. There are no reported confidence intervals, standard deviations across seeds, or ablations on architectural components (e.g., removing multi-view attention vs. reference attention), which would materially clarify each module's contribution.
- **Datasets & Evaluation Metrics:** The WikiArt subset and manually collected images provide reasonable stylistic diversity but lack explicit categorization or difficulty grading (e.g., structural vs. textural styles). Metrics are reported as averages, but no per-style or per-category breakdown is provided, making it difficult to identify where the method fails or excels.

### Writing & Clarity
- The pipeline in Figure 2 is informative and accurately reflects the methodology. Qualitative comparisons (Fig. 4, Fig. 6) are well-organized and visually support the claims of geometry preservation and style consistency.
- **Confusing Section:** Section 3.1 ("STYLE-TEXTURE PAIRS CREATION") contains the ambiguity highlighted above regarding same-object vs. cross-object training. Clarifying whether `reference` and `target` are sampled from the same or different Objaverse assets is critical.
- **Figure/Table Clarity:** Table 1 lacks standard deviation/error bars, making the "SOTA" claim harder to statistically validate. Figure 3 effectively shows the shuffling/masking trade-off, but the y-axis scaling for Gram/CLIP could be normalized to 0–1 for intuitive comparison.

### Limitations & Broader Impact
- The authors correctly identify the failure to preserve fine-grained text/symbols (Fig. 7) and attribute it to the SDXL backbone.
- **Missing Limitations:** 
  1. **UV Dependency:** The pipeline relies on pre-existing, often inconsistent, UV parametrizations. Performance on meshes with poorly laid-out UVs or severe stretching is not discussed.
  2. **Lighting/Albedo Assumptions:** The method treats multi-view outputs as pure albedo, but diffusion models inherently bake lighting/shadow priors. The "visibility-aware reprojection" step does not explicitly address baked illumination removal (inverse rendering), which could cause view-dependent shading artifacts when rotating the final mesh under new lighting.
  3. **Training Compute & Convergence:** The paper reports 10 epochs of training but omits the compute budget (GPU hours) and training stability analysis, making it hard to assess the "scalable" claim relative to fine-tuning adapters.
- **Broader Impact:** The ethical statement is standard but could briefly address the potential for style appropriation in commercial 3D asset pipelines and how the model's style-space interpolation behaves with copyrighted artistic corpora.

### Overall Assessment
Jigsaw3D proposes a practical, scalable pipeline for 3D style transfer that avoids per-asset SDS optimization and introduces an intuitive patch-shuffling mechanism to isolate stylistic statistics. The qualitative results are strong, and the multi-view conditioning plus UV baking pipeline is well-motivated for production use. However, the paper faces two critical hurdles for ICLR acceptance: (1) a fundamental ambiguity in the training paradigm—if reference and target views are sampled from the same 3D asset, the model learns texture reconstruction rather than general style transfer, and cross-asset generalization mechanisms must be explicitly detailed and justified; (2) the quantitative evaluation relies on outdated metrics without statistical variance, modern perceptual measures, or architectural ablations that would rigorously support the claimed contributions. Clarifying the data pairing strategy, incorporating LPIPS/user evaluations, and ablation studies on the attention modules would significantly strengthen the paper. The core idea is sound and aligns with ICLR's interest in efficient 3D generation, but addressing these methodological and evaluative gaps is essential to meet the venue's standards for novelty and empirical rigor.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces Jigsaw3D, a multi-view diffusion framework for 3D style transfer that addresses style-content entanglement and computational bottlenecks in existing optimization-heavy or training-free methods. By applying a patch-shuffling and masking (jigsaw) transform to reference images, the authors construct scalable pseudo style-texture pairs from uncurated 3D datasets, which train a reference-attention-guided diffusion model conditioned on explicit geometric cues. The generated multi-view renderings are baked onto input meshes using a visibility-aware projection and seam-aware blending pipeline, yielding view-consistent stylized assets at low inference latency.

### Strengths
1. **Scalable Data Strategy via Jigsaw Disentanglement:** The core contribution of synthesizing supervised training pairs from Objaverse using patch shuffling and stochastic masking is elegant and practical. The paper provides a mathematical proof (Appendix A.2) and quantitative validation (Figure 3) that shuffling preserves first/second-order style statistics while suppressing semantics, effectively bypassing the lack of curated 3D style datasets.
2. **Strong Empirical Performance & Efficiency:** The method achieves state-of-the-art results on standard style fidelity metrics (Gram Matrix Similarity, AdaIN Distance) and competitive CLIP scores compared to baselines (Table 1). Inference is substantially faster (~40s vs 15min for SDS-based StyleTex), making it highly practical for iterative 3D content creation workflows.
3. **Well-Integrated Architectural Design:** The combination of a trainable geometry adapter, reference cross-attention, and row-wise multi-view attention is logically sound and effectively balances structural preservation with stylistic adaptability. Qualitative results (Figure 4, Figure 6) demonstrate strong multi-view consistency and accurate application of style attributes to corresponding geometric regions without warping base geometry.
4. **Comprehensive Ablation & Generalization:** The paper thoroughly ablates key design choices, including jigsaw application during training/inference (Figure 5), mask ratios, and patch sizes (Appendix A.5). It also successfully extends to downstream applications like partial stylization and tileable texture generation (Appendix A.6), showcasing robustness.

### Weaknesses
1. **Incremental Novelty Over Core Components:** While the application to 3D is well-executed, patch shuffling/masking for style isolation has been established in 2D literature (e.g., StyleAdapter, Gu et al., 2018), and the multi-view/reference attention modules closely follow prior works (MVDream, IP-Adapter). The paper claims novelty as the "first to incorporate image-jigsaw for 3D stylization," but does not sufficiently articulate the technical gap or fundamental ML insight beyond the system-level integration and data pipeline.
2. **Evaluation Metrics & Baseline Rigor:** 
   - The use of CLIP score as a proxy for "style-content disentanglement" is problematic. A lower CLIP score between reference and output could indicate successful disentanglement, but could equally imply poor style transfer or loss of meaningful semantics, making the metric ambiguous for this task.
   - Table 1 reports cost and time without hardware specifications, batch sizes, or variance measures. Comparing "~40s" to "15min" lacks fairness without standardized GPU specs and full pipeline timing (including baking).
   - Key recent baselines in open-world 3D texture generation (e.g., Meta3D-TextureGen, recent diffusion texture models) are omitted from quantitative comparison.
3. **Under-Specified Baking Pipeline:** Section 3.2.1 describes the style baking process at a high level. Critical details like multi-view visibility weighting functions, UV seam blending heuristics, and the exact algorithm for 3D inpainting are missing. Since ICLR reviewers expect algorithmic clarity, the current description reads more like a heuristic engineering step than a reproducible methodology.
4. **Training Details & Compute Transparency:** The paper states the model is trained for only 10 epochs with AdamW and LR=5e-5 on SDXL. It is unclear whether the entire U-Net is fine-tuned or only lightweight adapters. Without clarifying trainable parameters, VRAM usage, and total training compute, reproducibility and efficiency claims are difficult to verify.

### Novelty & Significance
**Novelty:** Moderate. The method is a strong engineering integration of existing 2D style disentanglement techniques into a 3D multi-view diffusion framework. The pseudo-pair generation strategy via jigsaw transforms is the most novel aspect, providing a scalable workaround for paired data scarcity in 3D stylization.
**Significance:** High for applied generative modeling and 3D graphics. The framework directly addresses two major pain points in current 3D texture generation: semantic entanglement in reference conditioning and the prohibitive compute cost of test-time optimization. If packaged properly, it represents a practical, production-friendly advance.
**Clarity:** Good. The manuscript is well-structured, the pipeline diagram is clear, and mathematical formulations are standard and consistent. Minor parser artifacts do not detract from comprehension.
**Reproducibility:** Fair to Good. Hyperparameters, dataset sources, and architectural components are listed, and code is promised. However, missing hardware specs, exact trainable parameter lists, and formal baking algorithms limit full reproducibility.

### Suggestions for Improvement
1. **Clarify Novelty & Positioning:** Explicitly contrast the proposed method with 2D jigsaw/shuffling baselines. Detail the specific ML challenges in adapting these to 3D multi-view diffusion (e.g., view-consistent attention routing, 3D geometry conditioning, UV-space continuity) and highlight how Jigsaw3D's architecture uniquely solves them.
2. **Strengthen Evaluation Rigor:** 
   - Replace or supplement the CLIP disentanglement claim with structure-aware content preservation metrics (e.g., LPIPS or DINO similarity between original and stylized mesh renderings) to better quantify geometry preservation.
   - Include exact hardware specs, variance/standard deviations, and end-to-end timing (including baking) in Table 1. Consider adding a recent diffusion-based texture generation baseline for a more complete quantitative comparison.
   - Include a user study evaluating human perception of style fidelity, content preservation, and multi-view consistency, as perceptual metrics poorly correlate with subjective quality in stylization tasks.
3. **Detail the Baking Algorithm:** Provide pseudo-code or explicit formulas for the visibility-aware reprojection, confidence-weighted blending, and seam-aware UV inpainting. If heuristic, justify design choices with an ablation (e.g., effect of random view count on baking quality, as hinted in Fig 19).
4. **Improve Training Transparency:** Specify exactly which modules are frozen vs. trainable (e.g., are VAE/U-Net backbone weights frozen, or only adapters fine-tuned?). Report total training compute (GPU hours) and VRAM requirements to validate the efficiency claims and aid reproduction.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add a controlled semantic-mismatch experiment using reference-target pairs with disjoint object categories** (e.g., animate reference on rigid target) and report a content-leakage metric, because Gram/AdaIN only capture low-level texture statistics and cannot verify the core claim of style-content disentanglement.
2. **Include a naive 2D-to-3D baseline that applies a strong 2D style transfer method per-view before UV projection**, because the paper claims multi-view architectural superiority but lacks this fundamental control to justify the added complexity of cross-view attention modules.
3. **Report standard perceptual generative metrics (FID, LPIPS) alongside the existing statistical ones across a standardized test set**, because diffusion models can artificially optimize first/second-order feature statistics without producing perceptually high-quality or realistic textures, making the SOTA claim unverifiable.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze high-level feature distributions (e.g., t-SNE or linear probes on frozen CLIP/VGG activations) pre/post jigsaw**, because the mathematical proof of mean/variance preservation in Appendix A.2 does not account for non-linear deep representations where style and semantics routinely entangle.
2. **Quantify the attribution contribution of each attention branch (self vs. multi-view vs. reference) via gradient-based importance or activation magnitude analysis**, because without isolating each mechanism, readers cannot determine if disentanglement stems from the novel reference attention or simply from SDXL's default priors and geometric conditioning.
3. **Evaluate error propagation during the 3D baking stage by measuring projection misalignment rates and UV distortion across varying parameterization qualities**, because the method assumes seamless baking works reliably, yet UV space artifacts are a known failure mode that could entirely negate the 2D generation quality.

### Visualizations & Case Studies
1. **Show cross-view correspondence heatmaps for identical geometric points across different rendered angles**, because side-by-side viewpoint images cannot prove true multi-view consistency and often hide texture sliding or feature mismatch.
2. **Overlay Reference Attention activation maps onto generated outputs**, because it is critical to visually confirm that style features are being localized and recombined with target geometry rather than applied as a global, content-agnostic wash.
3. **Present failure cases on highly non-uniform UV unwraps or complex topological objects** (e.g., characters with open topology, thin structures), because the current figures predominantly show simple or well-parameterized shapes, masking the method's true robustness limits.

### Obvious Next Steps
1. **Conduct a formal user study measuring perceptual style fidelity, content preservation, and 3D coherence**, because ICLR expects human evaluation for subjective generative tasks and standard computational metrics correlate poorly with artistic judgment.
2. **Provide exact architectural specifications for the Style U-Net, explicitly listing which SDXL blocks are frozen, which are adapted, and how cross-attention queries/keys are dimensionally matched**, because vague module definitions prevent reproducibility and leave reviewers unable to assess whether the method introduces substantive architectural novelty or relies heavily on off-the-shelf diffusion priors.
3. **Release the generated pseudo-pair dataset construction pipeline or provide a script that replicates the jigsaw operation on Objaverse assets**, because the paper's training paradigm relies entirely on this synthetic data curation, and without it, the claimed dataset advantage cannot be validated.

# Final Consolidated Review
## Summary
Jigsaw3D proposes a fast, training-free (at inference) 3D style transfer framework designed to bypass per-asset optimization and mitigate style-content entanglement. The method constructs pseudo style-texture pairs by applying patch shuffling and masking to multi-view renders of existing 3D assets, training a diffusion model that conditions on explicit geometric maps, multi-view consistency modules, and a reference cross-attention mechanism. Stylized renderings are subsequently baked onto input meshes to yield view-consistent textured assets.

## Strengths
- **Scalable pseudo-pair generation via jigsaw disentanglement:** Utilizing stochastic patch shuffling and masking to synthesize style-texture supervision from uncurated 3D datasets is an elegant workaround for the scarcity of paired 3D style corpora. Figure 3 and the ablation in Figure 5 demonstrate that this operation effectively suppresses spatial semantics while preserving local stylistic statistics, enabling supervised training without manual curation.
- **Practical efficiency and multi-view consistency:** The pipeline achieves competitive style fidelity with inference times of ~40 seconds, drastically outperforming heavy SDS-based baselines (~15 min). The explicit integration of geometric conditioning (normal/position maps) and multi-view attention yields strong qualitative consistency across viewpoints and accurate style localization on complex geometries, as evidenced in Figures 4 and 6.

## Weaknesses
- **Fundamental misalignment in the training data construction:** Section 3.1 states that target views and reference images are rendered from the *same* 3D asset, using the jigsawed view as input and the original view as ground truth. This effectively trains a view-conditioned texture reconstruction model rather than a cross-domain style transfer model. It remains entirely unclear how this regime teaches the network to map novel, out-of-distribution styles (e.g., WikiArt) to arbitrary, unseen geometries without catastrophic content leakage or reverting to the training distribution's native style. This gap fundamentally undermines the claimed style-content disentanglement and generalization capability.
- **Insufficient evaluation rigor and missing reproducibility details:** The quantitative evaluation relies exclusively on low-level statistical metrics (Gram Matrix, AdaIN) that poorly correlate with human perception of modern diffusion-generated textures, and is conducted on a small test set (20 objects, 70 styles) without variance reporting, perceptual metrics (LPIPS/FID), or user studies. Furthermore, critical architectural and procedural details are omitted: the paper does not specify which SDXL blocks are frozen vs. trainable, how single-view reference features are spatially aligned with multi-view queries in the cross-attention module, nor does it provide algorithmic clarity (formulas or pseudo-code) for the visibility-aware UV baking pipeline. These omissions severely hinder reproducibility and make the SOTA claims statistically unverifiable.

## Nice-to-Haves
- Incorporate a controlled semantic-mismatch experiment where reference styles are explicitly paired with disjoint target categories during training or evaluation to bound the generalization capability.
- Provide gradient-based or activation magnitude analysis for the three attention branches to empirically isolate the contribution of the reference attention module versus the base diffusion priors.
- Release the exact data generation script and a formal specification of which network parameters are updated during the 10-epoch training regime.

## Novel Insights
The paper pragmatically repurposes patch shuffling not as a theoretical disentanglement tool, but as a constructive data-generation mechanism that forces a multi-view diffusion model to ignore global layout and attend to local statistical cues. By treating jigsawed renders as explicit style conditions paired with geometric projections, the framework sidesteps the need for invertible style embeddings or per-asset SDS optimization, offering a clean, production-oriented inference path. However, the technical contribution remains a system-level integration of established 2D shuffling priors into a 3D-aware conditioning architecture; the work does not establish new theoretical bounds on style-content separation or introduce fundamentally novel attention mechanisms beyond careful module routing and geometric injection.

## Potentially Missed Related Work
- StyleAdapter (Wang et al., 2023) and Deep Feature Reshuffle (Gu et al., 2018) — These works pioneered patch shuffling for 2D style isolation. Explicitly detailing how Jigsaw3D's adaptation differs technically when extended to multi-view 3D diffusion (e.g., handling view-consistent feature routing and preventing cross-view semantic collapse) would better contextualize the novelty.
- Meta 3D TextureGen (Bensadoun et al., 2024) — A recent large-scale texture generation framework that serves as a relevant baseline for modern diffusion-based 3D texturing pipelines and could strengthen quantitative comparisons.

## Suggestions
- Redesign or explicitly justify the training paradigm: either incorporate cross-asset/cross-domain style pairing to learn actual transfer mappings, or formally prove/empirically demonstrate that same-asset reconstruction inherently yields transferable style priors without memorizing training object appearances.
- Expand the evaluation to include high-dimensional perceptual metrics (LPIPS, DINO for geometry preservation) and conduct a controlled user study on style fidelity, geometry preservation, and multi-view consistency to validate statistical improvements against human perception.
- Publish complete architectural specifications (frozen vs. trainable parameter counts, cross-attention dimension matching) and explicit algorithmic steps for the UV baking and seam-aware blending procedures to satisfy ICLR reproducibility standards.

# Actual Human Scores
Individual reviewer scores: [8.0, 2.0, 4.0, 4.0]
Average score: 4.5
Binary outcome: Reject
