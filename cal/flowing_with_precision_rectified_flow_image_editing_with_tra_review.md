=== CALIBRATION EXAMPLE 30 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:**
The title "Flowing with Precision: Rectified Flow Image Editing with Trajectory and Frequency Guidance" is appropriate and reflects the dual-domain contribution. The abstract clearly states the proposed Starting Point Optimization (SPO) and Trajectory Optimization (TO) strategies, and the claims about handling complex multi-object scenes are supported by the experiments. However, the abstract mentions "experiments show that our method significantly outperforms existing methods" and "higher user preference," but does not quantify this significance (e.g., p-values), which is an ICLR expectation for strong claims.

**Introduction & Motivation:**
The introduction effectively sets up the problem: editing complex scenes with rectified flow models is challenging due to semantic entanglement. It correctly identifies limitations of prior inversion-based and attention-modification methods. The contributions are clearly listed. A minor weakness is that the core insight—that the editing process naturally separates into distinct phases (Chaos, Layout, Refinement)—is introduced somewhat abruptly. More motivation or preliminary evidence for this tri-phase model would strengthen the narrative. The connection between this observation and the proposed SPO could be made more explicit.

**Method / Approach:**
This is the core section and contains several points that require clarification for reproducibility and scientific rigor.
1.  **Phase Definitions (Sec 3.2):** The definitions of the Chaos, Layout, and Refinement phases are qualitative and based on an observation of cosine similarities (Fig 4). The precise criteria for identifying the transition points \(T_0\) and \(T_{turn}\) algorithmically are not fully specified. The text states \(T_0\) is where "the low frequency and overall values of CosSim2 are first equal," but the process for determining this from an image pair needs a clearer, step-by-step description.
2.  **Starting Point Optimization (SPO):** The description is insufficient. It's stated that SPO "adaptively determines the optimal editing start point by calculating the low frequency Mean Squared Error (MSE) between the source and target images." However, the target image is unknown during editing. Presumably, this refers to the low-frequency MSE between the source image and some intermediate prediction? Or is it calculated during a preprocessing inversion? The method description must unambiguously define all inputs and operations.
3.  **Trajectory Optimization - Frequency Scaling (Eq. 11):** The derivation and intuition for the adaptive frequency scaling coefficient \(\lambda_{type}(t_i)\) are unclear. The equation seems to scale components based on their relative spectral energy. The parameter \(\alpha\) is introduced but its effect, typical value, and ablation are not discussed in the main paper (though it might be in the appendix). More importantly, the *justification* for this specific form of scaling is missing. Why is "1 + α * (1 - relative_energy)" a good choice? How does it connect to the observed "stage-specific spectral characteristics" mentioned in the abstract?
4.  **Attention Remapping (Sec 3.4):** The process is better explained, but the mapping function \(\phi(j)\) is critical. How is token correspondence established between source and target prompts? Is it a simple string match? This is a non-trivial issue for edits involving synonyms or paraphrases. The choice of layers for injection (5-20 for layout, 20-45 for refinement) is justified in the appendix, which is good, but the main text should at least summarize the rationale.
5.  **Mathematical Notation and Flow:** There are some inconsistencies. Equation (4) uses \(X_t^{edit}\), \(X_t^{tar}\), \(X_t^{src}\), but the relationship between \(X_t^{tar}\) and the target prompt \(c^{tar}\) is not defined. Is \(X_t^{tar}\) the latent trajectory for a *new* target image generation? The description around Eq. (5) and (6) could be streamlined for clarity. The term "cross-prompt" and "cross-trajectory" in Eq. (6) is helpful, but the subsequent orthogonalization step (Eq. 7) needs a clearer geometric or semantic interpretation.

**Experiments & Results:**
1.  **Baselines:** The selection is comprehensive, covering both RF-based and DM-based SOTA methods, including recent multi-object editors like OIR. This is a strength.
2.  **Metrics:** A standard and comprehensive set is used. It's good to see both structure preservation (LPIPS, SSIM) and text alignment (CLIP) metrics. Reporting CLIP for both the whole image and the edited region is appropriate.
3.  **Quantitative Results (Tables 1 & 2):** The results show a strong trend favoring the proposed method, particularly on structure preservation metrics (Distance, LPIPS, SSIM). The text alignment (CLIP) scores are competitive, often best or second-best. However, for ICLR, it's crucial to report measures of statistical significance (e.g., standard deviations over multiple runs/seeds, or p-values from statistical tests). The tables present only mean scores. Without variance, it's difficult to assess if the improvements are statistically reliable.
4.  **Ablation Study (Table 3 & Fig 8):** This is well-structured, ablating the three core components (SPO, Injection, TO). The quantitative and qualitative results clearly show each component's contribution. However, the ablation for the frequency scaling parameter \(\alpha\) and the layer selection ranges is relegated to the appendix. Given the novelty of the frequency-aware scaling, its ablation should be in the main paper.
5.  **User Study:** The reported user preference rates (50.5% for single-object, 54.8% for multi-object) are impressively high. The description in the appendix is adequate, but the main text should include the number of participants and trials to allow the reader to gauge the study's power.

**Writing & Clarity:**
The paper is generally well-structured. The main clarity issues are concentrated in the Method section, as detailed above. Specifically, the procedural description of SPO and the justification for the frequency scaling formula are the most opaque parts. The figures are helpful, but Figure 2's caption is cut off and references an equation not in the snippet, which is confusing (this may be a parser artifact). The phase diagram (Fig 3/4) is a good conceptual aid.

**Limitations & Broader Impact:**
The paper lacks a dedicated limitations section. Key limitations should be explicitly discussed: (1) The method's performance likely depends on the accuracy of the token mapping function \(\phi(j)\) for attention remapping, which may fail for complex linguistic edits. (2) The computational cost of the frequency-domain operations and the need to determine \(T_0\) and \(T_{turn}\) per image pair should be mentioned. (3) The reliance on the specific architectural properties of MM-DiT (e.g., layer-wise frequency patterns) may limit generalizability to other transformer-based diffusion/flow models. A broader impact statement, while often formulaic, is standard for ICLR and is missing.

### Overall Assessment
The paper presents a novel and well-motivated approach to image editing with rectified flow models, introducing the insightful concept of phase-aware editing and a dual-domain (time and frequency) optimization strategy. The core ideas—adaptive start point selection, semantic vector orthogonalization, and frequency-aware scaling—are interesting and appear effective. The experimental evaluation is thorough, and results are promising. However, for acceptance at ICLR, the manuscript requires significant revisions to improve clarity and rigor. The method description, particularly for SPO and the frequency scaling mechanism, must be made unambiguous and reproducible. Quantitative results need measures of variance or significance testing. A discussion of limitations is mandatory. If these issues are adequately addressed, the paper could make a solid contribution to the field.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a dual-domain framework for text-guided image editing using Rectified Flow (RF) models. The method introduces a Starting Point Optimization (SPO) strategy to adaptively determine the optimal timestep to begin editing based on an image's structural complexity, and a Trajectory Optimization (TO) strategy that operates in both time (via semantic-aware vector orthogonalization) and frequency (via adaptive re-weighting of residuals) domains. Additionally, it selectively injects structural priors from the source image into specific attention layers of the MM-DiT architecture. The approach aims to improve multi-object editing by better balancing structural preservation and semantic alignment.

### Strengths
1. **Well-motivated technical contributions**: The SPO and TO strategies are grounded in an analysis of the denoising process (partitioned into Chaos, Layout, and Refinement phases) and frequency-domain characteristics. The adaptive start point selection based on low-frequency MSE and the frequency-aware scaling are novel and address key challenges in inversion-free editing.
2. **Comprehensive evaluation**: The paper provides extensive quantitative results on both single-object (PIE-Bench) and multi-object (PIE-Bench++, OIR) editing benchmarks, demonstrating state-of-the-art or competitive performance across multiple metrics (e.g., structure distance, LPIPS, SSIM, CLIP similarity). A user study further validates strong human preference.
3. **Thorough ablation studies**: Ablation experiments (Table 3, Fig. 8) clearly demonstrate the contribution of each component (SPO, attention injection, TO). Additional analysis (e.g., attention layer selection in Fig. 11-12, hyperparameter effects in Fig. 18-19) provides valuable insights into design choices.

### Weaknesses
1. **Limited discussion of related frequency-based work**: While the paper cites some prior work on frequency analysis in diffusion models, it does not sufficiently differentiate its frequency-adaptive scaling from existing techniques (e.g., FreeInv, Yu et al. 2023). A more detailed comparison would strengthen the novelty claim.
2. **Hyperparameter sensitivity and reproducibility concerns**: The method involves several tuned hyperparameters (e.g., injection layers 5-20 and 20-45, scaling factor β=4, adaptive coefficient α). The paper lacks a systematic sensitivity analysis, making it difficult to assess robustness across different settings or architectures.
3. **Clarity and organization issues**: The writing contains grammatical errors and occasionally unclear passages (e.g., the description of Eq. (7) and (8) is terse). The abstract and introduction are somewhat repetitive. While OCR artifacts are not held against the paper, the final version would benefit from careful proofreading and streamlining.

### Novelty & Significance
The paper presents novel contributions in adaptively determining the editing start point via low-frequency analysis and jointly optimizing trajectories in time and frequency domains. The selective attention injection leveraging MM-DiT's frequency-aware properties is also a new insight. The work is significant as it advances text-guided image editing for complex multi-object scenes using modern RF models, offering a principled approach to balance structural preservation and editability. The phase-based analysis and frequency-domain manipulations could inspire future research.

### Suggestions for Improvement
1. **Clarify novelty relative to frequency-based methods**: Expand the related work discussion to explicitly contrast the proposed frequency-adaptive scaling with prior frequency manipulation techniques in diffusion/flow models, highlighting the distinct contributions.
2. **Include sensitivity analysis**: Conduct experiments or add a discussion on how performance varies with key hyperparameters (e.g., injection layer ranges, number of denoising steps, α values) to improve reproducibility and provide guidance for adaptation.
3. **Improve writing and organization**: Revise the abstract and introduction to be more concise and avoid repetition. Expand technical explanations where needed (e.g., derivations of orthogonalization and frequency scaling) for better clarity. Ensure the final version is thoroughly proofread.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Compare against state-of-the-art inversion-free RF editing methods.** The paper does not include recent, relevant baselines like **FlowAlign (Kim et al., 2025)** or **DNA-Edit (Xie et al., 2025)**. Without this, the claim of significant outperformance is incomplete and unconvincing for the current literature.
2.  **Rigorously evaluate multi-object editing claims.** The paper asserts superiority in multi-object editing, but quantitative results on PIE-Bench++ (Table 2) show OIR beats the method on key metrics (LPIPS, MSE). A direct, comprehensive comparison on a dedicated **multi-object benchmark with localization metrics** (e.g., IoU for edited objects) is missing, undermining the core claim.
3.  **Conduct a sensitivity analysis for critical hyperparameters.** The method introduces several key hyperparameters (e.g., the injection layers [5-20, 20-45], scaling factor *β*=4, frequency scaling coefficient *α*). No ablation study shows how performance degrades with suboptimal choices, making the method seem fragile and the design choices unjustified.
4.  **Test on a more diverse and challenging dataset.** Evaluations are limited to curated benchmarks (PIE-Bench, OIR). The method's efficacy on **real-world, internet-sourced images** with complex layouts, occlusions, or unconventional compositions is not demonstrated, leaving generalizability in doubt.

### Deeper Analysis Needed (top 3-5 only)
1.  **Quantify and analyze the "frequency-aware" claim.** While a frequency-domain operation is proposed (Eq. 9-11), there is **no analysis showing what the adapted frequency weights (*λ*) actually look like across timesteps or image types**. Without visualizing these weights or correlating them with editing outcomes, the mechanism's contribution is not validated.
2.  **Justify the heuristic for attention layer selection.** The choice to inject into layers 5-20 and 20-45 is justified with a single qualitative figure (Fig. 10,11,12) but lacks a **quantitative ablation across all layers**. A plot showing editing performance (CLIP/SSIM) versus injection layer index is needed to prove these are optimal and linked to frequency patterns.
3.  **Analyze failure modes.** The paper shows successes but does not systematically categorize or analyze **when and why the method fails** (e.g., for specific object types, prompt complexities, or structural changes). This analysis is critical for understanding the method's limitations and scope.
4.  **Clarify the SPO mechanism.** The Starting Point Optimization uses low-frequency MSE to find *T_0*. However, the relationship between this image-specific metric, structural complexity, and the optimal start point is not rigorously established. An analysis showing the correlation between computed *T_0* and human-judged edit quality across diverse images is missing.

### Visualizations & Case Studies
1.  **Show quantitative visual comparisons (e.g., heatmaps) for multi-object edits.** For claims of precise multi-object control, side-by-side results with **error maps or attention map visualizations** highlighting what changed and what stayed the same are necessary. This would expose leakage or inaccurate localization not captured by global metrics.
2.  **Visualize the "orthogonalization" effect.** The core time-domain operation (Eq. 7) should be illustrated. A **vector diagram or feature space visualization** showing the cross-prompt and cross-trajectory components before and after orthogonalization would make the abstract concept concrete and verify its intended effect.
3.  **Provide a detailed case study on a complex edit with breakdowns.** Include a step-by-step visualization of the intermediate latents *X_t* through the three claimed phases (Chaos, Layout, Refinement) for a challenging edit. This would directly substantiate the phased editing narrative in Section 3.2.

### Obvious Next Steps
1.  **Perform a statistically significant user study.** The reported user study (n=28, 14 questions) is too small. For ICLR, a **larger-scale, statistically validated A/B test** (e.g., on a platform like Amazon Mechanical Turk with >50 participants and significance tests) is expected to substantiate strong claims of user preference.
2.  **Benchmark against recent mask-based methods for a fair comparison.** The paper critiques mask-based methods but does not compare against strong, contemporary ones like **RichControl (Zhang et al., 2025)** on their own terms (i.e., using provided masks). A comparison showing the trade-off between mask-free convenience and mask-aided precision is necessary.
3.  **Explore the integration of the proposed frequency guidance into the inversion process of other methods.** A compelling next step would be to apply the trajectory/frequency optimization module to other editing frameworks (e.g., RF-Inversion) to demonstrate its utility as a general plug-in component, strengthening its contribution.

# Final Consolidated Review
## Summary
This paper proposes a dual-domain framework for text-guided image editing with Rectified Flow models. It introduces a Starting Point Optimization (SPO) strategy to adaptively determine when to begin editing based on image complexity, and a Trajectory Optimization (TO) method that performs semantic-aware vector orthogonalization in the time domain and adaptive frequency re-weighting in the spectral domain. The method also selectively injects source image features into specific attention layers of the MM-DiT architecture.

## Strengths
- **Novel phase-based analysis and adaptive start point selection.** The paper provides a clear, empirical analysis of the editing process, dividing it into Chaos, Layout, and Refinement phases. The proposed SPO strategy, which uses low-frequency MSE to find the optimal transition point \(T_0\), is a simple yet effective solution to a previously unaddressed problem of when to begin editing for different images.
- **Comprehensive and strong empirical evaluation.** The method is evaluated on standard single-object (PIE-Bench) and multi-object (PIE-Bench++, OIR) benchmarks against a wide array of recent baselines. It demonstrates superior performance on most structure preservation metrics (e.g., Structure Distance, LPIPS, SSIM) while maintaining competitive text-alignment scores, and is supported by a user study showing strong preference.

## Weaknesses
- **Methodological clarity and reproducibility concerns.** The description of the core SPO mechanism is ambiguous: it states SPO calculates low-frequency MSE "between the source and target images," but the target image is unknown during editing. The process for algorithmically determining the key transition points \(T_0\) and \(T_{turn}\) from an image pair is not specified step-by-step, making exact replication difficult.
- **Lack of statistical validation for quantitative claims.** The results tables report only mean scores without any measure of variance (e.g., standard deviation over multiple runs) or statistical significance tests. For a conference with high standards like ICLR, this omission makes it impossible to judge whether the reported improvements are reliable or due to random variation.
- **Insufficient analysis and justification for the frequency-domain mechanism.** While frequency-adaptive scaling is proposed, the paper lacks analysis to validate its core "frequency-aware" claim. There is no visualization or analysis of what the adapted weights \(\lambda_{low}(t_i)\) and \(\lambda_{high}(t_i)\) actually look like across timesteps or how they correlate with editing outcomes, leaving the mechanism's contribution and intuition unclear.

## Nice-to-Haves
- A sensitivity analysis for key hyperparameters (e.g., injection layer ranges, frequency scaling coefficient \(\alpha\)) to improve reproducibility and provide guidance for adaptation.
- A more systematic analysis of failure cases or limitations, which would help users understand the method's boundaries.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "The abstract does not quantify significance (e.g., p-values)." -> *While related to the missing statistical validation, this is a specific formatting nitpick about the abstract.*
- **Weakness:** "The core insight is introduced somewhat abruptly." -> *This is a stylistic critique, not a substantive flaw in the contribution.*
- **Weakness:** "The derivation for the frequency scaling coefficient \(\lambda_{type}(t_i)\) is unclear and unjustified." -> *This is partially valid but is absorbed into the broader, more substantive weakness about insufficient analysis of the frequency mechanism.*
- **Weakness:** "The mapping function \(\phi(j)\) for attention remapping is non-trivial and may fail for synonyms." -> *The paper acknowledges token mapping (Sec. 3.4) and this is a generic challenge for all attention-based methods, not a specific flaw of this work.*
- **Weakness:** "Lacks comparison to very recent baselines (FlowAlign, DNA-Edit)." -> *It is unreasonable to demand comparison against every concurrently emerging method; the paper compares against a comprehensive set of state-of-the-art methods as of its submission.*
- **Weakness:** "Needs evaluation on internet-sourced images." -> *Evaluation on standard, curated benchmarks is the norm; demanding "internet-sourced" images is scope creep.*
- **Weakness:** "User study is too small." -> *The study size (28 participants) is acceptable for a preliminary preference study; a larger study would be a "nice-to-have."*

## Novel Insights
The paper provides a novel, empirical analysis framing the rectified flow editing process into three distinct phases (Chaos, Layout, Refinement), linked to measurable transitions in feature similarity. This phase-based perspective directly motivates the adaptive Starting Point Optimization strategy. Furthermore, the insight that the MM-DiT architecture exhibits layer-wise frequency patterns, which can be leveraged for phase-specific feature injection (low-frequency structural priors in early layers, high-frequency details later), is a genuine contribution for designing controls within this modern architecture.

## Suggestions
- Revise the method section to provide an unambiguous, step-by-step algorithmic description of how SPO determines \(T_0\) in practice, clarifying what inputs are used if the "target image" is not available.
- Report variance measures (e.g., standard deviation) for quantitative metrics across multiple runs or seeds, or perform statistical significance testing to bolster the claims of improvement.
- Add an analysis or visualization (e.g., a plot of \(\lambda_{low}(t)\) and \(\lambda_{high}(t)\) across timesteps for representative edits) to substantiate the operation and effect of the proposed frequency-adaptive scaling.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 2.0, 4.0, 2.0]
Average score: 2.8
Binary outcome: Reject
