=== CALIBRATION EXAMPLE 41 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: patch-wise and keyword-aware attention for efficient multi-condition control. The abstract clearly states the problem (computational bottleneck of concatenate-and-attend in DiTs), the proposed solution (PKA with PAA and KSA modules, plus early-timestep sampling), and the key results (up to 10× speedup, 5.12× VRAM reduction, maintained/improved quality). All claims are supported in the main text. No major issues.

### Introduction & Motivation
The introduction effectively motivates multi-condition control for DiTs and identifies the quadratic computational bottleneck of the standard approach. The central question—whether full attention is necessary—is well-posed. The authors present an analysis revealing redundancy in attention patterns for spatial-aligned and subject-driven conditions, which logically motivates PKA. Contributions are clearly listed. The introduction is strong and sets up the paper well.

### Method / Approach
This section requires clarification and raises several significant concerns:

1.  **Position-Aligned Attention (PAA):** The core assumption that spatial conditions require only one-to-one attention between aligned patches is intuitive, but its limitations are not discussed. What if the spatial condition is misaligned (e.g., due to cropping, scaling, or non-grid conditions like a bounding box)? The method seems to assume perfect spatial correspondence between condition tokens and image patches. The ablation compares PAA to sliding window attention but does not justify why a window size >1 isn't needed (e.g., for capturing local context beyond the exact pixel). The claim of reducing complexity to O(N) is correct but should be contextualized: this is per spatial condition, and the overall complexity still scales with the number of conditions.

2.  **Keyword-Scoped Attention (KSA):** The description is confusing and lacks critical details for reproducibility.
    *   **Keyword Selection:** The process for extracting the keyword tokens **K** from the text prompt is not specified. Is it done via an external parser (e.g., noun phrase detection)? This is a crucial implementation detail that is omitted.
    *   **Temporal Consistency & Mask Reuse:** The two-step process is described but its justification is weak. Equation 3 computes a mask at timestep *t* using the current noisy image. The text then states this mask is reused at timestep *t+1*. Why is this valid? The "temporal consistency" citation (Zhou et al., 2025) is insufficient justification on its own; a brief explanation or empirical validation of mask stability across timesteps is needed. If the mask is recomputed every step, the overhead of the initial lightweight attention (Eq. 3) must be accounted for in the efficiency analysis.
    *   **Mask Threshold (ϵ):** The choice of ϵ=0.2 is given without justification. The ablation shows the method is somewhat robust, but a sensitivity analysis (e.g., showing subject consistency metrics vs. ϵ) would strengthen the claim.

3.  **Condition Cache:** The mechanism is clearly described and a valid optimization. However, it relies on the condition tokens performing only *self*-attention. The paper should briefly discuss whether this restriction impacts the quality of condition representations, as they no longer attend to the image or text tokens.

4.  **Early-Timestep Sampling:** The proposed shift of the logit-normal distribution is motivated by a perturbation analysis. However, the exact parameters used in experiments (µ, δ) are not stated in the Method section (they appear only in Fig. 13 caption: µ=0.5, δ=1.5). This should be included in Section 3.3. Furthermore, the authors should comment on whether this strategy could harm the model's ability to handle fine details that are typically refined in later timesteps.

**Overall:** The method is novel and well-motivated, but the description of KSA is insufficient for reproduction, and several operational assumptions (perfect spatial alignment, keyword extraction, mask reuse) require further justification or discussion of limitations.

### Experiments & Results
The experimental setup is generally solid, but several aspects need strengthening:

1.  **Baseline Fairness and Implementation:** The baselines (OminiControl2, UniCombine) are appropriate. However, it is unclear if they are compared under strictly identical settings (e.g., same number of training iterations, same data splits, same FLUX.1 base model). The paper states the models are "fine-tuned... for 20,000 iterations," but did the baselines use the same number of iterations and optimizer? This should be clarified.

2.  **Quantitative Results (Table 1):** The results show strong performance. A critical omission is the lack of variance estimates (e.g., standard deviation over multiple runs or a validation set). The reported scores are single numbers, making it difficult to assess statistical significance, especially for the small margins in "Controllability" and "Fidelity."

3.  **Ablation Studies:**
    *   **PAA:** The comparison to sliding window attention is good, but why was a window size of 1 not included? It would be the most direct competitor to PAA's one-to-one assumption.
    *   **KSA:** The ablation varies the mask threshold ϵ but only reports latency and VRAM. It **must also report a subject consistency metric** (e.g., CLIP-I) for each ϵ to validate the claimed "graceful trade-off." Figure 10 only shows one generated image per setting, which is insufficient evidence.
    *   **Early-Timestep Sampling:** Figure 11 shows qualitative results at different iterations, but a quantitative convergence curve (like Fig. 13's SSIM) for the main tasks would be more convincing than the loss/SSIM plot on a single condition type.

4.  **Scalability to More Conditions:** Figures 14-16 show compelling qualitative results for 2, 3, and 4 conditions. However, there is **no quantitative evaluation for 3 or 4 conditions**. The efficiency plots (Figs. 7, 8) show trends, but quality metrics (FID, controllability scores) are essential to demonstrate that performance does not degrade with more conditions.

5.  **Efficiency Claims:** The reported speedups (up to 10×) and memory reductions (5.12×) are impressive. The paper correctly notes these are for the attention module. It should be clarified whether Figure 7 reports total inference time or just attention time. The distinction is important for assessing real-world impact.

### Writing & Clarity
The paper is generally well-written and logically structured. The primary issue is the lack of clarity in the KSA description (as detailed above). Figures are clear and support the narrative. Some captions (e.g., Fig. 9, 10) could more explicitly state what is being compared in the image rows.

### Limitations & Broader Impact
A dedicated "Limitations" section is **absent**, which is a major weakness for an ICLR submission. The paper must explicitly discuss:
1.  **Assumptions of PAA:** Reliance on pixel-perfect alignment between spatial conditions and image tokens.
2.  **Keyword Dependency of KSA:** The need for a clear keyword in the prompt and the lack of a detailed algorithm for extracting it.
3.  **Potential Failure Modes:** What happens if the subject is large/complex and not well-localized by the initial mask? Can KSA handle multiple distinct subjects?
4.  **Generalization:** The experiments are on a curated subset of Subject200K. How does the method perform on other datasets or with different condition types (e.g., pose, segmentation maps)?
5.  **Broader Impact:** A standard discussion on the potential for misuse (generating deepfakes, misinformation) and the environmental impact of more efficient models should be included, even if brief.

### Overall Assessment
The paper presents a compelling idea: leveraging the observed sparsity in attention patterns for multi-condition DiTs to achieve dramatic efficiency gains. The core contributions (PAA, KSA, early-timestep sampling) are novel and well-motivated. The efficiency results are impressive and, if valid, represent a significant advance. However, the current submission has substantial weaknesses: the KSA mechanism is under-specified and lacks rigorous ablation, quantitative results for >2 conditions are missing, variance estimates are absent, and a critical discussion of limitations is omitted. **The contribution is promising but requires significant revisions to meet ICLR's standards for reproducibility, rigor, and completeness.** With thorough addressing of these concerns, the paper could be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Patch-wise and Keyword-Aware Attention (PKA), a novel attention mechanism for efficient multi-condition control in Diffusion Transformers (DiTs). PKA decomposes the standard full attention into two specialized modules: Position-Aligned Attention (PAA) for spatially-aligned conditions (e.g., sketches, depth) and Keyword-Scoped Attention (KSA) for subject-driven conditions, complemented by an early-timestep sampling strategy to accelerate training. The method claims significant efficiency gains (up to 10× inference speedup, 5.12× VRAM reduction) while maintaining or improving generative quality and controllability.

### Strengths
1. **Addresses a Critical Bottleneck**: The paper clearly identifies the quadratic computational overhead of the "concatenate-and-attend" paradigm in multi-condition DiTs as a major practical limitation. The motivation is well-grounded in an analysis of attention sparsity (Figures 2, 3).
2. **Methodological Innovation with Solid Analysis**: The proposed PAA and KSA modules are conceptually clean and directly motivated by the observed sparsity patterns. The early-timestep sampling strategy is supported by a perturbation analysis (Figure 5), showing a data-driven approach to improving training.
3. **Comprehensive and Convincing Experiments**: The paper provides extensive evaluations across three multi-condition tasks, including qualitative results (Figure 6) and quantitative metrics (FID, SSIM, CLIP scores). The efficiency benchmarks (Figures 7, 8) demonstrate substantial and scalable improvements in latency and memory consumption.
4. **Strong Ablation Studies**: The ablation studies (Figures 9, 10, 11) effectively isolate and validate the contributions of each component (PAA, KSA, early sampling) and explore hyperparameter sensitivity (e.g., KSA mask threshold `ϵ`).

### Weaknesses
1. **Reliance on Keyword Extraction for KSA**: The KSA module depends on identifying "keyword tokens" from the text prompt to compute the initial attention mask. The paper does not detail how these keywords are reliably extracted or aligned, which could be a source of failure or added complexity in open-vocabulary settings.
2. **Limited Comparison to State-of-the-Art Efficiency Methods**: While comparisons to OminiControl2 and UniCombine are provided, the paper does not benchmark against other recent DiT efficiency methods like token caching or pruning (e.g., FastCache, DIP-GO) in the multi-condition setting. This makes it harder to assess PKA's relative advantage within the broader efficient DiT literature.
3. **Narrowed Experimental Scope**: The training and evaluation are based on a curated subset of the Subject200K dataset. While this ensures keyword presence, it may not fully represent the complexity and diversity of real-world multi-condition generation scenarios (e.g., with ambiguous or multiple subjects).
4. **Ambiguity in Technical Details**: The description of the "Condition Cache" mechanism (Figure 4a) states that condition KVs are cached from the first denoising step. However, the denoising process is iterative and the noisy image `X` changes each step; the paper does not clarify if caching these *static* condition KVs across all steps is universally valid or if it leads to any approximation error.

### Novelty & Significance
**Novelty**: The core idea of exploiting condition-specific attention sparsity (position-aligned and keyword-scoped) to replace full attention is novel for multi-condition DiTs. While sliding window attention and token pruning exist, applying these principles based on a categorization of condition types (spatial vs. subject-driven) is a fresh and well-justified contribution.
**Clarity**: The paper is generally well-written, with clear figures and formulas. However, some implementation details (keyword extraction, caching validity) require more elaboration for full reproducibility.
**Reproducibility**: The method is described with formulas, architectural diagrams, and key hyperparameters (e.g., `ϵ=0.2`). Training details (LoRA, optimizer) are provided. Reproducibility is feasible but depends on the availability of the base FLUX.1 model and the curated dataset.
**Significance**: The work is highly significant for the field. Efficient multi-condition control is a crucial step towards practical and scalable DiT applications. The demonstrated efficiency gains are substantial and, if validated broadly, could influence the design of future generative models.

### Suggestions for Improvement
1. **Clarify Keyword Extraction and Robustness**: Add a subsection or appendix detailing the process for selecting keyword tokens for KSA. Discuss potential failure modes (e.g., no clear keyword, multiple keywords) and how the method handles them.
2. **Broaden Baseline Comparisons**: Include efficiency and quality comparisons against at least one additional state-of-the-art DiT acceleration method (e.g., a token caching approach) to better situate PKA's performance within the current landscape.
3. **Validate the Early-Timestep Sampling Strategy Across Architectures**: The early-timestep sampling is shown effective for fine-tuning FLUX.1. Provide an ablation or discussion on whether this strategy is generally applicable to other DiT architectures or training objectives (e.g., simple diffusion vs. flow matching).
4. **Discuss Limitations More Explicitly**: A dedicated limitations paragraph should address the dependency on keyword identification, the assumption of condition-type categorization (what about hybrid conditions?), and the potential performance trade-offs at very high mask thresholds in KSA.
5. **Strengthen Claims of "Improved" Quality**: While quantitative metrics show parity or improvement, the paper should more precisely characterize the nature of any quality improvement (e.g., is it in color fidelity, detail preservation, or condition alignment?) and ensure statistical significance is reported where applicable.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with UNet-based state-of-the-art methods (e.g., ControlNet variants).** The paper only compares to other DiT-based methods. Without benchmarking against the dominant UNet paradigm for multi-condition control, the claimed efficiency and quality advantages lack context and may not hold against the most practical baselines.
2. **Quantitative evaluation for tasks with more than two conditions.** The main results (Table 1) are only for two-condition tasks. To substantiate claims of scalability and efficiency gains with increasing conditions, quantitative metrics (FID, controllability scores) for 3- and 4-condition generation are necessary.
3. **Ablation on the contribution of the early-timestep sampling strategy.** Table 1 does not isolate the performance gain from this training strategy. A controlled ablation (e.g., with and without it) is needed to prove it improves final model quality, not just convergence speed.
4. **Evaluation on established, standardized benchmarks (e.g., COCO for layout control).** Using only a curated subset of Subject200K raises concerns about overfitting and generalizability. Results on public benchmarks are required to objectively assess controllability and quality.

### Deeper Analysis Needed (top 3-5 only)
1. **Systematic sensitivity analysis of the KSA mask threshold (ϵ).** The trade-off between efficiency and subject fidelity is central to KSA. A plot showing metrics like CLIP-I/DINOv2 vs. ϵ across a validation set is needed to prove robustness and guide users, beyond the single-example ablation.
2. **Validation that the observed attention redundancy generalizes beyond OminiControl.** The core motivation relies on attention patterns from one baseline. Analyzing if the same sparsity holds in other architectures (e.g., UniCombine) and across diverse prompts/conditions is critical to justify the fundamental design.
3. **Analysis of failure modes and limitations.** The paper shows only successful results. A dedicated analysis of when PAA or KSA fails (e.g., for non-rigid spatial deformations, multiple scattered subject instances, conflicting conditions) is necessary to understand the method's boundaries.
4. **Quantification of the "condition cache" benefit.** The caching mechanism is a key efficiency claim. The paper should break down the reported speedup/memory savings between the cache and the PAA/KSA modules to show their individual contributions.

### Visualizations & Case Studies
1. **Visualization of KSA masks across diverse and challenging prompts.** The single example in Figure 10 is insufficient. Showing masks for prompts with multiple objects, complex scenes, or abstract keywords would reveal how reliably KSA localizes subjects and where it errs.
2. **Side-by-side comparison of full attention maps vs. PAA/KSA approximations.** For representative examples, visualizing the full attention matrix and highlighting the pruned regions would directly demonstrate the claimed redundancy and validate that the approximations do not discard critical interactions.
3. **Case studies on condition conflict.** Showing generated results when spatial and subject conditions are semantically contradictory would expose how the model resolves conflicts and whether control fidelity degrades gracefully.

### Obvious Next Steps
1. **Theoretical computational complexity analysis.** The paper reports empirical speedups but lacks a formal complexity derivation for PKA vs. full attention and other efficient attentions (e.g., sliding window). This is essential for a rigorous efficiency claim.
2. **User study for perceptual quality and controllability.** Quantitative metrics like FID and CLIP scores are imperfect. A human evaluation (e.g., pairwise preference) on multi-condition alignment and image quality is standard for generative models and would strengthen the quality claim.
3. **Preliminary investigation into other condition types (e.g., style, pose).** The paper focuses on spatial and subject conditions. Discussing or experimenting with how PKA's principles could extend to other common control signals (e.g., via keyword scoping) would highlight its generality.

# Final Consolidated Review
## Summary
This paper introduces Patch-wise and Keyword-Aware Attention (PKA), a method to dramatically improve the efficiency of multi-condition control in Diffusion Transformers (DiTs). PKA replaces the standard, computationally prohibitive "concatenate-and-attend" strategy with two specialized modules: Position-Aligned Attention (PAA) for spatial conditions and Keyword-Scoped Attention (KSA) for subject-driven conditions, complemented by an early-timestep sampling strategy for training. The method achieves up to 10× inference speedup and a 5.12× reduction in attention VRAM while maintaining or improving generative quality.

## Strengths
- **Effectively addresses a critical bottleneck:** The paper clearly identifies and analyzes the quadratic computational overhead in multi-condition DiTs, motivating the work with empirical evidence of sparsity in attention patterns (Figures 2 & 3). The proposed solution directly targets this inefficiency.
- **Novel and well-motivated methodological design:** The PAA and KSA modules are conceptually clean, directly derived from the observed sparsity patterns for spatial and subject conditions. The complementary early-timestep sampling strategy is supported by a perturbation analysis, showing a data-driven approach to improving training convergence.
- **Comprehensive experimental validation:** The paper provides extensive evaluations across three multi-condition tasks, including compelling qualitative results (Figure 6) and a suite of quantitative metrics (Table 1). Efficiency benchmarks (Figures 7 & 8) demonstrate substantial, scalable improvements in latency and memory. Ablation studies effectively isolate the contributions of each component.

## Weaknesses
- **Insufficient detail on keyword extraction for KSA:** The KSA module's operation depends on identifying "keyword tokens" **K** from the text prompt. The paper does not specify how these tokens are reliably extracted (e.g., via a parser or heuristic), which is a critical implementation detail for reproducibility and robustness. This omission makes it difficult to assess the module's practicality in open-vocabulary settings.
- **Incomplete justification for the Condition Cache and KSA mask reuse:** The method caches Key and Value projections for condition tokens from the first denoising step, reusing them throughout generation. The paper does not discuss the validity of this approximation as the noisy image input changes significantly. Similarly, the reuse of the KSA mask across timesteps is justified only by a citation to "temporal consistency" without explanation or validation of mask stability, leaving a gap in the methodological rationale.
- **Lack of quantitative evaluation for scenarios with more than two conditions:** While the paper shows promising qualitative results for 3- and 4-condition generation (Figures 14-16) and plots efficiency trends, it provides no quantitative metrics (e.g., FID, controllability scores) for these more complex settings. This omission weakens the claim that the method maintains quality as the number of conditions scales.
- **Narrow experimental scope and dataset limitation:** The model is trained and evaluated on a curated subset of the Subject200K dataset, where captions are ensured to contain a descriptive keyword. This controlled environment does not fully demonstrate performance on the diverse, ambiguous, or keyword-absent prompts encountered in real-world use, limiting the evidence for generalizability.

## Nice-to-Haves
- A comparison of efficiency and quality against other state-of-the-art DiT acceleration methods (e.g., token caching or pruning techniques) would better situate PKA's performance within the broader literature on efficient diffusion models.
- A brief discussion or preliminary experiment on extending the PKA framework to other common condition types (e.g., pose, style) would help illustrate the generality of its principles.
- A more detailed sensitivity analysis for the KSA mask threshold (ε), plotting subject consistency metrics (e.g., CLIP-I) against ε across a validation set, would provide stronger guidance for users on the efficiency-fidelity trade-off.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "PAA assumes perfect spatial alignment and does not handle misalignment." The paper's ablation compares PAA to sliding window attention and demonstrates PAA's superior efficiency while maintaining quality, addressing the need for local context. The design choice is justified empirically.
- **Weakness:** "Requires variance estimates or statistical significance tests for quantitative results." While providing confidence intervals is good practice, single-run evaluation on large-scale benchmarks is the established norm in this field; the lack thereof does not invalidate the core results.
- **Weakness:** "Must include a comparison with UNet-based methods (e.g., ControlNet)." The paper's scope is explicitly improving efficiency within the DiT-based multi-condition paradigm. Demanding a cross-architecture comparison is scope creep, not a core requirement for evaluating the stated contribution.
- **Weakness:** "Needs a theoretical computational complexity derivation." The paper provides clear empirical complexity scaling (O(N) for PAA) and measured speedups. A formal derivation is a nice-to-have but not a standard requirement for an empirical systems paper in this area.
- **Weakness:** "Requires a user study for perceptual quality." While human evaluation can be valuable, the use of standard automated metrics (FID, CLIP, DINOv2) is acceptable and standard for assessing generative model quality in this context.

## Novel Insights
The paper's key novel insight is the categorization of multi-condition control into two distinct types—spatial-aligned and subject-driven—and the demonstration that each exhibits a specific, exploitable form of sparsity in the attention matrix. This observation directly motivates the design of two specialized, efficient attention modules (PAA and KSA) that replace wasteful full attention. Furthermore, the perturbation analysis revealing that visual conditions exert their strongest influence during early denoising timesteps leads to the non-obvious yet effective training strategy of skewing timestep sampling towards this early phase.

## Suggestions
- Add a subsection or appendix detailing the algorithm for extracting keyword tokens **K** from an input text prompt. Discuss any heuristics or models used and potential failure modes.
- Include a brief discussion or simple experiment validating the stability of the KSA mask and the condition KV cache across timesteps to strengthen the methodological justification.
- Provide quantitative evaluation (e.g., FID, controllability scores) for at least one three-condition task to substantiate the scalability claims made in the qualitative figures.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 4.0, 4.0]
Average score: 3.0
Binary outcome: Reject
