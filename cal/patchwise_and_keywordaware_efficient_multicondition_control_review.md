=== CALIBRATION EXAMPLE 44 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contributions: patch-wise and keyword-aware attention for efficient multi-condition control in DiTs. The abstract clearly summarizes the problem (computational bottleneck of concatenate-and-attend), the proposed solution (PKA with PAA and KSA, plus early-timestep sampling), and the key results (up to 10× speedup, 5.12× VRAM reduction, maintained quality). The claims are specific and appear supported by the later content. No issues.

### Introduction & Motivation
The introduction effectively frames the problem: fine-grained multi-condition control is needed, but the standard approach in DiTs leads to quadratic scaling. The central research question ("Does effective multi-condition control truly require such massive attention computation?") is well-motivated. The authors present preliminary analysis (Figures 2, 3) showing attention sparsity for spatial-aligned and subject-driven conditions, which directly motivates the design of PAA and KSA. Contributions are stated clearly. No major concerns.

### Method / Approach
This is the core technical section. The overall framework (PKA) is clearly depicted in Figure 4. However, there are several points requiring clarification and justification:

1.  **Equation and Terminology Consistency:** The text references Equation 2 for PAA, but the equation block appears incomplete in the provided text (it cuts off mid-sentence). The description of PAA as "one-to-one" attention is clear, but the formal definition must be complete. Similarly, the description of KSA involves equations (3) and (4) but the flow is somewhat fragmented. The authors must ensure all equations are fully and correctly stated.
2.  **Keyword Extraction for KSA:** The method hinges on identifying "keyword tokens" **K** for subject-driven conditions. The paper states **K** "typically contains just 1 to 2 tokens" but does **not** specify how these tokens are selected from the text prompt. Is this a manual annotation, an automated process using a parser or LLM, or does it rely on the text encoder's tokenization? This is a critical implementation detail for reproducibility. The robustness of the method to inaccurate keyword selection should also be discussed.
3.  **Mask Reuse in KSA:** KSA relies on temporal consistency to reuse a mask \(M^t\) computed at timestep \(t\) for step \(t+1\). However, the initial mask at the first denoising step (high noise) is computed from the noisy image. The authors should briefly discuss or empirically verify that the mask generated from a very noisy latent is sufficiently reliable to bootstrap the process. An ablation on the frequency of mask recomputation could strengthen this point.
4.  **Assumption of Static Condition KV Cache:** The Condition Cache mechanism assumes the Key and Value projections for condition tokens (SP, SJ) are static across the entire denoising trajectory and can be computed once and cached. This is a non-trivial approximation. The authors should provide empirical evidence (e.g., by analyzing the variation of these projections across timesteps) or a citation to justify this design choice, which is key to the efficiency gains.
5.  **Early-Timestep Sampling Justification:** The perturbation analysis (Figure 5) is conducted on a baseline model (Ominicontrol). While the insight that visual conditions are most influential early is plausible, the authors should explicitly connect this finding to their own training scheme. Does the early-timestep sampling strategy provide benefits when fine-tuning their own PKA architecture, or is it primarily beneficial for the baselines? The ablation in Figure 11 shows results for different \(\mu\), but it's unclear which model (baseline or PKA) is being fine-tuned here. This should be clarified.

### Experiments & Results
The experimental setup is generally solid, but several aspects need strengthening for an ICLR-level evaluation:

1.  **Fairness of Efficiency Comparison:** Figures 7 and 8 show impressive gains. The authors state "each condition is represented by 1024 tokens." It must be explicitly confirmed that all compared methods (OminiControl2, UniCombine, and Ours) use the **exact same** tokenization and sequence length for each condition. Any difference here would invalidate the comparison.
2.  **Quantitative Results Depth:** Table 1 shows PKA outperforms baselines on most metrics. However, the results are presented as single numbers without any measure of variance (standard deviation, confidence intervals). Given the stochastic nature of generation, reporting the mean and std over multiple runs or a large sample of generated images is essential to assess significance, especially where margins are small (e.g., CLIP-T). This is a standard expectation.
3.  **Ablation Study on KSA Threshold \(\epsilon\):** Figure 10 shows the trade-off with threshold \(\epsilon\). This is good. However, the claim that \(\epsilon\) is "not a sensitive hyperparameter" is based on a qualitative assessment of a single example. A quantitative ablation across the test set, showing metrics like subject consistency (CLIP-I/DINOv2) vs. \(\epsilon\), would substantiate this claim more convincingly.
4.  **Scalability Evaluation:** The appendices (Figures 14-16) show promising qualitative results for 2, 3, and 4 conditions. However, there is **no quantitative evaluation** for scenarios with more than two conditions. The paper's central claim is efficiency for multi-condition generation; thus, it is crucial to report standard quality and controllability metrics (FID, CLIP-I, F1, etc.) for the 3- and 4-condition tasks to demonstrate that performance does not degrade with condition count.
5.  **Baseline Selection and Implementation:** OminiControl2 and UniCombine are appropriate recent baselines. The paper should specify whether these baselines were re-implemented or if official implementations/code were used. For fairness, all methods should be fine-tuned from the same base checkpoint (FLUX.1) for the same number of iterations on the same dataset split.

### Writing & Clarity
The paper is mostly well-written. The main issue is the disjointed presentation of equations and some methodological steps (noted above). The narrative flow from problem to analysis to method is logical. Figures are helpful. Some minor clarifying edits are needed but do not impede understanding.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. Important limitations to acknowledge include:
*   **Condition-Type Assumption:** PKA is designed for spatial-aligned and subject-driven conditions. Its effectiveness on other condition types (e.g., global style, color palette) is not explored and may be limited.
*   **Keyword Dependency:** KSA's performance is contingent on correctly identifying keyword tokens. The method may struggle with complex prompts where the subject is not easily isolated to a few tokens.
*   **Cache Assumption:** The static KV cache for conditions is an approximation whose error is not quantified.
*   **Societal Impact:** The standard boilerplate on potential misuse for generating deepfakes or misinformation should be included, alongside positive uses in creative design.

### Overall Assessment
This paper presents a well-motivated and potentially impactful approach to a clear problem: the quadratic computational bottleneck in multi-condition Diffusion Transformers. The core idea of leveraging observed attention sparsity via specialized modules (PAA, KSA) is sound and novel. The reported efficiency gains are substantial. However, the current evaluation has notable gaps: lack of variance in quantitative results, missing quantitative analysis for >2 conditions, and insufficient detail on key implementation steps (keyword extraction, cache justification). Furthermore, the methodological description requires tightening, particularly around the equations and the training strategy's connection to the proposed architecture. If these issues are adequately addressed in a revision, the paper could meet ICLR's acceptance bar. In its current form, it is promising but requires significant strengthening of the experimental rigor and methodological clarity.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Patch-wise and Keyword-Aware Attention (PKA), a novel and efficient attention mechanism for multi-condition control in Diffusion Transformers (DiTs). PKA replaces the standard, computationally prohibitive "concatenate-and-attend" strategy with two specialized modules: Position-Aligned Attention (PAA) for spatially-aligned conditions (e.g., sketches, depth) and Keyword-Scoped Attention (KSA) for subject-driven conditions (e.g., reference images). Complemented by an early-timestep sampling strategy for training, the method achieves substantial reductions in inference time (up to 10×) and attention memory usage (up to 5.12×) while maintaining or improving generation quality on several multi-condition tasks.

### Strengths
1.  **Well-Motivated and Insightful Analysis:** The paper begins with a strong, empirical analysis of attention patterns in existing multi-condition DiTs (e.g., OminiControl), convincingly demonstrating the sparsity and redundancy in full cross-attention matrices. The observation that attention is highly localized (diagonally concentrated for spatial conditions, keyword-scoped for subjects) provides a clear and principled foundation for the proposed efficiency improvements.
2.  **Significant and Well-Validated Efficiency Gains:** The core contribution delivers impressive practical benefits. The experiments demonstrate speedups of 3.9× to 10× and VRAM reductions of 2.46× to 5.12× for the attention mechanism compared to strong baselines (UniCombine, OminiControl2). These metrics are critical for the deployment of complex multi-condition models.
3.  **Comprehensive Experimental Evaluation:** The paper validates its method thoroughly across three distinct multi-condition tasks (Subject-Canny, Subject-Depth, Canny-Depth) using a suite of standard metrics (FID, SSIM, CLIP-I, DINOv2, controllability scores). The qualitative results in figures are compelling and show clear visual improvements over baselines. The ablation studies are detailed and effectively isolate the contribution of each component (PAA, KSA threshold, early-timestep sampling).
4.  **Novel Training Strategy:** The proposed early-timestep sampling strategy, motivated by a perturbation analysis showing the outsized importance of early denoising steps for visual conditioning, is a novel and effective contribution. Figure 11 and Appendix A.3 show it accelerates convergence and improves final model fidelity.

### Weaknesses
1.  **Assumption of Input Alignment and Structure:** The effectiveness of PAA critically depends on the spatial alignment of condition maps (e.g., depth, sketch) with the noisy image tokens. The paper does not discuss the implications or handling of misaligned conditions, non-grid conditions, or conditions at different resolutions, which may limit the generalizability of this module in less structured scenarios.
2.  **Under-Specified Keyword Extraction:** The KSA module relies on identifying "keyword tokens" **K** from the text prompt to compute the initial relevance mask. The method for selecting these tokens is not detailed in the main text (e.g., is it manual, based on a noun parser, or learned?). This lack of clarity makes the method's automation and reproducibility for arbitrary prompts ambiguous.
3.  **Limited Discussion of Integration and Broader Context:** While the efficiency gains are compared against other DiT-based methods, there is no direct comparison or discussion of how the final performance/efficiency trade-off compares to the established, feature-fusion paradigm of UNet-based controllers (e.g., ControlNet, IP-Adapter). Placing the contribution within this broader landscape would strengthen the significance claim.
4.  **Justification for Temporal Mask Reuse in KSA:** The KSA module reuses the attention mask computed at timestep *t* for step *t+1*, justified by "temporal consistency." While plausible, this is a non-trivial approximation. The paper would benefit from a brief analysis or citation to validate that this reuse does not degrade subject fidelity compared to recomputing the mask at each step.

### Novelty & Significance
**Novelty:** The core idea of decomposing monolithic multi-condition attention into condition-type-specific, efficient modules (PAA and KSA) is novel within the context of Diffusion Transformers. Leveraging the observed sparsity of attention for efficiency is a known concept, but its application here—with a *position-aligned* one-to-one attention for spatial maps and a *keyword-scoped* dynamic masking for subjects—represents a fresh and clever synthesis. The early-timestep sampling strategy for flow matching fine-tuning is also a novel and insightful training improvement.

**Significance:** The work addresses a critical and timely bottleneck in scaling up controllable generation with DiTs. The demonstrated order-of-magnitude efficiency improvements are highly significant for practical applications, making complex multi-condition generation more accessible. The paper provides a clear, practical path forward for efficient DiT-based controllers, which aligns well with ICLR's focus on impactful algorithmic advancements in machine learning.

### Suggestions for Improvement
1.  **Clarify Keyword Selection and Robustness:** Explicitly describe the process for extracting keyword tokens **K** from a text prompt in the main method section. An ablation or analysis showing the sensitivity of results to the chosen keyword(s) would strengthen the robustness claim of KSA.
2.  **Discuss Scope and Limitations of PAA:** Add a brief discussion in the method or experiment section about the requirements for spatial condition alignment. Acknowledge potential limitations and propose or cite possible solutions (e.g., learnable warping, cross-attention for coarse alignment) for handling non-aligned conditions.
3.  **Expand Comparative Context:** Include a high-level discussion (potentially in the Related Work or Experiment sections) comparing the efficiency/performance profile of the proposed DiT-based PKA approach versus state-of-the-art UNet-based multi-condition methods. This would better position the contribution within the entire field of controllable generation.
4.  **Strengthen KSA Mask Reuse Justification:** Provide a small experiment or quantitative analysis in the appendix (e.g., comparing subject consistency scores when recomputing the mask every step vs. reusing it) to empirically support the temporal consistency assumption for mask reuse in KSA.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **End-to-end inference efficiency comparison.** The paper reports speedup and VRAM reduction for the attention module only, not the full generation pipeline. Without measuring total inference latency and memory (including non-attention operations), the practical efficiency claim is unconvincing.
2. **Comparison with state-of-the-art efficient DiT methods.** Baselines are limited to OminiControl2 and UniCombine. The paper must compare against other recent efficient DiT techniques (e.g., token pruning, caching methods like FastCache, DIP-Go) to substantiate claims of superior efficiency without quality loss.
3. **Quantitative evaluation for more than two conditions.** Scalability claims are supported only by qualitative examples for 3–4 conditions. Quantitative metrics (FID, controllability scores) on these harder settings are essential to validate that performance does not degrade.
4. **User study for perceptual quality and controllability.** Automated metrics (FID, CLIP) are insufficient for multi-condition generation. A user study is required to assess whether the improved efficiency comes at the cost of perceived image quality or condition adherence.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of failure cases and limitations.** The paper shows only successful examples. To trust the method, we need to see when it fails—e.g., with misaligned spatial conditions, multiple subjects, ambiguous keywords, or conflicting conditions. This reveals the method's robustness.
2. **Ablation on keyword selection for KSA.** The process for identifying keyword tokens is not described (manual vs. automatic). If automatic, its accuracy and impact on mask quality must be analyzed; if manual, the method’s practicality is severely limited.
3. **Effect of early-timestep sampling on text fidelity.** The sampling strategy is motivated by visual conditions, but its impact on textual prompt adherence (CLIP-T) across timesteps is not examined. It may harm text alignment, undermining the multi-condition claim.
4. **Interaction between multiple conditions of the same type.** The method treats conditions independently, but conflicts (e.g., sketch vs. depth) can arise. An analysis of how the model resolves such conflicts is necessary for real-world use.

### Visualizations & Case Studies
1. **Visual comparison of attention matrices (full vs. PKA).** Side-by-side visualizations of the full attention matrix and the sparse approximations from PAA and KSA are needed to validate the redundancy claim and show that PKA preserves essential interactions.
2. **Evolution of KSA masks across denoising timesteps.** Showing how the subject mask changes during generation would illustrate temporal consistency and whether the initial mask remains valid, which is critical for KSA’s correctness.
3. **Case studies on challenging or contradictory conditions.** Visual examples where conditions are noisy, misaligned, or semantically conflicting would demonstrate the method’s limits and help users understand its behavior.

### Obvious Next Steps
1. **Support for a broader range of condition types.** The method is designed for spatial and subject conditions; extending it to other controls (e.g., style, color palette) is a natural and expected generalization for a multi-condition framework.
2. **Dynamic condition handling during generation.** The current KV cache assumes static conditions. Enabling condition changes mid-sampling (e.g., interactive editing) is a practical next step for real applications.
3. **Combination with other efficiency techniques.** PKA could be integrated with token pruning or layer skipping for further gains; a preliminary experiment would showcase its composability.
4. **Preliminary application to video generation.** Given the emphasis on efficiency and temporal consistency, a video extension is a logical next step that should have been explored to demonstrate broader impact.

# Final Consolidated Review
## Summary
This paper introduces Patch-wise and Keyword-Aware Attention (PKA), an efficient framework for multi-condition control in Diffusion Transformers (DiTs). PKA replaces the standard quadratic-cost "concatenate-and-attend" strategy with two specialized modules—Position-Aligned Attention (PAA) for spatial-aligned conditions and Keyword-Scoped Attention (KSA) for subject-driven conditions—and an early-timestep sampling strategy for training. The method achieves up to 10× inference speedup and 5.12× reduction in attention VRAM while maintaining or improving generative quality across several multi-condition tasks.

## Strengths
- **Well-motivated design based on empirical analysis:** The paper begins with a convincing analysis of attention sparsity in existing multi-condition DiTs, showing that spatial conditions activate primarily along the diagonal and subject conditions activate only in keyword-relevant regions (Figures 2, 3). This directly motivates the design of PAA and KSA.
- **Substantial and well-demonstrated efficiency gains:** PKA delivers order-of-magnitude improvements, with measured speedups of 3.9–10× and attention VRAM reductions of 2.46–5.12× compared to strong baselines (Figures 7, 8), addressing a critical bottleneck for practical deployment.
- **Comprehensive evaluation across multiple tasks:** The method is validated on three distinct multi-condition tasks (Subject-Canny, Subject-Depth, Canny-Depth) using a suite of standard metrics (FID, SSIM, CLIP-I, DINOv2, controllability scores) and shows competitive or superior performance (Table 1, Figure 6). Ablation studies effectively isolate the contributions of each component.

## Weaknesses
- **Key implementation details are under-specified:** The KSA module requires identifying "keyword tokens" from the text prompt to compute the relevance mask, but the paper does not describe how these tokens are selected (e.g., manual annotation, automated parsing). This omission affects reproducibility and raises questions about robustness to arbitrary prompts.
- **Insufficient justification for critical approximations:** The method relies on two non-trivial approximations—caching static Key/Value projections for condition tokens across all denoising steps, and reusing the KSA mask computed at one timestep for the next—without empirical validation or citation to justify their reliability. These choices are central to the efficiency gains and require stronger support.
- **Scalability claims lack quantitative backing:** While the paper shows promising qualitative results for 3–4 conditions (Figures 15, 16), it provides no quantitative evaluation (e.g., FID, controllability scores) for scenarios beyond two conditions. This is a significant omission given the paper's core claim of efficient multi-condition generation.

## Nice-to-Haves
- Reporting variance estimates or confidence intervals for the quantitative metrics in Table 1 would strengthen the statistical significance of the results.
- A brief discussion comparing the efficiency/performance trade-off with established UNet-based multi-condition methods (e.g., ControlNet, IP-Adapter) would better contextualize the contribution within the broader field.
- Visualizing the attention matrices of PAA/KSA versus full attention would further validate the redundancy claim.

## Novel Insights
The paper provides a novel and insightful decomposition of multi-condition attention into two sparse, condition-type-specific operations. The analysis revealing that spatial-aligned conditions require only position-aligned attention and subject-driven conditions can be confined to keyword-activated regions is a genuine insight. Furthermore, the early-timestep sampling strategy, motivated by perturbation analysis showing visual conditions exert strongest influence early in denoising, is a novel and effective training improvement for flow-matching fine-tuning.

## Suggestions
- Explicitly describe the keyword selection process for KSA in the method section (e.g., using a noun phrase extractor or CLIP-based relevance scoring) and include a brief ablation on its robustness.
- Add a short analysis or citation to justify the static KV cache assumption and the temporal mask reuse in KSA, perhaps by measuring the variation of condition token projections across timesteps or comparing subject consistency with vs. without mask reuse.
- Include quantitative evaluation (using the same metrics as Table 1) for at least one 3-condition task to substantiate the scalability claim.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 4.0, 4.0]
Average score: 3.0
Binary outcome: Reject
