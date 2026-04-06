=== CALIBRATION EXAMPLE 32 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution. The abstract clearly states the problem (cost of dense annotations), the proposed solution (first weakly-supervised LTA using only transcripts), the key methodological components, and the main experimental finding (competitive performance). The claim of being the "first" weakly-supervised approach for dense LTA appears valid based on the related work cited. The abstract is well-structured and supports the paper's narrative.

### Introduction & Motivation
The introduction effectively motivates the problem by highlighting the scalability issue of dense frame-level annotations for LTA. It correctly positions prior work, including the semi-weakly supervised approach of Zhang et al. (2021), to justify the novelty of using *only* transcripts. The contributions are listed clearly and align with the content of the paper. The motivation for using transcripts—their semantic abstraction aligns with the logical progression needed for LTA—is convincing.

### Related Work
This section is comprehensive, covering TAS, action anticipation, LTA, and sequence-to-sequence modeling. It adequately distinguishes the proposed work from prior weakly-supervised TAS methods and the semi-supervised LTA approach. One minor gap: a more direct comparison or discussion of how recent unsupervised/weakly-supervised TAS methods (e.g., CLOT, OTAS) might relate to or differ from the LTA task could add depth. The connection to sequence alignment techniques (CTC, DTW, CRF) is appropriate and sets the stage for the method.

### Methodology
The problem definition is precise. The overall architecture (Fig. 2) is logical, integrating an encoder, temporal alignment, cross-modal attention, and a decoder.

*   **Temporal Alignment Module:** The use of ATBA (Xu & Zheng, 2024) to generate pseudo-labels is a sensible choice from the weakly-supervised TAS literature. However, the paper does not sufficiently analyze the sensitivity of the overall framework to the quality of these pseudo-labels. This is a critical dependency. An ablation or analysis of how alignment errors propagate to anticipation performance would strengthen the method's validation.
*   **Cross-Attention Layer:** The proposed local cross-attention mechanism, gated by pseudo-labels, is a novel and interesting way to ground video features with transcript semantics. The ablation study (Tables 3,4) confirms its importance.
*   **Loss Functions:** The combination of alignment (ATBA-based), segmentation (CTC), and anticipation (CRF + duration) losses is well-motivated. The use of CTC for transcript-level supervision is standard but appropriate. The proposed self-supervised affinity-based duration loss (Eq. 7) is clever, as it avoids direct duration supervision. However, its reliance on class duration priors estimated from the (noisy) segmentation head predictions needs more discussion. How robust is this to mis-segmentation in the observed part? The CRF for anticipation coherence is a good addition.
*   **Reproducibility:** The description is generally sufficient. Missing details include the specific values or tuning process for the loss weights (γ1, γ2, γ3) and more specifics on the "progressive training scheme" (e.g., learning rates for each stage). The number of learnable queries for the decoder is given, but the rationale for these specific numbers (8 for Breakfast, 20 for 50Salads) is not explained.

### Experiments & Results
The experimental setup is solid, using standard benchmarks, features, protocols, and metrics.

*   **Main Results (Tables 1 & 2):** The results are compelling. TbLTA significantly outperforms the prior weakly-supervised baseline (WS-DA) and is often competitive with fully-supervised methods, especially on Breakfast. This strongly supports the core claim. The performance gap on 50Salads is honestly presented and discussed (attributed to denser actions and weaker temporal regularities). The EGTEA results show a promising trend for rare classes.
*   **Ablation Studies (Tables 3, 4, 5):** These are necessary and well-executed. They clearly demonstrate the contribution of key components: CTC loss, cross-attention, and the duration head. The finding that duration loss helps more on Breakfast than 50Salads is insightful and discussed.
*   **Baselines:** The chosen supervised baselines (FUTR, ActFusion) are strong and recent. It would be beneficial to also include a comparison to a *supervised* method that uses a similar architectural backbone (e.g., a transformer encoder-decoder) to better isolate the cost of weak supervision versus architectural differences.
*   **Statistical Significance:** Results are averaged over multiple splits, which is good practice. Reporting standard deviation or confidence intervals would further strengthen the claims.
*   **Qualitative Results:** The provided examples and the admission that duration estimation remains a challenge are honest. More failure case analysis would be valuable.

### Writing & Clarity
The paper is generally well-written and logically organized. Figures 1 and 2 effectively illustrate the framework and data flow. Some sections, particularly the detailed description of the cross-attention mechanism (Eq. 1, 2) and the loss functions, are dense but manageable. The use of placeholder text like "- means stochastic protocol" in Table 1 and incomplete sentences ("- means stochastic protocol") are likely parser artifacts and not the authors' fault.

### Limitations & Broader Impact
The conclusion mentions the challenge of duration estimation, which is a key limitation. The paper could more explicitly discuss other limitations: 1) The reliance on a pre-trained temporal alignment module (ATBA) and its potential failure modes; 2) The assumption of a correct and complete transcript during training (how robust is the method to transcript errors or omissions?); 3) The computational cost of the progressive training scheme. A broader impact statement is absent; while the work is foundational, a brief note on positive societal impacts (e.g., reducing annotation burden for assistive robotics) and potential negatives (e.g., biases in transcripts) would be appropriate for ICLR.

### Overall Assessment
This paper presents a novel, well-motivated, and technically sound contribution: the first framework for dense Long-Term Action Anticipation using only video transcripts as supervision. The core idea is impactful, as it addresses a significant scalability bottleneck. The methodology intelligently combines and adapts techniques from weakly-supervised segmentation (alignment, CTC) and anticipation (decoder, CRF), with the novel addition of pseudo-label-gated cross-modal attention. The experimental validation is thorough, demonstrating competitive performance with fully-supervised methods on established benchmarks, which is a remarkable result. The main weaknesses are the insufficient analysis of pseudo-label error propagation and some missing implementation details for full reproducibility. However, the contribution stands as significant and likely to influence future work in scalable video understanding. It meets the high bar for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces TbLTA, the first fully weakly-supervised framework for dense Long-Term Action Anticipation (LTA), requiring only video transcripts (ordered action lists) during training instead of costly frame-level annotations. The method uses a temporal alignment module to generate pseudo-labels, a cross-modal attention mechanism to ground video features with transcript semantics, and combines CTC and CRF losses for sequence alignment and coherence. Experiments on Breakfast, 50Salads, and EGTEA benchmarks show competitive performance compared to fully supervised methods.

### Strengths
1. **Novel Problem Formulation**: The paper successfully identifies and addresses a significant gap by proposing the first fully weakly-supervised method for dense LTA, using only transcripts. This is a clear step towards scalable anticipation models (Sections 1, 3).
2. **Comprehensive Experimental Validation**: The method is evaluated on three standard benchmarks (Breakfast, 50Salads, EGTEA) under multiple observation ratios, with results often competitive with supervised baselines, convincingly demonstrating the viability of transcript-based supervision (Tables 1, 2, Section 4.2).
3. **Well-Designed Integration of Techniques**: The architecture cleverly integrates several components: the ATBA module for pseudo-label generation, cross-modal attention for feature grounding, and CTC/CRF losses for sequence learning. The ablation studies justify these design choices (Sections 3.1, 3.2, 4.3).

### Weaknesses
1. **Heavy Reliance on Pseudo-Label Quality**: The core supervision comes from pseudo-labels generated by the ATBA module. While ablations show its importance, the paper does not deeply analyze the failure modes or error propagation when this alignment is noisy, especially for complex activities in 50Salads where performance lags more (Section 4.2, Table 1).
2. **Simplistic Duration Modeling**: The affinity-based duration loss (Eq. 7) relies on class-wise priors estimated from the observed segment. This is a strong assumption that may not hold for actions with highly variable durations or for unseen activity contexts, which is a noted remaining challenge (Section 4.3, 5).
3. **Incremental Technical Contribution**: While the problem formulation is novel, many core techniques (ATBA for alignment, CTC loss, cross-attention) are adaptations from prior weakly-supervised segmentation works. The primary novelty lies in their novel combination and application to LTA (Sections 2, 3).

### Novelty & Significance
**Novelty**: The paper's main novelty is conceptual: formulating and demonstrating that dense LTA can be learned with transcript-only supervision, a previously unexplored setting. The technical architecture is a thoughtful synthesis of existing alignment and attention mechanisms tailored for this new task.
**Significance**: The work is significant as it provides a path towards more scalable LTA by drastically reducing annotation cost. The results are promising, showing that high-level semantic supervision can effectively guide long-horizon prediction. For ICLR, this represents a meaningful advance in making video anticipation models more practical.

**Clarity**: The paper is generally well-written and the model (Fig. 2) is clearly explained. Some details, like the exact training scheme progression and the "global-local contrast" loss, are relegated to supplementary material.
**Reproducibility**: The method is described in sufficient detail, and the promise of code release enhances reproducibility. The use of standard datasets and features (I3D) facilitates comparison.

### Suggestions for Improvement
1. **Error Analysis and Limitations**: Include a dedicated analysis of failure cases. For instance, qualitatively show videos where pseudo-label alignment fails and how this impacts anticipation, or analyze which action classes suffer the most in duration prediction. This would provide deeper insight into the method's current limits.
2. **Refine Duration Modeling**: Propose and evaluate a more robust duration prediction mechanism. Instead of simple class-wise priors, could the model learn a distribution over durations conditioned on the contextual video features or the predicted action sequence?
3. **Ablation on Transcript "Quality"**: Conduct an experiment to test robustness to imperfect transcripts (e.g., missing or permuted actions). Since transcripts are cheaper but not free, understanding the sensitivity to transcript errors would be valuable for real-world applicability.
4. **Clarify Training Stability**: The progressive training scheme (pre-training, alignment, full training) seems crucial. Provide more analysis on why this is needed and its impact on final performance. A sensitivity analysis on the number of epochs per stage or the loss weights (γ1, γ2, γ3) would be helpful.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with recent weak/semi-supervised LTA or TAS methods.** The paper only compares to one weak method (WS-DA, 2021). To properly situate the contribution, it must be compared to contemporary methods that also reduce annotation cost (e.g., other transcript-based TAS methods applied to LTA). Without this, the claim of being the first and competitive is not fully substantiated.
2. **Ablation on the temporal alignment module.** The entire pipeline depends on pseudo-labels from the ATBA module. An ablation comparing ATBA to other alignment techniques (e.g., differentiable DTW, CTC alone) is critical to show the chosen component is necessary and optimal for the task.
3. **Experiments on non-procedural or less structured datasets.** The method is evaluated only on cooking activities (Breakfast, 50Salads, EGTEA), which have strong procedural regularities. Testing on datasets with less rigid action order (e.g., sports, surveillance) is needed to assess generalizability beyond recipe-like activities.
4. **Robustness to transcript noise/incompleteness.** Real-world transcripts may contain errors (missing actions, wrong order). The method's sensitivity to such noise is unexplored; experiments with synthetic transcript corruption are required to judge practical utility.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of pseudo-label quality.** The model is trained on pseudo-labels, but their frame-wise accuracy (vs. ground truth) is not reported. This is essential to trust that the learning signal is reliable and to diagnose error propagation.
2. **Detailed error analysis for anticipation.** The paper reports MoC scores but does not break down whether errors are due to incorrect action predictions, wrong durations, or mistaken action ordering. This analysis is crucial to understand the method's limitations and guide future work.
3. **Analysis of cross-modal attention effectiveness.** It is claimed that cross-attention grounds video features with transcript semantics. Without visualizing attention weights or quantifying how well attended frames correspond to action segments, the mechanism's contribution remains unverified.
4. **Performance analysis as a function of anticipation horizon.** The paper evaluates fixed future percentages but does not analyze how accuracy degrades as the model predicts further into the future. This would test the claim of modeling long-range dependencies.

### Visualizations & Case Studies
1. **Side-by-side timelines of ground truth, pseudo-labels, and predictions.** Visual comparisons for multiple videos would immediately reveal the quality of temporal alignment, segmentation, and anticipation, highlighting where the method succeeds or fails (e.g., duration errors, action swaps).
2. **Heatmaps of cross-modal attention weights.** Showing which video frames are attended to for each transcript action token would demonstrate whether the model achieves meaningful semantic grounding, as claimed.
3. **Failure case studies.** Displaying examples where anticipation is grossly incorrect (e.g., wrong action sequence, severe duration misestimation) would concretely illustrate the method's current limitations.

### Obvious Next Steps
1. **Iterative pseudo-label refinement (self-training).** The pseudo-labels are generated once. An obvious improvement is to iteratively refine them using the model's own improved predictions, which could boost performance and is a standard step in weakly-supervised learning.
2. **Incorporating large language models (LLMs) for temporal common sense.** Transcripts are text; using an LLM to infer likely action durations, ordering constraints, or typical progressions could provide a stronger prior and improve anticipation, especially for rare actions.
3. **Extension to an online/streaming setting.** The current method assumes a fixed observation ratio. Adapting it to process streaming video and update anticipation dynamically is a logical next step for real-world deployment.

# Final Consolidated Review
## Summary
This paper introduces TbLTA, the first fully weakly-supervised framework for dense Long-Term Action Anticipation (LTA). It learns to anticipate future actions using only video transcripts (ordered action lists) during training, eliminating the need for costly frame-level annotations. The method integrates a temporal alignment module to generate pseudo-labels, a cross-modal attention mechanism to ground video features with transcript semantics, and a combination of CTC and CRF losses for sequence learning.

## Strengths
- **Novel and impactful problem formulation.** The paper successfully addresses a significant scalability bottleneck by demonstrating, for the first time, that dense long-term action anticipation can be learned with transcript-only supervision. This opens a new, more practical direction for the field (Sections 1, 4.2).
- **Strong empirical validation.** The method achieves performance competitive with fully-supervised baselines on established benchmarks (Breakfast, 50Salads), particularly at longer observation horizons. This convincingly supports the core claim that high-level semantic supervision is sufficient for the task (Tables 1, 2).
- **Effective integration of a novel cross-modal component.** The proposed local cross-attention mechanism, gated by pseudo-labels to contextually ground video features with transcript semantics, is a specific and effective design choice justified by clear ablation studies (Section 3.1, Tables 3, 4).

## Weaknesses
- **Inherent dependency on pseudo-label quality.** The framework's supervision stems from pseudo-labels generated by an external temporal alignment module (ATBA). The paper does not analyze how errors in this alignment propagate to the anticipation decoder, which is a core vulnerability, especially for complex datasets like 50Salads where performance lags more noticeably (Section 4.2).
- **Limitations in duration modeling.** The proposed affinity-based duration loss relies on class-wise priors estimated from the (noisy) observed segment, a strong assumption that struggles with actions of highly variable length. This is acknowledged as a remaining challenge and is a genuine weakness of the current approach (Section 4.3, 5).

## Nice-to-Haves
- A more detailed analysis of failure cases (e.g., visualizing where pseudo-label alignment breaks down) would provide deeper insight into the method's current limits.
- Exploring iterative refinement of pseudo-labels (self-training) is a natural next step that could improve performance and robustness.

## Novel Insights
The primary novel insight is that the procedural narrative captured in a simple action transcript contains sufficient high-level semantic structure to supervise dense, long-horizon action anticipation. The paper demonstrates that by combining temporal alignment (to generate frame-level pseudo-signals) with semantic grounding (via cross-modal attention), a model can learn to forecast future actions competitively without any temporal boundary annotations. This establishes a new, more scalable paradigm for the LTA task.

## Suggestions
- In the limitation or future work section, include a dedicated discussion on the error propagation from the temporal alignment module and propose directions for more integrated or robust alignment strategies.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
