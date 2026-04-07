=== CALIBRATION EXAMPLE 33 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title and Abstract
The title clearly reflects the paper's core contribution: a weak-to-strong (W2S) learning paradigm for generalized video quality assessment (VQA). The abstract succinctly states the motivation (annotation cost, poor generalization), the core finding (W2S effect in VQA), the proposed enhancements (integrating diverse teachers via ranking, iterative training), and the outcome (state-of-the-art results, especially OOD). All claims are supported in the main text.

### Introduction and Motivation
The introduction effectively motivates the problem: the generalization challenge in supervised VQA due to limited and costly human annotations, and the shortcomings of existing self-supervised methods. It clearly outlines the two research questions (effectiveness of W2S in subjective VQA, and how to enhance it) and summarizes the contributions. The use of Figure 1 to illustrate the OOD performance drop is compelling.

### Related Work
Covers key areas: supervised VQA, self-supervised VQA, ranking-based VQA, and weak-to-strong generalization. The discussion is appropriate and sets the context. A minor gap: the related work on iterative self-training or bootstrapping in quality assessment could be mentioned to better situate the iterative W2S component.

### Section 3: Weak-to-Strong Learning for VQA (Problem Setup & Initial Evidence)
This section is crucial for establishing the basic W2S effect. The problem setup is clear. The choice of weak teachers (five SOTA models) and the strong student (LLaVA-OneVision-Chat-7B) is justified, though the massive parameter disparity (Table 4) raises the question of how much gain stems from capacity alone. However, the results in Figure 4 and Appendix Table 5 convincingly show that students trained on pseudo-labels can match or exceed their teachers, with **significant OOD gains** (e.g., MinimalisticVQA(VII)-labeled student improves SRCC on LIVE-YT-HFR from 0.061 to 0.318). The comparison with a supervised student (trained on LSVQ human labels) in Table 5 is particularly insightful, showing that W2S can yield better OOD generalization even when in-domain performance is slightly lower. This provides strong empirical evidence for the W2S effect in VQA.

**Concerns:**
1.  **Model Capacity Confound:** While the results are promising, the dramatic OOD improvements could be partly attributed to the student's vastly larger architecture and pre-trained knowledge (from LLaVA). A more controlled ablation—e.g., training a student with similar capacity but without the weak supervision—would help isolate the contribution of the W2S knowledge transfer itself. The paper mentions using the same student architecture for the supervised baseline (LSVQ-labeled), which is good, but a comparison using a smaller student model (closer to teacher capacity) would strengthen the claim of a "W2S effect" beyond just scaling up.
2.  **Dataset Construction Reproducibility:** The process of collecting 3M videos and selecting 200k via mixed-integer programming to match LSVQ's low-level feature distributions (Appendix A) is non-trivial. While described, the exact implementation details (e.g., the optimization objective, library used) are critical for reproducibility. The provided code link should ideally clarify this.

### Section 4: Improving Weak-to-Strong Learning
This section presents the core methodological innovations: unifying supervision via ranking and iterative training.

*4.1 Unifying Diverse Supervision Signals*
The ranking formulation is a sensible way to merge signals from different teachers (homogeneous ensemble and synthetic distortion simulators). The use of statistical significance thresholds (based on ensemble variance) for label assignment is clever. The inclusion of synthetic distortions (spatial, temporal, streaming) enriches the supervision space. Figure 5 and the accompanying description are clear.

*4.2 Iterative W2S Training Strategy*
The idea of recycling the strong student as a new teacher is intuitive and aligns with self-training paradigms. The difficult-sample selection strategy—using gMAD for teacher-ensemble pairs and ground-truth distortion levels for synthetic pairs—is well-motivated to focus the model on challenging cases. Equation (1) is clear.

*4.3 Training Strategy*
The combined loss (cross-entropy + confidence loss) follows prior W2S work and helps mitigate noise. The adaptive weighting mechanism (Eq. 25-26) is adequately described.

**Concerns:**
1.  **Error Propagation in Iterative Training:** The iterative process (3 stages) risks amplifying errors or biases from the initial weak teachers. While the confidence loss and difficulty sampling are mitigations, a deeper analysis of how the pseudo-label distribution evolves across iterations (e.g., calibration plots, analysis of samples selected by gMAD) would increase confidence in the stability of the approach.
2.  **Synthetic Distortion Realism:** The synthetic distortions, while diverse, are relatively simple and may not capture the complex, non-linear degradations in real-world videos (as acknowledged in the Introduction regarding prior self-supervised methods). The paper shows they help (Table 1, Model II), but it's unclear if their benefit plateaus or if more advanced simulators (e.g., learned generative models) would yield further gains.
3.  **Computational Cost:** The full pipeline (training a 7B-parameter LMM for multiple iterations on 200k+ video pairs) is extremely resource-intensive (8xA800 GPUs for ~2 days per stage?). This is a practical limitation for widespread adoption and should be more explicitly discussed in the limitations.

### Experiments & Results (Section 3.3, 4.4, and Appendix D)
The experimental design is comprehensive, using 10 benchmarks split into in-domain and OOD categories. Table 1 is the centerpiece, showing incremental improvements from each component (ensemble, synthetic teachers, confidence loss, iterative stages). The final model (V) achieves SOTA or competitive results on most datasets.

**Strengths:**
- Clear ablation study demonstrating the value of each proposed component.
- Impressive OOD gains, e.g., SRCC on LIVE-YT-HFR improves from 0.329 (Q-Align teacher) to 0.683 (final model).
- Outperforms recent supervised LMM-based methods (VQA², VQAThinker) without using human labels, a strong result.

**Concerns:**
1.  **Statistical Significance:** The paper reports correlation coefficients but does not indicate statistical significance (e.g., confidence intervals, p-values, or multiple runs with different seeds). Given the variability inherent in VQA benchmarks and training, it is important to show that the reported improvements (especially the modest in-domain gains) are statistically reliable.
2.  **Fairness of SOTA Comparison:** The comparison with VQAThinker and VQA² is essential, but the paper does not detail if the evaluation protocols (e.g., train/test splits, pre-processing) are identical. A brief statement confirming alignment with the original papers' evaluation setups would prevent concerns about unfair comparison.
3.  **Lack of Qualitative Analysis:** While quantitative results are strong, the paper would benefit from qualitative examples or error analysis. For instance, showing video pairs where the final model correctly ranks quality but the initial teacher fails would illustrate the "challenging cases" the iterative strategy aims to address.
4.  **Anchor Selection for Inference:** Appendix C.3 describes the inference procedure which relies on comparisons with five anchor videos. The choice and representativeness of these anchors can impact the final score. The paper should specify how these anchors were selected (e.g., from the training set?) and discuss the sensitivity of results to this choice.

### Discussion and Conclusion
The discussion appropriately contextualizes the work, highlighting its potential as a pathway toward scalable VQA foundation models by integrating diverse supervision sources. The conclusion summarizes contributions accurately.

**Limitations:** The paper briefly mentions computational cost in the training details but does not have a dedicated "Limitations" section. A subsection explicitly discussing limitations (computational cost, potential error propagation, reliance on the initial teacher ensemble's biases, and the simplified nature of synthetic distortions) would strengthen the paper's rigor and self-awareness.

### Writing and Clarity
The paper is generally well-written and logically structured. Figures are informative. Some minor clarity issues:
- In Section 3.2, the description of the student model architecture references Figure 3 and Appendix C.1, but the figure is somewhat busy and the caption could better guide the reader.
- The prompt templates in Appendix B.2 are clear, but it would be helpful to see an example of a full input sequence during training.

### Reproducibility and Ethics
The reproducibility statement is good, providing a code link. The dataset filtering pipeline (public content with permissive licenses) and ethics statement are appropriate.

## Overall Assessment

This paper presents a novel and well-executed study on weak-to-strong learning for video quality assessment. The core finding—that a strong model can learn from weak VQA teachers and achieve superior generalization, especially on out-of-distribution data—is empirically solid and significant for the field. The proposed enhancements (teacher ensembling, synthetic distortions, iterative training with difficulty sampling) are thoughtful and yield substantial performance gains, culminating in state-of-the-art results without human annotations. The work is timely, addressing the critical problems of annotation cost and generalization in VQA.

The main concerns that need addressing are: (1) better isolation of the W2S effect from pure model capacity scaling, (2) analysis of error propagation and stability in iterative training, (3) reporting statistical significance of results, and (4) a more explicit discussion of limitations (computational cost, synthetic distortion simplicity). If these points are adequately addressed in a revision, the paper makes a strong contribution suitable for ICLR, offering a promising new paradigm for scalable and generalized VQA.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a weak-to-strong (W2S) learning paradigm for no-reference video quality assessment (VQA), aiming to improve generalization without relying on human annotations. The core idea is to train a strong student model (a large multimodal model) using pseudo-labels from multiple weaker VQA models and synthetic distortion simulators, formulated as a ranking task, and to iteratively refine the student by recycling it as a teacher for subsequent cycles. The method achieves state-of-the-art performance on both in-domain and, more notably, out-of-distribution (OOD) benchmarks.

### Strengths
1. **Clear Problem Formulation and Motivation**: The paper convincingly identifies and articulates a critical problem in VQA: the poor generalization of supervised models to unseen content/distortions and the high cost of human annotation. The proposed W2S paradigm is a direct and motivated response to this challenge.
2. **Comprehensive Experimental Validation**: The paper provides extensive, well-structured experiments across ten diverse benchmarks (five in-domain, five OOD). The step-by-step ablation studies (Table 1) clearly demonstrate the incremental gains from each proposed component (teacher ensemble, heterogeneous teachers, confidence loss, iterative training). The results are compelling, showing consistent improvements, particularly strong gains on challenging OOD datasets (e.g., SRCC on LIVE-YT-HFR improves from 0.329 to 0.683).
3. **Practical Impact and Reproducibility**: The work offers a practical pathway to scalable VQA model development. The methodology is described in detail, including dataset construction (200k videos, matched distributions), training protocols, and inference procedures. The authors provide a code link and include an extensive appendix, which significantly aids reproducibility and aligns with ICLR's emphasis on these aspects.

### Weaknesses
1. **Limited Theoretical Justification for W2S Efficacy in VQA**: While the empirical results are strong, the paper provides limited theoretical or conceptual insight into *why* W2S generalization works so well for the subjective task of VQA. The introduction raises this as a critical question, but the analysis remains primarily empirical (e.g., attributing OOD gains to a larger training dataset). A deeper discussion connecting the properties of weak teachers (e.g., their failure modes) to the student's learned representations would strengthen the contribution.
2. **Insufficient Analysis of Failure Cases and Limitations**: The paper highlights successes but offers minimal analysis of where the proposed method still falls short or fails. For instance, while OOD performance improves dramatically, absolute performance on some OOD datasets (e.g., initial SRCC of 0.061 on LIVE-YT-HFR for a teacher) remains a challenge. A qualitative analysis of videos where the student fails to outperform the teacher or where the iterative strategy plateaus would provide a more balanced view of the method's limitations.
3. **Complexity of the Final Pipeline**: The final proposed system involves multiple stages: collecting a large dataset, running five teacher models and multiple distortion simulators, performing three iterative training cycles with difficulty sampling, and using a complex ranking-based inference. While effective, the computational and engineering cost is non-trivial. The paper would benefit from a discussion of this trade-off between performance and complexity, and perhaps a simpler baseline comparison (e.g., training on a mix of synthetic distortions alone).

### Novelty & Significance
**Novelty:** The work successfully adapts the emerging concept of weak-to-strong generalization—previously explored in NLP and reward modeling—to the domain of perceptual video quality assessment. The integration of heterogeneous teachers (off-the-shelf VQA models *and* synthetic distortion simulators) under a unified ranking formulation and the iterative, difficulty-aware training strategy are novel contributions to the VQA field.

**Significance:** The significance is high. The paper demonstrates a viable path to break the annotation bottleneck in VQA, potentially enabling the development of more robust and generalizable models. Achieving SOTA results without human labels is a notable milestone. The findings and framework could influence broader areas involving subjective evaluation and model alignment, as suggested by the authors.

### Suggestions for Improvement
1. **Deepen the Analysis of "Why It Works"**: Conduct and present a more detailed analysis to explain the W2S effect in VQA. This could involve visualizing the feature spaces of teachers vs. students, analyzing the types of "difficult samples" selected by gMAD, or discussing how the ranking loss might help calibrate predictions across different teacher scales.
2. **Include a "Cost vs. Performance" Analysis**: Add a section or table discussing the computational requirements (e.g., GPU hours for pseudo-labeling, training iterations) of the full pipeline versus supervised baselines. This would help readers assess the practical feasibility and trade-offs of the proposed method.
3. **Strengthen the Comparison to Semi/Un-supervised Baselines**: While compared to strong supervised SOTA, the paper could more directly compare against recent self-supervised or unsupervised VQA methods (briefly mentioned in Related Work) to better position the gains of the W2S paradigm within the annotation-free landscape.
4. **Improve Figure and Table Readability in the Main Paper**: Some figures (e.g., Fig. 1, 4) and the large Table 1 are essential but could be made more reader-friendly in the main text. For example, consider moving the per-dataset results for the initial W2S experiment (currently in Appendix D.1, Table 5) to the main paper to immediately support the claim of a W2S effect. Ensure axis labels and legends in graphs are clearly legible.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on model capacity**: The "strong" student is a much larger LMM (7B params) versus teachers (e.g., 87M-8B). A critical experiment is to train a student with the *same architecture* as a teacher (e.g., Swin-B) using pseudo-labels. Without this, the claimed "weak-to-strong generalization effect" is conflated with simply using a larger model.
2. **Comparison to other weakly/self-supervised VQA methods**: The paper only compares to supervised SOTA. To substantiate the claim of a "new paradigm," results must be shown against recent self-supervised (e.g., ConvIQT, QPT-v2) or other label-free methods. Otherwise, it's unclear if the gains are novel.
3. **Cross-dataset generalization test**: To prove generalization beyond teacher biases, train a student using pseudo-labels from models trained on *one* dataset (e.g., KoNViD) and evaluate on *others* (e.g., LIVE-VQC). The current "OOD" test uses datasets where teachers were all trained on LSVQ, which is insufficient.
4. **Human evaluation correlation**: The ultimate metric is alignment with human perception. A small-scale human study on videos where the student and teachers disagree most would validate that improved SRCC/PLCC translates to better human judgment.

### Deeper Analysis Needed (top 3-5 only)
1. **Error analysis and failure modes**: The paper lacks analysis of where the method *fails*. For example, on which distortion types or content categories does performance degrade? This is critical for understanding the method's limitations and for OOD claims.
2. **Quantifying what the student learns beyond teachers**: Analysis (e.g., feature similarity, attention maps) comparing student and teacher representations on "difficult" samples selected by gMAD is missing. This would validate the claim that iterative training captures knowledge beyond teachers.
3. **Impact of synthetic distortion diversity**: The paper claims synthetic teachers "enrich supervision." An analysis is needed to show which distortion types contribute most to OOD gains, and whether simply adding more synthetic data of a single type yields similar benefits.
4. **Sensitivity to teacher ensemble composition**: The performance likely depends on which teachers are ensembled. An analysis of how performance varies with different subsets (e.g., removing the strongest teacher) is needed to assess the robustness of the ensemble strategy.

### Visualizations & Case Studies
1. **Qualitative comparisons on high-disagreement videos**: Visualize video pairs where the student's ranking contradicts the teacher ensemble's ranking, alongside the predicted scores. This would concretely show the "weak-to-strong" effect and whether student judgments are perceptually plausible.
2. **Visualization of "difficult samples"**: Show examples of videos selected by the gMAD strategy across iterations. This would reveal what "challenging" means (e.g., specific distortions, content) and whether the selection strategy is meaningful.
3. **Failure case gallery**: Display videos where the student model performs significantly worse than the best teacher (especially on in-domain data). This is essential for a balanced assessment and to identify boundaries of the method.

### Obvious Next Steps
1. **Ablation study on framework components**: The paper progressively adds components (ensemble, synthetic teachers, confidence loss, iteration) but does not ablate their individual contributions in a controlled way (e.g., removing synthetic teachers while keeping everything else fixed). This is necessary to justify the design.
2. **Analysis of computational cost and efficiency**: The method uses a 7B LMM and iterative training on 200k+ videos. A discussion of training/inference cost relative to teacher models is missing, which is critical for practical adoption and for understanding the trade-off for performance gains.
3. **Exploration of alternative strong backbones**: The paper uses one specific LMM. Trying other large vision-language models (e.g., InternVL, LLaVA-NeXT) as the strong student would test the generality of the approach and whether gains are backbone-specific.

# Final Consolidated Review
## Summary
This paper proposes a novel weak-to-strong (W2S) learning paradigm for no-reference video quality assessment (VQA). It first empirically demonstrates that a strong student model (a large multimodal model) can learn from pseudo-labels generated by weaker, off-the-shelf VQA teachers and achieve superior generalization, especially on out-of-distribution (OOD) data. The core contribution is a framework that enhances W2S learning by (1) integrating supervision from both homogeneous VQA models and heterogeneous synthetic distortion simulators via a learn-to-rank formulation, and (2) employing iterative training where each strong student becomes the teacher for the next cycle, progressively focusing on challenging samples. The final model achieves state-of-the-art results across ten benchmarks without using human annotations, with particularly large gains on OOD datasets.

## Strengths
- **Empirically Validates a New Paradigm:** The paper provides clear, extensive evidence for a weak-to-strong generalization effect in VQA. The initial experiment (Section 3.3, Table 5) shows students can match teacher performance in-domain and significantly surpass it OOD, establishing a foundation for annotation-free learning.
- **Effective and Comprehensive Methodological Innovations:** The proposed enhancements—ensembling teachers, incorporating synthetic distortions via ranking, and iterative difficulty-aware training—are well-motivated and demonstrably effective. The ablation study (Table 1) shows consistent, cumulative gains from each component, leading to SOTA results.
- **Strong Practical Impact:** The method achieves superior OOD generalization compared to recent supervised models (VQA², VQAThinker) without using human labels, directly addressing a major bottleneck (annotation cost) and poor generalization in the field. The work includes detailed dataset construction, training protocols, and a public code release, supporting reproducibility.

## Weaknesses
- **Limited Analysis of the W2S Mechanism and Failure Modes:** While the empirical results are strong, the paper offers limited insight into *why* the W2S effect is so pronounced for this subjective task. A deeper analysis—e.g., examining the types of samples where the student diverges from or corrects the teacher, or characterizing the failure cases where iterative training plateaus—would strengthen the conceptual contribution and provide a clearer understanding of the method's limits.
- **Incomplete Isolation from Model Capacity Effects:** The student is a 7B-parameter LMM, while teachers range from ~30M to 8B parameters. Although a supervised baseline with the same architecture is provided, an ablation using a student model with capacity closer to the teachers (e.g., a Swin Transformer) would more cleanly isolate the knowledge transfer effect from simply scaling up model size.
- **High Computational Complexity without Full Cost-Benefit Discussion:** The full pipeline involves running multiple teacher models on 200k+ videos, multi-stage iterative training of a large LMM, and a complex ranking-based inference. While the performance gains are substantial, the paper lacks a direct discussion of the computational trade-offs (e.g., GPU hours for pseudo-labeling and training vs. supervised baselines), which is important for assessing practical adoption.
- **Synthetic Distortions Are Simplified:** The synthetic teachers (spatial/temporal/compression distortions) are relatively simple and may not fully capture the complex, non-linear degradations in real-world videos. The benefit of this component is shown empirically, but its ceiling and the potential for more advanced simulators remain unexplored.

## Nice-to-Haves
- A brief analysis of statistical significance for the reported correlation improvements, though single-run evaluation is common in the field.
- A qualitative visualization of high-disagreement video pairs or difficult samples selected by the gMAD strategy to illustrate the "weak-to-strong" effect concretely.
- A more direct comparison against recent self-supervised or unsupervised VQA methods to better situate the gains of the W2S paradigm within the annotation-free landscape.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Strengths:** "The paper is well-written" and "The topic is important." (Generic)
- **Weaknesses:** "Missing related work on iterative self-training." (Scope creep; the related work covers W2S generalization adequately.)
- **Weaknesses:** "Dataset construction reproducibility is unclear." (The appendix details the mixed-integer programming approach and low-level metrics; a code link is provided.)
- **Weaknesses:** "Fairness of SOTA comparison is unverified." (Standard evaluation protocols are assumed unless stated otherwise.)
- **Weaknesses:** "Anchor selection for inference is unspecified and may affect results." (The inference method is described in Appendix C.3; anchor selection is a standard detail for ranking-based VQA.)
- **Weaknesses:** "Lack of theoretical justification." (This is an empirical systems paper; demanding theoretical proofs is not standard.)
- **Weaknesses:** "Requires a human evaluation study." (This is a methodological paper; human studies are not required for algorithmic contributions in this domain.)
- **Weaknesses:** "Needs a cross-dataset generalization test where teachers are trained on a different dataset." (The OOD benchmarks already test generalization to datasets with different content and distortions; the proposed test is an interesting extension but not a core flaw.)

## Novel Insights
The paper's primary novel insight is the empirical demonstration and systematic exploitation of the weak-to-strong generalization effect for the subjective task of video quality assessment. This shows that a strong model can learn effectively from the imperfect, aggregated judgments of multiple weaker models and synthetic simulators, not merely matching but surpassing their performance—especially on challenging, out-of-distribution data. This finding shifts the paradigm for improving VQA generalization from collecting more human labels to strategically leveraging and refining existing automated assessors. The iterative, difficulty-aware training strategy further reveals that the student can progressively learn to resolve ambiguities and errors present in the initial teacher ensemble, enabling cumulative knowledge transfer beyond any single supervisor.

## Suggestions
- Conduct and report a controlled capacity ablation: train a student model with an architecture (e.g., Swin-B) similar to a teacher using the same W2S protocol to better disentangle the effect of knowledge transfer from pure model scale.
- Add a dedicated "Limitations" subsection discussing: (1) the computational cost of the full pipeline relative to supervised training, (2) the potential for error propagation in iterative training and how the confidence loss mitigates it, and (3) the simplified nature of the synthetic distortions compared to real-world degradations.
- Include a brief qualitative analysis or case study showing video examples where the final student model's ranking contradicts the initial teacher ensemble, illustrating the type of "challenging cases" the method learns to handle.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
