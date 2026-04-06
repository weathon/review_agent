=== CALIBRATION EXAMPLE 17 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the core contribution: a systematic evaluation that aims to eliminate threshold bias. The abstract makes strong, specific claims (e.g., "threshold choice alone can alter rankings by over 200 percentage points," "XRAI achieves 31% improvement over LIME"). These claims are supported by the results presented later. The abstract effectively summarizes the problem, proposed solution, key findings, and implications.

### Introduction & Motivation
The introduction effectively frames the problem: the arbitrariness of threshold selection in attribution evaluation undermines reliable comparison. It connects this to broader known issues in XAI evaluation (input invariance, sanity checks). The contributions are clearly stated: a threshold-free AUC-IoU framework, systematic evaluation revealing contradictory single-threshold results, and size-stratified performance analysis. The motivation for a threshold-free metric is well-argued and grounded in the distinct response characteristics of different attribution paradigms.

### Methodology
*Overall Design*: The use of the HAM10000 dataset, a standard ResNet-18 model, and a representative set of seven attribution methods is appropriate. The creation of a dedicated 500-image evaluation subset is reasonable for statistical power.

*Threshold-Free Protocol*: The core methodological contribution—computing AUC over IoU values across 19 thresholds from 0.05 to 0.95—is clearly described. The handling of the edge case (union=0) is noted. However, **a significant conceptual issue is not addressed**: IoU is a set similarity metric that requires binarization. The paper critiques the choice of a *single* binarization threshold as arbitrary, but the proposed AUC-IoU still relies on the same binarization process at each of the 19 thresholds. While this integrates over the threshold parameter, it does not fundamentally escape the need to binarize. The paper would benefit from a theoretical justification for why integrating over thresholds is the correct way to "eliminate" the bias, rather than, for example, using a continuous metric like the Pearson correlation coefficient between the attribution map and a continuous ground truth (if one existed). The claim of being "threshold-free" is somewhat overstated; it is more accurately a "multi-threshold integration" approach.

*Statistical Analysis*: The use of Wilcoxon signed-rank tests with Holm-Bonferroni correction is appropriate for paired, non-parametric data. The description is clear.

*Missing Detail*: The methodology states that segmentation masks were "binarized for evaluation." It is crucial to know the binarization threshold used for these *ground truth* masks, as this is another potential source of bias. This information is not provided in the main text or the appendix. The appendix (B.1) mentions masks were "binarized at threshold 127," but it's unclear if this is the final ground truth used for IoU calculation. This needs to be explicit.

### Experiments & Results
*Model Performance*: The model performance (91.75% accuracy) is sufficient to make attribution analysis meaningful.

*Comprehensive Results (Table 2)*: The presentation of mean AUC-IoU, standard deviation, and confidence intervals is good. The reported performance values (e.g., XRAI mean AUC-IoU of 0.1844) are strikingly low. While the absolute value is less important than the relative comparison, a discussion on why even the best method achieves such a low alignment with segmentation masks would be valuable context. Is this due to inherent limitations of attribution methods, a mismatch between the segmentation task (lesion area) and what the model actually uses for classification, or the evaluation metric itself?

*Statistical Significance (Table 3)*: The statistical validation is rigorous. It is helpful that non-significant pairs (GradCAM vs. SmoothGrad IG, Blur IG vs. Guided IG) are identified, preventing overinterpretation of small differences.

*Size-Stratified Analysis (Table 4)*: This is a strong and insightful component of the paper. It moves beyond aggregate performance and reveals important dependencies. The dramatic variation (e.g., GradCAM's 269% improvement from small to large lesions) is a significant finding. However, the term "Improvement" in the table header is ambiguous—it appears to be the percentage increase from the small to large category. This should be labeled more clearly (e.g., "Small-to-Large % Increase").

*Threshold Bias Analysis (Table 5)*: This table effectively demonstrates the core problem. The large relative differences and performance swings (e.g., +202.7% for Vanilla IG at τ=0.7) powerfully illustrate how threshold choice can distort evaluation. The note that all differences are statistically significant after correction strongly supports the claim that single-threshold evaluation is systematically biased.

*Figures*: The figures referenced (e.g., showing confidence intervals, statistical matrix, threshold spectra) are described in the text but were not included in the provided content. Their descriptions suggest they would support the claims.

*Baseline Fairness & Missing Ablation*: The set of attribution methods is comprehensive. A critical **missing ablation** is an analysis of whether the *number* or *spacing* of thresholds (19 uniform steps) impacts the AUC-IoU ranking. A sensitivity analysis here would strengthen the methodological contribution. Additionally, while the paper argues single-threshold evaluation is flawed, it does not compare AUC-IoU against other proposed "threshold-robust" or continuous metrics from the literature (e.g., rank correlation measures). Establishing that AUC-IoU is a better solution than existing alternatives would strengthen the case for its adoption.

### Writing & Clarity
The writing is generally clear and well-structured. The flow from problem statement to method to results is logical. Some specific points:
- In Section 3.4.1, the header "THRESHOLD-FREE EVALUATION PROTOCOL" is present, but the subsequent text is under "3.5 EVALUATION METRICS." This minor structural oddity does not impede understanding.
- The use of "~~I~~" in method names (e.g., "SmoothGrad ~~I~~ G") in tables appears to be a parsing artifact from the original PDF and should be corrected to "IG".
- The description of LIME as "threshold-invariant" (Tables 5, 9 description) is clear and is a key observation.

### Limitations & Broader Impact
The limitations section is appropriate and acknowledges key points: single dataset/task, computational overhead, and that IoU may not capture all aspects of attribution quality. It also offers constructive recommendations for the community. The Ethics and Reproducibility statements are standard and adequate. A broader limitation not mentioned is the assumption that the binarized segmentation mask is the correct "ground truth" for what the model should be attributing to. In medical imaging, the clinically relevant features may not coincide perfectly with the lesion segmentation area. This is a common challenge in the field, but a brief acknowledgment would be prudent.

### Overall Assessment
The paper identifies a genuine and important problem in attribution method evaluation—threshold selection bias—and proposes a concrete, implementable solution (AUC-IoU). The experimental demonstration is thorough, using multiple methods, rigorous statistics, and insightful stratified analysis. The primary weakness is a lack of deep theoretical justification for why AUC-IoU is the right way to eliminate threshold bias, and a missed opportunity to ablate the design choices of the framework or compare it to other continuous metrics. The contribution stands as a valuable, empirically-driven methodological advance that provides clear evidence against single-threshold evaluation and offers a practical alternative. For ICLR, which values both algorithmic innovation and rigorous empirical analysis, this paper is likely above the acceptance bar, provided the authors can address the conceptual justification for their metric and the missing details about ground truth mask binarization.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies and addresses a critical methodological flaw in the evaluation of attribution methods: the arbitrary selection of a single threshold for binarizing continuous attribution maps, which can dramatically reverse performance rankings. The authors propose a threshold-free evaluation framework based on computing the Area Under the Curve (AUC) for Intersection over Union (IoU) across a spectrum of thresholds. They validate this framework on a dermatological imaging task, demonstrating that single-threshold evaluations yield contradictory results, while their AUC-IoU metric provides stable, statistically robust comparisons, revealing that XRAI consistently outperforms other methods, with performance heavily dependent on lesion size.

### Strengths
1.  **Well-Motivated Problem and Clear Contribution:** The paper compellingly identifies "threshold selection bias" as a fundamental, underexplored flaw in attribution method evaluation. The proposed AUC-IoU framework is a direct, theoretically sound, and practical solution to this problem. The evidence that threshold choice alone can alter rankings by over 200 percentage points is a powerful motivator for the community to adopt more robust evaluation practices.
2.  **Rigorous and Comprehensive Experimental Design:** The evaluation is methodologically sound. It uses a substantial dataset (HAM10000, with a 500-image test subset), a well-trained model (ResNet-18 with reported performance metrics), and a representative set of seven attribution methods across major paradigms. The inclusion of size-stratified analysis is a significant strength, revealing important performance dependencies that aggregate metrics mask.
3.  **Thorough Statistical Validation:** The paper employs appropriate non-parametric statistical tests (Wilcoxon signed-rank) with rigorous multiple comparison correction (Holm-Bonferroni). Reporting effect sizes, confidence intervals, and corrected p-values provides a high level of confidence in the claimed performance differences (e.g., XRAI's superiority) and non-differences (e.g., between GradCAM and SmoothGrad IG).
4.  **Strong Reproducibility and Clarity:** The paper includes a detailed reproducibility statement, appendix with training specifics, and clear descriptions of the evaluation protocol (19 thresholds, fixed seeds). The methodology is sufficiently detailed for replication.

### Weaknesses
1.  **Limited Exploration of the Metric's Limitations:** While the AUC-IoU framework addresses threshold bias, it inherits the fundamental limitations of IoU as an evaluation metric. The paper acknowledges this briefly but does not deeply discuss issues such as IoU's sensitivity to object size or its assumption that the ground-truth segmentation mask is the "correct" explanation—a significant caveat in XAI where models may rely on valid but non-annotated features.
2.  **Narrow Empirical Scope for a General Claim:** The entire empirical validation is conducted on a single binary classification task (melanoma vs. non-melanoma) using a single dataset (HAM10000) and a single model architecture (ResNet-18). While the results are convincing for this domain, the paper's strong claims about eliminating evaluation artifacts "in medical imaging and beyond" would be more persuasive with evidence from additional modalities (e.g., X-rays, text) or non-medical tasks.
3.  **Insufficient Comparison to Related Threshold-Free Ideas:** The related work section mentions evaluation challenges but does not thoroughly situate the AUC-IoU idea within the broader context of threshold-free or ranking-based metrics used in other fields (e.g., Average Precision in object detection, which also integrates over thresholds). A deeper discussion of this lineage would clarify the novelty of the *application* of this concept to attribution evaluation.
4.  **Missing Discussion on Computational Cost:** The framework requires computing IoU at 19 thresholds per image-method pair. For large-scale evaluation, this represents a non-trivial increase in computational cost over single-threshold evaluation. While the paper mentions "computational overhead" as a limitation, a more quantitative analysis or suggestion for efficient approximations would be helpful for practical adoption.

### Novelty & Significance
**Novelty:** The core novelty lies in the systematic identification, quantification, and proposed solution for threshold selection bias in attribution method evaluation. While the concept of AUC is well-known, its application to solve this specific problem in XAI evaluation is novel and timely. The size-stratified analysis providing insights into method-dependent performance patterns is also a valuable contribution.
**Significance:** The work is highly significant for the XAI community, particularly for applied domains like medical imaging. It provides a concrete methodological standard that can improve the reliability and comparability of future research. By exposing how prior comparisons may have been biased, it encourages a re-evaluation of existing conclusions and fosters more rigorous science in explainable AI.

### Suggestions for Improvement
1.  **Expand Empirical Validation:** To support claims of general applicability, include experiments on at least one additional dataset from a different domain (e.g., a natural image benchmark like ImageNet or a text classification task) and/or with a different model architecture (e.g., a Vision Transformer).
2.  **Deepen the Analysis of Metric Trade-offs:** Add a subsection discussing the pros and cons of AUC-IoU versus other potential threshold-free metrics (e.g., computing AUC for metrics like correlation coefficient). Acknowledge more explicitly that while it solves the threshold problem, it does not solve the "what is a good explanation?" problem.
3.  **Strengthen the Related Work Section:** Explicitly compare and contrast the proposed AUC-IoU with threshold-free metrics from other fields (e.g., AUC-ROC, Average Precision). This will better position the contribution as an adaptation of a robust statistical idea to a new problem.
4.  **Provide Practical Guidelines and Cost-Benefit Analysis:** Offer clearer recommendations for researchers: e.g., "For a quick benchmark, a single threshold at X may be sufficient, but for rigorous comparison, AUC-IoU over at least N thresholds is recommended." A brief analysis of how the number of thresholds affects result stability versus compute time would be very useful.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No validation on non-medical or multi-class datasets.** The paper's claim that threshold bias is a general problem and its solution is broadly applicable rests entirely on a single binary medical imaging task. Without experiments on natural image datasets (e.g., ImageNet) or multi-class settings, the generality of the findings is unsupported.
2. **No comparison to existing threshold-free or threshold-robust metrics.** The proposed AUC-IoU is not compared to other integrated metrics (e.g., Average Precision, AUC for saliency detection) or multi-threshold averaging. This omission leaves it unclear if the proposed metric is superior or necessary compared to simpler alternatives.
3. **Missing ablation on the number and range of thresholds.** The choice of 19 thresholds from 0.05 to 0.95 is arbitrary. A sensitivity analysis is needed to show that the method ranking is stable with respect to these hyperparameters; otherwise, the evaluation itself may be sensitive to a new arbitrary choice.
4. **No experiment on model faithfulness or causal metrics.** The paper equates "attribution quality" solely with spatial alignment to ground-truth masks. It does not test if threshold-free evaluation correlates better with faithfulness metrics (e.g., insertion/deletion) or human evaluation, which are critical for assessing explanation utility.

### Deeper Analysis Needed (top 3-5 only)
1. **Lack of analysis linking attribution value distributions to threshold sensitivity.** The paper states gradient methods have "concentrated" values but does not quantitatively characterize these distributions (e.g., kurtosis, entropy). Without this, the claimed mechanistic explanation for threshold bias is merely descriptive, not explanatory.
2. **No investigation of how model confidence/correctness affects threshold bias.** It is unknown if the observed bias and method rankings hold for correctly vs. incorrectly classified cases, or for high- vs. low-confidence predictions. This is critical for clinical trust, as explanations for errors may be most important.
3. **Insufficient statistical grounding for size-stratified claims.** While improvement factors are reported, there is no statistical test (e.g., interaction test) to confirm that performance differences between size categories are significant for each method. The claim that "method selection cannot rely on aggregate metrics" needs this formal support.

### Visualizations & Case Studies
1. **Side-by-side visual examples of attribution maps at different thresholds, with IoU curves.** The paper needs a figure showing, for a few representative images, how binarization at different thresholds dramatically changes the overlap with the ground truth for each method, directly illustrating the ranking reversal problem.
2. **Case studies of failures.** Show specific images where single-threshold evaluation gives a misleading ranking (e.g., method A beats method B at τ=0.3 but loses at τ=0.7) and how the AUC-IoU ranking aligns (or not) with a qualitative assessment of explanation plausibility.

### Obvious Next Steps
1. **Benchmark on at least one additional dataset and task** (e.g., ImageNet classification or a different medical modality) to demonstrate the framework's generality, which is a core claim.
2. **Compare AUC-IoU to a simple baseline of averaging IoU over the same set of thresholds** to isolate the benefit of area-under-curve integration versus just avoiding a single threshold.
3. **Provide a public code implementation and benchmark suite** to allow community adoption and validation. The paper's impact depends on others using the proposed framework, which requires accessible, standardized code.

# Final Consolidated Review
## Summary
This paper identifies threshold selection bias as a critical flaw in the evaluation of attribution methods, where arbitrary choice of a single binarization threshold can reverse method rankings. It proposes a threshold-free framework using Area Under the Curve for Intersection over Union (AUC-IoU) to eliminate this bias, validated on a dermatological imaging dataset with comprehensive statistical analysis and size-stratified performance evaluation.

## Strengths
- **Clear demonstration of threshold bias and a practical solution:** The paper quantitatively shows that threshold choice alone can alter method rankings by over 200 percentage points (Table 5), and introduces AUC-IoU as a straightforward protocol that integrates over the threshold spectrum, providing reliable comparisons (Tables 2 and 3).
- **Rigorous and insightful experimental analysis:** The evaluation includes seven diverse attribution methods, thorough statistical validation with Wilcoxon signed-rank tests and multiple comparison correction, and a novel size-stratified analysis revealing that performance varies dramatically with lesion scale (e.g., GradCAM shows 269% improvement from small to large lesions in Table 4).
- **Strong reproducibility and clarity:** The methodology is detailed with fixed seeds, dataset splits, and hyperparameters (Section 3, Appendix B), enabling replication, and the results are presented with confidence intervals and effect sizes to support claims.

## Weaknesses
- **Limited empirical validation for claimed generality:** The framework is evaluated only on a single binary classification task using one medical imaging dataset (HAM10000) and one model architecture (ResNet-18). While the results are convincing for this domain, the paper's assertion of applicability "in medical imaging and beyond" is not adequately supported, weakening broader implications.
- **Lack of robustness analysis for the AUC-IoU protocol:** The choice of 19 uniformly spaced thresholds from 0.05 to 0.95 is arbitrary, and no sensitivity analysis is provided to show that method rankings are stable with respect to the number or range of thresholds. This omission risks replacing one arbitrary choice (single threshold) with another (threshold set), undermining the protocol's reliability.
- **Incomplete statistical grounding for size-stratified claims:** Although performance differences across size categories are reported (Table 4), there is no formal statistical testing (e.g., interaction tests) to confirm that these variations are significant for each method. This weakens the conclusion that "method selection cannot rely on aggregate metrics," as observed improvements might be due to noise.

## Nice-to-Haves
- Comparison of AUC-IoU to other threshold-free or continuous metrics (e.g., correlation-based measures) to further justify its advantages over alternatives.
- Experiments on additional datasets or tasks (e.g., natural images or multi-class classification) to demonstrate broader applicability without requiring full validation.
- Deeper analysis linking attribution value distributions (e.g., kurtosis) to threshold sensitivity, providing a mechanistic explanation for the observed biases.

## Novel Insights
The paper's key novel insight is that threshold selection bias in attribution evaluation is not a minor issue but can fundamentally reverse method rankings, rendering many comparative studies unreliable. By integrating over thresholds, the AUC-IoU framework uncovers consistent performance patterns—such as XRAI's superiority across lesion sizes and LIME's threshold-invariant behavior—that were obscured by single-threshold metrics. Additionally, the size-stratified analysis reveals that attribution methods exhibit dramatic performance variations based on object scale, suggesting that method efficacy is context-dependent and challenging one-size-fits-all approaches in explainable AI.

## Suggestions
- Conduct a sensitivity analysis on the number and spacing of thresholds used in AUC-IoU to ensure robustness and provide guidelines for choosing these parameters in practice.
- Include statistical interaction tests to formally validate the performance differences across size categories, strengthening the size-stratified conclusions.
- Acknowledge more explicitly in the discussion that while AUC-IoU addresses threshold bias, it does not solve all evaluation challenges, such as the appropriateness of IoU or ground truth masks for assessing explanation quality.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0]
Average score: 1.0
Binary outcome: Reject
