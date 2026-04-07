=== CALIBRATION EXAMPLE 19 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title accurately reflects the core contribution: a systematic evaluation that eliminates threshold bias and reveals method-dependent patterns. The abstract clearly states the problem, proposed solution (threshold-free AUC-IoU), key results (performance rankings, size-stratified analysis), and implications. Claims are specific and are supported in the paper. The abstract is well-structured for ICLR.

**Introduction & Motivation**
The introduction effectively motivates the problem of threshold selection bias in attribution evaluation, citing prior work on evaluation failures. The core issue—that threshold choice alone can reverse rankings—is well-articulated. Contributions are listed clearly. A minor point: the claim that threshold choice can alter rankings by "over 200 percentage points" is strong and is substantiated later, but the introduction could briefly explain what constitutes a "percentage point" change in this context to avoid ambiguity.

**Related Work**
The section adequately covers major attribution paradigms and known evaluation challenges (input invariance, sanity checks, metric contradictions). It appropriately cites recent benchmarking efforts and identifies the gap regarding systematic analysis of threshold bias. However, the connection between threshold bias in medical image segmentation (Müller et al.) and its specific manifestation in attribution evaluation could be more deeply developed. The related work sets the stage but does not fully engage with prior attempts (if any) to mitigate threshold sensitivity in attribution metrics.

**Methodology**
- *Dataset and Model*: Use of HAM10000 and a fine-tuned ResNet-18 is standard and well-documented. The creation of a dedicated 500-image attribution evaluation subset with balanced class representation is a strength.
- *Attribution Methods*: The selection of seven methods across paradigms is comprehensive. Implementation details are provided, though some parameters (e.g., LIME's kernel width, Ridge regression α) are given without justification. This is acceptable given the use of a standard library.
- *Evaluation Framework*: The core proposal of AUC-IoU over 19 thresholds is sensible. However, critical details are missing:
    1. **Normalization**: The paper states attribution maps are "normalized to [0,1]" but does not specify the method (e.g., min-max, absolute max, percentile). This choice significantly impacts thresholding and must be explicitly defined.
    2. **Edge Case Handling**: The statement that IoU returns 1.0 when the union is zero is problematic. If both attribution and ground truth are empty, IoU=1 is conventional. However, if only one is empty, the union is not zero, and IoU should be 0. The description needs clarification to ensure correct implementation.
- *Statistical Analysis*: The use of Wilcoxon signed-rank tests with Holm-Bonferroni correction is appropriate for paired, non-parametric data. The analysis plan is rigorous.

**Results**
- *Model Performance*: The model achieves strong performance (91.75% accuracy), providing a reliable basis for attribution analysis.
- *Comprehensive Evaluation*: Table 2 and Figure 1 clearly show performance rankings. XRAI is superior, with tight confidence intervals. The reported improvements (31% over LIME, 204% over Vanilla IG) are derived from the mean AUC-IoU values and are consistent.
- *Statistical Significance*: Table 3 provides corrected p-values and effect sizes. The findings that some method pairs (GradCAM vs. SmoothGrad IG, Blur IG vs. Guided IG) are not statistically significant are important and honestly reported.
- *Size-Stratified Analysis*: Table 4 reveals compelling performance variations across lesion sizes. The dramatic improvement for GradCAM (269%) and the near-invariance of Blur IG are interesting. This analysis adds depth.
- *Threshold Bias Analysis*: Table 5 powerfully demonstrates the bias introduced by single-threshold evaluation. The large performance swings (e.g., 235.6 percentage points for Vanilla IG) support the central claim. However, the surprising threshold-invariance of LIME (identical IoU at τ=0.3, 0.5, 0.7) requires explanation. Is this due to the discrete superpixel weights producing a step-like function? This phenomenon should be discussed, as it is central to the claim of "method-dependent patterns."

**Discussion**
The discussion effectively interprets the results, linking threshold-response patterns to theoretical mechanisms (concentrated vs. diffuse attributions). It situates the work within broader ML evaluation challenges and provides practical clinical implications. Limitations are appropriately acknowledged (single dataset, computational overhead, IoU's potential inadequacies). The recommendations for the community are concrete and reasonable.

**Reproducibility & Ethics**
The reproducibility statement is detailed, covering seeds, dataset splits, hyperparameters, and evaluation protocols. The ethics statement correctly addresses dataset usage and cautions against clinical reliance on attributions. Both are sufficient for ICLR.

**Appendix**
The appendix provides necessary additional details on preprocessing, training, and calibration. The supplementary figures (threshold spectra, size-stratified trends) are valuable and support the main claims.

### Overall Assessment
This paper makes a strong methodological contribution by rigorously identifying and addressing threshold selection bias in attribution evaluation. The proposed AUC-IoU framework is simple, intuitive, and effectively eliminates an arbitrary hyperparameter that can reverse method rankings. The experimental evaluation is extensive, employing multiple methods, rigorous statistical testing, and insightful size-stratified analysis. The core findings are well-supported by the data. The primary weaknesses are the lack of specification for attribution map normalization and insufficient explanation for LIME's threshold-invariance. Additionally, the handling of IoU edge cases needs clarification. These are important but addressable issues. The paper meets ICLR's standards for novelty, rigor, and clarity, and its focus on improving evaluation practices is highly relevant to the community. With minor revisions to clarify methodological details, this paper would be a solid contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies and addresses a critical methodological flaw in the evaluation of attribution methods: the arbitrary selection of a single threshold for binarizing continuous attribution maps, which can dramatically reverse performance rankings. The authors propose a threshold-free evaluation framework using Area Under the Curve for Intersection over Union (AUC-IoU) to assess attribution quality across the full threshold spectrum. They empirically demonstrate that single-threshold evaluations lead to contradictory results on a dermatological imaging dataset (HAM10000), while their threshold-free framework provides stable rankings, revealing XRAI as the superior method. They further conduct a size-stratified analysis showing performance varies substantially with lesion scale.

### Strengths
1.  **Clear Problem Formulation and Motivation:** The paper convincingly establishes that threshold selection is a major, under-addressed source of bias in attribution evaluation, citing evidence that rankings can swing by over 200 percentage points. This directly addresses a known pain point in the XAI community.
2.  **Well-Defined and Reproducible Methodology:** The proposed AUC-IoU metric is straightforward and effectively eliminates the arbitrary threshold choice. The experimental setup is detailed, including dataset splits, model training (ResNet-18), implementation specifics for seven attribution methods, and a comprehensive statistical analysis plan (Wilcoxon signed-rank tests with Holm-Bonferroni correction). The reproducibility statement is strong.
3.  **Rigorous and Multi-Faceted Analysis:** The evaluation goes beyond aggregate scores. The size-stratified analysis is a significant strength, revealing that method performance is not uniform and depends on lesion characteristics (e.g., GradCAM shows a 269% improvement from small to large lesions). This provides nuanced, practical insights for clinical deployment.
4.  **Compelling Empirical Evidence:** The results robustly support the core claims. The paper shows that single-threshold evaluations yield contradictory rankings, while AUC-IoU provides a consistent order. The statistical significance of the pairwise comparisons between methods is thoroughly validated.

### Weaknesses
1.  **Limited Scope of Generalization:** The empirical validation is conducted on a single dataset (HAM10000), a single task (binary melanoma classification), and a single model architecture (ResNet-18). While the methodological argument is general, the paper does not provide evidence that the observed performance hierarchy (XRAI > LIME > etc.) or the magnitude of threshold bias holds in other domains (e.g., NLP, tabular data), modalities (e.g., 3D medical images), or with different model families (e.g., Vision Transformers).
2.  **Dependence on Segmentation Ground Truth:** The evaluation relies entirely on IoU against expert segmentation masks, which assumes the ground truth mask perfectly encapsulates all "important" features. This does not evaluate other critical aspects of attribution quality like *faithfulness* (does the attribution reflect the model's actual reasoning?) or *human usability*. The discussion mentions this but does not explore it.
3.  **Incomplete Discussion of Metric Trade-offs:** While AUC-IoU solves the threshold bias problem, it may introduce others. For instance, it equally weights all thresholds, but thresholds at the extremes (near 0.05 or 0.95) may have less practical relevance. The paper would be strengthened by a brief discussion of the potential limitations or implicit assumptions of the AUC-IoU metric itself.
4.  **Presentation of Baseline Model Performance:** The clinical imbalance (melanoma recall of 0.60) is noted, but the implications for attribution evaluation are not deeply discussed. For example, how might attribution quality differ for correctly vs. incorrectly classified cases? A brief analysis here could add depth.

### Novelty & Significance
**Novelty:** The core novelty lies in the systematic formalization and quantification of *threshold selection bias* for attribution map evaluation and the promotion of a threshold-free AUC-based protocol as a standard. While the use of AUC to aggregate over thresholds is a known technique in other fields (e.g., ROC analysis), its application to solve this specific problem in XAI evaluation is novel and timely. The size-stratified analysis is also a fresh and valuable perspective.
**Significance:** The work is highly significant for the XAI community, particularly for applied fields like medical imaging. It provides a simple, implementable fix for a flaw that likely undermines many comparative studies. By establishing a more rigorous evaluation standard, it can improve the reliability of future research and help practitioners make evidence-based choices of explanation methods. The findings also caution against one-size-fits-all method selection, advocating for context-aware (e.g., size-aware) deployment.

### Suggestions for Improvement
1.  **Expand the Validation Scope:** To bolster claims of generality, include a small experiment on a second dataset from a different domain (e.g., a natural image dataset like ImageNet with bounding boxes, or a different medical imaging modality). This would demonstrate that the threshold bias problem and the utility of the solution are not dataset-specific.
2.  **Complement IoU with a Faithfulness Metric:** To address the reliance on ground truth masks, incorporate one threshold-free faithfulness evaluation (e.g., the area under the deletion curve or a correlation-based measure like Spearman's ρ between attribution magnitudes and prediction drop upon perturbation). This would show the framework's applicability to other important attribution qualities.
3.  **Analyze Attributions for Misclassified Cases:** Include a preliminary analysis of whether the attribution quality trends (and method rankings) hold for cases the model gets wrong. This could reveal if explanations for failures differ and is relevant for real-world debugging.
4.  **Refine the Discussion on Metric Choice:** Briefly discuss why uniformly spaced thresholds and the trapezoidal rule for AUC were chosen, and acknowledge if alternative weighting schemes (e.g., focusing on a clinically relevant threshold band) might be useful in future domain-specific adaptations.
5.  **Clarify Computational Cost:** The paper mentions computational overhead but does not quantify it. Providing the approximate runtime for the AUC-IoU evaluation (19 thresholds) versus a single-threshold evaluation on the 500-image set would be helpful for practitioners assessing feasibility.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Lack of multi-dataset validation:** The paper evaluates only on HAM10000 (dermatology). Without validation on diverse datasets (e.g., natural images from ImageNet, other medical modalities like chest X-rays), the claim that the threshold-free framework provides "practical guidance for robust comparison... beyond" is unsupported and undermines generalizability for ICLR.
2. **Missing comparison to other threshold-free or robust metrics:** The paper claims AUC-IoU solves the bias problem but does not compare it to other potential threshold-robust metrics (e.g., ranking-based metrics, Optimal IoU, or metrics from segmentation like Average Precision). This gap leaves open whether AUC-IoU is the best solution or if the observed effects are metric-specific.
3. **No ablation on the framework's components:** The choice of threshold range [0.05, 0.95] and the number of thresholds (19) is arbitrary. An ablation study is needed to show performance is stable w.r.t. these choices; otherwise, the framework itself may introduce a new form of range-selection bias.
4. **Absence of a user study or correlation with downstream task performance:** The ultimate test of an attribution metric is its correlation with human utility or model debugging efficacy. The paper assumes higher AUC-IoU implies a "better" explanation for clinical use, but provides no human evaluation or faithfulness test (e.g., insertion/deletion) to validate this claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Causal analysis of why XRAI performs best:** The paper reports XRAI's superiority but lacks a mechanistic analysis. Is it due to the region-based approach, or is it an artifact of the evaluation (IoU favors contiguous regions)? An analysis correlating method properties (e.g., attribution sparsity, smoothness) with AUC-IoU is needed to move from observation to insight.
2. **Interpretation of low absolute AUC-IoU scores:** The best method (XRAI) only achieves a mean AUC-IoU of 0.1844. This is very low. The paper does not discuss whether this indicates a fundamental limitation of attribution methods versus the ground truth, or if the metric is extremely stringent. Without this context, the practical significance of the reported "204% improvement" is unclear.
3. **Analysis of threshold-bias interaction with model confidence/uncertainty:** The bias might be more pronounced for correct vs. incorrect predictions or high vs. low confidence cases. Analyzing performance stratified by model confidence would reveal if threshold-free evaluation is most critical for uncertain predictions, which are clinically important.

### Visualizations & Case Studies
1. **Visualization of failure cases for high AUC-IoU methods:** Show examples where a method achieves high AUC-IoU but the saliency map is clearly wrong or uninterpretable to a human. This would test the core assumption that AUC-IoU faithfully captures explanation quality.
2. **Case studies illustrating ranking reversals:** The paper claims single-threshold evaluation can reverse rankings. It should visually demonstrate a specific image where, at threshold τ=0.3, Method A beats Method B, but at τ=0.7, the ranking flips, alongside their AUC-IoU values. This would make the central problem concretely evident.

### Obvious Next Steps
1. **Validate on a synthetic dataset with known ground-truth attributions:** To conclusively prove the framework correctly identifies better methods, it must be tested in a controlled setting where the "true" explanation is known (e.g., a simple model with clear important pixels). This is a standard practice for evaluating evaluation metrics.
2. **Benchmark against recent evaluation frameworks:** The paper cites EvalAttAI and SaliencyBench but does not quantitatively compare its proposed AUC-IoU framework against these comprehensive benchmarks. A direct comparison is necessary to position its contribution.
3. **Provide clear guidelines on when to use threshold-free vs. single-threshold evaluation:** The paper strongly advocates for always using threshold-free evaluation but does not acknowledge potential trade-offs (e.g., compute cost, simplicity). A discussion or analysis of when the bias is negligible would make the recommendation more practical and nuanced.

# Final Consolidated Review
## Summary
This paper identifies and addresses a fundamental methodological flaw in the evaluation of attribution methods: the arbitrary selection of a single threshold for binarizing continuous saliency maps, which can dramatically reverse performance rankings. The authors propose a threshold-free evaluation framework using Area Under the Curve for Intersection over Union (AUC-IoU) and demonstrate on a dermatological imaging dataset that single-threshold evaluation yields contradictory results, while their framework provides stable rankings and reveals that method performance varies substantially with lesion size.

## Strengths
- **Clear identification and solution of a critical evaluation bias:** The paper convincingly demonstrates that threshold selection alone can alter method rankings by over 200 percentage points, formalizing a known but under-addressed problem. The proposed AUC-IoU framework is a simple, direct, and effective solution that eliminates this arbitrary hyperparameter.
- **Rigorous and multi-faceted experimental analysis:** The evaluation is comprehensive, testing seven attribution methods across paradigms. The analysis includes rigorous statistical validation (Wilcoxon tests with multiple comparison correction) and a particularly insightful size-stratified analysis, which reveals that method performance is not uniform and depends critically on lesion scale (e.g., GradCAM shows a 269% improvement from small to large lesions).
- **Strong empirical support for core claims:** The results robustly show that single-threshold evaluations lead to ranking instability and contradictory conclusions, while the threshold-free framework provides a consistent performance order, with XRAI emerging as the superior method. The documentation is detailed, supporting reproducibility.

## Weaknesses
- **Unclear specification of critical pre-processing steps:** The paper states attribution maps are "normalized to [0,1]" but does not specify the normalization method (e.g., min-max, absolute max). Furthermore, the handling of IoU edge cases ("returning a score of 1.0" when the union is zero) is ambiguously described and could lead to incorrect implementation if the case where only the attribution or ground truth is empty is not handled properly. These omissions undermine the reproducibility and precise interpretation of the metric.
- **Limited empirical validation of generality:** The core methodological argument is general, but the empirical validation is confined to a single dataset (HAM10000), a single task (binary melanoma classification), and a single model architecture (ResNet-18). While not invalidating the proposed framework, this limits the confidence that the observed performance hierarchy (XRAI > LIME > ...) or the exact magnitude of threshold bias translates to other domains, modalities, or model families.
- **Evaluation is limited to localization (IoU) against segmentation masks:** The assessment relies entirely on spatial alignment with expert-drawn segmentations, which does not evaluate other critical attributes of a good explanation, such as *faithfulness* to the model's reasoning. The paper acknowledges this but does not explore how the threshold-bias problem manifests or could be addressed for faithfulness metrics.
- **Insufficient discussion of the practical meaning of low absolute scores:** The best method achieves a mean AUC-IoU of only 0.1844. The paper does not contextualize whether this indicates a fundamental limitation of current attribution methods, an extremely stringent metric, or a mismatch between the metric and practical utility. This makes the practical significance of the reported large percentage improvements (e.g., 204%) difficult to interpret.

## Nice-to-Haves
- A brief analysis of whether attribution quality trends differ for correctly versus misclassified cases could provide additional insight into the framework's utility for model debugging.
- Quantifying the computational overhead of the AUC-IoU evaluation (19 thresholds) compared to a single-threshold evaluation would be helpful for practitioners.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness (Harsh Critic):** "The introduction could briefly explain what constitutes a 'percentage point' change." *Removed as a minor stylistic nitpick that does not affect understanding.*
- **Weakness (Spark Finder):** "Lack of multi-dataset validation... undermines generalizability." *Weakened and moved to main weaknesses as a scope limitation, but removed the strong claim that it "undermines" the core contribution, which is the framework itself.*
- **Weakness (Spark Finder):** "No ablation on the framework's components (threshold range, number of thresholds)." *Removed as demanding non-standard methodological practice; the chosen range [0.05, 0.95] with uniform sampling is a reasonable and standard design choice for an AUC metric.*
- **Weakness (Spark Finder):** "Absence of a user study or correlation with downstream task performance." *Removed as imposing an arbitrary rigor requirement; user studies are not the standard for purely algorithmic/methodological contributions in this area.*
- **Weakness (Spark Finder):** "Validate on a synthetic dataset with known ground-truth attributions." *Removed as demanding validation outside the paper's scope and empirical focus.*
- **Weakness (Spark Finder):** "Benchmark against recent evaluation frameworks (EvalAttAI, SaliencyBench)." *Weakened; while a comparison could be interesting, the paper's primary contribution is exposing a bias within a standard metric (IoU), not benchmarking against every comprehensive evaluation suite.*
- **Suggestion (Spark Finder):** "Causal analysis of why XRAI performs best." *Weakened to a nice-to-have; the paper provides a theoretical discussion linking patterns to method mechanisms, and a full causal analysis is beyond its stated scope.*

## Novel Insights
The paper provides novel insights into attribution method behavior through its threshold-free lens. It systematically reveals that gradient-based methods (e.g., Integrated Gradients variants) exhibit concentrated attribution patterns that are penalized by high thresholds, while perturbation-based methods (LIME) show threshold-invariant performance due to their discrete, superpixel-based approach. Furthermore, the size-stratified analysis uncovers a previously unquantified dependency: method performance is not uniform and can vary dramatically (0–269%) with the scale of the feature being explained, challenging the assumption that a single "best" method can be selected from aggregate scores.

## Suggestions
- **Clarify the methodology:** Explicitly define the normalization technique used on attribution maps (e.g., "normalized by dividing by the maximum absolute value"). Precisely describe the IoU edge-case logic in an equation or pseudocode to avoid ambiguity.
- **Contextualize the low absolute scores:** In the discussion, add a brief analysis or commentary on why the AUC-IoU values are so low (e.g., is it due to inherent noise in attributions, stringent ground truth, or metric design?) to help readers interpret the practical meaning of the reported improvements.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0]
Average score: 1.0
Binary outcome: Reject
