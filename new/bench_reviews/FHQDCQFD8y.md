## Summary

This paper presents Grad-TopoCAM, a gradient-based interpretability method for EEG decoding that applies Grad-CAM formulations to deep learning models for EEG and maps channel-level attribution scores to brain topographies. The method is evaluated across 8 architectures on 4 datasets, with visualizations claimed to align with neuroscientific knowledge and a demonstration of channel selection for model efficiency.

## Strengths

- **Broad empirical validation:** Tables 1-3 demonstrate the method applied across 8 architectures (ShallowConvNet through Conformer) on 4 datasets (motor imagery, inner speech, silent reading in Chinese/English), showing awareness of methodological diversity.
- **Clear mathematical formulation:** Equations 1-3 provide an accessible framework for mapping gradient-based feature importance to EEG spatial channels, bridging DL feature maps and brain topographies.
- **Practical application scope:** Section 5.2 applies the method to channel selection, showing computational parameter reductions (e.g., Table 4 shows reductions across models), which is practically relevant for BCI deployment.

## Weaknesses

### Fatal

### Major

- **The core methodological contribution is incremental—standard Grad-CAM with trivial temporal averaging.** Equations 1 and 2 are mathematically identical to the standard Grad-CAM formulation (Selvaraju et al., 2017). The only modification is Equation 3, which averages class activation maps over the temporal dimension $T$ to produce a per-channel score. Applying gradient-based attribution to sequential data by pooling over non-spatial axes is standard practice, not an algorithmic innovation. The paper frames this as a "universal" and "innovative" method, which misrepresents the technical contribution and mischaracterizes the novelty.

- **The central interpretability claim is validated solely through qualitative post-hoc rationalization, with zero quantitative faithfulness testing.** Section 4.3 validates Grad-TopoCAM by claiming the heatmaps "align with existing research" that "identifies motor-related areas, especially C3 and Cz" for motor imagery. Without quantitative metrics (insertion/deletion curves, causal ablation, gradient randomization tests, or correlation with established neurophysiological markers), there is no empirical basis to distinguish whether highlighted regions drive model decisions or reflect spurious correlations. The claim that the method reveals "true decision-critical features" remains unverified.

- **Table 4 reports physically impossible parameter/FLOP values that undermine experimental credibility.** EEGNet is listed with 130.245M parameters but only 213.748K FLOPs; LMDA-Net with 288.759M parameters but only 8.388K FLOPs. FLOPs for a single forward pass cannot be orders of magnitude lower than parameter count—each parameter must be read at least once. Either the units are wrong, the calculation method is fundamentally flawed, or the numbers are fabricated. This calls into question the channel selection results in Table 5, since the claimed efficiency gains are based on these corrupted values.

- **Datasets III and IV contain near-chance accuracy models, making their visualizations scientifically meaningless.** The paper states these datasets contain data from "a single participant." Table 3 shows accuracies around 12-19%, which for 7-class Chinese (~14.3% chance) and 9-class English (~11.1% chance) is near-random. Generating interpretability maps from a model that hasn't learned meaningful decision boundaries is invalid. The paper claims validation across "four datasets," but 1 of 4 is a single-subject dataset where models fail to learn, undermining the comprehensiveness claim.

### Minor

- **The spatial mapping process from channel-level attribution scores to 2D topographic images is unspecified.** EEG topographies typically require spherical spline interpolation or electrode coordinate projection to render channel scores onto continuous brain imagery. Equation 3 describes temporal averaging but doesn't specify how per-channel scores are spatially interpolated. This creates a reproducibility gap for anyone attempting to implement the topographic visualization.

- **The channel selection methodology lacks essential details.** Section 5.2 doesn't specify whether models were retrained from scratch on the selected channels, how cross-validation was structured to prevent information leakage during gradient-based selection, or how the method compares to standard feature selection baselines. The accuracy fluctuations in Table 5 are therefore uninterpretable.

### Trivial

- **Table 5 uses inconsistent subject labeling** ("501, 502..." instead of the standard S01-S10 nomenclature used elsewhere in the paper), creating confusion.

## Nice-to-Haves

- Adding quantitative faithfulness metrics (e.g., AUC-ROC of region deletion curves or model confidence decay under CAM-guided masking) would strengthen the interpretability claims.
- Comparing Grad-TopoCAM against standard Grad-CAM with naive time-pooling and EEG-specific attribution methods (Integrated Gradients, LRP) would isolate whether the temporal averaging and topographic mapping provide genuine improvements.
- Providing causal intervention visualizations showing model output when attributed channels are zeroed out would demonstrate functional necessity.

## Removed Points

These points are flagged to be removed, treat them with caution:
- *The claim that "existing methods lack extensibility to other DL models" contradicts Grad-CAM's model-agnostic nature.* The paper does acknowledge Grad-CAM and related methods exist, but argues existing EEG-specific methods require 2D conversions or custom architectures. This is a scope-clarification issue, not a factual error.
- *Section 5.1's claim about "deeper layers capture more task-specific features" based solely on visual inspection.* This is a standard limitation of deep learning visualization papers and doesn't invalidate the paper.
- *Mixed bolding in Table 5 obscuring which model performs best.* This is a formatting issue, not a substantive criticism of the research.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Clearly acknowledge that Grad-TopoCAM is an application of standard Grad-CAM to EEG with temporal averaging, positioning this as a domain application rather than claiming algorithmic novelty.
- Replace or complement the qualitative "aligns with neuroscience" validation with quantitative faithfulness metrics.
- Correct the parameter/FLOP reporting in Table 4 with verified calculations.
- Remove or clearly disclaim the single-subject datasets (III/IV) where models perform at chance level, or explain why near-random model visualization is still informative.
- Specify the interpolation method for mapping channel-level scores to continuous topographic images.

## Calibration and Scoring

I compared this paper against several calibration anchors:

- **Low-score anchors (3-4):** Papers like zPxlHOLxmh (scores 3,1,1,3) and XsYJ6yvgEC (scores 3,1,6) with qualitative-only validation, near-chance performance, and incremental novelty received similar rejections. The EEG interpretability paper B5i88Tj1nk (scores 3,3,8) was also withdrawn despite solid experiments, because the incremental contribution was outweighed by methodological gaps.

- **Borderline anchors (5-6):** Papers like K1VLZ5rNuZ (scores 3,3,5,3) had novelty concerns but stronger experimental grounding. The channel selection EEG papers V5lBNcD65H (scores 3-6) and p30YulvDbj (scores 1-3) showed that EEG channel-selection papers receive low scores when novelty and experimental rigor are weak.

- **High-score anchors (7-8):** Papers like oSQiao9GqB and lvw3UgeVxS with strong empirical validation scored high due to genuine methodological contributions and rigorous quantitative evaluation—both absent here.

This paper sits below the borderline anchors. The fatally weak validation (qualitative-only, near-chance model visualizations, impossible FLOP reporting) combined with incremental novelty (standard Grad-CAM + temporal pooling) makes it weaker than papers that received scores of 5-6. It aligns more closely with the 3-scoring papers that had qualitative-only validation and incremental contributions. The broad empirical scope across 8 models is a positive, but doesn't compensate for the lack of quantitative faithfulness testing and the corrupted numerical reporting.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>