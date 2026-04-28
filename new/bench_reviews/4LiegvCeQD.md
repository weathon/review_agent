## Summary
This paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method where ensemble members dynamically teach each other via a majority-vote-based soft target. The method updates all model parameters on single samples without labels and is evaluated on CIFAR-10/100-C and ImageNet-C corruption benchmarks.

## Strengths
- **Clear mathematical formulation**: Equation (1) precisely defines the cross-entropy loss between all ensemble members and the dynamically-selected soft target, making the method straightforward to implement.
- **Comprehensive corruption evaluation**: Tables 1-3 report results across all 15 corruption types on three datasets (CIFAR-10-C, CIFAR-100-C, ImageNet-C), providing detailed per-model and ensemble-level accuracy changes.
- **Individual model improvements demonstrated**: The results show that constituent classifiers improve individually (e.g., Table 1 shows ResNet20 improving by +20.36% on Zoom Blur), validating the claim that IEL differs from conventional static ensembles where individual models remain unchanged.

## Weaknesses

### Fatal
None

### Major
- **Missing comparison to standard TTA baselines**: Section 2.2 discusses established TTA methods (TENT, EATA, COTTA), yet Section 4 compares IEL only against static pre-trained models. Without quantitative comparison to adaptive baselines, it is impossible to determine whether IEL offers advantages over standard entropy minimization or ensemble-based TTA techniques. This gap prevents assessment of whether the contribution is novel or incremental. Calibration papers with similar missing baseline issues (ev9OcnAHOI.md, bEaUEFTT3N.md) typically score 3-4 and are rejected.

- **Evaluation metric masks instability**: Tables 1-3 report "Highest accuracy improvements (%) over all epochs," but Figure 3 and Section 3.1 explicitly show accuracy peaks around Epoch 4-5 and degrades thereafter, sometimes falling below the static baseline ("In some experiments we found that the average model accuracy of the IEL ensemble reduced below the static model accuracy by the final epoch"). In online TTA, cumulative or final performance is the relevant metric, not peak performance. Reporting peak accuracy obscures the catastrophic forgetting that the paper itself acknowledges.

### Minor
- **No reliability filtering on pseudo-labels**: Algorithm 1 updates all models on every incoming sample without confidence thresholds or consistency checks. Standard TTA methods (EATA, COTTA) mitigate pseudo-label noise through filtering or regularization. The lack of filtering leads to negative accuracy improvements on Noise corruptions (Gaussian, Shot, Impulse) in Tables 1-2, causing the method to fail precisely in high-shift regimes where TTA is most needed.

- **Tension between method design and ensemble theory**: Section 3 states "by minimizing the cross-entropy distances between a majority voted model output and all others... we minimize the diversity of the ensemble." Ensemble theory suggests diversity is crucial for robustness. Forcing agreement on potentially incorrect pseudo-labels (when the majority is wrong) is theoretically unsound for robustness under shift, yet the paper offers no mechanism to preserve diversity while adapting.

- **Single-sample claim contradicts multi-epoch protocol**: The Abstract and Introduction frame IEL as a solution for "single sample... inference," implying strict online adaptation. However, Section 4 states "We apply IEL for several epochs on the corruption types... 90% split of tuning set samples used for IEL." The performance gains rely on multiple passes over target data, undermining the practical motivation for scenarios where batch statistics are hard to acquire.

### Trivial
- **Conclusion overclaims consistency**: Section 5 claims "significant and consistent improvement," but Table 2 shows performance degradation on 4 of 15 CIFAR-100C corruption types (Gaussian Noise, Shot Noise, Glass Blur, Impulse Noise) where average accuracy decreases.

## Nice-to-Haves
- Report mean/online accuracy over the data stream in addition to peak accuracy to honestly evaluate stability.
- Add ablation on ensemble size to show whether the majority vote mechanism scales reliably.
- Include calibration metrics (ECE) to verify whether models become overconfident on wrong predictions as entropy decreases.
- Provide confusion matrices for failure cases (e.g., Shot Noise) vs. success cases (e.g., Zoom Blur) to reveal bias patterns.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 1 (Evaluation Metric)**: KEPT as Major weakness - verified against Figure 3 and Section 3.1 which explicitly acknowledge degradation.
- **Harsh Critic Point 2 (Unfiltered Updates)**: KEPT as Minor weakness - Tables 1-2 confirm negative improvements on noise corruptions.
- **Harsh Critic Point 3 (Missing TTA Baselines)**: KEPT as Major weakness - Section 4 confirms only static baseline comparison.
- **Harsh Critic Point 4 (Single Sample Contradiction)**: KEPT as Minor weakness - Section 4 confirms multi-epoch protocol.
- **Strength Finder "Effective Single-Sample Adaptation"**: REMOVED - conflicts with verified weakness about multi-epoch protocol; the paper uses multiple epochs over a tuning set, not strict single-sample online adaptation.
- **Strength Finder "Implicit Entropy Reduction"**: KEPT as it aligns with Figure 1 showing correlation between IEL loss and Shannon entropy.
- **Any criticism about unreleased models/datasets**: None found - all cited datasets (CIFAR-10-C, ImageNet-C) are standard benchmarks.
- **Formatting/typo criticisms**: REMOVED per hard rules - parser artifacts, not author errors.

## Novel Insights
The paper's core idea of dynamic mutual learning among ensemble members during inference is conceptually interesting and differs from static ensemble usage or one-way knowledge distillation. However, this exact approach (ensemble soft pseudo-labels with majority-vote teacher selection) was previously explored in CLESP (99K0EoKrCu.md), which scored 2.50 and was withdrawn due to nearly identical weaknesses: missing TTA baselines, unaddressed catastrophic forgetting, and peak-accuracy reporting that masked degradation. The IEL paper does not cite or differentiate from CLESP, raising novelty concerns. The explicit acknowledgment of forgetting without proposing mitigation distinguishes it slightly from CLESP, but the fundamental methodological gaps remain the same.

## Suggestions
1. **Add TTA baseline comparisons**: Include quantitative results against TENT, EATA, COTTA, or SAR under the same single-sample protocol to establish whether IEL offers advantages over existing adaptive methods.
2. **Report honest online metrics**: Supplement peak accuracy with mean accuracy over the data stream or final accuracy to transparently show stability and forgetting behavior.
3. **Implement reliability filtering**: Add confidence thresholds or consistency checks before updating weights to prevent confirmation bias on noise corruptions.
4. **Clarify the adaptation setting**: Either revise the "single sample" claims to reflect the multi-epoch protocol, or add a true single-pass experiment to validate strict online deployment.
5. **Analyze failure modes**: Provide deeper analysis of why the method fails on noise corruptions (e.g., is the majority vote confidently wrong?) and propose mitigations rather than deferring to future work.

## Score and Decision

**Calibration anchors consulted:**
- **99K0EoKrCu.md (CLESP, avg 2.50, Withdrawn)**: Nearly identical method (ensemble soft pseudo-labels with majority-vote teacher), rejected for missing TTA baselines, catastrophic forgetting without mitigation, and peak-accuracy reporting. IEL is slightly better presented but shares the same fundamental flaws.
- **ev9OcnAHOI.md (avg 4.00, Reject)**: Missing direct comparison with EATA and outdated baselines.
- **bEaUEFTT3N.md (avg 2.80, Reject)**: Missing TENT/EATA comparison, noted as critical weakness.
- **x6jHZYhnhL.md (ZeroSiam, avg 5.00, Accept)**: Addresses entropy minimization collapse with proper TTA baseline comparisons.
- **dTWfCLSoyl.md (In-Place TTT, avg 7.33, Oral)**: Strong TTA paper with comprehensive baselines and ablations.

**Reasoning**: IEL shares the same critical weaknesses as CLESP (2.50), which was withdrawn: missing TTA baselines, unaddressed catastrophic forgetting, and evaluation metrics that mask instability. However, IEL is somewhat better organized and explicitly acknowledges the forgetting issue (albeit without solving it). Papers missing TENT/EATA/COTTA comparisons typically score 3-4 and are rejected. IEL's comprehensive corruption evaluation and clear formulation prevent it from scoring as low as CLESP, but the major methodological gaps prevent acceptance.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>