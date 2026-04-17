Now I have sufficient calibration information. Let me consolidate the review.

## Summary

VIBEFACE introduces a novel facial biometric dataset (50 subjects, 2,250 images, 1,550 videos) designed specifically for eKYC (electronic Know Your Client) verification scenarios. The dataset features demographically balanced subjects across gender, race, and age, captured across five sessions varying lighting and occlusion conditions using three consumer smartphones, with 18 scenarios ranging from standardized photos to eKYC-style verification videos. The paper demonstrates dataset utility through face detection (MTCNN, RetinaFace, MediaPipe) and face verification (ArcFace, MagFace) benchmark tasks.

## Strengths

- **Fills a genuine gap for eKYC-style video data.** The paper convincingly argues that no publicly available dataset combines eKYC-style verification video sequences with demographic balance and ethical sourcing. Table 1 clearly shows VIBEFACE is the only dataset satisfying all of: eKYC videos, demographic data, gender/race/age balance, and ethical collection. This is a real contribution to the community.

- **Thoughtful acquisition protocol with multiple real-world variability dimensions.** The 18-scenario design systematically covers standardized photos, selfies, selfie videos, and seven distinct eKYC verification actions (head rotation, blinking, expression change, etc.). Five sessions manipulate lighting and glasses conditions. Three different smartphones introduce device heterogeneity. This controlled-variability design is well-suited for analyzing how individual factors affect verification performance.

- **Strong ethical and legal compliance.** Full informed consent, GDPR and EU AI Act compliance, controlled-access licensing for non-commercial research only, anonymized identifiers, and explicit withdrawal rights. This is a model for responsible biometric dataset creation and addresses a genuine community concern about ethically sourced data.

- **Demographic balance within its scale.** With 50 subjects split 50:50 by gender, roughly equal across four racial categories (13/13/12/12), and following ISO age distribution standards, the paper achieves as much balance as possible at this scale. Explicit metadata on facial hair, piercings, and skin tone (Fitzpatrick scale) is provided.

## Weaknesses

### Major:

- **Verification benchmark lacks impostor comparisons — a fundamental deficiency.** Section 4.2 evaluates face verification using only genuine (mated) comparisons: a single flash reference image compared against other images/videos of the same subject. No non-mated (impostor) pairs — comparing different subjects — are evaluated. Without this, there are no false accept rates, no EER, no ROC/DET curves, and no threshold-invariant metrics. The paper reports "percentage of frames in which the face was correctly authenticated" at a fixed threshold of 0.5, which is not a standard biometric verification metric. A system that accepts everyone would score 100% on this metric. This is a critical omission for a paper claiming to "establish a benchmark" for face verification, as the results in Table 4 are uninterpretable as verification results without impostor rates. This fundamentally undermines the verification evaluation and, by extension, the demographic fairness claims derived from it.

- **Dataset scale (50 identities) is insufficient to support fairness and generalization claims.** The paper repeatedly positions VIBEFACE as enabling "robust and fair" evaluation and as a "benchmark" for studying demographic bias (Abstract, Introduction, Conclusions). With only 50 subjects — ~12–13 per racial group, further subdivided by age and gender — any subgroup-level performance differences (e.g., ArcFace OAV by race: 0.460–0.519) are based on extremely few identities and are indistinguishable from noise. No confidence intervals, standard errors, or statistical tests are reported anywhere. The dataset can support qualitative or controlled-condition studies, but asserting fairness conclusions from N≈12 per group is not defensible.

- **Most distinctive eKYC scenarios (17–18) are excluded from evaluation.** Scenarios 17 (partial face occlusion by hand) and 18 (sequential face touching) are precisely the eKYC-specific content that differentiates VIBEFACE from generic face datasets. They are excluded from all experiments "because they involve occlusions that significantly reduce facial visibility" — yet understanding failure under such conditions is the very purpose of having eKYC scenarios. Excluding the dataset's most unique content from evaluation creates a disconnect between the stated motivation (realistic eKYC robustness) and demonstrated utility.

- **Benchmark findings are largely unsurprising.** The core results — that glasses and weak lighting degrade performance, off-angle poses are harder, and MTCNN is weaker than modern detectors — are well-established in the literature. The paper does not leverage VIBEFACE's unique properties (multi-session cross-device data, temporal video structure, eKYC actions) to produce novel insights. No cross-device verification, no temporal score aggregation, no cross-session analysis, and no presentation attack detection experiments are conducted.

### Minor:

- **Fixed threshold of 0.5 without justification.** The choice of similarity threshold is not calibrated, and different thresholds could produce very different relative patterns across demographic groups. Since ArcFace and MagFace have different score distributions, the single threshold may favor one over the other. No sensitivity analysis or threshold-invariant metric accompanies this choice.

- **Near-ceiling detection performance limits benchmark utility.** RetinaFace achieves 1.000 detection rate across nearly all conditions; MediaPipe is close behind. This suggests the detection benchmark has limited discriminative value for modern models and offers little diagnostic utility beyond showing that MTCNN is outdated.

- **Controlled studio environment vs. real-world claims.** The paper emphasizes "realistic operational settings" and "unconstrained conditions," but data was collected in a "controlled studio environment" with "standardized instructions" and "continuously supervised by trained operators." Real eKYC sessions occur in uncontrolled home environments with variable backgrounds and no supervision. This gap should be acknowledged as a limitation.

- **No temporal separation between sessions.** All five sessions for a given subject appear to be recorded in close temporal proximity. Real eKYC systems must handle appearance changes over time (haircuts, aging), which this dataset cannot assess. The paper does not discuss this limitation.

- **Missing basic dataset characterization.** Video duration per scenario, native capture frame rates, and total storage size are not reported. These are essential for other researchers to assess suitability and reproducibility.

### Trivial:

- No train/test/enrollment split protocol is defined, making the dataset a data collection rather than a standardized benchmark for reproducible comparison.

## Nice-to-Haves

- Compute proper biometric verification metrics (EER, ROC curves, TAR at fixed FAR) with genuine and impostor comparisons — this would transform the evaluation from marginally informative to genuinely useful.
- Conduct at least one experiment leveraging the dataset's unique cross-session/cross-device structure (e.g., enroll on one phone, verify on another).
- Test temporal score aggregation for eKYC video scenarios (score fusion across action frames vs. single frame).
- Include baseline PAD experiments using scenarios 14–18 as bona fide samples, as the paper itself suggests this direction.
- Report video durations, native frame rates, and per-scenario frame counts in the dataset documentation.

## Removed Points

- **Demand for more verification models beyond ArcFace and MagFace.** Reviewers suggested testing more recent models. However, two SOTA models is reasonable for a dataset paper. The weakness is the evaluation protocol itself, not the number of models.
- **Demand for presentation attack detection experiments as a required contribution.** The paper explicitly scopes its benchmark as face detection + verification. While PAD is a natural extension, requiring it goes beyond reasonable scope. The paper already suggests it as future work.
- **Criticism that zero-prescription glasses for non-glasses wearers are "unrealistic."** This is a deliberate experimental control to isolate the occlusion effect of glasses, which is a methodological strength, not a weakness. The paper explains this choice clearly.
- **Complaint that the dataset is "too small to train deep models."** This misunderstands the purpose. Evaluation datasets do not need to support training; VIBEFACE is explicitly positioned as an evaluation benchmark, and existing face verification models are used off-the-shelf (pretrained on larger datasets).

## Novel Insights

The observation that ArcFace verification rates for the 18–30 age group are consistently the *lowest* across both models and multiple scenarios (e.g., ArcFace OAV: 0.466 for 18–30 vs. 0.469 for 51–70) is counterintuitive, since younger faces typically produce higher recognition rates in the literature. This could reflect the specific demographic composition of the small sample, or could be a genuine interaction with the dataset's acquisition conditions. However, with only ~17 subjects in the 18–30 group, this finding cannot be generalized — underscoring why the paper's fairness claims are premature.

## Suggestions

- **Add impostor comparisons and standard biometric metrics.** This is the single most impactful change: generate all cross-subject pairs, compute similarity scores, and report EER, ROC curves, and TAR@FAR=0.1%. Without this, calling the paper a "verification benchmark" is inaccurate.
- **Include scenarios 17–18 in evaluation.** Even if detection/verification rates drop significantly, reporting failure modes under occluded eKYC conditions is the most informative content the dataset can provide.
- **Add confidence intervals or permutation tests for demographic comparisons.** Even with N=50, bootstrap CIs or non-parametric tests would demonstrate whether observed group differences are distinguishable from chance.
- **Clearly scope the contribution**: present VIBEFACE as a well-designed, ethically collected, controlled-condition dataset for preliminary eKYC-style evaluation, not as a fairness benchmark. Reduce claims about "generalization" and "demographic bias analysis" to the level the data supports.
- **Define a standardized evaluation protocol** (enrollment/reference images, query sets, cross-validation folds) so other researchers can reproduce and compare results.

## Evaluation

**Originality:** Moderate. The combination of eKYC-style videos, demographic balance, and ethical compliance in a single dataset is novel and fills a real gap. Individual components (mobile capture, video, demographic metadata) exist in other datasets but not in this specific configuration. However, the benchmark experiments are standard and do not push methodological boundaries.

**Importance of research question:** Moderate-to-high. eKYC verification is a significant real-world application, and the ethical positioning is timely and important.

**Claims support:** Weak. The verification evaluation is fundamentally incomplete (no impostor trials), and the demographic fairness claims are statistically unsupported at N=50. The detection results show near-ceiling performance for modern models, providing limited diagnostic value.

**Experimental soundness:** Weak. The verification protocol is not a proper biometric verification benchmark. The exclusion of the most distinctive scenarios from evaluation further weakens the experimental contribution.

**Clarity:** Good. The dataset description is thorough and well-structured. The scenario and session design is clearly documented.

**Value to community:** Moderate. The dataset itself — even at 50 subjects — is a useful controlled-condition resource for researchers studying eKYC scenarios, especially given its ethical compliance. But the paper oversells what the data can demonstrably support.

## Score and Decision

Calibration references:
- **DeepFaceGen** (770K+ samples, large scale but overclaimed findings): scores 5,6,6,5 → avg 5.5, rejected
- **HiDF** (30K images, limited evaluation): scores 5,3,3,6 → avg 4.25, rejected
- **VIBeID** (100 subjects, novel modality): scores 5,6,6,6 → avg 5.75, rejected
- **Mineral Fertilizer Dataset** (1,608 images, weak evaluation): scores 3,3,6,3 → avg 3.75, rejected
- **ADOPD** (large-scale, comprehensive evaluation): scores 8,6,6,6 → avg 6.5, accepted

VIBEFACE has a stronger ethical contribution and more thoughtful design than HiDF or Mineral Fertilizer, but its verification evaluation is fundamentally incomplete (no impostor analysis), which is a more severe flaw than what HiDF or VIBeID suffered from. VIBeID had evaluation issues but at least had proper experimental protocols. The paper also overclaims fairness/generalization benchmarking capability with 50 subjects. The dataset itself is a reasonable resource, but the paper's claims significantly exceed what the experiments and data can support.

This places VIBEFACE below VIBeID (which at least had proper benchmark protocols despite small N) and at a similar or slightly higher level than the Mineral Fertilizer dataset (which also had a useful but limited dataset with weak evaluation). The ethical rigor and eKYC novelty add value, but the core experimental contribution is too weak for the claims made.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>