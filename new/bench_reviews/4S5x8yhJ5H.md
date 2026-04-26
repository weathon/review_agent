## Summary

VIBEFACE introduces a novel multimodal facial biometric dataset distinguished by its inclusion of eKYC-style verification videos (head rotation, blinking, expression changes, partial occlusion) alongside still images and selfie recordings. Collected from 50 participants across multiple consumer smartphones and under varied lighting and eyeglass-occlusion conditions, the dataset is ethically sourced with informed consent and GDPR compliance, targeting a genuine gap in publicly available eKYC facial video data.

## Strengths

- **Novel eKYC-style scenario taxonomy.** Scenarios 12–18 define a concrete, application-relevant sequence of compliance-driven actions (circular head rotation, tilting, blinking, facial touches) that is not present in prior public benchmarks and directly supports liveness-detection and video-verification research (Section 3.2, Table 2, Figure 3).
- **Strong ethical and legal documentation.** The paper provides thorough informed-consent procedures, anonymization, GDPR/EU AI Act compliance, and a controlled-access research license—addressing a real and timely weakness in the field given the withdrawal of web-crawled datasets (Sections 2, 3.4, 3.5).
- **Systematic factorial design.** The five-session acquisition protocol independently varies lighting (artificial, flash, natural, weak natural) and eyeglass occlusion, including zero-lens glasses for non-wearers to ensure consistent occlusion conditions, enabling clean covariate analysis (Section 3.3, Table 2).
- **Clear gap analysis.** Table 1 explicitly compares VIBEFACE against MOBIO, Replay-Mobile, OULU-NPU, SOTERIA, and others, making the targeted contribution transparent.

## Weaknesses

### Fatal
None.

### Major

- **Face verification protocol is methodologically invalid for biometric benchmarking.** Section 4.2 evaluates verification using only mated (same-subject) comparisons at a single fixed threshold of 0.5, reporting "percentage of frames correctly authenticated." No impostor (cross-subject) comparisons are performed, and no standard biometric metrics—ROC/DET curves, TAR@FAR, or EER—are reported. This is not a recognized biometric verification protocol. For a dataset paper whose central claim is to "establish a new benchmark for evaluating the robustness and fairness of biometric verification systems," this omission severely undermines the benchmark's interpretability and scientific validity.
- **Dataset scale is incompatible with demographic fairness claims.** With only 50 subjects (12–13 per racial group), subgroup performance estimates in Tables 3–4 have extremely wide confidence intervals and are statistically unreliable. The paper positions VIBEFACE as supporting "demographic fairness" and "generalizable verification systems," yet the sample size is orders of magnitude below what is needed for meaningful fairness analysis. This limitation is inherent to the current corpus and cannot be fixed without substantial additional data collection.

### Minor

- **Controlled studio acquisition contradicts "realistic remote eKYC" framing.** The Abstract and Introduction repeatedly stress "unconstrained conditions," "at home," and "variable lighting," yet Section 3 states data were collected in a "controlled studio environment" with "trained operators continuously supervising participants." While the eKYC action taxonomy is valuable, the actual capture conditions are supervised and standardized, not unsupervised remote eKYC.
- **Face detection benchmark saturates modern detectors.** RetinaFace and MediaPipe achieve near-perfect accuracy (≥95 %, often 100 %) across most conditions (Table 3), indicating the dataset is not challenging for contemporary detectors. The inclusion of the 2016 MTCNN does not compensate for this saturation.
- **Demographic subgroup analyses lack uncertainty quantification.** With cell sizes as small as N≈12, reported subgroup differences (e.g., "Caucasian subgroup" or "female participants") are likely noise. The paper should provide confidence intervals, bootstrapped variance estimates, or at minimum acknowledge the low statistical power.
- **Speculative claims about PAD and deepfake research.** The Conclusions claim suitability for presentation-attack detection (PAD) and deepfake detection, yet the paper includes no attack data, artifacts, or PAD baselines—making these claims unsupported.

### Trivial
None.

## Nice-to-Haves

- Add a valid biometric verification protocol with impostor comparisons, ROC/DET curves, and threshold-independent metrics (TAR@FAR, EER).
- Include subject-wise similarity heatmaps or qualitative failure cases to substantiate claims about challenging conditions.
- Expand the subject pool to several hundred identities if the paper continues to position itself as a fairness benchmark.
- Add a classical texture-based PAD baseline on the occlusion/action scenarios to substantiate cross-task utility.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Comparison with million-scale web datasets unfairly favors baselines.** The harsh critic criticizes VIBEFACE for being smaller than VGGFace2 or MS-Celeb-1M, but the paper never claims to compete on scale; it targets a specific niche (eKYC video with ethical sourcing). Criticizing a specialized dataset for not matching the scale of general-purpose web-crawled datasets is scope mismatch and favors the baseline.
- **"These invalidate the paper's core contribution."** The harsh critic overstates the case. While the verification protocol and small N are serious weaknesses, the eKYC scenario taxonomy, ethical documentation, and systematic condition variation remain genuine, specific contributions. The paper is flawed but not structurally void.
- **Missing appendix/proofs references.** Parser artifacts, not author errors.

## Novel Insights

The tension between the paper's genuine novelty—its eKYC action taxonomy and ethical data collection—and its methodological shortcomings highlights a recurring pattern in dataset papers: compelling data design can be undermined by weak downstream evaluation. The most useful insight for the authors is that the value of VIBEFACE lies primarily in its *scenario design* and *ethical provenance*, not in its ability to serve as a large-scale fairness benchmark. Reframing the paper honestly as a small, controlled, ethically sourced pilot dataset with a unique eKYC scenario taxonomy would better align claims with evidence and likely improve reception.

## Suggestions

1. **Redesign the verification experiment.** Add impostor comparisons, compute ROC/DET curves, and report TAR@FAR and EER rather than a fixed-threshold accuracy metric.
2. **Reframe the scale claim.** Either collect substantially more subjects (several hundred) or honestly present VIBEFACE as a controlled pilot dataset valuable for its eKYC scenarios and condition variation, not as a large-scale fairness benchmark.
3. **Add statistical rigor to demographic analyses.** Report confidence intervals or bootstrap CIs for all subgroup comparisons, and include a frank power analysis.
4. **Align the abstract/introduction with the actual capture setting.** Remove language implying unsupervised remote "at home" collection if the data was captured in a supervised studio.
5. **Substantiate cross-task claims.** Include at minimum a classical PAD baseline (e.g., texture-based liveness detection) on the occlusion/action scenarios before claiming utility for PAD or deepfake detection.

## Score and Decision

Calibration papers used for comparison:
- `lAhQCHuANV.md` (avg 6.33): Strong face-recognition fairness paper with proper theoretical ROC analysis. VIBEFACE has weaker methodology but a novel dataset contribution.
- `4YzVF9isgD.md` (avg 5.25): Synthetic face dataset (HyperFace) with decent methodology but limited novelty, accepted as poster. VIBEFACE has weaker experiments but real data and targeted novelty.
- `E6EbeJR20o.md` (avg 5.50): Large 3D face mesh video dataset (NeuFace), rejected for limited novelty despite large scale and reasonable experiments. VIBEFACE has smaller scale but more targeted application relevance.
- `hWRc2L2hc5.md` (avg 4.50): Face recognition augmentation method, rejected for missing comparisons and limited gains. VIBEFACE is more substantial as a dataset contribution.
- `q7XxKp2rHs.md` (avg 3.00): Small-scale FR method with marginal improvements. VIBEFACE is more novel and concrete.
- `TJU9J8iQXL.md` (avg 2.33): Fairness metric critique with non-novel core claim. VIBEFACE exceeds this in concrete contribution.

VIBEFACE is more original and concrete than the low-scoring anchors, but its central benchmark claim is undermined by a non-standard verification protocol and a sample size far too small for the fairness and benchmarking claims made. Compared to the accepted `4YzVF9isgD` (5.25), VIBEFACE has weaker experimental rigor despite having real data. It sits below the borderline: the two major methodological gaps (invalid verification protocol and insufficient scale) pull it down to a clear reject, though not a catastrophic one.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>