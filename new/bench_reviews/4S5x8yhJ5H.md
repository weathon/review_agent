Now I have a thorough understanding of the paper and relevant calibration anchors. Let me write the final review.

## Summary

VIBEFACE introduces a novel, ethically collected, demographically balanced face biometric dataset comprising 2,250 images and 1,550 videos from 50 subjects, designed to support face verification in eKYC (electronic Know Your Client) scenarios. The dataset features five acquisition sessions varying in lighting and eyeglass occlusion, 18 scenarios mimicking eKYC workflows (head rotation, blinking, expression changes, etc.), and three consumer smartphones. Benchmarks are provided for face detection (MTCNN, RetinaFace, MediaPipe) and face verification (ArcFace, MagFace).

## Strengths

- **Ethically and legally collected demographically balanced dataset**: VIBEFACE achieves 50:50 gender balance, near-balanced racial distribution across four groups (13 African, 13 Caucasian, 12 East Asian, 12 South Asian), and age distribution following ISO 19795-2 — a genuine contribution after the withdrawal of web-scraped datasets like VGGFace2 and MS-Celeb-1M. GDPR and AI Act compliance with informed consent, anonymization, and controlled-access licensing is documented (Section 3.4).

- **Well-structured scenario and session design**: The five sessions systematically vary lighting (artificial, flash, natural, weak natural) and occlusion (glasses), while the 18 scenarios include both still images and seven eKYC-style video actions (head rotation, blinking, expression changes, mouth opening, face covering, face touching). This provides a principled framework for studying environmental and behavioral variability (Sections 3.2–3.3, Table 2).

- **Honest demographic reporting in detection results**: Table 3 shows MTCNN's reduced detection rate for African-descent subjects (0.610 OAV) vs. East Asian (0.812 OAV) and for older age groups, providing useful evidence of model bias even at limited scale.

## Weaknesses

### Fatal

None.

### Major

- **Verification benchmark reports only genuine acceptance rates without impostor comparisons (Section 4.2, Table 4)** — The paper defines verification success as "the similarity score exceeded a fixed threshold of 0.5" and measures "the percentage of frames in which the face was correctly authenticated" — but no impostor (cross-subject) comparisons are reported. Standard biometric verification evaluation requires both genuine and impostor score distributions to compute metrics like EER, TAR@FAR, or ROC curves. Without these, one cannot determine whether the models actually discriminate identity at all, making Table 4's claims about verification performance incomplete. The relative comparisons across demographics and conditions remain informative, but the paper presents this as a full "verification" benchmark when it is only a genuine acceptance rate study.

- **Dataset scale of 50 subjects (12–13 per racial group) limits statistical reliability of demographic fairness claims** — With only 12–13 subjects per racial subgroup, the demographic fairness insights (e.g., Table 3's racial detection gaps, Table 4's age/race verification differences) rest on extremely small sample sizes. The paper does not acknowledge this limitation or provide confidence intervals/significance tests. Findings at this scale may not be reproducible or generalizable.

- **Claims of "realistic operational settings" and "authentic eKYC-style" recordings are overstated** — The abstract describes "realistic operational settings such as electronic Know Your Client (eKYC) procedures," but Section 3 reveals that all data was collected "in a controlled studio environment" with "standardized instructions" and "continuously supervised by trained operators." Real eKYC sessions involve unguided users in uncontrolled home environments. While the behavioral scenarios (turning head, blinking, etc.) mimic eKYC workflows, the ecological validity of operator-supervised studio recordings differs substantially from actual eKYC deployments.

### Minor

- **Detection benchmark is near ceiling for modern detectors** — RetinaFace achieves near-perfect detection (~99–100% across all conditions), and MediaPipe similarly performs above 94% in almost all cells (Table 3). Only MTCNN shows interesting variation, but it is an older detector. This limits the diagnostic value of the detection benchmark for state-of-the-art methods.

- **Threshold of 0.5 for verification is not justified and the similarity metric is unspecified** — The paper does not explain why 0.5 was chosen, nor whether this is cosine similarity, Euclidean distance, or another metric. Different metrics and thresholds would yield very different genuine acceptance rates, making absolute numbers hard to interpret.

- **Acquisition device randomized per session confounds session and device effects** — Because the device was "randomly chosen before each session" (Section 3.3), performance differences across sessions cannot be disentangled from device variability. This limits what can be concluded about cross-session generalization.

- **Conclusion overclaims about other datasets' demographic imbalance** — The claim that findings "highlight that many existing datasets lack demographic balance, which can result in biased model performance" (Section 5) is not supported by any comparison with other datasets. The paper evaluates only VIBEFACE, not whether training on other datasets produces biased performance on VIBEFACE-style tasks.

## Nice-to-Haves

- Add impostor comparisons and standard verification metrics (EER, TAR@FAR) to make Table 4 a proper verification benchmark.
- Include statistical significance tests or confidence intervals for per-demographic-subgroup comparisons.
- Add at least one PAD baseline, as the paper mentions this as a potential application but provides no evaluation.
- Compare verification/detection performance on at least one existing dataset under the same protocol to contextualize VIBEFACE's difficulty.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Missing ArcFace/MagFace checkpoint details**: This falls under reproducibility nitpicks (undisclosed hyperparameters/trivial implementation details); the paper cites the models and thresholds used, which is sufficient for a dataset paper.
- **The abstract's claim that "there are no publicly available datasets that include authentic eKYC-style facial videos alongside still images" may be too strong**: This is effectively a missing-related-work criticism, which is disallowed by the rules.
- **Per-frame thresholding inflates success rates for longer videos**: This is a minor methodological concern, but it's a standard evaluation approach in video-based face biometrics and the paper is consistent in how it applies the threshold across all conditions.
- **Session B's flash image as reference is "the easiest possible condition"**: Flash photography actually creates harsh lighting and specular reflections that can make matching harder, not easier — so this criticism rests on an incorrect assumption. Using a standardized operator-taken flash photo is a reasonable simulation of document-based enrollment.
- **Formatting/artifacts issues**: These are parser artifacts, not author errors.

## Novel Insights

The verified weakness that verification results lack impostor comparisons is significant but partially mitigated by the fact that the genuine acceptance rates do enable meaningful *relative* comparisons across demographics and conditions — the paper's demographic gap findings (e.g., MTCNN's 0.610 OAV for African subjects vs. 0.812 for East Asian subjects) have directional value even at small scale. The core tension is between VIBEFACE's genuine contribution (an ethically collected, demographically balanced eKYC-style dataset in a field that urgently needs one) and its limited experimental validation, which underrepresents the dataset's potential utility.

## Suggestions

- Add impostor comparisons to Section 4.2 — even a simple all-pairs cross-subject comparison with the same threshold would transform Table 4 into a proper verification benchmark.
- Report confidence intervals or bootstrap standard errors for demographic subgroup results to acknowledge small-sample limitations.
- Soften claims about "realistic operational settings" to acknowledge that supervised studio collection differs from real eKYC deployments.

## Score and Decision

**Calibration anchors compared:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| SctfBCLmWo (dataset bias, strong methodology) | 8.0 | Far above VIBEFACE — rigorous experiments, no methodological gaps |
| lAhQCHuANV (face recognition fairness) | 6.33 | Above VIBEFACE — proper ROC/EER metrics, larger scale |
| 2d734s2WDb (VIBEID, biometric dataset, 100 subjects) | 5.75 | Most similar; VIBEFACE has fewer subjects and a worse verification methodology flaw, but better demographic balance |
| CCo8ElCT7v (face recognition benchmark) | 4.25 | Similar topic; VIBEFACE has a more focused niche contribution but similar experimental limitations |
| UwbX8KOZgK (PixelProse dataset, ethical, limited experiments) | 4.2 | Roughly comparable — both address ethical data needs but have limited experimental validation |
| 6nnWnLK8If (Mineral Fertilizer Dataset) | 3.75 | Below — VIBEFACE addresses a more important need with better domain alignment |

VIBEFACE sits below the VIBEID biometric dataset (5.75) because it has fewer subjects, a more serious methodological gap in verification evaluation, and overclaimed ecological validity. It sits above purely incremental dataset papers (3.75) because it addresses a genuine need in a high-stakes domain with careful ethical and demographic design. The verification methodology flaw is the main differentiator from a higher score — with proper impostor comparisons, this could be a solid contribution.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>