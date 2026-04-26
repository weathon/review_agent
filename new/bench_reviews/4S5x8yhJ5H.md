Now let me search for calibration anchors.Now I have enough information to write the consolidated final review.

---

## Summary

VIBEFACE is a multimodal biometric dataset of 50 participants, comprising 2,250 still images and 1,550 short videos, captured under five session conditions designed to mimic eKYC (electronic Know Your Client) authentication workflows. The dataset emphasizes demographic balance across gender, race, and age, and demonstrates dataset utility via face detection and face verification benchmarks evaluated across three detectors and two verifiers with per-demographic breakdowns.

---

## Strengths

- **Well-specified eKYC action protocol**: Scenarios 12–18 (circular head rotation, blinking, mouth opening, face occlusion, sequential face touching, expression changes) directly map to real eKYC liveness prompts with a granularity that exceeds existing datasets in Table 1. This is a concrete and original design contribution.
- **Detection benchmark reveals genuine fairness gaps**: Table 3 shows MTCNN's detection rate falling to 0.610 for African-descent subjects vs. 0.984 for East Asian subjects on frontal views — a 37 pp disparity that is actionable and non-trivial, providing real evidence of dataset utility.
- **Above-average ethical and legal compliance**: Explicit GDPR and EU AI Act adherence, informed consent, anonymized identifiers, controlled-access non-commercial licensing, and zero personally identifiable metadata constitute a genuinely thorough framework, especially relevant for biometric data.
- **Cross-device capture**: Three distinct consumer smartphones (Xiaomi Redmi Note 13, iPhone 13, Samsung Galaxy A35 5G), randomly assigned per session, provide meaningful device variability matching real mobile deployment conditions.
- **Demographic balance**: Exact 50:50 gender split, near-balanced four-way racial distribution (13/13/12/12), and ISO 27553:2011-compliant age structure are implemented carefully and exceed comparable datasets.

---

## Weaknesses

### Fatal

None that fully invalidate the dataset itself, but see the Major section — one flaw fatally undermines the verification benchmark specifically.

### Major

- **The verification benchmark contains only genuine pairs (no impostors), rendering Table 4 metrics uninterpretable as verification performance.** Section 4.2 states: "Verification was considered successful when the similarity score exceeded a fixed threshold of 0.5. Performance was measured as the percentage of frames in which the face was correctly authenticated." Every query frame is from the *same* subject matched against their own reference. No cross-subject (impostor) comparisons are run. As a result, the metric is simply genuine-match recall at a single operating point — not verification discrimination. A system that always outputs a similarity score of 0.6 would score ~100% by this metric while being completely useless at distinguishing identities. Standard biometric evaluation requires genuine/impostor score distributions to compute EER, TAR@FAR, or any meaningful ROC. The claim "ArcFace consistently outperformed MagFace across scenarios, sessions, and demographic groups" cannot be supported by this data, since only the genuine-side score distribution is characterized. This is the most severe flaw in the paper, as it invalidates one of the two benchmark tasks used to demonstrate dataset utility.

- **Single fixed threshold τ = 0.5 is used for both ArcFace and MagFace without normalization.** ArcFace and MagFace use the same cosine-similarity range in principle, but magnitude-aware margin training in MagFace produces different score distributions. Applying the same absolute threshold compares models at non-equivalent operating points (different implicit FMR/FNMR tradeoffs). Even the model-level comparison — and all the demographic-level findings in Table 4 — are confounded by this. This cannot be fixed by discussion alone; calibrated thresholds or ROC curves are required.

- **The "realistic/unconstrained" framing of the introduction directly contradicts the controlled studio collection.** The introduction motivates VIBEFACE by describing eKYC "users recording short videos under unconstrained conditions — at home, in variable lighting, and across heterogeneous mobile devices." Section 3 then discloses that "Data acquisition was conducted in a controlled studio environment, each session in a separate room specifically arranged to ensure consistent experimental conditions ... participants received standardized instructions and were continuously supervised by trained operators." The variability comes from scripted lighting manipulations, not from naturalistic capture. This is a legitimate methodological constraint (consent and control were necessary), but the abstract and introduction repeatedly claim the data "reflects realistic authentication scenarios" in ways that are not accurate given the collection protocol. The framing should be corrected to: the dataset *simulates* key variability factors rather than capturing them naturalistically.

### Minor

- **Front-facing vs. rear-facing camera asymmetry in the reference image is unacknowledged.** The reference (Session B) is captured with the rear-facing camera under flash (back camera, as noted in Table 2), while all query samples use the front-facing camera. This introduces a systematic domain gap (sensor, focal length, image geometry) between reference and query that is not mentioned in the paper. This is a confound that users of the dataset need to know about.

- **50 subjects is too small for the demographic claims made.** With 12–13 subjects per racial group and roughly 17 per age group, all differences in Tables 3 and 4 across demographic subgroups are based on single-digit to low-double-digit sample counts. No confidence intervals, bootstrap estimates, or significance tests are reported. Demographic findings should be presented as preliminary observations rather than reliable empirical conclusions.

### Trivial

- RetinaFace achieves 1.000 in all image-based cells and near-perfect scores in video cells (Table 3), making it uninformative for discriminating conditions. This could be noted briefly to save space for the more informative MTCNN/MediaPipe comparison.

---

## Nice-to-Haves

- **Include impostor pairs in the verification benchmark** (50×49 = 2,450 cross-subject pairs minimum) and report EER and TAR@FAR=0.01; this would convert Table 4 from descriptive recall into a real biometric evaluation.
- **Cross-device analysis axis**: The three smartphones used are randomized per session but never broken out as an analysis dimension in Tables 3 or 4; this is one of the stated motivations and would add value.
- **Failure case analysis**: Qualitative examples of frames where detection or verification fails would help users understand dataset challenges beyond aggregated numbers.
- **Calibrated model comparison**: Plot ROC curves or find threshold at a target FMR for each model separately, allowing a fair ArcFace vs. MagFace comparison.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic's suggestion that this venue is wrong (ICLR)**: Removed as an editorial judgment. Dataset papers with benchmark contributions are within scope for ICLR, even if infrequent.
- **PAD/deepfake application claim is "severely oversold"**: The paper is explicit that it provides "high-quality bona fide samples" and frames PAD as a "potential" future direction, not a demonstrated capability. This is reasonable hedging, not a factual overclaim. Removed.
- **Strength: "37 pp MTCNN disparity proves dataset utility"**: Kept — this is grounded in Table 3 data.
- **Strength Finder's point about youngest age group yielding lowest verification**: Technically true per Table 4 (ArcFace OAV: 18–30 = 0.466 vs. 31–50 = 0.469 vs. 51–70 = 0.519) but the differences in the genuine-recall metric are not interpretable as verification discrimination without impostor pairs. Retained only in the detection context.

---

## Novel Insights

The most important observation surfacing from synthesis is that the verification benchmark as designed measures only one side of the biometric operating characteristic (genuine recall) while presenting it as verification performance. This is a subtle but consequential error: dataset papers in the biometric domain are evaluated in part by whether their benchmark protocol is sound enough for others to adopt. Because VIBEFACE's verification protocol lacks impostor pairs, a practitioner replicating Table 4 cannot extract any of the standard biometric metrics (EER, ROC, TAR@FAR) that would actually let them compare systems or assess demographic fairness in verification. The detection benchmark (Section 4.1) is entirely sound and reveals a real fairness gap — the pity is that the verification benchmark, which is more directly relevant to eKYC authentication, is not.

---

## Suggestions

1. **Immediately**: Add all 50×49 impostor pairs to the verification experiment and replace Table 4 with EER/TAR@FAR breakdown; this single change rescues the paper's core benchmark claim.
2. Adopt model-specific thresholds (derived from a calibration subset or a published operating point) rather than a single τ = 0.5 for ArcFace and MagFace comparison.
3. Revise introduction/abstract framing from "unconstrained/realistic" to "controlled simulation of real-world variability" to be accurate.
4. Add a short discussion of the front-to-back camera domain gap in the reference sample, and consider adding a front-camera frontal reference alternative.
5. Flag all demographic group comparisons as preliminary and underpowered (n ≈ 12–13 per group) pending future data collection.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Human Score | Comparison |
|---|---|---|---|
| VIBeID (biometric dataset, 100 subjects, valid benchmark) | `/home/wg25r/review_agent/human_reviews/2d734s2WDb.md` | 5.75 (Reject) | Most comparable: biometric dataset paper with valid benchmark, 2× more subjects; no broken evaluation |
| Person Detection Bias (dataset bias analysis, weak methodology) | `/home/wg25r/review_agent/human_reviews/tC1b9DBWww.md` | 2.50 (Reject) | Low anchor: has dataset + bias analysis but fundamentally insufficient analysis of results |
| NeuFace (face video dataset + method) | `/home/wg25r/review_agent/human_reviews/E6EbeJR20o.md` | 5.50 (Reject) | Medium anchor: face video dataset with methodological contribution, valid benchmark |
| Face Recognition ROC Fairness | `/home/wg25r/review_agent/human_reviews/lAhQCHuANV.md` | 6.33 (Accept) | High anchor: face recognition + fairness, theoretical soundness, valid evaluation |
| F3Set (video benchmark, large scale) | `/home/wg25r/review_agent/human_reviews/vlg5WRKHxh.md` | 7.00 (Accept) | High anchor: strong benchmark paper with comprehensive evaluation |

**Reasoning**: VIBeID (5.75, Reject) is the best analog — a biometric dataset with a valid benchmark at a methods venue. VIBEFACE falls below VIBeID on every dimension: half the subjects, a broken verification benchmark, and a framing inconsistency. The detection benchmark is solid but accounts for only one of two demonstrated uses. The broken verification benchmark is a concrete methodological flaw that a practitioner would immediately encounter when attempting to use Table 4, putting VIBEFACE in the 4.0 range — meaningfully below VIBeID (5.75) and above the genuinely weak papers like tC1b9DBWww (2.50). The paper's strong ethical scaffolding and eKYC design are real contributions, but they do not compensate for the evaluation flaw in the context of demonstrating benchmark utility.

**Originality**: Moderate — the eKYC scenario design is novel; the dataset-plus-benchmark format is standard.
**Importance of research question**: Moderate — eKYC fairness is a real applied problem.
**Claims well-supported**: Partially — detection claims hold; verification claims do not.
**Soundness of experiments**: Detection: sound. Verification: structurally flawed.
**Clarity of writing**: Generally clear with some framing inconsistencies.
**Value to community**: Genuine potential, but only if the verification protocol is corrected.

**Final Score: 4.0**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>