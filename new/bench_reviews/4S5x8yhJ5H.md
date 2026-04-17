Now I have enough calibration context. Let me synthesize my final review.

Key calibration points:
- DeepFaceGen (large-scale face benchmark, 776K samples, 34 gen techniques): Score 5.5 avg → Reject. Its evaluation was criticized for unclear presentation and unsupported conclusions.
- VIBeID (biometric dataset, 100 subjects, limited evaluation clarity): Score ~5.75 → Reject. Similar issues with limited subjects, evaluation protocol, and missing details.
- ILLUSION (deepfake dataset, 1.3M samples, balanced demographics): Score 6.0 → Accept Poster. But had much more scale and diversity.
- HiDF (deepfake dataset, limited baselines): Score ~4.25 → Reject.
- dEGYODD6iU (skin tone PAD fairness): Score ~3.7 → Reject. Weak baselines and fairness evaluation gaps.

VIBEFACE has genuine strengths (ethical data collection, demographic balancing, eKYC scenario novelty) but serious weaknesses (verification evaluation without impostor analysis, arbitrary threshold, 50 subjects, no standard biometric metrics). These are not minor — the verification benchmark claim is fundamentally unsupported by the experiments.

Let me write the final review.</think>## Summary

The paper introduces VIBEFACE, a multimodal facial biometric dataset comprising 2,250 images and 1,550 videos from 50 demographically balanced subjects (gender, race, age) captured under five lighting/session conditions using multiple consumer devices, including eKYC-style verification video sequences. The dataset fills a gap in publicly available resources by combining identity photos, selfie photos/videos, and structured eKYC verification actions with detailed demographic metadata, all collected under GDPR/AI Act compliance.

## Strengths

- **Ethically rigorous data collection.** Full GDPR and EU AI Act compliance, informed consent, anonymization, and controlled-access licensing make this one of the most carefully sourced facial biometric datasets available—a genuine strength given the well-documented withdrawal of Internet-scraped datasets (MS-Celeb-1M, VGGFace2, MegaFace).

- **Thoughtful demographic balancing.** 50:50 gender split, four balanced racial categories (≈12–13 per group), age 18–69 following ISO 19795-2, and Fitzpatrick skin tone coverage. This intentional design is a clear improvement over most existing datasets for fairness-oriented evaluation.

- **Novel eKYC video scenario design.** Scenarios 12–18 (head rotation, blinking, expression change, mouth opening, hand occlusion, face touching) directly mimic real eKYC verification workflows. This is a practical and previously unavailable contribution; no other public dataset combines these structured liveness-oriented video sequences with still identity photos.

- **Well-documented acquisition protocol.** Device types, lighting conditions, eyeglasses sessions, resolution, orientation, and format details are clearly described, enabling reproducibility of the collection methodology.

## Weaknesses

### Major

- **The "face verification" benchmark is methodologically incomplete: no impostor comparisons or standard biometric metrics are reported.** Section 4.2 evaluates verification using only genuine (same-subject) comparisons with a fixed similarity threshold of 0.5, reporting "percentage of frames in which the face was correctly authenticated." There are no impostor trials, no FAR/FRR, no ROC/DET curves, no EER, and no TAR@FAR. This means the core claim—that VIBEFACE supports "face verification" benchmarking and enables evaluation of "robustness and fairness of biometric verification systems"—is empirically unsupported. Without impostor comparisons, one cannot assess security (false accept rate) at all, and the fixed threshold of 0.5 is applied to two different models (ArcFace, MagFace) with different embedding spaces and score distributions, making cross-model comparisons invalid. This is not a minor methodological gap; it undermines the central benchmark claim of the paper.

- **Demographic fairness claims are not statistically supported.** Tables 3 and 4 report per-frame averages by subgroup, but with only 50 subjects (≈12–13 per racial group, ≈8–17 per age group), these descriptive statistics are extremely noisy. Frames from the same subject and session are highly correlated, inflating the apparent sample size. No confidence intervals, hypothesis tests, effect sizes, or subject-level aggregation are provided. Statements like "minimal variation across gender and age" (§4.1) or "Caucasian subgroup performed worse" (§4.2) are anecdotal rather than statistically valid, yet the abstract and conclusions present fairness evaluation as a key contribution. The dataset *may* enable fairness research, but the current evidence does not demonstrate it does so reliably.

- **The dataset scale (50 subjects) limits its utility as a "benchmark" for robust verification and fairness.** With 50 identities, even a proper impostor evaluation would yield limited and potentially unstable results, especially when sliced by demographics (e.g., ≈6 subjects per race×gender cell). The paper repeatedly calls VIBEFACE "a benchmark" and "a new benchmark" without acknowledging that N=50 is well below the scale at which benchmark-grade conclusions about verification robustness and fairness can be drawn. The paper does not discuss this limitation.

### Minor

- **No cross-device breakdown in benchmark results.** The paper explicitly claims the dataset reflects "cross-device variability" and used three smartphones, but the benchmark tables report results pooled across devices without any per-device analysis. This makes it impossible to evaluate whether device type affects detection or verification performance—a core claimed capability of the dataset.

- **Controlled studio conditions vs. "realistic operational settings."** The introduction emphasizes that eKYC sessions involve "users recording short videos under unconstrained conditions—at home, in variable lighting, and across heterogeneous mobile devices." Yet all data were collected in a "controlled studio environment" with "trained operators" and "standardized instructions" (§3). The paper does not discuss the significant gap between studio collection and real in-the-wild eKYC conditions, despite repeatedly using language like "realistic operational settings" and "compliance-driven environments."

- **Near-ceiling detection results for RetinaFace and MediaPipe.** RetinaFace achieves 1.000 detection rate across nearly all image-based conditions, and MediaPipe is close behind. This suggests the face detection benchmark may not be sufficiently challenging to differentiate state-of-the-art methods, limiting its usefulness as an evaluation benchmark for detection.

- **No defined evaluation protocol (enrollment/gallery/probe splits).** The paper does not specify standardized train/test splits, which sessions/scenarios serve as enrollment vs. query, or how cross-session evaluation should be conducted. For a benchmark paper, this reduces reproducibility and comparability of future work.

### Trivial

- Scenarios 17–18 (face occlusion) were excluded from detection evaluation. While understandable for a baseline, these are precisely the scenarios where robustness matters most for eKYC; this should at least be acknowledged more explicitly as a current gap.

## Nice-to-Haves

- Run a proper verification evaluation with impostor pairs and report EER, TAR@FAR, and DET curves per demographic subgroup.
- Include a baseline PAD/liveness detection experiment using the eKYC action videos—the paper itself identifies PAD and deepfake injection as key applications, but provides no evidence of this utility.
- Add an uncontrolled "at-home" session to complement the studio data and validate the "realistic operational settings" claim.
- Provide t-SNE/UMAP visualizations of embeddings colored by demographic attributes to visualize whether fairness issues manifest in representation space.
- Discuss per-cell sample sizes (e.g., race × age group) and their implications for intersectional fairness analysis.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Outdated or limited baseline models"** (cited by human finder comparing to HiDF/ILLUSION reviews demanding more recent models like AdaFace, ElasticFace): While adding more models would strengthen the paper, MTCNN, RetinaFace, MediaPipe, ArcFace, and MagFace are legitimate, widely-used baselines. The issue is not *which* models were chosen but *how* they were evaluated—the missing impostor analysis and arbitrary threshold are the real problems, not the model selection.

- **"Absence of presentation attack data"** (human finder weakness #7): The paper does not claim to include attack samples. It states PAD is a *potential future application* ("VIBEFACE holds potential for broader applications… in emerging areas such as detecting injection attacks"). Criticizing a dataset for not including what it explicitly scopes out is scope creep. This is a nice-to-have, not a weakness.

- **"Limited device variety"** (human finder weakness #6, neutral reviewer weakness #6): Three devices is a reasonable starting point. The real weakness is that device metadata isn't used in the analysis, not the number of devices per se.

- **"No intersectional demographic analysis"** (human finder weakness #6): With 50 subjects, intersectional analysis (e.g., older African females) would have cell sizes of ≈3–4, making any such analysis statistically meaningless. Requesting intersectional breakdowns is unrealistic given the dataset scale; the more fundamental problem is that even the single-attribute analyses lack statistical support.

## Novel Insights

The critical insight across all reviewers is that VIBEFACE has a genuine and valuable design contribution (ethically sourced, demographically balanced eKYC video data), but the paper's *benchmark evaluation* severely undersells and misrepresents what the dataset can do. The verification experiment without impostor trials is not just incomplete—it actively undermines the claim that VIBEFACE serves as a verification benchmark. The irony is that the dataset's scenario design (eKYC actions, multiple sessions, cross-device capture) would genuinely support a rich evaluation protocol *if* the authors had designed one. The paper would be substantially stronger with a proper biometric verification protocol (genuine + impostor comparisons, standard metrics) even on 50 subjects, rather than the current frame-level genuine-only accept rate.

## Suggestions

1. **Redesign §4.2 with a proper verification protocol.** Define gallery and probe sets (e.g., session B flash frontal as gallery; all other sessions as probes). Generate all impostor pairs. Report EER and TAR@FAR=0.1% per scenario, session, and demographic subgroup. This single change would transform the paper's contribution from "dataset description with preliminary evaluation" to "dataset with a meaningful benchmark."

2. **Temper benchmark claims proportionally.** In the abstract and conclusions, replace "benchmark for evaluating the robustness and fairness" with more measured language acknowledging the current scale (50 subjects) and the preliminary nature of the evaluation.

3. **Add subject-level aggregation and confidence intervals** to Tables 3 and 4, reporting per-subject means with bootstrap CIs rather than frame-level percentages.

4. **Include a cross-device breakdown** in at least one table, since the device metadata is available and this is a claimed strength of the dataset.

5. **Acknowledge the controlled vs. uncontrolled gap** (studio vs. real eKYC) as a limitation, and discuss plans for or value of supplementary at-home data.

## Score and Decision

**Calibration comparison:** VIBeID (biometric dataset, 100 subjects, limited evaluation, unclear protocol) received scores of 5/6/6/6 → Reject. DeepFaceGen (large-scale face benchmark with substantive evaluation issues but 776K samples and 34 techniques) received 6/5/6/5 → Reject. HiDF (face dataset with limited/outdated baselines) received 5/3/3/6 → Reject. The ILLUSION dataset (1.3M samples, demographic balance, multi-modal) received 5/5/8/6/6 → Accept Poster.

VIBEFACE shares core evaluation weaknesses with VIBeID (incomplete protocol, limited subjects) and has even more fundamental issues with the verification evaluation (no impostor analysis at all). However, it has genuine dataset design contributions (eKYC scenarios, ethical compliance, demographic balance) that are real and novel. The paper is a reasonable *dataset description* but fails as a *benchmark paper* due to the missing verification methodology. The scale (50 subjects) is also a real limitation for a benchmark. Overall, this is below the VIBeID quality bar because the evaluation protocol is more fundamentally flawed, and below the DeepFaceGen bar because of scale. It sits in the 3–4 range.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>