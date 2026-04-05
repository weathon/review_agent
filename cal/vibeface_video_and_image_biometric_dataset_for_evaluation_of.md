=== CALIBRATION EXAMPLE 10 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is indeed about a video-and-image biometric dataset for face evaluation. However, “evaluation of faces” is a bit vague; the core contribution is a controlled-access dataset for face verification and detection, not face understanding broadly.
- The abstract clearly states the dataset size, modality mix, subject count, and demographic balancing. It also identifies the intended use case gap: eKYC-style verification workflows.
- The main issue is that the abstract makes a stronger contribution claim than the paper fully substantiates. “A new benchmark for evaluating the robustness and fairness of biometric verification systems” is plausible, but the paper only evaluates three detectors and two recognizers on a very small dataset, with limited statistical analysis. The fairness claim is especially not yet convincingly established as a benchmark property rather than a dataset design intention.

### Introduction & Motivation
- The problem is well-motivated: eKYC-style face verification is operationally important, and there is a reasonable gap in existing public datasets regarding realistic mobile recording conditions plus demographic balance.
- The gap is identified, but somewhat overstated. The introduction claims there are “no publicly available datasets that include authentic eKYC-style facial videos alongside still images.” That may be true for exactly this combination, but the paper does not sufficiently distinguish “authentic eKYC-style” from other mobile video verification datasets that partially cover similar conditions. This needs a tighter and more defensible positioning against prior work.
- Contributions are reasonably stated: a balanced dataset, realistic scenarios, ethical collection, and benchmark experiments. However, the introduction slightly over-sells the breadth of the fairness contribution. With only 50 subjects, balanced at the group level, the dataset can support exploratory demographic analysis, but it is not large enough to make strong claims about fairness benchmarking across populations.
- ICLR expectation: for a dataset paper, the novelty should be clearly tied to a research need and accompanied by a convincing demonstration that the dataset enables analyses not possible before. Here the motivation is solid, but the paper needs a more precise articulation of why this dataset meaningfully advances methodological research rather than just providing a small controlled collection.

### Method / Approach
- The dataset construction is described with useful detail: capture devices, portrait orientation, five sessions, and the specific scenarios in each session. The scenario design is one of the paper’s strongest aspects.
- Reproducibility is only partial. The paper explains what kinds of photos/videos were collected, but not enough operational detail to fully reconstruct the protocol: e.g., exact subject instructions, duration of each video, number of frames per video, whether prompts were standardized, how “neutral expression” was enforced, whether session order was randomized, and how many samples per scenario per subject are ultimately included in each split of the released dataset.
- The assumptions behind the benchmark setup deserve more justification. For face verification, using a frontal image from the flash session as the reference sample may emulate a document/selfie setup, but it is also a very specific choice that likely advantages certain methods and does not probe cross-session generalization thoroughly. The paper should explain why this reference choice is representative and whether alternative reference protocols were considered.
- The verification evaluation uses a fixed threshold of 0.5 for ArcFace and MagFace similarities. This is a major methodological weakness. A fixed, uncalibrated threshold across models is not standard and can substantially distort comparative conclusions, especially because the two methods may not be on the same similarity scale. If the goal is benchmarking, thresholds should be calibrated on a validation split or results should be reported using threshold-independent metrics such as ROC-AUC, TAR@FAR, EER, or DET curves.
- There are logical gaps in the benchmark methodology: the paper reports “percentage of frames correctly authenticated,” but does not explain how frame-level decisions relate to video-level verification. Since the dataset includes videos, a video-level evaluation would be much more meaningful than aggregating frame-level outcomes, which can inflate performance and ignore temporal consistency.
- Edge cases/failure modes are under-discussed. Scenarios 17 and 18 are explicitly occlusion-heavy, but they are excluded from the detector benchmark and not evaluated in verification. That is understandable for detection, but the paper misses an opportunity to characterize them as hard cases for verification or liveness-related analysis.
- The “legally compliant” and “ethically sourced” claims are important, but the method section does not explain the governance process in enough detail to support those claims beyond consent and controlled access. For a biometric dataset, details about IRB/ethics approval and data retention/access policies matter.
- No theoretical claims are central here, so proof completeness is not relevant.

### Experiments & Results
- The experiments partially test the dataset’s intended claims: they show that common face detectors and recognizers behave differently across lighting, occlusion, pose, and demographic slices. That is a useful sanity check.
- However, the experiments are too limited to substantiate the dataset’s broader claims about robustness and fairness benchmarking. Only three detectors and two recognizers are tested, all off-the-shelf and apparently without tuning. That is acceptable for a dataset paper, but the paper should be more modest in what it concludes.
- Baselines are reasonable choices in spirit, but the comparison is incomplete. For detection, MTCNN, RetinaFace, and MediaPipe are fine. For verification, ArcFace and MagFace are plausible, but the paper does not say which pretrained checkpoints were used, whether they were trained on overlapping data, or whether any preprocessing/alignment was applied consistently. Those details are necessary for fair comparison.
- The biggest experimental concern is the absence of statistical rigor. There are no error bars, confidence intervals, or significance tests. Since the dataset has only 50 subjects, variance across identity subsets could be substantial. Claims like “female participants consistently achieved slightly higher verification rates” or “Caucasian subgroup performed slightly worse” may be unstable without uncertainty estimates.
- There are missing ablations that would materially strengthen the paper:
  - performance by session type separated from scenario type,
  - cross-session generalization,
  - effect of eyeglasses versus no glasses under matched lighting,
  - effect of reference image choice,
  - aggregation at the subject/video level rather than frame level,
  - sensitivity to threshold choice in verification.
- The results appear to support only a limited set of claims. For example, Table 3 and Table 4 do show that weak natural light and eyeglasses degrade some methods, especially MTCNN and MagFace. But some reported differences are small, and because the metrics are frame-level percentages, it is hard to know how practically meaningful they are.
- The dataset comparison in Table 1 is helpful, but it is not entirely clear how “eKYC” is defined across datasets, nor whether some listed datasets have similar controlled mobile videos that were simply not labeled that way. A more careful comparison would improve credibility.
- For ICLR, the dataset contribution would be stronger if the paper included a clearer benchmark protocol and stronger evaluation metrics. As written, the experiments are informative but not yet compelling enough to validate the full ambition of the dataset.

### Writing & Clarity
- The overall structure is understandable, and the scenario/session organization is easy to follow once described.
- The main clarity issue is not grammar but methodological ambiguity in the benchmark section. The paper needs to distinguish more clearly between:
  - sample-level detection,
  - frame-level verification,
  - video-level verification,
  - and subject-level fairness analysis.
  Right now these are blurred together in the text and tables.
- Tables 3 and 4 are informative in layout, but the interpretation is difficult because they aggregate across many axes at once. It would help to separate session effects, scenario effects, and demographic effects more cleanly. As presented, it is hard to tell whether observed differences are due to lighting, pose, glasses, age, or their interactions.
- Figure 1 is useful in principle for demographics, and Figure 2/3 help illustrate scenario design. The figure captions are mostly adequate, though the narrative around them could be tighter. The paper would benefit from a concise visual summary of the acquisition protocol and the count of samples per scenario/session.

### Limitations & Broader Impact
- The authors mention ethical handling, consent, controlled access, and non-commercial use, which is appropriate and important.
- They do not sufficiently acknowledge the core scientific limitation: 50 participants is small for claims about fairness, demographic robustness, or generalization. The group balancing is helpful, but it does not substitute for scale.
- Another missed limitation is the studio-like controlled acquisition environment. Although the scenarios mimic eKYC behavior, the sessions were still collected in a controlled setting with trained operators. That weakens the claim that the dataset fully reflects “real-world” eKYC conditions, since true operational use would include more variability in background, device handling, and user compliance.
- The paper also under-discusses privacy and misuse risks. Even with controlled access, facial biometrics are inherently sensitive, and the paper should more explicitly address the risks of identity leakage, secondary use, and model inversion or re-identification.
- The broader impact section is incomplete in spirit. The dataset could help fairer biometric research, but it could also support more powerful face surveillance or identity systems. That dual-use risk should be acknowledged more explicitly, especially for an ICLR submission.

### Overall Assessment
VIBEFACE is a genuinely relevant dataset contribution with a well-chosen application focus: mobile, eKYC-style face verification under controlled but varied conditions, with deliberate demographic balancing and multimodal still/video capture. That said, the paper currently falls short of ICLR’s stronger bar for empirical rigor and methodological clarity. The dataset design is promising, but the benchmark evaluation is limited, relies on a fixed threshold that weakens the verification comparison, lacks uncertainty estimates, and does not convincingly establish fairness or robustness claims at the level implied by the abstract and conclusion. I think the contribution is real and potentially useful, but the paper needs stronger protocol justification, threshold-independent verification metrics, and more careful limitation of claims before it would be competitive at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces VIBEFACE, a small but carefully designed face biometrics dataset intended for eKYC-style evaluation. The dataset contains 50 subjects, 2,250 still images, and 1,550 short videos captured across five sessions with controlled variation in lighting, eyeglasses, pose, and interaction scripts, and includes demographic metadata intended to support fairness analysis. The paper also provides baseline experiments for face detection and face verification using off-the-shelf models.

### Strengths
1. **Relevant and timely application focus for ICLR standards.**  
   The paper targets a concrete real-world problem—mobile/eKYC face verification—where robustness, demographic balance, and controlled capture conditions matter. This is a meaningful dataset contribution rather than a routine benchmark on scraped Internet data.

2. **Clear emphasis on ethics, consent, and controlled access.**  
   The authors explicitly describe informed consent, GDPR/AI Act compliance, anonymization, withdrawal rights, and restricted non-commercial access. For a biometric dataset, this is a substantive and necessary strength, especially given the privacy sensitivities around facial data.

3. **Multimodal and scenario-rich design.**  
   The dataset is not limited to static portraits: it includes standardized photos, selfie photos, selfie video, and verification videos with specific actions such as head rotation, blinking, expression changes, and partial occlusions. This is useful for studying both verification and liveness-related behavior in practical workflows.

4. **Demographic metadata and balancing are explicitly considered.**  
   The dataset provides gender, age, race, facial hair, hair color, and piercing metadata, and the paper states that the collection was balanced across gender, age, and race. ICLR reviewers often value datasets that enable downstream fairness analysis; this is a genuine contribution.

5. **Baseline evaluations are useful and aligned with the dataset’s purpose.**  
   The face detection and verification experiments help demonstrate that the dataset is non-trivial and can reveal performance variation across conditions. The paper reports results across sessions and demographic groups rather than only aggregate metrics, which is appropriate for a fairness-oriented benchmark.

### Weaknesses
1. **Limited scale and potential narrowness of the benchmark.**  
   The dataset has only 50 participants. For an ICLR dataset paper, this is relatively small, which limits statistical power, diversity of identities, and the breadth of possible research use. It may be useful as a controlled pilot dataset, but the paper does not convincingly argue why this scale is sufficient for broad community benchmarking.

2. **Novelty is somewhat incremental relative to existing biometric datasets.**  
   The main novelty is the combination of consented data, eKYC-style videos, and demographic balance. However, similar themes already appear in prior mobile biometric and PAD datasets cited by the paper (e.g., MobiBits, SOTERIA, WMCA/HQ-WMCA, Replay-Mobile). The paper claims uniqueness of “authentic eKYC-style facial videos,” but the distinction from prior mobile short-video datasets is not deeply justified.

3. **Evaluation methodology is too limited for strong ICLR acceptance.**  
   The benchmarks only test two detection models and two verification models, all off-the-shelf. There is no dataset-specific protocol design, no cross-session/cross-device generalization analysis, no statistical significance testing, and no comparison to human performance or calibration. The experiments show utility, but they do not thoroughly establish VIBEFACE as a rigorous benchmark.

4. **The verification protocol is underspecified and potentially weak.**  
   The paper uses a fixed similarity threshold of 0.5 for ArcFace and MagFace, but it is unclear whether this threshold was tuned on a validation split, taken from prior work, or chosen arbitrarily. For a verification benchmark, ICLR reviewers would expect clearer protocol definition, possibly ROC/EER/AUC, TAR@FAR, and subject-disjoint splits.

5. **Some methodological claims appear stronger than the evidence supports.**  
   The paper repeatedly emphasizes fairness and bias analysis, yet the experimental evidence is limited to descriptive subgroup performance tables on a very small cohort. This is suggestive, but not enough to substantiate broad claims about algorithmic fairness or demographic bias.

6. **Lack of deeper dataset documentation and reproducibility detail.**  
   Important details are missing or only briefly stated: exact per-session counts, per-subject contribution structure, train/test or evaluation splits, annotation format, subject recruitment criteria, capture protocol timing, and access mechanics. For a dataset paper, stronger documentation is essential.

7. **The comparison table and related-work framing are not fully convincing.**  
   The paper asserts that no public dataset includes authentic eKYC-style videos, but the comparison is coarse and does not sufficiently disentangle “eKYC-style,” “mobile face video,” and “verification workflow.” ICLR reviewers may see this as an overclaim unless the differences are precisely defined.

### Novelty & Significance
**Novelty: Moderate.** The dataset introduces a useful combination of consented, demographically balanced, mobile-captured face images and eKYC-style videos, which is practically relevant. However, the core idea is more an integration and curation contribution than a fundamentally new methodological advance, and the small scale limits the breadth of novelty.

**Significance: Moderate.** As a resource for controlled studies of eKYC verification, lighting, eyewear, and demographic effects, the dataset could be useful to the community. Under ICLR standards, though, its significance is somewhat constrained by size and by the limited depth of evaluation; it is likely more compelling as an applied biometrics/data resource paper than as a high-impact ICLR contribution.

**Clarity: Fair.** The acquisition scenarios are described reasonably clearly, and the dataset structure is understandable. That said, several important evaluation and protocol details remain under-specified.

**Reproducibility: Fair to limited.** The scenario design is reproducible in principle, but the paper does not provide enough detail for complete reproduction of benchmarks or fair reimplementation of the evaluation protocol. Controlled access further limits immediate reproducibility unless comprehensive documentation accompanies release.

**ICLR acceptance bar assessment:** This paper has a solid practical motivation and responsible dataset design, but it likely falls below the typical ICLR bar for methodological novelty and experimental depth. ICLR generally expects either a substantial new learning method, a strong theoretical contribution, or a dataset/benchmark paper with broad scale, strong methodological rigor, and convincing evidence of community impact. Here, the contribution is useful but relatively modest and not yet backed by sufficiently deep analysis.

### Suggestions for Improvement
1. **Strengthen the benchmark protocol substantially.**  
   Add standard verification metrics such as ROC curves, AUC, EER, TAR@FAR, and cross-session/cross-device evaluations. Use subject-disjoint splits and clearly document how thresholds are selected.

2. **Provide a more rigorous fairness analysis.**  
   Move beyond subgroup average accuracy. Report confidence intervals, statistical significance, intersectional analyses, and error breakdowns across combinations of gender, age, race, lighting, and eyeglasses.

3. **Clarify and expand dataset documentation.**  
   Include a data card-like section: recruitment process, exact counts per session/scenario, per-subject contributions, file naming conventions, metadata schema, and detailed access instructions. This would materially improve reproducibility.

4. **Justify the novelty relative to existing datasets more carefully.**  
   Explicitly contrast VIBEFACE with MobiBits, SOTERIA, WMCA, and Replay-Mobile in terms of capture protocol, authenticity of eKYC workflow, demographic balancing, and consented collection. Avoid broad claims unless they are precisely supported.

5. **Increase the empirical value of the release.**  
   Consider adding more baselines, especially lightweight mobile models, PAD-oriented models, or video-based verification methods. A deeper benchmark suite would make the dataset more valuable to the ICLR audience.

6. **Discuss limitations candidly.**  
   Acknowledge that the dataset is small, controlled, and limited to 50 subjects, and explain what questions it can and cannot answer. This would improve credibility and help readers interpret the results appropriately.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add verification benchmarks against standard public face-verification datasets and protocols (e.g., LFW, CFP-FP, AgeDB-30, IJB-B/C, or a mobile/eKYC-oriented benchmark) because the claim that VIBEFACE is a “new benchmark” is not credible without showing how its difficulty and failure modes compare to established evaluation settings.  
2. Add strong baselines beyond ArcFace/MagFace and beyond face detectors: evaluate current mobile-friendly and SOTA verification pipelines with explicit alignment, quality filtering, and tracking across video frames, because the paper claims practical eKYC relevance but only tests two embedding models with a single threshold rule.  
3. Add ablations isolating what VIBEFACE contributes: remove eKYC motions, remove demographic balance, remove glasses, remove lighting variation, and compare performance shifts, because without these controlled comparisons it is unclear which claimed dataset properties actually drive the observed results.  
4. Add a cross-session/cross-device protocol and report identity-level metrics, not frame-level success rates, because the current evaluation can be inflated by correlated frames and does not demonstrate that the dataset supports realistic deployment claims.  
5. Add a liveness/PAD or spoof-vs-bona-fide benchmark if the paper wants to argue eKYC utility, because eKYC datasets are expected to support presentation-attack and motion-based challenge evaluation, and this is currently only asserted, not shown.

### Deeper Analysis Needed (top 3-5 only)
1. Add statistical significance testing and confidence intervals for all detection/verification comparisons, because with only 50 subjects the reported subgroup gaps may be sampling noise and the fairness claims are not trustworthy without uncertainty estimates.  
2. Analyze performance at the identity level and per-subject level, not only aggregated by demographic bins, because the paper claims demographic fairness but does not show whether a few hard identities or capture conditions dominate the results.  
3. Quantify the effect of each nuisance factor separately and jointly (pose, glasses, lighting, motion, camera type), because the current tables conflate them and do not show which aspect of the dataset is actually challenging for modern systems.  
4. Report how balanced the dataset remains after filtering/detection failures, because any missingness or frame dropouts could distort subgroup comparisons and undermine the fairness narrative.  
5. Justify the fixed 0.5 similarity threshold with ROC/DET/EER analysis and threshold calibration by session, because the current verification conclusion depends on an arbitrary cutoff and may not generalize across models or operating points.

### Visualizations & Case Studies
1. Add ROC, DET, and score-distribution plots split by session and demographic group, because these would reveal whether the claimed fairness/difficulty differences are real or just threshold artifacts.  
2. Add per-scenario failure galleries for detection and verification, especially for eyeglasses, weak light, and off-angle motion, because the main claim is about realistic eKYC failure modes and the paper currently shows only a few success examples.  
3. Add a subject-level matrix or heatmap showing which identities are hardest across sessions and models, because that would expose whether errors are systematic or concentrated in a few individuals.  
4. Add representative false-accept/false-reject cases for ArcFace and MagFace with similarity scores and conditions, because the paper’s practical claim depends on understanding failure modes, not just aggregate percentages.  

### Obvious Next Steps
1. Release a formal benchmark protocol with train/validation/test splits and evaluation scripts, because a dataset paper should provide a reproducible standard, not just ad hoc frame-level scores.  
2. Expand evaluation to modern video face recognition and quality-aware methods, because the dataset’s main novelty is video/eKYC capture and the current experiments do not test methods designed for that setting.  
3. Add a fairness study using calibrated metrics such as equalized error rates, subgroup AUC, and performance parity gaps, because the paper explicitly claims demographic balance and fairness relevance but does not measure them rigorously.  
4. Add PAD/deepfake/injection-attack experiments, because the authors themselves claim this as a future application and it is the most natural next validation of eKYC-style data.

# Final Consolidated Review
## Summary
This paper presents VIBEFACE, a controlled-access facial biometrics dataset with 2,250 images and 1,550 videos from 50 subjects, captured across multiple mobile devices and sessions designed to mimic eKYC-style verification workflows. The dataset is explicitly balanced across gender, age, and race, and the paper includes baseline face detection and face verification experiments to illustrate utility.

## Strengths
- The dataset design is genuinely relevant to a real deployment setting: it includes eKYC-style video prompts, selfie capture, pose variation, lighting changes, and eyeglasses, all of which are important nuisance factors in mobile face verification.
- The authors have made a serious effort on ethics and governance: the paper describes informed consent, anonymization, controlled-access release, non-commercial use restrictions, and compliance with GDPR/AI Act framing, which is substantive for a biometric dataset.

## Weaknesses
- The dataset is small for the breadth of claims being made: 50 subjects is enough for a pilot resource, but not enough to support strong conclusions about fairness, robustness, or demographic bias at the level implied by the abstract and conclusion. This limits statistical power and makes subgroup findings fragile.
- The benchmark methodology is weak and too ad hoc for a dataset paper that claims to be a “benchmark.” The verification evaluation uses a fixed 0.5 similarity threshold across models, reports frame-level success rather than identity/video-level metrics, and provides no confidence intervals or significance testing. As a result, the reported comparisons are hard to interpret and may be threshold-dependent artifacts.
- The evaluation is too narrow to validate the paper’s broader claims. Only three detectors and two recognizers are tested, all off-the-shelf, with no cross-session/cross-device protocol, no ROC/EER/TAR@FAR analysis, and no deeper ablations isolating the contribution of glasses, lighting, pose, or video motion. This makes the “robustness and fairness benchmark” claim overstated.
- The novelty is somewhat incremental relative to existing mobile biometric and PAD datasets. The paper does combine consented collection, demographic balancing, and eKYC-style videos, but it does not convincingly distinguish VIBEFACE from prior mobile face/video datasets beyond this specific workflow packaging.

## Nice-to-Haves
- Add a formal dataset card with exact per-session/per-scenario counts, per-subject contributions, file naming conventions, and metadata schema.
- Provide subject-disjoint evaluation splits and standard verification metrics such as ROC, AUC, EER, TAR@FAR, and DET curves.
- Include uncertainty estimates and statistical tests for subgroup comparisons.
- Expand the benchmark suite to include video face recognition, quality-aware methods, and PAD-oriented baselines.

## Novel Insights
The most interesting aspect of VIBEFACE is not its scale, but its attempt to operationalize a realistic mobile eKYC workflow with explicit scenario scripting: selfie capture, guided verification motions, glasses/no-glasses sessions, and lighting variation on consumer smartphones. That combination is practically useful, but the current paper does not yet turn it into a rigorous benchmark. The results are mostly descriptive, and because the evaluation is frame-based and threshold-driven, the evidence for fairness and robustness remains suggestive rather than convincing.

## Potentially Missed Related Work
- MobiBits — relevant because it includes mobile biometric data with demographic metadata, making it a close comparator for the dataset and fairness motivation.
- SOTERIA — relevant because it is explicitly discussed as a balanced, responsible mobile face dataset with demographic diversity.
- WMCA / HQ-WMCA — relevant because they cover mobile biometric conditions and challenge factors such as occlusion and real-world capture variation.
- Replay-Mobile — relevant because it is a mobile face video dataset and helps contextualize what is actually new about the eKYC-style workflows here.
- PAD and mobile face presentation-attack datasets more broadly — relevant because the paper itself suggests liveness and attack detection as future uses, but does not benchmark them.

## Suggestions
- Replace the fixed-threshold verification setup with a proper benchmark protocol: subject-disjoint splits, calibrated thresholds, ROC/EER/TAR@FAR reporting, and identity/video-level metrics.
- Make the claims more precise and modest: present VIBEFACE as a controlled, consented eKYC-style dataset with useful demographic balance, rather than as definitive evidence of fairness or robustness.
- Add a clearer experimental section that separates the effects of session, scenario, camera, glasses, and demographic group, ideally with confidence intervals and failure-case visualizations.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
