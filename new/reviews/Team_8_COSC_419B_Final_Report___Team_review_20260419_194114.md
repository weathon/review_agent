# Review of Team_8_COSC_419B_Final_Report___Team.md

Now let me do calibration searches to anchor my score.Now I have sufficient information to write the final consolidated review. Let me synthesize everything.

---

## Summary
This paper presents a tracklet-based jersey number recognition pipeline for broadcast soccer footage that replaces a multi-stage 2024 baseline (legibility classification + pose estimation + PARSeq OCR) with a streamlined system combining Real-ESRGAN super-resolution, OWLv2 open-vocabulary detection, collage-based multi-frame aggregation, and Qwen VLM with LoRA fine-tuning. The main reported result is an improvement from 49.7% zero-shot accuracy to 76.41% after two epochs of LoRA fine-tuning on a 1,840-sample evaluation set, along with a ~7.6× reduction in recognition-stage inference calls versus the baseline PARSeq path.

---

## Strengths

- **Sensible and coherent system design** (Section IV, Figure 8): Replacing pose-estimated torso crops with direct open-vocabulary number detection, and replacing per-frame confidence aggregation with a collage presented to a VLM, is a principled architectural direction that addresses real pain points in the baseline. The design is well-motivated and clearly described.

- **Significant LoRA fine-tuning gain** (Section V-C.1): The 49.7% → 76.41% accuracy improvement after two epochs of LoRA on a collage-input formulation is a genuine and substantial result. The convergence visible in Figure 12 (loss dropping ~77% in first 100 steps) supports that the collage representation is learnable with parameter-efficient adaptation.

- **Concrete recognition-stage speedup** (Table I): On a single test tracklet, the collage path requires 5 Qwen inference calls totaling 1.19 s vs. 635 PARSeq calls totaling 9.05 s — a ~7.6× wall-clock reduction at the recognition stage. The paper is explicit that this does not include OWLv2 runtime.

- **Honest and specific error analysis** (Section IV-D): The paper identifies five concrete failure modes — single/double-digit confusion, OWLv2 detection misses, ESRGAN hallucination artifacts, ground-truth label errors, and aggressive legibility filtering — grounded in actual pipeline outputs rather than speculation. Notably the authors flag dataset labeling errors (Figure 14) rather than silently accepting lower accuracy numbers.

- **Clear visual evidence for Real-ESRGAN benefit** (Figures 9–10): The before/after comparison at single-crop and collage levels shows markedly sharper digit boundaries, making the preprocessing motivation concrete.

---

## Weaknesses

### Fatal
None. The paper does not contain invalid methodology that invalidates an existing result.

### Major

- **No end-to-end comparison against the stated baseline on a common test protocol.** This is the most critical gap in the paper. The central framing throughout (Abstract, Sections II, IV, VII) is that the proposed pipeline replaces and improves on the Koshkina & Elder 2024 system. But the paper never reports what that baseline achieves on the same 1,840-sample evaluation set. The 76.41% figure cannot be interpreted as an improvement without a reference point — it could be worse, equal, or better than the baseline. The only quantitative comparisons offered are: recognition-stage timing on one tracklet (Table I, explicitly partial), and component-level legibility comparisons on very small subsets. This means the paper's core comparative claim is entirely unsupported by data. A reader cannot determine whether the proposed pipeline is a useful replacement for the baseline.

- **Evaluation sample sizes are too small to support design decisions.** The CNN legibility comparison is run on the *first five* test tracklets (Section III-A.1); the Swin-T vs. ResNet34 full-pipeline comparison is run on only *two* tracklets, which the paper itself notes represent "24.28% of all ground-truth tracklets assessed in that evaluation slice" (Section III-A.2). The decision not to adopt Swin-T is attributed to a 0.34% accuracy difference on two tracklets — a finding that carries no statistical meaning and is explicitly acknowledged as limited, yet still used as a design justification. These evaluation sizes are inadequate to inform architecture decisions.

### Minor

- **Runtime conclusion overstates what was measured.** The abstract uses the phrase "faster recognition-stage inference," which is accurate. But the runtime section concludes "the proposed recognition architecture is faster in practice" (Section V-D, emphasis added) while the same section states OWLv2 runtime is excluded and Section IV-C notes the full pipeline runs "20h for the whole dataset" unbatched. The conclusion should be restricted to recognition-stage speedup only.

- **LoRA evaluation setup is underspecified.** The 76.41% accuracy figure is the paper's most important result, but the evaluation protocol around it is vague: what are the training/validation/test split sizes from the 1,840 samples, how are classes distributed, and are the 1,840 samples tracklets or collages? Without this, the number cannot be reproduced or interpreted in context.

- **Collage size inconsistency.** Figure 8's caption states "up to 25 images per tracklet," while the same flowchart text says "15 crops per jersey," and Figure 11 shows a 5×5 = 25-image collage. This internal inconsistency raises questions about what the system actually does.

### Trivial

- The Figure 12 text describes loss dropping from 1.447 to 0.327 "in the first 100 steps," but Figure 12 itself starts at step 100 with training loss 0.325. The early-training trajectory is not plotted, making the text description inconsistent with the figure. This is a minor presentation confusion, not a substantive error.

---

## Nice-to-Haves

- **Ablation of pipeline components**: Isolating the contribution of Real-ESRGAN, OWLv2 localization, collage vs. single-best-frame, and LoRA vs. zero-shot independently would clarify which components drive the 76.41% result and whether the collage truly provides multi-frame reasoning benefits or simply gives the model more chances to see one good crop.

- **Systematic label noise quantification**: The paper anecdotally identifies ground-truth errors. A small manually-verified subset (e.g., 50–100 tracklets) with estimated mislabel rate would let readers calibrate how much the reported accuracy is depressed by dataset noise.

- **Performance breakdown by single vs. double digit numbers**: Since single/double-digit confusion is identified as the dominant failure mode, breaking down accuracy by number type would meaningfully characterize where the system succeeds and fails.

- **Qualitative collage success/failure cases with OWLv2 boxes**: Showing examples where the collage aggregation rescues a correct answer that any single frame would have missed — versus examples where it fails despite containing a readable crop — would directly test the "sequence-level reasoning" claim.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — Legibility metrics are "largely uninterpretable"**: The paper explicitly and proactively discloses the weak-supervision labeling scheme and its limitations (Section III-A.2: "introduced label noise, since some frames within a legible tracklet may still be unreadable"). The legibility results are used for backbone selection, not as the core claim. The paper's own acknowledgment neutralizes the force of this criticism. Removed as a strawman against a limitation already addressed.

- **Harsh Critic — "First five tracklets" CNN evaluation is too arbitrary**: This comparison is clearly framed as preliminary motivation for moving to modern architectures (Section III-A.1), not as the paper's core experimental claim. Criticizing its size for a motivational scan is scope creep. Removed.

- **Harsh Critic — EfficientNetV2 and MobileNet fairness unspecified**: The paper uses pretrained legibility classifier weights evaluated (not trained) on the five tracklets — this is a fair observation but applies only to the motivational scan in Section III-A.1, which is already downgraded. Removed as tied to an already-removed point.

- **Harsh Critic — Introduction oversells "sequence-level reasoning"**: The paper uses this phrase but also frames the collage as a way to present more information rather than claiming true temporal reasoning has been proven. This is a reasonable framing choice, not a factual error. Removed as a style preference.

- **Strength Finder — "Comprehensive empirical comparison of legibility backbones"**: Given that image-level labels are inherited weak proxies (acknowledged in the paper), and the full-pipeline comparison uses only two tracklets, describing this as "comprehensive" overstates it. Removed.

- **Strength Finder — "No charset limit" as a key strength**: This is a generic presentation point about using a general-purpose VLM vs. PARSeq; it does not constitute an empirically demonstrated advantage. Removed.

---

## Novel Insights

The framing of collage-as-input as a replacement for per-frame confidence aggregation is conceptually clean: rather than hand-engineering an aggregation heuristic, the VLM is directly exposed to the tracklet's visual evidence and asked to reason across it. The insight that failed detection by OWLv2 serves as an implicit legibility filter — consolidating two separate pipeline stages into one — is an elegant design principle that could transfer to other sports analytics domains. However, neither insight has been fully validated in the current submission; demonstrating them rigorously would elevate the paper significantly.

---

## Suggestions

1. **Run the baseline (Koshkina & Elder 2024) on the same 1,840-sample test set and report its accuracy alongside 76.41%.** This single addition would transform the paper from one that shows an absolute result into one that actually supports its comparative claim.
2. **Increase the full-pipeline evaluation to at least 50–100 tracklets** and report variance. At minimum, revisit the Swin-T pipeline comparison on a larger and representative slice.
3. **Resolve the 25 vs. 15 crops-per-collage inconsistency** in the text/figures.
4. **Restrict the runtime conclusion to recognition-stage speedup only** to avoid overstating the practical benefit.
5. **Add a data table describing the 1,840-sample evaluation split** (train size, val size, test size, class balance) so the 76.41% figure can be interpreted and compared by others.

---

## Evaluation on Key Axes

- **Originality**: Moderate. Collage-as-VLM-input and OWLv2-as-implicit-legibility-filter are fresh ideas in this application domain, but the components themselves (ESRGAN, OWLv2, Qwen, LoRA) are all existing. The combination is novel and sensible.
- **Importance of research question**: Moderate-high. Automated jersey number recognition supports many sports analytics downstream tasks and is a legitimate computer vision challenge.
- **Claims well-supported**: Weak. The core comparative claim (proposed pipeline improves on baseline) is not supported — no baseline number on the same test set is provided. The 76.41% is a genuine result but lacks context.
- **Soundness of experiments**: Weak-to-moderate. LoRA result is real and reasonably described, but evaluation sample sizes are very small for the design-level conclusions, and the end-to-end comparison is missing entirely.
- **Clarity of writing**: Good. The paper is readable, well-structured, and honest about its limitations, which is commendable.
- **Value to research community**: Limited in current form. The system idea is interesting, but a reader cannot tell from this paper whether the approach is actually better than the baseline it aims to replace.

---

## Score and Decision

**Calibration anchors used:**
- *3iJ7eSj2rE* (Synergistic Weak-Strong Collaboration, scores 5/5/3/3, avg 4.0, withdrawn): missing end-to-end evaluation, core claims inadequately supported by experiments — close structural parallel to this paper's main weakness.
- *SPORTU* (x1yOHtFfDh, scores 5/6/6/5, avg 5.5, accepted poster): a sports video benchmarking paper with clearer empirical scope and more complete evaluation.
- *HilIIP4yzw* (scores ~1.8, rejected): pedagogical paper with no empirical evaluation — clearly weaker than this paper.
- *qpz84ykqgv* (EarthquakeNPP, scores avg 5.25, rejected): novel dataset/method but lacks qualitative validation — comparable profile.

The paper under review is more complete than the 1–3 range papers (it has a real system, real results, and honest limitations) but falls below the 5+ range anchors because its central comparative claim is unsubstantiated. The 3iJ7eSj2rE anchor (avg 4.0, rejected for "missing end-to-end evaluation") is the closest structural match. This paper has slightly better presentation and a clearer experimental result, but the missing baseline comparison is if anything more critical here because the paper's entire framing is as a pipeline replacement. Placing it at **3.5** — slightly below the 4.0 anchor because the end-to-end comparison gap is the very heart of the contribution rather than a supporting experiment.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

**Predicted score: 3.5**

**Acceptance rate @ score=3.5: 0.00% (n=2)**

**Acceptance rate @ score=3.5±0.5: 11.11% (n=9)**

**Percentile of score=3.5: 14.7% (n=34)**
