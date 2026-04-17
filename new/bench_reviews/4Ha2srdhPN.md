Now I have sufficient context to write the consolidated review. Let me synthesize everything carefully.

## Summary

GRAID is a framework for generating high-quality spatial reasoning VQA data from real images using only 2D bounding box geometry from object detectors, deliberately avoiding error-prone single-view 3D reconstruction and hallucination-prone LLM-based generation. Applied to three autonomous driving datasets (BDD100k, NuImages, Waymo), it produces over 8.5M VQA pairs across 22 question templates, with human evaluation reporting 91.16% validity versus 57.6% for SpatialVLM's community implementation. Fine-tuning VLMs on GRAID data yields consistent improvements on external benchmarks like BLINK, A-OKVQA, and VSR, with evidence of transfer to held-out question types.

## Strengths

1. **Clean, well-motivated core insight**: The argument that qualitative spatial relationships can be reliably determined from 2D bounding box geometry—avoiding cascading errors from depth estimation and hallucinations from generative models—is clearly articulated and empirically validated. The approach is conceptually simple but practically effective, and the comparison in Table 1 shows meaningful architectural advantages over prior work.

2. **SPARQ is a genuine engineering contribution**: The predicate-based early rejection system achieving up to 1400× speedups (App. Table 3, with concrete timing: 5.17ms predicates vs. 46.95ms full realization for RightOf) demonstrates careful systems thinking that enables scaling to millions of VQA pairs. This is not just a conceptual paper—the framework is designed for practical deployment.

3. **Scale and breadth of output**: 8.5M+ VQA pairs across three real-world datasets and 22 template types, spanning five cognitive categories (spatial relations, counting, ranking, localization, size/aspect), is a genuinely substantial resource contribution. The hierarchical breakdown in Figure 2 shows meaningful coverage.

4. **Compelling generalization evidence**: RQ2's result—training on only 6 question types and improving on 10+ held-out types across two datasets—is strong evidence that models learn transferable spatial primitives rather than mere template memorization. The cross-dataset transfer (BDD→NuImages, +29.1%) and the fact that only 10/143 BLINK Spatial Relations questions contain "car" further support genuine concept transfer.

5. **Consistent benchmark improvements across multiple backbones**: GRAID-based SFT improves performance on BLINK, A-OKVQA, RealWorldQA, VSR, and NaturalBench across Llama 3.2 11B, Gemma 3 4B, Qwen2.5 VL 3B, and Qwen3 VL 8B, outperforming OpenSpaces-tuned variants. These improvements are particularly notable in spatial subtasks (+41.13% Relative Depth, +30.77% Spatial Relations on BLINK).

6. **Substantially higher data quality than the evaluated baseline**: The 91.16% vs. 57.6% gap in human-validated accuracy for OpenSpaces is large and directionally persuasive, even accounting for methodological concerns about the comparison.

## Weaknesses

### Major:

- **The human evaluation comparison with OpenSpaces/SpatialVLM is methodologically weak and the strong comparative claims are not well supported by the evidence.** The paper evaluates only 250 VQA pairs from OpenSpaces (a community implementation, not the original SpatialVLM pipeline) and 317 pairs from GRAID-BDD, out of millions. The OpenSpaces evaluation explicitly notes that many SpatialRGPT questions could not be evaluated due to masked-region queries, and the GRAID evaluation allowed annotators to view bounding boxes while the baseline evaluation lacked equivalent grounding information. More fundamentally, the causal claim that "low-fidelity training data" is "the cause of a model underperforming its size class" is not demonstrated—no controlled experiment isolates data quality as the variable. The direction of GRAID being higher quality is plausible, but the specific 91.16% vs 57.6% comparison is not apples-to-apples, and the generalization to "current training data generation pipelines" from a single community implementation is overextended.

- **Claims about learning generalizable "spatial reasoning concepts" vs. template-specific pattern matching are insufficiently supported.** RQ2 trains on LeftOf, RightOf, HowMany, AreMore, LargestAppearance, and IsObjectCentered, then tests on held-out types. However, many held-out types (e.g., LeftOf vs. FarLeftOf, or other directional variants) share strong lexical and geometric overlap with training types. The paper provides no experiments with paraphrased questions, out-of-domain object categories, or adversarial variations that would distinguish genuine spatial concept learning from surface-level template generalization. The cross-dataset result (BDD→NuImages) is the strongest evidence but still involves the same template engine on similar driving scenes.

- **Experiments use ground-truth bounding box annotations rather than detector predictions, creating an unrealistic deployment gap.** The paper explicitly states (Section 4) that they "select to directly leverage these high-quality labels in GRAID's generation rather than train our own object detectors" to evaluate in isolation. While methodologically clean for evaluating GRAID's template logic, this means the reported 91.16% human validity and all downstream VLM experiments rely on near-perfect detections. In practice, users must run their own detectors, and detection errors (missed objects, mislabeled classes, inaccurate boxes) will propagate into VQA errors. No experiments assess how robust GRAID is to detector noise, which directly impacts the "any detector" claim.

### Minor:

- **Limited human evaluation scope and no inter-annotator agreement metrics.** Only 317 VQA pairs (from 5.3M+ in BDD alone) and no per-question-type stratification or Fleiss' kappa/Cohen's kappa across the four annotators. The human evaluation was conducted only on GRAID-BDD without depth; the paper claims quality across all 8.5M+ pairs and all datasets without independent verification.

- **Post-hoc corrections inflate reported quality relative to baseline.** The paper acknowledges that annotator feedback allowed template corrections, so "the current public datasets have these corrections and thus even higher validity" than the reported 91.16%. The baselines were not afforded analogous corrections, creating a structural advantage.

- **The "similar planes" heuristic in Algorithm 1 is underspecified.** The paper mentions checking whether bounding boxes "lie on similar planes" to avoid ambiguity (e.g., objects at different heights), but this check is not formalized. For a method whose core insight is avoiding ad-hoc heuristics from 3D reconstruction, the opacity around what constitutes "similar planes" and how it's determined from 2D data is an unaddressed gap.

- **Some regressions are insufficiently analyzed.** RQ2 reports regressions on LessThanThresholdHowMany and MoreThanThresholdHowMany, attributing them to overfitting on common question types without further investigation. These regression patterns could reveal important limitations in the data or training approach.

### Trivial:

- No confidence intervals or variance across multiple training runs for the VLM experiments. Single-run evaluation is common in this community, so this is a minor concern.

## Nice-to-Haves

- Run the full pipeline with off-the-shelf detector outputs (e.g., YOLO) and compare dataset quality against GT-based generation to quantify the practical deployment gap.
- Conduct a controlled, apples-to-apples human study evaluating both GRAID and SpatialVLM data under identical annotation protocols and ground-truth access.
- Test on non-driving source images (e.g., COCO indoor scenes) to validate the domain-agnostic claim, even at smaller scale.
- Analyze per-question-type failure rates from the human evaluation to identify which spatial relations are most error-prone under the 2D framework.
- Conduct ablation on depth questions to determine whether they help or hurt downstream performance, given the tension with the paper's anti-3D-reconstruction stance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Inherent limitation of 2D-only spatial reasoning — can't capture 'in front of', 'under', 'on top of'"**: The paper explicitly includes depth-based questions (Closer, Farther) and uses configurable margin ratios to handle depth ambiguities. It also acknowledges this limitation. The paper's claim is about *qualitative* spatial relationships, which many of these are in practice when viewed from a fixed camera perspective. This criticism overstates the limitation since the paper does address depth-related relations.

- **"Domain-agnostic claim undermined by driving-only datasets"**: The paper explicitly states "GRAID is domain-agnostic; we instantiate on driving datasets because they provide among the largest openly available, high-quality object detection annotations at scale, not due to any AV-specific assumption in the method." The framework is demonstrably domain-agnostic (it takes any image + detections). The criticism confuses the demonstration instantiation with the framework's capability. The strong cross-domain transfer to indoor scenes in BLINK further undermines this concern.

- **"Template-based questions produce unnatural spatial reasoning"**: This is standard for synthetic VQA datasets and doesn't constitute a unique weakness. The paper's contribution is showing that these templates teach transferable spatial primitives (demonstrated by held-out type improvements and cross-benchmark gains).

- **"SpatialRGPT not included as a training baseline in RQ3"**: The paper explains that SpatialRGPT's dataset (OpenSpatialDataset) could not be reliably evaluated due to masked-region queries, which is a legitimate methodological barrier. Removing this baseline is reasonable, not an unfair omission.

- **"No ablation on depth questions"**: The paper does provide depth and no-depth variants (Table 2: "With Depth" vs "Without Depth"), though it doesn't run a training ablation comparing them. This would be nice but is not a core flaw.

- **"Dependency on ground-truth detection labels propagates errors"**: This is partially addressed above as a major weakness (unrealistic deployment gap), but the version from the human finder about "cascading errors" from detection is somewhat overblown—the paper found only 5 labeling errors in the BDD GT, and modern AV datasets have well-validated labels.

- **"Unfair comparison — GRAID evaluators could view bounding boxes but SpatialVLM evaluators could not use equivalent grounding"**: This is a legitimate methodological asymmetry but is partially addressed by the fact that GRAID questions are about relationships between objects that are explicitly named in the question (e.g., "Is there a car to the right of a person?"), while OpenSpaces questions sometimes had masked region identifiers that were not interpretable. The conditions differ because the data formats differ, not because of bias in evaluation design.

## Novel Insights

The paper's most insightful contribution is the demonstration that 2D qualitative spatial relationships, extracted purely from bounding box geometry, can serve as effective training signal for VLM spatial reasoning that transfers across domains and question types. The SPARQ predicate-realization separation is an elegant engineering contribution that could be useful beyond this specific framework. The finding that 57.6% of a widely-used community SpatialVLM dataset has incorrect answers is attention-grabbing, though the evidentiary basis for generalizing this to "current training data generation pipelines" needs strengthening.

## Suggestions

1. **Reframe the human evaluation comparison fairly**: Present the OpenSpaces result as "a community implementation of SpatialVLM's approach produces data with significant noise (57.6% invalid in our sample)" rather than as a definitive quality metric for the method, and acknowledge the methodological differences in annotation protocols between datasets.

2. **Add a small experiment with predicted detections**: Even 1-2K VQA pairs generated from YOLO detections (with documented detection metrics) on a subset of images, followed by human evaluation, would dramatically strengthen the "any detector" practical claim.

3. **Soften "concept learning" claims**: Replace "models learn spatial reasoning concepts that generalize" with more precise language such as "models learn spatial reasoning skills that transfer across question types and datasets" until paraphrase/robustness experiments are added.

4. **Report inter-annotator agreement**: With 4 annotators, Fleiss' kappa should be trivial to compute and would strengthen the human evaluation significantly.

## Score and Decision

Calibration:

- **COMFORT (Accept Oral, scores 5-10)**: High-quality spatial reasoning benchmark paper with rigorous evaluation methodology, cross-lingual evaluation, and systematic analysis. GRAID has a similar scope but weaker evaluation methodology and overclaims.

- **Sparkle (Withdrawn/Reject, scores 3-5)**: Similar paper addressing VLM spatial reasoning through synthetic data generation on basic spatial primitives. GRAID is significantly stronger: larger scale, real-world data, human evaluation, multiple backbones, and external benchmark validation. GRAID would clearly score above Sparkle.

- **STUPD (Reject, scores 3-5)**: Synthetic spatial reasoning dataset. GRAID is notably stronger: larger scale, real-world images, human quality evaluation, and VLM fine-tuning results on external benchmarks.

- **DataEnvGym (Accept Spotlight, scores 6-8)**: Strong data generation framework paper. GRAID's contribution is comparable in scope (framework + data + evaluation) but has weaker methodological claims in the evaluation section.

- **ADOPD (Accept Poster, scores 6-8)**: Large-scale dataset paper with practical utility. GRAID has a similar profile: useful framework + large dataset + practical value.

GRAID makes a solid contribution: a clean framework, a genuinely large and practical dataset, and clear empirical improvements. The main weaknesses are (1) the overclaimed human evaluation comparison (the directional finding is almost certainly correct, but the specific numbers are not apples-to-apples), (2) the gap between 2D-only template learning and true "spatial concept" generalization claims, and (3) the deployment gap from using GT rather than predicted detections. These are real but not fatal—the core framework is sound, the data quality is directionally much better than prior work, and the practical utility is demonstrated.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>