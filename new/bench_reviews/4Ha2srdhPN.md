Now I have a thorough understanding of the paper. Let me write the consolidated final review.</think>## Summary

GRAID introduces a framework for generating large-scale spatial reasoning VQA data using only 2D bounding box geometry from object detectors, deliberately avoiding 3D reconstruction errors and LLM hallucinations. Applied to BDD100k, NuImages, and Waymo, it produces 8.5M+ VQA pairs across 22 question types, with a human-verified validity of ~91% versus ~58% for the SpatialVLM community dataset, and demonstrates that models fine-tuned on GRAID data show transfer to held-out question types and external benchmarks.

## Strengths

- **Core insight is well-motivated and sound**: Operating on 2D bounding boxes instead of noisy 3D reconstruction or hallucination-prone LLMs is a genuinely useful design choice that produces demonstrably higher-quality data, as validated by human evaluators (~91% validity). The qualitative-versus-quantitative question framing cleverly sidesteps depth estimation errors.

- **Strong scale and practical contribution**: 8.5M+ VQA pairs across three driving datasets is one of the largest high-quality spatial VQA resources available. The SPARQ predicate-sieve architecture achieves up to 1400× speedup, making large-scale generation tractable. The framework is domain-agnostic and supports standard detection packages (Detectron2, MMDetection, Ultralytics).

- **Compelling transferability evidence (RQ2)**: Training on only 6 question types and evaluating on 10+ held-out types with substantial accuracy gains (e.g., +47.5% on BDD, +37.9% on NuImages for Llama 3.2 11B) across an entirely unseen dataset provides meaningful evidence that the model learns generalizable spatial primitives rather than memorizing templates.

- **Consistent improvements across multiple backbones and benchmarks (RQ3)**: GRAID-tuned models improve on BLINK (+15.94% overall for Llama 3.2 11B), A-OKVQA (+32.5%), and other external benchmarks, across four VLM backbones, including on tasks with minimal domain overlap (only 10/143 BLINK Spatial Relations questions contain "car").

## Weaknesses

### Fatal

None.

### Major

- **RQ3 comparison limited to a single weak baseline**: The only SFT data comparison in RQ3 is against OpenSpaces (the community SpatialVLM implementation), which the paper itself shows has only ~58% answer validity. While the failure to evaluate against SpatialRGPT's data is partly justified (masked regions making evaluation difficult), the absence of comparison against SpaRE's dataset leaves a significant gap. Showing GRAID outperforms a demonstrably low-quality baseline does not fully establish that GRAID is the best available alternative — only that it is better than one bad one. This does not invalidate the quality result, but it weakens the claim of state-of-the-art performance.

- **No non-spatial VQA control in RQ2**: The RQ2 experiment shows that training on 6 spatial question types improves performance on 10+ held-out spatial types. However, without fine-tuning on a comparable amount of generic (non-spatial) VQA data, it is difficult to distinguish "learned transferable spatial concepts" from "learned general VQA answering patterns that happen to help on spatial tasks." The cross-dataset and cross-benchmark results provide some evidence, but a non-spatial VQA control would substantially strengthen the conceptual transfer claim.

- **Algorithm 1 omits the "similar planes" check described in prose**: The text at lines 247-253 states that the RightOf realizer checks "they should lie on similar planes" to avoid ambiguous cases (e.g., objects at different heights). However, Algorithm 1 only checks (1) ≥2 distinct classes, (2) x_min(b1) > x_max(b2), and (3) IoU = 0. The "similar planes" condition is absent from the pseudocode. This inconsistency affects reproducibility and raises questions about whether the 8.5M generated pairs actually include this disambiguation filter, which is important for the claim of high-quality qualitative spatial data.

### Minor

- **Human evaluation uses different protocols across methods**: GRAID evaluators could view images with and without bounding boxes, while SpatialRGPT evaluation was limited by masked region queries. The 317-sample (GRAID) and 250-sample (SpatialVLM) evaluations are small relative to millions of generated pairs. While the quality gap is large enough to be convincing, the headline comparison of "91.16% vs 57.6%" conflates different metrics: GRAID reports a combined validity rate while SpatialVLM reports separate question validity (41.6%) and answer accuracy (57.6%).

- **Dataset generation uses ground-truth annotations, not detector outputs**: The paper explicitly uses GT bounding boxes from the driving datasets to evaluate GRAID "in isolation." While this is a reasonable methodological choice for evaluation, practical deployment requires running detectors, whose errors would propagate into GRAID's output. The human evaluation figure of 91.16% is therefore optimistic relative to real-world use scenarios.

- **Regressions in counting questions acknowledged without analysis**: The paper notes regressions in LessThanThresholdHowMany and MoreThanThresholdHowMany in RQ2, attributing them to "overfitting" because they are the most common question types, but provides no supporting analysis.

### Trivial

- The Waymo subset selection heuristic (balancing object count and largest-object-to-image ratio) is mentioned but not evaluated.

- The paper's "key insight" is repeatedly stated as a self-evident virtue, when it is better understood as a pragmatic trade-off (giving up true 3D spatial relationships for data quality and scalability).

## Nice-to-Haves

- An experiment measuring data quality when using detected (rather than ground-truth) bounding boxes, even on a small subset, would directly address the practical deployment gap.
- Comparison with SpaRE's data in RQ3, even on a subset of benchmarks, would substantially strengthen the SOTA claim.
- Analysis of how often 2D spatial relations (left/right/above/below) disagree with 3D ground truth in the driving datasets would inform whether this is a meaningful limitation or a negligible concern for the target domains.
- Categorization and presentation of the 28 flagged invalid/confusing instances from GRAID's human evaluation would provide valuable qualitative error analysis.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"2D approach conflates image-plane relations with 3D spatial reasoning" (Harsh Critic point #1)**: The paper explicitly frames its contribution as generating *qualitative* spatial relationships from 2D geometry — this is the stated design choice, not an oversight. The paper's title and claims are about "spatial reasoning of VLMs" and "qualitative spatial relationships," and the RQ3 benchmarks (BLINK, VSR) test 2D spatial reasoning as commonly understood. This is a scope limitation worth noting (see Nice-to-Haves), not a fatal flaw. The driving scene images are mostly from forward-facing cameras where 2D spatial relations are a reasonable proxy for 3D, and the paper deliberately chooses qualitative rather than metric spatial queries.

- **"Headline comparison not apples-to-apples" (Harsh Critic point #2)**: While the metrics differ (combined validity for GRAID vs. separate metrics for SpatialVLM), this actually understates rather than overstates the gap. Computing a combined metric for SpatialVLM would yield an even lower rate. The concern about the community implementation is addressed by the paper's own acknowledgment and by corroboration with recent independent findings (Ogezi & Shi, 2025).

- **"Missing comparisons against SpaRE/SpatialRGPT" (Harsh Critic structural concern)**: Partially retained as Major weakness above — the comparison against only OpenSpaces is a gap. However, the SpatialRGPT evaluation limitation is genuine (masked regions), and SpaRE requires captions that don't exist for these datasets, making direct comparison difficult. Downgraded from the harsh critic's framing that this makes "the advantage over SOTA unsubstantiated" to Major rather than Fatal.

- **"Depth questions criticize depth models while using them"**: The paper addresses this by making depth questions qualitative (which is closer?) and configurable with thresholds, which is a reasonable mitigation. This is more of a scope limitation than a contradiction.

- **Formatting and notation nitpicks**: Removed per hard rules.

## Novel Insights

The most interesting insight is how the "2D-only" constraint, while seemingly limiting, may actually be a feature rather than a bug: by generating spatial Q&A pairs that are qualitatively determinable from 2D geometry, GRAID avoids the cascading error accumulation that plagues 3D-reconstruction-based and LLM-hallucination-based approaches. The SPARQ predicate-sieve design (shared lightweight predicates that enable early rejection) is an underappreciated engineering contribution that makes the 8.5M-scale generation tractable. The RQ2 finding that training on 6 spatial primitives transfers to 10+ held-out types aligns with emerging evidence that compositional spatial understanding can emerge from simple primitives, though the lack of a non-spatial control weakens how definitively this can be claimed.

## Suggestions

- **Add a non-spatial VQA control**: Fine-tune on a comparable amount of generic VQA data (e.g., from VQAv2 or GQA) and show that the spatial improvements from GRAID exceed what any VQA fine-tuning would provide. This single experiment would resolve the transferability question.

- **Include the "similar planes" check in Algorithm 1** (or explicitly note it is deferred to the appendix/additional templates) to resolve the prose-code inconsistency.

- **Report detector-based generation results**: Even a small-scale experiment with YOLO/Detectron2 detections rather than GT annotations would show how performance degrades in real deployment and strengthen the practical claims.

## Score and Decision

Comparing against calibration anchors:

| Anchor | Avg Score | Relation to GRAID |
|---|---|---|
| EQA-MX (8M+ QA, novel architecture, embodied QA) | 8.0 | GRAID is less novel (no new architecture) but has comparable scale and strong empirical results |
| SPACE (spatial cognition benchmark) | 6.75 | Similar spatial reasoning domain; SPACE is more novel (cognitive science grounding) but GRAID has a practical generation framework |
| Sparkle (spatial reasoning from synthetic data, limited scope) | 4.5 | GRAID is substantially more complete — real images, larger scale, more benchmarks, multiple backbones |
| Euclid (synthetic geometric data, overclaimed) | 5.0 | GRAID has similar concerns about claimed transfer but much stronger empirical backing |
| RelationVLM (overclaimed generalization, limited comparisons) | 5.0 | Similar pattern — limited baselines but GRAID has more thorough evaluation |
| 3D Spatial VLM reasoning (limited evaluation, 2D scope) | 4.0 | GRAID is far more comprehensive and practical |

GRAID is clearly above the 4-5 range (more substantial than Sparkle, Euclid, RelationVLM) and below the 7-8 range (less architecturally novel than EQA-MX, less conceptually novel than SPACE). The main limitations are the single-weak-baseline comparison and the prose-algorithm inconsistency, but the core contribution — a simple, scalable, high-quality data generation framework with strong empirical evidence of transfer — is genuine and useful.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>