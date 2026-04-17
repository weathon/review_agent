## Summary

This paper introduces GraphDoc, a graph-based document structure analysis (gDSA) task and dataset that extends DocLayNet (80K images, 1.10M instances) with 4.13M relation annotations across 8 relation types (4 spatial: up/down/left/right; 4 logical: parent/child/sequence/reference). The authors also propose DRGG (Document Relation Graph Generator), a plug-and-play relation prediction module that achieves 57.6% mAP_g@0.5 on the gDSA task when combined with RoDLA and InternImage backbone.

## Strengths

- **Novel and well-motivated task formulation.** The gDSA task meaningfully extends traditional DLA by unifying layout detection, reading order, hierarchy, and reference prediction into a single graph prediction problem. Table 1 clearly shows the gap: no existing dataset combines spatial and logical graph relations with both textual and non-textual elements at paragraph level.

- **Large-scale annotation effort.** Adding 4.13M relation annotations across 8 categories to DocLayNet is a substantial effort, and the separation into spatial and logical relation types provides interpretable and fine-grained structural information.

- **Comprehensive per-category evaluation.** Table 3 provides per-relation-category breakdowns that reveal important performance gaps (e.g., Left/Right at ~99% vs. Reference at 16.8%), clearly identifying where future work is needed.

- **New evaluation metrics designed for gDSA.** The threshold-based mR_g and mAP_g metrics (Algorithm 1) address legitimate issues with standard SGG metrics: variable relation counts per document, severe class imbalance, and multiple coexisting relations per element pair.

## Weaknesses

### Major

1. **Heuristic annotation pipeline lacks quantitative quality validation.** The entire 4.13M-edge dataset is produced by a rule-based system (Sec. 3.1.4): spatial relations by nearest-neighbor geometry, reading order via Recursive X-Y Cut, hierarchy via category-based rules, references via string matching. The only quality assurance mentioned is that "most of the results have been manually verified and refined"—with no quantification: no inter-annotator agreement, no sampled accuracy rates, no error breakdowns by relation type. This is a critical gap for a dataset paper: the training signal models receive encodes these heuristics, not necessarily human-judged document structure. Without a validated human-truth subset, we cannot assess whether the dataset measures genuine structural understanding or merely the learnability of the authors' geometric rules.

2. ~~**Near-perfect scores on spatial relations undermine the headline mAP_g metric.** Table 3 shows 99.0% AP for Left/Right relations across multiple backbones—trivially learnable from bounding box geometry since these edges are defined as "nearest adjacent bounding box in each direction." Since spatial relations constitute 64.06% of all edges, the aggregate mAP_g@0.5 of 57.6% is heavily inflated by these geometrically trivial predictions. The paper does not report logical-relation-only metrics or discuss how class imbalance affects interpretability of the headline number. This directly impacts the credibility of the "strong baseline" claim.~~ **Edit: On reflection, the near-perfect Left/Right scores are a feature of the task design—the relations ARE defined geometrically, and confirming that the model can learn them is expected. The more important concern is that the aggregate metric obscures the substantial difficulty gap between spatial and logical relations. This is an interpretability concern rather than a fatal flaw, but it remains significant.**

3. **No ablations or simple baselines for the DRGG architecture.** Every gDSA result in Table 2 uses DRGG; there is no comparison against simpler relation-prediction alternatives (e.g., a pairwise MLP on concatenated box features, or an adapted SGG method). There are also no ablations on DRGG's design: pooling+upsampling branches vs. direct features, single-layer vs. multi-layer aggregation, weighted vs. unweighted aggregation. Without these, it is impossible to determine whether DRGG's specific architecture is necessary or whether *any* relation head on top of a strong detector would achieve similar results. This is an evidential gap for a method paper.

4. **Evaluation metrics are under-specified.** Algorithm 1 defines instance matching and relation filtering, but the actual computation of mR_g and mAP_g is left to opaque functions f_mR and f_mAP. Key details are missing: how multi-label relations per pair are handled in AP computation; whether APs are per-class then averaged; how class imbalance across the 8 relation types is addressed. For a *new benchmark* proposing *new metrics*, complete specification is essential for reproducibility and meaningful comparison.

5. **Vision-only model evaluated on text-dependent relations.** DRGG operates purely on visual features (acknowledged in Limitations), yet the task includes *reference* relations (e.g., "see Figure 3") and *sequence* relations (reading order) that fundamentally require textual understanding. The 16.8% AP on references confirms this mismatch. This is not just a limitation—it raises questions about whether the right inductive biases are present for the more semantically meaningful relations in the dataset.

### Minor

- **No DLA-only vs. joint training comparison.** The paper jointly trains DLA and gDSA but never reports whether this multi-task setup helps or hurts DLA compared to the InternImage+RoDLA baseline (80.5% mAP without DRGG vs. 81.5% with DRGG—only a 1% gain). The DLA improvement from DRGG is marginal.

- **Single-page scope limitation.** Restricting relations to within-page excludes cross-page references and sequences common in real documents, limiting practical applicability. The paper acknowledges this but the dataset's utility for multi-page document understanding is inherently limited.

- **No qualitative examples of predicted graphs.** For a graph generation task, visualizing predicted vs. ground-truth graphs (especially for logical relations) is essential for understanding failure modes. The paper includes only aggregate metrics.

### Trivial

- The gDSA loss in Eq. 2 writes node classification loss inside the edge summation, which is a minor notational oddity but does not affect interpretation.

## Nice-to-Haves

- A small human-annotated subset with inter-annotator agreement statistics to validate the rule-based labels.
- An oracle experiment feeding ground-truth bounding boxes to DRGG to isolate relation prediction from detection errors.
- Separate metrics for spatial vs. logical relations to give a clearer picture of difficulty.
- A simple MLP baseline on concatenated bounding-box geometry features to establish whether DRGG's architecture provides meaningful benefit over trivial alternatives.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that no comparison with SGG methods is required**: The harsh critic demanded comparison with Neural Motifs, MSDN, etc. While such comparisons would strengthen the paper, this paper is about document structure analysis, not scene graph generation. Demanding adapted SGG baselines as a condition for acceptance goes beyond reasonable scope. *However*, comparing against any simple baseline pair predictor would be appropriate and is kept as a major weakness above.

- **Claim that the gDSA loss equation is "misleading"**: The notation in Eq. 2 includes classification loss inside the edge sum, but this is a minor mathematical presentation issue, not a substantive error. Removed as formatting nitpick.

- **Claim about "error amplification across pipeline steps"**: While plausible, this is speculation without evidence. The paper does not claim the pipeline is error-free, and a responsible reviewer should flag the absence of error analysis, not invent failure modes.

- **Demand for confidence intervals / statistical variation**: Standard practice for large-scale detection benchmarks is single-run evaluation; requesting confidence intervals is a nice-to-have, not a core flaw.

- **Criticizing the paper for not evaluating on HRDoc/ReadingBank**: These datasets have different annotation formats (line-level, text-only) and different task definitions; direct comparison is not straightforward and is outside the paper's scope.

## Novel Insights

The most revealing insight from the evaluation is the stark difficulty gradient between spatial and logical relations: Left/Right at ~99% AP confirms these are geometrically trivial, while Reference at 16.8% AP reveals that the most semantically meaningful relations remain essentially unsolved by vision-only models. This suggests that the gDSA task as currently formulated may decompose into two sub-problems of vastly different character—a geometric adjacency problem that is largely solved, and a semantic reasoning problem that requires textual understanding. Future work on this benchmark should likely decouple these and prioritize the semantic component.

## Suggestions

1. **Add a validated human-annotated subset.** Even 500–1000 pages with dual-annotator ground truth would transform the dataset's credibility. Report Cohen's kappa or agreement rates per relation type.
2. **Report logical-relation-only metrics separately.** This would give a clearer picture of the genuinely challenging part of the task and prevent the aggregate metric from being dominated by trivially easy spatial edges.
3. **Add a trivial geometric baseline.** A simple rule predicting "Left" / "Right" from bounding box coordinates would establish what portion of performance comes from geometry rather than learned features.
4. **Specify mR_g and mAP_g computation completely.** For a benchmark paper, the metric definition must be fully reproducible, including multi-label handling and per-class aggregation.

## Evaluation

**Originality:** The gDSA task formulation is a genuine and meaningful contribution—unifying spatial and logical relations into a graph structure for documents fills a real gap. The dataset construction, while derived from DocLayNet, adds substantial new annotation. DRGG's architecture, however, is incremental (standard encoder-decoder features + pairwise prediction head).

**Importance of research question:** Document structure understanding beyond bounding-box detection is important and timely. The graph-based framing could catalyze new research directions.

**Claim support:** The headline mAP_g@0.5 of 57.6% is difficult to interpret due to class imbalance, trivial spatial relations, and under-specified metrics. No ablations or simple baselines are provided for DRGG.

**Soundness of experiments:** Insufficient. Missing ablations, no annotation quality validation, no oracle DLA experiment, under-specified metrics.

**Clarity:** Generally well-written and clear. Task definitions and dataset statistics are well-presented.

**Value to community:** The dataset has potential value, but its current form—rule-based annotations without quality assessment and under-specified metrics—limits its immediate utility as a benchmark.

**Calibration:** Compared to DocGenome (CI9JMBAsPg, rejected, avg 6.25)—which also had auto-labeled data with limited novelty and questions about annotation quality—this paper faces similar concerns but has a more focused task definition. Compared to ADOPD (x1ptaXpOYa, accepted poster, avg 6.5)—which featured human-in-the-loop annotation and thorough quality validation—this paper's annotation quality is substantially weaker. The paper sits in a similar quality range to DocGenome: meaningful effort and interesting framing, but with significant validation gaps.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>