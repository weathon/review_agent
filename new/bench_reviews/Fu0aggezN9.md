Now I have a thorough understanding of the paper. Let me construct the final consolidated review.

## Summary

The paper introduces the graph-based Document Structure Analysis (gDSA) task, which extends traditional document layout analysis by requiring models to predict both document element locations and their structural relations (spatial and logical) in the form of a graph. To support this task, the authors construct the GraphDoc dataset, built on top of DocLayNet, with 80K document images and 4.13M relation annotations across 8 relation categories. They also propose the Document Relation Graph Generator (DRGG), a plug-and-play relation prediction module that attaches to standard encoder-decoder object detectors, achieving 57.6% mAP_g@0.5 as a baseline on the new benchmark.

## Strengths

- **Novel and comprehensive task formulation**: The gDSA task unifies document layout analysis with structural relation prediction (spatial + logical) into a single graph-generation problem. Table 1 demonstrates that GraphDoc is the only dataset combining all six modalities (V, T, L, O, H, G) and supporting non-textual instances for graph structure analysis, filling a genuine gap in existing benchmarks.

- **Well-designed evaluation metrics (mR_g, mAP_g)**: The paper identifies that standard SGG top-k metrics are inadequate for documents (class imbalance, variable relation counts, multiple coexisting relations) and proposes threshold-based metrics that require correct instance detection before evaluating relations (Algorithm 1). This is a sensible and principled design choice.

- **Dataset scale and diversity**: GraphDoc provides 4.13M relation annotations across 8 categories on 80K documents, providing sufficient training data. Building on DocLayNet ensures reliable base layout annotations and diverse document types (financial reports, manuals, scientific papers, legal documents).

- **Plug-and-play DRGG architecture**: DRGG integrates with multiple existing detectors (DETR, Deformable DETR, DINO, RoDLA) without modifying the base architecture (Figure 5, Section 3.2), and Table 2 demonstrates this versatility across four detectors.

## Weaknesses

### Fatal
None.

### Major

- **Unexplained extreme asymmetries in Table 3 undermine confidence in the evaluation**: For Deformable DETR, "Left" achieves 99.0% AP while "Right" achieves only 11.9%. For Swin+RoDLA, the asymmetry flips: "Left" is 33.7% while "Right" is 99.0%. Left and Right are directional counterparts on a plane; such extreme, model-dependent asymmetries are inexplicable without a bug in evaluation code, a systematic labeling bias, or some other artifact. The paper provides no explanation or even acknowledgment of these anomalies. Since spatial relations constitute 64% of all annotations, any systematic error in their evaluation invalidates a large portion of the claimed results. This must be resolved before the benchmark results can be trusted.

- **No rule-based baseline for gDSA establishes whether the task is non-trivial**: The annotations are generated entirely by heuristic rules (Section 3.1.4): spatial relations by nearest-neighbor pixel scanning, reading order by Recursive X-Y Cut, hierarchical structure by category-based heuristics, reference by text matching. Since 64% of relations are spatial (deterministic geometric computations), a rule-based baseline applying these same heuristics would likely match or exceed the model on spatial relations. Without such a baseline, it is impossible to assess whether DRGG learns anything beyond replicating the annotation rules, and the headline 57.6% mAP_g@0.5 cannot be interpreted meaningfully. This is essential for a dataset/benchmark paper.

- **Annotation quality is unquantified**: The paper states "most of the results have been manually verified and refined" (Section 3.1.4) but provides no inter-annotator agreement scores, no sampling methodology, no characterization of what fraction "most" represents, and no quality metrics for the verification process. For a dataset/benchmark paper where annotation quality is the primary contribution, this is a critical gap. The reader has no way to assess whether the rule-generated annotations are reliable enough to serve as ground truth.

### Minor

- **Headline mAP_g@0.5 is dominated by near-trivial spatial relations**: The best model achieves 99.0% AP on Left/Right spatial relations but only 16.8–18.8% AP on Reference relations (arguably the most interesting and practically important relation type). The aggregate 57.6% mAP_g@0.5 thus misrepresents the genuine difficulty of the task. The paper should decompose results into spatial-only and logical-only components to give readers an accurate picture of where the challenge lies.

- **DRGG's claimed DLA improvement is marginal and untested for significance**: Adding DRGG to RoDLA with InternImage improves DLA mAP from 80.5% to 81.5% (Table 2)—a 1.0% gain within typical random variation, with no variance reported across runs. The paper should be more cautious about claiming DRGG "improves" DLA without establishing statistical significance.

- **Fundamental mismatch between annotation method and model capability for Reference relations**: Reference relations are constructed by matching annotation text (Section 3.1.4: "We match annotation texts to construct reference relation"), yet the model operates purely on visual input without text access. The 16.8% AP on Reference confirms this is a core limitation, not a minor one. The paper acknowledges the vision-only limitation but underplays its impact—the most distinctive relation type in the benchmark is largely undetectable by the proposed model.

### Trivial
None.

## Nice-to-Haves

- A downstream task evaluation showing that predicted document graphs improve some application (e.g., information extraction, document QA) would substantiate the claim that gDSA enables "deeper comprehension."
- A text-based or multimodal variant of DRGG would directly test whether the identified limitation on Reference relations is addressable.
- Cross-page relation annotations would increase the practical utility of the dataset, as cross-page references are common in multi-page documents.

## Removed Points

*These points were flagged for removal—treat them with caution:*

- **Up/Down minor asymmetries (Harsh Critic #2)**: The reviewer flagged ResNet Up=15.1%/Down=17.2% and Swin Up=18.8%/Down=19.8% as "unexplained inconsistencies." These are 1–2% differences between directional counterparts and are well within normal variation—not anomalous. Removed as a weakness.

- **No comparison to adapted SGG/document relation extraction methods (Harsh Critic #3 partial)**: The reviewer demanded comparisons to HRDoc's hierarchical approach or neural motif models. For a new task/dataset paper, the absence of directly comparable external methods is somewhat expected. The rule-based baseline gap (kept above) is the critical missing comparison; the absence of adapted SGG methods is a lesser concern.

- **DRGG architecture similarity to SGG methods (Harsh Critic Section-by-Section)**: The reviewer claimed the paper doesn't acknowledge the connection to scene graph generation. In fact, the paper discusses SGG in the related work (Section 2), evaluation metrics (Section 3.3), and explicitly states "Inspired by the graph generation from computer vision...we propose a graph-based task for document analysis" (Section 2). While deeper architectural comparison would strengthen the paper, the claim of no acknowledgment is overstated.

- **Missing downstream task evaluation (Harsh Critic)**: While valuable, the paper's stated scope is task/dataset formulation and baseline establishment—not demonstrating downstream utility. Moved to nice-to-have.

- **Missing text/multimodal variant (Harsh Critic)**: The paper explicitly scopes to vision-only and acknowledges this as a limitation. Moved to nice-to-have.

- **Cross-page relation limitation (Harsh Critic)**: The paper explicitly acknowledges this limitation (Section 5). This is not an undisclosed weakness.

- **Reproducibility concerns about hyperparameters, T_R thresholds (Harsh Critic)**: The paper provides implementation details and references Appendix F. Demands for justifying specific threshold choices and calibration analysis are standard nitpicks.

## Novel Insights

The Table 3 Left/Right asymmetry pattern—where different models fail on opposite directions—is more informative than simply noting the asymmetry exists. The fact that InternImage+RoDLA and ResNeXt+RoDLA achieve symmetric near-perfect Left/Right scores, while Deformable DETR and Swin+RoDLA show extreme opposite asymmetries, suggests this is not a dataset bias (which would affect all models similarly) but rather a model-specific convergence or evaluation artifact. This pattern could indicate that certain model architectures become trapped in suboptimal local minima that favor one spatial direction, or that there is an interaction between the model architecture and the evaluation pipeline that differentially affects directional predictions.

## Suggestions

- **Add a rule-based baseline**: Implement the same heuristics used in the annotation pipeline (nearest-neighbor spatial extraction, X-Y Cut reading order, category-based parent/child assignment) as an evaluation baseline. Report its performance alongside DRGG to establish the non-triviality of the gDSA task.

- **Explain the Table 3 asymmetries**: Investigate and report why Deformable DETR achieves 99.0% Left / 11.9% Right while Swin achieves 33.7% Left / 99.0% Right. At minimum, report the per-category distribution of relations in the dataset to rule out labeling bias, and verify the evaluation code produces symmetric results on synthetic symmetric data.

- **Quantify annotation quality**: Report inter-annotator agreement on a sample, and characterize what fraction of rule-generated annotations survive manual verification unchanged. Even a small-sample study (e.g., 100–200 documents) would meaningfully address this gap.

- **Decompose aggregate metrics**: Report mAP_g separately for spatial and logical relations so readers can interpret results without the distortion from near-trivial spatial predictions.

<context>
**Paper summary**: The paper introduces the gDSA task (graph-based Document Structure Analysis), which extends document layout analysis by requiring models to predict both element locations and their structural relations as a graph. It constructs GraphDoc, a dataset built on DocLayNet with 80K images and 4.13M relation annotations across 8 categories (4 spatial: Up/Down/Left/Right; 4 logical: Parent/Child/Sequence/Reference). Annotations are generated via heuristic rules (nearest-neighbor pixel scanning for spatial, Recursive X-Y Cut for reading order, category heuristics for hierarchy, text matching for references) with claimed manual verification. The proposed DRGG module attaches to encoder-decoder detectors and predicts pairwise relations between object queries, achieving 57.6% mAP_g@0.5 as a baseline.

**Original reviewer signal**: Harsh Critic found fundamental problems (rule-based annotations trivialize the benchmark, Table 3 asymmetries suggest bugs, no external baselines). Strength Finder emphasized the novel task formulation, dataset comprehensiveness, and well-designed metrics. Direct disagreement on whether the paper provides sufficient evidence for its claims.

**What was dropped and why**: (1) Up/Down minor differences (1-2%) flagged as "inconsistencies"—within normal variation, not anomalous. (2) Demands for SGG-method comparisons—paper does reference SGG in related work; for a new task, external method comparisons are less critical than the rule-based baseline. (3) Missing downstream evaluation—outside the paper's stated scope of task/dataset formulation. (4) Multimodal variant demand—paper explicitly scopes to vision-only and acknowledges the limitation. (5) Cross-page limitation—explicitly acknowledged by authors. (6) Reproducibility/threshold nitpicks—standard nitpicks not substantive weaknesses.

**Cross-checks performed**: (1) Verified Table 3 asymmetries directly: Deformable DETR Left=99.0/Right=11.9 and Swin Left=33.7/Right=99.0 confirmed in paper text. These are genuinely unexplained and anomalous. (2) Verified "most of the results have been manually verified and refined" claim—appears only once (line 147) with no quantification. (3) Verified annotation pipeline is entirely rule-based (Section 3.1.4). (4) Verified Reference relations built by text matching while model is vision-only—confirmed mismatch. (5) Verified paper does discuss SGG in related work and evaluation metrics sections—reviewer's claim of no acknowledgment was overstated.

**Review construction notes**: The three Major weaknesses (Table 3 asymmetries, no rule-based baseline, unquantified annotation quality) are independent and all bear on the core question of whether this benchmark is reliable and meaningful. The Table 3 anomaly is particularly concerning because the asymmetry flips direction by model, suggesting something beyond dataset bias. The rule-based baseline absence is critical because 64% of annotations are deterministic geometric computations. Annotation quality is standard table-stakes for dataset papers. Minor weaknesses note that the headline number is misleading and the DLA improvement claim is marginal.
</context>