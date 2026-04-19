## Summary
The paper introduces Learning Embedding Linear Projections (LELP), a knowledge distillation method that extracts pseudo-subclasses from teacher embeddings via PCA-based linear projections to improve student performance on binary and few-class tasks. The method modifies the student to predict an expanded output dimension ($S \times C$ classes), effectively transferring embedding-structure information without requiring teacher retraining. The approach demonstrates strong empirical results across multiple large-scale NLP benchmarks, including cases where the student outperforms a teacher with 20x more parameters.

## Strengths
- **Practical advantage over prior subclass-based methods:** The paper convincingly demonstrates that LELP achieves performance on par with or exceeding Subclass Distillation while eliminating the need for teacher retraining, which is a significant efficiency gain for large teacher models where iterative hyperparameter tuning is prohibitive.
- **Strong empirical gains on NLP tasks:** On several large-scale sentiment and review datasets (e.g., Amazon Reviews, Sentiment140), LELP provides meaningful improvements over baselines; notably, the ALBERT-Base student trained with LELP outperforms the ALBERT-XXL teacher, confirming the method successfully captures embedding information beyond logit-level knowledge.
- **Robust and architecture-agnostic design:** The method handles mismatched embedding dimensions and diverse architecture families (e.g., ResNet to MobileNet, ALBERT to MLP over frozen T5) without learnable projection layers, simplifying the training pipeline compared to embedding-regression methods.
- **Validated design via Oracle Clustering:** The synthetic CIFAR experiments and Oracle Clustering baseline effectively demonstrate that subclass-aware distillation can, in principle, provide gains, and that LELP's linear projection approach is a strong practical approximation of this upper bound.

## Weaknesses

### Fatal
// None identified.

### Major
- **Experimental protocol ($\alpha=0$) limits relevance to standard supervised distillation:** All main comparisons are conducted with $\alpha=0$, meaning the student is trained purely on teacher targets without any ground-truth label supervision. While the paper motivates this with semi-supervised settings and claims to isolate the distillation loss, the core claim targets "few-class problems" in realistic applications like sentiment analysis, which are almost always solved with standard supervised fine-tuning ($\alpha > 0$). In supervised KD, the hard label term often dominates or interacts non-trivially with the distillation term; performance in the $\alpha=0$ regime does not necessarily translate to superior results when labels are available. The paper's conclusions are therefore scoped to pure/immitation distillation, but the framing suggests broader applicability to supervised few-class KD.
- **Comparison to Subclass Distillation is confounded by teacher accuracy differences:** The paper acknowledges that Subclass Distillation requires retraining the teacher, which results in different teacher accuracies compared to the fixed teacher used for LELP. However, the paper still uses these mixed-teacher comparisons to claim LELP "exceeds or performs as well as" Subclass Distillation. Since student performance is highly sensitive to teacher quality, confounding the student method with the teacher retraining effect makes it impossible to attribute gains purely to the LELP mechanism. The computational efficiency argument is valid, but the accuracy superiority claim is not cleanly established.

### Minor
- **Mechanistic motivation does not align with NLP task properties:** The method is motivated by Neural Collapse and hidden subclass structures (e.g., recovering fine-grained classes from coarse labels), which is validated on binarized CIFAR tasks. However, the headline NLP results are explicitly on datasets "without subclass structure." The paper does not analyze what the PCA projections capture in these settings; it remains unclear whether the linear projections are discovering task-relevant variations (e.g., sentiment intensity, topical subgroups) or merely fitting to noise. While the method works empirically, the explanatory narrative is disconnected from the primary evaluation domain.
- **Marginal gains and inconsistent averaging in Table 2:** On several tasks, margins are small (e.g., QGLUEval is 81.43 vs 80.85 with standard deviations that approach overlap), yet the language claims consistent superiority. Furthermore, the "Avg. gain over the best baseline" rows in Table 2 (values like +0.02, +0.04) appear inconsistent with the raw differences (e.g., 90.22 vs 89.24 is a ~1.0 gap, not 0.02), which creates confusion about how gains are being aggregated or reported.

### Trivial
- Baseline naming discrepancies between Section 4.1 and Table 2 (e.g., "Feature," "VD," "Retained KD" vs. FitNet, VID, Relational KD) could be clarified in the main text.

## Nice-to-Haves
- **Evaluation in the standard $\alpha > 0$ supervised regime:** Running a subset of experiments with $\alpha > 0$ would significantly strengthen the paper's relevance to practical few-class KD and clarify whether LELP's gains persist when hard labels are present.
- **Ablation of null-space projection and random rotation on NLP tasks:** These design choices are shown to help on vision tasks in the appendix; demonstrating their impact on the main NLP datasets would bolster confidence in their general utility.
- **Analysis of pseudo-subclass semantics:** Visualizations or case studies of high-probability subclasses on text datasets would help interpret what LELP is actually learning beyond raw accuracy gains.

## Removed Points
These points are flagged to be removed, treat them with caution:
- *Critic's claim of "structural misalignment" making the paper "not about KD":* The paper explicitly discusses semi-supervised distillation and isolates the distillation loss; this is a valid experimental choice, even if it narrows the scope. It is not a fatal flaw.
- *Critic's demand for baseline adaptation (CRD/DKD for text):* The paper already discusses why CRD is limited for text (augmentation difficulty) and why DKD collapses to Vanilla KD for binary tasks, which is a standard justification in this domain.
- *Critic's request for significance tests:* For a paper with large datasets and consistent directional wins, lack of formal t-tests is a minor presentation issue, not a validity threat. Results with large margins (e.g., +1.0) are practically significant.
- *Strengths about "principled design" or "Unified single-loss":* These are descriptive of the method rather than evidence-backed strengths. The unified loss is a byproduct of the subclass formulation, not a distinct contribution.

## Novel Insights
The paper offers a pragmatic insight for few-class distillation: when logit-level information is sparse due to low cardinality, recovering structure from the teacher's embedding space (via PCA-derived pseudo-subclasses) can effectively increase the information bandwidth of each training example. The finding that a student can outperform a 20x larger teacher by learning from embedding geometry rather than just output logits suggests that few-class performance bottlenecks may stem from information compression in the logit layer rather than capacity limits in the student.

## Suggestions
- **Scope clarification:** Explicitly frame contributions around "label-free" or "pure distillation" settings in the abstract and introduction, and temper claims about general supervised few-class KD.
- **Table 2 calculation correction:** Verify the "Avg. gain" rows in Table 2 for arithmetic consistency with the raw numbers, or clarify the aggregation method.
- **Fairer comparison:** Include a matched-teacher baseline for Subclass Distillation where possible, or clearly separate the "compute cost" benefit from the "accuracy" benefit in the discussion to avoid confounding.

## Score and Decision
I calibrated this paper against several KD anchors:
- Compared to **MiniPLM** (Scored 6), which also proposed a practical KD framework but was noted for limited cross-architecture evaluation; LELP demonstrates broader architectural flexibility and stronger empirical wins (beating the teacher) but has a narrower experimental scope due to $\alpha=0$.
- Compared to **CKA Hidden State Matching** (Scores 6, 8, 3), which introduced a mathematically novel similarity metric for distillation; LELP is less theoretical but more empirically robust in its domain.
- The paper is clearly superior to low-scoring anchors (incremental work, missing baselines) and does not reach the depth/novelty of high-scoring (8+) papers that introduce new paradigms or comprehensive scaling studies.

The method is simple and effective, and the practical advantage (no teacher retraining) is substantial. However, the $\alpha=0$ limitation is a major scope mismatch for a paper claiming to improve "few-class distillation" in general, and the confounded comparison to the closest competitor weakens the empirical narrative. These issues prevent a higher score but do not invalidate the contribution.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>