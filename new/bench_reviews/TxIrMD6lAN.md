Now let me search for calibration anchors.Now I have enough information to write the final review. Let me compile it.

---

## Summary

This paper proposes augmenting classical regularization-based incremental learning (IL) methods with task-specific bottleneck adapter modules co-trained alongside the backbone. The core idea is to decompose representation learning into invariant features (backbone) and task-specific features (adapters), and to integrate this design with both weight-regularized methods (EWC, MAS, PathInt — by excluding adapter parameters from penalties) and prediction-regularized methods (LwF, LwM — by adding an auxiliary backbone distillation loss). Experiments on CIFAR-100 task-IL across five base methods and multiple task orderings show consistent ~3–5% improvements; ImageNet results are mixed.

---

## Strengths

- **Broad empirical compatibility across two families of regularization methods (§3.2.1)**: The paper genuinely distinguishes the integration strategies for weight-regularized (excluding adapter params from Fisher penalties) vs. prediction-regularized (adding backbone-level KD loss $R_\varphi^t$, Eq. 1) methods. This is a concrete, method-aware design, not a one-size-fits-all hack. Gains of ~3% (weight-regularized) and ~5% (prediction-regularized) across all five baselines on CIFAR-100 (Figure 3) are consistently shown over all 10 tasks.

- **Robust evaluation across task orderings and scales**: Figure 5 shows the adapter advantage holds under the harder coarse-grained ordering (higher inter-task diversity) and the iCaRL random ordering. Figure 4 confirms advantages at 5, 10, and 20 classes per task. Running 10 random seeds and reporting full learning curves (not just final accuracy) is good practice.

- **Ablation of co-training vs. frozen backbone (Table 2)**: The comparison between co-trained LwF-A (74.0%) and frozen-backbone LwF-A-FrB (72.9%) directly validates the co-training design choice over prior frozen-backbone adapter approaches.

---

## Weaknesses

### Fatal
None.

### Major

- **Headline claim of "consistent outperformance" is directly contradicted by Table 1**: Verified against Table 1: at task 10, LwF-A (67.2%) < LwF (68.2%), and LwM-A (56.9%) < LwM (58.0%). Two of the five methods — specifically the prediction-regularized ones, for which the most elaborate integration (Eq. 1 with $R_\varphi^t$) was designed — actually *regress* on ImageNet. The paper states "methods with adapters yield the best performance across all incremental tasks" (§4.2) — this is factually incorrect by the paper's own table. The paper acknowledges hyperparameters were not re-tuned for ImageNet, but this makes the situation worse: it shows the proposed benefit is brittle to hyperparameter transfer precisely for the methods requiring the most specialized design effort. This does not invalidate the CIFAR-100 contribution, but it does undermine the broader generalizability claim.

- **The claim to "eliminate the stability-plasticity dilemma" is not supported by results**: The abstract, §3.2, and conclusion all claim the method "eliminates" the dilemma. However, Figure 3 shows all adapter variants still exhibit monotonically declining accuracy curves over tasks — significant forgetting remains in all cases. The paper demonstrates meaningful *mitigation* (~3–5%), not elimination. Overclaiming a qualitative resolution of a fundamental open problem without any supporting evidence is a substantive error that will undermine credibility with readers.

- **Core mechanistic claim — backbone learns invariant features, adapters learn task-specific features — is never empirically verified**: The entire theoretical justification for the design (§3.2, §3.2.1, Conclusion) rests on this decomposition. Yet there is no analysis of backbone feature similarity across tasks (e.g., CKA, t-SNE), no measurement of how much task-identifying information resides in adapters vs. the backbone, and no diagnostic experiment demonstrating the claimed specialization. Without such evidence, the approach could simply be adding model capacity and an output regularization term, with the functional decomposition as a post-hoc narrative.

### Minor

- **Main evaluation confined to task-IL with oracle task-ID**: The paper correctly notes class-IL is "more practical yet challenging" but places all class-IL results in the appendix, focusing the entire main body on the easier task-IL setting where task identity is provided at inference. For adapter methods, this sidesteps a fundamental question: how would adapters be selected/combined at inference without a task oracle? This scoping decision weakens practical relevance.

- **Bottleneck width inconsistency**: Figure 6 shows width 256 consistently performs best among widths tested (16–256) for both EWC and LwF. However, Table 1's caption states width 128 was used for ImageNet, with no justification for the deviation. This inconsistency affects the reported ImageNet results.

- **Diminishing returns with larger task granularity unexplained**: §4.2 notes that "the benefits of utilizing adapters diminish as the number of classes increases within each task" and dismisses this as "understandable." However, the regime of fewer tasks with more classes per task is arguably more practical, and the paper offers no analysis of why the method's advantage degrades under this condition.

### Trivial

- **TAMiL comparison uses the authors' cherry-picked best configuration** (Table 2 footnote: "the best method-adapter pair we yielded"): while this is disclosed, it means the comparison is not head-to-head under matched conditions. The margin (74.7 vs. 71.4) should be interpreted cautiously.

---

## Nice-to-Haves

- Verification of the backbone/adapter functional decomposition using feature-space analysis (e.g., CKA between task representations before and after training), which would either substantiate or refute the core conceptual claim.
- A controlled parameter-count baseline to disentangle capacity effects from the structural design: at bottleneck width 256 on ResNet-34, 10 tasks add ~2.6M parameters (≈12% overhead), which is not negligible.
- Per-task accuracy matrices rather than only the aggregate $A_t$ metric, to cleanly separate the stability and plasticity contributions claimed to both improve.
- Adapter selection strategy for class-IL deployment, even if only discussed as future work.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic's claim about adapter placement being "significantly different from Houlsby et al."**: The paper is transparent about where adapters are placed (after the backbone feature extractor, before the label predictor, §3.2). This is a design choice that is clearly described and motivated; it is not a weakness.

- **Harsh Critic's claim that co-training showing only 1.1% advantage implies gains come from capacity not mechanism**: The 74.0% vs. 72.9% comparison does support the design choice. The inference that this "reveals" the gains are from capacity alone is speculative and unfair — both components may contribute.

- **Strength Finder's claim that Figure 1 "provides direct evidence that inter-task diversity causes more forgetting"**: Figure 1 shows correlation between task ordering (coarse-grained vs. alphabetical) and forgetting, not causation. This is a reasonable motivation but overstated as "direct evidence." Removed as a generic/overstated strength.

- **Strength Finder's claim about "lightweight parameter overhead" as a strength**: The paper does not report actual parameter counts in the main text, and as noted above the overhead is not trivially negligible at larger widths. Removed as unverified.

---

## Novel Insights

The paper's most transferable insight — that adapter integration into weight-regularized vs. prediction-regularized methods requires qualitatively different strategies (excluding adapter params from penalty vs. adding a backbone-specific distillation term) — is concrete and practically useful for anyone extending these classical baselines. However, the mechanistic claim about invariant vs. task-specific feature decomposition remains an open empirical question that future work should verify directly. The observation that adapter gains shrink as the number of classes per task grows is also worth further investigation, as it may point to a regime-dependence of the method's benefits.

---

## Suggestions

1. Replace all instances of "eliminate the stability-plasticity dilemma" with "mitigate" or "substantially reduce" — this is more accurate and no less impactful.
2. Restate the abstract's claim of "consistently outperform non-adapter counterparts" to acknowledge the ImageNet caveat for prediction-regularized methods.
3. Add at least one diagnostic experiment (e.g., CKA between backbone representations for different tasks) to provide direct evidence for the invariant/task-specific decomposition.
4. Move a condensed class-IL result (or at minimum, a discussion of adapter selection without task-ID) into the main paper.
5. Clarify the bottleneck width choice for ImageNet (why 128 rather than 256?).

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Score | Comparison to paper under review |
|---|---|---|
| `/human_reviews/HCCkCjClO0.md` | 3.0 (Reject) | Weak continual learning paper with unclear motivation, weak baselines, and poor writing. The paper under review is substantially better in clarity and experimental scope. |
| `/human_reviews/gV0Moskp7k.md` | 4.4 (Reject) | Parameter-efficient CL for LLMs; stronger motivation than this paper but also rejected for overclaiming and limited baselines. Roughly comparable in quality. |
| `/human_reviews/1nHQRsb3Ze.md` | 5.0 (Reject) | "Auxiliary Classifiers Improve Stability in CL" — very close comparator: adds a module to multiple CL baselines, shows consistent improvements, no strong theoretical grounding, class-IL performance is limited. Borderline rejected for limited novelty and empirical contribution — a near-identical profile to this paper. |
| `/human_reviews/sSyytcewxe.md` | 7.0 (Accept) | "Divide and not forget" — accepted, MoE approach to class-IL with stronger novel design. Cleaner and more novel contribution than this paper. |
| `/human_reviews/5U1rlpX68A.md` | 7.5 (Oral) | SD-LoRA for class-IL — accepted Oral, with theoretical grounding, novel mathematical decomposition, addresses harder class-IL setting. This paper is clearly below that bar. |

**Reasoning**: The paper under review maps most closely to the avg-5.0 "Auxiliary Classifiers" anchor in terms of contribution profile: a modular add-on to multiple existing CL baselines with consistent empirical improvements but limited theoretical depth, no class-IL in the main paper, and some overclaiming. The additional problems (ImageNet regression for the primary new integration strategy, unverified core mechanistic claim) push it slightly below that anchor. The paper is not as weak as the avg-3.0 to 4.4 papers, which have fundamental methodology or writing problems. I place it at **4.5**, consistent with the borderline-reject cluster.

**Axis evaluation:**
- *Originality*: Low-to-moderate. Adapters for CL exist; the co-training twist and compatibility across two regularization families is the novelty.
- *Importance of research question*: Moderate. IL is an important problem.
- *Claims well-supported*: Weak. Core mechanistic claim unverified; headline claim falsified by Table 1 for ImageNet; "eliminate" framing not supported.
- *Soundness of experiments*: Moderate. CIFAR-100 experiments are thorough; ImageNet is incomplete.
- *Clarity of writing*: Good.
- *Value to community*: Limited. Adding adapters to old IL methods without class-IL support and without deeper analysis is a modest contribution.

**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>