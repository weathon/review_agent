=== CALIBRATION EXAMPLE 39 ===

# Final Consolidated Review
## Summary
This paper proposes DISCO, an efficient model-evaluation pipeline that selects benchmark items with high inter-model disagreement and predicts full-benchmark performance from the target model’s output signature on those items. The main empirical result is strong: with only 100 samples, DISCO substantially improves MAE and rank correlation over prior efficient-evaluation baselines on four multiple-choice language benchmarks and also transfers well to ImageNet classification.

## Strengths
- **A specific and practically strong empirical recipe emerges clearly from the paper:** replacing scalar subset accuracy with a learned predictor over **model signatures** is highly effective. In Table 1, even `Random + Sig. + RF` already improves markedly over `Random + Direct eval.` and over tinyBenchmarks variants on several datasets, showing that the richer response pattern is genuinely useful rather than a cosmetic reformulation.
- **The proposed disagreement-based selection further improves the signature approach in a consistent way.** On MMLU, moving from `Random + Sig. + RF` (1.81 MAE / 0.933 rank) to `High PDS + Sig. + RF` (1.07 / 0.987) is a large gain; similar improvements appear on HellaSwag, Winogrande, ARC, and ImageNet. This supports the paper’s central intuition that samples that separate models are more useful than merely representative ones.
- **The paper contributes a meaningful theoretical lens for why disagreement is informative.** Proposition 1 and Appendix G establish that, for a single sample, the mutual information between model identity/statistics and the model output equals a generalized JSD across model predictive distributions. This does not fully prove greedy subset optimality, but it does provide a principled justification for using disagreement as an informativeness signal.
- **The evaluation protocol is more realistic than the usual random meta-split.** The paper explicitly uses a chronological split of source vs. target models and also reports additional bootstrapped chronological results in Appendix D.2, which is a better test than purely IID splitting for this class of metamodels.
- **The factor analysis is unusually useful and actionable.** The paper does more than report a headline result: it studies predictor choice, dimensionality reduction, stratification, source-model count, and split choice. In particular, the observation that PCA is important for preventing overfitting and that kNN becomes preferable at extreme compression is practically valuable.
- **The vision extension is a real cross-domain check, not just a re-run on another language dataset.** Applying the same two-stage idea to ImageNet and obtaining 0.63 MAE / 0.969 rank with 100 examples suggests the core mechanism is not narrowly tied to one benchmark family.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overstates the theoretical support for the actual greedy subset-selection algorithm.** Proposition 1 is about the informativeness of a **single sample**: it shows that for one item, disagreement/JSD equals the mutual information with a model statistic under specific assumptions. But DISCO selects a **set** of samples greedily, and the paper does not show that ranking items by pointwise JSD/PDS maximizes the information in the resulting concatenated signature, nor does it address redundancy between selected items. So the theory justifies the *signal* used for scoring, but not the stronger implication that this yields an information-theoretically optimal greedy subset-selection rule at the set level. This matters because the paper’s framing leans heavily on theoretical optimality.
- **The main experiments do not fully isolate how much of the gain comes from selection versus prediction architecture.** The headline claim is that disagreement-based selection is better than representativeness-based selection, yet Table 1 changes both the selection mechanism and the predictor across methods. The paper does include an important partial ablation—`Random + Sig. + RF` versus `High PDS + Sig. + RF`—which shows selection matters. However, it does **not** compare alternative anchor sets under the **same signature-based predictor** (e.g., `Anchor-corr + Sig. + RF` or `Anchor-IRT + Sig. + RF`). As a result, the evidence strongly supports the full DISCO pipeline, but less cleanly supports the narrower claim that disagreement-based selection itself is the decisive improvement over prior anchor-selection strategies.
- **Robustness to source-model distribution shift remains a real limitation, and the paper’s own appendices reveal a notable failure mode.** Section 6 acknowledges dependence on the source-model population, and Appendix F shows that under a performance-gap split the gain over direct evaluation shrinks sharply (+1.8 rank points versus +6.5 to +8.7 elsewhere). The paper argues this scenario is unrealistic, but for a meta-predictor intended for rapidly evolving model families, stronger stress tests across model-family, training-objective, or capability shifts would be important. This weakness does not negate the reported results, but it does limit how broadly one should trust the method’s claimed robustness.
- **The offline cost and source-pool requirements weaken the inclusivity/democratization narrative.** Appendix B reports a substantial one-time offline cost (around 3284 GPU-hours on MMLU) to evaluate hundreds of source models and build the predictor. The paper does discuss amortization and break-even, which is reasonable, but this means the method is most compelling for ecosystems with a large pre-existing pool of evaluated models and repeated future evaluations—not necessarily for new or niche benchmarks with little initial infrastructure.

### Minor
- **The method is less plug-and-play than the main narrative suggests.** Appendix I states that on MMLU, using all available source models to compute disagreement performed worse, so the authors randomly subsample source models and tune this number as a hyperparameter (`M=100` for MMLU). This is a legitimate design choice, but it means performance depends on source-pool composition and tuning in a way that should be surfaced more prominently in the main paper.
- **The applicability is restricted to tasks with predefined answer choices and access to predictive probabilities.** The paper is explicit about this limitation in Section 6, so this is not a hidden flaw, but it substantially narrows the scope relative to the broad motivation around modern model evaluation costs.
- **The “domain-agnostic” claim is somewhat broader than what is actually shown.** The language experiments cover multiple-choice QA-style benchmarks, and the vision experiment is ImageNet classification. That is a useful multimodal validation, but not yet evidence for broad task-agnosticity across, e.g., structured prediction, open-ended generation, or more heterogeneous vision tasks.

### Trivial
- **JSD and PDS are empirically very close in the reported experiments, so the practical guidance on when one should prefer one over the other is limited.** This does not hurt the main contribution, but a clearer recommendation would improve usability.

## Nice-to-Haves
- Evaluate alternative anchor-selection methods using the **same signature-based predictor** to cleanly separate “selection improvement” from “prediction-model improvement.”
- Add stronger stress tests across genuinely shifted target populations, such as new architecture families, reasoning-specialized models, or other capability-distribution shifts.
- Provide a more systematic analysis of source-model redundancy/composition, since Appendix I already suggests this materially affects performance.
- Include qualitative analysis of the selected samples: are the highest-PDS items genuinely discriminative capability boundaries, or sometimes ambiguous/noisy items?
- Give clearer guidance for choosing source-pool size, PCA dimension, and predictor type on a new benchmark.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Chronological split showing little degradation means the split must be trivial or unrealistic.”** This is speculative. The paper does use a chronological split and additionally reports bootstrapped chronological results; one cannot infer from strong performance alone that the split is invalid.
- **Complaints that some cited baselines/methods cannot be verified or may not be available.** Not admissible under the review rules.
- **Missing related work / requests to compare to additional contemporaneous papers not evaluated here.** I cannot verify completeness of external coverage, so this should not be used as a substantive criticism.
- **Formatting/wording complaints about clustering being described as “complex and sensitive.”** This is mostly framing rather than a technical issue, and the paper’s contribution does not hinge on proving clustering is bad in general.
- **Generic reproducibility complaints about omitted trivial details.** The appendix already includes substantial implementation details, hyperparameters, cost numbers, and variance estimates.
- **Claims of unfair comparison because Metabench uses more anchor points.** The paper itself explicitly marks these results as “not directly comparable” in Table 1, so the concern is already disclosed. It is still fair to say this particular SOTA comparison should be interpreted cautiously, but not to treat it as hidden unfairness.

## Novel Insights
The most important synthesis across the reviews is that the paper’s strongest contribution is probably **not** the disagreement score alone, but the combination of two ideas with very different roles: (1) disagreement-based selection provides a principled way to find high-leverage examples, while (2) the much bigger conceptual simplification may be the shift from scalar corrected accuracies/IRT summaries to **response signatures as metamodel inputs**. The paper’s own numbers show that signatures already close much of the gap even under random selection, after which disagreement-based selection adds a further substantial boost. This suggests a useful reframing: DISCO is best viewed as a strong end-to-end pipeline whose predictive backbone is unusually powerful, with disagreement serving as a targeted improvement on top—not as a pure victory of selection alone. That interpretation makes the empirical results more convincing and the remaining gaps more precise.

## Suggestions
- Reframe the theoretical claim more carefully: say that Proposition 1 justifies **sample-wise disagreement as an informativeness proxy**, not optimal greedy subset selection in general.
- Add a controlled ablation where prior anchor sets are paired with the **same signature + RF** predictor used by DISCO.
- Promote the source-model subsampling/tuning detail from Appendix I into the main paper and analyze its sensitivity more systematically.
- Expand robustness tests to stronger source/target shifts, especially across model families or training paradigms.
- Temper the broadest claims (“domain-agnostic,” “information-theoretically optimal rule”) to match the actual evidence.


# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
