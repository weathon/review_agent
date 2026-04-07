## Summary

The paper investigates "benign relearning" in machine unlearning, where forgotten information resurfaces after fine-tuning on benign data. The authors challenge the prevailing view that topical relevance drives relearning, arguing instead that syntactic similarity between the forget set and relearning data is the primary driver. Through controlled experiments on TOFU and re-evaluation of the BLUR benchmark, they demonstrate that syntactically similar data consistently triggers higher recovery than topically relevant data. They provide mechanistic evidence via representation/gradient alignment and a novel loss ratio analysis showing that unlearning suppresses template tokens more than keyword tokens. They propose "syntactic diversification"—paraphrasing forget queries into heterogeneous structures—which suppresses relearning, accelerates forgetting, and improves model utility.

## Strengths

- **Novel insight about syntactic similarity as a relearning driver:** The identification of syntax (rather than topical relevance) as the key vector for benign relearning is a genuine contribution that reframes how the community should think about unlearning robustness. This is supported by controlled experiments across multiple unlearning methods (GA, NPO, SCRUB).

- **Strong mechanistic analysis:** The loss ratio analysis (Figure 6) showing that unlearning disproportionately suppresses template tokens over keyword tokens—and the causal template-injection experiment in Appendix F confirming that keyword knowledge remains retrievable—are particularly compelling. The finding that attack success rate remains ~0.9 under template injection while free generation fails provides direct causal evidence for the claimed mechanism.

- **Valid methodological critique of BLUR:** The identification of confounds in BLUR's evaluation—different dataset sizes leading to different training budgets, and non-monotonic recovery trajectories—has merit. The re-evaluation (Figure 3) shows that D_low (Lorem ipsum) achieves comparable recovery to D_hi in WHP, undermining the topical relevance hypothesis.

- **Practical mitigation with empirical validation:** Syntactic diversification is conceptually simple and shown to work across methods. Table 2 shows consistent utility improvements (ROUGE, Truth Ratio) compared to standard unlearning.

- **Evidence that relearn sets contain no target information:** Table 5 demonstrates that neither D_relearn[topic] nor D_relearn[syntactic] enables recovery in a perfectly retrained model, ruling out information leakage as an alternative explanation.

## Weaknesses

- **Core claim is overstated relative to evidence:** The paper claims syntactic similarity is "the primary driver" of benign relearning, but the strongest controlled experiments are limited to TOFU—a benchmark where syntactic homogeneity is essentially baked in by construction (questions follow rigid templates like "What is the full name of the author born in X on Y?"). While WHP and WMDP experiments provide supporting evidence, they lack the same controlled comparison between topic and syntax conditions. In WMDP (Figure 2a), some ordering D_hi > D_mid > D_low persists even after re-evaluation, suggesting topical relevance still plays a role. The paper should moderate its claims.

- **Conflated experimental design:** D_relearn[topic] and D_relearn[syntactic] differ in both topic *and* format simultaneously. D_relearn[topic] uses non-name questions about target authors, while D_relearn[syntactic] uses name-format questions about different authors. This conflation makes it difficult to attribute results purely to syntax—the design also introduces differential task activation (fact retrieval vs. name retrieval). A cleaner design would include a condition where topically relevant questions *also* share the same format.

- **Utility comparison may confound forgetting levels:** Table 2 compares model utility at fixed unlearning steps, but Figure 9 (bottom) shows that D_forget' reaches full forgetting faster. If diversification simply accelerates forgetting, then comparing utility at the same step number may favor D_forget' because it has achieved more forgetting with less collateral damage at that point. A fairer comparison would match forgetting levels rather than step counts.

- **No ablation against general data augmentation:** We do not know whether syntactic diversity specifically drives the benefits, or whether any augmentation (e.g., adding more training examples, random paraphrasing) would achieve similar results. The paper should isolate whether breaking syntactic rigidity is the active ingredient versus simply increasing data diversity.

- **Missing limitations discussion:** The paper lacks a dedicated limitations section. Key limitations that should be acknowledged include: (1) focus on parameter-optimization methods only, not guardrail-based or in-context unlearning; (2) adversarial relearning is not studied—all relearn sets are described as "benign"; (3) no practical method is proposed for detecting syntactic similarity at deployment time, despite noting this is a regulatory risk.

## Nice-to-Haves

- Include experiments on non-synthetic benchmarks (e.g., MUSE, LAMA) where syntactic patterns emerge naturally rather than being constructed, to strengthen generalization claims.
- Compare syntactic diversification against other robust unlearning methods (e.g., ERL, Forget-RL) to ensure improvements stem from syntax breaking rather than data augmentation in general.
- Provide a quantitative threshold for syntactic similarity below which relearning risk becomes negligible—this would make the finding actionable for deployment.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **GPT-4o dependency as a major weakness:** Appendix G.4 explicitly shows that Llama-3-8B achieves comparable diversification results, addressing scalability and reproducibility concerns. While GPT-4o is the default, an open-source alternative exists.

- **Missing statistical tests/error bars:** While variance reporting would be welcome, this is a common limitation in ML papers and does not invalidate the results. The qualitative patterns across multiple methods and benchmarks provide internal validation.

- **Demand for theoretical bounds on relearning probability:** This is an unreasonable standard for an empirical paper. The mechanistic analysis and controlled experiments already provide substantial support for the claims.

- **Claim that representation similarity finding is unsurprising:** While it may be intuitive that same-format questions activate similar representations, the gradient analysis and loss ratio decomposition go beyond this obvious point and provide novel mechanistic insight.

- **Request for attention map visualizations and embedding trajectories:** These would be nice additions but are not necessary for the current claims, which are already well-supported by the existing analyses.

## Novel Insights

The loss ratio analysis—demonstrating that unlearning disproportionately suppresses template tokens while leaving keyword knowledge largely intact—is the paper's most mechanistically satisfying contribution. This explains *why* syntactically similar relearning succeeds: it restores the suppressed templates, providing a pathway for dormant keyword knowledge to resurface. The causal template-injection experiment (Appendix F) directly validates this mechanism, showing that explicitly providing the answer template enables keyword recovery even when free generation fails. This finding has implications beyond unlearning: it suggests that current unlearning methods may primarily learn to suppress surface patterns rather than remove underlying knowledge, which has ramifications for how we evaluate unlearning efficacy more broadly.

## Suggestions

- Moderate the "primary driver" claim to "an important and underappreciated driver" or similar, acknowledging that topical relevance may still contribute in some settings.
- Add an ablation comparing syntactic diversification against (a) adding the same number of random training examples and (b) using paraphrases that preserve syntax but change semantics, to isolate whether syntactic breaking specifically drives the benefits.
- Match forgetting levels (rather than unlearning steps) when comparing utility between D_forget and D_forget' to ensure fair comparison.
- Include a brief limitations paragraph acknowledging the focus on parameter-optimization methods, the synthetic nature of TOFU, and the absence of adversarial relearning analysis.