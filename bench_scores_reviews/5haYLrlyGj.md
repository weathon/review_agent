## Summary
MetaSD frames the problem of selecting among multiple specialized speculative-decoding drafters as a Multi-Armed Bandit (MAB) problem, where each SD round is one bandit round and the total number of rounds is stochastic (determined by how many tokens are accepted). The paper's two main technical contributions are (1) a Block Divergence (BD) reward that estimates the normalized TV distance between drafter and target distributions—producing a denser, lower-variance signal than the standard Block Efficiency (BE) reward—and (2) a novel regret definition adapted to the fixed-target-length, stochastic-rounds nature of SD, with a log-linear regret bound for MetaSD-UCB that improves with larger draft length N_max. The framework is evaluated in both black-box (MetaSpS) and white-box (MetaEagle) settings across five diverse tasks and five multilingual pairs.

---

## Strengths

- **Tailored regret objective (Definition 2 / Theorem 2):** The reformulation from "fixed T rounds" to "fixed B tokens, stochastic T" is a non-trivial adaptation. The equivalence argument (B = τ + ΣN_acc) cleanly links round minimization to token acceptance maximization, and the resulting bound inherits the 1/N_max variance-reduction benefit of BD. This is more than a superficial reuse of standard UCB analysis.

- **Block Divergence reward design:** The BD reward (Eq. 1) directly addresses the sparsity problem with BE (Table 2 shows ~50% zero-reward rate for BE on the best drafter). The theoretical link through Theorem 1 (informal) and Lemma 5 (relating E[r^BE] and E[r^BD] via α_i) plus empirical validation in Table 2 and Table 6 constitute a coherent and convincing case. This is the most original and practically impactful piece of the paper.

- **Breadth of empirical coverage:** The evaluation spans black-box and white-box SD paradigms, five diverse NLP tasks, five multilingual translation pairs, greedy and temperature-sampled decoding, and multiple GPU platforms, with ablations on N_max, reward design, and best-arm convergence. Few papers in this space match this scope.

- **Concrete practical motivation via Table 1:** The cross-drafter performance table (where the Ja-drafter achieves 1.757 on Ja→En but only 1.012 on De→En) makes the motivation for adaptive selection immediately tangible and credible, and is more sharply argued than typical motivation sections.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing offline routing baseline—the critical ablation.** All three reviewers independently flag the absence of a simple prompt-level classifier or task-ID router that selects a drafter *once per query* with zero exploration cost. MetaSD's online bandit re-initializes from scratch per query and spends its first K rounds in forced exploration (Phase 1 of Algorithm 2). If a lightweight classifier achieves nearly the same drafter selection accuracy as UCB at zero bandit cost, the main justification for online learning collapses. This is especially pressing because (a) the tasks are labeled (Code, Math, QA, Sum, Trans, language pair), making routing trivially implementable, (b) the paper itself re-initializes the bandit per query, so the "context evolves during generation" motivation is not exercised in practice. Without this baseline, the claim that bandit-based selection adds value beyond one-shot routing is empirically unsubstantiated.

- **Internal contradiction between Figure 4 caption and main text.** Section 4.3 states "Across all configurations, UCB consistently identifies the best arm more rapidly than other bandit algorithms." However, the paper's own Figure 4 caption reads: "In all plots, the 'sh' algorithm consistently achieves a higher best arm ratio than 'ucb' and 'exp3', indicating better performance in identifying the optimal drafter." These two claims directly contradict each other within the same paper. The most likely resolution is that SH achieves a higher best-arm *selection ratio* but UCB achieves higher overall *speedup* due to its regret-optimal exploration policy—which would be a meaningful and nuanced finding—but the paper never makes this distinction explicit. As written, it is an unacknowledged internal inconsistency that undermines confidence in the analysis.

- **Overclaiming in abstract, main text, and conclusion.** The abstract says "superior results compared to traditional single-drafter approaches"; the conclusion says "superior performance of MetaSpS and MetaEagle compared to both specialized drafters and other state-of-the-art methods." However: (a) MetaSD-UCB does *not* consistently beat the task-matched specialized drafter oracle—this is the norm across Tables 3–5, not an exception. (b) PLD substantially outperforms MetaSpS-UCB on Summarization (2.501 vs 1.971), a ~27% gap that is never acknowledged or analyzed. (c) In the multilingual setting (Table 5), SH beats UCB on Ja→En (1.368 vs 1.161), while UCB still substantially trails the matched specialist (1.757). The paper's actual and meaningful result—that MetaSD-UCB approaches oracle-specialist performance while consistently beating fixed generalist (OFA) and non-adaptive baselines—is strong enough to stand on its own; the overclaiming weakens rather than strengthens the narrative.

- **Training budget between OFA and specialists not controlled or discussed.** The OFA drafter is a crucial baseline representing "single drafter" approaches. If each of the K=5 specialists is trained on N task-specific tokens and OFA is trained on a mixed set of the same total N tokens (= N/5 per task), the comparison is not compute-controlled. If instead OFA sees K×N total tokens, the setup may inadvertently favor the specialists. Because the entire empirical motivation for multiple drafters rests on "specialists > generalist," this methodological ambiguity must be resolved. The main paper defers to Appendix F but provides no summary of dataset sizes or training steps in the main text.

### Minor

- **BD reward practical computation not fully explained.** Definition 1 requires computing TV distance between the full token distributions of the drafter and target at each drafted position. In black-box SD, it is not obvious whether the target model already produces full logits over the vocabulary during verification (which would make BD "free") or whether additional computation is required. The paper states BD "utilizes the empirical mean of the acceptance rate" but does not reconcile this with Equation 1, which requires per-token distribution comparisons. Given that BD is the paper's main reward contribution, its exact computational protocol and overhead versus BE should be stated clearly in the main paper.

- **UCB hyperparameter β not analyzed.** Algorithm 2 uses β as an exploration-strength parameter, but the paper never reports what value is used, whether it is the same across tasks and settings, or how sensitive results are to it. Since the paper emphasizes low overhead and minimal tuning, the absence of this ablation is notable.

- **Inconsistency in multilingual claims.** Section 4.2 states "MetaSps-UCB consistently outperforms other bandit-based selection strategies (EXP3, SH)" for multilingual tasks. Table 5 shows SH=1.368 vs UCB=1.161 for Ja→En, directly falsifying this claim. No explanation is offered for why UCB fails on this particular language pair. Understanding when the method underperforms is important for users of the framework.

- **Table 2 reward statistics cover only the Japanese dataset.** BD is central to the paper's contribution, yet Table 2 validates its statistical properties on a single language-pair setting. A reader cannot verify that the lower variance and larger mean gaps generalize across the diverse-task suite without similar statistics for Code, Math, QA, etc.

### Tiny

- **"Dynamically allocate computational resources across various drafters"** (abstract) is misleading: the framework selects *one* drafter per round sequentially; it does not allocate compute across multiple drafters in parallel. A more accurate phrasing would be "dynamically select among drafters."
- The forced exploration cost during Phase 1 (K rounds) is unanalyzed for short-generation tasks. While K=5 may be negligible for long outputs, a brief discussion of the break-even sequence length would be informative.
- Mixed GPU hardware across experiments (A5000, A6000, A100) makes cross-table comparisons of absolute speedup figures difficult to interpret.

---

## Nice-to-Haves

- **Prompt-level task-routing classifier as an explicit baseline** (priority high; already flagged as Major above, but if the authors find it matches MetaSD, the new insight would be interesting in its own right—online learning adapts during generation; routing commits at query start).
- **Absolute wall-clock latency breakdown** including bandit overhead, KV cache I/O, and drafting/verification split, to validate the "negligible switching cost" claim quantitatively rather than asserting it.
- **Scaling analysis for K > 5 drafters** to understand how exploration cost and memory overhead grow, since the paper currently only evaluates K=5.
- **Per-query drafter selection trajectories** for individual examples: does the bandit converge within a single query (beneficial for generation-level adaptation), or does it simply learn a task-level assignment that a classifier could achieve more cheaply?
- Formally controlling N_max's role in the regret bound (Theorem 2) by showing that a larger draft window during Phase 1 meaningfully accelerates identification of the best arm in practice.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Criticism that non-stationarity is inadequately addressed:** The paper explicitly scopes its main theory to the stationary, per-query setting and re-initializes per query in all experiments. This is a stated design decision and a reasonable scope limitation, not an oversight. Section 3.3 discusses extensions. Criticizing this as a weakness conflates the paper's stated scope with what the reviewer wanted the paper to be.
- **Demand for statistical confidence intervals on main tables (Tables 3–5):** Single-run evaluation is the norm in LLM inference acceleration papers at this scale, and the main tables are consistent in their directional trends. Table 6 does provide multi-run statistics for the reward ablation. This demand is not standard for this community.
- **"OFA comparison is unfair"** (one reviewer claimed MetaSD benefits from using more total parameters): MetaSD uses 5 drafters that are *not simultaneously active*; only one drafts at a time, and the KV caches are the additional overhead. This is not a parameters-in-parallel comparison; it is sequential selection. The fairness concern is about training budget, which is addressed as a Minor weakness above, not about architectural unfairness during inference.
- **Criticism that the paper does not beat the task-matched oracle as a standalone weakness:** The oracle drafter (knowing in advance which drafter is best) is a strict upper bound that no online method can be expected to match. Failing to beat the oracle is not a weakness; failing to clearly acknowledge this in the claims is (addressed in Major).
- **Demand for cost-quality tradeoff analysis / explicit verification of losslessness:** Speculative decoding with exact verification is provably lossless; this is a well-known property of the framework. Demanding explicit confirmation in every paper using it is excessive.
- **Serving complexity / training cost as a limitations weakness:** These are real practical considerations but are not new to multi-drafter methods and are not central to the paper's algorithmic claims. They belong in a limitations note rather than as scored weaknesses.

---

## Novel Insights

The spark-finder report draws attention to one genuinely interesting implicit observation in Figure 4 that the paper fails to make explicit: **SH achieves higher best-arm selection ratio while UCB achieves higher overall speedup.** This is consistent with theoretical expectations—SH commits to the estimated best arm faster (lower exploration regret) but UCB's cumulative allocation policy is more efficient for the round-minimization objective. If this distinction were made explicit and tied back to Definition 2 (the round-minimization regret), it would both justify why UCB is the preferred algorithm despite lower best-arm ratio *and* explain the Ja→En case where SH outperforms UCB on speedup (where the gap between drafters may be large enough that early commitment by SH happens to win). This thread, if pursued, could meaningfully sharpen the theoretical narrative.

---

## Suggestions

1. **Add a one-shot router baseline.** Train a lightweight classifier (or use task-label lookup) to select the best drafter at query time, with no exploration. Compare speedup and convergence against MetaSD-UCB. If the router wins, the paper should discuss why online learning adds value in cases where task labels are unavailable or context shifts; if MetaSD-UCB is competitive, the comparison strengthens the paper.

2. **Reconcile the Figure 4 / text contradiction explicitly.** State clearly: "SH converges to the best arm more rapidly in terms of selection frequency, but UCB's exploration policy leads to higher cumulative speedup because it does not over-commit during the early high-uncertainty phase." Then revisit the Ja→En multilingual anomaly (Table 5) in this light.

3. **Provide a training budget table for OFA vs. specialists** in the main paper (even a one-row summary: dataset sizes, steps, total tokens per drafter). This directly addresses the central fairness concern for the OFA baseline.

4. **Clarify BD computation cost in the main paper.** A one-sentence explanation of whether the target model already produces full-vocabulary logits during verification (making BD free in white-box settings) and what is done in black-box SD (Appendix D presumably covers this; a forward reference with summary suffices).

5. **Correct overclaiming language** throughout the abstract, Section 4.2, and conclusion: replace "superior compared to specialized drafters" with "approaches oracle-specialized performance while consistently outperforming fixed generalist (OFA) baselines." Acknowledge the PLD gap on summarization explicitly and offer a hypothesis.

6. **Report the β value** used in Algorithm 2 and include a sensitivity plot (even in the appendix), as this is the only free hyperparameter in the algorithm.

---

## Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate-to-good. Applying MAB to drafter selection is conceptually natural, but the BD reward design and the adapted stochastic-rounds regret objective are genuine original contributions that go beyond "just apply UCB." |
| **Importance of research question** | High. Multi-task and multi-domain LLM serving is a central practical challenge; adaptive drafter selection addresses a real and under-studied bottleneck. |
| **Claims well-supported** | Partially. Core empirical claims are broadly supported, but specific claims of "consistent UCB superiority" and "superior to specialized drafters" are contradicted by the paper's own tables and internal captions. These need correction, not just qualification. |
| **Soundness of experiments** | Moderate. The coverage is broad, but the absence of an offline routing baseline is a significant gap, and the training-budget parity for OFA is unverified. Within its scope, the experimental design is reasonable. |
| **Clarity of writing** | Moderate. The high-level story is clear, but the paper contains at least one explicit internal contradiction (Figure 4 caption vs. Section 4.3 text), several overclaims, and an underspecified account of how BD is computed in practice. |
| **Value to research community** | Good. The BD reward and stochastic-rounds regret adaptation are reusable ideas. The empirical demonstration that multiple specialized drafters + bandit selection consistently beats a single generalist drafter is a useful practical result. |
| **Contextualization relative to prior work** | Adequate for the main baselines (Eagle, Medusa, PLD, Lookahead). The absence of an offline routing comparison is the main gap—not a missing citation, but a missing class of approach. |

Overall, MetaSD is a solid piece of work with a clear practical contribution and genuinely novel methodological pieces (BD reward, stochastic-rounds regret). It is **close to the ICLR acceptance bar** but not clearly above it in its current form. The internal contradictions between text and figures, the uncorrected overclaiming in the abstract and conclusion, the missing routing baseline, and the uncontrolled OFA comparison are the issues that most need to be addressed before the paper can be confidently accepted.