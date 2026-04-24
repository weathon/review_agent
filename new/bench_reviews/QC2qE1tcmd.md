## Summary

The paper proposes a unifying framework that views simplicial and cellular complexes as relational structures, enabling the extension of graph-theoretic oversquashing analysis to higher-order message passing. The authors introduce *influence graphs* and *aggregated influence matrices* to bound Jacobian sensitivity (Lemma 3.2), connect local geometry to oversquashing via an extended Forman curvature (Proposition 3.4), and analyze the impact of depth (Theorem 3.5) and hidden dimensions (Section 3.4). They also propose a heuristic rewiring algorithm for relational structures and evaluate it on TUDataset benchmarks and a synthetic RINGTRANSFER task.

## Strengths

- **Clean unifying axiomatic framework.** The paper recasts simplicial message passing (Equations 1–2) as an instance of relational message passing (Definition 2.5), explicitly showing how boundary, co-boundary, lower, and upper adjacencies map to relations of varying arity (Remark 2.6). This provides a useful conceptual bridge between topological and relational deep learning.
- **Novel conceptual tool for analysis.** Definition 3.1 (influence graph) and the aggregated influence matrix (Equations 5–7) offer a compact formal device for translating graph-theoretic quantities to relational message passing, which may help practitioners reason about information flow in higher-order architectures.
- **Systematic multi-model experimental design.** Table 1 evaluates eight architectures (SGC, GCN, GIN, RGCN, RGIN, SIN, CIN, CIN++) across three lifting strategies on five TUDataset benchmarks, providing broad coverage of how different model classes respond to rewiring.

## Weaknesses

### Fatal
None.

### Major

- **Cherry-picked rewiring evaluation undermines empirical claims.** Table 1 reports a "Best Rew." column that selects the best-performing rewiring method (SDRF, FoSR, or AFRC) post-hoc per cell. The paper states in Section 5.1 that all three methods were run, but does not disclose in the main text or table caption that the reported result is the maximum across methods per (dataset, model, lifting) tuple. This practice inflates apparent gains and obscures cases where rewiring degrades performance (e.g., SIN on ENZYMES with clique lifting drops from 51.0 to 46.5; GCN on ENZYMES without lifting drops from 32.2 to 30.7). Because the failures are hidden, the reader cannot assess whether rewiring is reliably beneficial or whether the improvements are simply due to selecting the lucky best case among three alternatives.

- **Synthetic validation does not include the simplicial networks the paper purports to study.** The abstract claims "empirical studies on simplicial networks," and the title emphasizes "simplicial message-passing." However, the RINGTRANSFER experiments in Section 5.2 and Figure 2 — which are presented as validation of Theorem 3.5 (depth) and Section 3.4 (hidden dimensions) — use only GIN and RGCN with clique/ring liftings. RGCN is explicitly classified in Section 5.1 as a *relational graph* model, distinct from the *topological* models (SIN, CIN, CIN++). While RGCN on lifted structures falls within the broad relational framework, it does not validate whether the derived bounds govern the behavior of the actual simplicial architectures that are the paper's stated focus. This is a significant gap between the object of analysis and the object of evaluation.

### Minor

- **No diagnostic evidence links real-world performance changes to oversquashing.** The paper claims that rewiring mitigates oversquashing on TUDatasets, but provides no measurements of gradient norms, influence-graph curvature, effective resistance, or bottleneck statistics on these benchmarks to establish that (a) oversquashing is present, and (b) the observed accuracy changes are caused by its mitigation. Without such diagnostics, the practical contribution in Section 4 rests on a plausible but unverified causal story.

- **Theoretical contribution is thinner than presented.** The influence graph framework aggregates higher-order shift operators into a pairwise matrix $\tilde{\mathbf{A}}$ (and then $\mathbf{B}$), after which Lemma 3.2, Proposition 3.4, and Theorem 3.5 adapt existing GNN proofs by substituting $\mathbf{B}$ for the adjacency matrix. The paper claims these are "novel extensions ... where existing methods for analysis do not apply," but the method is to reduce the relational structure to a derived graph and apply existing bounds. While the reduction itself is a useful conceptual device, the analytical machinery does not genuinely extend beyond what is already known for graphs.

### Trivial

- Citation inconsistency: "Papamakarios et al., 2024" in the introduction becomes "Papamakou et al., 2024" in Section 6.

## Nice-to-Haves

- Replace the post-hoc "Best Rew." column with a single, principled rewiring strategy (e.g., FoSR) run with a fair hyperparameter search budget, and report all results including cases where rewiring fails, with proper statistical testing.
- Run the RINGTRANSFER benchmark with actual simplicial networks (SIN, CIN) to validate depth and hidden-dimension predictions on the architectures emphasized in the title and abstract.
- Measure actual Jacobian norms during training on a controlled task to verify that they correlate with $(\mathbf{B}^t)_{\sigma,\tau}$ as predicted by Lemma 3.2.
- Provide theoretical justification for Algorithm 1 (e.g., showing how adding $E_{\text{new}}$ as a new relation improves the relevant bound in Lemma 3.2), rather than presenting the heuristic as self-evident.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Definition 2.5 is a straightforward arity-generalization of Schlichtkrull et al. (2018)."** The paper explicitly states this in Section 2.2. The value of the definition lies in the unification with simplicial message passing, not in claiming the relational MP scheme itself is revolutionary. This is a minor observation, not a substantive weakness.
- **"Proposition 3.4 is directly ported from Fesser & Weber (2023)."** The paper acknowledges it is "inspired by" Fesser & Weber (2023, Proposition 3.4). The adaptation to weighted directed influence graphs is nontrivial and correctly cited.
- **"Fixed hyperparameters are a significant confounder."** The authors explicitly acknowledge this limitation in Section 5.1 and note that hyperparameter tuning can impact performance. This is an honest disclosure of a study design choice, not a hidden flaw.
- **"Missing proofs in appendix."** Per instructions, appendix sections are stripped by the parser; proofs deferred to the appendix exist in the original submission.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Restructure Table 1** to report results for a single rewiring method with hyperparameter tuning, or report all three methods separately. If reporting "Best Rew." is retained, the caption must explicitly state that it is the maximum across SDRF, FoSR, and AFRC per cell, and the discussion must honestly characterize how often rewiring hurts performance.
2. **Add RINGTRANSFER experiments with SIN and/or CIN** to align the synthetic validation with the paper's stated focus on simplicial networks.
3. **Add diagnostic measurements** (e.g., gradient norms, influence-graph curvature) on at least one benchmark to empirically connect rewiring-induced performance changes to oversquashing.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/EzjsoomYEb.md` (avg 8.00, Accept Oral): A topological deep learning paper with rigorous theoretical analysis, novel architectures (MCN/SMCN), and new benchmarks. The paper under review is weaker on both theoretical depth and empirical rigor.
- `/home/wg25r/review_agent/human_reviews/qkBBHixPow.md` (avg 6.00, Accept Poster): PIORF, a physics-informed rewiring method for oversquashing with focused experiments on three benchmarks. The paper under review has broader model coverage but suffers from cherry-picking and lacks the focused empirical grounding of PIORF.
- `/home/wg25r/review_agent/human_reviews/iAmR7FfMmq.md` (avg 5.50, Accept Poster): CoED GNN, a novel but polarizing idea with mixed reviewer reception. The paper under review has a clearer conceptual framework but more serious empirical methodological flaws.
- `/home/wg25r/review_agent/human_reviews/YkR9UFlQ1s.md` (avg 4.40, Reject): NBA-GNN, rejected partly for reporting the maximum of two variants and for limited novelty. The paper under review shares the cherry-picking issue but has a stronger unifying framework.
- `/home/wg25r/review_agent/human_reviews/xMxHJxp192.md` (avg 4.80, Reject): DeltaGNN, rejected for weak experiments and theoretical confusion. The paper under review is cleaner theoretically and has more comprehensive experiments.

**Comparison:** The paper under review sits between the accepted 5.50 anchor (CoED) and the rejected 4.40 anchor (NBA-GNN). Its conceptual framework is genuinely useful, but the empirical evaluation has serious methodological issues — particularly the post-hoc selection of the best rewiring method per cell and the absence of simplicial networks in the synthetic benchmark — that undermine the reliability of its practical claims. These issues are addressable in revision but weigh against acceptance in the current form.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>