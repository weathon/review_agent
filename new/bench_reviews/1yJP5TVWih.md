Now I have a comprehensive understanding of the paper and all the claims. Let me write the final review.

## Summary

This paper extends the theory of rank collapse from transformers to State Space Models (SSMs) using the unifying sequence model framework of Ali et al. (2024) and Dao and Gu (2024). It introduces lambda-skip connections — a parameterized version of standard skip connections with a controllable strength parameter λ — and provides a sufficient condition on λ (Theorem 4.1) that guarantees a lower bound on the rank collapse measure μ across transformers, LTI SSMs, and selective SSMs. The paper also establishes that selective SSMs suffer exponential rank collapse without skip connections (even with LayerNorm) and doubly exponential collapse without both, and empirically identifies gating mechanisms as important for rank collapse prevention.

## Strengths

- **Genuine extension of rank collapse theory to SSMs.** Theorems 4.2 and 4.3 (and the appendices on LTI SSMs) establish that selective SSMs suffer from analogous rank collapse dynamics as transformers — exponential with LayerNorm, doubly exponential without. The identification of the quadratic input-dependence of M^(k) as the cause of doubly exponential collapse in selective SSMs (Section 4.2.1, Appendix A.9) is a genuine insight that fills a real gap, as SSMs had not been analyzed from this perspective.

- **The unifying framework is well-motivated and clean.** Using Equation 6 (Y^(k) = D^(k)(M^(k-1)Y^((k-1)}C_V^{(k-1)} + λY^(k-1))) to treat both transformers and SSMs under one recurrence is elegant and allows Theorem 4.1 to apply across architectures through shared quantities C_M and S. This unification is a meaningful organizational contribution.

- **The lower bound in Theorem 4.1, while conservative, is non-trivial to derive.** Obtaining any non-vacuous lower bound on μ that degrades only exponentially (rather than doubly-exponentially) with depth, for a general class of architectures with simplified LayerNorm, requires careful accounting. The tightness result (Proposition 4.3.2) shows the bound cannot be improved without additional assumptions, which is a meaningful completeness result.

- **Empirical identification of gating mechanisms' role in rank collapse.** Figure 3 demonstrates that gating mechanisms in Mamba-2 play a crucial role in preventing rank collapse — a novel empirical observation connecting gating's original purpose (memory) to a new role (rank collapse prevention).

## Weaknesses

### Fatal
None.

### Major

- **The "prevention" framing systematically overstates what Theorem 4.1 delivers.** The paper's title ("The Architectural Component That Prevents Rank Collapse"), abstract ("guarantees for rank collapse prevention"), and Section 4.1 ("Sufficient to Prevent Rank Collapse") all claim prevention. However, Theorem 4.1 requires a < 1 (as the paper acknowledges in the discussion following the theorem: "the only way to guarantee a solution to [7] is by having 1 − a > 0"), meaning the lower bound μ(Y^(K))² ≥ a^K μ(Y^(0))² decays exponentially with depth. In the infinite-depth limit, this IS rank collapse — just at a slower rate. The paper defines rank collapse as "convergence in the limit of infinite depth" (Section 3.2), so technically, the theorem does not prevent what the paper itself defines as rank collapse. For finite depth, μ > 0 is trivially satisfied by almost any architecture (the paper itself notes "the rank of the output layer will be full rank with probability one"). The meaningful contribution is showing that the convergence *rate* is at most exponential rather than doubly exponential — a real but qualitatively different claim from "prevention." Remark 4.1 acknowledges a^K ≈ 0.993 for K=64 with a=0.9999, but this only works because the specific numerical setting makes a^K close to 1; it does not generalize as a "prevention" guarantee. This overclaiming affects the title, abstract, introduction, and conclusion.

- **The sufficient condition on λ is so conservative it provides no actionable design guidance.** For the Mamba case (C_V = I, S = 1, C_M = √N), with N = 128 and a = 0.9999, the condition |λ| > (a + √a)SC_M/(1−a) requires |λ| > ~226,000. The experiments (Figure 1) show λ ≈ 10–20 suffices in practice. This four-orders-of-magnitude gap means Theorem 4.1 cannot inform architecture design. The paper acknowledges this ("our condition on λ in Theorem 4.1 is too conservative"), but the severity is understated: a sufficient condition this loose provides zero practical content and represents a fundamental gap in the usefulness of the main theoretical result.

- **Experiments do not establish that rank collapse prevention (as measured by μ) matters for practical model quality.** Table 1 shows that learning λ does not consistently improve accuracy — performance *decreases* in 5 out of 8 settings (e.g., Mamba-2/Image: 42.28 → 38.92; Transformer/MQAR: 99.6 → 98.9). The paper claims "learning λ does not affect the performance and even outperforms," which is a misleading characterization of mixed results. More fundamentally, no experiment links the rank collapse metric μ to any downstream consequence: there is no evidence that models with low μ train worse, that preventing rank collapse via λ improves training dynamics, or that the theoretical guidance on λ improves any practical outcome. The experiments on pre-trained models only confirm that changing λ changes μ — not that this matters.

### Minor

- **The theory operates on unnormalized μ while experiments report normalized μ.** Theorem 4.1 bounds μ(Y^(K)) = ‖Y^(K) − 𝟏γ_Y^(K)‖_F (Equation 5), but Section 5.1 measures "the rank collapse measure μ (normalized by the norm of the layer output)." The relationship between these two quantities is not discussed, and the theoretical guarantee for unnormalized μ does not directly transfer to the normalized version, potentially creating a gap between theory and experimental validation.

- **The input condition μ(Y^(0))² ≥ b may be restrictive.** Theorem 4.1 requires the input to satisfy μ(Y^(0))² ≥ b where b = (1/a^K)(2λNdSC_M)/(λ² − a(SC_M + |λ|)²). Since the 1/a^K term grows for a < 1, deeper networks require increasingly high μ at the input. The paper does not discuss whether typical tokenized inputs satisfy this condition or how restrictive it is.

- **Gating mechanisms are excluded from the theoretical analysis but shown to be crucial empirically.** The paper acknowledges this as a limitation, but the disconnect is significant: Figure 3 shows gating is the most important component for rank collapse prevention in Mamba-2, yet the theory that forms the paper's core contribution does not account for it.

- **The necessity analysis provides suggestive examples rather than formal necessary conditions.** Section 4.2's title asks "Necessary to Prevent Rank Collapse?" but delivers only ablation studies (showing collapse without skip connections) and 2×2 counterexamples (Propositions 4.3.1 and A.10.1). The paper is honest about this ("Although we do not provide a formal necessary condition"), but the framing implies more than is delivered.

### Trivial
None.

## Nice-to-Haves

- Training models from scratch with different λ values and tracking both μ dynamics during training and final task performance would establish the practical relevance of rank collapse prevention.
- Including MLPs in the lower bound analysis could substantially tighten the bound and might narrow the theory-practice gap, as the paper itself anticipates.
- A correlation analysis between μ at initialization and final training loss/accuracy would establish whether the rank collapse metric is worth optimizing in practice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "Simplified LayerNorm deviates from standard practice."** The paper explicitly acknowledges this ("we consider a slightly simplified version of LayerNorm, similar to Wu et al. (2024a)") and this is standard practice in this line of theoretical work. Downgraded to a minor note within the gating exclusion point.

- **Harsh critic: "Spike in Figure 3 is numerical instability not discussed."** This is speculative — the paper describes it as "a dramatic increase" which could have multiple interpretations. Without concrete evidence it's numerical instability, this is an unfounded specific claim.

- **Strength Finder: "Trainable λ experiments demonstrate practical viability."** Table 1 results are mixed (5/8 cases show decreased accuracy), contradicting the claim of "practical viability." Dropped as a strength.

- **Strength Finder: "Control-theoretic interpretation of λ."** While Remark 4.1 makes this connection, it is a brief speculative remark without developed analysis. Too thin to count as a substantive strength.

- **Harsh critic: "Necessity analysis only establishes skip connections are necessary, not the specific λ condition."** The paper is transparent about this: "Although we do not provide a formal necessary condition, we explore this idea in two ways." The examples in Section 4.2.2 do show that the specific choice of λ matters, which goes beyond just showing skip connections are necessary. The criticism is partially valid but overstates the gap.

## Novel Insights

The paper's identification that selective SSMs suffer doubly exponential rank collapse due to the quadratic input-dependence of the M matrix — analogous to the attention mechanism in transformers — is a genuine and insightful finding. It reveals a structural similarity between the two architectures that goes beyond the surface-level unification framework: the key shared property driving extreme rank collapse is that the mixing matrix M depends quadratically on the input, whether through the softmax attention mechanism or through the Y W_C W_B^T Y^T structure in selective SSMs. This suggests that any architecture with this quadratic dependency will face similar vulnerabilities, a useful design principle for future architectures.

## Suggestions

- Re-title the paper to accurately reflect the contribution: e.g., "Lambda-Skip Connections and the Rate of Rank Collapse in Sequence Models" — this would convey the genuine contribution (rate analysis) without the overclaim of "prevention."
- Quantify the tightness gap explicitly: for a specific model used in experiments, compute C_M and S, derive the required λ from Theorem 4.1, and compare with the empirically sufficient λ. This would turn the abstract acknowledgment of conservatism into a concrete, informative analysis.
- Add at least one experiment training from scratch with different λ values, measuring both μ dynamics during training and final accuracy, to establish whether the rank collapse metric has practical consequences for model quality.

## Score and Decision

**Calibration comparison:**

| Anchor Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Emergence of meta-stable clustering (mean-field transformer PDE) | eBS3dQQ8GV | 7.80 | Much more rigorous mathematical contribution; also has gap to practice but the theory itself is substantially deeper |
| Setting the Record Straight on Transformer Oversmoothing | OCx7dp58H1 | 5.75 | Most similar: rank collapse analysis with simplified assumptions, overclaimed generality. This paper has similar issues but additionally has the very conservative bound problem |
| Rademacher complexity bounds for CNNs | Y7lc4aZ4iP | 4.00 | Overclaimed "non-vacuous" bounds that were questioned; similar pattern of theoretical overclaiming |
| SGD complexity with unrealistic assumptions | lK0WxHeups | 2.50 | Low anchor; fundamentally flawed theoretical claims |
| SSM expressivity limits (parity/counting) | UvTo3tVBk2 | 8.00 | High anchor; also about SSM expressivity but with clean, tight results |
| Effective state-size (ESS) for SSMs | DHVjLvSps6 | 5.60 | SSM theory paper with limited practical impact, rejected |

This paper sits between OCx7dp58H1 (5.75, Reject) and Y7lc4aZ4iP (4.00, Reject). It shares OCx7dp58H1's issues of simplified assumptions and limited practical generality, but has a more severe theory-practice gap (the bound is off by 4 orders of magnitude, whereas the OCx paper at least had a reparameterization with some practical potential). It is above Y7lc4aZ4iP because the extension to SSMs is a genuine and needed contribution, and the theoretical analysis, while conservative, is non-trivial and correct. The ESS paper (5.60, Reject) is a reasonable comparison — a theoretical contribution to understanding SSMs that didn't demonstrate sufficient practical impact. This paper's overclaiming ("prevention") is worse than the ESS paper's framing, pushing the score down. The genuine novelty of studying SSM rank collapse pushes it up from the low range.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>