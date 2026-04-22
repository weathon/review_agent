Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final review.

## Summary

This paper extends the theory of rank collapse from transformers to State Space Models (SSMs) using a unifying framework (Ali et al., 2024; Dao and Gu, 2024). It introduces λ-skip connections—parametrized skip connections with a tunable strength λ—and derives a sufficient condition under which this parameter, together with LayerNorm, guarantees a lower bound on the rank collapse metric μ across all layers for any architecture expressible as Y^(k) = D^(k)(M^(k-1)Y^(k-1)C_V^(k-1) + λY^(k-1)). The paper also proves that selective SSMs suffer exponential (with LayerNorm) or doubly exponential (without LayerNorm and skip connections) rank collapse, and makes the novel empirical observation that gating mechanisms help prevent rank collapse.

## Strengths

- **First analysis of rank collapse in SSMs.** Theorem 4.3 proves that selective SSMs without skip connections suffer exponential rank collapse: μ(Y^(K)) ≤ √N(1 − ε²λ²_min α^(2N))^K (Section 4.2.1). Combined with Theorem A.10 (doubly exponential collapse when both skip connections and LayerNorm are ablated, due to the quadratic input-dependence of M), this extends rank collapse phenomena from transformers to the SSM family—a previously open question.

- **Unified theoretical framework across architectural families.** Theorem 4.1 derives a λ-dependent lower bound μ(Y^(K))² ≥ a^K μ(Y^(0))² for all architectures expressible in the form of Equation 6, generalizing rank collapse theory from transformers to SSMs using the Ali et al. (2024)/Dao and Gu (2024) unifying framework.

- **Tightness result for the lower bound.** Proposition 4.3.2 constructs a specific architecture where μ(Y^(k))² = O(a^k μ(Y^(0))²) when λ satisfies Equation 7, proving the bound cannot be improved without additional assumptions—this is an honest and useful finding (Section 4.2.3).

- **Novel connection between gating mechanisms and rank collapse.** Figure 3 shows that removing gating from Mamba-2 causes μ to approach zero (with LayerNorm) or exhibit instability (without LayerNorm), while the full model remains stable. This links gating's original memory purpose to rank collapse prevention, a connection the paper identifies as novel (Section 5.2).

- **Learnable λ does not hurt performance.** Table 1 shows training with variable λ across four architectures maintains or slightly improves accuracy compared to fixed λ=1 (e.g., Mamba-2 on MQAR: 97.3% → 99.1%), suggesting the proposed modification is practically viable.

## Weaknesses

### Fatal

None. The theory is mathematically sound; the issue is one of framing and looseness, not incorrectness.

### Major

- **"Prevention" framing overclaims what is actually a rate bound.** Theorem 4.1 states μ(Y^(K))² ≥ a^K μ(Y^(0))², where a < 1 is structurally required for any architecture with SC_M > 0 (since λ² - 1·(SC_M + |λ|)² < 0 always). The paper acknowledges "the only way to guarantee a solution to 7 is by having 1 − a > 0" (line 154) and notes in Remark 4.1 that a can be very close to 1, but the paper's title, abstract, and conclusions use the word "prevents" repeatedly. Since a^K → 0 as K → ∞, the theorem guarantees controlled decay rather than prevention of rank collapse. For practical depths (e.g., K=64 with a=0.9, a^64 ≈ 0.001), the bound still permits severe collapse. The paper should explicitly state that the result bounds the *rate* of collapse and does not guarantee μ stays above a fixed threshold for arbitrary depth. This matters because practitioners using the word "prevention" would expect μ remains bounded away from zero.

- **The input condition μ(Y^(0))² ≥ b is potentially vacuous for deep networks and is not discussed.** The quantity b contains the factor 1/a^K, which grows exponentially in K since a < 1. For deep networks where rank collapse matters most, this condition may require input diversity exceeding the maximum possible value of μ(Y^(0))². The paper provides no discussion of whether this condition is satisfiable for any realistic architecture or input sequence, which limits the practical applicability of the main theorem.

- **Large gap between theoretical predictions and empirical observations.** The paper acknowledges the bound is "too conservative" (line 246), but does not quantify the gap. For the Mamba architecture with S=1 and C_M = √N (e.g., N=128), the theorem requires |λ| on the order of tens of thousands to achieve a close to 1, yet experiments show λ ≈ 20–100 already suffices (Figure 1). Moreover, the experiments do not verify whether Theorem 4.1's specific conditions (Equation 7 and the input condition) are met. Without measuring C_M and checking the conditions on actual models, it is impossible to assess whether the theorem provides non-vacuous guarantees in practice.

### Minor

- **The necessity framing in Section 4.2 is somewhat misleading despite the question mark.** Although the section title uses "?" and the paper states "we do not provide a formal necessary condition" (line 162), Section 4.2 shows only: (a) rank collapse occurs without skip connections (known for transformers, new for SSMs), and (b) specific 2×2 counterexamples where collapse occurs for certain λ (Propositions 4.3.1–4.3.2). These do not establish that the sufficient condition in Theorem 4.1 is tight in any general sense—only that λ matters on a case-by-case basis. The section title creates expectations the content does not deliver.

- **Theorem 4.3 (selective SSM collapse) requires Assumption 4.1 (A not input-dependent), which is restrictive for Mamba.** The paper acknowledges this and points to Figure 1 for empirical validation of the input-dependent case (line 186), but this means the theoretical collapse result for the most important architectural variant (full selective SSMs with input-dependent A) rests entirely on empirical evidence. The paper already flags this limitation, which is acceptable, but it deserves somewhat more emphasis.

- **Experiments primarily measure forward-pass behavior of a pre-trained model under modified architecture.** Taking a 2B-parameter pre-trained Mamba-2 model and changing λ post-hoc (Section 5.1) measures forward-pass rank collapse under a parametrization the model was never trained under. This follows the protocol from prior work (Dong et al., 2023; Wu et al., 2024a) but leaves open whether training from scratch with appropriate λ improves optimization and final performance. Table 1 partially addresses this but only with learnable λ on small-scale tasks, not with fixed λ values spanning the theorem's threshold.

- **Figure 3 "instability" interpretation needs clarification.** The red curve (Gating=False, LN=False) shows μ increasing dramatically after initially dropping, which the paper describes as "instability." But in the paper's framework, high μ means diverse (non-collapsed) tokens—so a sudden increase in μ seems to contradict the collapse problem. The paper should clarify that this overshoot represents numerical/oscillatory instability rather than healthy diversity, which is a different failure mode than rank collapse.

### Trivial

- The simplified LayerNorm (Equation 4) removes both centering and learnable affine parameters (γ, β), which is a known simplification from prior work (Wu et al., 2024a) but could change the dynamics of the "mean token" direction. The paper attributes this to prior work but does not discuss implications.

## Nice-to-Haves

- Overlay the theorem's predicted lower bound on experimental plots (plotting both empirical μ and the theoretical a^K μ(Y^(0))² for actual computed C_M, S values) to make the gap visually apparent.
- Train models from scratch with fixed λ values spanning the theorem's threshold and below, measuring both rank collapse during training and final task performance, to directly test the theorem's predictive power.
- Include gating in the theoretical analysis as a per-element λ, which would substantially strengthen the contribution for Mamba-like architectures given that the empirical results (Figure 3) already highlight its importance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"First study that provides a general guarantee" is overstated relative to Dong et al. (2023).** The harsh critic claims Dong et al. (2023) already established skip connections provide infinite parametrizations avoiding rank collapse. However, the current paper's guarantee is different in kind (a λ-dependent quantitative lower bound with an explicit sufficient condition), so calling it the "first general guarantee" is defensible as a distinguishable contribution, even if the framing could be more precise.
- **Demand for missing related works.** Removed per hard rules—cannot verify existence of unspecified related work.
- **Formatting nitpicks.** Removed per hard rules—parser artifacts.
- **Reproducibility concerns about undisclosed hyperparameters.** Removed per hard rules—trivial implementation details.
- **Missing appendix/proofs.** Removed per hard rules—the parser strips appendix content.

## Novel Insights

The identification of the quadratic input-dependence of M in selective SSMs as the structural cause of doubly exponential rank collapse (paralleling the attention matrix's role in transformers) is a genuinely novel observation. This creates a clean conceptual parallel: just as softmax attention's row-stochastic property drives doubly exponential collapse in transformers, the quadratic dependence M = ISS(α) ⊙ (YW_C W_B^T Y^T) drives the same phenomenon in selective SSMs when both skip connections and LayerNorm are removed. The connection between gating mechanisms (originally designed for memory) and rank collapse prevention is also an insightful empirical finding that suggests gating serves a dual purpose in Mamba architectures.

## Suggestions

- Reframe "prevention" as "rate control" or "mitigation with a bounded decay rate" throughout the paper—this would align the language with what the theorem actually delivers and make the contribution more honest while still meaningful.
- Quantify whether the input condition μ(Y^(0))² ≥ b is satisfiable for any realistic setup (even a single worked example with typical parameter values), or acknowledge explicitly when it becomes vacuous.
- Measure C_M and S on the pre-trained Mamba-2 model used in experiments and compute the theorem-predicted λ threshold, then compare this with the empirically observed threshold from Figure 1. This single computation would clarify whether the gap is 10x or 1000x.

## Evaluation Axes

- **Originality**: Moderate-to-good. Extending rank collapse theory to SSMs and identifying the doubly exponential collapse mechanism are novel. The λ-skip connection concept itself is straightforward.
- **Importance of research question**: High. Rank collapse affects training stability and model expressivity, and SSMs are increasingly important architectures.
- **Claims well supported**: Partially. The theory is sound but conservative, and the "prevention" claim overreaches what the theorem delivers. The input condition is not validated.
- **Soundness of experiments**: Fair. Qualitatively consistent with theory but disconnected from specific predictions. The forward-pass-only protocol follows prior work but limits conclusions about training dynamics.
- **Clarity of writing**: Good. The paper is well-structured and the unified framework is clearly presented.
- **Value to research community**: Moderate. The conceptual contributions (SSM rank collapse analysis, gating connection) are valuable; the practical guidance from the main theorem is limited due to conservatism.

## Calibration

**Anchors compared:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| GNN oversmoothing + skip connections | i8vPRlsrYu | 7.0 | Similar topic (residual + normalization prevent collapse) but more complete theory and practical method. Our paper has looser bounds and overclaimed framing. |
| Mean-field token clustering | eBS3dQQ8GV | 7.8 | More rigorous math on token collapse. Our paper covers more architectures but with less depth. |
| Hyper-connections | 9FqARW7dwB | 6.25 | Similar topic (modifying skip connections), more empirical. Our paper has more theory. |
| Transformer oversmoothing eigenspectrum | OCx7dp58H1 | 5.75 | Similar (analyzing simplified architecture, rank collapse), rejected for simplified assumptions and unvalidated reparametrization. Our paper has similar issues. |
| Effective state-size for SSMs | DHVjLvSps6 | 5.6 | Similar domain (SSM analysis), rejected for unclear practical implications. Our paper has clearer theory but with the looseness/vacuity issue. |
| Conservative generalization bounds (LLM) | MF7ljU8xcf | 6.0 | Similar weakness pattern (conservative bounds). Accepted poster despite looseness. |
| Grokking dynamical systems | a8XwgTZzE0 | 2.0 | Overclaimed theory, disconnected experiments. Our paper is far stronger—math is sound. |
| DNNs as dynamical systems | 4YK1e3Ehdy | 2.6 | Incoherent presentation, disconnected. Our paper is clearly better. |

Our paper is clearly above the low-scoring anchors (sound math, meaningful framework) but below the high-scoring ones (conservative bounds, overclaimed framing, experiments don't directly validate predictions). It's comparable to the medium-scoring anchors, slightly below MF7ljU8xcf (6.0, accepted) because the overclaiming is more central to the paper's identity, and comparable to OCx7dp58H1 (5.75, rejected). The paper's real contributions (SSM rank collapse analysis, gating connection) add value that the rejected anchors didn't have, but the gap between what's claimed and what's delivered in the main theorem remains significant.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>