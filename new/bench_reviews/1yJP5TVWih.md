Now let me search for calibration papers.Now I have enough to write the final review. Let me synthesize everything.Now I have everything I need to write the final review.

---

## Summary

This paper extends the theory of rank collapse from transformers to State Space Models (SSMs) using the unifying framework from Ali et al. (2024) and Dao and Gu (2024). The central contribution is a parameterized skip connection ("λ-skip connection") together with a sufficient condition (Theorem 4.1) that provides a lower bound on the rank collapse metric across transformers, LTI SSMs, and selective SSMs. The paper also proves that selective SSMs suffer exponential (and doubly exponential, without LayerNorm) rank collapse when skip connections are removed, and empirically identifies gating mechanisms as a novel architectural component that prevents rank collapse.

---

## Strengths

- **First rank collapse analysis for SSMs** (Theorems 4.3 and A.10): The paper establishes exponential rank collapse for selective SSMs without skip connections and doubly exponential collapse without both skip connections and LayerNorm. The doubly-exponential regime is traced to the quadratic input-dependence of M^(k) in selective SSMs (‖M^(k)‖_F ≤ √N‖Y^(k)‖_F²‖W_BC‖_F), creating a direct mechanistic parallel to the transformer case. This fills a genuine gap in the literature.

- **Unified framework (Equation 6)**: The recursion Y^(k) = D^(k)(M^(k-1)Y^(k-1)C_V^(k-1) + λ^(k)Y^(k-1)) unifies transformers, LTI SSMs, and selective SSMs in a single formalism, enabling a single proof to cover all architectures. This is an elegant and reusable contribution.

- **Tightness of the lower bound (Proposition 4.3.2)**: The paper constructs a specific system achieving μ(Y^(k))² = O(a^k μ(Y^(0))²) when λ satisfies Equation 7, confirming the bound in Theorem 4.1 cannot be improved without additional assumptions. This is proper theoretical due diligence.

- **Novel empirical connection between gating and rank collapse (Figure 3)**: The four-condition ablation on a pretrained Mamba-2 (gating±, LayerNorm±) cleanly shows that the gating mechanism independently prevents rank collapse, a connection not previously made in the literature and explicitly noted by the authors as first-of-its-kind.

---

## Weaknesses

### Fatal
None.

### Major

- **Theorem 4.1's sufficient condition is not evaluable for the primary target architecture (selective SSMs).** The condition λ² − a(SC_M + |λ|)² > 0 (Equation 7) requires knowing C_M = sup_k ‖M^(k)‖_F. For transformers, this is C_M = √N (from the row-stochastic structure of softmax attention). For selective SSMs, however, footnote 2 only provides a *lower* bound on C_M — not the upper bound required to verify Equation 7. The paper acknowledges in Section 5.1 that the theoretical condition is "too conservative" and that practitioners need to tune λ empirically. If the condition cannot be evaluated for the architecture the paper targets most prominently (Mamba, Mamba-2), then Theorem 4.1 does not provide actionable design guidance for that architecture. The result functions as a mathematical consistency check rather than a usable sufficient condition for the experiments presented. This significantly limits the theorem's practical value for the SSM case.

- **Table 1 inconsistency: the stated conclusion does not match the data.** The paper concludes "learning λ does not affect the performance and even outperforms the models with fixed λ in some cases." Yet for Mamba-2 on Image LRA, the learnable λ underperforms fixed λ by 42.28% → 38.92% (−3.36 percentage points). No variance estimates or statistical significance are reported anywhere in Table 1, so it is impossible to determine whether any difference is meaningful. With only two tasks and mixed results (performance decreases in two out of four conditions), the stated conclusion of parity or improvement is not supported by the evidence as presented.

### Minor

- **Title and abstract overstate the theoretical guarantee.** Theorem 4.1 establishes μ(Y^(K))² ≥ a^K μ(Y^(0))², where a < 1 by necessity (Remark 4.1 correctly points this out for Mamba: a = 1 is impossible when SC_M > 0). The lower bound decays to zero as K → ∞. The paper frames this correctly in context ("in the finite layers setting" in the introduction; Remark 4.1 provides the a = 0.9999, a^64 ≈ 0.993 justification), and this framing is reasonable. However, the title "The Architectural Component That *Prevents* Rank Collapse" and the abstract's claim of "a general guarantee to *prevent* rank collapse" go somewhat beyond what Theorem 4.1 achieves — which is making rank collapse negligibly slow for practical depths, not eliminating it in a strict sense. "Mitigates" or "practically prevents in finite-depth networks" would be more accurate.

- **The initial condition μ(Y^(0))² ≥ b is under-discussed.** The quantity b depends on C_M, S, a, N, d, and K. If the input is low-rank (small μ(Y^(0))), the theorem gives no guarantee. The paper does not discuss when this condition holds in practice or what it implies for initialization schemes common in SSMs.

### Trivial

- Section 4.2 is titled "Necessary to Prevent Rank Collapse?" but opens by explicitly stating "we do not provide a formal necessary condition." The framing builds expectations that the section does not meet; Proposition 4.3.1 is an example for one specific 2-token system, not a necessity result. This should be more clearly labelled as an illustrative example.

---

## Nice-to-Haves

- **Training-from-scratch experiments with learnable λ.** All rank collapse validation is done on a pretrained Mamba-2 with gating removed (an out-of-distribution forward pass). Training a model from scratch with learnable λ alongside fixed-λ baselines would directly test whether the theory's predictions improve training stability or performance, and would cleanly separate the contributions of λ from the pretrained weights.

- **Layer-by-layer comparison of predicted vs. observed lower bound.** Plotting a^k μ(Y^(0))² against measured μ(Y^(k)) for specific λ values satisfying Equation 7 would make the tightness argument (Proposition 4.3.2) concrete and visually anchor theory to experiment.

- **Theoretical explanation for the negative-λ asymmetry.** Figure 2 shows that negative λ yields higher μ than positive λ of the same magnitude. Theorem 4.1 treats |λ| symmetrically. A brief theoretical account of this discrepancy (possibly related to the "negative feedback" interpretation mentioned informally) would strengthen the paper.

- **Training experiments keeping gating intact.** The natural downstream question — does adding learnable λ to a standard Mamba-2 (with gating) improve any performance metric? — is not answered. Even negative results would be valuable.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "C_M depends on weight matrices in a complex, input-dependent way — cannot be bounded in practice."** Partially valid for selective SSMs but overstated: the paper explicitly provides C_M = √N for transformers, which is computable and immediately usable. The SSM case is a real limitation (retained as Major weakness) but the critique as applied to transformers is wrong. Kept as a Major weakness for SSMs only.

- **Harsh Critic: "Experiment in Figures 1–2 is completely confounded by using a pretrained model."** This is partially valid but overstated as "structural confound." The same methodology (pretrained models with ablated components) is used in Dong et al. (2023) and Wu et al. (2024a) to validate rank collapse theory. The purpose is to isolate the effect of λ on the forward-pass rank collapse measure, not to claim task performance improvements.

- **Harsh Critic: "The a < 1 bound asymptotically goes to zero — this is the same category as Wu et al. (2024a) for LayerNorm."** This is a valid framing issue but overstated as "structural misrepresentation." The paper explicitly scopes to "finite layers setting" in the introduction and provides Remark 4.1 as a quantitative defense. The title overclaims, but the paper body is reasonably honest. Retained as a Minor weakness (title/abstract framing).

- **Harsh Critic: Figure 3's clean result means gating is the "operative mechanism" and λ-skip connections are merely compensating for its removal.** This alternative interpretation is interesting but ignores the fact that Figures 1–2 are explicitly designed to test λ in isolation. The paper's claim is that λ can substitute for gating as a collapse-prevention mechanism — not that it's superior to gating. The experiment is a reasonable test of that claim.

- **Strength Finder: "Remark 4.1 provides concrete design guidance."** The guidance (choose a = 0.9999, set λ accordingly) is partly dependent on being able to compute the threshold from Theorem 4.1 — which is not possible for selective SSMs (see Major weakness). For transformers it is actionable. Kept as a partial strength embedded in the theoretical section but not listed as a standalone strength.

- **Strength Finder: "Consistent empirical validation across architectures (Figures 1–2)."** This is partially true but the experimental design has the confound discussed above. Kept as supporting evidence, not a standalone strength.

---

## Novel Insights

The paper's most genuinely novel insight is that the doubly-exponential rank collapse in selective SSMs — despite their different architecture from transformers — arises from the quadratic input-dependence of M^(k), mirroring the transformer mechanism through a structurally different route. The unifying Equation 6 makes this parallel visible: both architectures suffer doubly-exponential collapse, but for different reasons that both reduce to the same recursive relationship. This suggests a deeper architectural principle connecting collapse behavior to the degree of input-dependence in the mixing matrix M, which the paper does not fully articulate but lays the groundwork for. The empirical finding that gating mechanisms (originally designed for memory) independently prevent rank collapse is a second novel observation that could motivate formal gating-collapse theory as future work.

---

## Suggestions

1. Compute and report the theoretical λ threshold from Theorem 4.1 for the transformer case (where C_M = √N is known) and compare it to the empirically effective λ in Figure 2. This would make the "too conservative" claim precise and quantify the gap.
2. Replace the conclusion in Table 1 with a statistically qualified statement, or add variance estimates and reframe around the cases where learnable λ does not hurt.
3. Rename Section 4.2 to "Examples: Rank Collapse Can Occur With λ-Skip Connections" or similar, to avoid implying a necessity result that is not proved.
4. Add a paragraph in Section 4.1 discussing conditions under which μ(Y^(0))² ≥ b holds (the initial condition of Theorem 4.1), especially for common initialization schemes.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Setting the Record Straight on Transformer Oversmoothing | OCx7dp58H1 | 5.75 (Reject) | Most topically similar: extends rank collapse analysis, proposes architectural modification, similar experiment limitations. This paper's contribution (SSMs as a new domain) is more novel. |
| Unlocking State-Tracking in Linear RNNs | UvTo3tVBk2 | 8.00 (Accept Oral) | SSM theory paper; more rigorous, cleaner experiments, higher practical impact. Much stronger anchor. |
| Emergence of meta-stable clustering in transformers | eBS3dQQ8GV | 7.80 (Accept Oral) | Deep theoretical analysis of transformer token collapse; rigorous PDE analysis. Far stronger than paper under review. |
| Deep Network Partition Density / Neural Collapse papers | O8fUZfC4GT | ~4.0 (Reject) | Weak theoretical foundation, limited practical value. Paper under review is clearly stronger than these. |

**Reasoning:**

The paper's core theoretical contributions — first rank collapse analysis in SSMs, the unifying framework, tightness result, and gating finding — are genuine and well-executed. These exceed what OCx7dp58H1 achieved (which only reparametrized transformers and got 5.75, rejected). The SSM extension is timely and fills a real gap.

However, the paper's two major weaknesses weigh against acceptance: (1) Theorem 4.1's main condition is not computable for selective SSMs, the primary target architecture, limiting the theorem to a mathematical consistency result for that case; and (2) Table 1 reports a notable regression (−3.36%) for Mamba-2 on Image LRA without error bars, while claiming the method "does not affect performance." These weaken the connection between theory and practice.

Positioned between OCx7dp58H1 (5.75, rejected) — which this paper moderately exceeds in novelty — and a standard solid-accept threshold (6.5+), the appropriate score is **5.5**. The paper is borderline: the theoretical contributions deserve publication, but the overstated claims, non-computable conditions for the target architecture, and weak empirical validation in Table 1 need addressing before acceptance.

**Originality:** Good — first SSM rank collapse analysis, novel gating finding.
**Research question importance:** High — rank collapse is a real and underexplored issue for SSMs.
**Claim support:** Moderate — theory is sound but overclaimed in title/abstract; experiments are incomplete.
**Experimental soundness:** Moderate — rank collapse visualization experiments are reasonable; Table 1 is weak.
**Writing clarity:** Good overall.
**Community value:** Solid — both theorists (framework, theorems) and practitioners (λ tuning, gating insight) benefit.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>