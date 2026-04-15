Now let me read the referenced human reviews for calibration:Now I have enough information to produce the final consolidated review. Let me synthesize everything.---

## Summary

This paper extends the theory of rank collapse from transformers to State Space Models (SSMs) by leveraging the unified sequence-model framework of Ali et al. (2024) and Dao & Gu (2024). The authors introduce *lambda-skip connections*—a parameterized generalization of standard skip connections—and derive sufficient conditions on λ under which a lower bound on the rank-collapse metric is guaranteed not to vanish over a finite depth. They also show that without skip connections, selective SSMs exhibit exponential (and under certain conditions doubly-exponential) rank collapse, and they provide an empirical finding that gating mechanisms also help prevent rank collapse in Mamba-style architectures.

---

## Strengths

- **Genuine novelty: first rank-collapse study for SSMs.** Prior work (Dong et al., Wu et al.) covered transformers; this is the first paper to formally extend the analysis to LTI SSMs and selective SSMs using the O = MV unifying template.
- **General sufficient condition (Theorem 4.1).** The lower-bound result applies to any architecture expressible in the unified form of Eq. 6, covering transformers, LTI SSMs, and selective SSMs in a single theorem. The LayerNorm normalization decouples C_M from the specific input, making the condition architecturally interpretable.
- **Novel empirical connection between gating and rank collapse.** Figure 3 and Section 5.2 demonstrate—apparently for the first time—that gating mechanisms (originally designed for memory) also serve to prevent rank collapse in Mamba-2. This is a practically useful observation even if the theory does not yet cover it.
- **Tightness result (Proposition 4.3.2).** The paper shows that without additional assumptions, the lower bound cannot be improved, providing an honest account of what can and cannot be proven within this framework.
- **Collapse results for selective SSMs without skip connections (Theorem 4.3 and Appendix A.9/A.10).** The derivation adapts Wu et al. (2024a) to the selective SSM setting and explains why doubly-exponential collapse occurs (input-dependent M^(k) introduces a quadratic dependence on the input norm).

---

## Weaknesses

### Fatal
*None triggered.*

### Major

- **Abstract/title overclaim vs. what Theorem 4.1 actually proves.** The paper defines rank collapse as an *infinite-depth* convergence to rank-1 (Sec. 3.1) and the abstract claims "a general guarantee to prevent rank collapse." However, Theorem 4.1 delivers only a finite-depth lower bound: μ(Y^(K))² ≥ aᴷ μ(Y^(0))². For Mamba specifically, Remark 4.1 acknowledges that a < 1 is the *only* feasible choice, meaning the bound still decays exponentially with depth. The practical argument in Remark 4.1 (a = 0.9999, K = 64 → aᴷ ≈ 0.993) is reasonable, but it is a heuristic mitigation, not a proof of prevention in the sense defined. The contributions section correctly scopes the result to "the finite layers setting," so the result itself is sound — but the title and abstract consistently claim more than the theorem delivers. This mismatch should be corrected for honesty and reproducibility.

- **Theory omits gating, yet experiments show gating is the dominant mechanism for Mamba.** Section 3 explicitly states "we ignore [gating] in the theoretical part of this paper for simplicity," but Section 5.2 shows that, in the fully pretrained Mamba-2 model, gating is the component primarily responsible for maintaining the rank-collapse metric near 1.0. The λ-skip experiments in Section 5.1 are conducted on a *gating-free* version of the model, which is an ablated and off-distribution architecture. The theory thus characterizes a simplified system, while the dominant effect in the real architecture is entirely outside its scope. The limitations section acknowledges this, but the framing that the paper provides "a unifying theory for transformers and SSMs" is overstated given that gating—standard in every deployed Mamba variant—is excluded.

- **Mixed Table 1 results and ambiguous practical conclusion.** The paper concludes that "learning λ does not affect the performance and even outperforms the models with fixed λ in some cases." However, Table 1 shows several notable degradations: Mamba-2 on Image LRA drops from 42.28 to 38.92, Transformer on MQAR drops from 99.6 to 98.9, and Linear Transformer on Image drops from 34.10 to 32.80. On a 2-task, 4-model table with no variance estimates, these cannot be dismissed as noise. The claim that learnable λ is safe or beneficial is not convincingly supported.

### Minor

- **The input threshold condition μ(Y^(0))² ≥ b (Theorem 4.1) is depth-dependent and never analyzed in practice.** The threshold b depends on K, λ, C_M, and S. There is no discussion of whether this condition is typically satisfied for realistic inputs or trained weights, or how it fails. The paper implicitly assumes it is fine, but it constitutes a hidden premise of the main result.

- **Section 4.2's framing of "necessity" is misleading.** The section explicitly says it does not provide a formal necessary condition, yet the heading asks whether lambda-skip is "necessary" and the section proceeds with ablations and hand-constructed counterexamples. What is shown is: (a) collapse can occur without skip connections, and (b) collapse can occur for specific λ values in particular systems. Neither establishes a necessary condition. The heading should be reworded to reflect what is actually proven.

- **The sufficient-condition bound is acknowledged as too conservative without quantification.** The paper notes in Section 5.1 that "our condition on λ in Theorem 4.1 is too conservative, in practice much lower values of λ are good enough." However, there is no analysis of *how* conservative—the empirically effective threshold versus the theoretical threshold—leaving practitioners unable to use the theorem for guidance on λ selection.

### Trivial

- The simplified LayerNorm (Eq. 4, omitting the bias/shift term) follows Wu et al. (2024a) and is appropriate for theoretical tractability.

---

## Nice-to-Haves

- **Report learned λ values across layers in Table 1 experiments.** This would reveal whether the model naturally satisfies or violates the theoretical condition, bridging the theory-practice gap identified as a weakness.
- **Train models from scratch with fixed λ values** at various settings to validate that the rank-collapse control translates to stable or improved training dynamics, not just inference-time proxy metrics.
- **Provide at least a qualitative/conservative bound that incorporates gating**, even if only for a scalar gating multiplier. Even a coarse result would significantly strengthen the Mamba-specific claims.
- **Show the empirical λ threshold for rank collapse vs. the theoretical threshold as a function of depth**, to make the conservativeness concrete rather than anecdotal.
- **Add confidence intervals/multiple seeds to Table 1** (differences on some tasks are <1%).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Harsh critic, Issue 3 framing as fatal ("experimental evidence is not suitable to validate")**. The use of post-hoc perturbation of pretrained models to study rank-collapse behavior is consistent with methodology in Dong et al. (2023) and Wu et al. (2024a), which this paper explicitly follows. The criticism that it "cannot support causal conclusions" is a methodological standard not uniformly applied in this subfield — this is a limitation, not a fatal flaw.

2. **Harsh critic, claim that the paper should be rejected because a < 1 in the Mamba case**. Remark 4.1 directly addresses this, showing that a = 0.9999 over K = 64 layers gives a^K ≈ 0.993. This is a practical and transparent resolution. The claim that this "leap is not justified" ignores the explicit quantitative argument provided in the paper.

3. **Harsh critic, characterizing Section 5.2 as undermining the theoretical framework entirely**. The paper explicitly frames gating as a separate, empirically studied phenomenon outside the theoretical scope (see Sec. 3 footnote and limitations). The theory and empirical findings are presented in complementary roles, not as contradicting each other.

---

## Novel Insights

The most genuinely novel observation from the combined reviews is the tension between the paper's strongest empirical finding and its theoretical contribution: the gating mechanism—not λ-skip connections—appears to be the primary rank-collapse-prevention mechanism in the dominant real Mamba-2 architecture, but gating is entirely excluded from the theoretical analysis. This creates a productive research gap: a theory that covers the lambda-skip mechanism is presented, while the empirical observation that gating is at least as important (Figure 3) is noted. The paper is essentially motivating the next theoretical step (incorporating gating into the rank-collapse framework) through its own experiments, even if it does not achieve it. The tightness result (Proposition 4.3.2) is also underemphasized — it establishes that no general lower bound can do better without additional structural assumptions, which is a hard boundary result useful for future work.

---

## Suggestions

1. **Revise the abstract and title** to accurately reflect the finite-depth scope of Theorem 4.1 and to distinguish "slows/bounds rank collapse over practical depths" from "prevents rank collapse."
2. **Quantify the conservativeness gap** in Section 5.1 by plotting the theoretical λ threshold from Eq. 7 against the empirically measured threshold across multiple architectures and depths.
3. **Restructure Section 4.2** to clearly distinguish "collapse occurs when skip connection is entirely removed" from "lambda-skip is necessary," and rename the section accordingly.
4. **Investigate learned λ values** from Table 1 experiments and evaluate whether they satisfy Eq. 7 — this would validate or refute the practical relevance of the theorem.
5. **Address Table 1 instabilities**: add multiple seeds or variance estimates, and analyze why Mamba-2 on Image LRA degrades with learnable λ.

---

## Score and Decision

**Calibration:**

- *pymXpl4qvi* (SSM bottleneck paper, accepted, 6/6/6/6): Analyzes SSM limitations theoretically and empirically, with practical mitigation. Accepted partly because it identified SSM issues first. The paper under review is comparably novel but has weaker practical validation (conservative bounds, mixed Table 1).
- *X6xzYP2cMk* (rank collapse in transformers, rejected, 5/5/6/3): Analysis limited to the first layer only; no general result. The paper under review is strictly stronger in theoretical scope (all depths, both architectures).
- *cxKLRM3KhC* (residual connection weighting, rejected, 6/5/5/6): Related idea of parameterizing skip connections, rejected due to lack of systematic λ guidance and mixed results. The paper under review has similar issues in Table 1.

**Positioning**: This paper is above X6xzYP2cMk (rejected, ~4.8 average) in scope and rigor. It sits below pymXpl4qvi (accepted, 6.0 average) in practical impact and experiment quality. The conservative bounds, gating exclusion, and mixed Table 1 results prevent a clear accept, but the novelty (first SSM rank-collapse study, general sufficient condition, gating finding) prevents a clear reject. The abstract/title overclaiming is the most fixable issue.

**Score: 5.0** — Borderline/weak reject. The paper contains real and novel contributions but overclaims in the abstract, underdelivers on practical validation, and leaves its most important empirical finding (gating) outside the theoretical scope.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>