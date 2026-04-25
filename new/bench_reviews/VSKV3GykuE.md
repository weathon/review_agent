Now let me do calibration searches to properly score this paper.Now I have enough to write the final review.

---

## Summary

RAC-LoRA proposes a theoretically grounded low-rank adaptation framework that fixes one of the two LoRA matrices randomly and trains the other, converting the otherwise intractable bilevel LoRA optimization into a projected gradient descent problem admitting standard convergence analysis. The paper proves O(1/T) sublinear rates for smooth non-convex objectives and linear rates under the PL condition, demonstrates that LoRA and COLA can diverge on a concrete quadratic example, and extends the framework to federated learning.

---

## Strengths

- **Counterexample genuinely motivating (Figure 1, Eq. 2):** The 3×3 quadratic example with M = Diag(10, 1, …, 1) concretely and reproducibly shows that LoRA and COLA diverge at the theoretical step size 1/L, while AsymmLoRA converges to a suboptimal stationary point. This is a real negative result that directly motivates the work.

- **Key theoretical insight: asymmetric design yields projected GD (Section 5.1, Eq. 3–4):** Fixing one matrix randomly reduces the subproblem to a convex quadratic in the trainable matrix, whose minimizer takes the closed-form Â = −η(B_S^⊤B_S)†B_S^⊤∇f(W). This makes the outer update W^{t+1} = W^t − γH∇f(W^t) with H an orthogonal projection — a clean connection to sketched/projected gradient descent that is the correct way to analyze LoRA-style methods.

- **Explicit convergence rates recovering classical GD at full rank (Theorems 5.3 and 5.5):** Setting λ_min^H = 1 recovers standard GD rates; setting λ_min^H = r/n for isotropic distributions recovers the correct rank-scaled bound. The machinery is self-consistent and the rates are verifiable.

- **MNIST chaining experiment (Table 3):** With the same 133 trainable parameters, RAC-LoRA achieves 92.0% vs. AsymmLoRA's 62.3% (Gaussian A setting), and with 912 parameters achieves 96.1% vs. 81.6%. This is the paper's clearest empirical evidence that chaining the asymmetric update substantively helps, and the comparison is fair (matched budget, matched parameters).

- **Breadth of theoretical coverage (Table 1):** The framework analyzes GD, SGD, Random Reshuffling, and a federated variant (Fed-RAC-LoRA), both under non-convex and PL conditions — a comprehensive theoretical contribution for a single paper.

---

## Weaknesses

### Fatal
None.

### Major

- **Practical algorithm (AdamW) diverges from the analyzed algorithm (GD/SGD/RR) — no formal justification for this substitution.** Algorithm 1 explicitly allows "any iterative solver" for the subproblem, and the paper analyzes GD, SGD, and RR (Appendix C–E). In practice, both experiments (RoBERTa/GLUE and MNIST, Sections 6.2.1–6.2.2) use AdamW with 10 epochs per block. AdamW is not analyzed anywhere in the paper. Unlike standard GD, AdamW applies adaptive per-parameter second-moment scaling and decoupled weight decay, which fundamentally changes the update trajectory relative to the analyzed closed-form projective step. There is no formal bound or informal argument guaranteeing that the approximate inner-loop solution via AdamW leads to the same guarantees as the analyzed GD step. The experiments therefore validate a heuristic, not the proven algorithm. This gap limits the extent to which the theory is "supported by experimental results" as claimed in the abstract.

- **Primary NLP results underperform plain LoRA (Table 2).** RAC-LoRA with 10 chains × 10 epochs achieves an average GLUE score of 77.0, compared to 78.5 for locally reproduced LoRA (1 chain × 100 epochs) and 82.8 for Hu et al.'s LoRA. The paper explains this by asserting the GLUE tasks are "easy" and a single LoRA is "already enough to obtain performance close to that of FPFT." While this framing is plausible, it is circular: the benchmark was chosen by prior work (AsymmLoRA, COLA) as the standard evaluation, and the paper's explanation amounts to claiming the method only helps where single-block LoRA is clearly insufficient. This regime is never precisely characterized, and no standard NLP benchmark is offered where RAC-LoRA is clearly superior to LoRA.

### Minor

- **"Bridge to FPFT" claim is only partially supported.** The Remark after Assumption 5.1 correctly computes λ_min^H = r/n for isotropic distributions, and Section 6.1 (linear regression) explicitly notes "convergence speed is proportional to r/n." This degradation is therefore acknowledged. However, the practical implication — that for typical LoRA settings (n ~ 4096, r = 2–8) the bound is hundreds to thousands of times looser than standard GD — is not discussed in the context of the "bridge to FPFT" framing. Given realistic dimensions, the chaining required to match FPFT convergence speed could be prohibitively long. The abstract and introduction should more carefully qualify what "bridge" means given this penalty.

- **Fed-RAC-LoRA is listed as a contribution but has no experiments.** Section 2.3 and the abstract mention federated learning as a contribution; the algorithm (Algorithm 2) and theory (Appendix F) are provided. But no federated experiments appear anywhere in the paper. For a system-level contribution, the absence of empirical validation is a notable gap.

- **Comparison scope in Table 2 is limited.** COLA and RAC-LoRA are run for 10 chains × 10 epochs while LoRA* from Hu et al. uses a different training budget (30–80 epochs per task). The authors note this (asterisk in the caption), but do not attempt to match budgets to isolate method effectiveness from total training time. The starred results should either be excluded from this comparison or reproduced under identical conditions.

### Trivial

- The right panel of Figure 1 illustrates that LoRA and COLA *do* converge with small enough step sizes — only to suboptimal points. The paper could more clearly distinguish between the divergence failure (step-size sensitivity) and the convergence-to-suboptimum failure (structural). Both are real, but the former is a lesser practical issue since practitioners always tune step sizes.

---

## Nice-to-Haves

- An experiment plotting convergence speed (loss vs. wall-clock gradient evaluations) for RAC-LoRA vs. FPFT across different ranks r on the linear regression problem, demonstrating the n/r factor empirically and helping readers understand when chaining becomes practical.
- Federated learning experiments for Fed-RAC-LoRA, even on a small synthetic dataset, to provide at least minimal empirical backing for the federated contribution.
- An experiment in a high-rank-deficiency regime on an NLP task (e.g., fine-tuning on a domain-shifted dataset where FPFT clearly outperforms single-block LoRA by a large margin), where the method's chaining mechanism could show a tangible advantage over vanilla LoRA.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic — "Convergence rate degradation is never acknowledged or discussed"**: This is factually incorrect. Section 6.1 explicitly states "convergence speed is proportional to r/n" and Figure 2 illustrates this across ranks. The Remark after Assumption 5.1 computes the exact value λ_min^H = r/n. The rate degradation is acknowledged; what is arguably underdiscussed is its implication for the "bridge to FPFT" framing in the abstract. This was kept as a Minor weakness rather than a Major one.

- **Harsh Critic — The NLP results should exclude starred LoRA numbers as "not on equal footing"**: The paper explicitly marks results from Hu et al. with an asterisk and explains the training budget difference. Noting this discrepancy in the caption is appropriate disclosure. This is not a "structural weakness" requiring excision of those results.

- **Strength Finder — "General framework supporting multiple optimizers is a presentation/practical strength"**: This is partially redundant with the theoretical coverage strength (Table 1) already listed. Merged into the main strength point.

- **Strength Finder — "Identification of Lipschitz smoothness loss under LoRA reparameterization"**: This result is attributed to Sun et al. (2024), Theorem 2, and is explicitly cited as such. The paper uses it as motivation but does not claim it as its own contribution. Dropped as a strength — it belongs to prior work.

---

## Novel Insights

The most genuinely novel technical observation is the equivalence between fixing one LoRA matrix randomly and performing projected gradient descent on the weight space. Prior LoRA theory either ignored the low-rank structure (COLA's analysis in Xia et al.) or worked with the non-smooth (A, B) parameterization directly. RAC-LoRA's derivation shows that the low-rank constraint, when properly handled via a random sketch, maps directly onto a well-conditioned sketched GD step, with the sketch distribution controlling the effective conditioning of convergence. The concrete quantification that isotropic sketch distributions yield λ_min^H = r/n, and therefore an effective convergence rate degradation proportional to n/r, is the right lens through which to evaluate any low-rank adaptation method — and this paper is the first to state and prove it clearly for LoRA-style updates.

---

## Suggestions

1. **Qualify the "bridge to FPFT" claim** in the abstract with the r/n rate penalty; this would make the theoretical claims more honest and more useful to practitioners.
2. **Provide an approximation analysis for AdamW inner-loop**, even informally, or cite a result bounding the error of approximate inner-loop optimization — this would close the most significant theory-experiment gap.
3. **Add at least one experiment where RAC-LoRA outperforms vanilla LoRA on a standard NLP benchmark**, or clearly scope the abstract to the regime where the chaining benefit is demonstrable (i.e., low-rank bottleneck tasks).
4. **Add minimal federated learning experiments** (even synthetic) to back the federated contribution.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Comparison to paper under review |
|------|-----------|----------------------------------|
| `/home/wg25r/review_agent/human_reviews/VpeAsLmcvg.md` | 3.75 | LoRA theory paper (SVA), rejected; had deeper theoretical flaws than RAC-LoRA whose proofs are sound. Below this paper. |
| `/home/wg25r/review_agent/human_reviews/c2OtbtZXFC.md` | 4.75 | Stiefel manifold + LoRA theory, rejected; had theory-application mismatches and experiments not supporting method. Closer to this paper but RAC-LoRA's theory is more self-consistent. |
| `/home/wg25r/review_agent/human_reviews/IZjBfdVRB0.md` | 5.00 | Circular convolution LoRA variant, rejected; primarily empirical with incremental gains. This paper has stronger theory but weaker NLP experiments. Roughly comparable. |
| `/home/wg25r/review_agent/human_reviews/fAGEAEQvRr.md` | 5.50 | GD with large initialization for matrix factorization, rejected; clean theory but modest novelty. Similar profile. |
| `/home/wg25r/review_agent/human_reviews/N0gT4A0jNV.md` | 6.00 | Alternating minimization for low-rank matrix completion, accepted; clean theory + experiments on realistic problems, no NLP underperformance. Somewhat stronger than this paper. |
| `/home/wg25r/review_agent/human_reviews/e0rQRMUhs7.md` | 6.60 | Federated LoRA (FRLoRA), accepted; stronger empirical coverage across 9 benchmarks vs. this paper's mixed NLP results. Clearly above this paper. |

**Assessment:** This paper sits between c2OtbtZXFC (4.75, rejected) and N0gT4A0jNV (6.00, accepted). The theoretical contribution is more coherent and self-contained than the Stiefel-LoRA paper, placing it above ~4.75. However, the mixed NLP results (underperforming plain LoRA on GLUE), the AdamW theory-experiment gap, and the absence of federated experiments prevent it from reaching the quality level of accepted low-rank optimization papers around 6.0. Centrally, the paper makes a genuine and novel theoretical contribution but fails to empirically demonstrate that the proposed method is practically advantageous over the baselines it is meant to supersede.

**Score: 5.0 (Borderline Reject)**

**Axes:**
- *Originality*: Moderate-high. The projected-GD interpretation of asymmetric LoRA updates is genuinely novel.
- *Importance*: Moderate. The convergence theory for LoRA is an open problem; this is a meaningful step.
- *Claims supported*: Partially. Theory is sound; empirical claims about practical superiority are not well-supported.
- *Experimental soundness*: Weak. GLUE results underperform LoRA; AdamW not analyzed; no federated experiments.
- *Clarity*: Good. Paper is well-organized and the derivation in Section 5.1 is clear.
- *Community value*: Moderate. The framework and counterexample are useful to the optimization community; the practical algorithm needs stronger empirical backing.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>