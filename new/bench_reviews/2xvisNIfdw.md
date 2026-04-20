Now let me do calibration searches to anchor my score.Now I have enough calibration information to write the final review. Let me synthesize everything.

---

## Summary

This paper proposes a framework for achieving global convergence in bilevel optimization by analyzing the penalized objective $L_\gamma(u,v) = f(u,v) + \gamma(g(u,v) - g^*(u))$ rather than the nested objective $F(u)$. It introduces two bilevel-specific PL conditions—joint and blockwise—and proves (for the first time) that PBGD converges to a global optimum with $\mathcal{O}(\log^2(\epsilon^{-1}))$ complexity on two specific problems: two-layer linear representation learning and data hyper-cleaning with MSE loss. The analysis is algorithm-dependent, verifying the PL conditions along the PBGD trajectory via induction and matrix perturbation theory.

---

## Strengths

- **First global convergence guarantee for a gradient-based bilevel method (Theorems 2 & 3).** Prior work (Ji et al., 2021; Hong et al., 2023; Chen et al., 2021) only establishes convergence to stationary points or local minima. Obtaining genuine global convergence—even for linear models—is a qualitatively different and significant advance.

- **Technically non-trivial trajectory analysis (Lemma 1, Theorem 2).** The induction argument that maintains $\sigma_{\min}(W_1^k) > 0$ and $\sigma_{\min}(W_2^k) > 0$ throughout PBGD's trajectory, combined with acute matrix perturbation theory to produce $k$-independent lower bounds on $\mu_k$, is a substantive technical contribution not present in prior bilevel work.

- **Illuminating landscape visualizations (Figures 1–2).** Example 1 and Figures 1–2 concretely demonstrate that even when both $f$ and $g$ satisfy the joint PL condition, the nested objective $F(u) = \frac{1}{2}(u - 2\sin u)^2$ violates PL and has spurious local solutions, while $L_{10}(u,v)$ retains a benign bowl-shaped landscape. This is pedagogically and scientifically valuable.

- **Clean structural match between problem type and algorithm variant.** The identification of isomorphic vs. heterogeneous bilevel structure as determining whether the Jacobi or Gauss-Seidel update is appropriate, matched with joint vs. blockwise PL conditions respectively, is conceptually clean.

- **Almost-linear convergence rate of $\mathcal{O}(\log^2(\epsilon^{-1}))$ (Theorem 1)**, which is vastly better than $\mathcal{O}(\epsilon^{-1})$ or $\mathcal{O}(\epsilon^{-2})$ complexity bounds typical of stationary-point bilevel analyses.

---

## Weaknesses

### Fatal
None.

### Major

- **Orthogonal data assumption for data hyper-cleaning is very strong and load-bearing.** Lemma 2 requires $X_\text{trn}X_\text{trn}^\top$ to be diagonal, and Theorem 3 requires $[X_\text{trn}; X_\text{val}][X_\text{trn}; X_\text{val}]^\top$ to be diagonal — i.e., all pairs of data vectors (both training and validation) are mutually orthogonal. This is not a mild regularity condition; it is essentially never satisfied by real datasets. The paper derives the closed-form expression for $\ell_\gamma(u, W)$ in Eq. (12) via Lemma 23 precisely because $\mathcal{S}(u)$ is independent of $u$ under this condition. Without diagonal $X_\text{trn}X_\text{trn}^\top$, the form of $\ell_\gamma$ does not simplify and the blockwise PL argument over $u$ collapses. This limits Theorem 3 to a highly stylized synthetic regime. It is not a proof gap but a genuine structural scope limitation that the paper's discussion does not adequately foreground.

- **Representation learning result is confined to two-layer linear networks in an overparameterized regime ($m \geq \max\{N, N'\}$, $h \geq \max\{m, n\}$) with MSE loss.** The paper explicitly justifies this ("even analyzing linear models can capture the essence"), which is appropriate for a pilot study, but the claim that "our analysis is adaptable to multi-layer neural networks" (Section 4) is stated without supporting argument. The induction argument relies critically on tracking $\sigma_{\min}(W_1^k)$ via matrix perturbation theory for linear maps; this technique does not extend to nonlinear activations. This is a gap between stated scope and proved results.

### Minor

- **All experiments are on synthetically generated data.** The convergence curves in Figures 3–4 confirm theoretical predictions on synthetic instances but give no signal about whether the theory covers practically relevant settings. At minimum, a simple real-data experiment (e.g., MNIST-based data hyper-cleaning with label noise) would quantify how the orthogonality and linear-model assumptions affect practical behavior.

- **The `$\arg\min_v L_\gamma(u,v)$ is independent of $u$` assumption in Theorem 1 (Gauss-Seidel case) is a strong hidden requirement.** It is listed as a theorem hypothesis but its verification for data hyper-cleaning is deferred to Lemma 23, without an upfront warning that this is a structurally special property (it means the lower-level optimal $v$ does not depend on $u$, which constrains the class of bilevel problems to which the Gauss-Seidel version applies). The paper should flag this more prominently.

### Trivial

- Section 6 concludes "our local PL-based analysis can be extended to other penalty reformulation-based algorithms" based solely on the empirical observation that F²SA and BOME also converge in Figure 4. This is an observation, not a theoretical claim, and should be worded accordingly.

---

## Nice-to-Haves

- An experiment showing how PBGD's convergence degrades as $X_\text{trn}X_\text{trn}^\top$ deviates from diagonal (e.g., varying condition number of the off-diagonal entries) would help calibrate whether the orthogonality assumption is merely a proof artifact or reflects a genuine difficulty.
- A discussion of what class of problems the diagonal assumption actually covers and whether it can be relaxed to near-diagonal $XX^\top$ with bounded condition number would substantially strengthen the data hyper-cleaning contribution.
- Extension of the induction argument to establish PL conditions for even a single class of nonlinear models (e.g., single-layer ReLU with homogeneous data) would meaningfully extend the "pilot study."

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Harsh Critic Point 3 (PL constant positivity gap):** The critic raises concern that the PL constant $c(W)$ in Lemma 2 may vanish if all positive mismatches go to zero. The paper explicitly states (Section 5, after Lemma 2): "we can derive a uniform positive lower bound of $\mu_u := \min_{u \in \mathcal{U}} c(W_\gamma^*(u))$ based on the acute matrix perturbation theory." This is addressed in the main paper; the proof is in the appendix (stripped by the parser). Not a gap in the submitted work.

2. **Harsh Critic: "Introduction over-promises RLHF/NAS/healthcare applications."** The paper uses "pilot study" in the title and explicitly scopes to two linear problems in Section 1.1. The broad motivation is standard framing for theoretical bilevel papers and does not rise to a verifiable weakness.

3. **Harsh Critic: Cold-start of inner loop ($w^{k,0} = w^0$).** Algorithm 1 cold-starts at a fixed $w^0$; Algorithm 2 does not cold-start (no such line). The complexity bound $T_k = \mathcal{O}(\log(\epsilon^{-1}))$ is set accordingly. The question of whether warm-starting would improve constants is a nice-to-have, not a gap.

4. **Strength Finder: "Comprehensive experiments with multiple algorithms and ablations."** This is somewhat overstated—the experiments are entirely synthetic and limited to two problems. Dropped from the strengths; the experiments are supportive but not comprehensive.

---

## Novel Insights

The most genuinely novel insight in this paper is the identification that the *penalty objective* $L_\gamma(u,v)$ preserves PL-type structure even when the nested bilevel objective $F(u)$ does not, and that this is not merely a cosmetic change but reflects a fundamental difference in landscape geometry arising from the extra dimension in $v$. The paper further shows that for linear bilevel problems with MSE loss, this PL structure can be *maintained uniformly throughout the optimization trajectory* via an induction argument — converting a local, iterate-dependent guarantee into a global one. This trajectory-based verification approach, rather than verifying a static landscape condition, is the key conceptual advance over prior PL-condition-based single-level analyses, and provides a template for future work on richer model classes.

---

## Suggestions

1. **Add a prominent remark in Section 5 explicitly stating the practical implications of the diagonal $XX^\top$ assumption** — what data geometries it covers, and what is conjectured (not claimed) about the non-diagonal case.
2. **Add at least one real-data experiment** (even a small-scale one like MNIST with synthetic label noise) with a disclaimer about which assumptions are violated, to calibrate empirical behavior vs. theoretical predictions.
3. **State the independence of $\arg\min_v L_\gamma(u,v)$ from $u$ as a named assumption** in Section 3.2, not buried in the theorem statement, since it is a structurally restrictive requirement.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Avg Score | Decision |
|---|---|---|---|
| A4aG3XeIO7 | Tuning-free bilevel (stationary point) | 6.5 | Accept (poster) |
| cyPMEXdqQ2 | Constrained bilevel, gap function reformulation | 6.5 | Accept (poster) |
| 06lrITXVAx | Dropout bilevel, overfitting | 7.0 | Accept (spotlight) |
| vIHmkF5rnC | Penalty bilevel, weak experiments | 4.25 | Reject |
| SXTmAdGjlg | Adaptive bilevel mirror descent | 4.6 | Reject |

**Reasoning:** This paper achieves something qualitatively stronger than the accepted poster-level papers (A4aG3XeIO7, cyPMEXdqQ2), which only prove stationary-point convergence — this paper proves global convergence for the first time. That is a genuine step up in the quality of the theoretical result. However, the data hyper-cleaning result requires an exact pairwise orthogonality condition that is never satisfied by real data, and all experiments are synthetic. The representation learning result is technically sound within the linear overparameterized regime.

The paper is clearly above the rejected bilevel papers (vIHmkF5rnC, SXTmAdGjlg), which had soundness issues and unclear contributions. It is at or slightly above the accepted poster cluster (~6.5) given the novelty of global convergence. The orthogonality limitation and synthetic-only experiments prevent it from reaching spotlight level (~7+).

**Final assessment:** The paper makes a genuine and technically non-trivial first-of-its-kind contribution for a specific setting, with appropriate scope qualification ("pilot study"). The major limitations (orthogonal data assumption, linear models only, synthetic experiments) are real but do not invalidate the core results — they define the paper's honest scope. The contribution warrants acceptance as a poster with the expectation that authors improve the discussion of assumptions.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>