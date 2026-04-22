Now I have a thorough understanding of the paper and the reviewer comments. Let me write the final review.

## Summary

The paper proposes OBCD, a row-wise block coordinate descent method for nonsmooth composite optimization under orthogonality constraints ($\mathbf{X}^\top\mathbf{X} = \mathbf{I}_r$). At each iteration, OBCD updates $k$ rows of the solution via a constraint-preserving multiplicative scheme with $\mathbf{V} \in \text{St}(k,k)$. The key theoretical contributions are: (1) a block-$k$ stationary point (BS$_k$-point) optimality hierarchy strictly stronger than standard critical points (Theorem 3.6); (2) $\mathcal{O}(1/\epsilon)$ ergodic and KL-based non-ergodic convergence rates; and (3) a breakpoint searching method (BSM) that globally solves the $k{=}2$ subproblem for $\ell_0$-penalized objectives, enabling the stronger BS$_2$-point guarantee in practice.

## Strengths

- **Elegant constraint-preserving update scheme (Lemma 2.1):** The update $\mathbf{X}^+ = \mathbf{X} + \mathbf{U}_B(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_B^\top \mathbf{X}$ maintains feasibility for any $\mathbf{V} \in \text{St}(k,k)$, eliminating the need for retraction or projection steps that infeasible methods (ADMM, penalty methods) require. This is a clean, effective design.

- **BS$_k$-point optimality hierarchy (Theorem 3.6):** The theorem establishes a strict hierarchy {critical points} ⊇ {BS$_2$-points} ⊇ {BS$_k$-points} ⊇ {global optima} with reverse inclusions not always holding. This is a meaningful theoretical advance over prior work (Wen & Yin, 2013; Chen et al., 2020) that only guarantees convergence to standard critical points, and it provides principled justification for why block updates can escape poor critical points.

- **Breakpoint Searching Method for $k=2$ (Section 5):** The BSM reduces the $k{=}2$ subproblem with $\ell_0$ regularizer to a one-dimensional optimization over at most $2r+4$ breakpoints, enabling global subproblem solutions that realize the BS$_2$-point guarantee in practice. This is a concrete algorithmic contribution.

- **Dominant empirical performance (Table 1):** OBCD-R(id) achieves the best objective value (relative value 0.00e+00) across all 10 datasets against LADMM and SPM, and Figure 1 shows it consistently escapes poor local minima where other methods plateau.

- **Complete convergence theory:** The combination of ergodic $\mathcal{O}(1/\epsilon)$ rate (Theorem 4.2), Riemannian subgradient convergence (Theorem 4.6), and KL-based non-ergodic rates (Theorem 4.11) provides a thorough theoretical characterization.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed generality of BS$_k$-point optimality guarantee:** The paper's central claim—that OBCD converges to block-$k$ stationary points offering "stronger optimality than standard critical points" (abstract, contributions, conclusion)—only materializes when subproblems are globally solved (Asm-iii). As Remark 2.4(b) concedes, for general $k$ and $h$, subproblems can only be solved locally, yielding the same critical-point convergence as existing methods. The guarantee is realized in practice only for $k=2$ with the BSM (Section 5), or for $h=0$ with diagonal $\mathbf{Q}$ (Remark 2.4(a)). The abstract states this stronger stationarity unconditionally, and the experiments only test $k=2$, so the paper effectively proposes and validates a $k=2$ method while presenting it as a general framework. The paper should prominently qualify this gap between the general framework's theory and its practical realization.

- **Missing comparisons with the most relevant baselines:** The paper compares only against LADMM and SPM (operator-splitting methods), while the related work section discusses more directly comparable approaches: proximal gradient methods (Chen et al., 2020; Li et al., 2024) and BMM/BCD Riemannian methods (Li et al., 2024, 2023; Breloy et al., 2021) that address the same nonsmooth + orthogonality problem class. These are natural competitors whose absence significantly weakens the empirical claim of "superior performance across various tasks." Without these comparisons, the paper shows OBCD beats two older splitting methods but not the current state of the art.

### Minor

- **Error in the $h$-decomposition (Equation 10):** The paper writes $h(\mathcal{X}_B^t(\mathbf{V})) = h(\mathbf{U}_B\mathbf{U}_B^\top \mathbf{X}^t + \mathbf{U}_B\mathbf{V}\mathbf{U}_B^\top \mathbf{X}^t) = h(\mathbf{U}_B^\top \mathbf{X}^t) + h(\mathbf{V}\mathbf{U}_B^\top \mathbf{X}^t)$. From Equation (4), the correct decomposition should use $\mathbf{U}_{B^c}\mathbf{U}_{B^c}^\top$ (rows not being updated) instead of $\mathbf{U}_B\mathbf{U}_B^\top$, giving $h(\mathbf{U}_{B^c}^\top \mathbf{X}^t) + h(\mathbf{V}\mathbf{U}_B^\top \mathbf{X}^t)$. This affects the constant $\tilde{c}$: with the paper's constant, $\mathcal{K}(\mathbf{I}_k) + \tilde{c} \neq F(\mathbf{X}^t)$. Since $\tilde{c}$ is constant w.r.t. $\mathbf{V}$, the algorithm and key convergence results (which depend on $\mathcal{K}(\bar{\mathbf{V}}^t) \leq \mathcal{K}(\mathbf{I}_k)$, not on $\tilde{c}$) are unaffected, but the incorrect derivation should be corrected and the claim that $\mathcal{K}(\mathbf{I}_k) + \tilde{c} = F(\mathbf{X}^t)$ should be verified with the corrected constant.

- **Experiments only test $k=2$:** The method is presented for general $k \geq 2$ but all experiments use $k=2$. Results for $k=3,4$ (even with approximate subproblem solutions and the resulting critical-point convergence) would demonstrate whether the BS$_k$ hierarchy yields practical benefits beyond $k=2$ and whether the framework is useful beyond its currently solvable instantiation.

### Trivial
None.

## Nice-to-Haves

- Add sensitivity analysis for $\alpha$: the choice $\alpha = 10^{-5}$ is quite small, and $\alpha$ controls both the majorization quality and convergence speed.
- Include a small illustrative example where a BS$_2$-point demonstrably differs from a critical point, motivating the optimality hierarchy beyond the 2×2 examples in Remark 2.6.
- Compare with the proximal gradient and BMM methods (Chen et al., 2020; Li et al., 2024) that are discussed in the related work section.

## Removed Points

- **"First BCD method for nonsmooth orthogonality constraints" overclaim:** The harsh critic flags this as inaccurate because BMM/BCD methods (Li et al., 2024, 2023; Breloy et al., 2021) also apply BCD to this problem class. However, the paper already distinguishes its contribution as *row-wise* BCD vs. prior *column-wise* BCD, and argues that column-wise methods are limited to smooth objectives with $k=2, r=n$. The "first" claim is qualified in context (first row-wise BCD), and the distinction is valid. This is more a presentation issue than a substantive error, and Section 1.1 makes the distinction clear.

- **Sensitivity to $\alpha$ and thresholded $\ell_0$ as severe weaknesses:** The critic demands analysis of $\alpha$ sensitivity and worries about the thresholded $\ell_0$ count. These are reasonable suggestions for improvement but not weaknesses that threaten the core claims. The small $\alpha$ is standard practice in majorization-minimization methods, and the thresholded count is a numerically stable replacement widely used in practice.

- **"Time limit=30 disadvantages full-gradient methods":** This is a design choice, not a weakness. Time-based comparison is the standard approach for evaluating methods with different per-iteration costs, and the paper explicitly justifies this choice.

- **KL-based convergence results are "standard applications":** While the KL framework is well-established, applying it to derive non-ergodic convergence rates for this particular problem structure (nonsmooth composite on Stiefel manifold with BCD) requires non-trivial technical work in the sufficient decrease condition and subdifferential analysis. The results are not mere boilerplate applications.

## Novel Insights

The paper reveals an interesting structural property: the row-wise update $\mathbf{X}^+ = \mathbf{X} + \mathbf{U}_B(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_B^\top \mathbf{X}$ with $\mathbf{V} \in \text{St}(k,k)$ preserves the entire Stiefel manifold constraint globally (not just along a tangent direction), yielding feasible iterates at every step. This contrasts with infeasible methods (ADMM, penalty) and even with feasible retraction-based methods that only approximate the manifold locally. The resulting BS$_k$-point hierarchy genuinely captures a notion of "how locally optimal are you if you can permute/transform $k$ rows simultaneously," which is strictly stronger than critical-point optimality and aligns with the empirical observation that OBCD escapes poor local minima. However, the practical realization of this hierarchy currently depends on globally solving the St(k,k)-constrained subproblem, which is only achieved for $k=2$.

## Suggestions

- Prominently qualify the BS$_k$-point claim: state clearly in the abstract and introduction that the stronger optimality guarantee requires global subproblem solutions, and that practical realizations currently exist only for $k=2$ with BSM and for smooth cases.
- Correct the $h$-decomposition in Equation (10) by replacing $\mathbf{U}_B\mathbf{U}_B^\top$ with $\mathbf{U}_{B^c}\mathbf{U}_{B^c}^\top$ and updating $\tilde{c}$ accordingly. Verify that all subsequent proofs use the correct constant.
- Add comparisons with at least one proximal gradient method and one BMM method for nonsmooth Stiefel optimization to strengthen the empirical contribution.

## Score and Decision Calibration

Anchor papers reviewed:
- **xGvPKAiOhq** (avg 8.0, Accept spotlight): Matrix sensing with over-parameterization on Stiefel-like manifold structure; strong theory with clean results.
- **5mtwoRNzjm** (avg 6.5, Reject): Landing method for generalized Stiefel manifold optimization; good theory but missing baselines, rejected despite one reviewer giving it 10.
- **K98byXpOpU** (avg 5.0, Reject): Nonsmooth bilevel optimization with convergence analysis; unsound technical analysis and limited experiments.
- **ogKE7LcvW6** (avg 7.0, Accept poster): Nonsmooth stationary points analysis in ReLU networks; novel theoretical contribution with focused scope.
- **1NYhrZynvC** (avg 2.5, Reject): Overclaimed convergence guarantees only work under very special unstated conditions; fundamental issues.
- **c2OtbtZXFC** (avg 4.75, Withdrawn/Reject): Stiefel manifold optimization with poor baseline comparison and questionable novelty.

Compared to these anchors: This paper has a more substantial theoretical contribution than c2OtbtZXFC (4.75) or K98byXpOpU (5.0), with a genuine BS$_k$ optimality hierarchy and BSM algorithm. However, it shares with 5mtwoRNzjm the issue of missing relevant baselines and some overclaiming. The overclaiming of generality (general framework but only k=2 validated) is more significant than in ogKE7LcvW6 (score 7), where the scope was appropriately narrow. The paper's core contribution for k=2 with BSM is solid, but the presentation overreaches. The decomposition error in Eq (10) is non-fatal but concerning for proof rigor.

I place this paper between the mid-range anchors (5-6.5) with clear theoretical merit but two major issues (overclaimed generality and missing baselines). The solid k=2 contribution justifies a score above the pure-reject threshold but below clear-accept territory.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>