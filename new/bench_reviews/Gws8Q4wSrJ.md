Now I have all the information I need. Let me write the consolidated review.

## Summary

The paper proposes OBCD, a block coordinate descent method for nonsmooth composite optimization under orthogonality constraints (Stiefel manifold). OBCD updates $k$ rows of the solution per iteration via a constraint-preserving scheme, solves small subproblems on $\text{St}(k,k)$, and defines block-$k$ stationary points (BS$_k$-points) that are provably stronger than standard critical points when subproblems are solved globally. A breakpoint searching method for exactly solving subproblems when $k=2$ and KL-based convergence rate guarantees are also provided.

## Strengths

- **Constraint-preserving row-wise update scheme (Lemma 2.1):** The update $\mathbf{X}^+ = \mathbf{X} + \mathbf{U}_B(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_B^\top \mathbf{X}$ with $\mathbf{V} \in \text{St}(k,k)$ cleanly maintains Stiefel manifold feasibility while updating only $k$ rows. This generalizes prior column-wise BCD methods (Shalit & Chechik, 2014) that were restricted to $k=2, r=n$.

- **Inclusion of Jacobi reflections alongside Givens rotations (Lemma 2.5, Remark 2.6):** The paper shows that $\text{St}(2,2)$ requires both rotations and reflections, and provides concrete 2×2 examples where using only rotations yields strictly suboptimal solutions. This corrects a gap in prior work.

- **Breakpoint searching method for $k=2$ (Section 5):** The BSM finds the global optimum of the nonsmooth subproblem by checking at most $2r+4$ breakpoints for $\ell_0$ regularization (Lemma 5.1 and subsequent analysis). This is efficient, non-trivial, and enables OBCD to actually achieve the stronger BS$_2$-point stationarity in practice.

- **BS$_k$-point optimality hierarchy (Theorem 3.6):** The hierarchy {critical points} ⊇ {BS$_2$-points} ⊇ {global optima} with strict inclusion possible provides a useful framework for reasoning about stationarity quality. This is a meaningful theoretical advance.

- **Comprehensive convergence theory (Section 4):** Both ergodic ($O(1/\epsilon)$, Theorems 4.2, 4.6) and non-ergodic KL-based rates (Theorem 4.11: finite/linear/sublinear convergence depending on KL exponent) are established. The KL-based last-iterate rates go beyond what is typical for BCD on Riemannian manifolds.

## Weaknesses

### Fatal
None.

### Major

- **Missing contemporary baselines undermines empirical claims.** The experiments (Section 6) compare OBCD only against LADMM and SPM, two operator-splitting methods from circa 2014. The related work section itself cites more recent and directly competitive methods—ManPG/AManPG (Chen et al., 2020; Li et al., 2024) and BMM-based Riemannian methods (Li et al., 2024; Breloy et al., 2021; Cheung et al., 2024)—that solve the same nonsmooth composite problem on the Stiefel manifold. None appear as baselines. The abstract claims "superior performance," and Section 6 claims OBCD "surpasses existing solutions," but these claims cannot be evaluated without comparisons to the most relevant contemporary methods. This is the difference between demonstrating an advance and demonstrating that one method beats two outdated strawmen.

- **Error in the majorization derivation (Inequality (10)).** The paper writes: $h(\mathcal{X}_B^t(V)) = h(\mathbf{U}_B\mathbf{U}_B^\top \mathbf{X}^t + \mathbf{U}_B\mathbf{V}\mathbf{U}_B^\top \mathbf{X}^t)$. However, from Equation (4), $\mathcal{X}_B^t(V) = (\mathbf{U}_B\mathbf{V}\mathbf{U}_B^\top + \mathbf{U}_{B^c}\mathbf{U}_{B^c}^\top)\mathbf{X}^t$, so the first term should involve $\mathbf{U}_{B^c}$, not $\mathbf{U}_B$. This propagates to the constant $\tilde{c}$, which incorrectly uses $h(\mathbf{U}_B^\top \mathbf{X}^t)$ instead of $h(\mathbf{U}_{B^c}^\top \mathbf{X}^t)$, and to the first line of (10), which is missing the $h(\mathbf{U}_{B^c}^\top \mathbf{X}^t)$ constant term entirely. Fortunately, the subproblem $\mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})$ (which contains only $\mathbf{V}$-dependent terms) is unaffected, and the sufficient decrease condition and convergence results can be derived correctly. Nevertheless, this error in the paper's main derivation raises concerns about proof rigor in the appendix, particularly for the more delicate KL-based convergence results.

- **"Stronger optimality" claim is conditional on globally solving subproblems, which is only guaranteed for $k=2$; this conditionality is not disclosed in the abstract.** The abstract states that OBCD's limiting points "offer stronger optimality than standard critical points" without qualification. However, Definition 3.5 requires globally minimizing $\mathcal{K}(\cdot; \tilde{\mathbf{X}}, \mathbb{B})$ for all $\mathbb{B}$, and the paper provides a global subproblem solver only for $k=2$ (Section 5). For $k>2$ with nonsmooth $h$, Remark 2.4(b) concedes that "strong optimality may be compromised"—meaning OBCD then converges to the same critical points as other methods. The abstract and introduction should clearly state this scope limitation.

### Minor

- **All experiments use $k=2$ despite the method being presented for general $k \geq 2$.** The paper presents OBCD for arbitrary $k$ but every experiment uses the breakpoint searching method that requires $k=2$. Whether OBCD with $k>2$ (using local subproblem solutions) still outperforms baselines is untested, leaving the generality of the method empirically unsupported.

- **C++/MATLAB implementation asymmetry complicates timing comparisons.** The breakpoint searching is implemented in C++ (due to inefficient element-wise loops in MATLAB), while baselines use MATLAB. The paper argues this is fair because baselines use optimized BLAS/LAPACK, but the fundamental operations differ (element-wise loops vs. matrix algebra), making this defense only partially convincing. Reporting per-iteration flop counts would strengthen the comparison.

- **Thresholded $\ell_0$ norm in experiments.** The experiments use a thresholded $\ell_0$ (counting elements with $|X_{ij}| > 10^{-6}$) rather than the actual $\ell_0$ norm. While standard practice for numerical stability, this means the algorithm optimizes a slightly different objective than what the theory covers. The discrepancy should be acknowledged more explicitly.

- **The cyclic strategy convergence bound (Theorem 4.2(c)) involves $C_n^k = \binom{n}{k}$**, making it vacuous for realistic $n$. While the random strategy bound avoids this, the cyclic bound's practical uselessness should be noted.

- **BS$_k$-point definition is tied to the majorization surrogate $\mathcal{K}$, not the original function $F$.** A BS$_2$-point means $\mathbf{I}_k$ minimizes $\mathcal{K}(\cdot; \tilde{\mathbf{X}}, \mathbb{B})$ for all $\mathbb{B}$—this does not mean $\mathbf{I}_k$ minimizes $F(\mathcal{X}_\mathbb{B}(\cdot))$ for all $\mathbb{B}$, due to the majorization gap. The paper does not quantify this gap.

### Trivial
None.

## Nice-to-Haves

- Compare against ManPG (Chen et al., 2020), AManPG (Li et al., 2024), and recent BMM/Riemannian BCD methods to validate empirical claims against the state of the art.
- Report results for $k > 2$ to empirically validate the general-$k$ formulation.
- Include nonnegative PCA and $\ell_1$-SPCA results in the main text, not just the appendix, to demonstrate generality of the nonsmooth term handling.
- Quantify the gap between BS$_2$-points and "true" block-2 stationary points of $F$.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Small computational footprint" claim critique (harsh critic):** The harsh critic argues that accessing $k$ rows of $\nabla f(X) = -CX$ still requires $O(nk)$ work, and cycling through all $C_n^k$ blocks makes total cost comparable to full-gradient methods. However, OBCD does not cycle through all $C_n^k$ blocks per "effective iteration" — it selects one block per iteration (randomly or greedily), so the per-iteration cost is genuinely $O(nk)$, which is much cheaper than the $O(nr)$ full-gradient cost when $k \ll r$. The claim is reasonable as stated.

- **KL convergence results "follow standard templates" (harsh critic):** The harsh critic dismisses the KL-based results as following standard templates from Attouch et al. (2010). While the proof technique is indeed standard, applying it to the specific BCD-on-Stiefel-manifold setting with the particular sufficient decrease structure is non-trivial and constitutes a legitimate contribution.

- **OBCD-R(id) achieving 0.00e+00 on all datasets is "suspicious" (harsh critic):** The harsh critic suggests the uniform dominance could reflect experimental confounds rather than genuine superiority. While the concern about baselines being weak is valid, the 0.00e+00 values simply mean OBCD found the best objective among all compared methods on every dataset—which is plausible when competing only against weaker methods.

- **Formatting/presentation nitpicks (harsh critic):** Various suggestions about visualization and case studies (plotting final objective vs. $k$) are nice-to-haves rather than substantive weaknesses.

## Novel Insights

The most insightful observation across the reviews is that the paper's theoretical and empirical contributions are mismatched in scope: the strongest theoretical result (BS$_2$-point optimality) is achieved only when $k=2$ with globally solved subproblems, and all experiments use exactly this setting. For the general-$k$ case that the paper claims to address, both the stronger optimality guarantee and the exact subproblem solving are unavailable, reducing OBCD to yet another method converging to standard critical points. The paper would be more honest and potentially stronger if framed primarily as a $k=2$ method with a path to generalization, rather than presenting a general framework and then only validating the special case.

## Suggestions

- Add comparisons to at least ManPG and one BMM-based method as baselines; this is the single most important improvement for the empirical contribution.
- Add a brief remark in the abstract or introduction explicitly noting that global subproblem solving (and hence the stronger BS$_k$-point guarantee) is currently implemented only for $k=2$, with local solutions for $k>2$ yielding standard critical-point convergence.
- Correct the $h$-decomposition in the derivation of Inequality (10): replace $\mathbf{U}_B\mathbf{U}_B^\top$ with $\mathbf{U}_{B^c}\mathbf{U}_{B^c}^\top$ and update $\tilde{c}$ accordingly.

## Calibration

**Anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Probabilistic Geometric PCA | mkDam1xIzW | 7.33 | Novel method extending PCA to manifolds with EM derivation; clean theory, limited comparisons but acceptable. Our paper has comparable novelty but weaker empirical validation and a derivation error. |
| Prodigy (parameter-free learner) | WpQbM1kBuy | 4.25 | Missing key online learning baselines, invalid lower bounds, overclaimed novelty. Our paper's situation is similar (missing key baselines, derivation error, overclaimed scope) but has more genuine algorithmic contributions. |
| Adaptive Moment via Preconditioner Diagonalization | NdNuKMEv9y | 4.00 | Missing second-order baselines, incorrect theoretical claims. Our paper is somewhat stronger: the derivation error doesn't invalidate results, and the algorithmic contributions (breakpoint searching, BS$_k$ framework) are more substantial. |
| Offline Optimization via Generalized Diffusion | K9Elg2JrvY | 5.67 | Incorrect derivations but engineering improvements. Our paper has genuine theoretical contributions beyond engineering, but the missing baselines are a comparable weakness. |
| Exact Linear-Rate GD | 1NYhrZynvC | 2.50 | Fundamentally flawed theory (depends on unique x*), no proper comparisons. Our paper is clearly stronger—our derivation error is fixable and non-fatal. |

Our paper sits between the 4.0–5.7 range of the medium-band anchors. It has genuine contributions (BS$_k$ framework, breakpoint searching, constraint-preserving update) that the low-band papers lack, but the missing contemporary baselines and derivation error prevent it from reaching the high band.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>