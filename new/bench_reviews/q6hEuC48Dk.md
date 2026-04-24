## Summary

The paper introduces the Radial Basis Operator Network (RBON), a shallow operator-learning architecture built entirely from radial basis functions. RBON uses fixed RBF features with centers determined by K-means clustering and solves for network weights exactly via the Moore–Penrose inverse. The authors claim that RBON achieves exceptionally low test errors on standard PDE benchmarks, generalizes to out-of-distribution inputs from different function classes, and is the first operator network capable of learning with complex-valued frequency-domain inputs (F-RBON).

## Strengths

- **Conceptual simplicity and compactness.** RBON employs at most 15 nodes per sub-network (capping hidden-layer products at 225) and trains non-iteratively via a linear solve. This is a genuinely different design point from deep neural operators and could be valuable for interpretability and fast training (Section 2.2).
- **Strong Burgers OOD benchmark.** Testing on polynomial initial conditions $u_0(x)=bx(x-1)$ after training exclusively on sinusoids $u_0(x)=a\sin(\pi x)$ is a more challenging OOD test than simple parameter extrapolation and distinguishes the experimental suite from much of the operator-learning literature (Section 3.1.2).
- **Code availability.** The authors provide an anonymous code repository, which supports reproducibility.

## Weaknesses

### Fatal
None.

### Major
- **Corollary 2.1.1 for normalized RBON is vacuous.** Equation 3 defines $\tilde{\xi}_i^k = \xi_i^k \sum_{i=1}^M\sum_{k=1}^N g(\lambda_i\|u^m-\mu_{ik}^m\|)g(\omega_k\|\mathbf{y}-\mathbf{c}_k\|)$, which depends explicitly on the input pair $(u^m,\mathbf{y})$. Substituting this into Equation 4 causes the normalizing sum to cancel exactly, yielding $\sum_{i,k} \xi_i^k g(\cdot)g(\cdot)$ — i.e., the original unnormalized approximation from Theorem 2.1. The corollary therefore does not establish a normalized network with constant parameters; it merely restates the theorem using input-dependent pseudo-weights. As written, the alleged theoretical extension is invalid.
- **The core training heuristic lacks theoretical or empirical justification.** For each query point $y_\ell$, the authors solve an independent least-squares problem $\xi_\ell^T\Phi_\ell = [v_1(y_\ell),\dots,v_J(y_\ell)]$ and then element-wise average the resulting $\xi_\ell$ vectors across all $L$ query points (Section 2.2). Averaging the solutions of $L$ independent regressors is not the minimizer of any global loss over the operator space, and the paper provides no sensitivity analysis or ablation comparing this heuristic against a principled global objective (e.g., stacking all $(y_\ell,v_j(y_\ell))$ pairs to learn a single $\xi$). This is a significant methodological gap for the paper’s central algorithmic contribution.
- **Baseline comparisons are inadequately documented and suspiciously poor.** LNO achieves $5.6\times10^{-1}$ in-distribution error on the Wave equation (Table 1) — roughly 560$\times$ worse than FNO ($9.9\times10^{-4}$) on the same benchmark and inconsistent with LNO’s status as a benchmark standard. The authors note that early stopping improved DeepONet’s OOD errors but do not report those improved numbers. No hyperparameters, training budgets, or optimization details are provided for any baseline. These omissions severely undermine the credibility of the headline claim that RBON outperforms state-of-the-art neural operators by orders of magnitude.
- **High variance from K-means initialization undermines reported results.** Section 4 acknowledges that K-means convergence can cause errors that “differ by several orders of magnitude between runs.” Table 1, however, reports only point estimates with standard-error margins across the test set, not across training runs. The reader cannot tell whether the reported low-error values are reproducible or cherry-picked.

### Minor
- **The CO$_2$-to-temperature task is finite-dimensional vector regression, not operator learning.** The inputs are 12 monthly CO$_2$ measurements and the outputs are 12 monthly temperatures (Section 3.2). Because $t\in\{1,\dots,12\}$, the mapping $u_n(t)\mapsto T_n(t)$ is a 12-dimensional vector-to-vector regression problem. Framing it as a scientific application of infinite-dimensional *operator* networks is misleading.
- **Beam equation results report meaningless precision.** The Euler-Bernoulli beam equation is linear, so exact linear regression is expected to perform extremely well. More problematically, the reported in-distribution error is $4.1\times10^{-8}\pm3.3\times10^{-6}$: the margin of error is two orders of magnitude larger than the point estimate, and the 95% confidence interval comfortably includes values comparable to competing methods. The reported sub-$10^{-7}$ precision is therefore not meaningful.
- **Abstract overstates the breadth of OOD generalization.** The abstract claims small error on “OOD data from entirely different function classes,” but only the Burgers benchmark uses a truly different base function (polynomial vs. sinusoid). The Wave and Beam OOD experiments are parameter extrapolations of the same base functions.

### Trivial
- Figure 1 labels the product of two vectors as a Kronecker product; for vectors this is simply an outer product.
- The relation between the scalar center $\mu$ in the general RBF definition and the $m$-dimensional vector $\mu_{ik}^m$ in Theorem 2.1 is not explicitly explained.

## Nice-to-Have
- A principled global least-squares ablation comparing per-query-point averaging against solving for a single weight vector that minimizes error across all query points simultaneously.
- Condition-number analysis of the $\Phi_\ell$ matrices and the effect of Tikhonov regularization, especially given the extremely small reported errors on the linear Beam problem.
- Visualization of how the per-query weights $\xi_\ell$ vary across locations $y_\ell$ to justify (or refute) the averaging heuristic.

## Removed Points
These points are flagged to be removed, treat them with caution.
- The claim that the method “is not learning a unified operator representation but an ad-hoc aggregate of pointwise regressors” is factually incorrect: the final weight vector $\xi$ is shared globally across all query points, even if the training procedure is heuristic.
- The criticism that FNO already learns in the frequency domain misunderstands the paper’s claim. FNO applies Fourier transforms internally but learns a spatial-to-spatial operator; F-RBON learns an operator whose inputs and outputs are frequency-domain (complex-valued) functions. The distinction is narrow but technically valid.
- “The paper should not be accepted without a complete reformulation” is excessive; the core architecture and several empirical results are sound, but the training procedure and theory need significant refinement.
- Various formatting, typo, and appendix-related nitpicks.

## Novel Insights
None beyond the paper's own contributions. The paper’s most genuinely novel observation is that a compact, shallow RBF architecture with exact linear training can achieve competitive or better empirical performance than deep iterative operators on some standard PDE benchmarks, provided the training heuristic is replaced with a more principled objective.

## Suggestions
- Remove or correct Corollary 2.1.1. If a normalized representation with fixed weights is desired, it must be derived independently, not by defining input-dependent pseudo-weights.
- Replace the per-query weight averaging with a single global least-squares objective, or at minimum provide empirical evidence that the $\xi_\ell$ vectors are sufficiently similar across query points to justify averaging.
- Report cross-run statistics (mean, median, best-of-$k$) accounting for K-means variance, or explicitly state that best-of-$k$ selection was used for Table 1.
- Re-run or thoroughly document baseline hyperparameters and training protocols, and explain the anomalously poor LNO Wave result.

## Score and Decision

**Calibration anchors used:**
- *Neural Spectral Methods* (avg 6.75, Accept poster): solid theory, strong baselines, rigorous spectral formulation. RBON is clearly below this.
- *MgNO* (avg 6.50, Accept poster): novel UAP and multigrid parameterization, strong experiments. RBON lacks this theoretical rigor.
- *clawNOs* (avg 5.00, Reject): decent empirical results but concerns about missing baselines and novelty. RBON has a more severe theoretical flaw (vacuous corollary) and more suspicious baseline results.
- *KNO* (avg 4.75, Withdrawn): kernel-based operator with comparison concerns and limited problem complexity, but solid methodology. RBON’s baseline issues and math error make it comparable or weaker.
- *FEONet* (avg 3.00, Withdrawn): fundamentally not genuine operator learning. RBON is substantially better.
- *Sub-Scaling Law* (avg 4.50, Withdrawn): massive empirical study with severe theoretical limitations and misleading claims. RBON has a narrower scope but a concrete mathematical error.

RBON has real empirical successes (especially the Burgers OOD test) and an interesting compact-architecture story, but it is undermined by a mathematically vacuous corollary, an unjustified core training heuristic, suspiciously poor baseline results, and unquantified K-means variance. These issues place it below the medium-quality operator-learning papers (clawNOs, KNO) and well below the strong accept band. It is, however, a more focused operator-learning contribution than the very low-scoring anchors. A score of **4.5** reflects serious methodological and theoretical gaps that would need to be addressed for acceptance.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>