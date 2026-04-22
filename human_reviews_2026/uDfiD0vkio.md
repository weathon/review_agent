# A Proximal Stochastic Gradient Method for Doubly-regularized Spectral Risk Minimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Spectral risk minimization (SRM) is an important category of distributionally robust optimization. Recent works in this field elaborate on either distribution shift regularization (DSR) on the spectrum or non-differentiable regularization (NDR) on the parameters. However, few methods can simultaneously handle double regularization. The main difficulty lies in suppressing the bias and variance of the stochastic gradient when double regularization is present. To solve this problem, we develop a novel proximal stochastic gradient method (PSG-SRM) that simultaneously handles double regularization, reduces bias and variance along with iterations, and achieves linear convergence. It has lower computational complexity than two state-of-the-art methods that handle DSR or NDR separately. Experimental results indicate that it achieves competitive performance in both regression and classification tasks, and shows stable performance with respect to randomness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the Spectral risk minimization (SRM) problem with double regularizations: distribution shift regularization (DSR) and  non-differentiable regularization (NDR). A new method, proximal stochastic gradient method (PSG-SRM), is proposed to solve the problem and both theoretical and empirical results are shown.

### Strengths
1. This paper solves the challenging double regularization problems. With the non-differentiable regularization, it is challenging to apply the existing bias and variance analysis skills to the problem. To solve this problem, this paper proposes a new method, PSG-SRM, to solve the problem.

2. The authors provide the theoretical analysis for the proposed algorithm, which achieves  lower computational complexity than the two state-of-the-art methods that handle DSR or NDR separately.  Also the numerical results show that the proposed algorithm has good empirical performance.

### Weaknesses
1. The presentation of this paper is unclear and difficult to follow. The introduction contains too many equations and definitions, which would be more appropriately placed in the Preliminaries section. Additionally, there are extensive discussions and comparisons with existing works on objective formulations and convergence rates; it would be clearer to summarize these in a table for better illustration.

2. Some important definitions, such as those on lines 51 and 193, are located in the appendix. Since the Lipschitz property plays a crucial role in the theoretical proofs and lemmas, its definition should be moved to the main text rather than being placed in the appendix.

### Questions
While the main issue of this paper lies in its presentation, I also have a few technical concerns:

1. The paper claims that “standard proximal (stochastic) gradient methods like SOREL can handle NDRs but cannot handle DSR.” However, in Section 2.3, SOREL formulates the problem as a minimax optimization. Generally, with DSR regularization, the objective should be concave in  q, which would typically make the problem easier to solve rather than more difficult.

2. It would be helpful if the authors could provide a table summarizing the assumptions used in SOREL, PROSPEC, and the proposed method. Additionally, please clarify whether achieving similar linear convergence in this paper incurs any extra computational cost compared to PROSPEC.

3. From an algorithmic standpoint, is there any substantial difference between the proposed PSG-SRM and PROSPEC, aside from the proximal stochastic gradient step on  w?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PSG-SRM to solve distribution shift regularization (DSR) problems with nonsmooth regularization terms. Unlike previous methods, the proposed approach can simultaneously handle problems that involve both a nonsmooth convex regularizer and distribution shift regularization. The authors prove the linear convergence of the algorithm theoretically. Experimental results demonstrate that the algorithm exhibits stronger robustness than other methods in most cases.

### Strengths
This paper introduces an interesting problem related to DRO and fairness machine learning. It is well written and easy to follow. The authros  develop an Lyapunov function to prove the convergence of their proposed algorithm. The empirical validation uses a reasonable diversity of datasets and kinds of spectral risks.

### Weaknesses
1. The proposed algorithm is mainly based on Prospect, and it seems that the only modification is replacing the gradient descent on parameter $w$ with proximal gradient descent.

2. The authors should clearly specify the assumptions used in each Lemma and Theorem.

3. In Line 192, the authors assume that each loss function is both strongly convex and Lipschitz continuous. However, these two assumptions cannot hold simultaneously unless the function is restricted to a compact set.

4. In Line 325, the authors state that the proximal gradient method can only achieve a sublinear convergence rate because the regularizer is nonsmooth or non-contractive. However, in [1], the authors study the spectral risk minimization problem without distribution shift regularization, which is a strongly convex–concave minimax problem. The author in [1] claims that the lower complexity bound is sublinear. Could the authors further clarify this point?

5. The authors claim that their algorithm has an advantage in runtime. However, according to Table A.2, its runtime does not show a significant improvement compared to Prospect.


Reference

[1] Ge, Y., & Jiang, R. (2024). SOREL: A Stochastic Algorithm for Spectral Risks Minimization. International Conference on Learning Representations, abs/2407.14618. https://doi.org/10.48550/arXiv.2407.14618

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PSG-SRM, a novel proximal stochastic gradient method for optimizing spectral risk measures. The algorithm incorporates double regularization, simultaneously reduces bias and variance over iterations, and achieves linear convergence under appropriate assumptions. Compared to the prior work PROSPECT, PSG-SRM enjoys lower per-iteration computational complexity, making it theoretically more scalable to large datasets. Empirical results demonstrate that PSG-SRM delivers consistently better and more stable optimization performance.

### Strengths
1. The paper provides a rigorous theoretical analysis showing that PSG-SRM achieves linear convergence, and clearly establishes its lower computational complexity compared to PROSPECT.

2. The experimental evaluation is conducted on widely accepted benchmarks in the field, and effectively demonstrates that PSG-SRM attains superior and more stable optimization performance in practice.

### Weaknesses
1. Readability: The writing suffers from several issues, including:
1.1 Excessive use of abbreviations (e.g., SRM, DSR, NDR, DRO, PAV) without sufficient introduction or justification;
1.2 Confusing mathematical notation (e.g., Equation 5 uses notation for DSR and NDR that appears as if it belongs in a denominator, which is misleading);
1.3 Arbitrary use of boldface text, which disrupts the flow and professionalism of the presentation.

2. Mismatch between theory and experiments: The theoretical contribution centers on faster convergence (linear rate) and reduced computational complexity relative to PROSPECT. However, the experiments only report final performance metrics and do not validate the claimed convergence speed or iteration complexity. For instance, there is no plot showing loss vs. epoch or wall-clock time.

3. Weak motivation for the core formulation: The key contribution (Equation 5) appears to be a straightforward combination of PROSPECT’s objective with an additional strongly convex, non-differentiable regularization. The explanation “Since NDRs are widely used to handle complicated tasks, …, it motivates us to unify NDR and DSR…” is overly heuristic. The authors should provide a deeper rationale for this specific unification.

### Questions
1. The paper theoretically establishes that PSG-SRM converges faster than existing methods. Can this be empirically verified? For example, does PSG-SRM reach the same objective value or test performance in fewer epochs or less wall-clock time?

2. The Introduction introduces five distinct optimization objectives, which feels overwhelming. In contrast, the cited baselines (PROSPECT, LSVRG, SOREL) typically frame their problems using only two objectives. Could the exposition be simplified to better highlight the specific problem PSG-SRM addresses?

3. Section 3.1 is dedicated solely to prove the smoothness of f. However, this property is standard in stochastic optimization, and the section does not appear to add novel insight. Is this section necessary?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies doubly-regularized spectral risk minimization (SRM) problems, which combine distribution-shift regularization (DSR) on the spectrum and a potentially non-differentiable parameter regularizer (NDR). The authors propose PSG-SRM, a proximal stochastic gradient algorithm that integrates a variance and bias control scheme (extended from PROSPECT-style estimators), leverages the non-expansiveness of proximal operators to handle NDR, and establishes linear convergence under the assumption that each sample loss $\ell_i$ is strongly convex. The paper claims superior computational complexity compared to methods that handle only DSR or NDR individually, and presents experimental results on regression and classification tasks showing competitive and stable performance.

### Strengths
1. The paper addresses a practically relevant problem by integrating both distributional robustness and nonsmooth regularization, filling a gap in existing methodology.
2. Theoretical contributions include lemmas bounding bias and variance and the construction of a Lyapunov function to establish linear convergence.
3. Experiments cover regression and classification tasks with multiple baselines, demonstrating stable performance across settings.

### Weaknesses
1. The proposed algorithm is incremental compared to PROSPECT, with its main novelty lying in the proximal stochastic gradient step.
2. On Page 4, Line 191, the authors assume that each $\ell_i(w)$ is $\mu$-strongly convex, $G$-Lipschitz continuous, and $L_\ell$-smooth. These conditions are often violated in real-world loss functions. The paper should more clearly discuss the practical regimes where these assumptions hold.
3. In Table 3, the evaluation of out-of-distribution (OOD) performance uses inconsistent metrics: "worst group test error" is reported for the 'amazon' dataset, while "median group test error" is used for 'iwildcam'. This selective reporting impedes direct and rigorous comparison and may suggest cherry-picking of favorable metrics.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
