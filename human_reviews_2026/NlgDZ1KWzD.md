# On the Convergence of FedProx with Extrapolation and Inexact Prox

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Enhancing the FedProx federated learning algorithm (Li et al., 2020) with server-side extrapolation, Li et al. (2024a) recently introduced the FedExProx method. Their theoretical analysis, however, relies on the assumption that each client computes a certain proximal operator exactly, which is impractical since this is virtually never possible to do in real settings. In this paper, we investigate the behavior of FedExProx without this exactness assumption in the smooth and globally strongly convex setting. We establish a general convergence result, showing that inexactness leads to convergence to a neighborhood of the solution. Additionally, we demonstrate that, with careful control, the adverse effects of this inexactness can be mitigated. By linking inexactness to biased compression (Beznosikov et al., 2023), we refine our analysis, highlighting robustness of extrapolation to inexact proximal updates. We also examine the local iteration complexity required by each client to achieved the required level of inexactness using various local optimizers. Our theoretical insights are validated through comprehensive numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents an inexact analysis on the convergence of the FedExProx (FedProx with extrapolation, Li et al., 2024a) algoithm under practical inexact updates. For smooth and strongly convex problems, it shows that inexact updates lead to convergence to a solution neighborhood, and relative approximation eliminates bias for exact convergence. The authors further refined their analysis by linking proximal inexactness to biased compression theory, validating the robustness of extrapolation even with imperfect client computations. The theoretical findings are complemented with a set of empirical evidence on federated quadratic optimization and CNN training tasks for validation.

### Strengths
1. The paper addresses a practical gap of FedExProx by relaxing the unrealistic exact proximal operator assumption, establishing convergence under inexact updates, which bridges theory and practical FL applications.

2. The paper is generally well organized and clearly presented, with sufficient technical details and proactive supplementary analyses offered to enhance readability and applicability.

3. The idea of linking proximal inexactness to biased compression is somewhat interesting, and it turns out to be useful for showing the robustness of extrapolation to inexact proximal operator evaluations.

### Weaknesses
The major concern goes to the novely of analysis and significance of results, given that this work essentially represents a theoretical contribution to FL. 

1. The convergence analysis is incrementally novel, mostly extending existing proof techniques without any particularly new ideas/tools developed. While the core results are intuitive and interesting, they are not expected to generate significant impact on FL research, both in thoery and practice. 

2. The analysis relies on overly strong assumptions, such as smoothness, global strong convexity, and shared client solutions, which rarely hold in real-world settings (e.g., non-convex, non-smooth deep federated learning). These assumptions, especially when combined, greatly limit the generalizability of the results.

3. The provided experimental study is illustrative but rather weak, in the sense that it fails to align with real scenarios and theoretical scope. Concerning the experiment on quadratic programming (in the main text), only synthetic data is used without comparisons to closely relevant baseline methods like FedAvg (which also inexactly minimizes local objectives with multiple iterations of SGD). Concerning the DL experiments on ResNet training (in the appendix), there exists a clear disconnect between theory and experiment as the key assumptions like smoothness/strong convexity are not fulfilled by the considered DL models. Such a gap weakens the convincingness of this group of experimental results, though they were observed in a more realistic setting.

### Questions
Is there any way to analyze the (inexact) FedExProx method in non-convex (or non-smooth) settings that are commonly considered in FL studies?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies FedExProx, an existing proximal-point method for federated optimization, with inexact local solvers. The analysis of the method is provided under the assumptions of: 1) individual convexity and smoothness, 2) global strong convexity, and 3) Interpolation. Finally, the number of local gradient oracle calls required to achieve the established communication complexity is provided for GD and AGD.

### Strengths
1. The motivation of the paper is clear. The first paper that analyzes FedExProx assumes exact subproblem solvers, which cannot be implemented. 
2. The theory is backed by reasonable proofs.

### Weaknesses
The contribution is overall a bit limited. 

> Scope 

The main contribution of the paper is to extend the analysis of FedExProx to allow inexact local solutions in the proximal steps. This has already been partially done in [1] (Appendix E) using the definition of Absolute approximation, under PL conditions.

FedExProx was originally proposed and analyzed in [1,2]. These papers also study client sampling, adaptive stepsize, and demonstrate the benefits of extrapolation in terms of communication speed-up under interpolation.

In contrast, the current submission focuses only on the strong-convexity-with-interpolation setting (full participation and constant stepsize), which is a bit more restrictive. (Note that the current theory breaks when $\mu \to 0$.) Moreover, the proof strategy for the outer iterations (communication) follows [1] almost directly, relying on the (S)GD reformulation and Moreau envelope. The main difference is the addition of an error term due to the inexact proximal-point step. 

> Novelty and results. 

FedExProx can be reformulated as a standard GD iteration (eqns (10)–(11)), with the only change being that the gradient is now biased. The bias term appears in eqn (11), and the analysis follows that of [1, 2] by additionally bounding this bias under the stated assumptions. These error terms then accumulate and appear in the final convergence rate. In this sense, the novelty is limited to handling an additional and simple bias term.

Furthermore, the analysis of GD (or SGD) with biased gradients — and its connection to compression — is well-established in the literature [3]. More connections and better theory can be done under approximately smooth assumptions [4], which is not considered here.
 
>  Comparisons with other methods

Given that this work targets federated optimization, the comparisons with other FL methods are a bit limited, both in theory and in experiments. Several state-of-the-art federated proximal-point methods can achieve stronger convergence guarantees (including acceleration), better local computation complexity, under weaker assumptions. The discussions about other federated proximal-point methods, except FedProx, seem to be missing.





[1] The Power of Extrapolation in Federated Learning, Neurips 2024.

[2] Tighter Performance Theory of FedExProx, arxiv 2024. 

[3] On the Convergence of SGD with Biased Gradients, ICML workshop 2020.

[4] First-order methods of smooth convex optimization with inexact oracle, Mathematical Programming 2013

### Questions
1. It would be great to validate the theory by comparing AGD against GD as different local solvers in numerical experiments.

### Soundness
3

### Presentation
2

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
The paper presents an improved convergence analysis of the FedExProx algorithm by removing the assumption that the proximal operator is exactly computed. FedExProx is a recently proposed extension of the FedProx algorithm that combines the proximal local updates of FedProx with the global stepsize schedule proposed in FedExp.

### Strengths
- The paper fills a gap in the analysis of the previous work that introduced FedExProx by removing the assumption of exactness of the local proximal operator.

### Weaknesses
- While the paper's analysis is sound and well-presented, its contribution is narrowly focused on a single variant of FedAvg—the FedExProx algorithm. It is unclear how the insights from the analysis apply more broadly and how they guide the design of federated optimization algorithms in general.
- In the literature review, please cite previous analyses of FedProx, such as FedNova (https://arxiv.org/abs/2007.07481), which includes as a special case the convergence analysis of FedProx (with $\tau$ local updates at each client, and thus inexact solution of the proximal operator). I also have questions (see below) about how the analysis of this work relates to that analysis of FedProx.
- The FedExProx algorithm considers an extrapolation parameter $\alpha_k$ that varies with the round number $k$. However, Theorem 3.2 and Theorem 4.2 assume that $\alpha_k = \alpha$, fixed across rounds. Wouldn't this reduce the algorithm being analyzed to FedProx?
- The simulation result plots are not easily readable. Please increase the font size.

### Questions
- In practice, clients are likely to simply perform $\tau$ local SGD updates where gradients are computed for the local objective plus the proximal penalty term. I don't quite understand how one would find the inexactness $\epsilon_1$ of the local solution. In the case of local FedProx-style SGD updates, the convergence analysis in https://arxiv.org/abs/2007.07481 can be applied. How does the analysis in this paper (and FedExProx) relate to the prior work? Would you need to consider an $\epsilon_1$ that decays with the number of communication rounds completed?

### Soundness
3

### Presentation
3

### Contribution
2
