# Non-differentiable Regularization for Heavy-tailed Differentially Private Stochastic Convex Optimization

- Decision: Reject
- Scores: 2, 4, 8

## Abstract
Recently, differentially private stochastic convex optimization (DP-SCO) with heavy-tailed (HT) (sub)gradients has attracted increasing attention. Weaker than the uniform Lipschitz continuity assumption, the HT assumption on the stochastic (sub)gradients aligns more closely with real-world data distributions. However, few existing methods can handle the non-differentiable regularization (NDR) for HT DP-SCO. The main difficulty lies in achieving a low excess population loss and a low computational complexity simultaneously. In this work, we propose a novel forward-backward splitting approach to tackle NDR for HT DP-SCO, abbreviated by NDR-HT. It satisfies concentrated differential privacy, achieves an asymptotically optimal excess risk bound (up to logarithmic factors), and requires only $\mathcal{O}(n \log n)$ gradient evaluations, which is a lower computational complexity than those of existing state-of-the-art approaches. Furthermore, NDR-HT achieves linear convergence up to an additive approximation error and avoids solving complex subproblems in each iteration. Extensive experiments on both synthetic and real-world data show that our approach is effective and efficient in solving HT DP-SCO with NDR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces NDR-HT, a new forward–backward splitting method for non-differentiable regularization (NDR) in heavy-tailed (HT) differentially private stochastic convex optimization (DP-SCO). Unlike existing approaches that rely on the Lipschitz assumption or cannot handle non-differentiable regularizers, NDR-HT is designed to operate under the weaker HT assumption while achieving ρ-concentrated differential privacy (ρ-CDP), asymptotically optimal excess population loss (EPL) (up to logarithmic factors), and O(n log n) computational complexity. The method constructs a contractive forward–backward splitting operator whose fixed point corresponds to the optimal solution, enabling linear convergence up to an additive approximation error. Experiments on synthetic linear regression and real-world logistic regression tasks demonstrate improved objective values compared with state-of-the-art baselines (LNCSM and NCSGD) across a range of privacy budgets.

### Strengths
1. The paper provides detailed proofs and theoretical guarantees, including convergence, privacy, and excess population loss bounds.

2. The paper is generally clearly-structured

3. The approach is built on operator-splitting and proximal methods, adapted carefully to the heavy-tailed and DP setting.

### Weaknesses
1. The proposed method is essentially a standard forward–backward splitting (proximal gradient) framework with noise injection for privacy. While the adaptation to heavy-tailed gradients and the use of ρ-CDP are reasonable, these extensions appear incremental relative to existing work such as LNCSM and NCSGD (Lowy & Razaviyayn 2023) and the proximal formulations in Asi et al. 2024. The paper lacks a clear conceptual leap or fundamentally new idea beyond applying known techniques in combination.

2. The paper does not convincingly explain why non-differentiable regularization under heavy-tailed assumptions is practically important or insufficiently handled by existing smooth or proximal DP methods. Without concrete applications or examples where NDR is essential, the problem setting feels somewhat artificial.

3. Experiments are minimal (one synthetic and one small UCI dataset) and offer limited insight into real-world effectiveness or scalability. There is no ablation, runtime, or sensitivity analysis to demonstrate robustness or the claimed computational advantage.

4. The main results, including privacy guarantee, convergence, and EPL bound, follow relatively directly from established results in contractive operator theory and standard DP composition arguments. The improvement (up to logarithmic factors) seems modest and not clearly demonstrated as practically meaningful.

5. Given the lack of strong motivation, empirical validation, or substantial methodological innovation, it is difficult to see broad impact on the differential privacy or optimization communities.

### Questions
1. What concrete scenarios or machine-learning models truly require handling non-differentiable regularizers under the heavy-tailed DP setting?

2. How does NDR-HT differ in essence from a straightforward application of a noisy proximal gradient method with gradient clipping?

3. Can the authors provide empirical evidence that existing DP proximal methods fail in the NDR-HT regime, beyond theoretical statements?

4. The asymptotic optimality claim depends on matching lower bounds “up to logarithmic factors.” Are these factors fundamental or merely analysis artifacts?

5. The assumption of bounded iterates is strong. How is this enforced or verified in practice, especially when noise accumulates over many iterations?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes NDR-HT, a forward–backward splitting algorithm for non-differentiable regularized (NDR) stochastic convex optimization under heavy-tailed data distributions with differential privacy guarantees.  
The authors reformulate the empirical objective as a fixed-point problem involving proximal operators of both the non-smooth regularizer \(g\) and the indicator function of a constraint set \(X\).  
They introduce a new operator \(T_C\) that is shown (under certain strong convexity and smoothness assumptions) to be $\kappa$-contractive, enabling a linearly convergent fixed-point iteration.  
The algorithm adds Gaussian noise each iteration to achieve $\rho$-Concentrated Differential Privacy while maintaining near-optimal Excess Population Loss (EPL) up to logarithmic factors.

Theoretical results claim that NDR-HT:
- Satisfies $\rho$-CDP for any $\rho$ > 0.
- Achieves asymptotically optimal EPL in the 2-heavy-tailed regime.
- Requires only $O(n log n)$ gradient evaluations—lower than LNCSM and PLLS.
- Converges linearly up to an additive approximation error.

### Strengths
I really appreciate the writing style of this paper. It presents a clear and coherent narrative — first identifying the core problem to be addressed, and then progressively delving into the technical challenges encountered along the way. The paper indeed proposes a novel and well-motivated approach to handle non-differentiable regularization (NDR). Moreover, the inclusion of complete proofs for all propositions, detailed algorithm pseudocode, and explicit parameter settings makes the work both transparent and reproducible.

### Weaknesses
As claimed in Table 1, you do not use that Lipschitz assumption at all, then why do you list as one of your assumptions?

### Questions
I notice that strong convexity is a fundamental assumption in your theoretical framework, and one of your motivations is that non-differentiable regularization is more practically relevant. However, the strong convexity assumption itself can also be quite restrictive in real-world applications, as it is often violated in many practical models. Moreover, in your experimental section, the algorithm is evaluated only on relatively simple regression tasks rather than on more complex settings such as deep learning models. This may further highlight the gap between the theoretical assumptions and their applicability in realistic scenarios.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles differentially private stochastic convex optimization under assumption of heavy-tailed and the objective includes a non-differentiable regularizer $\boldsymbol{g}$. It transforms the proximal gradient operator into a contractive forward-backward splitting operator such that reformulates the problem as finding a fixed point of a contractive operator. The method attains an asymptotically optimal excess population loss (matching known lower bounds up to logs) while using only $O(n \log n)$ gradient queries-improving on prior approaches that either require Lipschitz assumptions, cannot handle non-differentiable regularization, or incur higher complexity.

### Strengths
1. The paper is comprehensive, well written and well structured.
2. It provides improved bounds and computational efficient results in DP-SCO with heavy-tailed (sub)gradients .
3. The main idea is elegant. Reformulate DP stochastic convex optimization with a non-differentiable regularizer and heavy-tailed gradients as a fixed-point problem for a proximal-gradient map, then turn that map into a contractive forward–backward splitting operator so you can run a simple noisy iteration that (i) satisfies $\rho$-CDP, (ii) achieves asymptotically optimal excess risk (up to logs) with only O(n log n) gradient calls, and (iii) enjoys linear convergence up to a noise floor—all without solving inner subproblems.

### Weaknesses
1. Some notations should be claimed in advance, for example, $G_2^2$ in line 68 and $\bar{\kappa}$ in Table 1.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3
