# Adaptive Extrapolated Proximal Gradient Methods with Variance Reduction for Composite Nonconvex Finite-Sum Minimization

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
This paper proposes {\sf AEPG-SPIDER}, an Adaptive Extrapolated Proximal Gradient (AEPG) method with variance reduction for minimizing composite nonconvex finite-sum functions. It integrates three acceleration techniques: adaptive stepsizes, Nesterov's extrapolation, and the recursive stochastic path-integrated estimator SPIDER. Unlike existing methods that adjust the stepsize factor using historical gradients, {\sf AEPG-SPIDER} relies on past iterate differences for its update. While targeting stochastic finite-sum problems, {\sf AEPG-SPIDER} simplifies to {\sf AEPG} in the full-batch, non-stochastic setting, which is also of independent interest. To our knowledge, {\sf AEPG-SPIDER} and {\sf AEPG} are the first Lipschitz-free methods to achieve optimal iteration complexity for this class of \textit{composite} minimization problems. Specifically, {\sf AEPG} achieves the optimal iteration complexity of $\mathcal{O}(N \epsilon^{-2})$, while {\sf AEPG-SPIDER} achieves $\mathcal{O}(N + \sqrt{N} \epsilon^{-2})$ for finding $\epsilon$-approximate stationary points, where $N$ is the number of component functions. Under the Kurdyka-Lojasiewicz (KL) assumption, we establish non-ergodic convergence rates for both methods. Preliminary experiments on sparse phase retrieval and linear eigenvalue problems demonstrate the superior performance of {\sf AEPG-SPIDER} and {\sf AEPG} compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper develops AEPG-SPIDER and AEPG, which combine some well-established techniques, such as adaptive stepsizes, Nesterov extrapolation, and the SPIDER variance reducer, for composite nonconvex finite sum optimization. The authors provide theoretical analysis proving optimal iteration complexity and non-ergodic convergence rates under the KL condition.

### Strengths
+ The combination of adaptive stepsizes (based on iterate differences), Nesterov extrapolation, and variance reduction in a single framework for composite nonconvex problems is novel. The proposed update for the extrapolation parameter \sigma^t is creative and central to the analysis. Moving away from gradient norms to iterate differences for stepsize adaptation is a clever way to handle the nonsmooth term h(x).
+ Significant theoretical results in optimal iteration complexity and non-ergodic convergence rates under KL.
+ Clear presentation and comparison with existing work.

### Weaknesses
- Assumption 3.1 is a standard but potentially restrictive assumption. While commonly used in existing analysis, its necessity should be discussed since many real-world problems are unconstrained. Assumption 4.5/4/10 require the stepsize to be sufficiently large, which is non-standard and not sufficiently validated in numerical and practical experiments. 
- The chosen problems (sparse phase retrieval, linear eigenvalue problems) are standard testbeds that have been used to validate proximal methods. They do not demonstrate the algorithm's performance on large-scale, modern machine learning problems (e.g., training regularized DNNs). The algorithm is a bit complicated requiring several hyperparameters (\alpha, beta, \theta, v0, and q, b). How to tune these parameters for optimal performance and the incurred computational burden shall be discussed.
- The requirement to compute a generalized proximal operator (Assumption 2.1) can be expensive for complex h(x).

### Questions
- To show the scalability and practility, can the algorithms be evaluated on a more contemporary DNNs (with billions/millions of parameters) learning problem? 
- Why Adam or its modified/improved recent versions not included as baselines in the tests?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles constrained optimization problems of the form $\\min_x F(x):= f(x)+h(x)$, where $F$ is possibly nonconvex, $f$ is smooth and $h$ is a (nonconvex) regularizer with bounded domain. The authors propose a learning rate-free algorithm to tackle the problem in the deterministic setting, and incorporate a SPIDER type estimator for the finite sum setting. The convergence rate of the proposed methods to stationary points of $F$ in the nonconvex setting is shown and the (linear) convergence under the standard KL assumption is presented. Finally, the method is compared against state-of-the-art variance reduction methods in a number of relevant experiments.

### Strengths
1. The topic of the paper is interesting, since adaptive learning-rate free methods are of particular interest to the machine learning community.
2. The proposed method encompasses various techniques such as Nesterov momentum, AdaGrad type updates and variance reduction, potentially leading to faster algorithms. 
3. The proposed method seems to outperform the rest of the methods in most experiments.

### Weaknesses
1. The quality of the presentation could improve substantially. There are a lot of inconsistencies in the notation and repetitions especially in the appendix. Please see the Questions section for some examples.
2. While many techniques are combined in the paper, the theoretical improvement over existing works in the unconstrained case is only in terms of eliminating a logarithmic term, as described in Remark 3.9.
3. The assumptions of the paper are not discussed enough and are not compared against the ones from recent papers. For example, recent works focus more on problems beyond traditional Lipschitz smoothness assumptions. More discussion is also required for Assumption 4.10 which seems tailored to the convergence analysis of the paper.
4. Although the theoretical guarantees are strong, the variance reduction techniques utilized in the paper are impractical for modern machine learning applications.

### Questions
**Major questions:**

1. In the experiments, how were the other methods tuned? Since the parameters of the proposed algorithms $(v, \alpha, \beta)$ differ a lot between the two experiments, the authors should report how they obtained these parameters and the difficulty of this procedure.

**Minor questions and typos:**
- Line 034: $f(\cdot)$ is used as a function and $h(x)$ which is the value of $h$ at $x$. It is better to use either $h$ or $h(\cdot)$ to denote a function. See also Line 095.
- Line 035: generalized proximal operator is stated but not defined yet. It is better to provide a link to its definition.
- Line 148: Vector $v$ in the generalized norm definition should have nonnegative elements so that the quantity is well-defined.
- Assumption 2.1: missing a verb, probably "computed".
- Line 183: "Given any solution $y^t$"?
- Line 829: Clearly, $g(0) \neq 0$ and in fact $g(0) = 1-1/p$. The result still holds since $g(0) \leq 0$ as $p \leq 1$.
- Line 834: Better to use $(x-y)f'(y)$ instead of the inner product notation.
- Line 879: "uses We have".
- Lemma A.8: the assumption should hold for all $t \in N$.
- Lemma A.10: For the r.h.s. of the inequality to be well-defined, the sequence should also be nonincreasing. This holds for the sequence where the Lemma is applied.
- Line 1075: $s^t$ is the sum of vector and scalar, a $\mathbf{1}$ is missing.
- $Z_t$ is defined in line 1159 and in line 1176 and in line 1216. This happens with many quantities in the paper.
- Line 1196: $a^t = y^t - \nabla f(y^t) / v^t$ or $a^t = y^t - g^t / v^t$?

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
3

### Summary
This paper proposes an *Adaptive Extrapolated Proximal Gradient* method with variance reduction for solving composite nonconvex finite-sum optimization problems. The method integrates three mechanisms—adaptive stepsizes, Nesterov extrapolation, and the SPIDER estimator—into a unified proximal framework. The authors claim optimal iteration complexities \(O(N\epsilon^{-2})\) for AEPG and $O(N+\sqrt{N}\epsilon^{-2})$ for AEPG-SPIDER without requiring Lipschitz constants. Theoretical results under the Kurdyka–Łojasiewicz (KL) assumption provide non-ergodic convergence rates, and preliminary experiments on sparse phase retrieval and linear eigenvalue problems demonstrate faster empirical convergence.

Overall, while the paper contains several interesting elements, it is difficult to follow, motivations are not well articulated, and the technical novelty—especially in terms of complexity improvement—is limited compared to prior variance-reduced methods such as SPIDER and ADA-SPIDER.

### Strengths
The proposed framework unifies adaptive stepsizes, Nesterov extrapolation, and variance reduction. This shows an effort to generalize various optimization components into a single algorithmic structure.

### Weaknesses
-  The method combines several existing techniques (adaptive stepsize + extrapolation + variance reduction) but the motivation for this combination is vague. It is not clear what specific deficiency of prior algorithms is being addressed. The presentation is also dense and difficult to interpret, which hinders accessibility. 
-    The claimed iteration complexities \(O(N\epsilon^{-2})\) and \(O(N+\sqrt{N}\epsilon^{-2})\)  are identical to existing optimal results achieved by prior methods such as SPIDER [Fang et al., 2018] and ADA-SPIDER [Kavis et al., 2022b]. Therefore, the contribution is incremental, and the extrapolation and adaptivity do not yield theoretical acceleration.

### Questions
Could the authors clarify the *main motivation* for combining adaptivity, extrapolation, and SPIDER? What specific limitation of ADA-SPIDER or ProxSARAH does AEPG-SPIDER overcome?

### Soundness
2

### Presentation
1

### Contribution
2
