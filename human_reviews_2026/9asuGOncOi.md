# Fast Frank–Wolfe Algorithms with Adaptive Bregman Step-Size for Weakly Convex Functions

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4, 6

## Abstract
We propose Frank–Wolfe (FW) algorithms with an adaptive Bregman step-size strategy for smooth adaptable (also called: relatively smooth) (weakly-) convex functions. This means that the gradient of the objective function is not necessarily Lipschitz continuous, and we only require the smooth adaptable property. Compared with existing FW algorithms, our assumptions are less restrictive. We establish convergence guarantees in various settings, including convergence rates ranging from sublinear to linear, depending on the assumptions for convex and nonconvex objective functions. Assuming that the objective function is weakly convex and satisfies the local quadratic growth condition, we provide both local sublinear and local linear convergence with respect to the primal gap. We also propose a variant of the away-step FW algorithm using Bregman distances over polytopes. We establish faster global convergence (up to a linear rate) for convex optimization under the Hölder error bound condition and local linear convergence for nonconvex optimization under the local quadratic growth condition. Numerical experiments demonstrate that our proposed FW algorithms outperform existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposed Frank-Wolfe (FW) algorithms with adaptive Bregman step-size strategies for a class of constrained optimization problems. The objective function satisfies the smooth adaptable property with respect to some Bregman distance function and the $q$-Holder error bound condition. For convex optimization, the proposed FW algorithms achieves global linear or sublinear convergence. For weakly-convex optimization, the proposed algorithms achieves global linear or sublinear convergence if the objective function satisfies the quadratic growth condition.

### Strengths
- The paper is well-motivated, organized and well-written.

- The proposed algorithm is parameter-free (with adaptive strategy for parameters) and provides global convergence guarantees under convex and nonconvex scenarios.

### Weaknesses
See questions below.

### Questions
- The global linear convergence rates of the proposed algorithms only holds for specific classes of functions: $q=2$ or $q= 1+ \nu$ or $\nu = 1$ or for initial iterates. Though the footnotes give explanation, the presented results in Table 1 may be misleading at the first glance. I suggest to add more description in the table.


- Theorem 5.2 requires $\rho < \mu$ and Theorem 5.2 requires $\rho < \mu \leq L$. These assumptions seem nontrivial.  

(i) It is nice to see that Example D.4 satisfy the assumption.Is there a general class of functions satisfying the assumption? 

(ii) $\rho, \mu$ and $L$ characterize the convexity of $f$ to some extent: $\rho$ is basically the $\rho$-smad parameter when $\phi$ is the quadratic function (typical Euclidean distance). $\mu$ is requiring the strong convexity over the solution set, which is related to the star-convexity. Is there relation between $\rho, \mu$ and $L$? For example, would some property of $\phi$ guarantee that $\rho \leq L$? Inituitive discussions are also welcome. 

- Typo: Line 223, 'right-hand size'

### Soundness
3

### Presentation
4

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
This paper proposes new Frank–Wolfe (FW) algorithms that incorporate a Bregman-based adaptive step-size rule, removing the need for Lipschitz continuity of the gradient of the objective function. The authors extend FW to L-smooth adaptable functions, broadening applicability beyond standard L-smooth settings. They achieve sublinear to linear convergence rates under weaker conditions (e.g., Hölder error bound or local quadratic growth). This submission also provide both convex and weakly convex convergence analyses and an away-step variant for polytopes with provable linear rates. The paper include numerical validation on ℓp-loss and phase retrieval problems, showing empirical improvements over classical FW and its variants. In summary, it generalizes FW theory to handle non-Lipschitz and weakly convex objectives while retaining theoretical rigor and empirical competitiveness.

### Strengths
This submission has significant theoretical generalization. The relaxation from Lipschitz smoothness to relative smoothness is well motivated and aligns FW with modern Bregman-based optimization. It unifies and extends several prior frameworks, offering linear convergence under weaker assumptions.

The authors provide rigorous analysis. The theoretical sections in this submission are mathematically precise, covering: convex and weakly convex settings, both FW and away-step variants, and multiple growth conditions (HEB, quadratic growth). The mathematical proofs appear comprehensive and grounded in established geometric constants (pyramidal width, etc.). The authors also provided the adaptive Bregman step size. This adaptive scheme is elegant by extending previous results  and self-tunes both L and ν parameters. The termination proof (Remark 3.2) ensures practicality. 

Moreover, the authors conduct numerical experiments and the empirical results are consistent with the theoretical analysis. The Experiments on non-Lipschitz settings convincingly show faster convergence and robustness where Euclidean FW fails.

### Weaknesses
There are no major weakness about this submission. The following is just some minor weaknesses:

The authors should add more numerical experiments. Only two primary experiments are shown (ℓp loss and phase retrieval). While results are positive, including comparisons on structured convex problems (e.g., LASSO, matrix completion) would better demonstrate generality.

This submission need more clarifications. Definitions (e.g., L-smad, kernel generating distances) are presented quickly with minimal intuition. Some long theorems could be summarized qualitatively before stating full formulas. Adding geometric or schematic illustrations (e.g., showing Bregman vs. Euclidean geometry) would enhance readability.

The authors should add more discussion of related work. The paper references many FW variants, but comparison to mirror descent or relative smoothness-based proximal algorithms is somewhat limited. Highlighting differences in oracle requirements and computational cost would clarify its niche.

### Questions
There are no other questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a Frank–Wolfe (FW) algorithm with an adaptive Bregman step-size strategy. The proposed algorithm covers relatively smooth and weakly convex setups, which are broader than the conventional $L$-smooth convex setup. It achieves local linear convergence under weak convexity and the local quadratic growth condition. Moreover, when the constraint set is a polytope, the paper proposes a variant of the away-step FW algorithm that overcomes the zigzagging issue of the classical FW algorithm. The latter algorithm also achieves local linear convergence under the aforementioned assumptions. While the convergence is local in the nonconvex case, both algorithms achieve global linear convergence when the objective function is convex and satisfies the Hölder error bound condition. Finally, the paper provides numerical experiments demonstrating its efficiency.

### Strengths
- To the best of this reviewer's understanding, the γ-update in line 5 of Algorithm 2 and line 8 of Algorithm 3, which is motivated by equation (2.2), is a novel aspect of the proposed algorithm. It seems that this idea enables the proposed algorithms to establish convergence guarantees for the considered broad setup, and the fact that the paper indeed provides corresponding convergence results is meaningful. The paper also presents convincing experimental results.
- The considered setup, L-smooth adaptable (i.e., both Lφ − f and Lφ + f are convex on the constraint set C), indeed appears to be an extension of the conventional setup, as mentioned in the paper.
- The paper seems to be overall well structured.

### Weaknesses
- **W1.** While this reviewer believes that the results are meaningful and worth the effort, the reviewer is not fully convinced of the overall technical novelty. In short, to the best of the reviewer’s understanding, this paper leverages line search to develop an adaptive step size, which is arguably a classical technique in optimization. The reviewer is curious whether there were any new challenges the authors needed to overcome to handle this setup. The reviewer would be happy to be corrected if something has been overlooked, as noted in the related question Q1.

### Questions
- **Q1.** Could the authors elaborate on any novel proof techniques or specific technical challenges they had to overcome while establishing the results, particularly those related to the extension of the setup?

- **Q2.** There exists a line of research that develops parameter-free methods without employing line search in other setups [1–5]. Do the authors anticipate particular challenges in removing the line search component from the proposed framework? If so, could they elaborate on the underlying reasons? (The reviewer thinks that this question may be out of the scope of this paper and is mainly motivated by the reviewer’s interest in hearing the authors’ intuition.)


[1] Yura Malitsky and Konstantin Mishchenko. Adaptive gradient descent without descent. International Conference on Machine Learning, 2020.

[2] Yura Malitsky and Konstantin Mishchenko. Adaptive proximal gradient method for convex optimization. Neural Information Processing Systems, 2024.

[3] Puya Latafat, Andreas Themelis, Lorenzo Stella, and Panagiotis Patrinos. Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient. Mathematical Programming, 2024.

[4] Tianjiao Li and Guanghui Lan. A simple uniformly optimal method without line search for convex optimization. Mathematical Programming, 2025.

[5] Danqing Zhou, Shiqian Ma, and Junfeng Yang. AdaBB: Adaptive Barzilai-Borwein method for convex optimization. Mathematics of Operations Research, 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposed a Frank-Wolfe algorithm for constrained optimization whose objective function is convex/weakly-convex and relatively smooth, the algorithm is further equipped with adaptive Bregman stepsize and away-step specifically for polytope cases. Convergence rates are provided both convex and weakly convex cases. Numerical experiments are complemented to verify the effectiveness of the proposed algorithms.

### Strengths
1. Extend the scope beyond convex and L-smooth, which is more general and fit more practical problems.
2. The stepsize is adaptive and "drop-in", which is easy to use.
3. The writing is clear, the flow of the work is easy to follow.

### Weaknesses
1. Even though extending into nonconvexity, the results still require some strong conditions like HEB, such conditions are still a bit strong, and lacks nontrivial examples throughout the work to verify the effectiveness.
2. The work seems to be a combination of FW with Bregman divergence, also many existing works on EB/QG conditions and relative smoothness, the novelty in terms of techniques may be limited a bit.
3. Line 304, "We will now establish faster convergence rates than O(1/t) up to linear convergence depending on the choice of parameters.", but I may argue that the acceleration comes from the problem setting (additional EB condition compared to the vanilla convex setting), rather than your parameter setting.
4. For the nonconvex part, Theorem 5.2 and 5.3 further require $\rho<\mu$, which has not been verified, it would be helpful to include discussion or examples illustrating when this inequality holds, or how one might estimate these quantities in practice. This would clarify the scope of applicability of the nonconvex guarantees.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies Frank-Wolfe algorithm under relatively smooth and (weakly-) convex assumptions. The authors proposes new stepsizes that utilize Bregman distance, which generalizes standard Euclidian setting. The paper derive convergence guarantees: sublinear and local linear rates under weaker assumptions than classical Lipschitz‐gradient smoothness and strong convexity, and demonstrate experimentally that their methods outperform existing FW algorithms.

### Strengths
1. The paper extends the analysis of FW‐type methods to the class of (L-smad) functions and weakly convex objectives. These results are new and correct.
2. The proposed Adaptive Bregman step-size strategy automatically adapts to L-smad constant, which does not require extensive hyper-parameter search or estimation of $L$.
3. The paper shows not only global sublinear convergence but also local linear convergence in the convex case under a Hölder error-bound condition (HEB) and in the nonconvex case under a local quadratic growth condition. This gives stronger theoretical guarantees than many prior FW analyses.

### Weaknesses
1. Weak-convexity, quadratic growth and HEB assumptions while being more general then previous assumption, still are strong. Under these assumptions, linear convergence rate is not surprising.

### Questions
1. How sensitive is the performance of the adaptive Bregman step‐size strategy (Algorithm 2) to the parameters $\beta, \tau$,  (which control the inner loop for estimating $M$ and $\kappa$)? Do the authors provide guidelines on tuning those for new problems?
2. What is a complexity of Procedure step_size in Algorithm2?
3. Do short step-sizes perform better than adaptive step-sizes on numerical experiments?
4. In the nonconvex (weakly convex) setting, the local linear convergence assumes a local $\mu$-quadratic growth condition. Practically, how can one check or ensure this condition holds in a given application? Also, given that there is a linear local convergence of the method, how one can identify when the algorithm reach this local neighborhood?

### Soundness
3

### Presentation
2

### Contribution
3
