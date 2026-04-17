# Hom-PGD+: Fast Reparameterized Optimization over Non-convex Ball-Homeomorphic Set

- Decision: Reject
- Scores: 6, 4, 6, 4, 8

## Abstract
Optimization over general non-convex constraint sets poses significant computational challenges due to their inherent complexity. 
In this paper, we focus on optimization problems over non-convex constraint sets that are homeomorphic to a ball, which encompasses important problem classes such as star-shaped sets that frequently arise in machine learning and engineering applications. 
We propose \textbf{Hom-PGD$^+$}, a fast, \textit{learning-based} and \textit{projection-efficient} first-order method that efficiently solves such optimization problems without requiring expensive projection or optimization oracles.
Our approach leverages an invertible neural network (INN) to learn the homeomorphism between the non-convex constraint set and a unit ball, transforming the original problem into an equivalent ball-constrained optimization problem. This transformation enables fast projection-efficient optimization while preserving the fundamental structure of the original problem. We establish that Hom-PGD$^+$ achieves an $\mathcal{O}(\epsilon^{-2})$ convergence rate to obtain an $\epsilon + \mathcal{O}(\sqrt{\epsilon_{\rm inn}})$-approximate stationary solution, where $\epsilon_{\rm inn}$ denotes the homeomorphism learning error. 
This convergence rate represents a significant improvement over existing methods for optimization over non-convex sets. Moreover, Hom-PGD$^+$ maintains a per-iteration computational complexity of $\mathcal{O}(W)$, where $W$ is the number of INN parameters. Extensive numerical experiments, including chance-constrained optimization popular in power systems, demonstrate that Hom-PGD$^+$ achieves convergence rates comparable to state-of-the-art methods while delivering speedups of up to one order of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Hom-PGD+ for constrained optimization over ball-homeomorphic sets. An invertible neural network (INN) is trained to learn a homeomorphism from the unit ball to the constraint set, after which the problem is reformulated as an approximately ball-constrained one and solved via projection gradient descent with bisected projection. The final solution is mapped back using the inverse homeomorphism.

The method requires INN training and incurs $O(W)$ run-time complexity for the bisected projection, where $W$ is the number of INN parameters. Assuming the learned mapping has approximation error $\epsilon_{\mathrm{inn}}$, Hom-PGD+ with a constant step size finds an $\epsilon + \sqrt{L_H \epsilon_{\mathrm{inn}}}$-approximate KKT point in $O(L_H \epsilon^{-2})$ iterations.

Experiments on QCQPs and power-grid optimization problems, along with comparisons to baseline methods, support the effectiveness of the approach.

### Strengths
The proposed algorithm handles constrained sets with ball-homeomorphic structure by learning the homeomorphism using an invertible neural network. This reformulation reduces the problem to a ball-constrained one, enabling the use of projection gradient descent to solve the transformed optimization.

### Weaknesses
Assumption 2 for the convergence guarantee appears difficult to verify in practice. In particular, it is unclear how to estimate $\epsilon_{\mathrm{inn}}$.

### Questions
- How should the $k$’s (splitting components) be chosen in each layer of the INN?

- What does $L_H$ stand for in Theorem 1?

- In Appendix B.5, Eq. (5), should the supremum be taken instead of the infimum?

- In Appendix B.5, Eq. (8), the index $i$ is unused in the summation and should be clarified.

### Soundness
3

### Presentation
3

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
This paper studies parametric constrained optimization problems where the constraint set is compact and homeomorphic to the unit ball. The authors propose a transformation of the problem, where the constraint becomes simply the unit ball and study the equivalence of the stationary points between the two problems. Moreover, they propose a method that estimates the homeomorphism using an invertible neural network (INN), leading to a constraint set that only approximates the unit ball and thus requires a different projection that is then handled by the BP operator (Algorithm 2). The convergence rate of the method to approximate stationary points is proven and finally numerical simulations that showcase the superiority of the method compared to other constrained optimization solvers are presented.

### Strengths
1. The main idea of the paper is interesting and easy to follow. The problem that is considered is interesting and the proposed way of tackling constrained problems seems different from standard optimization approaches.
2. The experimental results show that the proposed algorithm performs similarly or better than state-of-the-art methods.

### Weaknesses
1. The contribution seems incremental compared to prior works, mainly combining approaches from the related literature, namely [1, 2]. The proofs are mostly straightforward applications of existing techniques.
2. The obtained complexity results are not easy to compare to existing results, since the authors utilize a neural network to approximate the homeomorphism. Although one can approximate such functions using NNs, the complexity of this procedure may not be reflected in the obtained guarantees. In that sense the phrase “it achieves a per-iteration complexity of $O(W)$, where $W$ is the number of INN parameters and setting $W = O(n^2)$ is sufficient to achieve strong performance in practice” is not suitable to describe the supposedly better performance of the method in theory, but rather the better performance in some practical applications. Moreover, further discussion on Assumption 2 is required.
3. The authors claim that the proposed method can find application in many practical scenarios and yet provide experimental results for standard quadratically constrained quadratic problems and electrical power systems. Are there any relevant examples of optimization problems from Machine Learning where the method is applicable?

---

References:

[1] Enming Liang, Minghua Chen, and Steven H. Low. Low complexity homeomorphic projection to ensure neural-network solution feasibility for optimization over (non-)convex set. In International Conference on Machine Learning. PMLR, 2023.

[2] Enming Liang and Minghua. Chen. Efficient bisection projection to ensure neural-network solution feasibility for optimization over general set. In International Conference on Machine Learning. PMLR, 2025.

### Questions
Minor questions and typos:
- In (P), the unit ball $\mathcal{B}$ is the Euclidean norm ball, $\\{x \in \mathbb R^n: \|x\|_2 \leq 1\\}$?
- In general, the phrase projected gradient descent is more common than “projection gradient descent” that is found throughout the text.
- Line 222: “the Lipschitz of the homeomorphism”. Should it be the Lipschitz continuity constant of the homeomorphism?
- Line 225: “to train the homeomorphism” should be phrased better.
- Definition 3.2: symbol $\mathcal{H}^n$ not defined. It is better to provide a link to the notation. The same holds for the assumptions of the paper which are found in the appendix.
- Line 257: What is a non-perfect and non-convex ball? Do the authors mean it in a topological sense? I suggest adding further details on such concepts which are currently missing.
- Line 323: "this results".
- Line 366: "The final convergent solution".
- Line 1390: $L_f$-smooth instead of $L_f$ smooth.
- Line 871: "Nerual" -> "Neural"
- Line 1558: The KKT conditions are presented and then the Lagrangian of the problem is presented in line 1588 and the KKT conditions for P follow. I would suggest switching the order.
- Lemma D.4: $ z^* = \\psi^{-1}(x^*) $ is missing. 
- Prood of Lemma D.4: the statement of the lemma is that $ z^* $ is a KKT point of $ H $, while the proof starts with $ z^* $ being a stationary point of P. Lines 1628-1631 follow by multiplying 1594 with $ J_{\\psi}(z^*)^T $.
- Line 1649: should be “implies”.
- Line 1659: What does "$\lambda^*$ as eq.18 mean"?
- Line 1664: "holds" -> "hold".
- Line 1820: by definition of the normal cone.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies optimization over non‑convex constraint sets that are homeomorphic to a ball. It proposes Hom‑PGD+, which learns a homeomorphism from a unit ball to the feasible set with an invertible neural network (INN), solves the transformed problem with projected gradient descent in the ball space, and maps the result back. The algorithm uses a bisected projection along the origin–point ray to handle the approximate ball produced by the learned map.

### Strengths
1. The idea is very clear: onece the INN is trained, the online method is standard PGD in a ball with a cheap ray‑bisection projection.
2. The convergence rate mathcing convex‑like first order methods
3. The overall writing is well-organized.

### Weaknesses
1. In general, it's very hard to verify whether the set is homeomorphic to ball (requires exponential sample complexity).
2. The approximation assumption used in the paper is very strong: In Assumption 2, it requires a uniform bounded on both the function and its Jocabian.

### Questions
1. For JCC/QCQP, could you estimate $\epsilon_{inn}$ from held-out samples and show how good $\epsilon+O(\sqrt{L_H}\epsilon_{inn}) $ bound is?

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
3

### Summary
This paper introduces Hom-PGD+, a method designed for problems with non-convex constraint sets. 

The core idea is to use an Invertible Neural Network (INN) to learn a mapping between the complex, non-convex constraint set and a simple unit ball. This transformation allows the algorithm to perform efficient projections onto the unit ball instead of dealing with the original non-convex set. 

The proposed method is a first-order, projection-efficient algorithm that achieves an iteration complexity of $\mathcal{O}(\epsilon^{-2}$. 

The paper demonstrates the method's effectiveness on problems like non-convex QCQP.

### Strengths
1. The idea of using INNs to learn homeomorphisms for reparameterizing constrained optimization problems seems new. It effectively bridges topological concepts with practical machine learning techniques.

2. This paper provides a convergence analysis, establishing an iteration complexity that is competitive with state-of-the-art methods for non-convex constraints. The analysis carefully accounts for the approximation error introduced by the INN.

3. The paper validates its claims on both synthetic (QCQP) and real-world chance-constrained power grid problems.

### Weaknesses
1. Assumption 2 appears overly strong, and it is unclear how the QCQP and other constrained optimization problems satisfy it.

2. The performance relies on the accuracy of the learned homeomorphism $\epsilon_{\text{inn}}$; the method may degrade if the INN fails to learn an accurate mapping, particularly for complex constraint sets.

3. Training the INN offline incurs additional computational cost, and verifying that a constraint set is ball-homeomorphic—a prerequisite for the theory—is non-trivial and expensive in high dimensions.

4. The method’s performance may be sensitive to the INN’s depth. A more detailed discussion or guidance on selecting an appropriate INN architecture would enhance its practical applicability.

5. Experiments primarily compare with first-order methods for non-convex constraints. Including a comparison with an industrial solver like IPOPT (for QCQP) or analyzing time-to-solution scaling would provide a clearer view of the method’s competitiveness.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the difficult setting of constrained optimization under non-convex constraints. Noting that several practically relevant constraint classes are homeomorphic to a ball, the authors leverage this to transform the original problem into an equivalent and readily solvable ball-constrained one. Their approach incurs a one-time cost of learning this transformation with an invertible neural network (INN), which is subsequently used for carrying out approximate projections within the classical Projected Gradient Descent scheme. The approach is supported by theory in terms of reformulation equivalence, correspondence of KKT points, and convergence guarantees, as well as extensive numerical experiments.

### Strengths
The paper is well-motivated as it proposes a tractable solution for the difficult problem of handling constrained problems with non-convex constraints. The equivalent reformulation into a ball-constrained optimization problem is elegant and highlights a great use case for invertible neural networks. The proposed approach is supported by both theory and extensive experiments, each of which is carried out soundly and with a great amount of detail. The problem and approach description are clear and concise, and supported by explanatory graphics. The related literature is mostly well-covered, to the best of my knowledge.

### Weaknesses
1. **Related literature**
	* The idea of using INNs for approximating operators used by classical optimization schemes is not tracked through the literature. Specifically, while the relevant works are cited, the connection is not made explicit. Since this is central to the proposed solution, a discussion is warranted. E.g., the current approach is a bridge between works relying on known unit ball homeomorphisms for convex sets [2] and INN-learned ball-homeomorphisms for arbitrary sets that fulfill sufficient conditions for the existence of such homeomorphisms [1]. In particular, a technical comparison between this work and [1] would be valuable. 

2. **Technical assumptions**
	* Lipschitz constants of line 338 are not defined in the main text (deferred to Appendix C2). 
	* The technical assumptions on $f_{\theta}$ and $g_{\theta}$ and $\psi_{\theta}$ are also fully deferred to appendix C2. They should be present in the main text, since they are essential to deriving the theoretical results and dictate the applicability of the results. 


3. **Minor**
	* Typos: line 222 "the Lipschitz constant", line 1846 "1/L_H"

[1] Liang, Enming, Minghua Chen, and Steven H. Low. "Homeomorphic projection to ensure neural-network solution feasibility for constrained optimization." Journal of Machine Learning Research 25.329 (2024): 1-55.

[2] Chenghao Liu, Enming Liang, and Minghua Chen. Fast projection-free approach (without linear
minimization oracle) for optimization over general compact convex set. Advances in Neural
Information Processing Systems, 2025a.

### Questions
* For Fig 5. in the appendix, the last two rows of figures: the log volume quantity exhibits an initial dip followed by an increase. Do the authors know why this happens? Is it due shift in the dominating terms of the loss (as a function of gradient norm)?
* Currently, the technical assumptions on the objective's components are stated as global requirements. Specifically, $f_{\theta}$ is $L_{f,0}$ Lipschitz continuous and $L_f$-smooth .This is quite restrictive, since, for example, the common mean-square type objectives $\frac{1}{2}\\|\cdot\\|^2$ do not satisfy global Lipschitz continuity. Given the compact constraints, can you refine/alleviate these assumptions?

### Soundness
4

### Presentation
4

### Contribution
3
