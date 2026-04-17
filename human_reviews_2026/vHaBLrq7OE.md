# Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise

- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
Constrained optimization is a powerful framework for enforcing requirements on neural networks. These constrained deep learning problems are typically solved using first-order methods on their min-max Lagrangian formulation, but such approaches often suffer from oscillations and can fail to find all local solutions. While the Augmented Lagrangian method (ALM) addresses these issues, practitioners often favor dual optimistic ascent schemes (PI control) on the standard Lagrangian, which perform well empirically but lack formal guarantees. In this paper, we establish a previously unknown equivalence between these approaches: dual optimistic ascent on the Lagrangian is equivalent to gradient descent-ascent on the Augmented Lagrangian. This finding allows us to transfer the robust theoretical guarantees of the ALM to the dual optimistic setting, proving it converges linearly to all local solutions. Furthermore, the equivalence provides principled guidance for tuning the optimism hyper-parameter. Our work closes a critical gap between the empirical success of dual optimistic methods and their theoretical foundation in the single-step, first-order regime commonly used in constrained deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies dual-only optimistic ascent applied to the standard Lagrangian. The central result establishes a precise equivalence between this method and gradient descent–ascent on the Augmented Lagrangian (ALM) when the optimism and penalty parameters are aligned. This equivalence holds at the iterate level for equality-constrained problems (under specific initialization) and, for inequality constraints, extends to locally stable stationary points, transferring local linear convergence and damping properties. The work thus generalizes the theoretical foundation of ALM-based gradient methods to the dual-only optimistic ascent framework, explains the empirically observed stabilization effect of dual optimism, and offers practical guidance on parameter tuning (e.g., matching optimism and penalty parameters) through the ALM perspective.

### Strengths
This paper reduces dual-only optimistic ascent to the Augmented Lagrangian Method (ALM), establishing a theoretical foundation for its practical use in constrained deep-learning pipelines. The theoretical derivation seems sound.

### Weaknesses
The generality of the results appears limited, as the analysis focuses on the specific setting of dual-only optimism with single-step first-order primal updates. Since the reviewer is not deeply familiar with the literature on constrained optimization via dual optimistic ascent, it is somewhat difficult to assess the broader significance of the contribution, especially given that the problem formulation seems rather specialized.

### Questions
Could the authors provide further justification or point the reviewer to references explaining why existing theoretical analyses of general min–max optimization are insufficient for this setting? It would also be helpful to clarify what specific advantages the proposed analysis offers over prior frameworks.

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
This paper establishes a theoretical equivalence between dual optimistic ascent (PI control) and the Augmented Lagrangian Method (ALM). Specifically, it shows that for equality-constrained problems, dual optimistic ascent on the Lagrangian is identical (in primal iterates) to gradient descent–ascent on the Augmented Lagrangian when the optimism coefficient ω equals the penalty parameter c. For general inequality constraints, the two methods share the same locally stable stationary points (LSSPs), implying the same convergence targets. This equivalence allows transferring ALM’s robust theoretical guarantees—especially linear convergence—to the dual optimistic framework, previously known mainly for empirical success. The paper further analyzes implications for convergence, stability, oscillation damping, and hyperparameter tuning.

### Strengths
This submission has novel and elegant theoretical contribution. The main equivalence result is new and insightful: it unifies two popular but seemingly different optimization paradigms. The insight that “dual optimism ≈ augmented regularization” is elegant and of clear theoretical and practical significance. This bridge justifies the strong empirical performance of PI-control methods used in deep RL and constrained deep learning.

This submission is mathematically sound and grounded in classical references. The local stability analysis via Jacobians is carefully argued and well contextualized. Theorems systematically extend the equivalence to convergence, oscillation damping, and conditioning, providing a unified picture.

The practical implications for tuning the optimism coefficient ω using ALM’s penalty intuition are particularly valuable. The discussion connects to recent empirical work in reinforcement learning and fairness-constrained training. This paper has some practical use in deep learning and LLM.

### Weaknesses
The most significant weakness of this submission is that this paper lacks empirical verification from numerical experiments. Although the theory neatly explains prior empirical results, no new experiments are provided. The reviewer suggest adding a simple numerical example (e.g., a 2D quadratic constraint problem or other simple constrained optimization problem ) showing the equivalence in iterates and the damping of oscillations. With the numerical experiments results, this can significantly improve the quality of this submission.

Other minor weaknesses include limited scope for inequality constraints. The equivalence for inequalities is only local, and the intuition behind the “two projections vs. one” distinction could be expanded with geometric visualization or numerical illustration. 

Also the assumptions (strict complementarity, LICQ, SOSC) are standard but a little strong. The paper could benefit from discussing whether relaxed conditions (e.g., weak constraint qualification) still preserve equivalence.

This submission didn't have discussion on Multi-step or Second-order ALM Variants The authors explicitly limit equivalence to single-step first-order methods. It is suggest that adding brief discussion about how multi-step variants (as used in practical ALM solvers) might diverge, or outline what breaks in the proof.

### Questions
There are no other questions.

### Soundness
2

### Presentation
3

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
This paper considers constrained optimization problems with both equality and inequality constraints. After describing three primal-dual approaches for solving the problem (Lagrangian, augmented Lagrangian, and dual optimistic ascent), the paper then establishes theoretical equivalence results between the augmented Lagrangian method and the dual optimistic ascent approach, hence allowing the desirable properties of augmented Lagrangian to be transferred to dual optimistic ascent (also known as PI control). Discussions are also provided on the practical implications of this equivalence, including the choice of the optimism hyperparameter in dual optimistic ascent.

### Strengths
- The paper is very well written and provides adequate background for an audience who may not be readily working in constrained optimization/learning.
- The provided equivalence results are interesting and could lead to significant follow-up theoretical and empirical research in constrained optimization, especially in the context of modern deep learning problems.

### Weaknesses
- There was no mention of how the results connect to the *parameterized* setting, where the primal variables are the parameters of a non-convex deep neural network architecture. Do you anticipate the results to have parameterized analogs under assumptions (such as near-universality of the parameterization)?
- The results are limited to a single primal step per iteration (although this is, I believe, what is mainly used in practice).
- The exact equivalence of primal iterates is limited to problems with only equality constraints.
- No empirical results are provided to corroborate/complement the theoretical findings.

### Questions
- What is the downside of augmented Lagrangian (and dual optimistic ascent) compared to regular Lagrangian? If convergence properties are more desirable, why is the regular Lagrangian the most commonly used approach in practice?
- Could you please clarify the statement on lines 224-225 on the replacement of the first dual update in (Lag-GD-OA) in practice? Is there a way to get around evaluations of $g$ and $h$ at $x_{t-1}$?
- Is there a way to better formalize the Jacobian "becoming real" in Corollary 5 (e.g., via characterizing the rate of decline of the imaginary eigenvalues)?
- I understand that this paper's contribution is primarily on the theoretical side. However, I believe some simple experiments on the differences between augmented Lagrangian and dual optimistic ascent in problems with inequality constraints, as well as dynamic tuning of the optimism hyperparameter by mirroring augmented Lagrangian penalty scheduling methods (as mentioned in lines 460-461), would be interesting and helpful additions to the paper.
- I would suggest swapping the discussion on OGDA and ALA in Section 2 (i.e., discussing OGDA first and then ALA) for a better flow.

### Soundness
4

### Presentation
4

### Contribution
3
