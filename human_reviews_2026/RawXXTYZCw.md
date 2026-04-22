# Bridging Constraints and Stochasticity: A Fully First-Order Method for Stochastic Bilevel Optimization with Linear Constraints

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
This work provides the first finite-time convergence guarantees for linearly constrained stochastic bilevel optimization using only first-order methods—requiring solely gradient information without any Hessian computations or second-order derivatives. We address the unprecedented challenge of simultaneously handling linear constraints, stochastic noise, and finite-time analysis in bilevel optimization, a combination that has remained theoretically intractable until now. While existing approaches either require second-order information, handle only unconstrained stochastic problems, or provide merely asymptotic convergence results, our method achieves finite-time guarantees using gradient-based techniques alone. We develop a novel penalty-based framework that constructs hypergradient approximations via smoothed penalty functions, using approximate primal and dual solutions to overcome the fundamental challenges posed by the interaction between linear constraints and stochastic noise. Our theoretical analysis provides explicit finite-time bounds on the bias and variance of the hypergradient estimator, demonstrating how approximation errors interact with stochastic perturbations. We prove that our first-order algorithm converges to $(\delta, \epsilon)$-Goldstein stationary points using $\Theta(\delta^{-1}\epsilon^{-5})$ stochastic gradient evaluations, establishing the first finite-time complexity result for this challenging problem class and representing a significant theoretical breakthrough in constrained stochastic bilevel optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper establishes a theoretical milestone by bridging the gap between linear constrained and stochastic bilevel optimization, delivering a purely first-order, provably convergent algorithm. It lays the groundwork for extending efficient bilevel solvers to more realistic, noisy, and constrained ML applications like meta-learning, RL, and data reweighting.

### Strengths
1. This paper provides a finite-time stochastic convergence with linear constraints and first-order access.

2. This paper also has strong theoretical grounding (bias/variance analysis and Goldstein stationarity), and the proposed method has superior scalability for high-dimensional problems.

### Weaknesses
1. This paper announces that it provides the first finite-time convergence guarantees. However, there are several works about constraints in bilevel optimization, such as Overcoming Lower-Level Constraints in Bilevel Optimization: A Novel Approach with Regularized Gap Functions. Can the author provide some comparison?

2. It looks like the Assumption 3.1 (ii) asks lower-level $g$ to be strongly convex and also have a bounded gradient. Can the author verify this assumption?

3. This paper also employed additional assumptions compared to other bi-level works. Such as Assumption 3.1 (iii) and Assumption 3.2. 

4. The paper is not well organized and is hard to read. Such as $\lambda^*(x)$ in line 151 is used before defined.

### Questions
1. Why is Assumption 3.2 necessary? In traditional bilevel optimization, this condition typically appears as a lemma rather than an assumption. Could the authors clarify what specific difficulty prevents deriving a similar lemma in the constrained setting?

2. Please compare the role and strength of this assumption with those used in other bilevel optimization works, particularly in constrained bilevel formulations.

3. Does the proposed problem have any practical applications? The current experiments appear overly simplified and resemble toy examples, which raises concerns about the real-world relevance of the proposed method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies stochastic bilevel optimization with linearly constrained lower-level (LL) problems, a setting where no prior work provides finite-time guarantees. The authors propose F2CSA (Fully First-order Constrained Stochastic Approximation)—a fully first-order method requiring only noisy gradients from the upper- and lower-level objectives. The key idea is to construct a stochastic inexact hypergradient oracle via a smoothed Lagrangian/penalty formulation with scaling parameters $\alpha_1 = \alpha^{-2}$ and $\alpha_2 = \alpha^{-4}$, together with inexact primal–dual LL solves.
They prove that the oracle has bias $O(\alpha)$ and variance $O(1/N_g)$, and that when used in a clipped nonsmooth outer loop, the algorithm converges to a $(\delta,\epsilon)$-Goldstein stationary point with total complexity $\tilde{O}(\delta^{-1}\epsilon^{-5})$—the first finite-time result for this class.

### Strengths
1. First finite-time guarantee for stochastic bilevel problems with linearly constrained LL subproblems, using a fully first-order method.
2. The presentation is clear.

### Weaknesses
1. The experimental evaluation is limited; additional large-scale experiments would be valuable to demonstrate the method’s scalability and practical relevance.
2. The LICQ assumption appears somewhat strong. Could the authors consider relaxing it to a weaker constraint qualification, or provide more discussion on why this assumption is essential for the current analysis?

### Questions
1. What other stationarity notions (beyond Goldstein stationary points) have been adopted in prior literature? A more comprehensive literature review on alternative stationarity metrics would strengthen the paper’s context.
2. I wonder whether variance-reduction or momentum techniques could further improve the theoretical complexity bounds within the proposed framework.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the linearly constrained stochastic bilevel optimization problem with first-order methods. The authors propose the algorithm F2CSA and provide convergence analysis.

### Strengths
1. The first-order methods for the constraint bilevel optimization problem have not been fully studied before.
2. The authors provide proof sketches for better understanding.

### Weaknesses
**There are lots of presentation problems that I doubt the correctness of the proof.**
1. In line 191, there is an incomplete sentence.
2. There is no explanation of Algorithm 1 before Remark 4.1. Therefore, there are a lot of undefined notations in it.
3. There is no update rule for $\tilde{\lambda}(x)$.
4. For the stochastic algorithm, are the authors sure that we can get $\\|\\tilde{y}^\ast(x)-y^*(x)\\|\\leq\mathcal{O}(\delta)$ rather than  in the expectation form with samples ($\mathbb{E}[\\|...\\|]\leq...$ with some samples $\xi$)? The same problem exists for $\lambda$.
5. In line 5 of Algorithm 1, what does "$\\|\leq\delta$" mean? 
6. In line 246, $\alpha\geq\frac{2C_f}{\mu}$. This means that $\alpha$ is at a constant order. However, later, $\alpha$ is set to the $\epsilon$ order so that the algorithm converges. This contradiction makes me highly doubt the correctness of the proof.
7. In Lemma 4.3, $L_{H,y}$ and $L_{H,\lambda}$ are not defined, and the formulation is not given either. 
8. For this linear constraint BO problem, I do not see how $h(x,y)$ impacts the convergence.

There are more problems that I have not listed yet. At this point, I believe the paper is far from ready.

### Questions
Please check the weakness part.

### Soundness
2

### Presentation
1

### Contribution
2
