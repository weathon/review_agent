# Computationally Efficient RL under Linear Bellman Completeness for Deterministic Dynamics

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
We study computationally and statistically efficient Reinforcement Learning algorithms for the *linear Bellman Complete* setting. This setting uses linear function approximation to capture value functions and unifies existing models like linear Markov Decision Processes (MDP) and Linear Quadratic Regulators (LQR).  While it is known from the prior works that this setting is statistically tractable, it remained open whether a computationally efficient algorithm exists. Our work provides a computationally efficient algorithm for the linear Bellman complete setting that works for MDPs with large action spaces, random initial states, and random rewards but relies on the underlying dynamics to be deterministic. Our approach is based on randomization: we inject random noise into least squares regression problems to perform optimistic value iteration. Our key technical contribution is to carefully design the noise to only act in the null space of the training data to ensure optimism while circumventing a subtle error amplification issue.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers the problem of learning in an MDP that is endowed with a property known in literature of _linear Bellman completeness_. Intuitively, this property ensures that, whenever a function is linear in a given feature map, it remains linear after applying the  Bellman optimality operator.

Even if existing algorithms (Eleanor) achieve optimal regret guarantees for this setting, no algorithm is known to solve it with polynomial _computational complexity_ in every parameter. 

In this paper, the authors solve the problem in case the process is also deterministic.

### Strengths
The paper introduces an algorithm based an a very original and novel idea, that is to perturb the parameters of the algorithm with a noise that is sampled from the null-space of the features spanned up to the given episode. This idea, which the authors call Null Space Randomization, may have a broader impact in the statistical learning community.

The paper also introduces some novelty in the proofs of the statistical complexity bounds. Related to the previous point, the authors introduce a Span Argument: indeed, the new features of the state-action pairs visited in the process do not lie outside of the span of the past features too often. In fact, for every time-step h, this can happen at most $d$ times. Using this simple observation, it is possible to avoid exponential blowout in $H$. 
As the problem of exponential blowout in the horizon appears in a wide family of RL with function approximation works, and is usually solved by truncating the value function, which needs to introduce unrealistic assumptions to ensure that truncation does not bring us outside of the function class, I think this is also a major contribution.

Lastly, it is remarkable that the papers deals also with the case of Low Inherent Bellman Error, where the linear model is affected by a misspecification. This scenario, while particularly challenging, is by far more general than assuming exact linearity.

Overall, it is worth mentioning that the algorithm proposed enjoys a regret guarantee which is order-optimal, i.e. scales with $\sqrt T$ (which is the best dependency) in the case of no misspecification. At the opposite side [1], which is perhaps the most recent "competitor" of this work, proves a statistical complexity bound that scales as

$$\varepsilon^{-O(A)},$$

being therefore exponentially dependent on the number of actions and potentially heavily suboptimal even for $A=2$ (due to the $O(\cdot)$). I see this as a major point for this work over the state of the art.


[1]Linear bellman completeness suffices for efficient online reinforcement
learning with few actions.

### Weaknesses
The only big weakness of the paper is that the setting is very restrictive, and the contribution does not _at first sight_ seem a big improvement over the state of the art.

1. Assuming deterministic dynamics is very restrictive and, most importantly, does not marry very well with Bellmann completeness. Indeed, given the for of the Bellman optimality operator,
$$\mathcal Tf (s,a)=\int_S p(s'|s,a)\max_{a'\in A}f(s',a')\ ds',$$

One may hope that a well-behaved density function $p(s'|s,a)$ helps the result to approach the span of the features by smoothing everything (a smooth function is indeed easier too approximate with a linear feature map) thus reducing the inherent bellmann error. This does not happen in case of deterministic dynamics, as we just get

$$\mathcal Tf (s,a)=\max_{a'\in A}f(p(s,a),a'),$$

which may be only piecewise smooth due to the maximum.

2. As explained in the paragraph "Deterministic Rewards or Deterministic Initial State", an efficient algortihm was already known for the case where the initial state was fixed; generalizing to the case of random $s_1$ is usually easy for general MDPs. Still, it must be noted that the assumption of linear Bellman completeness does not allow for an easy reduction: just creating a fictitious initial state does not work in this case.

Overall, I think these limitations should not divert the attention for the value of the paper, as they are just a consequence of the hardness of the setting.

### Questions
In appendix E: LINEAR MDPS AND LQRS IMPLY LINEAR BELLMAN COMPLETENESS, the authors prove a result about the Linear Quadratic Regulator.

At line 2040, the authors claim that, due to the structure of the problem, the optimal actions writes as $Kx$, where $K$ is some matrix. Even if it is true that in the LQR the optimal policy takes this form, I am not convinced from that passage: indeed, what the authors are showing is that

$$E[\phi(x',\pi^*(x'))^\top \theta] = \phi(x,u)^\top \theta',$$

for some $\theta'$, where $\pi^*$ is the optimal policy. While in fact, what they needed to prove is that

$$E[\min_u \phi(x',u)^\top \theta] = \phi(x,u)^\top \theta'.$$

There is a subtle difference between the two things, as first we are using the optimal policy of the MDP, while in the second case $u$ is given by the best action _with respect to the given $\theta$._ Please, correct me if I am wrong.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a learning algorithm for linear Bellman-complete MDPs with deterministic dynamics. The algorithm randomizes value function estimates to facilitate efficient exploration of the feature space, a commonly used technique. However, the distinguishing feature of this algorithm is that the random noise is injected only into the null space of the collected feature vectors, selectively exploring previously unvisited regions of the feature space and preventing the value function parameters from growing exponentially with the horizon length. Furthermore, the algorithm employs a random walk-based constrained optimization solver, enhancing the computational efficiency of the method. The derived regret bound for the algorithm is optimal in the number of episodes.

### Strengths
* The paper achieves computational efficiency alongside a low regret bound. As the authors point out, computational considerations in MDP learning are crucial for practical applicability, particularly given that many recent works focus only on the statistical analysis of MDP learning.
* The idea of injecting random noise exclusively within the null space of the data matrix is brilliant. The insight that this approach promotes efficient exploration of unexplored feature space without redundancy is well-explained throughout the paper. This strategy may be potentially adapted to settings beyond linear Bellman completeness.

### Weaknesses
* To illustrate the computational efficiency of the method, including a simple numerical demonstration would be beneficial.
* One concern is with the presentation style of the paper. The assumption of deterministic dynamics appears to be critical for the regret analysis, yet it is unclear why stochastic dynamics would impede the success of the method. Although it is mentioned that the deterministic nature of the system enables value decomposition as shown in Lemma 10, the necessity of this condition is not explicitly explained in the main body of the paper.

### Questions
* There are some typos: e.g., "to the learn parameters" in p.2, and "where the it is" in p.6.
* Why are the constrained squared loss minimization problems first converted to the convex feasibility problem, and then solved by random walk method that depends on a separation oracle? It seems that when $\mathcal{S}$ and $\mathcal{A}$ are infinite, then it is challenging to design an appropriate separation oracle for $\mathcal{K}$. Even when  $\mathcal{S}$ and $\mathcal{A}$ are finite, then each minimization problem has $d$ decision variables and $2|\mathcal{S}||\mathcal{A}|$ constraints, requiring any separation oracle to evaluate the membership for $2|\mathcal{S}||\mathcal{A}|$ inequality constraints.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies RL under linear Bellman completeness (a generalization of low-rank MDPs) and deterministic transition dynamics. It shows how to design the first algorithm that is both statistically and computationally efficient for this setting, and important open question given the intractability of all existing approaches. The main idea is to drive exploration by injecting noise into the least squares estimates, with the main novelty being that this is done only on the null space of the training data to avoid estimation erros from exploding.

### Strengths
1. Designing RL algorithms with function approximation for large state-action spaces that are not only statistically efficient but also computationally efficient is an important open problem, and this paper makes a good step forward in this direction
2. The "span argument" is interesting and novel
3. Although I only quickly went through the proofs, all results seem sound
4. The paper is very well written: the problem and its technical challenges are well motivated from the very beginning, notation is clear, and the adopted techniques are well justified
5. The examples in 3.1 are interesting to motivate why bounded norm assumptions are unreasonable, and further motivate the paper's contributions

### Weaknesses
1. The "span argument" is quite interesting, but it also seems quite specific for MDPs with deterministic transitions. Not clear if it extends to stochastic MDPs
2. While a good step forward and a common setting in practice, the assumption of deterministic dynamics may be a limiting factor
3. There is no experiment despite the algorithm being computationally efficient (and, I hope, implementable)
4. Not clear if the dependences on d and H are optimal in the main theorems, or if there is still work to do in that direction

### Questions
1. The paper mentions that "Many efficient algorithms can be applied to find approximate D-optimal designs such as the Frank-Wolfe". I think this is the case when S,A are finite, but what if they are infinite? Is there any efficient algorithm?
2. It is not clear why constrained least squares regression, instead of an unconstrained one, is needed since there is no assumption on boundedness of the true parameters. Could you clarify?
3. Even though it acts only on the null space, it was not clear to me why the injected noise (ie \sigma_h) has to be exponential in H
4. Does the algorithm need to know that the MDP has deterministic transitions? In other words, if applied to a stochastic MDP, would the algorithm still make sense or would it break for some reason?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper designs the first computationally and statistically efficient algorithm for linear Bellman complete MDPs with deterministic transitions. Instead of adding a quadratic exploration bonus and truncating the value function, this paper adds carefully designed random noises to the regression results so that it can (a) achieve optimism, and (b) prevent error amplification.

### Strengths
-	This paper designs the first computationally and statistically efficient algorithm for linear Bellman complete MDPs with deterministic transitions, which serves as a solid step toward solving the long-standing open question about the statistical-computational gap in the Bellman complete setting.
-	This paper is well written, and the algorithm is straightforward and easy to understand. The technical proofs in the appendix is also well written and easy to follow.
-	This paper clearly discussed the limitations of prior works in the linear Bellman complete setting (Section 3.1 and 3.2). Although the idea of adding noise to the regression result is not new, applying this idea to the specific setting requires non-trivial analysis on the exponential blowup of the norm of the parameters, which is novel in this paper.

### Weaknesses
-	My major concern is the organization of this paper --- although the high-level ideas are discussed in detail, all the proofs, including the proof sketch, are deferred to the appendix. I encourage the authors to at least discuss how the key designs of the algorithm (e.g., adding noise to the null space of the current data) enable the regret analysis.
-	One of the major assumptions in this paper is that the transition is deterministic (Assumption 1). However, this assumption is not very well discussed in the paper. The only place where Assumption 1 is used is Lemma 10, which only computes the value decomposition. If I understand correctly, when the transition is not deterministic, a variant of Eq (10) still holds where one only needs to take an expectation w.r.t the randomness of the trajectory on the RHS. Therefore, it is unclear to me what is blocking the analysis from extending to MDPs with stochastic transitions.
-	In the proof sketch (line 885 in particular), the definition of U_t is deferred to Appendix C.2. However, without its definition, it is impossible to understand the proof outline in B.3.
-	Line 1662: typo in the displayed equation, mismatched parenthesis.

### Questions
-	Example 1 in Section 3.1 is confusing. First of all, is \mu(s_2) a typo in Line 195? It seems to me that Example 1 proves that, there exists a tabular MDP and a particular feature mapping, such that the norm of the ground-truth parameter has unbounded l2 norm. In other words, this example does not prove that, linear Bellman complete MDPs with unbounded ground-truth parameters is a strict superset of those with bounded ground-truth parameters, since it is still possible that the same MDP can be represented with a different set of features.

### Soundness
4

### Presentation
4

### Contribution
4
