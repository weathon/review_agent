# Convergence of an actor-critic gradient flow for entropy regularised MDPs in general spaces

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
We prove the stability and global convergence of a coupled actor-critic gradient flow for infinite-horizon and entropy-regularised Markov decision processes (MDPs) in continuous state and action space with linear function approximation under Q-function realisability.
We consider a version of the actor critic gradient flow where the critic is updated using temporal difference (TD) learning while the policy is updated using a policy mirror descent method on a separate timescale.
For general action spaces, the relative entropy regularizer is unbounded and thus it is not clear a priori that the actor-critc flow does not suffer from finite-time blow-up.
Therefore we first demonstrate stability which in turn enables us obtain a convergence rate of the actor critic flow to the optimal regularised value function.
The arguments presented show that timescale separation is crucial for stability and convergence in this setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies infinite state-action space actor-critic with policy iteration and entropy regularization. It proves stability of the system and proves exponential convergence.

### Strengths
The analysis is done for continuous state-action space.

### Weaknesses
Presentation of results is bit cluttered, its little difficult to understand the contribution. See questions.

### Questions
Q1) Theorem 1.1 and Lemma 1.1 are insightful but I am not sure how novel it is. If it is established then I urge the authors to state it a proposition or property. And lemmas and theorems  be reserved for the results which are novel contribution of the paper.

Q2)  Policy Mirror Descent requires updating all states at each iteration, how can it be extended to online setting where we have access to only samples?

Q3) Corollary 5.1 (stabiliity): The constant $a_2$ >0, then how does the upper bound of $e^{a_2t}$ indicates stability, it is diverging very fast? Same for Corollary 5.2?

Q4) Corollary 5.3: At the limit $\tau \to\infty$, RHS converges to $a_1$. However, intuitively we expect that when the temperature becomes very high (so does the penality of deviating from the current policy) the policy should not change at all. In other words, at this limit $a_1$ should be zero. Is it the case?

Q5)  Authors mention that entropy regularization boosts exploration but policy iteration requires updating all states at each iteration. What benefits  entropy regularization provides to policy iteration?

Q6) In general, policy iteration conveniently convergence to the global optimal policy, and entropy regularization makes it even more stable ()

### Soundness
3

### Presentation
1

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
The paper analyzes an actor-critic method in continuous time with (i) TD learning as the critic, and (ii) policy mirror descent for policy optimization to solve entropy-regularized MDPs with general (Polish) state-action spaces. Under a realizability assumption on Q-functions throughout the trajectory, and for sufficiently small $\gamma$, an exponential convergence rate up to a critic error was established.

### Strengths
- The paper is very well-written and mathematically rigorous. The technical contribution is solid.
- Convergence of an actor-critic method based on TD-critic and PMD-actor was analyzed for MDPs with general (Polish) state-action spaces. The existing works analyze entropy-regularized actor-critic methods for finite action spaces, where the optimality gap has a term $\sqrt{\frac{\log |A|}{t}}$. Since this bound becomes vacuous for continuous action spaces, the contribution of this paper is significant.
- The paper establishes a stability analysis that explicitly captures the interplay between entropy regularization and timescales.

### Weaknesses
- The paper makes a strong assumption that $\gamma$ should be sufficiently small, given that the practical $\gamma$ values are typically larger than 0.9. Indeed, the range of $\gamma$ shrinks as $\tau$ increases (e.g., in Corollary 6.1).
- Equation (40) indicates that the decay rate of the optimality gap is $O(t^{-1/2})$. This makes the exponential convergence argument in the abstract a little confusing. I would recommend revising that statement in the abstract to indicate that it is exponential convergence up to an additional error term that stems from the critic.
- The impact of stochasticity is not characterized as the analysis is fully deterministic. For TD learning, this impact can be particularly significant, and may have significant implications for policy optimization.
- The realizability assumption is strong, but it can be relaxed by including an additional projection error term.
- A continuous action space $A$ would make the computation of an exact softmax policy intractable due to the integration in Line 188-189. As such, in order to sample from this policy, one may either compute an inexact policy (via an approximate integral), or use Langevin dynamics to sample from $\pi$ without explicitly computing it. In any case, this brings an additional error term. For this reason, in practice, this extension of finite-action softmax can be intractable and variants such as deterministic policy gradient (Silver et al., 2014) are used.
- (Minor) There is a continual switching between "regularisation" and "regularization". I recommend making it consistent.

**References**

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. "Deterministic Policy Gradient Algorithms." In International Conference on Machine Learning (pp. 387-395), PMLR, 2014.

### Questions
- An interesting connection between critic error and the policy optimization was established in (Kakade, 2001), which is known as the compatible function approximation. One can formulate the NPG updates as the minimizer of $L_\theta(w) = E_{s,a}[|\nabla \log \pi_\theta(a|s) w - A^{\pi_\theta}(s,a)|^2]$. For linear function approximation, this may roughly be thought as minimizing $E|\phi({s,a})\cdot w - Q^{\pi_\theta}(s,a)|^2$. Since one also uses TD learning with linear function approximation using the same feature map $\phi({s,a})$ to approximate $Q^{\pi_\theta}$, which gives us an exact solution under the realizability assumption, the insight from (Kakade, 2001) also applies here nicely. Is this idea used in the paper in an related/alternative form? The joint Lyapunov analysis that connects the actor and critic may possibly have a connection with this.
- In conjunction with the above point, the realizability seems to be a significant contributor to the exponential convergence rate. What would happen without this realizability assumption? More clearly, how would the projection error propagate?
- Is the small-$\gamma$ assumption inevitable? Additional discussion on its cause would be useful.

**References**

Kakade, S. M. "A Natural Policy Gradient." Advances in Neural Information Processing Systems 14, 2001.

### Soundness
3

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
The paper proves the convergence of an actor-critic algorithm for entropy-regularized reinforcement learning (RL) in the setting with continuous states, actions and time dynamics. Concretely, the paper shows that the algorithm is stable and converges at an exponential rate.

### Strengths
If the proofs are correct and novel, I believe that the paper makes an important contribution.

### Weaknesses
I am well familiar with the theory of entropy-regularized RL, but I am not an expert on continuous-time dynamics, which makes some of the proofs difficult to follow.

### Questions
The authors choose to regularize on a fixed action distribution \mu rather than on a full stochastic policy that selects a different action distribution in each state. Several existing algorithms such as TRPO regularize on full policies, so this seems like a limiting choice. Why do you not regularize on a full policy, and would it be difficult to extend the result to such policies?

What do you mean by a policy being "equivalent" to the reference action distribution \mu? My interpretation would be that the policy selects the same action distribution \mu in each state, but I believe that this is not what you mean.

Why did you choose to minimize the MSBE rather than some other objective? There are other objectives that seem to have better properties for entropy-regularized RL, e.g. Logistic Q-learning.

Equation (23) is a non-regularized objective, so Lemma 4.1 seems to imply that we can upper bound the regularized gradient by a non-regularized gradient. This is curious to me since I thought that bounding the regularized gradient directly would be easier. Why did you choose this particular proof strategy?

Lemma 5.1 looks familiar to me as a flow constraint, but I have a hard time interpreting Lemma 5.2, which looks like a differential equation. What is the interpretation of this lemma?

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
This paper presents theoretical analyses of the Actor-Critic algorithm for continuous action spaces in entropy-regularized Markov Decision Processes (MDPs). In addition, it provides explicit gradient expressions for both the critic and actor updates.

### Strengths
1. The theoretical results presented in this work appear to be sound and logically consistent with the proposed framework. The derivations are coherent, and the assumptions, although somewhat idealized, support the validity of the main conclusions. Overall, the theoretical analysis contributes to a solid understanding of the algorithm’s behavior and convergence properties.

2. In addition, the definitions and problem descriptions are generally clear and well-structured. The authors provide sufficient background and notation, making the paper accessible to readers familiar with reinforcement learning and KL-regularized optimization. The clarity of exposition facilitates understanding of both the motivation and the technical development of the method.

### Weaknesses
1. Some assumptions, such as Assumption 4.1, seem overly strong and may not hold in practical continuous or high-dimensional settings. The convergence results appear to rely heavily on these assumptions.


2. The technical novelty and improvements over prior work are unclear, and the contribution would benefit from a clearer explanation of how it advances existing methods.

### Questions
1. Assumption 4.1 appears to be quite strong and may not hold in practical scenarios, particularly in environments with continuous state and action spaces (potentially high-dimensional). Does the convergence guarantee in this work critically rely on this assumption?

2. Could the authors elaborate on the technical challenges encountered when extending the analysis or algorithm to continuous state and action spaces? How were these challenges addressed in the proposed framework?

### Soundness
3

### Presentation
3

### Contribution
2
