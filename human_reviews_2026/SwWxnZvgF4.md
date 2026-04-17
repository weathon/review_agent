# Reinforcement learning for saddle-point equilibria without full state exploration

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
We introduce a new fixed-point condition on the state-action-value $Q$-function for zero-sum Markov turn games that suffices to construct saddle-point and security policies, but is less restrictive than the classical condition arising from the Bellman equation. We then propose an iterative algorithm that guarantees convergence to a function satisfying this less restrictive condition. The key benefit of the new condition and algorithm is that convergence to a saddle-point can (and typically will) be reached without full exploration of the state-space; generally enabling the solution of larger games with less computation. Our algorithm is based on a limited form of exploration that gathers samples from repeated attempts to certify the current candidate policies as a saddle-point, motivating the terminology "saddle-point exploration" (SPE).  We illustrate the use of the new condition/algorithms in several combinatorial games that can be scaled in terms of the size of the state and action spaces. Numerical results, using both tabular and neural network $Q$-function representations, consistently show that saddle-point policies can be formally certified without full state exploration and, for several games, we can see that the fraction of states explored decreases as the size of the game grows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies finding saddle-point equilibria in a zero-sum Markov Game. The paper shows that exact optimality can be achieved without global convergence of the Q-function—it suffices to converge locally on the reachable subspace where the policies interact strategically. Based on this new finding, the authors propose SPE. For deterministic finite games, the algorithm provably converges to a restricted fixed point and terminates in finite time (Theorem 3).

### Strengths
1. The authors propose a new fixed-point condition for the state–action value function. Unlike the standard Bellman equation (which must hold for all states), this condition only needs to hold on a restricted subset of the state space — specifically, the states reachable under the current candidate policies.

2. The authors propose the SPE algorithm and prove the convergence to a restricted fixed point and termination in finite time for deterministic finite games.

3. In all experiments, SPE reached a certified saddle-point without exploring the entire state space. As game size increased, the fraction of explored states decreased, showing strong scalability.

### Weaknesses
1. The main convergence theorem (Theorem 3) only holds for deterministic zero-sum turn-based games with finite termination. However, in many real-world applications, the game is stochastic. 

2. Although the authors claim their algorithm does not need to search the whole state space, the paper does not derive theoretical sample complexity or convergence rates, showing their method is better than previous methods. 

3. In the experiments, all test domains are turn-based, finite board games. Moreover, the experiments benchmark against a hypothetical lower bound rather than comparing to concrete algorithms

### Questions
Please see the weakness part for my concern.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies Q-learning in turn-based two-player Markov zero-sum games. The authors introduce the notion of restricted fixed points for an operator on the Q-function and show that a saddle point can be derived from such a fixed point. Furthermore, an advantage of the proposed algorithm is that it only explores a subset of the entire state space, leading to improved efficiency compared to previous Q-learning algorithms.

### Strengths
- The paper characterizes sufficient conditions under which the Q-learning operator is contractive, thereby establishing sufficient conditions to compute such restricted fixed point.
- The computation of those restricted fixed points only requires visiting a subset of the entire state space.
- Numerical experiments are provided to validate the theoretical results.

### Weaknesses
- The paper focuses exclusively on deterministic zero-sum Markov games, where given the state and action, the next state is deterministic. This setting is significantly simpler than standard stochastic games, in which one of the main challenges arises from the uncertainty in state transitions.

- Theorem 3 only holds for games that terminate in finite time (which is different from Markov games with finite horizon). This assumption appears quite restrictive as it imposes strong structure constraints on the reward function of each state. In standard RL literature, the $\gamma$ discount factor is commonly used to ensure the value function remains bounded (and thus is absolutely summable). it would be helpful for the authors to justify why they adopt this assumption instead of following the conventional discounted setting.

- Despite claiming that the explored state space is a subset of the entire state space, the paper provides no theoretical results on the upper bound of the number of states visited by the algorithm. Thus it is difficult to theoretically quantify the improved state dependency of the proposed algorithm.

- Only an asymptotic convergence guarantee is provided for the proposed algorithm.

### Questions
- In line 244-246, the authors state that every fixed point is a restricted fixed point but the converse is not true. Can the author provides a specific game example where the converse fails to hold?

- In line 296, the authors mention that SPE is applicable to general zero-sum turn games. Can the author specify with more details? Is there any convergence guarantee when the operator fails to be contractive?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies two-player zero-sum turn-based Markov games. While the saddle point of this problem can be found by finding the fixed point of the Bellman operator over all state-action pairs, this paper shows that finding a fixed point over a certain subset of state-action pairs is sufficient. The paper proposes an algorithm, SPE, that finds the saddle point, possibly without exploring all state-action pairs.

### Strengths
The paper is nicey formatted and is easy to follow. The idea, theoretical guarantees, and the experiement are presented clearly.

### Weaknesses
1. The contribution of this paper does not seem strong enough for acceptance.

- Given the definition of a saddle points, specifically by Eqs. (6a) and (6b), Theorem 2 seems quite straightforward.

- The discussion is limited to deterministic games, which limits its generality.

- The theoretical sample complexity of Algorithm 1 is not provided other than that it is finite, which is limited against the compared works.

I think alpha-beta pruning is a more systematic and efficent way to implement the same idea of not searching the entire state-action space for two-player zero-sum deterministic finite-time turn Markov games that this paper focuses on. It prunes states that can only be reached by clearly suboptimal actions.
It is proved that its number of searches (which corresponds to number of sampled states in this paper) can be as small as $\mathcal{O}(\sqrt{S})$ for favorable instances, while maintaining correctness [A].
Although the setting may be slightly different, the difference seems negligible and the proposed algorithm and its theoretical guarantees seem inferior to this well-known algorithm. At least, I think some comparison must be made.

2. The proof of Theorem 2 is slightly confusing. In the proof of Theorem 2, it seems like the notations $\pi_ 1^\ast$ alternates denoting the true saddle point and the induced policy of the restricted fixed point. Lines 684-705 regard it as the induced policy from the restricted fixed point, but Line 706 states that the pair $(\pi_ 1^\ast, \pi_ 2^\ast)$ is in $\Pi^* $, which I thought was the goal of the proof. From then on it seems like $\pi_ 1^\ast$ and $\pi_ 2^\ast$ denote the true optimal policy.

[A] Knuth, D. E., & Moore, R. W. (1975). An analysis of alpha-beta pruning. Artificial intelligence, 6(4), 293-326.

### Questions
Theorem 2 introduces a new concept "feedback saddle point", however it is not explained in the paper.  
In line 828, $s < s'$ would hold if $s'$ is reachable from $s$, but I don't think it is "if and only if".

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies two-player zero-sum turn-based Markov games and proposes a new restricted fixed-point condition on the Q-function that guarantees saddle-point and security policies without requiring full state-space exploration. Based on this condition, the authors develop the Saddle-Point Exploration (SPE) algorithm, which alternates between best-response learning and Q-function updates. For deterministic finite games, the authors prove that SPE converges in finitely many steps to a restricted fixed point and yields a pair of saddle-point policies.

### Strengths
The paper introduces a theoretically grounded perspective by relaxing the full Bellman fixed-point requirement to a restricted subset of states. The resulting formulation is conceptually interesting and could inspire further work on certifiable learning under partial exploration. I didn’t check the full proof, but the finite-time convergence guarantees for deterministic games seems to be rigorous and technically solid.

### Weaknesses
The overall novelty appears limited. The main idea—constructing saddle-point policies from a restricted fixed-point condition—closely resembles existing work, particularly Anderson et al. (2025), which already studied finite-computation Q-learning for zero-sum turn games. The differences here mainly concern framing and sample selection, which seem incremental and insufficiently contrasted with prior work. There’s no explicit comparisons with previous works in terms of the problem settings and assumptions, theoretical results and empirical performance (note that in the Related Work section, there’s only comparison and statement for general Q-learning and zero-sum Markov games, which is not the focus of this paper.) 

Further, the paper’s framing and exposition could also be improved. The introduction overstates generality by referring to Markov games broadly before restricting to turn-based settings later, which seems a bit deceiving. The contribution relative to existing literature should be articulated more precisely.

The central claim of “learning without full state exploration” is also overstated. In practice, the restricted set still corresponds to the reachable states under certain policies, which are typically the states any standard learning process would visit. This makes the contribution conceptually modest, since avoiding unreachable states is expected and does not drastically improve efficiency.

The theoretical scope is narrow, as all guarantees assume deterministic, finite-horizon, turn-based dynamics, without discussion of approximate results under stochastic or continuous settings.

### Questions
Could the authors clarify how the proposed “restricted fixed-point” condition fundamentally differs from the standard Bellman fixed point beyond the scope restriction? Specifically, how does it change the theoretical or algorithmic implications compared to prior formulations such as Anderson et al. (2025)?

The paper claims to learn “without full state exploration,” but the restricted set still seems to include all reachable states under the equilibrium policy. Could the authors elaborate on how to identify this restricted set and whether this reduction offers measurable computational savings?

Have the authors considered extending the theoretical results to stochastic or partially observable settings? Even a discussion of potential obstacles or approximate guarantees in such cases would help assess the broader applicability of the approach.

### Soundness
3

### Presentation
2

### Contribution
2
