# Convergence and Connectivity: Asymptotic Dynamics of Multi-Agent Q-Learning in Random Networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Beyond specific settings, many multi-agent learning algorithms fail to converge to an equilibrium solution, instead displaying complex, non-stationary behaviours such as recurrent or chaotic orbits. In fact, recent literature suggests that such complex behaviours are likely to occur when the number of agents increases.
In this paper, we study Q-learning dynamics in network polymatrix games where the network structure is drawn from classical random graph models.  In particular, we focus on the Erdős-Rényi model, which is used to analyze connectivity in distributed systems, and the Stochastic Block model, which generalizes the above by accounting for community structures that naturally arise in multi-agent systems. 
In each setting, we establish sufficient conditions under which the agents' joint strategies converge to a unique equilibrium. We investigate how this condition depends on the exploration rates, payoff matrices and, crucially, the probabilities of interaction between network agents. 
We validate our theoretical findings through numerical simulations and demonstrate that convergence can be reliably achieved in many-agent systems, provided interactions in the network are controlled.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the convergence of Q-learning dynamics (QLD) - a model of continuous time dynamics which balances exploitation with exploration through a parameter T_k for each player k - in random networks, i.e., networks in which nodes are agents (players) and games between agents are played if an edge is randomly assigned between them. For the theoretical results, the paper assumes that the game across all edges is the same, i.e., a bimatrix game with matrices A, B. The paper shows that a sufficient condition for QLD to converge is that the exploration parameter is large enough for all players in the network, and specifically if it is larger than the product of the highest eigenvalue of the adjacency matrix and a "similarity of payoffs" index. The paper sharpens this result in two settings of random networks - one in which each edge is assigned randomly with probability p and one in which nodes form clusters with probabilities of edges being different within and between these clusters. Finally, the paper contains experiments in 3 types of games which demonstrate that the sufficiency condition provided in the main results can be likely further generalised.

### Strengths
The paper is mathematically rigorous and the results are correct to the extent that I could verify. There are enough details to reproduce the experiments and the main result (Lemma 1 and its two instantiations in Theorems 1 and 2) are neat. The main finding that convergence is easier with less connections in the networks is also were formulated.

### Weaknesses
- The theoretical results are established under limiting assumptions, i.e., that the game played across each edge is the same. Experiments in a game with different payoffs across edges aim to demonstrate the contrary. However, the experimental and theoretical analyses as presented seem to not disentangle the effect of cooperation or competition in the underlying game from the other effects that are studied, i.e., exploration parameter, number of agents, and density of connections.
- The result that convergence of Q-learning ensues for higher exploration rates is already well-known in simpler settings and the technical contribution to establish this result in random networks doesn't seem to substantially generalise existing techniques or insights.

### Questions
- Can the authors provide a more detailed comparison to the predecessor paper of this work, Hussain et. al (2024)? In particular, I think
- For the Sato game, the similarity index should be high (or not?) since the payoffs are aligned, but convergence occurs even for low exploration rates. Does this imply that the sufficient condition in Lemma 1 can be significantly tightened in some cases? 
- Related to the above, the authors could discuss more the choice of games and how they cover interesting cases. 
- The paper mentions in the introduction (lines 79-80) about an apparent contrast with the finding in Sanders. That statement confused me and it would be great if they could clarify what they mean.

In general, I find the paper clean, stating its assumptions and mathematics in a transparent way, and I like the condition derived in Lemma 1 (despite being not tight in many cases). I found the heatmaps maybe not the optimal choice to plot the boundaries since the boundary is essentially a line in all of them (except maybe in Figure 7 in the appendix where the boundary seems less well defined). My concern is that the paper - at least to the extent that I understood it - doesn't provide a significant extension over existing knowledge in the field and the selected experiments although well aligned with previous work, don't seem to explore all interesting cases (or at least are not discussed sufficiently). While I may have misunerstood some parts, I think that disentagling the impact of cooperative/competitive underlying game on convergence and doing more extensive experiments in games not covered by (the limiting) assumption 1, would strengthen the paper. In general, while extending the analysis to other models of random graphs - unless they provide substantially different motivations and challenges - as mentioned in the conclusions may be interesting, it seems to me that going deeper into the present theoretical results may be more interesting - the breadth could be covered with more experiments.

### Soundness
3

### Presentation
3

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
This paper investigates the convergence properties of multi-agent Q-learning dynamics in network polymatrix games. The central contribution is to establish a theoretical link between the convergence of the learning dynamics and the structure of the underlying interaction network, particularly for large-scale systems. The authors focus on networks generated from random graph models (Erdős-Rényi and Stochastic Block models) and derive sufficient conditions for convergence to a unique Quantal Response Equilibrium (QRE). The key result is that the minimum exploration rate required for guaranteed convergence scales with the expected node degree of the network. This implies that convergence can be reliably achieved even in systems with a very large number of agents, provided the network is sufficiently sparse. The theoretical findings are validated through a series of numerical simulations that demonstrate this trade-off between network density, exploration rate, and the number of agents.

### Strengths
*   The paper tackles the challenge of ensuring convergence in multi-agent learning as the number of agents grows, a key barrier to deploying learning agents in real multi-agent scenarios.
*   It provides a clear and theoretically grounded relationship between network sparsity (expected degree) and the convergence of Q-learning.
*   To control network connectivity as a practical lever for ensuring stability, could be useful for practitioners and multi-agent systems.
*   The theoretical claims are backed by suitable experiments that clearly illustrate the predicted effects across different games and network models.

### Weaknesses
*   The main theoretical results (Theorems 1 and 2) depend on Assumption 1, which implies that every edge in the network represents the same bimatrix game. This is a strong simplification, and the homogeneous graph structure also limits applicability.
*   The analysis is performed on the continuous-time Q-learning dynamics (QLD), which is an idealized model. The paper does not discuss the potential gap between these results and the behavior of the discrete-time Q-learning algorithm that would be implemented in practice.
*   The convergence guarantees are asymptotic. While the experiments confirm the trend for finite N, a discussion on how tight the bounds are for practical, non-asymptotic cases could be useful.

### Questions
1.  Your theoretical results are derived under the strong Assumption 1 (homogeneous payoffs). Could you comment on the primary theoretical hurdles to relaxing this assumption? And how well would the methods work for general graphs from datasets?
2.  The analysis is based on the continuous-time QLD model. How do you expect these guarantees to translate to the discrete-time Q-learning algorithm?
3.  Theorems 1 and 2 provide a lower bound on the exploration rate for convergence. How does this theoretical bound compare to the empirical convergence boundary you found in your experiments (e.g., in Figure 2)?

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
4

### Summary
This paper investigates the convergence performance of continuous-time Q-learning dynamics in network games, focusing on random graphs generated from the Erdős–Rényi model and stochastic block model. When learning the game, each agent independently executes Q-learning updates with softmax strategies. Given sufficient exploration (sufficiently high softmax temperature),  the authors show the asymptotic convergence of the Q-learning dynamics to the quantal response equilibrium—a solution concept that approaches the Nash equilibrium as the exploration rate vanishes—when both the algorithm running time and number of agents tend to infinity.
Experiments on three game models are provided, discussed, and compared with the theoretical results.

### Strengths
- The paper is well-written and the presentation is clear.
- The perspective of exploiting the graph structure for convergence analysis is novel and important.
- The proposed method (Q-learning) and solution concept (quantal response equilibrium) make sense for the problem.

### Weaknesses
### Theoretical Claims

The first concern is more on the presentation of the convergence results (Theorems 1 and 2). The results are asymptotic in terms of both the algorithm running time and the number of players $N$. However, as $N\to \infty$, $T_k \to \infty$, making the convergence point a trivial strategy that uniformly randomizes over the action space, as noted in the paper.

Nevertheless, non-asymptotic convergence results are more desirable. Could the authors comment on the feasibility of deriving non-asymptotic results for either the number of agents or the running time?

The asymptotic nature of the results also cast shadows on the validity of the solution concept QRE, as the results may only hold for large $N$, which corresponds to a high exploration rate and thus renders the QRE uninformative.

### Related Work

Hussain et al. (2024) appears to be a closely related work based on the authors' description and is not a concurrent work as it is heavily cited in the paper. However, the only difference between the two works discussed in the paper is that Hussain et al. (2024) "consider deterministic graphs with specific structure". This is a vague comparison and casts doubt on the significance of the work. The authors should provide a more detailed comparison of the two works to clarify the novelty of the current work, including but not limited to what graph structures are considered in Hussain et al. (2024), how are they fundamentally different from the random graphs considered in this paper, and what results are obtained in Hussain et al. (2024).

Some works on learning dynamic graphon games might be relevant to the paper, as learning unknown payoff functions is the key setting of the paper and extending to Markov games.

### Other

Only continuous-time Q-learning dynamics are considered. The authors may want to discuss the feasibility of extending the results to discrete-time Q-learning algorithms, which are used in practice.

Quadratic payoff with Assumption 1 is a bit restrictive. Moreover, the authors may want to discuss how Assumption 1 affects the players' action space. If all players share the same payoff matrices $(A,B)$, at least it implies all players share the same action space cardinality $|\mathcal{A}_{k}|$. What other implications may Assumption 1 have on the action space?

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes the convergence of continuous-time Q-learning dynamics (QLD) in network polymatrix games where each pairwise interaction shares identical payoff structure. Under this symmetry assumption, the authors prove that if the exploration (temperature) rate T exceeds the product of the "intensity of identical interests" and the spectral radius of the adjacency matrix, then the game is strongly monotone and admits a unique quantal response equilibrium (QRE) to which QLD converges almost surely (Lemma 1). They further derive lower bounds on the exploration rate by establishing asymptotic upper bounds on the spectral radius for Erdos-Renyi and Stochastic Block Model networks, relying on Vu (2007) for the almost-sure spectral bounds.

### Strengths
- The paper draws an explicit analytical link between network connectivity and learning stability.
- The combination of graph spectral analysis with learning dynamics is relevant to multi-agent RL communities.

### Weaknesses
- Overstated scope in the abstract and introduction:
Q-learning primarily studied in the context of Markov decision processes or Markov games with discrete-time updates. The paper's limited focus on normal-form games and the continuous-time counterpart is not explicit in the abstract and introduction until lines 64-72. More importantly, the restrictive Assumption 1 is not clear in the abstract and introduction.
- Implicit information structure:
It is not clear whether agents know the model or can observe the others' strategies. For example, (1) involves r_ki(x_{-k}(t)). Then, how agent k can compute it without knowing r_ki and x_{-k}(t). Note that x_{-k}(t) is the probabilistic strategy rather than the action taken. Furthermore, does that update take place for all actions i. Then, how the agent can know the reward associated with actions not taken without knowing the model. If the agent knows the model, this should be explicit since the literature review criticizes existing works using model knowledge.
- Restrictive symmetry assumption:
The core Lemma 1 depends critically on Assumption 1 assuming identical bimatrix payoffs on all edges. This limits generality. Furthermore, under homogenous formulations, mean-field games (with continuum of agents) and mean-field-type games (with many agents) also have convergent learning dynamics. The statement at line 042-043 saying "...as the number of players grows, non-convergent behavior becomes the norm." is misleading in that regard.
- Presentation of sufficient vs. necessary conditions:
Statements like "higher exploration rates are necessary for convergence" at lines 242-243 are misleading. The results only prove sufficiency, and counterexamples could exist where QLD still guaranteed to converge under weaker conditions.
- Lack of rigorous in asymptotic bounds:
The paper cites spectral radius results from Vu (2007) to establish almost-sure upper bounds but does not quantify the non-asymptotic behavior for finite N. A clear statement would say that for any \epsilon > 0 there exists sufficiently large N such that if T \geq Bound(N) + \epsilon then the convergence of QLD is guaranteed. 
- Ambiguity in the "p\rightarrow 0" statement at liens 304-305:
The claim "if p \rightarrow 0 faster than 1/N, QLD converges almost surely as N\rightarrow \infty" lacks clarity and mathematical rigor. The dominant term p(N-1) can remain bounded, yet we also have \epsilon \sqrt{N}. Therefore, it is not clear with what rate QLD can converge. Furthermore, such a statement could be misinterpreted as by making p go to zero we can ensure convergence while a change in p implies a different game.
- Limited literature review:
The literature review should discuss the seminal works: Shapley (1964) provided a counterexample of two-agent three-action game in which fictitious play does not necessarily converge to unique equilibrium. There is also famous Jordan's game in which fictitious play may not converge. Hart and Mas-Colell showed that for any uncoupled learning dynamics (not incorporating opponent payoffs), there exists a game in which they do not converge to equilibrium. More importantly, some citations are not provided in the references. For example, Leonardos et al (2024) is cited at lines 94 and 192. However, there is no such reference in the references. In the references, there are two papers from Leonardos yet both also were not cited within the main draft.
- Minor comments:
-- Line 158: T_1,\ldots, T_N > 0. 
-- Line 165: "the QRE converges to the Nash Equilibrium" is not a rigorous statement. Do you mean the set of QRE converges to the set of Nash equilibrium?
-- Line 175: ... a discounted history ... is misleading since in the RL context, the discount refers to a geometric decay.
-- Line 184: ... denote ...
-- Line 295: Lemma 6 is in the supplementary material. This should be highlighted. Furthermore, an explicit proof of Theorem 1 and 2 would make the paper more complete. Stating these theorems first and then proving them using auxiliary lemmas would improve the accessibility of the paper.

### Questions
In the supplementary material:
- Lines 965-970: The statement: "We decompose each edge into a half-edge along which A is played and a half-edge along which B is played" is not clear. Can you explain this and the following adjacency matrix decomposition over an example? For example, if we only have two agents, then N = [O, -A; -B, O] and N+N^T = [O, -A-B^T; -B-A^T, O]. Then, what does G_{k\rightarrow l} and G_{l\rightarrow k} refer to?
- Theorem 1.4 in Vu (2007) also has a lower bound on \sigma^2. Why is it not included in Lemma 8?

### Soundness
2

### Presentation
2

### Contribution
2
