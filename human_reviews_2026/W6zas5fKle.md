# Safety-Biased Policy Optimisation: Towards Hard-Constrained Reinforcement Learning via Trust Regions

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Reinforcement learning (RL) in safety-critical domains requires agents to maximise rewards while strictly adhering to safety constraints. Existing approaches, such as Lagrangian and projection-based methods, often either fail to ensure near-zero safety violations or sacrifice reward performance in the face of hard constraints. We propose Safety-Biased Trust Region Policy Optimisation (SB-TRPO), a new trust-region algorithm for hard-constrained RL. SB-TRPO adaptively biases policy updates toward constraint satisfaction while still seeking reward improvement. Concretely, it performs trust-region updates using a convex combination of the natural policy gradients of cost and reward, ensuring a fixed fraction of optimal cost reduction at each step. We provide a theoretical guarantee of local progress toward safety, with reward improvement when gradients are suitably aligned. Experiments on standard and challenging Safety Gymnasium tasks show that SB-TRPO consistently achieves the best balance of safety and meaningful task completion compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Existing safe RL methods either fail to ensure near-zero safety violations or severely degrade reward performance in the hard-constraint setting. The paper proposes extending TRPO to the CMDP setting. The key idea is not to explicitly enforce that the cost is zero (or below a fixed threshold) at every update. Instead, the method introduces a surrogate cost constraint requiring that the cost of the new policy be no greater than that of the previous policy. The algorithm then uses the conjugate gradient method from TRPO to separately compute the update directions for the reward objective and for the surrogate cost objective. It then forms a convex combination of these gradient directions. The paper proves that this convex combination maximizes a linearized objective that balances reward improvement and surrogate cost reduction.

### Strengths
1.The paper extends TRPO to a hard-constraint setting and provides a proof of a performance improvement guarantee.

2.It clearly discusses the difficulty of safe RL under hard constraints in CMDPs, and shows that many existing methods fail in this regime.

3.The proposed method is conceptually simple and appears easy to implement.

### Weaknesses
1. CPO already extends TRPO to constrained MDP (CMDP) settings. Therefore, the paper should make explicit what the key differences are between CPO and the proposed method, and which design choices lead to better performance in practice. I did not find a clear discussion of this.

2. My understanding is that CPO provides a performance improvement guarantee, but still performs poorly in hard-constraint settings. This may be due to its approximate solution procedure and the fact that it tries to maintain feasibility at every step. The proposed method also claims a performance improvement guarantee, but I do not see a clear explanation of why it would behave better than CPO in practice under strict (near-zero) safety constraints. This needs to be articulated more clearly.

3. The training curves in Figure 4 suggest that the proposed method still does not consistently achieve both near-zero safety violations and high reward in the hard-constraint setting. It appears that the problem is not fully solved. Please explain, task by task, what is happening in Figure 4 (for example, when cost remains non-zero or reward collapses).

### Questions
1. Why do you not report a metric that reflects task completion under safety, such as the cost per successful episode (i.e., episodes that actually reach the goal / complete the task)? The “safety probability” and “safe rewards” metrics you define focus on constraint satisfaction, but if the agent fails to complete the task, then having near-zero cost is not very meaningful. In many safe RL algorithms, when the constraint is enforced as a hard (near-zero) constraint, the agent simply fails to solve the task at all.

2. I am familiar with the Safety Gymnasium tasks. In your Table 1, many tasks have very low or even negative reward values (e.g., Point Push, Point Button, Point Goal, Car Push, Car Circle, Car Goal). This suggests that the policies often fail to complete the task. How should we interpret “safety probability” in such cases? Is the agent just behaving very conservatively and not solving the task?

3. There are no video rollouts or qualitative demonstrations comparing different methods. This makes it hard to judge whether the learned policies are actually completing the tasks versus just being safe by doing nothing. Please provide qualitative evidence (e.g., videos or behavioral descriptions) to show what the learned policies are doing for each algorithm.

4. In Lemma 3, you assume ⟨g_c, Δ_c⟩ ≤ −ϵ. This assumption may not hold in general. Could you justify why this is reasonable in practice? As written, it is not obvious to me that this condition always holds, and the later argument depends critically on it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript proposes a new optimizer for constrained Markov decision processes, which constitute an important model for safe reinforcement learning. The new method, called Safety-Biased Trust Region Policy Optimization (SB-TRPO) updates the policy's parameters according to an adaptive convex combination of the natural policy gradients of the reward and cost. Both, a theoretical analysis and empirical comparison to some existing methods are presented.

### Strengths
+ Very well writte, easy to understand and parse. 
+ Addresses a timely and important problem of safe policy optimization. 
+ Proposes a new way of adaptively weighting reward and cost natural policy gradients. 
+ Clean and simple method, which provides promising empirical results.

### Weaknesses
I mainly see one weakness: The manuscript is completely missing any discussion of a recent line of works on constrained TRPO (C-TRPO) and its proximal version (C3PO). Here, the trust regions consisting of only safe policies are designed through a mixture of the common KL geometry and a cost-dependent term (Milosevic et al. 2025, Milosevic et al. 2025). 

+ Embedding Safety into RL: A New Take on Trust Region Methods, Milosevic et al., 2025
+ Central Path Proximal Policy Optimization, Milosevic et al., 2025

### Questions
+ Can you elaborate on the connection of SB-TRPO and C-TRPO? Both theoretically as well as empirically. I know this is a big task to ask in a short rebuttal, but as the two approaches seem very close in nature (not necessarily identical), I feel that a comparison would be very interesting and is required for publication. I am willing to raise my score, if you can incorporate the following: 
	+ Comparison of obtained update directions: How does a mixture of geometries relate to a mixture of the two updates? The resulting updates can be plotted for some problems. 
	+ Comparison of theoretical guarantees
	+ Most importantly: empirical comparison
+ What is the intuition behind forcing a decrease of the cost during optimization? From my understanding $L_{c, \pi_{old}} (π) \le L_{c,\pi_{old}} (π_{old}) − \epsilon$ is stronger than usual. In particular, if $\pi_{old}$ is safe, then the cost can no longer be decreased. I wonder, whether it would be more flexible to define the CMDP as usually that the cost is required to be below a certain threshold $b$ and to only require the reduction of the cost as long as it is above the threshold $b$. Note that one obtains the current formulation by setting $b=0$ and considering a non-negative cost. 
+ How do you solve the trust-region formulations?

### Soundness
2

### Presentation
3

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
The paper introduces SB-TRPO, a modification of Trust Region Policy Optimization (TRPO) for constrained reinforcement learning. The authors extend the TRPO objective by adding a penalized term to jointly account for reward maximization and cost minimization within a constrained Markov decision process (CMDP) framework. The goal is to improve stability and constraint satisfaction compared to existing approaches. The paper provides theoretical analysis and reports experiments on benchmark environments to demonstrate improved trade-offs between performance and safety metrics.

### Strengths
- Clarity and structure: The paper is well organized and clearly written, making it easy to follow the main ideas.
- Principled approach: The modification to the TRPO objective is logically motivated and mathematically consistent with the trust-region framework.
- Relevance: Addressing stability and safety in policy optimization is an important and timely problem in reinforcement learning.
- Theory and implementation: The theoretical discussion of the proposed surrogate objective adds rigor and helps frame the contribution.

### Weaknesses
1. Incremental contribution.
The method extends an existing algorithm (TRPO) in a relatively straightforward way by adding a penalty term to handle constraints. The conceptual novelty and theoretical differences from prior constrained TRPO or penalty-based methods are limited and should be clarified.
2. Comparative evaluation.
The experimental section would benefit from stronger and more diverse baselines. The paper appears to compare primarily with older constrained RL variants (e.g., TRPO-Lagrangian). Including a wider range of recent or more competitive baselines would make the results more convincing. See for example the methods discussed in Milosevic et al. 2024 (https://arxiv.org/abs/2411.02957) 
3. Empirical depth.
The evaluation seems limited to a few standard environments. Additional experiments or analyses (such as sensitivity to penalty parameters, ablation of components, or convergence stability) would strengthen the claims.
4. Conceptual clarity.
The motivation section criticizes CMDP formulations for tolerating constraint violations but then employs the same CMDP structure. The authors should explain why this formulation remains appropriate in their setting and how their modification mitigates the stated limitations.
5. Potential bias in the surrogate.
Since the objective introduces a new penalty term, it would be helpful to discuss whether this affects convergence to the true constrained optimum or introduces bias.
6. Figures and intuition.
Some theoretical sections could be complemented by short intuitive explanations or clearer figures illustrating how the penalty affects the policy update.

### Questions
- Contradictory motivation: the authors criticize CMDP formulations for tolerating nonzero costs but still adopt one for their solution. This should be addressed more clearly. Can you clarify how the proposed approach addresses the CMDP limitations mentioned in the introduction?
- Does the modified surrogate guarantee convergence to the true constrained optimum, or to an approximate penalized solution?
- How sensitive is the algorithm’s performance to the penalty coefficient and trust-region size?
- How does the method behave under stricter or multiple constraints?

### Soundness
3

### Presentation
3

### Contribution
2
