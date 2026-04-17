# Optimal Robust Subsidy Policies for Irrational Agent in Principal-Agent MDPs

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 2

## Abstract
We study a principal-agent problem in a Markov Decision Process where the principal provides subsidies to influence the agent's policy, which in turn determines the accrued rewards. Our focus is on designing a robust subsidy scheme that maximizes the principal’s cumulative expected return, even when the agent displays bounded rationality and may deviate from the optimal action policy after receiving subsidies.

As a baseline, we first analyze the case of a perfectly rational agent and show that the principal’s optimal subsidy coincides with the policy that maximizes social welfare, the sum of the utilities of both the principal and the agent. We then introduce a bounded-rationality model: the globally $\epsilon$-incentive-compatible agent, who accepts any policy whose expected cumulative utility lies within $\epsilon$ of the personal optimum. In this setting, we prove that the optimal robust subsidy scheme problem simplifies to a one-dimensional concave optimization, revealing that optimal subsidies concentrate along social-welfare-maximizing trajectories. We also bound the associated loss in social welfare. Finally, we investigate a finer-grained, state-wise $\epsilon$-incentive-compatible model. In this setting, we show that under two natural definitions of state-wise incentive-compatibility, the problem becomes intractable: one definition results in a non-Markovian agent action policy, while the other renders the search for an optimal subsidy scheme NP-hard.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper considers the Principal-Agent problem on Markov Decision Processes (MDP) where the Principal sets subsidies to the Agent in a form of an additive factor to the Agent's reward function. In turn, the Agent optimises its subsidised cumulative reward over a finite horizon MDP.
The main contributions of this paper are theoretical insights into optimal subsidy schemes under the assumption of bounded rationality of the Agent. In particular, three cases are considered; perfect rationality, $\epsilon$-incentive compatible Agent over episodes, and $\epsilon$-incentive compatible Agents over states.
In the first case, the authors show that the Principal's optimal payoff is the difference between the social welfare payoff and the Agent's unsubsidised payoff and provide an analytical solution to the optimal payoff scheme that only needs to subsidies actions corresponding to the social welfare maximizing actions.
In the second case, the authors show that the Principal's problem can be reformulated as a one-dimensional concave optimization, therefore, optimal subsidy schemes can be found with first-order optimization methods.
In the third case, the authors show that the Principal's optimization problem is NP-hard.

### Strengths
1. The paper is easy to follow and has a clear structure that builds from section to section.
2. Considering non-rational Agents in the Principal-Agent problem is a fundamental issue and having Principal actions that are robust to bounded rationality is crucial for applications.

### Weaknesses
1. The paper lacks discussion on how results connect to previous publications and what is the novelty of the contributions. While many recent publications are mentioned briefly in the Related Works section, without reading each of them it is hard to position the paper's contribution to the field. I strongly suggest the authors to provide more details for each Proposition and Theorem they prove what is the novelty compared to previous results.
2. The setting is limited to finite-horizon and finite state-action space MDPs with additive subsidy schemes. While this is a nice setup for toy experiments, for the results to be more relevant for the applications of Principal-Agent problems this is a clear limitation without any guidance on how the results might generalise to more complex spaces or Principal actions. Furthermore, as the authors also highlight the Conclusion this work is limited to settings where the Principal knows the Agent's reward function which does not hold for most applications, therefore, results when the Principal has to learn the Agent's incentives would be more relevant.
3. The paper lacks experimental results demonstrating the theoretical results. The Principal-Agent problem has many applications as the authors also highlight in the Introduction, therefore, empirical evaluations on some of the applications would solidify the theoretical results. Especially, ablations on the parameter $\epsilon$ to show how bounded rationality affect the outcomes of the problem.

### Questions
1. The authors claim the the Agent's optimal policy is deterministic in the perfectly rational case but might be stochastic in the bounded rational setting. Could they elaborate on this claim and at least provide pointers that show these results formally?
2. On page 7 Line 355-373, the authors discuss Markovian vs. Non-Markovian policies and claim that the Principal and the Agent adopt non-Markovian strategies in the globally $\epsilon$-IC case. Could the authors formalise the notion of non-Markovian in the case of finite-horizon MDPs and elaborate on what is the difference between non-Markovian and time-dependent?

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
4

### Summary
The paper studies a principal-agent problem. The agent performs actions in a Markovian environment. The principal offers subsidies to change the agent's reward. The agent acts optimally (either exactly or approximately) based on the subsidized rewards. The goal of the principal is to find a subsidy scheme so that the agent's best response gives the highest possible value to the principal. The authors studied both the case where the agent responds in a perfectly optimal way and the case where it does so only approximately. For both cases, the paper characterizes optimal subsidy schemes. The paper also analyzed the relation between the social welfare gap and the agent's rationality bound. Moreover, the paper considers a stronger constraint which arises when the gap between the agent's maximum attainable value and the policy they use is bounded below an epsilon threshold at every time step. The authors argue that in this case, the agent's induced policy may be non-Markovian, which introduces additional complexity to the problem. To work around, they considered an alternative greedy state-wise epsilon-IC condition, but even with that the problem remains NP-hard.

### Strengths
I think the problem studied is interesting and natrual. The paper is very clear and easy to follow. I didn't check the full proofs but the proof sketches are helpful for understanding the ideas, and the arguments look sound. The characterization results about optimal subsidy schemes and their relation to social welfare maximizing policies are very interesting, so is the relation between the social welfare gap and epsilon. The paper is overall solid and I enjoy reading it.

### Weaknesses
- The part in Section 5 could be explored a bit more. The authors introduced an alternative type of agent in an attempt to overcome the complexity caused by history-dependency, but even that is intractable. The paper would be stronger if a tractable alternative is introduced.

- Some related works are missing, but this is minor. The idea of subsidizing the agent's reward is a special case of the reward design problem. There are many papers in this direction which also adopt a worst-case perspective in the face of uncertainty about the agent's response; see, e.g., Admissible Policy Teaching through Reward Design, Banihashem et al., 2022. Similar principal-agent subsidy problems have also been studied under fairness considerations: Envy-free Policy Teaching to Multiple Agents, Gan et al., 2022. And there are similar principal-agent models based on Markov games, e.g. Stochastic Principal-Agent Problems: Efficient Computation and Learning, 2023.

### Questions
- Does your proof of Theorem 4.1 also imply that in the fully rational case, it is also always optimal for the principal to use a Markovian subsidy scheme.

- Would you be able to show that the problem against a state-wise epsilon-IC agent (Definition 5.1) is NP-hard?

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
4

### Summary
This paper studies a principal-agent Markov Decision Process (MDP) problem where the principal can offer state-action-dependent subsidies to influence the agent's action policy.  The paper considers both perfectly rational (best-responding) agent as well as $\epsilon$-best-responding agent.  The paper shows that:

1. For perfectly rational agents, the principal's optimal subsidy can be characterized easily: it is the highest achievable social welfare minus the agent's default value. 
2. For $\epsilon$-best-responding agents, taking the worst case for the principal among the $\epsilon$-best-responding policy sets, the paper shows that the principal's maximin problem can be converted into a single-dimensional concave maximization problem that admits an efficient solution. 
3. Lastly, the authors consider a special case of $\epsilon$-best-responding -- state-wise $\epsilon$-IC -- in which case the optimal subsidy turns out to be NP-hard to compute.

### Strengths
(S1) The theoretical result for $\epsilon$-best-responding agent, where the authors show that the maximin problem can be converted into a single-dimensional concave problem (using a duality approach) is very interesting.  The fact that such an optimization problem can be solved efficiently is not obvious at all, especially given some previous works on $\epsilon$-best-response in other principal-agent problems, such as information design [Yang & Zhang, NeurIPS 2024, Computational Aspects of Bayesian Persuasion under Approximate Best Response].

(S2) Good theoretical coverage.  The main $\epsilon$-best-response model is in the ex-ante sense: the agent computes the overall expected utility.  The authors also consider the state-wise $\epsilon$-best-response model, establishing a clear boundary between tractable and intractable cases.  And both Markovian and non-Markovian policies are discussed.  

(S3) The paper is very well written, with rigorous definitions, helpful examples, and intuitive proof sketches.

### Weaknesses
(W1) A related literature is "steering agents in games by payments", e.g., [1, 2, 3], which studies how a principal can add payment (subsidy) to a game to induce certain outcomes among the players.  This literature studies multi-player games, which are more general than the single-agent problem in this paper.  But on the other hand, this literature usually considers static or repeated game, which is less general than this paper's MDP model.  [2, 3] consider no-regret learning agent, which is a form of $\epsilon$-best-responding agent considered by this paper.  Given the conceptual similarities, I think a discussion to this literature is needed.  Nevertheless, I view this as a minor weakness because the techniques are different due to modeling differences.  



---------------------------------

[1] Monderer & Tennenholtz, 2004, K-Implementation.

[2] Zhang et al, EC 2024, Steering No-Regret Learners to a Desired Equilibrium.

[3] Zhang et al, 2025, Learning a Game by Paying the Agents.

### Questions
## Questions for the authors

(Q1) Do you have any results for **Markovian** state-wise $\epsilon$-IC agent?  Namely, if the agent uses a Markovian policy that is $\epsilon$-optimal at every state, can the principal find the optimal subsidy efficiently?  (I am not sure whether Theorem 5.1 applies to Markovian state-wise $\epsilon$-IC.) 



## Suggestions

* For the future direction "consider scenarios in which the principal does not have prior knowledge of the agent's reward function or the value of $\epsilon$, such as in a learning-based setting", [3] is very related.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the principal-agent MDP problem where the agent is assumed to be boundedly rational in the sense that the agent may take any suboptimal responses up to epsilon gap. This problem is modeled as a robust, minimax game: the principal tries to find a subsidy policy that maximizes social welfare, while the agent responds with the worst-possible, sub-optimal deviation. The paper shows that, if the agent response is globally epsilon suboptimal, the optimal subsidy scheme can be effectively determined. Meanwhile, if the agent response is the state-wise ϵ-IC, finding the optimal subsidy is NP-hard.

### Strengths
This paper makes a good effort to extend the setting of principal-agent MDP problems. Modeling agent’s bounded rationality moves beyond the perfect-world assumptions of classical game theory and makes the work directly applicable to real-world systems where users are not perfect optimizers.

### Weaknesses
It is unclear why the paper chose to focus on the welfare maximization problem instead of the more common objective to maximize the principal’s utility. I suspect the results of this paper will no longer hold under the objective to maximize the principal’s utility. Therefore, the authors should clarify on this modeling choice and explain how the current solution might rely on the welfare maximization objective.

### Questions
What's the motivation or is there any technical result that this work focus on the welfare maximization objective?

### Soundness
3

### Presentation
2

### Contribution
2
