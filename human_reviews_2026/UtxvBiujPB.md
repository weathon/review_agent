# Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Partial agent failure becomes inevitable when systems scale up, making it crucial to identify the subset of agents whose compromise would most severely degrade overall performance. In this paper, we study this Vulnerable Agent Identification (VAI) problem in large-scale multi-agent reinforcement learning (MARL). We frame VAI as a Hierarchical Adversarial Decentralized Mean Field Control (HAD-MFC), where the upper level involves an NP-hard combinatorial task of selecting the most vulnerable agents, and the lower level learns worst-case adversarial policies for these agents using mean-field MARL. The two problems are coupled together, making HAD-MFC difficult to solve. To solve this, we first decouple the hierarchical process by Fenchel-Rockafellar transform, resulting a regularized mean-field Bellman operator for upper level that enables independent learning at each level, thus reducing computational complexity. We then reformulate the upper-level combinatorial problem as a MDP with dense rewards from our regularized mean-field Bellman operator, enabling us to sequentially identify the most vulnerable agents by greedy and RL algorithms. This decomposition provably preserves the optimal solution of the original HAD-MFC. Experiments show our method effectively identifies more vulnerable agents in large-scale MARL and the rule-based system, fooling system into worse failures, and learns a value function that reveals the vulnerability of each agent. Code available at https://anonymous.4open.science/r/VAI-5F61/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new problem formulation of adversarial attacks in large scale multi-agent reinforcement learning. The problem formulation focuses on adversarial attacks of mean field control. The attacker can select a subset of agents to be adversarial and design an adversarial policy for them with bounded similarity against the intact policy. Adversarial agents execute that adversarial policy, and the rest agents are victims which execute the intact policy.

This paper further proposes a bi-level hierarchical algorithm to solve the problem: (1) on the upper level, the algorithm incrementally selects adversarial agents following the objective of minimizing the worst performance that can be achieved with the selected agents as adversarial; (2) on the lower level, given the subset of adversarial agents selected by the upper level, the algorithm finds adversarial policies for them that can lead to the worst performance. The lower level problem is solved by mean-field Bellman operator derived from Fenchel-Rockafellar duality. The effectiveness of the algorithm is empirically justified by out-performing baselines on 17 of 18 tasks of finding the strongest adversarial attack under the problem formulation.

### Strengths
The problem formulation of adversarial attack in large scale multi-agent reinforcement learning proposed by the paper is useful for diagnosing the robustness of a policy: The set of worst case adversarial agents can indicate which agent is vulnerable, and the worst performance can reveal the robustness of the policy.

The paper proposes an effective algorithm to tackle the problem, justified by their empirical study. Moreover, the mean-field Bellman operator derived from Fenchel-Rockafellar duality can efficiently estimate the worst performance given the set of adversarial agents, in terms of both accuracy and running time.

### Weaknesses
1. The impact of this paper is limited, as:

(a) it only studies under the setting of mean field control, which assumes agents are heterogeneous and can be approximated by mean field approximation. This assumption is no longer true under the setting of adversarial attack, where the behavior of adversarial agents are different from victim agents;

(b) in the problem formulation (Assumption 3.2), the difference between adversarial and intact policies of each adversarial agent is bounded by the same factor $\epsilon$. This formulation excludes adversarial attack where different adversarial agents are perturbed by different $\epsilon^i$. The latter can potentially lead to stronger adversarial attacks that are hard to detect in terms of overall behavior of all agents;

(c) the paper is limited to tasks with discrete action space.

2. Estimation of upper level reward faces sampling error caused by sampling trajectories. The paper lacks the analysis of how error in estimating upper level reward affects the performance of the adversarial attack. The paper can be better if make both theoretical and empirical discussion on this.

3. In empirical evaluation, environments are quite simple (scale of state and action is not big). It is unclear how the method performs in more complicated environments.

4. A few places are not well written, which makes the paper difficult to understand:

Line 197: How is the trajectory collected?

Line 266: Why is the first inequality true? By definition, $(\hat{\mathcal{B}}^{\hat{\pi}}V^i)(s^i,\mu)=\min_{\pi_\alpha}(\mathcal{B}^{\hat{\pi}}V^i)(s^i,\mu)\le(\mathcal{B}^{\hat{\pi}}V^i)(s^i,\mu)$. Also there may be a typo at the end of line 264.

Figure 1: What do scattered points and curves mean?

Table 1: What is the rule to bold text? It should be consistent, e.g. only bold the best method.

### Questions
1. Can the method be extended to find adversarial attacks for more general multi-agent reinforcement learning set up: (a) no assumption of heterogeneous agents; (b) potentially different $\epsilon^i$ for different agents; and (c) multi-agent reinforcement learning tasks with continuous action space?

2. Is there a bound for the error in estimating upper level reward? How many trajectories need to be collected to ensure the error is within an acceptable range? How does the performance (overall reward of following adversarial-attacked policy) change with respect to magnitude of error?

3. How does the method perform in more complicated environments, e.g. SMAC[1]?

4. Please address confusing parts in writing that are mentioned in **Weaknesses**.

5. In Table 1, VAI-Greedy is better than VAI-RL on some tasks. Do the authors have any insight on why this happens?

[1] Samvelyan, Mikayel, et al. "The starcraft multi-agent challenge." arXiv preprint arXiv:1902.04043 (2019).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the detection of individual agents that could potentially have a significant impact on the performance of all other agents if taken over by an adversary. The authors formalize this problem as a bi-level optimization problem: the upper level selects the most vulnerable agents and the lower level learns the worst-case policies for those agents. They demonstrate that selecting the most vulnerable agents is NP-hard and that learning adversarial policies is a mean-field multi-agent reinforcement learning (MARL) task. The authors additionally propose a practical algorithm and demonstrate the effectiveness of attacks conducted using their approach.

### Strengths
1. The paper explores identifying vulnerable adversaries by assuming weaker adversaries and larger-scale systems than previous work. This results in a relevant problem with reasonable threat modeling.
2. The assumptions are clearly stated and well-defined. The assumptions used in the experiments are reasonable.
3. The method is derived from a regularized mean-field Bellman operator and is shown to learn the optimal selection of vulnerable agents and adversarial policies.
4. The empirical evaluation of the method is extensive and shows strong results in three environments.

### Weaknesses
1. Including more complex environments, such as the ones chosen here, would be beneficial. Currently, it is unclear how the method scales to more complex systems. However, I don't think this is a significant issue given the compelling nature of the current experiments.
2. It would have been interesting to see qualitative results showing how the policy of an adversarial agent differs from that of a benign one. Specifically, can the selected agents only degrade performance because they were originally achieving high rewards, or can they also influence other agents in the system to perform poorly?
3. I find the paper's writing to be slightly confusing. The abstract and introduction read more like a paper on detecting vulnerable agents as a defense mechanism, while the methods and experiments are written more like an attack against MARL systems. The authors should update their writing or presentation of the results to address this discrepancy.
4. Related to the previous point, investigating simple defense methods would contribute to the paper. For example, one could investigate adversarial training methods, in which agents are trained alongside adversarial agents in the same environment.

### Questions
1. What does the behavior of an adversarial agent look like on a qualitative level? Can they influence other agents, or are they simply the most effective agents in the system?
2. How do you envision the practical application of this method? Is it intended to be an attack method against large-scale systems or a defense mechanism against adversarial policies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Given a set of agents, this paper presents an approach to identify the most vulnerable subset, including the subset identification and the training of an adversarial policy for that subset. The problem is modeled and studied theoretically, and two algorithms are proposed (a RL approach, and a greedy approach). Experiments are then performed in 3 different domains.

### Strengths
- The problem is interesting and relevant.

- The theoretical study and the model seems ok.

- The results are significantly better than baselines.

- Additional analysis are conducted to help explain the results.

- The writing is clear, overall.

### Weaknesses
By coincidence, I already reviewed an earlier version of this paper, and my main concerns of the previous version were (i) only the case \epsilon = 1 was studied, when the attacker has full control over the subset of agents; (ii) the computational cost was not studied; (iii) there were typos and grammar issues in the text, although the writing overall was ok; (iv) the source code was not available at that time.

I see that in this version the authors released the source code anonymously, and added appendix D.1 and appendix D.2, with the study of the computational cost, and the study of different values of \epsilon (i.e., different levels of control by the attacker).

However, these new additions seem to have been added somewhat quickly, and a few issues still remain:

- In appendix D.1, which environment was used? And what is the variance, confidence interval, etc? Also, only 2 cases are studied, so it is hard to say how it scales as the agent number grows.

- In appendix D.2, the variance is also not shown. Additionally, authors run experiments only in one environment (Battle), while the experiments in the main paper uses 3 environments. It is unclear why we would study different values of \epsilon only in the Battle environment.

- Additionally, the text in the appendix has several writing issues, e.g.:

-- "agents) since it do not requires additional training.", "Our VAI-RL is fast since it do not need"-> "it does not"

-- "First, VAI-RL works constantly better for VAI-Greedy" -> you meant "VAI-RL works constantly better than VAI-Greedy"?

-- "the result also shows our VAI-RL and VAI-Greedy works better in various perturbation budgets ϵ" -> You meant better than baselines?

-- "the effect of our VAI is consistent with this constained budget." -> You meant "constrained"?

Hence, in general, although the paper does have new additions, they seem rushed, and are not in the same level of quality as the previous experiments in the main paper.

### Questions
- Which environment was used in appendix D.1?

- What is the variance of the results in appendix D.1 and D.2? Are the results statistically significant?

- Why only study different values of \epsilon in the Battle environment? Would the results still be positive in different environments?

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
This paper studies an optimal attack problem in multi-agent reinforcement learning (MARL). The authors consider the setting where a subset of the agents can be attacked, whose policy will be modified by the attacker with certain budget constraint. The authors show that this optimization problem is NP-hard. The authors propose a method that efficiently solves the lower-level problem for finding the worse-adversary policy. For the upper-level problem, the authors formulate the corresponding combinatorial optimization problem into an MDP problem, which can then be solved using standard RL algorithms.

### Strengths
1.	This paper provides a complete study of the problem, including its complexity analysis and algorithm design and analysis.
2.	Comprehensive numerical results are provided to support the performance of the proposed algorithm.

### Weaknesses
1. The motivation of the attack model introduced in Section 3.1 needs some further explanation. In particular, why the perturbed attack model is considered, and why the adversary faces a budget constraint $\epsilon^i$ (i.e., is there any physical meaning of this constraint on the adversary)? Does the adversary need to now the agent’s original policy $\pi_{\beta}^i$. In general, a worst-case adversary could potentially alter the agent’s original policy in an arbitrary manner.

2. Proposition 3.3 and its proof need to be checked again. In particular, when you want to prove a problem is NP-hard, the reduction is from a known NP-hard problem to the problem that you want to prove. The direction seems to be reversed in the paper.

3. While the authors prove that the optimization problem in Eq.(2) is NP-hard, the authors the prove in Proposition 4.5 that the algorithm they propose solve the problem optimally. Either the algorithm proposed is not a polynomial algorithm or the solution must lose optimality. An explicit analysis of the computational complexity (e.g., run time) of the proposed algorithm needs to be provided. In addition, in Section 4.2, the authors propose a greedy-type algorithm, i.e., at each step $k$, the algorithm selects the most vulnerable agent, adds it to the selected set of vulnerable agents, and proceeds to the next step. In general, such kind of greedy algorithm for NP-hard optimization problem would return only an approximate solution. Based on the above, it would be good to check again the result in Proposition 4.5 and its proof.

4. Typos: line 325, is it correct to refer to Eqn. 4.2? line 286, to prove that.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
