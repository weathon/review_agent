# Imitation Learning as Return Distribution Matching

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
We study the problem of training a risk-sensitive reinforcement learning (RL) agent through imitation learning (IL). Unlike standard IL, our goal is not only to train an agent that matches the expert’s expected return (i.e., its *average performance*) but also its *risk attitude* (i.e., other features of the return distribution, such as variance). We propose a general formulation of the risk-sensitive IL problem in which the
objective is to match the expert’s return distribution in Wasserstein distance. We focus on the tabular setting and assume the expert’s reward is *known*. After demonstrating the limited expressivity of Markovian policies for this task, we introduce an efficient and sufficiently expressive subclass of non-Markovian policies tailored to it. Building on this subclass, we develop two provably efficient algorithms—RS-BC and RS-KT —for solving the problem when the transition model is unknown and known, respectively. We show that RS-KT achieves substantially lower sample complexity than RS-BC by exploiting dynamics information. We further demonstrate the sample efficiency of return distribution matching in the setting where the expert’s reward is *unknown* by designing an oracle-based variant of RS-KT. Finally, we complement our theoretical analysis of RS-KT and RS-BC with numerical simulations, highlighting both their sample efficiency and the advantages of non-Markovian policies over standard sample-efficient IL algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the problem of risk-sensitive imitation learning (IL) in tabular finite-horizon Markov Decision Processes (MDPs). Traditional IL focuses on matching the expert's occupancy measure, which only imitates the expected return (average performance) and is inherently risk-neutral, ignoring aspects like variance or tail behavior in the return distribution. The authors reformulate IL as "Return Distribution Matching" (RDM), where the goal is to train an agent that matches the expert's entire return distribution (encoding risk attitude) in Wasserstein distance, rather than just the expectation or limited risk measures like Conditional Value at Risk (CVaR). They consider settings where the expert's reward is known (primary focus) or unknown, and the transition model is either unknown (no-interaction/offline) or known. The aim is to develop provably sample-efficient algorithms that handle non-Markovian experts, who may exhibit complex risk-sensitive behaviors not capturable by Markovian policies.

### Strengths
1. The analysis carried out is sound and well-motivated. 
2. The paper is polished and easy to read. 
3. The problem setting is interesting. 
4. The empirical results are encouraging.

### Weaknesses
1. Primary focus assumes known r^E, which may not hold in real IL (e.g., human demos); unknown-reward variant relies on oracles, potentially impractical. If the expert’s rewards were known, we could have solved the forward RL, which allows us for the full access to the rich literature: function approximation, generalization, etc. Assumptions: Requires known rewards (or oracle for unknown); no online interaction for learning transitions.
2. There is no compelling empirical evidence presented in the main body that validates whether the correct distribution is being learned (the results presented in the main body only show the overall performance). 
3. There is a figure in the appendix that seems to show the learned distribution, however this figure is hard to understand, and can likely be significantly improved. I strongly encourage the authors to improve this figure and include it in the main body.

### Questions
1.It is not clear what about Algorithm 1 is ‘risk-sensitive’. It seems to just be learning a policy, without a clear link to the return distribution.
2.The paper could benefit from a broader discussion as to how the proposed methods connect to the distributional RL literature. For example, how does the proposed method relate to a categorical or quantile representation.
3.Since we know the expert's rewards, wouldn't it be better to solve the forward RL with the known reward to completely restore the expert's policy? What are the benefits of the proposed methods over the forward RL?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents the risk-sensitive imitation learning problem as matching the expert's return distribution in Wasserstein distance, and designs two types of algorithms, RS-BC (without model, based on behavior cloning non-Markov strategy class) and RS-KT (known transition, constructing LP search occupancy measure to match distribution), for the cases with/without known transition models. It provides theoretical sample complexity bounds (Theorem 4.3/4.4/5.1/5.2) and conducts a large number of simulation experiments.

### Strengths
1. Directly promoting risk-sensitive imitation learning to return distribution matching is more comprehensive in depicting "risk attitude" than merely matching expected value or several quantiles, which has both theoretical and intuitive appeal.
2. This paper proposed a  subclass of non-Markovian policies $\Pi(r_\theta)$ that can be stored and computed. The sample complexity bounds for RS-BC (no interaction) and RS-KT (known transition) are provided, with clear analysis and comparison. The theorem gives upper bounds for various scenarios.
3. The experimental design is clear, aiming to answer four specific research questions (Q1-Q4).

### Weaknesses
1. This paper proposes the main algorithms (especially RS-BC and RS-KT), which all assume that the expert's reward function $r^E(s,a)$ is known, or at least the reward for each step can be accurately calculated from the expert's trajectory. Section 5 of the paper explicitly points out that this is an important limitation, and it specifically discusses the "unknown $r^E$" case, but this part only stays at the theoretical analysis level (oracle-based) and does not propose an actual runnable algorithm.
2. The theoretical guarantees depend on strong assumptions (e.g., known transitions, exact LP solution), yet the experiments use approximations without quantifying the induced errors, leaving the practical scalability and validity of the theoretical bounds unclear.

### Questions
1. The RS-KT algorithm needs to solve an LP with a scale of $\mathcal{O}(SAH|\mathcal{Y}^{\theta}|)$, which seems difficult to scale in practice. Have the authors considered extending it to an approximate method in the continuous domain (for example, using function approximation to replace tabular occupancy measures)?
2. Can the algorithm be extended to an online interactive version, which can learn unknown transitions and unknown expert rewards from the environment. Maybe the author can add more discussion on this approach and the comparison with risk-sensitive RL.

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
3

### Summary
The paper studies the problem of risk-sensitive imitation learning. The goal is to train an risk-sensitive agent that matches the expert’s return rather than merely the expected return. They show that Markovian policies are in general insufficient for this task,  and introduce an efficiently storable subclass of non-Markovian policies based on discretized cumulative reward. They propose  two provably efficient algorithms, RS-BC and RS-KT, when the transition model is unknown and known, respectively for the tabular setting. They prove sample complexity guarantees for both algorithms, show a polynomial-sample result for the unknown-reward robust variant. They also provide simulation evidence that RS-BC and RS-KT better match expert return distributions than standard baselines in a range of synthetic MDPs.

### Strengths
- The problem reformulation of risk-sensitive IL is clear and well-motivated. Since return contains for relavant distribution  for risk-sensitive decision-making, matching return distributions can generalize expectation/CVaR matching to general risk-attitude imitation.
- The paper is well organized and the main ideas are presented clearly. The transitions are fluent and follow a logical flow.
- All three algorithms are conceptually simple and intuitive. Both RS-BC and RS-KT are computationally efficient and  come with  sample complexity guarantees. For the known reward setting, the RS-KT algorithm attains a strong sample bound independent of S,A. These sample complexity results with different dependence regimes, e.g., with and without transition knowledge are novel and interesting.
- The paper provides empirical validations across many random MDPs, which shows the proposed methods reduce Wasserstein error relative to standard baselines.

### Weaknesses
- One major limitation is the computational cost of RS-KT. It requires solving an LP with size scaling as $O(S A H |Y_\theta|)$. For moderate/large tabular MDPs this can become expensive.
- Unknown-reward setting lacks practical algorithm. Theoretical sample-efficiency for the unknown-reward case is established by using an oracle-based argument, but a concrete practical algorithm for this setting is deferred. This is an important practical case since expert reward unknown.

### Questions
questions/comments
-  Could the authors provide an experiment comparing RDM to a strong CVaR-matching IL method across a range of \alpha levels, to show where full distribution matching yields material improvement?  
- how to choose \theta in practice is still not very clear
- Lower bounds is missing (the authors have noted this as a limitation)
- There are some small typos and places where exposition could be tightened 
- Missing related works, to name a few,  sample complexity/regret guarantees on risk-sensitive RL [1][2], CVaR RL [3][4], risk-sensitive distributional RL [5][6]

[1]Fei, Yingjie, et al. "Risk-sensitive reinforcement learning: Near-optimal risk-sample tradeoff in regret." Advances in Neural Information Processing Systems 33 (2020): 22384-22395.

[2]Fei, Yingjie, et al. "Exponential bellman equation and improved regret bounds for risk-sensitive reinforcement learning." Advances in neural information processing systems 34 (2021): 20436-20446.

[3]Bastani, Osbert, et al. "Regret bounds for risk-sensitive reinforcement learning." Advances in Neural Information Processing Systems 35 (2022): 36259-36269.

[4]Wang, Kaiwen, Nathan Kallus, and Wen Sun. "Near-minimax-optimal risk-sensitive reinforcement learning with cvar." International Conference on Machine Learning. PMLR, 2023

[5]Liang, Hao, and Zhi-Quan Luo. "Bridging distributional and risk-sensitive reinforcement learning with provable regret bounds." Journal of Machine Learning Research 25.221 (2024): 1-56.

[6]Chen, Yu, et al. "Provable risk-sensitive distributional reinforcement learning with general function approximation." arXiv preprint arXiv:2402.18159 (2024).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies training a risk-sensitive reinforcement learning agent through imitation learning. The classical imitation learning methods only focus on matching the expert's expected return (average performance), which may not capture risk attitudes. To overcome this, the authors propose matching the entire expert return distribution, such as variance and tail risks. The approach formulates a return distribution matching (RDM) problem using the Wasserstein distance between distributions.

The authors propose two provably efficient algorithms, RS-BC (Risk-Sensitive Behavior Cloning) for unknown transition models and RS-KT for known transition models that learn a class of non-Markovian policies that efficiently express risk-sensitive behaviors. These algorithms improve over standard imitation learning by matching the full return distribution, accommodating risk attitudes. They provide theoretical guarantees on sample complexity and policy approximation, and then empirically validated that RS-BC and RS-KT outperform baseline methods.

### Strengths
**The following are the strengths of the paper:**
1. This paper considers training a risk-sensitive reinforcement learning agent through imitation learning while matching the entire expert return distribution to capture the risk attitudes. 

2. The authors propose two provably efficient algorithms, RS-BC and RS-KT, that learn a class of non-Markovian policies that efficiently express risk-sensitive behaviors, while having theoretical guarantees on sample complexity and policy approximation.

3. Finally, the authors empirically validated that RS-BC and RS-KT outperform baseline methods.

### Weaknesses
**The following are the weaknesses of the paper:**
1. The proposed algorithms are restricted to finite tabular RL environments, limiting their scalability to RL problems with large or continuous state spaces. 

2. It is unclear how expensive Step 2 of RS-KT is, especially for large LP. How practical it is to assume the expert's reward function (also the transition model for RS-KT) is known.

### Questions
Please address the weaknesses of the paper. I am open to changing my score based on the authors' responses.

### Soundness
4

### Presentation
3

### Contribution
3
