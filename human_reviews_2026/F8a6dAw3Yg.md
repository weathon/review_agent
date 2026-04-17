# Safe In-Context Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
In-context reinforcement learning (ICRL) is an emerging RL paradigm where the agent, after some pretraining procedure, is able to adapt to out-of-distribution test tasks without any parameter updates.
The agent achieves this by continually expanding the input (i.e., the context) to its policy neural networks.
For example,
the input could be all the history experience that the agent has access to until the current time step. 
The agent's performance improves as the input grows, without any parameter updates.
In this work,
we propose the first method that promotes the safety of ICRL's adaptation process in the framework of constrained Markov Decision Processes.
In other words,
during the parameter-update-free adaptation process,
the agent not only maximizes the reward but also minimizes an additional cost function.
We also demonstrate\footnote{All the implementations will be made publicly available and are now located in the supplementary materials.} that our agent actively reacts to the threshold (i.e., budget) of the cost tolerance.
With a higher cost buget, the agent behaves more aggressively, and with a lower cost budget, the agent behaves more conservatively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents the first approach to achieving safe ICRL under CMDP, enabling safe adaptation to OOD scenarios without requiring parameter updates. The proposed method first performs safe supervised pretraining in an offline manner, followed by further refinement through EPPO in the online environment. Experiments conducted in the SafeDarkRoom and SafeDarkMujoco environments demonstrate the method’s strong OOD adaptation capability.

### Strengths
- This work investigates a novel problem setting, Safe ICRL, and proposes a principled solution to address it.
- To account for the unique characteristics of ICRL, where multiple trajectories are used as history, the authors improve upon the conventional primal–dual framework by proposing the EPPO algorithm, which updates the Lagrange multiplier based on the maximum cost observed across multiple trajectories.
- Experiments conducted in manually designed environments demonstrate the method’s ability to achieve robust OOD adaptation.

### Weaknesses
- The authors should more clearly and explicitly motivate the study of this problem. From my perspective, Safe ICRL does not appear to be a compelling research problem. First, the authors use a “dark” setup in their experimental environments—turning off all radar and local observation information and only retaining the agent’s own state—and create OOD scenarios by changing obstacle and target locations under this restricted setting. However, I doubt such tasks occur in real-world applications, since sensors like radar are typically available. With radar, changes in obstacle distribution might not constitute true OOD scenarios. Second, ICRL requires continual exploration and trajectory collection to adapt to new environments. While this is acceptable in standard RL, in Safe RL such exploration repeatedly generates new unsafe interactions, which can be risky.
- The motivation for the proposed method is insufficiently clear. The stated goal of Safe ICRL is to enable more efficient adaptation to OOD scenarios, yet the main technical contribution appears to be online reinforcement pretraining—a technique that is common in online safe RL. The authors should more thoroughly explain why reinforcement pretraining improves OOD adaptation. Is it because the online phase yields more or higher-quality safe trajectories for training? If the benefit is merely due to increased quantity or quality of data, then the claimed motivation rooted in ICRL may be inconsistent.
- The experimental environments are relatively simple and too few. The authors should evaluate on more tasks in SafetyGymnasium, e.g., construct multi-task scenarios in MuJoCo Velocity tasks by varying safe velocity thresholds, etc.
- The experimental baselines are insufficient.
    - The supervised pretraining baseline is purely offline, whereas the proposed method is offline + online; the settings and amounts of training data therefore differ, making the comparison unfair and insufficient to demonstrate the advantage of the proposed method.
    - The paper does not adequately compare EPPO with other online safe RL algorithms (e.g., RCPO [1], CVPO [2], CAL [3], RESPO [4]); only a comparison with a naive primal–dual method in the appendix for a single environment is shown, and cost results are not provided, which is unconvincing.
    - The authors should also compare with more recent baselines in the ICRL literature to demonstrate the effectiveness of their cost-oriented design in the safe setting.
    - Additionally, in OOD experiments, comparisons should be made to meta-safe-RL methods under consistent offline/online settings.

[1] [Reward constrained policy optimization](https://arxiv.org/abs/1805.11074)

[2] [Constrained variational policy optimization for safe reinforcement learning](https://proceedings.mlr.press/v162/liu22b.html)

[3] [Off-policy primal-dual safe reinforcement learning](https://arxiv.org/abs/2401.14758)

[4] [Iterative reachability estimation for safe reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dca63f2650fe9e88956c1b68440b8ee9-Abstract-Conference.html)

### Questions
Were the reported cost values in the experiments normalized? If yes, what kind of normalization was used? If not, given that the cost values are already small in the zero-shot setting, does this indicate that the experimental environments are too simple in terms of safety constraints?

### Soundness
2

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
3

### Summary
This paper proposes Safe In-Context Reinforcement Learning (Safe ICRL) by introducing Constrained Markov Decision Processes (CMDPs) into the in-context RL framework.  The approach consists of two parts: Safe Supervised Pretraining and Safe Reinforcement Pretraining, each with its own optimization algorithm designed to handle CMDP constraints.  Experiments are conducted mainly on the SafeDarkRoom and SafeDarkMujoco environments.

### Strengths
-   The paper provides an attempt at theoretical analysis of the proposed optimization method.

### Weaknesses
1.  The CMDP formulation in the paper appears to consider only a single cost function, which limits generality.
    
2.  In Theorem 1, the logical order of the proof is somehow  problematic — the existence of the fixed point  should be established before proving that it corresponds to a primal optimal solution.
    
3.  The experimental evaluation is insufficient. The paper does not compare against established ICRL baselines, making it difficult to assess effectiveness and safety guarantees relative to the state of the art. 
    
4.  The language and structure of the paper need further polishing and clarification.

### Questions
1.   For the proposed iterative algorithm:  Does it converge to a fixed point?  
2.  Supervised pretraining behavior: In SafeDarkMujoco-Point and SafeDarkMujoco-Car, supervised pretraining does not produce the expected monotonic increase in reward with episode return. What explains this mismatch ？
3.  Cost-to-go vs. return: Figure 2b shows no clear correlation between cost-to-go and return, suggesting that cost-to-go does not effectively control cost tolerance under supervised training. Why does this occur？

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces and formalizes the problem of Safe In-Context Reinforcement Learning. The goal is for an agent, after a pretraining phase, to adapt to new, out-of-distribution tasks and satisfy safety constraints without any parameter updates. The agent must rely solely on its context to learn and adapt its behavior. Authors propose safe supervised pretraining that imitates learning histories conditioned on return- and cost-to-go, similar to algorithm distillation and decision transformer; and safe reinforcement pretraining that iteratively updates the policy and Lagrangian multipliers in an online manner, called Exact Penalty Policy Optimization (EPPO). Experiments on variants of previous safe/meta-RL benchmarks show that EPPO can achieve good performance while adapting to cost constraints, but safe supervised pretraining struggles in some environments.

### Strengths
- Tackles an interesting problem with potential importance in areas like robotics.
  
- Evaluation is performed in OOD environments, a rather challenging setup.
  
- Theoretical guarantees are provided.

### Weaknesses
- Both methods the authors proposed seem to be rather straightforward extensions of existing algorithms with limited technical novelty.
  
- Comparison with baselines is lacking. While it is stated that methods in this paper are parameter-update-free, they can arguably still be compared with safe meta-RL approaches with parameter updates under fair setups (e.g. same online interaction budget).
  
- The authors discussed the issue of the number of evaluation episodes in the methods section but did not conduct any OOD experiments in this regard.

### Questions
- How does the methods compare with existing safe meta-RL approaches?
  
- Can EPPO maintain its performance when given OOD episode number $K$?

### Soundness
4

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
2

### Summary
This paper studies the safety problem in in-context reinforcement learning. The problem is formalized as maximizing the accumulated return from different episodes in context, while ensuring each episode does not violate the cost constraint. The paper proposes EPPO, a safe reinforcement pre-training algorithm that solves the problem and demonstrates its effectiveness through experiments. In particular, EPPO modifies the dual problem of the original optimization problem so that the optimization does not need the knowledge of in-context episodes, while being provably equivalent to optimizing the original problem.

### Strengths
1. This paper studies a relevant problem to an emergent training paradigm, a.k.a. the safety problem in ICRL.
2. The proposed algorithm EPPO provably solves the original optimization problem (maximizing reward constrained by fixed cost budget), while being parameter-free in the number of evaluation episodes.

I am not familiar with the literature on ICRL, so I give a low confidence score.

### Weaknesses
1. Though the paper mentioned applications of ICRL in language tasks, the safety framework does not seem appropriate for languages. RL in language models is usually not formalized in the way of MDP, and it is usually hard to define a cost function.
2. The approach requires that in the pre-training phase, we already know what the cost function is, whereas this is not realistic in a lot of practical scenarios.
3. The writing is confusing in some paragraphs. For example, I do not understand why the policy in reinforcement pre-training also depends on CTG and RTG.

Minor typo:
1. Line 163: can depend

### Questions
1. Is the framework described in the paper applicable to language tasks? If yes, how would you solve the problem of defining the cost function, and that RL in language model is usually not formalized in MDP?
2. Success of pretrained models greatly depends on the usage of offline data, and their general capability to adapt to different downstream tasks. Is it possible to find a way so that we do not need knowledge about the safety constraints during pre-training?
3. Compared to regular training (without considering safety), how much does safety pre-training impact the capabilities of the models?

### Soundness
3

### Presentation
2

### Contribution
2
