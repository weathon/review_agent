# Learning Multiple Coordinated Agents under Directed Acyclic Graph Constraints

- Decision: Reject
- Scores: 3, 6, 3, 6

## Abstract
This paper proposes a novel multi-agent reinforcement learning (MARL) method to learn multiple coordinated agents under directed acyclic graph (DAG) constraints. Unlike existing MARL approaches, our method explicitly exploits the DAG structure between agents to achieve more effective learning performance. Theoretically, we propose a novel surrogate value function based on a MARL model with synthetic rewards (MARLM-SR) and prove that it serves as a lower bound of the optimal value function. Computationally, we propose a practical training algorithm that exploits new notion of leader agent and reward generator and distributor agent to guide the decomposed follower agents to better explore the parameter space in environments with DAG constraints. Empirically, we exploit four DAG environments including a real-world scheduling for one of Intel’s high volume packaging and test factory to benchmark our methods and show it outperforms the other non-DAG approaches.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a Multi-Agent Reinforcement Learning (MARL) approach designed to train multiple coordinated agents, taking into account the constraints of directed acyclic graphs (DAG). The highlighted features of this solution include: 

1. From a computational standpoint, the authors leverage the inherent properties of DAG structures to enhance the solution's efficiency. This is realized by introducing innovative concepts such as the leader agent, reward generator, and distributor agent. These elements guide the subordinate follower agents in a more targeted exploration of the parameter space in environments governed by DAG constraints.
 
2. On a theoretical level, the proposed solution serves as a lower bound for the optimal value function. This is made possible by incorporating a surrogate value function supplemented with synthetic rewards. 

Furthermore, the authors validate the efficacy of their method on four distinct DAG environments, underscoring its advantages in comparison to non-DAG strategies.

### Strengths
- This paper introduces an approach to address multi-agent coordination challenges within the framework of directed acyclic graphs (DAG), a context particularly relevant to real-world scenarios like industrial process control.

- The solution proposed is very interesting, with a special emphasis on the reward generator. This component effectively shapes rewards at the multi-agent coordination level, aligning with the specificities of DAG constraints.

- From a theoretical standpoint, the method appears robust and well-founded.

- To underscore the effectiveness of their approach, the authors have carried out comprehensive experiments on tasks under DAG constraints.

### Weaknesses
- The motivation of this work, as it stands, could be better articulated. While I recognize that problems with multiple subtasks are important for multi-agent coordination, it's not clearly conveyed why these specifically fall under multi-agent reinforcement learning or what challenges they present. Given that the work's contribution is centered on the DAG structure, it might be beneficial for the authors to use an illustrative example for a clearer understanding.

- In line with the above point, I find it somewhat challenging to connect the proposed solution with the problem described. For instance, the leader is trained to generate goal vectors for followers, but how does this align with the DAG constraint of the problem? How can we determine that a proposed goal-vector results in "better coordination"? A clearer outline of the challenges and more detailed motivations for the solution would be helpful.

### Questions
1. If different agents deal with different subtasks within the DAG, then do there exist interactions between agents in this setting? 

2. Is it possible to directly derive the optimal value function for the DAG? If not, what impediments exist in directly computing this function? Might it be feasible in a more simple setup, such as in a tabular setting?

3. The proposed approach maintains "abstract" messages between the leader agent and its followers. Why this level of abstraction? Could this characteristic of non-interpretability pose challenges to the method's practical application?

4. With respect to the synthetic reward, how is an agent's contribution to the team reward assessed quantitatively? If a particular subtask significantly influences the coordination task but has several preceding subtasks, are these ancestors also assigned with high synthetic rewards?

5. The study appears to incorporate several assumptions:
a)	In P3: “the team reward is the sum of the rewards obtained from sinks”
b)	In P4: “there can be a function $f_{ik}$ that measures the contribution of agent $i$ to sink agent $k$’s reward and …”
c)	In P6: “the synthetic reward for the follower $i$ is determined based on its contributions to the sink followers among its descendants”

Do these assumptions still hold in real-world scenarios? Might they limit the broader applicability of the proposed method? A more in-depth discussion on this matter would be appreciated.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the problem of a Markov decision process with DAG constraints, which is prevalent in some real-world problems. To deal with the issues of delayed rewards and lack of individual incentives, the paper proposes the idea of synthetic rewards, which are based on each agents' contributions to the down-streaming cooperative tasks. Theoretically, it proves that the surrogate value function based on the synthetic rewards is the lower bound of the  optimal value function. Empirically, the paper proposes a training algorithm to generate the synthetic rewards. The proposed algorithm is tested on several MARL environments.

### Strengths
- The setting is very interesting. Many practical problems have inter-dependent subtasks which can be framed by the proposed MDP with DAG constraints. 
- The idea of synthetic rewards is well motivated, since the team reward cannot well capture the individual contribution. Further, it is theoretically justified by the paper. 
- A practical algorithm to generate the synthetic rewards is proposed and has achieved better performance on several MARL environments.

### Weaknesses
- The DAG constraints are predetermined, which may need some domain-knowledge and human annotations.
- Compared to decentralized algorithms such as independent Q-learning, the proposed practical algorithm requires a centralizer to generate the synthetic rewards, which may not always be available.
- In the practical algorithm, the goals generated by the leader are not interpretable. It is not clear why it is beneficial for the MARL problems.

### Questions
- There are no error bars in the Figure 3 and 5. Are the experiments tested on multiple seeds?
- Can the authors visualize the generated synthetic rewards and how does it captures the individal contribution? For example, when does the RGD generate low rewards and high rewards in the provided testing environments?
- The purpose of the goals generated by the leader is not clear to me. Can the authors explain why it is beneficial?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the challenges of Multi-agent Reinforcement Learning (MARL) in scenarios with complex subtask dependencies, represented by Directed Acyclic Graphs (DAGs). To tackle the issues of delayed rewards and reward distribution among agents, the authors introduce a novel algorithm featuring a "leader" that generates abstract goals for agents, and a "Reward Generator and Distributor (RGD)" to coordinate agents based on their contributions. The approach aims to enhance agent coordination and optimize their performance in intricate, real-world applications.

### Strengths
This paper studied a novel problem that hasn't been considered before. The idea is very novel. The empirical results show that their method could be efficient.

### Weaknesses
1. The writing could be improved. Currently, the problem-setting section is very confusing.

2. There is a GitHub url in this paper, which violates the double-blind principle.

### Questions
1. Please further decertify the problem-setting.
2. Is this DAG setting common in some real-world scenarios?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel framework to solve the MDP-DAG problem with synthetic rewards as the rewards of the agent itself and its contribution to its descendants. Two new learning components, the leader agent and the reward generator and distributor, are introduced to solve the problem. The proposed method shows improved performance on multiple different tasks compared to several baselines.

### Strengths
The paper is well-written and the contributions are novel. The motivation behind the proposed learning components is clear. Decomposing agents' contributions to team rewards makes the inner learning simpler. The experiment results are comprehensive, with various environments and baselines.

### Weaknesses
As mentioned in the discussion, theoretical analysis is missing for the convergence and optimality of the learned reward distributor.

The proposed method is very complex multiple inner and outer cycles, as well as additional learning components. May need more results or analysis on the stability and sensitivity to hyper-parameters of the proposed method.

### Questions
Can more results or analysis on the stability and sensitivity to hyper-parameters of the proposed method, in addition to the analysis of the goal period length in Figure 6?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
