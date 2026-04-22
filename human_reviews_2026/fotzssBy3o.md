# DISTRIBUTED MULTI-AGENT DEEP REINFORCEMENT LEARNING

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Centralized training with decentralized execution (CTDE) has been the dominant paradigm in multi-agent reinforcement learning (MARL), but its reliance on global state information during training introduces scalability, robustness, and generalization bottlenecks. Moreover, in practical scenarios such as adding/dropping teammates or facing environment dynamics that differ from the training, CTDE methods can be brittle and costly to retrain, whereas distributed approaches allow agents to adapt using only local information and peer-to-peer communication. We present a distributed MARL framework that removes the need for centralized critics or global information. Firstly, we develop a novel Distributed Graph Attention Network (D-GAT) that performs global state inference through multi-hop communication, where agents integrate neighbor features via input-dependent attention weights in a fully distributed manner. Leveraging D-GAT, we develop the distributed graph-attention MAPPO (DG-MAPPO) -- a distributed MARL framework where agents optimize local policies and value functions using local observations, multi-hop communication, and shared/averaged rewards. Empirical evaluation on the StarCraftII Multi-Agent Challenge (SMAC) and Multi-Agent Mujoco demonstrates that our method consistently outperforms strong CTDE baselines, achieving superior coordination across a wide range of cooperative tasks with both homogeneous and heterogeneous teams. Our distributed MARL framework provides a principled and scalable solution for robust collaboration, eliminating the need for centralized training or global observability. To the best of our knowledge, DG-MAPPO appears to be the first to fully eliminate reliance on privileged centralized information, enabling agents to learn and act solely through peer-to-peer communication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a fully distributed multi-agent reinforcement learning (MARL) framework, DG-MAPPO, which aims to eliminate the need for the centralized training components common in the CTDE paradigm. Here, a Distributed Graph Attention Network (D-GAT) performs multi-hop, attention-based message passing over a dynamic communication graph so each agent infers a global context from local exchanges. Agents use this inferred global state, along with their local observations, to train local policy and value networks. The authors evaluate their framework on the StarCraft Multi-Agent Challenge (SMAC) and claim that it consistently outperforms strong CTDE baselines.

### Strengths
1. Introducing a graph attention module (D-GAT) tailored for distributed MARL is a novel and appealing idea.
2. The presentation is clear. The method and its components are described in a straightforward manner.

### Weaknesses
1. Reported baseline outcomes do not align with prior literature. In particular, the near-zero HAPPO performance on several SMAC tasks conflicts with Heterogeneous-Agent Reinforcement Learning (Zhong et al., JMLR 2024) where HAPPO does not collapse in this way. This raises concerns about configuration, implementation, or evaluation protocol for baselines, and weakens the empirical claims.
2. Evaluation scope is narrow. All experiments are on SMAC; there is no validation on other standard domains. Most figures focus on the proposed method’s learning curves; side-by-side training curves against baselines are missing.
3. Practicality and overhead are not analyzed. D-GAT is stacked for multi-hop propagation, which can make per-step runtime and bandwidth scale poorly with team size. There is no comparison of wall-clock time, communication volume, parameter counts, or sample efficiency versus CTDE baselines. Without cost analysis, claims of scalability are hard to assess.
4. Communication graphs appear very dense. In many SMAC tasks, the average node degree is close to n−1 (effectively fully connected), which benefits attention aggregation but is unrealistic and costly. 
[1] Zhong, Y., Kuba, J. G., Feng, X., Hu, S., Ji, J., & Yang, Y. (2024). Heterogeneous-agent reinforcement learning. Journal of Machine Learning Research, 25(32), 1-67.

### Questions
1. What is the communication and compute overhead of D-GAT vs. CTDE methods?
2. How sensitive is performance to the number of hops K, average degree, and other parameters? Please add ablations.

### Soundness
1

### Presentation
2

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
This paper proposes a distributed deep MARL algorithm under the setting that global state information is not available to individual agents. GATs are employed for agents to perform global state inference through local communication. Simulation results on the SMAC benchmark demonstrate the effectiveness of the proposed algorithm.

### Strengths
(1) The requirement for centralized critics or global information is removed in this work.

(2) By leveraging GATs, each agent can make better decisions based on the information aggregated from its neighbors instead of only on the local observation.

(3) Simulations are conducted on standard MARL benchmarks, and solid CTDE-based baselines are compared.

### Weaknesses
(1) This work assumes the team reward can be observed by all agents, and the joint policy can be used in the local critic parameter update, which contradicts the distributed training setting considered in most existing distributed MARL works.

(2) The proposed algorithm is computationally inefficient due to its multi-hop communication.

(3) This work only considers fixed communication graphs.

(4) Both the loss function design and the notations are unclear.

See the Questions section for more details.

### Questions
(1) In (11), do you only consider the consensus of the feature vectors $\hat{h}^i$ obtained from one-hop communication?

(2) Please provide the specific form of the function $l^i$ in (9). What are the relationships among $\psi^i$, $\phi^i$, and $\theta^i$?

(3) It seems that incorporating the consensus update step in the GAT’s parameter update is useless. Note that there is no guarantee that $\hat{h}^i = \hat{h}^j$ when all agents share the same GAT parameters.

(4) It is very hard to understand the design of $\tilde{o}_t^i$ in (13). Note that it is not used in the policy update.

(5) The joint policy is required in the critic parameter update, which contradicts the distributed training setting.

(6) Please explain how to obtain the policy update loss function in this work based on the MARL policy gradient theorem (Th. 1) from a mathematical perspective.

(7) In your experiment, the comparison between the proposed algorithm and CTDE baselines is unfair, since the input of the policy contains more environmental information due to the GAT. The authors are recommended to include other baselines that also employ GNN-based policies.

(8) Some confusing sentences:

(Page 1) Third, CTDE methods often suffer from a train–test mismatch, … generalization.

(Page 7) Training the value function … which risks destabilizing training.

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
4

### Summary
In the paper, the authors consider a distributed MARL problem. In particular, a distributed training scheme is considered, where each local agent only use local information to train its policy. Graph attention network (GAT) is used to infer and gather global information from local communications.  GAT is further extended to MAPPO and the authors show strong performance of the proposed method against many CTDE baselines.

### Strengths
The paper is easy to follow. The derivations of the method and equations are clear and smooth. Extensive simulations have been done to benchmark the proposed method.

### Weaknesses
I found the method of this paper is quite straightforward and the contribution is limited. First, CTDE does not always mean that it does not scale well as the global information can be compressed. During evaluation, CTDE can work by using local information (or with neighbors). Nevertheless, the adoption of representing the relations via a graph is a promising direction to better gather information locally to infer nearly-global information. One biggest drawback of this paper is the literature review. Graph + MARL is not new in the field and it has been studied extensively in the past 7-8 years. The authors only mentioned the  Jiang et al.
(2018) proposed graph convolution paper in the intro; this is far from enough as there are a lot of following and parallel works in this direction. Overall, the contribution in terms of novelty of this paper is limited.

### Questions
Are you aware of  any other Graph + MARL works? If yes, please include and discuss them,

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a fully distributed alternative to the Centralized Training with Decentralized Execution (CTDE) paradigm in Multi-Agent Reinforcement Learning (MARL). The authors identify key limitations of CTDE, such as its reliance on global state information during training, which can lead to scalability, robustness, and generalization bottlenecks. To address this, they introduce two main contributions: a Distributed Graph Attention Network (D-GAT) for global state inference via multi-hop peer-to-peer communication, and DG-MAPPO, a distributed MARL framework built upon D-GAT that learns policies and value functions using only local observations and communicated information. The method is evaluated on the challenging StarCraftII Multi-Agent Challenge (SMAC) benchmark, where it is shown to achieve competitive or superior performance compared to strong CTDE baselines like MAPPO, HAPPO, and MAT-Dec, particularly in complex, heterogeneous scenarios.

### Strengths
1. On SMAC benchmark, comparison against multiple strong baselines, and reporting across a wide range of scenarios (from easy to super-hard) provides compelling evidence for the method's efficacy.
2. The integration of Graph Attention Network into a MAPPO framework (DG-MAPPO) is clean and well-explained.

### Weaknesses
1. While the overall framework is well-evaluated, a more detailed ablation study would be beneficial. How critical is the D-SGD averaging versus the consensus loss? How does performance degrade if we use a simpler aggregation method (e.g., mean) instead of attention? Understanding the contribution of each component would provide deeper insights.
2. The claim of scalability is supported empirically, but a more formal analysis of the communication overhead (bytes per step, convergence time w.r.t. number of agents) would be valuable, especially since multi-hop communication is used at every timestep.
3. Several related multi-agent RL baselines are missing. Like [1][2][3] need to clarify contribution comparison and consider including them in experimental evaluation.

[1] Jing Xu, Fangwei Zhong, Yizhou Wang. Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks. NeurIPS 2020.
[2] Jiechuan Jiang and Zongqing Lu. Learning Attentional Communication for Multi-Agent Cooperation. NeurIPS 2018.
[3] Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson. Counterfactual Multi-Agent Policy Gradients. AAAI 2018.

### Questions
1. Given the significant recent progress in using large foundation models (LLMs) as the core of multi-agent systems, how do you see the role of your gradient-based, distributed MARL framework? Is it a competing approach, or could it be integrated with LLM-based agents (e.g., for learning low-level coordination policies that an LLM planner oversees)?
2. SMAC is a classic but mature benchmark. What novel scientific insight does achieving a new State-of-the-Art (SOTA) on it provide, beyond incremental performance improvement? Does the success of DG-MAPPO reveal a fundamental limitation of CTDE that was previously underestimated?
3. The experiments are comprehensive but limited to SMAC. Can you provide evidence or a discussion on how your method would perform in environments with more complex, open-ended communication demands, or in partially observable environments where the "global state" is inherently non-reconstructible from local views?
4. In the one scenario (25m) where DG-MAPPO underperforms, what is your hypothesis for the reason? Is it due to the increased difficulty of long-range credit assignment in a fully distributed setting, or is it a limitation of the D-GAT's capacity to propagate information effectively in large teams?

### Soundness
2

### Presentation
2

### Contribution
2
