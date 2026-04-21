# Learning Team-Level Information Integration in Multi-Agent Communication

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
In human cooperation, both individual knowledge and group consensus play important roles in accomplishing tasks. However, existing multi-agent reinforcement learning (MARL) communication methods commonly focus on individual-level communication, which lacks the necessary global information for well-grounded decision-making. Meanwhile, individual-level communication is often infeasible when the communication bandwidth is limited. To tackle these problems, we propose a group-level information integration model called Double Channel Communication Network (DC2Net). DC2Net highlights the significance of independent group feature learning by separating individual and group feature learning into two independent channels. In this model, agents no longer communicate with each other in a peer-to-peer paradigm; instead, all interactions are carried out in the group channel. By combining individual and global features, decisions are made collaboratively. We conduct experiments on several multi-agent cooperative environments and the results show that the DC2Net not only outperforms state-of-the-art MARL communication models but also reduces the communication costs. Furthermore, the two independent channels enable adaptive balancing of individual and group feature learning based on task requirements.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies communications between agents in multi-Agent reinforcement learning. Specifically, it focuses on team-level communications instead of commonly used peer-to-peer communications. By doing so, the communication bandwidth is reduced. The proposed algorithm is tested on Traffic Junction Environment (TJ) with different levels of difficulties and Multi-agent Particle Environment (MPE).

### Strengths
- The paper is well-motivated, since reaching team consensus is an efficient strategy for human cooperation.
- The team communication channel in section 3.3 is permutation-invariant, which is reasonable for homogenous settings and can potentially reduce the search space.
- There are ablation studies over the design choices of different components of the proposed algorithm (DC2Net).

### Weaknesses
- DC2Net needs the access to a centralizer, which may not be available in practice.
- The paper claims "peer-to-peer communication may not provide sufficient IG for effective agent decision-making". However, peer-to-peer communication is not taken as a baseline in the experiments.
- The empirical study is focusing on limited scenarios. There are many other challenging tasks, such as Cooperative Navigation and Cooperative Push in MPE, SMAC[1] and GRF[2]. The benefit of DC2Net is more convincing by additional experiments on these tasks.

[1] The StarCraft Multi-Agent Challenge

[2] Google Research Football: A Novel Reinforcement Learning Environment

### Questions
- What's the difference between all-to-all communication, where each agent receive all other agents' information, and DC2Net? It seems that team consensus can be reached by all-to-all communication, and if so DC2Net is just a special case of peer-to-peer communication. 
- Does DC2Net still outperform the baselines in the heterogeneous setting where permutation invariance does not hold?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This authors in this paper propose to enhance MARL by allowing both individual and team information to be used.  Specifically, they propose the DC2Net as depicted in Figure 1.  They show that for Traffic Junction and Predator-Prey problems, the newly proposed model outperforms IQL, CommNet and DGN.

### Strengths
The authors propose a new Model that outperforms a number of model proposed in the past a few years for Traffic Junction and Predator-Prey problems.

### Weaknesses
I think the main weakness of the paper is the lack of significance.  The authors criticise that previous work only use team information as a kind of supplementary information, but I do not think that the new Model is not among them.  Worse, there is no direct comparison of performance/ communication cost between the new model and these previous works.

In fact, personally I do not feel comfortable to consider models such as that depicted in Figure 1 should be used for MARL.  This is because the whole model has to be trained centrally--which defeat the purpose of MARL in some scenarios.

Finally, the authors seem to have incorrectly used the citation style.

### Questions
The authors are expected to elaborate and justify the significances of the paper.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents Double Channel Communication Network (DC2Net), a multi-agent reinforcement learning communication model that addresses the limitations of individual-level communication by integrating team-level information. DC2Net separates individual and team feature learning into two independent channels, allowing for collaborative decision-making while reducing communication costs and enabling adaptive balancing between individual and team learning based on task requirements.

I have several major concerns, comments, and questions:

- The method proposes team-level communication channel in order to avoid individual inter-agent communications, supposedly due to low bandwidth constraints. Nevertheless, my biggest concern is that, this approach imposes centralization which is a major limiting factor in multi-agent systems and introduces scalability problems. This also contradicts the Dec-POMDP formulation adopted by the paper, since a centralized team-level communication channel is required.

- limited bandwidth can be resolved by learning efficient communication. For instance, see [1]. This related work, which to my surprise was not even mentioned in the paper, learns an efficient binarized communication policy which works well in low bandwidth scenarios AND it can be executed distributed. They also consider scenarios where some agents have extremely low visibility and team-level coordination is a requirement for accomplishing the multi-agent task. This is only one example of such approaches and you can find more of such works in the recent emergent multi-agent communication literature.

[1] Seraj, Esmaeil, et al. "Learning efficient diverse communication for cooperative heterogeneous teaming." Proceedings of the 21st international conference on autonomous agents and multiagent systems. 2022.

- The employed benchmarks are said to be “state-of-the-art”. However, this is not true. The employed baselines, IQL, CommNet, and DGN at best, are some standard baselines and not the SOTA, as they are relatively old and have been repeatedly outperformed by recent methods such as MAGIC, HetNet [1], MAPPO, TarMAC, and many others (see the list of benchmarks in mentioned papers). This weakens the evaluation process and the presented results. The method needs to be experimented and evaluated against more recent SOTA methods for the employed domains.

- The employed particle domains, i.e., Traffic Junction and Predator Pray are very simplistic. To draw firm conclusions on the usability and efficacy of the proposed method, more advanced domains are needed to be tested, especially in scenarios where the centralization process (as required in the proposed method) may pose a challenge.

- What are the limitations of this approach? For instance, the centralization required by the team-level communication channel must be discussed as a potential limitation.

At current states I vote weak rejection mostly due to the centralization issue, missing important related prior work, and weak benchmarks and domains, although the algorithm seems to be sound and working. I’d be happy to increase my score when authors satisfactorily addressed my comments and questios.

### Strengths
See above.

### Weaknesses
See above.

### Questions
See above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper seeks to address the problem of communication-based multi-agent reinforcement learning. The new idea is in the proposed Double Channel Communication Network (DC2Net) that utilizes two separate channels for individual and team-level learning to achieve the mixed learning process. Compared to most existing work, the proposed DC2Net eliminates the peer-to-peer communication setting, and leaves all the communication for the team-level channel to achieve centralized joint learning. Experimental results against multiple baseline algorithms are provided to demonstrate the effectiveness of the proposed framework.

### Strengths
+ The idea of separating individual learning and team-level communication is interesting.

+ The summary of representative literature in learning-enabled multi-agent RL with communication is adequate.

### Weaknesses
- While the idea of separating individual and team-level communication is quite interesting, the proposed DC2Net lacks key insights to properly justify the current design. For example, it seems DC2Net needs to assume centralized training and execution with strong assumptions on the underlying communication topology, i.e. every agent should be able to communicate to a centralized team-level channel at all times. Besides, the communication process simply aggregates all the agent’s team-level information with gradient truncation, thus making it more like a decentralized implementation of a centralized MARL without special treatment of communication component, which is less comparative to peer-to-peer communication where a centralized agent or node may not exist. 

- Although it’s claimed in the paper that the existing work with mixed learning of individual and team-level information does not perform well, there is no evidence to support that. For example, more insights could be provided to highlight what information should be considered as an individual learning-related component and what should be considered as a team-level component, and how they would interact with each other so that it will indeed outperform the existing mixed learning process.

- There is no direct comparison against the MARL framework that uses peer-to-peer communication or centralized computation. In the Ablation Study, it would be more helpful to provide results from DC2Net with mixed learning to justify the influence of the team-level communication component, rather than the provided DC2Net-T that boils down to a MARL without joint learning among agents.

### Questions
1. Can authors provide more insights on the particular examples where it’s better to use the DC2Net instead of the existing mixed communication learning process of MARL? What would be the individual feature and team-level features, and how that could be properly observed from the given simulation results?

2. What is the assumption on the communication topology in DC2Net? Does it require a centralized communication channel where the team-level features have to be aggregated for centralized training?

3. Is there any theoretical analysis in terms of policy convergence of the proposed DC2Net?

4. Could authors give a fair experiment comparison against peer-to-peer communication-based MARL?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
