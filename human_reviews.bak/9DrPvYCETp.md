# Shared Memory for Multi-agent Lifelong Pathfinding

- Decision: Reject
- Scores: 5, 8, 3

## Abstract
Multi-agent reinforcement learning (MARL) demonstrates significant progress in solving cooperative and competitive multi-agent problems in various environments. One of the main challenges in MARL is the need to explicitly predict other agents' behavior to achieve cooperation. As a solution to this problem, we propose the Shared Recurrent Memory Transformer (SRMT), which extends memory transformers to multi-agent settings by pooling and globally broadcasting individual working memories, enabling agents to implicitly exchange information and coordinate actions. We evaluate SRMT on the Partially Observable Multi-Agent Path Finding problem, both in a toy bottleneck navigation task requiring agents to pass through a narrow corridor and on a set of mazes from the POGEMA benchmark. In the bottleneck task, SRMT consistently outperforms a range of reinforcement learning baselines, especially under sparse rewards, and generalizes effectively to longer corridors than those seen during training. On POGEMA maps,  including Mazes, Random, and Warehouses, SRMT is competitive with a variety of recent MARL, hybrid, and planning-based algorithms. These results suggest that incorporating shared memory into transformer-based architectures can enhance coordination in decentralized multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a global shared recurrent memory transformer (SRMT) mechanism for multiagent reinforcement learning to address the multiagent pathing finding problem. Specifically, SRMT uses self-attention to aggregate agent memory and observation history while utilizing cross-attention to aggregate the shared memory from other agents to help coordination. Results on a toy bottleneck navigation task and a set of maze environments from the POGEMA benchmark show that SRMT outperforms various baselines.

### Strengths
1.	The motivation for using a global shared memory to help coordination and the idea of using the transformer to implement it are clear.
2.	The background is clearly explained and the related works are well discussed.

### Weaknesses
1.	It seems that a lot baselines are missing. For example, in the Bottleneck Task, only some basic memory mechanisms from single-agent RL are compared while more advanced memory mechanisms such as relational memory [1] and AMRL [2] from the single-agent RL domain are not compared.
2.	At the same time, although some works about MARL memory such as RATE and ATM are discussed in Section 2.2, they are not compared in the experiments.
3.	The ablation study to validate each component of the proposed SRMT is not given.
4.	There are some typos. In Line 36, “MAPF” is not defined.

References

[1] Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Théophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, and Timothy Lillicrap. Relational Recurrent Neural Networks. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.

[2] Jacob Beck, Kamil Ciosek, Sam Devlin, Sebastian Tschiatschek, Cheng Zhang, and Katja Hofmann. Amrl: Aggregated memory for reinforcement learning. In International Conference on Learning Representations, 2020.

### Questions
1.	Could the authors give the number of network parameters of each method? As SRMT uses transformers and ResNet, it may obtain advantages by more network parameters.
2.	Could SRMT scale well with the number of agents? If the number of agents increases, will the training time become much longer?
3.	Why does MAMBA with discrete communication protocol outperform SRMT in some scenarios? Does it mean that the global shared memory is not always the best choice? If yes, how could we choose the right method for the multiagent path-finding problem?

### Soundness
2

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
4

### Summary
This paper introduces the Shared Recurrent Memory Transformer (SRMT), a novel model in multi-agent reinforcement learning designed for multi-agent lifelong pathfinding tasks. SRMT extends memory transformers to decentralized multi-agent environments by pooling individual agent memories into a shared memory space, allowing agents to indirectly share information and coordinate. The model is tested in various pathfinding tasks, including bottleneck navigation and complex environments from the POGEMA benchmark. SRMT demonstrates superior performance in coordination and generalization, particularly in high-density and partially observable environments.

### Strengths
1. The SRMT model is an adaptation of memory transformers to multi-agent settings, facilitating indirect communication among agents through a shared memory. This approach addresses a significant challenge in decentralized coordination by leveraging shared recurrent memory, which is unique compared to conventional communication strategies.
2. The paper provides a rigorous evaluation of SRMT on multiple benchmark tasks, including POGEMA and bottleneck navigation. The use of diverse reward settings (e.g., sparse, directional) further strengthens the experimental framework, revealing SRMT’s adaptability in various coordination scenarios.
3. The architecture and methods are clearly explained, supported by diagrams and flowcharts that help clarify SRMT’s working mechanism. The comparisons with baselines and the explanation of the multi-agent Markov decision process formulation are presented in a straightforward and understandable manner.
4. SRMT’s ability to handle decentralized pathfinding without explicit communication protocols has considerable implications for real-world applications, particularly in settings where communication might be unreliable or costly. Its effectiveness across different maps and scenarios demonstrates potential for scalability in complex, large-scale environments.

### Weaknesses
1. While SRMT performs well on small to medium-sized environments, its scalability to very large maps or highly dense environments remains uncertain. The evaluation could be extended to more challenging settings, particularly with greater agent populations or larger obstacles, to fully assess SRMT’s scalability.
2. While SRMT is designed for decentralized systems, it would be beneficial to see comparisons with centralized approaches on key metrics to understand the trade-offs better, particularly in environments that demand high coordination.
3. While the paper claims that shared memory improves coordination, additional analysis on how shared memory influences individual agent behavior would provide a deeper understanding. An ablation study removing the shared memory aspect could further validate its impact on SRMT’s performance.
4. The model's performance varied across different reward structures, and while this is discussed, a more detailed exploration of how reward shaping influences learning would strengthen the analysis. This would help in tailoring SRMT to tasks where only sparse rewards are available.

Missing references (MARL with local information). I believe these are quite recent papers and work in a similar setting as mentioned in the related works section.

[1]: Hu, Y., Fu, J., & Wen, G. (2023). Graph soft actor–critic reinforcement learning for large-scale distributed multirobot coordination. *IEEE transactions on neural networks and learning systems*.

[2]: Nayak, S., Choi, K., Ding, W., Dolan, S., Gopalakrishnan, K., & Balakrishnan, H. (2023, July). Scalable multi-agent reinforcement learning through intelligent information aggregation. In *International Conference on Machine Learning* (pp. 25817-25833). PMLR.

### Questions
1. How well does SRMT scale with an increased number of agents or more complex map structures? Additional experiments in larger environments could help evaluate its robustness in real-world applications.
2. Would SRMT benefit from combining shared memory with limited explicit communication for certain high-density environments?
3. How does shared memory impact the decision-making process for individual agents? Further analysis on memory usage patterns and shared memory dynamics could provide insights into SRMT’s internal coordination mechanisms.
4. Does SRMT allow for integration with hierarchical pathfinding methods, such as combining local and global pathfinding strategies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work considers the application of a shared memory mechanism to the MAPF setting.

### Strengths
- The writing is generally clear and polished. 
- The approach is well-grounded in prior literature, and the algorithmic details are well-explained.  
- Figure 1 is a useful complement to the written algorithmic details, and makes it easy to understand the method at a glance. 
- Figure 10 analysis is nice.

### Weaknesses
* It is hard to get a relative sense of the competitiveness of this approach. The baselines did not feel particularly well-motivated, and MARL communication works, which I'd argue share a similar goal, were not used as baselines (e.g. \[1\])
* More generally, I am left not knowing exactly what I should take away from the results—Figure 5 seems to show that SRMT and variants achieve modest results compared to baselines (and the baselines used are not motivated or described in sufficient detail).
* \[2\] I consider this a necessary work to acknowledge, given it is one of the first works discussing the use of attention in MARL
* Nitpicks: 
	* I cannot interpret the error bars in Figure 4—it is too muddled.
	* Despite the writing overall being clear, the language could be tightened somewhat; e.g. L043: "has to reach its goal" is quite colloquial; also contraction in L497. I recommend combing through the paper and essentially asking each word/phrase to justify itself—and to be as specific as possible, avoiding colloquialisms. 

\[1\] Jakob Foerster, Ioannis Alexandros Assael, Nando de Freitas, and Shimon Whiteson. Learning to communicate with deep multi-agent reinforcement learning. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett (eds.), *Advances in Neural Information Processing Systems, volume 29*. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper_ files/paper/2016/file/c7635bfd99248a2cdef8249ef7bfbef4-Paper.pdf.

\[2\] Iqbal, S. &amp; Sha, F.. (2019). Actor-Attention-Critic for Multi-Agent Reinforcement Learning. <i>Proceedings of the 36th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 97:2961-2970 Available from https://proceedings.mlr.press/v97/iqbal19a.html.

### Questions
- Following up on a weakness above: Why was this approach not evaluated against any MARL baselines that implement communication channels between agents?

### Soundness
2

### Presentation
3

### Contribution
2
