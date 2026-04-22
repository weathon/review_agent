# Learning to Communicate Locally for Large-Scale Multi-Agent Pathfinding

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 8, 4

## Abstract
Multi-agent pathfinding (MAPF) is a widely used abstraction for multi-robot trajectory planning problems, where multiple homogeneous agents move simultaneously within a shared environment. Although solving MAPF optimally is NP-hard, scalable and efficient solvers are critical for real-world applications such as logistics and search-and-rescue. To this end, the research community has proposed various decentralized suboptimal MAPF solvers that leverage machine learning. Such methods frame MAPF (from a single agent perspective) as Dec-POMDP when at each time step an agent has to decide an action based on the local observation and typically solve the problem via reinforcement learning or imitation learning. We follow the same approach but additionally introduce a learnable communication module tailored to increase the level of cooperation between the agents via efficient feature sharing. We present the Local Communication for Multi-agent Pathfinding (LC-MAPF), the method that applies multi-round communication between the neighboring agents to exchange information and improve their coordination. Our experiments show that the introduced method outperforms the existing learning-based MAPF solvers, including IL and RL based approaches, across diverse metrics in a diverse range of (unseen) test scenarios. Remarkably, the introduced communication mechanism does not compromise the scalability LC-MAPF, which is a common bottleneck for communication-based MAPF solvers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a learnable communication module for Multi-Agent Path Finding (MAPF). The communication module leverages a transformer-based approach with 2 million parameters, used to amplify a Decentralized MAPF approach trained via Imitation Learning of a centralized method. Experiments were conducted in the POGEMA environment, and support the hypothesis.

### Strengths
- Detailed explanation of the methodology

- The experiments show that the proposed method is able to outperform baselines in large scales.

- Extensive ablation studies show that multi-round communication improves performance.

### Weaknesses
- Limited literature review in MAPF problems, especially in MARL and Communication-based approaches.

- While the experimental results show that the method is able to outperform the two baselines in 192 agent scale, it is not able to achieve similar performance in smaller scales.

- Limited number of Baselines (2).

### Questions
- It is unclear how the Communication model is pretrained (as GPT implies Generative Pretrained Transformer).

- How does the proposed approach compare against other learning based communication methods in MARL settings presented in [1] such as [2]?

[1] M. Bettini, A. Shankar, and A. Prorok, “Heterogeneous Multi-Robot Reinforcement Learning,” Jan. 17, 2023, _arXiv_: arXiv:2301.07137. doi: [10.48550/arXiv.2301.07137](https://doi.org/10.48550/arXiv.2301.07137).

[2] E. Seraj _et al._, “Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming,” in _Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems_, in AAMAS ’22. Richland, SC: International Foundation for Autonomous Agents and Multiagent Systems, 2022, pp. 1173–1182.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents LC-MAPF, a novel decentralized learning framework for multi-agent pathfinding (MAPF) that incorporates learnable local communication among agents. The proposed approach builds upon imitation learning, training agents on expert demonstrations from centralized solvers (LaCAM*), but introduces a multi-round message-passing mechanism that allows agents to exchange information with nearby peers without explicit communication supervision.

### Strengths
- The paper introduces a learnable communication framework for decentralized MAPF without direct supervision on communication messages—an elegant and conceptually novel design. The multi-round local communication implemented via a transformer resembles end-to-end differentiable message passing, bridging imitation learning and graph-based communication paradigms.
- The methodology is solid and well-grounded in both MAPF and multi-agent learning literature. The derivation of gradient flow for communication learning (Eq. 10–11) is mathematically clear and demonstrates how communication content is implicitly optimized for coordination. The ablation and scalability analyses provide convincing empirical support.

### Weaknesses
- The model uses a fixed number of communication rounds and neighbors, which may not optimally adapt to different map densities or task complexities. Adaptive mechanisms for determining when and with whom to communicate could further enhance efficiency.
- Although the paper compares to leading IL-based solvers, it would strengthen the contribution to also benchmark against communication-learned RL frameworks (e.g., CommNet, DIAL, or QMIX with communication extensions).
- Although LC-MAPF scales linearly, its per-step latency is higher for smaller agent populations. A discussion of optimization or deployment strategies (e.g., asynchronous communication, sparse attention) would improve the practical relevance.

### Questions
- Have the authors considered dynamic adjustment of the number of communication rounds based on environment complexity or agent density? This could improve efficiency without sacrificing coordination.
- Since the communication vectors are learned implicitly, have the authors attempted to visualize or interpret what information the messages encode (e.g., goals, conflict zones, local intents)?
- The method assumes homogeneous agents. How difficult would it be to extend LC-MAPF to heterogeneous agent systems with different capabilities or goals?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a new learning-based MAPF method (LC-MAPF) that focuses on improving coordination through agents' communication. The model employs a transformer-based backbone, similar to prior work such as MAPF-GPT, but extends it by concatenating message embeddings from neighboring agents along with local observations as model input. This is a simple yet effective idea, supported by the results on the POGEMA benchmark.

### Strengths
- The contribution is well-executed (albeit coming across as a bit incremental)

### Weaknesses
1. The authors describe LC-MAPF as a foundation model, but this term is reserved for architectures trained on large-scale, diverse datasets that generalize across multiple tasks. Since LC-MAPF is trained exclusively on MAPF data and evaluated only on MAPF environments, this characterization seems overstated.

2. Theoretical novelty is limited as the backbone architecture, data collection procedure, and training pipeline closely mirror MAPF-GPT. The main innovation lies in incorporating message passing, which, although useful, is a relatively incremental extension. 

3. While Figure-3 and the Appendix provide partial insights, it would be valuable to include a comprehensive comparison table for the Maze and Puzzle environments summarizing success rate, makespan, and collision rate. This would help readers clearly assess performance trade-offs against existing baselines.

### Questions
1) Could you clarify how this paper represents a leap over previous work?
2) Could you clarify whether for training, a foundation model was used or was it just MAPF data?

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
Then paper introduces LC-MAPF, which is a decentralized framework for MAPF that is designed to improve large-scale multi-agent coordination through local communication mechanism. In this framework, each agent generate messages and exchanges them with nearby agents. Such communication proceeds in sequential rounds. The messages are learned end-to-end. The authors collect 32M samples and train LC-MAPF with 4 rounds of communication. Evaluation on the POGEMA dataset shows state-of-the-art performance among other decentralized MAPF solvers.

### Strengths
1. It is a novelty to explore end-to-end learned communication in MAPF. The authors adapt a multi-round comminucation in MAPF settings.
2. Experimental evaluation is comprehensive and strong in comparison agains baselines.

### Weaknesses
1. There is a lack of interpretability. The messages are learned end-to-end, but there is no analysis of what information is actually encoded.
2. Data collection is skewed towards maze-like maps, but without good justification. What is the motivation for this? Would it cause overfitting to the maze maps? Could you break down the results by map types?

### Questions
1. What is $L$ on line 186?
2. You provide the code but not the model weights. Do you plan to open source them as well?
3. On line 318, is it 15 billion of 1.5 billion?
4. Could you break down the results by map types?

### Soundness
3

### Presentation
3

### Contribution
2
