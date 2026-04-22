# Learning to Interact in World Latent for Team Coordination

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
This work presents a novel representation learning framework, interactive world latent (IWoL), to facilitate *team coordination* in multi-agent reinforcement learning (MARL). Building effective representation for team coordination is a challenging problem, due to the intricate dynamics emerging from multi-agent interaction and incomplete information induced by local observations. Our key insight is to construct a learnable representation space that jointly captures inter-agent relations and task-specific world information by directly modeling communication protocols. This representation, we maintain fully decentralized execution with implicit coordination, all while avoiding the inherent drawbacks of explicit message passing, *e.g.*, slower decision-making, vulnerability to malicious attackers, and sensitivity to bandwidth constraints. In practice, our representation can be used not only as an implicit latent for each agent, but also as an explicit message for communication. Across four challenging MARL benchmarks, we evaluate both variants and show that IWoL provides a simple yet powerful key for team coordination. Moreover, we demonstrate that our representation can be combined with existing MARL algorithms to further enhance their performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present IWoL, a unified representation learning and communication framework for cooperative multi-agent RL (MARL). The proposed method jointly learns a latent space that captures inter-agent relations and helps in addressing partial observability in MARL. The method features two variants, one with implicit communication and the other with explicit communication at execution/test time. The authors perform experiments over different scenarios/benchmarks, comparing the performance of the proposed method against other baselines.

### Strengths
The paper features a good discussion of related work in Sec. 2. The experimental protocol seems solid, with the authors performing multiple runs and reporting confidence intervals. The experimental results seem reproducible. The experiments are also extensive, considering several different environments. The experimental results seem to be in favour of the proposed method. The authors also provide an ablation study.

### Weaknesses
I think the clarity of the paper could be slightly improved in some parts, particularly in Sec. 4.2.

The proposed method seems to feature some limitations (deterministic transitions and access to the underlying state at training time).

The contributions of the paper appear somewhat incremental, as it builds upon and combines previously proposed techniques in the MARL field. Could the authors further clarify if there are key contributions (in terms of the proposed method) besides combining previously proposed techniques? Nevertheless, I believe this should not be a reason to reject the paper alone, as the paper has its own merits.

### Questions
- "(...) we assume a deterministic MDP throughout this work" - this means the state transition dynamics are deterministic? This is quite a restrictive assumption.
- "(...) we introduce a learnable protocol" - the learnable protocol function P needs to be trained/applied in a centralized fashion, right? I understand that for the implicit world model this is not a problem, but for the explicit world model it may be.
- line 205, "Next, the communication protocol block" -  this corresponds to the red block in Fig. 2, right? If so, please clarify it in the text. I got a bit lost while reading Sec. 4.2.
- line 214, "Lastly, the previously produced vectors, e.g., communication message and intermediate embedding, are used as input to policy and value function networks." - is this correct? Looking at Fig. 2 it seems that this is indeed the input to the value function, but the input to the policy is z_i^t. Also, z_i^t is only explained around line 244. I suggest the authors explain it earlier in the text to make the discussion clearer.
- line 247, "The world decoder reconstructs the agent’s privileged state s^t_i" - so, this assumes access to the true state of the Dec-POMDP during training? This is quite a restrictive assumption, right? What about reconstructing joint observations?
- Why would Im-IWoL outperform Ex-IWoL - E.g., looking at Fig. 7 this seems to be the case. Why? I would expect Ex-IWoL to be in advantage in comparison to Em-IWoL since it features explicit communication between the agents at execution time, right?

Minor comments/typos:
- sentence in line 17 of the abstract starting with "This representation" needs to be revised.
- sentence in line 61 "if desired, the same backbone can expose as explicit messages at test time." is a bit hard to understand.
- line 197: "by a self-attention, (...)" - "layer" missing in the sentence?

### Soundness
4

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
The paper develops a technique to learn inter-agent communication for cooperative MARL tasks with a Transformer-based communication architecture and an encoding-decoding loss. 
The technique requires inter-agent communication during training (centralized training) and is flexible to turn on (“explicit”) or off (“implicit ”) the communication during execution.

### Strengths
Overall, the paper is well-written with clarity.
The empirical study is systematic and presented in an organized way.

### Weaknesses
1. Some key structural assumptions on top of Dec-POMDPs are not explicitly state:
    - Section 3 introduces the standard Dec-POMDP formulation but additional structures are needed in Section 4, such as agent position and minimum distance for communication (line 236). 
    - Even, agent "privileged state” (line 247) is critical for training but not defined.

2. The novelty of the proposed method, IWoL, is fairly limited. 
    - Prior works did the learning of communication topology (e.g.,MAGIC,I2C) and used Transformer-based communication modules (CommFormer, TarMAC).
    - The DecoderW module helps IWoL a lot according to Figure 8. However, such a module seems to be flexible to be a plug-and-play module for any existing prior method. For example, CommFormer/MAGIC in Figure 8 is much better with DecoderW than without. 

3. It is unclear what the paper wants to say about implicit vs explicit communication: 
    - Section 4.1 argues that explicit communication is worse than implicit, but the failure example of explicit communication is caused by bandwidth limit and message-corruption, which should equally affect implicit communication (during training).
    - In principle, explicit should subsume implicit because explicit agents can choose to send null messages. In Table 1,  Im-IWoL outperforms Ex-IWoL, and the paper does not explain this performance difference.

### Questions
All my concerns/questions are in the Weaknesses section.

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
This paper proposes a novel representation learning framework called Interactive World Latent (IWoL) to enhance team coordination in multi-agent reinforcement learning (MARL), which captures both inter-agent relations and task-specific world information while supporting both implicit (message-free) and explicit (message-rich) coordination modes without additional modules. Across four MARL benchmarks, IWoL variants outperform existing MARL baselines, showing strong robustness under incomplete observations and good scalability with increasing agent numbers.

### Strengths
1. The paper proposes the IWoL framework that jointly captures inter-agent relations and task-specific world information, supporting both implicit (message-free) and explicit (message-rich) coordination modes, avoiding drawbacks of traditional explicit communication
2. IWoL outperforms existing MARL baselines across four benchmarks with higher success rates or rewards and also show good scalability.

### Weaknesses
1. The implicit mode (Im-IWoL) relies on a well-trained latent representation, and its performance may degrade if the training data fails to cover diverse scenarios.
2. Compared to the simple MARL baseline (MAPPO), IWoL still has slightly higher training overhead, especially in complex tasks like bimanual dexterous manipulation.
3. The selected baselines are not comprehensive compared to what the authors listed in the related work.

### Questions
1. Can the authors show the performance degradation of Ex-IWoL similar to Figure 1?
2. Can you explain why implicit world latent is needed if the agents do not exchange messages during execution, and how can it keep generalizable?

### Soundness
3

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
3

### Summary
This paper proposes Interactive World Latent (IWoL), a representation learning framework for multi-agent reinforcement learning (MARL) that aims to facilitate team coordination. The key idea is to learn a latent representation that jointly captures inter-agent relations and task-specific world information through a communication protocol based on graph-attention mechanisms. The framework supports both implicit (no messages at test time) and explicit (messages fed to policy) modes. The authors evaluate their approach across four MARL benchmarks: MetaDrive, Robotarium, Bi-DexHands, and multi-agent quadruped environments, demonstrating improved performance over several baselines.

### Strengths
1. **Well-motivated problem**: The paper addresses an important challenge in MARL - building effective representations for team coordination under partial observability. The motivating example (Figure 1) effectively demonstrates the fragility of explicit communication under realistic constraints.

2. **Clear presentation**: The paper is generally well-written with clear diagrams that effectively communicate the architectural design.

3. **Comprehensive experimental evaluation**: The experiments cover 10 tasks across 4 different environments, including autonomous driving, swarm robotics, dexterous manipulation, and quadruped coordination.

4. **Strong empirical results**: IWoL variants consistently achieve top or second-best performance across all tasks, with particularly notable improvements on challenging tasks where baselines struggle (e.g., Go1Sheep, Two Catch Underarm).

5. **Thorough ablation studies**: The paper includes ablations examining the role of world representation, incomplete observations, scalability, and latent dimensionality.

### Weaknesses
1. **The title is a bit misleading**: 
   - The title "Learning to Interact in World Latent" sounds like model-based RL, which is confusing
   - The term "interactive world latent" is also confusing - what makes it "interactive" vs just "world latent"?

2. **Some design choices are unexplained**: Several important architectural decisions lack justification:
   - Why use self-attention in the observation encoder?
   - Why use additive-attention and GumbelSoftmax in the communication protocol?
   - Why use decentralized value functions for IWoL when other CTDE methods use centralized critics? This seems to give up an advantage of the CTDE framework
   - How is the interactive world encoder (for Im-IWoL) designed?

3. **The communication protocol is unclear**: The communication mechanism needs clarification:
   - Does the system execute L rounds of communication before every action? Line 209 mentions "This mask is a relationship graph that guides a Transformer block that performs L rounds of attention-based message aggregation and refinement", I didn't fully get the procedure
   - What is the relationship between the adjacency mask (line 208 and 233) and the topology graph (line 235)? The adjacency mask is computed from agent features, while the topology graph appears to be based on physical distance (d_comm). Do they both refer to G^t?
   - Besides, does ∥x^t_i − x^t_j∥ ≤ d_comm mean agents can communicate when they are closer than d_comm, or when they are farther? Line 237 mentions that "d_comm is a minimum distance for communication."
   - How exactly does "Ex-IWoL" work? Since computing the relationship graph requires all agents' features, does this mean broadcasting all agent features and messages for L rounds? The notation P: M^0_1 × ... × M^0_I → M_1 × ... × M_I suggests a function of all agents' initial messages, but the architecture description suggests per-agent processing

### Questions
see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
