# Adaptive Reinforcement Learning for Unobservable Random Delays

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
In standard Reinforcement Learning (RL) settings, the interaction between the agent and the environment is typically modeled as a Markov Decision Process (MDP), which assumes that the agent observes the system state instantaneously, selects an action without delay, and executes it immediately. In real-world dynamic environments, such as cyber-physical systems, this assumption often breaks down due to delays in the interaction between the agent and the system. These delays can vary stochastically over time and are typically _unobservable_ when deciding on an action. Existing methods deal with this uncertainty conservatively by assuming a known fixed upper bound on the delay, even if the delay is often much lower. In this work, we introduce the _interaction layer_, a general framework that enables agents to adaptively handle unobservable and time-varying delays. Specifically, the agent generates a matrix of possible future actions, anticipating a horizon of potential delays, to handle both unpredictable delays and lost action packets sent over networks. Building on this framework, we develop a model-based algorithm, _Actor-Critic with Delay Adaptation (ACDA)_, which dynamically adjusts to delay patterns. Our method significantly outperforms state-of-the-art approaches across a wide range of locomotion benchmark environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of reinforcement learning under random, unobservable interaction delays in cyber-physical systems. The authors propose the "interaction layer", a framework that enables agents to generate a matrix of future actions to handle time-varying delays and potential packet loss. Building on this framework, they develop ACDA (Actor-Critic with Delay Adaptation), a model-based RL algorithm that uses distribution embeddings and a heuristic for action selection under delay uncertainty. The method is evaluated on MuJoCo locomotion tasks and demonstrates superior performance compared to state-of-the-art delayed RL algorithms.

### Strengths
1. The paper tackles a practically important problem of unobservable random delays in real-world RL applications, moving beyond the restrictive constant-delay assumption used in prior work.

2. The interaction layer framework is well-motivated and provides a principled way to handle varying delays by generating action matrices that anticipate different delay scenarios.

3. The experimental results show consistent and substantial performance improvements over strong baselines (BPQL, VDPO, DCAC) across multiple environments and delay processes.

4. The paper provides comprehensive experimental details including rigorous ablation studies, hyperparameters, and justification for design choices such as action noise.

### Weaknesses
1. Limited Experimental Scope. The evaluation is restricted exclusively to MuJoCo locomotion tasks (Ant, Humanoid, HalfCheetah, Hopper, Walker2d), which represent a narrow class of continuous control problems. The generalizability to other important domains such as manipulation, navigation, discrete action spaces, or real-world robotic systems remains unclear.

2. Computational Complexity Analysis. While Appendix F.2 provides execution time measurements, the paper lacks a thorough complexity analysis comparing ACDA's computational requirements (both time and space) against baselines.

3. Heuristic Assumptions and Limitations. The heuristic in Section 4 assumes delays are temporally correlated (constant over short windows), which may not hold in many real-world scenarios such as wireless networks with bursty interference or cloud computing with variable load.

### Questions
1. Generalization beyond locomotion: Can you provide results or discussion on how ACDA would perform in other RL domains such as manipulation tasks, partially observable environments beyond delays, or discrete action spaces? What modifications would be necessary?

2. Action buffer horizon: How sensitive is performance to the choice of horizon h and prediction length L? Is there a principled way to set these parameters, or do they require manual tuning per environment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work aims to solve the problem of unobservable random delays in reinforcement learning. This paper proposes a novel "interaction layer" framework, where the agent no longer generates a single action but instead produces an action matrix to dynamically adapt to different delay situations. The proposed algorithm, ACDA, utilizes this framework and has been shown in experiments to achieve significantly superior performance.

### Strengths
1. This paper tackles a practical problem in the field of RL with delays, that the delay is random and cannot be observed or predicted. This stands in sharp contrast to the unrealistic priori assumptions about delay found in existing works in this field. The reviewer affirms the contribution of this work.
2. The paper's experimental design is comprehensive. The authors introduce action noise to induce stochasticity into the environments. Furthermore, the results in Appendix E.2 and E.3 serve to reinforce the paper's methodological design and central claims.

### Weaknesses
1. The paper's most significant weakness lies in its comparative experimental design, as both the choice of baselines and their setup are unreasonable. The authors selected BPQL and VDPO, which are methods primarily designed for constant delay. However, in random delay environments, providing the worst-case delay as a priori knowledge to these algorithms is, in the reviewer's opinion, an unreasonable setup. The reviewer believes a more reasonable design would be to provide the mean of the random delay distribution as the priori delay knowledge for this class of methods; that is, the agent would be making decisions based on the average-case delay.
2. Given the paper's focus on the random delay problem, the authors have omitted a comparison with recent state-of-the-art methods that also specifically address random delays, such as DFBT [1] and State Augmentation-MLP [2]. Benchmarking against this relevant class of methods would be necessary to further substantiate the claimed effectiveness of the proposed method.
3. The paper's presentation is not good enough. The notation introduced is somewhat complex, and the process described in the text for Figures 2 and 3 is difficult to follow. For example, in Figure 2, the most recent observation received by the agent is labeled o_{t-2}, yet the description in lines 193-194 refers to o_t as the agent's observation, which is confusing. Furthermore, when Figure 3 is first introduced (lines 210-211), the definition for the counter c_t, which is essential to the figure caption, has not yet been provided, making it difficult to understand.

### Questions
1. In lines 418-419, the authors state, "we expect ACDA not to perform as well with the M/M/1 queue delays that fluctuate more." However, in Table 1, ACDA's performance under the MM1 Delay process is exceptionally strong, in most environments even exceeding its performance on the GE1,23 and GE4,32 processes. This result appears to contradict the authors' statement. Can the authors explain this phenomenon?
2. The parameter L does not appear to be explicitly defined in the main text. The reviewer found in Appendix F.2 that L represents the "prediction length," but how this parameter is determined is a significant concern for the reviewer. If L is set to the maximum value of the random delay, this implies an assumption that the maximum random delay is known, which would weaken the advantage of this work in handling unobservable random delays. If L is set as a fixed empirical value, the reviewer hopes the authors can provide the corresponding justification.
3. A very interesting result in Table 1 is why DCAC, a method explicitly designed for random delays, performs so poorly in this random delay environment—far worse, in fact, than the constant-delay methods BPQL and VDPO. Is this poor performance attributable to the stochastic transitions induced by the authors' use of action noise, which perhaps DCAC is not well-suited to handle?

Reference
[1] Directly Forecasting Belief for Reinforcement Learning with Delays. ICML 2025
[2] Addressing Signal Delay in Deep Reinforcement Learning. ICLR 2024

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Actor-Critic with Delay Adaptation (ACDA), a model-based reinforcement learning algorithm designed to handle random, unobservable, and time-varying delays between the agent and the environment. The authors introduce an interaction layer that models delayed interactions as a POMDP, allowing the agent to generate a matrix of future actions and adapt dynamically when delays vary. ACDA employs a heuristic to estimate previously applied actions and a model-based distribution embedding to represent future state distributions.
Extensive experiments on multiple MuJoCo locomotion benchmarks with various stochastic delay processes show that ACDA consistently outperforms state-of-the-art baselines such as BPQL, VDPO, and DCAC. The method demonstrates strong robustness to delay variability.

### Strengths
1. The paper tackles an important and practical challenge in reinforcement learning, handling random, unobservable, and time-varying delays, which is highly relevant to real-world systems such as networked and robotic control.

2. The proposed interaction layer is conceptually elegant, bridging MDP and POMDP formulations for delayed environments and offering a general framework for asynchronous agent-environment interactions.

3. The ACDA algorithm integrates a heuristic delay adaptation mechanism with a model-based distributional embedding, enabling the agent to remain robust to stochastic delays without explicit delay estimation.

4. The experimental evaluation is extensive and well-executed, covering multiple MuJoCo tasks and diverse delay processes, with thorough ablations.

Overall, the paper is clearly written and easy to follow, with strong motivation, a clear problem setup, and a well-structured methodology. The experimental evaluation is extensive and thorough, providing solid empirical support for the proposed approach.

### Weaknesses
1. The method assumes that delays remain approximately constant within short time windows. This assumption may not hold under highly dynamic or non-stationary delay patterns, potentially reducing the method's robustness.

2. The paper does not provide formal convergence analysis, stability proofs, or error bounds. As a result, the robustness of ACDA is supported primarily by empirical evidence rather than theoretical justification.

### Questions
1. Is there a potential direction toward providing formal guarantee, such as performance bounds or stability proofs, under certain classes of stochastic delays?

2. How might the interaction-layer framework generalize to discrete-action, multi-agent, or large-scale RL settings?

3. Could future work explore learning the action-buffer management policy itself (rather than fixing its size or update rule), allowing the system to dynamically adjust to varying network conditions?

I find this paper to be creative and well-executed, with a clear motivation and a solid methodological contribution. While the lack of theoretical guarantees slightly limits the depth of the work, the paper presents a meaningful step forward.
Overall, I hold a positive opinion and would be happy to see the paper accepted.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies unobservable random delays in MDPs, where action execution may be randomly delayed, and the agent observes the delays only in hindsight. The authors introduce an interaction layer to handle these delays. Specifically, at each timestep, the agent generates an action matrix containing a sequence of actions to be executed until the next action packet, for every possible delay length. They then build a model to predict the future state from the current state and the future actions to be executed, and use it with a version of SAC combined with BPQL for training. Experiments conducted on five MuJoCo environments under three random delay processes demonstrate superiority over the baselines.

### Strengths
- The problem setting is important, as delays are often stochastic and action delays are not immediately observable.
- The paper is well-motivated and provides a good overview of existing work.
- The proposed interaction layer, which represents actions as a matrix, is novel.

### Weaknesses
- A key weakness of the approach is that representing the action packet as a matrix can be computationally and communicationally expensive. The work is motivated by improving efficiency under delays; however, computing the action matrix itself is significantly more time-consuming than generating a single action at each timestep. In particular, while the agent executes only $T$ actions, the framework requires computing and transmitting $T \times L^2$ actions over the network. As a result, many actions are never used, yet they add computational overhead, which could further increase delays in practice.


- The presentation of the method needs improvement. I had to read some parts multiple times and cross-reference earlier parts to understand the details. Some sections include irrelevant discussions, while key aspects are missing. For example, the POMDP formulation is disconnected from the rest of the paper, and the claim that a single delay $d_t$ is equivalent to other types of delays is neither proved nor sufficiently discussed.

- While the overall approach is novel, the two main components: generating sequences of actions and employing a model-based distributional agent, have been extensively explored in prior work. The idea of generating a sequence of actions is closely related to *action chunking*, an active research topic in the RL community. Similarly, model-based distribution agent idea has appeared in previous works on delayed RL[1, 2]. The authors could have focused more on explaining the unique contributions of their method rather than discussing previously established ideas in detail.

- I also have concerns regarding the experimental evaluation. First, at least one baseline result does not match the values reported in prior work (see under *Questions*). Second, the choice of delay distributions appears to favor the proposed method. The constant-delay baselines, adapted to handle random delays, assume the worst-case delay at every step. Although the authors partially address this concern in the appendix by modifying the assumed delay upper bound (showing improved results for baselines) this adjustment still favors their method. A more fair evaluation would include the uniform delay distribution, where baseline methods operate under the correct assumption.

[1] Wang, Wei, et al. "Addressing signal delay in deep reinforcement learning."

[2] Liotet, Pierre, et al. "Learning a belief representation for delayed reinforcement learning."

### Questions
- I assume L or h is the upper bound of the delay. What is the other variable?
- What happens if an outdated action packet arrives at the action buffer?
- It is not clear to me how $d_t$ replaces both observation and action delays. Could you elaborate? In particular, under both types of delays, there are timesteps where the agent receives no new observations. How is this situation equivalent in your formulation?

- The reported DCAC performance on HalfCheetah-v4 is 35.60 for the MM1 delay process. Based on the MM1 process shown, all delays are less than 21, with a mean around 5. However, previously reported DCAC results [3] on constant delays of 5 and 25 are approximately 2000 and 600, respectively, on the same environment without noise. From my experience, introducing 5% noise is unlikely to cause such a large performance gap. What is the source of this discrepancy?

[3] Wu, Qingyuan, et al. "Variational delayed policy optimization."

### Soundness
3

### Presentation
2

### Contribution
2
