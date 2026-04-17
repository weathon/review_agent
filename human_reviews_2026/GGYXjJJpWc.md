# Interactive Learning for LLM Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR (Interactive Learning for LLM Reasoning), a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5\% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies, providing useful insights toward future multi-agent design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ILR (Interactive Learning for LLM Reasoning), a co-learning framework designed to improve the independent reasoning ability of Large Language Models (LLMs) through training-time multi-agent interaction.
ILR combines Dynamic Interaction, which adaptively switches between cooperative and competitive modes using Item Response Theory (IRT), and Perception Calibration, which integrates peer reward distributions via Group Relative Policy Optimization (GRPO).
Experiments on several reasoning and coding benchmarks show consistent improvements (up to 5%) over single-agent baselines, suggesting that interaction during training can enhance autonomous inference performance.

### Strengths
1. The paper explores whether multi-agent interaction can improve the independent reasoning ability of LLMs — a novel and valuable direction beyond traditional system-level optimization.

2. The framework is evaluated across multiple reasoning and coding benchmarks, showing stable and consistent improvements (up to 5%) over strong single-agent baselines.

3. The paper includes insightful ablation and robustness studies, such as comparing cooperative vs. competitive modes and analyzing the effect of dynamic interaction on stronger LLMs.

### Weaknesses
1. The estimation of the reasoning ability parameter $\gamma$ is unclear, and the proportion of samples assigned to each strategy (cooperation vs. competition) after IRT-based classification is not reported.
If the estimation of  $\gamma$ is performed only once offline, its reliability may be questionable, as model reasoning ability can evolve throughout training.

2.  While the paper emphasizes adaptive switching between cooperative and competitive strategies, in real human collaboration these two modes often coexist within the same reasoning process. Given that multiple samples are already drawn per question, it would be natural to mix both interaction types within a single batch and compute a unified advantage. Without theoretical or empirical evidence that adaptive switching is superior to parallel coexistence, the contribution of this design choice appears less convincing.

3.  The training procedure involving GRPO and multi-agent interaction is insufficiently detailed. It is unclear whether the interaction phase replaces or complements GRPO rollouts, and how the reward is assigned to individual agents’ actions. If the reward covers the full Idea³ process, it is uncertain how the model achieves autonomous inference afterward. Moreover, the alignment between training and inference prompts is not discussed, raising potential concerns about off-policy distributional shift. Including end-to-end pseudocode for the rollout and GRPO update would significantly improve clarity.

4. Insufficient Motivation and Notation Ambiguity in Equation (5):

   4.1  Should the subscript notation be $l = M  \setminus \{ M_i \}$ ？Is it  $l \in M \setminus \{ M_i \}$ instead?

   4.2 What is the motivation for using the range normalization term $(R_{l,\max} - R_{l,\min})$? 

   4.3 The scaling term before the clip function depends on raw differences in magnitude — what proportion of rewards is actually clipped, and could this excessive truncation suppress informative signals?

    4.4 Conceptually, how does incorporating another model’s relative advantage causally enhance the independent reasoning ability of the current agent? The underlying mechanism is unclear.

   4.5 If the formula is adapted from prior work, please provide an explicit citation. Otherwise, conduct an ablation study comparing alternative normalization methods.

5. Reward model implementation not specified.

6. Training cost and cost-effectiveness remain unclear, as the paper lacks quantitative analysis of computational overhead versus performance gains.

7. The use of only two agents provides limited evidence to support the claim of a “multi-agent system.”
In multi-agent interaction settings, both methodological complexity and computational cost increase substantially as the number of agents grows, yet the paper does not evaluate how ILR scales beyond two agents.

8. The paper assumes agents with different backbones, but it is unclear whether the framework would still work with identical models initialized by different prompts or roles.

### Questions
See Weakness.

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
3

### Summary
The paper investigates whether multi-agent interaction can enhance a single LLM’s independent problem-solving ability. The proposed ILR framework comprises two components: Dynamic Interaction and Perception Calibration. In Dynamic Interaction, the strategy (cooperation vs. competition) is adaptively selected based on problem difficulty and model capability. The LLMs then communicate via Idea3 (idea sharing, idea analysis, and idea fusion). In Perception Calibration, ILR adjusts rewards using the collaborator LLMs’ reward distribution features and trains the LLMs with GRPO. Evaluations on five benchmarks, compared against single-agent and multi-agent fine-tuning methods, demonstrate the effectiveness of ILR.

### Strengths
1. This work proposes the problem of enhancing a single agent’s capability via multi-agent collaboration and designs a novel learning framework named ILR.
2. The Dynamic Interaction module automatically chooses cooperation or competition by estimating the problem difficulty and model capability, avoiding the cost of manually setting a cooperation ratio. Ablations show that pure cooperation or pure competition is suboptimal, while dynamic selection is more robust.
3. By using collaborators’ reward distribution features (max/min/avg) to continuously shape the agent’s own reward and optimizing the single agent with GRPO, the method achieves strong experimental performance.

### Weaknesses
1. The motivation for Dynamic Interaction—“hard problems benefit from cooperation, easy problems from competition,” inspired by classroom observations—may not hold across all task settings. More conditioning variables may need to be incorporated to improve robustness.

2. The paper builds on GRPO with multi-agent discussion to optimize training but does not compare against single-agent GRPO variants such as [1] and [2]. What is the essential advantage of introducing multi-agent enhancement over improved single-agent GRPO baselines?

   [1] CPPO: Accelerating the training of group relative policy optimization-based reasoning models.

   [2] COG-RETHINKER: Hierarchical Metacognitive Reinforcement Learning for LLM Reasoning.

3. Using MATH-500 as an example, untrained collaborative inference already boosts Llama-3.1-8B from 49.80 to 64.00/66.20 (Table 4 Debate), whereas after ILR training the single-agent score on this benchmark rises only to 55.80/55.20 (Table 1). This “multi-agent at training, single agent at inference” setup may underutilize the potential advantages and ceiling of multi-agent systems at deployment.

### Questions
1. Training cost vs. single-agent learning methods: Is the performance gain partly due to more sampling? In ILR, each question involves multiple agents generating sampled answers, which could increase trajectories and reward evaluations relative to single-agent training.
2. Experiments are limited to two agents. Does increasing the number of agents in the discussion further improve training effectiveness?

### Soundness
2

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
This paper proposes a multi-agent learning framework that aims to enhance individual LLMs' reasoning capabilities through interaction, rather than building collaborative multi-agent systems for deployment. The key innovation lies in two components: (1) Dynamic Interaction that adaptively chooses between cooperation and competition based on question difficulty (estimated via self-ranking and Item Response Theory), followed by a communication paradigm (Idea Sharing, Analysis, Fusion), and (2) Perception Calibration that automatically incorporates peer reward distributions into each agent's reward function using GRPO. Experiments on Llama-3.1-8B and Qwen2.5-7B/14B across math and coding benchmarks show consistent improvements of up to 5% over single-agent baselines.

### Strengths
- The distinction between training for system-level vs. individual-level improvement is thoughtful and practically relevant. The paper correctly identifies that existing multi-agent work (MALT, ReMA, MAPoRL) focuses on building collaborative systems that require multiple agents at inference, which differs from human learning, where individuals improve through interaction but solve problems independently afterward.


- Using self-ranking combined with Item Response Theory to dynamically determine cooperation vs. competition is theoretically grounded. The IRT formulation with the 1.7 coefficient is well-established in educational assessment. The ablation study in Figure 3 effectively demonstrates that dynamic interaction outperforms pure cooperation (p=1.0) or pure competition (p=0.0), validating this design choice.


- The evaluation across multiple model families (Llama and Qwen), scales (7B-14B), and benchmarks (5 math + 1 code) is thorough

### Weaknesses
- While the combination is novel, the individual pieces are incremental. The Idea³ framework is essentially a structured variant of existing debate/reflection paradigms. Self-ranking for difficulty estimation and Reward shaping in multi-agent RL have been explored.


- The "up to 5%" improvement claim appears only for Llama-3.1-8B on one metric (average across benchmarks). Most models show 1-2% gains, and on individual benchmarks, improvements are often within noise. I feel there is some overclaim about results. The AIME results look impressive (e.g., 1.67→10.00 for Llama) but these are on tiny test sets with high variance. The paper doesn't report confidence intervals or significance tests


- The paper claims to be "the first to investigate whether multi-agent learning can more effectively enhance an LLM's independent reasoning capability" but doesn't compare against the extensive literature on self-reflection and self-improvement[1,2, 3, 4]


[1] Renze, M., & Guven, E. (2024). Self-Reflection in LLM Agents: Effects on Problem-Solving Performance. arXiv:2405.06682

[2] Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366

[3] Dou, Z., Yang, C., Wu, X., Chang, K., & Peng, N. (2024). Re-ReST: Reflection-Reinforced Self-Training for Language Agents. In Proceedings of EMNLP 2024.

[4] Zhang, W., Shen, Y., Wu, L., Peng, Q., Wang, J., Zhuang, Y., & Lu, W. (2024). Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives. In Proceedings of ACL 2024, pp. 3602-3622.

### Questions
Why is IRT the right framework here? The assumption that P(solve) follows a logistic function seems arbitrary for LLMs. Did you try simpler heuristics like directly using self-ranking scores?


Did you try ILR with more than 2 agents? With heterogeneous role assignments (like MALT's generator/verifier/refiner)?


Can you provide confidence intervals, especially for low-data benchmarks like AIME?

### Soundness
3

### Presentation
3

### Contribution
2
