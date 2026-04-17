# LLM Coaching LLM in Self-Play Training

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Large language models (LLMs) have achieved impressive progress in domains such as mathematics and programming, supported by verifiable outputs and robust benchmarks. However, their potential in games—longstanding benchmarks for reinforcement learning (RL) research—remains underexplored. The most prominent challenge is no ground truth–high variance, especially in complex and strategic games like Texas Hold'em, where naively optimizing for the highest observed payoff risks training dead loops, while payoff estimation itself is highly resource-intensive. To address these issues, we propose the LLM Coach, which transforms raw self-play (SP) payoffs into class-wise reward functions by leveraging payoff data, state information, and the current policy. This design stabilizes training and accelerates learning. Within an RL+SP framework, our Qwen2.5-32B agent significantly outperforms strong baselines (e.g., Grok4, GPT-o3) in Texas Hold'em Poker, while also exhibiting improvements in broader capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the "no ground truth–high variance" challenge in applying self-play reinforcement learning (SPRL) to large language models (LLMs) for strategic games, with a focus on Texas Hold'em Poker. It introduces an "LLM Coach" that classifies states from self-play data and provides class-level policy updates, integrated into a CFVFP-based RL framework. Using Qwen2.5-32B, the authors demonstrate improved performance over baselines like Grok4 and GPT-o3 in poker, with generalization to games such as Chess and Liars Dice.

### Strengths
Inspired by traditional Poker AIs, the LLM Coach innovatively leverages an LLM   to cluster states (e.g., by hand strength, board texture) and refine policies at a class level, drawing inspiration from human expert strategies. This helps address high variance in complex games without excessive rollouts.

The Qwen2.5-32B agent shows strong results, outperforming baselines in Texas Hold'em after seven self-play iterations.

### Weaknesses
1. **Insufficient Engagement with Related Work:** The paper's related work section is notably incomplete, missing critical discussion of closely related advances in LLMs for game-theoretic self-play. For example, it does not address "Game of Thoughts: Iterative Reasoning in Game-Theoretic Domains with Large Language Models" (Kempinski et al., AAMAS 2025), which proposes iterative refinement algorithms (BRIRLM, FPIRLM, PSROLM) for LLMs in normal-form games, sharing conceptual similarities with the LLM Coach. Similarly, PokerBench (Gupta, AAAI 2025) provides a poker-specific benchmark that could validate results, yet it is not cited. Other relevant works, such as SeRL (arXiv:2505.20347) for limited-data self-play and LLM-Gomoku (arXiv:2503.21683) for self-play in Gomoku, are also absent. This omission weakens the paper's positioning and reduces its perceived novelty, as these works address similar challenges in multi-agent strategic reasoning.
2. **Rigorous Proof for the Convergence to NE is Missing:** The paper's claim that the proposed method "preserves convergence guarantees to NE" (as stated in Section 1 and Section 3.1) is somewhat inconsistent with the practical realities of the POLICY-BASED LLM SPRL FRAMEWORK and the LLM Coach integration. For example, in the POLICY-BASED LLM SPRL FRAMEWORK, the equation (7) for computing counterfactual Q-value of each action at information set is simplified to Eq.(8), and the use of a fixed α=0.1 (Section 5.1) instead of the standard 1/(t+1) deviates from vanilla FP/CFVFP to stabilize LLM training. All of these alterations lack theoretical justification for preserving convergence.
3. **Missing Important Baselines:** Without this rigorous proof for the convergence to NE, this proposed method is just a heuristic RL approach to NE, similar to existing works on LLM with self-play discussed in the related work section. These works on LLM with self-play thus should be used as baselines in experiments. CoT and the related LLM enhance techniques should be used as baselines as well.

### Questions
What is SP-3?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors proposed an LLM-based SPRL framework that effectively addresses the challenges of lacking ground truth and high variance. To this end, the authors explicitly prompt the LLMs to output the probability as players' policies and introduce an LLM coach for state clustering and class-level refinements. Experiments in Texas Hold’em Poker showed that the method can train an LLM to surpass strong baselines.

### Strengths
- The idea of decomposing sequential games into node-level updates and leveraging the coach for state clustering and class-level refinements is well-motivated and reasonable.

- The experiments show that the method has strong superiority over baselines.

### Weaknesses
Due to the lack of sufficient knowledge in this field, I have some concerns (not very confident and please correct me if I make mistakes):

- The self-play framework has been widely used to train LLMs in recent works, but not all these related works have been discussed in the paper. Why do the authors preclude these works? Like [1-3]. Is it possible to compare the proposed method with these related works?

- The authors mentioned that the closely related paper is SPIRAL, but there is no comparison with this method. Can the authors provide some explanations?

- Minors: Abbreviations should be given the full name when they are presented in the paper in the first place, e.g., Line 59 CFVFP, Line 64 GTO.

---

[1] Wenkai Fang, Shunyu Liu, Yang Zhou, Kongcheng Zhang, Tongya Zheng, Kaixuan Chen, Mingli Song, Dacheng Tao. SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data. arxiv 2025.

[2] Guanting Dong, Guanting_Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou. Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models. ICLR 2025.

[3] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. ICML 2024.

### Questions
See **Strengths** and **Weaknesses**.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a self-play reinforcement learning (SPRL) framework for LLMs that combines the game-theoretic CFVFP algorithm with a novel LLM Coach module to address the challenges of missing ground truth and high payoff variance in strategic games. The LLM Coach emulates human-like abstraction by clustering similar game states and providing class-level policy improvements, significantly enhancing training stability and sample efficiency. Applied to complex games such as Texas Hold’em, the framework enables Qwen2.5-32B to outperform strong baselines like Grok-4 and GPT-o3 without GTO supervision, while also demonstrating strong generalization across multiple game domains.

### Strengths
1. This paper proposes the novel concept of an LLM Coach that mimics human-like strategy abstraction for policy refinement in self-play training. 
2. This paper builds a rigorous and principled framework by integrating CFVFP with LLM-specific adaptations such as policy-level outputs and game-node decomposition. 
3. Experimental results demonstrate strong performance gains in complex games and generalization across multiple domains, offering a scalable approach for LLM-based game agents.

### Weaknesses
1. The effectiveness of the LLM Coach is not evaluated independently, making it unclear how much it contributes relative to baseline CFVFP updates.
2. The reported regressions in general capabilities (e.g., math, coding) are acknowledged but not thoroughly analyzed or addressed.
3. The training and evaluation focus heavily on Texas Hold’em, with limited coverage or analysis of more diverse or structurally different games.

### Questions
1. Can the authors provide a more rigorous ablation study isolating the contribution of the LLM Coach from the rest of the framework (e.g., comparing models trained solely with CFVFP updates vs. LLM Coach updates over multiple iterations)?
2. How exactly does the LLM Coach perform state clustering and reasoning? Are there consistent latent groupings observed (e.g., by hand strength, betting history), and how stable are these across training iterations? 
3. Are there specific scenarios or state types (e.g., bluff situations, short-stack play) where the model consistently fails or regresses? 
4. The current evaluation is primarily in two-player zero-sum settings. Do the authors anticipate any challenges when extending the framework to multi-agent or general-sum games?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on addressing the two key challenges of LLMs in game-playing SPRL, i.e., no ground truth and high payoff estimation variance. First, it constructs an LLM-specific game environment leveraging DeepMind OpenSpiel, enhanced with the Elo rating system to enable quantitative evaluation. And then it proposes an LLM-based SPRL framework that integrates the CFVFP algorithm with an LLM Coach: CFVFP decomposes sequential games into independent subgames to enable node-level LLM policy optimization and ensure convergence to Nash Equilibrium (NE), while the LLM Coach transforms raw self-play payoffs into class-wise reward functions via state clustering and policy refinement, mimicking human experts. Experiments in Texas Hold’em Poker show that Qwen2.5-32B outperforms strong baselines, while also delivering improvements across diverse games.

### Strengths
1. The paper technically adapts the CFVFP algorithm to LLM-based SPRL. It decomposes sequential games into independent CFVFP nodes, enabling direct optimization of LLM policies at the node level via counterfactual Q-values. 
2. The paper introduces a novel LLM Coach mechanism, which addresses high payoff variance via a linguistically grounded, class-wise optimization pipeline for complex games.

### Weaknesses
1. I generally do not agree about the bottleneck the absence of ground truth and the high variance of payoff estimates. The success of DeepCFR and AlphaGo demonstrate that current algorithms can handle this, as well as the recently DeepSeek-R1, i.e., GRPO, can handle this case, where the reward only received at the end of the game. We have the ground truth during training, i.e., the reward received from the game, We may do not have the supervised policy, but this is also for some RL problems. Also, about the high variance, what exactly this mean?
2. The framework is tested on Texas Hold’em and smaller games, e.g., Leduc Poker, but provides no technical analysis of scalability to games with exponentially larger state spaces. Besides, in Figure 3, the random agent achieves 1569, while your agent achieves 1606 and the random agent is top 4. This is quite surprising, does that means your model is not that strong even compared with the random agents? And your model also scarifies the general reasoning capabilities, as demonstrated in Table 2. 
3. It only take less than one page for the approach, i.e., section 5. Many key technical components are not explained, e.g., CFVFP decomposition, reward function weights. These components are also not ablated in the experiments. Also, about the imitation learning, why this is needed, why directly using self-play? It seems that imitation learning plays a critical role in the training due to the format issue, so the self-play cannot handle the format issue? I believe some analysis is required for understand the influences of all components in your framework.

### Questions
Please refer to weakness section.

### Soundness
3

### Presentation
2

### Contribution
2
