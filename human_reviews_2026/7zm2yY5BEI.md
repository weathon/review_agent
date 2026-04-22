# Incentivizing Multimodal Reasoning via Progressive Curriculum Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Reinforcement learning has shown strong potential in improving the reasoning abilities of large language models, and recent studies extend this paradigm to multimodal reasoning. However, the complexity and diversity of multimodal tasks often lead to unstable performance across domains and difficulty levels. To address these challenges, we introduce VL-COGITO, a multimodal reasoning model trained with a multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL gradually increases task difficulty, enhancing reasoning robustness in diverse contexts. It features two key innovations: (1) an online difficulty–aware weighting mechanism that dynamically adjusts task difficulty across training stages, and (2) a dynamic length reward that encourages adaptive control of reasoning path length to balance efficiency and accuracy. Experiments demonstrate that VL-COGITO achieves state-of-the-art performance on 8 out of 10 benchmark tasks spanning mathematics, science, logic, and general understanding, while matching comparable results on the remaining 2 tasks, validating the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes **VL-COGITO**, a MLLM designed for multimodal reasoning, trained via a **Progressive Curriculum Reinforcement Learning framework (PCuRL)**. PCuRL comprises two core components:

1. **ODSW**: Dynamically adjusts sample difficulty weights across different training stages, progressively enhancing the model’s reasoning capability from easy to hard examples.
2. **DyLR**: Adaptively modulates the length of reasoning chains based on task complexity, preventing over-reasoning on simple questions and under-reasoning on complex ones.

### Strengths
1. Introduces a curriculum-based reinforcement learning approach to equip the model with the ability to handle tasks of varying difficulty levels.

### Weaknesses
1. **Limited performance gains**: Although the method achieves state-of-the-art results on most benchmarks, the absolute improvement over the strongest baselines (e.g., VL-Rethinker, MMR1) is typically only 1–3 percentage points. On certain tasks, it even performs slightly worse, casting doubt on the statistical or practical significance of the reported gains.
2. **Scalability concerns**: PCuRL has only been validated on a 7B-parameter model. It remains unclear whether this framework generalizes effectively to much larger models (e.g., 30B+ parameters).

### Questions
1. Since the same dataset is used across all three training stages, could this lead to overfitting during the "hard" stage? Are there validation-set evaluations or cross-dataset generalization experiments to mitigate this concern?
2. Given the relatively modest performance improvements (mostly <3%), is the added complexity of integrating ODSW + DyLR with a multi-stage training pipeline justified?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes PCuRL (Progressive Curriculum Reinforcement Learning) and the model VL-COGITO. PCuRL organizes GRPO training into three stages—easy → medium → hard—over the same data, reshuffled per stage. There are two key mechanisms: (1) Online Difficulty Soft Weighting (ODSW): A stage-specific, continuous weighting of prompts by online rollout accuracy, (2) Dynamic Length Reward (DyLR): A cosine-shaped reward that targets, per prompt, the average length of correct rollouts (or a capped maximum when none are correct).

### Strengths
1. The paper is generally well-written and easy to follow, with a clear description of the method.
2. The paper provides intuitive visual demonstrations to help better understand the paper.

### Weaknesses
1. **Limited novelty.** The paper’s two core contributions: difficulty-aware weighting and a length reward within a curriculum RL framework have already been extensively explored in prior work [1,2,3]. Moreover, even the data curation choice of an open-ended response format has precedent (e.g., NoisyRollout) [4]. As a result, the manuscript appears to build primarily on established conclusions, and its incremental contributions seem limited relative to the existing literature.
2. **Design choice for online difficulty soft weighting.** In each curriculum stage, the authors reuse the same dataset and intervene via reweighting rather than performing the standard curriculum step of pre-partitioning samples by difficulty and staging them accordingly. Why not adopt the conventional approach of separating easy/medium/hard subsets and advancing through them? In addition, letting every stage train on the full data makes it difficult to attribute gains solely to the proposed soft weighting; improvements may stem from repeated exposure to all examples rather than from the weighting mechanism per se.
3. **Scope and evidence for the length reward.** The length reward is only activated in the hard stage. According to `Fig.3` (Training curves of PCuRL), the only clearly diverging trajectory from the baseline appears in the Average Length Curve during stage three. This design does not convincingly substantiate the claim in `Line 240–242` that the method “avoids fixed targets that risk over- or under-thinking.” A more interpretable analysis (*e.g.*, per-task length/accuracy Pareto fronts, error taxonomy vs. chain length) and a complete ablation across stages (length reward on/off in easy/medium/hard) are needed to validate this claim.
4. **Potential zero-gradient issue without filtering**. At `Line 172–173`, the authors emphasize that their approach “retains more prompts” instead of filtering as in other works. Does this choice reintroduce the all-correct / all-incorrect (i.e., zero-variance) groups that can yield zero or near-zero gradient under GRPO-style normalization?
5. **Fairness of experimental comparisons.** Fairness remains unclear in `Tab.1`. It is not specified whether baselines share the **same base model**; for instance, early versions of R1-VL [5] are based on **Qwen2-VL**, which differs from Qwen2.5-VL used here. Additionally, the paper samples **16** responses per prompt during training, whereas many compared methods use **8 or 12**. This hyperparameter materially affects group advantage estimation stability and accuracy in GRPO. Please report an ablation on the group size (G) and ensure base model parity to enable a fair comparison.
6. **Attribution to algorithm vs. data scale/overlap.** The training corpus spans many domains, with several sources closely related to the evaluation benchmarks (e.g., MathVista). By contrast, MM-Eureka [6] and NoisyRollout [4] train with substantially **smaller datasets** (≈15K and ≈3K, respectively). It remains unclear how much of the observed gains arise from the PCuRL algorithm versus data scale and domain overlap. Please provide experiments controlling for training data size and composition (e.g., fixed subsets, down-sampling to match [6]/[4]) to disentangle algorithmic benefits from data effects.

**References**

[1] GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models, https://arxiv.org/abs/2504.09696 

[2] L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning, https://www.arxiv.org/abs/2503.04697 

[3] Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning, https://arxiv.org/abs/2503.07065 

[4] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation, https://arxiv.org/abs/2504.13055 

[5] R1-VL: Learning to Reason with Multimodal Large Language Models via  Step-wise Group Relative Policy Optimization, https://arxiv.org/abs/2503.12937v1 

[6] MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning, https://arxiv.org/abs/2503.07365

### Questions
See the `Weaknesses` part.

### Soundness
3

### Presentation
3

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
The paper proposes VL-COGITO, which uses a three-stage progressive curriculum to reinforce learning (easy → medium → difficult) to train a multimodal reasoning model: a dynamic length reward is introduced in the difficult stage, with the average thought length of each correct sample as the target, which encourages deeper thinking for complex questions and avoids verbose thinking for simple questions; it achieves 8 state-of-the-art results on 10 benchmarks and close results for the rest, showing that accuracy and stability can be significantly improved without SFT.

### Strengths
This paper employs a course-based reinforcement learning approach, progressing from easy to medium to difficult tasks. It first stabilizes and optimizes quickly on easier problems, then gradually focuses on more challenging tasks, significantly reducing training oscillations while improving gradient and data utilization.

In the difficult phase, a dynamic reasoning length reward is introduced, setting a target based on the average inference length of correct samples per problem. Complex problems encourage deeper thinking, while simple problems maintain conciseness.

Eight of the ten multimodal benchmarks achieved SOTA results, with the rest approaching or surpassing the benchmarks.

### Weaknesses
Are there any automated switching methods for multi-stage training, or what quantitative metrics can guide multi-stage training?

Does encouraging longer, more challenging problem-solving sessions increase the likelihood of issues like repeated output or random output? Should we add penalties for this?

Figure 3 shows that the improvement in the last stage is not significant, and the increase in the estimated length in the third stage has no obvious effect.

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a multimodal reasoning model named VL-COGITO, whose core is a novel PCuRL framework. This framework aims to address the performance issues arising from the complexity and diversity of multimodal tasks. PCuRL features two key innovations: 1. an ODSW mechanism that dynamically adjusts task difficulty during training. 2. a DyLR mechanism that incentivizes the model to adaptively control the length of its reasoning path according to the complexity of each problem.
Experimental results show that VL-COGITO achieves state-of-the-art performance on 8 out of 10 benchmark tasks, and the authors further conduct extensive ablation studies to verify the effectiveness of each component.

### Strengths
1. This paper combines curriculum learning with reinforcement learning to specifically tackle complex multimodal reasoning problems. The two core components ODSW and DyLR are designed to address challenges in training stability and reasoning depth across tasks of varying difficulty. This approach provides valuable guidance for training multimodal reasoning models.
2. The paper conducts extensive experiments on diverse multimodal benchmarks spanning mathematics, science, logic, and general understanding. VL-COGITO achieves SOTA performance on eight benchmarks, outperforming both general models and other SOTA reasoning models. The authors also perform comprehensive ablation studies to demonstrate the effectiveness of the proposed methods.

### Weaknesses
1. The authors claim in line 60 that the model “bypasses the cold-start SFT phase.” However, the backbone is Qwen2.5-VL-7B-Instruct, which has already undergone extensive SFT on large-scale instruction-following data. Thus, this statement is misleading, as the model benefits from a SFT procedure.
2. During data curation, the authors discard samples with pass@8 > 50%, keeping only data with higher "learnability". I am curious how a simple SFT on these selected  samples would perform compared to the proposed PCuRL pipeline.
3. The paper states (around line 350) that applying DyLR directly to GRPO destabilizes training. However, in Table 2, the model “+DyLR” actually outperforms vanilla GRPO (58.1 → 58.6). This discrepancy is confusing and needs clarification.
4. DyLR (Eq. 5 & Eq. 8)
   When Acc = 0, the DyLR encourages the model to generate longer responses via L_max, while Equation 8 simultaneously down-weights these samples with w = 0.25. This appears contradictory: on one hand incentivizing exploration, on the other reducing its significance.

   I would like to know:

   a. Has the authors verified whether this design truly produces conflicting optimization signals?

   b. Since DyLR is only applied in the final stage, where accuracy should already be higher (if not, please clarify), are these potentially conflicting cases (i.e., Acc = 0) too rare to matter, and thus masked by the accuracy gains from earlier stages?

5. The difficulty-dependent weighting functions used in ODSW appear heuristic. The paper lacks an in-depth analysis for these functional forms.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
