# Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
Large Language Models (LLMs) have recently demonstrated strong potential in generating ‘believable human-like’ behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments. Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline. The project page is available at https://damon-demon.github.io/shop-r1.html.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
A user simulation of online shopping, Shop-R1, is proposed in this work. The user model will generate actions with reasoning, and is improved by supervised learning and reinforcement learning. To prevent reward hacking and accurately represent the difficulty of specific actions, a hierarchical reward signal is proposed. However, the method is only tested with Qwen variants models and compared with limited baselines, which should be further improved in the future version.

### Strengths
* The authors conduct a detailed ablation study across different training configurations, e.g. different reward signals and temperature of sampling.

### Weaknesses
* Leveraging supervised learning and reinforcement learning for user simulation is a well-studied area. For example, the user model in GenTUS [1] is first trained with supervised learning and then further optimised through reinforcement learning. Similarly, USimAgent [2] simulates user clicking behaviours, including reasoning processes. Therefore, the novelty and contribution of this work should be clarified and elaborated further, which means the first contribution mentioned in the introduction could be an overstatement, where the authors claim they are the first to introduce RL into simulation-oriented human behaviour modelling.
* Building on the previous point, existing reinforcement learning and user simulation methods should be more carefully examined, as they may already address some of the challenges discussed in this paper. For instance, to prevent the model from selecting trivial actions (e.g., always choosing termination), a user goal and goal-failure penalty can be incorporated. Moreover, the choice of hyperparameters (e.g., the weight of the type reward) requires stronger justification. In addition, “accuracy” or “F1 score” may not be ideal metrics for evaluating user models, since multiple user behaviours can be equally reasonable. More suitable alternatives could include direct evaluation (e.g., human judgments of naturalness) or indirect evaluation (e.g., assessing the downstream system trained on simulated users).
* The model selection is also limited (only Qwen variants are tested), making it difficult to assess the generalizability of the proposed framework. Furthermore, the absence of proper baselines weakens the validity of the claimed contributions.

[1] Lin et al. GenTUS: Simulating User Behaviour and Language in Task-oriented Dialogues with Generative Transformers. SIGDIAL 2022.

[2] Zhang et al. USimAgent: Large Language Models for Simulating Search Users. SIGIR 2024.

### Questions
* In L081, the human behaviour is divided into two stages. Is there any theory or research supporting this setting? Is this really how humans act during web shopping?
* Typo: The "Sec. A" should be "Appendix A" in L177. In addition, the details of the action space are still unclear, since only a prompt is provided in the appendix. 
* Why is there no negative reward in the proposed training framework?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the use of LLMs to simulate human behaviors in online shopping scenarios. It introduces Shop-R1, a RL framework designed to optimize the task. Shop-R1 integrates a self-certainty score, an internal metric for assessing rationale generation, and employs a hierarchical reward scheme for evaluating action prediction. Additionally, it incorporates a difficulty-aware reward mechanism that assigns greater rewards to correct actions deemed more challenging.

### Strengths
1. The idea of leveraging LLMs to simulate human decision-making in online shopping is interesting and promising.

2. The methodological presentation is generally clear, particularly the explanation of the reward design.

### Weaknesses
My main concern is that the task definition is unclear, making it difficult to assess the task's difficulty or determine whether the benchmark genuinely reflects real-world challenges. In addition, the experimental results do not provide strong evidence supporting the effectiveness of the proposed method. Specifically:

1. Datasets:
  - The paper lacks sufficient detail on how the datasets were constructed, and provides no illustrative examples, making it difficult to assess task difficulty or data quality.
  - Basic dataset statistics (e.g., action distribution) are missing.
  - The paper does not describe how the training and test sets are split. 
2. Experiments:
  - The experiments are limited to a 3B-parameter model. It is unclear whether Shop-R1’s effectiveness generalizes to larger models.
  - In the ablation study (Table 4), components such as reward scale, rationale reward, and hierarchical reward only improve exact action accuracy by less than 1%. The absence of multiple experimental runs further weakens the robustness of these results, making the improvements appear marginal.

### Questions
1. According to the definitions provided in the evaluation metrics (line 299), the accuracy of Exact Action should always be lower than that of Action Type. However, in Table 4, for the setting 'w/o reward scale', the accuracy of Exact Action is larger than that of Action Type. Could the author clarify this?

2. Human shopping behavior tends to be highly personalized. Given the same context, individuals may act differently depending on personal traits, purchase history, or preferences. In this paper, the agent’s context includes only past actions. Does this simplified setting can capture the problem in real-world?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the limitations of Supervised Fine-Tuning (SFT) for simulating human web-shopping behavior, which is capped by the quality of synthetic rationales. It proposes Shop-R1, an SFT-RL framework, which decomposes the task into rationale generation (guided by a self-certainty reward) and action prediction (guided by a hierarchical, difficulty-scaled reward). Experiments show a 65% relative improvement in exact-match accuracy over the SFT baseline.

### Strengths
1. The work tackles the important and practical challenge of simulating nuanced, human-like behavior in web environments, moving beyond simple task completion.

2. The paper introduces a sophisticated, hierarchical reward scheme with difficulty-aware scaling that effectively mitigates common reward-hacking failure modes.

3. The empirical results are strong, demonstrating a significant 65% relative improvement in exact-match accuracy over the SFT baseline.

### Weaknesses
1. Limited Novelty: The core training paradigm, which combines SFT with subsequent RL optimization, is a common practice in the field. The contribution is thus an incremental (though effective) application of established methods rather than a fundamentally advancement.

2. The proposed hierarchical reward function is highly complex and task-specific, with numerous hard-coded weights and rules. This intricate, hand-crafted design is brittle and would be difficult to generalize to new simulation environments without significant re-engineering.

The paper's contribution to ML or RL appears limited. However, as I do not work in the specific sub-field of human behavior simulation, I cannot fully assess the potential impact of the authors' claim to be the "first to introduce RL into a simulation-oriented human behavior modeling task." My evaluation is therefore borderline.

### Questions
Have you qualitatively or quantitatively evaluated the quality of the rationales themselves? Do the rationales from Shop-R1 appear more "human-like" than the SFT baseline, or do they simply score higher on the self-certainty proxy metric?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Shop-R1, an RL training pipeline for simulating human-like shopping behavior on web UIs. The task is decomposed into two heads per step: (i) rationale generation and (ii) action prediction. Rewards are hybrid: a format reward enforces JSON outputs; a self-certainty reward for rationales measures average KL divergence from a uniform distribution; and a hierarchical action reward scores both action type and sub-action fields with difficulty-aware reward scaling (DARS). Training uses SFT as a cold start followed by RL. On a proprietary dataset of 52,137 real-world shopping sessions, Shop-R1 improves exact-match action accuracy from 16.76% (SFT) to 27.72%, a ~65% relative gain, with ablations for temperature, component importance, and context usage.

### Strengths
- The task is unique and useful and well-motivated; and this approach is a useful step for scalable UX evaluation, A/B testing simulators, and recommendation research.

- Introduces a simulation-oriented RL design for human-behavior replay rather than task completion, with a two-head reward structure (rationale + action).

- The hierarchical + DARS scheme is a pragmatic way to densify rewards and disincentivize “terminate”-spamming. In addition, the reward is a well-motivated reward design. The hybrid structure (format + self-certainty + hierarchical+DARS) addresses parseability, sparse credit, and reward hacking.

- Paper is well written and organized.

### Weaknesses
- All results rely on a proprietary single-site dataset; there’s no evaluation on public benchmarks or cross-datasite generalization.

- Self-certainty reward risks overconfidence. Using KL-to-uniform for rationales may incentivize confident but incorrect chains of thought; calibration is not reported.

- Reward design sensitivity. DARS is fixed at 1000, but sensitivity curves and thresholding for ROUGE-L (the text says “e.g., 0.75”, p. 5) are not systematically explored.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
