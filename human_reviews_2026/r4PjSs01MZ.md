# RePAIR: A Rule-based Process-Adaptive Reinforcement for Large Language Model Training

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Although reinforcement learning (RL) has demonstrated promise in enhancing the reasoning capabilities of Large Language Models (LLMs), the difficulty of reward design has prohibited exploiting the full potential of RL. Previous methods mainly fall into two categories: training a reward model based on human preferences, or designing verifiable outcome rewards. However, reward models often suffer from poor interpretability and require extensive annotation for effective training. Verifiable outcome rewards provide sparse signals only, which leads to an ambiguous credit assignment and low training efficiency in RL. These limitations necessitate rewards that provide more efficient, fine-grained supervision. In order to address these, we propose Rule-based Process-AdaptIve Reinforcement (RePAIR) that constructs adaptive verifiable process rewards through symbolic reasoning rules. These rules are automatically derived through the integration of common pattern mining and semantic summarization over the reasoning trajectories of LLMs. For stable training purposes, RePAIR defines a reward informativeness metric that dynamically adjusts the rule's weights based on policy updates. Extensive experiments across three reasoning tasks demonstrate that RePAIR achieves a 6.03% improvement on average and combines well with various advantage functions. Code and data will be available at https://anonymous.4open.science/r/RePAIR-8EFC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RePAIR, a Rule-based Process-Adaptive Reinforcement framework aimed at improving reinforcement learning for LLMs. It tackles the challenges of sparse and ambiguous reward signals in traditional RLHF and RLVR by incorporating symbolic reasoning rules that yield verifiable and adaptive process-level rewards. These rules are automatically extracted from model-generated reasoning trajectories via frequent subgraph mining and semantic summarization, enabling interpretable and fine-grained feedback throughout training.

### Strengths
1. The introduction of symbolic, rule-based process rewards enhances interpretability, verifiability, and adaptability, effectively addressing the limitations of outcome-based and black-box reward models.

2. The framework automatically derives reasoning rules from LLM trajectories, reducing dependence on manual reward engineering and human annotation.

3. RePAIR achieves consistent and substantial improvements across multiple reinforcement learning algorithms and reasoning benchmarks.

4. The paper is well written, with clear algorithmic descriptions, implementation details, and a stated plan for open-source release.

### Weaknesses
1. In Table 3, the performance gain over the unverified RULE baseline is modest for GRPO and REINFORCE++, suggesting that improvements may primarily stem from adaptive weighting rather than rule validation.

2. Experiments are conducted mainly on small-scale models (≤3B parameters), leaving it unclear whether RePAIR scales effectively to larger models (e.g., 7B+).

3. The influence of the informativeness update parameters ($\alpha$, $\beta$, $\eta$) is not systematically analyzed, which may limit reproducibility.

### Questions
1–3. See weaknesses above.

4. How does RePAIR generalize to tasks beyond mathematical or structured reasoning, such as commonsense or dialogue-based reasoning?

5. What is the computational overhead of rule extraction and verification relative to standard RL training?

6. The adaptive weighting mechanism depends on the success rate, which requires access to the final outcome. Does this constrain applicability in settings with only process-level feedback (i.e., without a final answer)?

7. In tasks where intermediate reasoning cannot be easily decomposed into discrete steps, how would graph construction and rule matching be adapted?

### Soundness
2

### Presentation
3

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
In this paper, the authors proposed an innovative framework called RePAIR. This framework can automatically extract symbolic rules and transform them into dynamic, fine-grained process reward signals. The authors compare the method with a series of baselines and evaluate its effectiveness.

### Strengths
1. In this paper, the authors provide sufficient experimental details and hyperparameters. These settings cover the settings for reproducibility.

2. The structure and format of this paper are clear and easy to follow. The authors also provide clear figures and tables to enhance the readability.

### Weaknesses
1. In the related work section, the authors mention PRM but fail to discuss these methods. In recent research, there are many methods of LLM-as-a-judge using strong LLMs or token-level process reward. These methods are direct competing solutions for the method. The lack of comparison between these methods weakens the persuasiveness of the method.

2. The experiments in this paper are primarily based on the Qwen model. Although they cover different scales of LLM, the model architecture is singular. This setting limits the generalizability of the method and contradicts the model-agnostic assumption mentioned in the paper. In addition, the method relies on external strong teacher models like GPT-4o or Deepseek-R1. This raises the question of whether the method remains effective when using weaker models for rule extraction.

### Questions
1. The authors mention that rules can be extracted from successful and failed trajectories. However, the authors only discuss positive rules. An issue worth discussing and studying is how negative rules have an impact. The authors may need to present a case study of negative rules.

2. In this paper, the authors use first-order logic to represent rules. This strategy is clear and verifiable. However, this hard-coded approach may limit the LLM's reasoning ability since the process of LLM reasoning may be more ambiguous. An open question is whether more ambiguous soft rules would lead to better results.

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
4

### Summary
The paper introduces a new approach to accelerate LLM-based reasoning by specifying an intermediate reward signal learned from successful reasoning trajectories. The approach involves first constructing multiple symbolic graphs from these trajectories, where each graph represents a rule-based reward signal. At each step, these signals are combined using adaptive weights to form a single intrinsic reward. These weights are updated based on the hit rate and success rate of each rule. The paper demonstrates the empirical benefits of this method on several reasoning tasks, including GSM-8k, AIME-23, and AIME-24, using a 1.5B Qwen model.

### Strengths
- The general idea of constructing an intrinsic reward signal to accelerate the learning process is a promising direction, particularly for problems where the true reward signal is sparse and obtained after long trajectories. The paper introduces a valuable contribution in this area.
- The proposed approach is algorithm-agnostic, demonstrating its effectiveness with GRPO, Reinforce, and Dr. GRPO on various reasoning benchmarks.
- The experiments section is well-organized, and the paper aims to answer several interesting questions regarding the proposed approach, such as its compatibility with several RL algorithms, the generalizability of the reward signals, and its comparison to handcrafted rules.
- In general, the paper is well-written and easy to follow.

### Weaknesses
- The primary weakness of the paper is its questionable technical soundness:
    - It is well-known in RL that constructed intrinsic reward signals must follow specific rules for the optimal policy to remain invariant [1]. However, it is unclear whether the proposed intermediate reward signal satisfies these requirements.
    - The advantage term (Eq. 5), which combines the intrinsic and extrinsic rewards, appears mathematically unsound and biased. The paper lacks explanation or proof of how it leads to the update described in Eq. 1.
    - The reward informativeness metric and the adaptive weight update rule lack sufficient mathematical explanation. It is unclear why Eq. 4 has the desirable effect. Intuitively, the rules that appear in unsuccessful trajectories are equally emphasized as long as the same rule appears in successful trajectories, as the metric does not use information about unsuccessful trajectories.
- The overall empirical results are underwhelming, partly due to the algorithmic inconsistencies noted above and partly due to the training procedure. A 1.5B model achieves a score of 25+ on AIME-24 (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), but the presented results are significantly lower. Furthermore, the benchmarks used may be too simple, as many are known to be relatively easy and might not adequately highlight the usefulness and scalability of the approach. A more thorough evaluation would also consider baselines from process reward modelling (PRMs).
- Some parts of the paper require more explanation; for instance, the concept of “frequent subgraph mining” is not clearly defined for the reader.

[1] Ng, Andrew Y., Daishi Harada, and Stuart Russell. "Policy invariance under reward transformations: Theory and application to reward shaping." Icml. Vol. 99. 1999.

### Questions
- The paper should provide more details on the rule-generation process. Since the approach uses an LLM to both extract semantic features and generate executable rules, it is unclear how the validity and reliability of these rules are ensured.
- The authors should elaborate on the mechanisms that prevent the generation of numerous, question-specific rules. It is unclear how the method maintains a bounded total number of rules and, by extension, ensures the generalizability of these symbolic rules beyond the specific questions used for training.
- The methodology of the generalization experiment requires clarification. The paper should explicitly state whether this experiment was designed to test the generalizability of the generated symbolic rules themselves or the final model trained with them. The former is more interesting.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework called RePAIR to improve LLM's reasoning ability. To deal with the challenges in human preference reward models and verifiable outcome reward models, RePAIR extracts symbolic reasoning rules from model reasoning trajectories, and turn them into verifiable process-level rewards which guide training at finer granularity.

### Strengths
1. The proposed idea is novel. RePAIR can extract symbolic reasoning rules from LLM-generated trajectories, which formalize common reasoning patterns as a computable function to provide a verifiable and interpretable basis for process supervision.
2. The dynamic weighting strategy ensures that the most informative rules guide training.
3. The rule extraction is lightweight and computationally inexpensive.

### Weaknesses
1. The convergence of the adaptive rewards needs justification. As  RePAIR continuously adjusts rule weights based on “reward informativeness,” the reward function itself changes during training. With a dynamically changing reward, the RL policy will get confused and divergent. 
2. The scalability is limited. The rule extraction pipeline is based on subgraph mining and symbolic reasoning. It is effective for structured reasoning (e.g., math or logic) but may not scale efficiently to open-ended tasks such as dialogue, coding, or multimodal reasoning.
3. Small models tend to memorize symbolic rules instead of learning true reasoning patterns. It causes overfitting to specific rule structures and poor generalization to new tasks. As a result, training appears successful on seen data but fails to transfer to diverse reasoning scenarios.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
