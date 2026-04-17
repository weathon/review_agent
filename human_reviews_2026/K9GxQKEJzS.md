# Bootstrapping Language and Numerical Feedback for Reinforcement Learning in LLMs

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Reinforcement learning for large language models (LLMs) often relies on scalar rewards, a practice that discards valuable textual rationale buried in the rollouts and hampers training efficiency. Naive attempts to incorporate language feedback are often counterproductive, risking either memorization from leaked solutions or policy collapse from irrelevant context. To address this, we propose Language-And-Numerical Policy Optimization (LANPO), a framework that cleanly separates the roles of feedback: language guides exploration, while numerical rewards drive optimization. LANPO builds a dynamic experience pool from past trials and introduces two principles to ensure feedback is effective: Reward-Agnostic Reflection for safe intra-sample self-correction and Relevant Abstraction to distill generalizable lessons from inter-sample experiences. Across mathematical reasoning benchmarks, LANPO enables 7B and 14B models to significantly outperform strong baselines trained with GRPO in test accuracy. Our work provides a robust method for integrating historical experiences into the LLM RL loop, creating more effective and data-efficient learning agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method to integrate natural language feedbacks in on-policy RL training. Specifically, this method introduces two methods: reward-agnostic reflection and relevant abstraction. The former serves to do self-reflection on its previous roll-outs (without reward feedback) and facilitate more exploration; the latter summarizes policies from similar tasks and shrinks exploration space. On mathematical reasoning tasks, this work outperforms GRPO based methods.

### Strengths
1, Combining language and scalar feedbacks is a very interesting direction. And this work proposes an reasonable solution, which could be an important contribution to the research community.

2, I especially appreciate the empirical exploration on challenges in introducing language feedback (section 3). These experiments provide valuable insights on how pure language feedback affects training and test performance.

3, In section 5, empirical results show that this method could effectively improve vanilla GRPO performance by introducing language feedbacks.

### Weaknesses
1, My biggest concern is with the writing in Section 4. While the proposed framework is sound in general, there are unclear points after reading section 4. (I will list my questions in the question section.) I would recommend the authors to describe their method using more mathematical notations. 

2, As pointed in the discussion section, the authors also admit that long context would be induced during training time, which will significantly increase training time. Considering that RL training is already time-consuming, further training time increase could be intimidating. Could you compare the wall-clock training time of vanilla GRPO and your time (using the same hardware)?

3, The last point is less concerning, but worth pointing out that the reflection and summarization actually requires LLMs with decent language capabilities. So, the proposed method might not be effective for small size LMs. Could you run an experiments with Qwen3-4B model?

Minor:
1, The “strong ICL benefit” in Figure 2 caption is unclear by reading the figure. Please consider using “inter-sample feedback”, or other appropriate terms.

### Questions
1, The first question is that whether the reflection and the contexts will be used to compute the gradient and back-propagated?

2, For the self-reflection process, what are the input and outputs? What languages/tokens are added to the RL training context? Will there be a distribution shift if no intra-sample reflection is conducted during inference time?

3, Details in Section 4:
- in line 261, What is the “feedback”?
- in line 263, what is the “high-level solution plan”?
- in line 283, what do you mean “preserving independence”?
- in line 286 to line 288, the loss/negative reward is significantly different from that of GRPO. Can you explain why you make those changes?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes LANPO, a framework that integrates both language and numerical feedback into RL for LLMs. Traditional LLM-RL methods rely solely on scalar rewards, discarding valuable reasoning signals embedded in textual outputs. LANPO addresses this by introducing two key mechanisms: Reward-Agnostic Reflection, which enables safe intra-sample self-correction, and Relevant Abstraction, which filters and summarizes inter-sample experiences into transferable reasoning principles. Together, these mechanisms allow the model to reuse its own past rollouts as structured guidance during training. Experiments on multiple mathematical reasoning benchmarks (AIME24/25, AMC, MATH) demonstrate that LANPO consistently outperforms strong baselines such as GRPO across both 7B and 14B models, achieving stable optimization and better zero-shot generalization.

### Strengths
1. The paper addresses an underexplored yet important problem: how to effectively integrate language feedback into RL for LLMs, which is highly relevant to advancing reasoning and RL sample efficiency.
2. The paper introduces two well-motivated components: Reward-Agnostic Reflection, enabling safe and label-free intra-sample self-correction, and Relevant Abstraction, which distills inter-sample experiences into transferable principles for more robust generalization.
3. Consistent performance improvements: LANPO achieves notable gains over strong baselines such as GRPO across multiple reasoning benchmarks and model scales, under both 7B and 14B Qwen models.

### Weaknesses
1. My primary concern is that the experiments are conducted solely on Qwen models and mathematical reasoning tasks. Recent studies [1] have highlighted potential overfitting or evaluation bias in Qwen+math evaluations. The paper would be substantially strengthened by including results from additional models or domains.
2. LANPO introduces multiple auxiliary components (e.g., summarization, reflection, retrieval), which substantially increase computational cost and token consumption. The reported performance improvements may therefore be partially attributable to greater training time. A fairer comparison would normalize along the x-axis (training time or total token budget) and report performance on the y-axis.
3. LANPO updates the policy twice using the same data: once through parameter optimization and once through prompt modification (feedback injection). This effectively makes the training partially off-policy, since the prompt update alters the policy distribution between rollouts and updates. The authors do not discuss this issue, yet in practice, such prompt-level changes can substantially shift the policy distribution and degrade the stability of GRPO-like on-policy RL algorithms.
4. Recent works have shown that self-reflection mechanisms often fail to improve model outputs in the absence of external supervision signals [2]. A key conceptual concern remains: if the model can solve a problem correctly after reflection, why couldn’t it solve it correctly in the first pass? Is the improvement simply due to longer context or increased reasoning steps?
5. As shown in Table 2, the model attains 44.64 without filtering and 44.83 with filtering, indicating that filtering offers only a minor benefit when the testing configuration is consistent with the training setup.

[1] Spurious Rewards: Rethinking Training Signals in RLVR

[2] When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs

### Questions
1. Have you evaluated LANPO on models or tasks beyond Qwen and math reasoning to test generalization?
2. Can you report results normalized by total training tokens or compute to ensure fairness?
3. Does updating prompts between rollouts introduce off-policy effects or instability?
4. Are the reflection gains due to better reasoning or simply longer context?

### Soundness
3

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
4

### Summary
This paper proposed LANPO, a framework that incorporates language feedback in RL to guide exploration. LANPO builds and experience pool to provide 2 types of feedback: 

1. Reward agnostic reflection where the model is is shown its prior response to a prompt and asked to critique it before producing a refined response. This is different from naive intra-sample feedback where the model is shown the label instead. By making intra-sample feedback reward agnostic, LANPO prevents the label leakage issue. 

2. Relevant abstraction for inter sample feedback. To prevent the behavior collapse induced by naively incorporating raw solutions, LANPO first finds sufficiently similar trajectories from the experience pool and then condenses them into high-level principles that generalize across problems. 

During RL, the model alternates between from scratch and feedback aware rollouts to ensure that the model can perform well with and without drawing from the experience pool during inference. 

LANPO consistently outperforms GRPO on sample efficiency and achieves an absolute performance improvement of 9.27% on the AIME25 test set after the same number of training steps.

### Strengths
- The paper introduces a novel and thoughtful way to incorporate language feedback into RL. Utilizing the experience pool to provide two different types of feedback allows for the model to learn introspection and how to generalize transferable learnings across examples.
 -  Intra-example feedback in particular consistently improves zero-shot and self-corrected performance across all benchmarks even without language feedback training.
- LANPO achieves better sample efficiency than GRPO, while improving both zero-shot and feedback augmented inference. 
 - The paper is well written with a number of insightful ablations that study the impact of filtering on inter-example feedback, and the balance between feedback aware rollouts and from scratch generation during RL,

### Weaknesses
- LANPO's effectiveness is studied in a relatively limited setting. Only Qwen 7B and 14B models were included in all experiments and all the experiments were conducted on math tasks. For better generalizability, it would be helpful to see results on a broader range of model sizes. It's interesting that the 7B model benefits more from LANPO than the 14B model, being able to study any trends here would be nice.  

- Including experiments with LANPO on at least one other type of tasks like coding, safety, general instruction following etc would go further in proving its generalizability. It could be difficult to find relevant examples from the experience pool for broader tasks complicating the relevant abstraction step.

- The computational overhead introduced by LANPO should be discussed further. In particular, the relevance calculation requires an LLM inference which would add significant overhead during training. The paper would be improved with precise numbers on step time increases introduced by each part of the pipeline. 

- The relevance filtering ablation studies the effect of no filtering vs. $\gamma = 0.9$ threshold with mixed results. AIME 24 and MATH and better without filtering whereas AIME25 and AMC are better with filtering. This indicates inconsistent utility of filtering.

### Questions
- How many prompts with feedback are truncated during training? Similarly how many intra example feedback rollouts are truncated?

### Soundness
2

### Presentation
3

### Contribution
1
