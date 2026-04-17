# RewardAnything: Generalizable Principle-Following Reward Models

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Reward Models, essential for guiding Large Language Model optimization, are typically trained on fixed preference datasets, resulting in rigid alignment to single, implicit preference distributions. This prevents adaptation to diverse real-world needs-from conciseness in one task to detailed explanations in another. The standard practice of collecting task-specific preference data and retraining reward models is resource-intensive, often producing biased rewards, and limits practical application. We introduce generalizable, principle-following reward models. We propose that RMs should understand and adhere to dynamically provided natural language specifications of reward principles, similar to instruction-following in LLMs. To measure this capability, we develop RABench, a comprehensive benchmark for RMs focusing on generalization across diverse principles. Evaluations on RABench reveal poor generalization of current RMs. As a solution, we present RewardAnything, a novel RM designed and trained to explicitly follow natural language principles. We achieve SotA performance with RewardAnything in traditional RM benchmark simply by specifying a well-defined principle, and results on RABench show we excel in adapting to novel principles without retraining. Furthermore, RewardAnything integrates seamlessly with existing RLHF methods and we show by a case study on how to automatically and efficiently align LLMs with only natural language principles.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces REWARDANYTHING, a reward modeling framework that enables language models to follow arbitrary, natural language-specified principles at inference time. It functions as a language model judge trained on preference data using Group Relative Preference Optimization (GRPO). To evaluate this capability, the authors propose RABENCH, a new benchmark designed to assess how well reward models generalize to diverse and previously unseen evaluation principles. Their results show that existing reward models struggle with such generalization, whereas REWARDANYTHING performs strongly, often matching or surpassing the performance of models like GPT-4.1 on both traditional benchmarks and tasks requiring principle generalization.

### Strengths
1. The paper addresses an important challenge in reward modeling: enabling flexible alignment of language models through natural language principles rather than static preference datasets.
2. The paper is mostly well-written and organized. Key methods like GRPO and the listwise reward structure are clearly explained. The paper is technically competent, and the proposed method is well-implemented.
3. The proposed dataset and benchmark are open source, which could be useful tools for future research in this space.

### Weaknesses
1. The paper lacks in-depth analysis regarding whether the model’s outputs, when conditioned on natural language principles, truly align with those principles. Section 4.2 provides only a very high-level description of how the quality of the preference data is ensured, offering no details on how each component of the dataset, such as the principles and the chosen/rejected responses, is evaluated. There is also no information about the annotator recruitment process or qualification criteria.
2. The paper does not clearly explain how the principles were curated. Since the dataset is a critical component of this work, a more thorough analysis of its quality is necessary. I only found a vague statement on line 163: “We manually curated 200 distinct principles,” along with some generic analysis in Appendix A. However, there is no clear explanation of how the principles were curated. Did you write the principles yourself, or were they sourced from existing datasets?
3. The idea of using LLMs as generative reward models conditioned on natural language and producing structured reasoning has already been explored in prior work [1, 2, 4]. Among these, [1] appears to be the most similar to the proposed approach. Based on my reading, the primary methodological difference is the use of GRPO for training the reward model in this work, where as [1] used pairwise reward model loss. Although [1] is listed in the references, it is neither discussed nor even mentioned in the main text, which is a significant omission. Please correct me if I’m mistaken. Furthermore, similar methods are already widely used in alignment research. For example, [3] aligns models using what they term “specs,” which are effectively equivalent to the natural language principles used here. Likewise, [5] introduces “checklist-guided evaluation” for alignment, where the checklists are derived from in-situ user interactions and feedback. The distinctions claimed in this work, such as GRPO training or explicit principle-following, appear superficial without a direct comparison to these existing approaches.

---
Related Work

[1] Improving Context-Aware Preference Modeling for Language Models, NeurIPS 2024, https://openreview.net/forum?id=52r4XJYzjg

[2] CARMO: Dynamic Criteria Generation for Context-Aware Reward Modelling, ACL 2025, https://arxiv.org/abs/2410.21545

[3] Deliberative Alignment: Reasoning Enables Safer Language Models, OpenAI 2024, https://arxiv.org/abs/2412.16339

[4] Generative Reward Models, Arxiv 2024, https://arxiv.org/abs/2410.12832

[5] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback, Arxiv 2024, https://arxiv.org/abs/2408.15549

### Questions
1. How each component of the dataset, such as the principles and the chosen/rejected responses, is evaluated?
2. How the principles were curated or how they influenced the generation of preference data and the training process?
3. How does your work differ from
- Improving Context-Aware Preference Modeling for Language Models, NeurIPS 2024, https://openreview.net/forum?id=52r4XJYzjg
- Deliberative Alignment: Reasoning Enables Safer Language Models, OpenAI 2024, https://arxiv.org/abs/2412.16339?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses a limitation in traditional Reward Modeling for RLHF: the rigidity of RMs trained on static preference datasets. Current RMs learn implicit preferences and struggle to adapt to new criteria without costly retraining. The authors propose a paradigm shift towards "principle-following reward models," which can dynamically adjust their evaluation criteria based on natural language instructions

### Strengths
Timely and Important Problem: The paper tackles a highly relevant problem. The inflexibility and bias of current RMs are major bottlenecks in deploying aligned AI systems. 

Efficiency and Interpretability: The listwise ranking approach of REWARDANYTHING is computationally efficient compared to pairwise comparisons, requiring only a single inference call to rank multiple candidates. Furthermore, the model's generation of explicit reasoning steps enhances the interpretability of the reward signal.

### Weaknesses
Unfair Comparison Methodology on RM-Bench: The claimed SOTA performance on RM-Bench (Section 6.1, Table 2) appears to rely on a flawed comparison methodology. The authors state they provided REWARDANYTHING with a clear principle prioritizing accuracy, and that "this principle is also passed to other RMs as system prompt" (L355-356). However, the majority of the baselines, particularly the Discriminative Reward Models (e.g., Skywork-Reward, FsfairX, Nemotron), are fine-tuned sequence classifiers or regression models; they are not architecturally designed to interpret system prompts or dynamically adjust their learned preferences based on inference-time instructions. By giving REWARDANYTHING explicit instructions on the evaluation criteria, while the baselines are forced to rely on their implicit learned preferences, the comparison is heavily biased. This undermines the SOTA claim.

Complexity and Stability of GRPO Training: The proposed training method (GRPL) uses GRPO, an online RL algorithm, with a complex, multi-component reward function (Equations 2-4) involving numerous hyperparameters. This is likely significantly more complex and potentially less stable than standard SFT or offline methods like DPO. While the ablation (Table 4) shows GRPO outperforms SFT, the paper lacks analysis of the training costs, stability, and hyperparameter sensitivity of GRPO in this context. The necessity of this complexity over potentially adapting DPO for principle-following is not established.

The Burden of "Principle Engineering": The analysis in Section 6.1 (Figure 3) highlights that performance drops significantly when principles are vague or when multiple objectives lack clear prioritization. This suggests that the success of the method heavily depends on the user's ability to craft high-quality, structured principles. This potentially shifts the challenge from "data collection" to "principle engineering," which presents its own difficulties, and may limit the claimed generalization if the model fails on ambiguous real-world principles.

### Questions
In addition to the weakness section, I have the following questions:

How exactly were the discriminative RMs prompted with the principle in the RM-Bench evaluation? Given their architecture, what evidence do you have that they are capable of utilizing this input? How would REWARDANYTHING perform if it were not provided with an explicit principle tailored to the benchmark?

Q2. Regarding W2: Given that the training data relies solely on LLM consensus without human oversight, how do you ensure that REWARDANYTHING is learning generalizable principle-following capabilities rather than overfitting to the specific patterns and biases of the LLM judges (e.g., GPT-4.1) used for data generation?

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
The paper identifies two major flaws in current reward models: (1) limited generalization and adaptability to new tasks, and (2) the lack of explicit bias control and interpretability due to reliance on implicit preference learning. To address these limitations, the authors propose a principle-following reward model that extends beyond fixed reward criteria. Analogous to instruction-following large language models, this framework enables the reward model to interpret and generalize over arbitrary principles (P) — which may represent task specifications, ethical guidelines, or combinations of multiple criteria with defined priorities. The model is expected to handle unseen principles at varying levels of specificity and abstraction.

To systematically evaluate this capability, the authors introduce RABENCH, a comprehensive benchmark designed to test reward models on their ability to adapt evaluation criteria according to explicitly provided natural-language principles.

Building on this, they develop Reward-Anything, a reward model trained via on-policy reinforcement learning. The objective is to encourage the model to produce correctly ranked lists consistent with the guiding principles, thereby promoting both adaptability and principled generalization.

### Strengths
The paper presents a clear and coherent narrative — progressing logically from the motivation of improving reward model generalization to the development of RABENCH and the proposed principle-following reward model. The methodology is well-structured and easy to follow. The authors further strengthen their contribution by conducting on-policy training with GRPO, featuring an intuitive reward design. Experimental results demonstrate promising generalization of the learned reward model when applied to diverse downstream tasks.

### Weaknesses
Didn't report performance on other popular benchmark like RewardBench to show generalization, and some other popular work uses reward model for test time compute (verifier for best of N), reporting numbers on these would make the generalization statement more robust.

### Questions
1. Are there any existing reward models trained via reinforcement learning (RL) that the authors compare their approach against? It would be helpful to understand how the proposed method performs relative to prior RL-based reward models in terms of generalization, adaptability, and bias control.

2. Could you report the REWARDANYTHING on REWARD bench?

3. How does instruction following capability correlates with the principle adherence capability, if we start with a IF tuned model (verifiy-if) would that results in even better performance? 

4. How could we combine this with rubrics based rewards ideas? The model seemed to follow a set of principles which are analogous to rubrics (checks) which gains popularity recently for evaluating the response quality.

### Soundness
2

### Presentation
2

### Contribution
2
