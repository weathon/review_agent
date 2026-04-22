# Can LLMs Serve as Causal Inference Agents? A Study on Post-Training Methods

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Causal inference is essential for decision-making but remains challenging for non-experts. While large language models (LLMs) show promise in this domain, their precise causal estimation capabilities are still limited, and the impact of post-training on these abilities is insufficiently explored. This paper examines the extent to which post-training can enhance LLMs’ capacity for causal inference.
We introduce CausaGym, a comprehensive dataset comprising seven core causal tasks for training and five diverse test sets. Using this dataset, five post-training approaches—SFT, DPO, KTO, PPO, and GRPO are systematically evaluated.
Across five in-domain and four existing benchmarks, our experiments demonstrate that appropriate post-training enables smaller LLMs to perform causal inference competitively, often surpassing much larger models. Our 14B-parameter model achieves 93.5% accuracy on the CaLM benchmark, compared to 55.4% by OpenAI o3. Furthermore, the post-trained agents exhibit strong generalization and robustness under real-world conditions such as distribution shifts and noisy data. Collectively, these findings provide the first systematic evidence that targeted post-training can produce reliable and robust LLM-based causal inference agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author introduced the DeepCausa dataset (with variants) for training large language models in causal reasoning and evaluated the effectiveness of popular post-training methods in enabling models to perform this task.

### Strengths
1. The work explores the suitability of using LLMs for certain causal tasks, representing a non-trivial effort that provides valuable insights and lays the groundwork for further research.
2. The experiments are comprehensive and well-documented, offering detailed analyses across multiple aspects, including generalization, internalization, and robustness of the models.

### Weaknesses
1. The causal tasks and datasets used in this paper assume that the required statistics (e.g., probabilities for do-operators) are explicitly provided in the questions, meaning the LLM’s role is largely retrieving numbers and performing simple arithmetic. This assumption limits the applicability of the framework to more complex causal reasoning tasks.
2. In the motivation example in Figure 1, only two probabilities are provided: P(B=1|A=0) and P(B=1|A=1). Probabilities like P(B=0|A=1) or P(B=0|A=0) are omitted, which seems sufficient for computation. If all necessary information is provided in this way, the problem could be solved heuristically without requiring an LLM, raising questions about the necessity of the model.
3. One of the main contributions of this work is the training dataset. However, without a clearer understanding of its quality (such as question diversity and entity variety), its usefulness for further research is limited. Providing supplementary material with detailed dataset statistics would strengthen the contribution.

### Questions
N/A

### Soundness
3

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
5

### Summary
This paper studies the influence of post-training techniques on the LLMs' causal inference abilities. Authors introduce the DeepCausa benchmark, a comprehensive dataset that contains seven core training causal tasks and five test tasks. In the experiments, the authors evaluate five post-training methods, including SFT, DPO, KTO, PPO, and GRPO. Finally, authors conclude that post-training approaches can enhance LLMs' causal inference abilities.

### Strengths
1. The studied problem is important and practical. Both post-training methods and causal reasoning are essential for LLMs.

2. The results analyses are comprehensive, and the authors analyze diverse aspects of agents, including the generalization, internalization, and robustness.

### Weaknesses
1. The proposed benchmark and source codes are not open-sourced.

2. I think the motivation may be a little contradictory. Specifically, in the introduction, the authors claim that "we can develop a causal
inference agent that explains its assumptions and reasoning in plain language." However, in Figure 1, it seems that LLMs are still mainly doing the numerical calculation rather than a detailed language explanation. Besides, the term "backdoor adjustment set" may still be hard to understand for non-experts in causal inference. 

3. Since there already exist other formal causal reasoning benchmarks (e.g., the CLADDER [1]), I would suggest that authors add an individual section on the differences between their newly proposed DeepCausa and existing benchmarks. Why can‘t other benchmarks test the abilities of post-training methods?

4. The related work section is too short. I think there should be at least two separate parts: post-training methods for LLMs, and LLMs' causal inference abilities. Authors should consider revising the related work.

5. I think Table 1 is unclear and could be misleading. Authors should consider listing other models, post-training methods, and base model (DeepSeek-R1-Distill-Qwen-14B) with separate lines. Currently, it's hard for readers to tell which one is the baseline simply from the table.

6. I believe only one base model (DeepSeek-R1-Distill-Qwen-14B) is not enough. Authors should consider including more base models to verify the generality of their findings.

> [1] Jin Z, Chen Y, Leeb F, et al. Cladder: Assessing causal reasoning in language models[J]. Advances in Neural Information Processing Systems, 2023, 36: 31038-31065.

### Questions
Please refer to the weaknesses part.

### Soundness
3

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
This paper introduces **DeepCausa**, a new benchmark and dataset for evaluating and training LLMs on causal inference tasks. It formulates seven core causal estimation problems (ATE, CDE, ETT, NDE, NIE, PN, PS) as natural-language reasoning questions, and enables reinforcement learning (especially GRPO) by providing programmatically computable rewards. Experiments show that a 14B model trained with GRPO reaches ~93% accuracy on the benchmark, surpassing larger models, while maintaining performance on math reasoning tasks.

### Strengths
1. Introduces the first causal dataset that supports RL-based training with automatic rewards.
2. Provides quantitative evidence that GRPO can improve causal reasoning accuracy without degrading general reasoning (math).

### Weaknesses
1. The “agent” claim is overstated; the model remains a passive CoT generator without environment interaction or intervention ability.
2. Evaluation is confined to the same synthetic distribution used for training; no results on external causal benchmarks (CLadder, CLEAR, CaLM) are reported.
3. No ablation on reward shaping or robustness to real data.

### Questions
1. How well would the trained model transfer to unseen causal benchmarks such as CaLM or CLadder?
2. Could the benchmark be extended to allow environment-level interaction (e.g., tool-based causal discovery)?
3. Does the model’s performance degrade when SCM variables have realistic semantics rather than random symbols?

### Soundness
2

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
This paper investigates whether post-training methods can enhance the causal inference capabilities of large language models (LLMs). The authors introduce DeepCausa, a comprehensive benchmark comprising seven causal tasks for training and five test sets for evaluation. They systematically compare five post-training methods—SFT, DPO, KTO, PPO, and GRPO—and demonstrate that online RL methods, particularly GRPO, significantly improve the causal reasoning abilities of smaller LLMs, enabling them to outperform larger baseline models. The study also evaluates generalization, internalization, and robustness under distribution shift and noisy data conditions.

### Strengths
1. The experimental design is thorough, covering multiple causal tasks, training methods, and evaluation dimensions.

2. The introduction of the DeepCausa benchmark is a valuable contribution, providing a structured dataset for training and evaluating causal inference agents.

3. The analysis is comprehensive, with clear comparisons across methods and detailed ablation studies.

### Weaknesses
1. The abstract could be more concise and formal. For instance, the phrase “To this end, this paper investigates whether post-training can turn LLMs into effective causal inference agents” could be rephrased to better align with academic tone.

2. The introduction repeatedly uses “we” and could be structured more objectively. For example, “We then systematically evaluate…” could be replaced with a more formal passive or impersonal construction.

3. The paper lacks a discussion on the practical utility of using LLMs for formal causal reasoning, especially given the existence of specialized causal inference tools (e.g., DoWhy, CausalML). The authors should clarify the real-world scenarios where an LLM-based agent would be preferable.

4. While the DeepCausa dataset is introduced, its naming and branding could be more distinctive (e.g., “CausalAgent-Bench” or similar) to enhance recognition and reuse.

5. There is no direct comparison with existing causal reasoning benchmarks (e.g., CLADDER, CLEAR). A dedicated section explaining how DeepCausa differs and why it is better suited for evaluating post-training methods would strengthen the contribution.

6. The figures (e.g., Figure 1) use light colors and small text, making them difficult to read. The samples are overly detailed; a more abstract and summarized visualization would improve clarity and impact.

### Questions
1. How does DeepCausa compare to existing causal reasoning benchmarks in terms of task coverage and difficulty?

2. Can the authors provide more insight into why online RL methods (especially GRPO) outperform offline methods so significantly?

3. What are the limitations of using synthetic SCMs for training, and how might this affect real-world applicability?

### Soundness
3

### Presentation
2

### Contribution
2
