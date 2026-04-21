# SIKeD: Self-guided Iterative Knowledge Distillation for Mathematical Reasoning

- Avg Score: 5.20
- Decision: Reject
- Scores: 5, 3, 6, 6, 6

## Abstract
Large Language Models (LLMs) can transfer their reasoning skills to smaller models by teaching them to generate the intermediate reasoning process required to solve multistep reasoning tasks.  While LLMs can accurately solve reasoning tasks through a variety of strategies, even without fine-tuning, smaller models are not expressive enough to fit the LLMs distribution on all strategies when distilled and tend to prioritize one strategy over the others.
This reliance on one strategy poses a challenge for smaller models when attempting to solve reasoning tasks that may be difficult with their preferred strategy.
To address this, we propose a distillation method *SIKeD*: **S**elf-guided **I**terative **K**nowledge **D**istillation, where the LLM teaches the smaller model to approach a task using different strategies and the smaller model uses its self-generated on-policy outputs to choose the most suitable strategy for the given task. The training continues in a *self-guided* iterative manner, where for each training iteration, a decision is made on how to combine the LLM data with the self-generated outputs. Unlike traditional distillation methods, *SIKeD* allows the smaller model to learn *which* strategy is suitable for a given task while continuously learning to solve a task using different strategies.
Our experiments on various mathematical reasoning datasets show that *SIKeD* significantly outperforms traditional distillation techniques across smaller models of different sizes.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the challenge of knowledge transfer from large language models (LLMs) to smaller, more efficient models. The authors identify that smaller models typically demonstrate constrained capabilities and tend to favor single reasoning strategies. To address this limitation, the authors introduce SIKED, an approach that leverages self-generated outputs to create iteratively mixed training datasets. The method promotes the development of diverse reasoning strategies during the knowledge transfer process. The effectiveness of SIKED is demonstrated through empirical evaluations on several mathematical reasoning datasets, showing improvements over baseline approaches.

### Strengths
The proposed problem is worth exploring.
The paper is well-written and easy to follow.

### Weaknesses
1. Although the method demonstrates effectiveness, the motivation requires further clarification. While iteratively forming new training datasets based on model outputs is an existing approach, the paper's contribution lies in showing this method can enable diverse reasoning strategies in smaller models. However, the underlying mechanism for how this approach promotes strategy diversity needs clearer explanation.
2. The experimental evaluation is currently limited to mathematical reasoning tasks. Exploring the effectiveness of the proposed method in other scenarios would provide valuable insights into its generalization capabilities.

### Questions
1. Can the authors elaborate on the underlying reasons for smaller models' bias towards specific strategies? Does the preferred strategy vary?
2. The paper states that mixing LLM-generated data with self-generated outputs helps align smaller models with their learned knowledge (L83). However, discarding samples with mismatched outputs creates a different form of bias? Please clarify this.
3. Please explain the mechanism by which mixing the data promotes diverse strategy selection in smaller models.
4. How to judge if the generated r_i is correct? (L294)
5. Is it a typo in L211 & L215 that the same notation appears in different contexts?
6. Could the authors clarify the strategy sampling process for smaller models? Specifically regarding L413, "if both CoT and PoT are sampled correctly, our biased strategy choice is PoT" - is this strategy determined by the smaller model's output? Additional details on the strategy sampling mechanism would be appreciated and helpful

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes **S**elf-guided **I**terative **K**nowledge **D**istillation (SIKeD), which utilizes outputs of LLMs (teacher) and small models (student) to Iterative train the student model. The main process can be summarized as follows:

source data (GSM8K training set) → LLMs generate Cot, PoT and other format reasoning → training small models → small models augment data → filtering → training small models → small models augment data → filtering → … … n …

Source data: GSM8K training set (7,473 samples)

Model: Qwen2 0.5B, 1.5B, SmolLM 1.7B, Gemma 2B / 7B,

### Strengths
This paper carefully presents the relevant method analysis and explores how generation strategies evolve across multiple iterations in small models, which is a very intriguing phenomenon.

### Weaknesses
1. The mathematical notation is overly verbose. There is no proof or any other theoretical contributions. However, this paper employs overly complex notation. And there is also no revision on loss function. The overly complex notation will prevent quick understanding. The main process of proposed method can be summarized in one or two sentences. And all operations performed at the dataset level.

2. Limited generalization. The approach only enhances the GSM8K dataset, but reasoning tests should be conducted on more realistic datasets, such as MATH, Arc-Challenge and so on. And more reasoning tasks also need to be evaluated, such as commonsense reasoning and symbolic reasoning.

3. The absence of important references. The self-distillation in small models is already studied in [1,2,3].

### Questions
**Do self-distillation really work in small model continual training ?** 

Fig. 4 in your paper and Fig. 7 in [3] indicated  the performance decreased, when the n of iteration became large. From a theoretical perspective on synthetic data, the data variance decreases with multiple generations n [4]. To prevent variance reduction, i.e., to enhance data diversity, this paper incorporates data synthesized by LLMs throughout the iterative process. This operation is very important. From data perspective, authors should analyze the data distribution shifting across n. 

**Is there any new theoretical insights ?**

The authors observed shifts in generation strategies over iterations n. What causes this phenomenon? Additionally, as more data is generated, the overall dataset size increases.

------

[1] Ho N, Schmid L, Yun S Y. Large language models are reasoning teachers[J]. arXiv preprint arXiv:2212.10071, 2022.

[2] Fu Y, Peng H, Ou L, et al. Specializing smaller language models towards multi-step reasoning[C]//International Conference on Machine Learning. PMLR, 2023: 10421-10430.

[3] Zhu X, Qi B, Zhang K, et al. PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning[J]. arXiv preprint arXiv:2305.13888, 2023.

[4] Mobahi H, Farajtabar M, Bartlett P. Self-distillation amplifies regularization in hilbert space[J]. Advances in Neural Information Processing Systems, 2020, 33: 3351-3361.

[5] Dohmatob E, Feng Y, Yang P, et al. A tale of tails: Model collapse as a change of scaling laws[J]. arXiv preprint arXiv:2402.07043, 2024.

### Soundness
2

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
4

### Summary
It is hard for a small LLM to learn multiple correct trajectories towards the same ground truth answer, which might affect its reasoning capability, especially the generalization to OOD reasoning tasks. The paper has tested distilling multi-trajectory training data  for post-training smaller LLM from large LLMs and observed unsatisfactory results. To alleviate this problem, the paper proposes a mix of self-generated multi-trajectory training data together with the distilled data for SFT. Surprisingly it achieves better reasoning performance compared to existing methods, especially on OOD tasks.

### Strengths
1. intuitive and effective method

2. thorough experimental analysis

### Weaknesses
The only concern I  have on the experiment part is that only testing the small model’s preference on COT, POT and L2M is a bit constrained. I’m curious to see that among the three methods,  1)  training with the proposed method, 2) pure distilling  and 3) pure self-generating, which method can make the model generate the most diverse trajectories and whether the diversity is aligned with model’s performance on OOD tasks. Because in each strategy, for example in COT, a model can also generate multiple cot trajectories that lead to the correct answer. I’m curious to see which of the methods can improve the general diversity of the model’s output trajectories the most and whether this diversity is aligned with model’s OOD performance.

### Questions
Please see weakness

### Soundness
3

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
3

### Summary
This paper proposes SIKeD, a novel knowledge distillation approach for transferring multistep reasoning skills from large language models (LLMs) to smaller models, particularly for mathematical reasoning tasks. Traditional distillation methods struggle with strategy selection, often resulting in smaller models biased towards a single strategy. SIKeD addresses this by allowing the smaller model to iteratively learn and apply various strategies, combining LLM-generated data with its self-generated outputs to refine strategy selection. The method demonstrates improvements over single-strategy and combined-strategy distillations, achieving superior results across in-distribution and out-of-distribution mathematical reasoning datasets.

### Strengths
- The idea is novel. SIKeD leverages the idea of constructivist learning theory and uses an iterative self-guided approach for multi-strategy distillation, compared to previous single-step distillation.
- SIKeD shows generalization across in- and out-of-distribution mathematical reasoning datasets, demonstrating its effectiveness in diverse contexts.
- SIKeD is evaluated using various small model types, showing consistent improvements over the baselines across different model types.
- The paper is well-written and easy to follow.

### Weaknesses
- W1: The proposed method is only evaluated on mathematical reasoning tasks. It’s unclear how well SIKeD would generalize to other domains that require more nuanced strategy selection.
- W2: The paper lacks comparison with knowledge distillation methods.
- W3: The code is not available for reproduction.

### Questions
- Q1: Have the authors considered applying SIKeD to tasks outside of mathematical reasoning to test the generalizability of the strategy selection mechanism?
- Q2: Does SIKeD require the ground-truth answers of the training data? If not, how does it handle the tasks where the ground-truth answers are not available?
- Q3: Is there any knowledge distillation baseline that could be used to compare the performance of SIKeD on mathematical reasoning tasks? Current experiments only compare SIKeD against CoT, L2M, PoT and Combined.
- Q4: Can the authors explain why does the improvement on Qwen 1.5B model is less significant compared to the other base models? What are the potential reasons for this discrepancy?
- Q5: The small models are tuned with LoRA, what if the parameters of the small models are fully tuned? Would the performance of SIKeD be further improved?
- Q6: "The iterative training is stopped when accuracy shows only marginal improvements or declines." What specific criteria is used to determine the optimal number of iterations?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents SIKeD, a knowledge distillation approach to enhance smaller models with reasoning skills from Large Language Models (LLMs). SIKeD employs an iterative process that enables smaller models to learn multiple strategies and select the most appropriate for a given task, addressing the issue of strategy bias found in conventional distillation methods. The paper reports that SIKeD outperforms traditional techniques on mathematical reasoning tasks across various smaller model sizes.

### Strengths
1. The paper presents a unique approach to knowledge distillation by introducing the concept of self-guided iterative training. This method allows smaller models to dynamically adjust their strategy preferences, which is a creative solution to the challenge of strategy distribution mismatch in traditional distillation.

2. The experiments are well-designed and conducted across various mathematical reasoning datasets, providing a thorough evaluation of SIKeD's effectiveness. The improvements in performance metrics are substantial and clearly demonstrated.

3. The paper is well-organized, with a clear problem statement and a logical flow of ideas. The methodology is explained in detail, making it easy for readers to understand the proposed approach and its implications.

### Weaknesses
1. The paper's methodology, SIKeD, is contingent upon the quality of the initial LLM data. There is a need for further exploration on how fluctuations in LLM data quality might influence the distillation process and the performance of the resulting smaller models.

2. The study primarily focuses on mathematical reasoning tasks, with less clarity on the transferability of SIKeD to other reasoning domains such as commonsense or symbolic reasoning. Additional investigation into the broader applicability of SIKeD could be valuable.

### Questions
1. Can the authors comment on the potential of SIKeD to be effective in domains outside of mathematical reasoning? Have there been any preliminary experiments or considerations in this direction?

2. Could the authors elaborate on the computational efficiency of SIKeD, especially in terms of the number of iterations required for convergence and the resources needed for each iteration?

3. The paper discusses various smaller models, but does not discuss how the size of the smaller model affects the distillation process and the final performance. Are there any insights on how SIKeD scales with different model sizes?

### Soundness
3

### Presentation
3

### Contribution
3
