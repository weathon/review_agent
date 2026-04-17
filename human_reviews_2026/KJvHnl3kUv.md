# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
We aim to improve the reasoning capabilities of language models via reinforcement learning with verifiable rewards (RLVR). Recent RLVR post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RLVR alone to improve reasoning on inherently difficult tasks is less effective due to sparse rewards. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across diverse datasets and models demonstrate that E2H Reasoner substantially enhances LLM reasoning. Code is available at - https://github.com/divelab/E2H-Reasoning

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the E2H Reasoner, a reinforcement learning (RL) method for large language models (LLMs) inspired by curriculum learning. The method schedules tasks from easy to hard during training, aiming to accelerate LLM learning. Theoretical analysis shows that curriculum-based reinforcement learning (CRL) requires fewer total samples than directly training on the final task, and experimental results are generally positive.

### Strengths
1. The paper provides theoretical justification for why CRL can achieve sample efficiency, requiring fewer total samples than direct learning on the final task.
2. The experimental results are sound and well-presented.

### Weaknesses
1. The idea of using curriculum learning to improve RL efficiency is not novel. The paper acknowledged prior work—e.g., Chen et al., Foster et al., Bae et al., Zeng et al. which used curriculum learning ideas. The paper should also cite Yu et al. (DAPO: An Open-Source LLM Reinforcement Learning System at Scale). 
2. In the experimental results, E2H does not consistently outperform baselines such as GRPO or Self-Evolve. 
3. The paper does not clearly articulate the advantages of E2H over adaptive filtering methods such as DAPO or Self-Evolve. In fact, adaptive filtering—where pass rate determines sample filtering—has several appealing properties: 
1) Model-dependent difficulty: “Easy” and “hard” samples are relative to the specific model; what is easy for one model may be hard for another. Thus, classifying samples by difficulty a priori can be problematic. 
2) Lack of synchronization: Without adaptive scheduling, the scheduler and the model may become misaligned—for instance, the scheduler may advance to harder tasks before the model is ready.

### Questions
See weakness

### Soundness
3

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
4

### Summary
This paper proposes using a curriculum learning approach where they schedule tasks from Easy to Hard during RLVR which shows better performance at the end of training. They provide convergence guarantees for this algorithm in an approximate policy iteration framework and derive finite-sample complexity bounds which show the this is more sample efficient than training without any curriculum.

### Strengths
The paper proposes a simple method of using curriculum learning. The curriculum implicitly assumes some grouping of tasks, but they also  show that the grouping is not necessary because tasks can be clustered just using pass rates of the initial model. They also compare with different baselines and the empirical results seem sound.

### Weaknesses
The only weakness that comes to mind is not comparing with DAPO [1] which also has an implicit curriculum because the model keeps filtering prompts that are either too easy or too hard. Could the authors compare with DAPO as well and show results on the benchmarks? 

Also the paper doesn't cite Paprika [2] which also proposes a curriculum when tasks can be grouped. 


[1] DAPO: An Open-Source LLM Reinforcement Learning System at Scale (https://arxiv.org/abs/2503.14476)

[2] Training a Generally Curious Agent (https://arxiv.org/abs/2502.17543)

### Questions
Please look at the weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes E2H Reasoner, a curriculum reinforcement learning (CRL) approach for enhancing LLM reasoning capabilities. It decomposes complex tasks into easier subtasks, uses probabilistic schedulers (cosine and Gaussian) to gradually shift from easy to hard tasks during RL post-training, and provides empirical improvements on benchmarks like Blocksworld, Countdown, and arithmetic tasks, achieving SOTA results. Theoretically, it analyzes CRL via approximate policy iteration, proving convergence and reduced sample complexity compared to direct learning.

### Strengths
1. The method creatively combines task decomposition with probabilistic scheduling in CRL, addressing rollout inefficiencies in difficult reasoning tasks by building skills incrementally, which makes intuitive sense and extends prior RL post-training like DeepSeek-R1.

2. Theoretical analysis provides finite-sample bounds and convergence guarantees, grounding the approach in approximate policy iteration.

3. Well-structured presentation with illustrative figures (e.g., task decomposition in Fig. 2, schedulers in Figs. 3-4) and precise definitions of reasoning as generalization; methods and experiments are logically sequenced.

### Weaknesses
1. Risk of Overfitting in Task Decomposition: Decomposing hard tasks into varying difficulty levels may cause repeated exposure to similar knowledge patterns across subtasks, increasing overfitting risks, especially if subtasks overlap significantly without explicit regularization.


2. Lack of Implementation Details for Reproducibility: Key details are missing, such as prompts used for automatic difficulty estimation (e.g., in AQuA/GSM8K) or exact hyperparameters for task grouping, raising concerns about replicating the reported effects.


3. Limited Scope of Experiments: Evaluations focus on relatively lower-difficulty tasks like Blocksworld and arithmetic benchmarks; lacks experiments on highly challenging ones like AIME, LCB, or agent-based tasks, limiting evidence for broader applicability.

### Questions
1. Advantages of Task Decomposition Over Traditional Curriculum Learning: What specific advantages does your task decomposition offer compared to standard curriculum learning (e.g., fixed-stage switching)? Is there a theoretical comparison on generalization, perhaps extending your API framework?


2. Inconsistencies in Model Trends in Figure 1(a): In Figure 1(a) for Countdown, why do Qwen 1.5B and LLaMA 3.2 3B show inconsistent relative performance trends under E2H vs. base models (e.g., one benefits more at low k)? Could this relate to architectural differences or training artifacts?

### Soundness
3

### Presentation
3

### Contribution
3
