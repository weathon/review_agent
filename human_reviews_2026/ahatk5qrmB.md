# BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel framework designed to empower LLMs with budget-aware reasoning. The method can provide precise control over the length of their thought processes. The main idea is to inserts special control tokens to inform the model of its remaining token budget. They conduct two training phases, first SFT then RL.

### Strengths
1. The paper focuses on an interesting question. 
2. The paper is well-written and easy to follow.
3. The experiments show reasonable improvements.

### Weaknesses
1. The length setting appears to be manually selected. As shown in Table 1, the experiments were even evaluated under different length budgets for different dataset. However, users would not know the optimal length for each dataset in practice. Therefore, this setup seems somewhat unrealistic.
2. The paper mainly focuses on controlling the maximum output length. However, for relatively simple questions, forcing the model to produce unnecessarily long responses is neither efficient nor meaningful. It would be more realistic to allow adaptive length generation based on task difficulty.
3. As described in the paper, the proposed method appears to require huge computational resources for training. The SFT data generation process itself seems costly, and the full training pipeline involves both an SFT stage and two phases of RL. This raises concerns about the overall training efficiency and practicality of the approach.
4. I think some baselines are missing. Such as [1,2,3] all focus on reasoning with a length budget.
[1] Aggarwal, Pranjal, and Sean Welleck. "L1: Controlling how long a reasoning model thinks with reinforcement learning." arXiv preprint arXiv:2503.04697 (2025).
[2] Qi, Penghui, et al. "Optimizing anytime reasoning via budget relative policy optimization." arXiv preprint arXiv:2505.13438 (2025).
[3] Zhang, Xuechen, et al. "Making small language models efficient reasoners: Intervention, supervision, reinforcement." arXiv preprint arXiv:2505.07961 (2025).
[4] Li, Zhong-Zhi, et al. "TL; DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression." arXiv preprint arXiv:2506.02678 (2025).

### Questions
1. How to choose the number of special control tokens K. 
2. What would happen if the method were applied to a model that does not naturally produce overly long reasoning traces, and instead aims to explore longer reasoning budgets? Would the length constraint still provide benefits, or would it hinder the model’s ability to develop deeper reasoning skills?
3. For SFT, is it important to control the length distribution? Will it also help if the SFT data length distribution is very different from the length budget?
4. In the curriculum stage, the paper adopts a long-to-short training schedule. Moreover, how would the method adapt if the model’s current average generation length is already shorter than the budget?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces BudgetThinker, a framework designed to give Large Language Models (LLMs) and Multimodal LLMs (MLLMs) precise control over the length of their Chain-of-Thought (CoT) reasoning processes. The core innovation is the periodic insertion of special control tokens during inference, which continuously inform the model of its remaining token budget. This is supported by a comprehensive two-stage training pipeline: 1) Supervised Fine-Tuning (SFT) on a curated dataset that includes these control tokens and spans a wide range of reasoning lengths, and 2) Reinforcement Learning (RL) using a curriculum and a length-aware reward function to optimize for both accuracy and strict budget adherence. Experiments on mathematical and multimodal reasoning benchmarks demonstrate that BudgetThinker achieves a better trade-off, maintaining high accuracy while precisely following specified token budgets, outperforming several baselines.

### Strengths
* The use of the periodic special tokens to remind the LLM its remaining budget is somewhat novel, and incorporating it into SFT and RL is reasonable.
* Experiments cover both LLM and VLM.

### Weaknesses
* The paper shows that the model adheres to the budget, but offers less insight into how the reasoning strategy changes. A qualitative analysis comparing the CoT traces at different budgets (e.g., does it skip certain reasoning steps? does it use more abbreviations?) would be very valuable.
* The method is a bit complex and costly due to the SFT stage.

### Questions
* When conducting GRPO, we assume the data is sampled from \pi_old and we compute the token ratio using \pi_\theta and \pi_\old. In this method, since you insert special tokens while sampling, you essentially change the distribution of \pi_old. How did you deal with the GRPO loss computation?

* I recently came across this paper Optimizing Anytime Reasoning via Budget Relative Policy Optimization (https://openreview.net/pdf/8136e4668a09f8c47a2454d9e72728d4fdea055e.pdf), which shares very similar motivation (their paragraph 2 v.s. your paragraph 2) and method (they truncate at some sampled budget while you insert at the budgeted locations). It would be nice if you can discuss your differences and how your method is fundamentally better than the baseline.

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
2

### Summary
This paper presents BudgetThinker, a framework that enables LLMs to perform budget-aware reasoning by periodically inserting special control tokens during inference. The approach consists of two key components: (1) a ratio-based control token insertion strategy using a fixed set of K=8 tokens to signal remaining budget fractions, and (2) a comprehensive two-stage training pipeline combining supervised fine-tuning with curriculum-based reinforcement learning that incorporates a length-aware reward function. The authors demonstrate improvements over strong baselines (ThinkPrune, Laser) on mathematical benchmarks (MATH-500, AMC 2023) and multimodal reasoning tasks (EQRA, SAT), showing an average accuracy improvement of 4.9% across various budgets while maintaining better budget adherence.

### Strengths
Originality:
- The ratio-based control token design is elegant and scalable, requiring only K fixed tokens regardless of budget size, which addresses a practical limitation of fixed-interval approaches
- The training methodology is thoughtfully designed, with a well-motivated balance of three data types (long reasoning, iteratively compressed, non-thinking) and a sensible curriculum learning strategy
- The plug-and-play nature of the framework makes it potentially valuable as a complementary module for existing test-time scaling methods

Quality:
- The experimental evaluation spans multiple model scales (1.5B, 7B), modalities (text-only and vision-language), and task types, demonstrating breadth
- The ablation studies systematically examine key design choices (control token strategies, iterative training, RL contribution), providing insight into what drives performance
- Figure 2 effectively demonstrates the practical advantages in both budget following and budget utilization compared to baselines

Clarity:
- The motivation is well-articulated, clearly establishing the need for precise budget control in latency-critical applications
- The method description flows logically from token insertion mechanics through data curation to RL training
- The visualizations, particularly Figure 1, help readers quickly grasp the core idea

Significance:
- The work addresses a genuine practical need for deploying reasoning models in resource-constrained settings
- The evaluation on both mathematical and multimodal reasoning demonstrates applicability beyond a single domain
- The improved budget adherence could be particularly valuable for production systems with hard latency requirements

### Weaknesses
- The task coverage is somewhat focused on mathematical reasoning. I'm interested in how the approach might generalize to other important applications like code generation, open-domain QA, or multi-step agentic tasks
- For the 7B experiments, it would be helpful to include ThinkPrune comparisons if feasible
- The budget range tested (50-10K tokens) is reasonable, though I wonder about behavior in more extreme regimes
Beyond Pass@1 accuracy, it might be illuminating to examine additional dimensions like reasoning quality, consistency, or calibration

### Questions
1. Task generalization: I'd be interested to know how the approach performs on:
- Code generation (where completeness vs. brevity tradeoffs differ)
- Open-ended QA (without clear correct answers)
- Multi-step agent tasks (requiring planning + execution)

2. Failure mode characterization:
- Which problem types are most challenging for budget control? Could you provide quantitative analysis?
- Figure 2 shows lower budget following ratios at small budgets for BudgetThinker—what drives this behavior?

3. Alignment with human preferences:
- Have you conducted human evaluations? Do users always prefer shorter reasoning?
- In some contexts, detailed reasoning might be valuable for trust and debugging—how does the framework accommodate this?

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
The paper proposes budget thinking, a combination of test time control token insertion and RL training reward that teaches the model to make use of its test time budget. Across the board it seems that budget thinking is better at trading off compute vs. performance and controls token generation at run time.

### Strengths
The paper is interesting in introducing a combo of test time behavior and RL training reward to better elicit token control behavior for length-controlled generation. The method sounds novel and has rather solid tech contributions.

### Weaknesses
One weakness is that comparison against baseline seems relatively weak in that there is no alternative RL training method that gets compared, and it will be useful to have extra ablations to better understand the mechanism of the improvement.

### Questions
=== *budget think with no training* ====

I think it's useful to compare the current method vs. a baseline that just inserts the control token at test time - without training, would the model be able to already in-context take the hint and follow instructions better? This comparison will highlight whether there is a genuine gain in training model against such behavior vs. just in-context hint.

=== *baseline comparison in Fig 2* ====

I think the flagship comparison in Fig 2 shows improvements from budget thinker but it seems that the pruning think baseline does not make use of extra RL training? So this is a comparison between test time algorithm vs. test + train time algorithm. If the proposed method is quite novel and I understand there is no similar training in the literature yet, maybe we can ablate various ways that the reward function is set up to better understand the performance gains. 

One idea is to train the system using typical reward but with insertion of the control token - this will essentially not penalize the model at training time when it violates the token budget. This comparison will show whether penalizing the token budget violation at RL time is necessary, or it suffices to just prompt the model + let it in-context learn. My hunch is that if we train long time without penalizing the violation, the model will learn to ignore such control tokens and the violation rate goes up over time, but it's good to verify if possible.

==== *SFT vs RL* ====

Fig 3 shows limited gain of using RL vs. SFT in combination with budget thinking, is that right? Does that mean that budget thinking constrains the RL performance vs. SFT, how about we just do regular RL vs. SFT and we should typically expect much larger gains from RL?

### Soundness
2

### Presentation
2

### Contribution
2
