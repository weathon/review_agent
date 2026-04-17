# J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
The progress of AI is bottlenecked by the quality of evaluation, making powerful LLM-as-a-Judge models a core solution. The efficacy of these judges depends on their chain-of-thought reasoning, creating a critical need for methods that can effectively optimize this reasoning process. In this work, we introduce J1, a reinforcement learning framework for teaching LLM judges to think before making decisions. Our core contribution lies in converting all judgment tasks for nonverifiable and verifiable prompts into a unified format with verifiable rewards, enabling direct optimization of evaluation quality while mitigating positional bias. We then use RL to train thinking-judges at scales of 8B, 32B, and 70B and show that they obtain state-of-the-art performance across multiple benchmarks. In particular, J1-Qwen-32B, our multitasked pointwise and pairwise judge also outperforms o1-mini, o3, and a much larger 671B DeepSeek-R1 on some benchmarks, while only training on synthetic data. Through comprehensive ablations of pairwise, pointwise, and multitask J1 variants, we demonstrate the effectiveness of our approach across seed prompts, reward strategies, and training recipes. Qualitative analysis reveals that J1 develops systematic evaluation strategies, including dynamic criteria generation, reference answer creation, iterative self-correction of initial assessments, and feedback generation for low-quality responses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an RL-based method for teaching an LLM-as-a-Judge to think before delivering a verdict/score. The authors investigate and compare various pairwise and pointwise strategies, and conduct meaningful analysis and ablations over these settings

### Strengths
1. The writing is clear and easy to follow.
2. The evaluation is comprehensive, spanning many relevant datasets and baselines. 
3. The approach, particularly the formulation in Section 3.2, is clear and well-motivated. Many meaningful ablations over the approach are provided.

### Weaknesses
1. There is some disconnect between Sections 2.3 and Section 4.1. For example, it's not clear which formulations from Section 2.3 were used to train the models evaluated in Section 4.1. Section 3 could be used to provide more information about the trained models. 
2. The role of the consistency reward is unclear. From Figure 5, it appears to slightly underperform the other pairwise settings, but a more thorough analysis (e.g., as a setting in Table 4) is needed.
3. In real-world settings, automatic evaluations often span multiple dimensions (e.g., helpfulness, factuality, safety). How could the J1 framework be adapted for this multi-task setting?
4. A couple of statements could use some qualification. For example, in L.180, the authors claim that "pointwise judges are inherently consistent", which is true in reference to positional consistency (because pointwise judges have no notion of position), but more broadly incorrect (e.g., sampling from the same pointwise judge is not guaranteed to produce consistent results). Similarly, in L.311, it states that J1-Qwen-32B-MultiTask outperforms "all previous methods" on RewardBench, but this is only true against other generative approaches; a number of classified approaches outscore J1-Qwen-32B-MultiTask [1].

[1] https://huggingface.co/spaces/allenai/reward-bench

### Questions
See weaknesses. Also,
1. [2] is recent work that trains a pointwise judge with RL, but they observe that response length explodes to over 10k tokens during training (similar to DeepSeek-R1-Zero [3]), whereas you observe that it initially falls and settles around 500 tokens. Why is this? Would performance be better if the models reasoned for longer before providing a score/verdict?
2. In L.262, the authors state the results are statistically significant "(p<0.0001)". How was this statistical testing/resampling done?

[1] https://huggingface.co/spaces/allenai/reward-bench

[2] https://arxiv.org/pdf/2504.10337

[3] https://arxiv.org/pdf/2501.12948

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes J1, a reinforcement-learning (RL) framework to train “LLM-as-a-Judge” models to think before deciding. The key idea is to convert both verifiable (e.g., math) and typically non-verifiable judging tasks into a unified, verifiable formulation; then optimize the thinking traces and final decisions using GRPO, with rewards for verdict correctness and order-consistency to mitigate position bias. J1 is instantiated on Llama-3.x and Qwen3 (8B/32B/70B) and evaluated on PPE, RewardBench, JudgeBench, RM-Bench, and FollowBenchEval, reporting SOTA or competitive results—especially for J1-Qwen-32B-Multitask—despite training on ~22k synthetic preference pairs. The paper includes ablations (pairwise vs pointwise vs multitask; reward variants; seed prompts) and qualitative analyses of learned evaluation strategies.

### Strengths
- Unified, verifiable training recipe across “verifiable” and “non-verifiable” prompts enabling direct optimization via RL, not just DPO/SFT.
- Bias mitigation at training time via batching both orderings and an order-consistency reward; clean and effective.
- SOTA or competitive results on multiple judging benchmarks at reasonable model sizes, with notably strong PPE Correctness and RewardBench numbers; comprehensive comparison tables.
- Thoughtful ablations (rewards, prompts; pairwise vs pointwise vs multitask) and qualitative traces illustrating learned strategies (criteria, reference answers, self-correction).

### Weaknesses
- “Non-verifiable” to verifiable conversion is underspecified. For subjective tasks, the paper leans on synthetic construction and pairwise labels; the validity of these labels as ground truth for reward is not rigorously validated with humans.
- Limited causal analysis of bias reduction. While order-consistency improves, the paper doesn’t isolate the effect of batched dual-ordering vs consistency reward vs prompt phrasing with confidence intervals on all benchmarks.
- A minor detail is that while prompts and some details are provided, RL stability (seed variance, KL control, learning rate sweeps, early stopping criteria) could be reported more completely to ensure replicability of gains.

### Questions
- How sensitive are results to adding graded thought-quality rewards (e.g., rubric adherence, factual checks) vs the current binary signals?
- For “non-verifiable” tasks, what fraction of pairwise labels are ambiguous/tied under careful human review?
- How does J1 fare on unseen domains/languages and with adversarial perturbations (e.g., equal-quality or equal-error pairs) that should produce ties?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work uses RL to train thinking LLM judges and show that it achieves SOTA performance on multiple benchmarks. Here are the contributions:
1. a recipe that converts different judgement tasks into a unified format with verifiable reward to train with RL
2. RL training on the thinking judges that shows strong performance improvement across various benchmarks
3. Develop methods to address positional bias through consistency reward.

### Strengths
1. Strong Empirical Results: The model trained using RL showed strong and consistent improvement across benchmarks, and is able to match frontier model (e.g. o3-mini, Deepseek-R1) that is an order of magnitude bigger.
2. Comprehensive Ablation & Analysis: The author provides thorough analysis on things like positional bias in the ablation study, which helps better understand the behavior of the model, and show that through the consistency reward, the "Verdict Flip/Ties" rate decreases.

### Weaknesses
1. Training Complexity: while J1 shows stronger performance than EvalPlanner, which uses offline DPO training, it is not a fully apple-to-apple comparisons. It is unclear how these two methods (GRPO vs DPO) differ under the same compute budget.

### Questions
1. It seems like for Qwen32B, there is a much lower improvement (e.g. +3.5 overall, +0.7 on JudgeBench) compared to the base model, whereas Llama model shows a much stronger improvement. Can the author comment on why this is the case?
2. Can the authors elaborate on the compute cost (e.g., GPU-hours, convergence time) of J1 (GRPO) versus the DPO-based EvalPlanner baseline? This would provide a more complete picture of the trade-offs involved.

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
This paper proposed J1 model which combines verifiable rewards and reinforcement learning, enabling LLM judges to develop deeper reasoning, fairer evaluation, and more consistent decision-making across diverse tasks.

### Strengths
The paper introduces a unified framework that converts both verifiable and non-verifiable evaluation tasks into verifiable formats using synthetic preference pairs; applies online RL to directly optimize the chain-of-thought reasoning in LLM judges; showing a novel consistency-based reward that enforces the same judgment regardless of response order; and develops a multitask J1 model that jointly learns from both pairwise and pointwise supervision. Although similar ideas have been employed in other papers, their application to LLM Judge is novel.

### Weaknesses
1.	The current setup focuses solely on pairwise and pointwise evaluation, without exploring extensions to multi-response or listwise judgment
2.	The work defines both Verdict Correctness and Verdict Consistency rewards, but lacks any reward weighting or sensitivity analysis
3.	The data used for training and evaluation primarily covers conversational and reasoning domains, with no experiments on diverse areas such as code or multimodal judgment
4.	This paper omits any discussion of training cost or computational resources

### Questions
see above weakness

### Soundness
3

### Presentation
3

### Contribution
3
