# Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Large language models trained with reinforcement learning on verifiable rewards often inflate response length—trading brevity for accuracy. While longer reasoning can help on hard problems, many extra tokens are filler: verbose text making little progress. We introduce GFPO (Group Filtered Policy Optimization), which curbs this length explosion by sampling larger groups per problem and only training on responses filtered by (1) length and (2) token efficiency (reward per token). By sampling more during training time, GFPO teaches models to think less at inference time. On Phi-4-reasoning, GFPO cuts GRPO’s length inflation by up to 85\% across STEM and coding benchmarks (AIME 24/25, GPQA, Omni-MATH, LiveCodeBench) while preserving accuracy. We find that GFPO also outperforms Dr. GRPO in both accuracy and length reduction and generalizes across model sizes and families. We further propose Adaptive Difficulty GFPO, which allocates more training exploration to harder problems, yielding better efficiency-accuracy trade-offs on challenging questions. With only a 7\% increase in training time, GFPO reduces end-to-end latency by $\sim$30\%, cutting response time on hard queries by 90 seconds. GFPO trades modest training-time increases for lasting gains in inference—an effective recipe for efficient reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Group Filtered Policy Optimization, a simple modification to GRPO for RL with verifier rewards: for each question, sample a larger group of trajectories, rank by a target metric (e.g., length or reward-per-token), and update only on the top-k while zeroing advantages for the rest. Experimental results on Phi-4 reasoning demonstrate the effectiveness in reducing the generation length, latency, and training time.

### Strengths
1. A simple yet effective approach to dynamically adjusting the group size can broaden the response pool, incorporating more candidates with desirable traits and thereby optimizing toward the desired response properties.

2. The authors present comprehensive experimental results, evaluating not only mathematical tasks but also out-of-domain scenarios, and reporting both token-level efficiency and pass@1 accuracy.

3. The authors demonstrate that GFPO reduces latency by generating shorter responses than GRPO, while the token-efficiency variant achieves this with nearly the same training cost.

### Weaknesses
1. There is a line of research on token-efficient GRPO methods [1–4]. Although the authors mention several of these works in the related work section, they do not provide a direct comparison in the paper. Including such comparisons would strengthen the paper’s results and claims.

2. Additional training costs introduced during the process may limit the use of large values of G or K in GFPO.


[1] Aggarwal, Pranjal, and Sean Welleck. "L1: Controlling how long a reasoning model thinks with reinforcement learning." arXiv preprint arXiv:2503.04697 (2025).

[2] Luo, Haotian, et al. "O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning." arXiv preprint arXiv:2501.12570 (2025).

[3] Huang, Chengyu, Zhengxin Zhang, and Claire Cardie. "HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization." arXiv preprint arXiv:2505.11225 (2025).

[4] Arora, Daman, and Andrea Zanette. "Training language models to reason efficiently." arXiv preprint arXiv:2502.04463 (2025).

### Questions
1. What is the performance of GFPO on the Qwen series model?

2. Could you conduct an ablation study by scaling up K (e.g., to 64 or 128)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the critical issue of length inflation in large language models trained with reinforcement learning, where models tend to produce excessively verbose outputs without corresponding accuracy gains. The authors propose Group Filtered Policy Optimization (GFPO), a simple yet powerful modification to the GRPO training process. The key idea is to sample a larger group of responses during training and selectively update the policy using only a filtered subset of the most desirable ones—based on metrics such as conciseness or token efficiency. This in-training filtering serves as an implicit form of reward shaping, effectively teaching the model to be more concise at inference time.

### Strengths
1.GFPO’s core idea of in-training filtering is clear and easy to implement. By sampling a larger pool of responses and selectively training on the best subset, it avoids the complexities of explicit reward engineering. Framing this as implicit reward shaping provides an intuitive and generalizable way to steer model behavior toward desirable attributes like conciseness.

2. The paper demonstrates a rare and highly desirable outcome: improving inference efficiency while maintaining or even enhancing accuracy. The reported gains—up to an 85% reduction in excess length and a ~30% reduction in end-to-end latency are substantial and consistent across challenging STEM and coding benchmarks. 

3. The method generalizes well beyond the training domain. On the out-of-distribution LiveCodeBench benchmark, GFPO continues to reduce response length where GRPO inflates it, sometimes even improving accuracy.

### Weaknesses
1. The Adaptive Difficulty GFPO relies on the average reward of sampled responses to estimate problem difficulty. While conceptually clever, this heuristic can be unstable and noisy, especially early in training or on high-variance problems. Randomly poor rollouts may cause easy instances to be misclassified as “hard,” leading to inefficient resource allocation.

2. The method’s success hinges on the retention fraction k/G, which determines how aggressively responses are filtered. Although the paper explores different ratios empirically, it offers limited theoretical or heuristic guidance for setting this parameter in new settings. This may require costly tuning for practitioners.

3. GFPO’s core trade-off sample more to think less implies higher computational expense during training. While the authors show that a 7% increase in training time yields a 29% reduction in inference latency, this additional cost may still be significant in resource-constrained environments.

4. Filtering based on conciseness or shortness may inadvertently penalize useful but verbose reasoning paths. This could suppress exploration and hinder the model from discovering complex problem-solving strategies that are initially long but later refinable into concise solutions.

### Questions
1. How stable is the average-reward–based difficulty estimate throughout training? Did you explore smoothing or momentum-based approaches to mitigate early-stage noise?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
GFPO (Group Filtered Policy Optimization), which curbs this length explosion by sampling larger groups per problem and only training on responses filtered by length and token efficiency (reward per token). This can make LLM generate shorter reasoning traces during inference.

### Strengths
1. The intuition of sampling more and using more computational resources on hard questions is straightforward and reasonable.
2. The experiment shows the effectiveness of GFPO in different tasks, including mathematical, STEM, and coding reasoning.
3. The method itself is simple and easy to plug into any other RL post-training framworks.

### Weaknesses
1. The evaluated model size and model family are limited, which only contain the 14B Phi-4 model.
2. Considering this method is based on GRPO, maybe an analysis of training stability is needed.

### Questions
1. Is it possible to add some experiments on other sizes of models, besides 14 B?
2. Is it possible to add some experiments on other model families, such as Qwen, DeepSeek, and Llama?
3. Is it possible to include some analysis about the training stability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose GFPO (group filtered policy optimization), a method to encourage concise responses in reasoning models by using both token efficiency and length as filters to select the best examples over a larger set of rollouts. Through this larger amount of oversampling they find better examples to optimize for token efficiency. It adds 7% to the training time but saves 90 seconds on average over long queries.

The adaptive version dynamically scales the number of samples taken from the model based on an estimate of the difficulty. If rollouts continue to fail, more are sampled until a maximum. There are a few insights guiding this: easy questions don’t need as much reinforcement, so we can stick to just giving a few short demonstrations. Harder questions may have different ways of expressing them. The authors claim theirs is the first RLVR method to do something like this.

They test using GFPO to RLVR Phi-4-reasoning against a baseline that used GRPO with 72k math problems. Their approach uses the same data setup as Phi-4-reasoning-plus, so it’s a fair comparison. All parameters are kept the same. GFPO delivers slightly better accuracy and significant improvements in length over GRPO.

I’m not really the right person to review this as I don’t work on RL, so this is a low confidence review.

### Strengths
- Simple, elegant, but novel (I think) idea
- High-validity experiment by testing against an existing baseline with everything else held equal
- Clear demonstration of improvement in performance
- Lengthy and detailed analysis

### Weaknesses
I am pretty convinced by the results, but it would be nice to see the experiment replicated for at least one other model so we can know it’s not a fluke.

### Questions
N/a, sorry my review is late

### Soundness
4

### Presentation
4

### Contribution
3
