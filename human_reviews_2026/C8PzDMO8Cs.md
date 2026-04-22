# Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Reinforcement Learning with Verifiable Reward (RLVR) is a powerful method for enhancing the reasoning abilities of Large Language Models, but its full potential is limited by a lack of exploration in two key areas: \textbf{Depth} (the difficulty of problems) and \textbf{Breadth} (the number of training instances). Our analysis of the popular GRPO algorithm reveals a bias that down-weights difficult, low-accuracy problems, which are crucial for improving reasoning skills. To address this, we introduce Difficulty Adaptive Rollout Sampling (DARS), a method that re-weights difficult problems by using targeted, multi-stage rollouts. This approach increases the number of rollout outcomes for these harder problems according to our proposed re-balancing schedules and leads to consistent gains in \textit{Pass@K}. We also found that simply enlarging the rollout size isn't effective and can even harm performance. We also investigated the role of breadth by scaling the batch size and using full-batch updates. This significantly improved \textit{Pass@1} performance by maintaining high token-level entropy, which indicates continued exploration and reduced gradient noise. Finally, we present DARS-Breadth, a combined approach that uses DARS with a large breadth of training data. This method demonstrates simultaneous gains in both \textit{Pass@K} and \textit{Pass@1}, confirming that depth (adaptive exploration) and breadth (scaling the training data) are orthogonal and essential dimensions for unlocking the full reasoning power of RLVR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper analyzes RLVR and argues that existing methods such as GRPO underweight hard problems (depth bias). It proposes Difficulty-Adaptive Rollout Sampling (DARS) to allocate more rollouts to low-accuracy samples and examnines breadth scaling (large-batch updates) as a complementary factor. The authors calim that combining the two (DARS-breadth) yields simultaneous improvements in Pass@1 and Pass@K across math reasoning benchmarks.

### Strengths
1. The paper provides the intuition that GRPO (variants) emphasizes medium-difficulty problems and train less on challenging problems, through math derivation by showing A_{group} = 2 N u (1 - u). 
2. The paper evaluates the DARS and variants on AIME, OlympiadBench, Minerva, MATH500 with multiple model sizes, showing consistent performance trends.

### Weaknesses
1. The performance comparison between GRPO and DARS is not controlled for solution length. Since solution length is generally positively correlated with correctness, and DARS places more emphasis on hard problems that naturally require longer reasoning, it is possible that DARS merely encourages longer generations rather than genuinely improving reasoning ability.
2. The “breadth scaling” results are unsurprising—larger batch sizes often stabilize RL training—and the idea itself is not novel.
3. The claimed synergy between depth and breadth is not convincingly demonstrated. The improvements appear largely additive rather than reflecting a meaningful interaction between the two dimensions.
4. The methodology section is difficult to parse, with inconsistent notation and somewhat heavy presentation. The paper would benefit from clearer mathematical exposition and cleaner notation.

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
2

### Rating Number
2

### Confidence
4

### Summary
The paper firstly tries to present a bias in GRPO which allocates more attention to medium-difficulty problems. Then, they present DARS: Difficulty Adaptive Rollout Sampling, a technique that allows simultaneous improvements in Pass@1 and Pass@k metrics during RLVR training.

### Strengths
The paper is presented well and studies an important problem, that is, trying to mitigate diversity collapse during training with GRPO which has been observed in a couple of different papers in the literature. Their technique is also novel and doesn't seem too hard to implement. They also provide code in the supplementary material which is useful for the reviewer.

### Weaknesses
This paper has a couple of issues:

1. Firstly, they do not discuss how they estimate pass@k. This is an extremely important thing to figure because this quantity has a couple of different estimators. Digging through the code, I could figure out that when there are 128 rollouts, and you want to estimate pass@128, the estimator essentially becomes $$\mathbb{1} \left[\ r_1 = 1 \lor r_2 = 1 \lor \cdots \lor r_{128} = 1\right]$$. This is important to place in the paper.

2. It is very important to note that the above estimator has high variance (compared to a plugin or bootstrap estimator) especially when pass rates for a prompt are very low. Therefore, it is very hard to believe the gains are consistent unless there are confidence intervals for pass@128. Could the authors make confidence intervals and specifically describe how they made them for the numbers in Table 1? 

3. The numbers for the base model in Table 1 for the 1.5B on Math500 seems rather low. For instance, take a look at the Table 2 of Qwen2.5 Math Technical Report [1]. The reported number in this paper is 35.1 and the number in the tech report is 49.8. This seems to be a rather large discrepancy. Could the authors try to replicate their numbers and see how far they can get? 

4. What is the 'cumulative advantage' term that the authors define supposed to represent? There is no mathematical justification of this term. The authors mention that this term is supposed to represent the bias in GRPO. However, there is no clear definition of what quantity is exactly biased here. Could the authors elaborate on what this term is supposed to mean.

5. The number of extra rollouts for the harder prompts is mostly heuristic and lacks any solid mathematical basis. 


[1] QWEN2.5-MATH TECHNICAL REPORT: TOWARD MATHEMATICAL EXPERT MODEL VIA SELF-IMPROVEMENT (https://arxiv.org/abs/2409.12122)

### Questions
Please address the concerns above.

### Soundness
2

### Presentation
4

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
The paper studies RLVR for reasoning LLMs and shows that two important factors: depth (pushing on hard problems) and breadth (batch size)—must be optimized together. It first shows that GRPO/Dr.GRPO’s group-based cumulative-advantage underweights high-difficulty items, capping Pass@K.   The authors then propose DARS, which does a light pre-rollout to estimate per-question difficulty and then reallocates extra rollouts to harder items via Equal-Treatment (ET) or Hardness-Weighted (HW) schedules; in parallel, they scale breadth by replacing PPO mini-batches with full-batch updates across multiple epochs to sustain exploration entropy and raise Pass@1.

### Strengths
1. The main concepts are well defined, and the analysis of both is clear. I also appreciated Figure 2, which succinctly illustrates the issues with default sampling and how DARS resolves them.

2. The paper is well written and easy to follow.

3. I really like the Pass@1-Pass@k visualization idea and the analysis is very clear.

### Weaknesses
1. The evaluation protocol is potentially problematic. For the baseline, the authors pick the checkpoint with the highest Pass@1, which (per Figure 7) may not correspond to the best Pass@128. For DARS, they then choose the checkpoint that surpasses the baseline on Pass@1 and has the highest Pass@128. This selection strategy may inflate the reported Pass@128 improvement.

2. In Table 1, please report uncertainty (e.g., error bars or standard deviations). Also, which sampling temperature is used for each model? Prior work suggests RLVR sharpens the distribution [1], so using a single temperature across models may be unfair [1, 2, 3]. It would be more informative to report the best Pass@128 under by performing a temperature sweep per model/checkpoint. Do the gains remain meaningful with error bars and sampled at model-specific optimal temperatures?


3. While Eq. 2 (cumulative advantage) is used to characterize training effects, it would help to also discuss/update magnitudes (e.g., gradient norms or per-example gradient contributions). A large cumulative advantage does not necessarily imply a large training impact if the underlying gradients are small.

### Questions
1. In Table 1, “Pass@1 (Avg@128)” is unclear. Are these the same quantity? Pass@1 can be computed via greedy decoding (temp=0) or estimated from 128 rollouts at nonzero temperature using an unbiased estimator. Which definition are you using?

2. In Figure 9, the gains appear more substantial and consistent for Llama-3, while Pass@K improvements seem to diminish for Qwen. Do you have comparable baselines for Llama-3? Any hypotheses for why the effect is stronger on Llama-3?

3. Why focus on Qwen-Math models given their extensive math pretraining? Have you tried non-math tasks (e.g., Reasoning Gym) on Qwen models, or alternative bases such as Qwen-Instruct or other families?




[1] Reasoning with Sampling: Your Base Model is Smarter Than You Think https://www.arxiv.org/abs/2510.14901

[2] Decomposing Elements of Problem Solving: What "Math" Does RL Teach? https://arxiv.org/pdf/2502.17356v1

[3] Can GRPO Help LLMs Transcend Their Pretraining Origin? https://arxiv.org/abs/2510.15990

### Soundness
2

### Presentation
3

### Contribution
2
