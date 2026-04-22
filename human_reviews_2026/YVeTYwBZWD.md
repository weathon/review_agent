# Improving Sampling Efficiency in RLVR through Adaptive Rollout and Response Reuse

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Large language models (LLMs) have achieved impressive reasoning performance, with reinforcement learning with verifiable rewards (RLVR) emerging as a standard paradigm for post-training. A representative algorithm, group relative policy optimization (GRPO) (Shao et al., 2024), computes advantages by normalizing outcome rewards within response groups, but suffers from a vanishing advantage issue when all responses in a group receive identical rewards. To address this issue, we propose Adaptive Rollout and Response Reuse Policy Optimization (AR3PO), a sampling efficient RLVR algorithm that introduces two novel techniques: adaptive rollout, which dynamically allocates more responses to difficult prompts while saving computation on easier ones, and response reuse, which leverages previously generated correct responses to provide useful training signals. We compare AR3PO with strong RLVR baselines on multiple representative benchmarks using two different families of base models. Across the 7B and 8B models, AR3PO consistently outperforms GRPO and matches or surpasses DAPO (Yu et al., 2025), reducing rollout cost by up to $4.2\times$. On the larger 32B model, AR3PO achieves comparable performance to DAPO at similar training steps while maintaining substantially lower rollout cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce AR3PO, a sampling-efficient RLVR method that fixes GRPO’s vanishing advantage problem, where all responses in a group get identical rewards and yield zero gradient, by combining two ideas: adaptive rollout and response reuse. Adaptive rollout allocates more generations to hard prompts in staged rounds while saving compute on easy ones, and response reuse supplies training signal by leveraging previously generated correct answers with careful handling. Multiple experiment results over different benchmarks is presented, benchmarking against previous methods DAPO and original GRPO.

### Strengths
- The paper tries to tackle an important aspect of RLVR vanishing gradient obstacle, which is timely
- Experiments are conducted over multiple sets of model families (Qwen/Llama)
- The motivation is clear for building a sampling-efficient method

### Weaknesses
**[w1] The experiment setup and presentation is unclear and problematic:**
- Please report the baseline performance before training, this is necessary to assess the training improvement.
- Improvement is very marginal (<1pp) when comparing to GRPO, which has the relatively similar response sampling efficiency.
-  I am concerned that whether the baseline results are replicated in its actual performance, as the full experiment setup is different from DAPO paper. Given only single-run results are reported, and the credibility of the result is less persuasive. Please consider to report avg & std running results across multiple runs of the baselines along with your methods to increase credibility of your results.
- Also, the training curve for actual accuracy, e.g., the one for the validation set, only includes your method. I would suggest to plot the validation curve for the other method as well along with yours to provide a direct comparison, which could greatly strengthen the credibility and effectiveness of your method. 
- You are working on 32B model, but why not disclose the result in Table 1?

**[w2] clarity should be improved; some explanation and presentation read confusing**
- For the result in Figure 3, it reads very surprising, as DAPO requires 2000 steps while yours only need 130 steps. Such surprising results would also confuse readers and raise the concern regarding training setup, as in the previous experiment, the difference between yours and DAPO is very minor. **Also, in DAPO paper, at around 100 steps, avg@32 on AIME24 is only around 5%, but in your Figure 3, it reaches 15% even for the first validation point; this raises strong concerns regarding the evaluation protocols**. To improve clarity, please include the DAPO results trained using your setup and plot the curve till 150 steps. This is the best way to consolidate this surprising result.
- Check Q1-Q3 in question sections, please clarify. 
- Sampling numbers wouldn't fully reflect the training time; I suggest to include the actual training time for methods in table 1 to better highlight your training efficiency.

### Questions
- For adaptive rollout, I cannot see any technique difference between yours and DAPO, as both will stop sampling until one correct response is sampled out. The only difference I see is that you will remove these response from the prompt space, while DAPO won't. Also, DAPO can also be treated as an "adaptive weighting" as they will also spend more rollouts over the difficult questions. 
- I still don't understand how the response reuse could resolve the problem of all-negative-sample group. The reuse technique is based on the assumption that the prompt has been resolved, which is a minor case for the problem I believe; for unsolved problems, it has no correct response to use.
- Para from line 216 to line 222 reads confusing, please consider to re-write it to make it clear.

### Soundness
2

### Presentation
1

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
This paper addresses the rollout costs of using group relative policy optimization (GRPO) in the context of reinforcement learning with verifiable rewards (RLVR). The work presents a two-pronged approach to improving sample efficiency in RLVR with:
* Adaptive rollout: dynamically allocates more responses to “difficult prompts” than easier ones
* Response reuse: uses previously generated correct responses as training signals
Empirical evaluation of the methods centers on mathematical reasoning benchmarks using Qwen and Llama base models, showing comparable or better performance compared to GRPO and DAPO baselines while improving rollout costs.

### Strengths
* Motivation for the two complementary ideas suggested by the authors—adaptive rollout and response reuse—are motivated by “GRPO’s potential inefficiency” (l 74). 
* Algorithm presentation is simple and straightforward. Direct comparison of the difference between AR3PO and GRPO creates an easy-to-follow illustration of the extension of the author’s proposed approach (Sec 3.2).
* The paper seems promising, with a clear method and potentially promising initial empirical evaluation, though the reviewer believes broader, more thorough empirical substantiation is required (refer to Weaknesses section).

### Weaknesses
* Table 1: it is hard to assess whether AR3PO’s performance differences are statistically significant compared to the two other methods compared against. Are these metrics across a single run of the algorithm? Can you provide error bars for these results? Additionally, these would be useful in Table 2. 
* Baseline methods for comparison yield a somewhat limited empirical assessment of performance. While the two methods that are compared against are appropriate choices—vanilla GRPO, which the proposed method is an extension of, and DAPO, a recent method with dynamic sampling strategy—additional points of comparison would help create a more convincing story of the performance benefits the authors claim AR3PO provides. 
* The two models used are of similar capacity (7B vs 8B params). It would be useful to know how this method performs on a smaller and much larger model.

### Questions
* Did you assess the three methods on pass@X for any of the benchmark tasks? While for lower X, this is a noisier, higher variance metric, it would be interesting to know how the methods compare in terms of yielding more diverse rollouts.
* For transparency, can you please provide a breakdown of the resulting benchmark task makeup, including number of examples, after you apply the filtering?
* Why did you only evaluate Qwen2.5-7B as the base model in Table 2, instead of also evaluating Llama-3.1-8B?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes AR3PO, a novel RLVR framework for improving post-training efficiency in large language models (LLMs). Existing RL-for-reasoning methods rely on fixed rollout budgets and discard previously correct responses, thus they have severe sampling inefficiency issues and advantage vanishing problems. This work introduces a rollout strategy that dynamically allocates sampling resources with historically correct outputs. It mainly contains  1) an adaptive scheduler, to assign more rollouts to harder prompts based on difficulty estimation, 2) a response reuse mechanism, to maintain effective gradient signals under uniform reward conditions. The authors provide experimental results to show that AR3PO effectively prevents advantage vanishing and significantly reduces sampling cost.

### Strengths
The motivation of improving sampling efficiency is reasonable. Most existing work focuses on fixed-budget RL for reasoning, but these approaches cannot handle prompt-specific difficulty or reuse prior correct responses. This would limit training efficiency and stability. This paper dynamically allocates rollouts and reuses verified correct samples, which improves gradient utilization and reduces cost in RLVR.

The workflow is well-structured. This approach balances computational allocation across tasks and further stabilizes policy optimization by mitigating the advantage vanishing issue.

Open code. Most of the information is included.

### Weaknesses
According to the descriptions, this AR3PO relies heavily on difficulty estimation and response caching during rollout process. This brings additional implementation complexity and even instability if the difficulty metric is poorly calibrated. There is no discussion (or limitation part) of this extra computational overhead. The trade-off between the computational cost and efficiency gains is a very important problem in RL for reasoning, which should not be avoided.

The baselines are limited. The paper only compares with GRPO and DAPO. Adding other efficient RL baselines would make the results more convincing, as this paper is not the first one targeting efficiency issues.

The experimental design is not complete. 1) No ablation study to isolate the individual contributions of each component; 2) The experiments are conducted on only two models, without statistical measures such as variance or confidence intervals.

### Questions
See above. And the authors may consider using Qwen 2.5's 1.5B or Qwen 3's smaller models for more results. For me, the 7b and 8b models are almost the same size, and adding a time report would strengthen the claims.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method to improve sampling efficiency in reinforcement learning with verifiable rewards (RLVR) through AR3PO. The authors specifically compared AR3PO with GRPO and DAPO on multiple mathematical reasoning benchmarks using Qwen and Llama base models. They provide results showing that AR3PO achieves comparable or better accuracy than DAPO while reducing rollout cost.

### Strengths
1. The paper is clearly written; the experimental design is easy to follow, and the presentation is complete.

2. The motivation and the method proposed in this paper are easy to understand.

3. The idea of combining adaptive rollout and response reuse is intuitive and aligns well with the efficiency goal in RLVR training.

### Weaknesses
1. Limited novelty. The algorithmic design is standard and common practice. The main change is to adjust rollout allocation adaptively and reuse previous correct responses. Beyond this incremental modification, the method does not introduce a new learning paradigm.

2. The model comparisons are relatively narrow. The proposed AR3PO shows efficiency gains, whereas its performance advantage is marginal over DAPO. Without broader benchmarks or stronger baselines, the claimed improvements are hard to fully justify.

3. Authors did not conduct comprehensive ablation studies for the proposed method.

### Questions
How sensitive is AR3PO to the choice of hyperparameters, eg number of rollout stages?

Including mcode generation or logical QA tasks would help.

### Soundness
3

### Presentation
3

### Contribution
2
