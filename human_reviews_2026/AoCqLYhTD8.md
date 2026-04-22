# Policy Optimization with Experience Replay: Guiding Reasoning Models to Complete the Reasoning Path

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 4

## Abstract
To our knowledge, in the field of large language models, all existing reinforcement fine-tuning algorithms require generating a complete reasoning process starting from the question, which results in a substantial time overhead during the rollout phase of training.Challenging this conventional approach, we propose the assumption that during reinforcement fine-tuning, the model only needs to generate part of the reasoning process. We analyze the impact of different segments of the reasoning path on the correctness of the final result, and based on these insights, we introduce \textbf{Policy Optimization with Experience Replay (POER)}, a plug-and-play reinforcement fine-tuning algorithm. Unlike traditional reinforcement fine-tuning algorithms that generate full reasoning paths, POER trains the model by generating suffixes of the reasoning path using experience caching, thereby significantly reducing training time while improving training stability.From evaluations during the rollout phase of training, POER reduces token generation in this phase by approximately 95\%, greatly lowering the theoretical time overhead. In practical training, compared with full-path reinforcement fine-tuning algorithms, POER reduces the training time of the 1.5B model by 90\% and the 7B model by 72\%, while maintaining performance comparable to typical algorithms such as GRPO and DAPO.
We have open-sourced the code in an anonymous repository: \url{https://anonymous.4open.science/r/POER-4BF2}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies the substantial computational cost and instability of RLFT for LLMs as a key bottleneck, primarily due to the need to generate complete reasoning paths during the sampling phase. To address this, the authors propose a method that leverages a cache of previously generated high-quality answers. Instead of generating a response from scratch, it retrieves a cached answer for a given question, truncates the final tokens, and tasks the model with generating only the suffix. This approach is claimed to greatly reduce training time and improve stability. The authors also introduce Length-Aware Reward Shaping mechanism to encourage more concise reasoning. The method is evaluated on different model sizes across several math reasoning datasets, demonstrating significant reductions in training time and stable, competitive performance against baselines.

### Strengths
1. The work tackles a problem of high practical importance. The computational expense of RLFT is a major barrier to its widespread adoption and research. The claimed training time reductions of up to 90% are dramatic and, if robust, would represent a significant practical contribution to the field.

2. The paper's core concept is intuitive and clearly articulated. The motivation for avoiding full-path generation is well-established, and the proposed solution is easy to understand. Moreover, the solid and detailed theoretical analysis in this paper can explain the motivation of experience replay well.

3. The empirical evaluation is extensive. The authors validate their method on multiple model sizes and a range of standard reasoning benchmarks. The experiments are well-designed, including ablation studies on key hyperparameters and transferability of cache between different sizes of models.

### Weaknesses
1. Potential negative effect on the baseline methods. In Table 1, in some benchmarks, the performance of model with POER drops a little, showing the potential negative effect of POER on the reasoning performances of LLMs.

2. The experiments are exclusively focused on mathematical reasoning datasets.  These tasks often have highly structured reasoning paths where the correctness of a prefix strongly predicts the correctness of the full solution. This is a best-case scenario for POER. The method's viability is far less certain on more open-ended or creative tasks where diverse, valid reasoning paths exist. The approach implicitly assumes a single "good" prefix per problem, which is a limiting assumption that may not hold in other domains.

3. The proof of gradient stability in Appendix B essentially formalizes that conditioning on more information reduces variance.  However, it fails to address the more critical issue of bias. By constraining the sampling distribution, POER introduces a potentially significant bias, and this trade-off is never discussed. The reduced variance might simply be a byproduct of the policy learning a less diverse set of behaviors, which is potentially suboptimal.

### Questions
1. Given the heavy reliance on cached prefixes, how can you be sure the model is not simply overfitting to these specific reasoning paths? 

2. Could you provide a more thorough explanation for why providing a 1.5B model with superior reasoning paths from a 7B model yields no significant benefit?

3. The performance of POER seems highly dependent on the quality of the initial cache. How does the algorithm perform if the initial model fails to find correct solutions for a significant portion of the training set, leaving parts of the cache empty or filled with incorrect paths?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper aims to reduce training time while improving training stability. The paper proposes the assumption that during reinforcement fine-tuning, the model only needs to generate part of the reasoning process. Instead of always generating full reasoning paths, POER trains the model by generating suffixes of the reasoning path using experience caching.

### Strengths
1. The paper focuses on an interesting question.
2. The paper conducts ablation and additional experiments to understand the impact of length-aware reward design and the sensitivity of hyperparameters.
3. Experimental details are provided for reproduction

### Weaknesses
1. I think some related work and baselines are missing. For example, experience replay has been proposed by other papers recently [1-3]. 
2. The experiments only use Deepseek-r1-qwen-distill-1.5b and Deepseek-r1-qwen-distill-7b as base models. The models are already well fine-tuned for the reasoning task. It's unclear if the method will still work using less powerful base models.
3. The writing can be improved for better clarity. For instance, Table 1 is somewhat confusing. Demonstrating that GRPO/DAPO improves over the base model does not directly support the main claim. We care more about how GRPO/DAPO compares with GRPO/DAPO + POER. The same issue applies to the “w/o R” results. In addition, the main message of Figure 3 is unclear and should be explained more explicitly.
4. The results appear to be quite sensitive to hyperparameters. As mentioned in the paper:  "The evaluation results are shown in Table 5: under a fixed L, performance first improves as α increases and then declines." This indicates that the method’s stability and generality may depend strongly on fine-tuning specific parameters.
5. Also, the paper says "POER demonstrates lower exploration ability compared to GRPO, and the gap between the two methods
gradually widens as max_length increases." This seems to reveal a clear limitation of the proposed method.
[1] Zhang, Hongzhi, et al. "Rlep: Reinforcement learning with experience replay for llm reasoning." arXiv preprint arXiv:2507.07451 (2025).
[2] Zhang, Xuechen, et al. "BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning." arXiv preprint arXiv:2506.17211 (2025).
[3] Dou, Shihan, et al. "Improving rl exploration for llm reasoning through retrospective replay." arXiv preprint arXiv:2504.14363 (2025).

### Questions
See weakness

### Soundness
2

### Presentation
2

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
Existing reinforcement learning (RL) fine-tuning methods typically require generating a complete reasoning trajectory from the beginning, which introduces substantial computational overhead. To address this challenge, the paper proposes POER (Policy Optimization with Experience Replay) — a reinforcement fine-tuning algorithm that leverages partial reasoning prefixes from a cache of previous trajectories to generate sufficient completions. In addition, the method introduces a length-aware reward to encourage more effective policy gradient updates. The paper provides both theoretical analysis, showing that POER yields a more stable training process, and empirical results demonstrating that across six reasoning benchmarks, POER significantly reduces training time while achieving comparable or slightly improved accuracy relative to GRPO and DAPO.

### Strengths
1. The training speed improvements are quantitatively convincing.

2. The paper is well-organized, providing a clear step-by-step methodological exposition. Figures and tables effectively illustrate both the conceptual workflow and empirical results.

### Weaknesses
1. The idea of learning from partial rollouts has been extensively discussed in prior literature [1], and the use of length-based rewards [2] is also not a new technique. However, the paper does not adequately discuss the differences or connections between these existing works and the proposed method.

2. The average sequence lengths used for both training and evaluation appear to be quite short, which limits comparability across methods. For instance, DeepScaleR-1.5B-Preview [3], trained with pure GRPO on DeepSeek-R1-Distill-Qwen-1.5B, achieves 43.1% on AIME2024, 87.8 on MATH, 73.6 on AMC2023, and 30.2 on MinervaMath, higher than the results reported here. This discrepancy likely stems from shorter evaluation lengths, but the paper does not clarify this. As a result, POER might make the model more myopic, performing optimistically well on short-length reasoning but losing extrapolation capability for longer chains.

3. Moreover, the pass@k performance of POER is substantially worse than the baseline, suggesting that the method sacrifices exploration even with limited training. This raises concerns about its long-term training dynamics and ultimate performance ceiling.

4. The assumption: “for the same question, a reasoning path that reaches the correct answer more concisely should be rewarded with a higher value” doesn’t seem to be supported by any evidence. From a reinforcement learning perspective, excessively concise reasoning paths may encourage memorization rather than adaptive exploration. In contrast, longer reasoning chains often include more trial-and-error behavior, potentially fostering more generalizable reasoning patterns that perform better on harder or unseen problems.

[1] Qu, Yuxiao, et al. "Optimizing test-time compute via meta reinforcement fine-tuning." arXiv preprint arXiv:2503.07572 (2025).

[2] Yeo, Edward, et al. "Demystifying long chain-of-thought reasoning in llms." arXiv preprint arXiv:2502.03373 (2025).

[3] Luo, Michael, et al. DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL. 2025, pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2.

### Questions
1. The paper claims that POER yields a more stable training process than GRPO by comparing variance metrics. However, the theoretical proof of variance reduction (particularly the decomposition shown in Equation (11)) hinges on assumptions that do not hold in the presented setting. Specifically, in POER, $a_k$ is off-policy, sampled from a cache composed of prior policies. Therefore, the conditional expectation $\mathbb{E}_{\left(q_k, a_k\right) \mid q_k}$ in the proof is not taken under the same distribution as the current gradient estimator. This discrepancy invalidates the steps from Eq. (11) → (13) → (18) unless the cache distribution is explicitly modeled and corrected, e.g., via importance sampling.

2. Without such corrections, the POER gradient estimator becomes biased, making a “variance reduction” claim theoretically unfounded. Empirically, the results, reported only up to four epochs, are insufficient to convincingly demonstrate improved stability over GRPO or DAPO, especially for longer training runs where such bias may accumulate.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Policy Optimization with Experience Replay (POER), a plug-and-play reinforcement learning fine-tuning algorithm. The main contribution of this work is to use partially correct reasoning process hints to guide the LLM RL training to improve the training efficiency and reach a more stable training.

### Strengths
1. The motivation and the logic of the algorithm are reasonable.
2. The presentation of the method is clear.
3. The experiments evaluate 6 datasets and 2 different model sizes, which makes this part more comprehensive.

### Weaknesses
1. The model family is limited, only containing DeepSeek models.
2. The task is limited, only containing mathematical reasoning tasks.
3. There is no study about the generalization ability of the models after this training method. 
4. The usage of LLM writing is obvious; although the authors clarify the usage, it is better to rewrite some parts, such as those parts using '-', and also some obvious typos, such as model names.
5. While the method is generally reasonable, it is hard to encounter one problem several times during large-scale training in practice, which I personally think will limit the practicality of this method.

### Questions
1. Is it possible to evaluate some other reasoning tasks, such as coding and commonsense reasoning?
2. Is it possible to include some study about other model families, such as Qwen3 and Llama?
3. Is it possible to add some studies about out-of-domain tasks?
4. Is there a possibility that this method introduces a distribution shift during training because of the usage of cache and the special reward shaping to diversify the reward of sampled generations?

### Soundness
3

### Presentation
2

### Contribution
2
