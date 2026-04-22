# EFRame: Deeper Reasoning via Exploration-Filter-Replay Reinforcement Learning Framework

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Recent advances in reinforcement learning (RL) have significantly enhanced the reasoning capabilities of large language models (LLMs). Group Relative Policy Optimization (GRPO), a lightweight variant of Proximal Policy Optimization (PPO), improves efficiency but suffers from limited exploration and training instability, limiting its effectiveness on complex reasoning tasks. To address these challenges, we introduce EFRame, an Exploration-Filter-Replay framework that augments GRPO across three dimensions: additional rollouts enable deeper and more targeted exploration, online filtering removes low-quality samples to stabilize gradients and accelerate training, and experience replay amplifies rare yet informative trajectories for stable convergence. This unified framework establishes a principled training cycle that balances exploration, efficiency, and stability. Experiments on diverse reasoning benchmarks and evaluation settings  demonstrate that EFRame achieves consistent gains, including a 37.9\% improvement on Geometry3K over GRPO, and exceeds RL baselines under Pass@K settings. EFRame further supports fine-grained sample categorization and precise entropy control, highlighting it as a robust solution for advancing deeper reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EFRame, a Exploration–Filter–Replay reinforcement learning framework designed to enhance the reasoning capability and stability of large language models (LLMs) during post-training. Building upon Group Relative Policy Optimization (GRPO), EFRame introduces three complementary modules: additional rollouts to promote targeted exploration for difficult prompts, online filtering to remove low-quality or zero-advantage samples (stabilizing gradients and improving efficiency), and experience replay to amplify the influence of rare but informative trajectories (mitigating entropy explosion and ensuring stable convergence).

### Strengths
- The paper is well written and easy to follow.
- The experiments are conducted on three diverse datasets and the gains are strong.
- The framework has three distinct parts which the authors conduct ablations by isolating the effect of each component.

### Weaknesses
- While well-engineered, the framework primarily combines known components (resampling, filtering, replay buffer) on top of the existing GRPO framework rather than introducing a fundamentally new optimization principle.
- The paper lacks theoretical justifications, and some claims are poorly supported:
    - In lines 243 - 248, "low-quality samples are significantly more numerous than high-quality ones, ... the informative signal from high-quality samples may be drowned out by chaotic updates from low-quality ones". I am not sure if it is correct. Although there are more low-quality samples, the absolute value of their advantages is much closer to 0 than the high-quality ones. In other words, we assign larger weights to the high-quality ones when updating than to the low-quality ones. As you stated in Theorem 3.4, the sum of their advantages should be 0, which means that we put the same amount of "weight" onto low and high quality responses.
    - Theorem 3.5 uses a theorem under the tabular setting with NPG. I am not sure if it can be directly applied here.
- The experiment settings are a bit "toy" and not really realistic. The paper uses Qwen 2.5, which is a bit old, a relatively short context of 2k tokens, and the number of rollouts ($G_1$) is 5, while DAPO and other recent recipes set it to 16. The scale of the experiments is not large enough to showcase the effectiveness of the method.
- Inconsistent experiment results.
    - Could the authors clarify how evaluation is performed on AIME and MATH? Is it based on pass@1 only? The AIME results suggest this may be the case, since the reported numbers are multiples of 1/30 (AIME has 30 questions). If so, the variance of pass@1 is quite high, and it would be more robust to report pass@32 or a similar metric.
    - For MATH, if the metric is indeed pass@1, it is unclear how results such as 65.5 and 78.3 were obtained. The test set contains 500 questions, so the accuracy should be a multiple of 1/500, and fractional correctness (e.g., half a question) is not meaningful. You cannot answer a question half correctly.
    - From Figure 5, we can see that, at step 100, the reward of the orange line is much higher than the blue line, while the accuracy is the other way around. I wonder if it is also due to the large variance of the evaluation.
    - DAPO takes more time for each step as it keeps resampling. However, under the same number of steps, the performance of DAPO should be higher than GRPO since, for each batch, it contains more gradient signals as zero-advantage prompts are all filtered out. Could the authors provide an explanation for why DAPO performs worse than GRPO in their experiments?
- Small typos:
    -  Figure 3 (b) and (c), EFR should be EFRame.

### Questions
Please refer to the weaknesses.

### Soundness
1

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
This paper proposes EFRame, a framework that enhances the reasoning ability of LLMs through an exploration–filter–replay mechanism built on top of GRPO. The idea is to generate more diverse samples, filter low-quality responses online, and replay high-quality trajectories to improve stability and exploration efficiency. Experiments are conducted on Qwen models across several reasoning benchmarks.

### Strengths
1. The proposed Exploration–Filter–Replay framework is conceptually clear and easy to follow.

2. The method improves training stability and reasoning accuracy compared to GRPO baselines.

3. The ablation experiments provide useful insight into the contribution of each component.

### Weaknesses
1. Limited novelty: Similar mechanisms have already been explored in RLEP [1], RePO [2] and VL-Rethinker [3], which all employ replay-based or filtering strategies to stabilize reinforcement learning for reasoning tasks.

2. Baseline insufficiency: The paper does not compare against these closely related works [1–3], making it unclear how much gain is attributable to EFRame itself.

3. Lack of exploration metrics: The claimed improvement in exploration is not supported by pass@k, a standard evaluation metric for reasoning diversity.

4. Model limitation: Experiments are restricted to two Qwen models, with no tests on other LLM families (e.g., Llama, Deepseek).

5. Benchmark limitation: The paper omits newer reasoning benchmarks such as AIME25, MMT-Feb24, HMMT-Feb25, and CMIMC25.

6. Data contamination risk: Recent research shows that Qwen2.5 is susceptible to data leakage on certain reasoning benchmarks, raising doubts about evaluation improvements [4].

7. Lack of theoretical explanation: The paper provides no deeper analysis of why the combined exploration–filter–replay design leads to consistent improvement beyond empirical evidence.

### Questions
1. How does EFRame differ algorithmically from RLEP [1] , RePO [2] and VL-Rethinker [3]

2. How about the results on pass@k?

3. Have the authors tested EFRame on other model families to confirm generality?

4. Would the conclusions hold on newer benchmarks such as AIME25 or HMMT-Feb25?



[1] Zhang et al., RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning, arXiv:2507.07451

[2] Li et al., RePO: Replay-Enhanced Policy Optimization, arXiv:2506.09340

[3] Wang et al., VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning, arXiv:2504.08837 

[4] Wu et al., Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination, arXiv:2507.10532

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Vanilla Group Relative Policy Optimization (GRPO) suffers from limited exploration and training instability. To address these, the authors introduce EFRame that augments GRPO via three components: (1) additional rollouts to promote exploration, (2) online filtering removes low-quality samples to stabilize gradients, and (3) experience replay for stable convergence. Through these mechanisms,  EFRame balances exploration, efficiency, and stability, finally achieving 4.6% improvement on Geometry3K over vanilla GRPO.

### Strengths
1. Authors provide recipe for stable RL training which includes additional rollouts with higher temperature, online filtering, and experience replay. I believe it's a promising research direction.

2. This paper provides detailed analysis of each introduced mechanism based on the current challenges of GRPO, which is well motivated and reasonable.

3. This paper is well organized and easy to follow.

### Weaknesses
I discuss the weaknesses of originality and experiments. Weaknesses marked with **W** are key concerns that might affect the final rating, while weaknesses marked with **M** may have minor impact on my rating.

### Originality
**[M1]** The core ideas used in this work, *i.e.*, adaptive sampling for hard problems [1][2], online filter [2][3] and experience relay [4][5], have been explored in prior literature. This work combines these existing ideas well, but it's not very inspiring to me.

### Experiment
**[W1] Limited evaluation domains.** I notice that the training dataset includes math domain (DAPO-17k) and other general domains (*e.g.*, ScienceQA, ChartQA). However, the evaluation only focuses on mathematical reasoning domains, which raises concerns about its generalization ability. I suggest adding general reasoning tasks like MMLU-STEM, MMLU-Pro, GPQA, MMMU, and DocVQA to further validate the effectiveness of the proposed method.

**[W2] Limited model backbones.** This paper only use Qwen2.5-7B series (*i.e.*, Qwen2.5-math-7b, Qwen2.5-VL-7B-Instruct) to conduct the experiments. However, recent studies reveal that there may be potential data contamination in the Qwen model [6]. Consequently, conclusions derived from contaminated benchmarks (MATH-500, GSM8K) on Qwen2.5 series may be unreliable. The transferability of the proposed method to different models like Llama or Gemma warrants a more in-depth investigation.

**[M2] Concerns on Performance Gains.** I notice that EFRame improves ~1.0% over baselines on most benchmarks (Table 1). Does this suggest a possible limit to the power of this method to discover new reasoning patterns?

**[W3] Concerns on Experimental Settings.** I notice the maximum response length is set to 2,048 in RL training (Appendix A). But in DAPO, the default maximum response length is 20,480. So I wonder if the settings unintentionally impaired the exploration capabilities of baselines like GRPO and DAPO.

**[M3] Missing hyperparameters.** I don't find any information about the hyperparameters for the evaluation. What's the maximum response length, temperature, and sample numbers in evaluation? Besides, I don't see the prompt template used in training.

**[W4] Missing computational costs discussion.** In the vanilla GRPO baseline, what is the number of responses sampled for each question? It seems that the introduction of additional rollout and experience replay may bring more computational overhead. I suggest reporting relevant computational costs clearly. 

---

[1] Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL. NeurIPS, 2025

[2] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv preprint arXiv:2503.14476

[3] MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning. arXiv preprint arXiv:2503.07365

[4] RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning. arXiv preprint arXiv:2507.07451

[5] Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945

[6] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination. arXiv preprint arXiv:2507.10532

### Questions
1. What's the setting of GRPO-1 and GRPO-2 in the Introduction? Are there any differences?

2. How many numbers of old responses are used to replay?  Can authors provide more details on the process of replay?

3. How will EFRame handle samples without positive signals after additional rollout?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper propose a framework of exploration and filter sample in RLVR. The experimental results show that it can enhance the performance of RL reasoning.

### Strengths
- This article focuses on an important issue, the significance of exploration for RL.
- The idea is very simple and extensible to prior methods.

### Weaknesses
- The experimental design is relatively weak, with too few baselines — aside from the fundamental algorithm GRPO, the comparison includes only one method of the same type (DAPO). The experimental analysis is also insufficient.
- The paper does not verify scalability, such as testing across different model architectures or sizes.
- All chosen benchmarks are standard math tasks, without any out-of-distribution (OOD) tasks to demonstrate the effectiveness of exploration.

### Questions
As noted in the weaknesses, the paper need to add more experiments and analyses:
- Provide clearer differentiation from concurrent work with detailed comparisons. 
- Expand experiments to include more baselines, models, and diverse domains.
- Conduct more thorough case analysis.

If the authors can address  the above concerns in a revision, I would be willing to reconsider my assessment.

### Soundness
2

### Presentation
3

### Contribution
2
