# Learn to Reason Efficiently with Adaptive Length-based Reward Shaping

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large Reasoning Models (LRMs) have shown remarkable capabilities in solving complex problems through reinforcement learning (RL), particularly by generating long reasoning traces. However, these extended outputs often exhibit substantial redundancy, which limits the efficiency of LRMs. In this paper, we investigate RL-based approaches to promote reasoning efficiency. Specifically, we first present a unified framework that formulates various efficient reasoning methods through the lens of length-based reward shaping. Building on this perspective, we propose a novel **L**ength-b**A**sed **S**t**E**p **R**eward shaping method (LASER), which employs a step function as the reward based on target length. LASER surpasses previous methods, achieving a superior trade-off between performance and efficiency. Next, we further extend LASER based on two key intuitions: (1) The reasoning behavior of the model evolves dynamically during training, necessitating reward specifications that are also adaptive and dynamic; (2) Rather than uniformly encouraging shorter or longer chains of thought (CoT), we posit that length-based reward shaping should be difficulty-aware i.e., it should penalize lengthy CoTs more for easy queries. This approach is expected to facilitate a combination of fast and slow thinking, leading to a better overall tradeoff. The resulting method is termed LASER-D (**D**ynamic and **D**ifficulty-aware). Experiments on five open-weight models from 1.5B to 32B demonstrate that our approach significantly enhances both reasoning performance and response length efficiency. For instance, LASER-D achieves a **5.3** improvement on AIME2024 while reducing token usage by **64**%. Further analysis reveals that our RL-based compression produces more concise reasoning patterns with less redundant ``self-reflections''. All resources (Models, Code, Data)
are available at https://github.com/hkust-nlp/Laser.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of inefficiency in Large Reasoning Models (LRMs) caused by over-thinking, where models generate unnecessarily long reasoning chains. The authors propose LASER and LASER-D, two adaptive length-based reward shaping methods that dynamically adjust reasoning length according to task difficulty. The approach provides a unified framework that formalizes existing CoT compression techniques and introduces a fully automated RL method that balances conciseness and reasoning depth. Experiments show significant improvements in both accuracy and token efficiency across multiple models and benchmarks.

### Strengths
1. The paper presents a unified conceptual framework that reformulates various length-control methods as instances of length-based reward shaping. This brings theoretical clarity to prior work and establishes a strong foundation for the proposed methods.

2. The design of LASER and LASER-D is elegant and well-motivated. The difficulty-aware mechanism that adjusts reasoning budget based on in-batch correctness is intuitive and largely automated, reducing the need for manual hyperparameter tuning.

3. Empirical results are very strong. LASER-D achieves large gains in both accuracy and efficiency (e.g., +5.3% accuracy with a 64% token reduction on AIME2024), demonstrating a clear improvement in the accuracy-efficiency trade-off.

4. The evaluation is thorough and convincing. The authors show consistent gains across model scales and domains, and the behavioral analysis offers meaningful insights into how the method reduces unproductive reasoning.

### Weaknesses
1. The difficulty estimation based on in-batch correctness is a noisy heuristic. It may misclassify easy problems as hard, leading to inefficient allocation of reasoning budget and potential instability during training.

2. The “automatic” mechanism introduces new hyperparameters (e.g., monitoring size, update frequency) whose tuning is not well discussed. This adds complexity and may limit practical applicability to new domains.

3. The analysis of reasoning behavior relies on keyword counting, which is a coarse proxy for reflective thinking. The model could learn to express similar unproductive reasoning with different phrasing, weakening the interpretability of this analysis.

4. The step-function reward creates a sharp incentive boundary at the target length. This may encourage minimal valid responses rather than richer reasoning, potentially sacrificing explanation quality in borderline cases.

### Questions
1. How stable is the in-batch difficulty estimation across training, and have you tried combining it with more static difficulty signals?

2. Did you observe cases where the model learned to bypass the reflection keyword filter using alternative phrasing?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a simple framework, LASER, and an advanced version of i,t named LASER, which use reward shaping and dynamic reward shaping methods to do reinforcement learning, respectively, to improve the large reasoning models' efficiency and reasoning ability. The paper starts from a bunch of findings, which motivate their method contribution with difficulty dynamic reward shaping.

### Strengths
1. The paper conducts a lot of evaluation experiments, which point out the deficiency of the current reward shaping methods for efficient LRM training.
2. The design of the method, especially how to monitor the difficulty dymanics, and the logic of the paper are reasonable. 
3. The experiment section includes a lot of baselines and different length settings, and different sizes of models.

### Weaknesses
1. While the method includes some novelty from the length penalty findings, it seems like an improved version of the previous method.
2. The task in the experiment is limited, which only contains mathematical reasoning.
3. The model family is limited, which only contains 2 versions of DeepSeek models.
4. While the tables in the experiment section show the effectiveness of this method, it is still a little bit hard to compare methods because sometimes other methods are better at token usage.

### Questions
1. Is it possible to include experiments with other model families, such as Qwen3 and LLama
2. Is it possible to include some experiments in other reasoning tasks, such as coding or commonsense reasoning?

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
This paper addresses the "over-thinking" issue in Large Reasoning Models (LRMs), where models generate excessively long reasoning traces. The authors propose a unified framework for viewing various RL based chain-of-thought compression methods through the lens of length-based reward shaping. Building on this, they introduce two novel methods: LASER, which uses a step function to reward shorter correct responses, and LASER-D, which dynamically adapts the target length based on question difficulty and the model's evolving state during training. Experiments on several math reasoning benchmarks show that the proposed methods achieve significant improvements in token efficiency while also showing some improvements in reasoning accuracy compared to baselines.

### Strengths
1. The paper presents empirical results, demonstrating a strong trade-off between efficiency and performance. The proposed methods, consistently reduce token usage by a good margin across multiple models and benchmarks while often improving accuracy.
2. The introduction of a unified framework for RL-based CoT compression is a good conceptual contribution. It provides a clear and structured approach to understanding and comparing different reward shaping strategies.

### Weaknesses
1. Framing and Claims: The framing of performance improvements and baseline comparisons could be strengthened. The paper claims adaptive methods like AutoThink and Thinkless remain "verbose on hard ones (AIME)." However, the data in Table 4 shows these methods do reduce token usage on AIME compared to the original model, albeit less than LASER-D. Furthermore, AutoThink achieves higher accuracy on AIME than several LASER and LASER-D variants, which complicates the claim. 
2. Deepseek-r1 distilled models are typically known to be very verbose shown in several recent works. Results on Qwen3 or even DeepScaleR would make the results more impactful.
3. For fair comparison, results should also be compared against officially released, downloadable models from prior work when available (e.g., L1-Max) that have the same starting model, since there may be small changes in training setup.    
4. The unification framework is a strong point, but its coverage doesn't seem to be exhaustive. For instance, It is unclear how it would accommodate methods where a budget is enforced structurally, for example, by inserting an "end think" token after a certain number of steps (eg, in case of Qwen3).

### Questions
1. Could you clarify the choice of alpha=0.01 for the L1-Max baseline, as the original paper reports alpha=0.002? Was this value tuned, and if so, how?
2. Can you confirm that all reproduced baselines used the identical GRPO implementation and hyperparameters (including the "clip-higher" strategy), with the reward shaping term being the only difference?
3. How does the difficulty-aware mechanism handle the "cold-start" problem at the beginning of training, where many prompts may have zero correct rollouts, potentially making the difficulty estimation noisy? Maybe this is not an issue for the training dataset used, but could this be a limitation for training on harder datasets?

### Soundness
2

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
This paper **(LASER)** aims at addressing the _overthinking_ phenomenon in Large Reasoning Models (LRM) through a novel RL reward shaping method. Specifically, it first draws inspiration from truncation-based baseline methods and existing length rewards in efficient reasoning by proposing a step function reward. It then takes problem difficulty and the RL training dynamics into account by further introducing an adaptive module **(LASER-D)** that controls the reward shaping hyper-parameters on-the-fly. Experiment results on various in-domain and out-of-domain benchmarks demonstrate that LASER effectively reduces reasoning length with minimal influence on the accuracy.

### Strengths
- The paper is well motivated and shows a good understanding of existing methods in LLM efficient reasoning.
- The experimental study seems comprehensive (applying the method to 5 LLMs and evaluating on both in-domain and out-of-domain tasks).
- The paper is rigorous in terms of including important details like hyper-parameter choices and abundant ablation studies in the main text & appendix.

### Weaknesses
1. My main concern lies in the numerous *manual design choices* (often appearing as hyperparameters) despite the paper’s claim that most modules are *automatically adaptive*. For instance, the criterion for defining $L_A$ is set as the smallest value ensuring $ECR_d \geq 1$, where “1” represents the threshold for “at least one complete and correct response.” Why fix this threshold as a constant rather than make it depend on the rollout size $K$? Similarly, the lower bound length $L_T$ appears to be manually chosen, and the experiments suggest that its value is adjusted for different model (e.g., 4096 for the 8B model and 8192 for the 1.5B model). Other examples of such manual tuning include the choice of **three** difficulty levels and the update interval of $N$ steps. For practitioners, what concrete guidance is provided for selecting these parameters in new settings?


2. While the overall presentation is clear, I find the explanation in Section 4.2 (*Automatic Adapting Mechanism*) particularly difficult to follow. In particular, the term “group” is confusing, as it can refer to both the “rollout group” (i.e., the $K$ rollouts for a given prompt) and the “difficulty group.” As a result, any statement that simply mentions “group” without specification becomes ambiguous. Additionally, the concept of $ECR_d$ seems to represent a per-difficulty quantity, yet its introduction follows immediately after the sentence “we sample $K$ responses for each query in the monitoring set, and $ECR$ is computed as…,” which makes it sound like a per-query statistic instead. I recommend a thorough revision of this subsection to improve clarity and precision.

3. [Minor] Based on Table 5, I notice that while LASER does work on non-Qwen models as well, the magnitude of length reduction is perceivably smaller than the two Qwen models highlighted throughout main text (on most benchmarks the reductions are $> 50$% on Qwen models, but only around $40$% for other models). Is there any intuitive reason that can explain such discrepancy?

### Questions
Most questions are raised in the weaknesses. Other than above:

1. [Question] During inference time, does LASER further uses the difficulty assessment module to evaluate the difficulty of a new query, or simply let the RL-finetuned model to generate the reasoning without any pre/post-processing?

2. [Typo] **L144:** lack of a verb after "In this section, we []".

### Soundness
3

### Presentation
2

### Contribution
2
