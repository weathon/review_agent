# RewardBench 2: Advancing Reward Model Evaluation

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Reward models are used throughout the post-training of language models to capture nuanced signals from preference data and provide a training target for optimization across instruction following, reasoning, safety, and more domains. The community has begun establishing best practices for evaluating reward models, from the development of benchmarks that test capabilities in specific skill areas to others that test agreement with human preferences. At the same time, progress in evaluation has not been mirrored by the effectiveness of reward models in downstream tasks -- simpler direct alignment algorithms are reported to work better in many cases. This paper introduces RewardBench 2, a new multi-skill reward modeling benchmark designed to bring new, challenging data for accuracy-based reward model evaluation -- models score about 20 points on average lower on RewardBench 2 compared to RewardBench, a widely-used existing reward model evaluation-- while being highly correlated with downstream performance. Compared to most other benchmarks, RewardBench 2 sources new human prompts instead of existing prompts from downstream evaluations, facilitating more rigorous evaluation practices. In this paper, we describe our benchmark construction process and report how existing models perform on it, while quantifying and providing new insights on how performance on the benchmark correlates with downstream use of the models in both inference-time scaling algorithms, like best-of-N sampling, and RLHF training algorithms like proximal policy optimization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RewardEval, a benchmark and methodology for evaluating reward models. It provides a standardized, interpretable, and scalable way to assess how well reward models capture human preferences across diverse tasks. Compared with RewardBench, RewardEval focuses on unseen, in-the-wild human prompts sourced from WildChat and applies decontamination to avoid overlap with common downstream evaluations. It also includes analyses of reward-model-guided best-of-n sampling and reinforcement learning from human feedback (RLHF). In practice, RewardEval’s scores show strong correlation with best-of-n downstream performance and reveal an important RLHF insight: for PPO, the alignment and distributional match between the policy and the reward model are critical, and high RewardEval scores alone do not guarantee PPO improvements when the reward model is off-policy or out-of-distribution. Overall, RewardEval serves as a more challenging and downstream-relevant successor to RewardBench, with particular emphasis on instruction following, math, and factuality, areas where many leading reward models continue to struggle.

### Strengths
1. Presents a well-designed and comprehensive benchmark that goes beyond pairwise preference accuracy to capture more realistic aspects of reward model performance.
2. Uses unseen, in-the-wild human prompts and diverse domains, improving robustness and reducing contamination compared to prior benchmarks like RewardBench.
3. Provides insightful analyses on best-of-n sampling and RLHF, highlighting practical implications of distribution and policy mismatch in reward-guided optimization.
4. Demonstrates strong empirical validation, with RewardEval scores correlating closely with downstream performance.
5. Offers clear presentation, transparent methodology, and open-source resources that enhance reproducibility and long-term research impact.

### Weaknesses
1. The paper feels somewhat incremental compared to RewardBench, as it builds upon a similar benchmarking foundation. Although it includes additional analyses on reward-model-guided training and inference, these studies are not comprehensive enough to establish deeper or more general conclusions.
2. The analysis of RLHF training dynamics is limited to experiments using the TULU 3 8B model, which restricts the generality of the reported insights across architectures and scales.
3. Additional evaluation dimensions such as reward model robustness and reward hacking resistance would be highly valuable to the community. In practical RLHF setups, reward models often become ineffective after short on-policy training, as the policy quickly learns to exploit their weaknesses. This limitation is also reflected in the paper’s own findings (Section 5.2), where all evaluated reward models, regardless of their RewardEval scores, lead to similar final policy performance after RL training. Addressing this issue directly would make the benchmark far more impactful and less incremental compared to RewardBench.

### Questions
1. How do you plan to extend RewardEval to better capture reward model robustness and resistance to reward hacking? Given that on-policy RL training often leads to rapid overoptimization and degradation of reward signal quality, have you considered incorporating adversarial or on-policy evaluation settings into the benchmark?
2. In Section 5.2, you observe that all reward models, regardless of their RewardEval performance, produce similar final outcomes after PPO training. Could you elaborate on whether this suggests that current reward model quality is not the primary bottleneck in RLHF, or that the RL optimization dynamics overpower the reward signal?
3. The analysis of RLHF training dynamics is based solely on the TULU 3 8B model. Do you expect similar trends for other model families? and do you have preliminary evidence to support that expectation?

### Soundness
4

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
This paper introduces REWARDEVAL, a large-scale benchmark for evaluating reward models (RMs) used in RLHF and inference-time selection (e.g., best-of-N sampling). The benchmark spans six domains—three familiar ones (Focus, Math, Safety) and three new ones (Factuality, Precise Instruction Following, and Ties). It is constructed from unseen, high-quality human prompts, with four candidate completions per prompt, enabling more granular accuracy measurement and margin-based calibration testing. The authors train 120 Bradley-Terry reward models and show that REWARDEVAL correlates strongly with downstream PPO and BoN performance, while also revealing that RM–policy lineage alignment is crucial for stable RLHF outcomes. Compared to prior datasets such as RewardBench and PPE, REWARDEVAL claims to provide cleaner, harder, and more diagnostic evaluations.

### Strengths
- Ambitious, comprehensive benchmark spanning six diverse domains.

- Strong empirical study with over a hundred RMs and multiple baselines.

- Identification of practical phenomena such as the importance of model lineage in RLHF.

- Systematic comparison to RewardBench, PPE, and other RM datasets clarifies positioning.

- High reproducibility through public release and clear experimental pipeline.

### Weaknesses
- The Ties metric lacks invariance to scaling/temperature and may over-penalize calibrated models.

- Domain averaging ignores differing sample sizes, reducing statistical interpretability.

- Heavy reliance on LLM-as-judge for factuality/safety labels introduces label bias and potential leakage.

- Correlation analyses are based on a single policy distribution, limiting generality.

- “Stronger correlation with BoN” may partially stem from shared data lineage rather than intrinsic benchmark quality.

- No formal error analysis or confidence intervals are reported.

These weaknesses do not invalidate the idea but suggest that the benchmark’s mathematical rigor and external validity remain limited.

### Questions
1. How robust are the REWARDEVAL correlations when evaluated on policies outside the Tulu family (e.g., Mistral or Llama3 or any other family)?

2. Can the authors provide a scale-invariant version of the Ties metric (e.g., based on ranking or normalized variance)?

3. Were the factuality and safety labels cross-checked with independent human annotators to mitigate LM-as-judge bias?

4. Could domain weighting or bootstrapped confidence intervals be added to report uncertainty in the overall score?

5. How do results change if the number of completions (`N`) increases beyond four?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces REWARDEVAL, a new multi-skill benchmark aimed at evaluating reward models, a key contribution of REWARDEVAL is its use of newly collected, unseen human-authored prompts, rather than reusing prompts from downstream evaluation datasets. This deliberate decontamination strategy ensures the benchmark provides a clean, unbiased evaluation of reward model generalization and prevents data leakage from overlapping with RLHF or inference datasets. It also demonstrates strong correlation with downstream performance, including in best-of-N sampling and PPO-based RLHF training, highlighting its value as a predictive diagnostic tool for real-world effectiveness. Beyond benchmarking, the paper offers actionable insights for improving reward model training, for example, finding that training for more than one epoch can enhance performance in certain regimes, counter to common assumptions in preference model fine-tuning.

### Strengths
The paper presents a well-designed benchmark that significantly improves upon RewardBench by introducing unseen, human-written prompts to ensure data decontamination and incorporating new task categories for broader coverage. It conducts comprehensive experiments analyzing correlations with downstream tasks, yielding several insightful findings: (1) combining multiple data sources enhances average performance, (2) the choice of base model influences reward model effectiveness, and (3) training reward models for multiple epochs does not inherently degrade downstream performance, challenging common assumptions. The structure is clear and experiments are well-thought, and the insights are easy to understand backed by comprehensive experiments.

### Weaknesses
"For RLHF, the reward model should be based on a model of the same lineage as the policy
model or else downstream performance can degrade significantly, so simply taking the
highest scoring reward model on a benchmark will not ensure a good post RLHF model." this seemed to be a very strong statement, I don't see experiments conducted across various model types, a study on different architectures might be beneficial or make this statement less affirmative might be a better consideration?

### Questions
Do we have results to backup this claim : "the reward model should be based on a model of the same lineage as the policy model or else downstream performance can degrade significantly"? Sorry I didn't find it easily in paper? And I assume it means the policy model needs to be the same as the reward model? I don't see a table indicates that correct me if I am wrong.

### Soundness
2

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
4

### Summary
- Introduces a tougher reward model benchmark with six domains, a best-of-4 selection format that lowers the random baseline to 25%, and mostly unseen human prompts to reduce contamination.
- Shows strong correlation with best-of-N sampling and highlights that transfer to PPO depends on on-policy or lineage-matched reward models.
- Reports that top models score notably lower than on prior benchmarks, indicating increased difficulty and headroom.
- Provides practical training insights, including benefits from more than one epoch and lineage matching for RLHF.

### Strengths
- Principled evaluation design with a lower random baseline that better matches downstream selection.
- Comprehensive domains, including calibration via ties, and strong empirical validation against best-of-N.
- Scaled experiments across many trained and existing RMs yield practical insights.
- Clear practitioner guidance on training and deployment.

### Weaknesses
- Candidate pool may bias difficulty and favor models similar to generators.
- Mixed metrics across domains, with the ties metric blending correctness and calibration, can reduce comparability.
- Limited policy diversity and small subsets in places may restrict generality and statistical power.
- Heavy reliance on frontier models for filtering could introduce systematic biases.

### Questions
- How stable are rankings if the best-of-4 candidate set is regenerated with a different generator pool or temperatures?
- Can lineage matching be quantified more continuously to predict PPO transfer beyond a binary on-policy label?
- What is the computational cost tradeoff of best-of-4 compared to pairwise setups for broad adoption?
- How robust is the ties subset to subtle quality differences and score distribution shifts?

### Soundness
3

### Presentation
3

### Contribution
3
