# Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 6, 4

## Abstract
Reinforcement Learning (RL) has emerged as a pivotal method for improving the reasoning capabilities of Large Language Models (LLMs). However, prevalent RL approaches such as Proximal Policy Optimization (PPO) and Group-Relative Policy Optimization (GRPO) face critical limitations due to their reliance on sparse outcome-based rewards and inadequate mechanisms for incentivizing exploration. These limitations result in inefficient guidance for reasoning. Specifically, sparse rewards fail to deliver sufficient feedback, particularly for challenging problems. Furthermore, such rewards induce systematic biases that prioritize exploitation of familiar trajectories over novel solution discovery. These shortcomings critically hinder performance in complex reasoning tasks, which inherently demand iterative refinement across intermediate steps. To address these challenges, we propose an Intrinsic Motivation guidEd exploratioN meThOd foR LLM Reasoning (i-MENTOR), a method designed to deliver dense rewards and amplify exploration in the RL-based paradigm. i-MENTOR introduces three innovations: trajectory-aware exploration rewards that mitigate bias in token-level strategies while maintaining computational efficiency; error-conditioned reward allocation to ensure efficient exploration on challenging samples while intrinsically stabilizing training; and advantage-preserving integration that maintains advantage distribution integrity while incorporating exploratory guidance. Experiments across 4 public datasets demonstrate i-MENTOR's effectiveness, achieving a 22.23\% improvement on AIME 2024.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses the lack of exploration in RL fine-tuning of reasoning LLMs, where models like PPO and GRPO rely on sparse, outcome-only rewards. The authors introduce i-MENTOR, a lightweight framework that adds (1) a trajectory-level curiosity signal computed by an RND-style module, (2) an error-conditioned gating that only applies bonuses when the final answer is wrong, and (3) an advantage-preserving integration that adds intrinsic rewards after advantage computation to avoid gradient sign flips. Experiments on GSM8K, Countdown-34/4, and AIME-24 with Qwen2.5-3B and DeepSeek-7B show steady improvements, particularly on harder datasets.

### Strengths
- Clear and simple idea: exploration defined at sequence level fits the nature of reasoning tasks.
- Implementation is easy to reproduce; curiosity module is small and cheap.
- Consistent gains across benchmarks and backbones; stronger effect on the hardest dataset.
- Ablations show the effect of each component, and sensitivity analysis helps understand tuning.
- The method seems robust and does not require additional supervision or annotation.
- Code is available and reproducible.

### Weaknesses
- Limited novelty, the paper is mainly limited to adding a curiosity reward to GRPO
- Limited domain coverage: only math and logic tasks are tested, could extend evaluation to other domains.
- Group-size analysis: GRPO group size is fixed (G=5) and never varied, though this can affect both reward scaling and exploration diversity.
- Intrinsic head details: predictor/target architecture is given only briefly, and there is no ablation on the encoder choice.
- Lack of robustness tests: no evaluation under prompt wording perturbations or noise, which is common in reasoning benchmarks.
- No cost table: paper says module is “lightweight” but does not report training-time or overhead
- Missing references to prior curiosity-driven methods, such as below:


@inproceedings{pmlr-v260-bougie25a,
  title     = {Exploring Beyond Curiosity Rewards: Language-Driven Exploration in RL},
  author    = {Bougie, Nicolas and Watanabe, Narimasa},
  booktitle = {Proceedings of the 16th Asian Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {260},
  pages     = {127--142},
  year      = {2025},
  month     = {Dec},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v260/bougie25a.html}
}

@article{song2025outcomeExploration,
  title   = {Outcome-Based Exploration for LLM Reasoning},
  author  = {Song, Yuda and Kempe, Julia and Munos, Remi},
  journal = {arXiv preprint arXiv:2509.06941},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.06941}
}

@article{bamba2025xrpo,
  title   = {XRPO: Pushing the Limits of GRPO with Targeted Rollout Allocation},
  author  = {Bamba, Ulysse and others},
  journal = {arXiv preprint arXiv:2510.06672},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.06672}
}

### Questions
1. Clarify which intrinsic-reward normalization is used in all experiments and provide code/configs.
2. Report group-size sensitivity, intrinsic head ablation, and cos of adding i-MENTOR.
3. Extend experiments beyond math to at least one code or QA domain.
4. Include missing references 
5. Explain more contributions and novel aspects of the method

### Soundness
3

### Presentation
2

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
This paper proposes i-MENTOR, an intrinsic-motivation mechanism to improve RL fine-tuning of LLMs for multi-step reasoning. i-MENTOR (1) replaces token-level RND with a trajectory-aware RND to avoid sequence-length bias and reduce per-token cost, (2) applies exploration bonuses only to incorrect trajectories , and (3) injects exploration rewards after advantage computation to avoid destabilizing PPO/GRPO updates. Experiments on GSM8K, Countdown-34/4 and AIME 2024 show consistent gains vs PPO/GRPO, ablations that attribute gains to each component, and a short training-time overhead analysis.

### Strengths
- The motivation for incorporating RND-based intrinsic rewards to encourage exploration is clear.
- The presentation is well-structured, and the empirical results are clearly organized and easy to follow.

### Weaknesses
1. The experimental settings are rather limited. Both GSM8K and Countdown are relatively simple benchmarks, lacking evaluations on more challenging or large-scale reasoning tasks.

2. The paper omits critical implementation details of RND. For example, it is unclear what the input to the predictor and target networks is, or how sequences are represented.

3. The idea of using RND for intrinsic exploration reward is not particularly novel, and most of the other design elements appear to be practical implementation tricks rather than fundamental innovations. 

4. The paper lacks comparisons with other exploration-enhancing methods, such as entropy-based mechanisms[1] or DAPO’s clip-higher strategy[2].

[1] The entropy mechanism of reinforcement learning for reasoning language models
[2] Dapo: An open-source llm reinforcement learning system at scale

### Questions
see weaknesses

### Soundness
3

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
The paper introduces i-MENTOR, a novel approach that leverages intrinsic motivation to guide exploration specifically within sequential reasoning tasks for Large Language Models (LLMs). The proposed method successfully adapts established exploration techniques, such as those based on Random Network Distillation (RND), for the LLM domain. The authors demonstrate that i-MENTOR achieves improved performance over baselines, including PPO and GRPO, across challenging arithmetic and logic datasets like GSM8K, Countdown, and AIME 2024.

### Strengths
1. The proposed i-MENTOR method successfully demonstrates improved performance over both baselines (PPO and GRPO) across multiple reasoning datasets.
2. The paper provides a clear and well-articulated motivation for incorporating each technical component, making the design choices behind i-MENTOR easy to follow.

### Weaknesses
The paper's core claim rests on enhancing exploration quality, yet it fails to provide compelling empirical evidence or quantitative metrics that directly confirm the proposed method effectively increases or improves the quality of exploration compared to the baselines.

### Questions
Questions
1. In Table 2, if i-MENTOR effectively enhances exploration, it should naturally lead to increased answer diversity. This would typically manifest as a similar performance score at Pass@1 and a significantly higher score at Pass@32. Why do the empirical results show that the performance improvement remains similar or even diminishes as the response count is increased?
2. What accounts for the numerical difference between the results reported in the Countdown-4 rows of Table 1 and Table 4, given that they appear to share identical experimental settings?

Suggestions
1. It would be better to include the standard deviation of the accuracy metrics in all tables to ensure the statistical significance of i-MENTOR's improvements can be properly assessed. Given the inherent variance from random initialization of the RND network and the stochastic sampling of tokens from the base LLM, including the standard deviation is crucial for validating the statistical significance of the reported improvements.
2. It would be better to Introduce a specific metric designed to quantify the diversity or efficiency of the exploration process in the thought/token space to directly validate the central hypothesis.

### Soundness
2

### Presentation
3

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
This paper proposes i-MENTOR, a method for fine-grained reward assignment that aims to address the problem of sparse rewards and insufficient exploration in current RL algorithms. By integrating i-MENTOR, models like Qwen-2.5-3B and DeepSeek-7B achieve improvements over GRPO or PPO.

### Strengths
1. This paper clearly identifies the problem of sparse reward in current RL methods, where current RL algorithms generally adopt the outcome reward but overlook the differences of among important segments or tokens.

2. This paper evaluates i-MENTOR on 2 models from different families, 2 mainstream RL algorithms, and 4 benchmarks, which is comprehensive to reveal the performance improvement from using i-MENTOR

### Weaknesses
1. This paper lacks analysis for the problem they aim to address: the sparse reward and inadequate exploration. It would be better for the authors to show some empirical results and analysis to demonstrate the impact of these methods on the optimization performance.

2. The use of RND (Random Network Distillation) is heuristic. It would be better to deeply discuss the selection of RND as the process reward model.

3. The experiments show the results of two models with different sizes, which is good. However, there are no results from models of the same family with different sizes (e.g., Qwen-2.5 (1.5B, 3B, 7B)); we cannot see the scalability of i-MENTOR.

4. Although math problems are good to evaluate the reasoning performance of different RL methods, it is better to show empirical results from benchmarks across different domains, such as code and logic.

5. The presentation of the experimental results is misleading. For example, in Table 2, the AIME 2024 only has 30 questions, so the improvement of i-MENTOR-GRPO is limited to only two more correct solutions. The significance of improvement could be misleading in understanding the performance of i-MENTOR.

6. In abstract: Group-Regularized Policy Optimization -> Group Relative Policy Optimization

### Questions
1. What is the connection between the motivation “guide exploration” and the method involving the RND framework to provide the process reward? More justification of the connection should be presented in this paper.

2. How does the performance of i-MENTOR compare with other improvement methods for RL algorithms, such as DAPO, or simply applying a process reward model trained from PRM800K? Both of them can adapt to the RL process and acquire extra training time to improve performance, which is similar to i-MENTOR?

3. How does the performance compare when using alternative models instead of RND to compute trajectory-aware exploration rewards?

4. Is it necessary to retrain the RND model when applying it to a new benchmark? If the RND trained on one benchmark cannot generalize to others, its efficiency may be significantly reduced.

### Soundness
2

### Presentation
2

### Contribution
2
