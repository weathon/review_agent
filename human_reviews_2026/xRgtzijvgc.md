# Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Critic-free methods like GRPO reduce memory demands by estimating advantages from multiple rollouts but tend to converge slowly, as critical learning signals are diluted by an abundance of uninformative samples and tokens. To tackle this challenge, we propose the **Dynamic Dual-Level Down-Sampling (D$^3$S)** framework that prioritizes the most informative samples and tokens across groups to improve the efficiency of policy optimization. D$^3$S operates along two levels: (1) the sample-level, which selects a subset of rollouts to maximize advantage variance ($\text{Var}(A)$). We theoretically proved that this selection is positively correlated with the upper bound of the policy gradient norms, yielding higher policy gradients. (2) the token-level, which prioritizes tokens with a high product of advantage magnitude and policy entropy ($|A_{i,t}|\times H_{i,t}$), focusing updates on tokens where the policy is both uncertain and impactful. Moreover, to prevent overfitting to high-signal data, D$^3$S employs a dynamic down-sampling schedule inspired by curriculum learning. This schedule starts with aggressive down-sampling to accelerate early learning and gradually relaxes to promote robust generalization. Extensive experiments on Qwen2.5 and Llama3.1 demonstrate that integrating D$^3$S into advanced RL algorithms achieves state-of-the-art performance with generalization while requiring fewer samples and tokens across diverse reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This study uncovers that in group-relative advantage-based RL algorithms (e.g., GRPO), the theoretical upper bound of the policy gradient norm scales with advantage variance. Building on this insight, the authors propose Dynamic Dual-level Down-Sampling (D3S), a novel training framework that improves efficiency and performance through two key mechanisms:

Sample-level: selects rollouts that maximize advantage variance to enrich signal diversity;
Token-level: prioritizes tokens with high entropy × |advantage|, focusing updates on regions that are both uncertain and influential.
To prevent overfitting, D3S employs a curriculum-inspired dynamic down-sampling schedule, gradually easing selection criteria as training progresses. Experiments on Qwen2.5 and Llama3.1 demonstrate consistent gains over baselines across multiple reasoning benchmarks.

### Strengths
(1) The thesis is well-written
(2) The theoretical proof is reasonable and supports the contribution of this paper
(3) The proposed algorithm is relatively concise and has certain generalization potential
(4) The experiment was conducted from multiple perspectives and achieved a good empirical analysis

### Weaknesses
Recently, there have been many works on trajectory screening based on entropy and advantage, and the methods are relatively homogeneous. Please distinguish the differences between this work and previous ones through detailed comparative experiments and discussions. 

In my view, the amplitude of the advantage value represents the difficulty of the problem (like [3]), and the positive impact of high-entropy trajectories on the optimization of strategies has also been discussed in a large number of studies [1,2]. The combined influence of advantage and entropy was also discussed in the paper [4].

[1]: Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
[2]: The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models
[3]: Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay
[4]: RETHINKING ENTROPY INTERVENTIONS IN RLVR: AN ENTROPY CHANGE PERSPECTIVE

### Questions
Please refute the above weaknesses through discussion or experiment. I will adjust the score based on the author's feedback and the opinions of other reviewers.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a framework called Dynamic Dual-Level Down-Sampling ($D^{3}S$), aimed at making critic-free reinforcement learning algorithms such as GRPO more efficient. The core problem is that these methods are often slow to converge because critical learning signals are diluted by an abundance of uninformative samples and tokens.
Its core contributions are:
1) Theoretical Justification: It first theoretically proves that the upper bound of the policy gradient norm is positively correlated with the advantage variance ($Var(A)$) of the sampled subset. This justifies its strategy of selecting samples to maximize this advantage variance, rather than reward variance ($Var(R)$) as other methods do.

2) Dual-Level Down-Sampling: The $D^{3}S$ framework operates at the sample-level and token-level to select only the most valuable data for updates.

3) Dynamic Sampling Schedule: To prevent overfitting to this high-signal data, $D^{3}S$ employs a dynamic schedule inspired by curriculum learning. It starts with aggressive down-sampling (using fewer, high-signal samples) to accelerate early learning and then gradually relaxes this (using more data) to promote robust generalization.

Experiments on models like Qwen2.5 and Llama3.1 show that $D^{3}S$ achieves state-of-the-art performance on reasoning benchmarks while using significantly fewer tokens and achieving major training speedups.

### Strengths
1) This paper clearly defines the problem and presents a clear methodology, proposing a three-part strategy ($D^{3}S$) within critic-free frameworks like GRPO/GSPO to focus on the most informative samples and tokens from the source.
2) This paper provides sufficient theoretical proof, presenting the upper bound of the GRPO gradient norm (Proposition 1). It further proves that when maximizing advantage variance on a subset, this upper bound increases with the advantage variance (Proposition 2), which forms the theoretical basis for the sample-level selection. Subsequently, the dual selection levels are integrated into a PPO-style objective.
3) The experimental validation is thorough, conducted on models including Qwen2.5-Math-7B/1.5B and Llama3.1-8B, across a total of 7 benchmarks.

### Weaknesses
1) This paper introduces the Dynamic Down-Sampling Schedule, which in turn brings at least four key hyperparameters: $N_{init}$ (initial sample size), $N_{final}$ (final sample size), $K_{init}$ (initial token ratio), and $K_{final}$ (final token ratio). The paper demonstrates in Figure 3 that this dynamic schedule is crucial for preventing overfitting. This implies that the model's final performance may be highly dependent on the careful selection of these four new hyperparameters. However, the paper does not provide a sensitivity analysis for these hyperparameters. This could make the cost of reproduction and application to new tasks very high.

2) In Figure 4, $D^{3}S$ lowers policy entropy in well-aligned models (Qwen, OpenMath2) but sharply increases policy entropy in the unaligned Llama model. The paper attributes both of these contradictory behaviors to the ability of $D^{3}S$ to effectively balance exploration and exploitation. Why does the exact same algorithmic mechanism lead to exploration in one model and exploitation in another? The underlying mechanism for this is not fully substantiated.

### Questions
See weaknesses above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper focuses on enhancing the learning efficiency of current LLM RL algorithms by training on subsets of rollouts with high advantage variance and tokens that exhibit both high advantage value and high entropy. Meanwhile, the algorithm is equipped with a dynamic schedule that includes more rollouts and tokens to prevent overfitting. An empirical comparison with GRPO, GSPO, and PODS is provided on multiple math reasoning benchmarks.

### Strengths
The algorithm is intuitive and well-explained. 

The paper presents a thorough ablation study, demonstrating the contribution of each component.

### Weaknesses
PODS already proposed to downsample rollouts. The advantage of $D^3S$ over the previous model is not immediately apparent.

The analysis of token masking and dynamic scheduling is not thorough enough to understand their effects. (Check the Question section)

With the additional cost of hyperparameters, the performance improvement from PODS is not obvious.

The paper can be better prepared, for example, by adding a comparison to PODS for analysis and including y-axis labels for Figures 3 and

### Questions
1. How is the sample usage rate compared to PODS?
2. Is there any analysis on which tokens are masked off? Do they contain less information?
3. Is there any evidence that, without a downsampling schedule, the model is overfitting? As shown in Figure 2(a), the sample usage is low again. Why can such zero-advantage samples help if they do not modify the gradient?
4. The KL result where $D^3S$ is closer to the reference model is interesting. Why does it happen? Does it contribute to the performance of $D^3S$?
5. Usually, entropy decreases during RL training, but why does Figure 4 show even increasing entropies?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a down-sampling approach for fine tuning LLMs with RL to speed up learning and convergence called D3S. The authors argue that using all the available interaction data can slow down learning because critical learning signals are overshadowed by many uninformative transitions. The proposal is to down-sample training data at two levels: (1) select a small set of trajectories to have large advantage variance, and (2) within each trajectory select transitions whose advantage magnitude multiplied by policy entropy is large. Finally, to avoid overfitting to the sub-selected data, the sub-sampling is decreased over training to promote large updates in early learning, while maintaining diversity and generalization toward the end of training.

The contributions are as follows,
1. Proposing a down-sampling framework to improve training efficiency of critic-free policy optimization algorithms for fine tuning LLMs.
2. Theoretical analysis to prove that maximizing the advantage variance increases the upper bound of the policy gradient norm, which can speed up learning.
3. Experiments show improved sample efficiency, and better final performance on benchmarks compared with previous methods
4. Ablation studies show different components of the algorithm contribute to its improved performance

### Strengths
- I believe this paper is a good contribution to the RL and LLM literature and should be considered for acceptance. Down-sampling training data to improve training efficiency is not common in prior work and may be interesting to the community. Furthermore, maximizing advantage by sub-selecting trajectories is a novel idea and it is nicely investigated in this work.

- The paper's overall quality is good. The writing and content organization is good. The exposition and motivation is clear. The paper mostly does a good job covering prior work and placing their contribution in the context of the broader literature. The proposed algorithm is supported by intuitive and theoretical justification and the experiments are accompanied by sufficient details and analysis.

### Weaknesses
- The paper may benefit from further discussion of prior work such as coresets for machine learning. For example: Mirzasoleiman, B., Bilmes, J.A., & Leskovec, J. (2019). Coresets for Data-efficient Training of Machine Learning Models. International Conference on Machine Learning.
- The paper may also benefit from some discussion of prioritizing transitions in RL. For example: Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized Experience Replay. arXiv: Learning.

- some typos in the abstract and introduction
  - line 017 “efficient”
  - line 019 “proven”
  - line 078 “data pool is gradually expanded”

### Questions
- The paper claims state of the art generalization for D3S. I don't think this claim is supported by the evidence. Maybe this claim needs further elaboration on what is meant by generalization in this case.
- D3S also introduces some hyperparameters such as N_init, K_init, N_final, K_final. Are these values hard to choose or require additional resources to find good values for? Are they sensitive to the task?

### Soundness
3

### Presentation
3

### Contribution
3
