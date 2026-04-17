# FOCUS: High-Dimensional Model-Based RL via Focused Control Unit Sampling

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Reinforcement learning (RL) in high-dimensional continuous state and action spaces often struggles with low learning efficiency and limited exploration scalability. To address this, we introduce FOCUS, a novel model-based RL framework that leverages the insight that effective policies often rely on dynamically focused, sparse control. FOCUS learns preferences over action dimensions to facilitate more targeted and efficient policy learning. It employs a hierarchical decision-making strategy, in which a high-level policy generates binary prompts to activate control units that have more impact on the task performance, while a low-level policy produces actions conditioned on these prompts. To promote behavioral diversity guided by different control-unit preferences, we integrate a diversity-driven objective into the model-based policy optimization process. FOCUS significantly outperforms existing methods on multiple visual control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a hierarchical approach for model-based reinforcement learning in which a high-level policy outputs binary prompts intended to focus exploration on certain action dimensions and a low-level policy outputs actions conditioned on these prompts.

### Strengths
- Intuitive approach solving a relevant problem.
- Strong performance across a suitable number of experiments and baselines.
- Well-written.

### Weaknesses
- The computational trade-off compared to "vanilla" DreamerV3 should be discussed more, especially in the context of B.4.
- The number of candidate trajectories is an important hyperparameter. The ablation appears to be based on a single run and the discussion is superficial. 
- The explanation in Sec. 4.4 for why FOCUS w/o high-level performs worse than DreamerV3 is not convincing.
- Minor: Figure 6 should be enlarged

### Questions
- Is the presented observation wrt. the impact of the number of candidate trajectories robust and can you provide insights on why a higher number of samples would lead to worse performance?

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
4

### Summary
This paper proposes FOCUS, a hierarchical model-based RL framework for visual continuous-control tasks with high-dimensional action spaces, where a high-level “preference policy” samples binary control-unit prompts (one per action dimension) and a low-level policy conditions on these prompts to focus exploration and execution on salient dimensions. Empirically, FOCUS outperforms strong baselines (DreamerV3, TD-MPC2, DrQv2, Director) on DMC Humanoid/Dog and MyoSuite hard tasks under matched interaction budgets, with ablations showing all components contribute. Main questions are robustness of the binary prompt design (vs. softer/structured sparsity), the need for predefined action factorization, and fuller reporting of compute costs; nonetheless, the approach is a practical step toward scalable MBRL in very high-dimensional actions.

### Strengths
1. The idea is well-motivated—only a sparse subset of joints often matters at a time—and the mechanism is clearly specified, including the Bernoulli-coded prompts and planner weighting/update rules.

### Weaknesses
1. Insufficient benchmarking. The authors only picked 4 tasks and didn't include Humanoidbench, which, to my knowledge, could be the most suitable benchmark for this paper.

2. Inappropriate setting. In my understanding, the most important contribution of this paper is identifying the significant action dimensions. It is not necessary to bind this paper to visual input, or at least, authors should consider the proprio inputs as one of the experimental settings.

### Questions
See above,

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
FOCUS is a hierarchical model-based RL method for high-dimensional continuous control. A high-level “preference” policy samples a binary prompt over control units, and a low-level policy outputs actions conditioned on that prompt. During Dreamer-style imagination, diversity and consistency regularizers make selected units behave differently and unselected ones similarly, so prompts actually influence actions. After that, the agent does preference-based SMC planning: it samples several prompts, rolls out short imagined futures with each fixed prompt, weights them by exponentiated advantages, and executes the first action, yielding stronger performance and lower decision latency than baselines on benchmarks.

### Strengths
1. The paper is generally well written.
2. The proposed method is novel and shows promising results compared to the baselines.

### Weaknesses
1. Prompts are sampled from independent Bernoulli variables, yet the paper claims to capture inter-unit correlations; the modeling and empirical effect of such dependencies are not clearly demonstrated.
2. The claim that prior knowledge about important dimensions can be incorporated is not evaluated with targeted experiments.

### Questions
1. How exactly are “control units” defined per task?
2. Could you explain the motivation for exponentiating the advantage when updating particle weights? This choice can increase variance. How do you address stability?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents FOCUS, a model-based reinforcement learning (MBRL) framework for high-dimensional continuous control that enhances efficiency by focusing on key action dimensions. It employs a hierarchical policy, where a high-level module samples binary prompts to select important control units, and a low-level policy generates conditioned actions. A diversity-driven objective promotes exploration, and a preference-based Monte Carlo planner improves decision quality. Experiments on DeepMind Control Suite and MyoSuite show FOCUS outperforming baselines like DreamerV3 and TD-MPC2 in sample efficiency and performance.

### Strengths
1. Well-written and clearly presented, concepts like preference prompts, interaction policy, and focused sampling are explained intuitively and supported by clear figures.

2. Target a core challenge in reinforcement learning: scaling model-based methods to high-dimensional action spaces.

### Weaknesses
1. It lacks a thorough discussion of prior studies that also tackle high-dimensional control challenges ([1], [2], [3]). A more comprehensive literature review would provide clearer connections to existing research and better highlight the paper’s contributions.

2. The two-level policy design, combined with diversity regularization and SMC-based planning, introduces additional structural and computational complexity compared to single-policy MBRL approaches. The trade-off between performance gains and computational cost is not clearly analyzed or quantified.

3. Although the framework’s effectiveness is demonstrated empirically,  it lacks formal theoretical justification regarding how preference prompts influence optimization dynamics, stability, or convergence. 

[1]  Schumacher, Pierre, et al. "DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems." The Eleventh International Conference on Learning Representations.

[2] Zhong, Dianyu, Yiqin Yang, and Qianchuan Zhao. "No prior mask: Eliminate redundant action for deep reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.

[3] Zhang, Wenbo, and Hengrui Cai. "Where to Intervene: Action Selection in Deep Reinforcement Learning." Transactions on machine learning research (2025).

### Questions
1. Could the authors provide more fine-grained statistics or visualizations of the selection frequency across action dimensions over time? This would help illustrate how the preference policy dynamically shifts focus among control units.

2. Could the authors elaborate on the time cost associated with world model learning versus environment interaction? A breakdown of computational cost would clarify where the main bottlenecks occur.

### Soundness
2

### Presentation
2

### Contribution
2
