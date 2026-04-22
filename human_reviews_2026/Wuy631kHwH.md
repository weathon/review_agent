# $\mu$P for RL: Mitigating Feature Inconsistencies During Reinforcement Learning

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
The maximal update parameterization ($\mu P$) has been influential in supervised and unsupervised learning, with fixed data distributions, owing to its ability to maintain feature learning across larger parameter scales. This causes more consistent learning dynamics and learned features across model sizes. In addition, optimal hyperparameters such as learning rate approximately transfer from small to larger models, minimizing the computational overhead of hyperparameter sweeps. However, it remains elusive if these benefits readily transfer to the reinforcement learning framework, where the model's learning dynamics are coupled to the shifting data distribution. Reinforcement learning agents must continually adapt to non-stationary data distribution shifts throughout training. We empirically study how two regimes of reinforcement learning agents under the "rich" CompleteP and "lazy" Neural Tangent Kernel (NTK) parameterizations affect hyperparameter transfer, feature and policy consistency. Ultimately, we show that agents trained using the CompleteP parameterization consequentially improves compute and reward efficiency compared to the NTK parameterization over 16 continuous control tasks and variants e.g. normalization and sparse rewards. Hence, we argue that adopting the CompleteP parameterization minimizes learning inconsistencies across model sizes to improve compute efficiency when scaling up.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how network parameterization affects learning consistency and computational efficiency in reinforcement learning (RL). It compares the Neural Tangent Kernel (NTK) parameterization with a proposed alternative, CompleteP, across a suite of continuous control tasks. The authors argue that RL’s non-stationary data distribution makes NTK scaling rules suboptimal, leading to instability when transferring hyperparameters across network widths and depths. CompleteP, in contrast, maintains more consistent feature learning and enables efficient scaling without costly hyperparameter sweeps. Empirical results on up to 16 continuous-control benchmarks show that CompleteP improves both reward efficiency and compute efficiency relative to NTK.

### Strengths
* *Timely and relevant topic*: Scaling laws and parameterization effects in RL are under-explored compared to supervised learning. The paper addresses an important gap with potential practical implications.
* *Comprehensive empirical analysis*: The authors evaluate multiple architectures, scaling factors, and optimization settings, and measure both reward and compute efficiency.
* *Clear empirical advantage*: CompleteP shows consistent improvements over NTK across tasks and widths, supporting its claim of better hyperparameter transferability.
* *Strong writing and organization*: The paper is clearly structured, with detailed appendices and well-designed figures summarizing trends in feature consistency.

### Weaknesses
* *Logical gap in motivation*: The paper attributes NTK’s failure in RL primarily to non-stationary data distributions, yet never establishes how non-stationarity causes NTK’s scaling rules to break down. Without a mechanistic or empirical link (e.g., gradient variance growth, kernel drift, or feature collapse under distribution shift), this reasoning remains speculative.
* *Mismatch between motivation and evidence*: The experiments focus on continuous-control benchmarks (mostly DeepMind Control Suite and MuJoCo), which feature relatively stationary dynamics and dense rewards. These settings do not strongly exhibit the non-stationarity that motivates CompleteP. Hence, the experimental results demonstrate better scaling efficiency, not necessarily stability under non-stationary data.

### Questions
* Could the authors clarify the precise mechanism by which RL’s non-stationary data affects NTK dynamics? Is the instability empirical (e.g., learning divergence) or theoretical (e.g., loss of kernel invariance)?
* Would CompleteP still provide benefits in explicitly non-stationary or multi-task settings (e.g. multi-agent environments), where environment distribution shifts are more severe?
* Are there diagnostic results (e.g., kernel alignment or feature CKA over time) that directly support the claim of improved “feature learning consistency”?
* How sensitive is CompleteP to optimizer choices or different scaling laws (e.g., under SGD versus Adam)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates whether feature-learning parameterizations (CompleteP/µP with depth scaling) improve consistency and efficiency in RL compared to lazy (NTK) scaling. Using PPO on 16 continuous-control tasks/variants, the authors show (i) learning-rate transfer across width/depth with CompleteP; (ii) more consistent feature evolution and seed-to-seed policy alignment; and (iii) better compute and reward efficiency (Pareto frontiers) at large scales. They provide kernel/CKA analyses on a synthetic probe set and report results with multiple widths/depths and seeds.

### Strengths
- Learning-rate transfer across width/depth with CompleteP reduces sweep cost—useful for scaling PPO on hard tasks
- Feature consistency: width-independent feature evolution and higher seed-alignment under CompleteP (CKA/eigenspectrum evidence).
- Efficiency gains: clear Pareto analysis showing lower compute to reach the same reward and higher reward at fixed compute on challenging tasks (e.g., Humanoid Maze).
- Comprehensive scaling study: multiple widths (4→2048), depths, and tasks; includes sparse-reward variants and ablations like orthogonal init / layer norm notes.

### Weaknesses
- Compute metric simplification. “Compute = params × grad steps × env steps” ignores optimizer-dependent costs and simulator variance; wall-clock with hardware notes would improve fairness.
- Baselines & breadth. Little empirical contrast with Standard Param (SP) at scale or with mean-field/$\mu$P variants beyond CompleteP; vision/pixel-based tasks are left for future work. Minor typos (“ComplteP”), a few caption clarifications needed.

### Questions
- How sensitive are results to optimizer (Adam vs. SGD) given the parameterization tables? Any changes to the scaling rules alter conclusions?
- Can you include a small sweep to confirm NTK’s best LR per depth doesn’t close the gap, and a SP baseline at select widths?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the application of the CompleteP parameterization (a variant of μP) to reinforcement learning agents, demonstrating that it mitigates inconsistencies in hyperparameter transfer, feature learning, and policy evolution across model widths and depths. The authors empirically show that CompleteP enables stable learning rate transfer, consistent feature representations, and improved compute/reward efficiency on 16 continuous control tasks compared to the NTK parameterization, arguing for its adoption to enhance RL scaling.

### Strengths
- Investigates the use of μP/CompleteP in RL, tackling non-stationary data with empirical evidence of stable hyperparameter transfer and feature consistency.
- The claims are well-supported by extensive experiments on 16 continuous control tasks.

### Weaknesses
- Lack of Theoretical Justification for RL

The paper provides no new theoretical backing for why $\mu P$'s benefits, designed for stationary data, transfer so robustly to the non-stationary RL setting. It relies on borrowing the SL theory and providing empirical validation.

- Ambiguous definition of CompleteP

The method is introduced conceptually (“μP variant adapted for RL”) but without a formal scaling equation or algorithmic pseudocode.
It is unclear how its initialization and learning-rate rules differ from μP in exact terms.

- Questionable Necessity for Large-Scale RL

The paper assumes that scaling benefits from LLMs apply to RL, but RL often works with small networks—unclear if large models are frequently needed beyond niche cases like RLHF in LLMs.

- Minor Points

The paper contains presentation flaws, such as a missing reference "Appendix ??" on L172, which should be corrected.
OpenReview title (“μP for RL”) and PDF title (“CompleteP for RL”) mismatch.

### Questions
- Why focus on large-scale RL when most agents use small networks? How does this align with RL's sample inefficiency versus LLM scaling?
- Could you extend mean-field theory to prove CompleteP's advantages in RL's dynamic distributions?

### Soundness
3

### Presentation
3

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
This paper systematically investigates how Neural Tangent Kernel (NTK) and CompleteP parameterizations influence hyperparameter transfer, feature consistency, and policy consistency within the context of online Reinforcement Learning (RL). The authors demonstrate that adopting the CompleteP parameterization effectively mitigates scaling inconsistencies, resulting in consequential improvements in compute and reward efficiency compared to the lazy NTK regime across continuous control tasks.

### Strengths
- The authors conduct extensive experiments, consuming significant computational resources to compare the models' learning dynamics up to the point of achieving final rewards in various environments.
- Authors provide empirical evidence on the scaling effects of NTK and CompleteP in the non-stationary RL setting , where these parameterizations have been non-trivially studied before.

### Weaknesses
- The reliability of the consistency findings (e.g., parameter transfer, feature/policy consistency) is constrained as the detailed analysis and visualizations are often restricted to only one or two specific environments (e.g., HalfCheetah, HumanoidMaze, PandaPickCube). A more comprehensive, universally quantified summary across all tasks would strengthen the generalizability of the conclusion.
- The work primarily compares the two parameterizations (NTK vs. CompleteP) which are specific scaling theories. It lacks comparison against established baselines from the general RL community, such as methods designed explicitly to mitigate seed variance or enhance plasticity (e.g., using orthogonal initialization or layer normalization as a baseline in core experiments). This makes it difficult to ascertain the absolute benefit or difference in learning dynamics relative to standard industrial practices.
- The related work section is sparse regarding modern techniques for resolving RL inconsistencies. The authors should enrich this section.

### Questions
In terms of sample efficiency and compute efficiency, does training using the NTK and CompleteP parameterizations demonstrate superiority compared to other methodologies (i.e., standard RL practices like [1])?

[1] Lee et al, "SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning"

### Soundness
2

### Presentation
1

### Contribution
2
