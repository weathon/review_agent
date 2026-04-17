# Direct Preference Optimization for Primitive-Enabled Hierarchical RL: A Bilevel Approach

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
Hierarchical reinforcement learning (HRL) enables agents to solve complex, long-horizon tasks by decomposing them into manageable sub-tasks. However, HRL methods face two fundamental challenges: (i) non-stationarity caused by the evolving lower-level policy during training, which destabilizes higher-level learning, and (ii) the generation of infeasible subgoals that lower-level policies cannot achieve. To address these challenges, we introduce DIPPER, a novel HRL framework that formulates goal-conditioned HRL as a bi-level optimization problem and leverages direct preference optimization (DPO) to train the higher-level policy. By learning from preference comparisons over subgoal sequences rather than rewards that depend on the evolving lower-level policy, DIPPER mitigates the impact of non-stationarity on higher-level learning. To address infeasible subgoals, DIPPER incorporates lower-level value function regularization that encourages the higher-level policy to propose achievable subgoals. We introduce two novel metrics to quantitatively verify that DIPPER mitigates non-stationarity and infeasible subgoal generation issues in HRL. Empirical evaluation on challenging robotic navigation and manipulation benchmarks shows that DIPPER achieves upto 40% improvements over state-of-the-art baselines on challenging sparse-reward scenarios, highlighting the potential of preference-based learning for addressing longstanding HRL limitations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose DIPPER, a hierarchical reinforcement learning (HRL) framework that formulates HRL as a bi-level optimization problem. In this formulation, the higher-level “teacher” policy and lower-level “student” policy are jointly optimized while explicitly modeling their interdependence. To mitigate the non-stationarity caused by the evolving lower-level policy, the authors leverage Direct Preference Optimization (DPO) to train the higher-level policy on stationary human preference data rather than environment rewards. Additionally, they introduce a lower-level value function regularization term that encourages the higher-level policy to propose feasible and achievable subgoals.

### Strengths
The paper provides a rigorous bi-level mathematical formulation of hierarchical reinforcement learning (HRL).

The use of real human preference data is commendable. Conducting human-in-the-loop experiments introduces significant complexity and time overhead, yet provides stronger evidence for the method’s practical relevance

The authors empirically evaluate whether DIPPER actually addresses the two challenges they identify, non-stationarity and infeasibility of subgoal, rather than assuming these issues are solved. This careful, hypothesis-driven evaluation is a strong aspect of the paper.

They explore how varying key hyperparameters (λ, β) affects performance, showing an understanding of what aspects drive DIPPER’s behavior.

### Weaknesses
**Stationarity of Human Preferences**

The paper assumes that human preferences mitigate non-stationarity, yet human preferences themselves are not necessarily stationary.
Prior work suggests that people’s judgments can change over time [1], particularly when confidence is low. 

Question: How would DIPPER handle cases where human preferences shift if the same trajectories are presented multiple times? Would the algorithm adapt?

**Missing Related Work**

The related work section omits relevant approaches in automatic curriculum learning that use bi-level optimization to generate goals for lower-level policies.
Examples include Narvekar and Stone (2019) [2] and Muslimani et al. (2023) [3]. 
These should be discussed to contextualize DIPPER within existing curriculum learning frameworks.

**Failure Cases and Environment Suitability**

DIPPER fails in the simple maze task. The paper does not explain why.

Question: What types of environments does DIPPER work best in?

In environments where DIPPER outperforms baselines (Pick & Place and Push), success rates remain low (around 30%).

Question: Why does performance plateau at this relatively low level? Would longer training or additional preference data improve results?


**Baselines**

The authors remove the HER component from PIPER “for fairness,” but this substantially alters the original algorithm.
This change may make the comparison less meaningful.

Question: How would DIPPER perform against the original PIPER implementation (with HER) as described in the source paper?

The authors mention tuning DIPPER’s hyperparameters using grid search, but it is unclear whether the same effort was made for the baselines.
 This introduces a potential unfair advantage if baselines are not tuned equivalently.

**Evaluation Metrics and Interpretation**

The “lower Q-function metric” measures Q-values for subgoals proposed by the higher-level policy. However, this could be inflated if the higher policy proposes easy subgoals.

Question: Do the proposed metrics actually capture whether DIPPER learns progressively more difficult subgoals over time?


**Human Preference Data**

The paper provides limited details about the human annotators.

Questions:

Who were the annotators (experts or laypeople)?

Were they trained or provided with guidance?

What were their demographic backgrounds?

**Most importantly, was this data collection ethics-approved?**

These details are crucial to assess the reliability and reproducibility of the preference data.


**Sources:**
[1]Visser et al (2005) https://pubmed.ncbi.nlm.nih.gov/16022057/

[2] Narvekar and Stone (2019) https://arxiv.org/pdf/1812.00285

[3] Muslimani et al (2023) https://arxiv.org/pdf/2204.11897

### Questions
I included my questions in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present DIPPER, a hierarchical RL method that directly optimizes environment reward and human preferences. Given human preferences, the authors derive a bi-level optimization objective that train the policy to increase the likelihood of the preferred trajectories with the high-level policy while simultaneously updating the low-level policy to try to maintain value improvement for the same objective. The specific objective is derived from DPO, and helps mitigate non-stationarity common to most HRL methods.

### Strengths
**Comparisons:** The authors compare against a wide array of relevant baselines, demonstrating great performance against said baselines on manipulation tasks especially. 

**Motivation:** Non-stationarity in HRL is a big issue and this paper presents a well-motivated solution for it.

**Experiments:** The experiments are performed on tasks well-suited for HRL, and the analysis on goal distance prediction against HIER and HAC demonstrates that DIPPER’s objective encourages sampling reachable goals for the lower-level policy.

**Clarity:** The paper is overall quite clear and the walkthrough of how to obtain the objective was both interesting and easy to read.

### Weaknesses
**Figures**: All of the results figures have small font that make them harder to read, and are also clearly image files put into overleaf instead of vectorized PDFs. The results figures should have thicker lines for each baseline, more spacing between baselines on the legend (in fig 2), and larger font for the x and y axes labels and ticks labels.

**Annotation cost:** As authors admit, there is a high annotation cost to obtaining labels with human prediction. 

**Experiments:** Given the fact that the authors have human annotations, it seems that more challenging tasks could’ve been demonstrated in the experiments. This is not a reason to reject the paper, however I will list this as a slight weakness of the paper.

**Minor issues:**

- There’s related work which also mitigates non-stationary and infeasible subgoal generation by modeling *intrinsic* options: e.g., https://arxiv.org/abs/2101.06521, some comparison against this work and any follow-ups in the related works section would be beneficial
- There’s also related work on unifying low level and high level policy optimization: https://sites.google.com/view/hippo-rl
- L363: “we maintain empirical consistency across all baselines to ensure fair comparisons” — what does this actually mean? be specific here

### Questions
Instead of a subgoal distance metric (fig 3), why not directly measure the success rate of the lower level policy’s ability to reach goals? This is the actual metric that matters for addressing the subgoal feasability problem, right?

Re: annotation cost, prior work in reward learning has demonstrated that VLMs can be useful for obtaining preferences in place of humans: https://rlvlmf2024.github.io/, have the authors looked into this or tried out similar approaches at a small scale?

### Soundness
3

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
2

### Summary
This paper has proposed a novel hierarchical reinforcement learning framework, which employs preference-based learning in the high-level policy to mitigate the non-stationary issue in hierarchical policy learning. The proposed framework is evaluated in a set of simulated long-horizon tasks.

### Strengths
1.	The research problem of non-stationarity is significantly important in the hierarchical reinforcement learning domain.

### Weaknesses
1.	The technical contribution of this work is low, which is a direct application of preference-based learning to high-level policy optimization.

2.	Experiments are limited to simulated environments. Demonstrations in real robots or transfer scenarios would significantly strengthen the empirical validation.

### Questions
1.	How sensitive is DIPPER to the scale or bias of human preferences? Would synthetic or learned preferences generalize effectively?

2.	Could DIPPER be extended to fully autonomous preference learning (e.g., self-generated ranking signals) without human input?

### Soundness
2

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
4

### Summary
This introduces a hierarchical RL framework (DIPPER) aimed to tackles two HRL issues: 1. non-stationarity of the higher-level environment and 2. infeasible subgoal generation. The framework is composed of a high level goal-conditioned HRL with Direct Preference Optimization (DPO) and standard RL at the low level. This work formalize HRL as a constrained bi-level optimization problem. 

Experiments on four challenging sparse-reward navigation and manipulation tasks (random mazes, pick-and-place, push, and Franka kitchen) show that DIPPER substantially outperforms both standar and hierarchical baselines, including prior preference-based HRL methods. This work also uses subgoal distance and low-level Q value to empirically support claims about reduced non-stationarity and improved subgoal feasibility.

### Strengths
1. Originality: The paper gives a clean bi-level optimization formulation of goal-conditioned HRL, rewriting the hierarchy as an upper-level problem with a constraint that the lower-level policy is (locally) optimal. In contrast to PIPER, which learns a reward model and then uses RL/RLHF on top of it, this work directly trains the high-level policy via DPO on a preference dataset, which allows the framework to avoid modeling explicit rewards and avoids having a second RL loop. 

2. Significance: This paper provides a preference-based, DPO-driven recipe with a clear regularizer derived from a bi-level formulation that is not present in prior HRL work. The experimental results also suggest an improvement of around 40% in success rates on harder sparse-reward tasks, compared to state-of-the-art baselines that already use preferences and primitive-informed regularization. 

3. Clarity: The authors provide pseudo-code for their framework and clearly describe the alternation between preference-trajectory collection and updates of the high-level DPO objective, low-level value function, and low-level SAC policy. The introduction also clearly states the two central issues (non-stationarity and infeasible subgoals). 

4. Quality: The derivation from bi-level HRL to a constrained problem, then to a Lagrangian, and finally to a primitive-regularized DPO objective is detailed and internally consistent.

### Weaknesses
1. The bi-level formalization in this paper is in some sense rephrasing standard HRL coupling (optimal lower-level policy conditioned on higher-level subgoals) in the language of constrained optimization. The authors could consider adding more analysis of the resulting bi-level problem such as convergence guarantees, regret bounds, and when the relaxed constraint and approximate value function yield near-optimal behavior. 

2. This paper mentions that one benefit of DPO over RLHF is computational simplicity and stability. This statement is not quantified in terms of training time, memory etc for DIPPER vs RLHF-based alternatives (such as PIPER or a simple RLHF high-level baseline). ALso, sample efficiency is mainly reflected in success-rate vs timestep plots. There could be a more fine-grained analysis of preference-query complexity or number of environment transitions until convergence.

3. The paper claims their framework works on “long-horizon complex robotic tasks”, however the four environments are standard MuJoCo-style navigation/manipulation tasks (with some challenge enhancements like random mazes and sparse kitchen rewards). The work could consider adding vision-based setting with more complex hierarchical structures. Prior works like CRISP/PEAR do report real-robot experiments with more challenging perception and dynamics.

### Questions
1. Are preferences generated from ground-truth reward (e.g., via Bradley–Terry over cumulative environment reward), or from some kind of synthetic labeling procedure? How many preference queries are used per environment, and how does performance scale with the number of preferences?

2. The work approximates V^{L*} by updating V^L_m for m gradient steps between policy updates. How sensitive is DIPPER to m? Intuitively, if V^L_m is poor in the earlier portion of training, the regularizer might be actively pushing the high-level toward bad subgoals. Is there any safeguard (e.g., annealing λ) or empirical evidence that this does not happen?

3. Some suggestions on the presentation: 
The work could benefit from having a more clear overview diagram that explicitly shows the bi-level view and how it leads to the primitive-regularized DPO block, with arrows from lower-level value function to the high-level DPO loss.
In the experiments, it would help to explicitly list the baselines and their key differences in a table, including whether they use preferences, whether they use primitive-informed regularization, and whether they are hierarchical or flat.

### Soundness
3

### Presentation
3

### Contribution
3
