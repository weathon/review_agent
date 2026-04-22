# Improving Safe Offline Reinforcement Learning via Dual-Guide Diffuser

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 4, 8

## Abstract
In offline safe reinforcement learning (OSRL), upholding safety guarantees while optimizing task performance remains a significant challenge, particularly when dealing with adaptive and time-varying constraints. 
While recent approaches based on actor-critic methods or generative models have made progress, they often struggle with robust safety adherence across diverse and dynamic conditions. 
For diffusion models specifically, a key bottleneck is the reliance on unreliable cost classifiers for safety guidance.
To address these limitations, we propose SDGD (Safe Dual-Guide Diffuser), a novel framework that decouples safety and performance optimization. 
SDGD leverages classifier-free guidance (CFG) to strictly enforce cost constraints while simultaneously using classifier guidance (CG) to steer generation towards high-reward outcomes.
This dual-guide mechanism robustly handles cost limits that change dynamically within a single episode.
We provide an error bound of the estimation on reward and cost, offering performance and safety guarantees. 
Extensive evaluations on the DSRL benchmark demonstrate that SDGD establishes a new state-of-the-art, achieving safety in 94.7% of tasks (36/38). 
Crucially, our method extends the aggregate Pareto frontier in the reward-cost space, achieving a superior trade-off in the safe region.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SDGD, an offline safe RL method that employs a diffusion model as a planner. SDGD conditions the diffusion process on the cost target and applies the reward target as classifier guidance for planning. Extensive experiments on the OSRL benchmark demonstrate the effectiveness of the proposed approach.

### Strengths
- In real-world deployments with strict safety requirements, safe RL methods based on TD learning are rarely feasible, and planning-based safe control approaches such as MPC are typically preferred. This paper similarly explores recasting traditional TD-learning-based safe RL into a planning framework, which I believe is a promising research direction.
- The experimental results clearly demonstrate significant advantages of the proposed method over previous TD-learning-based approaches.
- The design choice of using classifier-free guidance to control cost and classifier guidance to control reward in the generation process is both reasonable and well-motivated.

### Weaknesses
- Some implementation details in the paper are rather vague, such as the hyperparameters involved in trajectory representation and those related to the diffusion process. Moreover, since no actual code implementation is provided (either in the main text or supplementary materials), the reproducibility of the reported results cannot be fully ensured.
- The experiments in this paper use different hyperparameters for different tasks, whereas baselines such as FISOR keep their hyperparameters fixed across tasks, leading to a potentially unfair comparison. In fact, based on my own reproduction using the official FISOR code, I observed that if FISOR were allowed to use task-specific expectile regression parameters (e.g., setting $\tau=0.8$ or $\tau=0.7$ for tasks with safety violations in the current experiments), it could also achieve safe performance across all tasks in Safety Gymnasium.

### Questions
No other questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem of learning a safe policy that can handle varying cost limits from an offline dataset. The proposed method, called SDGD uses classifier-free guidance to learn a safe diffusion policy as a base model, and uses classifier guidance to steer generation towards high-reward outcomes. The algorithm is tested on the DSRL benchmark against offline safe RL baselines, and results show that SDGD learns constraint-satisfying and high-reward policy.

### Strengths
1. The experiments are extensive, covering diverse tasks on a standard benchmark and including a broad range of baselines.
2. The writing of the paper is clear.

### Weaknesses
1. It is unclear what is the specific problem this paper tries to solve and why existing methods cannot solve it.
2. The proposed method has a flaw: reward guidance undermines safety. The authors use feasible trajectory re-labeling to deal with this problem, but it only considers short-term safety and cannot guarantee long-term safety.

### Questions
1. What is the specific problem this paper tries to solve? Is it how to handle varying cost limits or how to learn a safe and high-performance policy under a given cost limit? Why existing methods, including generative methods and diffusion models, cannot solve this problem?
2. In the introduction, the authors claim that making decisions step-by-step is suboptimal. However, in MDP, the optimal policy is exactly a step-by-step state feedback policy. Why is it suboptimal?
3. Why the cost classifier used by existing diffusion-based methods is unreliable? The proposed method in this paper also uses a classifier for reward and short-term safety guidance. Is this classifier also unreliable?
4. Reward guidance makes the policy unsafe. The feasible trajectory re-labeling only considers short-term safety. How can long-term safety be guaranteed in this case? In addition, adding penalty to the reward is essentially considering cost in the classifier, which means safety is still guided partly by the classifier.
5. In the experiment part, the claim that SDGD significantly outperforms all baselines is unfair. It still lies on the existing Pareto frontier from Figure 4, which seems a new trade-off between reward and cost.

### Soundness
1

### Presentation
3

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
SDGD (Safe Dual-Guide Diffuser) is a diffusion-based method for offline safe RL that decouples objectives: classifier guidance steers generation toward high reward, while classifier-free guidance conditions on target costs to enforce safety. It adds a feasible-trajectory relabeling scheme that penalizes unsafe prefixes so the reward signal doesn’t lure generation into high-cost regions. The method adapts to time-varying cost limits, and achieves state-of-the-art safety with strong returns on the DSRL benchmark.

### Strengths
- Works under different and time-varying cost limits without retraining.

- Strong empirical results on DSRL, staying within limits on 36/38 tasks and achieving top rewards on 21 tasks with the cost limit 10.

- Decoupling safety thorough (CFG) for costs and (CG) for rewards and feasible-trajectory relabeling, improves performance.

### Weaknesses
- The core claim that Classifier Guidance (CG) is unreliable for cost constraints and is the key bottleneck is not definitively proven within the SDGD model. The authors did not provide the direct ablation showing the performance degradation when swapping the roles (i.e., replacing the reliable Classifier-Free Guidance for cost with CG for cost).

- The choice and benefits of the Sparse-State Dense-Action (SS-DA) trajectory representation are mentioned but not clearly motivated or quantitatively ablated to demonstrate its contribution to the final performance.

- The Feasible Trajectory Re-labeling (FTR) mechanism employs an overly strict, binary definition: a trajectory is "Infeasible" if the cumulative cost in the first $f$ steps is merely greater than zero ($\sum_{t=0}^{f-1}c_t > 0$). This rigid rule is independent of the overall cost limit ($l$) and may conservatively discard potentially safe, high-reward trajectories that incur a small, necessary cost early in the episode.
- The paper’s Pareto-optimality claim is imprecise: the result is computed from averaged Normalized Reward and Normalized Cost over 38 heterogeneous DSRL tasks, and SDGD is said to set a “new frontier” at that aggregate point. In formal multi-objective optimization, Pareto optimality must be established within a single task (problem instance), where no policy can improve reward without worsening cost for that same instance. Averaging outcomes across tasks with different dynamics and scales cannot certify Pareto optimality for any one task; it only indicates a better aggregate trade-off.

- All empirical results, including the main performance table and ablation/adaptation figures, lack standard deviation or standard error, which is essential to assess the robustness and variance of the method across different seeds.

- The lack of specific, task-dependent optimal guidance scales ($s$ and $w$), which the authors acknowledge are sensitive, and the absence of shared code hurts reproducibility.

### Questions
- Please check weaknesses
- Since SDGD, CDT, TREBI, CAPS, and CCAC are all designed to be adaptive using a single model, please provide a table showing the aggregate results for all 38 tasks under the $L=20$ and $L=30$ cost limits, in addition to the $L=10$ results already shown in Table 1. This will provide empirical support that SDGD maintains its SOTA results across a range of safety requirements.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes SDGD (Safe Dual-Guide Diffuser) to target diffusion-based trajectory generation for offline safe RL. The key idea is to decouple safety and performance during the generation by using dual guidance mechanics: classifier-free guidance to condition on cost limits, and classifier guidance to steer toward high reward. It also explores a technique called Feasible Trajectory Re-labeling (FTR) to prevent reward gradients from pulling trajectories into unsafe regions and getting trained on preventing spurious high-reward/high-cost correlations. The overall method comes with strong theoretical guarantees on safety and optimality. 

Experiments show that on the 38 DSRL tasks, SDGD reports outperformance over several baselines in safety and performance, is able to do almost all tasks safely, and achieves a pareto-optimal reward cost frontier. Authors also show that it adapts well to changing (even intra-episode) cost limits.

### Strengths
Although diffusion based and classifier guidance has been explored previously, the dual-guide approach is novel for OSRL, and addresses known limitations (unreliable cost classifiers, sequential errors) in prior works by conditioning costs directly. 

The FTR heuristic also seems like a simple yet effective safety mechanism. It is 
“the specific cost limit applied for the current generation does not predetermine the limits for subsequent generations” - this is a noteworthy strength of the framework. The strong performance in handling intra-episode varying constraints also demonstrates this amply.

The authors perform comprehensive evaluation with several baselines, and ablations (covering guidance, scale, FTR). The flow is logical and the writing is easy to understand.

The paper provides clear inequalities bounding both cost violation and reward loss, and links safety guarantees to dataset coverage. This is somewhat rarely seen in OSRL prior works.

### Weaknesses
Some theoretical assumptions  concentration bounds) may be standard but may not hold in less data regimes. Maybe I missed it, but it would be good to see some validation on how they hold in simulated datasets in the paper or some real world ones. Some small discussion around the bounds’ behavior when these break would also be insightful.

“we evaluate all methods under a strict, uniform absolute cost limit of 10 across all tasks” - the reasoning behind this is not clear. It may also hide cost scales issues across environments, and seems a bit overfit to the framework.

Relatively minor:
Variance over the evaluations/seeds can also be included with the results. The paper notes 2.1 Hz generation but compute/memory comparison would also be insightful. The framework and techniques like FTR is somewhat hyperparameter sensitive, which limits plug-and-play use.

### Questions
Can forms of FTR be applied to other baselines? It would be particularly interesting to see the performance if so.

In case of model failure (I believe in ~5% of the tasks), have the authors done any error analysis on the possible causes or common patterns, like poor dataset coverage, complexity etc.? Maybe it highlights more limitations of the model.

### Soundness
3

### Presentation
4

### Contribution
4
