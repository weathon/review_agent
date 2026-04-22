# Learning Massively Multitask World Models for Continuous Control

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
General-purpose control demands agents that act across many tasks and embodiments, yet research on reinforcement learning (RL) for continuous control remains dominated by single-task or offline regimes, reinforcing a view that online RL does not scale. Inspired by the foundation model recipe (large-scale pretraining followed by light RL) we ask whether a single agent can be trained on hundreds of tasks with online interaction. To accelerate research in this direction, we introduce a new benchmark with 200 diverse tasks spanning many domains and embodiments, each with language instructions, demonstrations, and optionally image observations. We then present Newt, a language-conditioned multitask world model that is first pretrained on demonstrations to acquire task-aware representations and action priors, and then jointly optimized with online interaction across all tasks. Experiments show that Newt yields better multitask performance and data-efficiency than a set of strong baselines, exhibits strong open-loop control, and enables rapid adaptation to unseen tasks. We release our environments, demonstrations, code for training and evaluation, as well as 200+ checkpoints.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an architecture to perform online RL on a bigger scale, training one policy for many different tasks. The learning algorithm requires as a first step a behavioural cloning approach to warm-start the network and make the exploration problem simpler. It then performs online RL, outputting actions after an MPC planning step trough a world model. It hence combines model-based and model-free RL approaches.

### Strengths
The paper presents good results on continous  control benchmarks which is still an interesting problem. Especially training one policy over this big variety of tasks is interesting to see. The algorithm is presented clearly and is easy to follow, and the model-based MPC aspect of it is interesting. I really appreciate the effort of the authors of making the code and the checkpoints accessible, this makes it possible to reproduce the results and build on-top of them.

### Weaknesses
My major concern is the applicability of this to real-world continuous control problems. While in simulation the results look good, it requires over 100M steps to train this policy which would be unfeasable on a real-world application. I also think the paper would benefit from ablating the usefulness of the different components - specifically interesting would be to understand how useful is the mpc planning is, how much learning of the world model helps performance as well as  how much does the initial bc training helps.

### Questions
how useful is the mpc planning? 
how much does learning of the world model helps performance?
how much does the initial bc training help?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the scalability challenge of online RL for continuous control by introducing MMBench, a large-scale benchmark spanning 200 diverse tasks across 10 domains (e.g., DMControl, Meta-World, Atari, and a newly proposed MiniArcade). Each task includes language instructions, demonstrations, and multi-modal observations. The authors also propose Newt, a language-conditioned multitask world model built upon TD-MPC2. Newt first pretrains on demonstrations to obtain task-aware representations and action priors, then performs online optimization across tasks with architectural refinements and action supervision. Experiments show that Newt outperforms several baselines (BC, PPO, FastTD3) in multitask performance and exhibits generalization to unseen tasks.

### Strengths
The paper has several notable merits:
- MMBench provides a unified framework across 10 heterogeneous domains with consistent data handling and language-conditioned tasks.
- The paper introduces reasonable design choices such as discrete regression for reward/value prediction and per-task discount factors, with comprehensive ablations supporting their impact.
- The paper is well-organized, figures effectively illustrate key results, and open-sourced resources (200+ checkpoints, 4000+ demos) significantly enhance reproducibility.

Overall, the paper represents a solid step toward scalable, general-purpose control systems.

### Weaknesses
### 1. Novelty concerns in core contributions

While MMBench contains 200 tasks, most of them are directly inherited from existing benchmarks (e.g., DMControl, Meta-World). This limits its novelty compared to benchmarks such as ManiSkill3, which introduces fundamentally new task paradigms. Similarly, Newt—although incorporating CLIP/DINOv2 encoders and demonstration conditioning—builds incrementally upon TD-MPC2, without a clear paradigm shift.

### 2. Missing analysis of task scalability

The paper emphasizes scaling to hundreds of tasks but lacks quantitative analysis on scaling behavior. For example, performance is not evaluated as the task count increases (e.g., 50 → 100 → 200 tasks), leaving unclear whether multitask training indeed benefits from more tasks. Moreover, since MMBench spans disjoint domains (e.g., Atari vs. Meta-World), the paper should analyze cross-domain transfer—whether training on visually distinct domains interferes with or enhances learning in others—and include ablations comparing full multitask vs. domain-specific training.

### 3. Incomplete baselines in MBRL

Experimental comparisons primarily focus on model-free RL and behavioral cloning baselines, while omitting competitive model-based RL counterparts. In particular, DreamerV3, which is explicitly designed for multitask continuous control, is absent, as is a multitask-adapted TD-MPC2 baseline (used for demo collection). Without these, it is difficult to attribute performance gains to the proposed architectural innovations rather than inherited advantages from TD-MPC2.

### 4. Insufficient evaluation under state-limited conditions

Most evaluations assume access to full low-dimensional states, which is unrealistic in real-world control settings. The paper lacks experiments under state-limited or purely visual conditions (e.g., partial observations or agent-only states), which would better demonstrate robustness and practical applicability.

### Questions
1. The paper employs masking to handle inconsistent action/state dimensions. Could masking lead to sparse gradient updates or optimization inefficiencies for high-dimensional action spaces (e.g., humanoid control with 50+ joints)? Have you explored task-specific action embeddings or shared latent action representations as scalable alternatives?

2. Table 3 reports Newt’s training time, but it is unclear whether this includes the cost of training 200 single-task TD-MPC2 agents used for demonstration generation. Please clarify whether the total computation time includes both stages (demo collection + multitask training) to ensure fair comparison and full cost transparency.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MMBench, a multitask RL benchmark containing 200 different control tasks across 10 domains. TDMPC2 agents on each single task are trained to collect expert demonstrations, while both the model checkpoints and demonstrations are open-source. In addition, this paper substitute language descriptions for task indices for better distinguish among tasks, which makes Newt,  a multi-task world model based on TD-MPC2. Experiments show comparable results with a strong baseline FastTD3.

### Strengths
1. This paper proposes a benchmark which integrates domains that are popularly studied in the RL community, releasing single-task checkpoints and dataset which is significant not only for multi-task RL but also offline, O2O and continuous RL research.
2. The empirical results show advantages over baselines on ManiSkill and DMControl. 
3. Model info such as training time, model architecture is detailed presented.
4. The figures are well drawn and easy to understand.

### Weaknesses
1. There is no preliminary description so the problem setting confuses me at the beginning. In the multi-task RL setting a task label $n$ should be added to the original $(s, a, s', r)$ but in Line.275 there is only $(s, a, r)$. 

2. Newt only shows performance boost over baselines in DMC and Maniskill out of all 10 domains. In Meta-World, MuJoCo, Box2D, Robodesk and Atari it's just on par with FastTD3, while in OGBench and MiniArcade it's on par with behavior cloning. 

3. Selected baselines are not strong enough. FastTD3 is a strong baseline and in its original paper is compared with strong model-based baselines such as TDMPC2 and Dreamerv3, but it mainly reports results on humanoidbench, mujoco playground and Issaclab, neither of them are included in MMBench. Moreover, as long as Newt is built upon TDMPC2, it surprises me that TDMPC2 is not listed as a baseline in this paper.

### Questions
1. To make it clear, is there only one agent being trained to interact with all 200 environments and collect online trajectories?

2. What does the "Language Instructions: None" refer to in Line 399-405? Is there a task index provided for each trajectory?

3. Is there any results that support the claim in Line 147 that one-hot encoding limits the potential for transferring to unseen tasks?

### Soundness
3

### Presentation
2

### Contribution
3
