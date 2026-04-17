# Beyond Shallow Behavior: Task-Efficient Value-Based Multi-Task Offline MARL via Skill Discovery

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
As a data-driven approach, offline MARL learns superior policies solely from offline datasets, ideal for domains rich in historical data but with high interaction costs and risks. However, most existing methods are task-specific, requiring retraining for new tasks, leading to redundancy and inefficiency. To address this issue, we propose a task-efficient value-based multi-task offline MARL algorithm, Skill-Discovery Conservative Q-Learning (SD-CQL). Unlike existing methods decoding actions from skills via behavior cloning, SD-CQL discovers skills in a latent space by reconstructing the next observation, evaluates fixed and variable actions separately, and uses conservative Q-learning with local value calibration to select the optimal action for each skill. It eliminates the need for local-global alignment and enables strong multi-task generalization from limited, small-scale source tasks. Substantial experiments on StarCraft II demonstrate the superior generalization performance and task-efficiency of SD-CQL. It achieves the best performance on $\textbf{13}$ out of $14$ task sets, with up to $\textbf{68.9}$% improvement on individual task sets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the problem of multi-task offline multi-agent reinforcement learning (MARL), where agents must learn policies that generalize across multiple tasks using only static offline datasets. Existing offline MARL approaches are typically task-specific, requiring retraining when task configurations (e.g., number of agents) change, leading to inefficiency and poor generalization.
To address this, the authors propose Skill-Discovery Conservative Q-Learning (SD-CQL), a value-based algorithm that integrates observation-encoding-based skill discovery and conservative policy optimization to enhance task efficiency and cross-task generalization.
The proposed method is evaluated on 14 StarCraft II offline task sets derived from the SMAC benchmark, covering both Multi-to-Multi (training and testing on multiple tasks) and One-to-Multi (training on one task and testing on unseen ones) scenarios. Across datasets of varying quality, SD-CQL achieves the best performance on 13 out of 14 task sets, showing up to 68.9% improvement over existing baselines.

### Strengths
- The paper explores the under-investigated yet practical problem of multi-task offline MARL. While offline MARL itself has recently gained traction, the extension to an offline multi-task formulation—where agents must generalize across tasks with varying numbers of agents and configurations—remains relatively unexplored. 

- From an originality standpoint, the paper contributes a skill-based framework that moves beyond behavior-cloning-based action decoding. Prior skill-discovery methods in RL often relied on direct imitation from latent skills to actions, which limited their optimality and generalization. 

- The experimental evaluation is extensive, spanning 14 task sets in the StarCraft II SMAC benchmark. The results are consistently strong, with SD-CQL outperforming prior baselines in 13 out of 14 task sets, and showing large relative improvements.

### Weaknesses
1. Questionable definition of skill via next-observation prediction:
The authors claim z represents "skills" (high-level decision patterns), but the learning mechanism is simply next-observation reconstruction. This is not skill learning—it's observation representation learning.
- Traditional skill-based RL defines skills as temporally extended action sequences or sub-policies.
- SD-CQL's z is extracted by predicting o_{t+1} from o_t, which is a standard technique for learning task-relevant observation features, not for capturing behavioral patterns
- Existing skill discovery methods (even task-agnostic ones) explicitly model action sequences or behavioral diversity, not just observations
The paper lacks justification for why observation encoding equals skill representation. Without modeling action sequences, calling z a "skill" contradicts established RL terminology.

2. Task-Agnostic Skill Discovery vs Task-Dependent Action Selection:
The authors emphasize "task-agnostic" skill discovery, but action selection is task-dependent.
- z is learned without rewards (task-agnostic)
- But Q(tau, a | z) is optimized with task-specific rewards (Eq. 5)
How does this enable generalization? It seems generalization comes from the shared encoder/Q-network architecture, not from the skill discovery mechanism itself. The interaction between these components is unclear.

3. Section 3 is Too Implementation-Heavy:
Section 3 focuses excessively on implementation details (entity decomposition, variable action spaces, network architectures) rather than conceptual contributions:
- Section 3.2.1 largely repeats CQL from Section 2.2
- Separate Q-networks (Eq. 4) is an engineering choice for variable action spaces, not a core contribution

### Questions
Q1: Skill Definition:
Why is next-observation reconstruction equivalent to learning "skills" (decision patterns) rather than just compressed observation features?

Q2: Source of Generalization:
Which component drives generalization: the shared observation encoder, the skill vectors z, or local value calibration? Can you provide ablations isolating each?

Q3: "Task-Agnostic" Clarification
Skills are learned without rewards but evaluated with task-specific Q-values—in what sense are they truly task-agnostic?

I would also like to hear the authors' thoughts on the weaknesses raised above.

### Soundness
2

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
This paper tackles task-specific inefficiency in multi-task offline MARL. It proposes Skill-Discovery Conservative Q-Learning (SD-CQL), a value-based algorithm designed to learn generalizable policies from limited offline data. The method's core components are: 1) A skill discovery module usingnext local observation reconstruction to learn continuous skill vectors. 2) A skill-conditioned policy optimized with Conservative Q-Learning (CQL), using separate Q-networks for fixed and variable actions. 3) A "Local Value Calibration" (LVC) regularizer to enhance stability. Experiments on StarCraft II benchmarks show that SD-CQL significantly outperforms baselines in both multi-to-multi and one-to-multi (task-efficient) transfer scenarios.

### Strengths
- The paper addresses the critical and practical problem of zero-shot generalization in offline MARL, a key step for real-world applications with high data collection costs.
- The empirical evaluation is comprehensive, testing on 14 SMAC task sets across various data qualities and two transfer settings. The SOTA results (13/14 sets) are strong and supported by ablation studies in the appendix.
- The paper is well-structured. The method is presented logically.

### Weaknesses
1. Clarity of Notation/Figures: The paper contains ambiguities.
    1. Figure 1.a plots an undefined "CFCQL" baseline.
    2. The Q-function notation in Sec 3.2.1, such as $Q(a|o,z)$, is non-standard. The`|`(conditional) notation is incorrectly used and should be replaced with standard Q-function notation, e.g.,$Q(o, z, a)$.
    3. It is recommended that bolded entries in all result tables indicate only the best result. The meaning of bolded entries differs between the main text and appendix tables.
2. Hyperparameter Sensitivity: The LVC coefficient,$\eta$, appears highly sensitive, varying widely (0.1 to 0.9) across tasks (Table 8). This reliance on careful, task-specific tuning is a significant practical limitation, as confirmed by ablation in Table 14.
3. Conflated Generalization Metrics: The main claim is improved generalization, but Table 1 reports average performance across all source and unseen tasks combined. This metric conflates performance on known tasks with generalization to unknown tasks. Generalization should be demonstrated by reporting performance on unseen tasks separately.
4. Baseline Validation: The paper compares against re-implemented baselines. There is no validation (e.g., in the appendix) comparing these re-implementations to the results from the original papers. This is necessary to ensure the comparisons are fair and the baselines are robust.

### Questions
1. Encoder Notation (Lines 208-213): Please clarify the encoder notation. What do$K$, $K_i$, and $K_a$ represent? What is the precise meaning of $K_i+1$ in the expression $h=e_{i,K_i+1}$?
2. Hidden Variable`h` (Lines 214-215): What is the concrete implementation of `h`?  Why is`h`not also used in the decoder? The paper claims 'decoder enable the skill to capture temporal information'. Are there any relevant ablation? 
3. Stop Gradient (Fig. 3): What is the motivation for the stop-gradient (Line 285)? I hope there is relevant ablation. In addition, since L_rec and L_Q are completely independent, have the authors tried a two-stage optimization (i.e., pre-train/freeze the skill module, then train the Q-networks) to reduce noise, instead of joint optimization?
4. LVC Hyperparameter $\eta$ (Table 8): Is there any intuition for setting $\eta$? $\mathcal{L}_{LVC}$ is a BC loss. Intuitively, "Expert" datasets should warrant a higher $\eta$, but Table 8 contradicts this (e.g., Marine-Hard-Expert $\eta=0.5$ < -Medium $\eta=0.8$ ).
5. Metrics and Baselines: Can the authors provide a version of Table 1 that disaggregates performance, showing separate average win rates for *source tasks* and *unseen tasks*? Can the authors add a table comparing their re-implemented baseline results against those reported in the original baseline papers?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes SD-CQL, a value-based multi-task offline MARL framework that integrates skill discovery via observation reconstruction with conservative Q-learning. Each agent independently learns skills from local observation transitions without global state or reward supervision. Experiments on SMAC show improve in-domain zero-shot adaptation performance.

### Strengths
The paper is well-written and easy to follow, presenting a clear framework that integrates skill discover with conservative value learning for multi-agent offline MARL. It demonstrate consistent performance improvements on SMAC benchmarks and provides quantitative analysis of the learned skill representations.

### Weaknesses
1. The skill discovery mechanism relies solely on local observation, without access to global state or task-specific rewards to learn latent skills. In this way, the learned skills mainly capture individual behavioral modes rather than genuinely cooperative or team-level strategies. This local-only formulation fundamentally limits the framework's capacity to represent coordinated multi-agent behaviors.

2. The method essentially performs multiple instances of single-agent skill discovery rather than discovering cooperative skills, making the conceptual contribution to the MARL community relatively limited.

3. In the proposed framework, the shared semantic skills across agents implicitly assumes that agents possess homogeneous embodiments or action structures. How would the skill discovery mechanism generalize when agents differ in embodiment or role (e.g., marine & medic)?

4. The semantic expressivity of the proposed skill discovery is fundamentally limited by the SMAC environment's discrete action space, which consists mainly of move and attack primitives. As shown in Section 4.3, the discovered skills are restricted to two primary modes (retreat and attack). To demonstrate the emergence of diverse or compositional skills, the method should be further validated in continuous-control or more complex environments.

5. The evaluation is limited to in-domain variations, where only unit scales differ, providing weak support for true cross-domain generalization beyond the training distribution.

### Questions
1. How does the proposed skill discovery differ conceptually from existing self-supervised skill learning methods for single-agent learning?
	
2. Since the learned skills are derived from local observation transitions, how does the framework ensure that these skills contribute to coordinated multi-agent behavior?

3. How would the proposed method generalize to continuous-control environments?

### Soundness
2

### Presentation
3

### Contribution
2
