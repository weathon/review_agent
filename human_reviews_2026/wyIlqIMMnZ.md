# N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In mobile manipulation, the manipulation policy has strong preferences for initial poses where it is executed. However, the navigation module focuses solely on reaching the task area, without considering which initial pose is preferable for downstream manipulation.
We identify this critical, yet highly overlooked problem and introduce N2M, a strongly practical solution that guides the robot to a preferable initial pose after reaching the task area, thereby substantially improving task success rates. N2M features five key advantages: (1) reliance solely on ego-centric observation without requiring global or historical information; (2) real-time adaptation to environmental changes; (3) reliable prediction with high viewpoint robustness; (4) broad applicability across diverse tasks, manipulation policies, and robot hardware; and (5) remarkable data efficiency and generalizability.
N2M demonstrates state-of-the-art performance compared to prior methods, showing 3% to 54% performance improvement compared to reachability-based methods and 24% to 55% performance improvement compared to the only existing policy-aware alternative in PnPCounterToCab and CloseDrawer tasks, respectively.
Furthermore, in the Toybox Handover task, N2M provides reliable predictions even in unseen environments with only 15 data samples, showing remarkable data efficiency and generalizability.
**Anonymized project website: https://nav2manip.github.io**

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper identifies a critical misalignment in mobile manipulators: navigation modules deliver the robot to a task area, but not necessarily to a preferable initial pose for the manipulation policy. To bridge this gap, the authors introduce N2M, a lightweight transition module that uses only ego-centric observations to predict and guide the robot to a pose that maximizes manipulation success. Experiments in both simulation and real-world environment demonstrate the effectiveness of N2M.

### Strengths
1. Bridging off-the-shelf navigation and fixed-base manipulation policies with a separate fine-grained navigation model is of great application value.
2. N2M can handle dynamic scene, being viewpoint-robust and data-efficient.

### Weaknesses
1. More discussion is required with related works, such as MoManipVLA [1], Mobi-Pi and MoTo. All these methods empowers off-the-shelf fixed-base manipulation policy with mobile capability. A more comprehensive discussion is suggested to be placed in Introduction and Experiments.

2. The above mentioned methods are training-free, but N2M requires rollout from the manipulation policy to train. The overall data collection and training time should be reported to make a fair comparison.




[1] MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation, CVPR 2025.

### Questions
After predicting the camera pose, how to efficiently optimize a path to navigation to this location?

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
This paper identifies a critical misalignment in mobile manipulation pipelines, where navigation modules deliver the robot to a task area without considering the optimal initial pose for the subsequent manipulation policy. To bridge this gap, the authors introduce N2M, a transition module that re-positions the robot into a preferable initial pose after navigation. The core strength of N2M lies in its practical and efficient design, which operates on egocentric observations and demonstrates exceptional data efficiency and generalizability. Its key advantages are convincingly demonstrated through a dramatic performance improvement, elevating success rates from a baseline of 3% to 54% in one task, and showing reliable operation in unseen environments with only minimal data.

### Strengths
1. The field of mobile manipulation, which this paper addresses, represents a relatively nascent yet highly challenging research area.
2. The authors propose N2M, a transitional module designed to reposition the robot into a more favorable initial pose following the completion of navigation.
3. Extensive experimental results are provided to validate the effectiveness of the proposed method and the rationality of its design.

### Weaknesses
1. Does the pose in this paper refer to the position of the robot base or the position of the robot gripper?
2. The generalizability of the method to novel viewpoints should be further evaluated.
3. How would the system adjust the robot's position in case of partial occlusions?
4. Could the authors elaborate on the training methodology for the policy illustrated in Figure 5(b)?

### Questions
please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In mobile manipulation, directly start manipulation from where the navigation ends may be ineffective, as the manipulation policy has preferred starting locations. To address this issue, this paper proposes to rollout the manipulation policy at different locations and fitting a distribution of successful locations, which could be used to generate navigation end points.

### Strengths
1. The paper is well written, with clear description of its technical challenges and solutions. All figures are helpful for getting across the proposed method and the experiment setups.
2. The authors conduct a large number of experiments to validate the proposed method, both in simulation and in the real world.

### Weaknesses
My major concern is the novelty of the proposed method. Though I am not very familiar with mobile manipultion field, according to my limited understanding, finding an appropriate starting pose for manipulation policy is a classic problems in the filed, and there are already several prior papers focusing on this challenge [1][2]. It's unclear to me, compared to them, in which way the proposed method is more novel and contributes to solving this questions.

[1] Iriondo, Ander, et al. "Learning positioning policies for mobile manipulation operations with deep reinforcement learning." International journal of machine learning and cybernetics 14.9 (2023): 3003-3023.
[2] Jauhri, Snehal, Jan Peters, and Georgia Chalvatzaki. "Robot learning of mobile manipulation with reachability behavior priors." IEEE Robotics and Automation Letters 7.3 (2022): 8399-8406.

### Questions
See weakness.

### Soundness
3

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
3

### Summary
This paper addresses the misalignment between navigation and manipulation in mobile robots, where navigation focuses solely on reaching task areas without considering which initial poses are preferable for executing manipulation policies. The authors propose N2M (Navigation-to-Manipulation), a transition module that predicts multi-modal distributions of preferable initial poses using Gaussian Mixture Models from ego-centric RGB point clouds. The key technical contributions include: (1) learning pose preferences directly from policy rollouts, treating policies as black boxes without requiring access to internals or training data, (2) a viewpoint augmentation strategy that renders observations from multiple angles during training, enabling viewpoint robustness and data efficiency with only 10-15 rollouts, and (3) real-time adaptation through single forward pass prediction from ego-centric observations without global scene reconstruction. Experimental validation includes simulation results across 4 tasks and 3 policy architectures in RoboCasa showing dramatic improvements (3% → 54% success rate on PnPCounterToCab), and real-world deployment on 5 tasks demonstrating 10 consecutive successes with randomized configurations and generalization to unseen environments.

### Strengths
**Originality**

Novel problem formulation: The paper clearly identifies and formalizes the navigation-manipulation misalignment problem, which has been overlooked despite its criticality. The key innovation is **learning pose preferences directly from rollouts**, treating policies as black boxes—a creative departure from prior work requiring specific policy types (RL for value functions, IL for distributional similarity) or access to training data.

Technical innovation: The combination of GMM-based distribution modeling with viewpoint augmentation is elegant. Notably, viewpoint augmentation simultaneously addresses multiple challenges (robustness, data efficiency, generalization)—an emergent insight that represents a valuable contribution beyond the primary problem.

**Quality**

Comprehensive experiments spanning 4 simulation tasks with 3 policy architectures, systematic data efficiency analysis, controlled generalization studies (texture vs. layout), and 5 real-world tasks. The ablation study confirms viewpoint augmentation's importance, and attention visualizations (Figure 12) provide interpretability. The paper also acknowledges N2M exceeding oracle performance and discusses policy preferences versus training distributions. However, real-world tasks (b)-(e) use manual rules rather than actual policies, slightly weakening end-to-end claims. More failure case analysis would strengthen the work.

**Clarity**

Well-structured: Clear progression from problem to solution to validation. The five-challenge framework effectively organizes the approach, and visualizations strongly support claims—particularly the multi-modality illustration (Figure 2), success heatmaps (Figure 9), and generalization demonstrations (Figure 10).

Minor gaps: Some technical details need clarification—coordinate frame transformations, kernel number selection rationale (K=2 vs K=1), and the quantitative impact of augmentation ratio M=300 on final performance.

**Significance**

Practical impact: Addresses a fundamental bottleneck in mobile manipulation with dramatic improvements (3% → 54% success rates). The data efficiency (10-15 rollouts), broad applicability across tasks/policies/hardware, and ego-centric design lower adoption barriers significantly.

Research contributions: Opens research directions on viewpoint augmentation's effectiveness and rollout-based preference learning. The finding that policy preferences diverge from training distributions has broader implications for understanding learned behaviors.

Scope limitations: While valuable, the work improves existing modular pipelines rather than proposing transformative paradigms. End-to-end approaches may eventually supersede such bridging solutions. Real-world experiments remain in controlled lab settings. 

Overall, this is a solid, practical contribution that will benefit the community but may not be groundbreaking.

### Weaknesses
**Problem Importance is Questionable**

Base placement is already solved: Methods like B* [1] provide efficient and optimal base placement. The dramatic improvement over the "reachability baseline" (3% → 54%) likely reflects a poorly implemented baseline rather than fundamental limitations of geometric approaches. The paper must compare against state-of-the-art optimization methods (B*, modern inverse reachability maps) to justify that learning is necessary.

Ego-centric observation is not an advantage: This is presented as a benefit but is actually a limitation: (1) Modern robots have mapping capabilities, (2) Methods like Mobi-π plan optimal docking from anywhere using scene reconstruction, while N2M requires navigating close first, (3) No evidence that ego-centric is faster or more robust than global methods.

**Low Technical Novelty**

Standard components: Point-BERT (pre-trained) + MLP + GMM prediction. GMM for robotic poses is established (Bahl et al. 2023), and learning from binary rollout labels is basic preference learning. Architecture (Figure 3) contains no innovation.

"Learning from rollouts" overclaimed: This is standard reward-free imitation from success/failure labels. Predicting poses instead of actions (vs. Lee et al. 2019) is a simplification, not innovation.

Viewpoint augmentation under-analyzed: The only interesting aspect but: (1) No ablation on M=300, (2) No comparison with simpler augmentations, (3) No mechanistic explanation, (4) "10-15 rollouts" is misleading—actually 3000-4500 samples after M=300× augmentation.

**Insufficient Empirical Validation**

Missing critical baselines: No comparison with B* [1], Mobi-π, MoTo, or validation of Lee et al.'s "1000+ rollouts" claim.

Weak real-world validation: Only 1 of 5 tasks uses actual learned policy—tasks (b)-(e) use manual rules. Generalization tested on only 5 similar scenes (Section 4.4) or qualitative only (Section 5.2).

Misleading data efficiency: Claims "10-15 rollouts" but ignores M=300 multiplier. No human time comparison or empirical validation against alternatives.

No statistical rigor: Success rates lack error bars or significance tests despite high variance (Figure 6).


**Missing Ablations**

No ablations on: (1) Encoder choice (Point-BERT vs. alternatives), (2) Loss regularization terms and hyperparameters, (3) GMM kernel number K, (4) Fine-tuning vs. frozen encoder.

### Questions
**Questions**

**Q1: Quantitative Comparison with Optimization Methods**
How does N2M compare with state-of-the-art optimization methods like B* [1]? The 3% → 54% improvement over your reachability baseline suggests a poorly implemented baseline rather than fundamental limitations of geometric approaches. Please provide: (a) success rates for B* or modern inverse reachability methods on your tasks, (b) computational cost comparison, (c) specific scenarios where learning demonstrably outperforms optimization, or (d) acknowledgment if the reachability baseline doesn't represent state-of-the-art geometric methods.

**Q2: Viewpoint Augmentation Analysis and Data Requirements**
Why is M=300 necessary, and what are the true data requirements? Claiming "10-15 rollouts" when M=300 produces 3000-4500 training samples is misleading. Please provide: (a) ablation varying M ∈ {10, 50, 100, 300} showing performance vs. data trade-offs, (b) comparison with simpler augmentations (random jittering, noise), (c) total human time for data collection vs. alternatives, (d) mechanistic explanation of why geometric rendering improves generalization beyond just data quantity.

**Q3: Real-World Validation with Learned Policies**
Why do only 1 of 5 real-world tasks use actual learned manipulation policies? Tasks (b)-(e) use manual rules, weakening end-to-end claims. Please provide: (a) quantitative success rates with learned policies for at least 2 additional tasks, (b) validation that manual rules accurately approximate policy preferences, or (c) explicit acknowledgment as a limitation with discussion of why full policy validation wasn't feasible.

**Q4: Missing Comparisons and Ablations**
What are the quantitative comparisons with related work and ablations on design choices? Please provide: (a) success rate and efficiency comparison with Mobi-π and MoTo on shared tasks, (b) encoder comparison (Point-BERT vs. PointNet++/PointTransformer), (c) loss function regularization ablation with hyperparameter values (αw, αdist, αmode), (d) empirical validation of data efficiency claims vs. Lee et al. 2019, or acknowledge which comparisons/ablations are infeasible.

**Suggestions**

**S1: Provide Statistical Rigor and Failure Analysis**
All success rates lack confidence intervals despite high variance. Include: (a) error bars and significance tests for all results, (b) failure mode breakdown (why 0.0 success in some Figure 9 cells?), (c) conditions where viewpoint robustness or generalization fails, (d) sensitivity to depth noise and partial occlusions. This analysis is critical for understanding practical limitations and deployment reliability.

### Soundness
3

### Presentation
3

### Contribution
3
