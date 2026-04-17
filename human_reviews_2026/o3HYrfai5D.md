# RoVer: Robot Reward Model as Test-Time Verifier for Vision-Language-Action Model

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Vision-Language-Action (VLA) models have become a prominent paradigm for embodied intelligence, yet further performance improvements typically rely on scaling up training data and model size -- an approach that is prohibitively expensive for robotics and fundamentally limited by data collection costs. We address this limitation with $\mathbf{RoVer}$, an embodied test-time scaling framework that uses a $\mathbf{Ro}$bot Process Reward Model (PRM) as a Test-Time $\mathbf{Ver}$ifier to enhance the capabilities of existing VLA models without modifying their architectures or weights.
Specifically, RoVer (i) assigns scalar-based process rewards to evaluate the reliability of candidate actions, and (ii) predicts an action-space direction for candidate expansion/refinement. During inference, RoVer generates multiple candidate actions concurrently from the base policy, expands them along PRM-predicted directions, and then scores all candidates with PRM to select the optimal action for execution. Notably, by caching shared perception features, it can amortize perception cost and evaluate more candidates under the same test-time computational budget. Essentially, our approach effectively transforms available computing resources into better action decision-making, realizing the benefits of test-time scaling without extra training overhead. Extensive experiments demonstrate that RoVer consistently improves success rates across diverse manipulation tasks. Our contributions are threefold: (1) a general, plug-and-play test-time scaling framework for VLAs; (2) a PRM that jointly provides scalar process rewards and an action-space direction to guide exploration; and (3) an efficient direction-guided sampling strategy that leverages a shared perception cache to enable scalable candidate generation and selection during inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduce RoVer, a framework for test-time scaling in VLA models for robotics. It uses a PRM as verifier to evaluate and refine candidate actions from a frozen base policy without changing the model weights or needing more training data. The PRM gives scalar rewards and directions in action space to guide sampling better candidates. They cache perception features to make it efficient. Experiments on CALVIN sim and real robot show improvements over baselines like GR-1, Dita, MoDE, and Diffusion Policy.

### Strengths
1. Novel idea to shift scaling from training-time to test-time, which is important because robot data is expensive to collect. This plug-and-play approach can work with existing VLAs without retraining, that is practical.

2. The PRM design is clever, predicting both reward and direction allow for guided exploration instead of just random sampling. The action amplifier and anchor-centered training seem to help distinguish fine action differences.

3. Efficiency with shared perception cache is good, allow evaluating more candidates under same compute budget. Ablations in the method section show direction-guided sampling outperform random.

4. Experiments cover sim (CALVIN ABC->D) and real robot, with multiple baselines. Consistent gains, like improving success rates on long-horizon tasks.

### Weaknesses
1. The training of PRM still require the same dataset as the base policy, so it's not completely "without additional data" as claimed – it's reusing data but for a separate model. Could be clearer on this.

2. Limited to delta actions in end-effector space; not sure how it generalize to other action representations like joint angles or velocity control.

3. Real robot experiments only with Diffusion Policy, would like see on more advanced VLAs. Also, no latency numbers for inference – how much slower is it with N=64 candidates?

4. Direction prediction visualized on PushT, but PushT is easy (near 100% for DP), so gains are small. More analysis on when direction helps vs fails would be nice.

### Questions
1. In training, why choose RMSE to expert action as "better"? Is this process reward correlate well with actual task success? Maybe compare to outcome rewards.

2. For inference, how sensitive to noise scales σ_base and σ_adapt? Any hyperparam tuning details?

3. Compatibility: does RoVer work better with some base policies than others? E.g., diffusion vs transformer policies.

4. Appendix mentioned for distribution gap analysis, but not provided; can authors include more details?

### Soundness
2

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
2

### Summary
RoVer is a test-time scaling framework designed to enhance embodied Vision-Language-Action (VLA) models without additional data collection or retraining. It introduces a lightweight Process Reward Model (PRM) that both scores candidate robot actions and predicts refinement directions in the action space, at inference time. By generating and verifying many candidate actions, RoVer stabilizes long-horizon performance and unlocks latent capabilities in existing policies. Experiments in simulation and real-world manipulation tasks demonstrate consistent performance improvements in both simulation and the real-world setup.

### Strengths
- RoVer introduces a plug-and-play test-time scaling framework that complements traditional training-time scaling, providing a new horizontal improvement direction for existing VLA policies without modifying their architectures or weights.

- The method makes efficient use of perception caching, enabling the evaluation of a large number of candidate actions with minimal additional compute.

- RoVer directly addresses a key bottleneck in embodied AI, the high cost of collecting and annotating robotic data, by reallocating compute to inference-time verification and refinement, achieving meaningful performance gains without expanding the dataset with annotation.

### Weaknesses
- While the high cost of collecting robotic data is a clear motivation for test-time scaling, the claim that *"VLA success rates fluctuate due to stochastic decoding and manipulation brittleness, especially for long-horizon tasks"* remains underspecified. Please clarify how this observation motivates reallocating computation from training to inference.

- Some important baselines are missing. In particular, evaluation on stronger or more recent state-of-the-art VLA backbones (e.g., Pi-zero[1]) would better demonstrate generality and competitiveness.

- It is unclear whether $R_\phi$(h, a) outputs only a scalar reward or both reward and direction (lines 171 and 279).

- Figure 1 over-emphasizes the base inference module rather than RoVer’s novel components, and Figure 2 is overly dense and hard to interpret. Simplifying visuals to highlight the contribution would improve accessibility.


---
[1] Black, Kevin, et al. "$\pi_0 $: A Vision-Language-Action Flow Model for General Robot Control." RSS (2025).

### Questions
- Why is the way to compute adaptive noise scale necessary, rather than sampling noisy expert action $a_{en}$ around $a_e$ and distinguishing better and worse by computing distance to expert actions? Although the authors mention it yielded poor performance, what is the main reason that makes these two significantly different?

- RoVer's benefit appears dependent on the PRM being trained from small perturbations around expert actions. In cases where the base policy deviates significantly from the expert distribution (e.g., early failures in long-horizon tasks), does PRM guidance remain reliable? Do the authors have relevant observations in the experiments?

- Is a separate PRM trained for each manipulation task, or can a single model generalize across multiple tasks? For diverse tasks with different action distributions, how sensitive is the method to the chosen Gaussian noise scale during training? Results on multi-task PRM or noise-scale ablations would clarify scalability.

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
4

### Summary
This paper proposes an inference expansion framework named RoVer, which aims to enhance the performance of existing Vision-Language-Action (VLA) models without modifying their architectures or weights. RoVer adopts a Robotic Process Reward Model (PRM) as the runtime validator. During the inference phase, RoVer synchronously generates multiple candidate actions from the base policy, expands them along the directions predicted by the PRM, and selects the optimal action for execution through PRM scoring. Experiments demonstrate that RoVer consistently improves success rates across various manipulation tasks, providing a runtime expansion solution for VLA models that requires no additional training overhead.

### Strengths
1. The method proposed in the paper is highly lightweight and exhibits strong scalability.
2. The writing follows a clear and rigorous logical flow, making it easy to understand.
3. The experimental section is extremely detailed and comprehensive.

### Weaknesses
1. **The comparison with crucial related work is missing**: *Nakamoto, M., Mees, O., Kumar, A., & Levine, S. (2024). Steering your generalists: Improving robotic foundation models via value guidance. arXiv preprint arXiv:2410.13816.*
This paper also selects the action with the maximum Value during the inference stage, which is very similar to the core idea of the RoVer. Please make a detailed comparison of the similarities and differences between RoVer and V - GPS.

2. **The selection of $a_{better}$ may not be better**: The anchor action $a_{anc}$ is obtained by adding noise to the expert action $a_e$. Based on the anchor action $a_{anc}$, the action closer to $a_e$ is called $a_{better}$.
Suppose a successful trajectory is $a_1$, $a_2$, $a_3$ until the task is completed. If $a_2$ is taken as the current expert action, it is possible that $a_{anc}$ obtained after adding noise is closer to $a_3$. At this time, $a_{better}$ obtained from $a_2$ close to the expert action may be worse than the expert action instead.
What impact will this situation have on the learning of rewards?

### Questions
1. Is the current reward function compatible with the VLA algorithm for joint angle control?
2. How should the hyperparameters in Equation 14 be selected, and what impact do they have on the corresponding performance?

### Soundness
2

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
4

### Summary
This paper introduces RoVer, a test-time scaling framework for VLA models that enhances robotic policy performance without modifying the base model's architecture or weights. RoVer employs a lightweight PRM to score candidate actions and predict refinement directions in the action space. Experiments on the CALVIN benchmark and real-robot tasks demonstrate consistent improvements across multiple base policies (GR-1, Dita, MoDE, and Diffusion Policy), with significant gains in success rates and inference efficiency.

### Strengths
1. Rigorous experiments across simulation and real robots, with strong empirical results;
2. Offers a practical and efficient alternative to costly training-time scaling, with broad applicability to existing policies and VLA models.

### Weaknesses
1. The method is less effective for chunk-based policies like MoDE due to the step-chunk mismatch;
2. The PRM relies on expert action proximity as a supervision proxy, which may not always align with task success in more complex or long-horizon settings.

### Questions
1. The authors themselves also mentioned the limitation of the mismatch between chunk-based policy inference and PRM inference. Could the direction-guided sampling be extended to handle multi-step or hierarchical actions to better support chunk-based policies?
2. How does the performance of the authors' proposed method compare to the RL-based approach [1] that uses a trained value function to guide policy sampling?

[1] Nakamoto, et al. Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. CoRL, 2024.

### Soundness
2

### Presentation
3

### Contribution
2
