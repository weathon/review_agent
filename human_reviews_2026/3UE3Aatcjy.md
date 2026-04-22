# HWC-Loco: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Humanoid robots, capable of assuming human roles in various workplaces, have become essential to embodied intelligence. However, as robots with complex physical structures, learning a control model that can operate robustly across diverse environments remains inherently challenging, particularly under the discrepancies between training and deployment environments. In this study, we propose HWC-Loco, a robust whole-body control algorithm tailored for humanoid locomotion tasks. By reformulating policy learning as a robust optimization problem, HWC-Loco explicitly learns to recover from safety-critical scenarios. While prioritizing safety guarantees, overly conservative behavior can compromise the robot's ability to complete the given tasks. To tackle this challenge, HWC-Loco leverages a hierarchical policy for robust control. This policy can dynamically resolve the trade-off between goal-tracking and safety recovery, guided by human behavior norms and dynamic constraints. To evaluate the performance of HWC-Loco, we conduct extensive comparisons against state-of-the-art humanoid control models, demonstrating HWC-Loco's superior performance across diverse terrains, robot structures, and locomotion tasks under both simulated and real-world environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents HWC-Loco, a hierarchical whole-body control framework for robust humanoid locomotion. It combines a goal-tracking policy for task performance and human-like motion with a safety policy for handling safety-critical scenarios with a high-level planner that can dynamically switches between them to keep safe and precise. By formulating policy learning as a robust optimization problem under mismatched environmental dynamics, HWC-Loco ensures safety while maintaining efficiency, demonstrating superior performance across diverse terrains, robots, and tasks in both simulation and real-world deployments.

### Strengths
1. HWC-Loco quantitatively improves task success rates and demonstrates reasonable high-level policy switching to maintain safety.
2: The high-level policy and hierarchical architecture are well-designed, effectively balancing goal-tracking and safety recovery.
3. The paper presents extensive experiments, and the overall structure and clarity of the manuscript are strong.

### Weaknesses
1. The paper lacks comparison to commonly used baselines such as larger domain randomization or larger domain randomization combined with history-aware policies. While the authors argue that “excessive regularization can greatly affect the efficiency of control policy,” they do not provide evidence that HWC-Loco achieves a better trade-off between efficiency and safety relative to these methods. (If these issues are addressed with stronger evidence, I’d be happy to raise my score.)
2. Performance/Evaluation Concerns: 1) The terrain experiments do not clearly illustrate when or how the high-level policy switching occurs, making it difficult to understand the specific contribution of the safety policy. 2) The terrains and challenge cases used are relatively simple. The paper should include scenarios that highlight the necessity of the hierarchical design—i.e., cases where success cannot be achieved without it or with standard domain randomization alone. Providing qualitative comparisons would help distinguish HWC-Loco from these baselines. 
3. The reliance on ZMP constraints is not that generalizable or robust. ZMP-based control depends heavily on accurate robot dynamics modeling and sensing, which can undermine the reliability (though I agree that integrating it within hierarchical, rather than enforcing hard constraints, is a more flexible design). Moreover, for tasks involving rapid or aerial motions (e.g., jumping) or requiring more agile locomotion, enforcing ZMP-based safety objectives may conflict with task goals, thereby restricting the effectiveness of the safe policy. Finally, its applicability to loco-manipulation under external forces remains limited

### Questions
1. Why does the baseline Goal-tracking consistently outperform AHL/DreamWaQ?
2. Some videos on the project website (e.g., Soft and Slippy Terrain) were not visible at my review time. Is this an accessibility issue on my side, or were those videos not yet uploaded?

### Soundness
2

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
3

### Summary
The paper proposes HWC-Loco, a hierarchical approach for humanoid locomotion balancing safety and agility. The policy is switching between two policy classes: a goal-tracking policy which imitate human motion and a recovery policy trained with robust objective. Then, a high-level planner is learned to switch between two policies optimizing for overall tracking performance. The results is verified in sim and real world.

### Strengths
- the hierarchical approach that target for safety and performance trade-off is natural and well-motivated.
    
- the performance of controller is evaluated comprehensively with diverse metrics to show is robustness and naturalness.

### Weaknesses
- comparison mainly ephasize dreamwaq and hal, where recent strong baselines for humanoid locomotion without hierarchical design like [1] as well as the same style switch controller like [2] is missing.
    
- it would be beneficial if the author could further discuss the sim2real gap identified in the real world deployment and explains with quantitative result on how those sensing and actuation gap hurts the standard policy design and why the hierarchical design can improve on it.
    

[1] AdaMimic: Towards Adaptable Humanoid Control via Adaptive Motion Tracking, Huang et al

[2] Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion, He et al.

### Questions
- how sensitive the tracking/robustness trade off to the imitation multiplier lambda? would the increase of human-like behavior would lead to less recovery agility?
    
- is it possible to distill two policies into a single one to enable smooth transition by teacher-student training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present a learning based hierarchical control framework for humanoid robots that ensures task completion while maintaining safety requirements. A high level planning policy, trained using double DQN, selects between a goal tracking policy and a safety recovery policy based on robots state and historical observations. The paper presents impressive results in simulation as well as real world experiments, across various tasks and terrains. The extensive evaluation demonstrates superior performance over comparable baselines.

### Strengths
The paper is well written, structured and easy to understand. The paper tackles a challenging problem and the proposed approach is appropriately motivated and positioned well among related work. Real world robust humanoid locomotion is a challenging task and the results presented in this paper is quite impressive. 

The paper provides sufficient experimental details and extensive validation. Tests across a wide range of conditions such as terrains, task commands and disturbances provide insightful details about the policies performance. Comparisons against relevant baselines and ablation studies shed light on the effectiveness of the proposed approach well.

### Weaknesses
One of the weaknesses I can think of is the complex framework - this involves training two lower level policies separately, then a high level planner , GAN discriminator, and to enable real world deployment a VAE encoder that estimates privileged information that the policy has during training in simulation. These design choices might be hard to reproduce or deploy.

Since the networks are not trained jointly, there could be switching behavior between the two low level policies due to errors in the system. For example, if the VAE estimator has noisy estimates of privileged information.

### Questions
1) What are some of the failure cases and what caused them?
2) What sort of domain randomization was applied in simulation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper, "HWC-LOCO," proposes a novel hierarchical whole-body control (HWC) approach for robust humanoid locomotion. The core contribution is a hierarchical policy designed to dynamically resolve the trade-off between aggressive goal-tracking and conservative safety-recovery, particularly under environmental disturbances and Sim2Real mismatch. The authors frame the policy learning as a robust constrained reinforcement learning (CRL) problem, maximizing task rewards while ensuring worst-case feasibility constraints across an uncertainty set of transition dynamics. The hierarchical structure consists of a high-level planner that switches between a task-oriented goal-tracking policy (trained with a mimic learning objective for natural, human-like motion) and a stability-focused safety-recovery policy (enforcing ZMP-based constraints). The method is evaluated extensively in simulation and on a real-world humanoid platform, demonstrating superior robustness and performance compared to state-of-the-art baselines across various terrains and

### Strengths
The paper presents a novel and well-motivated combination of existing concepts. The formulation of humanoid locomotion as a robust constrained RL problem (Equation 4) is a significant and original contribution, moving beyond simple reward-shaping for safety. The hierarchical architecture, which explicitly and dynamically switches between a task-maximization mode and a safety-guarantee mode, is a practical and elegant solution to the classic robustness-vs-performance trade-off in safety-critical systems.

This work is highly significant for the field of humanoid robotics. Robust and reliable locomotion is a critical bottleneck for real-world deployment. By providing a principled way to integrate safety guarantees (via ZMP-based constraints in the recovery policy) with high-performance task execution (via the goal-tracking policy), HWC-Loco offers a foundational advancement. The demonstrated Sim2Real success and superior robustness under external disturbances suggest a practical and deployable control framework.

### Weaknesses
1. While the overall formulation is novel, the concept of a hierarchical controller switching between a task-policy and a recovery-policy is not entirely new in robotics (e.g., in model-based control or even some prior RL works). The paper's novelty rests heavily on the robust constrained RL formulation that trains this hierarchy. The authors should more explicitly discuss and contrast their hierarchical training approach with prior hierarchical execution methods to better highlight the distinction.

2. The robust optimization objective (Eq. 4) depends on the uncertainty set parameter $\alpha$ and the feasibility constraint $\epsilon$. The paper mentions that $\alpha$ specifies the scale of mismatch, but the sensitivity of the final policy's performance and robustness to the choice of $\alpha$ is not thoroughly explored. A poor choice of $\alpha$ could lead to either an overly conservative or insufficiently robust policy. More detailed analysis or guidance on selecting these critical parameters would strengthen the work.

3. Robust RL methods, especially those involving a max-min objective or sampling from an uncertainty set of dynamics, are typically computationally expensive. The paper does not provide sufficient detail on the training time or computational resources required for HWC-Loco compared to the baseline methods. Given the complexity of whole-body humanoid control, this is a crucial practical consideration that should be addressed.

### Questions
1. the paper compares HWC-Loco against standard baselines. Could the authors provide an ablation study comparing the full HWC-Loco (trained with Robust CRL, Eq. 4) against a version trained with the simpler Constrained RL (Eq. 2) or even a standard RL with a large penalty for constraint violation? This would isolate the benefit of the max-min robust optimization component specifically.

2. The high-level planner is key to the dynamic trade-off. What is the exact trigger mechanism for switching from the goal-tracking policy to the safety-recovery policy? Is it a simple threshold on a stability metric (e.g., ZMP distance from the support polygon), or is it a learned policy itself? Please elaborate on the input features and the decision logic of the planner.

3. The safety-recovery policy enforces ZMP-based constraints. Is this policy trained to be general across all tasks and terrains, or is it specialized? If it is a fixed, model-based controller, please state this clearly. If it is a learned policy, how is its robustness guaranteed, and how does it interact with the goal-tracking policy's learned dynamics?

4. The Sim2Real success is impressive. Were any specific system identification or domain randomization techniques used to tune the simulation parameters to match the real robot before training? If so, please detail these steps, as they are often critical for successful Sim2Real transfer in complex systems like humanoids.

### Soundness
3

### Presentation
3

### Contribution
3
