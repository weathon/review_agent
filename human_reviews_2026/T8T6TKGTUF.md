# A Bio-inspired Gradient-Free Learning Framework for CPG-based Neural Networks in Robot Locomotion

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Central Pattern Generators (CPGs) for animal locomotion have been adopted for robot locomotion control for three decades, but have rarely been used in reinforcement learning. It is partly due to the inherent recursive connections causing gradient vanishment or explosion during optimisation. In this paper, we propose a framework with a bio-inspired learning rule to optimize a recurrent neural network for robot reinforcement learning. The learning rule utilises weight fluctuation for parameter exploration and reward for convergence. With the framework, we trained a minimal network containing CPG and full connection layers for quadrupedal locomotion. Experiments suggest that our framework can train the network, while the Proximal Policy Optimization (PPO) fails. Compared to PPO with a feedforward network, our framework with a CPG is more robust to sensor failure. This work lowers the barrier to exploring the potential advantages of recurrent neural networks in robot reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a Dynamic Synapse (DS) learning rule, a gradient-free, biologically inspired optimization approach in which each network weight oscillates sinusoidally, and its center is updated based on received rewards. The authors use DS to train central pattern generator controllers for quadruped locomotion, both in simulation and on a real robot. They claim that DS can train architectures where PPO fails due to gradient instability and that DS-trained controllers exhibit robustness to sensor failures.

### Strengths
- Novel yet simple gradient-free mechanism derived from neurodynamics, implemented efficiently in PyTorch for parallel rollouts.
- Demonstrates stable training of CPG-RNN controllers that PPO struggles with.
- Includes real-robot results and robustness evaluations, which strengthen the empirical credibility.
- Provides an interesting connection between biologically inspired oscillatory dynamics and practical control learning.

### Weaknesses
1. The approach should be compared against stronger evolutionary algorithms (CMA-ES) and well-tuned PPO/SAC.
2. Provide ablations: (a) original vs. new DS formulation, (b) which parameters are DS-trained, (c) reward sensitivity.
3. The approach should also be tested on multi-skill or sparse-reward tasks to assess generality.
4. Clarify total learnable parameter count.
5. Investigate using one shared CPG topology conditioned on commands, rather than per-task networks.

### Questions
- “Therefore, we use the first structure (Fig.3(a)) to complete the backward
walking task, and others (Fig.3(b)(c)) to complete the forward walking task. “ -> seems complicated for just a walking task. How would this scale to other tasks? Would this suggest you need a new network for each task? 
- “We observed that the genetic algorithm (GA) often collapsed very early, sometimes as
soon as the fifth generation” -> I'm surprised by this. I would suggest trying a much larger population size
-  “his instability caused the robot to fall frequently or produced NaN states, terminating training prematurely” -> For me, this rather points to some kind of bug. There shouldn't be any NaN numbers because training converges.

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
2

### Summary
This paper presents a bio-inspired learning framework that enables effective training of recurrent neural networks with Central Pattern Generators (CPGs) for robot reinforcement learning. The proposed method achieves robust quadrupedal locomotion and improved resilience to sensor failure.

### Strengths
The paper is well written and has hardware demonstrations. The design choices are nicely ablated in both simulation and hardware. I also like the idea of integrating CPGs into RL (although I am unsure about its benefits as discussed in weaknesses below).

### Weaknesses
1. The paradigm in legged locomotion seems to use fast and efficient simulators, e.g., Mujoco playground (https://arxiv.org/abs/2502.08844) to train policies efficiently in simulation and transfer to real. I am not quite sure how the proposed approach would be better than the more common strategy and what its underlying benefits are. 

2. In addition, the proposed method does not benefit from fast parallel simulators, which are widely available and highly efficient. 

3. The hardware demonstrations and the simulation tasks considered in this work are not at all dynamic compared to the SOTA methods for legged locomotion (https://arxiv.org/pdf/2010.11251, https://ieeexplore.ieee.org/abstract/document/10225268)

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a bio-inspired gradient-free learning framework to address the issue that Central Pattern Generators (CPGs) – which are effective for robot locomotion control but rarely used in reinforcement learning (RL) due to gradient vanishing or explosion caused by their inherent recursive connections – are difficult to optimize via gradient-based methods. The core of this framework is a Dynamic Synapse (DS) optimizer, which realizes parameter exploration through weight fluctuations and convergence through reward feedback, and it is used to train a three-layer network (including Obs-layer, CPG layer with specific oscillators, and Act-layer) for quadrupedal locomotion on the Unitree go1 robot. Experiments show that the framework can successfully train the CPG-based network, whereas Proximal Policy Optimization (PPO) fails due to gradient problems; the CPG-based network is more robust to sensor failures than PPO-trained feedforward networks, and it avoids the early training collapse of Genetic Algorithms (GA). Although the DS optimizer supports parallel training, it does not fully utilize parallel simulation environments, and its performance still lags behind PPO-trained MLP networks, but the work overall lowers the barrier to applying recurrent neural networks in robot RL.

### Strengths
1. The paper proposes a bio-inspired gradient-free Dynamic Synapse (DS) optimizer that effectively addresses the gradient vanishing/explosion issue caused by the inherent recursive connections of Central Pattern Generators (CPGs), enabling successful training of CPG-based neural networks for quadruped robot locomotion— a challenge that gradient-based methods like Proximal Policy Optimization (PPO) fail to overcome , .
2. The CPG-based network trained with the proposed framework demonstrates stronger robustness to sensor failures compared to PPO-trained feedforward networks, which is a critical advantage for real-world robot applications where sensor malfunctions may occur.

### Weaknesses
1. The Dynamic Synapse (DS) optimizer, despite supporting parallel training through PyTorch implementation, fails to fully leverage parallel simulation environments, resulting in no significant reduction in training time, which limits its efficiency for large-scale or time-sensitive robot locomotion training tasks .
2. There remains a performance gap between the CPG-network trained by the DS optimizer and the traditional Multilayer Perception (MLP) network trained by Proximal Policy Optimization (PPO), indicating the proposed framework still lags behind existing mainstream methods in terms of locomotion control performance under complete observation conditions .
3. The framework’s validation is relatively limited to basic quadruped locomotion tasks (forward and backward walking) on the Unitree go1 robot, without extending to more complex scenarios such as locomotion over challenging terrains or multi-skill motion control, restricting the demonstration of its general applicability .
4. The DS optimizer’s hyperparameters (e.g., amplitude initialization, oscillation period) require careful tuning based on specific tasks (e.g., matching the oscillation period to the causality period between action and reward), and improper settings may lead to issues like early convergence, local optima, or prolonged training time, increasing the complexity of practical application , .

### Questions
1.  Reward hacking is a common issue in reinforcement learning. Have you observed any cases where the robot achieves high rewards through exploiting loopholes in the designed reward function (e.g., abnormal locomotion without effective movement)? If not, what evidence supports that the reward function can avoid such problems?
2.  The DS optimizer relies heavily on reward signals for parameter adjustment. How does noisy or inaccurate reward (e.g., from sensor errors) affect its convergence and stability? Do you have experimental evaluations on the optimizer’s robustness to reward noise?
3.  The DS-CPG network lags behind PPO-MLP in performance under complete observations. Could this gap be narrowed by adjusting the reward function’s weight distribution or adding new terms? Have you conducted relevant tests?
4.  The reward terms for locomotion metrics (e.g., trunk height, foot sliding) – are they designed based on prior knowledge or systematic optimization (e.g., hyperparameter search)? Would this design limit adaptability to other legged robots (e.g., hexapods)?
5.  You compared DS with GA and PPO in stability, but not in reward efficiency (steps to reach a target reward). Does DS require more training steps to converge than GA when both are stable? How does it compare to PPO (excluding gradient issues)?
6.  The DS optimizer’s design is tailored to periodic locomotion tasks. For non-periodic tasks (e.g., robotic manipulation), what key adjustments (e.g., fluctuation period, learning rules) would be needed? Is there preliminary evidence for its applicability?
7.  You mentioned underutilization of parallel simulation. What specific bottlenecks exist in the current parallel implementation? Do you have plans to optimize it (e.g., adjusting parameter sharing mechanisms) to reduce training time?

### Soundness
2

### Presentation
2

### Contribution
1
