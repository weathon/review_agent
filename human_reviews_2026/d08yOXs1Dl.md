# SpikePingpong: Spike Vision-based Fast-Slow Pingpong Robot System

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 0, 8, 8, 8

## Abstract
Learning to control high-speed objects in dynamic environments represents a fundamental challenge in robotics. Table tennis serves as an ideal testbed for advancing robotic capabilities in dynamic environments. This task presents two fundamental challenges: it requires a high-precision vision system capable of accurately predicting ball trajectories under complex dynamics, and it necessitates intelligent control strategies to ensure precise ball striking to target regions. High-speed object manipulation typically demands advanced visual perception hardware capable of capturing rapid motion with exceptional temporal resolution.
Drawing inspiration from Kahneman's dual-system theory, where fast intuitive processing complements slower deliberate reasoning, there exists an opportunity to develop more robust perception architectures that can handle high-speed dynamics while maintaining accuracy.
To this end, we present \textit{\textbf{SpikePingpong}}, a novel system that integrates spike-based vision with imitation learning for high-precision robotic table tennis. We develop a cognitive-inspired Fast-Slow system architecture where System 1 provides rapid ball detection and preliminary trajectory prediction with millisecond-level responses, while System 2 employs spike-oriented neural calibration for precise hittable position corrections. For strategic ball striking, we introduce Imitation-based Motion Planning And Control Technology, which learns optimal robotic arm striking policies through demonstration-based learning.
Experimental results demonstrate that \textit{\textbf{SpikePingpong}} achieves a remarkable 92\% success rate for 30 cm accuracy zones and 70\% in the more challenging 20 cm precision targeting. This work demonstrates the potential of cognitive-inspired architectures for advancing robotic capabilities in time-critical manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper is about a robotic system and related algorithms/models for playing pingpong using a robot manipulator. The system is based in the fast-slow principle by Kahneman, with a fast system that does rapid ball detection and physics-based prediction, and a slow system that improves the fast system with a spike-based calibration, then both systems are combined with a form of supervised imitation learning called IMPACT.

The contributions are:
-a comprehensive robotic table tennis system for high-speed manipulation via a cognitive-inspired architecture
- A fast-slow perception system combining imitation learning with accurate trajectory prediction.
- Extensive experiments showing that the proposed system improves the task to 92% success rate in 30cm zones.

### Strengths
- The paper is very well written and clear.
- I believe this is an interesting robotics problem, with some machine learning usage, definitely in the application domain of robotics this is an improvement over the state of the art and shows the power of learning systems.
- I believe the evaluation is correct, there are some baselines (not many choices in robotics), and there are good ablation results showing the contribution and necessity of the different components in the proposed method.

### Weaknesses
- My major complaing about this paper is that its contributions are in the robotics domain, and not in the machine learning domain. To me from the paper it is not clear what the machine learning community can learn from this paper, and while I think it is a great robotics paper, it does not seem to fix into a machine learning conference. Possibly this paper can be accepted by CORL or any other robotics conference or a journal.
- I believe the claim about a fast-slow system with two components that are working at different temporal frequency is a bit of a stretch, in terms that the system does not have cognition, so its not a cognitive architecture, it is just imitation of a well known cognitive principle (system 1 and system 2).
- There are no relevant machine learning model training details, for example there is no mention of data splits, cross-validation, or standard machine learning setup, and I believe this is because the paper focuses on the robotics concepts. I am sure the authors trained models properly and there was a validation set to check for overfitting, but this is not described in the paper so a future reader cannot check for generalization.

### Questions
- What is the machine learning contribution in this paper? What would machine learning people at ICLR learn from reading this paper?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents SpikePingpong, a robotic table tennis system that integrates spike-based vision with imitation learning. The system employs a cognitive-inspired Fast-Slow architecture where System 1 provides rapid ball detection and physics-based trajectory prediction, while System 2 uses spike camera data for refined trajectory corrections. The IMPACT module handles strategic ball striking through imitation learning. The system achieves high success rate in the real world.

### Strengths
- The Fast-Slow system architecture is well-motivated and technically sound, combining physics-based modeling with learned corrections.
- The spike camera integration for capturing millisecond-level ball-paddle contact is innovative and well-executed.
- The experimental validation includes solid real-world deployment and comprehensive comparisons and ablations.

### Weaknesses
- The task focuses solely on table tennis. There are many task-specific design choices with insufficient analysis of what methodology generalizes beyond table tennis (and applies to more general dynamic manipulation tasks).
- There lacks a in-depth failure analysis. Table 8 shows distributions but doesn't explain root causes of 79.1% near-miss failures or how failures vary by ball speed, spin, trajectory.
- Minor: Wrong citation in Table 3: "Diffusion Policy (Zhao et al., 2023)" should be "(Chi et al., 2023)".

### Questions
- Why is the reported diffusion policy inference time so slow (2437.22ms)? If using DDIM during inference with around 10 denoising steps, the inference should be around 100ms if using image input, and should be even faster if just using state input. 
- Do ACT and diffusion policy baselines implementations have exact same state-based inputs/outputs as IMPACT or do they use image inputs?
- Previous work like UMI [1] is able to do dynamic tasks like tossing with asynchronous inference with latency matching using diffusion policy. How is policy inference implemented for baseline policies? Was asynchronous inference with latency matching attempted as well?

[1] Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, Shuran Song.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a learning-based robotic system for autonomous ping pong playing. It consists of three main modules: a perception module for detecting ping pong ball trajectory, a module for detecting hitting positions supervised with GT obtained from privileged spike cameras, and a policy module for predicting parameters for ball hitting. The system is learned with successful demonstrations obtained from an automated pipeline with random hitting configurations. It is shown to be able to condition on desired landing region, as well as outperforming human baseline in consecutive ball hitting returns.

### Strengths
- The paper is overall well-written and easy to follow.
- It is impressive to see the demo included in the supplementary submission, which demonstrates the robustness of the system.
- The experiments are comprehensive to demonstrate the importance of the design choices made in this work as well as the overall robustness of the system.
- As a system-oriented paper, the design choices and their details presented in the paper can be useful references for future works in similar direction.

### Weaknesses
- No major weaknesses.
- Typo in Figure 2: “inversive kinamatics”

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a vision-equipped robotic ping-pong system using an ABB IRB-120 arm with a standard table-tennis racket. The framework consists of three main components: (1) System 1, which provides ball detection from rgbd and trajectory prediction; (2) System 2, which refines the predicted hittable position; and (3) an Imitation-based module (IMPACT) that learns optimal striking strategies from human demonstrations. Together, these modules form a closed-loop perception–planning–control system capable of perceiving, planning, and executing appropriate striking motions in response to the opponent’s incoming shots. Real-world experiments demonstrate the proposed method‘s effectiveness.

### Strengths
+ Well-structured and intuitive architecture. The modular pipeline from perception to control is clear and logically connected.

+ Detailed modular implementation. The paper provides comprehensive implementation details for each subsystem, enabling reproducibility and practical insights for future robotic applications.

+ Good real-world results. Experimental evaluation on the physical ABB IRB-120 setup demonstrates reliable tracking, accurate returns, and stable rally performance, confirming real-time feasibility.

### Weaknesses
The main weakness of this paper lies in its limited adaptation and generalization capability. Since the Stage-3 action generation (IMPACT) relies purely on imitation learning, the system lacks the ability to adapt to unseen or out-of-distribution ball trajectories, such as those with different spins, velocities, or bounce patterns. The evaluation also appears to be confined to in-domain scenarios, and it remains unclear whether the tests include unseen human/robot launch opponents or novel shot types. Moreover, the demonstration data are said to come from both a ball-launching machine and human players, but the paper does not specify which subsets were used for training and which for testing, nor does it clarify whether the demonstrator shown in the supplementary video is the same person who provided the training data. The robot’s striking behavior in the video further appears to be limited to only two distinct motion patterns, suggesting a lack of diversity and adaptability in the learned control policy. A deeper analysis of how the system handles different incoming trajectories and human play styles, given that variations in hitting points and trajectories across players can be substantial, would greatly strengthen the paper and highlight the system’s robustness beyond demonstration-specific conditions.

### Questions
see "weaknesses"

### Soundness
3

### Presentation
3

### Contribution
3
