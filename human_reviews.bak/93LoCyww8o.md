# Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response

- Decision: Accept (poster)
- Scores: 5, 8, 6, 6

## Abstract
Robust locomotion control depends on accurate state estimations. However, the sensors of most legged robots can only provide partial and noisy observations, making the estimation particularly challenging, especially for external states like terrain frictions and elevation maps. Inspired by the classical Internal Model Control principle, we consider these external states as disturbances and introduce Hybrid Internal Model (HIM) to estimate them according to the response of the robot. The response, which we refer to as the hybrid internal embedding, contains the robot’s explicit velocity and implicit stability representation, corresponding to two primary goals for locomotion tasks: explicitly tracking velocity and implicitly maintaining stability. We use contrastive learning to optimize the embedding to be close to the robot’s successor state, in which the response is naturally embedded. HIM has several appealing benefits: It only needs the robot’s proprioceptions, i.e., those from joint encoders and IMU as observations. It innovatively maintains consistent observations between simulation reference and reality that avoids information loss in mimicking learning. It exploits batch-level information that is more robust to noises and keeps better sample efficiency. It only requires 1 hour of training on an RTX 4090 to enable a quadruped robot to traverse any terrain under any disturbances. A wealth of real-world experiments demonstrates its agility, even in high-difficulty tasks and cases never occurred during the training process, revealing remarkable open-world generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work proposees Hybrid Internal Model which uses minimal sensor input for legged locomotion aiming to address the limitation of existing learning-based locomotion control methods. The Hybrid Internal Model learns the latent representation of velocity and implicit dynamics of the environment via contrastive learning. The method is evaluated in simulation and deployed in the real world.

### Strengths
1. The work provides a simple one-stage method for training blind locomotion controllers, with minimal sensory information.
2. This work provides impressive real-world performance in the supplementary video.
3. The t-SNE visualization shows that the proposed hybrid internal model learned to distinguish different terrain type in latent space.

### Weaknesses
1. The overall delivery of the work is not satisfactory in the following aspect:
* Many technical details are poorly described or totally missing: what’s the exact task legged robot need to solve (follow forward velocity or follow angular velocity or anything else), what command is included in the user-command, what’s the linear velocity and angular velocity range for the input. 
* Some parts are quite confusing, for instance: the user command and trainable prototypes are using the same notation: c_{i}. There are many typos in the paper, for example: “police network” in section 3.

2. The evaluation of the method is poor:
* There is no quantitative comparison with any baseline in the paper (no result in simulation and no result in real world). 
* The only comparison result (The training curve in Figure 6) provides limited information, since there is no specific description of the task. It’s hard to see the actual performance gap of different methods in terms of the “task target” like velocity tracking and angular.0
* There is no actual baseline compared in the experiment section, since the method use only proprioception observation of the robot, at least RMA, MoB should be compared, quantitatively, and provide corresponding training curve. Since this work claim the proposed method provide better performance and sample efficiency against two-stage methods, at least comparison is needed.
* There is no performance analysis the method or baselines across different terrains, or across different command range or different command types, or across different command target.
* This work claims contrastive learning is better than purely regression, but there is no experiment to support the claim.


Reference: 
[1] Kumar, et al. RMA: Rapid Motor Adaptation for Legged Robots

### Questions
Though this work provides a reasonable real-world demo, this work need to be improved in multiple ways to meet the standard of ICLR. 
1. Many technical details (including but not limited to the items mentioned in the weakness section) need to be cleared.
2. More baseline needs to be compared.
3. More ablation study need to be performed, for example regression or contrastive learning
4. More specific performance analysis is required: performance across terrain or different physical properties.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to use a learned model to improve learning policies for a quadrupedal robot. In particular, the learned model is trained via contrastive learning instead of the typical regressive learning in existing literature.

### Strengths
1. Using contrastive learning for learning the dynamics model seems interesting.

2. Real robot experiments.

### Weaknesses
1. The paper seems to be written in a rush with lots of typos and incorrect references. 

2. The need of the velocity model is a little bit unsatisfying as one would expect a perfect internal model should be able to replace it.

3. Need clarification for some questions that I will list below.

### Questions
1. Seems to me the proposed model is more of a state estimation model (or a state compression model) instead of a dynamics model. For example, one would expect a dynamics model to be used to do rollouts instead of using a simulator during training. It will be nice to have some comments about it.

2. A key missing ablations is to compare the proposed contrastive learning with a regression model. 

3. It would be nice to have more analysis of the learned latent model as this is the key contribution. For example, examining the effect of the number of prototypes,

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposed a new legged locomotion method called HIM (Hybrid Internal Model). The method uses only joint encoders and IMU observations (no cameras or range sensors). HIM has 2 modules, the "information extractor " which learns the environment dynamics, and the "policy network" which learns the policy. The HIM model is trained in a simulator (IsaacGym) and successfully deployed in the real world (Unitree Go1 robot).

### Strengths
The main strength of the paper is the proposal of the HIM (Hybrid Internal Model), which is able to address the legged locomotion problem using only joint encoders and IMU observations (no cameras or range sensors). This is achieved using two components, the "information extractor " which learns the environment dynamics, and the "policy network" which learns the policy. This is a very interesting approach and from my point of view is related to the "world models" concept.

The method worked successfully in different simulated and real environments. In the real world, experiments were conducted on various environments, showing agile and robust locomotion.

### Weaknesses
From my point of view "world models" also learn the dynamics of the environment, in simulation or without using simulators, and it is required that authors compares its proposal with world models. For instance, Wu et al. "Daydreamer: World models for physical robot learning" applies word models online and address legged locomotion. 

This is important in order to better assess the originality and contributions of the paper.

### Questions
How your work is related with "world models" that also learn the dynamics of the environment? For instance, [1] applies word models online and also addresses the legged locomotion problem. I know that in your work less sensor data is used, but in addition to this, how your approach is compared with world models? Can your approach be considered a world model?

[1] Wu et al. "Daydreamer: World models for physical robot learning" applies word models online and address legged locomotion.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors proposed a new auxiliary module, i.e. the hybrid internal model to help blind locomotion policy training. The hybrid internal model consumes past propiorceptive observation histories and produces a latent dynamic vector and estimated base body speed. Instead of using the regular regression loss, the latent dynamic vector is trained with a contrastive loss. The authors zero-shot sim2real transfer their policy to a Unitree robot and show that  their policy can outperform other blind baselines on a few difficult tasks including walking up/down large stairs and on soft terrains.

### Strengths
1) The paper shows great sim2real results. The learned gait looks very smooth, and more importantly, can solve difficult multi-terrains without using perception
2) Some novelties in introducing a contrastively learned environment/internal dynamics model.

### Weaknesses
1) There isn't a big delta from previous works, i.e. the ETH 2020 science paper also demonstrates similar capabilities. 
2) While it is applaud to see good results on blind locomotion walking, without perception the policy is still fundamentally limited on difficult terrains. For example, the policy has to hit the vertical edges to sense the terrain before the hybrid internal model can "sense" the change in the environment. And there are already good perceptive locomotion works there. 
3) The experiment result section of the paper is thin. For example, there is no quantitative results on success rate etc. 
4) In the ablation study, the oracle policy seems learning slower than the proposed method, which is surprising. 
5) Consider publishing the code with the paper. Right now there is a supplementary pdf which is a duplication of the main paper.

### Questions
Is the hybrid internal model co-trained with the policy or separately?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
