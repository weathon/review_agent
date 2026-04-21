# HuWo：Building Physical Interaction World Models for Humanoid Robot Locomotion

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 5, 6, 3

## Abstract
Reinforcement Learning control has been proved to be an effective approach for quadruped robot locomotion. However, locomotion tasks for humanoid robots are challenging, especially in complex environments. The main reason is that humanoid robots must maintain balance during movement and constantly engage in complex dynamic interactions with the environment. Understanding robot-environment interaction dynamics is key to achieving stable locomotion for humanoid robots. Since there is privileged information that the robot cannot directly access, to expand the observable space, previous reinforcement learning-based methods either reconstruct environmental information from partial observations or reconstruct robotic dynamics information from partial observations, but they fall short of fully capturing the dynamics of robot-environment interactions. In this work, we propose an end-to-end reinforcement learning control framework based on physical interaction$\textbf{Wo}$rld Model for $\textbf{Hu}$manoid Robots (HuWo). Our key innovation is to introduce a physical interaction world model to understand the interaction between the robot and environment, employing the hidden layers of transformer-XL for implicit modeling of this process across temporal sequences. The proposed framework can showcase robust and flexible locomotion ability in complex environments such as slopes, stairs, and discontinuous surfaces. We validated the robustness of this method using the $Zerith1$ robot, both in simulations and real-world deployments, and quantitatively compared our HuWo against the baselines with better traversability and command-tracking.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an approach to train a reinforcement learning-based controller for a humanoid robot. The key components are (1) Transformers for both actor and critic, which allows for efficient learning from the history of observation (2) in addition to learning the value function, the critic is also tasked with learning the system dynamics from the learned hidden state. (3) The actor is also tasked to match the hidden state of the critic.

### Strengths
The system is straightforward to understand, and the presentation is good.

The proposed method can learn robust control policies for a humanoid robot.

Analysis of the hidden state.

### Weaknesses
The paper should be revised for some typos, e.g., in the paragraph Terrain Passability Experiment, there is a missing reference to a figure. 

I am not sure about some of the points, which I will list below.

### Questions
From Fig 4, it does not look like learning a world model helps much. The yellow and green curves converge to similar values and in almost the same number of steps.

What is the difference between a world model and a dynamics estimator? It would be nice to clearly point out which part of the system is the world model and which part is the dynamics estimator in Fig. 2.

The ablation on the dynamics estimation module is missing. There is also no ablation on the latent variable regression. These ablations are mentioned in the text but are not in the figures.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces HuWo, an end-to-end reinforcement learning framework that leverages a physical interaction model using Transformer-XL to enhance humanoid robot locomotion in complex environments. The approach implicitly models dynamic interactions between the robot and its environment across temporal sequences. HuWo demonstrates robustness and flexibility in challenging terrains like slopes, stairs, and discontinuous surfaces, outperforming baseline models in traversability and command tracking.

### Strengths
Originality: The authors cleverly integrate the concepts of predicting states with a dynamics model within an RL training framework. The idea makes perfect sense to model the complex dynamics between the humanoid and the environment.

Quality: The gait appears robust in most parts of the video, and I commend the authors for a thorough evaluation across diverse terrains. The baseline comparisons in the simulation are also convincing.

Clarity: The paper is clear and easy to understand.

Significance: This method effectively combines the strengths of both world-model-based approaches and powerful RL frameworks with domain randomization. The learned dynamics model is well-suited for capturing the complex interactions between the humanoid and its environment.

### Weaknesses
1. The presentation could be improved with more attention to figure and text quality. For instance, the text in many figures is too small and difficult to read without close inspection, and there is a missing figure reference on page 8. Additionally, the formatting and choice of symbols in mathematical equations could be refined for better readability. Some claims in the paper are also questionable, such as: “However, for humanoid robots, these methods can only handle relatively simple environments and have not yet fully addressed dynamic control issues in complex terrains.” Prior research has shown that RL can handle challenging terrains for humanoid robots [1,2,3].

1. The evaluation would be more robust with additional quantitative results from real-world experiments. The sim-to-real gap can be significant, and while the authors show the policy performs in both simulation and real settings qualitatively, simulation results alone may not fully demonstrate real-world effectiveness without quantitative results, as performance could differ substantially.

1. Finally, the gait in the supplementary video is not particularly impressive, as humanoid walking videos have become common. The stride length is short, and the speed seems slow relative to the robot’s scale. In the stair-climbing experiment at the end, the walking also appears less robust.

[1] Gu et al., *Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning*, RSS 2024.

[2] Liao et al., *Berkeley Humanoid: A Research Platform for Learning-based Control*, ArXiv 2024.

[3] Zhuang et al., *Humanoid Parkour Learning*, CoRL 2024.

### Questions
1. Could you motivate more why the dynamics model helps learn humanoid locomotion? Since as I mentioned above, RL approaches seem to already work well enough with recent advances. Some reasons I can image include data efficiency or better generalization capabilities by incorporating a world model. But if that's the case, more evidence needs to be shown to support these claims.

1. Could you show more evidence that the proposed method is superior to baseline methods in the real world? For example, what is the velocity tracking performance in the real world? Have you considered *Gu et al.* as a baseline?

1. In Figure 4, what does terrain level mean? Is there a more convincing metric than mean rewards?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents an asymmetric actor-critic framework to train a policy for blind bipedal locomotion, based on the transformer-XL architecture.  In particular, the critic is trained (with privileged information) as a value function approximator and also as a world model that predicts the next state observation, based on the hidden variables $h_t$. Such privileged hidden variables are then used to supervise the hidden variables $z_t$ of the actor, which are generated from observations with corrupted information and are also used to obtain the next action. The method is trained in simulation and demonstrated on hardware, on a set of challenging tasks.

### Strengths
- Blind bipedal locomotion is a challenging problem, which the paper successfully addresses, bridging the sim-to-real gap and demonstrating a good performance of the trained policy on hardware. 

- The proposed method outperforms the evaluated baselines, in general, in terms of overall reward and velocity tracking performance.

### Weaknesses
- The paper is good as a systems paper. However, my main concern is that the conference might not be a good fit for the paper. The scope of the paper is very specific to one domain and while the proposed method is potentially useful for other applications, showcasing its usefulness in different domains will require further experiments on a broader set of environments and tasks, which is outside the scope of the rebuttal period. 

- The paper would benefit from qualifying,  justifying, or explaining in more detail statements like:

  - L 223: “After establishing an understanding of the interaction process with the environment, the robot should learn and comprehend interaction information during dynamic prediction.”  What do you mean by “understanding of the interaction process with the environment”? What do you mean by “comprehending interaction information during dynamic prediction?” How can you evaluate that? 

  - L 230. “However, partial observations cannot withstand the challenges posed by complex environments”. I guess, the sentence is trying to make a point about the fact that policies that receive only partial information do not perform well in complex environments and that a history of partial observations helps to improve the performance of the policy.  However, the sentence is not clear by itself.

  - L318: “Learn richer and more precise representations in the latent space”.  How can a latent representation be more precise? 



- The paper has several typos, many of them related to missing spaces before citations. To mention a few: 
  - L062: Missing word - complex non-linear* dynamics.
  - L107: Missing spaces - learning_Escontrela
  - L112: Typo - to solve* this problem.
  - L113: Missing space -  (2023)_that utilizes
  - L116: Missing space - such as gait_Margolis
  - L419: Broken reference Fig ??.

### Questions
- What is the rationale behind the order of the updates? After optimizing the dynamics models, why is the policy optimized first and not the regression module of the hidden variables? 

-  It is already common practice to use a buffer of past observations as input for the policy in locomotion applications [1][2]. However, the size of the buffer varies depending on the setting. How does the performance of your method change as the number of past observations increases? 

   - [1] Joonho Lee et al., Learning quadrupedal locomotion over challenging terrain.
   - [2] Gabriel Margolis et al., Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents a framework for humanoid robot locomotion that uses the Transformer-XL architecture to learn a blind locomotion controller for a range of different, flat and non-flat, terrains. The key contribution is the introduction of a physical interaction world model that "understands" the dynamic interaction between the robot and its environment. The model is trained in simulation and deployed both in sum and on a real-world robot. The authors presents a set of comparisons with other methods and a baseline approach. The results show how the proposed framework is able to walk over a set of trial terrains.

### Strengths
The results in simulation show that the approach outperforms the baseline and related methods, and the oracle-based approach in vehicle tracking.
The combination of the actor-critic with the transformer-xl approaches and the application to humanoid locomotion is somewhat novel in a blind locomotion context.
The deployment on a real robot platform is an added strength for the work.

### Weaknesses
The main weakness of the paper is the lack of robust evaluation especially in the case of the real robot examples. Traditionally for locomotion controllers one can use measures as robustness to disturbances and noise to evaluate the robustness of a given locomotion approach, this would help in quantifying the resulting approach.

Why is the baseline chosen a good representation of what is available as state of the art for humanoid locomotion?

Parts of the work are not presented in good detail and there are parts of the text that are unclear or need to be expanded on. For example, line 233, what is sufficient information? in line 250, "The actor hidden variables are guided by the critic hidden variables to learn more interaction information" this is unclear, in line 260, what is the contact mask and what are the clock inputs? lines 275 and 286 what do the critic_trans(.) and actor_trans(.) functions do?

The conclusion section is hard to follow, especially where many different vague terms are used in a single sentence, for example, "This framework incorporates the world model concept, forming self-awareness, environmental understanding, dynamic prediction, and observation expansion through historical sequences". It is not clear how self-awareness or environmental understanding has been addressed with the approach.

Line 071. Sentence is not clear.
Line 088. What does affordance-based mean in this context?
Related work section, many spaces are missing.
Line 261, system(\theta_{xy}) is not defined 
Line 317, Better capture with respect to what?

### Questions
The computational cost of the approach is not discussed in the paper, a discussion on training and deployment could add to the presentation of the approach.

Line 421, The method outperforms the oracle-based method that has access to privileged information. A discussion to how this is possible or what information is being exploited to achieve this can add to the depth of the analysis of the results.

Latent layer analysis, it is not clear what this represents. What do we expect to see here and how the plots support this?

Lin 470, it is unclear what self-awareness and environmental perception refers to.

### Soundness
2

### Presentation
1

### Contribution
1
