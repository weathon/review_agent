# Model-free reinforcement learning with noisy actions for automated experimental control in optics

- Avg Score: 5.25
- Decision: Reject
- Scores: 3, 5, 8, 5

## Abstract
Setting up and controlling optical systems is often a challenging and tedious task. The high number of degrees of freedom to control mirrors, lenses or phases makes automatic control challenging, especially when the complexity of the system cannot be adequately modeled due to noise or non-linearities. Here, we show that reinforcement learning (RL) can overcome these challenges when coupling laser light into an optical fiber, using a model-free RL approach that trains directly on the experiment without pre-training. By utilizing the sample-efficient algorithms Soft Actor-Critic (SAC) or Truncated Quantile Critics (TQC), our agent learns to couple with 90% efficiency, comparable to the human expert. We demonstrate that direct training on an experiment can replace extensive system modeling. Our result exemplifies RL's potential to tackle problems in optics, paving the way for more complex applications where full noise modeling is not feasible.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper discusses using RL for automated control in optical experiments, specifically for coupling laser light into optical fibers. The authors demonstrate that an RL agent, trained directly on the experiment without pre-training, can achieve coupling efficiencies comparable to human experts. The authors use standard SAC and TQC.

### Strengths
1. The paper presents a real-world deployment of reinforcement learning, which I find valuable given the challenges of implementing RL on hardware.

2. The experimental setup is thoroughly detailed, offering insights for practitioners and researchers looking to deploy RL in a similar application.

### Weaknesses
1. My main concern is that this paper reads more like a technical report on applying standard RL algorithms to a specific problem. The application itself does not appear to be novel, and given the lack of additional contributions beyond using off-the-shelf RL techniques, I believe this paper is not relevant to ICLR. I would recommend the authors consider submitting to venues focused on optics, sensors, or robotics.

2. The authors highlight challenges in applying RL to real-world problems due to poor sample efficiency and argue that their environment is particularly difficult due to partial observability and noisy actions. These are valid challenges that could inspire algorithmic innovations to tackle them. However, the authors simply use existing model-free RL algorithms without any modifications. If sample efficiency is a concern, model-based methods would be more appropriate, or the authors can consider pretraining the policy with expert data. Similarly, if partial observability is an issue, methods designed for POMDPs or architectures with memory should be considered. Lastly, the impact of noisy actions is not clear to me, even though it is in the tilte of the paper. In my opinion, noisy actions is nothing but stochasticity in the transition dynamics, and RL algorithms are by definition applicable to stochastic transition dynamic. Given the fact that standard RL algorithms have worked in this case, the authors should either downplay these challenges or address them with appropriate methods and more thorough investigations.

3. I identified some inaccurate statements in the paper:
* L43: “Only very few [RL applications] were done in real-world environments.” This is misleading as there are hundreds of papers on RL applied to real-world robotics, beyond simulations. Additionally, RL has been applied to complex real-world tasks like fusion reactor control [1], balloon navigation [2], and data center cooling [3].
* L474: “Our experiments are the first steps towards RL in optics.” While I am not an optics expert, several papers have already applied RL in this field, some of which are cited by the authors, though others are missing [4, 5, 6]. The primary distinction in this work seems to be the direct training of the RL agent on hardware, which, albeit challenging, is not considered a novel contribution. In fact, as stated in Point 2 above, perhaps offline pretraining of the policy on expert data would be an even more suitable direction.

4. The main body of the paper includes many experimental details that are unnecessary for readers unless they intend to replicate the exact setup. This makes the paper harder to follow and obscures key information. I suggest moving these details to the appendix. Additionally, the paper contains repetitive statements, with the same points being reiterated in slightly different ways (e.g., strengths of RL algorithms, challenges in applying RL, etc.) without adding new insights. Reducing these repetitions would improve clarity.

[1] Tracey, Brendan D., et al. "Towards practical reinforcement learning for tokamak magnetic control." Fusion Engineering and Design 200 (2024).

[2] Bellemare, Marc G., et al. "Autonomous navigation of stratospheric balloons using reinforcement learning." Nature 588.7836 (2020).

[3] Lazic, Nevena, et al. "Data center cooling using model-predictive control." Advances in Neural Information Processing Systems 31 (2018).

[4] Pou, B., et al. "Adaptive optics control with multi-agent model-free reinforcement learning." Optics express 30.2 (2022)

[5] Nousiainen, Jalo, et al. "Laboratory experiments of model-based reinforcement learning for adaptive optics control." Journal of Astronomical Telescopes, Instruments, and Systems 10.1 (2024).

[6] Mareev, Evgenii, et al. "Self-Adjusting Optical Systems Based on Reinforcement Learning." Photonics. Vol. 10. No. 10. MDPI, 2023.

### Questions
1. I am aware of the experiments on pretraining the policy on a virtual environment and transferring it to the real setup. But that may be too challenging because of the gap between simulation and reality. However, what is the issue with pretraining the policy on expert demonstrations (from humans, or classic controllers) from the actual setup and fine tuning online on the actual setup? This can be a very reasonable approach if deploying on the actual system is expensive. 
2. How does the performance of imitation learning on expert data compare to RL?
3. Are noisy actions anything beyond stochasticity in the transition dynamics? What was done specifically regarding noisy actions?
4. What do shaded regions in the figures represent? 
5. How many seeds have been used in the experiments?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
An applied contribution proposing the use of off-policy continuous action space RL algorithms (SAC, TQC, TD3, etc.) for controlling four stepper motors tilting two mirrors in optics experiments. The goal is to automate the alignment of a laser beam for coupling to an optical fiber.

In this setup the main algorithms tested appear to perform more or less the same, with the only difference being made by the value of the target power.

### Strengths
- good exposition of the problem being solved and the associated challenges
- extensive experiments demonstrating the feasibility of the application
- detailed description of the experimental setup. Without a background in optics experimentation I got the feeling that I could reproduce the experiments to some degree if required to. The paper contains thorough descriptions of design choices regarding the actions pace, processing of observations, formulation of the reward function, reset procedure, etc.

### Weaknesses
The only reserve I have is **whether ICLR is a good venue for this submission**. This contribution demonstrates that off-the-shelf RL algorithms (literally StableBaseline implementations) can be used for automating parts of the calibration and setup of optics experiments in the presence of noisy actuators.

This in itself is not surprising nor novel. DRL algorithms have been shown to perform control in real-life experiments with noisy actuators, partial observability and, in addition to this work, noisy observations, multiple times and even for higher-dimensional problems. In its current form the paper presents no findings that can be of general interest for the broader RL community.

It would probably be an excellent contribution to a more applied venue.

**Minor:**
- figures presenting multiple algorithms and power goals in the same pane can be confusing / difficult to follow (eg.: fig 2a-b).

### Questions
- Why evaluate only TQC in fig 2e and not SAC also?
- What is the interpretation of fig 2d? Why is the number of steps of the last episode important?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The coupling of laser light into optical fibers is a difficult task in optics labs because of its high precision and sensitivity requirements. This work investigates the use of model-free reinforcement learning (RL) to automate and optimize this process. The work shows that RL can effectively handle motor imprecision and experimental noise without the need for a comprehensive simulation model. By employing the Truncated Quantile Critics (TQC) and Soft Actor-Critic (SAC) algorithms, the authors get coupling efficiencies above 90%, which is on par with human experts. By demonstrating RL's promise in automated optical alignment tasks, this work opens the door for its application in intricate optical experiments where comprehensive noise modeling or conventional control techniques are impracticable.

### Strengths
Originality: In optical systems, where simulation is difficult due to noise and system complexity, this research uses model-free reinforcement learning in an area where traditional models fall short.

The RL agent's ability to attain high coupling efficiencies directly on physical hardware, matching or surpassing human performance, is demonstrated by rigorous experimental results.

Significance: The effective use of RL in this context may encourage the development of comparable methods in other physical and optical disciplines where intricate, noisy situations defy conventional modeling.

Clarity: Detailed experimental methods improve reproducibility, and the report clearly conveys its approach and the justification for algorithm selections.

### Weaknesses
Limited Generalization: Although the RL agent performs admirably, it is designed especially for this particular experimental configuration. The significance of the paper would be enhanced by greater generalizability or adaptability to different optical settings.

Training Time: The practical implementation of this strategy in time-sensitive settings may be limited by the time needed to reach high efficiency, which can take up to four days.

Dependency on Particular Hardware: The configuration is dependent on motorized mirrors and particular power measurements, which may restrict its direct application to other labs without comparable apparatus. Investigating different sensing techniques may increase flexibility.

### Questions
Could transfer learning from pre-trained models or virtual simulations speed up the training process?

How resilient is the RL agent to modifications in the optical configuration, such as slight misalignments or shifts in the location of the optical fiber?

Could the method be applied to tasks with more complicated noise distributions or higher dimensional control requirements?

Given the lengthy training procedures, is the agent's performance sustainable in terms of wear on mechanical components?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents an application of deep reinforcement learning algorithms to a physical task of optical fiber coupling. As designed by the authors, the RL agent controls two motorized mirrors, each with two degrees of freedom. The task is to move the mirrors to align the laser beam for the highest output power.

The control problem emphasizes practicality. Concretely, the authors employ a POMDP setting where the observations only consist of fields related to the measurable power output and historical actions. This design choice ensures the policy wouldn't rely on mirror position readings (subject to drift) and other hard-to-measure metrics. As a result, a policy is learnable directly on real hardware. The paper also details the experiment design, including the reward function, resetting scheme, and curriculum learning.

Experiment results show that the proposed RL framework derives a control policy that performs the optical fiber coupling task faster than a human expert.

### Strengths
- The authors have rigorously designed the experiments, performing method selection in simulation and experimentation on hardware.
- The presentation is very clear. The paper covers the problem of interest, potential challenges, design choices, experiment designs, and results in great detail.
- The main experiment results are convincing: the RL agent can overcome action stochasticity and perform from the limited observations to solve the optical fiber coupling problem above the human level.

### Weaknesses
- It is nice to see RL applied to a physical control system and have utilities for research in another domain. However, the control problem solved in the work is relatively simple, especially when the robotics community has trained RL policies on higher degrees of freedom systems, even from image observations.
- I sense there's not much representation-learning in this project. I think the insights will be more relevant if the authors can scale the experiment up to higher-dimensional observations. For example, is there a way to have a visual observation of laser paths as input to the policy?

### Questions
- Regarding Figure 2, my understanding is that the reward function differs for different goal power levels. In this case, plotting episode returns in the same figure is not the most appropriate. Is there a way to define a normalized or common metric?
- Prior work in RL has shown that a few human demonstrations can help speed up RL, even overcoming the sparse reward problem. I wonder if using a few human demos can accelerate learning and maybe even remove the need for the dense reward function.

### Soundness
4

### Presentation
3

### Contribution
2
