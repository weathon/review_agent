# Learning Diverse Quadruped Locomotion Gaits via Reward Machines

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 3, 5

## Abstract
Quadruped animals are capable of exhibiting a diverse range of locomotion gaits. While progress has been made in demonstrating such gaits on robots, current methods rely on motion priors, dynamics models, or other forms of extensive manual efforts. People can use natural language to describe dance moves. Could one use a formal language to specify quadruped gaits? To this end, we aim to enable easy gait specification and efficient policy learning. Leveraging Reward Machines~(RMs) for high-level gait specification over foot contacts, our approach is called RM-based Locomotion Learning~(RMLL), and supports adjusting gait frequency at execution time. Gait specification is enabled through the use of a few logical rules per gait (e.g., alternate between moving front feet and back feet) and does not require labor-intensive motion priors. Experimental results in simulation highlight the diversity of learned gaits (including two novel gaits), their energy consumption and stability across different terrains, and the superior sample-efficiency when compared to baselines. We also demonstrate these learned policies with a real quadruped robot.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a reward machine to allow learning different quadrupedal gaits for quadrupedal robots. Multiple gaits are learned for a real quadrupedal robot, including novel gaits such as Three-One.

### Strengths
1. A novel way to encourage policies to produce different gaits for quadrupedal robots. Including the learning of novel gaits.

2. Good ablation studies to evaluate the importance of different components.

### Weaknesses
It will be nice to also learn transitions between gaits.

### Questions
It will be interesting to evaluate energy consumption for different gaits at different speeds, e.g., one will expect certain gaits to be more energy efficient at high speed while less at low speed. It will also be fun to try out gaits that are typical at high speed in nature, like galloping, even if only demonstrated in simulation.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces reward machines for learning different quadrupedal gaits called Reward Machine-based Locomotion Learning (RMLL). The key to the proposed approach is introducing high-level gait specifications as automaton states and a counter to control gait frequency. The authors construct a automaton via LTL formulas representing foot-contacts. PPO is used to learn a policy which takes the state as a combination of automaton state, frequency counter, proprioception and commands to output target joint angles. More rewards are given for transitioning to subsequent automaton states. All the learned gaits are demonstrated in the real-world.

### Strengths
The presented approach shows a straightforward way of learning different gaits via adding automaton structure to reward function. High-level state conditions and transitions are characterized by foot contacts and used to motivate the desired automaton state transitions. The added benefit of controlling the gait frequency at execution adds to the contributions of RMLL.

### Weaknesses
The proposed method’s novelty is highly limited and is more a robotics application of the base method proposed for using reward machines with RL (Icarte et al. 2018, 2019, 2022). Using automaton states with the state of the MDP has been introduced in Icarte et al. 2018. The authors introduce the timestep counter for controlling automaton state transitions.

The authors have not mentioned the exact representation of the automaton state $u$ in the input to the policy. Is it a vector of boolean? How is it different when the RM state is replaced with that of foot contacts? In that ablation, do you still keep $\phi$ for the No-RM-Foot-Contacts case?

Can you please clarify?: With foot-contacts, two consecutive states can have two different foot-contacts with random policy. However, with reward machines the RM state is constant until a transition happens? Also, in all the ablation of state space, the reward structure is based on the RM right?

Learning policy for individual gaits is limited contribution in itself. How is the energy consumption study relevant to show the efficacy of the proposed approach? If you already know which gait consumes least energy while maintaining stability as a function of the terrain, why cannot a terrain based reward machine states be formulated?


[1] Rodrigo Toro Icarte, Toryn Klassen, Richard Valenzano, and Sheila McIlraith. Using reward ma- chines for high-level task specification and decomposition in reinforcement learning. In Interna- tional Conference on Machine Learning, pp. 2107–2116. PMLR, 2018.

[2] Rodrigo Toro Icarte, Ethan Waldie, Toryn Klassen, Rick Valenzano, Margarita Castro, and Sheila McIlraith. Learning reward machines for partially observable reinforcement learning. Advances in Neural Information Processing Systems, 32:15523–15534, 2019.

[3] Rodrigo Toro Icarte, Toryn Q Klassen, Richard Valenzano, and Sheila A McIlraith. Reward ma- chines: Exploiting reward function structure in reinforcement learning. Journal of Artificial In- telligence Research, 73:173–208, 2022.

### Questions
See weakness above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors introduce reward machine, a state machine based mechanism to shape complex, state/time dependent rewards for dynamic locomotion control problems. For each desired quadruped gait, an automata is constructed to modulate foot contact transitions and timings. Then, the automata state, proprioceptive state (including estimations) from the robot, as well as gait parameters are used as the state vector for reinforcement learning training. The authors train a few different gaits in simulation and transfer the policies to the real hardware.

### Strengths
The strengths of the paper include:

1) Introduction of the reward machine for locomotion control and gait specification.
2) Sim2real transfer of learned policies to the real robot

### Weaknesses
The weaknesses of the paper are:

1) While the concept of the reward machine is new especially in the locomotion learning community, in reality it is merely a fancy way of constructing a state machine which controls the gait transition. 
2) The tasks in this paper are not novel. I see only flat terrain locomotion with a few gaits, and it is hard to justify why a complex state machine is needed, given there are works that can also achieve diverse gaits with time based rewards: "Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior".
3) Other than that, the learning is conducted in Isaac Gym with PPO and there is limited novelty.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a reward design system via a set of state machines that consists of individual conditions on each of the foot contacts. This allows the user to specify different types of gaits easily. Diverse gait policies are trained via sim2real (in isaac gym) and deployed on hardware. The videos show gaits like jumping, trotting, running etc.

### Strengths
- The problem of automatic reward design is important for training more general policies
- The robot videos look good and the gaits are very diverse
- The motivation and approach are well-presented
- There is a lot of good analysis on the experiments and ablations, especially about the differences in the gaits

### Weaknesses
- I am unclear about the exact novelty of this approach - as it is already common in RL-based locomotion setups to use foot poses to generate different gaits. 

- It would be more interesting to see how these reward machines can be used to do more complex, long-horizon tasks, like walking with multiple gaits or imitating a long reference trajectory. Specifically, how can one transition between different machines?

### Questions
See weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
