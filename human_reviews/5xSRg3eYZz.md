# VVC-Gym: A Fixed-Wing UAV Reinforcement Learning Environment for Multi-Goal Long-Horizon Problems

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 8

## Abstract
Multi-goal long-horizon problems are prevalent in real-world applications. The additional goal space introduced by multi-goal problems intensifies the spatial complexity of exploration; meanwhile, the long interaction sequences in long-horizon problems exacerbate the temporal complexity of exploration. Addressing the great exploration challenge posed by multi-goal long-horizon problems depends not only on the design of algorithms but also on the design of environments and the availability of demonstrations to assist in training. To facilitate the above research, we propose a multi-goal long-horizon Reinforcement Learning (RL) environment based on realistic fixed-wing UAV's velocity vector control, named VVC-Gym, and generate multiple demonstration sets of various quality. Through experimentation, we analyze the impact of different environment designs on training, assess the quantity and quality of demonstrations and their influence on training, and assess the effectiveness of various RL algorithms, providing baselines on VVC-Gym and its corresponding demonstrations. The results suggest that VVC-Gym is suitable for studying: (1) the influence of environment designs on addressing multi-goal long-horizon problems with RL. (2) the assistance that demonstrations can provide in overcoming the exploration challenges of multi-goal long-horizon problems. (3) the RL algorithm designs with the least possible impact from environment designs on the efficiency and effectiveness of training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents VVC-Gym, a fix-wing UAV RL environment for multi-goal long-horizon tasks. The environment is built upon real dynamics and rooted in a real-world UAV control problem. Various simulation shows VVC-Gym is suitable for studying the influence of environment designs on multi-goal long-horizon problems, the necessity of expert demonstrations on the task and suitable RL algorithm factors for the environment.

### Strengths
Fix-wing UAV control inherently requires multi-goal and long-horizon RL. The biggest strength to me is that the problem and the environment is based on real need, not simulations. I also appreciate the explorations of the authors take on how to design the environment and how to make RL work in this context.

### Weaknesses
1. I am not a researcher on fixed-wing UAV control and multi-goal long-horizon problem. The biggest problem for me is that while the authors present an environment for future researchers, they failed to give a clear guidance on how to use this environment for benchmark. For example, as a user of the environment, I want to know:

* What are the number of tasks in this environment, and what are their difficulties, respectively? Seems to me there is only one task. In this way, users might not use this environment in their study since it is likely to be rejected by other reviewers by the reason like "small amount of experiments".

* What are the recommended configuration, hyperparameters and algorithms for me to start with, for researchers in the field of (1) GCRL (2) demonstration-based RL and (3) fixed-wing UAV control?

* As for demonstrations, if I want to generate trajectories to support my own task, do I have tools for that in the environment? Are there standardized demonstrations recommended by the authors?

2. If conventional algorithms such as PID/MPC can already solve the task, why should we use RL for this problem? Maybe we can significantly reduce the complexity of the problem by planning?

3. After searching on google, I find some repo doing similar tasks like this: https://github.com/liuqh16/LAG, which is also based on fixed wings. Can you also discuss existing simulators of fixed-wing UAV control?

4. Some minors: in page 7, ... with the results presented in Fig. 2b, which should be Table. 2b. Please also check others.

To sum up, as RL researchers, we always need real-world environments for simulations and I appreciate the large amount of work by authors to develop this environment and make it work. However, I'm not sure how to use the environment without pain. Maybe the authors could revise the paper by giving more recommendations to users?

### Questions
Please see "Weakness" section. I will read the rebuttal and adjust scores in reviewer-author discussion.

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
This paper propose a multi-goal long-horizon Reinforcement Learning (RL) environment based on realistic fixed-wing UAV’s velocity vector control, named VVC-Gym, and generate multiple demonstration sets of various quality. I suggest that the authors improve their academic writing skills, especially in organizing their ideas properly and expressing their points of view.

### Strengths
This paper propose a multi-goal long-horizon Reinforcement Learning (RL) environment based on realistic fixed-wing UAV’s velocity vector control, named VVC-Gym, and generate multiple demonstration sets of various quality.

### Weaknesses
I suggest that the authors improve their academic writing skills, especially in organizing their ideas properly and expressing their points of view. For example, the paper first presents the concept of multi-goal long-horizon problems. It does not clearly define the problem or list related works to support the issues, but it spends many words discussing the GCRL, which confuses me about the paper's writing logic. If the authors think the multi-goal long-horizon problems belong to the GCRL, they need to discuss the background of the GCRL and the current state-of-the-art. Then, they need to analyze their relationships and provide recent research on the challenge. Moreover, I believe this problem might belong to the multi-objective multi-agent area. Please check the paper: 
Rădulescu, R., Mannion, P., Roijers, D.M. et al. Multi-objective multi-agent decision making: a utility-based analysis and survey. Auton Agent Multi-Agent Syst 34, 10 (2020).

I feel it is more like a technical paper introducing a platform in reinforcement learning than a research paper. Additionally, I suggest the authors add more baseline algorithms to their experiments, not just PPO.

### Questions
Generally, I strongly recommend several papers, as shown below, in which authors can learn how to improve academic writing skills and organize corresponding ideas from them.

1) Yang, Q., & Parasuraman, R. Bayesian strategy networks based soft actor-critic learning. ACM Transactions on Intelligent Systems and Technology (TIST).

2) Mannion P, Devlin S, Duggan J, Howley E. Reward shaping for knowledge-based multi-objective multi-agent reinforcement learning. The Knowledge Engineering Review. 2018

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a multi-goal long-horizon Reinforcement Learning (RL) environment based on realistic fixed-wing UAV velocity vector control, named VVC-Gym. The proposed environment is studied through various ablation studies using different Goal-Conditioned RL (GCRL) algorithms and is equipped with multi-quality demonstration sets. Baselines are also provided on the environment.

### Strengths
1. The proposed benchmark is useful for the GCRL community. 

2. The benchmark is equipped with demonstrations with different quality and several baseline algorithms. 

3. The influence of the environment-related parameters is studied via ablations.

4. Baseline methods are provided with the benchmark.

5. The paper is well-written and easy to read.

### Weaknesses
1. Some writing issues:

    - The citation format is not very suitable. The authors may consider using `\citep` for most of the citations.
    
    - Line 349 and Line 357: I guess Fig. 2a and Fig. 2b should be Table. 2a and Table. 2b.

    - Figure 5 is provided but not mentioned in the text.

2. The only task provided is the velocity vector control. If various types of different tasks are provided, the paper will have a larger influence and contribution.

### Questions
1. In Fig. 6c, we can conclude that there are several distinct stages in training where different termination conditions are triggered. However, what can we conclude from Fig. 6d?

2. Seems that the success rates of all the baseline methods are often low (less than 50%). What might be the possible reasons that the GCRL algorithms fail to achieve a higher success rate?

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
3

### Summary
The Paper presents a Multi-Goal Long Horizon RL Environment based on Fixed Wing UAV Velocity Vector control. The paper further provides a set of various demonstrations, analyzing the quantity and quality of the demonstration and their effect of training. The paper claims that the environment presented is suitable for studing curriculum learning that combined imitation learning and RL, influence of the environment designs on multi-goal long-horizon problems.

### Strengths
The paper provides a detailed analysis of the presented environment, including evaluation of Demonstrations and training of various RL Algorithms with and without Curriculum Methods.

Clear problem formulation and transition function.

Detailed Appendix and Supplementary Materials containing demonstrations.

### Weaknesses
- It is unclear what the novelty of the environment is compared to other simulation based fixed wing control environments.

- It is unclear if the machine generated demonstrations can be substituted by human demonstrations.

### Questions
- It is unclear how the environment would work with the presence of Adversarial Fixed Wing Agents [1, 2].

- Is the environment applicable to Inverse Reinforcement Learning applications?

- How much reward engineering is required to train an agent to learn a different policy than what is provided?

- What are the challenges of transfering models trainined in this environment to real world UAVs?

[1] Strickland L. G., “Coordinating Team Tactics for Swarm-vs-Swarm Adversarial Games,” Ph.D. Thesis, Georgia Inst. of Technology, Atlanta, GA, July 2022, [https://smartech.gatech.edu/handle/1853/67090](https://smartech.gatech.edu/handle/1853/67090) 
[2] B. Vlahov, E. Squires, L. Strickland and C. Pippin, "On Developing a UAV Pursuit-Evasion Policy Using Reinforcement Learning," 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA), Orlando, FL, USA, 2018, pp. 859-864, doi: 10.1109/ICMLA.2018.00138.

### Soundness
3

### Presentation
3

### Contribution
3
