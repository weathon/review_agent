# RILe: Reinforced Imitation Learning

- Decision: Reject
- Scores: 3, 5, 3

## Abstract
Learning to imitate behaviors from a limited set of expert trajectories is a promising way to acquire a policy. In imitation learning (IL), an expert policy is trained directly from data in an efficient way, but requires vast amounts of data. On the other hand, inverse reinforcement learning (IRL)  deduces a reward function from expert data and then learns a policy with reinforcement learning via this reward function. Although this mitigates the data requirement of imitation learning, IRL approaches suffer from efficiency issues because of sequential learning of the reward function and the policy. In this paper, we combine the strengths of imitation learning and inverse reinforcement learning and introduce RILe: Reinforced Imitation Learning. Our novel dual-agent framework enables joint training of a teacher agent and a student agent. The teacher agent learns the reward function from expert data. It observes the student agent’s behavior and provides it with a reward signal. At the same time the student agent learns a policy by using reward signals given by the teacher. Training the student and the teacher jointly in a single learning process offers scalability and efficiency while learning the reward function helps to alleviate data-sensitivity. Experimental comparisons in reinforcement learning benchmarks against imitation learning baselines highlight the superior performance offered by RILe particularly when the number of expert trajectories is limited.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes RILe, which is a teacher-student imitation learning architecture by introducing an intermediary teacher agent into the GAIL optimization.

### Strengths
1. The framework seems interesting and new
2. The algorithm is simple and seems to be better than previous baselines, but I have questions about those results, see below.

### Weaknesses
1. The motivation and advantage of this work compared with previous methods is not clear.
2. Results are simply not good enough. For example, on Atari, most of the results do not solve the task (like -20 in Pong actually makes no difference with -21) and is similar to BC (or even worse); on Mujoco, the results are not consistent with those reported by the previous works.
3. Writing is clear the most of time.

### Questions
1. What is the motivation of such a teacher-student framework? By adding the complexity to GAIL, what is the advantage compared with it? Under Eq (12) the authors said
``This enables us to train the student agent in a standard RL setting where it receives rewards from the teacher to ensure that its policies mimic expert behavior. Thus, we can break the data-policy connection common to existing IL solutions and facilitate a less data-sensitive learning process that can generalize over the specific state-action pairs of expert trajectories.`` I do not see any `break the data-policy connection common` to GAIL, and I also do not understand why such a learning style `facilitate a less data-sensitive learning process`. GAIL also has such advantages.

2. `This reward structure allows us to utilize any single-agent reinforcement learning algorithm, instead of using supervised learning to optimize over loss functions defined in Equations 11 and 12.` What is the difference to GAIL? GAIL also allows any single-agent reinforcement learning algorithm. And why should we conduct `supervised learning over loss functions defined in Equations 11 and 12`? What are you specifying?

3. In Table 1, the Atari experiments
    - (a) Why the BC behaves better for traj. num. = 1 than traj. num. = 100? Especially on Asteroids and Qbert.
    - (b) Why the std for all baselines is 0? How many training/testing seeds are you using? How do you calculate the averaged results?

4. In Figure 2, the Mujoco experiments
    - (a) What is 2.5 expert trajectories?
    - (b) Why the results of baselines, like GAIL and AIRL is not consistent with the report from their paper or many previous imitation learning papers? What implementation are you using? What kind of expert demo are you using? From my experience, GAIL can easily solve Hopper and Walker.
    - (c) How many training/testing seeds are you using?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an interesting method to the imitation learning problem. It uses existing Adversarial Imitation Learning method to train a teach policy, and uses the action probability of the teach policy as a reward to train the (behavioral) student policy. Experiments are carried out on several Mujoco and Atari environments and outperform previous AIL/IRL methods like GAIL and AIRL.

### Strengths
1. The proposed idea is simple and straightforward.

2. The paper is easy to follow, and strong empirical performance is achieved on the Mujoco benchmark.

### Weaknesses
1. The major weakness of this paper is that is in its soundness. The teacher policy is trained to maximize the distribution matching between the student policy's state action visiting distribution and the expert's distribution, and its output probability is used as the reward for the student policy. It is unclear whether the student policy can still perform correct distribution matching. I would like to see some theoretical results on this. For the current version the method looks more like a heuristic but not a theoretically sound approach.

2. It is unclear to me why adding a teacher policy as a reward processor to train the behavior policy would be better than using AIL/IRL reward directly. Besides theoretical results, I would also like to see some intuitive & in-depth discussions on this point. Also, it looks like the choice of AIL/IRL reward for training the teacher policy should not be limited to GAIL, but the authors only base their method on GAIL. The authors can try to use different choices to see if their proposed approach can lead to some improvement.

3. I am also wondering the online sample efficiency of the proposed method. I think it would be necessary to add a return plot for comparison.

### Questions
See the weakness parts.

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
This paper proposes an approach to imitation learning that combines approaches from inverse reinforcement learning and more standard imiation learning methods. The approach involves training a discriminator to distinguish between a teacher's actions and the expert actions, then using the teacher to "distill" information into a student model by using the teacher's action as a reward function. The authors evaluate their method on Atari and MuJoCo tasks, where they show that their method outperforms GAIL, AIRL, and BC.

### Strengths
- I believe that the approach of combining GAN-like training of a teacher with distillation into a student model is novel. 
- The experimental results show that RILe has a substantial improvement over prior baselines on Atari and MuJoCo tasks.

### Weaknesses
- The presentation of this paper needs significant improvement, especially in the technical sections. It uses ambiguous and incorrect notation at times, and I am unsure what exactly the reward for the student is exactly. Several specific examples include:

- Abstract: "expert policy is trained directly from data in an efficient way, but requires vast amounts of data." -> This seems like a contradictory statement. Requiring vast amounts of data is normally not seen as "efficient".
- Sec. 3: Pi* is overloaded between equation 1 and equation 3 and refer to the optimum for different objectives. 
- The notation IRL(t) should be defined before equation 3.
- Sec 4: Thus, it evaluates the state action pair of the student agent s T = (sE, aS) and chooses an action aT that, in turn, becomes the reward of the student agent aT = rS. -> This is a syntax error - rewards are scalar values, which an action is generally not.
- Eq 8 -> Same issue. What does it mean to minimize an action? Do you mean to minimize MSE between the action of the student and the expert?
- "the action of the teacher is the reward of thestudent: rS = πT ((sE, aS)" -> missing a closing parentheses.

- The performance of the baselines looks extremely poor (worse than performance on the same tasks presented in the original papers).This seems indicative of poorly tuned baselines. I do not completely trust the results until the questions I have raised in the questions section have been addressed.

### Questions
Sec 5.3: What does it mean to use 2.5 expert trajectories? What does half a trajectory mean?

Experiments: What was the performance (total reward) of the expert trajectories used for each experiment? This would be a good number to show as an upper bound / oracle of performance.

The section mentions different amounts of expert trajectories used, but only one table is reported. Are all of the different # trajectories averaged into the same table?

Please report learning rates, other hyperparameters, and hyperparameter sweeping strategies in the experiments or appendix (esp for the different components: the student, teacher, and discriminator).

Why do the reported baselines (AIRL, GAIL) perform so poorly, even poorer than performance reported in the original papers? For example, GAIL reports expert performance on the Walker and Hopper tasks in their paper, yet in the baselines reported here the performance is very poor. Would appreciate if the authors shed some insight into the discrepancy in results (e.g. is it because of the reduced number of expert trajectories?)

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
