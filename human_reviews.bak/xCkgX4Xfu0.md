# A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
In this paper, we present empirical evidence of skills and directed exploration emerging from a simple RL algorithm long before any successful trials are observed. For example, in a manipulation task, the agent is given a single observation of the goal state (see Fig. 1) and learns skills, first for moving its end-effector, then for pushing the block, and finally for picking up and placing the block. These skills emerge before the agent has ever successfully placed the block at the goal location and without the aid of any reward functions, demonstrations, or manually-specified distance metrics. Once the agent has learned to reach the goal state reliably, exploration is reduced. Implementing our method involves a simple modification of prior work and does not require density estimates, ensembles, or any additional hyperparameters. Intuitively, the proposed method seems like it should be terrible at exploration, and we lack a clear theoretical understanding of why it works so effectively, though our experiments provide some hints.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work builds upon previous work in contrastive reinforcement learning. What's unique to this work is how their learning algorithm conditions on a single goal; this has implications on the distribution of data within the replay buffer which ultimately impacts the learning dynamics and exploration behaviour of the agent. The authors argue that this approach results in emergent skill learning without explicitly having to define a reward function or learning curriculum for the task being solved. This argument is supported with simulated experiments on robotic manipulation and maze traversing benchmarks. The performance of their approach is compared to contrastive reinforcement learning using a range of human-specified goal difficulties and existing approaches (SAC+HER and RIS). The authors justify their claim with empirical results, leading to the insight that contrastive reinforcement learning with conditioning on a single goal demonstrates advantageous exploration and skill learning behaviours relative to existing approaches that seek to learn effective exploration and skill learning strategies.

### Strengths
- The authors provide valuable insights into the effectiveness of training a policy conditioned on a single goal with contrastive reinforcement learning in the single-task setting. 

- The authors contribute to the discussion on skill learning through empirically demonstrating the emergence of skill learning with contrastive objective functions.

- The authors validate each of their claims and provide discussions on counter arguments (e.g. overfitting single-task) that support their empirical results. 

- The authors provide sufficient information and high-quality materials to reproduce their results.

### Weaknesses
- There could be more analysis of emergent skill learning, in the current draft the authors mention that fixing a checkpoint and examining the qualitative behaviour of the policy they see evidence of skill learning, further justification of this claim is required in my opinion. 

- The authors highlight an awareness of this point but further analysis on why this strategy works would be useful. 

- It would be interesting to see if this approach results in exploration strategies that are capable of solving more complex tasks. For example tasks that encounter Sussman's anomaly, where naive subgoal design may not work well. I am unsure how this approach would perform but it would be promising if it did well and might further justify the strength of this more general approach to learning to solve tasks with reinforcement learning. Also tasks with significant bottlenecks in which naive exploration strategies are often incredible sample inefficient, does this approach help improve performance on such tasks?

- While the authors address the concern of overfitting to the individual task being solved, they do not explore how well the learnt representations generalise. While there are examples of solving tasks with minor perturbations there isn't a discussion of more significant changes to the task environment. From a practical standpoint, it would be interesting to understand how well this approach can be used to learn value representations that generalise well across task environments. 

- This work suffers from ailments of reinforcement learning approaches more generally, most especially sample efficiency. It would be interesting to understand if the representations learnt in this work can be leveraged to address the issue of sample efficiency when learning multiple tasks (e.g. on the metaworld benchmark).

### Questions
Thank you for contributing this work. When reading the paper the emergent skill learning argument felt like it needed further justification. 


(1) I wished to clarify if the following pattern of skill emergence always held? The paper suggests yes but I wished to confirm this point. 

"For example, in all three environments, the agent (1) first learns how to move its end-effector to
varying locations, (2) then learns how to nudge and slide the object, and (3) finally learns to pick up
and direct the object. "

(2) I also wished to ask when you mention fixed sampling intervals for checkpoints, what fixed interval was used? Were there any exceptions to the skill learning trends identified in (1) (e.g. checkpoints with purely random behaviours) and if so how frequent were exceptions?

In answering (1) and (2) it would be useful to provide:

(a) Video's indicating the progression of skill learning with details of the checkpoints, this would be helpful for validating the qualitative behaviours claimed within the paper. 

(b) Not entirely necessary but it would be nice to have quantitative metrics for characterising the various skills. For instance when you mention learning how to move the end-effector, distinguishing between purely random movements and directed movement would be useful, there are certain quantitative metrics that could be used, such as correlations between actions. Contrasting this with naive exploration approaches such as SAC may yield insights and further validate the skill learning claim. Similarly for learning to interact with objects and eventually pick them, characterising these skills quantitatively would be useful. 

(c) It may be possible to perform clustering on policy rollouts data to identify distinct exploratory behaviours as learning progresses. This would be a valuable addition.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors describe a curious phenomenon observed in some simulated robot manipulation environments used for RL. When using contrastive RL, using a single hard goal (one far away from the robot corresponding to task success) led to better learning outcomes than using a human designed curriculum of easy and hard goals.

This is observed across 4 environments, a Sawyer bin placing task, box placing task, peg insertion, and a 2d maze navigating task.

The authors provide a number of ablation experiments trying to study why this phenomenon occurs.

### Strengths
The fact that a single hard goal is sufficient with contrastive RL is certainly surprising, and the paper is upfront in not having a good explanation why this occurs. The fact it occurs over a few environments provides some evidence it is not simply a one-off occurrence.

### Weaknesses
I am suspicious that the results are too good - that there was no environment where using multiple goals performed best. Broadly my concern is that maybe these environments are too easy. If the model is able to succeed at the task given just the final hard goal, perhaps it's too hard to design a good dense reward or curriculum of goals to speed up learning. This wouldn't be too surprising, it's often been remarked that good partial reward design is difficult.

I'm also not sure how "single goal" the final method is. In particular, Figure 10's caption was confusing to me. It seemed to suggest that in their method, the actor loss uses multiple goals, rather than a single goal? If so, this doesn't really seem like "1 goal is all you need". Essentially I think the authors may be overgeneralizing their conclusions.

My understanding so far is this:

* A contrastive critic is learned via contrastive RL, defining reward by $\phi(s,a)^T \psi(s_f)$, where $s_f \sim Geom(1-\gamma)$ steps in the future
* When generating data, we use a single hard goal $s^*$ and act according to $\pi(a|s,s^*)$
* When updating the actor, we sample a trajectory from the replay buffer of data generated according to $\pi(a|s,s^*)$. But, for each initial $s$, we then sample $s_f \sim Geom(1-\gamma)$ within the trajectory, and apply gradient updates as if we collected data according to $\pi(a|s,s_f)$, even though we actually collected data according to $\pi(a|s,s^*)$.

In which case, at most you could say 1 goal is the only requirement for data collection, but the policy still needs to be trained on every intermediate goal observed to achieve good performance.

### Questions
Do the authors have any statistics on eval performance over the same distribution of goals used to generate the replay buffer? One natural argument is that, in so far as each goal can be viewed as a separate task, the model will be best at the goal distribution that appears in the replay buffer. I believe this is distinct from the overfitting experiments, because in those experiments the final evaluation number is still only the final hard eval goal.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work aims to solve goal-conditioned RL (GCRL) tasks where only one goal is desired without access to rewards (dense or sparse), curriculums, or demonstrations. It proposes modifying standard Contrastive RL (CRL) by using only a single goal in rollouts while still using multiple goals while training. The paper runs experiments to show that the specific combination of single-goal rollouts, multi-goal training, and contrastive representations are needed to achieve good results. The paper also includes some additional analysis to examine single-goal CRL under impossible goals and to explore the representations learned by CRL.

### Strengths
Under the conditions set by the work, i.e. learning single difficult goals without rewards, demonstrations, curriculums, etc, the single-goal CRL clearly outperforms other methods. Furthermore, this method reduces assumptions compared to standard RL. Finally, an important finding of this paper is that a policy can be trained under contrastive learning to find a difficult goal without providing intermediate easier goals.

### Weaknesses
Firstly, it is unclear how the authors selected or defined “difficult goals”. Are the difficult goals sampled by failure cases of CRL? How many difficult goals are the results evaluated on? For example, if this method is effective on one particular goal, but not others, then it would not be effective in practice.

Secondly, the authors claim that the first ablation disproves the hypothesis that single-goal CRL “learns useful representations and an effective policy only for states along that goal path” (line 476). However, since the single-goal CRL fills the replay buffer with rollouts where the policy aims to reach goal $s^*$, and the actor is only being updated with states from the replay buffer, how does this prove that the single-goal rollout + multi-goal training policy is not overfit to/does not only contain good representations for states along the goal path?

Finally, Fig. 11 shows that a monolithic critic CRL (Range of Difficulties) can reach approximately 60% success rate on the single hard goal on Sawyer Box after 100,000 steps. However, Fig. 3 shows that standard CRL (range of difficulties) can only achieve up to approximately 40% success rate on Sawyer Box after 1,000,000 steps. It seems this might contradict the assertion that the separate representations are important? (It is possible that the methodology for selecting the “difficult goals” is impacting these results.)

### Questions
How are the “difficult goals” sampled, how many are sampled, etc? Why are thy chosen in this way?

Are there other experiments done (e.g. something like the perturbation test, or a test where the single-goal policy shows useful representations or reasonable success on goal states outside of those in the goal path) which would alleviate concerns that this method is simply overfitting to solve for this one goal?

### Soundness
2

### Presentation
3

### Contribution
2
