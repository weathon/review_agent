# Revisiting Familiar Places in an Infinite World: Continuing RL in Unbounded State Spaces

- Decision: Reject
- Scores: 6, 3, 5, 6

## Abstract
Deep reinforcement learning (RL) algorithms have been successfully applied to train neural network control policies for many  sequential decision-making tasks. However, prior work has shown that neural networks are poor extrapolators and deep RL algorithms perform poorly with weakly informative cost signals. In this paper we show that these challenges are particularly problematic in real-world settings in which the state-space is unbounded and learning must be done without regular episodic resets. For instance, in stochastic queueing problems, the state space and cost can be unbounded and the agent may have to learn online without the system ever being reset to states the agent has seen before. In such settings, we show that deep RL agents can diverge into unseen states from which they can never recover, especially in highly stochastic environments. Towards overcoming this divergence, we introduce a Lyapunov-inspired reward shaping approach that encourages the agent to first learn to be stable (i.e. to achieve bounded cost) and then to learn to be optimal. We theoretically show that our reward shaping technique reduces the rate of divergence of the agent and empirically find that it prevents it. We further combine our reward shaping approach with a weight annealing scheme that gradually introduces the pursuit of optimality and a log-transform of state inputs, and find that these techniques enable deep RL algorithms to learn performant policies when learning online in  unbounded state space domains.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work focuses on reinforcement learning in real-world reset-free environments where state is not necessarily bounded. It shows that vanilla policy gradient algorithms (ppo) diverge in these cases. It proposes a method, STOP, that learns to be stable at first and seek optimality afterwards, which is successful in three synthetic environments.

### Strengths
I think the problem that this work considers is very important, and one of the main issues that hinder the applicability of reinforcement learning in real-world systems.

### Weaknesses
There exists a line of work on using training wheels for RL in real systems, e.g., [Towards Safe Online Reinforcement Learning in Computer Systems](http://mlforsystems.org/assets/papers/neurips2019/towards_mao_2019.pdf), that this work ignores. Furthermore, no quantitative comparison is done with methods that learn to reset, some of which is mentioned in the related work section.

### Questions
1. How do you deal with exploration in the presence of stability objective? Aren't these two objectives at odds with each other?
2. How should one set the schedule for lambda? How does it interact with exploration hyper parameter schedules?
3. How can state get unbounded in a real problem? It seems to me that there is always going to be a physical limit to the state space of real systems, e.g., maximum queue lengths in the queuing environment that you explored.
4. As explained in the second paragraph of 5.2., the choice of cost function and its interplay with stability objective is problem specific. How should one choose a good cost function?
5. Are your evaluation environments based on real problems? If yes, can you elaborate on the real problems?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The reviewer is not an expert in average reward RL research field. This paper considers RL in average reward setting with unbounded state space and unbounded cost function. The paper first empirically demonstrated the failure or divergence encountered by existing average reward algorithm in an unbounded grid world, then proposed three techniques to stabilize learning. The techniques include a reward shaping, inspired by Lyapunov function, a scheduling process to balance the shaped and original cost, and state transformation to handle unbounded state space. The proposed method was compared with average-reward PPO in experiments, and ablation studies were conducted to compare different designs.

### Strengths
The originality of this paper is considering the average reward RL in an unbounded state and cost setting, which is challenging. This setting is well motivated and the authors take a lot effort to illustrate the difficulty when agent faced with unbounded state and cost. 

The paper is overall easy to follow. The related works are discussed. Their own method is generally clearly described.

### Weaknesses
1. The authors motivate the reward shaping from the perspective of Lyapunov function, but lacks a background or detailed explanation. How is it related to analyzing the stochastic system is not presented.

2. The paper seems to emphasize the average cost may be unbounded and difficult to optimize, e.g., in Section 4.1 "As we described in Section 3, $J^O$ is difficult to minimize directly; even keeping $J^O$ bounded is challenging". However, in this setting, generally we are not directly optimizing this objective but optimizing a differential return (also mentioned in the paper). The differential value function is  bounded, and trust region or PPO style algorithm can be applied.

3. The paper lacks an algorithm description for their own method. Though several changes are described in Section 4, it is unclear to readers what value functions are used for shaped reward and original reward, how the policy is updated. 

Suggestions
1. This paper considers the setting of average reward. It is better to use a subsection named "average reward MDPs" to give an overview. Though some basic terms are included in Section 2, a well organization is helpful for readers not familiar with this area.

### Questions
1. The unbounded grid world is unclear to me, i.e., I am unclear what is the action space of the agent. From the description in appendix and combined with arrows in Figure 1, my understanding is agent can only take actions towards the origin. For instance, in Figure 1 (a), agent can only take actions {up, left}? Please correct if I am wrong.

2. What is the intuition of Assumption 3 in Section 2? Why is divergence in the cost equivalent to the divergence in the state?

3. In section 4.3, authors mention the reward shaping and scheduling can guide agent to "re-visit regions of states with low costs". I am not very sure about this claim. Since reward shaping is dealing with a new cost $c(s_t) - c(s_{t-1})$, why learning policy on this cost can ensure agent to visit state $s$ with low cost $c(s)$?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to deal with unbounded state spaces and unbounded cost functions in continuing RL problems. A reward shaping approach and a state transformation technique are developed to tackle the two issues, respectively. The presented method is shown to outperform baseline methods significantly in several tasks.

### Strengths
- The proposed method achieves much better performance than the baselines.
- Many ablation studies are shown to prove the effectiveness of each component.
- There is a good related work section to distinguish the current method from existing works.

### Weaknesses
My major concern is that the proposed method seems to be designed for some specific tasks with special properties, such as Assumption 3 (norm equivalence).
It is true that the state space could be unbounded since we can not control it in most cases.
However, unlike the state space, the cost function is usually manually designed. 
Thus, if an unbounded reward function is bad, a natural solution would be designing a better reward function which seems to be an easier fix.
For the tasks mentioned in this work, the hardness of tasks could be mainly due to poorly designed reward functions.
For example, can you compare methods with the reward function $r(s,a,s') = \exp(-||s+a||_2^2)$ or $r(s,a,s') = \exp(-||s'||_2^2)$ in the gridworld task?

Other issues are listed below.

- The paper could be better motivated with more real-world tasks satisfying Assumption 1-3.
- A demonstration of the generalization ability of the proposed method to other tasks is missing. It will be great if authors can test their method and show its advantage in several classic control tasks, such as Catcher, Pendulum, Acrobot, Walker, Hopper, Reacher, and Swimmer mentioned in this [work](https://drive.google.com/file/d/1l2sr7HRkeaOdZNaWNjgrqBELnTF90e70/view).
- There is no direct evidence to support the claim that "introducing stability cost helps guide the agent to re-visit regions of states with low costs." Is it possible to plot visited states with different methods to show that? For example, in the gridworld task.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work focuses on a setting where the state-space is unbounded and learning must be done without regular episodic resets. The authors find that the unboundedness causes the agent diverges, requires the agent to extrapolate. The unbounded cost makes estimation with high variance, causing instability in policy learning.

The authors propose to encourage the agent to re-visit the states. A lyapunov-inspired reward shapping approach and a weight annealing scheme and state transformations are proposed.

The experiment shows the propose STOP method can learn in the environment with unbounded states & no reset.7

### Strengths
The experiments are conducted in various environments and tasks, including a toy environment, allocation queuing and traffic control. This is a big plus to the soundness of the experiments.

I like the way how this paper is written: find a problem and then solve it.

Nice to see the code is included in the supplementary material.

### Weaknesses
Section 4.3 state is poorly motivated. Why the state transformations can solve the extrapolate burden? I have no idea why just scaling the input to the network and make it "feels like a visited state" can improve anything. The claim "all these functions reduce divergence rate" is not grounded.

The proposed method is relatively trivial, the stability cost function is simply the temporal difference of the cost (the reverse of a reward function) function $g = c(s_t) - c(s_{t-1})$. The connection between "using this bonus term in the reward function" and "revisiting familiar places" is unclear to me. 

The theoretical results is not deep enough to motivate the paper. For example I have no idea the what the proposition 1 and 2 is doing. Yes we have a policy gradient on this $g$ function and yes now the $J^S$ is  rate stable. So what do these mean? Why proving these two propositions can prove the proposed method is better in any aspect?

The presentation of the experiment is confusing. I almost consider the SUMO experiment is omitted in the paper as the Figure 2 mixes result from 3 environments without the introduction and the discussion of the evaluation metrics. We should have a section (either in Appendix or main body and has clear annotation) to discuss the very detailed meaning of the evaluation metrics and when presenting in the experiment section we need to emphasize what metric in what environment is improved/affected by what.

### Questions
It will be good to unify the color scheme in Fig 2, like setting "O"'s color to green across all figures.

Eq 3 looks similar to the Lagrangian reward shaping method in Safe RL. Can we setting the cost as the primary objective and set the stability cost to be the constraint? We can write Eq 3 to be like $c(s_t)  - \lambda g(s_{t-1}, s_t)$ and use traditional RL to solve. So that we can avoid the adhoc design of the annealing scheme.

Why we use the world cost instead of reward. I guess it's because in this infinite horizon setting our goal is to reduce the "average-cost" like the Eq 1.

We are now using $g = c(s_{t-1}) - c(s_t)$. Would that possible to apply the idea of multi-step TD like $g = c(s_{t-10} - c(s_t))$ or even with the idea of $TD(\lambda)$? What would happen in this case.

The annealing is too adhoc. Can we have a mechanism to dynamically adjust the preference over cost & stability cost by observing the state uncertainty, either via a distributional Q values or via some kinds of representation learning approaches?

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
2 fair
