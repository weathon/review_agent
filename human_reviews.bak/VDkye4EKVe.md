# Discovering Minimal Reinforcement Learning Environments

- Decision: Reject
- Scores: 3, 3, 3, 3

## Abstract
Human agents often acquire skills under conditions that are significantly different from the context in which the skill is needed. For example, students prepare for an exam not by taking it, but by studying books or supplementary material. Can artificial agents benefit from training outside of their evaluation environment as well? In this project, we develop a novel meta-optimization framework to discover neural network-based synthetic environments.  We find that training contextual bandits suffices to train Reinforcement Learning agents that generalize well to their evaluation environment, eliminating the need to meta-learn a transition function. We show that the synthetic contextual bandits train Reinforcement Learning agents in a fraction of time steps and wall clock time, and generalize across hyperparameter settings and algorithms. Using our method in combination with a curriculum on the performance evaluation horizon, we are able to achieve competitive results on a number of challenging continuous control problems. Our approach opens a multitude of new research directions: Contextual bandits are easy to interpret, yielding insights into the tasks that are encoded by the evaluation environment. Additionally, we demonstrate that synthetic environments can be used in downstream meta-learning setups, derive a new policy from the differentiable reward function, and show that the synthetic environments generalize to entirely different optimization settings.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The work presents an extension to the synthetic environment learning framework. While keeping the meta-training loop the presented work proposes to learn initial state distributions instead of full transition dynamics when training synthetic environments. This approach is further extended with a curriculum learning approach to steadily increase evaluation rollout lengths in the target environment to stabilize the meta-training approach

### Strengths
The work presents an interesting extension to learning synthetic environments.
The idea to only learn an initial state distribution and fixing transition dynamics to always reach goal states is an interesting approach to environment design.
Further, the idea of using synthetic reward functions to extract an optimal policy that is learnable in the environment is very compelling as it provides a ground truth that is often not available. Thus, such learned environments could be used to provide more insights into the learning dynamics of different RL agents.

### Weaknesses
The work is often not clear enough in which parts are coming from the framework proposed by Ferreira et al. Thus, it took me some time to understand that basically the only change to the framework is the idea of learning an initial state distribution (with fixed dynamics) rather than learning the dynamics. This lack of clarity is particularly highlighted in the experiments where “the plain T setup” is declared as similar to Ferreira et al. but it is not elaborated on what the exact differences are. Further, in the experiments there is no clear comparison to the prior work and one is left to guess on the actual performance differences.

Figure 3 most often not show the learning curve for the proposed training setup. Only for hopper and Mountain car does it show the actual learning curve of the proposed IC training setup but for all other ablations only a dashed line of the final performance is shown. This omission of the learning curves seems rather strange and requires an explanation. Similarly, in this figure often the 95% confidence intervals seem wholly missing for some or even all methods. Lastly, in this ablation it seems rather strange to not include “no curriculum” as an option when showing the impact of different curricula. This is an important baseline to show. I understand that the difference in using curricula and no curricula is shown in the first row, but it is not easy to visually compare the ablated element in this way.

The writing could be improved as often statements are made which are not easy to follow. For example, already early in Section 3 it is stated that learning the transition dynamics is not as useful as only learning an initial state distribution but no reasons are given as to why. At least a reference to the later experiments would have shown that this statement is not just coming from thin air. Similarly, limiting an episode length to 1 seems to fall from thin air without any comparison if maybe a slight longer episode of 2 or 3 steps might perform better or not. It is simply stated that this does not lead to reduced performance except on pendulum, for which no explanation is given nor a hypothesis stated.
Another example is the statement that the SEs can train agents within 10k steps whereas in real environments this is often requires many more steps. Unfortunately, no learning curves are presented to visualize this difference. At least a reference to Fig 1 seems necessary to support this statement.

Overall the work seems interesting but not yet fully ready for publication.

### Questions
At the beginning of section 5 it is stated that “…, the episodes in the synthetic environment can be limited to one step without a quali- tative loss of performance. In this case, the reward received is equal to the return, the state-, and the state-action value function.“ Am I correct in understanding that learning an SE then is simply supervised learning of a reward function for particular policies? I don’t see why a learned initial state distribution is needed at all then. Would it not suffice to simply fix the initial distribution?

How did you set the hyperparameters of your method?

How much tuning is necessary to achieve good results?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduce a meta-optimization framework for synthetic environment discovery, where the parameters of the synthetic environment are meta optimized using SNES. Through extensive empirical study, the authors show that meta-training over a large range of inner loop tasks leads to synthetic environments that generalize across different RL algorithm and broad range of hyper-parameters. In addition, the authors shows that training contextual bandits achieves good enough result to generalize to the target environment. Compared to training as RL agents, contextual bandits are easy to interpret.

### Strengths
This paper present extensive experiment results to demonstrate and explain the proposed method. Firstly, the authors shows the performance of meta-trained synthetic environments on multiple openAI gym environments.  Further, the authors show that training contextual bandits is sufficient to train RL agents which significantly reduce the number of parameters needed through empirical study. The idea of simplify an RL problem to a CB by eliminating the transition probability makes the entire framework more feasible.  It does have a potential to be applied to more large scale real problems.

### Weaknesses
I found this paper very hard to follow. Some necessary background knowledge is not introduced in the paper. 
Although this paper gives empirical proof for the statements, the intuition behind the ideas are now explained. Without any explanation, I am not convinced with the claimed results. For example, the authors claimed that the state transition probability is not necessary to train well-performing agents. I read the experimental result, but I cannot understand the intuition behind it.

### Questions
Similar to the points I made in weaknesses session, I would read more on the intuition behind all your experiments.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a meta-learning approach to learning a synthetic model of the RL environment as a proxy for finding the optimal policy, instead of directly interacting with the real environment. The authors formulate the problem as a meta-learning problem and solve it using evolution strategy (ES). To overcome the difficulty of learning the dynamics of the actual environment, the authors propose that learning a synthetic *contextual bandit* (SCB) model is sufficient to achieve the goal and train the agent policy. They conduct experiments and provide ablation studies to analyze the choice and variants of the proposed method.

### Strengths
1. Using a simpler model (synthetic contextual bandit, SCB) as a proxy is an interesting idea when environment is complex.
2. Experiments on interpretability justify the choice of SCB as the proxy.

### Weaknesses
One of the main contributions of this paper is the use of a synthetic contextual bandit (SCB) as a proxy to the real environment. However, it is important to note that:

* A contextual bandit (CB) can be converted to a Markov decision process (MDP), but not vice versa, because CB is stateless. This means that the SCB model may not be able to accurately capture the dynamics of more complex environments, such as Go, where state is essential for planning and decision-making.
* It is also unclear how synthetic contextual bandit can work for partial observation environments, where the agent does not have access to all of the state information.

In other words, the SCB model may be able to learn to play simple games, such as Atari Breakout, where the state space is relatively small and statelessness is not a major issue. However, for more complex games, such as Go, where the state space is very large and statelessness is a major issue, the SCB model is unlikely to be able to learn to play at a high level. Additionally, it is unclear how the SCB model would perform in partial observation environments.

In addition to the limitations of the SCB model discussed above, there are several other issues with this paper:

* One important paper is not cited nor discussed, which is closely related:

Ferreira et al. (2022) proposed a similar approach of learning synthetic environments and reward networks for reinforcement learning. It would be helpful to discuss the relationship between this work and the proposed method.

* There is no comparison with the state-of-the-art results on the environments/tasks in this paper. It is understandable if the proposed method does not outperform the state of the art, as the inner policy optimization can be different. However, it would be interesting to see how the proposed method compares to other methods on the same tasks.

* The motivation for using a synthetic environment is not entirely clear. In the first paragraph of the introduction, the authors mention the biological fact that organisms can learn from artificial stimuli. However, it is not clear how this relates to the use of synthetic environments in reinforcement learning. Additionally, the authors do not explain why learning or meta-learning RL policies is not sufficient.

* The colors in Figure 5 should be revised so that synthetic curves have the same color, and the same for the real curves. This would make the figure easier to read and interpret.

Overall, the use of a SCB as a proxy to the real environment is a promising approach, but it is important to be aware of its limitations. For more complex environments, such as Go, and partial observation environments, models like MCTS are still necessary for learning to play at a high level. The authors need to address the issues raised above in order to strengthen their work.

Reference

Ferreira, F., Nierhoff, T., Sälinger, A., & Hutter, F. *Learning Synthetic Environments and Reward Networks for Reinforcement Learning*. In ICLR 2022.

### Questions
1. Why do the authors choose to evaluate the proposed method on Brax environments, instead of Gym or other popular environments?

2. The authors claim that limiting the episode length to one step in the synthetic environment does not qualitatively affect the performance of the agent (Section 4, last subsection; Section 5, beginning). However, I could not find any figure or table in the paper that shows this result. Can the authors please provide this information?

3. The authors claim that "even state-of-the-art RL algorithms such as PPO struggle with solving MountainCar" (Section 1, second paragraph). However, I am not sure if this is true. Can the authors please provide citations to support this claim?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to parameterize and optimize RL environments, thus "discovering" synthetic environments.
It is also remarked that this problem can be made much simpler by not learning a transition function for the synthetic environment, limiting the synthetic environment to a contextual bandit while preserving performance benefits.
Empirical results show that the synthetic environment can be used to train a variety of learning algortihms by meta-learning the synthetic environment over a distribution of inner-loop algorithms (different algorithms, as well as different hyperparameters).
The synthetic environments "are interpretable" and have other applications in down-stream tasks.

# Decision
While I like the overall direction of this research, I think there are several problems with the specifics of its execution and must recommend rejection.
The major flaw in this paper is that it seems to solve for the value function in a much more complicated way using meta-learning, without clearly demonstrating the benefits of doing so.
Some claims are not substantiated in the paper at all, and several other contributions are not clearly conveyed.
The paper would require a more careful treatment of the relevant baselines, as well as careful motivation for the proposed method, along with more evidence to support the claims.

### Strengths
- The research direction considered, discovering environments in which algorithms can learn quickly, seems novel and has potential to be quite interesting for deployment of RL algorithms. I especially like that use on further downstream tasks, beyond the prescribed inner-loop, are considered. This can have several benefits not only for training RL algorithms but in developing new algorithms with discovered environments for quicker iteration.

### Weaknesses
- There seems to a connection between the proposed method and simpler non-meta RL algorithms that just learn the value function, which is only mentioned in passing. The optimal reward of the contextual bandit seems to be the optimal value function at the context (state in the MDP). What is the benefit of meta-learning this, rather than just using monte-carlo returns? Or, what is the benefit of this over just learning the value function itself?
- Important claims made at the beginning of the paper, such as improved wall-clock time, are not substantiated.
- Overall, the empirical results fail to convince me of the importance of the contribution. The results in Figure 2 show that training in the synthetic environment is feasible to learn a policy (which is interesting, but also not entirely surprising because of the first point). The results, however, do not demonstrate that this provides a benefit over training in the real environment. The bottom row of figure 2 shows that baseline algorithms are underperformant on some hyperparameter distribution. But the performance on a particular hyperparameter distribution is also not entirely relevant. The benefit of an RL algorithm that performs well on a distribution of hyperparameters is robustness, and results in a less computationally expensive hyperparameter sweep. But the cost of this sweep can be much lower than the cost of the hyperparmaeter sweep and subsequent meta-learning of the synthetic environment itself. Your setup involves many hyperparameter choices in both the inner and outer loop. These design decisions are not abalated and do not provide enough confidence that the empirical results are reliable.

### Questions
- Section 2 (Curricula): you state that you learn a curriculum in this section, but in contribution 1 it seems that you in-fact design a curriculum. These two claims seem contradictory.
- Section 2 (Synthetic Data): I understand that there is some prior work that attempts to train reinforcement learning algorithms from synthetic data, but I do not think this is well-motivated. Why not just use the real data? You need it for evaluation anyway to learn the synthetic enviornment, so you may as well use it during meta-learning as well.
- In your experiments, is the meta-learned synthetic environment tuned to a specific real environment?
- Results, figure 3: why is there a discrepancy between the evaluation of the ablation and the results in Figure 2. Specifically for hopper? I understand ablations can be expensive, and may necessitate a smaller set of results, but this ablation does not explain the performance differences with respect to the original results.
- I do not see how synthetic environments provide any interpretability beyond what a learned value function can provide. It would be interesting to see whether the reward function meta-learned for synthetic environments differs in some qualitative way from that of a learned value function. But otherwise, this analysis does provide me with any additional insight.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
