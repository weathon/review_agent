# Bridging Sub-Tasks to Long-Horizon Task in Hierarchical Goal-Based Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
Hierarchical goal-based reinforcement learning (HGRL) is a promising approach
to learn a long-horizon task by decomposing it into a series of subtasks of achiev-
ing subgoals in a shorter horizon. However, the performance of HGRL crucially
depends on the design of intrinsic rewards for these subtasks: as frequently ob-
served in practice, short-sighted reward designs often lead the agent into undesir-
able states where the final goal is no longer achievable. One potential remedy to
the issue is to provide the agent with a means to evaluate the achievability of the fi-
nal goal upon the completion of the subtask; yet, evaluating this achievability over
a long planning horizon is a challenging task by itself. In this work, we propose
a subtask reward scheme aimed at bridging the gap between the long-horizon pri-
mary goal and short-horizon subtasks by incorporating a look-ahead information
towards the next subgoals. We provide an extensive empirical analysis in MuJoCo
environments, demonstrating the importance of looking ahead to the subsequent
sub-goals and the improvement of the proposed framework applied to the existing
HGRL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel intrinsic reward function design method in hierarchical goal-based reinforcement learning. In order to alleviate the short-sighted reward designs that often lead the agent into undesirable states where the final goal is no longer achievable, the forward-looking reward (FLR) proposed in this paper can enable the agent to evaluate the achievability of the next subgoal upon the completion of the subtask. FLR can be applied to existing hierarchical goal-based reinforcement learning methods, and the results of the corresponding variants obtained in the Mujoco environment show that FLR can improve the performance of the vanilla algorithm.

### Strengths
1. The paper is easy to follow. The description of motivation is clear and makes people feel reasonable and meaningful. It is indeed a problem worthy of attention that the intrinsic reward function designed in hierarchical goal-based reinforcement learning is too short-sighted.
2. The chosen experimental environment is reasonable. If the reward function designed for the agent in the AntMaze3Leg scenario is too short-sighted, it will indeed lead to task failure.
3. The analysis of the experimental results is sufficient. In the paper, the author gives the number of episodes where the agent falls and the look-ahead reward difference between the normal and fall states when the proposed FLR is applied, which clarify the role of FLR in reinforcement learning to a certain extent.

### Weaknesses
1. The difference between the performance of the algorithm using FLR and the vanilla algorithm is not significant. Especially in AntMaze4Leg, the performance of FLR+DHRL is almost the same as that of the original DHRL.
2. The experiment was only conducted in two scenarios. I think this is a more empirical paper and should be experimented in more different domains. The current experimental results are not convincing enough.
3. I think FLR is more about improving the original algorithm capabilities, rather than really giving it the ability to solve short-sighted problems. For example, in AntMaze3Leg, as training progresses, PIG cannot effectively suppress the number of episodes where the agent falls, and neither can FLR+PIG.
4. The code is not available, which is not conducive to the reproduction of experimental results.

### Questions
1. Why do the curves in Figure 4(b) show a downward trend in the later period?
2. How does FLR+PIG perform in AntMaze4Leg?
3. I think the comparison between FLR+DHRL with oracle-based DHRL in Appendix Figure 6 cannot be directly used to illustrate that FLR encourages the avoidance of undesirable transitions. Is the reason the author gives this conclusion because their final winning rates are similar? How about the learning curve for the original DHRL?

### Soundness
2 fair

### Presentation
2 fair

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
This paper investigates the issue of inefficient exploration in hierarchical goal-conditioned reinforcement learning (HGRL). The main idea proposed in this paper is to use a forward-looking reward (FLR) that guides the low-level policy to consider the achievability of the final goal with respect to the current subgoal or state. The FLR is based on a proxy model that estimates the value function of the next subgoal slightly distant from the current one. The paper claims that FLR can improve the performance and sample efficiency of HGRL, especially when the agent starts from unskilled states. The paper also claims that FLR can be easily applied to existing HGRL methods as an extension. Moreover, the empirical analysis on MuJoCo environments shows that FLR outperforms the selected baselines and demonstrates robustness and stability in complex tasks.

### Strengths
- It tries to address an important and challenging problem of reward design in HGRL, which has not been well studied in the literature. 
- It proposes a simple yet effective solution incorporating a forward-looking perspective and a proxy model for value estimation.

### Weaknesses
- The authors claim that their proposed method can work as a plug-in for existing HGRL methods. However, the experiments are conducted only on top of DHRL. Since RL algorithms often perform high-performance variance in different environments/tasks, I think it is necessary to make comparisons with more baselines. And also, I think it would be better if the authors make comparisons on different environments, not only the Maze.
- Key designs lack of explanation, e.g., how to determine the $sg$ in equation 5? Parameter searching via enumeration, or gradient-based?
- Missing introduction of environment details
- The investigation on the effectiveness of $\alpha$ is not thoroughly, as the authors only test a range of $\alpha$ when $\gamma = 0.99$. More trails should be included.
- The writing is not good enough. For instance, the supplementary should be improved, e.g., there should include some natural language description for the given algorithm tables, and it seems the authors didn't conduct repeated experiments in Figure 6; The paper mentioned FPS algorithm, but what is FPS algorithm? No reference and no introduction; Figure 3 shows the results of "initial random rollouts" settings, but there is no description for this setting.

And I suggest the authors improving the writing further, especially the details of the involved environments. As you should not suppose your readers are all familiar with the related work of goal-conditioned RL.

### Questions
1. What is FLW in the begining of section 2.3?
2. What is the upperbound of $\epsilon$?
3. Section 2.3, the authors include a reference to HER in wrong format. It should be a parenthetical citation.
4. Section 2.2, the authors claim "we may need to redeﬁne the low-level policy as $\pi_l(s, sg_i, g)$ rather than $\pi_l(s, sg_i)$ agnostic to $g$. This requires anadditional complication to train the low-level policy". How to do that? I cannot relate the corresponding context in this paper. Could you further explain it?
5. What is the mathematical format of $g$ in your work?

### Soundness
2 fair

### Presentation
1 poor

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
In this paper, the authors focus on the problem that short-sighted reward designs often lead the agent into undesirable states in HGRL. That is, with the completion of a simple given subtask, the agent may be misguided, leading to the ultimate goal becoming unattainable. The authors propose a subtask reward scheme to bridge the gap between the long-horizon primary goal and short-horizon subtasks. The motivation of this paper is reasonable, and there are relatively detailed method analysis.

### Strengths
+ The motivation of this paper is reasonable, and the introduction is well-organized.
+ The innovation of this paper is relatively strong.
+ The effectiveness of the proposed method is verified by experiments.
+ The pictures presented in this paper are clear and standardized.

### Weaknesses
+ Only in the simplest scenarios, the proposed method has obvious performance improvement. More experiments should be conducted in various scenarios(such as FetchPush, FetchPickAndPlace, AntMazeBottleneck, and UR3Obstacle) to highlight the superiority of the proposed method.
+ As can be seen from the ablation study, the experiment is sensitive to the hyperparameter $\varepsilon$, even in a simple task, not a robust hyperparameter is provided.
+ The proposed solution is only suitable for controlling the local reaching task of the robot, because when achieved goal is subsituted with the location of the object being operated, it has the same hindsight goal for all the goal relabeling using HER, and FLR will not yield results as expected.
+ It seems that the subscripts of $sg_{i}$ and $k_{i}$ are not standardized.
+ At the beginning, the value of $V_{\pi}$ is definitely biased, and these biased values will continue to exist in the experience buffer when updating, which will continue to degrade theperformance, and may be a reason for the unsatisfactory performance on complex tasks.
+ The proposed solution is only suitable for controlling the local reaching task of the robot, because when the operated object is not changed in position，it has the same hindsight goal for all the goal relabeling using HER, and FLR will not yield results as expected.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
