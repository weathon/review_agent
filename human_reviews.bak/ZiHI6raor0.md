# CAMMARL: Conformal Action Modeling in Multi Agent Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 6, 5

## Abstract
Before taking actions in an environment with more than one intelligent agent, an autonomous agent
may benefit from reasoning about the other agents and utilizing a notion of a guarantee or confidence about the behavior of the system. In this article, we propose a novel multi-agent reinforcement learning (MARL) algorithm CAMMARL, which involves modeling the actions of other agents in different situations in the form of confident sets, i.e., sets containing their true actions with a high probability. We then use these estimates to inform an agent’s decision-making. For estimating such sets, we use the concept of conformal predictions, by means of which, we not only obtain an estimate of the most probable outcome but get to quantify the operable uncertainty as well. For instance, we can predict a set that provably covers the true predictions with high probabilities (e.g., 95%). Through several experiments in two fully cooperative multi-agent tasks, we show that CAMMARL elevates the capabilities of an autonomous agent in MARL by modeling conformal prediction sets over the behavior of other agents in the environment and utilizing such estimates to enhance its policy learning.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a multi-agent reinforcement learning (MARL) algorithm CAMMARL, where each agent's policy is not only conditional on its own observation, but also on the estimate of other agents actions. Specifically, it uses conformal predictions to learn confident sets containing other agents' probable action estimates. CAMMARL consider two agents setting and is tested on two MARL tasks, cooperative navigation and level-based foraging.

### Strengths
- The paper is well-motivated. Modeling other agent's actions is intuitively helpful for learning a good policy.
- There are some relevant  statistical analysis over how CAMMARL works, such as the averaged set sizes outputted by the conformal predictor. 
- A range of related baselines are considered.

### Weaknesses
- The predicted conformal sets is exponentially growing with respect to $|\mathcal{A_{other}}|$, which may not be scalable when $|\mathcal{A_{other}}|$ is large.
- The paper is restricted to the two agents setting. It is not clear if the proposed method can generalize to more complex tasks with potentially many agents.
- The experiment results are preliminary in only two tasks. Further, the improvement over the baselines is not obvious, especially in CN in Figure 4.

### Questions
- Can CAMMARL work in settings with more than two agents? If so, it is beneficial to show its effectiveness in some more complex MARL benchmarks with more agents.
- Does $\mathcal{N_{other}}$'s action also condition on $\mathcal{N_{self}}$'s action?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes CAMMARL, a novel multi-agent reinforcement learning algorithm that uses conformal predictions to model the actions of other agents in the environment as sets that contain their true actions with high probability. The paper claims that these sets can inform the decision-making of an agent and improve its performance in cooperative tasks. The paper demonstrates the effectiveness of CAMMARL in two multi-agent domains and compares it with several baselines that use different types of information about other agents.

### Strengths
The paper addresses an important and challenging problem of reasoning about other agents in partially observable environments. The paper introduces a novel way of using conformal predictions to model the uncertainty and confidence of other agents' actions. The paper presents extensive experiments on two challenging cooperative tasks and demonstrates that CAMMARL improves over various baselines in terms of returns, learning speed, set sizes, and coverage. The paper also discusses some limitations and future directions for extending CAMMARL to more complex scenarios. The paper is well-written, clear, and well-structured. The code is provided for reproduction.

### Weaknesses
The paper has some limitations and possible areas for improvement.

- The paper could provide more thorough discussions on experiment details. For example, the paper does not explain how the regularization term in the conformal prediction model is chosen or tuned, or how it affects the performance of CAMMARL. A more transparent and rigorous analysis of this aspect would enhance the credibility and generalizability of CAMMARL. Also, what are the differences of the settings between different baseline algorithms compared in the paper? Since they have different dimensions of input, maybe they also need slightly different training settings to achieve their own best performance.
- The paper uses learning curves as the only evaluation metrics but does not provide any qualitative analysis or visualization of the learned policies or behaviors of CAMMARL agents, which would help to illustrate how they leverage conformal predictions to cooperate effectively.
- The paper only considers fully cooperative settings and does not explore how CAMMARL would perform in competitive or mixed scenarios where other agents' intentions may not be aligned or predictable. Especially, in the competitive robust RL setting, the conformal set could potentially be used for worst-case analysis.
- The paper does not discuss any potential drawbacks or challenges of using conformal predictions, such as computational costs, calibration issues, or sensitivity to hyperparameters.

### Questions
- Intuitively, the idea of using conformal predictions to model the actions of other agents is to augment the agent’s state with the historical memory of other agents' behavior, which is similar to fictitious play in game theory. However, fictitious play only converges in some specific game settings, and MARL is known to be hard to converge in general settings. Moreover, the conformal predictions are based on previous observations but the other agent’s policy is also evolving. Is there any observation where the algorithm does not converge well or even has cycling behavior?
- In the paragraph of Global-Information-Agent-Modeling (GIAM), the paper states that this can be infeasible in real-world scenarios. Why is that? Seems like the information used in this algorithm is the same as CAMMARL. The difference is that, the historical trajectories are used to first train a prediction model and then feed into the agent’s policy network, whereas in GIAM, it is directly used in the policy network in an end-to-end way.
- How does CAMMARL handle different set sizes produced by the conformal model? How does it encode them into input features for policy learning? How does it affect the stability and convergence of policy learning?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a multi-agent reinforcement learning algorithm that uses conformal prediction to explicitly reason about the behavior of other agents. To evaluate their proposed algorithm two experiments using cooperative simulated environments are conducted. The experiments show that CAMMARL outperforms alternative methods though is still short of a model with access to global information.

### Strengths
* Applying conformal prediction to reason about agents in a MARL problem is a simple but straight-forward idea that doesn't suffer from the drawbacks of alternatives methods which require considerable more compute and data (e.g., inverse reinforcement learning).
* The conformal approach not only supports reasoning about others but also provides confidence measures around its reasoning

### Weaknesses
* Beyond the idea of using conformal modelling for action prediction there does not appear to be large algorithmic advancements in solving the conformal problem in this setting
* The simulated environments used in the experiments are quite simple 
* Algorithm 1 is difficult to follow. For example, there are many lines with multiple assignments and updates.

### Questions
* What is $b_{conformal}$ in Algorithm 1?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose a multi-agent reinforcement learning (MARL) algorithm CAMMARL, which models the actions of other agents in different situations. The the estimates are used to inform an agent's decison-making. The experimental results illustrate that the proposed method elevates the capabilities of an autonomous agent in MARL.

### Strengths
+ The motivation of this paper is reasonable.
+ This paper is well-organized and well-written.
+ The main idea of the proposed method is described in great details.

### Weaknesses
+ It seems that the proposed method does not improve the performance greatly compared with other methods.
+ The scenarios selected in this paper are not so convincing to some extend. More experiments should be conducted to demonstrate the superiority of the proposed method against other methods.
+ The assumption of this paper is too ideal. Commonly, it is hard to obtain observations from other agents in MARL especially in the context of decentralization.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
