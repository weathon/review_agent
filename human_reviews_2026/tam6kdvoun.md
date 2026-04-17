# Toward Agents That Reason About Their Computation

- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
While reinforcement learning agents can achieve superhuman performance in many complex tasks, they typically do not become more computationally efficient as they improve. 
In contrast, humans gradually require less cognitive effort as they become more proficient at a task.
If agents could reason about their compute as they learn, could they similarly reduce their computation footprint? 
If they could, we could have more energy efficient agents or free up compute cycles for other processes like planning.
In this paper, we experiment with showing agents the cost of their computation and giving them the ability to control when they use compute.
We conduct our experiments on the Arcade Learning Environment, and our results demonstrate that with the same training compute budget, agents that reason about their compute perform better on 75\% of games.
Furthermore, these agents use 3 times less compute on average.
We analyze individual games and show where agents gain these efficiencies.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present a study on what happens when RL agents can decide how much compute to use, as measured by how many decisions they make, and as implemented by options of various durations. Their baseline comparison is to DQN, and they use the widely-used ALE set of environments. Results are positive, demonstrating that agents who can manage their own compute outperform agents who don’t.

1. **What is the specific question and/or problem tackled by the paper?**
    
    Put simply, “What happens when you let RL agents decide how to use their own compute?”
    
2. **Is the approach well motivated, including being well-placed in the literature?**
    
    Yes, very relevant work, cites relevant literature on RL including the options framework, DQN work, and related work on sample efficiency, etc.
    
3. **Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous.**
    
    Claims are straightforward, that compute-aware RL agents may perform better under the same compute budgets as baseline agents. Results are straightforward to support the claim that compute-aware agents perform better in 75% of ALE environments, save for 3.
    
4. **What is the significance of the work? Does it contribute new knowledge and sufficient value to the community?**
    
    Yes, a straightforward hypothesis, simple and interpretable math, fits neatly into previous work and cites it well, is relevant to modern questions about compute efficiency, and tested on a huge number of environments with decent visualizations and analysis. This paper could use a journal version.

### Strengths
Very clear framing, extremely easy to understand, well formulated, positive results, cites relevant RL literature.

### Weaknesses
Could use more analysis as to the 25% negative results.

### Questions
I do wish there was slightly more commentary on the exploration/exploitation divide. It seems that temporally-extended options can make exploration more difficult, so some information about how much coverage of the ground-truth MDP agents make over time would be interesting.

A comparison to the baseline DQN with a frameskip of 10 (6Hz?) instead of 5 would be interesting.

Some analysis of the worse performing environments would be welcomed. Any insights there? Why do compute-aware agents struggle so much in these environments? Perhaps this ties into the exploration issue mentioned above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a method for augmenting DQN with action repeat options, showing that this reduces the number of actions executed per frame and improves human-normalized score (HNS) compared to the DQN baseline policy on about three-quarters of the Atari games.

### Strengths
- Clean formalization and direct application to any value based rl method
- Experimental results on all atari environments, showing less average decisions and higher HNS compared to existing baselines.
- Analysis of decision rate change intra episode is interesting and novel.

### Weaknesses
- Compute is only accrued at decision steps, which does not cover simulator overhead/option execution. Since the proposed agent might see more frames/episodes than the baseline agent during decision-making, this may confound the reported gains.
- The framework seems to heavily rely on the existence of temporally extended options. Given the prevalent use of action repeat / frame skip in the atari literature, this narrowly makes sense. In an arbitrary RL environment however its not obvious to me that these assumptions hold. 
- Figure 8 shows HNS monotonically decrease wrt compute cost. This is surprising considering the overall claim of the paper that a nonzero c improves performance relative to DQN (c=0). 
- No comparison with wider baselines like Rainbow, QR-DQN, IQN, or PPO or ablations on various durations of cleanup-target episodes is provided.
- The field of efficient reasoning, particularly in the Reasoning LLM context, is under discussed.

### Questions
1. It is not clear how the ratio of 40M decisions to 200M DQN frames is determined. Please report the average number of environmental frames sampled for each environment. 
2. What differentiates the performance of the proposed method with c=0 from DQN?
3. How does compute savings hold when the cost of option execution / simulator cost is taken into account? An ablation here would improve the paper.
4. Run baselines on Rainbow/QR-DQN (with the fixed frame-skip ablation).
5. How do the author's position this paper in comparison to efficient llm reasoning line of work. Two papers which come to mind are:

Achille, Alessandro, and Stefano Soatto. "AI Agents as Universal Task Solvers." arXiv preprint arXiv:2510.12066 (2025).
Aggarwal, Pranjal, and Sean Welleck. "L1: Controlling how long a reasoning model thinks with reinforcement learning." arXiv preprint arXiv:2503.04697 (2025).

The first explicitly discusses the notion of the cost of time, which is analogous to this works notion of the cost of computation. The second can be seen as parameterizing cost as constraint rather then some lambda*t.  A discussion of this context, given the authors claim their method is, "a stepping stone toward agents that can reason about their own computational processes just as they reason about how they can bring about goals in the world," would strengthen the paper.

A discussion of this context along with clarifying my other concerns would move my score to an accept. I do not see any fundamental concerns with this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Compute DQN, an extension of Deep Q-Networks that enables agents to control and reason about their own computational effort. By integrating a compute cost into the reward function and allowing the agent to choose action durations, the model learns not only the policy but also the decision frequency. Evaluated on 46 Atari games, Compute DQN outperforms standard DQN in 75% of tasks while using much less computation on average. The agent adapts its decision rate dynamically, conserving compute in low-stakes moments and increasing it during critical gameplay phases. These findings demonstrate that agents can autonomously learn compute-efficient strategies without sacrificing performance, highlighting a path toward energy-efficient and adaptive reinforcement learners that optimize not only task outcomes but also their internal resource use.

### Strengths
1. The setup is clearly stated, and the overall flow is easy to follow. I enjoyed reading this paper a lot, and oftentimes I find answers to my questions/confusions lying just a few sentences away.

2. The idea of modeling deliberate control of computing budget with options under different frequencies is quite novel, and it turns out to be very effective. 

3. The experiment evaluation is thorough and convincing.

### Weaknesses
If I have to say sth here, the only thing I would say is if the authors can show similar results on some larger tasks like VLA, LLM fine-tuning, it would make the work perfect.

### Questions
1. What do you think might be the reason that compute-DQN performs badly in some environments like Gopher and DemonAttack? Is it because reward/computing cost scale? Did you clip both of them by [-1, 1]? Have you tried a finer-grained option set to see if those game performances improve?

2. Under the computing budget + limited set of frequencies, could the authors envision how to analyze the suboptimality of the learned policy theoretically?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method to expand the action space of an agent to decide to commit to taking the same action across a number of timesteps, so that no decisions need to be made for those timesteps. It also augments the reward to subtract a value for every time a decision is made. This then encourages the agent to make fewer decisions, thus reducing computation time. Results show that better performance can be obtained on the arcade learning environment, on average, while also reducing the number of decisions.

### Strengths
The paper’s methods are concise, and the results make sense. I believe the results can be easily replicated.

### Weaknesses
The paper positions itself about investigating algorithms that reason about their own computation. However, the method presented involves a straightforward modification to the reward given by the MDP. Therefore, this paper is related to reward shaping. However, relevant literature from this field is not mentioned and, therefore, the paper does not position itself within the relevant literature.

### Questions
What is meant by “reasoning” and can reward shaping be equated to enabling an algorithm to meta-reason, i.e. reason about its own computation?

### Soundness
1

### Presentation
1

### Contribution
2
