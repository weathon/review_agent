# Reinforcement Learning with Elastic Time Steps

- Avg Score: 3.75
- Decision: Reject
- Scores: 3, 3, 6, 3

## Abstract
Reinforcement Learning (RL) is usually modelled as a Markov Decision Process (MDP), where an agent goes through time in discrete time steps. When applied outside of simulation, virtually all existing RL-based control systems maintain the MDP assumptions and use a constant rate control strategy, with a time step that is empirically chosen according to the specific application environment. Controlling dynamic systems with learned policies at the highest, worst-case frequency to guarantee stability can require high computational and energy resources, which can be hard to achieve with on-board hardware. Following the principles of reactive programming, we posit that applying control actions $only$ $when$ $necessary$, can allow the use of simpler hardware and reduce energy consumption. To implement this reactive policy, we break the fixed frequency assumption and propose $RL$ $with$ $elastic$ $time$ $steps$, where the policy determines the next action as well as the duration of the next time step. We also derive a Soft Elastic Actor-Critic (SEAC) algorithm to compute the optimal policy in our new setting. We demonstrate the effectiveness of SEAC both theoretically and experimentally driving an agent in a simulation of a simple world with Newtonian kinematics. Our experiments show higher average returns, shorter task completion times, and reduced energy consumption.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The main contribution of this paper is the idea that, in RL, a policy can be made to specify both a control action to apply *and* the length of time an actuator should apply that action. The paper integrates this idea within an existing, popular algorithm for model-free RL (the SAC algorithm), and presents comparative results in a small example problem.

### Strengths
As far as I know this precise idea is novel, and it is certainly intuitive. Results and other details aside, I think the community should investigate this direction more deeply and this paper provides a nice starting point for that effort.

### Weaknesses
Weaknesses:
- The literature review is quite thin, and as a circumspect reader I do wonder how novel this idea really is, given how little literature is referenced. For example, a quick google scholar search reveals the following papers that seem to be very closely related: [1, 2]. I would also add that variable rate decision making is widely studied in the control theory literature. A key phrase to find this literature is “adaptive time step.”
- The paragraph immediately above section 3.1 indicates that there are a lot of loose ends that are not being discussed in detail, and which may strongly affect results. The imprecision of this discussion (e.g., what is a “partial MPC” and what role does the PID serve if you already use MPC?) suggests that the work may be somewhat immature.
- The reward structure discussed section 3.1 is *not* what one would properly call a “multi-objective optimization problem.” A distinguishing trait of such problems is the concept of “Pareto optimality” which encodes all of the tradeoffs among optimal performance with respect to each separate objective. By assuming a fixed weighting, this paper effectively reduces the problem to a standard optimization problem (and picks a single point on the Pareto frontier). I recommend consulting [3] for further details.
- Relatedly, the construction in Definition 1 is not as clear as it could be. For instance: are the R terms intended to be functions of state (and action)? If so, why does it make sense to only accrue reward at the times when actions are changed? Doesn’t that lead to some obvious opportunities for reward hacking? For example, could an agent decide to plow straight through some region of low reward for a bunch of (unactuated) time steps? Also, can R_t and R_\epsilon be evaluated at every time, or only at the end of an episode? Evidently, at every time t, but then I am lost as to why the agent is incentivized to minimize n, the length (in steps) of an elastic time step. I am lost.
- The details of the method are really not very clearly explained. For example, throughout the discussion of section 3 it appears that the there is some notion of an agent physically moving and the policy gets to access a measure of distance somewhere. This is unclear: everything up until this point (and in general) is framed around general MDPs, which have nothing to do with physical embodiment. How general-purpose is the proposed approach?
- Relatedly, the test environment is not very clearly explained, or at the very least, suggests a very basic question: wouldn’t it make more sense for the policy to output a force, rather than a target position? This would remove the need for lower-level tracking control (MPC, PID) and also mitigate the “measure of distance” question above, I believe. 
- I do not follow the “six dimensions of the state in the environment” - in fact, I count 9: 2 each for agent/obstacle/goal position, 2 for agent velocity, and 1 for duration. What am I missing? In the same paragraph, the discussion of semi-Markov processes and recurrence is rather opaque. Use of words like “might” and “could” lead me to wonder how clearly this point is understood. I suggest clarifying the language here.
- There are no discernible error bards in the plots, and the shaded areas appear to be traces of other plotted data - this needs to be explained precisely, and plots should show some measure of error in order to be interpreted statistically.
- More importantly, even: there is little to no interpretation of the behavior of the proposed policies. Results here indicate some differences in aggregate behavior (although the interpretation to that effect should really require some error bars as above), but it would really help to understand what is going on if the authors expanded upon Fig. 7 to illustrate what was going on in the environments in these situations and why it made sense to change the control rate as shown.

Other nitpicks:
- It seems like the main motivation here is one of saving computational resources. Obviously, most control systems are pretty lightweight and so I imagine these savings really come in from the perception side, e.g., if you no longer have to process big images at high frame rate. Experimental results to illustrate these savings more directly than the abstraction of “number of repeated actions” would be highly motivating.
- There are quite a few typos and other small syntax issues.
- The vertical axis labels are wrong in Fig. 5.
- Figures 5 and 6 could be more clear about indicating that the right hand sides are insets of the left. Also, why were the methods run for so long - it seems they all converged quite a bit earlier and then for some reason PPO destabilized. Something seems off here.
- Why does Fig. 7 say “epochs” instead of “configurations?”


[1] Chen, Y., Wu, H., Liang, Y., & Lai, G. (2021, July). VarLenMARL: A framework of variable-length time-step multi-agent reinforcement learning for cooperative charging in sensor networks. In 2021 18th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON) (pp. 1-9). IEEE.

[2] Sharma, Sahil, Aravind S. Lakshminarayanan, and Balaraman Ravindran. "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning." International Conference on Learning Representations. 2016.

[3] Deb, Kalyanmoy, and Kalyanmoy Deb. "Multi-objective optimization." Search methodologies: Introductory tutorials in optimization and decision support techniques. Boston, MA: Springer US, 2013. 403-449.

### Questions
Please see my comments above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents relaxes the fixed frequency assumption of MDP typically studied in RL and proposes RL with elastic time steps.  Also a Soft Elastic Actor-Critic algorithm is derived with theoretical and practical benefits.

### Strengths
1. The work is concisely summarized.
2. The use of elastic time is important in the tasks such as robotics etc.

### Weaknesses
1. There are many existing studies with varying time (e.g. option framework, action repetitions…)
Authors introduce some notions of options and semi-MDP in appendix, but without clear definitions of each notation, which makes it harder to see the clear connections to the main work and the option framework.  (It was not clear how the authors validated Bellman-like equations for elastic time case)  Assuming the algorithm is properly derived from the option framework, it is necessary to compare to the existing work based on the framework.  (Or at least it should show significant practical results compared to the existing work; it seems the experiments are not for sufficiently complex tasks.)
2. Existing environments such as OpenAI Gym can be easily adjusted to include time as information for states; I am not sure what the authors mean by “...additional input and output information that is not available within existing RL environments…”
(Note that simulators anyway need to run with small time interval to maintain accuracy, and action durations can be just a repetition of that.)
4.  Figure 5 is a bit hard to parse: why time in seconds are negative?  I could guess this but it is better to make them crystal clear.
5.  It would be better to show baseline with 100Hz (fixed) case, not 5.0 Hz since the elastic one uses 1 to 100 Hz.
6.  Figure 7 is also hard to interpret; why are there only 2 time steps…?  2 steps are enough to complete tasks…?
7.  Finally, it was not clear why the authors specifically used the reward defined in Definition 1.

### Questions
1.  Figure 4 right seems too sparse; what does it try to imply?
2.  What is the action space A?  Is it the Cartesian product of “action” and “time”?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a reactive reinforcement learning policy, which breaks the fixed time step assumption commonly adopted in RL and determines the next action and the duration of the next time step as input to the controller, thus integrating the temporal aspect into the learning process. The authors test their approach in a simulation of a simple word with Newtonian kinematics, showing its effectiveness in leading to higher efficiency in terms of speed and energy consumption.

### Strengths
The contribution is clearly stated and it is relevant to the development of real-world efficient and effective RL-based control systems. The paper structure is well organized and clear. Figures and schemes are helpful and explanatory. Limitations of the proposed approach (which components would be necessary for a real-world implementation) are clearly stated.

### Weaknesses
The contribution is relevant but it is limited compared to the existing state of the art. Since the contribution is mainly aimed to applying RL control outside of simulation, a proof of concept of the functioning of the proposed algorithm on a real-world application (rather than only in a simulation environment) would be important, in my view. 
Although the paper’s quality of presentation is generally fair, I found the comparison with the related works poor and lacking of an insightful discussion about existing time-sensitive RL tasks, which are only quickly listed at the end of section 2. Expanding such a paragraph could make the relevance and applicability of the paper’s contribution clearer.
The presentation of the results could also be improved (see specific comments on the next section).

### Questions
-	Fig 1: I don’t find Fig 1 completely effective, based on the description within the Introduction. Since one of the contributions of the Elastic Time Step RL is that of enabling the policy to output the time step duration, together with the action, this could be somehow explicitly indicated in the Figure. Also, even though I understand the intention of splitting the “learning” and “execution” part of a RL implementation, I find the brain-like icon confusing when used to indicate the “execution” rather than the “learning” component of the system.
-	I would be curious to know from which specific practical application (robotics, autonomous driving?) comes the authors’ inspiration for the paper.
-	Page 4, sentence preceding Definition 1: “The aggregate reward for task completion is represented by r”. Did you mean “R” (capital letter)? 
-	The paragraph after Definition 1 (“We validate our reward strategy…”) could be rephrased to highlight SEAC differences compared to SAC.
-	What do you mean when you say “…giving a high probability that the agent can discover the optimal solution to complete the task”?  Maybe this sentence can be rephrased to make the exploration strategy clearer.
-	In general, from the sentence starting “we assume the agent…” to the sentence ending with “…Bellman equation”, I find the flow of the text, which can be read while referring to the scheme on Fig 3, a little hard to follow, in the sense that it jumps from one block to another one (of the Fig.3) without a precise order. Incorporating more references to the visual scheme and aligning the text with the functional flow of the figure 3 (rather than simply listing the meaning of the symbols) could help the readability. 
-	You mention that one major contribution of the SEAC is to include the execution time of each action to the output, but this term is not explicitly indicated on Fig.3, together with the At.
-	The meaning of the double arrows in Fig.3 is not very clear to me. Maybe an explanation could be included either on the caption or on the main text.
-	The impact value of the execution is defined, based on the chosen environment, as the target movement distance. Do you have in mind some examples of different implementations for different problems?
-	In the end of paragraph 3.1, when you say “the controller will compute a range of control-related parameters”, is this represented by Mt?
-	In the end of paragraph 3.1, when you say “our objective is for the agent to learn the optimal execution time”, is the execution time equivalent to the action time, and therefore represented by Tt?
-	Typo: “but but” in the sentence starting with “it is worth noting…” in paragraph 3.2
-	What is the meaning of “p” in eq. (2)?
-	Since the SAEC loss functions are (if I understand well) equal to those of SAC, rather than simply reporting the definitions, I would suggest to reorganize Section 4 to better explain how your formulation of the reward function is included in the update steps of the RL algorithm.
-	Section 5: When you refer to the “three RL algorithms”, do you mean SEAC, SAC and PPO? In this case, you should first say that you are comparing SEAC results with SAC and PPO in the text, otherwise it is not clear to the reader.
-	What are you representing differently on the left and right side of Fig. 5 and 6? Is it the right side simply a y-axis zoom-in of the left side? You should specify it on the figures' captions. What is the legend for the lighter colored plots?
-	I think that Fig. 7, as it is, is not very informative. It shows that SEAC dynamically changes the control rate, but it doesn’t allow to evaluate whether it does it in a meaningful way. Showing the scenario and/or information about the corresponding actions would make the concept clearer.
-	I feel Fig.8 would be more readable by inverting x and y axes (evaluation metric on the y-axis). Furthermore, you mention the overall reward both in the section and in the figure caption, but is the overall reward shown somewhere?

### Soundness
2 fair

### Presentation
2 fair

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
This paper extends the classical RL setting, where there is no concept of the action execution time, to RL with elastic time steps. The authors propose SEAC to output the next action as well as the duration of the next time step.

### Strengths
The proposed problem is interesting. The figures are vivid, and the paper is easy to follow.

### Weaknesses
The contribution and novelty is vague. As for the traditional RL, the control frequency is only an abstract definition. I think the proposed framework can be seen as a special instance of the traditional RL framework given a reformulated action space / state space / reward function. The algorithm also seems quite like SAC with new state / actions. Also, what is the relationship between the proposed algorithm with HRL methods?

### Questions
See above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
