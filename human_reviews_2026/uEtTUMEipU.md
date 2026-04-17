# Towards Understanding Learning from Human Interventions

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
As AI systems are deployed in real-world environments, they inevitably make mistakes where human interventions could provide valuable corrective feedback. However, many of the optimality assumptions made by existing methods for learning from interventions are invalid or unrealistic when measured against how humans actually intervene in reality. We conduct a deeper analysis with intervention data from real human users, revealing that humans often intervene sub-optimally in both the timing and execution of interventions, often acting when they perceive the agent’s progress to stagnate. Building on these insights, we show that the current methods of simulating human interventions, and the corresponding methods to learn from these interventions, do not accurately capture the behavior modes of human users in practice. Based on these insights, we introduce an improved approximate model of human intervention that better captures this behavior, enabling accurate simulation benchmarking of learning algorithms and providing a more reliable signal to develop better algorithms in the future. As a start to building on these insights, we propose a simple algorithm that combines imitation learning and reinforcement learning with a regularization scheme to leverage corrections for exploration rather than directly making strong optimality assumptions. Our empirical evaluation on simulated robotic manipulation tasks demonstrates that our method improves task success by $\sim$52\% and achieves $\sim$2x reduction in real-human effort on average as compared to baselines, marking a significant step towards scalable, human-interactive learning for robot manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper suggests that in intervention learning, human interventions are non-Markovian and depend on recent progress over the past few steps. The paper proposes STEER algorithm, which combines the Bellman regression loss with a behavior cloning loss on expert interventions. On both simulated and real human interventions, STEER outperforms the baselines across three robotic manipulation tasks.

### Strengths
1. Figures 2, 3, and 4 intuitively illustrate the human takeover mechanism, showing that human interventions are non-Markovian.
2. The paper conducts experiments on three manipulation tasks in Robosuite simulators, where STEER consistently outperforms the baselines.
3. STEER is easy to implement and has high compuational efficiency.

### Weaknesses
1. **The connection between the motivation and STEER algorithm is unclear**. The motivation in Fig 2,3,4 highlights the non-Markovian nature of the human takeover mechanism, but the STEER algorithm itself simply combines two losses from two separate buffers. It's unclear how STEER's design (line 341, page 7) is related to the proposed intervention mechanism (line 301, page 6). I don't see why STEER should be better than the baselines at handling non-Markovian interventions.
2. For the **simulated human experiments**, the paper **only uses one gating model** based on recent task progress (line 301) with a sub-optimal RL expert. It’s possible that under different gating models (e.g., using the action difference between the agent action and the RL expert action for intervention), baselines like HG-DAgger could outperform STEER.
3. STEER **relies on the task reward** for training, whereas baselines such as HG-DAgger and RLIF do not require it.
4. It's unclear **whether the proposed gating model can help the agent succeed during training**.  If the interventions are poor and fail to help the policy reach the goal, STEER may gain an unfair advantage over baselines because STEER can access the task reward and has a vanishing coefficient $\lambda(i)$. In order to justify the gating model, I would like to see the success rate of the behavior policy during training (with proper expert interventions, I expect that this training success rate approaches the expert's success rate).

### Questions
1. What is the reward model used to compute Q* and V* in Figures 2, 3, and 4? Is the reward very sparse?
2. Figures 2, 3, and 4 claim that humans are suboptimal. Could this result from the difference between the task reward and the human's intrinsic reward?
3. In Fig. 5 (Task Square) and Fig. 8 (Sim Can), the performance of HG-DAgger decreases as the number of human interventions increases. Could this result from an improper gating model, where the simulated human only provides data in non-critical states? As the agent's performance improves, how does the behavior of the gating model change?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the problem of learning from human interventions in robotic control. 

The authors observe that prior works assume humans intervene optimally and instantaneously based on per-step suboptimality, which is unrealistic. 

Using data from human–robot interaction experiments, they find that real human interventions are suboptimal and non-Markovian, typically triggered by stagnation in task progress rather than instantaneous errors.

They propose 

(1) a data-driven model of human intervention that predicts intervention timing based on progress stagnation over a short temporal horizon, and 

(2) STEER a hybrid RL–IL algorithm that treats interventions as a signal to guide exploration rather than as expert demonstrations. STEER combines off-policy RL updates with a decaying-weight behavior cloning term on human corrections.

### Strengths
1. I like the empirical studies that showcase the characteristics of human intervention: human is not optimal, human is not deciding based on single step (non-Markovian). But I suspect if $1/k \log_{\gamma}\cfrac{V*(s_t)}{V^*(s_{t+k})}$ is a correct metric and if the result is valid as we don't know the expertise levles and the number of the human subjects involved in the experiment.

2. The paper proposes RL+IL method to learn from with human-in-the-loop data -- which is not novel.

### Weaknesses
1. **Critical experiment information is missing.**  See questions.
2. **Critical related works are missing**. 
    * In [1], reward and human intervention are both used to learn from online human-in-the-loop data.
    * In [2], reward-free human-in-the-loop learning can achieve high learning efficiency and the paper finds that adding RL reward actually harms final performance.
    * In [3], BC loss is added to accelerate reward-free human-in-the-loop learning.
    * In [4], TD3 + BC is used as an effective offline RL method, which is highly similar to the Section 5
3. The contributions of this paper is questionable. The core algorithmic innovation is "BC regularized off-policy RL" which is already be implemented in reward-free human-in-the-loop learning[3] and offline RL[4]. And the empirical study on the characteristics of the human intervention is not convincing. Also, the connection between the empirical study in Sec 4 and the algorithm in Sec 5 is unclear.


[1] Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization

[2] Learning from Active Human Involvement through Proxy Value Propagation

[3] Data-Efficient Learning from Human Interventions for Mobile Robots

[4] A Minimalist Approach to Offline Reinforcement Learning

### Questions
### **Critical experiment information is missing.** 

1. Is there a complete footage of the human subject experiments? 
2. What's human subjects performance if doing full intervention? 
3. What's their expertise in the task? 
4. Is it possible that 3D mouse is not a good control device? Have you tried different control devices? Have you study the action distribution comparing "simulated human" and real human?
5. What if you have multiple human subjects with different expertise on the task? Does the analysis in Sec 4 still hold?
6. Did you get IRB? 
7. How many repeat experiments are conducted with real human?
8. How to evaluate the trained policy? How many episodes are run? How diverse the training scenarios and the test scenarios are?



### **Section 4 The characteristics of human intervention**

I find this section interesting, but there are still many unresolved issues.

1. How to ensure that the optimal experts' $\pi^*$ and $V^*$ are optimal? Could you post the metrics for this expert policy in the experiment tables / figures?
2. Does Figure 3 plots all $Q^*(a_h)$ for all human actions during the training? Do you observe the Q values for human actions changes across the training? For example, when human becomes tired as the agent becomes performing, will the quality of human actions drop --- this explain why RL helps at the later stage of training?
3. Have you tried to use the action difference, or the probability of agent actions under the expert's action distribution, as the gated function to build the intervention predictor? How it performs? 
4. **The Table 1 is concerning.** Flipping a coin will give you 0.5 recall on predicting human intervention. The precision is too low to 0.141, which means the gated function asks human intervention unnecessary. I don't think the intervention model is valid and thus all downstream experiments on the simulated human are not convincing.
5. **How to ensure that the expert policy satisfies the human intention?** For example, in the "Can" task, what if the human prefers to pick up the can higher in order to ensure safety but the expert RL policy is just optimal and use the shortest path? The single success rate and the sparse reward is not good enough to capture the human intentions --- this might explain why the Q* values of human doing bad but the proposed gating function is still so bad.


### **Experiment**

Experiments are very concerning.

1. According to Appendix B.2.5: "We use the same parameters across all experiments and report the best result on 3 seeds." **I don't think it's a correct practice to report best result over multiple repeated runs**.
2. According to Appendix B.2.1, "To accelerate convergence, we initialized training with 25 demonstration trajectories, fixed the initial positions of both the robot endeffector and goal objects across episodes, and restricted the action space to only end-effector position deltas and gripper commands." 
    * What is the behavior cloning performance on these 25 demonstration? Given that there is no diversity in the environment, I suspect whether 25 episodes BC warmup can already achieve good performance.
    * If I understand correctly, **there is no difference between episodes in terms of initial position, objects, goals?** I believe this severely undermines the soundness of the experiment, as it becomes unclear whether the policy learns to react to observations or simply memorizes the data—especially given the BC baseline is not provided.
3. Important baselines are missing. This paper only compares HIL and RLIF, yet there are many important baselines should be included.
    * Behavior Cloning with human demonstration (with independently collected 15k data).
    * Behavior Cloning with expert RL policy (with 15k data, and 40k data, to have a fair comparison).
    * HG-DAgger
    * PVP
    * IWR


### The intuition

1. First, I can't see the connect between the discussion in Sec 4 (let's put aside that whether they are valid, even though I think the conclusion is reasonable (non-Markovion, suboptimal), but the results are not convincing) and the only technical innovation the BC+RL in Sec 5. 
2. So STEER is basically HIL + BC loss.
3. The decay BC weights basically suggests we want human intervention to guide exploration and RL reward to guide later training. But I think the reverse idea is more practical in real-world setting. I think mastering basic skills via large scale pretraining and finetune the model to address compounding error and OOD actions in closed-loop rollout via human-in-the-loop is more interesting as it avoid human cognitive cost in providing basic-level demonstrations.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper does an analysis of human intervention in the setting of robot learning and proposes a method for learning from human interventions that takes into account what is the best way to use human intervention to learn a policy.

### Strengths
Analysis on human intervention is original:
The idea to analyze human interventions is novel. The paper collects real intervention signal and performs an analysis of when humans intervene. The analysis is informative and the proposed model takes into account this analysis for better performance. 

The baselines are thorough:
The paper takes into account all of the relevant baselines.

### Weaknesses
Limited experimental evaluations:
The paper only evaluates on three simulated tasks from the same benchmark which is not enough. The baselines the method compares to all have real robot experiments to back up the methods, which in the setting of learning from human demonstrations is important. The sections of real world experiments is also misleading as it sounds like it is real robot experiments when it is only real human intervention. 

Lack of details on human study:
The paper does not mention details of the human study, which is central to the analysis. 

Generalization to different intervention styles:
The experiments is ran with two humans. Considering how different humans may have different intervention styles, more humans are needed to test the proposed model.

### Questions
What are the details of the human study? How many participants were there?

How would the method on a real robot, where the interventions might look a lot more different?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a detailed analysis of how humans interact with robots in a human-in-the-loop policy learning setting. The authors study when and how humans intervene or take over while supervising an autonomous agent operating in an environment. Building on this analysis, they propose a new **gating mechanism** that predicts intervention timing based on the recent progress over a horizon. This mechanism is designed to simulate human intervention behavior more realistically. Evaluations using real human intervention data show that this gating model better captures human behavior than existing baselines.

Furthermore, motivated by the observation that human interventions can be suboptimal, which are evaluated via a critic from a trained RL policy, the authors propose a **learning-from-intervention** framework. This framework incorporates human interventions as a decaying auxiliary supervision loss during off-policy RL training, guiding exploration without assuming intervention optimality. The approach is evaluated against imitation learning (IL) and RL-based human-in-the-loop baselines.

### Strengths
- The paper is **well-written and easy to follow**, offering a detailed analysis of how and when people intervene in human-in-the-loop manipulation tasks.

- The proposed **gating mechanism** is intuitive and well-motivated, and its validation with real human data is convincing.

- The **learning-from-intervention framework** is compared against a range of baselines and evaluated across multiple dimensions such as human effort and final task success, showing careful experimental design.

### Weaknesses
- **Ablations on the RL vs. Supervision Trade-off**: It is unclear how much of the final performance gain comes from the RL component versus the intervention-based supervision. It would strengthen the work to include ablations such as: (i) training only with sparse rewards, and (ii) using only the corrected segments for behavioral cloning without decay (still distinct from HG-DAgger since no offline dataset is used).

- **Intervention Modality**: In this work, all interventions are provided via a space mouse, which raises questions about the role of interface modality. The interventions appear to be non-Markovian (e.g., humans waiting for clear mistakes due to the cost of correction). It is unclear whether the proposed method would be as effective on intervention learning with kinesthetic teaching or VR teleoperation.

- **Real-World Validation**: While the proposed framework is tested with real human interventions in simulation, experiments on a physical robot would make the contribution significantly stronger, especially given the practical nature of human-in-the-loop policy learning.

### Questions
- **Effectiveness with Other Intervention Types**: Would similar intervention patterns emerge with more intuitive interfaces like kinesthetic teaching or VR teleoperation? Some discussion on this would be valuable.

- **Baselines for Reactive Interventions**: The paper’s analysis suggests that human interventions are largely reactive (i.e., following temporary policy failures). However, the comparisons against supervised learning methods mainly focus on HG-DAgger. Other methods, such as Sirius (Liu et al. 2023) [1], also model reactive supervision and would serve as more appropriate baselines. Why are only RL-based methods evaluated with real human interventions in the performance plots?

Overall, I found this to be a well-motivated paper that tackles an important aspect of human-in-the-loop learning. I appreciate the authors’ clear presentation and the effort put into both analysis and experimentation. I would be happy to hear the authors’ responses and clarifications to the questions and points raised above, as they will help me better understand the scope and impact of this work.

[1] Liu, Huihan, et al. ‘Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment’. Robotics: Science and Systems (RSS), 2023.

### Soundness
4

### Presentation
4

### Contribution
2
