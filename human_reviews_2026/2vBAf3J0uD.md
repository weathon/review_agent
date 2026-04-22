# Pretraining in Actor-Critic Reinforcement Learning for Robot Motion Control

- Avg Score: 3.60
- Decision: Reject
- Scores: 4, 4, 2, 4, 4

## Abstract
The pretraining-finetuning paradigm has facilitated numerous transformative advancements in artificial intelligence research in recent years. However, in the domain of reinforcement learning (RL) for robot locomotion, individual skills are often learned from scratch despite the high likelihood that some generalizable knowledge is shared across all task-specific policies belonging to the same robot embodiment. This work aims to define a paradigm for pretraining neural network models that encapsulate such knowledge and can subsequently serve as a basis for warm-starting the RL process in classic actor-critic algorithms, such as Proximal Policy Optimization (PPO). We begin with a task-agnostic exploration-based data collection algorithm to gather diverse, dynamic transition data, which is then used to train a Proprioceptive Inverse Dynamics Model (PIDM) through supervised learning. The pretrained weights are then loaded into both the actor and critic networks to warm-start the policy optimization of actual tasks. We systematically validated our proposed method with 9 distinct robot locomotion RL environments comprising 3 different robot embodiments, showing significant benefits of this initialization strategy. Our proposed approach on average improves sample efficiency by 36.9% and task performance by 7.3% compared to random initialization. We further present key ablation studies and empirical analyses that shed light on the mechanisms behind the effectiveness of this method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a way to pretrain RL policies by introducing an inverse dynamics model in the actor-critic architecture. The pretraining is valid for a single embodiment with arbitrary tasks.
The algorithm is evaluated in simulation on quadruped locomotion tasks.

### Strengths
- The problem setting is interesting and timely.
- The proposed method is well motivated and generally applicable.
- The evaluations are chosen well and clearly show the benefit of the proposed method.

### Weaknesses
- The evaluation is limited to a single embodiment. 
The idea of an inverse dynamics model is applicable to many (if not all) embodiments found in RL benchmark problems. Additional results on different embodiments, for example those found in Gymnasium, would strengthen the paper significantly. With the presented results it is difficult to judge whether and for which systems the proposed pretraining is helpful.

- The method is the reliance on well curated prior data set.

I would hypothesis that the quality of the inverse dynamics model and therefore the pretraining success highly depends on the training data. In fact, the collection of the dataset seems to be a major effort that includes many hyperparameters (including uncertainty quantification, domain randomization, and reward tuning). There is only a single ablation w.r.t. the quality of the data set. 

- Missing baselines. Model-based and offline RL can both be used for pre-training (for example [1]. There is generally a big body of work on "Offline-to-online reinforcement learning" [2] that can serve as baseline.

[1] Nakamoto, M., Zhai, S., Singh, A., Sobol Mark, M., Ma, Y., Finn, C., ... & Levine, S. (2023). Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems, 36, 62244-62269.

[2] Xie, Z., Lin, Z., Li, J., Li, S., & Ye, D. (2022). Pretraining in deep reinforcement learning: A survey. _arXiv preprint arXiv:2211.03959_.

### Questions
-  For Figure 3 and Section 4.5: I assume the figure and section only describes the actor and the critic gets a 'value synthesizer' head instead of an 'action synthesizer'? Are there other difference between the value and critic? Do they share the same backbone?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a pretraining method for actor–critic RL in robot motion control. A Proprioceptive Inverse Dynamics Model (PIDM) is trained on task-agnostic exploration data and used to initialize both actor and critic networks in PPO. On seven quadruped tasks, this warm-start improves sample efficiency by 40.1% and final performance by 7.5% over random initialization.

### Strengths
* Clear idea and solid execution.
* Experiments are broad and well-controlled, with ablations and weight-update analyses that convincingly show consistent gains.

### Weaknesses
* Limited to one embodiment (ANYmal-D); generalization to other robots remains untested.
* The PIDM’s low predictive accuracy (40–50%) raises questions about what is actually transferred.
* Missing stronger model-based baselines (like TD-MPC2).

### Questions
* Would results hold for manipulators or bipeds?
* Is the benefit due to dynamics knowledge or regularization?
* Could recurrent or attention-based PIDMs improve transfer?
* Is there any multi-modality issue when fitting one model with multiple source data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a pretraining–finetuning scheme for actor–critic RL in robot motion control. The authors first collect task‑agnostic exploration data and train a Proprioceptive Inverse Dynamics Model (PIDM). The pretrained PIDM weights are then used to initialize both actor and critic via a modular architecture, aiming to warm‑start PPO without changing task formulations or hyperparameters. On seven ANYmal‑D tasks (locomotion, pedipulation, and five parkour skills), the method reports an average +7.5% final performance increase and 40.1% fewer iterations.

### Strengths
Well written and easy to follow; the modular design makes integration into PPO straightforward.

Consistent gains across seven tasks with clear metrics

Ablations (actor vs. critic vs. both; exploration vs. task data) and probing analyses provide insight into why the initialization helps

Code shared with intent to open‑source; good reproducibility posture

### Weaknesses
Comparisons are limited: beyond a vanilla MLP and “PIDM random init” there are no direct baselines to other pretraining/weight‑sharing approaches in RL. This makes it hard to position the gain relative to prior art.

Missing control for “pretrain the policy itself”: it would be important to compare against pretraining a standard actor–critic on the same exploration data (e.g., behavior‑cloning warm start or supervised inverse‑model head inside the policy) to isolate the benefit of the PIDM module versus generic weight initialization.

Practical clarity: the repository would benefit from clearer “contribution scripts”/entry points for (i) data collection, (ii) PIDM pretraining, and (iii) task fine‑tuning to ease reproduction.

### Questions
How does performance change if you directly pretrain the actor–critic on the same exploration dataset?

Can you provide adapted baselines from prior pretraining/weight-sharing work in RL to strengthen positioning? Even if not perfectly matched, a careful adaptation in Isaac Lab would be informative.

PIDM’s normalized error is ~40–50% overall. Do higher‑accuracy PIDMs (e.g., via longer training or temporal models) correlate with larger RL gains? A small study varying pretraining budget could clarify whether accuracy or structure (i.e., the inductive decomposition) drives the benefit.

What happens if you freeze the PIDM backbone initially (or for a set number of iterations) and only train the Intention encoder/Action synthesizer? This could separate “good initialization” from “continual adaptation.”

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a method for utilizing pre-training in the context of reinforcement learning.
Specifically, it proposes to learn a task-agnostic Proprioceptive Inverse Dynamics Model (PIDM) from data collected by an exploration policy before the agent is trained on the actual task.
The PIDM is then used as part of the actor and critic networks during task-specific training.
They show that this approach improves both sample efficiency and final performance on 7 simulated locomotion tasks involving the ANYmal robot when trained with PPO.

### Strengths
- This work addresses an important challenge in reinforcement learning: improving sample efficiency through pre-training.
I believe pre-training could be an important step towards making RL more practical for real-world applications, especially in robotics.
By fully separating the pre-training phase from the task-specific training, the proposed method has the potential to reduce training time for a wide variety of tasks.

- Furthermore, the proposed method seems compatible with many existing RL algorithms, as it only proposes a modification to the actor and critic models and not to the training algorithm itself, except for the pre-training phase.

### Weaknesses
- In my opinion, the main weakness of this work is that the method is evaluated on a single type of robot (ANYmal) in simulation and for locomotion only. Yet, it is framed as a general method for pre-training in reinforcement learning for robot motion control. I believe that this kind of framing is not sufficiently supported by the experiments presented in the paper. Specifically, I would suggest the following improvements:
  - In the context of locomotion, the paper could benefit from evaluations on a wider variety of robots (e.g., see LocoMujoco [1] which provides a simulation of many different robots)
  - It should either be made clear that this method is intended for locomotion tasks, or the method should be evaluated on other types of tasks. In particular in manipulation tasks, the role of proprioceptive information may be less important than in locomotion tasks, and it would be interesting to see how the method performs in such a context.
  - In the conclusion, the authors state that "in contrast to related work, [their] modeling of the problem addresses the sim-to-real gap in robotics". However, no sim-to-real experiments are presented in the paper. I believe that either such experiments should be added, or this claim should be removed.
- Hypothesis 1, which states that neural network policies encode intended target states in the earlier layers and actions in the later layers is not convincingly supported by the experiments in Section 5.2 for two reasons:
  1. Layer 0 eventually surpasses layer 1 in prediction error, contradicting the statement that "the correlation between the future state diminishes as we progress deeper into the network". 
  2. The observed effect could also have explanations different from Hypothesis 1. E.g., if the output actions are motor torques, then in order to reconstruct the delta angles from them, the current velocities are needed. Information about those might be present in the first layers of the network and not in the later layers. Hence, in my opinion, further analysis is needed to support Hypothesis 1.

[1] Al-Hafez, Firas, et al. "Locomujoco: A comprehensive imitation learning benchmark for locomotion." arXiv preprint arXiv:2311.02496 (2023).

### Questions
- What is the state and action space of the experiments with ANYmal?
- Why does the PIDM model require a history of proprioceptive states? Are joint velocities not included in x?
- In Figure 6, how many environment steps does each iteration amount to?
- How is the PIDM used in the critic?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a method for pretraining neural network weights to warm-start actor-critic reinforcement learning (RL) in robotic tasks.
The core idea is to pretrain a Proprioceptive Inverse Dynamics Model (PIDM) using task-agnostic, exploration-based data.
This pretrained model is then integrated into the actor and critic networks of a PPO algorithm, providing a better initialization than random weights.
The method is evaluated on quadrupedal robot tasks in simulation, demonstrating improvements in final performance and sample efficiency compared to using randomly initialized networks.

### Strengths
1. A novel method of initializing the networks for RL training.
2. Clear presentation and thorough empirical analysis about the methods.
3. An interesting finding in section 5.2: it suggests that policy networks naturally learn a structure where earlier layers predict future states, and later layers invert this to compute actions. This reminds me of recent work in solving robot manipulation tasks by learning a world model to predict future observations (or states) first and predicting the actions from an inverse dynamics model.

### Weaknesses
1. While the method does not require expert demonstrations, the exploration-based data collection phase itself involves training a separate PPO policy. The computational cost and time required for this pretraining stage are non-trivial and are not quantitatively compared against the sample efficiency gains in the main RL tasks. A discussion on the net computational benefit would strengthen the paper.
2. The hypothesis (Section 4.1) and probing experiment (Section 5.2) suggest that policies first form an target state and then compute the inverse action. If this is the case, it is somewhat counter-intuitive to initialize the network with an inverse dynamics model (PIDM). A more aligned approach might be to initialize the earlier layers with a forward dynamics model and the later layers with the PIDM? The paper would benefit from a deeper discussion or an ablation on this point.
3. The paper motivates the method by mentioning its relevance to the sim-to-real gap (e.g., through noise and domain randomization). However, all experiments are conducted in simulation. For a robotics paper, real-world validation or a dedicated sim-to-real transfer experiment would significantly strengthen the claims about the practical effectiveness of this method.

### Questions
1. Why using $ \Delta x_{t+1}^* $ instead of $ x_{t+1}^* $? Does this design choice help bridge the distribution shift when switching the Delta Encoder to the task-specific Intention Encoder? Should using $x_{t+1}^*$ lead a smaller gap when replacing it with $o_t$?
2. From Figure 6, the PIDM with random initialization underperforms the vanilla MLP in some tasks (e.g., Locomotion) but performs similarly or better in others (e.g., Climb Up). What factors might explain this inconsistent performance? 
3. The paper states that the method addresses the sim-to-real gap by incorporating noise and domain randomization. Beyond this, are there specific properties of the pretrained PIDM that make it particularly suitable for sim-to-real transfer?
4. This method reminds me of offline-to-online RL method [1]. How does this pretraining approach compare to alternative ways of leveraging the exploration data, such as:
1) Continue to train the actor and critic with normal RL training after the first phase of data collection using exploratory policy.
2) Using the data to fill the replay buffer for an off-policy RL algorithm.
3) Using the exploratory policy as a behavior policy for off-policy RL?

[1] Efficient Online Reinforcement Learning with Offline Data. Philip J. Ball, Laura Smith, Ilya Kostrikov, Sergey Levine. ICML 2023.

Minor
1. Please ensure all notations are clearly defined upon first use. For example, the definition of $ \Delta x_{t+1}^* $ is not clear. $\Delta x_{t+1}^* = x_{t+1}^*-x_{t}$, is this correct? Why using a * here?

### Soundness
3

### Presentation
3

### Contribution
2
