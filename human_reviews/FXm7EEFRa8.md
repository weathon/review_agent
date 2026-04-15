# Reinforcement Learning with Action Sequence for Data-Efficient Robot Learning

- Decision: Reject
- Scores: 5, 5, 6

## Abstract
Training reinforcement learning (RL) agents on robotic tasks typically requires a large number of training samples. This is because training data often consists of noisy trajectories, whether from exploration or human-collected demonstrations, making it difficult to learn value functions that understand the effect of taking each action. On the other hand, recent behavior-cloning (BC) approaches have shown that predicting a sequence of actions enables policies to effectively approximate noisy, multi-modal distributions of expert demonstrations. Can we use a similar idea for improving RL on robotic tasks? In this paper, we introduce a novel RL algorithm that learns a critic network that outputs Q-values over a sequence of actions. By explicitly training the value functions to learn the consequence of executing a series of current and future actions, our algorithm allows for learning useful value functions from noisy trajectories. We study our algorithm across various setups with sparse and dense rewards, and with or without demonstrations, spanning mobile bi-manual manipulation, whole-body control, and tabletop manipulation tasks from BiGym, HumanoidBench, and RLBench. We find that, by learning the critic network with action sequences, our algorithm outperforms various RL and BC baselines, in particular on challenging humanoid control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes an extension to the Coarse-to-fine Q-Network (CQN) reinforcement learning algorithm that predicts Q-values for sequences of actions. Using sequences of actions instead of single actions is inspired by work on behaviroal cloning that demonstrated that predicting sequences of actions improves the model's ability to fit noisy and multi-modal distributions of expert demonstrations. The authors evaluate their method on simulated benchmark tasks from BiGym, HumanoidBench, and RLBench, where the method demonstrates performance gains compared to the original CQN method, as well as SAC and DrQ-v2+.

### Strengths
The paper is easy to follow and the modifications to the original CQN algorithm are explained clearly. The evaluation is conducted on a wide variety of simulated robotics tasks. The authors evaluate both the pure RL setting as well as RL with demonstrations and include ablations for key components of the algorithm.

### Weaknesses
The method focuses only on the CQN algorithm while the concept of predicting Q-values for action sequences rather than individual actions should be applicable to other reinforcement learning methods as well. It remains unclear whether such a modification would also lead to similar performance gains in other RL algorithms.

The method appears to train a large number of Q-functions (L * K, where L = 3 and K = 16 for some of the experiments). This large number of neural networks certainly reduces the computational efficiency of the method. Some comparison of the computational efficiency would be nice.

### Questions
1. I am unsure why giving the Q-function access to the action sequence is beneficial if the Q-function is still trained with just a one-step Bellman error. This way the Q-function is not really getting a learning signal for the rest of the sequence. For example, a valid solution to the loss from equation 4 could be to ignore all actions but $a_t$ and predict the Q-value only from $o_t$ and $a_t$, like a regular Q-function without access to the action sequence. Since this is the central component of the paper, I would appreciate greatly if the authors could expand on why the action sequence is beneficial if the Q-functions are trained with this loss and how the Q-functions are incentivized to make use of the entire sequence for the predictions.

2. There seems to be an ablation for the temporal ensemble missing. While the authors compare averaging the actions predicted at different timesteps and executing the entire action sequence, the authors do not show the results of predicting the actions at every step and always executing only the first action. The current "No temporal ensemble" ablation essentially reduces the control frequency of the task, which certainly can have a negative impact on the agent's performance, but this does not necessarily mean that it is beneficial to average actions predicted at multiple timesteps.

3. In some of the plots the methods are not trained for enough steps. E.g., in the Run plot of Figure 5, TD-MPC2 is trained for only 2e6 steps and at this point it seems to be on-par with CQN-AS, but it is unclear whether it will perform better or worse in the long run.

4. The plots in Figure 1 are lacking standard deviations, which makes it hard to assess the significance of the results. Furthermore, all experiments are repeated for only 4 runs. It would be good increase the number of runs since the performance of RL algorithms can be a quite stochastic.

5. According to equation 2, every Q-function only sees the bin that was chosen at the current level and one level before. The authors use the method with three levels (according to Table 3). How would the finest Q-function know which bin was chosen at the coarsest level? Clearly, the bin chosen at the coarsest level has a strong impact on actual action executed by the agent and consequently on the Q-values. If the method would learn a deterministic policy in an on-policy setting, the agent would always choose the same action for a given state and the finer-level Q-function would need to adapt to the changed meaning of the actions only when the agent is updated. However, the method is able to process data from expert demonstrations and therefore needs to be able to process off-policy data, so it is unclear to me how the Q-function at the finest level is able to infer which bin was chosen at the highest level. I am aware that this is rather a question about the original CQN method than CQN-AS, but since understanding CQN is essential for understanding the method, I would appreciate if the authors could clarify this issue.

6. In and around equations 2 and 4, some of the actions are bold, and some (e.g., lines 132, 141) are not. I was wondering what the difference between the bold and non-bold actions is.

7. What exactly is $\pi^l_K$ in equation 4? Is it a learned policy or does it just denote the maximization over the Q-functions?

8. Line 187: Why is the one-hot vector $e_k$ necessary? If I understand the architecture correctly, the GRU gets the $h^l_{t,k}$ in sequence, so it could just learn to count to get the k. Why is k nevertheless needed as an explicit input?

9. According to lines 187-189 $s^l_{t,k}$ contains no information about $a^{l}$, only $a^{l-1}$ and in line 191 the Q-function is defined as a mapping that takes only $s^l_{t,k}$ as input, but the Q-function is supposed to also have access to $a^{l}$. Where is $a^l$ added to the inputs?

10. Line 201: m is not defined. Is it a hyperparameter?

11. Line 201: The notation $a_{t-i}$ is a bit confusing. Previously $a_{t-i}$ would mean the action for timestep t-i, but here it is supposed to mean the action for timestep t calculated at timestep t-i, I think. I think the author should make that explicit.

12. Lines 203/204 state that computing the actions only every K steps and applying the K-step action sequence is beneficial for tasks that require reactive control. This is a bit puzzling to me since executing the action sequence open-loop should make the policy less reactive since the agent can only react every K steps, while in the temporal ensemble case, the agent can at least change the actions at every step to some extent.

13. Line 302: The claim "other RL baselines fail to effectively accelerate training with noisy human-collected demonstrations" is a bit misleading. It sounds as if the baselines perform on-par with CQN-AS if all methods are trained without demonstrations, and CQN-AS excels when given demonstrations, but Figure 4 does not contain experiments without demonstrations. It would be good to clarify that statement.

14. Line 405: The authors describe the "Open Oven" task as a "challenging, long-horizon task". While I have not worked with RLBench yet, I cannot imagine why the "Open Oven" task would have a longer horizon tasks than other tasks (e.g., "Open Door").

15. The claim "we find that CQN-AS often achieves competitive performance to model-based RL baselines, i.e., DreamerV3 and TD-MPC2, on tasks such as Run or Sit Simple" does not seem to be backed by the data for the "Run" task. On this task DreamerV3 achieves significantly higher performance and the TD-MPC2 run was stopped too early to allow for a proper comparison.


Typos:
1. Line 264: "of training" --> "of the training"

2. Line 377: "generic setup" --> "generic setups"

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper extends the existing Coarse-to-fine Q-Network (Seo et al. 2024) by predicting Q value over a sequence of actions. The intuition behind this design is that it may learn a more robust Q function for noisy demonstrations. The authors conducted a thorough comparison against many baselines on various types of control problems, including BiGym, HumanoidBench, and RLBench.

### Strengths
* The paper presents a simple-yet-effective idea that can improve the performance of CQN algorithm.
* The proposed method seems to be effective on various control problems, including manipulation and humanoid control, which is impressive.

### Weaknesses
* I am not convinced that Q-prediction with action sequences offers higher accuracy over noisy trajectories. I believe this is the most important claim in the paper, however, I cannot find a good discussion. The paper simply mentions that "our algorithm allows for learning useful value functions from noisy trajectories." However, this is not intuitive; I am not sure why adding more arguments to the Q network mitigates such problems. Examples and/or toy problems would be appreciated.
* In that sense, I am not sure how the proposed technique is generic to other RL algorithms. Is it only specific to CQN? If I combine a similar idea with other off-the-shelf algorithms, such as PPO or SAC, would it also offer a significant performance boost?
* The paper is less self-contained. The paper suddenly uses the term "level", which is never properly introduced in this paper. Therefore, I believe the current manuscript forces readers to visit the previous CQN paper, which is not desirable.
* The paper uses both RL and BC as problems, and I am not so sure it is a good idea. The paper may focus on one of the problems to consolidate the scenario.

### Questions
Please refer to the comments above.

### Soundness
3

### Presentation
2

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
This paper combines ideas from RL and Behavior Cloning (BC) by introducing a critic network that outputs Q-values over action sequences. The main idea is that estimating Q-values over action sequences could mitigate the effect of noisy data, especially in sparse-reward environments, and improve the agent’s anticipation of the cumulative impact of decisions over time.

The proposed Coarse-to-fine Q-Network with Action Sequence (CQN-AS) outputs Q-values for a sequence of actions, which helps the RL agent plan multiple steps ahead. This is especially useful in tasks with noisy or sparse reward signals.
By decomposing Q-values into levels $l$ and sequence steps $k$, this approach enables multi-level action consideration. The use of parallel computation for all sequence steps $K \in {1,...,K}$ is also efficient.

CQN-AS leverages a combination of MLP and GRU networks for feature encoding and processing, which is effective for handling high-dimensional input data. GRUs capture temporal dependencies, enhancing sequential learning.

In traditional reinforcement learning, actual rewards from subsequent steps are typically required for updates. However, CQN-AS addresses this differently, leveraging bootstrapping and multi-step returns to estimate future rewards without needing actual observed rewards for each step in the sequence. This allows the model to estimate returns for a series of actions based on current Q-value predictions.

### Strengths
- **Sequence-Based Q-Value Predictions**: Using Q-value predictions over action sequences is relatively novel afaik, particularly in reinforcement learning for robotics.
- **Strong Empirical Evaluation**: The use of tasks from BiGym, HumanoidBench, and RLBench provides a broad evaluation setup, which supports the paper’s claims of robustness.
- **Performance on Complex Robotics Tasks**: Outperforming majority of RL and BC baselines, especially in humanoid control, indicates that the method can handle complex, high-dimensional tasks.

### Weaknesses
- Although the approach aims to handle noise effectively, an ablation study on this can strenghten the claim. For instance, is there a mechanism to manage extreme cases of noisy trajectories, or is this approach limited in such scenarios?
- Sequential Dependency in Q-Values: The authors note that Q-values at subsequent steps are computed independently of previous Q-values, potentially overlooking dependencies across actions. While this simplifies training, it may reduce performance in tasks that require detailed planning or interdependencies between actions. Incorporating sequential dependencies between Q-values could better align with real-world robotic control needs. I'd like to know the author's thought on this.
- While the approach is empirically evaluated, there is limited theoretical analysis of why learning Q-values over action sequences should improve robustness to noise or sparse rewards. Providing some theoretical backing or intuitions for the coarse-to-fine structure could make the work more rigorous.
- If the goal is to apply this approach to real-world robotics, the paper can discuss transferability, as these are often major hurdles in robotics. Without this, it’s unclear how well CQN-AS would generalize outside simulated environments.

### Questions
- It can be helpful to include a brief discussion contrasting this approach with standard multi-step RL methods like $TD(\lambda)$. Traditional methods generally focus on handling delayed rewards rather than explicitly addressing noisy, multi-modal action distributions, which is a central aspect of this paper. Although the two frameworks tackle these issues differently, I believe this comparison would clarify the different approaches.
- HumanoidBench Results: The results for TD-MPC2 (Figure 5) appear to be cut off too early in the HumanoidBench experiments ? Is there a reason for this?

### Soundness
3

### Presentation
3

### Contribution
3
