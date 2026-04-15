# Online Pre-Training for Offline-to-Online Reinforcement Learning

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Reinforcement Learning (RL) has achieved notable success in tasks requiring complex decision making, with offline RL offering the ability to train agents using fixed datasets, thereby avoiding the risks and costs associated with online interactions. However, offline RL is inherently limited by the quality of the dataset, which can restrict an agent’s performance. Offline-to-online RL aims to bridge the gap between the cost-efficiency of offline RL and the performance potential of online RL by pre-training an agent offline before fine-tuning it through online interactions. Despite its promise, recent studies show that offline pre-trained agents often underperform during online fine-tuning due to inaccurate value function, with random initialization proving more effective in certain cases. In this work, we propose a novel method, Online Pre-Training for Offline-to-Online RL (OPT), to address the issue of inaccurate value estimation in offline pre-trained agents. OPT introduces a new learning phase, Online Pre-Training, which allows the training of a new value function that enhances the subsequent fine-tuning process. Implementation of OPT on TD3 and SPOT demonstrates an average 30\% improvement in performance across D4RL environments, such as MuJoCo, Antmaze, and Adroit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies offline-to-online reinforcement learning and finds that directly fine-tuning offline pretrained Q-value functions typically does not lead to good online fine-tuning performance. Therefore, this paper proposes an online pretraining phase for offline-to-online RL, which aims to enable a smoother transition and better fine-tuning results.

### Strengths
(1)This paper addresses an important problem, as direct offline-to-online transitions often do not work well. This work can be seen as a method for transitioning offline pretrained Q-values (which are biased due to the behavior cloning term) to adapt more effectively to the level of online values (training from scratch).

(2) The experiments and ablation studies are comprehensive and demonstrate strong results.

### Weaknesses
(1)The motivation to address inaccurate value estimation lacks analysis. The current discussions about inaccurate value estimation are derived from other works. I believe the authors should provide deeper insights into why inaccurate value estimation hinders online learning. Is it truly the inaccurate value estimation that causes problems for offline-to-online learning, or is it overfitting to the offline dataset? In some cases, inaccurate value estimation is not necessarily an issue. For example, using pessimistic estimates online does not prevent further improvement but instead helps stabilize value function learning and could lead to good performance [1] [2].

(2)I have some concerns about the sample efficiency of OPT. OPT could be considered a method for transitioning offline-learned value functions, which are based on behavior cloning and temporal difference learning (biased due to the behavior cloning term), to an online level. From the learning curves, it appears that OPT requires completing this adaptation before it starts showing improvements. While I understand that this adaptation inevitably requires some environment interactions, I wonder if this could be improved. For instance, during the pretraining phase, it seems strange that OPT only allows the offline pretrained policy to collect data. Since you are adapting Q with online pretraining (Q on-pt), why not allow a new policy to evolve based on Q on-pt and have it collect data as well? This way, Q on-pt could more easily calibrate its value estimation to the online level based on its own experiences. This might reduce the need to rely on offline data during the online pretraining phase, as trial-and-error could mitigate overestimation and enable further improvement beyond Q off during the transition phase.

(3)I am also concerned about the complexity of OPT. OPT, in its current form, combines different techniques: MEGA for online pretraining and balanced replay for online fine-tuning. This makes OPT seem overly complicated, in my view. For example, if OPT did not use MEGA, would it still work? If it works, then why use MEGA? If not, then why is MEGA necessary? Similarly, why use balanced replay? After the online pretraining phase, shouldn’t Q on-pt already capture the offline pretrained policy’s knowledge by training on B-on, then why still using the offline datasets is important during online phase? Also you are comparing to MEGA and balanced replay in experiments, but you are combining their strategies in some way, and it is not suprising that OPT could be better than MEGA and balanced replay alone (Off2On has the second highest total score in table 2).

[1]Mark, M. S., Sharma, A., Tajwar, F., Rafailov, R., Levine, S., & Finn, C. Offline RL for Online RL: Decoupled Policy Learning for Mitigating Exploration Bias.
[2]Luo, Y., Ji, T., Sun, F., Zhang, J., Xu, H., & Zhan, X. (2024). Offline-Boosted Actor-Critic: Adaptively Blending Optimal Historical Behaviors in Deep Off-Policy RL. arXiv preprint arXiv:2405.18520.

### Questions
(1)Why is Qoff-pt also being updated during online fine-tuning? Is updating Qon-pt not sufficient?

(2)Question about the coefficient k: Does it start from a small value? With a small k, it seems like you are adding another adaptation period for OPT. Why is this necessary? After online pretraining, shouldn't Qon-pt already retain the good state-action pairs from Qoff and adjust their values to the online level? Is this additional period introduced because the initial stage of online learning might cause Qon-pt to unlearn?

(3)I’m curious about the performance of OPT with different numbers of offline training gradient steps. Could you provide some insight into how this affects results?

(4)Could you provide more details on training Qon-pt using only Bon? Does Qon-pt produce highly overestimated Q-values for the state-action pairs in Bon, which hinders online improvement?

(5)For the ablation in Section 5.3: How is Qoff-pt trained during online pretraining? Does it continue to collect and store its experiences in Bon and then update using Equation (3)? Why is it that online pretraining with only Qoff-pt cannot correct the Q-values it learned during the offline phase to the level of purely online learning?

(6)How do the baseline methods use the offline dataset during online learning? They might benefit from using the offline dataset online, and this needs further explanation.

(7) Is the basic idea of OPT somehow connected to PEX?


I look forward to discuss with the authors, and increase my score if the questions are answered thoroughly.

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
This paper introduced the Online Pre-Training for Offline-to-Online RL (OPT) to address the issue of inaccurate value estimation in offline pre-trained agents. In particular, OPT introduced a new learning phase, named Online Pre-Training, to learn a new value function using both the offline dataset and new online samples. Later, OPT uses two value functions for online policy fine-tuning.

### Strengths
- The writing is clear and the paper is easy to follow.
- The idea of using two different value functions is interesting.
- OPT shows promising results in the experiments.

### Weaknesses
- Since OPT needs to learn an extra value function, it increases the computational cost.
- OPT introduces some new hyper-parameters, i.e., $N_\tau$, $\alpha$, $\kappa$ that need further tuning.
- Missing recent baselines, i.e., BOORL (https://github.com/YiqinYang/BOORL), SO2 (https://github.com/opendilab/SO2).
- The proposed method is mostly empirical, there is a lack of theoretical analysis of the proposed method.
- No Figure name & caption & xy label in Appendix G. The legend in the image is very obscure.

### Questions
- The paper claims OPT helps to learn a more accruate value function in offline-to-online RL. However, there is no direct proof for this point. Does OPT really learns more accurate value functions compared with other baselines? I would suggest the authers to plot the MSE error of the learned value function with an approximated optimal value function across different tasks for different baselines. We can first train a set of online agents until they can achieve near-optimal performance, and use the mean value function of the set of online agents as an approximation of the optimal value function.
- The paper claims OPT can be used with different offline RL algorithms. My question is if $\pi_{off}$ is quite conservative which mostly takes similar actions as in the offline dataset. Then there will be an exploration issue during the online pre-training stage. Because $\pi_{off}$ avoids to take ODD actions, then the data in the $B_{on}$ is likely to be very close to the data in $B_{off}$. I don't think pre-training on such $B_{on}$ will bring any significant improvement.
- From Appendix G, OPT sometimes performs worse than the other baselines in early stages, i.e., hopper-medium-v2, hopper-medium-replay-v2, walker2d-medium-v2, walker2d-medium-replay-v2, antmaze-umaze-diverse-v2, antmaze-large-play-v2. Can the author explain the reasons?
- What is the wall-clock running time of OPT?
- Which task does Figure 7 use? Is the optimal $N_\tau$ the same for all tasks?

### Soundness
2

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
The paper proposes a new improvement technique for offline-to-online RL algorithms to tackle the challenge of slow online fine-tuning from offline pre-training. The key idea is to use separate online pre-training phase to pre-train a new Q-function before fine-tuning the offline-pretrained Q-function and the new Q-function together using regular online RL objective. The online pre-training of the new Q-function leverages a meta-adaptation formulation where the objective is to make the learned new Q-function to be able to adapt to new data encountered during online fine-tuning better. Empirical results show that the proposed can be applied to existing offline RL algorithms like TD3+BC and SPOT to achieve SOTA performance on the D4RL benchmark.

### Strengths
- The paper is easy to follow and the empirical evaluations seem thorough and comprehensive.
- Empirical results show the proposed method is effective in improving offline-to-online RL sample efficiency.

### Weaknesses
- The online pre-training phase uses a meta-adaptation strategy from prior work that requires meta gradients. This seems to be the most complicated part of the algorithm design. How necessary is this design? Is it sufficient to simply train the new Q-function on the online data with a frozen pre-trained policy? Is that what the second initialization method in Section 5.1 is? It would be good to clarify so. 
- Figure 5 and Appendix G have very blurry figure legends.

### Questions
- How sensitive is the scheduled of $\kappa$ (the balance between the value functions) on the performance?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors consider the offline-to-online setting, where offline pretraining is followed by finetuning with online interactions. They propose an algorithm which adds another training phase between the offline and online ones---online pretraining---which makes use of online interactions and the offline dataset to learn a seperate value function. This algorithm is shown to improve performance on standard benchmark environments from D4RL.

### Strengths
The proposed algorithm and training procedure is clearly explained and shows impressive results on the benchmarks compared to the baselines. 
There is also a variety of additional experiments in Section 5 that nicely round out the paper.

### Weaknesses
The main issue for me is the complexity of the algorithm and the source of the benefits. 
The algorithm involves many new pieces and some important aspects are not studied at all.
For example, the meta-adapation strategy outlined in lines 198-199 is not explained and its contribution to the overall algorithm is unclear. Why is it necessary to train in such a manner instead of, for example, using  $\mathcal{L}^{on}_{Q^{on-pt}}(\psi)$ as the second term in eq.3?


There is also the $\kappa$ weighting parameter, balancing the offline and online losses, which seems important to the algorithm. Schedules for $\kappa$ are discussed in section 5.2. but there's no further description of how exactly it was chosen. Also, the sensivity to this parameter seems important to discuss.

While the benchmark results are good, it is hard to see how these different algorithmic choices impact it and how each of them contribute.

### Questions
- Regarding section 5.2: t-sne visualizations can be heavily influenced by its hyperparamters. It would be better to use a different assesmment of distribution differences which are more reliable. 

- In table 1, most of the other baseline algortihms (aside from Off2On) underperform vanilla TD3+BC, is this to be expected?

- In table 4, the confidence intervals sometimes overlap significantly. It could be more suitable to not bold any entry in those cases since the performance of the algorithms are similar. 


- In the paragraph starting at line 156, it is mentioned that the online pretraining phase helps adjust the value function before the online phase. It would be good to confirm that this is the case as an ablation. In other words, what happens when you directly use a new value function for the online finetuning phase without the pretraining?


- In TD7 [1], the policy gets updated with batches of collected data with the policy fixed to collect each batch.
Online pretraining sounds like something similar, where the current best policy is used to generate data which will be used to perform a batch of updates. This is only done once instead of repeatedly as in TD7 though. Any thoughts on this comparison?


- In fig2, the fire symbol with the red background gives the impression that there is something negative happening in that module. I see it is supposed to contrast the frozen symbol but perhaps a different choice would be better to avoid confusion. For example, using green instead of red and having a checkmark instead or even no symbol.



[1] "For SALE: State-Action Representation Learning for Deep Reinforcement Learning" Fujimoto et al.

### Soundness
3

### Presentation
3

### Contribution
3
