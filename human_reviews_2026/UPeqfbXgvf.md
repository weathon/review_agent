# Online Finetuning Decision Transformers with Policy Gradients

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Decision Transformer (DT) has emerged as a powerful paradigm for decision making by formulating offline Reinforcement Learning (RL) as a sequence modeling problem. While recent studies have started to investigate how Decision Transformers can be extended to online settings, online finetuning with pure RL gradients remains largely underexplored: most existing approaches continue to prioritize supervised sequence modeling losses during the online phase. We identify hindsight return relabeling---a component widely used in online DTs---as a key obstacle that, while beneficial for supervised objectives, hinders the performance of importance sampling-based RL algorithms such as PPO and GRPO. In this work, we present a new algorithm that enables online finetuning of Decision Transformers purely with reinforcement learning gradients. Our approach represents a novel adaptation of the classical GRPO algorithm to the online finetuning of Decision Transformers. To make GRPO efficient and compatible with DTs, we incorporate several key modifications, including sub-trajectory sampling, sequence-likelihood objectives, and an active sampling strategy. We conduct extensive experiments across diverse benchmarks and show that, on average, our method significantly outperforms existing online finetuning approaches such as ODT and ODT+TD3. This opens a new direction for advancing the online finetuning of Decision Transformers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel online finetuning method for Decision Transformer (DT) that uses only GRPO gradients for update. 
To adapt GRPO to DT, the authors propose several modifications to GRPO: 1) update based on intended Returns-To-Go (RTG) to 
avoid instability in importance sampling caused by RTG being off-policy; 2) update GRPO based on subtrajectories. To calculate
advantages within a group, several reset points are sampled along the trajectory based on the action variance (if reset is not possible, 
a TD3-based critic is used for advantage estimation instead); 3) the importance ratio term is calculated sequence-wise to ensuring consistency between the objective and the advantage signal. On several environments, the proposed method outperforms baselines such 
as ODT, IQL and TD3+ODT.

### Strengths
1. The paper is well-written and easy to follow. The motivation is clearly stated in the first page of the paper, followed by clearly listed contribution and paper organization section which helps a lot for readers to grasp the general idea. There are also summaries that helps understanding for sections such as Sec. 3.2 and Sec. 4; overall, the paper possesses a good layout. The pseudocode also helps for understanding the paper.

2. The proposed idea is well-motivated and intuitive. I particularly like the idea of sampling reset points with respect to variance, which reflects the insight of "crucial tokens" [1] in today's RL for LLM community. The sequence-level importance ratio also reflects recent improvements on GRPO, such as GSPO [2]. 

3. The empirical results is solid; the proposed method is tested on many environments, including very challenging ones such as Adroit. The detailed hyperparameter listed in Tab. 5 and 6 enhances the reproducibility of the paper. It is exciting that the proposed method is also applicable to alternative architectures such as Reinformer as suggested in Fig. 4.

4. I appreciate that the authors provide a fallback for environments that are unable to reset, which addresses a large concern in the RL community.

**References**

[1] S. Wang et al. Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning. ArXiv: 2506.01939, 2025. 

[2] C. Zheng et al. Group Sequence Policy Optimization. ArXiv: 2507.18071, 2025.

### Weaknesses
As shown in Fig. 3 and Tab. 6, In order to make the algorithm to work well on many MuJoCo environments, the $L_{\text{eval}}$ needs to be very long (400 steps), which significantly slows down the training process. The method could also potentially struggle with sparse reward / goal-based scenario where the reward is rarely gained in the middle of the episode.

**Minor Weaknesses**

There are several typos throughout the paper. To list a few:

1. The full name of GRPO and PPO are not introduced in the first use of abbreviation;

2. line 50, graidents -> gradients;

3. Fig. 1 b), And -> and; the caption misses a period.

4. Tab. 5, the caption misses a period.

5. line 84 misses a period.

6. Fig. 3 b), the legend should be $L_{\text{eval}}$ instead of $L_{\text{traj}}$.

### Questions
I have several questions:

1. While I agree that "prioritizing supervised learning" is a mindset brought by decision transformer itself that worths breaking, there is also a possible middle ground before "prioritizing supervised learning" and "pure RL": RL-led training (i.e. much smaller supervised learning coefficient for ODT+TD3, or adding a small supervised coefficient for the proposed GRPO finetuning). In fact, the unification of post-training stages (RL+SFT) is also adopted by many recent LLM papers [1, 2, 3]. I think this is also a potential baseline worth trying.

2. In line 370, the authors mention "Atari" environment. Why is Atari results not appearing in the paper? (There is only one appearance of "Atari" in the paper besides author names in the reference list.)

3. Why does GRPO and QGRPO have very different initial performance before training in Fig. 3 c)?

4. Could the authors specify line 317-318, "using $\lambda$-returns rather than discounted Monte Carlo returns"? Does this mean PPO uses TD($\lambda$)? If so, what is the value of $\lambda$ (which is not shown in Tab. 5)? This $\lambda$ seems to confllict with the entropy term coefficient $\lambda$ in Eq. 3.

I am open to increase my score if the concerns raised in the question and weakness section can be well-addressed.

**References**

[1] X. Lv et al. Towards a Unified View of Large Language Model Post-Training. ArXiv: 2509.04419, 2025.

[2] J. Yan et al. LUFFY: Learning to Reason Under Off‑Policy Guidance. In NeurIPS, 2025.

[3] L. Ma et al. Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions. ArXiv: 2506.07527, 2025.

### Soundness
3

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
3

### Summary
This paper addresses a central challenge in extending Decision Transformers (DT) to online reinforcement learning settings: how to perform online finetuning with pure RL gradients rather than supervised losses. The authors observe that existing approaches (e.g., ODT, ODT + TD3) rely heavily on supervised objectives and that the use of hindsight return relabeling breaks on-policy importance sampling. They propose a novel adaptation of the GRPO algorithm to Decision Transformers for critic-free finetuning. The method introduces four key components: (i) sub-trajectory optimization to reduce variance, (ii) consistent state resetting for grouped rollouts, (iii) sequence-level importance ratios to align with advantage definitions, and (iv) an active selection mechanism that focuses updates on high-uncertainty states. Extensive experiments on D4RL benchmarks show that this method outperforms ODT, ODT+TD3, and IQL across MuJoCo, Adroit, and AntMaze tasks, especially when offline data is of low quality.

### Strengths
1. Novel Contribution: The proposed GRPO bridges policy gradient methods with ranking-based loss, enabling DT finetuning without requiring a Q-function or supervised pretraining loss modification.
2. Critic-Free Optimization: By using actual rollout returns and relative advantage estimation, the method avoids instability and estimation bias common in value-based methods.
3. High Compatibility: GRPO can be applied to pretrained DTs with minimal architecture changes, offering plug-and-play flexibility.
4. Action Variance-Based Reset Sampling: This active sampling strategy focuses optimization on uncertain states, potentially improving learning efficiency.
5. Extensive Evaluation: Experiments span multiple environments and dataset qualities (medium, expert, random), with results showing strong average performance and robustness, especially in low-quality offline settings.

### Weaknesses
1. Motivational concerns: The paper presents critic-free finetuning as its main contribution, citing instability and complexity of value-based methods. However, this motivation appears incompletely grounded: 1) No theoretical or empirical evidence is provided to substantiate that critics (e.g., in TD3) cause consistent instability. 2) Recent DT-based methods have successfully employed critics without major issues. 3) The GRPO approach incurs significantly higher sample cost, shifting rather than solving the valuation problem. 4) In environments where resets are not possible, GRPO resorts to using value functions, contradicting the motivation.
Therefore, while the method is interesting, its justification for replacing value-based learning is not sufficiently convincing.

2. Strong Reset Assumption: The method assumes the ability to reset environments to arbitrary past states, which is not feasible in many real-world applications or simulators.
3. Disaster Risk from High-Variance States: Resetting to states with high action variance can lead to rollout trajectories with consistently poor rewards. This may cause catastrophic forgetting, as GRPO updates policy based on the best among poor trajectories.
4. Loss of Temporal Coherence: Using short sub-trajectories breaks the global sequence modelling that DT is known for, potentially harming long-horizon decision-making performance.
5. Relative Advantage Lacks Global Signal: The group-based normalized advantage only provides relative ranking within a local batch, and may reinforce suboptimal behaviours if the entire group performs poorly.
6. High Sample Complexity: GRPO requires a large number of rollouts per iteration (G × K × G), which leads to high environment interaction costs compared to alternatives like TD3.

### Questions
1. Can you clarify what specific failure modes or inefficiencies of critics your method addresses that are not already mitigated in prior work?
2. How do you ensure that the selected reset points actually lead to meaningful learning rather than reinforcing poor behaviours or inducing catastrophic forgetting?
3. Is there a risk that all trajectories in a group are poor, and yet one is still “reinforced” due to its relative return?
4. Have you observed any degradation in performance on tasks that require long-horizon credit assignment?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the online fine-tuning of Decision Transformer (DT). It first points out that the "hindsight return relabeling" mechanism in traditional ODT leads to inconsistency between the RTG in online interaction and training phases, undermining the optimization foundation of importance sampling. 
Then, it proposes the core solution: retaining the original target RTG of online trajectories to eliminate this relabeling mechanism. 
On this basis, the paper adapts the GRPO to the online fine-tuning scenario of DT, and puts forward four key modifications: sub-trajectory sampling, environment resetting, sequence-level importance ratio, and active selection.

### Strengths
The analysis of hindsight return relabeling is remarkably clear and thorough, with notably effective results from the associated ablation studies.The paper features a well-structured framework and high readability.

### Weaknesses
1. The motivation of this paper requires further elaboration. The adoption of pure RL gradients in LLMs stems from the extremely high memory overhead and training costs associated with value functions. However, this issue does not exist in offline reinforcement learning (offline RL).
2. The exploration of online fine-tuning capabilities lacks investigation into concatenation ability, particularly in environments such as AntMaze.
3. The performance of certain baselines is significantly lower—for instance, the results of IQL on AntMaze UD-v2 and P-C-v1.

### Questions
1. Is the explanation for Figure 1 missing? What do subfigures (c), (d), and (e) represent respectively?
2. What is the unit of the x-axis in Figure 2? And why does IQL start from scratch in the Hopper-Medium-v2 and Hopper-Medium-Replay-v2 environments?

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
4

### Summary
This paper proposes an online decision transformer method based purely on policy gradients. The key techniques include optimization on sub-trajectories, providing consistent states, sequence-level importance ratio, and active selection. Experiments on locomotion and manipulation tasks validate the effectiveness of the proposed method.

### Strengths
- Improving the online decision transformer is, in my view, an interesting and important direction, and this paper focuses on that topic.

### Weaknesses
- Requiring multiple environment resets at each policy update step is clearly a major drawback. In most real-world or complex simulation environments, it’s almost impossible to reset the system to an arbitrary state, which makes the proposed method impractical in such settings.
- I believe the technique of optimization on sub-trajectories is potentially problematic. GRPO removes the critic mainly because the rewards are extremely sparse in outcome-reward setting, which makes value estimation unreliable. As a result, it assigns the same advantage to all tokens. However, in your tasks the rewards are dense, so giving all sub-trajectories the same advantage wastes valuable reward information. I’m not sure what the benefit of using GRPO is in a dense-reward setting like this.
- In addition, the technical details of optimization on sub-trajectories are unclear. What exactly does "taking mean actions for another $L_{eval}$ steps" mean? Why do you distinguish between $L_{traj}$ and $L_{eval}$? Why not all sample on-policy?
- The baselines should include *online off-policy* algorithms such as SAC, instead of only IQL, which mainly belongs to the *offline* RL domain.

I think this paper still needs substantial clarification and is clearly not ready for acceptance at its current stage.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
