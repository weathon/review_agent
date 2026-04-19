# Imitation Bootstrapped Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 6, 5

## Abstract
Despite the considerable potential of reinforcement learning (RL), robotics control tasks predominantly rely on imitation learning (IL) owing to its better sample efficiency.
However, given the high cost of collecting extensive demonstrations, RL is still appealing if it can utilize limited imitation data for efficient autonomous self-improvement.
Existing RL methods that utilize demonstrations either initialize the replay buffer with demonstrations and oversample them during RL training, which does not benefit from the generalization potential of modern IL methods, or pretrain the RL policy with IL on the demonstrations, which requires additional mechanisms to prevent catastrophic forgetting during RL fine-tuning.
We propose _imitation bootstrapped reinforcement learning_ (IBRL), a novel framework that first trains an IL policy on a limited number of demonstrations and then uses it to propose alternative actions for both online exploration and target value bootstrapping.
IBRL achieves SoTA performance and sample efficiency on 7 challenging sparse reward continuous control tasks in simulation while learning directly from pixels. 
As a highlight of our method, IBRL achieves $\mathbf{6.4\times}$ higher success rate than RLPD, a strong method that combines the idea of oversampling demonstrations with modern RL improvements, under the budget of **10** demos and **100K** interactions in the challenging PickPlaceCan task in the Robomimic benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method named imitation bootstrapped reinforcement learning (IBRL) that uses a stand alone imitation learning model to support the learning process of a separate reinforcement learning model. The main motivation is that traditional imitation and reinforcement learning combined approach usually wants to get the best of the two methods, but often has limited benefits from the generalization of IL beyond the given demonstration data, and needs to apply additional regularizations on RL to ensure the pre-trained IL initialization does not get washed by RL fine-tune. One of the reason why these methods struggle is that the IL and RL share the same architecture, making it hard for IL to generalize to both demo and online generated data. 

To mitigate these issues, the approach proposed a way to separate the training of IL and RL. In particular, the IL is trained first on the demonstration data, and then followed by two phases of RL training. The first one is the IL and RL will both propose actions during online interaction phase, and the policy select the action by either policy that has a higher Q-value. The second is that during the optimization phase of RL, the target Q-value is the max among the actions proposed by IL and RL policies. Demonstration data is also used within the RL replay buffer to accelerate learning. The authors also introduce several practical techniques to further improve the performance of IBRL, including using multiple Q-networks, dropout in Q-networks, data augmentation via random shifts, and regularizing the policy network with dropout.  

This method provides the flexibility that the IL and RL model can take different architectures, and IL model also does not get confused by the noise online interaction data. The authors claimed they performed experiment to show that taking different architecture designs for IL and RL brings some benefits in terms of performance. 

Experiments are performed on 7 sparse reward continuous control tasks, with increasing difficulties. Baselines such as RLPD and MoDem are used for comparisons. The author showed that the proposed IBRL method has better sample efficiency than the baseline methods on all 7 tasks. Additional ablation study finds that using IL for bootstrapping target in training phase is important for some environments and different architecture designs for IBRL and BC have huge difference in terms of performance.

### Strengths
1. Improved sample efficiency: IBRL is designed to learn from a limited number of expert demonstrations and achieve better performance than using BC only with the same amount of data. The authors demonstrated that IBRL outperforms existing IL+RL methods such as RLPD+ and MoDem on a range of challenging control tasks, while requiring fewer interactions with the environment.

2. Flexible network architecture selection. The algorithm design separate the training of IL and RL, and the ablation study shows that using different network designs can benefit both BC and RL, instead of forcing them the share the same network. 

3. Good collection of experiments and ablation studies to support the claims. The authors performed studies to support their claim that the separate IL/RL training is beneficial, and that using IL to bootstrap the RL training target is beneficial. Ablation studies corresponding to network designs and regularization techniques also validated the design choices.

### Weaknesses
Some related works are not mentioned, such as "SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards" by Siddharth Reddy et al. Though this work is relatively old, but it does present a relevant approach. 

The separation of IL and RL first looks novel, but the notion that different algorithms (IL vs RL) has different performances when using different architectures is a well-known thing. Therefore, it is not surprising to see the ablation study results. The regularizations used here are all from existing work, though I appreciated that the authors did experiments to show their benefits. 

One confusing thing is in Figure 7, what is the point of using 10 demos to show the difference if you already have 200 demos results? I'm not sure what the authors were trying to show. On the other hand, if the cost of collecting more demos are not that high, why does it make sense to only provide one or slightly more demonstrations, which can definitely be very noisy, and then rely on RL to solve the entire problem? 

The proposed IBRL method does improve on some environments, such as lift, which seems to be considered as an easier task, while the improvement on the harder task is limited. Therefore, it gives the impression that the method may overfit the easier task while its performance improvements on more complicated tasks remains unclear. On the comparison with MoDem, since the network architectures are different between IBRL and MoDem, it is hard to evaluate whether the improvement in performance comes from network design or the new method.

Since IL part is only trained on demonstration data, and is frozen at the RL training stage, it becomes unstable when the environment changes dramatically due to distributional shift. At that time, the algorithm is solely relying on RL to improve and the supervision from IL becomes less meaningful. This put this algorithm at a challenge when testing this algorithm against testing environments that are different from training tasks or testing tasks that have different distributions from training tasks. It's unclear from the paper, how different is the testing task from the training task. If it is very similar environments, there is a risk of overfitting to the environments.

### Questions
How does the method compare with SQIL?
How does this approach perform on tasks where testing environments are different from training environments?
How much cost is involved when collecting demonstrations? Is it more cost to collect demonstrations or is it more cost to do RL training? If we provide more demonstrations, would the method still performs better than other IL+RL methods? Since in many real world robotics applications, in general we may have some nontrivial demonstrations, and the diversity of demonstrations affect the training performance a lot. How does varying the demonstrations quantities affect the IBRL performance?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors present a reinforcement learning algorithm, Imitation Bootstrapped Reinforcement Learning (IBRL), which uses an imitation learning policy to provide alternative actions for exploration in the environment and for value function iteration. Empirically, IBRL outperforms competitive benchmarks in challenging continuous control tasks.

### Strengths
1.The idea of maintaining a frozen imitation learning agent prevents the knowledge obtained from expert demonstrations from being washed out, and this approach has proven successful in practice.

2.The dropout policies and ViT-Based Q-Network, accompanied by ablation studies, provide useful observations for practical implementations.

### Weaknesses
The lack of theoretical analysis diminishes the paper's contribution. It would be beneficial to decompose the reinforcement learning problem into imitation learning and reinforcement learning components, aiming for a 'best of both worlds' guarantee, as seen in reference [1].

Using different policies for imitation and reinforcement learning might not be sufficiently motivating. This introduces extra overhead and complicates hyperparameter tuning.

The fixed imitation learning policy doesn't leverage additional observations from reinforcement learning, where more robust frameworks, like generative adversarial imitation learning, could be naturally incorporated.

The bootstrap proposal for RL is not novel and might not be the best choice to emphasize the innovation.

[1] Cheng C A, Yan X, Wagener N, et al. Fast policy learning through imitation and reinforcement[J]. arXiv preprint arXiv:1805.10413, 2018.

### Questions
Same as listed in the weakness above. Additionally, please provide full result plots for all experiments in the appendix to further justify the findings.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a sample efficient reinforcement learning framework where an imitation learning policy trained on demonstrations proposes an alternative action during online exploration and target value bootstrapping. The key difference from past work on this literature is the use of a standalone IL policy throughout the entire training process. The modular design of the proposed framework also enables different network architectures for IL and RL components. In addition, dropout in policy networks and ViT-based Q-network are technical improvements that enhance performance in sparse reward control tasks. Experiment results show that this framework outperforms previous baselines in 7 robot control tasks in simulation and that the flexible architecture benefits IL and RL policies by choosing different encoders, respectively.

### Strengths
- The paper is well organized and written.
- The authors provide a reasonable and flexible algorithm that integrates IL and RL policies while preserving their advantages. To my knowledge, this framework seems new and would be interesting to the community.
- The performance of this algorithm is promising in the experimental tasks chosen in the paper. Ablation results demonstrate the importance of each component within the framework.

### Weaknesses
My primary concern and questions lie in the experiments.
- This paper investigates 7 robot control tasks and employs two test environments for two baseline methods. However, the reasons for this setup are not clearly provided. I am curious about why these tasks are selected, why comparisons are not made with the two baselines on all tasks, and how IBRL performs on other various robot control tasks.
- One of the baselines used is RLPD+, which is an improved implementation of the RLPD algorithm by the authors. A further comparison with the original RLPD would make the experimental results more persuasive and demonstrate the effectiveness of the two techniques in RLPD.
- The authors provide videos of sample rollouts of one task, showing that trained IBRL can finish the task faster than humans. Presenting more demos would give a more intuitive understanding of the difficulty of these tasks and the performance of the proposed framework.
- One benefit of the proposed approach is that "the IL policy may generalize surprisingly well beyond the training data." Although an explanation is provided, I am still confused about the supporting experimental results and how the result is connected to the claim.

### Questions
Please refer to the questions raised in the Weakness part.

### Soundness
2 fair

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces Imitation Bootstrapped Reinforcement Learning (IBRL), an innovative approach that combines Imitation Learning (IL) with Reinforcement Learning (RL) to enhance sample efficiency and performance in RL tasks. IBRL utilizes expert demonstrations to train an IL policy, which then informs the action selection and target value bootstrapping in RL. The framework demonstrates good results on several continuous control tasks, significantly outperforming the selected baselines.

### Strengths
1. The paper is well-organized and written with a clear structure, making it accessible to readers with a background in machine learning and RL. The explanations of the IBRL framework, the role of expert demonstrations, and the architecture of the Q-network are coherent and logical. Figures and algorithmic descriptions aid in understanding the proposed method.
2. The significance of this work lies in its potential impact on the field of RL, particularly in domains where sample efficiency is critical, such as robotics or any environment where interaction is costly or risky.

### Weaknesses
1. While the paper demonstrates the effectiveness of IBRL, it will benefit from a broader range of baselines, including recent advancements in both IL and RL. At least the baseline algorithms that pretrain policies via behavior cloning should be implemented for the Robomimic benchmark.   Additionally, similar enhancements as applied in RLPD+ should be adopted for the MoDem benchmark to ensure consistency and a fair comparison.

2. The paper would benefit from a more detailed theoretical analysis that elucidates the necessity and efficacy of the proposed method.  The solution indeed generally makes sense but as an academic paper, we need more analysis or religious modeling to show its advantage from the methodology perspective. Experiments in Figure 6 also seem to indicate that the main benefit comes from the dropout mechanism and DrQ network instead of the actor selection and bootstrapping pipeline.

### Questions
What if the Q value is underestimated under some actions? In the current pipeline, it seems that these actions can never be selected and thus the better action can never be learned by RL, since the max-Q mechanism will filter these actions and replace them with the actions in IL policy.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
