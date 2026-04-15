# A Dual-Agent Adversarial Framework for Generalizable Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 6, 5

## Abstract
Recently, empowered with the powerful capabilities of neural networks, reinforcement learning (RL) has successfully tackled numerous challenging tasks. However, while these models demonstrate enhanced decision-making abilities, they are increasingly prone to overfitting. For instance, a trained RL model often fails to generalize to even minor variations of the same task, such as a change in background color or other minor semantic differences. To address this issue, we propose a dual-agent adversarial policy learning framework, which allows agents to spontaneously learn the underlying semantics without introducing any human prior knowledge. Specifically, our framework involves a game process between two agents: each agent seeks to maximize the impact of perturbing on the opponent's policy by producing representation differences for the same state, while maintaining its own stability against such perturbations. This interaction encourages agents to learn generalizable policies, capable of handling irrelevant features from the high-dimensional observations. Extensive experimental results on the Procgen benchmark demonstrate that the adversarial process significantly improves the generalization performance of both agents, while also being applied to various RL algorithms, e.g., Proximal Policy Optimization (PPO). With the adversarial framework, the RL agent outperforms the baseline methods by a significant margin, especially in hard-level tasks, marking a significant step forward in the generalization capabilities of deep reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a dual-agent adversarial framework aimed at improving the generalization capabilities of reinforcement learning (RL) models, which often struggle with overfitting and fail to adapt to minor variations in tasks. The proposed framework facilitates a game process between two agents that learn to perturb each other’s policies while maintaining their own stability, enabling them to focus on relevant features in high-dimensional observations. Extensive experiments on the Procgen benchmark demonstrate that this adversarial approach significantly enhances the agents’ performance, especially in challenging environments, outperforming traditional RL algorithms like Proximal Policy Optimization (PPO). Additionally, the authors theoretically prove that reducing an agent’s robustness to irrelevant features can improve its generalization performance. Overall, the study marks a significant advancement in addressing generalization challenges in deep reinforcement learning.

### Strengths
- The introduction of a dual-agent adversarial framework is an innovative approach that addresses the pressing issue of overfitting and generalization in reinforcement learning, offering a new perspective on how agents can improve adaptability in varying environments.
- The paper provides a strong theoretical foundation, proving that reducing an agent’s robustness to irrelevant features can lead to better generalization, enhancing the depth of the contribution.
- The experiments conducted on the Procgen benchmark show significant performance improvements over existing methods like PPO, demonstrating the effectiveness of the proposed framework in real-world, challenging tasks.
- By focusing on reducing overfitting and enhancing generalization, the paper addresses a critical gap in reinforcement learning research, providing solutions applicable to broader, more complex environments.
- The framework is well-designed to scale across different environments, making it applicable to a wide range of RL tasks with high-dimensional observations.

### Weaknesses
- While the framework performs well on the Procgen benchmark, its applicability to real-world tasks remains untested, leaving questions about how well it generalizes outside controlled environments.
- The dual-agent adversarial framework introduces additional computational complexity, which may pose challenges in terms of scalability and efficiency for resource-constrained systems.
- Although the framework is shown to improve generalization, more detailed ablation studies could have been included to clarify the contribution of individual components, such as the specific impact of the adversarial training mechanism.
- The paper assumes that irrelevant features can be identified and suppressed, but it does not sufficiently address how to detect these features in environments where their classification is unclear or context-dependent.
- The comparison with state-of-the-art methods is somewhat limited, with a stronger focus on performance gains rather than in-depth analysis of differences in behavior between approaches.

### Questions
- How does the adversarial training framework handle environments where the distinction between relevant and irrelevant features is not well-defined or context-dependent?
- What strategies could be employed to reduce the computational overhead introduced by the dual-agent setup, especially in more complex or resource-constrained environments?
- Could the method be extended or adapted to improve generalization in real-world tasks beyond the Procgen benchmark, and what modifications would be necessary to achieve this?
- What impact would the framework have in environments with continuous action spaces or higher-dimensional state representations, where irrelevant features may be harder to isolate?
- How would performance vary in scenarios where adversarial training results in catastrophic forgetting of useful features, and what mechanisms could prevent this?
- Are there any considerations for applying this approach to tasks with dynamic or evolving feature relevance, where the set of relevant features may change over time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The submission proposes an adversarial framework that involves a game process between two agents: each agent seeks to maximize the impact of perturbing the opponent’s policy by producing representation differences for the same state, while maintaining its own stability against such perturbations. The submission conducts experiments in the ProcGen environment with 3 random seeds in 8 different games and provides comparison against DAAC and PPO.

### Strengths
Generalization in deep reinforcement learning is a highly important research direction.

### Weaknesses
The theoretical claims of the submission follow almost immediately from previous work, and do not bring any additional new knowledge. 

Table 1 should include standard deviations. Three random seeds is relatively small to interpret the results reported. The results reported in Table 1 and Figure 5 are contradictory. Table 1 reports that DAAC in climber is 3.299 and PPO + Adv. (Agent 1) is 4.473. However, Figure 5 clearly reports that DAAC performance as the highest. How is this possible?

Why is it only compared to DAAC and original PPO? There are more studies on generalization in deep reinforcement learning. 

In the DAAC paper there is another algorithm called IDAAC that performs better. Why is the algorithm IDAAC not included in the comparison?

I would also recommend checking page 9 and page 10 of the ProcGen section of paper [1]. In particular, the paper [1] states for ProcGen that: 

*“We note that a number of improvements reported in the existing literature are only 50 − 70% likely.”*

Furthermore the paper [1] states:

*“Instead, we recommend using normalization based on the estimated minimum and maximum scores on ProcGen and reporting aggregate metrics based on such score.”*

As it has been reported in [1] and [3], the performance of PPG [2] is also quite high. It might be good to include PPG in the comparison baseline.

ProcGen seems to have 16 tasks. Both of these papers [1,2] test across the 16 games in the ProcGen environment. The submission tests their proposed algorithm in only 8 of them.

[1] Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021.

[2] Phasic Policy Gradient, ICML 2021.

[3] Decoupling Value and Policy for Generalization in Reinforcement Learning, ICML 2021.

More recent techniques report substantially higher scores in the ProcGen environment [1,2].

[1] DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck, ICML 2022.

[2] Explore to Generalize in Zero-Shot RL, NeurIPS 2023.


How adversarial learning is mentioned in the introduction is incorrect. In the introduction it is stated that: 

*“Adversarial framework facilitates the development of agents capable of adapting to new environments by emphasizing the distinction between relevant and irrelevant information.”*

by referring to these studies [1,2,3] as adversarial learning 

[1] State-Adversarial DQN for robust deep reinforcement learning, NeurIPS 2020.

[2] Robust adversarial reinforcement learning, ICML 2017.

[3] Robust Deep Reinforcement Learning through Adversarial Loss, NeurIPS 2021.

However, recent studies demonstrated that in fact adversarially trained policies cannot generalize, and furthermore the generalization skills of standard reinforcement learning training is substantially higher [1]. 

[1] Adversarial Robust Deep Reinforcement Learning Requires Redefining Robustness, AAAI 2023.


Since the submission proposes an adversarial training method it would have been good to test against adversarial examples as well. It does not have to be the most state-of-the-art adversarial attacks, but still it would have been good to include for reference.


Another thing I want to mention is that by employing the proposed adversarial learning framework the number of encoder parameters that needs to be trained is in fact doubled. This brings a new set of questions. Is it really a fair comparison to lower capacity models as previous ones? Would the prior methods perform also well if we simply just increased the parameters in the encoder?

### Questions
Please see above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a dual-agent adversarial framework to improve generalization in reinforcement learning (RL). In this setup, two agents interact adversarially, each attempting to disrupt the other's policy while maintaining stability in its own. This competition drives both agents to develop robust and generalizable strategies. The framework is efficient, adding only one hyperparameter, and shows strong performance improvements in challenging environments, especially when used with standard RL algorithms like PPO. This approach offers a promising solution for enhancing RL generalization without relying on complex data augmentations or human-designed biases.

### Strengths
1. **Solid Theoretical Background**: The approach is backed by strong theory, clearly explaining how it supports RL generalization.
2. **No Human Bias in Addressing Generalization**: The method achieves generalization without relying on human biases, such as hand-designed augmentations.
3. **No Extra Network Parameters**: The framework achieves its goals without adding network parameters, relying on just one hyperparameter for flexibility.
4. **Novel Idea**: The dual-agent adversarial setup is an innovative way to tackle RL generalization.
5. **Strong Performance**: The approach performs well across tested environments, demonstrating robust generalization and effectiveness.

### Weaknesses
1. **Limited Environments and Baselines**: Testing is somewhat limited in environments and baseline comparisons. Adding diverse environments, such as DMC-GB[1], and competitive baselines like PIE-G[2], SVEA[3], and ARPO[4] would provide a more complete comparison and further demonstrate the model's capabilities.

[1] Generalization in reinforcement learning by soft data augmentation., Hansen et al., ICRA 2021

[2] Pre-Trained Image Encoder for Generalizable Visual Reinforcement Learning., Yuan et al., NeurIPS 2022

[3] Stabilizing deep q-learning with convnets and vision transformers under data augmentation., Hansen et al., NeurIPS 2021

[4] Adversarial Style Transfer for Robust Policy Optimization in Deep Reinforcement Learning., Rahman et al., ArXiv 2023

### Questions
Please refer to the Weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel adverserial learning framework that involves a minimax game process between two homogeneous agents to  improve the generalization capability of these agents in RL. This framework integrates with existing RL algorithms such as PPO, leverages no additional human prior knowledge which can lead to poor robustness in generalization and has minimal hyperparameters allowing for effective applicability. The authors additionally derive lower bounds for the training and generalization performance of the agent and show that by minimizing the policy's robustness to irrelevant features, one can improve generalization performance. The authors evaluate their framework in the ProcGen environment, showing gains over algorithms such as PPO and DACC.

### Strengths
- Empirically show a significant improvement over prior work in the ProcGen environment with their adverserial learning framwork
- Provide theoretical insights about how a policy's robustness to irrelevant features improves generalization performance which is a novel contribution that can be generally applied to any algorithm.

### Weaknesses
- Several works such as DRAC and RARL consider a multi-agent/adversarial optimization process in RL. Would be good to include an extensive evaluation of these approaches as baselines and contextualize the novelty of your approach with respect to each baseline.
- The method is primarily evaluated in the ProcGen environment and could benefit from additional empirical evaluation with a larger set of RL benchmarks to further evaluate the efficacy of the approach.
- GANs and other adversarial optimization techniques commonly have issues with mode collapse, vanishing gradients and convergence issues, which all make optimization more difficult. Though this is controlled with the parameter $\alpha$, would be good to consider the tradeoff of the robustness to adversarial threats and the performance of the agent.

### Questions
- One claim is that the method can be widely used with a variety of algorithms. Would you be able to share results on how this approach transfers with different Online RL algorithms in the ProcGen benchmark?
- Would you be able to provide some qualitative analysis of the representations learned by your framework compared with those of PPO and DACC to further validate the claim that robust representations are being learned?
- Could you share results of the sensitivity of $\alpha$ and the selection criterion for it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a dual-agent adversarial policy learning framework to address generalization gaps in reinforcement learning (RL). The authors first derive a lower bound on generalization performance, showing that optimizing this bound corresponds to constrained optimization at each RL step. They then leverage some approximations, leading to the dual-agent adversarial framework proposed in this paper. It employs two identical policy networks that are updated alternately to minimize reliance on irrelevant features through a combined loss function, which includes both the primary task loss and a new adversarial loss that includes both adversarial attacks of the other and robust defences of itself. Experiments on the ProcGen benchmark show that this approach outperforms PPO and DAAC baselines in generalization.

### Strengths
The paper is well-written and easy to follow, with an effective and engaging presentation. The paper begins with a theoretical analysis, systematically deriving the motivation and key designs, and ultimately showing good results.

### Weaknesses
My main concern is the evaluation. Although the performance is promising, the paper lacks in-depth analysis and extensive discussion on several aspects.

Currently, it seems that even without generalization, the proposed method is showing good performance. Thus it is unclear if it is due to better convergence or better generalization capability. We should be careful about this when drawing the conclusion that the proposed method has better generalization performance. And it would be helpful to add some experiments/discussion to compare only the generalization if the two baselines have similar in-distribution performance. Also, one needs to check if the comparison is fair or not. It would be helpful to report the wall clock time and number of gradient steps between the proposed method and the baseline.

Second, the approach is evaluated on only one generalization setup. However, it is unclear how challenging such a generalization setting is. It would be beneficial to conduct further analysis, assessing the degree to which the proposed algorithm enhances generalization across varying levels of difficulty, such as easy, moderate, and challenging generalization cases.

Third, please also consider adding the training complexity compared to baseline methods, as well as a discussion on the impact of hyperparameter $\alpha$.

Additionally, the authors may want to compare and discuss their method relative to other approaches aimed at enhancing RL generalization, such as [1][2]. Given that these methods may also align with the two characteristics described in Section 4.2, further elaboration on the distinctions or similarities would strengthen the paper.

* [1] MaDi: Learning to Mask Distractions for Generalization in Visual Deep Reinforcement Learning
* [2] Policy Rehearsing: Training Generalizable Policies for Reinforcement Learning

In Section 2, the introduction of the concept of MDP state semantics and the subscript $𝑚$ in the MDP notation is not well-motivated or clearly explained. Consider improving clarity by first defining the distribution $𝑝𝑀$ explicitly and then introducing $m$. Furthermore, the limitations and future works should be discussed in the paper.

### Questions
* Will the theorem hold if $𝑀_{train}$ is unbounded? In RL, policies typically interact with the environment on the fly, allowing for the gathering of infinite samples.
* Are the two characteristics discussed in Section 4.2 sufficient to ensure robust generalization performance? If not, what additional considerations could be relevant?
* Are any weights needed in Equation 19? If not, why?
* Would it be beneficial to consider heterogeneous encoders in the proposed method?

### Soundness
3

### Presentation
3

### Contribution
3
