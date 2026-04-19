# The Case for Gradual Structured Pruning in Image-based Deep Reinforcement Learning

- Decision: Reject
- Scores: 5, 6, 3

## Abstract
Scaling neural networks in image-based deep reinforcement learning often fails to improve performance. While it was shown that unstructured pruning of scaled networks can unlock performance gains, we find that refining the architecture of the scaled network yields even greater improvements. However, scaled networks in deep reinforcement learning present a practical challenge: the increased computational demands can hinder deployment on embedded devices, as commonly encountered in robotics applications. To address this, we propose a novel gradual group-structured pruning framework that allows performance gains through scaling while maintaining computational efficiency. Our method preserves the network's functional integrity of inter-layer dependencies in groups, such as residual connections, while seamlessly integrating with standard deep reinforcement learning algorithms. Experiments with PPO and DQN show that our approach sustains performance while significantly reducing inference time, making it the preferred approach for resource-limited deployment.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Scaling neural networks has generally been ineffective for deep reinforcement learning, as larger networks do not necessarily improve performance. Recent work by Obando-Ceron et al. (2024) suggests that pruning a large network could overcome this limitation. By initially increasing the network size and applying pruning during training, they achieved performance gains beyond those possible with a standard dense network. However, their unstructured sparsity approach often fails to generalize well outside the original setting. This paper introduces an improved CNN architecture, called Impoola, along with a structured pruning algorithm that outperforms unstructured pruning in both generalization and computational efficiency.

### Strengths
- Exploring scaling laws for deep RL is an important challenge.
- New CNN architecture and structured pruning algorithm are proposed.
- Outperform both unstructured and naive structured pruning baselines.

### Weaknesses
**Clarification of objectives and contribution**

The primary goal of this paper—whether to build upon Obando-Ceron et al.'s (2024) approach for finding scalable architectures or to develop a structured pruning method specifically for deep RL—is somewhat unclear. If the objective is scaling, adding experiments that compare the proposed method with the original dense network and unstructured pruning across different scenarios would strengthen its claims. On the other hand, if structured pruning is the focus, revisiting the introduction and motivation to reflect this intent could help clarify the paper’s contributions in this area.

---
**Need for broader benchmark comparisons**

If this paper aims to address scaling issues in deep RL, it would benefit from broader comparisons with relevant methods, such as those presented in [1] and [2]. Including at least one of these as a baseline would provide a more comprehensive context for the paper’s approach. If these methods are excluded, providing a brief rationale would clarify the baseline selection choices for readers.

---
**Potential for alternative focus on structured pruning**

An alternative approach for the paper could be to frame it as a structured pruning study specifically for deep RL, rather than as an extension of scaling work. The proposed Impoola architecture appears specialized, so comparing it with existing structured pruning methods for deep RL, such as [3], would reinforce its relevance in this area. Additionally, if the technique is intended to be broadly applicable to CNNs, testing it on traditional CNN tasks (e.g., video classification) would better demonstrate its versatility. Including comparisons with CNN-focused pruning techniques, such as [4], would further highlight the method’s adaptability across different applications.



[1] Obando-Ceron et al. Mixtures of Experts Unlock Parameter Scaling for Deep RL. ICML 2024.\
[2] Sokar et al. Don't Flatten, Tokenize! Unlocking the Key to SoftMoE's Efficacy in Deep RL. arXiv 2024.\
[3] Su et al. Compressing Deep Reinforcement Learning Networks with a Dynamic Structured Pruning Method for Autonomous Driving. arXiv 2024.\
[4] He & Xiao. Structured Pruning for Deep Convolutional Neural Networks: A Survey. TPAMI 2023.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a gradual structured pruning method for deep reinforcement learning. The authors firstly analyzes the drawbacks of the unstructured pruning, then introduces a improved impoola-CNN model to replace the previous impala-CNN model, and a gradual structured pruning for image-based deep reinforcement learning tasks. The experimental results show that the proposed method has compariable perfermances comparing with unstrutured pruning mehtod, and it has good robustness when intruducing nosies. Moreover, the proposed pruning method results in less latency times comparing with original dense model and unstructured pruned model.

### Strengths
1. The gradual structured pruning method is well-defined, and the experimental results show that is has good computation efficiency

### Weaknesses
1. The experimental results did not include the comparsion of impala-CNN and impoola-CNN, thus the advantages of the new network cannot be proved. The authors may consider to provide a direct comparison of performance between Impala-CNN and Impoola-CNN under the same pruning conditions across multiple environments, as well as the un-pruned editions. This would give clearer evidence for the claimed advantages of the new architecture.

2. The advantages of the scoring function are not proved. The reason that use ||w||_1 should be carefully analyzed. For instance, the authors may consider to compare L1-norm, L2-norm or other regularization methods to show the advantages of the proposed scoring function.

### Questions
1. Is their some comparsion between impala-CNN and impoola-CNN when both of them are processed by using gradual structured pruning? The authors may use the proposed pruning mehtod to both impala-CNN and impoola-CNN under different configurations to show the improvements introduced by impoola-CNN. 

2. Is their any experimental results to compare the performance of using different scoring functions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces a group-structured pruning framework for image-based deep reinforcement learning (DRL), specifically designed to maintain performance while reducing computational costs. The authors first demonstrate that the benefits of unstructured pruning diminish when making architectural improvements to neural networks. They then propose a structured pruning approach that removes entire network structures (channels/neurons) while accounting for inter-layer dependencies. The method is extensively evaluated on the Procgen Benchmark using PPO and DQN agents, showing comparable performance to unstructured pruning while achieving significant reductions in inference time.

### Strengths
1. The overall writing is clear.
2. Efficiency in the domain of RL is interesting.

### Weaknesses
1. The motivation is not clear. Although efficiency in the domain of RL is interesting, the author should include the difference between "pruning in image classification" and "pruning in reinforcement learning." This motivation should serve the algorithm.

2. There are some other references [a] to do gradual pruning. What is the difference? Why not cite this?

3. Some arguments are wrong. For example, in line 15, the authors claim, "unstructured pruning merely zeroes out individual weights, the resulting networks usually retain high computational demands despite sparsity". This is not correct. For example, Thinet [b] can achieve realistic acceleration. There are so many other filter pruning methods to achieve practical acceleration.

4. The performance is not good. In table 1, if the up arrow means "higher is better", the proposed method is not as good as others.

[a] H. Wang, C. Qin, Y. Zhang, and Y. Fu, “Neural pruning via growing regularization,” in Proc. Int. Conf. Learn. Represent., 2022
[b] https://github.com/Roll920/ThiNet

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
