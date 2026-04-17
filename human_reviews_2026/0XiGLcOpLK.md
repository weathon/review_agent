# FitLight: Federated Imitation Learning for Plug-and-Play Autonomous Traffic Signal Control

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Although Reinforcement Learning (RL)-based Traffic Signal Control (TSC) methods have been extensively studied, their practical applications still raise some serious issues such as high learning cost and poor generalizability. This is because the ``trial-and-error'' training style makes RL agents extremely dependent on the specific traffic environment, which also requires a long convergence time. To address these issues, we propose a novel Federated Imitation Learning (FIL)-based framework for multi-intersection TSC, named FitLight, which allows RL agents to plug-and-play for any traffic environment without additional pre-training cost. Unlike existing imitation learning approaches that rely on pre-training RL agents with demonstrations, FitLight allows real-time imitation learning and seamless transition to reinforcement learning. Due to our proposed knowledge-sharing mechanism and novel hybrid pressure-based agent design, RL agents can quickly find a best control policy with only a few episodes. Moreover, for resource-constrained TSC scenarios, FitLight supports model pruning and heterogeneous model aggregation, such that RL agents can work on a micro-controller with merely 16{\it KB} RAM and 32{\it KB} ROM. Extensive experiments demonstrate that, compared to state-of-the-art methods, FitLight not only provides a superior starting point but also converges to a better final solution on both real-world and synthetic datasets, even under extreme resource limitations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a federated learning method to improve the generalizability and convergence speed of traffic signal control. The author introduce an imitation learning mechanism combined with a hybrid pressure-based agent design, enabling real-time imitation learning and smooth transitions to reinforcement learning, allowing RL agents to quickly achieve a high-quality solution in the first episode.

### Strengths
1. This paper proposed a federated imitation learning method to improve the RL convergence speed.

2. This paper enables to operate in TSC scenarios with extremely limited resources., which reduce the deployment costs.

### Weaknesses
1. In real-world deployment, there may has a lot of noise in perception. How to solve the noise environment? (FuzzyLight, RobustLight)

2. The author should add more newly TSC methods to demonstrate the performance.

3. Traditional method like MP and Advanced-MP also achieved quickly convergence speed. Why not just use simple traditional method?

4. More federated machine learning methods should be compared. I think the author just uses average aggregation, and more methods like FedProx should be tested as ablation experiments.

5. The code, network and hyperparameters are not reported, which reduces the availability.

### Questions
See weakness.

### Soundness
3

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
4

### Summary
This paper presents FitLight, a federated imitation learning framework for traffic signal control, aiming to address issues like high learning cost and poor generalizability in RL-based methods. The motivation is clear, and the potential for deployment on resource-constrained microcontrollers (e.g., 16KB RAM, 32KB ROM) is a notable advantage. However, several significant concerns prevent me from recommending acceptance in its current form.

### Strengths
1.	The motivation for addressing the high learning cost and poor generalizability of RL-based TSC methods is well-articulated.
2.	The claimed computational advantage of operating on a microcontroller with merely 16KB RAM and 32KB ROM is compelling and relevant for practical deployments.
3.	The core ideas – federated imitation learning, a plug-and-play capability for RL agents, real-time imitation, and a seamless transition to RL – are clear and potentially innovative.

### Weaknesses
1.	The paper does not mention the availability of code for reproducibility. Given the complexity of the proposed framework (combining federated learning, imitation learning, and RL), the inability to verify the implementation and results is a critical flaw.
2.	The description of the FitLight framework, presumably in Figure 1 and its accompanying text, is stated to be overly simplistic. A more detailed explanation and a more informative diagram are necessary for readers to understand the system's architecture and data flow.
3.	The experimental evaluation may not include comparisons with the most recent state-of-the-art TSC methods, such as TransformerLight. This omission raises questions about the true performance advantage of FitLight against contemporary benchmarks.

### Questions
1.	The framework involves communication between a cloud server and individual intersections. Was the potential impact of communication latency on the real-time control performance considered or evaluated? This is a crucial practical factor for real-world deployment.
2.	 Could the authors confirm and justify the selection of baseline methods? It is essential to compare against the latest advanced methods to robustly validate the claimed superiority.
3.	 The experiments were conducted on a high-performance computing platform (Intel Core i9-12900K, 128GB RAM, NVIDIA RTX 3090). However, a key claim is the ability to run on extremely resource-constrained microcontrollers (16KB RAM, 32KB ROM). Could the authors clarify this discrepancy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes FitLight, a learning strategy for traffic signal control. FitLight leverages federated learning and pruning techniques to enable information sharing among agents. Furthermore, FitLight integrates imitation learning into reinforcement learning to accelerate model convergence.

### Strengths
1.The paper consider the feasibility of real-world deployment, particularly the model’s performance under resource-constrained conditions.
2.The proposed modules are well-defined abd expressed clear.

### Weaknesses
1.The paper introduces imitation learning to reduce the high training cost of RL agents. Using imitation learning to accelerate RL cold starts or alleviate sparse reward signals in early training is a common practice in practical RL applications. The paper’s application of this approach to the traffic signal control (TSC) problem primarily addresses the difficulty of RL initialization, thus offering only limited novelty in this aspect.
2.Employing a neural network to approximate an expert’s policy distribution is a standard paradigm. Imitation learning and reinforcement learning can be implemented as two separate stages. Since imitation learning does not require real-world interaction, it can be conducted offline on large-scale data to obtain a well-performing pretrained TSC model. The rationale for combining imitation learning and reinforcement learning sequentially within the training process is not clearly justified.
3.Another contribution is the enhancement of model generalization. However, the evaluation of generalization in the experiments is insufficient. For instance, can a model trained on certain intersections be transferred to new intersections at zero or low cost? How does this transfer cost compare to that of retraining the model from scratch, which is already relatively low? Moreover, even the largest simulated network contains only 4×4 intersections, leaving the model’s performance on larger networks unclear.
4.In RL, the ultimate goal is typically defined through a reward function that aligns the model’s behavior with the target objective. If average travel time is used as the performance metric, why is it not directly adopted as the evaluation criterion? In Section 4.3, regarding the evaluation of the reward function, does this imply that using average travel time as the reward function would yield similar results?

### Questions
1.The paper introduces imitation learning to reduce the high training cost of RL agents. Using imitation learning to accelerate RL cold starts or alleviate sparse reward signals in early training is a common practice in practical RL applications. The paper’s application of this approach to the traffic signal control (TSC) problem primarily addresses the difficulty of RL initialization, thus offering only limited novelty in this aspect.
2.Employing a neural network to approximate an expert’s policy distribution is a standard paradigm. Imitation learning and reinforcement learning can be implemented as two separate stages. Since imitation learning does not require real-world interaction, it can be conducted offline on large-scale data to obtain a well-performing pretrained TSC model. The rationale for combining imitation learning and reinforcement learning sequentially within the training process is not clearly justified.
3.Another contribution is the enhancement of model generalization. However, the evaluation of generalization in the experiments is insufficient. For instance, can a model trained on certain intersections be transferred to new intersections at zero or low cost? How does this transfer cost compare to that of retraining the model from scratch, which is already relatively low? Moreover, even the largest simulated network contains only 4×4 intersections, leaving the model’s performance on larger networks unclear.
4.In RL, the ultimate goal is typically defined through a reward function that aligns the model’s behavior with the target objective. If average travel time is used as the performance metric, why is it not directly adopted as the evaluation criterion? In Section 4.3, regarding the evaluation of the reward function, does this imply that using average travel time as the reward function would yield similar results?

### Soundness
2

### Presentation
3

### Contribution
2
