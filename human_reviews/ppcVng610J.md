# Pretraining Decision Transformers with Reward Prediction for In-Context Structured Bandit Learning

- Decision: Reject
- Scores: 3, 3, 6

## Abstract
In this paper, we study the multi-task structured bandit problem where the goal is to learn a near-optimal algorithm that minimizes cumulative regret. The tasks share a common structure and the algorithm exploits the shared structure to minimize the cumulative regret for an unseen but related test task. We use a transformer as a decision-making algorithm to learn this shared structure so as to generalize to the test task. The prior work of pretrained decision transformers like \dpt\ requires access to the optimal action during training which may be hard in several scenarios. Diverging from these works, our learning algorithm does not need the knowledge of optimal action per task during training but predicts a reward vector for each of the actions using only the observed offline data from the diverse training tasks. Finally, during inference time, it selects action using the reward predictions employing various exploration strategies in-context for an unseen test task. We show that our model outperforms other SOTA methods like \dpt, and Algorithmic Distillation (\ad) over a series of experiments on several structured bandit problems (linear, bilinear, latent, non-linear). Interestingly, we show that our algorithm, without the knowledge of the underlying problem structure, can learn a near-optimal policy in-context by leveraging the shared structure across diverse tasks. We further extend the field of pre-trained decision transformers by showing that they can leverage unseen tasks with new actions and still learn the underlying latent structure to derive a near-optimal policy. We validate this over several experiments to show that our proposed solution is very general and has wide applications to potentially emergent online and offline strategies at test time. Finally, we theoretically analyze the performance of our algorithm and obtain generalization bounds in the in-context multi-task learning setting.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper employs a Transformer model for decision-making in multi-armed bandit problems. Rather than generating actions directly, the authors propose training the Transformer to predict the reward for each arm. Actions are then selected greedily, with occasional random exploration, based on the reward estimates. The Transformer is trained on trajectories from a near-optimal expert, such as standard UCB or Thompson sampling algorithms. Results demonstrate that the trained Transformer outperforms traditional learning algorithms across a variety of bandit problems with shared structural features. The authors also present theoretical results quantifying generalization and transfer errors in relation to the training loss.

### Strengths
The paper introduces an alternative pre-training method for simple decision-making environments, such as bandits. In my opinion, this design is well-suited for the bandit context, and the algorithm performs effectively in both synthetic and real environments. The authors also provide ample ablation studies to explore different setups and gain insight into the conditions under which this algorithm performs well. In general, I find no significant issues regarding the originality or clarity of this paper.

### Weaknesses
I think the biggest weakness of the paper is that it fails to provide sufficient understanding of why this algorithm works well—specifically, it lacks insights or supporting evidence on why this design would outperform other algorithms and what key takeaways it offers for theoretical researchers and practitioners. For example, I find that the experimental section and the theory section feel somewhat disconnected. The theoretical results focus on generalization errors and transferability with respect to the pre-training loss. However, this pre-training loss does not directly correlate with the regret, which is the focus of the empirical study. This paper feels like a mixture of observations and results on different aspects of the problem, instead of a systematic effort to understand this pre-training pipeline.

### Questions
1. Could you provide a systematic explanation of why this design benefits decision-making, particularly in the shared-structure bandit setting? After reading the paper, I am unclear about the authors' motivation for studying the bandit problem with a shared structure. In my opinion, there is a lack of insight into why a Transformer-based decision-maker would succeed even in a simple bandit environment.

2. While I understand that the proposed algorithm does not rely on the optimal action, it would be beneficial to include a comparison with the optimal action to assess the performance difference. This addition could help clarify how much of the performance improvement is attributable to the optimal decision in the training data itself. Furthermore, I believe the paper should cite the following related work, which is closely related.

Supervised Pretraining Can Learn In-Context Reinforcement Learning, Jonathan N. Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a new training phase for a pre-trained Transformer model for structured bandit problems. Unlike existing approaches, such as Algorithm Distillation (AD) and Decision Pretrained Transformer (DPT), which predict the next (near-)optimal action using in-context data during training and testing, the proposed framework focuses on predicting the reward vector and selecting the arm based on this prediction.  This paper gives several experiments and a theoretical analysis of the proposed approach.

### Strengths
The paper is well-written. The proposed training framework reduces the dependency on (near-)optimal solutions in training data/tasks, making it more flexible. Additionally, the authors provide a set of experiments and offer a theoretical analysis to understand the framework's effectiveness.

### Weaknesses
[1]The primary advantage claimed for the proposed framework is its ability to function without requiring near-optimal solutions in the training data. However, for bandit problems specifically, this requirement is often not a major limitation. In simulated tasks, optimal arms are readily available, or near-optimal actions can be efficiently obtained through established algorithms like UCB or Thompson Sampling. To convincingly illustrate the advantage, experiments on more complex tasks, where identifying near-optimal actions is genuinely challenging, would strengthen the claim.


[2] The proposed framework appears to be limited to situations with a finite set of actions, which restricts its applicability in problems with a continuous action space. Is there a potential adaptation that could allow the framework to handle continuous action spaces? Additionally, the experiments are conducted with a relatively small action space (20 arms), which raises questions about performance in larger action spaces, such as those with over 100 arms.

[3]Although the framework outperforms other benchmarks in most experiments, the assumption that each arm’s features remain constant across training and testing tasks is atypical for structured bandit problems. In standard settings, arm features are generally assumed to be known and vary between tasks. It is unclear why the authors chose this fixed-feature assumption while not using these features as inputs in the transformer model, which seems to reduce the framework’s applicability. Can this assumption be relaxed to better reflect standard structured bandit settings?

[4]When tested with new actions, the framework’s performance declines significantly, as seen in Figure 3(c) and (d). This suggests that the pre-trained Transformer struggles to generalize in cases with out-of-distribution (OOD) actions because the pre-training data lacks these new actions. My question is: if the pre-training can relax the fixed-feature assumption—e.g., by allowing arm features to be randomly sampled in training— would this improve the model’s generalization capabilities?

### Questions
Please see the weaknesses. In addition, Line 272 states, “DPT-greedy estimates the optimal arm using the reward estimates for each arm during each task,” which appears to differ from the definition in Section 2.3 and Lee et al. (2023). Could you clarify whether DPT-greedy indeed uses the reward estimate or the best-arm estimate?

### Soundness
3

### Presentation
3

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
This paper introduces a new supervised pretraining procedure to learn the underlying reward structure in a multi-task structured bandit setting, which can be particularly useful when the optimal action is unknown in the training phase. This training method leads to improved performance across various bandit tasks, including linear, nonlinear, bilinear, and latent bandits. The authors also provide a theoretical generalization bound for the proposed algorithm.

### Strengths
I think the idea of predicting reward function rather than the optimal action has its merit. The numerical experiment is thorough, and the intuitions are well-explained.

### Weaknesses
I mainly have two concerns of the paper:

A distribution shift issue that doesn't arise under the in-context learning setting:
- The paper draws the analogy between the pretrained decision transformer with in-context learning multiple times throughout the paper, for both algorithm design and the theoretical bound. For in-context learning problems, we can think the task from the pretraining phase as being i.i.d. as the task during the test phase. But for the bandits problem, things are quite different. Specifically, the actions in $H_m$ generated during the pre-training phase is from weak demonstrators, whereas the actions in the test phase are generated from the transformer model itself. This will result in a distribution shift between the training and the test environment. And it matters both numerically and theoretically:
- Numerically, if the transformer's policy takes actions similar to the demonstrators such as UCB and TS, then why would it has a better regret performance than them? If the transformer's policy doesn't take actions similar to the demonstrators such as UCB and TS, how does the training through samples of $H_m$ from demonstrators extrapolate to the cases where the actions in $H_m$ is sampled from the transformer's policy? Given the large space of $H_m$, I don't believe the samples of $H_m$ following demonstrators can cover the whole space.
- Theoretically, I don't see any discussion of this distribution shift matter in the theoretical results in Section 7 (please correct me if I miss anything). To derive generalization bound in the face of this distribution shift, I believe notions such as the distribution ratio in the definition 5 of  https://arxiv.org/pdf/2310.08566 or others are necessary. 

Medium- to large-size action space or continuous action space: As I read from the algorithm and the experiments, the proposed method does not well handle tasks with continuous actions or with a large action space well, which are typical in linear and bilinear bandits. For tasks with a large number of actions, the proposed method becomes challenging to train because it requires H_m to include some reward for every possible action.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
