# Mind the budget: Accelerating Deep Reinforcement Learning using Early Exit Neural Networks

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
The _"Bitter Lesson"_  from Richard S. Sutton emphasizes that AI methods leveraging computation tend to outperform those relying on human insight, underscoring the value of approaches that use computational resources efficiently. In deep reinforcement learning (DRL), this highlights the importance of reducing both training and inference time. While early exit neural networks, models that adapt computation to input complexity, have proven effective in supervised learning, their use in DRL remains largely unexplored. In this paper, we propose the use of Budgeted EXit Actor (BEXA), which is a novel actor-critic architecture that integrates early exit branches into the actor network. These branches are trained via the underlying DRL method and use a constrained value-based criterion to decide when to exit, allowing the policy to dynamically adjust its computation. BEXA is general, easy to tune and compatible with any off-policy actor-critic method. We evaluate BEXA using different DRL methods such as SAC and TD3 on a suite of MuJoCo tasks. Our results demonstrate a substantial improvement in inference efficiency with minimal or no loss in performance. These findings highlight early exits as a promising direction for improving computational efficiency in DRL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Budgeted EXit Actor (BEXA), which adds early exits to actor networks in DRL to reduce computation under a budget constraint. Exit selection is optimized via a linear program using Q-values, enabling dynamic trade-offs between speed and performance.

### Strengths
1. This paper address a very interesting issue
2. This method is novel

### Weaknesses
1. The network architecture used in the paper is quite shallow, making it hard to observe significant differences. The authors rely on a conventional setup—a two-layer MLP with 256 hidden units per layer. Later, they mention “we drastically reduce network capacity to 4–16 hidden units per layer,” which suggests that the reduced configuration is still a two-layer MLP, but with only 16 units per layer at most. In order make it deep, why not use 4 layer MLP with 16 hidden units?

2. Experiment need more diversity environments. I suggest auther may consider add some very simple tasks (montain car, carpole) which we know they do not need complex network and see if the method can recognize that.

3. Exist is like a network structure pruning in policy network. The authors could strengthen the paper by adding related work and discussing the distinction and significance of early exits compared to pruning-based approaches.

### Questions
1. The output layer of MLP is a linear layer and exit is also a linear layer. So right after layer 2 (K=2), If exit, then the network graph should be same as K = 3 right (from figure 1, k = 3 will be after output layer)? 

2. Since each exit result in an Q_i, why not apply the whole method in critic to creat Q_i? I guess apply early exit in Q may result in increaseing in estimation error.

### Soundness
3

### Presentation
2

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
This paper proposes a novel actor-critic architecture which integrates early exit branches into the actor network. These branches are trained via the underlying DRL method and use a constrained value-based criterion to decide when to exit, allowing the policy to dynamically adjust its computation. Experiments on MuJoCo tasks using SAC and TD3 demonstrate substantial improvements in inference efficiency with minimal or no loss in performance.

### Strengths
1. This paper successfully addresses the challenges of applying traditional confidence-based early-exit methods to DRL, offering a principled and tunable approach to effectively balance performance and computational speed.

2. The extensive ablation studies, comparing various alternative strategies, provide robust evidence and strong support for the proposed BEXA approach.

### Weaknesses
1. From the results shown in Figure 3, although BEXA achieves inference acceleration, its final average return, when compared to baseline methods (SAC/TD3) on tasks like Hopper-v4, is only marginal or on par. Furthermore, performance appears to decline on tasks such as HalfCheetah-v4 and Ant-v4 in certain runs.
2. This paper lacks a comparison of the total wall-clock training time during the learning phase. The method introduces significant overhead: jointly optimizing $K$ exit policies and $K$ Critic Heads, and solving a Linear Program (LP) at every update step. It is necessary to demonstrate that the increase in training time is far outweighed by the improvement in inference speed.
3. The hyperparameter $\lambda$ (Gate Loss Scale) in the objective function has a wide search range ($1e-3$ to $1e-1$). This paper lacks a necessary sensitivity analysis for $\lambda$.
4. The experiments are based on a very low number of seeds for the final evaluation (only three random seeds per environment).

### Questions
See Weaknesses above.

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
This paper addresses the high computational cost of Deep Reinforcement Learning (DRL) by proposing a method to incorporate Early Exit Neural Networks (ENNs) into off-policy actor-critic algorithms. They propose Budgeted EXit Actor (BEXA), an architecture where the actor network possesses multiple early-exit branches. The decision of which exit to use is governed by a novel, budget-aware gating mechanism. This mechanism formulates the exit selection as a linear program (LP) solved during each training update. The LP finds the optimal probability distribution over the exits that maximizes the expected Q-value while adhering to a user-defined computational budget.  The results demonstrate that BEXA can achieve significant actor-inference speedups while maintaining or even slightly improving the sample efficiency and final performance of the baselines.

### Strengths
The primary originality lies in the formulation of the early-exit gating mechanism for DRL. While ENNs are explored in supervised learning, their application to DRL is novel and non-trivial. The key insight is to frame the exit selection as a constrained optimization problem. Using a linear program (Eq. 2) to directly trade off expected return (Q-values) against a computational budget is a solution to this problem.

For the clarity of the paper, the problem is clearly stated, and the related work is well-contextualized. Section 4 provides a step-by-step derivation of the BEXA method, from the architecture (4.1) to the core budget-aware learning objective (4.2). The inclusion of pseudocode (Algorithm 1) further clarifies the complete training loop.

### Weaknesses
1) Lacking analysis on the trade-offs between performance and the computation cost

The paper convincingly demonstrates a reduction in actor inference FLOPs. However, the BEXA method introduces two new sources of computational overhead during the training update steps with multi-head critic and LP solvers. Since the paper considers online RL settings, the total computational cost should be taken into account. Moreover, the paper does not provide an analysis of the trade-off between total computational cost and the performance. It is unclear if the total computation (inference FLOPs + training-step FLOPs) is reduced, or if the wall-clock time for training is actually faster.

2) Scalability to deeper networks with more parameters and layers

One of the main interests of this paper is to maintain (or even improve) the performance of the policy given a computational budget, which hypothesizes that putting more computation (more parameters) will lead to performance improvement. However, the experiments are conducted on shallow MLPs (2 layers, K=3 exits). The paper's Future Work section mentions scaling to ResNets or Transformers, where the multi-head critic design seems to scale poorly. A ResNet-18 could have K=8 or more exits; would this require 8 separate critic heads? This linear increase in critic computation appears like a significant scalability bottleneck that might negate the actor's speedup.

### Questions
1)  The paper mentions the computational budget of policy, but in actual applications, especially for robotic control, the total wall-clock time will matter. Could you provide more details on the results, for example, on the total wall-clock training time for BEXA-SAC vs. SAC?

2) I am also curious about the scalability of the method. The multi-head critic seems to be a scalability bottleneck for deep networks. How do you envision BEXA being applied to a model with, for example, $K=10$ exits? Would this require 10 critic heads, and if so, is that computationally tractable?

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
2

### Summary
Applies Early Exit Neural Networks to the actor, where the exit probabilities are trained to match the solution of a linear program solved per state. The linear program maximizes $Q(s,a)$ over the probabilistic exits, subject to a hard constraint on the expected cost, where the cost of each exit is proportional to the FLOPs required to reach that exit.

### Strengths
The algorithm is somewhat simple, and is novel to my knowledge

I do not know this area, but the baselines seem reasonable

The budget hyperparameter is better than some alternative approaches, in that this hyperparameter guarantees any requested budget constraint is met, rather than being, e.g., an uninterpretable soft penalty

### Weaknesses
3 seeds total is not enough seeds, https://arxiv.org/pdf/2304.01315 for example suggests at least 15, preferably more

Narrow hyperparameter search range, with no indication of why those bounds were chosen

Training time can increase (because all exits are trained), as the paper notes

No wall clock results (the paper notes that the dynamic branching is not easily parallelized on GPUs, though maybe CPU inference would more easily show a speed-up?)

The algorithm introduces a `gate_loss_scale` hyperparameter and a `gate_loss_freq` hyperparameter. The paper claims "Despite introducing new hyperparameters, BEXA required no extra tuning budget relative to its baselines", but I am not sure that's fair because the gate_loss_scale range is 1e-3 to 1e-1. That range had to be chosen, and it is not clear to me that it should be the same for other tasks (i.e., beyond the continuous control MuJoCo tasks). Similarly, there is no discussion of the `gate_loss_freq`.

### Questions
I think a one-layer actor might be an additional important baseline


&nbsp;


> The ”Bitter Lesson” from Richard S. Sutton emphasizes that AI methods leveraging computation tend to outperform those relying on human insight, underscoring the value of approaches that use computational resources efficiently

The first quotation mark is reversed. But also, you might consider just removing this sentence entirely, as well as the sentence after it.


&nbsp;


Do the exit probabilities vary a lot per state?


&nbsp;


> Yet, these methods can be hard to tune and might lead to potential training overhead

This claim is made without any citation or evidence


&nbsp;


> fraction of floating point operations (FLOPs)

should be "fraction of the"


&nbsp;


> and its self-imitation perspective makes it a natural asset for DRL transfer

I don't understand this


&nbsp;


> represented by deep neural network

"represented by a"


&nbsp;


> We assume that each exit policy $\pi_i$ has an associated $Q$-function $Q_i$.

This is just a notational choice, not an assumption, right? You might consider rephrasing that as "Let $Q_i$ be the $Q$-function of each exit policy $\pi_i$."


&nbsp;


> negligible small

should be "negligibly"

&nbsp;

### Soundness
2

### Presentation
2

### Contribution
2
