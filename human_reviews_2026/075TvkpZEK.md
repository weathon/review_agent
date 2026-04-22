# SMARAN: Closing the Generalization Gap with Performance Driven Optimization Method

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 8, 2

## Abstract
Optimization methods have evolved significantly by introducing various learning rate scheduling techniques and adaptive learning strategies. Although these methods have achieved faster convergence, they often struggle to generalize well to unseen data compared to traditional approaches such as Stochastic Gradient Descent (SGD) with momentum. Adaptive methods such as Adam store each parameter's first and second moments of gradients, which can be memory-intensive. To address these challenges, we propose a novel SMARAN optimization method that adjusts the learning rate based on the model's performance rather than the objective function's curvature. This approach is particularly effective for minimizing stochastic loss functions, standard in deep learning models. Traditional gradient-based methods may get stuck in regions where the gradient vanishes, such as plateaus or local minima. Therefore, instead of only depending on the gradient, we use the model's performance to estimate the appropriate step size. We performed extensive experiments on standard vision benchmarks, and the generalization trends observed with SMARAN demonstrate compelling distinctions relative to adaptive and non-adaptive optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an optimization algorithm SMARAN for deep learning. SMARAN has two main characteristics, the first is that it normalizes the gradient before updating the first-order momentum, and the second is that it adopts the objective function value to update the second-order momentum. Then this work provides the analysis of the regret bound for SMARAN based on common assumptions. Finally, SMARAN is compared with several classical or adaptive optimizers in the experiments of CV tasks. SMARAN achieves great test accuracies on CIFAR datasets, and obviously a low generalization gap on Tiny-Imagenet.

The main contribution of this work is that it adopts the function value in the adaptive learning rate, which could reduce the memory cost of optimizer states.

### Strengths
This paper proposes an algorithm SMARAN which adopts the function value in the second-order momentum. This is a novel technique in the optimizer studies. The writing of this paper is clear. Most contents of this work are easy to understand.

### Weaknesses
The “Introduction” part lists a series of drawbacks of previous methods. However, the proposed method SMARAN seems not to overcome all these drawbacks, except for the large memory of Adam, which is not mentioned in the following parts. This work needs to emphasize the motivation for using the function value to calculate the learning rate in SMARAN. It is also not explained or discussed in the work why replacing the gradient with the function value could close the generalization gap.

Some statements in the work lack the support of references. For instance, it states that “Previous methods … because gradients give the curvature of the landscape. However, for a nonconvex setting, steep curvature results in slow learning, whereas in our approach”. I think some related works should be provided for these assertions.

The setting of the experiments is relatively simple, i.e. only conducting the CV tasks. SMARAN adopts an adaptive learning rate, and the common adaptive optimizers are good at training a transformer-based model. More experiments on this kind of model should be included.

The presentation of the experimental results needs to be improved. It would be better to summarize the specific values of the test accuracies in one list to make the results clearer. This work states that SMARAN is a memory-efficient optimizer. However, this point is not shown in the experiments.

In addition, the organization of this work is also poor. The formulas in the proof of theorems leave too many blanks in the paper.

### Questions
It is mentioned at the end of page 4 that “Since the learning rate scheduler is based on the objective function value over training data, if the optimizer tries to overfit the training data, the same proportion of regularization prevents the model from overfitting”. Could you give a more detailed explanation of why adopting the adaptive regularization factor, and what advantage it has over the constant regularization factor.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SMARAN, a novel optimization method for deep learning that adjusts the learning rate based on the model's performance (i.e., the objective function value) rather than the gradient's curvature, aiming to close the generalization gap often seen in adaptive optimizers. Unlike Adam, which uses exponential moving averages (EMAs) of gradients, SMARAN uses the EMA of past loss values to scale the learning rate and incorporates a form of adaptive weight decay to prevent overfitting. Experiments on vision benchmarks (CIFAR, Tiny ImageNet) show that SMARAN achieves better generalization, with lower test loss and smaller generalization gaps, compared to state-of-the-art optimizers like Adam, AdamW, and SGD with momentum.

### Strengths
1. The core strength is its demonstrated ability to achieve superior generalization performance, as evidenced by consistently lower test loss and significantly smaller generalization gaps across multiple datasets and architectures compared to popular baselines.
2. The key innovation of using the EMA of the loss value (rather than gradients) to adapt the learning rate is conceptually distinct from most existing methods. This approach allows the optimizer to be cautious when losses are high and accelerate when losses are low and decreasing, potentially avoiding pitfalls like vanishing gradients.
3. SMARAN is more memory-efficient than adaptive methods like Adam because its adaptive learning rate component is a scalar (based on the overall loss) rather than a vector (requiring storage of per-parameter moment estimates).

### Weaknesses
1. The experiments are confined to image classification tasks on standard vision datasets (CIFAR, Tiny ImageNet). The paper lacks evaluation on more complex tasks (e.g., language modeling, object detection) or larger-scale datasets (e.g., ImageNet), making it difficult to assess the method's broader applicability and scalability.
2. While the paper provides a regret bound in the online convex setting, deep learning involves highly non-convex optimization. The analysis does not fully address the behavior of SMARAN in this more relevant non-convex landscape, which is the primary context for its use.
3. The learning rate adaptation depends directly on the absolute value of the loss. If the loss function has a very different scale (e.g., due to different architectures or tasks), the hyperparameters (especially the global learning rate η) might need significant retuning, potentially reducing its claimed "adaptive" advantage in practice. The paper uses a fixed γ=0.9 and λ=0.01, but a more thorough ablation study on these hyperparameters would strengthen the claims.

### Questions
How does SMARAN’s performance-driven learning rate adaptation theoretically behave in non-convex landscapes, particularly near saddle points or flat regions where the loss value may remain nearly constant for many iterations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel optimization method which the authors term as SMARAN, that aims to bridge the generalization gap often seen with adaptive optimizers and improve memory efficiency. SMARAN uniquely adjusts its learning rate based on the model's performance (loss values), utilizing exponential moving average (EMA) of normalized gradients to determine the update direction and an EMA of squared loss values to dynamically scale the learning rate.

The authors argue that such a strategy based on loss performance allows for cautious learning with high losses and accelerated convergence in the regime with low and decreasing losses, thus preventing stagnation in flat regions. 

SMARAN also integrates an adaptive weight decay regularization, whose strength is tied to the performance-based learning rate, to address overfitting. Theoretical analysis provided along with extensive experiments on image classification problems using multiple model architecture types.

### Strengths
I find a few key strengths in this work:
- The most significant strength is SMARAN's innovative approach to adjust the learning rate based on the objective function value (model or loss performance) rather than solely gradient information. This represents a fresh and unique direction in optimizer design.
- Proposed method SMARAN directly tackles two critical issues in deep learning: the generalization gap associated with adaptive methods and their memory intensiveness due to per-parameter moment storage. SMARAN's scalar learning rate is a clear win for memory efficiency, and the empirical results also seem to strongly support improved generalization.
- The integration of an adaptive weight decay mechanism, which is dynamically controlled by the performance-based learning rate, is a clever design. This dynamic regularization is well-suited for mitigating overfitting as the model converges.
- Proposed method consistently shows compelling empirical superiority across diverse vision benchmarks (CIFAR-10, CIFAR-100, Tiny ImageNet) and architectures (ResNet50, DenseNet121) when compared to a comprehensive set of baselines, including SGD, Adam, AdamW, RAdam, DecGD, and Prodigy.

### Weaknesses
Some weaknesses in the paper still remain undressed at this point:
- The use of a normalized gradient (Equation 2) could introduce numerical instability if the gradient norm approaches zero even when a reasonable epsilon value is used.
- While the paper states it's "performance-driven," the "performance" metric is consistently defined as the loss function value. Although loss is a direct indicator, further discussion on whether other performance metrics (e.g., validation accuracy, task-specific metrics) could be effectively incorporated into the adaptive learning rate calculation would be interesting. Wonder if the results would change in any way if we had looked these alternative performance metrics?
- Would have been good to see some theoretical study or explanations for how method performs in the non-convex setting.
- While future work mentions extending to text and video, the current experiments are primarily on vision tasks., it would be good to obtain empirical insights on language models / tasks as well.

### Questions
Some qns for the authors:
- How robust is the proposed SMARAN method to extremely noisy gradients, particularly concerning the normalized gradient? Could situations arise where the gradient norm is very small but not precisely zero, leading to an amplified noisy direction?
- The adaptive weight decay is a key feature. Could the authors elaborate on scenarios where a fixed weight decay (like in AdamW) might still be preferred, or where SMARAN's adaptive weight decay might have limitations?
- Could the authors offer more insight into the specific cases where SMARAN showed the largest performance gains or where its generalization gap was most significantly reduced compared to other optimizers? Are there particular types of datasets or model complexities where SMARAN particularly shines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an optimizer that adaptively changes its effective learning rate based on the an EMA of losses, in addition to using a EMA on the gradient history. Modulation of the learning rate as a function of the loss, is their main contribution. They provide a regret analysis in the online convex optimization setting. Empirically, across standard vision benchmarks they report competitive convergence with stronger generalization than Adam-style methods and SGD, while avoiding per-parameter second-moment storage.

### Strengths
1) The paper is easy to follow and is presented in a clear manner.
2) They propose a loss-driven gain that avoids per-parameter second-moment buffers reduces memory and implementation complexity.

### Weaknesses
1) Missing reference and comparison to [1].

2) I am afraid the hyper parameters with which SGD experiments are run are not optimal. For instance in Fig 1(a) SGD on ResNet-50 does less than 70% test accuracy, however in this [2] github implementation's default values SGD achieves 93.5%. Providing some clarity on this would be helpful in gauging the proposed methods efficicacy.


[1] Rolinek, Michal, and Georg Martius. "L4: Practical loss-based stepsize adaptation for deep learning." Advances in neural information processing systems 31 (2018).

[2] https://github.com/kuangliu/pytorch-cifar

### Questions
In line 128, it is mentioned that when the recent losses are low the learning rate is increased leading to faster convergence. However, looking at the update equation 13,$$
x_{t+1} = x_t - \eta \left( \frac{f(x_t)}{\sqrt{v_t} + \varepsilon} \right)\,(m_t + \lambda x_t)
$$
looking at this equation, I am afraid with lower loss the learning rate is decreased and not increased as stated in the paper. Can you provide clarity on line 128.

### Soundness
2

### Presentation
2

### Contribution
2
