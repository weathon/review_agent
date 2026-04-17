# Learning to Regularize: A Meta-Learning Approach for Sharpness-Aware Optimization

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
The ideal regularization strategy for deep neural networks should adapt to the local geometry of the loss landscape, since solutions in high-curvature regions are sensitive to perturbations and often generalize poorly. Classic penalties are static and thus may over-regularize in flat regions while under-regularizing in sharp ones. We propose the Structural Risk Network (SRN),a lightweight dynamic regularizer learned by meta-optimization. SRN maps the current model parameters to a state-dependent surrogate $r(\Theta;\phi)$, whose gradient is added to the task gradient at every training step, without per-step inner maximization. The surrogate is meta-aligned to a composite signal that blends two sharpness-related observables---validation-loss sensitivity and the inverse classification margin—providing complementary global and local cues. Under standard smoothness assumptions, a margin–curvature link and a validation–Hessian decomposition explain why this composite target emphasizes low-margin/high-sensitivity neighborhoods, biasing updates away from dominant curvature directions. We assess SRN's effect on curvature via an out-of-loop evaluation of the largest Hessian eigenvalue and observe reduced spikes and lower late-epoch values. In a unified protocol on CIFAR-10/100 with ResNet-8/20/32 (identical backbones, optimizer, epochs, and light augmentations), SRN consistently improves Top-1 accuracy over strong static and dynamic baselines while incurring only moderate overhead, yielding a favorable accuracy–compute trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work the authors explore a new curvature regularization scheme based on metalearning. They posit that it is beneficial to have, in their words, a dynamical metric for curvature regularization --- one whose functional form can change over training. They define the structural risk network, a regularizer composed of a surrogate network whose output is an estimation of the curvature using local loss perturbations and an inverse margin on a held out test set. The authors provide some theoretical justification for why this form might approximate the loss, and propose a meta-learning approach where the parameters of the surrogate network are occasionally updated during training.

The authors then conduct experiments on CIFAR10 and CIFAR100 where they show that the proposed method seems to show generalization benefits over the other proposed methods when compared at equal step-time. The overhead of the algorithm is reported at a few percent.

### Strengths
The overall idea of taking a meta-learning approach to curvature regularization is an interesting one and, to my knowledge, understudied. The mixing of two methods of curvature estimation is also an interesting proposal. The initial experiments provide an interesting first result into this area, and seem to indicate that the method should be investigated further.

### Weaknesses
The theoretical justification of the methodology has some major issues. First off, during much of training networks are not at a stationary point; this means the analysis (which assumes a stationary point) is not correct. Second, the analysis of 3.4 suggests that the method *increases* $\lambda_{max}$ for fixed values of the learning rate $\eta$. This proposed mechanism is in contradiction to the claims of reduced curvature, so something is off there. In addition, it did not look like SRN affected the curvature much via FIgure 3 (which in its current state is very hard to parse).

With regards to the experiments: it was unclear what the learning rate tuning procedure was for the different methods. This can matter a lot in regularization settings, where sometimes part of the benefit of a method is that it induces a larger or smaller effective learning rate, bringing the method closer to the optimal LR than the baseline.

There are also concerns about the scalability of the method. With the experimental details provided, though the compute cost is amortized by infrequent metalearning updates, the memory cost of the SRN is potentially high. In addition, all experiments were on ResNet with CIFAR10 and CIFAR100 in a regime where training took O(100) epochs; there is a question of how well this method works with other architectures (e.g. ViT) on a larger dataset with more classes (e.g. ImageNet).

### Questions
How does the computational complexity and memory cost of the method scale with different architectures --- primarily, the transformer architecture?

What happens when the base learning rate is tuned for the different methods?

What do results look like with architectures like ViT, and datasets like ImageNet?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the Structural Risk Network (SRN) — a meta-learned dynamic regularizer for sharpness-aware optimization.
Instead of performing inner-loop maximizations as in SAM, SRN learns a small neural network that outputs a state-dependent surrogate $r(\Theta; \phi)$, whose gradient is added to the task gradient at each update step. Theoretically, the paper links SRN’s update to curvature suppression along top Hessian eigen-directions.

### Strengths
SRN adds $\nabla_\Theta r(\Theta;\phi)$ to the task gradient each step and does not perform per-step adversarial/ascent inner loops (contrast with SAM/ASAM)

The composite meta-target combining validation sensitivity and inverse margin is conceptually elegant and theoretically motivated via Hessian-margin links.

The paper provides clear theoretical justification for curvature-guided stability and suppression of sharp directions

The paper offers a general framework for dynamic regularization that could be extended to other curvature-aware or meta-learned objectives.

### Weaknesses
The conceptual foundation overlaps with existing sharpness-aware techniques. The main novelty lies in how the curvature signal is learned rather than the overall objective. Indeed, there are already existing tools for dynamic regularization (using learning rate) based on estimation of local landscape. Ex: SALR: Sharpness-Aware Learning Rate Scheduler for Improved Generalization, amongst others.

The analysis assumes smoothness and stable curvature proxies (validation loss and inverse margin), which may not generalize beyond small-scale image benchmarks.

Evaluaton is limited to CIFAR-10/100 and moderate-size ResNets. Results on larger or non-vision tasks (e.g., Transformers) would better support generality.

### Questions
Could you quantify the correlation between the SRN surrogate $r(\Theta; \phi)$ and actual curvature ($\lambda_{\max}$) across training epochs?

Can you visualize or interpret SRN’s learned outputs over training to show how it adapts the penalty dynamically?

How sensitive is performance to the outer-loop frequency (validation updates) or meta-learning rate $\beta$?

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
The authors present a meta-learner capable of steering deep learning algorithms towards loss landscapes regions with better generalization capabilities. The approach consists in adding a small additional network that takes the original network parameters as inputs and uses a combination of margin error and validation loss to improve the performance of the meta-learner and help the original network to generalize. The overall approach appears to be tested using simple gradient descent on both set of parameters. The authors test their approach with convnets on cifar. They compare their methods to many other regularizing approaches. They study the impact of their approach on the curvature dynamics. Finally they provide an ablation study of the approach

### Strengths
- Despite the title, the authors do no directly use the sharpness of the objective to improve the meta-learner. Since sharpness as a measure of stability has been criticized, their approach avoids such an issue. It uses simple generalization criterions.
- The authors compare their approach to many other methods.
- An ablation study helps understand the benefits of each design choice. 
- The appendix contains many additional results.

### Weaknesses
- The overall cost of the approach is unclear. In Table 3, times are presented. Label smoothing, which is extremely simple to implement, raises the time per epoch (compared to cross entropy only) by 25%, which I find quite surprising. On the other hand, it is unclear whether the time for SRN takes into account the training of the additional network. I believe, for fair comparison, that the additional time for training should be reflected with a larger budget for hyperparameter optimization for the other methods for example.
- Some design choices are strange like adding a clipping on top of the additional network (that said the authors provide a valuable ablation study).
- The generic approach (meta-learner) does not help deepening our understanding of deep learning algorithms. Adopting this method would rather make the optimization of deep networks an even darker black box that is currently the case. 
- The approach requires a validation set and is tailored to multiple epochs. It is typically not a setup of some recent models like llms. 
- The approach is only tested on convnets and not other architectures.
- We could hope that the approach can alleviate the need of e.g. tuning some weight decay hyperparameters. However it introduces many new hyperparameters. 
- (The approach does not seem to really modify the curvature dynamics. That said, as mentioned above, the sharpness dynamics may not give the full story, anyway).
- Learning rates were not tuned for each of the methods. The authors argue that they use a "lightweight protocol" but it is actually unfair to not take into account additional tuning. In particular: how was the learning rate selected in the first place? We see in Figure 3 that the network may not reach the edge of stability for the smaller resnet for example.

### Questions
Major:
- Why did the authors not compare their approach to sharpness aware minimization?
- Could the authors present comparisons with tuned learning rate at least for a baseline approach (so sgd with momentum and weight decay for example)?
- How are $z(L_val)$ and $z(1/M)$ defined?
- Could the authors try using also some warm-up? Warm-up may reduce sharpening dynamics of the Hessian.

Minor:
- What is GAP in the network definition?
- The existence of a clipping at the top of the architecture is never mentionned before the ablation study on $r_max$

### Soundness
3

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
3

### Summary
The paper proposes SRN (Structural Risk Network), a lightweight meta-learned surrogate regularizer that takes the main model’s parameters as input and outputs a scalar “sharpness-risk” signal used to augment the task loss. SRN is trained in an inner–outer meta-learning loop and is motivated by building a computable meta-target from validation-loss sensitivity and the inverse classification margin, intended as proxies for curvature/sharpness. The method is evaluated on CIFAR-10/100 with ResNet-8/20/32, tracks the largest Hessian eigenvalue during training, and includes ablations and time-overhead analysis.

### Strengths
1. I find theoretical framing links the validation loss sensitivity to the Hessian spectral norm (Eq. 5), justifying its use as a sharpness proxy and the meta-target combines this with inverse margin (Eq. 6). The derivation further shows SRN effectively reduces the step along the top-curvature direction, explaining observed stability improvements.

2. The meta-learned, optimizer-agnostic surrogate that avoids per-step inner maximization yet exhibits curvature-aware behavior.

### Weaknesses
1. I think the experimental section is severely lacking. Firstly, the datasets considered only include CIFAR-10 and CIFAR-100, which limits the ability to validate. Secondly, the baselines considered do not include sharpness-aware methods such as SAM, ASAM, and GSAM, so the experimental superiority of SRN cannot be demonstrated. Moreover, other backbones such as ViT could strengthen the solidity of the experiments.

2. Ranges/curves (not just points) for sensitivity normalization, margin temperature, and the weight are missing. It would help if reporting robustness to different meta-val splits.

### Questions
1. How exactly is “validation-loss sensitivity” computed and normalized? 

2. What temperature and class-competition rules are used; any failure cases on long-tail or borderline examples?

3. Behavior with AdamW, cosine/one-cycle, warmup/WD—can SRN shorten warmup or stabilize larger base LRs?

### Soundness
2

### Presentation
3

### Contribution
2
