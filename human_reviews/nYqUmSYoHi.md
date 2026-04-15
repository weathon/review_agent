# A Geometric Analysis of Multi-label Learning under Pick-all-label Loss via Neural Collapse

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3

## Abstract
In this study, we explore multi-label learning, an important subfield of supervised learning that aims to predict multiple labels from a single input data point. This research investigates the training of deep neural networks for multi-label learning through the lens of neural collapse, an intriguing phenomenon that occurs during the terminal phase of training. Previously, neural collapse (NC) has been investigated both theoretically and empirically in the context of multi-class classification. For last-layer features, it has been demonstrated that (i) the variability of features within classes collapses to zero, and (ii) the feature means between classes become maximally and equally separated. In this work, we demonstrate that the NC phenomenon can be extended to multi-label learning, revealing that the "pick-all-label" training formulation for multi-label learning exhibits the NC phenomenon in a more general context. Specifically, under the natural analog of the unconstrained feature model, we establish that the only global minimizers of the pick-all-label loss display the same equi-angular tight frame (ETF) geometry. Additionally, scaled average of the ETF are used to represent the features of samples with multiple labels. We also provide empirical evidence to support our investigation into training deep neural networks on multi-label datasets, resulting in improved training efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates the training of deep neural networks for multi-label learning through the lens of neural collapse. Its main contributions   are summarized as:

a. This paper shows that the last-layer features and classifier learned via overparameterized deep networks exhibit a more general version of neural collapse.

b. This paper studies the global optimality of a commonly used pick-all-label loss for M-lab and proves that the optimization landscape has benign strict saddle properties so that global solutions can be efficiently achieved.

### Strengths
a. This paper is well-written and easy to follow.

b.  I appreciate that this paper provides extensive experiments.

c. Interesting findings. This paper shows that the last-layer features and classifier learned via overparameterized deep networks exhibit a more general version of NC. The high-order Multiplicity features are scaled average of their associated features in Multiplicity-1.

### Weaknesses
My main concern is the novelty of results. The main results Theorems 1 and 2 are so similar with the reference [1], i.e. Theorem 1 corresponds to Theorem 3.1 of [1] and Theorem 2 corresponds to Theorem 3.2 of [1]. In my opinion, these results are the extended versions of [1] with a few improvements for multi label learning. It is OK to leverage them, but they are not enough to be the main contributions in this paper. 

[1] Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. Advances in Neural Information Processing Systems, 34:29820–29834, 2021.

### Questions
a. Please discuss more about the results of reference [1]

[1] Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. Advances in Neural Information Processing Systems, 34:29820–29834, 2021.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the training of neural networks for multi-label learning. The analysis aims to show a neural collapse-type phenomenon when minimizing a pick-all loss function to address the multi-label learning task. By treating the features of every sample as a free optimization variable, the paper formulates and analyzes the optimization problem in equation (4), for which they characterize the global optima (Theorem 1) and show that all local optimal will be globally optima (Theorem 2). Several numerical results are discussed in section 4 to measure the M-lab ETF in training the neural network on multi-label learning tasks.

### Strengths
1- The paper is well-written and easy to follow. The authors present their results clearly, and the writing is overall in great shape.

2- The paper focuses on the interesting subject of neural net training dynamics in multi-label learning tasks.

### Weaknesses
1- While I understand that analyzing the problem in (3) could be challenging in the general case, I think the simplification of treating every feature vector $h_i = \phi_\theta(x_i)$ as a free optimization variable is restrictive. Could the analysis extend to a more general choice of $\phi_\theta$, e,g, a one-layer overparameterized neural network with some activation function? The authors may still be able to show a weaker result on local optima or stationary points of the objective.  If not, the paper should discuss why the analysis will be challenging for a parameterized neural net function $\phi_\theta$ and some more evidence of why the assumption of treating the features as free variables would make sense for deep neural nets. 

2- The theoretical results only analyze the critical points of the loss function, and the paper has no statement on how a gradient-based optimization method will perform in solving (3) or (4). Stating a corollary or theorem on the convergence behavior of a first-order algorithm for optimizing (4) wold be a nice addition. I think Theorems 1,2 could connect the first or second-order stationary points of the objective to the global minima, and so the authors should be able to use the results in the optimization literature to state such a convergence guarantee for a gradient-based optimization algorithm.

### Questions
Please see my comments for weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper explores neural collapse (NC) phenomenon in multi-label (M-lab) learning using deep neural networks.The main content is：
1.The paper provides theoretical analysis to show multi-label NC is the global solution under the unconstrained feature model.
2.The paper introduces the concept of multi-label equiangular tight frames (ETF) to characterize NC geometry.
3.They  empirically demonstrate multi-label NC on practical networks trained on synthetic and real datasets.

### Strengths
1.The paper is well-written and easy to follow. The graphics in this article are very intuitive.
2.This article combines NC and M-lab for the first time，providing a new perspective for the study of multi-label.

### Weaknesses
The theoretical part of this article is very similar to [1], lacking significant technological innovation and not sufficiently novel.The two papers share a lot of similarities in  technical approach and theoretical analysis frameworks towards extending and understanding the representation geometry of neural networks in multi-label learning tasks via NC.I don't see any important work in your theoretical section that goes beyond what has been done in [1].So, despite the interesting combination of NC and M-lab， I don't consider it as a contributory work.

Ref:
[1] Zhu, Zhihui, et al. "A geometric analysis of neural collapse with unconstrained features." Advances in Neural Information Processing Systems 34 (2021): 29820-29834.

### Questions
Please see weekness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
