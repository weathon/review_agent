# Flat is the New Sharp: Flatness-Aware Regularization for Robust Learning

- Avg Score: 0.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 0, 0, 0

## Abstract
Understanding and improving the generalization of neural networks has been a
central focus in machine learning. One of the most significant efforts to ad-
dress this challenge revolves around the concept of the loss landscape in deep
neural networks (DNNs). While some researchers have posited that solutions lo-
cated in flatter regions of the loss surface tend to generalize better than those in
sharper regions, others have provided theoretical frameworks and empirical find-
ings suggesting that flat minima are not the sole, or even primary, reason for strong
generalization. Despite these advances, the relationship between loss landscape
geometry and generalization remains an open question. In this work, we con-
tribute to this open question by introducing Flatness-Aware Regularization (FA-
Regularization). This method explicitly penalizes the loss surface towards flatter
minima by incorporating an estimate of the trace of the squared Hessian into the
training loss. We present empirical results demonstrating that this Hessian esti-
mate effectively penalizes the curvature of the loss surface, enabling the optimizer
to converge to flatter regions. We tested our FA-Regularizer across a variety of
models (MLP and Logistic Regression) and datasets (CIFAR-100, IMDB Movie
Reviews, and Breast Cancer Wisconsin). Our FA-Regularization method consis-
tently leads to improved generalization on cifar-100 compared to a baseline loss
function without the penalty term. Our FA-Regularization method indicates that,
flatness is shown to correlate with, but not fully explain, generalization.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes to regularize neural network training with the trace of the loss Hessian to obtain flatter solutions. To make this regularizer computable, they propose to approximate the trace of the Hessian via Hutchinson's method.

### Strengths
- Given that flatness is a popular indicator for generalization, regularizing for it is a reasonable idea.

### Weaknesses
**Placement in the literature:**
- Using the trace of the loss Hessian is not sound, since it is not reparameterization-invariant [3]. Several reparameterization-invariant flatness-measures have been proposed since 2017, such as the Fischer-Rao-Norm [5] and Relative Flatness [6]. This is not discussed at all.
- The novelty of this work is unclear, since regularizing the loss with a suitable flatness measure has already been proposed [1].
- The paper fails to address several important works on flatness after the "reparameterization-curse" [3], such as [2, 7, 8, 4]

**Experiments:**
- The paper does not compare to any baseline. This alone is sufficient reason for rejection.
- The empirical results show no significant gain, except for the Breast Cancer dataset with a tiny MLP.
- The runtime for training even a tiny MLP on a tiny dataset like Breast Cancer is huge.
- No ablation on the use of Hutchinson's method to approximate the loss Hessian is made. This is particularly important because Hutchinson's method is known to be very inaccurate for deep learning.

### Questions
References: 

[1] Adilova, Linara, et al. "FAM: Relative Flatness Aware Minimization." Topological, Algebraic and Geometric Learning Workshops 2023. PMLR, 2023.

[2] Andriushchenko, Maksym, and Nicolas Flammarion. "Towards understanding sharpness-aware minimization." International conference on machine learning. PMLR, 2022.

[3] Dinh, Laurent, et al. "Sharp minima can generalize for deep nets." International Conference on Machine Learning. PMLR, 2017.

[4] Han, Ting, et al. "Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking." Advances in Neural Information Processing Systems, 2025.

[5] Liang, Tengyuan, et al. "Fisher-rao metric, geometry, and complexity of neural networks." The 22nd international conference on artificial intelligence and statistics. PMLR, 2019.

[6] Petzka, Henning, et al. "A reparameterization-invariant flatness measure for deep neural networks." Science meets Engineering of Deep Learning 2019. Neural Information Processing Systems (NIPS), 2019.

[7] Petzka, Henning, et al. "Relative flatness and generalization." Advances in neural information processing systems 34 (2021): 18420-18432.

[8] Walter, Nils Philipp, et al. "When Flatness Does (Not) Guarantee Adversarial Robustness." arXiv preprint arXiv:2510.14231 (2025).

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a regularization term based on the approximation of the Hessian of the loss in order to encourage training of a neural network towards flatter minima. It demonstrates the effect of the regularizer on three tasks (CIFAR100 with MLP, Newsgroups with linear regression and BreastCancer with MLP) reporting the flatness and the accuracy on the tasks. The conclusion of the work is that regularizer leads to flatter solutions according to the Hessian and might or might not lead to better accuracy, but introduces computational overhead.

### Strengths
The paper addresses interesting question of connection between flatness and performance of neural networks.

### Weaknesses
The paper is missing on all the recent developments in the field, including SAM, Relative Flatness and FAM based on it, Fischer-Rao Norm, etc.

The usage of Hessian-based metrics for measuring flatness of the loss surface is known to mislead the understanding of generalization because of reparametrizations.

The evaluation is (i) limited to very small models (ii) has severely low performance (iii) does not show any meaningful connection between performance and flatness.

### Questions
-

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes Flatness-Aware Regularization (FA-Regularization), a simple method that adds a penalty proportional to the squared Frobenius norm of the Hessian, estimated efficiently via the Hutchinson trace estimator using Hessian–vector products.

Experiments on (very) small models (MLPs, logistic regression) show that FA-Regularization reduces estimated curvature and slightly improves accuracy on CIFAR-100, but offers limited or no gains on text and tabular tasks while incurring significant computational overhead.

The paper positions FA-Regularization as a direct curvature regularizer related to sharpness-aware methods like SAM.

### Strengths
- $S_1$: The high-level idea is very clear directly penalizing the curvature using Hessian–vector products.

- $S_2$: The proposed regularization seems conceptually easy to integrate into standard training loops.

- $S_3$: The computational overheads are honestly acknowleded.

### Weaknesses
- $W_1$: The paper looks unfinished. The Discussion section is misformatted, the paper is under-length (7 pages out of 9 allowed), and several definitions (e.g., “Avg. Flatness”) are missing. Figures and tables lack sufficient detail (captions, scales, error bars), and experimental descriptions are incomplete.

- $W_2$: There are no empirical comparisons with strong contemporary methods explicitly targeting flatness or sharpness, such as SAM (Foret et al., 2021), ASAM/ESAM, or CR-SAM. This makes it impossible to assess whether the proposed regularizer meaningfully advances the state of the art.

- $W_3$: The paper does not cite or discuss recent theoretical analyses of sharpness-aware methods, including “Towards Understanding Sharpness-Aware Minimization” (Andriuschenko & Flammarion, 2022) and “A Modern Look at the Relationship between Sharpness and Generalization” (Andriushchenko et al. , 2023). These works directly address the link between curvature measures and generalization, which is central to this paper’s motivation. Their omission weakens the contextual framing and risks overstating novelty.

- $W_4$: The implementation and measurement protocol are unclear: batch sizes, Hutchinson probe count, penalty frequency, seed averaging, and metric computation are not specified. No ablations or sensitivity studies are reported.
Furthermore, the curvature regularizer is computed with dropout disabled, so the penalized function differs from the actual training objective. The additional $1/B$ scaling factor in the regularization term lacks derivation or justification. The empirical results and reports do not allow reproducibility.

- $W_5$: All experiments use small models (MLPs or logistic regression) and modest datasets (CIFAR-100, 20NG, tabular). The approach’s value for modern architectures (ResNets, Transformers) is therefore untested.

### Questions
I do not have precise questions apart from those directly linked to the weaknesses I underlined.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a method called "Flatness-Aware Regularization" (FARegularization), which aims to improve the generalization ability of deep neural networks by explicitly penalizing the curvature of the loss surface. The core idea of this method is to add a regularization term to the training loss, which is an estimate of the trace of the squared Hessian matrix. The authors use the Hutchinson random trace estimator to approximate this value and claim that the method can be seamlessly integrated into standard optimizers such as SGD and Adam, consistently improving generalization performance. However, the experimental evaluation of this paper suffers from serious flaws. The model used is too simplistic (a "toy model") to support any conclusions it makes about the generalization ability of modern deep learning. Furthermore, the computational cost of this method is extremely high, and the paper fails to make any comparisons with state-of-the-art flatness-aware optimization methods such as SAM.

### Strengths
1.  The relationship between the geometry of the loss landscape and its generalization ability is an important and unsolved problem in deep learning. This paper attempts to provide new insights into this area.

2.  The paper clearly articulates the proposed method: using $tr(H^2)$ as a measure of flatness and approximating it using the Hutchinson
estimator.

### Weaknesses
1. The authors used a simple two-layer MLP 10 on CIFAR-100 9. This was a "toy" experiment. The baseline model ($\lambda=0$) achieved a final accuracy of only 26.3%. Any meaningful discussion on CIFAR100 should begin with a convolutional network (such as ResNet) that achieves reasonable performance (e.g., >70%). The small improvement shown on such a poorly performing model (from 26.3% to 27.0%) is meaningless and does not demonstrate any generalization advantage on modern deep neural networks.

2.  For the text classification task, the authors used logistic regression, a linear model. This again does not represent modern deep learning in the NLP field (e.g., Transformer).

3.  The paper mentions "sharpness-aware minimization" (SAM) in the relevant work section. SAM is currently the most important and relevant baseline in the fields of flatness and generalization. However, the authors made no comparisons with SAM or any of its variants in their experiments. We need to know: how does FA-Regularization compare to SAM in terms of accuracy, computational cost, and final flatness?

4.  The experimental results clearly show that this method is extremely computationally expensive. In the CIFAR-100 MLP experiments, the baseline training time was approximately 60 seconds, while FA-Regularization required ~2200-2400 seconds, about 30-40 times slower. Given such high costs even on these "toy" models, this method is completely infeasible on any real-world CNN or Transformer.

### Questions
Please see the weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1
