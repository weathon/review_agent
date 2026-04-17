# Heterogeneous Transfer Learning with Feature Transformation-Based Adaptation for Modeling Dynamical Systems

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
In this work, a novel heterogeneous transfer learning framework is proposed for modeling dynamical systems, where the source and target domains have different feature spaces. A feature transformation scheme is implemented via customized adaptation layers integrated into the pre-trained model. We conduct theoretical analysis of heterogeneous domain adaptation, demonstrating the generalization performance of the pre-trained model on the target domain after feature transformation. Based on this analysis, a two-phase training strategy is proposed to improve the performance of the heterogeneous transfer learning model. The experimental results in four case studies across different application domains demonstrate the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates heterogeneous transfer learning for modeling dynamical systems, where the source and target domains have different feature spaces. The authors propose a novel framework that integrates customized adaptation layers into a pre-trained model to enable effective feature transformation across domains. A theoretical analysis is provided to evaluate the generalization performance of the transformed model in the target domain. Building on this foundation, a two-phase training strategy is introduced to further enhance adaptation. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper addresses a significant but underexplored problem: the generalization of neural network-based models for complex dynamical systems. Tackling this challenge is valuable and appreciated.
2. The authors provide theoretical guarantees regarding model generalization, which adds rigor to the proposed framework.
3. The designed feature transformation modules for enabling target domain adaptation is simple and efficient.

### Weaknesses
1. The theoretical analysis lacks a strong connection to the characteristics of dynamical systems, especially nonlinear dynamics. 

2. The experimental validation looks too weak. The paper would benefit from more diverse datasets and comprehensive ablation studies to validate the contribution.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel framework for heterogeneous transfer learning (HTL) tailored to modeling nonlinear dynamical systems when the source and target domains have mismatched feature spaces.

### Strengths
The authors provide a detailed derivation of generalization error bounds for HTL using statistical learning theory and derive a two-phase training strategy, in which the custom loss function incorporates multiple theoretical terms, balancing empirical performance with generalization. The framework is modular and easily integrable into existing neural architectures. Adaptation layers are lightweight and interpretable, making the method scalable and practical.

### Weaknesses
1. Remark 1: when the input and output dimensions differ between the source and target domains, how the fact that different features repeated or different strategies to add the zero vectors affects the efficiency of the proposed algorithm?
2. dx in (1) is the dimension of the state vector, and dxs in (2) is the dimension of the model input features of the source domain, how to ensure that Assumption 1 is meaningful?
3. Any function h(\cdot) in the set of hypothesis functions H (Line 163-164 in Page 4): the input of h(\cdot) is any vector in the subspace spanned by x1 to xt, or one of x1 to xt?
4. As shown in (2), Xs, Xt, Ys, and Yt are subspaces. In (4), the authors assume that the corresponding subspaces are equivalent. Are the matrices P and Q nonsingular?
5. What about when replacing the linear transformation in (4) as some nonlinear transformation?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

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
A method for heterogeneous transfer learning on dynamical systems is proposed. It is based on linear transformations of the inputs and outputs. A bound of the generalization error in the target domain is provided. The proposed loss function is based on the insight obtained from the bound. The experimental results support the utility of the proposed method compared to several baseline methods.

### Strengths
- The method is simple.
- The analysis of the generalization error bound may not be overly technically novel but does make sense, and the insight obtained from the bound is neatly utilized in the loss function.

### Weaknesses
Although the technical contribution looks solid, the paper seems to need some updates to clear the bar to appear in ICLR.

**(1)**
Most notably, no ablation studies are presented, due to which we cannot analyze how each part of the proposed method was effective. The loss function has multiple terms, and the proposed algorithm comprises two different phases. The contributions of each of these components of the method should be examined in more detail.

**(2)**
The writing in Section 6 could be polished much more. Currently it rephrases the same thing over and over, and it's hard to extract important information.

**(3)**
The necessity of Assumption 1 is unclear. The matrices $P$ and $Q$ can simply be of sizes $d_{xs} \times d_{xt}$ and $d_{xt} \times d_{xs}$, respectively.

Below are minor things:
- Around Eq. (7) (and on the other occasions too), quantities $B_{V,F}$, $B_{W,F}$, $B_{U,F}$ are used without definition.
- Line 345: "The second term measures the performance of $h^*$ on the empirical source and target domains." ... I don't think so, it instead measure the *difference* of the performances.

### Questions
Do you have any results of ablation studies, investigating the effect of each component of the method?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Heterogeneous Transfer Learning framework that aligns source and target feature spaces through linear adaptation layers added before and after a pre-trained model. Experiments showed the effectiveness of the proposed method.

### Strengths
1. A novel heterogeneous transfer learning framework is proposed to address the feature mismatch between the source and target domains.
2. Theoretical analysis is provided. 
3. Experiments showed the effectiveness of the proposed method.

### Weaknesses
1. Insufficient analysis of data scale and noise sensitivity;
2. Limit in the ablation study.

### Questions
1. All current experiments have been conducted under conditions with sufficient data volume and controlled noise levels. It remains unclear how the proposed HTL framework performs in scenarios with small sample sizes or high noise levels.
2. The paper proposes a two-stage training process (Phase 1 feature adaptation and Phase 2 global fine-tuning). Still, the current experimental section fails to fully validate the independent contributions and synergistic effects of the two stages.
3. The adaptation matrices P and Q are theoretically responsible for implementing linear mappings between the feature spaces of the source domain and target domain. However,  is there a significant difference among all competing ones?
4. All datasets are from continuous control systems. How is the performance of the proposed method in more cross-domain experiments?

### Soundness
3

### Presentation
3

### Contribution
3
