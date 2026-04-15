# Warped Convolutional Neural Networks For Large Homography Transformation with $\mathfrak{sl}(3)$ Algebra

- Decision: Reject
- Scores: 6, 5, 8

## Abstract
Homography has fundamental and elegant relationship with the special linear group and its embedding Lie algebra structure. However, the integration of homography and algebraic expressions in neural networks remains largely unexplored. In this paper, we propose Warped Convolution Neural Networks to effectively learn and represent the homography by $\mathfrak{sl}(3)$ algebra with group convolution. Specifically, six commutative subgroups within the $SL(3)$ group are composed to form a homography. For each subgroup, a warp function is proposed to bridge the Lie algebra structure to its corresponding parameters in homography. By taking advantage of the warped convolution, homography learning is formulated into several simple pseudo-translation regressions. Our proposed method enables to learn features that are invariant to significant homography transformations through exploration along the Lie topology. Moreover, it can be easily plugged into other popular CNN-based methods and empower them with homography representation capability. Through extensive experiments on benchmark datasets such as POT, S-COCO, and MNIST-Proj, we demonstrate the effectiveness of our approach in various tasks like planar object tracking, homography estimation and classification.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes Warped Convolutional Networks (WCN) to learn and represent homography by sl(3) algebra with group convolution. The homography is decomposed into six commutative subgroups within SL(3), and the corresponding warp functions are designed to effectively recover the one or two-parameter groups. The experiments on classification, homography estimation, and planar object tracking tasks have shown the superiority of the proposed method.

### Strengths
++ This paper shows a good connection between warped convolutions and the SL(3) induced homography.

++ WCN demonstrates better robustness and data efficiency against prior works through comprehensive experiments.

### Weaknesses
-- page 7: "improved performance from 0.69 to 14.72"

-- The MACE of PFNet* in Table 2 is 1.20, while it is 1.21 in the main text of page 8.

### Questions
1) Does the order of the decomposed subgroups affect the results?

2) Can the estimators be learned in parallel instead of sequentially? The current implementation seems that later estimators need to be conditioned on earlier estimators.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces warped convolutional neural network to deal with the task of large homography transformation estimation under SL(3) algebra. 6 commutative subgroups within the SL(3) group are composed to form a homography. For each subgroup, a warp function is proposed, bridging the Lie algebra
structure to its corresponding parameters in homography. Experiments are conducted on the tasks such as classification, homography estimation and planar object tracking.

### Strengths
The proposed Lie representation for homography estimation is new and interesting.

### Weaknesses
The current presentation is not clearly show its advantages over previous homo representations. Please see questions

### Questions
“homography learning is formulated into several simple pseudo-translation regressions”The 4pt representation (corner offsets) of homography estimation is already a 4 translation vector representation, where 4 translational vectors at the 4 corner of an image can uniquely defines a homography. A homography matrix is obtained by solving a DLT 

According to Eq. 5, b_i should be estimated in order to estimate a homography. However, these b_i is not with the same value range, for example, the translation b1 and b2 may orders larger than perspective components, b7 and b8, which introduces training difficulty.  This is why previous works adopt corner offsets that share a similar value range for the homography estimation instead of regressing homography matrix elements.  I'm not sure working in the Lie space could solve such problem. 

Assume it can, why it is better than corner offsets or homography flow representation is still not clear. Saying that "incapable of estimating the large transformation" is not accurate, given that some works already adopted corner offsets or homography flows for large homography transformations, e.g., 

Jiang et al. Semi-supervised Deep Large-baseline Homography Estimation with Progressive Equivalence Constraint, AAAI 2023

Nie et al. Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation, TCSVT 2021

For experiments, more recent homography dataset should be adopted, e.g., the CA-Homo dataset, or AAAI 23 dataset, should be compared with, in which various deep-based, traditional feature-point matching, deep feature matching based approaches are reported. 

Authors should put some efforts to demonstrate the limitations of previous homography representations, so as to show the advantages of the proposed new representation. These previous representations, either corner offsets, or homography flow, on the one hand, can achieve large baseline registration already, on the other hand, are flexible to be extended for multiple planes, e.g., mesh local homos, that can go beyond the single-plane transformation. The flexibility is also important.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel approach, named Warped Convolutional Neural Networks (WCN), for effectively learning and representing homography in neural networks through algebraic expressions. The proposed method enables the learning of features that remain invariant to significant homography transformations and can be easily incorporated into popular CNN-based methods. The paper thoroughly analyzes the proposed approach, including the warp function and its properties, implementation details, as well as extensive experimental results on benchmark datasets and tasks. The contributions of this paper encompass a fresh perspective on homography learning utilizing algebraic expressions, the introduction of a novel warped convolutional layer, and a comprehensive evaluation of the proposed method across various benchmark datasets and tasks.

### Strengths
1. This paper establishes a sophisticated and elegant relationship between homography and the SL(3) group along with its Lie algebra.
2. The formulation of the homography and the underlying warping functions proposed in this paper demonstrate technical soundness.
3. The proposed WCN method for estimating homography parameters is logical and well-founded.
4. Extensive experiments are performed on various tasks and datasets, successfully validating the effectiveness of the proposed method.

### Weaknesses
1. In terms of novelty, this paper bears resemblance to the work of Zhan et al. (2022). Zhan et al. employed a similar approach, utilizing two groups for estimation, whereas this paper proposes the use of six groups for the same purpose. As a result, the contribution of this work can be seen as somewhat incremental. More clear discussion should be given. 

2. It is desirable for this paper to provide additional elaboration on the mathematical aspects associated with the proposed method, with the aim of enhancing comprehension for individuals who are not familiar with this particular field. The inclusion of more accessible explanations and intuitive examples would be beneficial in ensuring that the content is more easily understood by a broader audience.

### Questions
Please check the weakness listed above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
