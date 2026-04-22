# Riemannian Networks over Full-Rank Correlation Matrices

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Representations on the Symmetric Positive Definite (SPD) manifold have garnered significant attention across different applications. In contrast, the manifold of full-rank correlation matrices, a normalized alternative to SPD matrices, remains largely underexplored. This paper introduces Riemannian networks over the correlation manifold, leveraging five recently developed correlation geometries. We systematically extend Multinomial Logistic Regression (MLR), Fully Connected (FC), and convolutional layers to these geometries. Additionally, we present methods for accurate backpropagation for two correlation geometries. Experiments comparing our approach against existing SPD and Grassmannian networks demonstrate its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper extends MLR, FC, and convolutional layers in Riemannien Networks to correlation manifold under five geometries: ECM, LECM, OLM, LSM and PHCM, and discuss backpropagation of Riemannian computations under OLM and LSM. The authors evaluate thier method on 3 benchmarks Radar, HDM05 and FPHA.

### Strengths
The paper is coherent and methodologically sound. The mathematical formulations are clearly presented and internally consistent. The extension of fully connected layers, multinomial logistic regression, and convolutional operations to the correlation manifold represents, in my view, an original contribution.

### Weaknesses
The experiments are limited, only main results and an ablation study are presented. The contributions are fair, adopting Correlation matrix in Riemannian Networks is mainly an extension of existing SPD or hyperbolic formulations.

### Questions
1. In main results, the authors report the best results comparing to methods, Grassmannian and SPD manifolds, I wonder if the authors compare the proposed CorNet with other methods in Correlation Manifold?
2. The authors claim accurate backpropagation of Riemannian computations under OLM and LSM as one main contribution, but I didn't see experiments evaluating this point.
3. The authors needs more interpretations on fig.2 and fig.5
4. Can the authors provide more evidence justifying the novolties

### Soundness
3

### Presentation
3

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
This paper introduces Correlation Networks (CorNets): a new family of deep learning architectures operating on the manifold of full-rank correlation matrices. The authors extend key neural network components to this manifold under five distinct Riemannian geometries (four zero-curvature and one hyperbolic). They also propose accurate backpropagation schemes for their layers. Experimental results on radar and human action recognition datasets demonstrate that CorNets outperform existing SPD and Grassmannian networks in both accuracy and efficiency.

### Strengths
- The paper introduces a wide framework to implement neural networks on full-rank correlation matrices. It considers multiple geometries and layer types and distinguishes between them. Thus it introduces a wide toolbox that could be useful in various scenarios.
- This framework is presented in a detailed manner. The paper includes all the necessary details to understand and reimplement both the method and the experiments.
- The experiments are thorough and well-documented.  
- The introduced CorNets show improved performance over SPD and Grassmanian method on a variety of datasets.

### Weaknesses
For me, the paper has two crucial weaknesses:

a) The presentation is in some parts very dense. Especially, when introducing the various layers it is mostly a quick listing of facts and for me it was hard to form a conceptual picture that lasts beyond the specific methods. At the same time, I also found this makes it less pleasant to read for me. Maybe the authors could introduce more structure lists and tables in the main text, move the propositions etc to the appendix and discuss more conceptual things in the main text?

b) To me the examples in the experimental validation do not seem to be that interesting. The datasets are rather old, and the performance of previous methods seems to already be very good on two of them. It is interesting to see, that the method is noticeably better than competitors on HDM05. However, to me, it is not clear why this is and I would recommend the authors to discuss this more in the paper and at least propose some hypothesis. Furthermore, I miss a comparison to method not based on matrix manifolds for the same tasks. This is, for me, a common issue with paper working on neural networks for matrix manifolds and makes it hard to learn when such networks are appropriate tools. To mitigate this, the authors could include a more thorough discussion on when correlation matrices are most appropriate, when other matrices, and when maybe purely Euclidean methods suffice.

I still think that the paper deserves to be published at ICLR. Nevertheless, these concerns move me closer to the decision boundary.

### Questions
- Suggestion: I think it would be helpful to provide pseudo-code on how to implement the method. For me, this would drastically reduce the time necessary for going from reading the paper to using it in my context. Furthermore, I believe that pseudo-code is also a more permanent way to document implementations than a link to some repository.
  - Suggestion: I think tables for the layers introduced in  (7) and (9)-(12) containing their parameters and formulas would be helpful.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper extends Multinomial Logistic Regression (MLR), Fully Connected (FC), and convolutional layers to the geometry of full-rank correlation matrices. The proposed networks are compared to Symmetric Positive Definite (SPD) and Grassmannian neural networks to show their effectiveness.

### Strengths
- This paper concerns with the manifold of correlation matrices that is underexplored in deep learning.
- The construction of network building blocks is supported by theoretical results.

### Weaknesses
- The novelty is limited.
- More experiments are needed to validate the proposed networks.

### Questions
The aim of this paper is to construct MLR, FC, and convolutional layers for neural networks on the manifold of correlation matrices. Unfortunately, I failed to see new ideas in the construction of these building blocks for the following reasons:
- Riemannian MLR layers on hyperbolic space were proposed in [Ganeal et al., 2018], then extended to matrix manifolds in [Nguyen & Yang, 2023].
- FC and convolutional layers on hyperbolic space were proposed in [Shimize et al., 2021], then extended to matrix manifolds in [Nguyen et al., 2024].

The present work follows the same approach in [Shimize et al., 2021; Nguyen & Yang, 2023; Nguyen et al., 2024] to adapt the layers on hyperbolic space to the setting of matrix manifolds. What is new in the present paper is that they authors deal with the geometry of correlation matrices. However, since correlation matrices are normalized SPD matrices, and the geometry of the former is thoroughly studied [David & Gu, 2019], it is straighforward to derive formulas and theoretical results in the considered setting. 

Regarding experimental evaluations, I am not convinced by the comparison of the proposed networks and SPD neural networks for the following reasons:
- All datasets are of small size.
- Intuitively, I can't quite see why the proposed neural networks can outperform SPD neural networks in terms of accuracy, since correlation matrices are nothing but normalized SPD ones. I think improvements can be obtained in specific cases but it is not systematic. 

To summary, the greatest weakness of the paper is its limited novelty which explains my rating for the paper. 

Question: 

1. Could you give an intuitive reason why neural networks on the manifold of correlation matrices can be more effective than their SPD counterparts in terms of accuracy ?

2. I am wondering if the proposed networks and their SPD counterparts share the same architecture ? This question is related to one of my comments above.

### Soundness
3

### Presentation
3

### Contribution
2
