# Why do Features of Multi-Layer Perceptrons Condense in Training?

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5

## Abstract
This paper focuses on the problem of feature condensation in early epochs of learning multi-layer perceptrons (MLPs). In fact, the feature condensation is related to many other phenomena in deep learning, and people have some empirical operations to avoid these problems. However, current studies do not well explain essential mechanisms that lead to the feature condensation, i.e., which factors will determine (or alleviate) the feature condensation. The explanation of determinants of feature condensation is crucial for both theoreticians and practitioners. To this end, we theoretically analyze the learning dynamics of MLPs, which clarifies how four typical operations (including batch normalization, momentum, weight initialization, and $L_2$ regularization) affect the feature condensation.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper describes the issue of feature condensation, which refers to the phenomenon that deep MLPs take a long time during training for the loss to decrease, and the network gradients (for each layer) become more and more similar across different samples. The authors build on the fact that the gradients for weight matrices become more and more similar when feature condensation occurs, and provide theoretical analysis as well as empirical evidence on how standard regularization techniques alleviate the feature condensation phenomenon on MNIST, CIFAR, and TinyImageNet.

### Strengths
This paper tackles the occurrence of feature condensation in deep MLPs. This paper is well-written, easy to follow, and the analysis seems correct with strong empirical evidence.

### Weaknesses
I am not sure I buy the contribution of this paper. In particular---
1. Could methods in this paper be extended to more common architectures, such as CNN, Transformer, etc in any way?
2. Does the problem of feature condensation still exist in other architectures like Transformers or even MLP with a residual connection?

I would be happy to change my score if the authors convince me either (1) that there are practical applications for deep MLP, or (2) that the theoretical framework developed in this paper could be extended to modern architectures. Otherwise, I am afraid not enough people in the community will be interested in this paper.

### Questions
See weaknesses

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the feature condensation problem in the training of multi-layer perceptron networks. It defines feature condensation as a phenomenon in which the feature gradient between different class samples becomes similar. It identifies that training of multi-layer perceptrons occurs in two stages. The first stage and the second stage. The paper identifies that feature condensation occurs during the first stage. As the model progresses into the second phase, the similarity in feature gradients for samples from different classes decreases.
Furthermore, the paper explores the impact of deep learning techniques such as batch normalization, $L_2$ regularization, momentum, and weight initialization on the feature condensation problem.
This work also provides an explanation for the potential causes of the condensation problem.

### Strengths
This paper provides detailed mathematical justifications for some of their claims.

### Weaknesses
The novelty of the question explored in this work is not obvious. Figure 1c provides a typical example illustrating what is referred to as 'feature condensation.' This figure showcases the training profile of deep networks, a problem that has been extensively studied under the umbrella of what is commonly known as the 'vanishing gradient problem.' In this scenario, it often takes several epochs before the synaptic weights begin to change significantly. In fact, some of the techniques listed as potential ways to mitigate feature condensation are already well-documented as methods for addressing the vanishing gradient issue or expediting the update process. However, this paper does not offer a comprehensive explanation of the differences or similarities between feature condensation and the vanishing gradient problem.

The concept of feature condensation may not be immediately clear, as the paper lacks a precise definition of the term on its initial pages. Furthermore, this paper mentions a resemblance between its findings and the work by Williams et al. (2019), which studied the 'neural collapse' phenomenon. It remains unclear whether there are significant distinctions between the terms 'neural collapse' and 'feature condensation

### Questions
1). Is feature condensation uniform across all the hidden layers, or is it more pronounced in some layers compared to others?

2). Can you provide insights into how the choice of activation function impacts the feature condensation problem? Are there specific activation functions that are better suited for deep networks due to improved gradient flow? Would you expect the same extent of feature condensation with these functions?

3). What distinguishes the illustration in Figure 1c from the vanishing gradient problem or the dying neuron problem associated with ReLU?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper looks into the phenomenon that intermediate features become increasingly similar in the early training of MLPs. It proposes to explain the phenomenon by noting that there is a "self-enhanced system": if weight vectors in an intermediate layer approximately share a common direction, then the strength of this common direction can become even stronger after further training.

It is also briefly discussed in the paper how common tricks (normalization, momentum, random initialization, L2 regularization) affect the feature condensation phenomenon and provide benefits.

### Strengths
1. Interesting insights into the training dynamics of MLPs in the angle of feature condensation;
2. Qualitative analyses of the effects of common tricks, which could potentially inspire future designs of DNNs.

### Weaknesses
Major Issues:
1. Section 2 begins with "It has been widely observed that the loss decrease of DNNs is likely to have two phases". However, in training SOTA models, the loss starts decreasing almost immediately after the first step, and the authors themselves also noted in appendix that training standard ResNets and ViTs do not suffer from the issue that the loss gets stuck in the beginning, and the feature condensation phenomenon does not appear either. In my understanding, the setting being studied in this paper deviates significantly from the standard practice (no normalization, Xavier init instead of He init, etc.), which could be the main reason that feature condensation can be reliably observed in the paper. I would recommend the authors add more discussion and introduce more background at the beginning of Section 2, instead of just claiming that loss getting stuck is "widely observed".
2. The theory part is not rigorous enough. I believe if the authors make more efforts to make the analyses more formal, they will find many missing gaps in theory. For example, why does weight condensation imply feature condensation? (different input + same weight can lead to different features!) The self-enhanced system of weight condensation makes sense (existence of a dominating common direction -> the common direction becomes even more dominating), but how does the dominating common direction appear from random initialization? How do the analyses ensure that the noise term at each iteration won't be accumulated to be something bigger than the weights in the common direction? I can sort of understand the intuitions provided in this paper for sure, but since the paper claims that it is providing a theoretical analysis of feature condensation, it is better to do it more formally and provide an end-to-end theorem that rigorously proves feature condensation based on assumptions. It would also be better to instantiate the analysis rigorously on some simple and solvable examples.
3. Again, the theoretical analyses of the common tricks have the same issue: many insights are stated without any quantitative analyses. These analyses are thus not strong enough to be called theoretical analyses.

Minor Issues:
1. In the introduction, the summary of Williams et al. (2019); Lyu et al. (2021); Boursier et al. (2022); Wang & Ma (2023) is not very fair: some of these papers are only assuming the data is linearly separable. "the feature direction of each arbitrary training sample could act as a perfect linear classifier" seems to be an even stronger assumption. Does it refer to [orthogonally separable data](https://openreview.net/forum?id=krz7T0xU9Z_)?
2. It is discussed in Section 3.3 how common tricks affect feature condensation, but what is missing in this part of the paper is a sanity check to see whether less feature condensation is indeed beneficial to generalization. At least for the effect of the initialization scale, it is not very clear whether reducing feature condensation by increasing the initialization scale can lead to better generalization --- larger initialization means the neural net will stick to random features, which may not be of good quality.

### Questions
My main concern is that the theoretical analyses are not rigorous enough and I wonder if the authors have any plan to resolve this issue.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
