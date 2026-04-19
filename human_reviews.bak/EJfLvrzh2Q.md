# Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
We investigate the mechanisms of self-distillation in multi-class classification, particularly in the context of linear probing with fixed feature extractors where traditional feature learning explanations do not apply. Our theoretical analysis reveals that multi-round self-distillation effectively performs label averaging among instances with high feature correlations, governed by the eigenvectors of the Gram matrix derived from input features. This process leads to clustered predictions and improved generalization, mitigating the impact of label noise by reducing the model's reliance on potentially corrupted labels. We establish conditions under which multi-round self-distillation achieves 100\% population accuracy despite label noise. Furthermore, we introduce a novel, efficient single-round self-distillation method using refined partial labels from the teacher's top two softmax outputs, referred to as the PLL student model. This approach replicates the benefits of multi-round distillation in a single round, achieving comparable or superior performance--especially in high-noise scenarios--while significantly reducing computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors proposed introduce a single-round self-distillation method using refined partial labels from the teacher’s top two softmax outputs, referred to as the PLL student model. This approach replicates the benefits of multiround distillation in a single round. Experiments on several datasets demonstrate the proposed method is effective.

### Strengths
The analysis of self-distillation in linear probing reveals that self-distillation effectively performs label averaging among instances with high feature correlations when generating predictions on the training data.

### Weaknesses
Here are a few questions I would like to ask the authors:

1. What are the clear insights or significances of Sections 2.2, 3, and 4 in relation to the proposed method in Section 5? Could the authors summarize the roles and conclusions of Sections 2.2, 3, and 4, and their implications for Section 5? These sections are not sufficiently clear.

2. In Section 5, the authors state, "selecting the top two labels with the highest values and assigning a weight of 1/2 to each, setting all the other entries to zero." What is the rationale behind this approach?

3. The experiments use ResNet34 as the backbone for linear probe experiments; however, training ResNet34 with all parameters is not particularly time-consuming or resource-intensive. Should a backbone with a larger parameter count be chosen for the experiments?

4. When comparing existing PLL methods, the authors claim, "Our method differs by directly employing the refined partial labels derived from the teacher’s outputs, achieving the same benefits as multi-round distillation in just one round." Does this imply that other methods incorporating the teacher's output could achieve similar or even better results? Are there corresponding experiments to support this?

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper examines the multi-round self-distillation for multi-class classification in the context of linear probing. By approximating the softmax function as a linear function and considering the feature correlation matrix as a low-rank structure, this paper derives a quantified closed-form solution for the output of the $t$-th student model, showing the effect of self-distillation can be interpreted as label averaging among highly correlated instances. The authors then derive the conditions of the label corruption matrix achieving 100% population accuracy for multi-round self-distillation, in the context of balance and superclass corruption. The authors further prove that distilling teacher's top-2 outputs enjoys better theoretical properties. Extensive experiments are conducted and show consistency with the proposed theories.

### Strengths
- The proposed theory is solid and valuable, especially providing insight into the effectiveness of distilling with partial labels.
- The authors provide numerical and visual analysis of the approximation of the softmax function and the feature correlation matrix, which is reasonable and convincing.
- This paper is well-organized and clearly written.

### Weaknesses
In the experiments, Generalized Cross Entropy loss is applied for PLL which differs from the setting of CE loss in the main theory. The authors need to explain whether applying GCE loss is fair for comparison or provide experimental results of CE loss for PLL.

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a theoretical analysis of the mechanisms behind self-distillation in a linear probing setting. The analysis reveals that after $t$ rounds of self-distillation, the model's predictions converge to a weighted average of the provided labels, with the weights determined by the Gram matrix. Building on this finding, the authors investigate the effects of label noise and the efficiency of the self-distillation method. Experiments demonstrate the effectiveness of proposed single-round self-distillation method.

### Strengths
* This paper is well-written, particularly in Section 2, which formulates self-distillation and provides a clear overview of the results.
* This paper presents an interesting result in Theorem 2.1, establishing a connection between the predictions of the $t$-th distilled model and the given (possibly noisy) labels.

### Weaknesses
The main weakness about this paper is the significance of research problem.

* This paper focuses on self-distillation with linear probing and provides a theoretical analysis in this context. I believe that both self-distillation and linear probing are valuable techniques, but I am unclear about the purpose of combining self-distillation with linear probing. As far as I know, linear probing is widely used in self-supervised learning as a method to evaluate learned features. Why should we combine linear probing with self-distillation, especially in scenarios involving label noise?

* The proposed theory assumes a fixed feature extractor and therefore cannot be applied to trainable feature extractors, which are more commonly used in task adaptation and transfer learning.

### Questions
See my questions in weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work aims to analyze self-distillation in linear probing with neural network feature extractors. The contributions of this work are threefold: (1) the analysis reveals that self-distillation effectively performs label averaging among instances with highly correlated features when generating predictions on the training data; (2) the analysis quantifies the number of distillation rounds needed to achieve 100% population accuracy in the presence of label noise; and (3) based on the theoretical analysis, the authors introduce a novel self-distillation approach that achievers similar benefits of multi-round distillation in a single round.

### Strengths
1. Solid Presentation
Overall, this manuscript is well-written and polished. The introduction provides a satisfying overview of the rich theoretical analysis, which is a strong aspect of the work. Some parts of the analysis are somewhat difficult to follow, but experts in this specific field likely won’t struggle to understand the details.

2. Interesting theoretical analysis
One of the main theoretical contributions of this paper is demonstrating that the effect of self-distillation can be interpreted as label averaging among highly correlated instances. This interpretation offers a new perspective on the mechanism of self-distillation, even in the absence of feature evolution. Initially, I thought this was merely a special case of Allen-Zhu and Li’s study. However, after reviewing previous work, I discovered a notable difference in how the effects of self-distillation are analyzed, which I believe is worth sharing with the community.

3. A New Approach to Converting Multi-Round Self-Distillation into a Single Round
The proposed method takes a different approach by directly using the refined partial labels derived from the teacher's outputs, achieving the same benefits as multi-round distillation in just one round. This is especially appealing, as one of the main drawbacks of self-distillation is the need for multiple rounds, which makes training very inefficient. A single-shot method is highly desirable, and we hope this approach will be widely adopted for training classifiers in noisy label settings.

### Weaknesses
1. Most sections are well-written, but some parts would benefit from clearer descriptions or revision. Please refer to the section below.

2. The experiments are somewhat weak, but this is understandable since this work is more theoretical.

### Questions
1. The second contribution of this work is to determine the number of distillation rounds required to achieve 100% population accuracy in the presence of label noise. On first reading, the term "100% population accuracy" may not be immediately clear. Introducing a brief explanation of this concept in the first place would be beneficial for readers who may be less familiar with the subject.

2. Could you explain the intuition behind the condition (Eq. 10) in Theorem 4.1? At first glance, it seems almost implausible to reduce training losses to zero in the presence of label noise, even with multiple rounds of self-distillation. 

3. It seems that the terms “true label” and “given label” are not properly defined. As I understand it, the “given label” refers to the target label, which may include noise, while the “true label” represents the oracle.

### Soundness
3

### Presentation
3

### Contribution
3
