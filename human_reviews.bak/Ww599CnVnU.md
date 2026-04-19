# Multi-scale Minimal Sufficient Representation Learning for Domain Generalization in Sleep Staging

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 5

## Abstract
Deep learning-based automatic sleep staging demonstrates strong performance as a promising solution for diagnosing sleep disorders. However, deep learning models often struggle to generalize on unseen subjects due to variability in physiological signals, resulting in degraded performance in out-of-distribution scenarios. To address this issue, domain generalization approaches have recently been studied actively to ensure generalized performance on unseen domains during the training. Among those techniques, contrastive learning has proven its validity in learning domain-invariant features by aligning samples of the same class across different domains. Despite its potential, many existing methods are insufficient for extracting truly domain-invariant representations, as they do not explicitly reduce domain-relevant information embedded in the features. In this paper, we argue that addressing superfluous information is a key to bridging the domain gap. Furthermore, existing methods often neglect the multi-scale nature of sleep signals, potentially missing important temporal and spectral characteristics. To address these limitations, we propose a novel Multi-Scale Minimal Sufficient representation learning (MSMS) framework, which effectively reduces domain-relevant information while preserving essential temporal and spectral features for sleep stage classification. We evaluate our method on publicly available sleep staging benchmark datasets, SleepEDF-20 and MASS. Experimental results demonstrate that our approach consistently outperforms state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a domain generalization method for sleep stage prediction, multi-scale minimal sufficient representation learning (MSMS).
The basic approach of MSMS is learning domain-invariant representation as existing domain generalization methods.
This paper argues that the representations learned by existing methods still contain domain-relevant information called superfluous information.
Thus, the proposed method explicitly incorporates the superfluous information into the objective.
Another idea of the proposed method is learning domain-invariant representation in the final encoder output space and intermediate feature spaces because capturing multi-scale characteristics of sleep signals is important in sleep stage prediction.

### Strengths
- S1: The motivation is clear and easy to understand.
- S2: Introducing the superfluous information is interesting.

### Weaknesses
- W1: Is the superfluous information minimization effective in the proposed method? In the final loss function in Eq. (10), only the terms corresponding to $I(z_i;d_i)$ and $I(z_i;v_p)$ in Eq. (3) are included.
- W2: Distinguishing the roles of $I(z_i;d_i)$ and $I(z_i;v_i|v_p)$ would make the proposed method more convincing since both terms are explained as domain-relevant information. It would also be beneficial to perform ablation for $I(z_i;d_i)$ and $I(z_i;v_i|v_p)$. Since the superfluous information strongly correlates with $I(z_i;d_i)$, clarifying the efficacy of using not only $I(z_i;d_i)$ but also $I(z_i;v_i|v_p)$ would be intriguing.
- W3: Comparing with the empirical risk minimization (ERM), i.e., merging the datasets of all domains and simply training with a supervised manner without domain labels, would make the result persuasive because Gulrajani et al. [a] reveal that ERM performs the best when the experiment is carefully designed.
- W4: The derivation of the minimal sufficient representation learning would be more understandable if the relationships between ${x}, {v_1}, {v_2}, {z_1}, {z_2}$ are illustrated as a graphical model.
- W5: Displaying the amount of superfluous information and domain-relevant information in Tab. 1 would make the result more convincing.
- W6: Describing the procedure of the proposed method in more detail, e.g., how to compute $H(z_i|d_i)$, would make the paper more easy to follow.
- W7: Although $\lambda_1$ is set to 1 for neglecting $H(z_i)$ in Eq. (25) to derive Eq. (4), checking the effect of $H(z_i)$ would be beneficial.

[a] Ishaan Gulrajani and David Lopez-Paz. "In search of lost domain generalization." ICLR 2021.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose a new sleep staging method based on contrastive learning called MSMS. This method is inspired by the model SleepPyCo and tries to remove superfluous information from the representation of samples. This should help to reduce the domain shift in the data. The paper proposes several experiments on two classical sleep staging datasets to show the benefits of their methods over competitors.

### Strengths
- The paper tries to tackle the distribution shift in sleep staging, which is a crucial issue in the biosignal field. 
- The illustrative figures look nice.

### Weaknesses
- Figure 1 is a bit hard to understand when not reading the paper
- The paper, in general, lacks clarity.
- Related work is unclear. The authors didn't appropriately introduce the basis of contrastive learning, making understanding their paper hard.
- The notation is a bit messy. For example, in the notation part (which should be before related work), $D_m$ represents all samples for one domain, while after $D$ seems to represent the concatenated domains. After that $D_m$ is never used.
- If adapting between subjects could be challenging, there is a more significant distribution shift between datasets. Several papers deal with adaptation between datasets [1, 2, 3]. Using sleep staging for domain generalization without generalizing it to another dataset does not seem engaging.


[1] Eldele et. al. ADAST: Attentive Cross-domain EEG-based Sleep Staging Framework with Iterative Self-Training, IEEE TETCI, 2022
[2] Gnassounou et. al., Convolutional Monge Mapping Normalization for learning on sleep data, Neurips, 2023
[3] Wang et. al., Generalizable Sleep Staging via Multi-Level Domain Alignment, AAAI, 2024

### Questions
- Why did you choose this architecture for your model? Does already existing work inspire you?
- The results of MSMS are slightly above other methods. Did you do a statistical test to see if the improvement is significant?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper solves the domain adaptation problem in the context of sleep stage classification using EEG. In particular, the authors proposed  a Multi-Scale Minimal Sufficient representation learning (MSMS) framework that reduces domain-relevant information while preserving essential temporal and spectral features. Evaluations are conducted on publicly available sleep staging benchmark datasets, SleepEDF-20 and MASS to demonstrate the effectiveness of the proposed method.

### Strengths
* This problem the authors aimed to solve is practical.
* This paper is easy to follow.

### Weaknesses
* a) Limited novelty. Although the authors spent majority of the space talking about mutual information (MI) and superfluous information. They ended up adding an extra term of domain conditioned entropy in addition to the well known contrastive loss. Adding regularisation in the context of domain adaptation is not new. For instance, Domain-Adversarial Neural Networks (DANN) added an inverse gradient to confuse features extracted from two domains. The so called MULTI-SCALE is essentially applying contrastive loss to intermediate features which is not new in both contrastive learning or domain adaptation.

* b) Poor clarity. The authors spent too much space on MI which is in the end intractable. Instead, the authors should put more focus on describing the details of how exactly their framework works. For instance, it seems that the proposed framework is a dual-stage framework, how is the second stage trained and using what data ? How is H(z|d) calculated? What is the network structure used ? 

* c) Marginal improvement. As seen in Table 1, on SleepEDF-20, the improvement over the second best approach is 0.4%. On MASS, the improvement is 0.3%. 

* d) Insufficient ablation studies. The authors proposed two novelties in this work, a regularisation term and the application of the loss function (which layers). These should be studied individually to see the contribution of each.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposed domain generalization method for sleep staging analysis automatically. The main way to learn domain-invariant features are through contrastive learning however the problem of existing methods is unable to reduce domain relevant information embedded in the features. In addition, the claimed challenge is existing methods neglect multi-scale nature of sleep signals, missing temporal and spectral features. The paper proposes a multi-scale minimal sufficient representation learning framework. This framework has two functions, including the domain-relevant information reduction; and the temporal and spectral features for sleep stage classification.

### Strengths
-	The method and theorem seem correct
-	The writing and organization seem clear

### Weaknesses
-	One of the two main issues is the unclear motivations. In the introduction, especially the fig1a and 1b, it is ambiguous and unclear to understand your proposals. How to estimate the real and genuinely domain-invariant representations? This needs very clear and exact computations or experimental proofs, rather than using the toy-example figures. 

-	The second of the two main issues is in the novelty. The information bottleneck combined with the contrastive learning has been widely and commonly used in general machine learning and representation learning. It is clear that this solution may combined the existing work and method into the sleep staging. In addition, how to handle the genuine sleep staging characteristics in the method design is missing. Integration of existing method into the application is the major issue.

### Questions
see weaknesses and answer please item by item.

### Soundness
3

### Presentation
2

### Contribution
2
