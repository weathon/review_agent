# Generalized Category Discovery  Utilizing Reciprocal Learning and Class-wise Distribution Regularization

- Decision: Reject
- Scores: 6, 6, 5, 6

## Abstract
Generalized Category Discovery (GCD) aims to identify unlabeled samples by leveraging the base knowledge from labeled ones, where the unlabeled set consists of both base and novel classes. 
Since clustering methods are time-consuming at inference, parametric-based approaches have become more popular. 
However, recent parametric-based methods suffer inferior base discrimination due to the unreliable self-supervision. 
To address this issue, we propose a Reciprocal Learning Framework (RLF) that introduces an auxiliary branch devoted to base classification. 
During training, the main branch filters the pseudo-base samples to the auxiliary branch. 
In response, the auxiliary branch provides more reliable soft labels for the main branch, leading to a virtuous cycle. 
Furthermore, we introduce Class-wise Distribution Regularization (CDR) to mitigate the leaning bias towards base classes. 
CDR  essentially increases the prediction confidence of the unlabeled data and boosts the novel class performance. 
Combined with both components, our method achieves superior performance in all classes with negligible extra computation. 
Extensive experiments on seven GCD datasets validate the effectiveness of our method, e.g. delivering a notable 2.1\% improvement on the Stanford Cars dataset.
Our codes will be available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a Reciprocal Learning framework, introducing an auxiliary branch for base classification and class-wise distribution regularization to mitigate learning bias towards base classes. Extensive experiments on various datasets demonstrate the effectiveness and benefits of the proposed method.

### Strengths
The paper is easy to read and follow, and the results are convincing.

The proposed method is interesting, especially class-wise distribution regularization.

### Weaknesses
1. Could you provide an intuitive understanding of $m_k$?
2. It seems that the KL loss is similar to the approach by [1], which distills knowledge from a supervised model. Could you offer a more detailed analysis of the differences?
3. Experiments with DINOv2 are needed to better demonstrate the advantages of your methods.


[1] Gu et al. Class-relation Knowledge Distillation for Novel Class Discovery. ICCV2023.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a novel approach to address the challenge of Generalized Category Discovery (GCD) by utilizing Reciprocal Learning (RLF) and Class-wise Distribution Regularization (CDR). In RLF, the authors address the issue of poor base discrimination in current methods by introducing a base-only classifier and distilling the distributions between this classifier and the main classifier on filtered pseudo-base data. In CDR, the authors enhance novel class performance by maximizing the similarity between class-wise expected distributions from two views. Experimental results highlight the superiority of this method over existing state-of-the-art techniques in both effectiveness and generalizability.

### Strengths
1. The motivation behind the Reciprocal Learning framework is clearly articulated, and the solution is both simple and efficient.
2. The design of class-wise distribution regularization is elegant, and its improvements are impressive.
3. The method achieves significant performance gains, demonstrating its effectiveness.
4. The paper is well-written and easy to follow.

### Weaknesses
1. While class-wise distribution regularization serves as an effective regularization term, its motivation is unclear. The authors claim it primarily addresses the learning bias towards base classes caused by RLF and sharpens model predictions. However, CDR is applied to both branches, including the base classifier branch where it affects only the pseudo base classes. This could potentially reinforce the bias towards base classes and is contradictory to the stated motivation. Further clarification and an ablation study focusing on the impact of CDR in each branch would be beneficial. For instance, it would be useful to compare the effects of applying CDR exclusively to the main branch versus its application across both branches.
2. The reciprocal learning framework bears a resemblance to the approach presented in [1]. In that work, an additional base classifier head is trained with labeled base class data, and the distribution between this base-class classifier and the main classifier is distilled with a weight function (Eq. 8 in [1]) to reduce the impact of noisy unlabeled data learning. While there are differences in details, such as the design of the weight function (specifically, $\max(\mathbf{p}_{b, i}^{aux})$ in Eq. 5 of this paper versus Eq. 8 in [1]) and the training method of the base classifier, the framework and the final loss function (Eq. 5 in this paper versus Eq. 9 in [1]) are quite similar. A thorough analysis of the necessity of these modifications and a discussion of the novelty claim would strengthen the contribution.

[1] Peiyan Gu, Chuyu Zhang, Ruijie Xu, and Xuming He. Class-relation knowledge distillation for
novel class discovery.

### Questions
Main questions and suggestions:

Please refer to the Weaknesses section for detailed questions and suggestions.

Minor questions and suggestions:
1. I noticed that the contrastive learning loss is applied only to labeled data (Eq. 3). Could you explain the reason for excluding it from unlabeled data?
2. In Eq. 5, is the item $\max\(\mathbf{p}_{b,i}^{aux}\)$  detached during the computation?
3. The method as a whole lacks a distinctive name. It would be beneficial for clarity and reference if the authors could provide a name for their approach.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel method called RLF to improve the reliability of pseudo-labels in Generalized Category Discovery (GCD) tasks. RLF incorporates an AUX token that learns base categories to enhance pseudo-label quality, along with a CDR module to regularize novel samples. Extensive experiments and ablation studies across multiple datasets demonstrate the effectiveness of this approach, showing superior performance in GCD tasks.

### Strengths
1.	This article is well-written and well-organized.
2.	The paper addresses the often-overlooked bias in pseudo-labeling through the proposed AUX token and regularization.
3.	The proposed method significantly improves performance and provides more reliable pseudo-labels for classifier learning.
4.	The paper includes comprehensive quantitative and qualitative visualizations to demonstrate the effectiveness of the method.

### Weaknesses
1.	Missing Relevant Works: The paper overlooks key recent works like CMS and InfoSieve.
2.	Missing Self-Contrastive Learning (CL): SimGCD also uses self-CL in representation learning, but this is absent in Section 3.1.
3.	Lacking Deep Reason Analysis: The authors aim to reduce bias in pseudo-labels but don't analyze the root causes deeply, even though they explain the motivation in Section 3.2-whether it's due to weak representations, class maximum regularization in SimGCD, or something else.
4.	Confusing Sentence: What is the meaning of K in line 234: ‘K is the number of base and novel classes, while K denotes the number of base classes in the auxiliary branch’

### Questions
See weaknesses 3 and 4.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents an improved knowledge distillation strategy for generalized category discovery (GCD), focusing on learning a better model of the known classes. Built on SimGCD, this method first introduces an auxiliary branch for the base class prediction, which distills its knowledge into the full classifier. Moreover, it incorporates a class-wise distribution regularization term to alleviate the bias toward known classes. The proposed GCD framework is evaluated on 7 datasets with comparisons to the prior arts and ablation studies on its main components.

### Strengths
1. The method is clearly motivated and its findings seem interesting for GCD. Based on this observation, the proposed design is reasonable. 
2. The experimental results indicate that the method achieves strong performance on the benchmarks, outperforming most of the baselines.

### Weaknesses
1. The novelty of this work is limited. The idea of distilling known-class knowledge to improve class discovery has been explored in Gu et al, ICCV 2023, which uses a separate branch for base/known classes. Moreover, the class-wise distribution regularization technique is similar to Zhang et al. ICML 2024, which includes several properties, such as Theorem 1 in this work. While this work integrates those ingredients into SimGCD, the overall contribution seems rather limited.
2. The discussion of the related work is insufficient. Section 2 simply lists previous methods without proper comparisons to the proposed strategy.
3. There are many hyper-parameters in this method and it is unclear how they are tuned.  
4. The experimental evaluation is lacking in the following aspects: 
    - The baselines are insufficient. A baseline derived from NCD (Gu et al., ICCV 2023) should be included to validate the design of the AUX token. 
    - The claim of this work on enhancing base classification is not well supported by the results in Table 1&2, where the base class performances are much lower than previous methods in several cases. 
    - The visualization in 4.3 is not very informative as it only includes baseline methods. It should compare with the SOTA methods. 
    - The ablation study shows that AUX often does not have a positive impact on the model performance. Is this component necessary for the method? It would be interesting to show the ablation results for the full model minus each component. 
    - It would be more convincing to include the full results on 7 datasets for the analysis on different regularizations (ln 426) and estimated class numbers (ln 454). 
5. The paper's presentation needs improvement. Some terminologies are used without proper explanation, for example: 
    - ln 052: "base prediction of base unlabeled data".
    - ln 072: "pseudo-base samples"?
    - ln 199: "concentrating it with CLS ..."? 
    - Theorem 2. "converges to zero": should be “equals to".

### Questions
Please address the comments in the weaknesses section above.

### Soundness
3

### Presentation
2

### Contribution
2
