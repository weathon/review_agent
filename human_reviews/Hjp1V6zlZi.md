# Reducing Bias in Feature Extractors for Extreme Universal Domain Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 6, 6

## Abstract
Universal Domain Adaptation (UniDA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain without prior knowledge of the label sets between the two domains. The goal of UniDA is to achieve robust performance under arbitrary label-set distributions. However, existing literature has not sufficiently explored performance across diverse distribution scenarios. Our experiments reveal that existing methods struggle when the source domain has significantly more non-overlapping classes than overlapping ones, a setting we refer to as *Extreme UniDA*. In this paper, we demonstrate that classical partial domain alignment, which focuses on aligning only overlapping-class data between domains, is limited in mitigating feature extractor bias in extreme UniDA scenarios. 
We argue that feature extractors trained with source supervised loss disrupt the intrinsic structure of target data due to the inherent differences between source-private-class data and target data. To mitigate this bias, we employ self-supervised learning to preserve the structure of target data.
This method can be easily integrated into existing frameworks. We apply the proposed approach to two distinct training paradigms—adversarial-based and optimal-transport-based—and show consistent improvements across various class-set distributions, with significant gains in extreme UniDA settings.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This article introduces a new setting for Extreme UniDA, aiming for UniDA methods to perform well even when the source domain has significantly more non-overlapping classes than overlapping ones. The authors analyze current methods and demonstrate their inability to achieve excellent results in such scenarios. They propose the use of self-supervised learning to explore domain-specific private knowledge to eliminate bias and enhance knowledge transfer. Experiments validate the effectiveness of this approach.

### Strengths
1. The problem and new setting proposed in this paper are practically significant and more aligned with real-world scenarios.
2. The motivation of the paper is supported by extensive experimental analysis, providing detailed and convincing evidence.

### Weaknesses
1. Lack of novelty: The fundamental issues discussed in the paper revolve around how to explore private knowledge of the target domain in an unsupervised manner free from the bias of source knowledge, and how to explore when there is no corresponding source domain knowledge for target domain-specific knowledge. However, these are not new problems, and the proposed solutions are not novel. The use of self-supervised learning to explore domain-specific private knowledge has already been discussed in [1][6], and the issue of distancing unique target domain classes from common source domain classes has been explored in [2].
2. The methods compared are too outdated; both the new setting proposed by the author and the general UniDA setting only compare work from 2022 and earlier, completely omitting many more recent studies like [3][4][5].
3. Lack of ablation study: Although only a self-supervised loss function is proposed, it is not compared with other self-supervised learning methods, which makes it difficult to provide convincing evidence.

[1] Sun Y, Tzeng E, Darrell T, et al. Unsupervised domain adaptation through self-supervision[J]. arXiv preprint arXiv:1909.11825, 2019.

[2] Hu J, Tuo H, Wang C, et al. Discriminative partial domain adversarial network[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII 16. Springer International Publishing, 2020: 632-648.

[3] Zhu D, Li Y, Yuan J, et al. Universal domain adaptation via compressive attention matching[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 6974-6985

[4] Zhu D, Li Y, Shao Y, et al. Generalized universal domain adaptation with generative flow networks[C]//Proceedings of the 31st ACM International Conference on Multimedia. 2023: 8304-8315.

[5] Qu S, Zou T, He L, et al. Lead: Learning decomposition for source-free universal domain adaptation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 23334-23343.

[6] Liang J, Hu D, Wang Y, et al. Source data-absent unsupervised domain adaptation through hypothesis transfer and labeling transfer[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(11): 8602-8617

### Questions
1. I hope the authors can engage in a discussion with the paper I mentioned and further demonstrate the novelty of the proposed method.

2. I would like the authors to compare propsoed approach with more advanced methods across multiple settings to demonstrate the performance of their method.

3. Why are experiments not conducted in other settings such as PDA/OSDA/SDA?

4. Please explain the performance differences between the proposed SSL method and other SSL methods (such as rotation/cropping) in these settings, and provide more evidence to demonstrate the superiority of the proposed method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper investigates the problem of Extreme Universal Domain Adaptation (Extreme UniDA), where the source domain contains significantly more non-overlapping classes than overlapping ones. To address this challenge, an in-depth analysis is conducted to understand why the widely-used partial domain alignment paradigm fails in the Extreme UniDA setting. Subsequently, a self-supervised learning plugin is proposed to regularize the feature extractor. Extensive experimental results demonstrate the effectiveness of this mechanism.

### Strengths
1.	The Extreme UniDA setting is new, and the mechanism sounds reasonable.

2.	The experimental results appear highly effective.

### Weaknesses
1.	In Figure 1, which specific methods do “Adversarial Based” and “Optimal Transport Based” refer to? Which dataset did you use?

2.	In section 2.2, which dataset did you use to obtain the results plotted in Figure 4? 

3.	In Figure 2, what are the directions of $e\_1$ and $e\_2$, respectively? Also, I find it difficult to understand what Figure 2 conveys, could you clarify it further?

4.	Can the proposed SSL plugin also be applied to the methods in (Saito&Saenko, 2021; Hur et al., 2023; Lu et al., 2024)?

The paper has several areas that could benefit from improvement. Please consider refining it carefully to address the following example issues:

1.	Line 52: prior works have mainly adhered to the experimental protocols established by by You et al. (2019) -> prior works have mainly adhered to the experimental protocols established by You et al. (2019)
2.	In the titles of Figures 2 and 3, it is better to clarify the notations of $| \overline{C}_s |$, and $| C |$.
3.	Line 206: $\hat{y} (x)$ -> $\hat{y} (\mathbf{x})$

### Questions
See weakness for details.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper underlines Extreme UniDA, a challenging sub-task of UniDA and illustrate that the difficulty of the task roots in the bias in the feature extractor. However state-of-the-art UniDA methods, mostly designed by partial domain alignment that removes irrelevant data by reweighting, cannot completely mitigate the bias on their own for Extreme UniDA. This paper utilizes self supervised learning to enrich the representation with the structural information of the source and target data. Extensive experiments verify that the proposed methodology effectively improves existing partial domain alignment methods across Extreme UniDA settings.

### Strengths
1). This paper underlines Extreme UniDA, a challenging sub-task of UniDA and illustrate that the difficulty of the task roots in the bias in the feature extractor.
2). This paper analyses the limitation of partial domain alignment across Extreme UniDA. 
3). This paper proposes incorporating target label information by self-supervised learning as a lightweight module for partial domain alignment, which can reduce feature extractor bias and significantly enhance robustness across varying class-set distributions.

### Weaknesses
1). Supervised loss is to improve the model performance but will make the model biased to the source domain data, while self-supervised loss is to make the model learn the representation of the target domain data but will affect the classification ability of the model. How to balance these two losses .
2).The comparative work lacks the most recent research efforts, with most being from 2022 and earlier.
3). Although the method proposed in this paper has achieved a good performance on Extreme UniDA, the effect on General UniDA is not obvious, and the method lacks innovation.

### Questions
See above

### Soundness
2

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
3

### Summary
This paper addresses Universal Domain Adaptation (UniDA) in scenarios where source-private classes significantly outnumber source-common classes, termed "Extreme UniDA." The authors propose incorporating self-supervised learning ''to reduce feature extractor bias'' through a consistency loss across augmented views of target samples. The method is evaluated on four standard domain adaptation benchmarks, showing improvements over existing approaches.

### Strengths
- The study addresses an interesting scenario: Extreme UniDA with a high Source-Private to Source-Common Ratio.
- The integration of Self-supervised loss is easy to implement on many frameworks.

### Weaknesses
- Figure 1 lacks essential details such as methods, dataset, transfer tasks, and the y-axis metric used (e.g., source-private vs. target performance). Figures 2 and 3 lack discernible differences; please clarify or consider revising with more distinguishable data points. Figure 4 lacks dataset information and total class count, making the experimental setup difficult to interpret fully.

- The baselines used are old; consider including more recent baselines, such as [1] and [2].

- Although the setup is specific, self-supervision has been previously applied in UniDA, Open-set DA, and partial DA. Could the authors clarify what differentiates their self-supervision approach from that in DANCE and why this choice is preferable?

-  The title references UniDA, but the method is presented through partial DA with SSL for open-set conditions. Additionally, Figure 2 addresses an extreme partial set problem, while Figure 3 shows a vanilla domain adaptation setting. Consistency across the paper could be improved. 

- The paper claims, “Given the distribution shift and label-set shift between the source data Ds and target data Dt, the learned feature extractor θf trained with Ls can become biased towards source-private classes. As a result, this bias may lead to the misclassification of target common-class data as belonging to source-private classes when evaluated on Dt.” This seems misleading. The misclassification is likely due to the dominance of source-private classes, not an inherent model bias. The larger presence of source-private classes increases the statistical likelihood of misclassification, as the model hasn’t learned any target-domain semantics. The noise experiments reinforce this point and should be highlighted more prominently.

-  SSL shows a larger improvement for adversarial methods compared to optimal transport under high SPCR. The paper could benefit from discussing why this difference occurs.

- For reproducibility, could the authors specify which classes are considered source-private, source-shared, and target-private across each dataset in the experimental setup?


[1] Liang Chen, Yihang Lou, Jianzhong He, Tao Bai, and Minghua Deng. Geometric anchor correspondence mining
with uncertainty modeling for universal domain adaptation. In CVPR, 2022

[2] LU, Yanzuo, SHEN, Meng, MA, Andy J., et al. MLNet: Mutual Learning Network with Neighborhood Invariance for Universal Domain Adaptation. In : Proceedings of the AAAI Conference on Artificial Intelligence. 2024.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper address the challenge of Extreme Universal Domain Adaptation (UniDA), where the source dataset has many more unique classes than the target dataset. In these situations, traditional domain adaptation methods often fail because they develop a bias, focusing too much on classes unique to the source data and misclassifying target data.

To fix this, the authors propose adding self-supervised learning (SSL), which helps the model learn the structure of the target data without needing labeled target examples. By integrating SSL into current training methods, the model can better balance both datasets, reducing bias and improving accuracy, especially in difficult, high-bias cases. The results show that SSL is a simple but powerful addition that makes models perform more reliably across different class setups.

### Strengths
This paper addresses the difficult issue of Extreme Universal Domain Adaptation (UniDA). The source dataset has unique classes absent in the target. It introduces self-supervised learning (SSL) to reduce bias in feature extraction and also presents a novel solution. This paper have experiments on benchmarks like Office-Home and DomainNet showing consistent performance improvements. The authors clearly explain SSL's role in feature alignment, making it easy to understand. This work is relevant for real-world applications with mismatched class distributions and offer a straightforward method to enhance model robustness in high-bias situations.

### Weaknesses
The paper addresses limitations in existing partial domain alignment methods for extreme Universal Domain Adaptation (UniDA) but lacks novelty in its self-supervised learning (SSL) approach. The authors should clarify how their application of SSL differs from prior work and emphasize more on unique aspects that tackle class imbalance challenges.

### Questions
How does your application of self-supervised learning (SSL) specifically differ from existing SSL methods in the context of domain adaptation?
How many runs were conducted for the reported experiments, and what measures were taken to ensure the stability of your results?

### Soundness
3

### Presentation
2

### Contribution
3
