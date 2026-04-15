# Representation Norm Amplification for Out-of-Distribution Detection in Long-Tail Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 5

## Abstract
Detecting out-of-distribution (OOD) samples is a critical task for reliable machine learning. However, this task becomes particularly challenging when the models are trained on long-tailed datasets,  as the models often struggle to distinguish tail-class in-distribution  samples from OOD samples. We examine the main challenges in this problem by identifying the trade-offs between OOD detection and in-distribution (ID) classification, faced by existing methods. We then introduce our method, called Representation Norm Amplification (RNA), which solves this challenge by decoupling the two problems. The main idea is to use the norm of the representation as a new dimension for OOD detection, and to develop a training method that generates a noticeable discrepancy in the representation norm between ID and OOD data, while not perturbing the feature learning for in-distribution classification. Our experiments show that RNA achieves superior performance in both OOD detection and classification compared to the state-of-the-art methods, by 2.36\%, 1.17\%, and 7.38\% in AUROC and 2.20\%, 0.95\%, and 2.84\% in classification accuracy on CIFAR10-LT, CIFAR100-LT, and ImageNet-LT, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper tackles the significant issue of out-of-distribution (OOD) detection within the realm of long-tail learning. It introduces an innovative approach known as Representation Norm Amplification (RNA), with the goal of separating OOD detection from classification. This effectively resolves the trade-offs between OOD detection and in-distribution (ID) classification commonly encountered by existing methods. The main idea of RNA relies on the norm of representation vectors as a new dimension to differentiate between ID and OOD data, ultimately delivering superior performance in both OOD detection and classification.

### Strengths
(1) This work is well-organized and features a clear presentation. The problem statement concerning the challenges of out-of-distribution (OOD) detection in the context of long-tail learning is precisely defined and effectively underscores the motivation behind the proposed method RNA.

(2) The RNA approach proposed in this paper, which leverages representation norms to separate OOD detection from classification, is both innovative and logically grounded. This approach addresses the trade-offs encountered in existing methods and exhibits enhanced performance.

(3) The study presents comprehensive experimental results across multiple datasets, showcasing the effectiveness of RNA in improving both OOD detection and classification accuracy.

### Weaknesses
(1)  This work lacks a more detailed description of the proposed RNA method, such as a pseudocode algorithm, which would enhance the clarity of the paper and help explain the key steps of the proposed approach.

(2) While the paper mentions existing methods, it would significantly benefit from a comparison with more recent state-of-the-art work in both OOD detection and long-tail learning, e.g., [1, 2]. This would elevate the quality and relevance of the paper, providing a more up-to-date context for the proposed method.

(3)  Including additional explanations and visualizations of how representation norms are amplified of the network architecture would improve the interpretability of the model's decisions. This would make the paper more valuable by providing insights into the inner workings of the RNA method.

(4) The work is expected to consider a more comprehensive evaluation of larger and more complex datasets, such as iNaturalist [3]. This would provide a broader assessment of RNA's performance and its applicability to real-world scenarios.

[1] Out-of-Distribution Detection with Deep Nearest Neighbors, ICML 2022

[2] POEM: Out-of-Distribution Detection with Posterior Sampling, ICML 2022

[3] The iNaturalist Species Classification and Detection Dataset, CVPR 2018

### Questions
(1) Could the authors provide a pseudocode algorithm or a more detailed step-by-step explanation of the RNA method? This would enhance the clarity of the practical implementation of the approach.

(2) It would be valuable if the paper could incorporate a comparison with more recent state-of-the-art methods in both OOD detection and long-tail learning to provide a more comprehensive context for the significance of the proposed RNA method.

(3) Could the authors offer additional explanations or visualizations illustrating how representation norms are amplified in the RNA method and how this amplification process influences the network's decisions? This would contribute to a better understanding of the model's interpretability.

(4) The paper discusses the trade-offs between OOD detection and long-tail recognition. It would be beneficial for the authors to provide further elaboration on how RNA specifically addresses these trade-offs and achieves a balance between the two objectives. Additionally, insights into how RNA's performance varies with different levels of class imbalance in the long-tail datasets would be valuable.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper discusses the problem of detecting out-of-distribution (OOD) samples in long-tail learning, where models struggle to distinguish tail-class in-distribution samples from OOD samples. The authors introduce a method called Representation Norm Amplification (RNA) to address this challenge by decoupling OOD detection and in-distribution classification. RNA uses the norm of the representation as a new dimension for OOD detection and develops a training method that generates a noticeable discrepancy in the representation norm between ID and OOD data. Experimental results show that RNA outperforms state-of-the-art methods in both OOD detection and classification on CIFAR10-LT, CIFAR100-LT, and ImageNet-LT datasets. The main contributions of the paper are the introduction of RNA as a novel training method and the evaluation of RNA on diverse OOD detection benchmarks.

Contributions:
1. RNA decouples OOD detection and in-distribution classification, different from existing methods struggling to distinguish tail-class in-distribution samples from OOD samples.
2. RNA uses the norm of the representation as a new dimension for OOD detection, which is a noticeable discrepancy in the representation norm between ID and OOD data.
3. Experimental results show that RNA outperforms state-of-the-art methods in OOD detection and classification.

### Strengths
Strengths
1. Originality: The proposed method, Representation Norm Amplification (RNA), introduces a new dimension, the norm of representation vectors, for OOD detection. This approach decouples the OOD detection problem from the long-tailed recognition problem, allowing for the simultaneous achievement of both goals without compromising each other.
2. Quality: The results presented in Tables 6, 7, and 8 demonstrate the superior performance of RNA compared to other baseline approaches across various OOD test sets. RNA achieves the best results for all six OOD test sets when trained on CIFAR10-LT and achieves the best results on four OOD test sets as well as on average when trained on CIFAR100-LT.
3.  Clarity: The information provided in the tables is clear and concise, allowing for easy comparison of the performance of different methods. The description of the proposed method, RNA, is also clear and provides a good understanding of how it addresses the limitations of previous approaches.
4.  Significance: The ability to simultaneously achieve high OOD detection performance and accurate long-tailed recognition is significant in various applications, especially in scenarios where both goals are crucial. The proposed RNA method offers a promising solution to this challenge and outperforms other baseline approaches in terms of both OOD detection metrics and classification accuracy.

### Weaknesses
1. Reliance on an auxiliary OOD dataset: The proposed RNA method relies on the availability of an auxiliary OOD dataset for effective performance. While the paper mentions the possibility of replacing the auxiliary OOD dataset with augmented training data, it does not provide a thorough exploration of this alternative or compare its performance with the original approach. Further investigation into alternative data augmentation techniques or methods that reduce the reliance on auxiliary OOD datasets would strengthen the proposed method's practicality and generalizability.
2. The study lacks a comprehensive comparison of model performance under different imbalance rates, including balanced datasets, limiting its applicability and understanding of class imbalance effects.

### Questions
1. The paper mentions that the proposed RNA method amplifies the norms of only ID representations while indirectly reducing the representation norms of OOD data by updating the BN statistics using both ID and OOD data. It would be beneficial to provide more details on the rationale behind this approach and how it effectively generates a noticeable discrepancy in the representation norm between ID and OOD data. Additionally, It is better to provide theoretical derivation and discuss the potential impact of this approach on the overall training dynamics and convergence behavior would provide further insights.
2. The paper mentions that the proposed RNA method addresses the overconfidence problem of OOD samples and the underconfidence problem of tail-class ID samples. It would be helpful to provide more insights into how the proposed method achieves this and how it compares to other techniques, such as ODIN (Liang et al., 2018) and LogitNorm (Wei et al., 2022), which also aim to mitigate overconfidence issues. Providing a more detailed analysis and comparison of these techniques would enhance the understanding of the proposed method's effectiveness.
3. The robustness of the RNA method to diverse auxiliary datasets remains unclear. Exploring alternative auxiliary datasets holds the potential for improving results, necessitating further investigation into their impact on the RNA method's performance.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new loss named representation norm amplification to increase the feature norm of ID data during training such that the OOD data could be distinguished from ID data by the feature norm. This method requires an auxiliary OOD dataset to be effective. The loss is applied to the long-tail classification together with the LA loss. Experiments on CIFAR10/100-LT and ImageNet-LT show that the proposed method not only increases the classification accuracy, but also improves the OOD detection performance.

### Strengths
- The paper is easy to follow. The trade-off between outlier exposure and logit adjustment is explained to motivate the method.
- Experiments are designed to explain the various design choices.
- The desgin of the RNA loss is new to me, and the observation that the 2-layer projection is better than a no-projection is interesting.

### Weaknesses
- The claim "only ID data contributes to the gradient for updating model parameters" (page 2 and page 5) is wrong. OOD data surely affects the model training and their intermediate features appear in the gradient of model parameters. Otherwise you can remove these auxiliary OOD data without affecting the resulting model.
- The motivation only considers the conflict with one of the OOD methods, outlier exposure. However, the motivation of using outlier exposure in the first place is not explained. There are a lot of post-hoc OOD methods, such as Mahalanobis [a], ViM [b], ASH [c], etc that does not requrire training. Does LA conflicts with these post-hoc methods? It is interesting to compare with the performance of LA + a/b/c (to show that exposing OOD in training is necessary) and LA + RNA + a/b/c (to show that the norm is a good OOD scoring function).
- There is only 1 OOD dataset of ImageNet-1K LT, which is inadequate. It would be great to test more OOD datasets such as Texture and OpenImage-O.
- Page 8 "The different trends may be due to the difference in the number of classes of each dataset ... ImageNet-LT generally produce low confidence scores...". It is not obvious to me that the number of classes affect the confidence score. Please provide evidence for this claim. 

[a] "Exploring the limits of out-of-distribution detection." NeurIPS 2021.

[b] "ViM: Out-Of-Distribution with Virtual-logit Matching" CVPR 2022.

[c] "Extremely Simple Activation Shaping for Out-of-Distribution Detection" ICLR 2023.

### Questions
See weakness. I will raise the score if the questions are reasonably answered.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to amplify the representation norm for detecting OOD samples. The OOD samples are used in training; and the method is specifically applied in the long-tailed classification scenario. The paper claims that the tail data is easily overlapped with the OOD samples. Using representation norm as the OOD detection signal can help to decouple the influence of the training with OOD samples on tail data classification.

### Strengths
- The motivation is reasonable. It is important to handle the OOD detection issue with long-tail classification tasks, where the tail data is usually overlapped with the OOD samples. 
- The experiments on multiple datasets and various ablation studies are conducted.

### Weaknesses
- The idea of using representation norm for OOD detection is not new. And the discussion with related works is missing, such as:

> Park, Jaewoo, Jacky Chen Long Chai, Jaeho Yoon, and Andrew Beng Jin Teoh. "Understanding the Feature Norm for Out-of-Distribution Detection." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1557-1567. 2023.

- The proposed idea is actually general for OOD detection, especially the setting with OOD samples available in training (OE). It is not specifically designed for long-tail data, although maybe its benefits can be more obvious in long-tail classification settings. 

- The experiments can be improved. The datasets and settings used in experiments can be improved. Please check PASCL paper for a reference. 

Why the reported performances of the compared methods (such as PASCL) are lower than in the original paper? Please clarify the experimental settings.

### Questions
- Please clarify the questions in weaknesses. 

- Why the reported performances of the compared methods (such as PASCL) are lower than in the original paper? Please clarify the experimental settings.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
