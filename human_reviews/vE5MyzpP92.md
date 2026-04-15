# Threshold-Consistent Margin Loss for Open-World Deep Metric Learning

- Decision: Accept (poster)
- Scores: 5, 8, 6

## Abstract
Existing losses used in deep metric learning (DML) for image retrieval often lead to highly non-uniform intra-class and inter-class representation structures across test classes and data distributions. When combined with the common practice of using a fixed threshold to declare a match, this gives rise to significant performance variations in terms of false accept rate (FAR) and false reject rate (FRR) across test classes and data distributions. We define this issue in DML as threshold inconsistency. In real-world applications, such inconsistency often complicates the threshold selection process when deploying large-scale image retrieval systems. To measure this inconsistency, we propose a novel variance-based metric called Operating-Point-Inconsistency-Score (OPIS) that quantifies the variance in the operating characteristics across classes. Using the OPIS metric, we find that achieving high accuracy levels in a DML model does not automatically guarantee threshold consistency. In fact, our investigation reveals a Pareto frontier in the high-accuracy regime, where existing methods to improve accuracy often lead to degradation in threshold consistency. To address this trade-off, we introduce the Threshold-Consistent Margin (TCM) loss, a simple yet effective regularization technique that promotes uniformity in representation structures across classes by selectively penalizing hard sample pairs. Large-scale experiments demonstrate TCM's effectiveness in enhancing threshold consistency while preserving accuracy, simplifying the threshold selection process in practical DML settings.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the issue of inconsistency in threshold determination for negative samples in threshold-based image retrieval. The authors propose a new metric called Operating-Point-Inconsistency-Score (OPIS) to measure inconsistency and introduce the Threshold-Consistent Margin (TCM) loss as a regularization technique to enhance consistency. The key contributions include identifying the problem with existing method, introducing an intuitive evaluation metric and regularization approach, and demonstrating improved threshold consistency without sacrificing accuracy in large-scale experiments.

### Strengths
- The paper is well-written, making it easy to understand while offering comprehensive comparisons with current methods.

- It clearly highlights issues in existing models and presents an intuitive metric and regularization technique to tackle them.

- The research goes a step further by demonstrating not just improved threshold consistency but also better performance in several instances.

### Weaknesses
- The biggest weakness in the paper is the lack of experiments related to face verification, where threshold importance is evident. While image retrieval mostly uses metrics like mAP or Recall@k, face verification relies heavily on thresholds and uses metrics like TAR@FAR. The introduced method appears more suited for face verification than image retrieval.

- The paper suggests that high accuracy doesn't always mean high threshold consistency. However, in face verification tasks, consistency in threshold often translates to high accuracy. This amplifies the sense that the paper might be focusing on an unrelated task.

- The paper mentions related works like Liu et al. (2022) and OneFace, but experiments comparing the proposed method to these in the realm of face recognition are missing. Such comparisons are necessary to understand the proposed method's improvements in threshold consistency.

- The paper needs to update state-of-the-art results on the CUB and Cars-196 datasets [1].
[1] Kim et al., HIER: Metric Learning Beyond Class Labels via Hierarchical Regularization, CVPR 23

### Questions
Figure 3 shows ProxyAnchor (ResNet50) with a low threshold consistency. It would be beneficial to compare the improvement in R@K and OPIS when using the proposed method. Ideally, the proposed method should show a significant OPIS improvement compared to others.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces and addresses the threshold inconsistency problem in Deep Metric Learning (DML). To tackle this issue, the authors present the Operating-Point-Inconsistency-Score (OPIS) metric, which is based on the variance of utility score derived from the F-score. Additionally, they propose the Threshold-Consistent Margin (TCM) loss, which selectively penalizes hard sample pairs. The experimental results on various deep metric learning benchmarks validate the efficacy of their proposed method.

### Strengths
1.The paper effectively identifies and defines the threshold inconsistency problem within the context of Deep Metric Learning (DML). 

2.To address this issue, the authors introduce a novel loss function, the Threshold-Consistent Margin (TCM) loss. 

3.Their proposed method is rigorously evaluated through comprehensive experiments.

### Weaknesses
1. The use of the term "large-scale" in this paper may be misleading as the experiment datasets do not contain a sufficiently large number of samples to be accurately characterized as "large-scale." Typically, datasets with more than 10 million or 1 billion samples could be considered as large-scale.

2. The threshold inconsistency problem, as described in this paper, is also referred to as the generalization problem and has been previously discussed in the deep metric learning (DML) literature [r1, r2]. In reference [r1], the authors proposed the adoption of a metric variance constraint (MVC) to enhance generalization ability, which is essentially a variance-based metric. Reference [r2] provided an in-depth discussion of the generalization problem in DML. It would be beneficial for this paper to incorporate discussions and comparisons with these existing works in the context of addressing the threshold inconsistency problem. 

[r1] Kan, Shichao, et al. "Contrastive Bayesian Analysis for Deep Metric Learning." IEEE Transactions on Pattern Analysis and Machine Intelligence (2022).

[r2] Karsten Roth, Timo Milbich, Samarth Sinha, Prateek Gupta, Björn Ommer, and Joseph Paul Cohen. Revisiting training strategies and generalization performance in deep metric learning. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 8242–8252, 2020.

### Questions
See the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the problem of threshold inconsistency in deep metric learning (DML) for image retrieval. Existing DML methods often result in uneven representation structures within and between classes, leading to significant variations in performance across different test classes and data distributions, measured by false accept rate (FAR) and false reject rate (FRR). To tackle this issue, the authors propose a novel variance-based metric called Operating-Point-Inconsistency-Score (OPIS) to quantify the inconsistency in threshold performance across classes. They observe a trade-off between accuracy and threshold consistency, where improving accuracy can negatively impact threshold consistency. To mitigate this trade-off, they introduce the Threshold-Consistent Margin (TCM) loss, a simple yet effective regularization technique that penalizes difficult sample pairs to encourage uniform representation structures across classes. Extensive experiments on large-scale datasets demonstrate that TCM enhances threshold consistency while maintaining or even improving accuracy, simplifying the threshold selection process in practical DML applications. The key contributions of the paper include the introduction of the OPIS metric, the identification of the accuracy-threshold consistency trade-off, and the proposal of the TCM loss as a solution to improve threshold consistency in DML. The approach outperforms state-of-the-art methods on various large-scale image retrieval benchmarks, achieving significant improvements in threshold consistency.

### Strengths
1. The proposed Operating-Point-Inconsistency Score (OPIS) and ϵ-OPIS provide valuable insights.
2. The experiments comparing high accuracy with high threshold consistency are objective.
3. The proposed Threshold-Consistent Margin (TCM) loss is relatively simple and easy to understand.
4. The visualization of the TCM effect is interesting.
5. The experiments are comprehensive, with detailed implementation and coverage of mainstream metric learning settings.
6. The ablation experiments are extensive, exploring margin, DML losses, different architectures, and time complexity. They also validate the proposed method against state-of-the-art approaches such as RS.

### Weaknesses
It is meaningful to pull the scores of positive pairs towards a fixed value and the scores of negative pairs towards another fixed value, even though it sounds simple.

Apart from that, I did not see any other weaknesses.

### Questions
1. Since you conducted experiments on the large-scale iNaturalist-2018 dataset, what are the differences between open-set metric learning and face recognition or re-identification (re-ID)? Can your method be applied in the field of face recognition?
2. If your method can use a single model to maintain the same threshold across multiple test sets, would it be meaningful, such as in this work[1].

[1] https://cmp.felk.cvut.cz/univ_emb/

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
