# Rethinking Fair Anomaly Detection From The Group Imbalance Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Anomaly detection (AD) has been widely studied for decades in many real-world applications, including fraud detection in finance, intrusion detection for cybersecurity, etc. Existing anomaly detection methods struggle in imbalanced group scenarios, where the unprotected group is significantly larger than the protected group. Specifically, fairness-unaware methods achieve high overall performance by misclassifying more protected group examples as anomalies, while fairness-aware methods overcompensate fairness by labeling excessive unprotected group examples as anomalies, sacrificing overall performance. To address these issues, we propose FADIG, a fairness-aware contrastive learning-based anomaly detection method designed for imbalanced groups. FADIG consists of two key modules: (1) an adaptively re-balanced autoencoder module that dynamically adjusts group contributions to balance fairness with performance and (2) a fairness-aware contrastive learning module that maximizes similarity between protected and unprotected groups to ensure fairness. Moreover, we provide a theoretical analysis showing our proposed contrastive learning regularization guarantees group fairness. Extensive experiments across multiple real-world datasets demonstrate the effectiveness and efficiency of FADIG in achieving both accurate and fair anomaly detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FADIG, a new method for fair anomaly detection considering imbalanced protected and unprotected groups. It tackles the issue through two main components: an adaptively re-balanced autoencoder that adjusts group contributions to the loss function and a fairness-aware contrastive learning module to align the data representations across groups.

### Strengths
1. The considered problem is essential in fairness studies.

2. The presentation is overall smooth and easy to follow.

3. The experiments are extensive.

### Weaknesses
1. In the experiment part, fairness-accuracy trade-off comparisons are missing.

2. The re-balancing loss is a well-established technique. The author mentions that the novelty lies in its hyperparameter-free and parameter-free adaptive weight.  While the proposed formulation in Equation (4) is interesting, the idea of dynamically adjusting weights based on training losses or model performance on different data subsets is not new in the fields of fairness, class imbalance, or hard-sample mining.

3. The paper's motivation hinges on the claim that existing fairness-aware methods "often overlook the underlying group imbalance that gives rise to such unfairness". This framing is confusing, as any method designed to enforce group fairness inherently must consider group imbalance. The limitation is not that prior work overlooks imbalance, but how it attempts to solve it but insufficient.

4. The motivation of the proposed methods is not well presented. The paper does not provide a clear explanation for why FADIG's specific components succeed where others fail. The experimental results show that FADIG outperforms the baselines, and the ablation study confirms its components are necessary. However, it does not show how/why the proposed methods can achieve their superior outcomes by resolving what kinds of limitations.

5. Authors should consider widely adopted fairness metrics such as equalized odds and equalized opportunity for results comparisons.

### Questions
1. At the beginning, the model is largely unfitted, meaning reconstruction errors are high and potentially close to the initial estimates. This could make the numerator and denominator small, potentially leading to unstable or erratic behavior.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies fairness in unsupervised anomaly detection under group imbalance. Existing fairness-unaware models often misclassify protected-group samples as anomalies, while fairness-aware methods overcompensate and degrade overall performance. To address this, the authors propose FADIG, a fairness-aware anomaly detection method that is inspired by reconstruction-based autoencoders. The authors define a new loss function for reconstruction error that has two components: an adaptively rebalanced autoencoder reconstruction loss that dynamically adjusts group contributions during training, and a fairness-aware contrastive learning loss that aligns the representations of protected and unprotected groups while maintaining within-group diversity. A theoretical analysis based on f-divergence shows that minimizing their contrastive regularization reduces group recall disparity. The authors provide extensive experiments on image, tabular, and graph datasets demonstrating that FADIG achieves higher recall with lower disparity than baselines.

### Strengths
- Introduces a novel adaptive reweighting mechanism that balances contributions from protected and unprotected groups, and a fairness-aware contrastive learning module that promotes cross-group alignment and within-group diversity.
- Provides extensive experiments across image, tabular, and graph datasets, showing improvements in both fairness and accuracy (higher recall).

### Weaknesses
- The core contribution is relatively modest, as FADIG ultimately modifies the training objective through a reweighted reconstruction loss and additional fairness regularizer.
- The fairness analysis only provides an indirect link between the theoretical risk bounds and the empirical fairness metric (recall disparity), limiting the strength of its claims.

### Questions
- The authors say that the training and the test datasets are the same. I don't know how common this is in anomaly detection, but this requires more justification than the task being unsupervised.
- I am quite surprised about the recall statistics of fairness-unaware methods. In general, there is a tradeoff between accuracy and fairness, but the algorithm proposed by the authors achieves higher accuracy and fairness simultaneously. This raises questions about whether the baselines are sufficiently strong or well-tuned, how authors explain FADIG's ability to achieve higher fairness without compromising accuracy.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on fairness in outlier detection in unsupervised learning specifically addressing imbalance data that naturally arises in presence of minority protected groups. The paper presents a method for addressing representation disparity due to imbalance by proposing a fairness-aware contrative learning criterion and a weighted reconstruction based network module to account for patterns from minority groups. Empirical results show the effectiveness of the proposed method across multiple real-world datasets when compared to exciting fairness-aware methods.

### Strengths
1. Rebalancing autoencoder with learnable weight for reconstruction loss is a simple way to encourage learning patterns from minority groups. I like the analytical calculation of \epsilon. 
2. Adapting contrastive learning for comparing the groups induced by protected attributes is simple and effective approach.
3. Paper is easy to read and follow.

### Weaknesses
1. Appendix G notes that the paper focuses on binary groups. How does the method scale with multi valued multiple protected attribute setup?
2. Paper shows the robustness of choices of hyperparameters. However, a discussion on how initial choice was arrived at would be helpful.

### Questions
I do not have specific questions. I reviewed this paper in an earlier cycle, and the authors have significantly updated it.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies fairness in unsupervised anomaly detection with group-wise imbalance. The authors argues that standard AD methods skew toward majority patterns and over-flag the protected group as anomalous, while prior fairness-aware methods often over-correct and hurt overall recall. To tackle the group imbalance, the authors propose FADIG, which combines (i) an adaptively re-balanced autoencoder that learns group weights to balance utility and fairness, and (ii) a fairness-aware contrastive learning regularizer that aligns group-wise representations without collapsing anomaly separability. Experimental results across multiple datasets shows that FADIG improves detection performance and group parity over baselines.

### Strengths
The proposed FADIG framework elegantly integrates adaptive re-weighting and fair contrastive learning, addressing both representation bias and imbalance without requiring group labels during inference.

The authors derive a provable bound showing that minimizing their contrastive regularizer reduces group-risk differences, lending theoretical support to the fairness claims.

### Weaknesses
The overall problem setup lacks clarity: the method assumes full access to sensitive attribute labels during training, while anomaly labels remain unavailable. Is this a common real-world scenario? Specifically, addressing partial or missing sensitive information seems to be more reasonable.

The proposed re-balanced reconstruction loss seems to rely on the strong assumption that the minority group also shares worse performance. In practice, however, this is likely to not hold true, which poses concerns on the applicability of the proposed method.

It remains unclear how the proposed method aligns with conventional fairness notions such as disparate impact or equalized odds. The paper primarily evaluates fairness through group-wise performance gaps in anomaly scores, but does not explicitly examine whether FADIG improves or preserves fairness under these established definitions.

Several claims are misleading or inaccurate. For example, the authors claim "a hyperparameter-free and parameter-free adaptively reweighted autoencoder" in line 481, which is clearly not the case.

It is unclear how the proposed method would perform relative to thresholding-based or post-processing fairness baselines, especially given the superiority of post-processing as suggested in existing work [1].

[1] Cruz, André F., and Moritz Hardt. "Unprocessing seven years of algorithmic fairness." arXiv preprint arXiv:2306.07261 (2023).

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
