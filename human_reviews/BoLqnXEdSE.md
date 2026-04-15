# Measuring Fairness Using Probable Segmentation for Continuous Sensitive Attributes

- Decision: Reject
- Scores: 3, 6, 5

## Abstract
Algorithmic fairness in machine learning aims to regulate the bias towards sensitive attributes. In the case of continuous sensitive attributes, however, defining and measuring fairness is a non-trivial task. For instance, estimating a maximum disparity of predictions within a continuous sensitive attribute is vulnerable for an extreme case, whereas a mean disparity of predictions underestimates the effect of the worst case, which is meaningful for testing the independence between the prediction and the sensitive attribute. We address this issue by developing a new definition of fairness, probable demographic parity, based on a maximum prediction disparity of probable segments. We only consider probable segments of the continuous sensitive attribute that have a higher probability than the preset minimum probability condition. Then, we compare the local prediction average of these segments to identify the maximum prediction disparity. By doing so, we ensure a consistent estimation error for the local prediction average of the segment and mitigate the risk of encountering missing data in the segment. We analyze the various theoretical features including stability and independence and experimentally prove the usefulness of the proposed metric.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes probable demographic parity as a generalized measurement of demographic parity on continuous sensitive attributes. The proposed metric measures the maximum prediction parity between segmentations of the sensitive attribute with higher than alpha probability. In this way, it is robust to outliers and noises.

### Strengths
1. Fairness with continuous sensitive attributes is relatively under explored. So this is a good topic.

2. The math behind is explained very clearly.

3. Based on the design of the probable demographic parity, I agree that it can be considered as a trade-off between mean prediction disparity and maximum prediction disparity and has the potential to resolve the problems of those two metrics. However, the demonstration of this is lack in the experiments.

### Weaknesses
1. The underestimation and overestimation problems in Paragraph 2 were not cleared explained via Figure 1 because neither of mean prediction disparity or maximum prediction disparity were explained before that. Need to adjust the presentation order (present the content in Background first).

2. Very limited related work is presented. E.g. are there literature studying the other metrics such as equalized odds on continuous sensitive attributes? Are mean prediction disparity and maximum prediction disparity the only approaches for continuous sensitive attributes for demographic parity? Part of the discussion in the introduction should be put in the related work section. There are also existing literature using distance covariance to measure independence between the sensitive attributes and the predictions [1, 2]. They also allow continuous sensitive attributes.

3. There is no guidance in what alpha value should be chosen for the proposed metric although the alpha value will greatly impact the metric. 

4. The authors have claimed superiority of the proposed metric over mean prediction disparity and maximum prediction disparity. However, there is no experiment demonstrating such superiority. The evaluation has no other baseline.

[1] Liu, Ji, Zenan Li, Yuan Yao, Feng Xu, Xiaoxing Ma, Miao Xu, and Hanghang Tong. "Fair representation learning: An alternative to mutual information." In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1088-1097. 2022.

[2] Guo, Dandan, Chaojie Wang, Baoxiang Wang, and Hongyuan Zha. "Learning Fair Representations via Distance Correlation Minimization." IEEE Transactions on Neural Networks and Learning Systems (2022).

### Questions
1. How do you choose the appropriate alpha for the proposed metric?

2. Can you show in the experiments that the proposed metric is better than mean prediction disparity and maximum prediction disparity?

3. What do you think of the distance covariance based metrics from [1] and [2] listed in my Weaknesses when compared to your proposed metric?

4. How can the proposed metric be adapted to multiple sensitive attributes? When the number of sensitive attributes grows, it becomes difficult to find the segmentations.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors address the question of measuring fairness when the sensitive attribute is continuous, e.g. weight or age.

In particular, the authors point out the weakness of the GDP measure introduced by Jiang et al., which is not sensitive to small groups, i.e. when a slice of population s \in [a, b] has small probability measure, its contribution to GDP will be negligible. This is a bad quality for a fairness measure, since small populations are often the source of biased classifiers.

The authors propose to consider a new measure, which better approximates the original DP. Unlike GDP, they sample the segments in a way that makes each of them of significant measure (hyperparameter alpha). They confirm their findings with theoretical analysis, as well as experimentally.

Despite sound motivation and the method proposed, I think there is room for improving this paper. Although the theory presents some insightful results, it does not study analyse the whole problem. Specifically, they show how \tilde{M} fluctuates around M, and then they show how DP-alpha based on M approximates \tilde{M}. However, the two results are not connected. For this particular reason, it is not clear what is a good choice of alpha, there is clearly some trade-off happening between Theorem 2 and Theorem 4, but the reader has to figure it out by themselves. Furthermore, they do not address the question of estimating the segments from the sample, h_alpha are assumed to be known from the true distribution. Finally, your solution is relevant to non-parametric regression with nearest neighbours, and this has to be pointed out, with appropriate references.

I also have a concern regarding Theorem 4. The authors propose that as alpha -> 0, the difference goes to zero. But it is not obvious to the reader, perhaps it is better to keep a potentially weaker result in the main part, and move the original one to appendix. I also think this difference (k - h_a(k)) depends on the distribution and the measure around the point k. I.e. if k at the end of a mode of a distribution, this difference will not go to zero. This nuance has to be highlighted.

### Strengths
Good motivation and feasible proposed solution.

### Weaknesses
Incomplete theory

No validation procedure for choice of alpha proposed

### Questions
How do you choose alpha?

Do you need to simultaneously control the difference between M and \tilde{M} for more than one point k?

What are the conditions that the RHS in Theorem 4 goes to 0 as alpha -> 0 and at what rate does it converge?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a new measure of fairness called probable demographic parity with a focus on continuous sensitive attributes. The proposed measure is based on a maximum prediction disparity of probable segments. They consider probable segments of the continuous sensitive attribute that have a higher probability than the minimum probability condition. 

DP for discrete attributes is defined as Pr(Y=1|S=s)-Pr(Y=1|S=s').
A possible relaxed metric (that would also apply to the continuous case) is defined as: max_s Pr(Y=1|S=s)- min_s'Pr(Y=1|S=s').
Other related metrics are Generalized DP. The measure proposed by the authors looks at the maximum/minimum of the expected value of Y over an interval such that the interval has a measure alpha>0 (scaled down by the probability measure in that interval).

They compare the local prediction average of these segments to identify the maximum prediction disparity. They analyze the various theoretical properties including stability and independence and experimentally demonstrate the benefits of the proposed metric.

### Strengths
This paper introduces an interesting extension of demographic parity (DP) for continuous sensitive attributes. 
The measure proposed by the authors looks at the maximum/minimum of the expected value of Y over an interval such that the interval has a measure alpha>0 (scaled down by the probability measure in that interval).
For discrete attributes, this definition reduces to DP as shown in Thm 1.

Then, they also employ empirical estimation techniques to compute the proposed measure which has benefits in terms of computational complexity. They also include additional theoretical results on how far the DP will be to Probable DP for Quantized sensitive attributes and also under Lipschitz assumptions.

### Weaknesses
Thm 1 statement alpha>0?

The experiment results only seem to demonstrate that the algorithm works and the measure is computable. However, not much insight is provided on how this way of computation is beneficial. 

Is there a distribution with ground truth DP and PDP known so that the performance of the estimation can be compared to it?

There is very little detail on Figure 2.

How is this definition better than the Generalized DP? Could the authors elaborate on this?

There is no discussion on the limitations of this strategy/definition. 

While the idea is interesting, I feel that given other related works, this work is of limited novelty in the context of related works.

### Questions
Included in Weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
