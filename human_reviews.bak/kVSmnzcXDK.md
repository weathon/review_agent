# Combine and Conquer: A Meta-Analysis on Data Shift and Out-of-Distribution Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6

## Abstract
This paper describes a universal approach to combining detectors and compares combination methods for data distribution shift and out-of-distribution detection. By aligning each individual detector score's distribution into p-values through a quantile normalization, we transform the problem into a multi-variate hypothesis test that we combine by leveraging established meta-analysis tools. The resulting test can effectively fuse the individual decision boundaries to create a more capable detector. Furthermore, we can obtain a fully interpretable criterion by reshaping the final statistics of the in-distribution score. Our framework is easily extensible to future development of detection scores. Through a comprehensive empirical investigation, we examine diverse kinds of shifts with different magnitudes and fractions of affected data, showing that our framework is advantageous in improving overall robustness and performance across domains and types of shift and out-of-distribution detection.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a universal method for integrating detectors and evaluating approaches for detecting out-of-distribution (OOD) data and addressing data distribution shifts. The authors propose a technique that normalizes detector scores into p-values using quantile normalization, transforming the problem into a multivariate hypothesis test. They combine these tests using meta-analysis tools, improving the effectiveness of the detector and consolidating decision boundaries. The authors also create an interpretable criterion by adjusting the final statistics of in-distribution scores. Through empirical investigation, they demonstrate that their approach enhances robustness and performance across various domains, shift types, and OOD detection scenarios. The paper contributes to the field of machine learning by providing a flexible framework for integrating detectors and addressing OOD detection challenges.

### Strengths
The paper presents a universal method for integrating detectors, which can be applicable across different domains, including OOD detection and two sample test. 

The use of quantile normalization to transform detector scores into p-values seems interesting. This transformation allows for treating the problem as a multivariate hypothesis test and enables effective combination for a set of predefined scoring strategies.

The paper proposes adjusting the final statistics of in-distribution scores to create a fully interpretable criterion. This feature is valuable as it provides insights and explanations for the detection decisions, enhancing the transparency and interpretability of the method.

Through empirical investigation, the paper demonstrates that the proposed method significantly enhances overall robustness and performance across various domains, shift types, and out-of-distribution detection scenarios. This finding highlights the practical effectiveness of the approach.

### Weaknesses
This paper mainly uses the scoring strategies from OOD detection to build their method, which mainly considers the concept shift. However, in the discussion, they actually present another two different distribution shift, i.e., covariate shift and prior shift. Then, a natural question is why the basic method that handle concept shift can also be useful to tackling covariate shift and prior shift? Is it the attributed to the combination of different p values? More detailed discussion and empirical evaluation seem to be important. 

Quantile normalization seems interesting, while there exist many other ways, typically more simple ways, in doing normalisation. For example, we can simply normalise the score to follow the N(0,1) Gaussian distribution, which can also normalise the data. Therefore, more discussion about the theoretical superiority in using quantile normalisation is interesting.

It seems that extra parameters are introduced when combine different scoring strategies in Section 4.2 and 4.3, while I cannot find how to tune the related parameters. Moreover, how to conduct hyper parameter tuning, epically for the choice of evaluation dataset, should be discussed. If the proposed method does not introduce additional parameters, seemingly different scoring strategies are treated equally in combination. I am not sure if it is a proper setup. 

Different scoring strategies are effective for varying data, so another interesting question is that if the proposed method can be instance dependent in combination when facing different data points.

### Questions
Please see the weaknesses above.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an approach to combine multiple arbitrary out-of-distribution (OOD) detection scores. In particular, quantile normalisation is applied to obtain p-values from the individual detectors. Then, scores from multiple OOD detectors are combined using Fisher or Stouffer meta-analysis to obtain a single OOD estimate. As Fisher's method assumes independence of the p-values, a Brown correction is proposed, using the scaled chi-squared distribution. The approach is evaluated on standard OOD detection benchmarks where it shows good performance compared to the prior work.

### Strengths
+ The idea of the paper is easy to follow. In addition, the problem is well presented, including a detailed explanation of the different types of data shifts.
 + The different single-score methods and methods combining different detection scores are evaluated in detail using different data shift scenarios.

### Weaknesses
- (Major) Although the results of combining existing methods are interesting, the paper does not show any new idea. It therefore has limited novelty. 

- (Major) The paper claims that a major contribution is the correction for the assumed independence of Fisher's method. Therefore, Fisher's method should be compared with and without the correction in the experiments. 

- (Major) It is claimed that the method is interpretable as the distribution of the combined scores is known in advance. However, this is not demonstrated in the paper.

- The clarity and notation of the method has room for improvement, e.g. in Section 4.1 the index i is used to iterate over the window examples and at the same time the doctor.

- The experimental protocol is not very clear. For example, it should be clearly stated which combination of detectors is used in the evaluation. Is this adjusted according to the shift or is it sample or window based?

### Questions
- The paper states that different detectors are ensembled. They have a study on the distillation of the best subset of detectors where different OOD detection scores are combined. Are different classifiers combined or are different OOD detection scores combined? So does the word "detector" refer to the different OOD detection scoring methods or does it refer to a classification model? This is not immediately clear. 

- It should be clarified how the indices in equation 6 are mixed up? Should they be W^r_1 and W^m_2?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the challenge of detecting distribution shifts in data streams that are inputted to deep neural networks. It emphasizes the importance of recognizing when the distribution of incoming data deviates from the distribution of the training data, which can impact the performance and reliability of the model.

### Strengths
1, Instead of instance-level discrimination on OOD samples, this paper considers the OOD detection from an interesting perspective: the windows from the streamed data. 
2, In the poposed detection framework, the author leverages empirical cumulative distribution functions yo effectively compare the distribtuion from two windows, reference window and the test one. 
3.  The transformation into p-values is reasonable, and the calibration across detectors is well motivated. 
4. Extensive experiments are conducted to verify the effectiveness of the proposed method.

### Weaknesses
This paper falls outside of my expertise sligtly, thus for now, I cannot find a clear weaknesses.

### Questions
1, In Fig 5 (c), I cannot find the curve correpsonding to Resnet-101.
2, For the ablation study on window sizes, will listing the proportion of window size towards the whole dataset helpful for understanding the impact of window sizes?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor
