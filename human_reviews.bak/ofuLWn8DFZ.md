# Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work studies robustness of conformal prediction to poisoning in a threat model that allows insertion and deletion which is a limitation of the other studies e.g. zargarbashi et al, 2024. For training the authors propose to divide the training data into $k_t$ disjoint sets so that by definition any poisoning will at worst break one model. This robustness approach was previously studied in image classification as well. The conformity score function is the softmax over the votes of each classifier on each class. While the conformal guarantee is safe w.r.t. poisoning in the training process, calibration data can be poisoned as well. The authors remedy that by a similar approach of dividing the calibration set, and producing prediction sets from each calibration group, and running a majority vote-based decision on the final output. 

Overall I don’t count section 5.1 as a novel approach tailored to conformal prediction. However it is an smart approach to reduce the poisoning effect on set size. However showing that Section 5.2 is a valid approach is an important contribution.

### Strengths
The other works designed to make CP robust to poisoning are not considering insertion and deletion in their threat model. This makes this work unique. Also their approach however repeated from the poisoning literature, still is novel to be adapted to conformal prediction framework. 

The score function designed by the authors is interesting and novel. It also can have interesting properties beyond the score of robustness. The interesting outcome of this is that it enables them to provide the guarantees on the set size which is novel.

The paper is theoretically solid. I think the theorems are well defined, and the results are quite far from obvious. Therefore I see this as a strong work.

The writing is also in a good shape; the story of the paper is easy to follow. However a strong recommendation is to add a discussion on differences between normal conformal coverage guarantee and coverage reliability.

### Weaknesses
**Writing.**  I expect the threat model to be mentioned more clearly in the introduction. Until the related work I did not understood that the authors are addressing a poisoning model different from zargarbashi et al 2024. I strongly suggest the authors to mention that in their threat model addition and deletion of data points is allowed which is a limitation of zargarbashi et al 2024 either in the introduction or in the corresponding section in related works.

**Wrong background.** The definition of the conformal prediction in Theorem 1 is not correct. Either you define the scores as capturing agreement between the data and the label (e.g. directly using softmax) then the for test points you add labels with scores above the threshold and the threshold is the $\alpha_n = \lfloor\alpha(n+1)-1\rfloor$ or the quantile is as authors specified in line 135, where the score should be non-conformity $1 - f_y(x)$ and the labels with score lower that the threshold should be picked. 

**Unnecessary, and unclear desiderata.** Although I agree on desiderata no. 1, 2, and 3, I still think that computational efficiency is not a necessity. Also if no. 4 means that there is an strict increasing requirement, then it contradicts the assumption of adding datapoints can be invariant to the final result. Other than that what is the concrete definition of providing small prediction sets? Is it in comparison to the same model trained on clean data? or is it in comparison to vanilla setup?

**Discrete score.**  Although the score is float, it is discrete with fixed number of possibilities. However working in practice there are corner cases where ties can become problematic. You need to add a random noise to the score in Eq. 2 which I think does not change that much in the efficiency of the output but simply avoids these small chances of failure. 

**Study of statistical efficiency.** The empirical coverage is not always $1 - \alpha$. This is on it’s own a random variable sampled from a Beta distribution concentrated around the nominal guarantee. This concentration is increased by the number of calibration points. The first question that is not addressed here is that how breaking the calibration set to smaller calibration sets influence this concentration? For sure it should decrease due to smaller calibration sets, while some of this can be remedied by the majority prediction set. Do you have any intuition on how this is affected? At least I expect a discussion on the lowerbound of this in the limitations of the work.

**Typos.** Sentence at the end of line 250 is grammatically wrong. Typos are seen in lines 349 (title). 

**Suggestions.** The intuition behind the proof of Theorem 2. is non-trivial. I would recommend the authors to add a short sketch of proof or an intuition on what is deciding the parameter of majority vote set in the manuscript.

### Questions
1. The definition of $\hat{\tau}$ is unclear to me. Can you elaborate? What is this capturing? Why the iteration variable is called $x$? Does it have any correlation with covariates?
2. I can not understand why poisoning in training data can influence the coverage reliability. Since the coverage is only dependent to calibration set the guarantee should not break with a poisoned model (CP is guaranteed with black-box model). Then why Fig. 2-b shows a decay on coverage reliable. In general there are no comprehensive discussions on what coverage reliability and coverage guarantee have in difference. So from the conformal guarantee changes in the training data should have no influence in the “marginal” coverage of the prediction sets. However I think this reliability is defined per point. Is that so? In case yes please update the manuscript to draw this difference line in a more clear way. 
3. I think the set-size reliability is misleading to show (at least as a solid line) for calibration poisoning. Under calibration poisoning the guarantee is already invalid and with an invalid guarantee by definition it is not important if the set sizes are small or not. Even in some cases expanding set sizes can be considered as an alert to the defender that the calibration set is invalid.
4. In Fig 3. I can not understand the jumps in the beginning of the x axis for empirical coverage? What is the reason behind that intuitively? I see a quick explanation in lines 473-475 but isn’t it due to the missing randomization in the score function?

If the points in questions and weaknesses are provided I would be happy to increase my score.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes reliable prediction sets (RPS) as an efficient method for constructing conformal prediction sets under training and calibration data poisoning. The proposed approach has two components: to improve reliability under training data poisoning, authors introduce smoothed score functions that reliably aggregate predictions from classifiers trained on distinct partitions of the training data; and to improve reliability under calibration data poisoning, authors calibrate multiple prediction sets, each calibrated on distinct subsets of the calibration data, and construct a majority prediction set. Authors propose a desiderata for reliable conformal prediction, and derive certificates for the reliability of prediction sets under worst-case data poisoning attacks. The paper presents empirical evaluation of the method on image classification tasks.

### Strengths
1. The paper is well-written, organized, and easy to follow for the most part. 
2. The paper is technically sound. I appreciate the paper clearly lays out the desiderata for reliable conformal prediction sets under data poisoning in the beginning. The claims regarding RPS are well supported by theory and empirical evaluation. 
3. The paper presents important results that can contribute toward improving the reliability of conformal prediction sets under data poisoning attacks in real scenarios.

### Weaknesses
1. It would help to show comparison with baselines in the empirical evaluation. I see there is some comparison with APS to understand the effect of randomization; however, it would be interesting to see worst-case reliability analysis comparison with existing conformal prediction methods, and discussing how RPS fares in the different desiderata.
2. Relationship of reliability with number of partitions: the paper briefly discusses how the number of partitions (and dataset size) will affect the reliability of prediction sets; however, I did not seem to find experiments that analyze this relationship. It would help to demonstrate this relationship in order to better understand the utility of prediction sets in different scenarios.
3. Section 6 can be made more readable and the notation becomes slightly hard to follow at times. 



**Minor comments:**
1. L176: typo, Desideratum VI -> Desideratum IV
2. L300: equation reference should probably be (5) 
3. L350: typo, coverave -> coverage

### Questions
Is there any reason why full experiments for CIFAR-100 and ImageNet were not included? I appreciate the paper mentions the limitation in terms of accuracy loss, and I understand if full training can be expensive but I still believe there is value in showing limited results here, especially under calibration poisoning. It will be good to understand performance and the various desiderata in these relatively harder datasets with larger number of classes.

### Soundness
3

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
They introduce a method called "reliable prediction sets" that constructs conformal prediction reliable sets under poisoning attacks. They consider both training and calibration data poisoning. In order to achieve robustness against training poisoning, they partition training data into different parts and train a classifier on each part, and then aggregate the predictions from these classifiers to derive a smooth conformal score function.

Furthermore, in order to be reliable against calibration poisoning, they calibrate multiple prediction sets on disjoint subsets of the calibration data and construct a majority prediction set that includes classes that are included in the majority of the independent prediction sets.

Their guarantees show that:
Under training poisoning, they show the conformal prediction set derived with smoothed score function is both size reliable and coverage reliable.
Furthermore, for calibration poisoning, they show under some conditions their majority prediction set method is coverage and size reliable.

Finally, they show the effectiveness of their approach experimentally, showing their method achieves reliability while maintaining utility on clean data.

### Strengths
I think overall, the idea of partitioning the training data and training different classifiers and aggregating their predictions is pretty natural and has been in around for both differential privacy and robustness against poisoning attacks. But anyways, they are the first to study it in the context of conformal prediction.

### Weaknesses
I think overall, the idea of partitioning the training data and training different classifiers and aggregating their predictions is pretty natural and has been in around for both differential privacy and robustness against poisoning attacks. But anyways, they are the first to study it in the context of conformal prediction.

### Questions
In equation 2, it might be good to remind the reader that K is the number of classes. 

 I was confused by Definition 1, I think here you mean that the adversary flips y_i for some examples x_i, and then as a result, the prediction set C(x_{n+1}) might inflate or shrink. Is this correct?

### Soundness
3

### Presentation
3

### Contribution
3
