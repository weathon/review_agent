# Hearing faces among homogeneous populations: improvement of cross-modal biometrics

- Decision: Reject
- Scores: 3, 3, 5

## Abstract
The relationship between voice and face is well-established in neuroscience and biology. Recent algorithmic advancements have yielded substantial improvements in voice face matching. However, these approaches predominantly achieve success by leveraging datasets with diverse demographic characteristics, which inherently provide greater inter-speaker variability. We address the challenging problem of voice face matching and retrieval in homogeneous datasets, where speakers share gender and ethnicity. Our novel deep architecture, featuring a weighted triplet loss function based on face distances, achieves state-of-the-art performance for voice face matching on these uniform populations. We evaluate our model on a sequence of homogeneous datasets containing  only voices and faces of people sharing gender and ethnic group. In addition, we introduce percentile-recall, a new metric for evaluating voice face retrieval tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors make improvements to the task of voice face matching (and retrieval). The authors claim that existing approaches are limited because their datasets do not have homogeneity in the population, which makes the task easier. In order to solve the task, the authors introduce a new dataset which contains samples of population which are similar in demographic characteristics. They also develop a new deep learning architecture which is claimed to attain better results than the existing methods.

The key contributions of the paper:
- Introduction of the homogeneous dataset.
- A novel architecture which the authors claim works better than existing methods.
- Introduction of the percentile recall metric for voice face retrieval.

### Strengths
1. The authors identify a weak point in existing literature (high variance makes task easier) and attempt to tackle it.
2. The equations for loss and the newly introduced metric are clearly illustrated.
3. Discussion on the influence of noise injection is factored in.
4. The authors introduce a new metric (percentile recall) which tackles the problem of exponential decay in existing metrics in voice face retrieval.
5. Section on reproducibility is included.

### Weaknesses
1. Table 2 in the results section is not very clear. The authors mention for table 2 that "1:2 accuracy on homogeneous data in present work compared to previous works." It is unclear what is meant by "homogeneous data". It appears that the metrics being reported are for the methods on different GNA-var removed datasets and NOT each method on the dataset that the authors introduced.
2. If the above is in fact true, then this seems like a problem – different methods applied on different datasets cannot lead to a fair assessment of the proposed method. What is ideally needed, is the metrics of existing methods on the dataset that the authors propose for comparison.
3. The authors correctly state that "As Nagrani et al. (2018a) note, human performance deteriorates markedly when assessed on voice face matching of speakers sharing gender, ethnicity, or age group." However, Nagrani et al. (2018a) also report metrics (which I think are provided in table 2) with GNA-var removed. Now, I am unclear on the value the homogenous dataset introduced adds? Is it the size?
4. If the answer to 3 is yes, then the key finding appears to be that training on more "relevant" data (or hard examples) is helpful. This itself as a contribution to the field seems weak in my opinion.
5. The authors only target one demographic. There is no discussion on why this demographic was chosen. I believe it would be important for proving the efficacy by applying the method on different groups individually. I also believe that it would be interesting to see the resulting patterns as well as reduce bias.
6. The network architecture diagram (Fig. 3) is very high level, and it could benefit with the addition of more details like layers, parameters, etc.
7. Lacks implementation details like learning rate, number of layers, etc.

### Questions
1. It is not clear to me why audio is split into 3-8 seconds intervals? Why is this variable? How is the decision made? 
2. N' is not clearly defined in section 2.3.1.
3. The authors write for the dataset – "The list of them is available upon request". It is not clear if this paper has a dataset contribution.
4. More information on how the data was collected (or more importantly filtered) is lacking.

### Soundness
2

### Presentation
2

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
The paper makes significant contributions to the field of cross-modal biometrics by proposing a novel deep architecture and a new evaluation metric tailored for homogeneous datasets. The weighted triplet loss function effectively improves the model's performance in challenging scenarios. However, the model's generalizability and practicality need further investigation.

### Strengths
This paper introduces several key innovations aimed at improving voice-face matching in a homogeneous dataset. The primary innovation lies in a homogeneous dataset White Robin, where speakers share gender and ethnicity. And then propose a deep architecture that leverages a weighted triplet loss function based on face distances. To better evaluate the performance of voice-face retrieval tasks, the authors propose a new metric called percentile-recall.

### Weaknesses
1. The innovation in this article is insufficient. (1) The proposed weighted triplet loss is very similar to the triplet loss
used in FaceNet [1]. (2) The mechanism of the introduced Percentile-Recall is closely related to existing retrieval
performance evaluation methods.

2. The experiments are not sufficiently comprehensive to fully support the work. How does the proposed method
perform on general metrics for voice-face matching, such as Binary Accuracy (ACC), Multi-way Accuracy (ACC),
and Verification Area Under the Curve (AUC) [2]？

[1] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[C]//
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.
[2] Wang J, Zheng A, Yan Y, et al. Attribute-guided Cross-modal Interaction and Enhancement for Audio-Visual
Matching[J]. IEEE Transactions on Information Forensics and Security, 2024.

### Questions
Why is it necessary to propose a homogeneous dataset? 
What is the correlation between human appearances and voices within the same gender and ethnicity？

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper investigates the problem of cross-modal biometric matching—specifically, associating voices with faces—within homogeneous datasets, focusing on populations sharing specific demographic traits, such as gender and ethnicity. The authors introduce a new deep architecture incorporating a face-distance-weighted triplet loss to optimize matching between faces and voices in datasets with reduced inter-speaker variability. They also propose “percentile recall” as a novel metric to evaluate voice-to-face retrieval accuracy in large galleries. Their experiments demonstrate that the approach achieves state-of-the-art results with their new model architecture and fine-tuning of homogeneous data, surpassing existing models in voice-face matching accuracy.

### Strengths
- **Originality and Novelty**: The focus on homogeneous datasets for voice-face matching is a valuable addition to cross-modal biometrics research, addressing a specific gap in the field.
- **Metric Innovation**:  The introduction of percentile recall provides a practical metric for real-world applications where retrieval from large datasets is necessary.
- **State-of-the-Art Results**:  The paper reports superior performance using their proposed model and metric, setting a new benchmark for cross-modal matching in homogeneous datasets.
- **Detailed Experimental Setup**:  Extensive testing with various noise levels and codec conditions shows the model’s robustness, a critical aspect for practical deployment.

### Weaknesses
- **Limited Generalizability Discussion**: While the homogeneous dataset approach is compelling, the paper could further discuss its potential limitations in generalizing across other homogeneous populations (e.g., different ethnic groups).
- **Sensitivity to Loss Hyperparameters**: The performance impact of different parameter settings, particularly for the triplet loss function, is briefly mentioned but not thoroughly analyzed. A deeper analysis could clarify its robustness.
- **Comparative Evaluation**: More detailed comparative analysis with traditional metrics (besides identification accuracy) across various heterogeneous models would provide more precise insights into the model’s unique contributions. 
- **Experiments**: More benchmarks, ablation studies, and result analysis (as mentioned above) could yield a more complete story. Improved figures and layout would help (use figures/plots to help tell the story and make them compelling and full of information, using space efficiently).

### Questions
1. **Generalizability of Model Across Different Homogeneous Groups**: How would the model adapt if trained on homogeneous datasets featuring different demographics (e.g., age or ethnicity)?

2. **Impact of Triplet Loss Weights**: Could you elaborate on the sensitivity of the weighted triplet loss function to variations in the face embedding space, especially for similar-looking individuals?

3. **Dataset Bias**: Given the heavy reliance on a particular homogeneous dataset (White Robin), how might this affect the model’s adaptability or performance on non-represented populations?

### Soundness
3

### Presentation
3

### Contribution
4
