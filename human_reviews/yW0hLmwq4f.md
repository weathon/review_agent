# Delving Deep into Sim2Real Transformation: Maximizing Impact of Synthetic Data in Training

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Synthetic data has the distinct advantage of building a large-scale labeled dataset for almost free. Still, it should be carefully integrated into learning; otherwise, the expected performance gains are difficult to achieve. The biggest hurdle for synthetic data to achieve increased training performance is the domain gap with the (real) test data. As a common solution to deal with the domain gap, the sim2real transformation is used, and its quality is affected by three factors: i) the real data serving as a reference when calculating the domain gap, ii) the synthetic data chosen to avoid the transformation quality degradation, and iii) the synthetic data pool from which the synthetic data is selected. In this paper, we investigate the impact of these factors on maximizing the effectiveness of synthetic data in training in terms of improving learning performance and acquiring domain generalization ability--two main benefits expected of using synthetic data. As an evaluation metric for the second benefit, we introduce a method for measuring the distribution gap between two datasets, which is derived as the normalized sum of the Mahalanobis distances of all test data. As a result, we have discovered several important findings that have never been investigated or have been used previously without accurate understanding. We expect that these findings can break the current trend of either naively using or being hesitant to use synthetic data in machine learning due to the lack of understanding, leading to more appropriate use in future research.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper explores different aspect into using synthetic data for human detection in UAV images. This includes the interplay between the quantity of real vs. synthetic images, quantifying the distribution gap and how this is related to performance, and the impact of different testing data presenting domain shifts. The domain gap of a trained detector is measured by summing the Mahalanobis distances between in- and out-of-domain data as represented by the penultimate layer of the detector. For training with synthetic data, the PTL scheme is used, where various degrees of synthetic data are sampled based on there domain similarity with the real data.

### Strengths
+ Comprehensive analysis on the studied task
+ Reveals some interesting findings related to the use of synthetic data

More details in questions/comments below.

### Weaknesses
- As the study is limited to one task, it is not clear if results will generalize
- There are only one synthetic dataset tested, and only one synthetic to real transformation

More details in questions/comments below.

### Questions
* While the study is comprehensive and highlights some interesting aspects related to the use of synthetic data, it is also limited to one application and one type of synthetic data. It is not clear that the same conclusions would hold for different tasks and datasets.

* It could also be a different situation if the synthetic data was generated differently (for example using computer graphics vs. generative deep learning), or if the sim2real operator was different (for example something more state-of-the-art than a vanilla CycleGAN).

* One aspect related to bridging the gap between synthetic and real data is domain randomization, which is not discussed in relation to the experiments. There could be benefits in using synthetic data with a large degree of domain gap compared to real data, but where the complexity can promote optimization and generalization of the trained model. Here, a discussion around the differences in regards of sample quality/fidelity vs. diversity would be of interest.

* Is the proposed measure of domain difference used in synthetic data selection for PTL? This is not clear from the explanations. If so, how does it compare to the original method used? And if not, would it be beneficial to use the proposed measure?

* The findings formulated in the introduction, and the conclusions discussed in Section 6, are not very clear, and the potential impact is limited. It is not clear how these would translate to different datasets and synthetic data generation methods.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work explored how synthetic data affects machine learning model. They measured the distribution gap via Mahalanobis distance in feature space. They investigated about (1) the effect of synthetic data in terms of distribution gap, (2) the importance of real data, (3) the importance of sim2real transformation quality, and (4) the characteristics of the synthetic data pool.

### Strengths
- Synthetic data plays an important role in the field where data acquisition is difficult. For example, synthetic data is mainly used in optical flow [A]. Since synthetic data is widely used, research is needed to understand how it is used. This work conducted it.
- Based on Sim2Real transformation, they conducted the experiments of synthetic data in terms of the size of the real dataset, the size of the synthetic dataset, data selection methods (random, PTL), and synthetic pool comparison.

[A] FlowNet: Learning Optical Flow with Convolutional Networks, ICCV 2015

[B] Fake it till you make it: face analysis in the wild using synthetic data alone, ICCV, 2021

### Weaknesses
- The authors expanded the domain gap used in PTL to the dataset comparison. The expansion is incremental.
- In Sec. 5.1, the authors commented, “…, particularly the impact of synthetic images more significant when the amount of real data is small”. However, in Fig. 2(a), |Performance of VisDrone w/ synth - Performance of VisDrone w/o synth| is larger in 200 samples.
- In Table 2, the more real data authors use the smaller the domain gap. For example, the distribution gap is changed to 35.1 (20) → 31.7 (50) → 23.2 (100) → 19.8 (200) in the 50% column of Okutama. Why is VisDrone used more, and has the distribution gap decreased?
- The authors commented, “random selection is generally more effective in reducing the distribution gap using synthetic data than PTL”. It seems that the measure of the distribution gap used is not sufficient to delve into Sim2Real transformation alone. To delve into the synthetic data, it is better to provide the various views using another distribution measure, the number of parameters used in learning, etc.
- PTL shows a higher performance than random selection despite the high distribution gap, which means the model is biased toward the reference dataset. The authors said the quality of Sim2Real transformation is important. What is the quality authors said?

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
3 good

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
The paper provides an analysis of using synthetic data along with real data to train a model, focusing on two scenarios: one where both training and test data come from the same domain (same-domain task), and another where they come from different domains (cross-domain task). It proposes an in-depth analysis of the domain gap present in the sim2real transformations by measuring the normalized sum of the Mahalanobis distances from the training set for each test data. The authors build on top of prior work (PTL) as their setup to perform the analysis. PTL is a framework that expands the training set by adding virtual images using a GAN to apply visual transformations (PLT -- Progressive Transformation Learning for Leveraging Virtual Images in Training). The paper shows the results of varying the amounts of synthetic data and the performance on five datasets.

### Strengths
- This paper is well-written and easy to follow. The motivation of the proposed work is compelling, given the significance of using synthetic data to train a model that can be used in real-set scenarios. 

- The supplementary materials provide an extended set of experimental results varying the number of sim + real data, along with the implementation details, which adds clarity and reproducibility.

### Weaknesses
- Limited analysis: as shown in [1], the Mahalanobis distance is useful for comparing a single synthetic data point against the distribution of real data. However, prior work has shown good metrics on how similar the representations learned from different datasets are [2], including sim2real distribution shifts [3].

- Limited contribution: the proposed work heavily relies on [1], and only proposes to train the sim2real transformer by initializing from the model learned in the previous PTL iteration.

- It is hard to measure the contribution of the proposed analysis and insights without comparing the proposed PTL and established domain adaptation techniques.

[1] Shen, Yi-Ting, et al. "Progressive Transformation Learning for Leveraging Virtual Images in Training." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[2] Kornblith, Simon, et al. "Similarity of neural network representations revisited." International conference on machine learning. PMLR, 2019.

[3] Mishra, Samarth, et al. "Task2sim: Towards effective pre-training and transfer from synthetic data." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

### Questions
- Is it plausible to affirm that improving the sim2real transformation quality is more important than reducing the distribution without considering domain adaptation techniques to bridge the gap between datasets?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper is focused on maximizing the impact of synthetic data during training. It explores the sim2real transformation and discusses three key factors that influence its quality: real data, the selection of synthetic data, and the synthetic data pool. The study examines how these factors affect the effectiveness of synthetic data in improving model performance and enhancing domain generalization abilities. Additionally, the paper introduces a new evaluation metric designed to measure the distribution gap between two datasets. Using human detection in UAV-view images as the target task, the paper provides a lot of discussions and findings regarding the use of synthetic data in training.

### Strengths
+ Investigating the influential factors when utilizing synthetic data for training is both interesting and valuable for a wide range of tasks.

+ A multitude of experiments have been conducted to examine the impact of various factors on the effectiveness of using synthetic data in training for human detection in UAV-view images 

+ The discussions and findings have the potential to offer valuable insights into optimizing the utilization of synthetic data during training

### Weaknesses
- It is unclear whether the discussions, conclusions, and findings of this paper can be generalized for other applications when using synthetic data in training.

   - The experiments conducted in this paper are focused on the task of human detection in UAV-view images, which raises questions about the applicability of the derived findings and conclusions to other tasks.
   - Given the limited scale of both real and synthetic data used in the experiments (in the thousands), it is worth considering whether the paper's findings hold true when applied to large-scale synthetic training data.


- The paper does not mention or utilize FID (Fréchet Inception Distance), a commonly used metric for measuring domain gap. It would be beneficial to mention why FID is not employed in this paper and to discuss the advantages of the proposed metric over FID.

-  Some of the findings in this paper have been demonstrated in existing work on Sim2REAL

### Questions
See Weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
