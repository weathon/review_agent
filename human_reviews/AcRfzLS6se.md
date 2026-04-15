# Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness

- Decision: Accept (poster)
- Scores: 6, 6, 5, 5

## Abstract
Effective out-of-distribution (OOD) detection is crucial for reliable machine learning models, yet most current methods are limited in practical use due to requirements like access to training data or intervention in training. We present a novel method for detecting OOD data in Transformers based on transformation smoothness between intermediate layers of a network (BLOOD), which is applicable to pre-trained models without access to training data. BLOOD utilizes the tendency of between-layer representation transformations of in-distribution (ID) data to be smoother than the corresponding transformations of OOD data, a property that we also demonstrate empirically. We evaluate BLOOD on several text classification tasks with Transformer networks and demonstrate that it outperforms methods with comparable resource requirements. Our analysis also suggests that when learning simpler tasks, OOD data transformations maintain their original sharpness, whereas sharpness increases with more complex tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a white-box OOD detection method: BLOOD, by using the fact that the tendency of between-layer representation transformations of ID data is smoother than the corresponding transformations of OOD data for Transformer network.

-Hypothesis: During the model’s training, smooth transformations between DNN layers are learned, which corresponds to natural and meaningful progressions between abstractions for ID data. And these progressions will not match OOD data, so the transformations will not be smooth.

-The smoothness is defined as the difference of the mapping between the current representation its infinitesimally close neighbourhood's representation.

### Strengths
1. Looking at the smoothness of the layer is novel and hasn't been explored in prior research for OOD detection.

2. An unbiased estimator for the smoothness is proposed to reduce the computation.

3. BLOOD demonstrates superior performance over various OOD detection methods for RoBERTa and ELECTRA in text classification datasets.

### Weaknesses
1. The experiments exclusively focus on Transformer-based models and text classification. I am curious about the performance of BLOOD when applied to image classification datasets and Convolutional Neural Networks (CNNs). Additionally, the study lacks a comparison with the latest state-of-the-art methods for out-of-distribution (OOD) detection in CNNs, such as ASH (Extremely Simple Activation Shaping for Out-of-Distribution Detection, ICLR23), ReAct (ReAct: Out-of-distribution Detection With Rectified Activations, NeurIPS21), and DICE (DICE: Leveraging Sparsification for Out-of-Distribution Detection, ECCV2022). Furthermore, there is no analysis provided to explain why the concept of BLOOD is restricted to text classification and not applicable to other domains.

2. While Figure 1 qualitatively illustrates the smoothness difference, the author did not offer an explanation for the observed difference between in-distribution (ID) and out-of-distribution (OOD) data. I wonder whether there is an analysis or understanding of this phenomenon, specifically why the learned progressions on ID data do not align with OOD data and how to define this misalignment.

### Questions
Please see the weaknesses section, I am willing to raise my rating if the author can addresses those issues.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new OOD detection method called BLOOD, designed for a white-box setting where only pre-trained weights and model architectures are accessible, but without access to training data. It leverages the smoothness of between-layer representation transformations in ID data compared to OOD data. The method is evaluated using datasets for text classification tasks, demonstrating its superior performance over other methods with comparable resource requirements.

### Strengths
The authors propose a novel and effective scoring method for OOD detection based on the smoothness between Transformer layers that is applicable to pretrained models without access to training data. They provide a detailed performance analysis, including both strengths and limitations. The paper also presents an interesting finding regarding the relationship between smoothness and the complexity of the training dataset. The paper is well-organized and clearly written overall.

### Weaknesses
The proposed method does not perform competitively on simpler datasets, such as those with fewer classes in ID, or semantic shift as OODs. This inconsistency in performance across different datasets and settings suggests that relying solely on the smoothness of the representation may not be fully optimal for general OOD detection tasks. 

In Table 2, the calculated effect size is greater in the “Mean” approach than in the “Last” approach, which appears to contradict the results in Table 1.

### Questions
- In Section 4.3, there are certain inconsistencies in the presented results (e.g., RoBERTa model on the MG and NG datasets), which seem to challenge the hypotheses by the authors. Given the variations in performance, can you provide additional evidence or discussion, such as whether certain architectures or settings are preferred by the proposed method? 
- I’m curious if BLOOD can be extended to open-box settings. 
- Is it possible to apply BLOOD to OOD detection tasks in computer vision? Any insights or preliminary findings on this would be valuable. 
- Could you provide more detailed explanation of how the CLES measurements in Tables 2 and 3 were computed?

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
The paper introduces an out of distribution detection method called BLOOD which measures the transformation smoothness between layers by using the frobenius norm on the jacobian matrix. The proposed approach is then evaluated across several text classification dataset for OOD detection task.

### Strengths
- The proposed method is simple and can be easily applied on different tasks and network architectures, where OOD detection is required.
- The paper is easy to read and overall the paper is well written.
- The proposed method has been shown to perform well across different text classification datasets in comparison to standard OOD methods.

### Weaknesses
- The experimental results are not very strong or convincing. I would have expected some experiments on vision datasets as there has been a significant amount of OOD detection work focussing on vision datasets. Also, this approach can be easily applied on those tasks and will give better understanding how well this approach works.
- Most of the baselines considered in the comparisons are very old for OOD detection. There has been a significant amount of work focusing on last layer contributions to OOD detection similar to this paper (for eg. [a,b]). The authors should show comparisons against SOTA OOD baselines.



References:
- [a] Sun et. al. React: Out-of-distribution detection with rectified activations. Neurips, 2021.
- [b] Djurisic et al. Extremely simple activation shaping for out-of-distribution detection. arXiv preprint arXiv:2209.09858, 2022

### Questions
Please address the weakness mentioned above.

Why does BLOOD_mean performance significantly worse than the baselines in most of the cases? Whereas in some case it performs better than BLOOD_{L-1}. Can the authors explain that?

### Soundness
2 fair

### Presentation
3 good

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
This paper tackles OOD detection problems by analyzing the smoothness of the feature transformation between intermediate layers in a network in a pre-trained network. The idea behind the notion of smoothness in this paper extends from Liptischitz continuity. The method is evaluated on popular pre-trained models such as RoBERTa and ELECTRA on multiple texts analysing data sets such as SST, SUBJ, AGN etc. The results are compared with competitive baselines, which demonstrates its effectiveness in most of the cases.

### Strengths
OOD detection is an important research problem that plays a significant role in developing trustworthy AI. Hence, the paper addresses the problem which can be of interest to a larger audience. 

The method itself relies on the transformation features between the layers, which are estimated from the pre-trained models. Hence, this method does not require labelled examples from downstream tasks which is another strength of this method.

### Weaknesses
In the paper, it is mentioned that the method is robust to complex tasks and less competitive for easy tasks. However, the explanations are just based on empirical performance, lacking clear insight and understanding behind such outcomes.  

This is a similar line of work on OOD detection employing variance of gradients [a], which play a direct role in the smoothness feature transformation in the intermediate roles. 
It is better this paper acknowledges such works and argues how such methods differ from their proposed method. 

[a] Agarwal, Chirag, Daniel D'souza, and Sara Hooker. "Estimating example difficulty using variance of gradients." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

### Questions
Please see the weakness.

I wonder how this method would work in vision tasks.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
