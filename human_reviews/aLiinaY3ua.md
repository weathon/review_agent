# Diversifying Deep Ensembles: A Saliency Map Approach for Enhanced OOD Detection, Calibration, and Accuracy

- Decision: Reject
- Scores: 3, 5, 3, 3

## Abstract
Deep ensembles are capable of achieving state-of-the-art results in classification and out-of-distribution (OOD) detection. However, their effectiveness is limited due to the homogeneity of learned patterns within ensembles. To overcome this issue, our study introduces **Saliency Diversified Deep Ensemble (SDDE)**, a novel approach that promotes diversity among ensemble members by leveraging saliency maps. Through incorporating saliency map diversification, our method outperforms conventional ensemble techniques and improves calibration in multiple classification and OOD detection tasks. In particular, the proposed method achieves state-of-the-art OOD detection quality, calibration, and accuracy on multiple benchmarks, including CIFAR10/100 and large-scale ImageNet datasets.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Saliency-Diversified Deep Ensembles (SDDE) as a new diversification technique. This method uses saliency maps produced by GradCAM and makes them as different as possible. Specifically, it computes the cosine similarities between the saliency maps and uses their mean as the diversity loss function. SDDE performs better than previous ensemble methods. In addition, it performs well in OOD detection.

### Strengths
- The proposed method is very simple and easy to understand.
- It shows good performance on the OpenOOD benchmark.

### Weaknesses
- W1. SDDE performs only with CNNs because it GradCAM that is applied to CNN layers. 
- W2. The main hypothesis that ensemble diversity is proportionally related to the diversity of saliency maps, might be invalid for other datasets. The validation process proposed in the paper might not work for other cases.
- W3. SDDE can be applied to classification algorithms because CAMs are computed on the predicted classes.
- W4. Table 1 is misleading. The authors adopted additional diversity metrics, but it is obvious that their method looks to have better scores because they specifically added an additional loss function for diversity, which is based on cosine similarity.
- W5. The paper requires re-writing. The final method named SDDE_{OOD} is presented at the end of the paper, right before Section 6. The authors should describe this final method in Section 4.
- W6. It looks like MAL (Maximum Average Logit) is one of the main contributions of this paper. However, there is not enough analysis on this.

### Questions
What is the total training time of the entire framework when compared to previous approaches? I think it takes more time and has more FLOPs because of the CAMs computation.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose a Saliency-Diversified Deep Ensembles (SDDE) method for classification and OOD detection. Different from previous works which often focus on diversifying the model output, the proposed method aims to diversify the feature space for improving model performance. Specifically, SDDE leverages distinct input features for predictions via computing saliency maps and applying a loss function for diversification.

### Strengths
The idea of using saliency to enhance the diversity of input features for OOD detection is interesting.

### Weaknesses
1.	The authors followed the experimental setup and training procedure from the OpenOOD benchmark (Zhang et al., 2023). I am confused as to why they did not also follow the same evaluation setup from the OpenOOD.
2.	The authors miss several state-of-the-art OOD methods [1-4] for comparison. 
[1] Yue Song, Nicu Sebe, and Wei Wang. Rankfeat: Rank-1 feature removal for out-of-distribution detection. NIPS 2022.
[2] Andrija Djurisic, Nebojsa Bozanic, Arjun Ashok, and Rosanne Liu. Extremely simple activation shaping for out-of-distribution detection. ICLR 2023.
[3] Jinsong Zhang, Qiang Fu, Xu Chen, Lun Du, Zelin Li, Gang Wang, xiaoguang Liu, Shi Han, and Dongmei Zhang. Out-of-distribution detection based on in-distribution data patterns memorization with modern hopfield energy. ICLR 2023.
[4] Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. ICML, 2022.
3. In Table 2, why does the proposed method show inferiority on the MINIST dataset while achieving superior performance on the rest of the datasets?

### Questions
Please see weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper suggests that deep ensembles are less effective due to the homogeneity of learned patterns. So, the authors try to diversify the saliency maps of the models involved. 

By doing so, the paper claims to attain SOTA results.

### Strengths
+ Good results
+ Clearly written

### Weaknesses
- As per my understanding, saliency maps should highlight the object regions to help classification. If we make them highlight different regions, as done in Fig.1, it defeats the purpose of saliency maps. I don't agree with the idea that we should diversify saliency maps spatially, to the extent they start highlighting backgrounds. 
-Technical contributions are very limited.

### Questions
Why do authors think diversifying saliency maps is the same as diversifying features?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents SDDE, an ensembling method for classification and OOD detection. SDDE forces the models within the ensemble to use different input features for prediction, which increases ensemble diversity. Improved confidence estimation and OOD detection make SDDE a useful tool for risk-controlled recognition. SDDE is further generalised for training with OOD data and achieved SOTA results on the
OpenOOD benchmark.

### Strengths
Originality: The new aspect in the paper is actually that the diversity loss is combined with cross-entropy during training in a single optimization objective.  

Quality: The paper structure seems adequate. The balance between the theory and experiments seems adequate. The proposed method has been examined and compared against the other state of the art technologies. This paper presents rich ablation results. 

Clarity: The proposed method sounds reasonable and easy to follow. 

Significance: The paper shows that a large number of experiments have been achieved with comprehensive discussion.

### Weaknesses
Originality: The diversity loss is combined with cross-entropy during training in a single optimization objective. This additional component sounds like an incremental change. More deep investigation on the incentive of using this combination is required.

Quality: The discussion on the weaknesses of the proposed method seems missing.  

Clarity: This paper does not present sufficient explanation to the introduction of the combination of diversity loss and cross-entropy. The introduced strategy sounds like adhoc solution and requires wide discussion on the underlying mechanism. 

Significance: The proposed method does not significantly outperforms the other state of the art technologies. In some of the metrics, the proposed method seems to work well but not all or large metrics.

### Questions
1. Why the combination of diversity loss and cross-entropy is the best way to take on board?
2. To explain the convergence property of the combined solution in the paper.
3. To provide computational complexity analysis of the compared algorithms.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor
