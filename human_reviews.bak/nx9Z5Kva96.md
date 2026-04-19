# Revisiting Source-Free Domain Adaptation: a New Perspective via Uncertainty Control

- Decision: Accept (Poster)
- Scores: 8, 6, 5

## Abstract
Source-Free Domain Adaptation (SFDA) seeks to adapt a pre-trained source model to the target domain using only unlabeled target data, without access to the original source data. While current state-of-the-art (SOTA) methods rely on leveraging weak supervision from the source model to extract reliable information for self-supervised adaptation, they often overlook the uncertainty that arises during the transfer process.  In this paper, we conduct a systematic and theoretical analysis of the uncertainty inherent in existing SFDA methods and demonstrate its impact on transfer performance through the lens of Distributionally Robust Optimization (DRO). Building upon the theoretical results, we propose a novel instance-dependent uncertainty control algorithm for SFDA.  Our method is designed to quantify and exploit the uncertainty during the adaptation process, significantly improving the model performance.  Extensive experiments on benchmark datasets and empirical analyses confirm the validity of our theoretical findings and the effectiveness of the proposed method. 
This work offers new insights into understanding and advancing SFDA performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The current SFDA methods that use source models for self supervised adaptation neglects the uncertainties involved in the transfer process. The authors conduct a comprehensive analysis of the two types of uncertainties arising from both positive and negative samples, providing theoretical insights for both. Based on these theoretical insights, the authors propose a dispersion control method to mitigate the uncertainty introduced by noisy negative samples. For the uncertainty arising from positive samples, the authors suggest leveraging partial labels to fully utilize the predictive uncertainty. By managing these two types of uncertainties, the authors significantly improve the model's performance.

### Strengths
Overall, this could be an important theoretical and algorithmic contribution. The paper comprehensively analyzes the two types of uncertainties caused by positive and negative samples during the adaptation process, proves their impact on model performance, and provides effective solutions.

### Weaknesses
The method of the paper requires handling both positive and negative sample uncertainties, which increases the computational complexity and training difficulty of the model.The theoretical analysis section may be too complex and difficult for general readers to understand.

### Questions
How are the weights λ-CL, λDC, and λPL for the different loss terms (L-CL, L-DC, L-PL) determined? What principles or guidelines were used to set these hyperparameter values?
How are the threshold and the number of retained labels for the partial label method determined?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies source-fee domain adaptation, with a specific focus on the uncertainty in the contrastive learning-based source-free domain adaptation (SFDA) solution. The authors comprehensively analyze two types of uncertainty including both negative and positive uncertainty through the lens of Distributionally Robust Optimization. Based on the theoretical framework the authors propose an uncertainty-control SFDA method (UCon-SFDA). Extensive experiments demonstrate the advantages of UCon-SFDA over existing SFDA approaches.

### Strengths
(i) Significance. This source-free domain adaptation problem studied in the submission is significant for real-world applications of deep learning models. The practical assumption of no source access is meaningful for privacy-sensitive scenarios such as medical data. 

(ii) Novelty. This paper provides an in-depth theoretical analysis of the overlooked uncertainty problem in previous SFDA methods within a unified DRO framework.

(iii) Quality. Extensive experiments and comparisons demonstrate that the proposed approach UCon-SFDA outperforms existing SFDA methods.

(iv) Clarity. The presentation is clear with detailed theoretical analysis and illustrative figures.

### Weaknesses
Although the approach is supported by in-depth theoretical analysis and impressive experimental results, I remain concerned about the hyperparameters setting.

Since source-free domain adaptation only has unlabeled target-domain data, it is quite difficult to accurately tune hyperparameters for the SFDA approach. Usually, one solution is to avoid involving many hyperparameters in the SFDA approach. However, the proposed UCon-SFDA necessitates the tuning of many hyperparameters including different $\lambda$ as shown in Table 6 in the Appendix. Moreover, Table 6 demonstrates that the authors adopt different hyperparameter values for different datasets. The challenge is that in practical applications, we would be given an unseen unlabeled target domain where existing hyperparameters can be worse. 

My specific questions are:

- How to tune various hyperparameters to ensure UCon-SFDA can work well in both this paper and other new SFDA tasks? 

- Hyperparameter sensitivity analysis is critical for understanding the method but missing in the paper.

### Questions
Please see the question in Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper presents a novel approach for source-free domain adaptation (SFDA), grounded in an uncertainty-guided theoretical analysis of contrastive learning-based SFDA methods. We introduce a distributionally robust optimization framework to elucidate the role of uncertainty. Additionally, the method incorporates augmentation-driven dispersion control and an optimal solution for partial label sets within a contrastive learning-based SFDA approach. The proposed method is thoroughly evaluated across four benchmark datasets.

### Strengths
- This paper induces a new aspect to solve the source-free domain adaptation problem.
- It induces an efficient method based on the guide of the proposed theoretical analysis.
- The experimental results prove the advantage of the proposed method

### Weaknesses
- Intuitively, applying cross-entropy loss on uncertain target samples with estimated pseudo labels can be problematic and may lead to negative transfer due to noisy pseudo label information, as certain samples are more likely to have accurate pseudo labels than uncertain ones..
- There are seven hyper-parameters, as shown in Table 6, which suggests that the proposed method requires specific parameter tuning to achieve promising results.
- The data augmentation alignment in Eq. 7 is, in fact, a consistency regularization between the weak and strong augmentations of the same data, which is not a novel technique as it has been well studied.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
