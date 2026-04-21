# Multiclass Alignment of Confidences and Softened Target Occurrences for Train-time Calibration

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 5, 5

## Abstract
In spite of delivering remarkable predictive accuracy across many domains, including computer vision and medical imaging, Deep Neural Networks (DNNs) are susceptible to making overconfident predictions. This could potentially limit their utilization and adoption in many real-world applications, especially involving security-sensitive decision making. Among existing approaches to model calibration, post-hoc based techniques are simple and effective, however, they require a separate hold-out data. Lately, train-time calibration has emerged as an alternate paradigm, in which the recent methods have shown state-of-the-art calibration results. Inspired by the train-time calibration direction, in this paper, we propose a novel train-time calibration method at the core of which is an auxiliary loss formulation, namely multiclass alignment of confidences with the gradually softened ground truth occurrences (MACSO). It is developed on the intuition that, for a class, the gradually softened ground truth occurrences distribution is a suitable non-zero entropy signal whose better alignment with the predicted confidences distribution is positively correlated with reducing the model calibration error. In our train-time approach, besides simply aligning the two distributions, e.g., via their means or KL divergence, we propose to quantify the linear correlation between the two distributions which preserves the relations among them, thereby further improving the calibration performance. Extensive results on several challenging datasets, featuring in and out-of-domain scenarios, class imbalanced problem, and a medical image classification task, validate the efficacy of our method against state-of-the-art train-time calibration methods.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper present a new train-time calibration method, MACSO, featuring an auxiliary loss formulation that achieves multiclass alignment of confidence distribution and the corresponding distribution of gradually softened target occurrences. However, this method is very similar to PSKD.

### Strengths
Prove the effectiveness of the self KD method works in calibration.

### Weaknesses
1. The innovation is weak. The method, (Targets softening) is very similar to PSKD [Self-Knowledge Distillation with Progressive Refinement of Targets
]
2. The evaluated architecture is limited, only resent involved.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The study introduces a methodology for calibrating a model. The approach is based on a train-time calibration process with an auxiliary loss formulation to achieve multiclass alignment of the confidence distribution and the related distribution of progressively softened target occurrences. The authors perform extensive experiments on various in-domain, class-imbalaced, and out-of-domain scenarios to demonstrate the effectiveness of their proposed method.

### Strengths
- Proposed multiclass alignment of the confidence distribution sounds interesting.
- The motivation of the study is clear, and the authors articulate their ideas in a lucid manner.
- The authors conduct comprehensive experiments on a range of scenarios to support their claims.

### Weaknesses
- The technical contribution of the paper is limited. The contribution stated in the paper has been published earlier [1] in some way, but no citation is provided for the publication.
- The manuscript has a problem with plagiarism (please see the Ethics concerns for more details).
- This paper mainly focused on the classification task, whereas the previously published paper was based on object detection.

### Questions
This paper, in my opinion, duplicates a significant amount of content with an earlier published paper [1]. I was unable to locate any references for the paper, though. Could the authors specifically point out how this study differs from the published one to which they contributed?

[1] B. Pathiraja, M. Gunawardhana and M. Khan, "Multiclass Confidence and Localization Calibration for Object Detection," in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023 pp. 19734-19743.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This manuscript proposes to use gradually softened ground-truth label to improve the model calibration at train time. The proposed approach also employs Pearson correlation between the distilled soft label and the predictions to replace commonly used KL-divergence. The proposed approach is evaluated on several public toy datasets, and it demonstrates improved calibration performance on some of the scenarios.

### Strengths
The task of model calibration under distribution shifts is of significant importance for high-stake predictions. 

The proposed approach demonstrates improved calibration performance on public datasets. 

Most of the manuscripts are sufficiently readable.

### Weaknesses
The proposed work does bear novelty: combining knowledge distillation for label smoothing and replacing NLL/FL with Pearson correlation for smoothed results. However, these contributions may not meet the high standard of ICLR, as both self-distillation and smoothness-oriented losses (e.g., FL) have been widely discussed in uncertainty estimation / calibration, while the authors have not made significant theoretical breakthroughs nor significantly better empirical results in the current shape of submission. 

There is unfortunately a lack of clarity in the very core arguments: Sec 3.2: “The loss formulation is inspired by the intuition that as training goes, a model’s prediction becomes refined, and thus the predicted confidence scores can be gradually combined with the ground truth, to form a smoothed target distribution which has an increased entropy compared to the one-hot encoded hard targets, potentially leading to a better calibrated model.” This should be the most important argument supporting the manuscript, but what does this mean? Can the authors please make breaks to improve the clarity?  

After Eq. 3: “Additionally, in multiclass calibration, we care about preserving the class relations. Pearson correlation-based loss function allows the model to be guided appropriately to distill those truly informative multiclass relations.” What does “class relation” here mean? The argument of this paragraph is not intuitive. Instead, for multi-class scenarios, KL divergence is also computed for multiple classes as well. 

Adding smoothness constraints often comes at the cost of losing sharpness, therefore hurting categorical accuracy: the authors are therefore encouraged to report acc for the upper part of Table 2. Even looking at the lower half of Table 2, the proposed approach sometimes loses sharpness compared with NLL + MbLS. Also, what does NLL/FL in the lower part of Table 2 mean? Are they from NLL or FL or from a linear combination of them?   

Despite special considerations for multi-class tasks, when checking SCE (measuring multi-class calibration) and Acc, it is difficult to see if the proposed approach yields better multi-class results than NLL + MbLS 

Given that knowledge distillation, including this work, has been widely applied for label smoothing, the authors are encouraged to discuss at least the following similar works, and make comparisons if they find it necessary: [2-4] 

[1] ACLS: Adaptive and Conditional Label Smoothing for Network Calibration 
[2] Self-Distribution Distillation: Efficient Uncertainty Estimation 
[3] Efficient Uncertainty Estimation in Semantic Segmentation via Distillation 
[4] Distilling Calibrated Student from an Uncalibrated Teacher

### Questions
Given that label smoothing has been employed for long and it can be implemented with a wide range of loss functions, as summarized by [1], what is the major contribution of the proposed approach over existing label smoothing approaches that are discussed in [1]? The authors are encouraged to highlight this in the manuscript.

Given that ECE can be easily abused by degenerative solutions, the authors are encouraged to also report Brier scores which are proper functions for measuring calibration. 

Sensitivity on $\alpha$'s should be moved to the main text: this would be a common question raised by most readers.

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
The authors propose an auxiliary loss named MACSO for train-time calibration of deep neural networks for image classification. The original label is gradually softened and the multiclass alignment of predictions is calculated to regularize the training process. The method is validated on in and out-of-domain datasets and compared with state-of-the-art models.

### Strengths
1. Gradual targets softening during training. 
2. Both theoretical and empirical analyses of the advantages of linear correlation over KL divergence.

### Weaknesses
1. The introduction contents and design overlap a lot with the cited MDCA work [1].
2. The experiments show marginal gains from the proposed gradual target softening. The benefits of this key component are not strongly demonstrated, weakening the overall contribution.
3. The overall performance is similar to NLL/FL+MDCA, while introducing an additional hyperparameter for the gradual softening. The improvements over existing methods are incremental, and not clearly significant. The authors should provide statistical significance testing and quantify the differences to prior work. Small incremental gains may not be adequately justified as a stand-alone contribution.

[1] Ramya Hebbalaguppe, Jatin Prakash, Neelabh Madan, and Chetan Arora. A stitch in time saves
nine: A train-time regularizing loss for improved neural network calibration. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16081–16090, 2022.

### Questions
The efforts of including the medical image classification task are appreciated as it represents a safety-critical scenario. Why the scale of the SCE and ECE metrics on the medical datasets appear substantially different from the other domains?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
