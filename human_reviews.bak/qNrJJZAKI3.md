# FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling

- Decision: Accept (poster)
- Scores: 6, 8, 6, 8

## Abstract
Fairness in artificial intelligence models has gained significantly more attention in recent years, especially in the area of medicine, as fairness in medical models is critical to people's well-being and lives. High-quality medical fairness datasets are needed to promote fairness learning research. Existing medical fairness datasets are all for classification tasks, and no fairness datasets are available for medical segmentation, while medical segmentation is an equally important clinical task as classifications, which can provide detailed spatial information on organ abnormalities ready to be assessed by clinicians. In this paper, we propose the first fairness dataset for medical segmentation named Harvard-FairSeg with 10,000 subject samples. In addition, we propose a fair error-bound scaling approach to reweight the loss function with the upper error-bound in each identity group, using the segment anything model (SAM). We anticipate that the segmentation performance equity can be improved by explicitly tackling the hard cases with high training errors in each identity group. To facilitate fair comparisons, we utilize a novel equity-scaled segmentation performance metric to compare segmentation metrics in the context of fairness, such as the equity-scaled Dice coefficient. Through comprehensive experiments, we demonstrate that our fair error-bound scaling approach either has superior or comparable fairness performance to the state-of-the-art fairness learning models. The dataset and code are publicly accessible via https://ophai.hms.harvard.edu/datasets/harvard-fairseg10k.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a dataset for retinal disc/cup segmentation with several pre-defined attributes, which should be useful for studying the fairness problem in the medical domain. Furthermore, the authors set a baseline for the problem and define the evaluation metrics in this scenario. Overall, this work is sound and meaningful.

### Strengths
[1] Providing a dataset for fairness-related research is meaningful for the current community, along with its baseline and evaluation setting.

[2] Good writing and clear motivation

### Weaknesses
[1] ICLR might not be the best place for this paper. Other medical journals or conferences would be more suitable.

[2] There are many evaluation ways to assess the fairness problem. The selected metrics might not be the most suitable one. Please elaborate more on the motivation of baseline setting and evaluation.

[3] Some current works should be included to make the experiments sufficient. See: FairAdaBN: Mitigating unfairness with adaptive batch normalization and its application to dermatological disease classification

[4] Since most of the attributes are only for the patient level, why use the pixel-wise weights?

### Questions
See the above weaknesses

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a fundus image dataset for benchmarking the fairness of medical image segmentation methods, which is the first dataset and benchmark in this field. The authors also proposed to rescale the loss function with the upper training error-bound of each identity group to tackle the fairness issue.

### Strengths
- Novel Dataset: The paper introduced FairSeg, a new dataset for medical image segmentation with a focus on fairness. The creation of such a dataset is valuable as it addresses a gap in the current availability of medical datasets with fairness considerations.

- Fairness-Oriented Methodology: The authors proposed a fair error-bound scaling approach and an equity scaling metric. These methods represent an advanced effort to integrate fairness directly into the model training process, which could lead to more equitable healthcare outcomes.

- Open Access: I like that the author released the dataset and code for reproducibility and further research, which is a strong aspect of this work.

### Weaknesses
- Dice and IoU are equivalent (https://www.sciencedirect.com/science/article/pii/S1361841521000815), which are not necessary to be reported simultaneously. Instead, please add NSD which is suggested by metrics reloaded (https://arxiv.org/abs/2206.01653).

- nnUNet is still the state-of-the-art in many segmentation tasks. It would be great to evaluate it on your dataset.

### Questions
- The dataset was released as npz format. Could you please also release the original format?

- It would be great if you could release the trained models as well.

- Where do you plan to host this benchmark? CodaLab could be a good platform.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors introduced the new FairSeg dataset, designed to address fairness concerns in the domain of medical segmentation. Their innovative methodology centers on a fair error-bound scaling technique, which recalibrates the loss function by considering the upper error-bound within each identity group. Furthermore, they designed a new equity-scaled segmentation performance metric to facilitate fair comparisons between different fairness learning models for medical segmentation. Extensive experimentation underscores the efficacy of the fair error-bound scaling approach, demonstrating either superior or comparable fairness performance when compared to state-of-the-art fairness learning models. Furthermore, The related dataset and code are both made publicly accessible by the authors.

### Strengths
+ The paper is well-written and easy to follow. 
+ The proposed framework is technically sound.
+ The experiments are comprehensive.

### Weaknesses
There is no visualization comparison between different methods.

### Questions
1. In equation (1), a parenthesis is missing in the formula.
2. The authors proposed a new the Dice loss with a novel Fair Error-Bound Scaling mechanism, however there are experiment results to show the differences between the new dice loss and common one.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a publicly available medical fairness segmentation dataset (FairSeg) that contains 10,000 subject samples of 2D SLO Fundus images. The paper also proposes equity-scaled segmentation performance metrics to facilitate fair comparisons.

### Strengths
1. The fairness concern is an important topic, especially in medical images and the lack of segmentation dataset is a big issue. The motivation of the proposed dataset is strong.

2. The dataset contains a large amount of segmentation ground truths (10,000) and is well evaluated by authors with several SOTA learning algorithms.

3. As described by the authors, the segmentation seems to undergo a rigorous process including a hand-graded annotation by a panel of five medical professionals after initial registration.

### Weaknesses
1. The accuracy of the Nifty reg needs to be investigated since it might not be the SOTA for image registration.

### Questions
1. Why validation set is not constructed/used in selecting models in training?

2. It would be helpful to report Hausdorff distance and average surface distance along with Dice to better evaluate the methods.

3. The details of how standard deviation is computed need to be elaborated. Is it computed across the mean of for each group?

4. How is the training/testing split performed? Is it just randomly sampled without considering sensitive attributes at patient level?

5. It would be helpful to discuss the importance of registration in preprocessing using NiftyReg.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent
