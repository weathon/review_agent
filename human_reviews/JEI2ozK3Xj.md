# Multi-Label Generalized Zero Shot Chest Xray Classification  Using Feature Disentanglement and Multi-Modal Dictionaries

- Avg Score: 1.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 1, 3

## Abstract
Generalized zero shot learning (GZSL) aims to correctly predict seen and unseen classes, and most GZSL methods focus on the single label case. However, medical images can have multiple labels as in the case of chest x-rays. We propose a novel multi-modal multi-label GZSL approach that leverages feature disentanglement and multi-modal dictionaries to synthesize features of unseen classes. Feature disentanglement extracts class specific features, which are used with text embeddings to learn a multi-modal dictionary. A subsequent clustering step identifies class centroids, all of which contribute to better multi-label feature synthesis.  Compared to existing methods, our approach does not require class attribute vectors, which are an essential part of GZSL methods for natural images but are not available for medical images. Our approach outperforms state of the art GZSL methods for chest x-rays. We also analyse the performance of different loss terms in ablation studies.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents a multi-label zero-shot Chest X-ray classification method, with many ideas being explored in a recent work [1].
 


[1] Mahapatra, D., Jimeno Yepes, A. J., Kuanar, S., Roy, S., Bozorgtabar, B., Reyes, M., & Ge, Z. (2023, October). Class Specific Feature Disentanglement and Text Embeddings for Multi-label Generalized Zero Shot CXR Classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 276-286). Cham: Springer Nature Switzerland.

### Strengths
Many ideas have been validated by a published work.

### Weaknesses
Many ideas have been validated by a published work.

### Questions
None

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on multi-label generalized zero shot learning on chest xray datasets. The propose method is the same as MICCAI'23 paper "Class Specific Feature Disentanglement and Text Embeddings for Multi-label Generalized Zero Shot CXR Classification" with modification on feature generation network from Mixup to WGAN. Experiments are performed under chest xray datasets: Chest X-ray14 and CheXpert.

### Strengths
No

### Weaknesses
This paper content is largely overlapped with MICCAI'23 paper. The similarities are too many, including introduction, contribution points, methods and even experiment results. What is even more controversial is that, with a complex WGAN introduced in this paper, replacing Mixup in MICCAI'23 version, yet Table 2 experiment results are basically the same. 

I have a serious doubt for this paper reproducibility and originality. I vote for strong reject.

### Questions
.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors suggest a multi-modal, multi-label generalized zero-shot learning (GZSL) method for medical images. The method synthesizes features of unseen classes by utilizing multi-modal dictionaries and feature disentanglement. The proposed method outperforms state-of-the-art GZSL algorithms for chest X-rays.

### Strengths
- Attempt to tackle the challenging task of synthesizing features for unseen classes.
- The motivation is clearly stated and the paper is easy to follow.

### Weaknesses
- The technical contribution of the paper is critically limited.
- The experiments are limited to datasets from chest X-rays. To back up the claim that the proposed model is robust, experiments could be expanded to include more datasets, such as skin lesion images.
- The manuscript has a problem with plagiarism (please see the Ethics Concerns for more details).
- Minor but a few typos, such as 'X-ray' (it is not 'x-ray').

### Questions
Could the authors precisely explain how their work is different from the following paper?

[1] Mahapatra, D. et al. (2023). Class Specific Feature Disentanglement and Text Embeddings for Multi-label Generalized Zero Shot CXR Classification. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14221. Springer, Cham.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
