# Unified Medical Image Pre-training in Language-Guided Common Semantic Space

- Decision: Reject
- Scores: 6, 6, 6, 3

## Abstract
Vision-Language Pre-training (VLP) has shown the merits of analysing medical
images, by leveraging the semantic congruence between medical images and their
corresponding reports. It efficiently learns visual representations, which in turn fa-
cilitates enhanced analysis and interpretation of intricate imaging data. However,
such observation is predominantly justified on single-modality data (mostly 2D
images like X-rays), adapting VLP to learning unified representations for medical
images in real scenario remains an open challenge. This arises from medical im-
ages often encompass a variety of modalities, especially modalities with different
various number of dimensions (e.g., 3D images like Computed Tomography). To
overcome the aforementioned challenges, we propose an Unified Medical Image
Pre-training framework, namely UniMedI, which utilizes diagnostic reports as
common semantic space to create unified representations for diverse modalities
of medical images (especially for 2D and 3D images). Under the text’s guidance,
we effectively uncover visual modality information, identifying the affected areas
in 2D X-rays and slices containing lesion in sophisticated 3D CT scans, ultimately
enhancing the consistency across various medical imaging modalities. To demon-
strate the effectiveness and versatility of UniMedI, we evaluate its performance
on both 2D and 3D images across 10 different datasets, covering a wide range of
medical image tasks such as classification, segmentation, and retrieval. UniMedI
has demonstrated superior performance in downstream tasks, showcasing its ef-
fectiveness in establishing a universal medical visual representation.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a visual-language pre-training method that can handle both 2D and 3D medical image data. It aligns different modalities of image data with their corresponding diagnostic reports and enhances the correlation between different modalities using MIM-based self-distillation.

### Strengths
This paper is well-organized and clearly described, and the figures are intuitive. Creating a unified model that can effectively handle various kinds of image data is a valuable problem. The strengths of the paper include:

1. The paper is well-written. The organization is clear, and the paper is easy to follow.
2. The studied problem is meaningful. The paper offers a unified framework to integrate multi-modal medical language-guided images into semantic space, facilitating the analysis and interpretation of complicated imaging data.
3. The approach obtains superior performance in the downstream classification and semantic segmentation tasks.

### Weaknesses
The weaknesses of the paper include:
1. The novelty appears to be somewhat incremental, as the method may seem to be a direct application of Attentive Mask CLIP proposed by Yang et al. (2023).
2. Although the method is capable of handling 3D image data, it treats them as separate 2D images and does not consider the structural aspects of 3D data.
3. Some implementation details need to be further explained. For instance, the structures of the segmentation decoder.

### Questions
1. As mentioned above, Please provide the decoder structure used for the segmentation task.
2. The medical semantic segmentation task on the RSNA and BCV datasets appears to involve predicting bounding boxes that indicate evidence of pneumonia, thus making it more akin to a detection problem rather than a segmentation problem.
3. Please elucidate the distinctions between the proposed method and Attentive Mask CLIP proposed by Yang et al. (2023).

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The UniMedI framework presents an innovative strategy for unifying the processing of diverse medical image modalities, especially 2D and 3D images, by employing diagnostic reports as a common semantic foundation. This approach facilitates the creation of consistent representations for different types of medical images. By harnessing the guidance provided by text, UniMedI excels in extracting pertinent visual information. It adeptly identifies impacted areas in 2D X-ray images and locates slices with lesions in more intricate 3D CT scans, thereby markedly improving coherence across various medical imaging formats.

### Strengths
The topic is new and clinically relevant.
It successfully develops unified representations for a variety of medical images, notably addressing both 2D and 3D image formats.
The framework's performance is thoroughly evaluated across 10 different datasets, encompassing a broad spectrum of medical imaging tasks including classification, segmentation, and retrieval.
The paper is well-structured and clearly written, making it accessible and easy to understand for readers.

### Weaknesses
The framework appears to be a multi-stage learning process rather than a true end-to-end 2D/3D multi-modal learning framework. It seems to involve selecting high-attention slices from 3D images and then training a 2D image-text encoder. If this understanding is correct, it is not as useful as a unified 2D/3D multi-modal learning framework.

While the use of t-SNE for problem definition is interesting, the paper lacks a comparative t-SNE plot in the results section to illustrate the impact of 2D and 3D co-learning. This could have provided clearer visual evidence of the model's effectiveness.

The paper does not explore straightforward alternative approaches, such as using a DENO or MAE, for learning all 2D sections from 3D volumes in conjunction with all 2D X-ray data together in a single self-supervised learning model.

There is a lack of clarity on why certain metrics, like AUC (Area Under Curve) and ACC (Accuracy), are chosen and reported in different sections. Particularly, I would suggest to use AUC over ACC in COVIDx, which could affect the clarity of the performance evaluation.

Many of the results, as shown in tables such as Table 4 (AUC), Table 5, and Table 7, indicate only marginal improvements. This raises questions about the practical significance and real-world applicability of the proposed framework.

### Questions
See the Weakness section. My further final decision will be decided based on the author's rebuttal.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a pre-training method to jointly learn 2D and 3D medical images with radiology reports. The method uses diagnostic reports as a common semantic space to create unified representations for 2D X-ray and 3D CT scans. To jointly incorporate 2D and 3D data, an attentive slice selection method was designed to select the disease-relevant 2D slices from 3D CT scans.

### Strengths
- The proposed UniMedI framework is designed to handle different imaging modalities (e.g., 2D X-rays, 3D CT scans), which is significant because medical imaging is inherently diverse, and most existing models are limited to single-dimension data.
- By using associated diagnostic reports, the framework can tap into the rich semantic information that is typically underutilized in image-only models, potentially leading to more context-aware representations.

### Weaknesses
- The description of the framework is not clear. It would be better if you could mark the (1)-(4) in Figure 3. 

- Based on the methodology description, it seems that the proposed framework is a combination of the existing methods except for the attentive slice selection. The motivation for introducing the auxiliary task is not clear. Directly saying “inspired by…” is not a good writing style. Please explicitly present the main motivation of all your methodology components. 

- The medical classification experiments are weak since most selected datasets are old and many other dedicated methods already achieved great performance. Please use the latest RSNA datasets for validation as well. 

- For the 3D segmentation tasks, the small BCV dataset cannot provided statistically significant results (other previous works also used this BCV dataset is not a good excuse). Please use larger datasets like MICCAI FLARE and AMOS.

### Questions
- The authors designed an attentive slice selection method to select the disease relevant 2D slices from 3D volume. What’s the accuracy of the method? It is not validate yet but it could be easily done by testing it on some lung/abdomen tumor CT datasets that have tumor segmentation masks, e.g., check whether the selected slices cover all the tumors and compute the accuracy. 

- Sec 3. “selected 2D input is fed into the network along with the 3D tokens..”, You already covert the 3D images into 2D key slices. What are the 3D tokens here? 

- Please explicitly clarify the unique motivations of EMA and auxiliary tasks rather than just saying "inspired by"

Minors: Please polish the writing:
Page 5. Sec. 3.3 “In Section 3.1…. Then in Section 3.1…”

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a unified framework for pre-training, unifying 2D (X-ray) and 3D (CT) modalities. The unification is realized by introducing language embeddings, which align the features of two unpaired images, but with the same pathological condition (e.g., pneumonia), into a similar space. Results show improved transfer learning performance in both 2D and 3D medical imaging tasks when pre-trained with language (pathological reports).

### Strengths
+ Table 4 demonstrates that leveraging a composite of 2D and 3D datasets during pre-training enhances performance.  This ablation analysis is clear and important.
+ Introducing language-vision framework into pre-training appears promising direction since the availability of pathological reports along with X-ray and CT images.

### Weaknesses
- The motivation of the paper, bridging features in 2D and 3D, is not validated (see Q1).
- The details of the method are confusing (see Q2).
- Lack of baseline methods for 2D and 3D pre-training (see Q3).

### Questions
1. From Figure 1, I fail to observe (c) is particularly better than (b) in terms of feature quality. I understood that the authors try to present the features of pneumonia get closer between 2D and 3D images, but this needs features of another class (e.g., nodule) as a comparison. It is possible that the proposed UniMedI makes features of all classes similar. I think the discrepancy between 2D and 3D modalities could be much larger than that among classes.

2. The illustration of Figure 3 and Figure 4 is unclear to me. What are the light orange boxes used for? What is the operation after T_3D and T_2D? The main text in the method section is not corresponding to Figure 3 or Figure 4. It is unclear how Equations (1-2) were integrated into the framework.

3. As a paper for pre-training, the authors did not compare with the many existing pre-trained models, neither 3D or 2D. For 2D X-ray imaging, there are many pre-trained models publicly available [e.g., 1-3]. For 3D CT imaging, a variety of public models are also missing [e.g., 4-6].

4. The authors did not provide sufficient reference for the baseline methods presented in Table 1. 

**Reference**

[1] Ma, DongAo, Mohammad Reza Hosseinzadeh Taher, Jiaxuan Pang, Nahid UI Islam, Fatemeh Haghighi, Michael B. Gotway, and Jianming Liang. "Benchmarking and boosting transformers for medical image classification." In MICCAI Workshop on Domain Adaptation and Representation Transfer, pp. 12-22. Cham: Springer Nature Switzerland, 2022.

[2] Yan, Ke, Jinzheng Cai, Dakai Jin, Shun Miao, Dazhou Guo, Adam P. Harrison, Youbao Tang, Jing Xiao, Jingjing Lu, and Le Lu. "SAM: Self-supervised learning of pixel-wise anatomical embeddings in radiological images." IEEE Transactions on Medical Imaging 41, no. 10 (2022): 2658-2669.

[3] Zhang, Xiaoman, Chaoyi Wu, Ya Zhang, Weidi Xie, and Yanfeng Wang. "Knowledge-enhanced visual-language pre-training on chest radiology images." Nature Communications 14, no. 1 (2023): 4542.

[4] Tang, Yucheng, Dong Yang, Wenqi Li, Holger R. Roth, Bennett Landman, Daguang Xu, Vishwesh Nath, and Ali Hatamizadeh. "Self-supervised pre-training of swin transformers for 3d medical image analysis." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20730-20740. 2022.

[5] Haghighi, Fatemeh, Mohammad Reza Hosseinzadeh Taher, Michael B. Gotway, and Jianming Liang. "DiRA: Discriminative, restorative, and adversarial learning for self-supervised medical image analysis." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20824-20834. 2022.

[6] Chen, Sihong, Kai Ma, and Yefeng Zheng. "Med3d: Transfer learning for 3d medical image analysis." arXiv preprint arXiv:1904.00625 (2019).

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
