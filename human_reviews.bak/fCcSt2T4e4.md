# SILC: Improving Vision Language Pretraining with Self-Distillation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 5

## Abstract
Image-Text pretraining on web-scale image caption dataset has become the default recipe for open vocabulary classification and retrieval models thanks to the success of CLIP and its variants. Several works have also used CLIP features for dense prediction tasks and have shown the emergence of open-set abilities. However, the contrastive objective only focuses on image-text alignment and does not incentivise image feature learning for dense prediction tasks. In this work, we propose the simple addition of local-to-global correspondence learning by self-distillation as an additional objective for contrastive pre-training to propose SILC. We show that distilling local image features from an exponential moving average (EMA) teacher model significantly improves model performance on several computer vision tasks including classification, retrieval, and especially segmentation. We further show that SILC scales better with the same training duration compared to the baselines. Our model SILC sets a new state of the art for zero-shot classification, few shot classification, image and text retrieval, zero-shot segmentation, and open vocabulary segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a pre-training framework for vision-language models, termed SILC, to enhance local image features by self-distillation. The experiments are conducted on several tasks including classification, retrieval and segmentation, and SILC achieves the comparable performance with the baseline methods.

### Strengths
This work focuses on improving the performance of vision-language models on dense prediction tasks by designing a pre-training strategy, which is critical in the real-world applications. Experiments on both global recognition and dense prediction tasks are conducted and reported.

### Weaknesses
The important technical details, especially about the local crop, should be further clarified and explained.

- The operation details of how to obtain the local crop are important and expected to be clarified, since only applying additional views has introduced significant performance improvement as shown in Table 4.

- The explanation of why the performance is improved by enforcing the embeddings of local crop to be similar to the teacher prediction. As the local view is ``a random crop over a small region of the image", it may only contain the background or part of the foreground.

- The proposed SILC enhances the model local representation by local-to-global correspondence learning, which promotes the applications of VLM on segmentation tasks. However, how does this training strategy improve the performance of not only dense prediction tasks but also global recognition tasks?

More convincing experimental results should be provided.

- Direct comparisons with the results reported by the existing works (e.g., system-level comparisons or results under their settings) are expected. This may avoid concerns about fair comparisons, such as whether the baseline methods are under-tuned when they are reimplemented.

- The results of SILC* should be used for comparison since the setting of SILC is unfair. Compared with the baseline methods, the improvements of SILC* on several tasks are marginal.

- In order to demonstrate the effectiveness of the proposed approach on dense prediction tasks, the results for open-vocabulary object detection are critical, which is missing.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a method called SILC (Self-Distillation for Improving Vision-Language Pretraining with Contrastive learning) that aims to improve vision-language pretraining models for dense prediction tasks such as segmentation. The authors propose the addition of local-to-global correspondence learning by self-distillation as an additional objective for contrastive pre-training. They show that distilling local image features from an exponential moving average (EMA) teacher model significantly improves model performance on various computer vision tasks, including classification, retrieval, and segmentation. SILC sets a new state of the art for zero-shot classification, few-shot classification, image and text retrieval, zero-shot segmentation, and open vocabulary segmentation.

### Strengths
1.	SILC applies local-to-global correspondence learning by self-distillation as an additional objective for contrastive pre-training. 
2.	Distilling local image features from an EMA teacher model improves model performance on various computer vision tasks.
3.	SILC sets a new state of the art for zero-shot classification, few-shot classification, image and text retrieval, zero-shot segmentation, and open vocabulary segmentation.

### Weaknesses
1.	The novelty of the paper is a little bit concerning, as the local-global consistency training has already been widely used in a number of works (e.g. Caron et al., 2021; Oquab et al., 2023; Zhou et al., 2022b). EMA is also a very common technique widely used in self-/semi- supervised learning community. The limitation is also pointed out by the paper.
2.	SILC claims to improve vision-language pretraining models for “dense prediction tasks”. However, only segmentation related tasks are validated in the experiments. Please also consider adding more experimental comparisons about open-vocabulary object detection (which is also very important topics for dense prediction). 
3.	Missing comparisons with some state-of-the-art CLIP variants. Please consider comparing with SOTA CLIP variants, e.g. EVA-02-CLIP[a]. 
4.	Recently, there is a concurrent work [b] which uses self-distillation for improving dense representation ability of CLIP, which is very related to this work. Please consider adding discussion about it in the related work section. 
5.	Please report the training cost (training time) of the model. 
6.	Please consider discussing the limitation of the paper. 

Minor: 
1)	Presentation of Figure1 can be improved. Please consider using different colors for the arrows and losses, to make it easier to understand. Especially, in the figure, the student takes both local crop and global crop as the input. But they are not used at the same process.

[a] Sun, Quan and Fang, Yuxin and Wu, Ledell and Wang, Xinlong and Cao, Yue. EVA-CLIP: Improved Training Techniques for CLIP at Scale. Arxiv2023.
[b] Size Wu and Wenwei Zhang and Lumin Xu and Sheng Jin and Xiangtai Li and Wentao Liu and Chen Change Loy. CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction. Arxiv2023.

### Questions
Please see above. The paper is well presented. However, the reviewer concerns the somewhat limited novelty, insufficient experiments (especially for open-vocabulary detection).

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This study introduces an innovative technique that incorporates self-distillation loss into the pre-training of a vision-language model, aiming to establish local-global consistency. It is achieved through the utilization of an exponential moving average (EMA) teacher model. The proposed approach demonstrates superior scalability compared to the conventional image-text pre-training method and yields enhanced performance in various tasks, including zero-shot classification, few-shot classification, image-to-text and text-to-image retrieval, zero-shot semantic segmentation, and open vocabulary semantic segmentation.

### Strengths
1. The method in this paper is simple and has shown to be effective on several downstream tasks including image classification, retrieval, and segmentation.

2. The authors have provided ablation studies on the components of the method and verified the effectiveness of the self-distillation across many downstream tasks.

3. The authors prove better scalability of the method over baseline image-text pre-training on the zero-shot classification on the ImageNet dataset.

### Weaknesses
1. The motivation is somewhat unaligned with the experiments. The authors discussed the challenge of open-vocabulary dense prediction tasks including image segmentation and object detection. Conceptually, the proposed method would be established as a solution to the mainstream open-vocabulary dense prediction tasks including semantic segmentation, instance segmentation, and object detection, by imposing the global-local consistency that has been proven effective for these tasks in the literature of Self-supervised Learning. However, the authors have only provided experimental support for the proposed semantic segmentation method and a wide range of image-level tasks.
2. This paper seems to be incremental and lacks novelty. The key observation that segmentation emerges from learning local and global consistency, which sets the ground for the self-distillation approach, is contributed to self-supervised learning works such as DINO. Also, an EMA teacher has been a common practice in the self-supervised learning domain. Therefore, the technical contribution of this paper is quite limited.
3. The authors have attempted to showcase the scalability of the proposed method, but only on the zero-shot classification on the ImageNet dataset. I believe such a claim should undergo a more rigorous validation on more downstream tasks, including the dense prediction ones considered in this paper.
4. The authors missed a closely related work published on CVPR 2023, i.e., PACL [1]. The latter applied a Patch Aligned Contrastive Learning approach to the image-text pretraining to enforce vision-language alignment at the dense level. Discussion and quantitative comparison with PACL is needed, especially on the zero-shot semantic segmentation benchmark. Note that PACL is trained on million-scale data instead of the billion-scale data used by SILC.
5. For the comparison in Table 2, the compared methods are trained with smaller datasets, e.g., GroupViT is trained on CC12M and YFCC(14M). Although we desire a larger-scale training of vision-language foundation models, a fair comparison with existing methods should also be presented.

[1] Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning.  Mukhoti et.al., CVPR2023

### Questions
1. The impact of SILC on object detection and instance segmentation needs to be verified.
2. How about CNN-based VLMs? There are recent works [2, 3] that reveal the effectiveness of CNN-based CLIP models for open-vocabulary detection and segmentation.  The frozen CLIP CNNs without any additional losses or finetuning can already achieve SOTA on open-vocabulary object detection and panoptic segmentation. Would this approach also apply to CNN-based models?

[2] F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models. Kuo et.al., ICLR2023.

[3] Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP. Yu et.al., NeurIPS2023.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a pre-training method, SILC, to improve vision-language tasks, especially zero-shot/open vocabulary semantic segmentation tasks. The method is a combination of CLIP and self-distillation methods (e.g. DINO), which enhance the local features of CLIP models for better performance on dense prediction tasks.

### Strengths
1. The method is simple and effective. Although it does not introduce novel objectives for pre-training, it shows a simple combination of CLIP and DINO can work well at a relatively large scale.
2. The presentation is clear and easy to follow.

### Weaknesses
1. The experiments are not convincing. In Table 1, the authors reproduce several baseline methods for comparison under a comparable setting. However, in Table 2, only the reproduced CLIP has been compared. In Table 3, only open-sourced CLIP has been compared. The adaptation costs of those reproduced methods are relatively small compared to the pre-training costs. I think the authors should also compare with those baseline methods in Table 2 and 3.
2. For large-scale pre-training, computation efficiency is also important. The proposed methods use 2 global views and 8 local views for pre-training, which should be much slower in practice. Using "example-seen" is not a good indicator. I think the authors should compare the computation costs with the referred baseline methods.

### Questions
See the weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
