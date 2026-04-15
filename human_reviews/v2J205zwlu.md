# UniPose: Detecting Any Keypoints

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 6

## Abstract
This work proposes a unified framework called UniPose to detect keypoints of any articulated (e.g., human and animal), rigid, and soft objects via visual or textual prompts for fine-grained vision understanding and manipulation. Keypoint is a structure-aware, pixel-level, and compact representation of any object, especially articulated objects. Existing fine-grained promptable tasks mainly focus on object instance detection and segmentation but often fail to identify fine-grained granularity and structured information of image and instance, such as eyes, leg, paw, etc. Meanwhile, prompt-based keypoint detection is still under-explored. To bridge the gap, we make the first attempt to develop an end-to-end prompt-based
keypoint detection framework called UniPose to detect keypoints of any objects. As keypoint detection tasks are unified in this framework, we can leverage 13 keypoint detection datasets with 338 keypoints across 1,237 categories over 400K instances to train a generic keypoint detection model. UniPose can effectively align text-to-keypoint and image-to-keypoint due to the mutual enhancement of
textual and visual prompts based on the cross-modality contrastive learning optimization objectives. Our experimental results show that UniPose has strong fine-grained localization and generalization abilities across image styles, categories, and poses. Based on UniPose as a generalist keypoint detector, we hope it could serve fine-grained visual perception, understanding, and generation.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a unified framework named UniPose to estimate keypoints for any object with visual and tuxtual prompts. The trained model could generalize to cross-instance and cross-keypoint classes, as well as various styles, scales, and poses.  The authors unified 13 keypoint detection datasets with 338 keypoints across 1,237 categories over 400K instances, and employed such large-scale dataset to train the UniPose model, and obtained a generalist keypoint detector.

### Strengths
1) This work arranges a large scale keypoint detection dataset.
2) The paper proposes a large keypoint detection model with visual and tuxtual prompts.
3) The proposed model shows good performance in generalization.
4) The experiments are numerous.

### Weaknesses
Overall, the current version of paper does not provide sufficient explanations and in-depth analyses on major points. There are lots of experiments on comparison, but the in-depth experimental analyses (e.g., on prompts) are more necessary and meaningful.

- The most important one weakness is that there is no clear explanation for the textual prompts of keypoints. For examples, clothing or table could have ambiguous keypoint names, how to employ textual prompts to detect their keypoints? 

- Another important issue is that all the visualizations are shown without prompts and thus difficult to analyse. It is critical for understanding the effects of various prompts.

- In Section 4.1, why not apply UniPose in the setting of class-agnostic pose estimation as in CapeFormer? The comparison with adapted CapeFormer is unfair and unclear for indicating the performance of UniPose on detecting unseen objects.

- Lacking analysis for Section 4.3. The performance gaps are trivial, considering the differences on data and tasks.

- In Section 4.4, how to evaluate the SOTA open-vocabulary object detector on keypoint-level detection? Is is feasible to represent each keypoint with a small bbox in fine-tuning?

- Why does CapeFormer drop a lot in Tab. 12? Are the training and test resolutions of CapeFormer consistent? 

- What are UniPose-T and UniPose-V in Tab. 4?

- In Section 4.4, what is CLIP score? How to analyse the CLIP scores of UniPose and CLIP in Fig. 7?

### Questions
See Weaknesses*

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper makes the first attempt to propose an end-to-end coarse-to-fine keypoints detection framework named UniPose trained on a unified keypoint dataset, which can detect the keypoints of any object from the instance to the keypoint levels via either visual prompts or textual prompts and has remarkable generalization capabilities for unseen objects and keypoints detection. This work unifies 13 keypoint detection datasets containing 338 keypoints detection over 400K instances to train a generic keypoint detection model. Compared to the state-of-the-art CAPE method, this method exhibits a notable 42.8% improvement in PCK performance and far outperforms CLIP in discerning various image styles.

### Strengths
* Originality：
The first attempt of an end-to-end keypoints detection framework is developed by combining visual and textual prompts. Through the mutual enhancement of textual and visual prompts, this method can have strong fine-grained localization and generalization abilities for class-agnostic pose estimation and multi-class keypoints detection tasks.

* Quality：
The method description is detailed, and it is illustrated in the accompanying figures. 

The experiment is fully configured, taking into consideration unseen objects, expert model, general model, and open-vocabulary model. Comparative tests are conducted.

* Clarity
The paper exhibits a well-structured logical flow, accompanied by an appendix that provides an extensive overview of the work, including the introduction of the dataset, supplementary experiments, algorithmic limitations, and more.

* Significance
This paper Unifies 13 datasets to build a unified keypoints detection dataset named UniKPT. And the authors say that each keypoint has its own text prompts, and each category has its default set of structured key points. This unified dataset with visual and textual prompts can provide data support for point detection tasks in future work.

### Weaknesses
There are no complete examples of textual prompts.

In 4.4, the Fig.7 should be Tab.7.

### Questions
For the Inference Pipeline, whether only one prompt is used as input, and whether it will work better if two prompts are used?

### Soundness
3 good

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a so-called UniPose to detect various types of keypoints. The UniPose takes text or image with keypoint annotations as prompts, in order to detect the corresponding keypoints in query image. In order to support both modalities of prompts, the visual prompt encoder, textual prompt encoder, and two decoders are developped. The model is trained on 13 keypoint detection datasets with 338 keypoints, and the results show it can detect varying types of keypoints to some extent.

### Strengths
1) A model which supports textual or visual prompts are proposed.

2) The visualization shows the effectiveness to some extent.

### Weaknesses
1) Reading through the paper, it lets reviewer feel that the writing requires significant improvement and the math symbols are in mess. Moreover, there are already many works in literature using visual prompts such few-shot keypoint detection [1,2,3,4], and also the works regarding textual prompts such as [5,6]. The paper should fully discuss these two types of related works. From the technical perspective, it shows little advance compared to existing works. The simple combination of both types of prompts make the contribution and novelty of this work weak. 

    [1] Metacloth: Learning unseen tasks of dense fashion landmark detection from a few samples @ TIP21

    [2] Few-shot keypoint detection with uncertainty learning for unseen species @ CVPR'22

    [3] Pose for everything+Towards category-agnostic pose estimation @ ECCV'22

    [4] Few-shot Geometry-Aware Keypoint Localization @ CVPR'23

    [5] Clamp: Prompt-based contrastive learning for connecting language and animal pose @ CVPR'23

    [6] Language-driven Open-Vocabulary Keypoint Detection for Animal Body and Face @ arxiv'23

2) Some claims may not be true. For example, ``Xu et al. (2022a) first proposed the task of category-agnostic...". Work [3] may not be the first work as [1-2] are earlier. Moreover, what is the meaning of "the keypoint to keypoint matching schemes without instance-to-instance matching are not effective"? The keypoint representation can also aggregate the global information or context.

3) Given the existence of a similar approach such as CLAMP [5], it's essential to evaluate the performance of your method when compared to CLAMP. This evaluation should be conducted on the Animal pose dataset within the context of a five-leave-one-out setup. In this setting, the model is trained on four different species while being tested on the remaining one species. The paper should include the results of PCK@0.1 to facilitate meaningful comparisons.

3) Coarse-to-fine strategy already appears in paper [2023-ICLR-Explicit box detection unifies end-to-end multi-person pose estimation]; while the ideas of using prompts based keypoint detection already appears in FSKD, CLAMP, etc.

4) Some details are missing. This work uses CLIP as image and text encoder. The CLIP generally takes an image with size of 224 as input, while in table 2 and 3, the UniPose takes original image or image with size of 800 as input. Will the high-resolution input slow down the speed? What is the size of feature map after passing image (e.g. 800) through CLIP? A step forward, will the CLIP retain its prior knowledge after using such a high-resolution input?

    Moreover, how to select the visual object feature and textual object feature to produce $Q_{obj}$? How to select if both exist?

5) In table 1, some of datasets are already included in MP-100. For example, COCO, AP-10K, etc. So what is the meaning of counting it again to build the dataset of UniKPT?

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper discusses a new framework called UniPose, which aims to detect keypoints in various objects, including articulated, rigid, and soft ones, using visual or textual prompts. Keypoints are pixel-level representations of objects, especially articulated ones. Current fine-grained promptable tasks focus on object instance detection and segmentation but struggle with identifying detailed structured information, such as eyes, legs, or paws. The UniPose framework is the first attempt to create an end-to-end prompt-based keypoint detection system that can be applied to any object. It unifies various keypoint detection tasks and leverages multiple datasets to train a generic keypoint detection model. UniPose aligns textual and visual prompts through cross-modality contrastive learning optimization, resulting in strong fine-grained localization and generalization capabilities across different image styles, categories, and poses. The framework is expected to enhance fine-grained visual perception, understanding, and generation.

### Strengths
1. This paper is well-written and easy to follow.
2. This article provides a comprehensive summary of previous keypoint research, highlighting both its strengths and weaknesses, and uses this as motivation to propose its own approach.
3. The methodology in this article is well-designed, combining both text and image prompts for keypoint detection .

### Weaknesses
This article combines two approaches to prompts, but lacks in-depth analysis of the strengths and weaknesses of both modalities for this task:

1.What are the advantages and disadvantages of each prompt individually, and can you provide some visual results?
2. Can the strengths and weaknesses of the two prompts complement each other?
3. Is it possible to dynamically weight the two prompts? For example, can text be prioritized when there are no suitable images to serve as prompts?

### Questions
Please see the weakness. If you address my concerns, I am willing to improve my score. Thanks.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
