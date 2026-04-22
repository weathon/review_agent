# Multi-Level CLIP Transfer for Open-Vocabulary Object Detection

- Avg Score: 5.20
- Decision: Reject
- Scores: 6, 4, 8, 2, 6

## Abstract
Open-vocabulary object detection (OVOD) aims to detect novel objects beyond the training categories. Recent approaches extend conventional detectors to OV detectors by combining their detector scores with zero-shot classification scores of pre-trained vision-language models such as CLIP, which is capable of identifying various visual concepts via language descriptions. However, such a simple score-level combination struggles to balance the localization and classification of novel objects: CLIP encodes global semantics for accurate classification but exhibits limited sensitivity to localization precision when scoring proposals, whereas the detector provides robust localization yet tends to misclassify novel objects as background. Instead of a trade-off, our goal is to leverage the complementary strengths of CLIP and the detector. To this end, we propose the Multi-level CLIP Transfer (MCT-Det) strategy, which effectively transfers context, alignment, and generalization knowledge from CLIP to the detector at three distinct levels. Specifically, for each region proposal: 1) At the feature-level, we refine region features by dynamically integrating CLIP’s global context via cross-attention to improve localization. 2) At the embedding-level, we integrate the region representations of CLIP and the detector into a unified embedding to couple image-text alignment with localization-awareness for reliable recognition. 3) At the score-level, we follow previous methods to exploit CLIP's zero-shot classification ability via the scores combination strategy. Building upon F-ViT, our MCT-Det achieves comprehensive improvements and outperforms state-of-the-art methods, with 52.9 AP50novel on OV-COCO and 39.8 mAPr on OV-LVIS using ViT-L/14.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a multi-level (feature level, embedding level, score level) knowledge transfer framework, which leverages CLIP to improve the localization and classification ability of the detector for novel classes.

### Strengths
**Originality**:
- The motivation for this paper is very clear, and Figure 1 intuitively illustrates the current problems in open-vocabulary object detection.

**Clarity**:
- This paper clearly elaborates the problems with existing open-vocabulary object detection methods, as well as the solutions it proposes to address them.

**Significance**:
- This paper addresses a key problem in open-vocabulary object detection and makes a significant contribution to this field.

### Weaknesses
- In Table 2, it seems that the proposed MCT-Det can only be attached to CLIPSelf, which may limit the applicability of MCT-Det.
- In Table 2, the proposed method is effectively only compared with F-ViT and DST-Det. The other methods use different backbones, resulting in an unfair comparison. I hope the authors can clarify this: are there no other methods available for a fair comparison?
- I think the title "CLIP Transfer" is not appropriate. The paper isn't about transferring CLIP itself, but about transferring knowledge from CLIP.

### Questions
- This paper uses AP50 and AP75 as evaluation metrics. However, considering the experiments aim to verify that "CLIP can improve the detector's localization ability," is there any metric that directly reflects localization capability? For example, by calculating AP using only binary target/background classification?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper finds that CLIP encodes global semantics for accurate classification but exhibits limited sensitivity to localization precision when scoring proposals, whereas the detector provides robust localization yet tends to misclassify novel objects as background. To leverage the complementary strengths of CLIP and the detector, the paper transfers the knowledge from CLIP to the detector from feature-level interaction, embedding-level fusion, and score-level combination. The resulting model outperforms its baseline F-ViT on OV-COCO and OV-LVIS.

### Strengths
1. The paper clearly demonstrates its motivation with a separate section with detailed experiment analysis, demonstrating CLIP is good at classification while the detector is good at localization, and training on the cls head with novel classes is the key to achieving high novel class performance.
2. The method is implemented on various backbones, showing that the method is scalable with the size of the backbone.
3. The ablation studies show that the performance is relatively robust to the combination weights on both novel classes and base classes, demonstrating that the method is not sensitive to the hyperparameters.

### Weaknesses
1. The contribution is incremental, as distilling or combining CLIP features for classification is well explored, including but not limited to  [1,2,3,4]. All of these papers also aim to combine the classification ability from CLIP and the localization ability from the detector. The paper does not compare with them to highlight the unique advantages of the proposed method.
2. The CLIP features in feature-level interaction and embedding-level fusion are the same (the only difference is the mean operation). But the benefits differ: the former is for better localization while the latter is for better classification. I don't get why the same feature can have totally different advantages. And there are no experimental results or analysis to support the claim.
3. Further, it is a consensus that CLIP is pretrained with an image-level objective thus it is not sensitive to the quality of bounding boxes [5,6,7]. The claim that feature-level interaction can improve the localization ability does not make sense.

[1] Open-vocabulary Object Detection via Vision and Language Knowledge Distillation. In ICLR 2022.

[2] Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection. In CVPR 2023.

[3] EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment. In ICCV 2023.

[4] A Hierarchical Semantic Distillation Framework for Open-Vocabulary Object Detection. In TMM 2025.

[5] RegionCLIP: Region-based Language-Image Pretraining. In CVPR 2022.

[6] Frozen-DETR: Enhancing DETR with Image Understanding from Frozen Foundation Models. In NeurIPS 2024.

[7] CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction. In ICLR 2024.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work aims to improve open-vocabulary object detection by better fusing conventional detectors and CLIP. Specifically, a Multi-level CLIP Transfer (MCT-Det) strategy is proposed, which introduces feature level and embedding level interactions between CLIP branch and detector branch, in additon to conventional score level fusion. Experimental results on OV-COCO and OV-LVIS demonstrate the effectiveness of the proposed method.

### Strengths
- In Figure 4, the analysis of detection AP and classification accuracy for CLIP and the detector clearly illustrates their respective strengths and weaknesses. Moreover, the analysis of individual modules within the detector clearly reveals which module’s learning is more crucial for novel classes. These empirical studies strongly support the paper’s motivation and method design, providing valuable insights.
- The proposed method achieves impressive performance on OV-COCO and OV-LVIS.
- The paper is clearly written and easy to read.

### Weaknesses
- Equation 6 is somewhat unclear. Does it require to know the groundtruth category of predicted bounding box before calculating the final classification score? Moreover, the paper does not explain how the background scores from the CLIP and Det branches are fused.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MCT-Det, a method that addresses the problem of Open-Vocabulary Object Detection. The method is based on the F-ViT architecture, with the contribution of adding the **feature-level interaction**, which is the cross-attention between the base and the novel branch, and the **embedding level fusion**, which is the fusion of the base branch with the novel branch for score regression. The experiments indicate the efficiency of the proposed method.

### Strengths
1. The feature injection from the novel branch to the base branch is interesting, since it can bring the knowledge from the novel category to the base branch for better classification of the base branch.
2. The experiment and ablation studies somehow indicate the efficacy of the proposed components on top of the F-ViT model.

### Weaknesses
There are several weaknesses in this paper:
1. Motivation: It seems the score level combination mechanism is one of the motivations and contributions of this paper. However, the geometric mean of the score combination is a well-established method in the field of OVOD [1], [2], [3].

2. Comparison with Large Vision-Language Model: Recent approaches for Large Vision-Language models, including [4], [5], can perform Object Detection for any class based on the prompt provided to the model. A comparison with their approach will help the reader understand why we need OVOD, which can only perform on a single task (Object Detection).

3. Unified head terms: The paper uses the term "unified head"; however, the classification score is the combination of the CLIP score and the classification score from the model, which can be treated as two heads. Also, it does not seem that the unified head can reduce the number of computations. Based on Table 1, the contribution of the Unified head is minor.

4. The contribution of **Feature-Level Interaction** , and **Embedding-Level Fusion** are two simple mechanisms (try to add the interaction of the novel branch to the base branch), most of the components (architecture, training, and inference strategies) rely completely on F-VLM and F-ViT. This weakens the overall strength of the paper’s contribution.

Reference:

[1] Kuo, Weicheng, et al. "F-vlm: Open-vocabulary object detection upon frozen vision and language models." arXiv preprint arXiv:2209.15639 (2022).

[2] Du, Yu, et al. "Learning to prompt for open-vocabulary object detection with vision-language model." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[3] Pham, Chau, Truong Vu, and Khoi Nguyen. "LP-OVOD: Open-vocabulary object detection by linear probing." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024.

[4] Rasheed, Hanoona, et al. "Glamm: Pixel grounding large multimodal model." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[5] Zhang, Tao, et al. "Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding." Advances in neural information processing systems 37 (2024)

### Questions
1. What is the difference in terms of performance between this proposed method for OVOD compared to Large-Vision Language Model?
2. What is the computation cost of this method using the unified heads compared with F-ViT and F-VLM, both of which use two heads?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study identifies the limitations of the score-level combination in existing open-vocabulary object detection systems (i.e., F-ViT). CLIP assigns similar scores to poor and well-located boxes, while the detector misclassifies novel objects as background. To more effectively leverage the complementary strengths of CLIP and the detector, the authors proposed MCT-Det, a Multi-Level CLIP Transfer strategy that additionally fuses CLIP and multi-level detector features at the feature and embedding levels. In the experiment, MCT-Det achieves remarkable performance gains on the two challenging open-vocabulary detection benchmarks: OV-COCO and OV-LVIS.

### Strengths
1. The motivation is clear, and the paper is easy to follow. The idea of exploiting multilevel features and employing earlier feature fusion of CLIP and object detectors is intuitive and straightforward.

2. The overall performance gains on mainstream OV benchmarks are significant, with an improvement of 8.6 mAP on OV-COCO and 5.1 on OV-LVIS.

3. The authors have provided comprehensive ablation studies on the core components of MCT-Det.

### Weaknesses
1. This paper hypothesizes the proficiency of object detectors' class-agnostic object localization, as stated in Line 185. It is recommended to provide quantitative support for this hypothesis. Moreover, it is also worth investigating whether the feature-level and embedding-level fusion can also benefit object localization (RPN).

2. The transfer performance of the object detectors on unseen datasets (e.g., objects 365) is absent in the experiment.

3. The authors use the self-distilled CLIP models in the experiment. It would be interesting to see the performance of the original CLIP and more recent SigLIP models.

### Questions
1. Is the embedding-level fusion actually an addition of CLIP embeddings and region embeddings? If so, it seems to be equivalent to multiplying scores in the score combination part.

2. Is MCT-Det also applicable to CNN-based frameworks?

### Soundness
2

### Presentation
3

### Contribution
2
