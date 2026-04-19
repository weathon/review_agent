# SOHES: Self-supervised Open-world Hierarchical Entity Segmentation

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Open-world entity segmentation, as an emerging computer vision task, aims at segmenting entities in images without being restricted by pre-defined classes, offering impressive generalization capabilities on unseen images and concepts. Despite its promise, existing entity segmentation methods like Segment Anything Model (SAM) rely heavily on costly expert annotators. This work presents Self-supervised Open-world Hierarchical Entity Segmentation (SOHES), a novel approach that eliminates the need for human annotations. SOHES operates in three phases: self-exploration, self-instruction, and self-correction. Given a pre-trained self-supervised representation, we produce abundant high-quality pseudo-labels through visual feature clustering. Then, we train a segmentation model on the pseudo-labels, and rectify the noises in pseudo-labels via a teacher-student mutual-learning procedure. Beyond segmenting entities, SOHES also captures their constituent parts, providing a hierarchical understanding of visual entities. Using raw images as the sole training data, our method achieves unprecedented performance in self-supervised open-world segmentation, marking a significant milestone towards high-quality open-world entity segmentation in the absence of human-annotated masks. Project page: https://SOHES.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces SOHES, a method designed to eliminate the need for manual annotation and facilitate the learning of hierarchical associations for open-world entity segmentation. Specifically, the authors have devised a novel approach for generating initial pseudo-labels and subsequently enhancing the segmentation model through their utilization. When compared to previous research on open-world entity segmentation, SOHES demonstrates a significant reduction in the gap between self-supervised methods and the supervised SAM.

### Strengths
1. The proposed method is highly motivated, and the results of SOHES demonstrate tremendous potential in open-world entity segmentation, even surpassing SAM's performance on certain datasets. 
2. The paper is excellently structured and provides a clear and easy-to-follow presentation.

### Weaknesses
1. The analysis of the hierarchical architecture is inadequate.
2. Certain details in the method lack clarity, and there is a noticeable absence of some ablation experiments.

### Questions
1. Regarding the generation of pseudo-labels, I am inquisitive about why the visual feature of merged patches is computed as the sum of original features rather than an average or any other operation?
2. Concerning the second local re-clustering step, I'm curious about whether it has an impact on the results for large entities and how the threshold for small regions affects the final results. 
4. In the ablation study, there is no information regarding how the ancestor prediction head contributes to the final results. Further analysis of the hierarchical architecture is needed for clarification.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces the Self-supervised Open-world Hierarchical Entity Segmentation (SOHES) approach for computer vision, which can segment entities in images beyond pre-defined classes without human annotations. SOHES uses three stages: self-exploration, self-instruction, and self-correction, generating high-quality pseudo-labels from visual feature clustering and refining them through mutual-learning. This method achieves a new standard in self-supervised open-world segmentation, eliminating the need for human-annotated masks.

### Strengths
1.	The task SOHES is seldom investigated.
2.	This paper proposed a new method to generate hierarchical proposals.
3.	This paper proposed an ancestor prediction head, which is novel.
4.	The proposed method significantly outperformed the previous methods.

### Weaknesses
1.	Although this paper divide the stages into self-exploration, self-instruction, and self-correction. But it looks like previous papers[1] that generate pseudo-labels, then training from pseudo-labels, and apply self-training to improve the model. The framework is actually quite common. So what is the core difference from the previous works in the framework?
2.     The authors claimed that “Existing segmentation models cannot predict the hierarchical relations among masks. ” However, methods like Groupvit[2] already can predict the hierarchical relations among masks.
3.	The proposed method introduced too many hyperparameters such as Theta_{merge}, Theta_{small}, Theta_{cover}. There is not explanation for the selection of some hyperparameters, such as Theta_{cover}. 
4.	Although the authors proposed the ancestor prediction head for the hierarchical segmentation. However, the mask prediction of masks at each hierarchy is independent.  I am curious if the relation modeling of hierarchy can contribute to better mask predictions. I am also curious about the effect of just predicting the hierarchical relationships. It seems that there is no ablation to verify if the ancestor prediction head can bring the performance improvements.

[1] Wang, Xinlong, et al. "Freesolo: Learning to segment objects without annotations." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
[2] Xu, Jiarui, et al. "Groupvit: Semantic segmentation emerges from text supervision." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

### Questions
See weakness

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
The authors introduced the Self-supervised Open-world Hierarchical Entity Segmentation (SOHES) method, a three-phase approach for entity segmentation. The first phase, Self-exploration, uses a pre-trained DINO model to produce initial pseudo-labels. By clustering visual features, it identifies regions representing meaningful entities. In the Self-instruction phase, a Mask2Former segmentation model refines the segmentation by training on these initial labels. Even with some label noise, the model effectively averages out inconsistencies, resulting in better mask predictions. The final Self-correction phase uses a teacher-student mutual-learning framework to further refine the model's predictions and adapt to open-world segmentation. This approach only uses raw images without human annotations. A standout feature of SOHES is its ability to segment not just whole entities but also their parts and sub-parts.

### Strengths
[Task] Unsupervised image segmentation holds significant importance, and this study successfully performs segmentation without human supervision, offering segmentation masks at multiple levels of granularity.

[The generation of hierarchical masks] The approach to generate unsupervised hierarchical masks is pretty interesting. And surprisingly, this method surpassed SAM in recall on some evaluation benchmarks. 

[Paper writing] The paper is well-articulated, effectively communicating the central ideas.

### Weaknesses
[Technical Contributions] The three phases proposed in this work are very similar to the Cut-and-Learn pipeline proposed by CutLER [1]. Self-exploration is pretty similar to the MaskCut stage in CutLER, which also leverages DINO feature for pseudo-label generation. Self-instruction is the same as the LEARN process of CutLER, which trains a model on pseudo-labels. And, the Self-correction stage can be viewed as a variant of CutLER's multi-round self-training, but with a teacher-student framework. I agree that while there are some implementation differences between the stages in SOHES and CutLER, their core concepts are largely analogous.

[Model performance] SOHES performs much worse than CutLER (9.8 vs. 2.1 on COCO and 3.6 vs. 1.9 on LVIS) in terms of the mask AP. This works show stronger performance than the previous SOTA CutLER and SAM on some benchmarks, however, the main results are AR (averaged recall). Recall is important, however, for many downstream tasks, the AP is still the most valuable evaluation metric. 

[Unfair comparison] The main baseline CutLER used Cascade Mask RCNN as the segmentation model, while this work used Mask2Former, a stronger segmentation model, as the base model. This makes the performance comparison unfair. 

[1] Wang, Xudong, Rohit Girdhar, Stella X. Yu, and Ishan Misra. "Cut and learn for unsupervised object detection and instance segmentation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3124-3134. 2023.

### Questions
My main questions are listed in the weakness section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
