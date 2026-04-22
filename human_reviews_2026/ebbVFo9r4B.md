# Object-level self-distillation with bounding-box weak supervision improves vision pretraining

- Avg Score: 4.00
- Decision: Reject
- Scores: 0, 6, 6

## Abstract
Self-distillation has become a central paradigm for pretraining vision transformers (ViTs). Existing approaches typically operate at the image level and assume that different augmentations of the same image preserve semantic content to be distilled. This premise breaks down in complex scenes with multiple objects with randomly sampled data augmentations. To tackle this, we introduce ODIS (Object-level Self-Distillation), a new framework that refines the self-distillation objective to the level of individual objects using bounding boxes that encapsulate objects. ODIS leverages object-aware cropping to ensure that teacher and student views depict the same object, and employs masked attention to focus the learning signal on objects. Applied to ImageNet-1K, ODIS outperforms image-level distillation methods such as iBOT across both image-level and patch-level benchmarks, and its features transfer better to downstream classification and retrieval tasks. Moreover, ODIS is robust to bounding box noise: using two different off-the-shelf extractors, it consistently improves over SOTA baselines. Our results highlight the importance of object-centric supervision in scalable representation learning and demonstrate how pretrained tools can be integrated into distillation pipelines to enhance generalization.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes using object bounding boxes to guide the learning process in self-distillation frameworks such as DINO and iBoT: Instead of using random crops, the paper uses annotated bounding boxes as image crops. In order to build bounding boxes-aware representation, the paper employs masked attention in each self-attention module to limit the attention of the [OBJ] token only to patches appearing in the bounding boxes.

The paper shows in experiments that using bounding boxes during training, the proposed method ODIS outperforms DINO and iBoT.

### Strengths
The paper provides evidences showing that using annotated bounding boxes to guide model training in self-distillation framework like DINO and iBoT brings some benefits.

### Weaknesses
In the current form, I am not sure that the paper brings any new insights into the literature. it is widely expected that using annotated bounding boxes -- which are expensive to obtain, several times more expensive than class label annotations -- would improve the model performance compared to models trained without using any annotations. The comparisons included in the paper are not fair at all.
If the central claim is "pretraining pipelines should default to using bounding boxes" (L.306) with the goal of making the best model possible regardless of the annotation cost, the paper should demonstrate the method's effectiveness at a much larger scale and compare to the best models such as SigLIP, DINOv3, etc. At this scale, the results presented in the paper look trivial.

The idea of the paper is very close to Mishra et al. (https://arxiv.org/pdf/2112.00319) which stays in the self-supervised setting by using unsupervised object discovery methods to obtain the boundign boxes. The paper could consider using a similar approach for generating bounding boxes.

I am not sure the 12 fine-grained datasets considered in section 5.2 could be considered as "out-of-distribution" with respect to ImageNet. There is a significant overlap in visual concepts between these datasets and ImageNet.

### Questions
I encourage the authors to consider using bounding boxes generated in an unsupervised manner to better demonstrate the method's effectiveness, or attempt at building the best possible model in a very large scale.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ODIS (Object-level Self-Distillation) for self-supervised learning in multi-object scenes, where DINO leads to inconsistent learning signals.
ODIS introduces two main components: (1) object-aware cropping that uses bounding boxes to ensure teacher and student views depict the same object, and (2) masked attention that constrains the [OBJ] token to attend only to patches within the object boundary. Experiments demonstrate improvements over strong baselines across ImageNet k-NN classification, linear probing, and transfer learning tasks using both ground-truth and automatically extracted bounding boxes from YOLO and MAVL.

### Strengths
- The paper identifies a limitation in current self-distillation methods where random crops in multi-object scenes can lead to semantic inconsistency between teacher and student views. The issue is meaningful and worth exploring.
- The object-aware cropping and masked attention mechanisms are technically sound and plausibly address the identified problem. The integration with existing methods like iBOT is clean and well-designed.
- Results show consistent improvements across multiple benchmarks and different model sizes (ViT-S, ViT-B, ViT-L).
- The paper is well written and easy to follow.

### Weaknesses
- While scaling is compute-intensive, scalability is increasingly expected for modern self-supervised learning approaches. The method would be significantly more compelling if its scalability were demonstrated on large, multi-object datasets such as SA-1B, OpenImages, or Objects365. Such experiments would greatly strengthen the significance and practical impact of this work, but are unfortunately absent in the current submission.
- Can the authors provide empirical comparisons to other multi-object self-supervised learning methods, such as Odin, SelfPatch, SlotMIM, and CAPI? Including these baselines would help clarify the relative advantages and limitations of the proposed approach.

References:
- [Odin] Object discovery and representation networks
- [SelfPatch] Patch-level Representation Learning for Self-supervised Vision Transformers
- [SlotMIM] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning
- [CAPI] Cluster and Predict Latent Patches for Improved Masked Image Modeling

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Object-level Self-Distillation (ODIS), a method that brings object-level supervision into self-distillation without requiring class labels. The key idea is to ensure that the student and teacher networks distill knowledge from the same semantic object, rather than arbitrary random crops that may contain different entities.
Two technical contributions define ODIS:Uses bounding boxes during training to repeatedly sample two augmentations until both the teacher and student views include the same object region;Introduces a dedicated [OBJ] token, guided by a binary attention mask derived from the bounding boxes, so that [OBJ] aggregates information only from patches inside the object region.
With this setup, ODIS explicitly shifts the distillation granularity from image-level to object-level. Experiments on ImageNet-1K and COCO show consistent gains over DINO and iBOT across multiple tasks.
At inference, the paper further explores how to utilize bounding boxes (blackout, cropping, masked attention) and finds that ODIS performs best when inference masking aligns with its training-time attention mechanism.

### Strengths
1.Well-motivated problem: The authors correctly identify a key limitation of existing image-level self-distillation (e.g., DINO, iBOT): random cropping may cause student and teacher to view different semantic entities, leading to inconsistent or entangled representations. ODIS directly resolves this by enforcing object-level alignment.
2.Simple yet effective method: The integration of object-aware cropping and masked attention is conceptually clean and practically compatible with existing self-distillation frameworks. It only requires bounding boxes and minimal architectural changes to ViT, making it accessible for replication.
3.Comprehensive experiments:
On ImageNet and COCO, ODIS outperforms DINO/iBOT on all image-level evaluation metrics (Table 1).
On 10 transfer benchmarks, ODIS yields an average +1.7 pp improvement in linear probing (Table 2).
On retrieval tasks (ROxf/RPar), it matches or surpasses DINO and significantly outperforms iBOT (Table 3).
On dense tasks (VOC/ADE20K/COCO-Stuff), both dense k-NN and linear segmentation improve notably (Fig. 5).
Even with noisy bounding boxes from YOLO/MAVL, ODIS still beats baselines (Table 4).
4.Insightful inference analysis: Figure 6 and Section 5.3 show that using bounding boxes during evaluation benefits all models, but ODIS benefits the most, validating its object-centric pretraining.
5.Practical implications: The paper highlights that object annotations—often freely available—can substantially enhance self-distillation without labels. This could inspire broader adoption of “weakly spatially supervised” SSL paradigms.

### Weaknesses
1.No training-time comparison for “cropping only” vs “masked attention.”
Figure 6 compares blacking-out, cropping, and masked attention at inference time, but not during training. An ablation with “object-aware cropping only (no masked attention)” under the same compute budget would clarify the independent contribution of each component. 

2.Computation and efficiency not quantified.
The method involves up to 20 resampling trials to ensure box hits and uses masked multi-head attention in every transformer layer. The paper does not report training throughput (images/s), memory usage, or wall-clock time compared to DINO/iBOT. This makes it hard to assess efficiency trade-offs.

3.Limited analysis of bounding box quality and quantity.
Although Section 5.3 uses YOLO/MAVL for pseudo-boxes, there is no systematic study of performance vs. box quality (IoU thresholds, jitter, missing/false boxes, small objects). Such an analysis would strengthen claims of robustness.

4.Missing comparison to stronger modern baselines.
While DINO/iBOT are appropriate, recent methods such as DINOv2 or MAE + DINOv2 hybrids are omitted. Even a smaller-scale controlled comparison would clarify whether ODIS still holds advantages beyond scale improvements.

5.Scalability to unlabeled, box-free datasets not fully explored.
The method relies on bounding boxes. The paper notes that auto-generated boxes help, but large-scale web datasets often lack or contain noisy localization. Quantifying cost, bias, and privacy aspects of large-scale box generation would help position the method’s practicality.

6.Limited component-wise ablations.
Appendix E only reports ablations on the auxiliary image-level term Li and mask source. More detailed component ablations (cropping-only, masking-only, both) and sensitivity studies (teacher temperature, EMA coefficient, mask ratio) are missing.

### Questions
1.What are the throughput, GPU memory, and training time per epoch compared to DINO/iBOT under identical batch and resolution? How much extra overhead is caused by 20× resampling?

2.Have you tried synthetic perturbations to bounding boxes (e.g., random shifts, scaling, false positives)? How robust is ODIS under such distortions?

3.Why does training only sample one bounding box at a time? Have you tried using multiple [OBJ] tokens simultaneously for multi-object distillation?

4.Could you provide an ablation separating the effect of object-aware cropping and masked attention, trained under equal compute? This would reveal whether the main gains come from the attention mechanism or simply better aligned crops.

5.Beyond frozen features, how does ODIS perform when fine-tuned on detection or segmentation tasks (e.g., Mask R-CNN, Deeplab)? This would concretely demonstrate the value of object-level pretraining.

6.Please report or analyze the effect of teacher/student temperatures, EMA decay λ, and whether [OBJ] and [CLS] heads share parameters.

### Soundness
3

### Presentation
3

### Contribution
3
