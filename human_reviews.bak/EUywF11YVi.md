# SimPLR: A Simple and Plain Transformer for Object Detection and Segmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6

## Abstract
The ability to detect objects in images at varying scales has played a pivotal role in the design of modern object detectors. Despite considerable progress in removing handcrafted components using transformers, multi-scale feature maps remain a key factor for their empirical success, even with a plain backbone like the Vision Transformer (ViT). In this paper, we show that this reliance on feature pyramids is unnecessary and a transformer-based detector with scale-aware attention enables the plain detector `SimPLR' whose backbone and detection head both operate on single-scale features. The plain architecture allows SimPLR to effectively take advantages of self-supervised learning and scaling approaches with ViTs, yielding strong performance compared to multi-scale counterparts. We demonstrate through our experiments that when scaling to larger backbones, SimPLR indicates better performance than end-to-end detectors (Mask2Former) and plain-backbone detectors (ViTDet), while consistently being faster. The code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents SimPLR. It shows that a plain vision transformer without feature pyramids but with scale-aware attention is able to achieve detection and segmentation performance comparable with pyramid designs.

### Strengths
+ The presented SimPLR detector maintain the design philosophy of simplicity in ViTs, with a plain architecture and single-scale features.
+ SimPLR indicates a clear advantage over the ViTDet baseline.
+ SimPLR reports good performance on three dense prediction tasks on COCO.

### Weaknesses
- The novelty of the work seems marginal. A very similar observation has been made recently. See (DETR Does Not Need Multi-Scale or Locality Design, ICCV’23). Learning an object detector with single-scale features is also not new. See (You Only Look One-Level Feature, CVPR’21)

- The proposed multi-head scale-aware attention is an incremental extension of the Box Attention by (Nguyen et al., 2022). The extension from the Box Attention to the Fixed-Scale Attention and Adaptive-Scale Attention is straightforward. This leads to rather limited technical contribution.

- The improvement over the BoxeR is marginal. In Table 1, SimPLR achieves the same performance as BoxeR (55.4 box AP), with also comparable FLOPs. While SimPLR is slightly efficient, the improvement is not significant.

- The claim of SimPLR outperforms the multi-scale Mask2Former is questionable. In Table 4, the comparison is unfair. SimPLR and Mask2Former use different backbones and pretraining. Such a comparison seems meaningless.

### Questions
I don't have additional questions for the authors. The paper is easy to understand, and the problem addressed is clear. 

While I appreciate the simplicity of the SimPLR and the substantial experiments performed, the results and the claim are not surprising enough or can find similar work in open literature. The technical contribution is also somewhat incremental.

### Soundness
3 good

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
This paper presents SIMPLR, a simple and straightforward transformer for object detection and segmentation. SIMPLR aims to eliminate the need for the Feature Pyramid Network structure and instead introduces scale-aware attention. This allows the backbone and detection head to effectively utilize single-scale features. Extensive experiments show that SIMPLR outperforms other detectors equipped with FPN while maintaining superior speed.

### Strengths
1. The motivation behind the proposed components, scale-aware attention, is clear and effectively addresses the issue of relying on FPN. 

2. By introducing the idea of cropping box features at different scales from a single-scale map, similar performance to FPN can be achieved. The idea is logical and makes sense in this context.

3. The experiments conducted in this study are robust and thorough. The author systematically evaluates the impact of various hyper-parameters and settings of the proposed multi-scale box attention method. Furthermore, the author compares the performance of the proposed method with state-of-the-art approaches on different datasets and tasks.

### Weaknesses
1. If I've understood correctly, the primary contribution of this article is the concept of scale-aware attention. However, I noted the absence of any graphical representation to thoroughly elucidate this complex concept. It would be immensely beneficial if a detailed, comprehensible image could be included to help readers better grasp the intricate technicalities of this key component.

2. While the paper presents some novel aspects, there are areas that fall short. From my understanding, the key innovation lies in the scale-aware attention, which builds upon box-attention [1] by assembling attention from boxes at different scales. While I appreciate this advancement, I believe it may not be sufficient on its own. I fear that it may not meet the rigorous standards expected of an ICLR paper.

3. While the author presents the performance of other methods utilizing single-scale feature maps in Figure 1, it might be more compelling to incorporate detailed, quantified comparisons in the Experiments Section. This would further validate the effectiveness of SIMPLR.

[1] BoxeR: Box-Attention for 2D and 3D Transformers

### Questions
I have stated my questions and doubts in the weakness section. Please correct me if I am wrong.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes an object detector that is based on single-scale plain ViT backbone. It follows the detector of DETR and adapts scale-aware attention for learning multi-scale features from single-scale backbone features for detection. Experiments are conducted on COCO to show the performance of both detection and segmentation. SimPLR outperform ViTDet and Mask2Former with single-scale backbone (plain ViT) on detection and segmentation task.

### Strengths
The motivation and exploration of using single-scale feature / backbone for object detection is highly meaningful. Previous attempt like ViTDet tries to keep the pretrained backbone plain, but still needs to produce feature pyramids for classic object detectors to perform well.  SimPLR uses multi-scale attention instead to get comparable performance. It extends box-attention with multiple reference windows of different scales, similar to RPN.

### Weaknesses
While SimPLR is able to get comparable/better performance with ViTDet, it is still behind sota object detector (e.g. DINO), it is not clear how SimPLR can be transferred to different advanced detectors.

### Questions
The proposed method is based on a specific box-attention approach,  can such method be applied to other transformer-based detector as well?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
