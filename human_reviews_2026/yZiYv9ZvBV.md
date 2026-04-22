# Vision Transformers Need Zoomer: Efficient ViT with Visual Intent-Guided Zoom Adapter

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Vision Transformers (ViTs) have made significant strides recently, but vanilla ViT models struggle with complex scenes, particularly multi-label images and occluded objects. Humans can extract specific visual intents from complex images to guide effective classification. Inspired by human visual attention mechanisms that selectively focus on regions of interest while ignoring class-irrelevant areas, we propose ZoomViT, a novel approach that introduces visual intent-guided zoom adaptation for efficient vision transformers. ZoomViT is based on two key observations: (1) Humans and advanced models can intelligently ignore class-irrelevant areas and focus on semantically important regions through visual intent. (2) Standard ViTs can achieve superior classification accuracy when guided by adaptive zooming into regions that align with visual intent. Our approach introduces the Zoomer, a lightweight adapter with only 0.8M parameters that generates visual intent-guided score maps for image regions and dynamically adjusts patch sizes accordingly. This bio-inspired component simulates human-like visual attention by increasing patch density in regions deemed important by the visual intent guidance, while using larger patches for less critical areas. The visual intent-guided adaptation enhances both efficiency and accuracy, especially in complex images. Experiments show ZoomViT, based on the DeiT-S framework, achieves 83.8%(+4.0%) top-1 accuracy on ImageNet-1k, surpassing existing efficient state-of-the-art (SOTA) ViTs in accuracy and efficiency. The code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ZoomViT, a ViT variant enhanced by a visual intent-guided Zoomer adapter, aiming to address the limitations of vanilla ViTs in handling complex scenes. Inspired by human visual attention, where humans focus on semantically important regions while ignoring irrelevant areas, ZoomViT distills the attention of a pre-trained model to train a zoomer to localize the intent-aligned regions, and uses a detailed patchfy to enhance the representation of the important regions. A pruning strategy is further incorporated to improve efficiency.

### Strengths
- The overall model achieves good performance and better efficiency. The motivation is interesting and makes sense to me. Additionally, token pruning is combined with the Zoomer to further enhance efficiency.

### Weaknesses
- It would also be helpful to test with different resolutions to evaluate the model’s generalization ability, especially regarding extending the model (trained with lower resolution and integrated with the Zoomer) to test scenarios with higher resolution.
- I am curious about the cost of pre-extracting class-specific attention maps from the ImageNet-1k dataset before training. It would also be valuable to report the actual computational cost of the entire training procedure.
- The authors mentioned that other visual tasks will be explored as future work. In my view, this framework indeed needs further validation on additional tasks such as object detection and tracking. Furthermore, can this framework be well transferred to segmentation tasks?

### Questions
- The motivation of the overall framework is somewhat similar to extracting attention maps from DeiT-B (as a teacher model) and then training a Zoomer to guide the patchification of the student model. If DeiT-B were directly used as a teacher model to enhance DeiT-S, would it also achieve similar effectiveness and performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ZoomViT to solve vanilla ViTs' poor performance on complex scenes via a visual intent-guided Zoomer. It uses two-stage training (train Zoomer for score maps, guide dynamic patching) and achieves 83.8% Top-1 Acc (+4.0%) on ImageNet-1k, outperforming efficient ViTs.

### Strengths
1.The paper is well-substantiated overall, with attractive figures and tables.

2.Numerous experiments were conducted, and the proposed method demonstrated superior performance in the figures and tables.

3.The ablation experiments appear to be well-substantiated.

### Weaknesses
(1) Why is this method focus on multi-object tasks? The proposed method appears to be applicable to many tasks.

(2) Which human mechanism inspired the authors to adopt the dynamic patch mechanism? There seems to be no direct connection.  

(3) This paper mainly focuses on adapters. Recently, there have been many representative works on visual adapters [1-7]; the authors should compare their proposed method with these works and even supplement necessary experiments to demonstrate the innovation of the proposed method.  

[1]Visual prompt tuning, ECCV'22;

[2]5%> 100%: Breaking performance shackles of full fine-tuning on visual recognition tasks, CVPR'25;

[3]Adaptformer: Adapting vision transformers for scalable visual recognition, NeurIPS'22;

[4]1% vs 100%: Parameter-efficient low rank adapter for dense predictions, CVPR'23;

[5]Sensitivity-aware visual parameter-efficient fine-tuning, CVPR'23;

[6]Gradient-based parameter selection for efficient fine-tuning, CVPR'24;

[7]Parameter-efficient is not sufficient: Exploring parameter, memory, and time efficient adapter tuning for dense predictions, MM'24.

(4) Based on the experimental results and visualization results provided by the authors, I cannot understand why the proposed method is better than other methods. It is suggested that the authors provide a convergence comparison of different methods under the same settings. In addition, the comparisons in the visualization part should be conducted using the same images.  

(5) The experiments are mainly focused on image classification tasks. The field of computer vision (CV) is developing rapidly, and results on downstream tasks (such as detection and segmentation) are more interesting. Although the authors added experiments on YOLO in the supplementary materials, this is not sufficient.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ZoomViT with a visual-intent guided zoom adapter (Zoomer) that dynamically adjusts patch sizes and provides token re-ranking/pruning. The ZoomViT learns a Zoomer from a teacher network to focus on image regions with more class information. The Zoomer's prediction is used to identify class-related regions and divide the regions into smaller patches. And then the tokens go through a score-based re-ranking and token pruning for efficient inference. The experiment is mainly performed on the DeiT-S model and ImageNet-1k dataset, comparing ZoomViT and multiple efficient ViT models.

### Strengths
- The paper is clearly written and well-structured, making it easy to follow. The motivation, methodology, and experiments are presented in a logical manner, supported by clear figures and explanations.
- The proposed ZoomViT is conceptually intuitive and grounded in a clear motivation inspired by human visual attention, which is based on well-proven observations on ViTs in prior works. The zoom-in mechanism provides an interpretable way to adaptively allocate computation. 
- ZoomViT achieves a favorable balance between accuracy and efficiency, outperforming several strong baselines. The reported results and ablation studies demonstrate the method’s effectiveness.

### Weaknesses
- The experimental evaluation is relatively limited. Aside from the main result in Table 1, most experiments are ablation studies. Including additional benchmarks with other DeiT or ViT variants would strengthen the empirical evidence.
- The main comparison in Table 1 is difficult to interpret, as several baselines have substantially lower FLOPs than the proposed method. A fairer evaluation should compare models under similar computational budgets.
- The paper would benefit from comparisons with more recent efficient ViT models, such as Cf-ViT [r1], to better contextualize the claimed efficiency gains.
- The overall performance improvement appears modest. When comparing this paper’s Figure 5 with Cf-ViT’s results (both based on DeiT), the proposed method offers limited gains, and Cf-ViT (particularly with LV-ViT) shows a noticeably better accuracy–efficiency trade-off.
- The introduction of the Zoomer module adds additional computation during both training and inference, which may offset part of the efficiency advantage.

[r1] Chen, M., Lin, M., Li, K., Shen, Y., Wu, Y., Chao, F. and Ji, R., 2023, June. Cf-vit: A general coarse-to-fine method for vision transformer. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 6, pp. 7042-7052).

### Questions
- Did the authors evaluate ZoomViT on additional backbone architectures (e.g., LV-ViT, or Swin-T)? This would help demonstrate the generality of the proposed zooming mechanism.
- Cf-ViT is a closely related coarse-to-fine approach. Could the authors elaborate on how ZoomViT differs from or improves upon Cf-ViT, both conceptually and empirically? In particular, how does the zoomer’s intent-guided design yield advantages beyond Cf-ViT’s coarse-to-fine refinement?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ZoomViT, a visual intent-guided adaptive scaling Vision Transformer. The authors observed that vanilla ViT performs poorly when handling multi-labeled images and occluded objects because the model's visual attention is misdirected. ZoomViT guides adaptive patch size adjustments by introducing a lightweight Zoomer adapter (0.8M parameters) to generate a class-deterministic score map. This method achieves 83.8% Top-1 accuracy on ImageNet-1k, a 4.0% improvement over the DeiT-S baseline.

### Strengths
1. The paper effectively connects human visual attention mechanisms to model design, demonstrating through simple experiments how zooming helps ViT classify correctly.
2. The Zoomer adds only 0.8M parameters while achieving 4.0% accuracy improvement, offering an attractive efficiency-performance trade-off.
3. Thorough evaluation across multiple datasets (ImageNet-1k, ImageNet-A), extensive comparisons with SOTA methods, downstream task transfer, and detailed ablation studies.

### Weaknesses
## Weaknesses

1. Visual attention and adaptive patch sizing are not new concepts. The main contribution is engineering combination rather than fundamental innovation, with insufficient differentiation from existing works like DynamicViT and PS-ViT.

2. The method depends on a pre-trained DeiT-B teacher model, limiting generalizability. The two-step inference overhead is not thoroughly analyzed. The inconsistency between training (random α∈[0,1]) and inference (fixed α=0.03) lacks theoretical justification.

3. Unfair baseline comparisons due to different training strategies. Missing comparisons with the most relevant adaptive patching methods.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
