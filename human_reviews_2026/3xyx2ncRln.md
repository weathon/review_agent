# MaskInversion: Localized Embeddings via Optimization of Explainability Maps

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Vision-language foundation models such as CLIP have achieved tremendous results in global vision-language alignment, but still show some limitations in creating representations for specific image regions. 
To address this problem, we propose MaskInversion, a method that leverages the feature representations of pre-trained foundation models, such as CLIP, to generate a context-aware embedding for a query image region specified by a mask at test time.
MaskInversion starts with initializing an embedding token and compares its explainability map, derived from the pretrained model, to the query mask.
The embedding token is then subsequently refined to approximate the query region by minimizing the discrepancy between its explainability map and the query mask. During this process, only the embedding vector is updated, while the underlying foundation model is kept frozen
allowing to use MaskInversion with any pre-trained model. 
As deriving the explainability map involves computing its gradient, which can be expensive, we propose a gradient decomposition strategy that simplifies this computation.
The learned region representation can be used for a broad range of tasks, including open-vocabulary class retrieval, referring expression comprehension, as well as for localized captioning and image generation. We evaluate the proposed method on all those tasks on several datasets such as PascalVOC, MSCOCO, RefCOCO, and OpenImagesV7 and show its capabilities compared to other SOTA approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces MaskInversion, a novel and practical test-time optimization method for generating localized embeddings from pre-trained, frozen vision-language models like CLIP. The core problem it addresses is that models such as CLIP are trained for global image-text alignment (via the [CLS] token) and thus struggle with tasks requiring region-specific understanding.

MaskInversion works by initializing a new, learnable "Localized Embedding Token" (LETm) and iteratively optimizing it. The key insight is to use the model's own explainability map (e.g., LeGrad) as a guide. The optimization objective is to refine the LETm such that the explainability map it produces becomes spatially aligned with a given input query mask. This is achieved by minimizing a Dice loss between the explainability map and the mask. The resulting LETm can then be used as a "drop-in replacement" for the global [CLS] token in various downstream tasks, such as localized classification, referring expression retrieval, and even localized captioning and image diffusion.

### Strengths
* Novel and Elegant Method: The core idea of optimizing a new embedding by forcing its explainability map to match a target region is very clever. It's an elegant way to "invert" the model's attribution mechanism to achieve spatial control.
* Training-Free and Practical: The method works entirely at test time and keeps the powerful foundation model (like CLIP) completely frozen. This makes it extremely practical, versatile, and broadly applicable to any model that can produce a differentiable explainability map. It avoids the high cost and potential for catastrophic forgetting associated with fine-tuning.
* Strong and Comprehensive Evaluation: The paper demonstrates the effectiveness of MaskInversion across a wide array of tasks: referring expression retrieval (PhraseCut, RefCOCO), localized classification (PascalVOC, COCO, OpenImagesV7), localized captioning, and localized diffusion. The method consistently outperforms strong baselines, including those that modify the input (e.g., cropping, FGVP) and even those that require extensive fine-tuning (e.g., AlphaCLIP).

### Weaknesses
The original CLIP [CLS] token is powerful because it is aligned with text in a massive, open-vocabulary embedding space. It is not fully clear if the MaskInversion optimization, which pulls the token towards a spatial objective, fully preserves (or ideally, enhances) this fine-grained semantic alignment for general open-vocabulary classification. The regularization term ($L_{reg}$) helps, but this is a trade-off.

### Questions
Following on from the weakness, could the authors comment on the zero-shot generalization of the LETm? A valuable experiment, perhaps for the final version or future work, would be to use the ImageNet-S dataset (919 classes), which provides high-quality segmentation masks. How does the zero-shot classification accuracy of the MaskInversion LETm (using ground-truth masks) compare to the standard global [CLS] token on this large-scale benchmark? This would provide a definitive answer as to whether this localization technique consistently enhances or potentially trades off with raw zero-shot classification power.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper makes a valuable and well-executed contribution to the field of localized vision-language representation learning. It addresses a critical limitation of contrastive vision-language foundation models (e.g., CLIP) — their focus on global rather than regional alignment — with a method that balances effectiveness, efficiency, and flexibility. The work’s broad applicability across downstream tasks (referring expression retrieval, class retrieval, localized captioning, diffusion) further strengthens its impact.

### Strengths
1. Avoiding the "domain gap" of input-modification methods (e.g., RedCircle, Masked Crop) and the "data hunger" of fine-tuning methods (e.g., AlphaCLIP). By keeping the foundation model frozen and only optimizing an embedding token, MaskInversion retains the pretrained model’s knowledge while enabling task-agnostic localization.
The gradient decomposition strategy to reduce computational cost for multiple masks. This addresses a practical bottleneck of gradient-based explainability (expensive second-order derivatives) and makes the method scalable for real-world use cases.
3. The regularization loss (α) to balance local region details and global image context. This tunable tradeoff is absent in baselines and adds flexibility for tasks with varying context needs.

### Weaknesses
1. Upscaling increases the number of visual tokens (n), which affects the gradient decomposition’s efficiency (Equation 5 depends on n). The paper’s runtime ablation (Table 5) uses standard resolutions but not high-res inputs, leaving a gap in practicality for high-detail tasks (e.g., medical imaging).
2. For images with ≥5 objects, does MaskInversion’s performance degrade (e.g., due to cross-mask interference)? The current class retrieval and referring expression tasks focus on single masks, not overlapping or dense masks.

### Questions
1. In the gradient decomposition (Equation 5), you assume LETm(k)​ is independent of activations AL. Can you formally prove this independence, or provide empirical evidence that violating it (e.g., for highly complex masks) does not harm performance?
2. How does MaskInversion perform when masks overlap ? The Dice loss may struggle to distinguish overlapping regions, but this scenario is not tested.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method called MaskInversion to obtain local representations for masked query objects during test time by optimizing the objects' heatmaps and the provided binary masks. MaskInversion can be used for obtaining such local object embeddings generally from different pre-trained foundation models and the learned local embedding can be used as a drop-in replacement for improving downstream region-based tasks. The authors conduct experiments on downstream tasks including referring expression retrieval, class retrieval for segmentation. They compare with other training-free methods and show consistent improvements.

### Strengths
(1) MaskInversion is training-free can be used generally for region-based retrieval tasks. 

(2) The authors provide many ablation studies to validate the proposed method.

### Weaknesses
(1) In the paper, the authors claim MaskInversion can be used for any pre-trained model. Have the authors conduct experiments on any vision-language models such as llava? 

(2) In the section 4.2, the authors mention that for optimization of MaskInversion, they set "10 gradient descent iterations". I'm curious how this number of iterations is decided? 

(3) Have the authors try other loss functions such as simple MSE loss?

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
