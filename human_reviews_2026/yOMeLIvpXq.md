# FusionFormer: Multi-Window Fusion for Efficient Real-Time Segmentation with Vision Foundation Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Recent advances in real-time segmentation has been driven by lightweight Transformer variants. To address the limitations of such designs, state-of-the-art (SOTA) methods employ bidirectional architectures with training-only Transformer branch for long-range contextual guidance. However, these approaches typically depend on task-specific pre-trained models with limited scalability, potentially limiting their maximum performance.
In this paper, we introduce the \textit{Multi-Window Fusion Transformer} (\textbf{\textit{FusionFormer}}) to effectively leverage vision foundation models (VFMs). Specifically, we constrain self-attention computation within different window sizes and aggregate tokens within each window using varying fusion ratios. Our attention mechanism approximates attended fields from local to global while maintaining linear computational complexity, enabling better utilization of global attention guidance in Vision Transformer (ViT)-based VFMs. We first evaluate FusionFormer on the ImageNet-1k classification task, demonstrating its potential as a versatile efficient backbone. Further,extensive experiments on the ADE20K and Cityscapes datasets, coupling FusionFormer with a simple light head and DINOv2-B/14, demonstrate its excellent trade-off between segmentation accuracy and computational cost. Compared to previous methods, FusionFormer achieves a more efficient utilization of global guidance from VFMs. Our code will be released soon after the paper is accepted.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work the authors propose a backbone architecture and training framework for efficient semantic segmentation.
The authors recognize the speed/accuracy tradeoff in windowed attention, where increases in window size improve global understanding while increasing computational complexity of attention mechanisms quadratically.
FusionFormer is designed to approximate global contextual dependencies while maintaining low computational complexity.

The main architectural novelty is the Multi-Window Fusion Attention (MWFA) mechanism.
MWFA addresses the quadratic complexity of global attention by restricting computations to local windows and fusing tokens adaptively.
It captures interactions from short-range (local details) to long-range (global context) using multiple window sizes and fusion ratios.
MWFA projects QKV using linear layers, in a way that is identical to standard self-attention.

For a given window configuration (window size w), the projected QKV tensors are split into non-overlapping windows.
In order to reduce computation in larger windows (note: computational complexity is a function of w^2), an adaptive token fusion mechanism is used: a convolution-based MLP is applied to the concatenated QKV to generate fusion weights.
Tokens are fused by groups of r^2 (r: fusion ratio), weighed by the fusion weights.
In order to promote diversity in fused features, a learnable per-head bias term is added in order to bias the weights.
MWFA combines multiple configurations of window sizes and fusion ratios by weighing them based on a heuristic approximation of confidence, using attention scores.

Another novelty of the paper is the VFM guidance, which enables knowledge distillation from (potentially much less efficient) backbones, such as DINOv2.

The overall backbone architecture consists of a convolutional STEM for strides 4/8/16, two MWFA stages at stride 16, alternating MWFA and (shifted) SMWFA, and a final stride-32 convolution block.

Experiments are showing competitive results on the speed/accuracy Pareto frontier. Experiments include results for image classification (ImageNet-1k) and Semantic Segmentation (ADE20k and Cityscapes).

### Strengths
The paper is very well written, and the motivation is clear.

The idea of mixing window sizes and fusion ratios is interesting and seems novel.

The distillation framework from Vision Foundation Models, although not ground breaking, is a strong addition to the method.

### Weaknesses
The baselines that are used for comparison are old.

Prior works also aim at addressing the tradeoff of local/global attention:
* EdgeViT (ECCV2022) decomposes self-attention into convolution-based local aggregation and sparse global attention on delegate tokens.
* FasterViT (ICLR2024) employs local-to-global attention with Hierarchical Attention (carrier tokens) and appears to show competitive results, although they are not easy to compare against FusionFormer due to the different compute (FLOPS) budgets.

For VFMs, RADIO2.5 (CVPR 2025) improves upon DINOv2 on semantic segmentation (linear probe), and DINOv3 further improves from there (DINOv3 is, to my knowledge, a pre-print, not yet published in a conference, thus the authors are excused for not mentioning it).

### Questions
Did you ablate the need for the final MBConv block used for downsampling to a stride of 32? It would seem like semantic segmentation could benefit from a smaller patch size, thus why not stick to a patch size of 16?

Did you try to use a more expressive (e.g. UperNet) semantic segmentation head?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes FusionFormer, an efficient real-time segmentation foundation model that features the novel multi-window fusion transformer layers. In multi-window fusion transformer, the computational complexity is linear by constraining self-attention within the window. The experiment is carried out on ImageNet-1K dataset and ADE20K and Cityscape segmentation benchmarks. The proposed method is able to reach a balance between computational cost and segmentation accuracy.

### Strengths
- The multi-window fusion transformer can capture dependencies from short to long-range interactions, which is a significant improvement over traditional window attention.
- The proposed method improves the FPS in segmentation benchmarks, realizing real-time processing.

### Weaknesses
- The efficiency of FusionFormer-L is no as good as other model settings.
- The proposed method is mainly demonstrated in segmentation task, the reviewer is wondering if the similar architecture can be extended to be used in foudation models like SAM.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FusionFormer, a lightweight Transformer architecture tailored for real-time semantic segmentation. The primary contribution is the Multi-Window Fusion Attention (MWFA) module.

### Strengths
The overall architecture, including the backbone, light head, and VFM guidance, is clearly illustrated and described.

### Weaknesses
Let me briefly describe my guideline of reviewing the “efficient transformer” paper. Efficient transformers are hot about 3 or 4 years ago. Currently, without major novelty introduced or significant performance gain, it is hard to convince me that this paper is suitable for the whole community to read. As changing the backbone model will make re-training for everything, it does not make sense to use a new model without a major improvement.

- The paper's main claim to novelty is the MWFA module. However, it is difficult to see this as a major innovation. The core ideas—window-based attention , multi-scale processing , and adaptive token fusion —are all well-established and widely adopted concepts in the vision transformer community. The MWFA module appears to be an incremental combination of these existing techniques rather than a fundamental new approach to efficient attention.
- While the method achieves SOTA-level performance (claimed by the authors), the gains over previous work are not significant. These results are not a "significant performance gain" that would justify a new architecture in a crowded field, especially when the novelty is limited.
- In sufficient Ablation of the Core Architecture: The paper lacks a detailed ablation study on the MWFA module's design, making it hard to understand its properties or gain new insights. The specific multi-window configurations are presented without justification. A strong paper would ablate these choices, showing why this combination is optimal compared to others. This lack of deep architectural analysis makes the paper feel more like a report on a well-tuned system rather than a research paper providing new scientific insights.
- The paper's performance relied on VFM guidance, which is essentially a distillation technique. The ablation study on VFM guidance (Table 4 , Table 6 ) is more detailed than the ablation on the core architecture. On ADE20K, VFM guidance provides a major performance boost (e.g., 45.5 to 46.7 for FusionFormer-L, a 1.2 gain ). This makes it difficult to disentangle the contributions: is the performance from the novel MWFA architecture, or is it just showing that distilling from a powerful DINOv2 teacher is effective? The latter is quite obvious.


In conclusion, I think the contribution of this paper and the results are not significant. I know that doing large-scale experiments is hard. However, for such a crowd research area, it is necessary to make a contribution.

### Questions
NA

### Soundness
3

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
This paper introduces a new model for real-time semantic segmentation. This FusionFormer backbone is based on a lightweight adaptation of ViT foundation models. The method relies on one main architectural contribution: a learnable multi-scale token fusion scheme. This mechanism encodes information from deep layers of the foundation model by combining the aggregation of tokens from multiple window scales and a guidance distillation loss. Evaluations of the method on ADE20k and CityScape are presented as well as ablation studies.

### Strengths
The method shows consistent gains FPS and accuracy/mIoU on ImageNet-1k and two semantic segmentation datasets (ADE20k, Cityscape). The parameters overhead is low, and the performances are consistently above the presented baselines. 

Each components of the method are well evaluated through dedicated ablations.

### Weaknesses
My main concern is related to the absence of comparison with foundation models dedicated to semantic segmentation, and in particular, SAM and SAM2. On natural images, these models tend to show similar performances to the ones presented here, with far better generalization capabilities.

The fusion strategy appears original here, but the idea to reduce the transformer's complexity with multi-scale fusion is not novel, with seminal work going back to SWIN and PVT backbones.

### Questions
What are the performances of the vanilla VFM + linear probing on ImageNet-1k?

It is not clear if the number of parameters reported also accounts for the parameters of the VFM.

### Soundness
3

### Presentation
2

### Contribution
2
