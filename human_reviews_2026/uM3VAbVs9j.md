# Looking Locally: Object-Centric Vision Transformers as Foundation Models for Efficient Segmentation

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Current state-of-the-art segmentation models encode entire images before focusing on specific objects. As a result, they waste computational resources - particularly when small objects are to be segmented in high-resolution scenes. We introduce FLIP (Fovea-Like Input Patching), a parameter-efficient vision model that realizes object segmentation through biologically-inspired top-down attention. FLIP selectively samples multi-resolution patches centered on objects of interest from the input. As a result, it allocates high-resolution processing to object centers while maintaining coarser peripheral context. This off-grid, scale-invariant design enables FLIP to outperform META's Segment Anything models (SAM, SAM2 and fast variants) by large margins: With more than 440$\times$ fewer parameters, FLIP-Tiny (0.51M parameters) reaches a mean IoU of 79.90\% while SAM2-L reaches 75.87\% IoU (224.45M parameters). FLIP-Large even achieves 83.26\% mean IoU (96.6M parameters), still running about $2    \times$ faster than SAM2-L. We evaluate on six benchmarks in total. 
In five established benchmarks (Hypersim, KITTI-360, OpenImages, COCO, LVIS) FLIP consistently outperforms SAM and various variants of it. In our novel ObjaScale dataset, which stress-tests scale invariance with objects ranging from 0.0001\% up to 25\% of the image area, we show that FLIP segments even very small objects accurately, where existing models fail severely. FLIP opens new possibilities for real-time, object-centric vision applications and offers much higher energy efficiency. We believe that FLIP can act as a powerful foundation model, as it is very well-suited to track objects over time, for example, when being integrated into slot-based scene segmentation architectures.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces FLIP (Fovea-Like Input Patching), a parameter-efficient model for object segmentation. 
Inspired by biological foveation, FLIP adaptively samples multi-resolution patches centered on objects of interest, allocating high-resolution processing to object centers while maintaining coarser peripheral context. 
Key contributions include:
1. Off-grid and scale-invariant design for efficient small-object segmentation in high-resolution scenes.
2. Exceptional parameter efficiency: FLIP-Tiny (0.51M params) outperforms SAM-H (641M params) by 2.83% mIoU while running 23.6× faster.
3. State-of-the-art results on 6 benchmarks (e.g., 80.33% mIoU vs. SAM-H’s 75.41% on ObjaScale, a new dataset stress-testing scale invariance).

### Strengths
1. The fovea-like patching mechanism is an interesting idea. It introduces a biologically plausible, attention-based sampling strategy.
2. The ObjaScale dataset is a valuable contribution, filling a gap in benchmarking scale-invariant segmentation, especially for very small objects.
3. The parameter efficiency is remarkable. FLIP-Tiny (0.51M params) beats SAM-H (641M params) in mean IoU.
4. The speedup (6× faster than SAM-H) is significant and practically relevant.

### Weaknesses
1. No mention of code release or reproducibility checklist. For ICLR, open-sourcing code and models is strongly expected, especially when claiming efficiency gains.
2. While SAM and its variants are strong baselines, the paper does not compare against other efficient segmentation models. All models are from 2023, no models in 2024 and 2025. 
3. The ObjaScale dataset is synthetic and Blender-rendered. While useful, it may not reflect real-world complexity (e.g., motion blur, occlusion, and lighting variation).

### Questions
No

### Soundness
3

### Presentation
3

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
This paper proposes FLIP for object-centric segmentation. FLIP samples multi resolution patches around a Gaussian prompt, encodes them with a ViT using position aware attention, and predicts masks via sparse pixel queries. Experiments on various datasets show reasonable accuracy and efficiency with fewer parameters and lower runtime, including stable performance on small objects.

### Strengths
1. The paper is well written, and the motivation is clear and convincing.

2. FLIP provides an effective way to reduce the patch count while preserving segmentation accuracy, and it includes an efficient low level implementation of fovea patching.

3. The experimental results demonstrate reasonable and consistent performance gains across datasets.

### Weaknesses
1. The fovea inspired input patching appears incremental. STT [1] presents a similar idea to speed up SAM. The task setting is very close. Both use a prompt to focus on the object and build multi level foveated patches. FLIP relies on less structured and partly random sampling. STT uses a more structured tokenization. The paper does not cite STT. STT is CVPR 2025 and is within scope for ICLR 2026. This omission weakens the novelty claim. 

2. The comparison may be unfair. FLIP uses a Gaussian derived from the GT mask. SAM and its variants (EfficientSAM, MobileSAM, FastSAM) use a GT bounding box. These prompts provide different information. This mismatch can bias results. To ensure fairness, the evaluation should use the same input across SAM-family baselines and FLIP. A reasonable protocol is to use the GT box for all SAM-family methods and convert the box to a Gaussian for FLIP within preprocessing, then compare. This setup also aligns with practical use with detector generated boxes.

3. The paper does not provide enough detail on how the fovea patches are constructed, which harms understanding of the core contribution of this work.

[1] Segment This Thing: Foveated Tokenization for Efficient Point Prompted Segmentation, CVPR 2025.

### Questions
Please refer to the Weaknesses section. This paper has some technical contribution, but STT has already presented a similar idea. Considering the fairness issue in the comparison, only the current score can be given at this time.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FLIP, an object-centric vision transformer that employs a fovea-like input patching mechanism for efficient segmentation. The core idea is dynamically sampling multi-resolution patches centered on objects, which is biologically inspired and aims to reduce computational cost while maintaining accuracy. The authors demonstrate good performance on multiple benchmarks, emphasizing parameter efficiency and scale invariance, particularly for small objects.

### Strengths
- The fovea-like patching mechanism is interesting, which addresses the inefficiencies of full-image encoding. 

- The scale-invariant design is robustly validated on the proposed ObjaScale dataset, where FLIP outperforms SAM variants by large margins.
 
- The hierarchical inference scheme further enhances practicality for real-time applications.

### Weaknesses
- The reliance on a 2D Gaussian prior derived from ground-truth masks during training raises concerns about its generalizability to real-world scenarios, where object shapes are often irregular or annotations are imperfect. This dependency warrants further discussion.
- While FLIP demonstrates excellence in handling small objects, its performance on large-scale or complex occluded objects remains underexplored. The comparison is limited to SAMv1, omitting SAMv2 and recent object-centric models such as SlotAttention or diffusion-based segmenters. This limited comparison undermines the claim of broad superiority.
- Although the paper presents computational efficiency claims supported by inference times on an RTX 4090, a comprehensive analysis of memory footprint across diverse hardware is lacking.
- The synthetic ObjaScale dataset may not accurately capture real-world noise and variability. The paper fails to discuss potential domain shift issues when applying FLIP to unseen natural images. Moreover, the dataset evaluation lacks sufficient detail on construction methodology and scale distribution, making it challenging to assess the claimed advantages for tiny objects.
- The sparse training strategy focuses on boundary regions, but the adaptive sampling heuristic is not compared to alternative methods, and its impact on long-tail object categories remains unclear.
- Recent vision models, such as TransNeXt (CVPR'24) and OverLoCK (CVPR'25), have explored foveal mechanisms, but are not included in the related work. Additionally, DAT (CVPR'22) has proposed a deformable attention mechanism that can focus on a limited number of salient tokens, which is very similar to the mechanism proposed in this paper although this work focuses on patch embedding.
- The paper contains numerous typos and the citation format does not conform to the style of ICLR.

### Questions
Please address the weakness section.

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
This study introduces a learnable fovea-like input patching (FLIP) scheme to focus on only important regions in an image before encoding. This design is scale-invariant and saves computation. FLIP is able to achieve 1000x fewer parameters while keeping the same accuracy with SAM.

### Strengths
- This paper is presented neatly with a clear motivation. It starts with a motivation from biological fovea structure and then introduces a learnable 2D Gaussian distribution as possible focal regions.
- The proposed fovea patching does not require encoding the full image, which is able to significantly accelerate inference and training. This is verified on different benchmarks as it achieves the same performance with SAM while using only 1000x fewer parameters and being 6x faster during inference.

### Weaknesses
- Novelty issues. The idea that using explicit samping patches at multiple resolutions and then embedding them with resolution-specific modules has been proposed in the literature on different tasks. The most relevant one to this study is STT[1], which proposes foveating the input as a way to tokenize images efficiently for point-prompted segmentation. However, this study is not cited nor referred to in the manuscript. 
- The structural modifications as described in Sec 3.2 and 3.3 are mostly intuitive and incremental. The motivation of the introduction of independent LNs for Q, K, and V is not described in the manuscript, which leads to confusion to the reviewer. According to Fig. 3b, independent LNs are sequentially applied to X.  However, the features have already been normalized after the first LN, and the last two are thus equivalent to merely adding a new scaling and bias fator compared to the first one, which can actually be also learned in the weights of Q, K, and V. The necessity of adding independent LNs sequentially is thus under concern. Moreover, various modifications in regards to the normalization in transformers have been discussed in the literature, e.g.  [2], and should be correctly cited in the manuscript. 
- The paper claims a 6x accelerance over SAM. However, the mere comparison with SAM is considered not comprehensive enough as SAM is designed on a much broader downstream application scales. Therefore, FLIP is recommended to validate itself over more datasets (e.g. the 23 datasets as in SAM). The generalization ability is a special concern to the reviewer as the Gaussian distribution parameters are learned from the training datasets and irrelevant patches are filtered out. 

 [1] Schmidt, Tanner, and Richard Newcombe. "Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
[2] Menary, Stephen, Samuel Kaski, and Andre Freitas. "Transformer Normalisation Layers and the Independence of Semantic Subspaces." arXiv preprint arXiv:2406.17837 (2024).

### Questions
The reviewer look forwards to the authors rebuttal in regards to its novelty issues and recommends the authors to compare the results with the network in SST as well.

### Soundness
3

### Presentation
3

### Contribution
2
