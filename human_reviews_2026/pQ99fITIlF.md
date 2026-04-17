# Supervised Optimization of a Context-Aware Color Transfer Model

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
The task of image color style transfer aims to apply the color characteristics of a reference image to a content image while preserving its texture and structural integrity. However, two key challenges hinder effective training: (1) the scarcity of high-quality ground truth (GT) images for supervised learning, and (2) color distortions from suboptimal feature fusion.
To address the first issue, we propose a novel GT generation strategy—the first systematic method, to our knowledge, for producing high-quality GT images. A source image is recolored into two variants, and identical regions are randomly cropped to form a content image, a style image, and a structurally aligned GT image with distinct color styles, enabling reliable and precise supervision.
For second issue, we propose the Context-Aware Color Transfer Network (CANet). Previous methods, due to the absence of GT images, often focused on more precise color space mapping to improve color fidelity while overlooking the role of network architecture. In contrast, we are the first, to our knowledge, to introduce a spatial-channel attention mechanism into the task of color style transfer. Specifically, CANet processes the content and style images through separate downsampling extractor to extract texture and color features, which are then fused via a spatial-channel attention module for more accurate and consistent transfer. An image reconstruction module further reintegrates texture, reducing degradation and preserving structural integrity.
By combining these two innovations, our method significantly outperforms state-of-the-art approaches, as extensive experiments demonstrate clear advantages in both quantitative evaluation and visual quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses color style transfer, which involves transferring the color of a reference style image to an input image. The proposed method consists of two key elements. First, it introduces a novel training strategy: applying two effects to a single image and extracting two patches from it to generate a pseudo set of reference, input, and ground truth images. Second, CANet is proposed, which combines spatial attention and channel attention. Experiments demonstrate that the proposed method achieves the optimal trade-off between style similarity and content similarity compared to existing methods.

### Strengths
1. The authors propose a novel training method. This framework is capable of generating ground truth without the need for additional data, which makes it a highly practical approach. Unlike SCTNet (Fang et al., 2024) and Neural Preset (Ke et al., 2023), which could only perform global color transformations due to the lack of training data, the proposed framework achieves color transfer that considers local semantics.

2. They also introduce an attention mechanism into color style transfer. While Modflows (Larchenko et al., 2025) and D-Lut (Li et al., 2025) adopt convolution-based architectures, this is the first work to incorporate an attention mechanism.

3. The proposed method achieves the best trade-off between style similarity and content similarity compared with existing approaches. While Modflows achieves high style similarity, it tends to lose content consistency. In contrast, DeepPreset maintains content well but achieves lower style similarity. Compared to these methods, the proposed approach offers the most balanced performance. Given the wide range of potential applications of color style transfer, the impact of this work can be considered significant.

### Weaknesses
1. Augmented-Self Reference [1, 2] has been proposed in previous studies on colorization as a method similar to the framework proposed by the authors for generating ground truth. To clearly establish the novelty of the proposed method, the authors should cite Augmented-Self Reference, explain the differences between their method and Augmented-Self Reference, and conduct experimental comparisons.

[1] Lee, Junsoo, et al. "Reference-based sketch image colorization using augmented-self reference and dense semantic correspondence." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[2] Zhao, Hengyuan, et al. "Color2Embed: Fast exemplar-based image colorization using color embeddings." arXiv preprint arXiv:2106.08017 (2021).

2. Style transfer using Transformers has already been proposed in StyTr2 [3]. To clearly highlight the novelty of the proposed method, the authors should cite StyTr2, explain the differences between CANet and StyTr2, and provide experimental comparisons of the network architectures.

[3] Deng, Yingying, et al. "Stytr2: Image style transfer with transformers." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

3. The authors utilize PI, NIQE, and BRISQUE to evaluate perceptual quality. However, these metrics are inherently designed for evaluating grayscale images and are therefore unsuitable for this task. More recent metrics capable of evaluating color images, such as Q-Align [4], LIQE [5], and CLIPIQA [6], should be considered.

[4] Wu, Haoning, et al. "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels." International Conference on Machine Learning. PMLR, 2024.

[5] Zhang, Weixia, et al. "Blind image quality assessment via vision-language correspondence: A multitask learning perspective." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

[6] Wang, Jianyi, Kelvin CK Chan, and Chen Change Loy. "Exploring clip for assessing the look and feel of images." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 2. 2023.

4. In Table 1, the authors should report the performance of the Neural Preset.

5. In Table 1, I have concerns regarding the Times column. To my understanding, SCTNet and SCTNet-GT should share the same model, differing only in training methods. Yet, SCTNet-GT is significantly faster. Furthermore, while the original paper of D-LUT states that D-LUT runs at 0.011 s on a single GTX 1080 Ti GPU, Table 1 reports 29.88 s on an NVIDIA RTX 4090 GPU, which seems unreasonably slow.

6. Since color style transfer is a subjective task, the authors should also conduct a user study.

7. The explanation of the proposed method should be improved. Specific improvement points are listed below.

- In equation (2), $X^i$ should be $X^1$.

- In equation (3), $i=1,2,...,n$ should be $i=1,2,...,n-1$.

- Line 196 states ${\bf X}^i \in {\mathbb R}^{c\times H\times W}$, but since content and style are concatenated, it should be ${\bf X}^i \in {\mathbb R}^{2c\times H\times W}$.

- In Figure 3, the images and symbols are inconsistent. Similarly, equation (12) is also confusing.

### Questions
Please add explanations regarding the weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes a color transfer method, named CANet, which achieves good image color transfer capabilities through the implementation of a spatial-channel attention mechanism. It also introduces a new strategy for constructing color transfer ground truth. However, the paper lacks overall novelty, and the GT generation strategy exhibits obvious shortcomings.

### Strengths
1. In general, the writing of the paper is good. It clearly outlines the composition of CANet and the strategy for color transfer GT design.

2. The Related Works section is highly detailed. It identifies the shortcomings of existing works.

### Weaknesses
1. CANet does not introduce any novel network design. it merely combines CNNs with scaled dot-product attention. This approach is similar to directly using a ViT backbone. The attention mechanism and ViT architecture have been widely employed in color transfer tasks [1, 2] for years. 

2. I^c and I^s are randomly cropped from the same image in different color distribution, leading to overlapping or semantically similar content and style patches. This contradicts real-world color transfer scenarios where content and style are typically uncorrelated. 

3. The GT generation strategy constructs training images using fixed LUTs. This construction strategy may reduce the network's generalizability. In addition, the results in Fig. 8 show only minor color differences compared to the example in Fig. 3. 

4. Tab. 1 shows that CANet does not achieve an obvious performance gain over the compared methods like DeepPreset, which was published back in 2021. 

5. The paper does not discuss the limitations of the proposed method or provide any failure cases. 


[1] Tumanyan, Narek, et al. "Splicing vit features for semantic appearance transfer." In CVPR 2022. 
[2] Deng, Yingying, et al. "Stytr2: Image style transfer with transformers." In CVPR 2022.

### Questions
Although color transfer and style transfer are generally considered as two separate tasks, their boundaries are often blurred. I would recommend the authors to compare CANet with SOTA style transfer methods.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the task of image color style transfer, addressing two major challenges: the scarcity of high-quality GT data for supervised training and color distortions caused by suboptimal feature fusion in existing models. The authors propose an annotation-free GT generation strategy and introduce CANet, the first model to incorporate spatial–channel attention for this task. CANet employs a dual-branch structure for feature extraction and fusion, along with a reconstruction module for refined output generation. CANet demonstrates superior performance compared to five SOTA methods. Ablation studies further validate the effectiveness of the attention mechanism and GT generation strategy, though the analysis remains somewhat incomplete.

### Strengths
1. The proposed annotation-free GT generation strategy effectively exploits single-image color consistency to automatically produce structurally aligned GT pairs through recoloring, thereby alleviating the scarcity of high-quality ground truth data for supervised color style transfer. This approach eliminates the need for labor-intensive manual annotation and avoids the limitations of indirect supervision commonly adopted in prior works.

2. CANet represents a novel contribution as the first model to integrate spatial–channel attention into color style transfer. By moving beyond conventional convolution-based backbones, its dual-branch architecture and SCAB-guided feature fusion enable context-aware and semantically consistent color transformation, effectively mitigating the color distortion issues observed in existing models.

3. The experimental design is comprehensive, leveraging diverse augmented datasets and benchmarking against five state-of-the-art methods. Both quantitative and qualitative analyses consistently demonstrate the model’s superiority, while its high inference efficiency highlights strong potential for real-world applications.

### Weaknesses
1. The GT generation strategy is limited by its reliance on recolored variants of a single source image and a fixed set of 15 predefined 3D LUTs. This design restricts the diversity of color styles and weakens the model’s adaptability to complex or cross-semantic content–style combinations, as no validation has been conducted on cross-domain or semantically diverse scenarios.

2. The key hyperparameters of CANet are neither empirically optimized nor theoretically justified. Without experiments exploring alternative configurations, it is difficult to assess how parameter choices influence the trade-off between efficiency and accuracy, leaving uncertainty about the model’s optimal design.

3. The self-constructed test set lacks alignment with standardized benchmarks, which compromises the reproducibility and comparability of results against existing SOTA methods. Moreover, the dataset’s focus on natural and architectural scenes—while omitting domains such as portraits or extreme visual conditions—limits the evaluation of the model’s robustness and generalization in diverse real-world contexts.

4. The paper employs a simple L1 pixel-wise loss, which effectively minimizes low-level reconstruction errors but does not explicitly capture high-level perceptual quality metrics such as color harmony, semantic consistency in textured regions, or human subjective aesthetic preference. As a result, the perceptual realism of generated results may be constrained.

### Questions
1. The proposed GT generation strategy, which relies on recolored variants of a single source image and a fixed set of 15 predefined 3D LUTs, may limit the model’s ability to handle complex or semantically diverse content–style pairs. Could the authors elaborate on the potential limitations of this approach in such scenarios? In particular, how does the model perform on images with low semantic similarity between content and style? Additionally, has the strategy been tested on more diverse datasets to verify its generalization capability?

2. The key hyperparameters in CANet are not clearly justified, either empirically or theoretically. Could the authors provide further explanation regarding the rationale behind their selection, explore alternative configurations, and clarify how these choices affect the model’s efficiency and accuracy?

3. The self-constructed test set focuses mainly on natural and architectural scenes and lacks evaluation under extreme or challenging conditions. Could the authors confirm whether CANet has been tested on portraits or extreme cases to demonstrate its robustness? Furthermore, how do the authors plan to address the reduced reproducibility and comparability arising from the use of a non-standard dataset?

4. The paper employs a pixel-wise L1 loss, which primarily enforces low-level reconstruction consistency. How does this loss function perform compared to other loss formulations in terms of maintaining content fidelity and style consistency? Have the authors considered comparing their approach with perceptual, style, or adversarial loss functions to assess potential improvements in perceptual quality?

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
This paper proposes a simple yet stable data acquisition pipeline for supervised conditioned color style editing: it tunes one image into two different tones, and crops two different patches from it and form a pair. It also proposes a new model that processes the content image and style image separately, and then fused via a spatial-channel attention module. The proposed model is trained on the proposed data in a supervised manner, and achieves leading performance compared to previous methods.

### Strengths
- The proposed data acquisition pipeline is simple and stable, leading to efficient color editing models.

### Weaknesses
- The proposed data acquisition pipeline is to crop one GT image into two different patches with different color styles. While this leverage the same basis for accurate color tone alignment, their content remains similar and might harm the transferability after training.

- The proposed method is only tested on nature and architecture data, while its quality on broader scenarios like indoor, human, objects etc., and cross-domain scenarios remain unclear.

- The comparing methods are relatively not up-to-date. More comprehensive models (like some MLLM) are necessary to demonstrate the superiority of the proposed task-specific approach.

### Questions
- How the style similarity is calcuated? Although it is adapted from a prior work, it still worths noting its details in the main paper or appendix for quick reference.

### Soundness
2

### Presentation
3

### Contribution
2
