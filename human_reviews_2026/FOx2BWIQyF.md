# DUPS: Dynamic upsampling for efficient semantic segmentation

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
We present \textbf{DUPS}, a coarse-to-fine vision transformer for semantic segmentation. Unlike models that begin with dense high-resolution tokens, DUPS starts at low resolution and dynamically upsamples only regions predicted to contain semantic boundaries, following a “one-token-one-class” principle. Mixed-resolution attention enables interaction between coarse and fine tokens, allocating computation to semantically complex areas while avoiding redundant processing in homogeneous regions.  Experiments on ADE20K, COCO-Stuff, and Cityscapes demonstrate that DUPS achieves state-of-the-art results on ADE20K and COCO-Stuff with substantially fewer FLOPs, and delivers competitive accuracy on Cityscapes at markedly lower compute. For example, DUPS-Base attains \textbf{54.6 mIoU} on ADE20K in the $\sim$110M-parameter class while using fewer FLOPs than comparable backbones.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DUPS, a coarse-to-fine vision transformer for semantic segmentation. DUPS employs an inverted U-Net design, which selectively allocates computation to semantically complex areas, thereby reducing computational cost. The method is evaluated on ADE20K, COCO-Stuff, and Cityscapes, demonstrating promising performance with reduced computational cost.

### Strengths
- The dynamic upsampling design is an interesting approach, utilizing a learned predictor to estimate semantic edge density and focus computational resources on boundaries and fine structures.   
- The proposed inverted U-Net architecture is a promising design choice for refining tokens while preserving multi-scale features.  
- Experiments on different datasets show good performance.

### Weaknesses
- The proposed upsampling strategy is not convincing enough. Specifically, the authors fail to provide detailed proof of the benefits of the inverted U-Net design in upsampling from low to high resolution. Moreover, the inverted U-Net architecture bears resemblance to conventional encoder-decoder networks, where inputs are fed into a ViT and progressively upsampled while combining multi-scale features. This raises questions about the originality of the proposed architecture.

- A significant concern is the discrepancy between training and inference in the dynamic upsampling mechanism. During training, a batch-wise upsampling ratio is used, whereas inference employs a sample-specific ratio. Although the use of a pre-computed ratio is a necessary engineering solution, the paper neglects to analyze the potential impact of this mismatch.

- The upsampling process appears irreversible, with decisions not to upsample a region at a coarse scale being final. This design choice introduces a risk of compounding errors, where the upsampling predictor's failure to detect thin structures or small objects at an early stage cannot be recovered later.

- The paper should provide latency comparisons for different methods.

- Is the proposed Dynamic Upsampling a generic method? The paper lacks experiments demonstrating the effectiveness of Dynamic Upsampling on different feature backbones.

- The paper is difficult to follow due to the lack of clear and concise visualizations. The authors rely heavily on textual descriptions to explain the missing information in the figures, making it challenging for readers to understand the proposed method. 

- Comparisons with some recent methods for semantic segmentation, such as SegMAN (CVPR'25) and FreqFusion (TPAMI'24), are missing. Note that I will not dismiss the contributions of this paper solely because its performance is not on par with SOTA methods. However, the authors should provide in-depth insights into how the proposed method can benefit the community compared to these SOTA methods.

### Questions
Please address the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work studies the efficiency–accuracy trade-off in semantic segmentation. Motivated by the uneven distribution of semantic content, the authors propose DUPS, an inverted U-Net architecture. In DUPS, semantically simple regions are represented by a single token, while semantically complex regions are represented by multiple tokens. Specifically, DUPS incorporates a shallow, coarse-to-fine encoder and a relatively deep decoder network. Quantitative and qualitative experiments demonstrate the effectiveness.

### Strengths
- The method is grounded in the well-recognized observation that content in natural images is unevenly distributed, motivating the straightforward solution. 
- The design of DUPS is concise and easy to grasp.

### Weaknesses
**Limited novelty**: 
- From Table 1, we see that the right branch is much deeper than the left. The right branch is essentially a conventional neural network, while the left branch is a shallow pathway that directly injects image data to preserve spatial details. This dual-path design has been extensively studied in the semantic segmentation community over the past decade.
- The overall design is also very similar to a class of "learning-to-downsample" methods that allocate fewer tokens to object interiors and more tokens near object edges. I cannot recall the exact citation, but there is a series of works following this approach.

**Poor presentation**: 
- DUPS can be viewed as the inverse of token-merging methods, which merge semantically similar tokens. Both approaches are based on the same assumption of uneven image-content distribution. However, the paper lacks any discussion of these token-merging methods, neither textual analysis nor quantitative comparison.
- DUPS selects patches for upsampling when their upsampling scores exceed a threshold. Therefore, introducing the notion of the *dynamic upsampling ratio* is redundant. Defining \(K\) as an indicator function (e.g. $\(K=\mathbf{1}(\{u>\tau\}))$) is sufficient and more concise.

### Questions
- Line 193, what is each feature $i$? It is suggested to use consistent terminology. Refer either to "each token" or to "each patch" throughout the manuscript.
- The effectiveness of the MSE loss is unclear because no ablation study is provided.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents DUPS, a backbone designed for semantic segmentation that use a coarse-to-fine vision transformer that begins with low-resolution tokens and dynamically upsamples only regions predicted to contain semantic boundaries. The dynamic upsampling block predicts per-patch edge-density scores to select tokens for 2×2 sub-patch expansion and fuses lightweight image features with learned scale/sub-patch embeddings. Mixed-resolution neighborhood attention enables interaction between coarse and fine tokens. The decoder adapts Mask2Former/AFF. The paper shows promising performance compared to baselines, but crucial experiments are missing and the comparison is not fair among backbone baselines.

### Strengths
- The selective upsampling and mixed-resolution neighborhood attention is efficient and allows explicit cross-scale interactions during attention. The inverted U-Net for coarse-to-fine resembles the human visual system.
- Results demonstrate promising performance across three standard benchmarks under comparable number of parameters and FLOPs.

### Weaknesses
Missing experiments:
- Since the main contribution of this paper is the encoder, it should be comared fairly with other encoders to isolate the performance gains introduced by the novel encoder. Currently, the baselines use different decoders (e.g., OverLoCK use UperNet) and I cannot see whether the performance gains of DUPS are due to the novel encoder or the Mask2Former decoder. The authors should report segmentation results of different encoders using the same decoder in Table 2 and Table 3.
- The upsampling approach proposed should be compared with previous dynamic upsampling methods, such as DySample. 
- Missing baselines and comparison: SegNeXt, SegMAN, EDAFormer (these models' decoders are a part of their contribution so comparing them directly with different decoders is fine). Also since DUPS uses edge map to train the dynamic upsampling block, it should be comapred with other methods that use edge maps such as SegFix.
- Latency/FPS of DUPS should be reported.
-  The proposed encoder performance on ImageNet-1k should be reported and compared to the encoders of other baselines.

Figures:
- The overall figures do not look very professional and should be polished (Figure 1, 2). Details of the upsampling block should be shown in the figure.

References

Learning to Upsample by Learning to Sample. ICCV 2023

### Questions
- I am curious on how many tokens are upsampled on each stage on average? Is it a sparse set of tokens?
- Can the method be extended to other dense prediction tasks?

Overall I like the idea of this work but I want to see the impact of the proposed encoder on mIoU when compared to other encoders with similar complexity and using the same decoder.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DUPS, a coarse-to-fine vision transformer for semantic segmentation, which addresses the inefficiency of uniform and high-to-low resolution architectures by starting with low-resolution tokens and dynamically upsampling only semantically complex regions. Its core innovations include a boundary-aware scoring module for targeted upsampling, a content-adaptive upsampling ratio policy compatible with minibatch training, and an inverted U-Net structure that enables multi-scale token interaction.
Empirically, DUPS demonstrates strong performance across three key benchmarks: it achieves state-of-the-art mIoU on ADE20K and COCO-Stuff with significantly fewer FLOPs, and delivers competitive accuracy on Cityscapes at a fraction of the compute of comparable baselines. Ablation studies further validate the necessity of dynamic upsampling, the inverted U-Net structure, and auxiliary image data, reinforcing the reliability of the proposed method.

### Strengths
It proposes a brand-new semantic segmentation architecture combining "dynamic upsampling and coarse-to-fine pipeline", which is different from existing "token pruning" and "fixed scale allocation" methods, showing significant innovation in technical routes.
The efficiency advantage is significant: the FLOPs are lower than those of baseline methods with the same accuracy. Moreover, it maintains competitiveness on real-scene datasets such as Cityscapes, verifying the effectiveness of the method in practical applications.

### Weaknesses
Although the paper fully verifies the advantages of the "low-to-high resolution processing pipeline" in computational efficiency (FLOPs), it fails to deeply explain why this mechanism can outperform the traditional "high-to-low resolution processing pipeline" in terms of accuracy (mIoU), resulting in an incomplete demonstration of the performance rationality of its core innovation.
The paper regards the "one-token-one-class" principle as the core logical support for the dynamic upsampling mechanism. However, it fails to provide a clear definition or systematic elaboration of this principle.

### Questions
The paper points out that "high-resolution initialization" leads to redundant computation in homogeneous regions. Then, is the dynamic upsampling mechanism starting from low resolution proposed by the authors only intended to reduce computational load? Does this mechanism contribute to improving model performance? Compared with models with high-resolution initialization, what advantages does DUPS have in feature extraction? Since DUPS can reduce redundant computation, can it also demonstrate significant accuracy advantages on test sets with high semantic complexity?

### Soundness
2

### Presentation
3

### Contribution
4
