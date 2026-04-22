# DispViT: Direct Stereo Disparity Regression with a Single-Stream Vision Transformer

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Deep stereo disparity estimation has long been dominated by a \textbf{matching-centric paradigm}, built on constructing cost volumes and iteratively refining local correspondences.
Despite its success, this paradigm exhibits an intrinsic vulnerability: visual ambiguities from occlusion or non-Lambertian surfaces invevitably induce errorneous matches that refinement cannot recover.
This paper introduces \textbf{DispViT}, a new architecture that establishes a \textbf{regression-centric paradigm}. 
Instead of explicit matching, DispViT directly regresses disparity from tokenized binocular representations using a single-stream Vision Transformer.
This is enabled by a set of lightweight yet critical designs, such as a probability-based disparity parameterization for stable training and an asymmetrically initialized stereo tokenizer for effective view distinction.
To better align the two views during stereo tokenization, we introduce a novel shift-embedding mechanism that encodes different disparity shifts into channel groups, preserving geometric cues even under large view displacements.
A lightweight refinement module then sharpens the regressed disparity map for fine-grained accuracy.
By prioritizing holistic regression over explicit matching, DispViT streamlines the stereo pipeline while improving robustness and efficiency.
Experiments on standard benchmarks show that our approach achieves state-of-the-art accuracy, with strong resilience to matching ambiguities and wide disparity ranges.
Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
1. A single-stream ViT framework that bypasses explicit matching to directly regress stereo disparity from tokenized binocular representations.
2. A disparity-aware rotary position embedding method is proposed to remaingeometry-consistentevenat large
 displacements. 
3. SoTA stereo matching accuracy is achieved on the KITTI benchmarks.

### Strengths
1. SoTA stereo matching accuracy on the public stereo matching benchmark.
2. Good real-time performance benefit from the single-steam framework. 
3. the proposed positiinal embedding method and shift-embedding tokenizer are novel.

### Weaknesses
1. Is the shift-embedding tokenizer designed to generate multiple tokens for the right view, or a token with a single channel that contains multiple groups with each corresponding to a predefined horizontal shift? The formula (2) is too simplified. Please provide the changes in feature dimensions during this process to help readers better understand it. 
2. The refinement module from NMRF-Stereo is not included in the ablation study. Thereby weakening the effectiveness of the originally proposed modules and methods.
3. The disparity colorbar in Figure 2 appears to be inconsistent across different methods, especially for the NMRF results in the third row. Please revise this figure to use a single, unified colorbar across all methods within each scene

### Questions
1. Please provide the changes in feature dimensions about the shift-embedding tokenizer.
2. Please provide ablation studies about the refinement module. 
3. Please explain the less comparative performance in the foreground regions on the KITTI 2015 benchmark.
4. Please revise this figure to use a single, unified colorbar across all methods within each scene in Fig. 2.

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
In this paper, the authors devleop a single-stream transformer (termed DispViT) for direct stereo disparity regression. Instead of explicit calculation of matching cost, DispViT directly regresses dispaity from tokenized binocular representations. To this end, a shift-embedding stereo tokenizer is developed and disparity-aware rotary position embeddings (DA-RoPE) is introduced. Experiments are conducted on KITTI 2012 and KITTI 2015 for performance evaluation. Results show the suprior performance as compared to the approaches included for comparison.

### Strengths
- The proposed method technically sounds.
- The motivation is clear and this paper is easy to follow.

### Weaknesses
- My major concern is about the technical novelty. Incorporating the powerfull learning capability of transformers for stereo matching (e.g., [c1]) has been studied for years. Compared with these methods, the proposed method is incremental. From this point of view, the contribution of this paper is rather limited.
[c1] Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers

- The experiments are insufficient. The cross-dataset generalization is a critical metric to assess the performance of a stereo matching model. Consequently, more datasets should be included for evaluation, including Middlebury and ETH3D. Currently, only results on KITTI 2012 and KITTI 2015 are reported, which is insufficient.

- Several important SOTA methods are missing ([c1-c2]). These recent methods should be included for comparison to better demonstrate the superiority of the proposed method. 

[c2] Monster: Marry monodepth to stereo unleashes power
[c3] Defom-stereo: Depth foundation model based stereo matching

- The computational efficiency is also critical to the practicality of a stereo matching model. Consequently, the computational cost (e.g., FLOPs, memory cost and runtime) should be included for evaluation.

- For shifted embedding, the maximum disparity range should be manually predefined, which requires prior information of the scene. I wonder whether the proposed method may suffer performance degradation under large disparities (e.g., >200 pixels on SceneFlow).

### Questions
See Weaknesses

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
5

### Summary
The authors propose DispVIT, a vision transformer for disparity regression. The key idea is to eliminate the left-right correlation step in stereo disparity estimation frameworks. The authors propose the use of shift-embeddings to encode multiple hypotheses by shifting the right image, an asymmetric initialization, a disparity-aware rotary positional embedding, and a probabilistic disparity parametrization. Experiments show that the proposed frameworks achieve competitive results with leading stereo networks.

### Strengths
The idea of eliminating the matching step is not new (DispNet), but this is the first attempt to execute a similar paradigm using ViTs. The design choices are clever and lightweight, and maintain the low-cost spirit of the proposed framework. The authors have conducted numerous experiments to evaluate their proposed method, and there are enough qualitative and quantitative results to draw conclusions. The paper is well written and presented. Strong quantitative results.

### Weaknesses
Major weaknesses 

1) The parameters for the right image shift are not really explored in the paper. It is unclear why a K of 8 is the best parameter for the shift. 
2) The proposed shift and summation of features to generate the right image tokens doesn't come with any scientific reasoning, intuition or physical justification. On the contrary, this operation removes any epipolar and geometric constraints.
3) Following on point 2, the paper could benefit by a direct comparison with a monocular metric depth framework. 
4) Based on the experiments, the proposed approach relies heavily on DAv2 with an attempt to utilize the probabilistic disparity parametrization to scale the depth. This implies that the network might learn depth priors instead of utilizing the right image to compute metric depth. In lines 362-364 the authors omit to present the results of zero-shot generalization from scene flow to real datasets; however, such results would strengthen the claims in the paper, showcasing that the network learns to utilize the right image cues.

Minor weaknesses 

1) What are the effects of stacking the refinement network more than 2 times? 
2) It is unclear from the text if DA-RoPE is just RoPE on both axes.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The manuscript proposed the DispViT, a new architecture that directly regresses disparity from tokenized binocular representations using a single-stream Vision Transformer. The architecture achieves state-of-the-art accuracy on standard benchmarks.

### Strengths
1. The manuscript is well written. And the related work is comprehensive and clear.
2. The comparison between the matching-centric paradigm and the regression-centric paradigm is an interesting topic.

### Weaknesses
1. Although the manuscript claims that the DispViT avoids constructing cost volumes, the shift-embedding tokenizer, which horizontally shifts the right view with a set of predefined offsets, is similar to the procedure of building cost volumes.
2. Another concern is whether the design is effective. As shown in Tab 1, the DispViT (without the refinement module from NMRF) achieves the worst accuracy than all methods except for RAFT-Stereo. Do the experiment results mean that the previous works would provide a more reliable regression prior than DispViT if they are also refined by NMRF?  Moreover, the Tab 2 shows that the DispViT relies heavily on pretrained weights from additional data. However, some matching-centeric methods are trained from scratch. Therefore the comparison with matching-centeric methods might be unfair.

### Questions
1. It would be better to provide the number of parameters in Tab 1 for a more comprehensive comparison.
2. In Eq 3, does alpha_{ij} means attention weights?
3. Would it be possible to present the quantitative experiment results on Middlebury and ETH3D?

### Soundness
2

### Presentation
3

### Contribution
2
