# SPREAD-GS: Scale-Progressive Representation Extraction and Detailing for 3D Gaussian Splatting

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
3D Gaussian Splatting (3DGS) has recently shown remarkable capability for high-fidelity scene reconstruction. However, its potential for object recognition remains under-explored. Existing approaches often extract 2D features from multi-view images and embed them into 3DGS, which limits the joint use of 3DGS geometric and appearance information.
To fully exploit both the structural and fine-grained details encoded in 3DGS, we propose Scale-Progressive Representation Extraction And Detailing for 3D Gaussian Splatting (SPREAD-GS), a framework for object classification that combines scale-aware sampling with detail-preserving feature propagation. 
SPREAD-GS has two key modules: Scale-Progressive Sampling (SPS), generating multi-scale subsets by progressively narrowing the visible region, and SpreadNet, encoding these subsets and propagating details across scales through noise-augmented feature upsampling.  On the texture-rich MACGS dataset, SPREAD-GS achieves 93.93\% overall accuracy, improving the SOTA by  2.02\%. On the geometry-centric ModelNet40GS, it matches the SOTA while significantly reducing parameters.
These results demonstrate the effectiveness of scale-progressive sampling and detail-preserving feature propagation for 3DGS recognition.
These results demonstrate the effectiveness of scale-progressive sampling and detail-preserving feature propagation for 3DGS recognition. Our code is available at https://anonymous.4open.science/r/noname-64BE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose a framework for object classification, named SPREAD-GS. Its key module SPS generate multi-scale subsets and by progressively narrowing the camera FoV, and then SpreadNet encodes these subsets and propagating details across scales through noise-augmented feature upsampling. The experiments show that SPREAD-GS achieves the best results on the MACGS and ModelNet40GS datasets.

### Strengths
1. The authors propose leveraging Gaussian attributes as additional information to enhance the object classification capability, which I consider effective and innovative.
2. The experiments demonstrate that SPREAD-GS achieves superior performance on the MACGS and ModelNet40 datasets.

### Weaknesses
1. Incorporating more Gaussian attributes increases the input storage size compared to other methods that use point cloud representations. It would be valuable if the authors could provide additional experiments to validate the model’s performance under conditions where the input storage size is comparable across different methods. This would help clarify whether the performance improvement of SPREAD-GS stems from the richer Gaussian attributes or merely the larger input size.
2. As the authors claim that Gaussian attributes enhance the object classification capability. To better support this claim, it is recommended that the authors present comparative experiments to demonstrate the impact of combined Gaussian attributes—such as scale, rotation, opacity, and color—versus using only position information. 
3. The design of SPS focuses on single object. However, multi-object interactive scenes are more common in practical Gaussian Splatting applications(e.g., autonomous driving). It would be better if the authors could provide more examples or experiments on multi-object scenes to evaluate SPREAD-GS’s performance in handling object interactions.

### Questions
The definition of the camera pose during SPS remains unclear, which causes confusion. As an object’s appearance varies with the observation direction, the distance and angle between the camera and the object can significantly affect the sampling results. Additionally, the robustness of SPS to different camera poses also raises confusion.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces SPREAD-GS, a new framework that aims to enhance 3D object classification using 3D Gaussian Splatting (3DGS) representations. The authors argue that existing approaches either rely on 2D features embedded into 3DGS, which fail to exploit native 3D information, or treat Gaussians as point cloud, which discards fine-grained per-primitive details. To address this, SPREAD-GS proposes two key components: (1) Scale-Progressive Sampling (SPS): A multi-scale sampling mechanism that progressively narrows the FoV to create hierarchical subsets of Gaussians. It captures both global geometric structure and localized appearance details. (2) SpreadNet: A hierarchical network that processes these subsets and propagates fine-scale features across scales via a Detail Propagation (DP) module. The network leverages EdgeConv layers for local structure encoding and fuses features progressively to build robust object-level representations. Experiments on MACGS and ModelNet40GS benchmarks show that SPREAD-GS outperforms state-of-the-art point cloud and Gaussian-based baselines.

### Strengths
1. Good writing and organization.

2. Effective idea with clear motivation.

3. The proposed scale-progressive sampling and SpreadNet are novel and with solid engineering.

### Weaknesses
1. Lack of efficiency report. While SPREAD-GS is presented as efficient, the paper does not report detailed runtime, memory usage, or FLOPs. Given the multiple hierarchical scales and EdgeConv-based architecture, quantitative efficiency comparisons with baselines (e.g., DGCNN, PointMLP) would strengthen the claim of practical implementation and scalability.

2. According to Figure 3, although SPREAD-GS adopts random offsets to alleviate the central bias, during the Scale-Progressive Sampling process, each time it extracts Gaussians from a smaller region, it still focuses only on a localized area instead of sampling a large number of local features across multiple smaller regions. This may lead to a significant bias in local feature extraction.

3. The component ablation of SpreadNet is not sufficiently fine-grained. Although the paper performs ablations on hierarchical feature aggregation and propagation, it does not conduct step-by-step ablations on its five major modules, including EdgeConv, Encoder, Detail Propagation, Center Sample, and Feature FPS, to verify the necessity and contribution of each component.

### Questions
See the weaknesses.

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
4

### Summary
This paper introduces SPREAD-GS, a method for classifying 3D Gaussians. It leverages SPS to sample 3DGS at multiple scales, feeding each scale into the proposed SpreadNet to extract corresponding features. The final global feature representation is then used for classification.

### Strengths
1.Demonstrates strong classification performance on datasets such as MACGS and ModelNet40GS, validating the method’s effectiveness.

2.Proposes SpreadNet with novel encoder and block designs, showing clear architectural innovation.

3.Integrates multi-scale 3DGS features, enhancing attention to local details and improving their contribution to classification accuracy.

### Weaknesses
1.The method is primarily focused on 3D Gaussian classification. Although Section 4 suggests potential application to semantic segmentation, no experiments are provided, leaving its effectiveness on segmentation tasks uncertain. This may limit its broader applicability.

2.The ablation study lacks evaluation of the impact of varying the number of detail scales and the use of Feature FPS on classification performance.

3.While the paper references "Mitigating Ambiguities in 3D Classification with Gaussian Splatting (CVPR 2025)" and uses its datasets, no direct comparisons with that method are reported.

### Questions
The visual results are quite limited, only one lego case is provided. Should provide more visual results of used benchmarks.

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
The paper proposes SPREAD-GS, a novel framework for 3D object classification using native 3D Gaussian Splatting (3DGS) representations, introducing Scale-Progressive Sampling (SPS) to generate multi-scale Gaussian subsets and SpreadNet, a hierarchical network with Detail Propagation (DP) to fuse local details and global structure across scales. 
Evaluated on MACGS (texture-rich) and ModelNet40GS (geometry-only), SPREAD-GS achieves 93.92% (+2.01% over SOTA) and 91.67%, respectively, with comprehensive ablations validating the efficacy of SPS, DP, and encoder design.

### Strengths
The paper presents a well-motivated and timely approach to 3D object classification using native 3D Gaussian Splatting representations, a largely underexplored direction. 
Its originality lies in the thoughtful combination of multi-scale sampling (SPS) and cross-scale detail propagation (DP), which creatively adapts hierarchical reasoning to the unique structure of 3DGS beyond treating it as a plain point cloud. The technical quality is high, with comprehensive experiments on two distinct benchmarks (MACGS and ModelNet40GS) and thorough ablations validating each component. The clarity of presentation is strong, with clear figures and logical flow, and the significance is notable—by demonstrating that 3DGS contains rich semantic cues in its native attributes, the work opens new pathways for 3D understanding tasks beyond reconstruction.

### Weaknesses
1 Questionable task motivation: 3D Gaussian Splatting (3DGS) is primarily a reconstruction output rather than a standard input modality, and it is rarely the starting point for classification in real-world pipelines, making the practical relevance of this task unclear.

2 Unfair comparison: The comparison with Gaussian-MAE is conducted without using its intended pretraining setup—the paper explicitly trains it from scratch, which undermines the fairness and validity of the comparison.

### Questions
Fairness of comparison with Gaussian-MAE: Gaussian-MAE is designed as a large-scale self-supervised pretraining framework for 3DGS, where downstream tasks like classification and segmentation are used to evaluate the quality of its pretrained features—not as standalone task-specific models. Training Gaussian-MAE from scratch (without pretraining) for classification, as done in this work, does not reflect its intended use and undermines the comparison. 
Since the proposed SPREAD-GS is trained from scratch and tailored specifically for classification, comparing it against a general-purpose pretrained backbone under a non-standard (scratch) setting is neither fair nor meaningful.

### Soundness
2

### Presentation
2

### Contribution
1
