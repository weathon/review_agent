# Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present **PG-Occ**, an innovative  **P**rogressive  **G**aussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that **PG-Occ** achieves *state-of-the-art* performance with a relative  **14.3\% mIoU improvement** over the previous best performing method. Code and pretrained models are available at: https://yanchi-3dv.github.io/PG-Occ.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents PG-Occ, a Progressive Gaussian Transformer for open-vocabulary 3D occupancy prediction. The method progressively densifies 3D Gaussian representations in a feed-forward manner to capture fine scene details while maintaining efficiency. An additional anisotropy-aware sampling strategy adaptively adjusts receptive fields across scales and time for better spatio-temporal feature aggregation. Integrating language-aligned features enables text-driven 3D reasoning. Experiments show that PG-Occ achieves improvement over previous state-of-the-art methods, delivering more detailed and scalable scene understanding.

### Strengths
1. The paper introduces a novel Progressive Gaussian Transformer that progressively refines 3D Gaussian representations through a feed-forward densification process, balancing detail capture and computational efficiency.
2. The proposed anisotropy-aware sampling adaptively adjusts receptive fields across directions and scales, enabling more accurate feature aggregation and fine-grained geometric modeling.

### Weaknesses
1. Although progressive densification effectively enhances scene details, the increasing number of Gaussians at higher stages inevitably raises inference time and memory consumption. While the paper acknowledges this issue and plans to optimize it in future work, no quantitative memory analysis or profiling results are provided, leaving the practical cost-performance trade-off unclear.
2. Experiments are conducted on two benchmark datasets, which is relatively limited. The evidence for cross-domain robustness and generalization remains insufficient.
3. One of the main contributions, the anisotropy-aware sampling, provides quantitative gains in the ablation study, but the improvement is relatively small.
4. The comparisons mainly cover works published up to 2024. Recent approaches from the past year in similar or related directions are missing from the comparison.

### Questions
1. Could the authors provide a detailed analysis or profiling of memory consumption across different densification stages?
2. Have the authors tested the method on additional datasets or unseen domains to further validate its generalization capability?
3. While AFS shows limited quantitative improvement in the ablation, could the authors provide qualitative examples or visualizations that better illustrate its specific benefits or effects?
4. Could the authors include comparisons with more recent methods (2024-2025)?

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
3

### Summary
The paper introduces PG-Occ, a progressive Gaussian transformer for open-vocabulary 3D occupancy. It starts from a coarse set of 3D Gaussians and, in a feed-forward way, adds new Gaussians only where depth reveals under-modeled regions, so the scene becomes denser exactly where it needs more detail. It further makes each Gaussian anisotropy-aware and keeps the model stable with asymmetric self-attention so newly added Gaussians do not disturb existing ones.

### Strengths
- i) The paper targets a clear gap in current Gaussian-based open-vocabulary occupancy. The proposed progressive, depth-guided densification is a straightforward and well-motivated way to increase capacity only where the scene is under-modeled.

- ii) The anisotropy-aware feature sampling makes good use of the Gaussian’s scale and orientation, which is sensible for driving scenes where many structures are not isotropic. 

- iii) The paper is well organized and clearly written.

### Weaknesses
- i) The whole progressive pipeline relies on the quality of the pseudo depth.  A short discussion on robustness to worse depth (or to LiDAR-sparse depth) would make the claim stronger.  

- ii) Because the Gaussian set can only grow, not shrink, the last layers still have to process the largest token set. This is fine at Occ3D-nuScenes resolution, but may need pruning for HD-maps or city-scale scenes.

### Questions
- i) The current densification is driven by a depth discrepancy threshold. Have you tried combining this with a text-feature uncertainty signal, so that regions that are geometrically covered but semantically ambiguous can also trigger new Gaussians?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes PG-Occ, a progressive Gaussian Transformer framework for open-vocabulary 3D occupancy prediction. The method couples (i) Progressive densification of Gaussians through an iterative feed-forward densification strategy, (ii) Anisotropy-aware feature sampling that selects sample points and projects them onto feature planes with varying receptive fields. Extensive experimental results demonstrate that PG-Occ achieves SOTA performance on Occ3D-nuScenes.

### Strengths
The strengths of this paper lie in its clear motivation and cohesive design: it addresses the trade-off in Gaussian representations via progressive densification, stabilizes the training dynamics through asymmetric attention, and improves feature alignment via anisotropy that matches each Gaussian. Empirically, main results, ablations, and efficiency comparisons reinforce one another, showing that under fixed or limited compute, the approach delivers a better balance of performance and speed than previous baselines.

### Weaknesses
However, I also have some concerns:
(1) The robustness of corner cases is underexplored: In autonomous driving, identifying corner cases is critical. Common classes like cars, trucks, and pedestrians are already well recognized by supervised learning, whereas less common categories, such as plastic bags or trash bins, are much harder to detect. Can PG-Occ recognize such corner cases (not limited to these two examples)?
(2) From the paper, the motivation seems somewhat trivial, more like a data augmentation extension of GaussTR, so it should explicitly articulate the deeper thought behind this motivation.
(3) The method shows strong performance in a camera-based pipeline. Does the method work in a multimodal setting?

### Questions
1. Could you show more corner-case examples, such as uncommon categories?
2. Could you provide a deeper intuition or theoretical analysis of the motivation to justify the necessity of the method?
2. Is it also effective in a multimodal setting with LiDAR and cameras? If experiments are not feasible due to dataset constraints, an analysis of the underlying rationale would suffice.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PG-Occ, a framework for open-vocabulary 3D occupancy prediction in autonomous driving. The central challenge it addresses is the trade-off between sparse Gaussian representations, which are efficient but miss fine-grained details, and dense representations, which incur high computational costs. It resolves this with two key contributions: 
1. Progressive Online Densification (POD): A feed-forward strategy that iteratively refines the 3D Gaussian scene representation. It starts with a coarse model and progressively adds detail to regions with higher perception errors, efficiently capturing fine-grained objects without modeling the entire scene densely.
2. Anisotropy-aware Sampling (AFS): A sampling method that adaptively assigns receptive fields to Gaussians based on their specific scale and rotation (anisotropy). This allows for more effective spatio-temporal feature aggregation.

The model is trained using only 2D supervision (pseudo-depth maps and text-aligned features) without requiring 3D LiDAR data. Experiments show PG-Occ achieves state-of-the-art performance, demonstrating a significant 14.3% relative mIoU improvement over the previous best method on the Occ3D-nuScenes dataset.

### Strengths
1. PG-Occ introduces an efficient, online method that adaptively adds Gaussians to "under-represented regions" identified by depth errors. This allows the model to start coarse and progressively capture fine-grained details.
2. The paper originally identifies a weakness in prior methods that treat Gaussians as simple points, ignoring their shape. The AFS module is a novel solution that samples features based on the Gaussian's specific scale and rotation. This allows for adaptive receptive fields and more effective spatio-temporal feature aggregation.
3. ASA ensures training stability by allowing new, under-optimized Gaussians to learn from established ones, but not vice versa.
4. The method achieves SOTA results on the challenging Occ3D-nuScenes dataset, with a 15.15 mIoU score.

### Weaknesses
1. The method is trained using only 2D supervision from sparse-view cameras. This setup creates inherent ambiguity. A small, nearby object can project to a 2D feature patch similar to a large, distant object. While multi-view consistency and pseudo-depth supervision  help, they don't fully resolve this.
2. The model is initialized using pseudo-depth maps from Metric3D V2 and also supervised using them. The authors should present experimental results using other depth prediction methods to demonstrate the robustness of the proposed method.
3. The number of total queries should be provided in Table 4.

### Questions
1. Table 4 reports the final method's speed as 2.40 FPS. However, Table 12 reports the inference time of the final progressive layer as only 60.6 ms, and the sum of all three layers as 146.3 ms (27.4 + 58.3 + 60.6). This sum suggests a throughput of ~6.8 FPS. What component is responsible for the major bottleneck that reduces the final speed to 2.40 FPS? Is it the ResNet-50 spatio-temporal backbone, the final Gaussian-to-voxel post-processing, or another part not listed in Table 12?
2. You acknowledge a key limitation: "constraining the Gaussian scale in depth is challenging, which can cause popping artifacts". Could you elaborate on the severity and frequency of these artifacts? For example, do they primarily affect distant/occluded objects, or is it a general instability? How much do you believe this instability impacts the method's reliability for downstream tasks like motion planning, which require temporally stable geometric representations?
3. The model is initialized and supervised by pseudo-depth. Have you investigated the model's robustness under different depth models?
4. The ablation study shows that removing ASA ("w/o ASA") degrades performance. Did you also experiment with using standard, symmetric self-attention instead of just removing it?

### Soundness
3

### Presentation
3

### Contribution
3
