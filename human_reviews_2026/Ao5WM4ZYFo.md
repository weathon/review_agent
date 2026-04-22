# EG3AD: An Efficient Geometry-Aware Encoding Framework for Reconstruction-Based Multi-Class Point Cloud Anomaly Detection

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Multi-class point cloud anomaly detection is a critical task that aims to identify anomalous patterns across various categories using a single, unified model.
Current reconstruction based methods predominantly rely on transformer encoders to extract high-level semantic features, aiming to filter out subtle defect features, and then use decoders to reconstruct them into normal patterns. However, this suffers from two limitations:  
(1) employing encoders based on global attention mechanisms, particularly on uniformly tokenized inputs, hinders the rapid extraction of fine-grained local features;
(2) high computational cost arising from stacking multiple encoding layers during semantic feature extraction.
Thus, we propose EG3AD, an Efficient Geometry-aware encoding framework for reconstruction-based multi-class 3D point cloud Anomaly Detection. 
To investigate how to obtain effective geometric representations under token and parameter constraints, we begin by introducing the Curvature-Aware Sampling module, which mitigates the distortions caused by uniform sampling in regions of high curvature.
Then, leveraging geometry prior bias of point cloud data, we design the Point Cluster Graph Convolution, which enables compact and effect local geometric aggregation through only limited lightweight layers.
Finally, to obtain anomaly-invariant semantic features without relying on deep encoding layers, we introduce the Feature Purification Module inspired by optimal transport theory. This module compresses features into a set of cluster centroids that preserve fundamental geometric patterns, thereby yielding representations robust to subtle anomalies.
Extensive experiments show that simply replacing the vanilla point transformer encoder with our proposed EG3AD yields state-of-the-art results on all PCAD benchmarks.
Our code will be made publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on multi-class point cloud anomaly detection. The motivation is to address the large computational cost of the Transformer architecture and to keep fine-grained details in the point cloud sampling step. To this end, it proposes a strategy to sample low, mid, and high curvature points at different ratios and proposes a KNN-enhanced network to reduce the computational cost. The proposed method achieves leading performance on two major 3D anomaly detection benchmarks, including Real3D-AD and Anomaly-ShapeNet.

### Strengths
1) The motivation of this paper is clear. It is reasonable to address the computational cost for 3D anomaly detection, as the number of 3D points can be very large for high-resolution point clouds of a large object.

2) The proposed method demonstrates SOTA performance on leading benchmarks for 3D anomaly detection.

### Weaknesses
1) While the paper aims for 'efficient' 3D anomaly detection, its claims regarding memory and computational cost are not fully substantiated due to a lack of comparison with previous methods. The analysis of computational cost is incomplete; although a figure shows that the KNN-based PCGC module has an O(N) cost, this analysis omits the decoder and anomaly calculation modules, which add overhead during inference. To validate the claim of efficiency, it is crucial to provide a holistic comparison of the entire model's computational, memory, and time costs against other state-of-the-art methods.

2) The proposed curvature-based sampling strategy is a heuristic method reliant on pre-defined hyperparameters. According to the ablation study, performance is highly sensitive to these hyperparameters, which must be tuned for different datasets. This raises concerns about the module's effectiveness and generalization capability. The dependence on 'cherry-picked' hyperparameters may lead to overfitting on benchmarks and limit the method's applicability to unseen, real-world scenarios.

3) The paper contains several typos, errors, and unclear statements that need to be addressed:

- Figure 2a: Appears to be incomplete, with data points clipped from view.

- Line 93: The term 'cross-class metric' is undefined. If this means training on one class and inferring on others, this experimental setup is not presented in the paper. Please clarify.

- Lines 157-158: The definitions of m and m' seem inconsistent with their usage in Figure 3. Please check and correct these notations.

- Line 157: The citation for PointNet appears to be incorrect.

- Equation 14: The notation e is used without being defined.

- Line 320: The hyperparameter γ (gamma) is not defined, and its sensitivity on performance is not discussed.

- Tables: The formatting convention of bolding both the best and second-best results simultaneously is confusing. A clearer distinction is recommended.

4) According to the implementation details, the model is pre-trained on ShapeNet55. Since the Anomaly-ShapeNet test set contains objects from the ShapeNet dataset, this raises a concern about potential data leakage. Please clarify whether any overlap exists and how this issue was handled.

5) Regarding the ablation study on the drop ratio settings for CAS, the baseline performance is missing. What is the model's performance when no points are dropped (i.e., a drop ratio of 0)? This baseline is necessary to evaluate whether the sampling strategy provides a definitive benefit.

### Questions
Please see the questions in the weaknesses section.

### Soundness
3

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
The paper identifies the high complexity of existing methods and proposes a multi-class point cloud anomaly detection method. The method consists of three modules: CAS, PCGC, and FPM. Experiments demonstrate the effectiveness of the proposed modules.

### Strengths
The paper is well-organized, and each component of the method is clearly described and easy to understand. The superiority of the proposed method is validated on public datasets, including Real3D and Anomaly-ShapeNet. Ablation studies further confirm its effectiveness.

### Weaknesses
The related work section is not sufficiently comprehensive. For instance, “Boosting global-local feature matching via anomaly synthesis for multi-class point cloud anomaly detection” also addresses a similar multi-class anomaly detection setting and should be discussed for completeness.

The novelty of the proposed method remains questionable. The CAS module appears highly similar to curvature-based downsampling, and it is unclear what the essential differences are. Likewise, the distinction between PCGC and PointNet++ is not clearly justified.

Although the paper claims to reduce the high computational cost of Point Transformer, the proposed architecture still employs multiple Transformer decoders, and the PCGC module contains numerous fully connected layers. These components introduce considerable computational overhead, and it is doubtful whether the method is indeed more efficient in practice.

The visualization in Fig. 4 also seems problematic. The curvature maps show a smooth radial transition from the center outward, which is unlikely to be realistic. For example, the regions between the arms of a starfish should exhibit noticeable curvature changes, yet they are visualized as uniformly blue.

In addition, the relationship between curvature estimation and anomaly detection performance is not clearly analyzed or discussed.

It is also recommended to provide more qualitative comparisons with other methods to more convincingly demonstrate the superiority of the proposed approach.

Finally, since the method is evaluated under a multi-class setting, it is unclear how the results for other methods were obtained. It would be more convincing if the authors also reported single-class performance and compared against the latest state-of-the-art baselines, for which more reported results are available.

### Questions
Please refer to the issues raised in the Weaknesses section.

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
The paper tackles 3D point-cloud anomaly detection with three components: Curvature-Aware Sampling (CAS) to adaptively densify complex regions, Point Cluster Graph Convolution (PCGC) as a locality-aware, hierarchical encoder, and a Feature Purification Module (FPM) to denoise anomaly-corrupted features. On Real3D-AD and Anomaly-ShapeNet, EG3AD reports top object-level AUROC (e.g., ~81.1% on Real3D-AD; 85.6% on Anomaly-ShapeNet), with module ablations supporting each component.

### Strengths
1. Consistent performance gains are observed across both PCAD benchmarks. The per-module ablation studies provide useful insights.

2. The focus on efficiency—through locality using PCGC and sampling via CAS—is well justified, especially when dealing with large-scale point sets.

### Weaknesses
1. The discussion of deployment scenarios remains limited, with little coverage of sensor noise, sparsity, partial scans, or mixed categories. Comparisons to RGB-3D fusion-based anomaly detection methods are also lacking.

2. The experimental results appear to be based on a single random seed. There are no uncertainty estimates, statistical significance tests, or cross-dataset transfer analysis.

### Questions
1. Can results include variance over multiple runs and statistical significance tests for the reported AUROC scores?

2. How does the method perform under severe occlusions or partial scans? Is there any evaluation under domain shift, such as cross-dataset tests?

3. How does inference time compare to transformer-based baselines when operating on the same number of points?

### Soundness
2

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
3

### Summary
This paper presents a novel anomaly detection approach for 3D point cloud data, aiming to develop a more efficient and effective solution to this challenging task. The proposed method comprises three key modules: CAS, PCGC, and FPM. Specifically, the CAS module enriches the geometric information embedded in the input tokens; the PCGC module refines local geometric features to enhance anomaly detection performance; and the FPM module employs optimal transport to extract informative features for reconstruction. Experimental results on the Real 3D-AD dataset substantiate the effectiveness of the proposed approach.

### Strengths
1. The paper proposes an efficient and effective method specifically designed for 3D point cloud anomaly detection, addressing a critical gap in this research domain.

2. The approach introduces the concept of optimal transport in a novel way to overcome the inherent challenges of 3D point cloud anomaly detection, representing a meaningful technical innovation.

3. The proposed method achieves SOTA performance on several classes of the Real 3D-AD dataset, highlighting its strong competitive advantage and empirical effectiveness.

### Weaknesses
1. The paper would benefit from a clearer and more comprehensive explanation of the PCGC module, including its internal architecture and the specific mechanisms that contribute to performance improvement.

2. The authors employ ShapeNet55 for pre-training and evaluate the model on the Anomaly ShapeNet dataset. Since the latter is synthesized from the former, there exists a potential risk of data leakage or overlap between the pre-training and test sets.

3. The proposed model requires a pre-training stage on ShapeNet55, followed by fine-tuning on the target dataset. Consequently, it utilizes substantially more data for training compared to the baseline methods. To make the comparison convincing, the authors are encouraged to include results against CLIP-based methods, such as PointAD, which adapts CLIP for point cloud anomaly detection. It would be valuable to investigate whether the proposed approach can still achieve superior performance when PointAD is evaluated under the same multi-class setting.

4. It is recommended to include additional supplementary experiments on the MVTec3D dataset to further validate the generalization ability of the proposed approach.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
