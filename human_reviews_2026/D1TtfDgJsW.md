# Generalizable 3D Edge Detection For Soft & Hard Features

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Understanding 3D objects based on their geometric and physical properties--independent of predefined labels--is essential for creating, modifying, and using the objects in diverse contexts. However, most machine learning approaches in the 3D domain rely heavily on semantic or primitive labeled-data to achieve these tasks. We present a 3D edge detection algorithm that decomposes point clouds into precise geometric components without relying on primitives or semantic labels. This enables us to tackle datasets of freeform, entirely unrestricted objects (as in the Thang3D dataset) that are challenging, and in many cases impossible,  for current models in the literature to segment, reconstruct, or produce parametrically. Additionally, we achieve state-of-the-art (SOTA) edge detection accuracy on both the complex Fusion360 Segmentation, Thang3D, and simpler standard ABC benchmarks. Our approach maintains reliable edge detection on soft features where most existing models fail. In addition, when the detected edges are used as input for segmentation, our method outperforms recent segmentation models on intricate geometries. This framework provides a robust and generalizable foundation for edge-aware analysis, segmentation, and generation of diverse 3D shapes well beyond what can be easily labeled by humans.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a 3D point cloud edge detection algorithm that can effectively detect the geometric components of complex 3D objects without the need for raw annotations or semantic labels. The method achieves state-of-the-art (SOTA) performance on multiple datasets, which demonstrates its effectiveness.

### Strengths
1.This paper improves edge detection performance by introducing abundant geometric features. Even though the proposed architecture is simple, it can still achieve optimal results on multiple datasets.

2. The formulas in the paper are complete, and the visualizations are clear.

### Weaknesses
1.There is a lack of reports on running time and computational overhead.

2.The ablation experiments lack the part related to the skip branch, making its role unknown.

3.The paper has poor readability and a rather confusing logic.

4.Using more explicit geometric information as input has been adopted by many previous methods and proven effective; therefore, the innovation of this method is relatively weak.

### Questions
1.This paper mentions dividing each point cloud into non-overlapping subsets of 10k points. Has this been proven to be a favorable choice through experiments? Will different point quantities affect the model performance?

### Soundness
3

### Presentation
3

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
The paper addresses the problem of 3D edge detection in point cloud representations of shapes, emphasizing how identifying regions of high or moderate curvature, such as ridges, valleys, or protrusions, can enhance the accuracy of downstream geometric or primitive segmentation.

The authors propose an edge detection framework that leverages low-level geometric cues, including point coordinates, normals, mean curvature, and curvature gradients at multiple scales. These features are processed by a point-based neural network that predicts, for each point, the probability of belonging to an edge/boundary region. To efficiently handle large point sets (hundreds of thousands of points), the model operates on 10K-point patches while maintaining a learned global context vector that is iteratively updated across patches to preserve global geometric consistency.

After per-point boundary prediction, the method applies a flood-filling clustering algorithm to segment the shape into distinct geometric components based on boundary connectivity. The model is trained using BREP-based CAD datasets, with dense point sampling around high-curvature regions to better capture fine geometric details.

The proposed approach is evaluated on edge detection and geometric segmentation tasks, where it outperforms existing methods.

### Strengths
The paper addresses a challenging problem in point cloud analysis, performing primitive-free geometric segmentation without relying on predefined semantic/primitive labels. By framing the task as boundary region detection, the approach eliminates the need for labeled part data beyond edge annotations, thereby improving generalizability across datasets and shape types. The use of low-level geometric features such as curvature and curvature gradients enables the network to accurately capture regions of high convexity or concavity, including soft transitions like fillets and chamfers. Furthermore, the introduction of a global context vector that is recurrently updated across point patches helps maintain global geometric consistency, allowing the model to process large point clouds while preserving overall shape coherence.

### Weaknesses
While the paper presents a practically effective method, its architectural and conceptual novelty is limited. The patch-based point cloud processing strategy closely resembles EC-Net, which also processes shapes in local patches, while the edge detection formulation is conceptually similar to PB-DGCNN [1]. 

From an evaluation perspective, several aspects raise concerns about fairness and completeness. The method is compared primarily against NerVE and PCED, yet these baselines are designed for much smaller point sets (around 20K points), whereas the proposed model is trained and tested on significantly denser clouds (approx. 200K points). This discrepancy potentially biases the results, since higher sampling density naturally enables better capture of fine geometric details. Moreover, although the paper states (lines 337–338) that experiments were conducted on the full F360+ and Thang3D datasets, it later clarifies (lines 356–358) that only about half of F360+ and one-fourth of Thang3D were actually processed. Additionally, qualitative visualizations comparing failure cases of competing methods are missing, which would have provided clearer insight into where and how the proposed approach improves over baselines.

For the segmentation benchmark, the evaluation includes only ParSeNet, while other relevant and publicly available alternatives such as HPNet [2], PrimitiveNet [4], and SED-Net [3] are omitted. The same issue regarding point cloud resolution mismatch arises here, as ParSeNet was trained on lower-resolution inputs (10K points). Furthermore, in the proposed flood-filling clustering stage, it remains ambiguous whether the adjacency matrix is treated as binary or weighted, and how this choice affects the resulting segmentation quality. Finally, additional quantitative and qualitative comparisons would strengthen the experimental section and better substantiate the claimed superiority of the proposed method.

Finally, the paper claims that a CNN model is applied, but essentially an MLP PointNet-based encoder is utilized.


[1] Loizou M. et al., "Learning part boundaries from 3D point clouds." Computer Graphics Forum. Vol. 39. No. 5. 2020.

[2] Yan S. et al., "Hpnet: Deep primitive segmentation using hybrid representations." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[3] Li Y. et al., "Surface and edge detection for primitive fitting of point clouds." ACM SIGGRAPH 2023 conference proceedings. 2023.

[4] Huang J. et al, "Primitivenet: Primitive instance segmentation with local primitive embedding under adversarial metric." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

### Questions
- How do the different neighborhood scales used for computing curvature and curvature gradients affect the model’s performance in edge detection??
- What are the values of the hyperparameters $k$, $\tau_{\text{reject}}$, $r_{\text{bdry}}$, $\theta_0$, and $\lambda$ in the boundary-aware segmentation step, and how do variations in these parameters influence the final segmentation results?

### Soundness
2

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
This paper proposes a primitive/semantic label-independent 3D edge detection method. The main idea is to incorporate various geometric features (such as mean curvature and curvature gradient) as inputs to a PointNet-like framework to classify edge points. The method is evaluated on three datasets, demonstrating superior performance compared to selected baseline methods.

### Strengths
- The  proposed method achieved sota in three datasets.
- The proposed method works better on soft edges.

### Weaknesses
- The significance and challenges of the task is not clearly convinced.
- The baseline methods are outdated and the evaluation is insufficient. For example, the work lacks comparisons with more recent approaches [1]. Moreover, there are no comparisons with recent point cloud segmentation methods.

* [1]Liu et.al., ToG’25, HoLa: B-Rep Generation using a Holistic Latent Representation,

### Questions
- There is no citation for the Thang3D dataset. Is it newly collected in this work? If so, please provide more information about the dataset (e.g., number of models, annotations of ground truth).
- The authors claim that the proposed method can “provide a robust and generalizable … beyond humans.” Are there any quantitative comparisons or qualitative examples to support this claim?
- Was the proposed method also only trained on the F360+ dataset?
- What is the loss function of the proposed method? How are the segmentation results derived in Figure 6?
- What does “precise geometric components” mean? Is there any visualization?

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
This paper presents a primitive-less 3D edge detection method for point clouds that handles both soft features (fillets, chamfers, bevels) and hard features (sharp edges). The approach processes large point clouds (>200k points) by chunking them into 10k-point batches and maintaining a global context vector across chunks. The method uses a two-branch CNN architecture that processes geometric features including point coordinates, normals, curvature (H), and curvature gradients (∇H). A deterministic flood-fill clustering algorithm then segments the point cloud based on detected edges. The authors evaluate on ABC, Fusion360+, and Thang3D datasets, demonstrating state-of-the-art edge detection accuracy and improved segmentation performance compared to state-of-art approaches. The key contribution is achieving robust edge detection on diverse geometries without relying on semantic labels or geometric primitives.

### Strengths
**Originality**: The paper addresses an important gap in 3D edge detection—handling soft features that most existing methods fail on. The combination of curvature-based features with a chunked processing approach using global context is a reasonable design choice for handling arbitrarily large point clouds.

**Evaluation**: The experimental evaluation is comprehensive, including three diverse datasets (ABC, Fusion360+, Thang3D), ablation studies examining each component's contribution, and comparisons against recent baselines (PCEDNet, NerVE). The generalization capability is impressive, as demonstrated by training only on F360+ and testing on ABC and Thang3D without fine-tuning.

**Clarity**: The paper is clearly written, with most algorithmic details clearly outlined.

**Significance**: The work has practical value for CAD, manufacturing, and 3D reconstruction applications where capturing fine geometric details is necessary.

### Weaknesses
**Limited architectural novelty**: the architecture is relatively simple making the paper read more as an engineering contribution. The reliance on hand-crafted geometric features (curvature, gradients) reduces the learning burden but also limits the model's ability to discover novel feature representations.

**Dependency on feature quality**: the ablation study shows dependence on high-quality normals (Table 4: approximated normals drops F1 from 0.87 to 0.73). For real-world scanning scenarios with noise and occlusions, normal estimation quality may be unreliable. The noise experiment shows only modest degradation at 0.002 noise level, but more extensive noise analysis across different noise types and levels would strengthen the claims. Alternatively, experiments showing results on real or simulated scanned objects could be helpful for this.

**Incomplete description and motivation for the chunking approach**: the method heavily relies on chunking the point cloud for processing, but it does not provide the motivation for why such approach is reasonable / optimal, not describe algorithmic details of the approach (how are chunks determined? is the chunk size dependent on sampling resolution? is there overlap between chunks? ablation for the chunk size? etc.).

**Incomplete experiment description and analysis**:
1. Some important details are missing: dataset splits, training hyperparameters beyond epochs, convergence criteria, heuristic algorithm hyperparameter selection.
2. Why is it important to have the "skip branch" in the proposed architecture? Ablation study is necessary.
3. The segmentation comparison is limited to only ParSeNet—more recent methods like SpelsNet are mentioned but not compared.
4. Computational cost analysis is missing—how does processing time scale with point cloud size? What are training / inference times?

### Questions
Some of my questions and suggestions are outlined above, in the Weaknesses section.

Additional questions are below.
1. Feature engineering vs. learning: how much of the performance gain comes from the curvature features versus the neural architecture? Have you tried end-to-end learning without pre-computed curvature features using a larger network? Similarly, the approximated normals ablation shows significant degradation. Have you experimented with learning-based normal estimation methods?
2. Computational efficiency: what are the runtime comparisons against PCEDNet and NerVE?
3. Failure cases: are there specific geometric configurations where your method fails?
4. Clustering hyperparameters: how were the flood-fill algorithm parameters selected? Are they dataset-specific or can one set of hyperparameters work across all datasets?

### Soundness
3

### Presentation
3

### Contribution
3
