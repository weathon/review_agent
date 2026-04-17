# GAP3D: Geometry-Aware Adaptive Planar Representations for 3D Occupancy Prediction

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We address the problem of 3D occupancy prediction from multi-view images, a task central to autonomous driving and embodied perception. Conventional methods typically employ fixed 3D voxel grids, which provide structured coverage of the entire scene but scale poorly in memory and computation as resolution increases. Sparse 3D representations, by contrast, improve efficiency  by focusing only on occupied regions, but risk missing critical areas and often converge slowly. To overcome these limitations, we propose $\textbf{Geometry-Aware Adaptive Planar Representations (GAP3D)}$, a new adaptive 3D representation that combines the comprehensive coverage of voxel grids with the efficiency of sparse methods. GAP3D has two key components: (1) $\textbf{Adaptive plane representations}$, where planes sequentially partition vertical space via a stick-breaking process, concentrating representational capacity in regions with high occupancy likelihood; and (2) $\textbf{Geometry-aware splatting}$, which lifts the plane features into the full 3D occupancy volume through a differentiable Gaussian kernel, producing dense, spatially consistent predictions while preserving scene geometry. To further guide plane placement during early training, we introduce a height regularization loss that encourages alignment with scene structure. Experiments on the Occ3D-nuScenes benchmark demonstrate that GAP3D achieves state-of-the-art performance while significantly reducing memory and computation compared to existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a geometry-aware adaptive planar representation (GAP3D) to combine the advantages of grid-based representations and sparse representations. GAP3D first adaptively places planes along the vertical dimension with a stick-breaking formulation, and then converts the planes to 3D occupancy with the geometry-aware splatting technique. Experiments show that GAP3D achieves state-of-the-art performance on Occ3D-nuScenes.

### Strengths
1. The motivation to combine the global coverage of grid-based representations and the efficiency of sparse representations is reasonable.
2. The performance of GAP3D on Occ3D-nuScenes is good compared with other methods.
3. The paper is well-written and easy to follow.

### Weaknesses
1. Lack of related work. The paper does not cite closely related work such as TPVFormer, GaussianFormer, etc.
2. Lack of novelty. The proposed GAP3D can be regarded as constraining the positions of Gaussian primitives in GaussianFormer to grid centers in the xy plane, and the paper fails to explain why such a constraint is beneficial to the 3D representation. Therefore, GAP3D is more like GaussianFormer rearranged in a grid manner rather than a novel planar representation.
3. Insufficient experiment. The paper only conducts experiments on Occ3D-nuScenes, which is quite limited.

### Questions
1. Since each primitive still adopts a Gaussian kernel formulation, I think they are actually Gaussian primitives as in GaussianFormer. Then the primitives of the same plane actually have little relationship with each other, except that the paper arranges them in a HxWxD format. What is the insight of this rearrangement?
2. How does the proposed method perform on other widely used benchmark, such as SurroundOcc?

### Soundness
2

### Presentation
3

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
I offer a different perspective about the paper: This paper introduces a novel method where each layer predicts a relative update to the height of its query. The subsequent layer then utilizes this incremental adjustment to refine the query's location. Based on this updated position, features are extracted through an adaptive mixing mechanism. Crucially, by representing the data in a relative structured way, the model avoids the need for unordered signals like a CD loss, which significantly facilitate and stabilize the convergence process.

### Strengths
The paper is well organized and the logic is easy to follow.

The experiments achieve the SOTA performance.

### Weaknesses
1.	Since I consider the paper as a variant of the OPUS architecture, I have some questions: where does the overall speed advantage come from compared to the original OPUS model? How are the features for the initial plane in the first layer obtained?

2.	The RayIoU metric, while valuable, has a notable limitation: it only evaluates accuracy on non-free rays and fails to account for false positives predicted on free rays. Therefore, I believe RayIoU alone is insufficient for comprehensive model evaluation. In contrast, the mIoU metric could potentially capture this aspect of performance. For a more balanced and community-accessible benchmark, I recommend supplementing the results with the mIoU metric to provide a fuller picture of model capabilities.

3.	Given that modeling long-range dependencies is a key and unique aspect of the method, a specific ablation on this component is advisable. A more thorough description of this mechanism in the methodology would also improve clarity for readers.

4.	The experimental evaluation would benefit from additional visual comparisons with other methods to better demonstrate the qualitative advantages of the proposed approach.

5.	It is noted that SparseOcc, which also employs a dense representation, was trained for only 24 epochs, whereas the proposed method was trained for 60 epochs. Consequently, it remains unclear whether the observed performance gains stem from the novel representation or the substantially longer training schedule. A controlled ablation study to disentangle these two factors would significantly strengthen the paper's claims.

6.	The description of feature splatting needs to be explicitly stated: is it applied to all voxels in the 3D space or constrained to a single plane? 

7.	The training strategy employs several unconventional design choices that require further justification. For instance, the rationale behind annealing the regularization loss weight to zero over the first six epochs, as well as the decision to change the batch size to 32 in the second training stage, should be explicitly explained.

### Questions
See weakness.

My final score is directly contingent upon the quality of the rebuttal. Should all of the raised concerns be adequately addressed, I will elevate the rating to an 8. Conversely, if the response fails to fully resolve any issues, the score will likely remain at its current level. However, a minor improvement to a 6 is also be considered if some of the concerns are satisfactorily clarified.

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
5

### Summary
This paper introduces GAP3D, a framework for 3D occupancy prediction from multi-view images. The method aims to solve the trade-off between traditional dense voxel grids, which offer full coverage but scale poorly in memory and computation, and sparse representations, which are efficient but risk missing critical regions and lack global consistency. GAP3D combines the strengths of both approaches with two key contributions: 
1. Adaptive Plane Representations: The model represents the 3D scene using a set of adaptive 2D planes. It employs a stick-breaking process to sequentially partition the vertical space, allowing the height of each plane to dynamically adapt to the scene's geometry. This focuses representational capacity on geometrically significant regions.
2. Geometry-Aware Splatting: These adaptive plane features are fused into a dense 3D occupancy volume using a differentiable Gaussian kernel. This operation preserves the learned geometric structure and produces spatially consistent predictions.

To aid training, the authors also introduce a height regularization loss ($\mathcal{L}_{height}$) to encourage the adaptive planes to align with the scene structure, especially in early training stages.
Experiments on the Occ3D-nuScenes benchmark demonstrate that GAP3D achieves state-of-the-art performance, attaining 42.8% RayIoU. It outperforms previous methods like STCOcc and OPUS while maintaining competitive efficiency.

### Strengths
1. The idea of using planes is not new (e.g., TPV ), but the method for their placement is highly original. The use of a stick-breaking process to sequentially and adaptively partition the vertical space is a clever mechanism. This allows the representation to dynamically allocate "representational capacity" to geometrically complex areas, unlike fixed grids.
2. The paper introduces a differentiable method to fuse these adaptively-placed 2D planes back into a 3D volume6666. Using a Gaussian kernel that is aware of the adaptively predicted plane heights ($h_{u,v}^{(i)}$) ensures that the geometric structure learned by the planes is coherently preserved in the final voxel output.
3. The results are state-of-the-art. GAP3D achieves the highest RayIoU (42.8%) on the benchmark, outperforming prior work like STCOcc. This SOTA performance is particularly impressive because it is achieved using fewer input frames (8 vs. 16) and a lower input resolution than the next-best method.

### Weaknesses
1. The proposed method using planes (TPVFormer) and Gaussian splatting (GaussianFormer) is not new.
2. The authors could run experiments on other benchmarks, such as nuScenes-OpenOccupancy and SemanticKITTI, to demonstrate the generalization ability of the proposed method.
3. It's unclear what the model learns if the scene lacks strong geometric features in a specific (u,v) column (e.g., open sky). The stick-breaking process still has to allocate $K$ planes. 
4. The former sparse methods, like SparseOcc (both ECCV 2025 and CVPR 2024 versions), GaussianFormer (v1&v2), which use free x-y/u-v coordinate sparse voxels/3D Gaussians. This paper tries to fix the x-y coordinates to form planes. This may leed to the resource waste on empty areas.
5. The ablation study on the number of queries should be provided.
6. Table 3 explores the number of planes, but it does so with a reduced channel dimension (64 instead of 80) due to computational cost. This is a confounding variable. The performance drop at 6 planes (38.4 RayIoU) compared to 4 planes (38.7 RayIoU)  might not be because 6 planes are worse, but because 6 planes with fewer channels is worse. The conclusion that "simply adding more planes is not always advantageous"  is not fully supported by this experiment.
7. The speed of the proposed method seems to be much slower than the previous methods.

### Questions
1. Could you provide more intuition on the learned behavior of the adaptive planes in geometrically empty regions, such as the sky?
2. Could you clarify if the 4-plane result in this table also used 64 channels? If not, could you provide a "cleaner" ablation where the channel dimension is held constant to isolate the true impact of $K$?
3. The "geometry-aware splatting" via a Gaussian kernel is a key component. This method shares conceptual similarities with other well-established techniques (e.g., GaussianFormer). Could you elaborate on the specific novelty of your splatting mechanism?

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
3

### Summary
This paper proposes a novel scene representation approach for the task of 3D semantic occupancy prediction. The core idea is to elevate the traditional BEV (Bird's Eye View) grid into spatial planes, and aggregate these plane features into voxel features using geometry-aware splatting within 3D space for subsequent occupancy prediction. Experiments conducted on the Occ3D-nuScenes benchmark demonstrate the effectiveness of the proposed method.

### Strengths
Originality：
GAP3D introduces a novel approach by lifting BEV features into 3D planes, which are then aggregated into voxel features via splatting. This representation paradigm enhances the spatial expressiveness of BEV features to some extent.

Quality：
The method is well-motivated and mathematically well-defined.

Clarity：
The presentation is clear and well-structured. Figures progressively illustrate the proposed representation and pipeline, notations are consistent, and the appendix provides helpful implementation details and information on numerical stability.

Significance：

This work proposes an improved representation paradigm for occupancy prediction in autonomous driving, and demonstrates noticeable performance gains. It can serve as a useful reference for future research in this area.

### Weaknesses
Novelty：
Although the paper claims to introduce a new plane-based representation, it essentially amounts to enhancing BEV features along the height dimension. Similar ideas have already been explored in the Pillar-based literature. Thus, the proposed representation does not appear sufficiently novel.

Experiments：
The paper only reports results on the Occ3D-nuScenes dataset. However, other occupancy prediction datasets such as Occ3D-Waymo and SurroundOcc, have been widely adopted. The current experimental evaluation is therefore insufficient.

### Questions
1. Limited Dataset Comparisons: As far as I know, there are several autonomous driving datasets for occupancy prediction beyond nuScenes, such as Waymo and KITTI. Additionally, multiple benchmarks exist besides Occ3D. The paper only evaluates on Occ3D-nuScenes and should include additional experiments for a more comprehensive comparison.

2. Evaluation Metrics: The paper only uses RayIoU for evaluation, but mIoU is a more widely used metric for occupancy prediction. This important metric is missing and should be included.

3. Plane Representation Concerns: The proposed plane representation merely adds one more dimension to the BEV grid and lifts it into 3D space. This can hardly be regarded as a truly sparse representation. In contrast, existing works like GaussianFormer have implemented  primitive-based occupancy prediction, which this paper fails to mention.

### Soundness
3

### Presentation
3

### Contribution
2
