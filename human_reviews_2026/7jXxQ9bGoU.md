# GaussianFusion: Unified 3D Gaussian Representation for Multi-Modal Fusion Perception

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
The bird’s-eye view (BEV) representation enables multi-sensor features to be fused within a unified space, serving as the primary approach for achieving comprehensive multi-task perception. However, the discrete grid representation of BEV leads to significant detail loss and limits feature alignment and cross-modal information interaction in multimodal fusion perception. In this work, we break from the conventional BEV paradigm and propose a new universal framework for multi-task multi-modal fusion based on 3D Gaussian representation. This approach naturally unifies multi-modal features within a shared and continuous 3D Gaussian space, effectively preserving edge and fine texture details. To achieve this, we design a novel forward-projection-based multi-modal Gaussian initialization module and a shared cross-modal Gaussian encoder that iteratively updates Gaussian properties based on an attention mechanism. GaussianFusion is inherently a task-agnostic model, with its unified Gaussian representation naturally supporting various 3D perception tasks. Extensive experiments demonstrate the generality and robustness of GaussianFusion. On the nuScenes dataset, it outperforms the 3D object detection baseline BEVFusion by 2.6 NDS. Its variant surpasses GaussFormer on 3D semantic occupancy with 1.55 mIoU improvement while using only 30% of the Gaussians and achieving a 450% speedup.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
GaussianFusion presents a unified 3D Gaussian-based framework for multimodal fusion, replacing discrete BEV with a continuous shared space. It aligns camera and LiDAR features via a shared encoder with deformable attention and iterative updates. A Gaussian-to-voxel module enables task-agnostic perception. On nuScenes, it achieves state-of-the-art results in detection and occupancy tasks with lower latency and memory.

### Strengths
1. Unified continuous 3D Gaussian representation eliminates early discretization and preserves fine geometric/semantic detail.  

2. Forward-projection initialization produces accurate, optimization-friendly Gaussians without random guesswork.  

3. Shared Gaussian encoder with deformable attention enables natural cross-modal alignment and iterative refinement.  

4. Gaussian-to-voxel pooling delivers task-agnostic features for both detection and dense occupancy, beating BEV baselines while using less memory and latency.

### Weaknesses
1. It lags behind sparse-detection specialists that optimize only for 3D object recall.  

2. No temporal modeling is incorporated, so multi-frame cues remain unused.  

3. Gaussian-to-voxel conversion still imposes a fixed voxel grid, re-introducing minor discretization error.  

4. The shared encoder assumes synchronized full-modal inputs; robustness to sensor dropout is not evaluated.

### Questions
1. Have experiments been conducted on datasets such as nuPlan, NavSim, Waymo, or CARLA?

2. Where is the real-time performance analysis that the autonomous-driving community cares about?

3. What are the advantages of Gaussian-to-Voxel Pooling over BEV Pooling and Voxel R-CNN's Voxel ROI Pooling, and why weren't corresponding ablation experiments conducted?

4. In Figure 2, the schematic diagrams in both the overall part (a) and part (b) do not clearly illustrate the correspondence with the inputs, leading to potential ambiguity and necessitating revision.

5. What exactly is the "&" operation in Figure 2? It is not explicitly defined in the paper and needs to be clarified by the authors.

6. The authors need to clarify whether the two tasks are trained separately or handled simultaneously by a single model; Figure 2 suggests joint inference, but the presentation is ambiguous.

7. This work proposes to encode features by combining BEV and Gaussian representations, yet the detection head is not described in detail. Moreover, the paper does not explore whether alternative heads could be plugged in to verify the generality of the proposed encoding.

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
3

### Summary
This paper proposes *GaussianFusion*, a multi-modal fusion framework for 3D perception in autonomous driving. The core idea is to replace discrete Bird’s-Eye View (BEV) representations with a unified 3D Gaussian representation to preserve fine-grained geometric and semantic details. Key contributions include:  

- A **forward-projection-based Gaussian initialization** strategy for camera and LiDAR data, avoiding random initialization used in prior work (e.g., GaussianFormer).  
- A **shared Gaussian encoder** with deformable attention and iterative property updates to align cross-modal features.  
- A **Gaussian-to-voxel fusion module** for task-agnostic 3D perception.  
- State-of-the-art results on nuScenes for 3D object detection (74.0 NDS) and semantic occupancy prediction (28.65 mIoU), with improved efficiency over GaussianFormer.

### Strengths
**Originality**:  

- The use of 3D Gaussian representations for multi-modal fusion is novel. While 3D Gaussian Splatting (3DGS) has been explored for reconstruction (Kerbl et al., 2023) and single-modal perception (H et al., 2024), its application to multi-modal fusion is a creative extension.  
- The forward-projection initialization addresses a critical limitation of random initialization in GaussianFormer, improving convergence and performance (Table 6).  

**Quality**:  

- Experiments are thorough, covering both object detection and occupancy prediction. The ablation studies (Tables 6–7) validate design choices, and comparisons with BEVFusion (Liu et al., 2023b) and GaussianFormer (H et al., 2024) are compelling.  
- The shared encoder and Gaussian mixture model (Eq. 4–5) elegantly unify cross-modal features.  

**Clarity**:  

- The paper is well-structured, with clear figures (e.g., Fig. 2) illustrating the pipeline. Mathematical formulations (e.g., covariance matrix derivation in §3.2) are precise.  

**Significance**:  

- The framework advances multi-modal perception by addressing BEV’s quantization limitations. The 450% speedup over GaussianFormer (Table 5) demonstrates practical value.

### Weaknesses
**1. Limited Novelty in Fusion Mechanism**:  

- While the Gaussian representation is novel, the fusion mechanism (shared encoder + deformable attention) closely follows prior work (e.g., BEVFusion’s CNN fusion, UniTR’s cross-modal attention). The paper does not sufficiently differentiate its fusion strategy from existing methods. For example, the deformable attention module (Fig. 2b) resembles Zhu et al. (2020) without novel adaptation for Gaussian properties.  

**2. Incomplete Comparison with Sparse Detectors**:  

- The authors acknowledge that sparse detectors (e.g., Zhang et al., 2024a; Li et al., 2024) outperform their method but dismiss them as “task-specific.” However, recent sparse fusion works (e.g., Wang et al., 2024b; Yin et al., 2024) achieve both high detection accuracy and occupancy prediction capability. A direct comparison with these methods is missing and weakens the claim of “task-agnostic” superiority.  

**3. Shallow Technical Details**:  

- Critical components lack implementation specifics:  
  - The forward-projection initialization for LiDAR (§3.2) is described as “BEV-based” but lacks mathematical formulation (unlike the camera case).  
  - The Gaussian mixture aggregation (Eq. 5) is presented as a summation without addressing computational complexity or optimization strategies for large-scale scenes.  

**4. Under-Explored Limitations**:  

- The computational overhead of Gaussian-to-voxel conversion (§3.4) is glossed over. For example, the “neighborhood radius” calculation for Gaussian-voxel pairing is mentioned but not analyzed (e.g., runtime vs. BEV resolution).  
- The method’s reliance on dense LiDAR data (nuScenes) raises questions about its applicability to low-cost, camera-only systems.  

**5. Weak Visualization**:  

- Qualitative results (Fig. 3) are limited to two scenarios and lack comparisons with key baselines (e.g., OccFusion). The claimed “sharper boundaries” are not quantitatively validated.

### Questions
**Questions**:  

1. How does the shared encoder resolve modality-specific conflicts (e.g., camera depth uncertainty vs. LiDAR precision)? Is there a mechanism to weight modalities dynamically?  
2. The Gaussian updating module (§3.3) predicts offsets (Δµ, Δs, Δr). How stable is this optimization? Does it suffer from gradient vanishing/explosion with multiple encoder layers?  
3. Could the framework support temporal fusion (e.g., multi-frame LiDAR/camera data)? The authors mention it as future work but provide no preliminary analysis.  

**Suggestions**:  

1. **Strengthen the fusion novelty**: Compare the deformable attention mechanism with prior Gaussian-guided attention (e.g., Zuo et al., 2024) and highlight differences.  
2. **Expand experiments**:  
   - Compare with recent sparse fusion methods (e.g., Wang et al., 2024b; Yin et al., 2024).  
   - Report computational metrics (FLOPS, memory) for Gaussian-to-voxel conversion.  
3. **Clarify technical details**:  
   - Provide equations for LiDAR Gaussian initialization.  
   - Discuss how Gaussian scale (**s**) and rotation (**r**) are regularized during training.  
4. **Enhance visualization**: Include failure cases (e.g., occluded objects) and per-class IoU curves for occupancy prediction.  

---

**Rebuttal Potential**: The authors could improve the rating by:  

- Demonstrating superior task-agnostic performance against sparse fusion baselines.  
- Providing deeper analysis of computational trade-offs.  
- Adding a ablation on the Gaussian mixture aggregation strategy.  

Overall, the paper presents a promising direction but requires stronger differentiation from existing fusion paradigms and more rigorous evaluation to meet the acceptance threshold.

### Soundness
2

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
3

### Summary
The authors presented GaussianFusion that replaces discrete BEV grids with a continuous 3D Gaussian space to fuse camera and LiDAR, aiming to preserve fine details and align modalities before any quantization. It introduces forward-projection camera Gaussian initialization, a shared Gaussian encoder with Gaussian-guided deformable attention and iterative property updates, then pools Gaussians to voxels for task heads; on nuScenes it reports +2.6 NDS over BEVFusion for detection and higher mIoU for occupancy with lower latency/memory

### Strengths
Continuous fusion before BEV: avoids early discretization loss and enables richer cross-modal interaction in Gaussian space.

Forward-projection init for cameras; shared encoder with Gaussian-guided deformable attention and incremental updates to µ/s/r.

Solid results + efficiency: NDS 74.0 / mAP 71.7 on val, latency 132 ms and memory 4271 MB vs. BEVFusion’s 156 ms/5140 MB; occupancy mIoU 28.65 SOTA among single-frame models.

### Weaknesses
Scope limited to nuScenes; no cross-dataset/domain validation shown, so generality beyond this setting is inferred rather than demonstrated.

Although fusion happens in continuous Gaussian space, the method must voxelize Gaussians for task heads, which can bring back quantization artifacts the paper set out to avoid.

At the same 200×200 BEV size, GaussianFusion shows higher memory than BEVFusion (5418 MB vs. 5140 MB), even though latency is lower.

Temporal modeling out of scope. Authors acknowledge that temporal fusion boosts SOTA in both detection and occupancy, but GaussianFusion leaves this to future work, so comparisons to temporal SOTA aren’t shown.

### Questions
Some questions are mentioned in the weakness section.

### Soundness
3

### Presentation
3

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
The paper introduces GaussianFusion, a new framework for multimodal 3D perception that replaces the traditional bird’s-eye view (BEV) grid representation with a 3D Gaussian representation. This approach unifies multimodal features (e.g., from different sensors) in a shared, continuous 3D Gaussian space, effectively preserving fine details and edges. The framework includes a forward-projection Gaussian initialization module and a cross-modal Gaussian encoder that iteratively refines Gaussian properties using attention mechanisms. GaussianFusion is task-agnostic and demonstrates strong effectiveness.

### Strengths
1. Clear and timely motivation. I initially thought BEV was the best in terms of representational capability and fusion suitability, but this work convincingly shows that 3D Gaussian representation is indeed a promising alternative.
2. The paper is well-structured and flows logically.
3. The proposed solution is intuitively convincing and demonstrates effectiveness in two visual representation use cases.

### Weaknesses
1. Missing ablation study on the impact of voxel size on task accuracy.
2. For readability, the citation format should be revised: use ~\citep or ~\citet instead of ~\cite.
3. Lacks discussion of failure cases where the Gaussian representation performs worse than the BEV representation.

### Questions
1. In Table 2, why is the modality listed as “C” when LiDAR (with VoxelNet backbone) is used?
2. Can this method be applied to anchor-aware object detection models (e.g., FUTR3D [1], TransFusion [2])?
3. Regarding Table 3, as far as I know, BEVFusion addresses long-latency issues for improved efficiency. Given that BEV-based methods quantize features early, I would expect them to have lower latency. However, GaussianFusion shows smaller latency than BEVFusion — where does this latency saving come from?
4. Could this approach generalize to other modality combinations (e.g., Camera + Radar, LiDAR + Radar, Camera + LiDAR + Radar)?
5. How effective is this approach on other datasets (e.g., Waymo) that have different sensor characteristics, such as varying LiDAR beam densities?

**Reference**

[1] FUTR3D: A Unified Sensor Fusion Framework for 3D Detection

[2] TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers

### Soundness
3

### Presentation
3

### Contribution
3
