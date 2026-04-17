# GaussianFusionOcc: A Seamless Sensor Fusion Approach for 3D Occupancy Prediction Using 3D Gaussians

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
3D semantic occupancy prediction is one of the crucial tasks of autonomous driving. It enables precise and safe interpretation and navigation in complex environments. Reliable predictions rely on effective sensor fusion, as different modalities can contain complementary information. Unlike conventional methods that depend on dense grid representations, our approach, GaussianFusionOcc, uses semantic 3D Gaussians alongside an innovative sensor fusion mechanism. Seamless integration of data from camera, LiDAR, and radar sensors enables more precise and scalable occupancy prediction, while 3D Gaussian representation significantly improves memory efficiency and inference speed. GaussianFusionOcc employs modality-agnostic deformable attention to extract essential features from each sensor type, which are then used to refine Gaussian properties, resulting in a more accurate representation of the environment. Extensive testing with various sensor combinations demonstrates the versatility of our approach. By leveraging the robustness of multi-modal fusion and the efficiency of Gaussian representation, GaussianFusionOcc outperforms current state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
GaussianFusionOcc is a multi-modal 3D semantic occupancy prediction framework that performs sensor fusion directly on learnable 3D Gaussians instead of BEV or voxel grids. Its GaussianFusion Block uses modality-agnostic deformable attention to extract fuse and iterativly refine features from camera, LiDAR, and radar for each Gaussian. The refined Gaussians encode geometry and semantics efficiently. Their results show GaussianFusionOcc achieve state-of-the-art performance in the nuScenes occ benchmark.

### Strengths
1. The paper introduces the first framework that performs multi-modal fusion directly in the 3D Gaussian space, thavoiding the memory/computation inefficiencies of voxel or BEV-based approaches.
2. GaussianFusionOcc achieves state-of-the-art accuracy on nuScenes while significantly reducing memory usage, parameter count, and inference latency.
3. The paper is generally wirtten well, easy to understand and Includes detailed ablations, efficiency analysis, and visualizations that support the claimed advantages and provide a clear understanding of the model’s behavior.

### Weaknesses
See questions.

### Questions
The paper is generally well presented, but there are several concerns that the reviewer would like to raise:

1. The evaluation is conducted only on the nuScenes occupancy benchmark, whereas recent competing works typically report results on multiple benchmarks (e.g., Occ3D, SemanticKITTI). Including at least one additional benchmark would strengthen the generalization claims.

2. The paper reports end-to-end latency including the Gaussian-to-occupancy splatting stage, which is appropriate for semantic occupancy evaluation. However, a breakdown of the latency contributions from splatting versus sensor encoders would improve understanding of computational bottlenecks.

3. The reviewer’s main concern is that the motivation and supporting evidence for the full Gaussian parameterization are underspecified. The paper predicts and refines Gaussian opacity as a learnable parameter, suggesting that explicit volumetric density modeling is important for occupancy reasoning; however, prior Gaussian-based perception methods (e.g., GaussianFormer) operate without opacity and still perform well for scene understanding. It remains unclear why opacity is essential in this setting and how much it contributes relative to other Gaussian attributes such as scale and rotation. A more thorough justification—ideally including ablations that remove opacity or simplify Gaussians to point-like primitives—would more convincingly validate the necessity of the proposed representation. These studies would arguably be more informative than the current ablations focusing on Gaussian count or feature channel size.

4. The iterative refinement of Gaussian parameters is an interesting component, but its motivation is insufficiently verified. It is unclear whether the observed gains stem from iterative information aggregation or simply from increasing network depth. The paper also lacks analysis of diminishing returns, stability, or how performance scales with different numbers of refinement blocks. Ablations that vary the number of iterations would help validate that iterative refinement is truly necessary and that four iterations is a principled design choice rather than an arbitrary selection. Additionally, reviewer think feed-forward Gaussian approaches such as MVSplat are relevant and should be discussed to clarify why in occ task an iterative refinement strategy is preferred over direct depth+unprojection based Gaussian estimation.

5. The strongest baselines reported are from early–mid 2024, which is somewhat outdated for a late-2025 submission. While the proposed approach appears competitive, the reviewer lacks sufficient visibility into the very latest occupancy prediction works to confidently assess its state-of-the-art standing; other reviewers may have better knowledge of recent advancements. Including/referencing more up-to-date baselines would help clarify the method’s competitiveness.

### Soundness
3

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
3

### Summary
Overall, this paper extends recent Gaussian-based 3D semantic occupancy methods to the multi-sensor setting by proposing GaussianFusionOcc, which takes per-modality feature maps, extracts per-Gaussian features through a “modality-agnostic” deformable-attention encoder, concatenates and MLP-fuses them, refines a fixed set of 3D semantic Gaussians over several blocks, and finally splats them into a voxel occupancy grid.

### Strengths
Overall,  the paper is clearly written and the direction is reasonable.

- i) The paper is well aligned with the ongoing shift and makes the natural next step: “what if we plug multi-sensor fusion into the Gaussian pipeline?” This is a reasonable research question.

- ii) The model design is easy to read and to implement. 

- iii) The experiments cover several realistic sensor combinations

### Weaknesses
- i) The novelty over very close prior work is limited. In substance, the method looks like taking an existing Gaussian-based occupancy model, adding multi-sensor deformable attention in front of it, concatenating features, and keeping the existing splatting stage. 

- ii) The method relies on fairly strong per-sensor encoders. Camera, LiDAR and radar branches all reuse good backbones. Because of this, it is hard to tell whether the gains over the baselines actually come from the proposed Gaussian fusion or simply from using better encoders. 

- iii) The SoTA claim is not fully supported. The main table mixes methods with different modality sets and different backbones. The camera plus LiDAR plus radar result is good, but Occlusion Fusion under a comparable setting is not far behind, and there is no comparison at equal runtime or equal backbone. 

Overall, the contribution in its current form is still below the bar because the idea is close to existing work, the fusion is quite shallow, and the experiments do not yet prove that fusing in Gaussian space is the essential part.

### Questions
i) Since the main claim is that “multi-sensor fusion in Gaussian space” is beneficial, it would be helpful to re-train at least one representative BEV-first or voxel-first multi-sensor occupancy method (e.g., OccFusion or a standard camera–LiDAR fusion model) under exactly the same setting.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces GaussianFusionOcc, a multimodal 3D semantic occupancy prediction framework that integrates data from cameras, LiDAR, and radar. By representing the surrounding scenes using 3D semantic Gaussians and employing modality-agnostic deformable attention, the method effectively fuses heterogeneous sensor information to improve precision and scalability. Experiments on nuScenes show that GaussianFusionOcc achieves higher accuracy, better memory efficiency, and faster inference than existing SOTA methods.

### Strengths
The proposed method effectively addresses the challenge of multimodal fusion for Gaussian-based 3D occupancy prediction. The overall pipeline is simple yet efficient, achieving a good trade-off between performance and computational cost. Experimental results further demonstrate the effectiveness of the multimodal fusion strategy.

### Weaknesses
However, I also have some concerns of this paper:
(1) Although the proposed method effectively tackles Gaussian-based 3D occupancy prediction under a multimodal setting, the approach itself is rather trivial. The fusion strategy and the use of deformable attention have already been widely adopted in 3D object detection. As a top-conference submission, the work does not provide sufficient conceptual depth or novel insight, so I consider it below the ICLR acceptance bar.
(2) The writing of this paper is somewhat weak, particularly in the introduction and the illustration of the overall pipeline in the method section, which appear rather rough and underdeveloped.
(3) The experiments are conducted only on the SurroundOcc dataset, lacking evaluations on mainstream benchmarks such as Occ3D and nuScenes-Occupancy.

### Questions
1. Could the authors further clarify and emphasize their core contributions, including the novel techniques introduced and the potential impact these innovations may have on the research community?
2. Could you provide results on more widely used benchmarks and include comparisons with a broader range of state-of-the-art methods?
3. Please refine this paper carefully to reach the bar of ICLR.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GaussianFusionOcc, a novel framework for 3D semantic occupancy prediction in autonomous driving. It addresses the high computational and memory costs of traditional dense-grid methods by instead using a sparse and efficient 3D Gaussian representation. The paper's main contributions are:
1. It is the first framework to apply 3D Gaussian splatting to multi-modal 3D semantic occupancy prediction, seamlessly fusing data from cameras, LiDAR, and radar. 
2. It proposes a modality-agnostic Gaussian encoder that uses deformable attention to extract relevant features for each Gaussian from all sensor types.
3. It introduces a fusion method to create a unified feature vector, which is used to refine the properties of the 3D Gaussians iteratively.

This approach achieves state-of-the-art performance on the nuScenes dataset (30.37 mIoU and 45.20 IoU). Compared to existing models, it significantly reduces memory usage, parameter count, and inference latency and demonstrates strong robustness in challenging conditions like rain and nighttime.

### Strengths
1. While prior works like GaussianFormer demonstrated the efficiency of Gaussians, they were limited to single-modality inputs. This work creatively addresses that limitation. The core novel components—the "modality-agnostic Gaussian encoder" and the "seamless sensor fusion mechanism" —represent a new and logical combination of existing ideas to solve a clear and present problem.
2. The authors conduct extensive testing on the nuScenes dataset, comparing GaussianFusionOcc not just against one category of model, but against a wide array of state-of-the-art methods.
3. This paper's primary significance is that it offers a solution that is both more accurate and more efficient than existing state-of-the-art fusion models. It achieves SOTA accuracy (30.37 mIoU) while significantly reducing memory usage, number of parameters, and latency.

### Weaknesses
1. The introduction of multi-modal actually is not a novel paradigm for occupancy prediction. And the pipeline of occupancy can be regarded as a multi-modal version of GaussianFormer, making the contribution of this paper fair.
2. In the main results (Table 1), the C+L model achieves 30.21 mIoU. The C+L+R model achieves 30.37 mIoU. This improvement of 0.16 mIoU is negligible and well within the range of training noise, suggesting the radar adds no meaningful information in the general case. The issue is more severe in the "Night scenario" (Table 3). The C+L model scores 18.66 mIoU, while the C+L+R model scores 18.45 mIoU. Adding radar actually degrades performance. This directly contradicts the text, which states "GaussianFusionOcc achieves higher scores with the addition of radar data". This contradiction is not discussed and represents a significant unaddressed finding.
3. The GaussianFusionBlock is the paper's core contribution. However, the fusion itself is a simple concatenation followed by an MLP. Given the failure of radar fusion (Weakness #2), it's highly likely this simple fusion is insufficient. The paper does not ablate this choice.
4. The presentation, such as the Figures, is quite crude.

### Questions
1. Could this be a limitation of the fusion mechanism? The current fusion (concatenation followed by an MLP ) might be too simple to effectively integrate sparse radar data, potentially allowing the denser camera and LiDAR features to "drown out" the radar signal. Did you experiment with other fusion methods (e.g., cross-attention) that might be better suited for this?
2. Could the authors provide experimental results on SemanticKITTI for generalization and robustness?
3. Could the authors provide more visualizations on comparison with previous methods, not merely on ablations?
4. Could you please confirm if this is the case? Specifically, are the refined Gaussian properties (mean, scale, etc.) from block $N$ used as the input Gaussians for block $N+1$? If so, was an ablation study performed on the number of refinement blocks? Understanding how this iterative process contributes to the final accuracy would be very helpful.

### Soundness
2

### Presentation
2

### Contribution
2
