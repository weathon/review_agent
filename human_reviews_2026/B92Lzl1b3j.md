# GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Reconstructing 3D scenes using 3D Gaussian Splatting (3DGS) from sparse views is an ill-posed problem due to insufficient information, often resulting in noticeable artifacts. While recent approaches have sought to leverage generative priors to complete information for under-constrained regions, they struggle to generate content that remains consistent with input observations. To address this challenge, we propose GSFixer, a novel framework designed to improve the quality of 3DGS representations reconstructed from sparse inputs. The core of our approach is the reference-guided video restoration model, built upon a DiT-based video diffusion model trained on paired artifact 3DGS renders and clean frames with additional reference-based conditions. Considering the input sparse views as references, our model integrates both 2D semantic and 3D geometric features of reference views extracted from the visual geometry foundation model, enhancing the semantic coherence and 3D consistency when fixing artifact novel views. Furthermore, we introduce a reference-guided trajectory sampling strategy that ensures both angular coverage and view quality, further enhancing reconstruction fidelity. Considering the lack of suitable benchmarks for 3DGS artifact restoration evaluation, we present DL3DV-Res which contains artifact frames rendered using low-quality 3DGS. Extensive experiments demonstrate our GSFixer outperforms current state-of-the-art methods in 3DGS artifact restoration and sparse-view 3D reconstruction. The Project will be made public.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces VGGT and DINOv2 as additional conditions to enhance the performance of leveraging video diffusion prior to 3D Gaussian Splatting. It also proposes a novel reference-guided trajectory strategy that improves view-angle coverage during sampling, thereby enhancing the final reconstruction quality. Extensive experimental results demonstrate that the proposed method achieves strong and consistent performance improvements.

### Strengths
1. This paper introduces features from VGGT and DINOv2 as additional conditioning inputs for video diffusion within the 3D Gaussian Splatting framework. The idea is novel and brings clear performance improvements over existing methods.

2.The proposed Reference-Guided Trajectory provides a valuable insight into how to generate more suitable camera poses for rendering, which can help produce better inputs for the diffusion model and ultimately improve 3D Gaussian Splatting performance. This is an interesting and meaningful direction.

3.The paper is clearly written, and the extensive experiments convincingly demonstrate that the proposed method is effective.

### Weaknesses
1. While the method is technically sound, I find the contribution somewhat incremental. The overall framework is similar to previous approaches such as Difix3D+, which also progressively integrate diffusion priors to improve 3D Gaussian Splatting. Using video diffusion models to enhance 3DGS is no longer new, and incorporating 3D information into diffusion models has already been explored in works like ReconX. Thus, the novelty of this paper is somewhat limited.

2. I would have liked to see a deeper discussion on why the diffusion model benefits from conditioning on features from DINOv2 or VGGT since it is the paper's key contribution. For instance, if we consider using geometry foundation models such as VGGT  MapAnything, or MVSanywhere [3]  to generate depth maps as a supervision for 3DGS, would this be more effective than using them as conditioning inputs to the diffusion model? Addressing such questions could provide a stronger understanding of the method’s design choices and its underlying reason.

3. Difix3D+ already mentioned that its diffusion model supports multiple reference views, and the recent paper FlowR [4] also demonstrates that using multi-view reference information can effectively improve 3D Gaussian Splatting. Therefore, the core idea of reference-guided is not entirely novel in current stage. 

Below are several relevant citations I included in my review that were not discussed in the paper.

[1] Liu, F., Sun, W., Wang, H., Wang, Y., Sun, H., Ye, J., ... & Duan, Y. (2024). Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767.

[2] Keetha, N., Müller, N., Schönberger, J., Porzi, L., Zhang, Y., Fischer, T., ... & Kontschieder, P. (2025). MapAnything: Universal feed-forward metric 3D reconstruction. arXiv preprint arXiv:2509.13414.

[3] Izquierdo, S., Sayed, M., Firman, M., Garcia-Hernando, G., Turmukhambetov, D., Civera, J., ... & Watson, J. (2025). MVSAnywhere: Zero-Shot Multi-View Stereo. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 11493-11504).

[4] Fischer, T., Bulò, S. R., Yang, Y. H., Keetha, N., Porzi, L., Müller, N., ... & Kontschieder, P. (2025). Flowr: Flowing from sparse to dense 3d reconstructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 27702-27712).

### Questions
1. Why is a new benchmark necessary, and what limitations of existing datasets does it address?
2. What are the training settings for your 3D Gaussian Splatting, including whether the points are initialized randomly or from colmap, and  the learning rate for 3DGS, difix3D and your method?
3. For Difix3D+, did you evaluate with the post-rendering refinement step, or report the score without it?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes GSFixer, a novel framework for improving 3DGS reconstruction quality in sparse-view scenes. Due to the underconstraint problem caused by sparse view input, traditional 3DGS is prone to geometric distortion and new-view artifacts. GSFixer inpaints these artifacts by introducing a reference-guided video diffusion model. During training, the model utilizes pairs of artifact-laden 3DGS rendered frames and high-quality GT frames. It also combines  DINOv2 feature extractor and 3D geometric features eature extractor from the sparse-view input to provide a hard constraint for the video diffusion model, thereby achieving semantically and geometrically consistent inpainting from new views.

### Strengths
1. **Originality-wise**: the paper proposes a novel framework for robust 3D reconstruction.
2. **Quality-wise**: the proposed method has combine Dinov2 and VGGT head to extract the 2D/3D features to fix the 3D scenes with gaussian splatting.
3. **Clarity-wise**: the manuscript is clearly written, with well-structured methodology, detailed explanations, and intuitive visualizations that enhance understanding.

### Weaknesses
1. **Limited generalization of camera trajectories.** The reference-guided trajectory strategy performs well on circumferential trajectories, but its adaptability and effectiveness on non-circumferential or sparse, discontinuous trajectories (such as extreme angles and non-closed-loop aerial photography) have not been fully verified. The model may be sensitive to the distribution of reference viewpoints and lack the ability to generalize to diverse trajectories. More exps and analysis in complex environments need to be conducted.

2. Concern about the dependency of poses and SfM initialization. The COLMAP results from all viewpoints were used as the initial point cloud, from which the sparse viewpoints required for training were selected. While this setup ensures high SfM initialization accuracy, it is inconsistent with real-world sparse scene applications. In real-world scenarios, input images are often few and viewpoints overlap less, directly using a sparse image set for COLMAP reconstruction may fail or significantly reduce accuracy and stability. I'm curious about the performance only with few input images without any eval images. 

---
I have listed my concerns, and the score will be adjusted based on the author's response.

### Questions
Please refer to Weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes GSFixer designed to improve the quality of 3D Gaussian Splatting (3DGS) reconstructions from sparse input views. 

GSFixer: A generative reconstruction pipeline integrating a reference-guided video restoration model and trajectory sampling strategy.

Dual Conditioning: Uses both 2D semantic and 3D geometric features from reference views to guide artifact correction.

DL3DV-Res Benchmark: A new dataset for evaluating 3DGS artifact restoration.

Performance: Good results in artifact restoration and sparse-view reconstruction across multiple benchmarks.

### Strengths
I think the most original contribution is the reference-guided video restoration model, which conditions a video diffusion model on both 2D semantic features (via DINOv2) and 3D geometric features (via VGGT) extracted from the input sparse views. Different from previous works, the condition is multimodal. 

The introduction of the RGT strategy is a clever way to refine the 3DGS systematically.

### Weaknesses
The scales, dimensions, and information densities of the geometric features (VGGT) and semantic features (DINOv2) may differ, and improper fusion may result in the weakening of one of the features. How to confirm that the fusion is correct for both features? Did not see the ablation studies of this part.

### Questions
Could you have some attention visualizations to confirm that the fusion works?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes GSFixer, which uses a DiT-based video diffusion model (CogVideoX) to “repair” artifact-ridden 3DGS novel views, conditioning the diffusion on 2D semantic tokens (DINOv2) and 3D geometric tokens (VGGT) extracted from sparse reference images. The restored frames are then distilled back into 3DGS in an iterative loop, and a reference-guided trajectory is introduced to balance view quality and angular coverage. The authors also release DL3DV-Res, a benchmark of artifact frames rendered from low-quality 3DGS, and report improvements over GenFusion and Difix3D+ on restoration and sparse-view NVS. Results on DL3DV and Mip-NeRF-360 show mostly modest gains in PSNR/SSIM/LPIPS gains, with ablations indicating that both 2D and 3D tokens are beneficial.

### Strengths
**Task framing + benchmark.** Explicitly casting “artifact restoration” for sparse-view 3DGS and introducing **DL3DV-Res** gives a concrete testbed; the task is well motivated, and the dataset construction is described. Reported scores show clear, if not dramatic, gains over prior generative baselines on this benchmark.

- **Dual conditioning (2D + 3D tokens).** Conditioning the video diffusion on DINOv2 (semantics) and VGGT (geometry) is a coherent way to push consistency to the fixed frames, and the injection via cross-attention is straightforward. The ablation results show that the full model outperforms the “w/o 2D” and “w/o 3D” variants across all metrics.
 
- **Reference-guided trajectory.** The paper identifies a hybrid camera path that balances interpolation quality with spherical coverage. Although the gains are small, the trajectory is at least validated with both qualitative and quantitative comparisons.

### Weaknesses
**Incremental relative to recent generative NVS systems** - The method largely leverages known techniques, including video diffusion restoration, iterative distillation back to the 3D representation, and simple camera path sampling, and is very close in spirit to GenFusion/Difix3D+. ** The paper’s own related-work section positions it as a variation rather than a conceptual step change. The only new contribution is the type of conditioning signal used in the fix step, making this an incremental improvement to an existing framework rather than a new one. The novelty claim boils down to adding dual token conditioning and a minor trajectory tweak.
- **Trajectory ablation has small effect sizes** (e.g., ≤0.2 dB PSNR) over usual elliptical trajectories, making it unclear whether the complexity is warranted.
- **Under-analyzed compute footprint** - The core model is CogVideoX-5B with ~50 diffusion steps, plus extra encoders (BLIP, DINOv2, VGGT) at inference/conditioning; there’s no runtime/memory breakdown or wall-clock profiling, and the method’s practicality is relegated to a brief limitations section. For a system paper, this lacks the expected engineering transparency.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1
