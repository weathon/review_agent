# SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments. This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world. Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data. Multi-sensor simulation poses additional challenges as existing methods handle cross-sensor inconsistencies by favoring the quality of one modality at the expense of others. To overcome these limitations, we propose SimULi, the first method capable of rendering arbitrary camera models and LiDAR data in real-time. Our method extends 3DGUT, which natively supports complex camera models, with LiDAR support, via an automated tiling strategy for arbitrary spinning LiDAR models and ray-based culling. To address cross-sensor inconsistencies, we design a factorized 3D Gaussian representation and anchoring strategy that reduces mean camera and depth error by up to 40% compared to existing methods. SimULi renders 10-20$\times$ faster than ray tracing approaches and 1.5-14$\times$ faster than prior rasterization-based work (and handles a wider range of camera models). When evaluated on two widely benchmarked autonomous driving datasets, SimULi matches or exceeds the fidelity of existing state-of-the-art methods across numerous camera and LiDAR metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
SimULi extends 3DGUT by incorporating LiDAR support, enabling simultaneous modeling of multimodal sensors in autonomous driving scenarios. The system achieves performance on both camera and LiDAR modalities that matches or surpasses the current state of the art.

### Strengths
- Achieves joint modeling of multimodal sensors (camera and LiDAR) in autonomous driving scenes, with each modality outperforming existing single-modality methods.

 - Incorporates modeling of fisheye cameras and rolling-shutter effects, demonstrating a highly comprehensive design.

 - Significantly improves rendering speed.

### Weaknesses
- The interaction between camera and LiDAR Gaussian primitives relies solely on the anchor loss, which may lead to redundant Gaussian points.

 - The system demonstrates strong engineering merit, but the novelty is relatively limited.

### Questions
- How is LiDAR integrated into 3DGUT? Are the seven points projected and weighted following the LiDAR’s ray model? If so, could the authors provide a visualization similar to Fig. 7 in 2DGS to illustrate the validity of this approximation? As nonlinearity of LiDAR projection is typically stronger than that of camera projection.

 - Is the interaction between camera and LiDAR Gaussian primitives only achieved through the anchor loss? Would such a weak coupling lead to redundant Gaussian points?

 - Could the proposed approach be combined with recent generative-aided reconstruction methods (e.g., StreetCrafter, DriveX)?

 - Which component contributes most to the significant improvement in rendering speed?

### Soundness
3

### Presentation
4

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
SimULi describes a framework for real-time simulation of LiDAR and camera for autonomous driving scenes. The focus of this work is to improve the rendering speed and support arbitrary LiDAR and camera configurations. To do so, this work builds on 3DGUT to support arbitrary spinning LiDAR configuration. Furthermore, a factorized 3D Gaussian representation and anchoring strategy was proposed to address discrepancies between simulated LiDAR and camera data. The proposed method was benchmarked on two datasets to showcase the fidelity and efficiency in relation to existing works.

### Strengths
1.	The paper is well-written and easy to follow. The limitations of prior works and their relation to the proposed work are clearly highlighted.
2.	The proposed method demonstrates strong rendering fidelity and significant boosts in speed compared to state-of-the-art methods.

### Weaknesses
1.	The proposed method has been evaluated on a relatively limited set of datasets. Common benchmarks used in prior works, such as nuScenes and Argoverse 2, would help demonstrate the robustness of the method across datasets and sensor setups.
2.	SplatAD encodes all sensor information into the same Gaussian set. SimULi proposes to encode each sensor into its own particle set to address the inconsistencies between LiDAR and camera. The impact of this on the training time and potentially memory requirements is not discussed.

### Questions
1.	How does factorizing the Gaussian set impact training speed? How does the training time of the proposed work compare against existing works?
2.	For the anchoring loss, the choice of 50 nearest neighbors and updating assignments every 1000 iterations are hyperparameters. How does varying the number of nearest neighbors and update frequency impact convergence?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
this paper introduces a simulator method for autonomous driving that can render both complex camera models and LiDAR data in real time. It builds on 3DGUT and improves cross-sensor consistency, making it more accurate and faster than existing methods. The paper conducted experiments on two public datasets and show state-of-the-art performances.

### Strengths
1. The paper is well written and easy to follow.
2. The proposed method consistently outperforms baselines according to both visualizations and tables. 
3. The proposed method achieves significantly faster rendering speed than baselines.
4. The  factorized representation is interesting and innovative, which improves both the camera and lidar rendering accuracy.

### Weaknesses
would the factorized representation largely increase the total number of Gaussians in the scene? Does this lead to significantly higher memory usage compared to unified representations?

### Questions
1. From what I understand, the authors equalize the elevation tiling using a 1D CDF of elevation angles, and then reuse the same azimuth tiling across the whole scan. Wouldn’t that implicitly assume that the LiDAR point distribution is separable between elevation and azimuth, and also static over time? In practice, the point density may vary a lot with azimuth (depending on the scene or motion), so a fixed azimuth tiling might lead to load imbalance — some tiles being overloaded and others almost empty. Did you observe this issue in your experiments, or did you use any mechanism to adapt the azimuth tiling dynamically?
2. Since each camera Gaussian is softly constrained to stay close to its nearest LiDAR neighbor, how sensitive is the method to that K-nearest-neighbor choice (K = 50)? Also, do you ever observe issues where the anchoring loss pulls camera Gaussians toward noisy or missing LiDAR points, especially around thin structures or reflective surfaces?

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
5

### Summary
This paper introduces a 3DGS-based simulator for real-time camera and LiDAR rendering in autonomous driving. The main contributions include a non-equidistant tiling strategy that efficiently handles arbitrary spinning LiDAR sensors, and a factorized 3D Gaussian representation that mitigates cross-sensor inconsistencies between camera and LiDAR modalities, thereby improving rendering realism. The experiments are comprehensive, with extensive comparisons on the Waymo and PandaSet datasets, demonstrating state-of-the-art performance and realism.

### Strengths
* The paper is well written and clearly motivated.
* Experiments are thorough and cover multiple datasets and baselines. The proposed method achieves strong quantitative and qualitative performance, demonstrating competitive or superior rendering realism and efficiency.

### Weaknesses
* The novelty is limited. The method feels like a natural extension of 3DGUT, and the way LiDAR rendering is supported is conceptually similar to SplatAD.
* The improvement in handling the camera–LiDAR accuracy tradeoff mainly comes from the decoupled representation, but the deeper issue—imperfect sensor modeling (e.g., motion blur, rolling shutter, or calibration)—is not really addressed. Prior work such as NeuRAD has shown that explicitly modeling these effects can significantly improve reconstruction quality.
* This paper also omits related efforts such as AlignMiF, which also tackle multimodal alignment in autonomous driving simulation. A discussion or comparison with AlignMiF would make the contribution clearer and better positioned.

### Questions
* The paper claims improved cross-sensor consistency, but how does the method perform when modeling more accurate physical effects such as rolling shutter, motion blur, or calibration errors? This could also be evaluated under controlled conditions, for example using CARLA.
* Can the proposed LiDAR tiling strategy generalize to non-spinning LiDAR sensors, such as solid-state LiDARs, where the sampling pattern is different?

### Soundness
3

### Presentation
3

### Contribution
2
