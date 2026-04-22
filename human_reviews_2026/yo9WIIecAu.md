# Efficient Multi-View 3D Representation via Fusion of View-Agnostic Transformations

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Bird's-Eye View representations are essential for 3D perception in autonomous driving, providing unified and spatially coherent scene understanding. While attention-based methods achieve strong performance through global cross-view attention, they suffer from computational inefficiencies due to redundant referencing and spatial ambiguity from ego-centric projections. To address these limitations, we introduce Mosaic View Transformation (MosaicVT), a modular framework that independently transforms multi-camera views into a unified BEV space. MosaicVT employs a camera-centric polar coordinate system, effectively resolving directional ambiguity and reducing cross-view redundancy. A novel view-agnostic positional embedding enables a single transformation module to generalize across heterogeneous camera configurations without retraining. Transformed camera-centric representations are then aligned and fused into a global BEV using a geometry-aware interpolation strategy, significantly reducing computational overhead compared to global attention mechanisms. Experimental results on the nuScenes benchmark demonstrate that MosaicVT achieves state-of-the-art performance in 3D object detection and BEV semantic segmentation while providing substantial reductions in latency and maintaining robust generalization across diverse camera setups.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel framework for efficient multi-camera BEV representation in autonomous driving. 
The method addresses redundancy in transformer-based view transformation by processing each camera view independently in a camera-centric polar coordinate system. 
Key innovations include View-Agnostic Positional Embedding (VAPE) for consistent 3D localization and a modular transformation-fusion pipeline that avoids cross-view attention. 
Experiments on nuScenes show that MosaicVT outperforms existing methods in 3D detection and BEV segmentation while reducing latency. 
It also demonstrates robustness to camera perturbations and configuration changes.

### Strengths
1. Computational Efficiency and Reduced Redundancy. MosaicVT eliminates the computationally expensive global cross-view attention used in transformer-based methods. 
2. Novel and Robust Formulation. The introduction of the camera-centric polar coordinate system and View-Agnostic Positional Embedding (VAPE) effectively resolves spatial ambiguity inherent in image-to-BEV transformation.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The method is based on the assumption that cross-view interaction is not important for global BEV. It might be true for nuScenes with little FoV overlap between cameras. However, the paper does not take cases into consideration where there might be considerable FoV overlap between surround cameras.
2. The paper does not compare against recent state-of-the-art methods for object detection and BEV segmentation.

### Questions
1. As in the weakness section, could you please discuss the validity of the assumption that cross-view attention is negligible?
2. Why do you only compare with methods that is more than one year from now?

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
This paper proposes a modular framework, MosaicVT, that generates a unified BEV representation by independently transforming multiple camera viewpoints. MosaicVT uses a central camera polar-coordinate system to address directional ambiguity and multi-view redundancy effectively. Through novel view-independent position embeddings, MosaicVT can generalize to different camera configurations without retraining. The generated camera-centered BEV representations are aligned and fused using a geometry-aware interpolation strategy, thereby significantly reducing computational overhead while maintaining high accuracy and robustness. Experimental results show that MosaicVT achieves state-of-the-art performance on 3D object detection and BEV semantic segmentation tasks on the nuScenes benchmark, while significantly reducing latency and performing robustly across different camera configurations.

### Strengths
Different from previous methods that uniformly process the global view, we propose a method that processes each camera view independently and then aggregates them into a unified BEV space, effectively reducing cross-view interference and spatial ambiguity while avoiding the computational overhead of the global attention mechanism.
The method introduction is clear and logical. The authors explain the core idea through detailed formula derivation and provide complete mathematical proofs for key details.
In the experimental part, the effectiveness of introducing camera-centered polar coordinates as position embedding is effectively proved by detailed comparison experiments and ablation experiments

### Weaknesses
1. Using polar coordinates for position encoding instead of Cartesian coordinates is actually a relatively common idea, which may not be very innovative.
2. When feature conflicts exist between different views, simple averaging and 2D convolution lack a dynamic arbitration mechanism to intelligently decide which view should be preferentially adopted, which may perform poorly when dealing with complex occlusions and view conflicts compared with more advanced fusion modules, such as attention-based fusion modules.

### Questions
Compared with the previous unified global-view processing method, is the weighted-average fusion method and 2D convolution adopted in this paper too simple? 
Although this lightweight design is efficient, is its ability to aggregate information sufficient? 
Are there more advanced fusion methods (e.g., cross-view attention) that can handle information interaction between views more robustly?

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
The paper proposes MosaicVT, a BEV view-transformation module that (i) transforms each camera independently in a camera-centric polar frame (angles θ, radial distance r, and height h), (ii) makes the transformation view-agnostic via a positional encoding that uses relative geometric cues so one shared module can generalize across heterogeneous camera setups, and (iii) aligns & fuses the per-camera BEV “tiles” into a global BEV using geometry-aware interpolation—avoiding global cross-view attention altogether

### Strengths
The paper pinpoints two concrete issues—redundant cross-view referencing and ego-centric ray ambiguity and ties each to a design choice (per-camera VT + camera-centric polar coordinates), giving a clean problem, method story. 

MosaicVT swaps only the view transformation stage inside standard pipelines (e.g., BEVDet for detection, BEVFusion for BEV segmentation), leaving heads/decoders unchanged—useful for real systems.

### Weaknesses
1. Motivation:
  a. The paper says prior view-transform “uses global cross-view attention,” but BEVFormer uses deformable (sparse) cross-attention from BEV queries, and LSS lifts frustums then splats to BEV—no global attention. Please clarify.
  b. The “A core challenge of this design is ensuring that a single transformation module can operate across heterogeneous camera setups” claim lacks a baseline analysis: why can’t BEVFormer/LSS work across heterogeneous camera?

2. Performance:
  a. Temporal length/stride/cache aren’t specified, and some numbers seem below commonly reported BEVFormer-base ≈ 0.517 NDS / 0.416 mAP on nuScenes—please detail the exact setting and explain the gap.
  b. Limited comparison coverage. Add more recent multi-view detectors (or justify exclusions) (e.g. Far3D, BEVNeXt, GeoBEV)

### Questions
1. In Table 9, why Camera $8\times 22$ results in poor performance?

### Soundness
3

### Presentation
2

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
This paper proposes MosaicVT, a modular framework for multi-camera BEV representation learning, aiming to address the computational inefficiency from cross-view redundancy and spatial ambiguity in ego-centric projections of existing attention-based methods. Extensive experiments on the nuScenes benchmark demonstrate that MosaicVT achieves competitive performance in 3D object detection and BEV semantic segmentation, with substantially reduced latency compared to transformer-based methods. It also exhibits strong robustness to variations in camera configuration and calibration noise.

### Strengths
1. MosaicVT processes each camera view independently using a camera-centric polar coordinate system, which avoids the unnecessary global cross-view attention in transformer-based methods.
2. The proposed VAPE embeds image features using relative geometric cues, abstracting away camera-specific parameters. This enables a single transformation module to adapt to diverse camera setups without retraining.
3. Experiments on simulated camera configuration changes and real-world calibration noise  show that MosaicVT outperforms WidthFormer and LSS in robustness.

### Weaknesses
1. The method of converting image features into Polar BEV and then obtaining BEV features through sampling is somewhat similar to the RC-Sample proposed by GeoBEV[1]. The advantages of MosaicBEV need to be further demonstrated.
2. MosaicBEV has not been compared with current SOTA methods, such as RayDN[2], BEVNext[2] and so on.
3. In Table 4, MosaicBEV is only compared with WidthFormer in terms of efficiency. The efficiency comparison with LSS-based methods should also be added.

[1] Zhang, Jinqing, et al. "Geobev: Learning geometric bev representation for multi-view 3d object detection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 9. 2025.\
[2] Liu, Feng, et al. "Ray denoising: Depth-aware hard negative sampling for multi-view 3d object detection." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.\
[3] Li, Zhenxin, et al. "Bevnext: Reviving dense bev frameworks for 3d object detection." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

### Questions
4. DFA3D also uses depth distribution to avoid the problem of depth uncertainty. What advantages does MosaicBEV have compared with DFA3D?
5. In Figure 7, why is the impact of vertical crop on LSS greater than that of horizontal crop?

### Soundness
3

### Presentation
3

### Contribution
2
