# Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering. Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering. To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes. However, our analysis reveals that significant redundancy still remains due to the lack of occlusion awareness. In this work, we propose Proxy-GS, a novel pipeline that exploits a proxy to introduce Gaussian occlusion awareness from any view.
At the core of our approach is a fast proxy system capable of producing precise occlusion depth maps at resolution 1000$\times$1000 under \SI{1}{ms}. This proxy serves two roles: first, it guides the culling of anchors and Gaussians to accelerate rendering speed. Second, it guides the densification towards surfaces during training, avoiding inconsistencies in occluded regions, and improving the rendering quality. 
In heavily occluded scenarios, such as the MatrixCity Streets dataset, Proxy-GS not only equips MLP-based Gaussian splatting with stronger rendering capability but also achieves faster rendering speed. Specifically, it achieves more than $2.5\times$ speedup over Octree-GS, and consistently delivers substantially higher rendering quality. Code will be public upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Proxy-GS, an occlusion-aware framework for MLP-based 3D Gaussian Splatting that uses a lightweight proxy mesh to (i) cull occluded anchors at test time via a fast hardware-rasterized depth map (~1k×1k under ~1 ms) and (ii) guide densification toward visible surfaces during training. The proxy-guided filter is fused with frustum culling in a single CUDA pass; proxy-guided densification projects errorful patches back to the proxy surface to grow anchors structurally. On occlusion-rich scenes (e.g., MatrixCity Streets), Proxy-GS improves quality and boosts FPS, with ablations showing best trade-offs at modest safety margins and confirming benefits from training-time consistency.

### Strengths
1. The paper introduces a proxy-guided Gaussian representation, which effectively leverages proxy geometry to guide Gaussian splatting for improved rendering quality and efficiency. This design elegantly integrates geometry awareness into the learning process, leading to better scene reconstruction and occlusion handling.
2. The paper is well written and clearly organized.

### Weaknesses
**Major**
1. The main idea of removing occluded Gaussians to accelerate rendering speed has already been proposed in OccluGaussian. The authors should cite, discuss, and compare their approach with this strong baseline.
2. The use of preprocessed proxy meshes seems quite tricky. First, I believe PGSR may not work well in very large scenes, such as the *MatrixCity small-city* dataset (which includes entire aerial or street-level environments). This is a significant limitation. Second, please provide details on the training time and storage requirements, including the time required to preprocess the proxy mesh, since users applying proxy-GS must start from scratch.
3. No demo has been submitted, so the performance of the proposed method cannot be effectively demonstrated.
4. The authors conducted experiments on only a few scenes (six in total), and the rendering metrics improve over the baselines by only 0.1 PSNR. I believe these results could easily be surpassed by tuning hyperparameters.

**Minor**
1. The paper lacks references to important related work on Level of Detail (LOD) and large-scale rendering. The authors should consider citing the following works:
    - *OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering*
    - *LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering*
    - *Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes*
    - *Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes*
    - *Vast 3D Gaussians for Large Scene Reconstruction*

### Questions
please see the weakness above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Proxy-GS, a framework for accelerating MLP-based 3D Gaussian Splatting (3DGS) by introducing a proxy-mesh–guided occlusion filter and densification mechanism. The core idea is to pre-compute a lightweight proxy mesh to perform depth-based occlusion culling via hardware rasterization (<1 ms) and use these depth maps both during training (to guide anchor densification) and inference (to cull occluded anchors). The authors claim that this approach improves efficiency and rendering quality simultaneously, reporting up to a 3× FPS increase and slight PSNR gains over Octree-GS and other baselines on datasets such as MatrixCity, Small City, and CUHK-LOWER.

### Strengths
1. The paper addresses a relevant efficiency problem in MLP-based 3DGS rendering, which is indeed computationally heavy.

2. The proposed integration of proxy meshes into training/inference is intuitive and could be useful for applied rendering systems.

### Weaknesses
1. No theoretical grounding: The proxy-guided densification lacks a clear optimization objective or loss formulation beyond heuristics; equations (2–9) only restate camera transformations.

2. Weak improvement over Octree-GS: The reported gains compared with Octree-GS are modest and often within the margin of experimental noise. Moreover, since the entire codebase and training pipeline are directly built upon Octree-GS, it is difficult to disentangle whether the improvements truly originate from the proposed proxy mechanism or from tuning and implementation differences. This raises doubts about the actual effectiveness and generality of the method.

### Questions
1. How robust is Proxy-GS to inaccurate or incomplete proxy meshes? Quantitative sensitivity analysis is missing.

2. Why is “occlusion-guided densification” superior to simply pruning via hierarchical Z-buffering or view-space LOD strategies?

3. Were all methods trained with identical data augmentations, learning rates, and iteration counts?

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
4

### Summary
This paper proposes Proxy-GS, an occlusion-aware and proxy-guided training and rendering framework for 3D Gaussian Splatting (3DGS). The central idea is to introduce a lightweight proxy mesh derived from SfM/MVS pipelines, and to use it as a structural scaffold to: 1) Guide densification toward geometrically meaningful regions. 2) Enable occlusion-aware pruning and rendering. 3) Reduce redundancy while maintaining reconstruction quality.

### Strengths
Introducing proxy-guided Gaussian growth is conceptually elegant and practically useful.

Results are good, particularly in occlusion-heavy environments.

Visualizations effectively reveal how proxies shape spatial distribution.

### Weaknesses
1. Dependence on Proxy Quality:

If the proxy mesh is coarse / noisy, how robust is Gaussian distribution?

Ablations on proxy resolution and noise are missing.

2. Fairness of Experimental Comparison:

Some baselines are not clearly described regarding training iterations, learning rate tuning, and hardware.

Include a wall-clock cost per iteration comparison.

3. Scalability Beyond Desktop-Scale Scenes:

Claims target “ultra-large VR walkthroughs” but datasets are still mid-scale.

Evaluate on a truly large-scale multi-block urban dataset (e.g., Tanks&Temples extended / UrbanScene3D).

4. LOD vs. Occlusion Interactions:

The method implicitly resembles geometric LOD.

Discuss when proxy-GS degenerates to simple LOD behavior.

### Questions
1. What failure behaviors emerge when the proxy mesh is inaccurate or incomplete? 

2. How sensitive is the densification policy to mesh vertex density and normal consistency?

3. Is the occlusion culling performed per-ray or per-fragment? Clarify computational impact.

4. Could proxy features replace Gaussians in distant regions entirely, forming a hybrid representation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces Proxy-GS, a large-scale 3D Gaussian Splatting (3DGS) framework designed for both rendering and training. The proposed method utilizes a proxy mesh for occlusion culling, which facilitates both proxy-guided rendering and densification. Experimental results on multiple datasets demonstrate the effectiveness of the approach.

### Strengths
This work on large-scale real-time scene rendering and training represents a timely and practical direction for 3D Gaussian Splatting.

### Weaknesses
Introduction  
1. Line 096. One of the main contributions 'We leverage engineering optimizations ...' seems to describe an implementation rather than a methodological contribution.



METHOD  
1. The framework is not clearly illustrated, and Figure 2 is not referenced anywhere in the manuscript.
2. The proposed method heavily relies on a pre-generated mesh as a proxy, which limits its practicality. Moreover, the use of such a proxy may result in an unfair comparison with mesh-free approaches.
3. The motivation for the proxy-guided filter is unclear. It is feasible to extract visibility or depth for occlusion estimation directly from 3DGS models. Why is an additional mesh necessary?
4. Equations (2) to (8) appear to replicate the culling implementation of the original 3DGS method, but the manuscript does not provide appropriate citations.


EXPERIMENT
1. It would be helpful to provide the computational or time budget required for obtaining the mesh.
2. It would be useful to include failure cases resulting from unsuccessful mesh reconstruction. For example, surface reconstruction may fail to recover fine details in large-scale scans.

Typo:

line 230: interpretability -> interpretability.

### Questions
1. This manuscript is not well-written, and it is recommended to restructure the manuscript.
2. It would be better clarify the necessity of proxy mesh.

### Soundness
2

### Presentation
2

### Contribution
2
