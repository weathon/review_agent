# Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Reconstructing dynamic 3D scenes from monocular input is fundamentally under-constrained, with ambiguities arising from occlusion and extreme novel views. While dynamic Gaussian Splatting offers an efficient representation, vanilla models optimize all Gaussian primitives uniformly, ignoring whether they are well or poorly observed. This limitation leads to motion drifts under occlusion and degraded synthesis when extrapolating to unseen views. We argue that uncertainty matters: Gaussians with recurring observations across views and time act as reliable anchors to guide motion, whereas those with limited visibility are treated as less reliable. To this end, we introduce USplat4D, a novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates reliable motion cues to enhance 4D reconstruction. Our approach estimates time-varying per-Gaussian uncertainty and leverages it to construct a spatio-temporal graph for uncertainty-aware optimization. Experiments on diverse real and synthetic datasets show that explicitly modeling uncertainty consistently improves dynamic Gaussian Splatting models, yielding more stable geometry under occlusion and high-quality synthesis at extreme viewpoints. Project page: https://tamu-visual-ai.github.io/usplat4d/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel framework for monocular 4D scene reconstruction, emphasizing the critical role of uncertainty estimation in dynamic Gaussian splatting. The authors address key challenges in reconstructing dynamic scenes from monocular videos, particularly under occlusions and extreme viewpoint changes, by explicitly modeling the uncertainty associated with each Gaussian component in the scene. The proposed approach constructs an uncertainty-aware graph that guides the selection of reliable scene components (key nodes) and propagates motion information accordingly, resulting in improved geometric consistency and rendering quality.

### Strengths
The primary contribution of explicitly modeling and utilizing uncertainty within the scene reconstruction process represents a significant advancement. This approach effectively addresses a persistent challenge in monocular 4D reconstruction—robustly managing occlusions and view-dependent ambiguities.

The paper presents a systematic development of the framework, encompassing uncertainty estimation, graph construction, and optimization. It offers a comprehensive and cohesive pipeline that integrates seamlessly with existing Gaussian splatting methods. The ablation studies convincingly demonstrate the importance of each component, underscoring how uncertainty modeling enhances reconstruction quality.

The authors validate their approach across multiple datasets, including challenging synthetic benchmarks designed to rigorously test robustness. Both quantitative metrics and qualitative results substantiate that USPLAT4D delivers superior performance, especially in maintaining scene details from extreme viewpoints.

### Weaknesses
The effectiveness of the proposed framework hinges critically on the precision of the uncertainty estimation. Any inaccuracies in this component could propagate through the graph construction and optimization stages, potentially compromising the stability and fidelity of the reconstructed scene.

While the paper demonstrates significant improvements under various conditions, it provides limited discussion on scenarios where the uncertainty modeling may underperform. A more comprehensive analysis of failure modes—such as highly textureless regions or rapidly changing scenes—would strengthen the robustness claims.

Incorporating per-Gaussian uncertainty estimation and uncertainty-guided graph optimization introduces additional computational complexity. The paper lacks detailed analysis regarding the impact on processing time, resource requirements, and scalability to larger or real-time applications.

### Questions
How does the uncertainty-guided graph approach specifically improve robustness in occluded or fast-moving regions? Are there quantitative metrics or qualitative assessments that distinctly highlight these enhancements?

What is the added computational cost of incorporating the uncertainty estimation and graph optimization steps? Are there insights into throughput or latency impacts, especially for real-time or large-scale scenarios?

Did the experiments include scenarios with severe occlusion, sparse observations, or highly ambiguous regions? How does the model's uncertainty estimation perform under these challenging conditions?

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
This work reconstruct 4D Gaussian Splatting for dynamic scenes from monocular videos. The proposed technique refines an initial 4DGS reconstructed by other method. They first classify each Gaussian into key and non-key nodes based on the uncertainty derived from rendering loss. Key nodes are finetuned to remain their original position and motion, while non-key nodes are regularized by the blended motion from nearby key nodes. Results on three datasets show improvement comparing to previous work.

### Strengths
I like the idea of to identify anchor Gaussians based on the rendering loss uncertainty. This may be useful for many other tasks which require some estimate of the importance of a Gaussian.

### Weaknesses
The original 4DGS seems to have much better quantitative and qualitative results from their paper, while the reported 4DGS looks much worse. For example, they report >30 PSNR on the synthetic dataset but the PSNR are all <20. I wonder the reason. Does the evaluated dataset in this paper exhibit some more challenge? Also, is will be even more convincing if the proposed method can be evaluated on the same set of datasets used in 4DGS.

I'm concern about the scalability of the graph pre-processing stage to scene with more Gaussians (e.g., larger scene, more complex scene, or long sequence). The existing graph pre-preprocessing already requires additional 10 more minutes. The computation and storage cost may increase even further and make it unpractical.

### Questions
Can the author provide some space and time complexity analysis for the graph pre-processing step (how they scale with the number of Gaussians)?

### Soundness
3

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
4

### Summary
This paper presents USPLAT4D, a method for reconstructing 4D scenes from monocular video that explicitly models per-Gaussian, per-frame uncertainty and integrates it into the optimization process. The approach consists of three main components: (1) uncertainty estimation based on color reconstruction variance and convergence indicators, (2) uncertainty-aware k-NN (UA-kNN) graph construction with key/non-key node partitioning, and (3) unified optimization using strong anchor losses for key nodes and Dual Quaternion Blending (DQB) interpolation with uncertainty-weighted losses for non-key nodes.

### Strengths
1. Clear conceptual contribution: The paper articulates a compelling and principled insight—"frequently observed Gaussians serve as stable motion anchors"—and successfully integrates this into a quantitative optimization framework. The uncertainty-aware formulation is well-motivated.
2. Effective spatial-temporal propagation: The UA-kNN approach with key/non-key partitioning provides an elegant mechanism to propagate motion from reliable nodes to uncertain ones, effectively suppressing drift in occlusion scenarios.
3. Depth-aware uncertainty modeling: The propagation of 2D errors to 3D anisotropic covariances via camera rotation explicitly accounts for depth-directional uncertainty, addressing a practical issue (depth distortion) that many methods overlook.

### Weaknesses
1. Dependence on initialization: The method relies on pretrained dynamic Gaussian models (SoM/MoSca) for initialization. When initial tracking completely fails (e.g., textureless regions, extremely fast motion), uncertainty estimation alone cannot recover from poor initialization. While the authors acknowledge this, more analysis of failure modes would be valuable.
2. Hyperparameter dependencies: Multiple design choices affect performance (key/non-key ratio ~2%, significant period ≥5 frames, threshold η_c). The automatic adaptability to different scene characteristics appears limited, and more ablation studies on these choices would strengthen the paper.
3. Potential graph connectivity issues: The uncertainty-weighted edge formation may exclude low-uncertainty nodes, potentially missing important scene structures that are simply less frequently observed. This could lead to incomplete propagation in certain scenarios.
4. More fundamentally, this approach appears inherently limited to single-object, near-rigid scenarios. The core assumption—that frequently observed Gaussians serve as stable motion anchors—explains why results are strong on object-centric scenes like "paper-windmill," "spin," and Objaverse datasets, but raises concerns about applicability to scenes with independent motions or highly non-rigid deformations. The authors should explicitly acknowledge this as a scope constraint and provide analysis beyond curated object-centric benchmarks.

### Questions
- What modifications would be needed to handle scenes that don't fit the single-object, near-rigid assumption?
- Are there quantitative metrics that could predict when the method will fail based on scene characteristics (e.g., rigidity, object count, motion complexity)?

### Soundness
3

### Presentation
3

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
This work introduces USplat4D, an uncertainty-aware dynamic Gaussian Splatting framework for monocular 4D scene reconstruction. The key idea is to explicitly estimate per-Gaussian, time-varying uncertainty and leverage it to organize Gaussians into a spatio-temporal graph. This uncertainty-aware graph informs the selection of reliable (“key”) and less reliable (“non-key”) nodes, guiding motion propagation and adaptive optimization for more robust geometry and novel view synthesis, especially under occlusions and extreme viewpoints. Experiments on several real and synthetic datasets demonstrate superior consistency, quality, and robustness versus state-of-the-art baselines in both quantitative and qualitative settings.

### Strengths
Strength:
1. The paper is well-written and easy to follow.
2. Usplat achieve good result against recent baselines.

### Weaknesses
Major Weakness:
1. The overall pipeline involves multiple stages and components, baseline reconstruction model Mosca is already complex, USplat makes it appear rather complex and add time complexity.
2. Usplat ommit a evaluation on the nvidia dynamic dataset.

Minor Weakness:
1. In line 60, The should be "the"

### Questions
1. How does the choice of the threshold $ \eta_c $ in Eq. 7, or the scaling constants $ (r_x, r_y, r_z) $ in Eq. 9, affect downstream geometry and rendering quality? How is the method robust to different settings? 
2. Can you provide more analysis on the time consumption.

### Soundness
3

### Presentation
3

### Contribution
3
