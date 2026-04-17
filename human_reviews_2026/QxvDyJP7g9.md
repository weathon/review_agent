# ARTDECO: Toward High-Fidelity On-the-Fly Reconstruction with Hierarchical Gaussian Structure and Feed-Forward Guidance

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
On-the-fly 3D reconstruction from monocular image sequences is a long-standing challenge in computer vision, critical for applications such as real-to-sim, AR/VR, and robotics. Existing methods face a major tradeoff: per-scene optimization yields high fidelity but is computationally expensive, whereas feed-forward foundation models enable real-time inference but struggle with accuracy and robustness. In this work, we propose ARTDECO, a unified framework that combines the efficiency of feed-forward models with the reliability of SLAM-based pipelines. ARTDECO uses 3D foundation models for pose estimation and point prediction, coupled with a Gaussian decoder that transforms multi-scale features into structured 3D Gaussians. To sustain both fidelity and efficiency at scale, we design a hierarchical Gaussian representation with a LoD-aware rendering strategy, which improves rendering fidelity while reducing redundancy. Experiments on eight diverse indoor and outdoor benchmarks show that ARTDECO delivers interactive performance comparable to SLAM, robustness similar to feed-forward systems, and reconstruction quality close to per-scene optimization, providing a practical path toward on-the-fly digitization of real-world environments with both accurate geometry and high visual fidelity. Project page: https://city-super.github.io/artdeco/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a unified framework for on-the-fly, high-fidelity 3D reconstruction from monocular image sequences. It addresses the longstanding tradeoff between efficiency, accuracy, and robustness in real-time 3D scene capture by combining the strengths of feed-forward 3D foundation models and SLAM-style geometric optimization, using 3D Gaussian Splatting (3DGS) as the scene representation.

### Strengths
The paper is clearly written and well-structured. The problem statement (monocular on-the-fly reconstruction) and core tradeoff (efficiency vs. accuracy vs. robustness) are crisply articulated in the introduction. Figures 2 and 3 provide intuitive visual summaries of the pipeline and LoD mechanism, respectively.

### Weaknesses
1. The paper describes three modules (frontend, backend, mapping) but lacks clarity on how tightly coupled they are and whether components can be independently replaced or ablated. For example: 
  - The frontend uses MASt3R for pose estimation, but the backend uses π3 for loop closure. Why not use $\pi_3$ in both? The ablation (Table 3) shows $\pi_3$ underperforms in pose estimation, but the paper doesn’t explain why this is the case (e.g., is it due to training data, architecture, or metric scale ambiguity?).
  - The mapping module relies on pointmaps from the backend, yet it’s unclear how errors in pose or loop closure propagate into Gaussian initialization.

2. The paper compares against LongSplat and OnTheFly-NVS, but omits very recent pose-free methods like AnySplat or No Pose, No Problem in key metrics. More critically, the evaluation protocol (every 8th frame held out) is applied retroactively to baselines, but it’s unclear if those methods were designed for such a protocol. For example: 
 - Feed-forward methods often assume all input frames are available at once; streaming evaluation may disadvantage them unfairly.
 - Conversely, SLAM methods like MASt3R-SLAM are inherently streaming—so the comparison may favor ARTDECO by design.

3. The core claim is that ARTDECO “combines the efficiency of feed-forward models with the reliability of SLAM.” However, the paper does not quantify how much of the performance gain comes from: 
 - The foundation models (MASt3R/$\pi_3$),
 - The LoD Gaussian representation, 
 - The SLAM-style optimization (BA, loop closure).

The ablation in Table 3 is helpful but incomplete. For instance: 
 - There is no ablation removing both loop closure and mapper frames.
 - The “w/o level-of-detail” ablation shows a PSNR drop (~1 dB), but it’s unclear whether this is due to rendering artifacts, geometry errors, or memory inefficiency.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes ARTDECO, a unified online reconstruction system that integrates the feed-forward 3D model with a real-time pose estimation and loop-closure pipeline. It converts multi-view features into a hierarchical level-of-details (LoD) Gaussian scene representation and continuously refines it using all frames. This integration strikes a balance between real-time performance and reconstruction fidelity, delivering globally consistent 3D scenes with quality comparable to offline optimization.

### Strengths
- The paper is well structured and easy to follow. The appendix provides more detailed experimental results and implementation details.

- The work demonstrates a coherent integration of pretrained feed-forward 3D models, pose estimation, and Gaussian-based mapping into a unified online system, validated on both indoor and outdoor benchmarks.

- The hierarchical Gaussian representation and LoD-aware rendering effectively balance reconstruction fidelity and computational efficiency, achieving practical real-time performance.

### Weaknesses
-The engineering efforts of the proposed system is appreciated; however, the system primarily integrates existing components/concepts from previous studies, i.e., composing feed-forward 3D foundation models (MASt3R-SLAM), Gauss–Newton BA, and hierarchical Gaussian splatting, into a single pipeline. The following points offer specific comments.

- The proposed front-end tracking system is built upon MASt3R-SLAM.  It switches to a pinhole camera model + reprojection residual that is a standard bundle-adjustment design in traditional SLAM systems. This is less general than MASt3R-SLAM’s generic central camera + ray-angle residual (which is explicitly designed to be robust to depth noise and varying intrinsics). The paper does not provide analysis of failure modes under non-central cameras, zoom changes, or intrinsic effect (e.g., rolling-shutter effects), nor evidence that the chosen residual yields measurable advantages over ray-angle residuals.

- LoD introduced without proper reference and sounds incremental w.r.t. previous structured/multi-level representations.
Multiple previous works have introduced hierarchical/structured scene representations and explicit LoD mechanisms (e.g., Scaffold-GS, Hierarchical 3D Gaussians, Octree-GS, Mip-Splatting, FLoD). However, at the first occurrence of “LoD” (at Line 73), there is no clear citation for this concept in this literature. E.g., what is the difference(s) of the proposed LoD with the same concept introduced in FLoD/Octree-GS? The proposed method mainly implements a 2D patch-budget–driven coarse-to-fine scheduling (e.g., per-level pixel coverage, fixed per-level scale growth like $1.4^{2l}$, and a distance-based gating with blending). While these choices are practical, they are engineering heuristics rather than a principled spatial hierarchy. Compared to methods that anchor hierarchy in 3D space/geometry (anchors/octrees/mip anti-aliasing), the proposed 2D scheduling is arguably less principled for large-scale geometry control and view-consistent refinement/anti-aliasing. The paper does not provide a sensitivity analysis (e.g., to the 1.4 scale law, pixel-budget per level, or $d_{max}$) or an ablation against established 3D-anchored LoD schemes.

- The proposed Gaussian initialization and densification seems like reusing established solutions. The method relies on residual-driven insertion to densify Gaussians, and a decoder to map multi-scale features to structured Gaussians (including scale/rotation regularization). These mirror previous practices such as base-scale setting and anisotropic covariance regularization (e.g., Meuleman et al; Wu et al, as cited).

- Loop closure via a foundation 3D model (Pi3) is not conceptually new and appears to be a dominant factor. Using a foundation model for loop detection is not new (e.g., CLIP-based loop closure has been shown for monocular Gaussian SLAM, Tian et al). Here, Pi3 is simply plugged in as the retriever. The ablation indicates that replacing Pi3 with VGGT degrades accuracy, implying the BA stack is heavily dependent on the chosen foundation model rather than on the SLAM backend system itself. Moreover, although the paper mentions memory concerns of baselines (at Line 70), it does not report the compute/memory cost of employing Pi3 (or alternative foundation models) in the loop-closure module.

- Please check the references; there are multiple duplicates, e.g., Longsplat: Robust unposed 3d gaussian splatting for casual long videos, On-thefly reconstruction for large-scale novel view synthesis from unposed images

References:

Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bernhard Kerbl, and George Drettakis. Onthe-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images. ACM Transactions on Graphics, 44(4), 2025b.

Songyin Wu, Zhaoyang Lv, Yufeng Zhu, Duncan Frost, Zhengqin Li, Ling-Qi Yan, Carl Ren,
Richard Newcombe, and Zhao Dong. Monocular online reconstruction with enhanced detail
preservation

Lan, Tian, Qinwei Lin, and Haoqian Wang. "Monocular gaussian slam with language extended loop closure." arXiv preprint arXiv:2405.13748 (2024).

### Questions
Please refer to the questions outlined in the weakness section.

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
I am not an expert in 3d vision and reconstruction, so the following comments and questions could be based on potentially inaccurate understandings or biased perspectives. From my understanding, this paper presents a system for efficient and high-fidelity 3D reconstruction performed "on-the-fly" from monocular image sequences. The method utilizes a \textit{structured scene representation} to achieve both speed and quality, demonstrating competitive performance against state-of-the-art SLAM and reconstruction approaches like DROID-SLAM and Go-SLAM regarding metrics like ATE and memory consumption. The goal is to deliver interactive, robust 3D reconstruction in real-time scenarios.

### Strengths
1. Achieves a desirable balance between reconstruction fidelity (low ATE/RTE, good PSNR) and computational efficiency (low time and memory footprint), crucial for real-time mobile applications. 

2. The system's design is robust for on-the-fly operation and across diverse scene types (indoor and outdoor). The results show strong quantitative improvements or parity with several established recent baselines.

### Weaknesses
1. The abstract and snippets are light on the explicit technical details of the "structured scene representation." Without knowing the specific structure (e.g., hash grid, implicit surface, etc.), it is difficult to judge the approach's novelty and engineering complexity. 

2. The comparison in the snippet only covers a limited subset of recent SLAM/Reconstruction methods; a more comprehensive benchmark, including performance on texture-less regions or under extreme motion, would be beneficial.

### Questions
Could the authors elaborate on the specific architecture of the "structured scene representation" and quantify how its design choices (e.g., sparsity, resolution, memory layout) explicitly contribute to the claimed efficiency gain compared to unstructured or purely implicit representations?

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
3

### Summary
ARTDECO provide a framework for real-time 3D reconstruction from monocular image sequences. ARTDECO aims to bridge the gap between high-fidelity per-scene optimization and feed-forward foundation models. ARTDECO achieves this by combining the efficiency of feed-forward inference with the accuracy and consistency of SLAM-based optimization, making it suitable for AR/VR, robotics, and digital twin applications.

### Strengths
Strength:
1. The paper is well-written and easy to follow.
2. ARTDECO achieve good result on the serveral benchmarks against recent baselines, providing a solid solution for 3D reconstruction

### Weaknesses
Major Weakness:
1. The overall pipeline involves multiple stages and components, making it appear rather complex.

Minor Weakness:
1. In line 176, “front-end” should be changed to “frontend” for consistency with the rest of the paper.

### Questions
Question:
1. How does ARTDECO compare with simpler baselines, such as $\pi^{3}$ combined with bundle adjustment?
2. In line 313, should the formula $\alpha = (d_r - d_{max})/ (d_{max})$ given that it occurs in the rendering process?
3. Can you provide more details on the ablation studies for mapper frames, level of detail, and structural Gaussians regarding the baseline and variant designs used?

### Soundness
3

### Presentation
3

### Contribution
2
