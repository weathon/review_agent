# EA3D: Event-Augmented 3D Diffusion for Generalizable Novel View Synthesis

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 8

## Abstract
We introduce **EA3D**, an Event-Augmented 3D Diffusion framework for generalizable novel view synthesis from event streams and sparse RGB inputs.  Existing approaches either rely solely on RGB frames for generalizable synthesis, which limits their robustness under rapid camera motion, or require per-scene optimization to exploit event data, undermining scalability.
EA3D addresses these limitations by jointly leveraging the complementary strengths of asynchronous events and RGB imagery. 
At its core lies a learnable EA-Renderer, which constructs view-dependent 3D features within target camera frustums by fusing appearance cues from RGB frames with geometric structure extracted from adaptively sliced event voxels. 
These features condition a 3D-informed diffusion model, enabling high-fidelity and temporally consistent novel view generation along arbitrary camera trajectories. 
To further enhance scalability and generalization, we develop the Event-DL3DV dataset, a large-scale 3D benchmark pairing diverse synthetic event streams with photorealistic multi-view RGB images and depth maps. 
Extensive experiments on both real-world and synthetic event data demonstrate that EA3D consistently outperforms optimization-based and generalizable baselines, achieving superior fidelity and cross-scene generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes EA3D, a generalizable event-augmented 3D diffusion framework for novel view synthesis from sparse RGB frames plus continuous event streams. An EA-Renderer fuses posed RGB appearance features with geometry features extracted from adaptively sliced event voxel grids via cross-attention, producing view-conditioned 3D features that condition a CogVideoX-style 3D-aware diffusion model. The authors also curate Event-DL3DV, pairing synthetic event streams with multi-view RGB and depth. Experiments on DL3DV and Tanks-and-Temples with 2/4/6 views, plus real events from DSEC, report consistent gains over optimization-based event methods (E-NeRF, Event3DGS) and RGB-only generalizable methods (ViewCrafter, NVS-Solver). Ablations indicate benefits from event-derived geometry, adaptive slicing, and a feature reconstruction loss.

### Strengths
1.The authors purposed really clear and robust architecture. EA-Renderer elegantly aligns pose-free event geometry with pose-conditioned RGB appearance features in camera trajectory space, directly addressing occlusion and large-baseline challenges. The proposed Event-DL3DV uses randomized thresholds/resolutions to diversify events and adds blur augmentation and temporal reversal to enhance robustness and trajectory generalization.

2.The authors systematically isolate the effects of event geometry, adaptive slicing, and reconstruction loss, showing that event features significantly stabilize NVS under extended baselines.

3.They report inference and memory comparisons with E-NeRF, Event3DGS, NVS-Solver, and ViewCrafter, showing clear runtime advantages.

### Weaknesses
1.The paper includes both event-based (E-NeRF, Event3DGS) and non-event diffusion-based (ViewCrafter, NVS-Solver) baselines, which is good coverage. However, it would further strengthen the empirical section to discuss or include newer event-augmented Gaussian frameworks. Considering that many latest works have not yet released code, this omission is understandable and not a fatal limitation, but at least a discussion of such methods would provide better context for positioning EA3D among concurrent developments.

2.While the proposed “geometry feature from event streams” introduces an interesting idea of leveraging voxelized event information to encode geometric continuity, the paper does not elaborate on its structure or behavior in sufficient depth. The feature appears to be embedded within the event encoder of EA-Renderer but lacks standalone analysis, visualization, or theoretical justification. Consequently, although this feature may contribute meaningfully to performance gains (as suggested by Table 3), its novelty and generalizability remain underexplored. A more detailed study would greatly strengthen this contribution.

3.Because the paper does not fully characterize the geometry feature or its underlying mechanism, the overall framework risks being perceived as a system-level composition of existing modules (event voxel encoding, cross-attention fusion, and diffusion-based rendering) rather than a fundamentally new algorithmic contribution. The work would benefit from a clearer articulation of what specific modeling insight or formulation distinguishes EA3D from prior event-guided 3D reconstruction methods.

### Questions
The paper introduces several key components, such as the geometry feature extracted from event streams, the adaptive slicing strategy within the event encoder, and the 3D-aware diffusion model. However, these are all described only briefly. Could the authors provide more detailed explanations, figures, or analyses to clarify how these modules operate and how they differ from existing event-based voxel encoders or diffusion frameworks?

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
3

### Summary
The paper introduces EA3D, a framework for generalizable novel view synthesis that uses sparse RGB frames and continuous event streams as input. It proposes a Event-Augmented Feature Renderer module, which is a learnable module that constructs 3D features for the target camera path. It extracts appearance cues from the sparse RGB frames and captures geometric structure from estimated camera parameters and depths. The 3D features from the EA-Renderer are used to condition a video diffusion model (based on CogVideoX) for novel view synthesis.

### Strengths
1. Extensive experiments: The paper provides comprehensive comparisons with baselines, investigates the impact of varying the number of input samples, and includes relevant ablation studies on model design.
2. The creation of the large-scale Event-DL3DV dataset would be a valuable contribution if it is made open source.
3. The proposed method achieves noticeable improvements in both visual quality and quantitative metrics.

### Weaknesses
1. Reliance on Upstream Models: The EA-Renderer requires camera parameters and depth maps for the input RGB frames, which are obtained from an "off-the-shelf multi-view stereo model". This means EA3D is not fully end-to-end, and its performance is dependent on the accuracy of this prerequisite model.
2. The idea of incorporating explicit 3D information into diffusion models has been explored in many previous works, and this paper does not present particularly novel aspects in this regard.

### Questions
In Table 4, "Computation cost comparison with the baselines," does the reported computation cost include the time required to estimate camera parameters and depths using an off-the-shelf multi-view stereo model?

### Soundness
3

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
The paper proposes an event-augmented framework for novel view synthesis. The core idea is to leverage event cameras to provide complementary information to strengthen the overall synthesis process, especially in scenarios with rapid camera movements. The framework consists of two main components: an event-augmented (EA) feature renderer and a feature-conditioned video diffusion model. The EA feature renderer fuses event information with RGB features using cross-attention in the 3D space, creating a strong feature prior for subsequence generation process. The diffusion model, adapted from CogVideoX, is then conditioned on these features to synthesize novel views in video sequence. Experiments demonstrate the advantages of adopting event features to boost the performance of novel view synthesis.

### Strengths
* The core idea of leveraging event cameras to provide complementary information for novel view synthesis is both novel and compelling.
* The proposed framework is straightforward and effective. The experiments clearly demonstrate the benefits of incorporating event data.
* The paper is well-written, clearly structured, and easy to understand

### Weaknesses
* A central claim of the paper is that event data is particularly helpful for fast camera movements. However, the experimental section seems to lack a direct evaluation to support this specific claim. It would significantly strengthen the paper to include an analysis comparing performance under fast camera motion.
* As event data has strong correlation with camera trajectories, I am curious about the model's generalization capabilities under different camera trajectories. How does the method perform when tested with camera trajectories that are different from those seen during training (e.g., different paths or motion patterns)? A discussion on this aspect would be very insightful.

### Questions
N.A.

### Soundness
2

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
3

### Summary
The paper proposes a generalizable novel view synthesis framework that fuses event streams and RGB images to achieve photorealistic rendering under fast camera motion. The method extracts appearance features from RGB frames using a multi-view stereo model, encodes geometry-aware event features through voxel-based event representation, and integrates them via attention module. Experimental results demonstrate that the proposed fusion module effectively achieves its intended goal and enables the framework to reach state-of-the-art performance in novel view synthesis.

### Strengths
1. This paper is the first to propose a generalizable novel view synthesis framework based on event camera,. The overall network design is clear, simple, and effective, leading to strong performance.

2. The use of 3D features to condition the DiT appears both effective and novel, offering a promising direction for improving 3D-aware generation.

3. The final results are quite impressive in term of extremely sparse input conditions, the proposed method even outperforms per-scene optimization approaches. 

4. Adaptive Slicing is an effectively strategy for encoding event feature.

5.The paper is well written, clearly structured, and easy to follow

### Weaknesses
1. The ablation study appears unfair to me, as the authors leverage VGGT + video diffusion, which could be the primary factor contributing to the performance improvement. From my perspective, as shown in works like FlowR [1], combining geometry foundation model + multi-view diffusion can already yield strong performance. However, the ablation study in this paper does not clearly demonstrate that the event information is fully utilized or that it significantly contributes to the performance gains.

2. From Table 1, the performance improvement becomes less significant when the number of input views increases (e.g., to six views), indicating that the method’s advantage is mainly evident in extremely sparse-view settings.

[1] Fischer, T., Bulò, S. R., Yang, Y. H., Keetha, N., Porzi, L., Müller, N., ... & Kontschieder, P. (2025). Flowr: Flowing from sparse to dense 3d reconstructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 27702-27712).

### Questions
1. How did you align the VGGT poses to the coordinate system used in your evaluation? I find this part a bit confusing, and I have some concerns about the fairness of the evaluation. If the authors could clarify this alignment process, I would consider increasing my score.

2. Would you mind add one more ablation about without event features? Which will help us to understand why we need to do generalizable event-based novel view synthesis.

3.Is your evaluation performed only between the input views, or over the entire scene including unseen regions? The results are surprisingly strong, so I’m concerned that the evaluation might be limited to the input-view range rather than the full scene.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a framework for generalizable novel view synthesis that uses a set of sparse RGB frames and a continuous event stream as input  to generate novel RGB views. The method conditions a video generative model (a Diffusion Transformer based on CogVideoX)  on 3D features. These 3D features are constructed by an "EA-Renderer," which uses an off-the-shelf 3D reconstruction method (VGGT) to get posed appearance features, and then enriches these features with geometric information from event-based voxels. Since poses for the event camera are not available , the method learns a feature fusion module (using cross-attention)  to integrate the un-posed event features into the posed 3D features. The paper provides comparisons on real-world and synthetic datasets against both RGB-only 3D generative models and optimization-based, event-camera-specific methods. The ablation study provides a much-needed demonstration of performance degradation when the event-based stream is removed.

Despite the technical contribution being somewhat incremental (combining existing MVS, video diffusion, and attention mechanisms), the application to event-augmented novel view synthesis is valuable and I am arguing for acceptance. However, I would strongly encourage the authors to address the weaknesses below -- the paper would be far more convincing if the authors added experiments on challenging scenes with real fast motion (e.g. fast-moving objects or drone footage) to truly demonstrate the advantage of using event cameras over standard generative 3D models.

### Strengths
- Important, interesting, and timely problem: The paper addresses the challenging problem of novel view synthesis from sparse inputs under fast camera motion -- a scenario where event cameras offer a clear theoretical advantage over standard RGB-only methods and where existing 3D generative model research hasn't focused on .
- Fusion Strategy, though obvious choice and computationally intensive, is an interesting technical contribution. The proposed architecture for fusing posed RGB features with un-posed event features via a cross-attention mechanism  is a logical way to combine these complementary data sources. 
- Effective Ablations: The ablation studies in Section 4.3 clearly validate the core contribution, and in particular, the main ablation shows that incorporating event-based geometry features significantly improves results, especially as the baseline (view range) between input frames increases.

### Weaknesses
- Dataset Limitations: My main concern is that the evaluation datasets do not fully demonstrate the scenarios where event cameras are organically needed. The paper uses "static sequences" from the real-world DSEC dataset , while other benchmarks (DL3DV, T&T) rely on simulated events. Most motion blur is artificially induced. The paper would be significantly stronger if it outperformed existing 3D generative models in datasets where event cameras are truly necessary (e.g. fast drone videos, rapid human motion).
- Missing Comparisons: The paper does not compare against EGVD (Zhang et al., 2025) or (Chen et al., 2024a). While the authors argue these methods are for interpolation along the trajectory, the evaluations in this paper also appear to synthesize views near the original camera path. A comparison seems feasible and would benefit to fully evaluate the approach.
- Minor: The method is heavily reliant on an off-the-shelf pretrained multi-view 3D reconstruction method (VGGT)  to extract the initial 3D point cloud and posed features.
- Minor: Inaccuracies in Related Work (Sec 2.2): The related work section needs revision for accuracy. 1) The claim that ReconFusion / ZeroNVS "struggles with pose control... due to implicit pose encoding"  is incorrect as there are explicit 3D method. 2) The use of "3D-aware diffusion model"  is confusing. This term typically refers to models that are 3D-consistent by design (e.g. by incorporating a renderer in the denoising loop, like EG3D DMV3D and RenderDiffusion, which haven't been cited), which is not the case here. Therefore, the claim "To ensure 3D consistency"  should be weakened to "To encourage 3D consistency," as a video diffusion model does not guarantee this. 3) The claim that other video diffusion models "degrade when the input views exhibit large viewpoint variations" is a weak argument; methods like CAT3D perform well from very sparse views. The primary argument against them should be their inability to leverage event data.

### Questions
- The feature fusion module performs cross-attention between each appearance feature and the entire event feature set. How does the computational cost of this fusion scale, and does it limit the practical length of the event streams you can process?
- How robust is the method to failures in the initial MVS (VGGT) step, especially in texture-less regions or under extreme motion?

### Soundness
3

### Presentation
3

### Contribution
2
