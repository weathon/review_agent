# Joint Optimization for 4D Human-Scene Reconstruction in the Wild

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Reconstructing human motion and its surrounding environment is crucial for understanding human-scene interaction and predicting human movements in the scene. While much progress has been made in capturing human-scene interaction in constrained environments, those prior methods can hardly reconstruct the natural and diverse human motion and scene context from web videos. In this work, we propose JOSH, a novel optimization-based method for 4D human-scene reconstruction in the wild from monocular videos. Compared to prior works that perform separate optimization of the human, the camera, and the scene, JOSH leverages the human-scene contact constraints to jointly optimize all parameters in a single stage. Experiment results demonstrate that JOSH significantly improves 4D human-scene reconstruction, global human motion estimation, and dense scene reconstruction by utilizing the joint optimization of scene geometry, human motion, and camera poses. Further studies show that JOSH can enable scalable training of end-to-end global human motion models on extensive web data, highlighting its robustness and generalizability. The code and model are available at [https://vail-ucla.github.io/JOSH/](https://vail-ucla.github.io/JOSH/]).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes JOSH, a unified optimization framework for 4D human scene reconstruction in the wild from monocular video. This method introduces a human-scene contact loss, which enforces physical plausibility and consistency between the human scene and the surrounding environment. Several experiments on the SLOPER4D, EMDB, and RICH datasets demonstrate that JOSH significantly improves global human motion estimation, dense scene reconstruction, and physical plausibility metrics compared to previous methods (e.g., SynCHMR, WHAM, and TRAM). Another interesting network is JOSH3R, an end-to-end model that generates scalable pseudo-labels for large-scale web video training.

### Strengths
1. Josh appears to have a lot of competing work these days, but in terms of timeline, Josh is indeed the first to simultaneously optimize camera pose, human motion, and scene geometry using a joint optimization framework.
2. Josh3R is a good model that demonstrates the feasibility of automatically labeling real web videos for large-scale model training.
3. Josh's extensive experiments on multiple datasets (SLOPER4D, EMDB, and RICH) validate the effectiveness of his approach, which I highly recommend.

### Weaknesses
1. The overall visualization and experimental results (WA-MPJPE, MPJPE) are acceptable. However, I observed that even with the foot contact loss, there remains some penetration in the visualization (the 41-second mark in the supplementary video). Additionally, given that your work builds on the Rich model, why not consider adding hand penetration constraints? For instance, the first two visualizations in the supplementary video exhibit significant hand penetration.  
2. The method relies heavily on the initialization model. While monocular camera scene reconstruction methods are advancing rapidly, JOSH still adopts MASt3R. Also, key experiments comparing with this year’s methods (e.g., VGGT, CUT3R) are missing. Including these experiments would better validate the work’s performance.  
3. Although optimization-based methods are effective, they are computationally costlier than purely learning-based methods. It would be valuable to see experiments on time complexity to evaluate JOSH’s computational performance.  
4. Theoretically, JOSH3R is capable of real-time performance. I would appreciate seeing results for JOSH3R—both qualitative visualizations and quantitative metrics.

### Questions
Please clarify why penetration remains despite foot contact loss and discuss the potential of adding hand penetration constraints. Provide results with other new monocular camera scene reconstruction methods. Add comparisons with recent methods (e.g., VGGT, CUT3R). Provide JOSH’s time complexity data and JOSH3R’s qualitative/quantitative results to validate real-time capability.

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
4

### Summary
This paper proposes a single-stage optimization framework that jointly estimates camera intrinsics/extrinsics, global multi-human motion (SMPL), and dense scene geometry from a in-the-wild monocular video. The key novelty is to explicitly enforce human–scene contact through two losses: (i) a contact-scene loss that pulls predicted human contact vertices toward matched scene points, and (ii) a contact-static loss that discourages contact sliding across time. The method also optimizes focal length jointly with root depth to correct metric-scale drift common in monocular setups. On SLOPER4D/EMDB/RICH, JOSH variants (with different off-the-shelf initializers) improve human trajectory errors, scene Chamfer distance, and physical plausibility (jitter, foot sliding) over strong baselines and a re-implementation of SynCHMR. They further show that pseudo-labels produced by JOSH enable scalable training of an end-to-end model that outperforms training on limited GT.

### Strengths
1. Clean contact formulation. Explicit vertex–scene distance with visibility + depth-prior gating + a temporal static contact term that mathematically targets sliding, which is practical and easy to implement without a simulator.
2. Focal-length optimization tied to root depth, addressing metric-scale failures when intrinsics are unknown, high leverage for web videos.
3. General wrapper. Boosts multiple scene/human initializers; evaluation spans human, scene, and physics plausibility metrics.

### Weaknesses
1. No force/stability reasoning. JOSH’s contacts are geometric (distance/static) without force, friction cone; expect residual artifacts under occlusion or weak depth priors (e.g., hand-on-sofa, compliant supports).
2. Contact detection & masking sensitivity. JOSH assumes reliable contact vertices and segmentation; occlusion/noisy masks may yield wrong correspondences, and the method’s robustness to such errors isn’t deeply quantified.
3. Dynamic scenes / non-static supports. geometric matching in JOSH presumes a static background for scene correspondences; cannot handle transient contacts and near-contacts through stability/dynamics.

### Questions
1. Evaluate foot-skate error, contact F1, and COM-support margin like PPR do.
2. Compare results with fixed f versus optimized f to show how intrinsics optimization resolves scale drift relative to SynCHMR.
3. Include contact visulizations with and without contact modeling.
4. Randomly perturb 20 % of BSTRO contact labels or depth priors; measure degradation in WA-MPJPE and sliding metrics to show robustness of Lc1/Lc2.
5. Provide inter-person collision or contact consistency metrics to substantiate multi-human capability.

I will reconsider my recommendation if the authors could kindly address these points.

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
4

### Summary
This paper proposes JOSH, an approach for reconstructing the 4D human motion and the 3D environment from a single video. The paper uses initialization information from off-the-shelf models and jointly optimizes the human motion and the environment. This is done by designing carefully the optimization objectives for this joint optimization. The complete system is benchmarked in different datasets and settings demonstrating strong quantitative performance.

### Strengths
- The proposed method achieves strong quantitative performance across the board. The paper provides comparisons with multiple baselines on a few benchmarks and JOSH outperform previous works in the majority of cases.
- Integrating signals for human contact is not easy, because these estimates are very noisy. The paper does this integration carefully, so that JOSH can benefit from the noisy predictions of the off the shelf systems.
- There is a helpful ablation that considers different aspects of the algorithm and shows the effect they have in the final result.
- The paper also introduces the JOSH3R model and trains it with pseudo-labels from running JOSH on in the wild videos. Although JOSH3R is not the main focus of the paper, it is a nice addition and shows the benefit of using JOSH for generating pseudo ground truth.

### Weaknesses
- The proposed approach is relatively straightforward for the most part. This type of optimizations are more traditional (e.g., Rempe et al, ICCV 2021 & Ye et al, CVPR 2023), so integrating better initial estimates from off-the-shelf models or refining the optimization objectives will often lead to better results.
- The use of human-scene contact relies on the contact being visible in the video. Such contact is often available in the simpler benchmark datasets used for evaluation but less common in in-the-wild videos. To be fair, this is something recognized by the paper as one of its weaknesses.
- The supplementary video does not provide a lot of results for in-the-wild videos. There are a few in the end, but they tend to be on the simpler side when it comes to the scene and/or the camera motion.

### Questions
- I appreciate the demonstration of using different off-the-shelf models for initializing the camera and human motion in Table 1 (e.g., VIMO, WHAM, etc), but it would be helpful to see some direct comparisons. I understand it might not be easy to do all the combinations, but how does VIMO+MonST3R compares with VIMO+MASt3R?
- I was underwhelmed from the demonstration on in-the-wild videos. It would be helpful to see results on more videos and more challenging ones. For example, how does the method performs on videos from the PoseTrack dataset?

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
5

### Summary
- The paper tackles joint scene and human-mesh reconstruction from video, with a focus on modeling human–scene contacts.
- The proposed approach, JOSH, is an optimization-based method with two stages: (1) initialize point maps, correspondences, human meshes, and contact labels using off-the-shelf models; (2) jointly optimize the scene point cloud, camera parameters, and mesh parameters using priors and contact losses.
- The method introduces heuristics and losses—static-contact loss, scene-contact loss, and focal-length optimization—to accurately model the camera and humans in contact with the scene.
- The evaluations are done on multiple datasets: SLOPER4D, EMDB and RICH.
- Key Baselines: SLAHMR, WHAM, TRAM.
- The quantative results show that JOSH consistently outperforms the baselines across various scenarios.

### Strengths
- The paper is well written, organized and easy to follow. The ideas are presented clearly and the introduction section particularly is an insightful read. 

- Joint human-scene modeling: The problem of jointly modeling scenes and humans is ambitious and forward-looking. The proposed approach, JOSH, is promising, shows strong in-the-wild results, and should encourage the community to design integrated human–scene methods.

- Simplicity of the approach: The core is simple—and that is its strength: initialize correctly and optimize reliably. This offers a promising path to obtain pseudo-3D (human+scene) supervision at scale from web videos. Although the supervision is noisy (compared to capture studios or synthetic), it remains to be seen how learning-based approaches (e.g., JOSH3R) generalize when trained at scale.

- Strong results: The qualitative results in the manuscript (and supplemental videos) clearly show the performance gap between JOSH and baselines such as TRAM and WHAM. The ability to model contacts over long durations without drift is impressive. Quantitatively, it outperforms both human- and scene-reconstruction baselines (MonST3R, MASt3R) across datasets.

### Weaknesses
- Limited Technical Novelty (without JOSH3R): I am suprised that JOSH3R's details (the learning based module built on top of the data collected using JOSH) is relegated to the appendix. Without JOSH3R as a core contribution, JOSH remains an optimization-based method for joint human–scene reconstruction; it can at best provide pseudo labels (although viable at scale). Standalone, it offers limited new technical insight. The optimization procedures largely mirror those in Hi4D (CVPR 2023), Ego-Exo4D (CVPR 2024), SLAMHR (CVPR 2023) with the key difference that the scene is included in the optimization. I acknowledge that incorporating the scene is nontrivial and requires precise execution at scale; however, this does not substantially increase the technical novelty. I strongly encourage including JOSH3R in the main paper narrative.

- Reliance on off-the-shelf initializations: The method acknowledges it can “suffer from bad initializations.” In practice, it stitches VIMO for humans and MASt3R for scenes, so failures propagate into JOSH. The optimization terms can refine but cannot recover from catastrophic errors.

- Comparison with Human only Methods on EMDB: The paper evaluates global human motion on EMDB, while other human/scene metrics are reported on SLOPER4D (less common in HMR). For a fair comparison to human-only methods, please report MPJPE (not world-MPJPE) on EMDB1 (see Table 3 of EMDB) to assess JOSH’s pose-alignment performance post-optimization against state-of-the-art human-only baselines.

### Questions
My questions are primarily centered around the weaknesses mentioned above:

1. Including JOSH3R in the main narrative. If not, justifications for novelty without it.
2. Robustness to poor initializations.
3. Pose-alignment comparison on EMDB-1.

### Soundness
3

### Presentation
2

### Contribution
2
