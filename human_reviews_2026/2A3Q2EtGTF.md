# Secondary Motion-Aware 3D Clothed Gaussian Avatars from Monocular Videos

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled animatable 3D human avatars from single videos with efficient rendering and high fidelity. However, current methods struggle with dynamic appearances, especially in loose garments (e.g., skirts), causing unrealistic cloth motion and needle artifacts. This paper introduces a novel approach to dynamic appearance modeling for 3DGS-based avatars, focusing on loose clothing. We identify two key challenges: (1) limited Gaussian deformation under pre-defined template articulation, and (2) a mismatch between body-template assumptions and the geometry of loose apparel. To address these issues, we propose a motion-aware autoregressive structural deformation framework for Gaussians. We structure Gaussians into an approximate graph and recursively predict structure-preserving updates, yielding realistic, template-free cloth dynamics. Our framework enables robust dynamic appearance modeling under the single-view constraint, producing accurate foreground silhouettes and precise alignment of Gaussian points with clothed shapes. To demonstrate the effectiveness of our method, we introduce an evaluation dataset featuring subjects performing dynamic movements in loose clothing, and extensive experiments validate that our approach significantly outperforms existing 3DGS-based methods in modeling dynamic appearances from monocular videos.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel, two-stage framework for creating animatable 3D Gaussian Splatting (3DGS) avatars from a single monocular video, with a specific focus on realistically modeling the "secondary motion" of loose-fitting garments. The authors identify two key failings of prior work: (1) poor initialization of Gaussians due to a mismatch between "naked body" templates and clothed geometry, and (2) an inability to model complex cloth dynamics due to deformation models that lack temporal context.

To solve this, this paper proposes:
1)Personalized Gaussian Initialization (PGI): A pre-processing stage that first trains a 4D NeRF to create a canonical, clothed representation of the subject, from which the initial 3D Gaussians are extracted. This ensures the Gaussians' starting positions match the actual clothed shape.

2)Secondary Motion-Aware Deformation (SMAD): A novel deformation model that represents the canonical Gaussians as a graph. A GNN then autoregressively predicts the position and velocity of the graph nodes, inspired by a second-order mass-spring-damper system. By explicitly encoding a buffer of past velocities, this model captures the temporal context crucial for realistic cloth motion.

This paper validates the method against several recent 3DGS-avatar baselines on three datasets, including the new "LoCo-Human" in-the-wild dataset they introduce. The experiments show state-of-the-art results, with quantitative Table 1 and qualitative Fig. 3, 4  data demonstrating a clear superiority in modeling loose clothing and avoiding common artifacts like skirt-splitting.

### Strengths
Clear Problem Definition: The paper does an excellent job of identifying, diagnosing, and illustrating a significant weakness in current monocular avatar creation: the failure to model secondary motion for loose clothing. The analysis of initialization mismatch and temporal-unaware deformation is insightful and directly motivates the proposed solution.

Novel and Sound Methodology: The PGI stage is a clever solution to the initialization problem. Using a deformable NeRF to build a subject-specific, clothed canonical space is a much more robust approach than trying to fit a generic, naked template (e.g., SMPL) to loosely-clothed subjects.

The SMAD module is the paper's core strength. Moving from a simple pose-conditioned deformation to a physics-inspired, autoregressive GNN is a significant and logical step. The "Velocity Encoding" (VE), which incorporates a buffer of past motions, is a direct and effective way to model the history-dependent nature of cloth dynamics.

Extensive and Convincing Experiments: The experimental validation is a major strength. The method is compared against multiple strong, recent 3DGS-based methods. Evaluation spans three distinct challenges: novel view synthesis (ZJU-MoCap), novel pose synthesis (4D-Dress), and in-the-wild generalization (LoCo-Human). The method achieves state-of-the-art quantitative results across the board (Table 1). The qualitative results (Fig. 1, 3, 4) are particularly compelling, clearly showing the elimination of artifacts (like skirts splitting or "needle" artifacts) that plague other methods.

Dataset Contribution: The introduction of the LoCo-Human dataset, featuring in-the-wild videos of subjects in loose clothing, is a valuable contribution to the community, which lacks such data.

### Weaknesses
Insufficient Ablation Experiments: There is no ablation experiments and no discussion about the selection of GNN-based autoregressive deformer with other models.

Missing Experiment Details: In the ablation study of Physics & Graph Design, there is no detailed explanation about the ablation content. For example, physics in A0 and Contact-aware cross-edges in A3 are ambiguous.

Descriptive Ambiguity:  In the experiment part, this paper describes that the LoCo-Human features five Loose-Clothed Humans performing 5 dynamic and 1 static motions per subject. However, in the data statistics of the Appendix, this paper states that the dataset comprises 5 unique subjects, each recorded across 5 sequences.

Cost of PGI Stage: The PGI stage relies on training a full 4D NeRF for each subject before training the main SMAD model. Table F shows that this stage takes 12.5 hours, nearly 3x longer than the 4.5-hour SMAD training. While the fast inference (26 fps)  is excellent, the total training time (17 hours) is substantial. This high "personalization" cost should be more clearly discussed as a trade-off.

Figure Quality Issue: Some texts in the figure are not correctly ordered, such as the text “Motion” in Fig. 1 (b). There is a redundant line on the right of Figure L.

Others: In the More Results part of the Appendix, the citation of the figure is not correct.

### Questions
Why does this paper choose GNN-based? Is there any model selection ablation experiment?

What do the physics in the ablation study represent? Are they the two physics-based losses in Equations 11 and 12?

And what do the hierarchical body–cloth graph and contact-aware cross-edges represent?

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
4

### Summary
The paper proposes a framework for modeling dynamic appearances of avatars from single monocular videos using 3DGS, with an emphasis on loose-fitting garments and secondary motion (e.g., skirt flutter). Instead of relying on pre-defined template-based initialization and skeletal skinning-based animation, this paper obtains dense Gaussian initialization through personalized 4D NeRFs, constructs a velocity-encoded Gaussian graph and finally learns a secondary motion-aware deformation  module that autoregressively predicts second-order dynamics. The authors also collected a new LoCo-Human dataset, containing in-the-wild videos with dynamic cloth motion. Experiments show strong improvements over state-of-the-art 3DGS-based avatar methods (e.g., 3DGS-Avatar, ExAvatar) on multiple datasets.

### Strengths
* The velocity-encoded Gaussian graph is a plausible design: it introduces physical intuition (mass–spring–damper) while maintaining differentiability for network learning. Such a design addresses a key gap in existing 3DGS-based avatars that only model the per-frame pose-to-deformation mapping and neglect the second-order dynamics.

* The LoCo-Human dataset addresses the lack of dynamic loose clothing under monocular capture in existing benchmarks. The dataset ethics statement is detailed, with informed consent and consideration for potential misuse.

* The paper is generally well-written and easy to follow. Figures are clear, with helpful comparisons and ablations.

### Weaknesses
* Constructing a node graph for modeling loose garments is not a new thing. Similar ideas have been well explored in previous methods using mesh-based representations like "Real-time Deep Dynamic Characters" (Habermann et al 2021). More discussions on the relationship between this paper and existing methods are necessary. 

* Although the paper draws analogies to second-order mass–spring systems, the GNN-based updates are learned implicitly, and no physical consistency (e.g., mass, stiffness calibration) is enforced. The “physics-inspired” term might overstate the grounding; clarification on whether parameters (k_ij, γ_i) are learned or derived would help.

### Questions
Missing citation: 

Li et al. Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling. CVPR 2024

### Soundness
3

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
4

### Summary
* This paper introduces a method for creating a person-specific animatable avatar from a single video.
* Method combines a commonly used approach of attaching gaussian splats to a human LBS model, and intorduces a secondary motion modeling element on top. Namely, the gaussian location and its moments are modeled as a dynamic system, parameterized by an autoregressive graph neural network conditioned on per-frame latent codes and representations of history/neighbors states and velocity. 
* Evaluation is conducted on a set of standard benchmarks (ZJU-Mocap, 4D-Dress), as well as a newly introduced dataset, comprised of videos more suitable for evaluation on subjects with loose garments (LoCo-Human).

### Strengths
*Clarity/quality:*
- Paper is relatively well-written and is easy to follow.
- Method seems easy to implement.

*Originality / significance:*
- Existing approaches for avatar modeling indeed lack realistic 2ndary
motion modeling, and modeling it with a hybrid physically inspired approach
(predicting physical properties with a NN) sounds like a technically sound
approach.

*Evaluation:*
- Quantitative and qualitative comparison indicates that the method
performs favorably compared to the chosen baselines.
- On the examples shown in the supp video clothing deformations indeed
look convincing.

### Weaknesses
*Method Limitations:*
- Method is person-specific, which means that a new model is trained per input video, and if the information (e.g. about the back of the body) is missing, there is no way to recover it from a prior.
- Similarly, one can be sceptical that complex physics of clothing
eformations can be learned from a single video without relying on any data-driven prior or a large dataset.
- (Arguable) This means that the method is unlikely to generate truly realistic motions for poses outside of training distribution, and is likely
overfitting to training sequences.


*Novelty / Significance:*
- The overall GS+LBS pipeline is not novel, not fully clear if the proposed dynamics formulation combined with GNN has
been done before.

*Experimental Evaluation:*
- It is unclear why the methods are different across different bencharks.
- Not fully related to the papers itself, but the quality of ZJU-Mocap
dataset is extremely poor to the point that results on that dataset
are not informative.
- (Minor) For ablation study, it would be useful to understand how the A0 performs compared to baselines (is it already better?). Also, was it conducted on a single subject? If so, it is unclear if the results
are very reliable.
- (Minor) It is actually unclear if the method is in any way specific to a single video setup (arguably, it is not - see method limitations), and there exists a variety of datasets of much higher quality (ActorsHQ, Goliath) which could inform whether the formulation provides
extra benefits in less noisy scenarios (and enable using more
baselines).

### Questions
1. Autoregressive formulation is known to be prone to error accumulation issues. Have authors considered evaluating their method on longer sequences
to confirm no "explosions" happens due to this?

2. Why different methods across different benchmarks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the challenging problem of creating animatable 3D human avatars from a single monocular video, with a specific focus on realistically modeling the secondary motion of loose-fitting clothing (e.g., skirts, coats). The authors identify two primary failings of current 3D Gaussian Splatting based methods: cloth shape-agnostic initialization and temporal context-unaware deformation. To solve this, the paper proposes a two-stage framework. The first is Personalized Gaussian Initialization (PGI) that avoids naked-body templates by first training a deformable 4D NeRF on the input video. A set of canonical 3D Gaussians is then extracted from this person-specific clothed shape. The second is Secondary Motion-Aware Deformation (SMAD) where the canonical Gaussians are structured into a graph. A GNN-based deformer then learns to animate these Gaussians.

The authors also introduce a new in-the-wild dataset, LoCo-Human, featuring subjects in loose clothing captured with smartphones. Experiments on LoCo-Human and other benchmarks (4D-Dress, ZJU-MoCap) show that the proposed method outperforms recent 3DGS-avatar baselines, particularly in the quality and realism of cloth animation.

### Strengths
1. The authors identify and illustrating (with Fig. 1) a key failure mode of modern animatable avatar methods, i.e., modeling varying clothing dynamics with time. This is in geenral a notoriously difficult and important problem, and the authors' analysis of why current methods fail (template mismatch and lack of temporal context ) is convincing. I would suggest adding some other relevant works [1,2,3,4] for trying to tackle the goal of modeling loose clothing with/without a template using point clouds and [5,6] that similarly use implicit representations for creating an initial representation to model clothing. [7] PhysGaussian is one of the works using physics to model dynamic Gaussians.
2. The proposed two-stage solution directly addresses the identified problems. The PGI stage creates a high-fidelity canonical representation of the clothed individual, which is a much better starting point for deformation. Note that this is explored in various other works as well [5,6]. The SMAD module's design is the paper's main strength. Moving from a per-frame, pose-conditioned model to an autoregressive, physics-inspired one is an important shift. Using velocity encoding to give the GNN a temporal state allows it to learn complex dynamics (like inertia and damping), which is something static models cannot do.
3. The experiments are thorough and are done on standard benchmark (ZJU-MoCap) as well as on more challenging and relevant datasets (4D-Dress, LoCo-Human) that contain subjects wearing loose clothing. The quantitative results (Table 1)  show improvement over all baselines across all three datasets. The qualitative results (Fig. 3, 4) visibly demonstrate the method's ability to avoid common artifacts like skirt-splitting and needle artifacts seen in competing work.
4. The ablation studies (Table 2, Fig. 6) effectively isolate the contributions of the key components, showing that Velocity Encoding and the graph-based SMAD deformer are essential to the performance gains.
5. The proposed LoCo-Human dataset is a valuable contribution. As it is captured in-the-wild with standard smartphones, it lowers the barrier to entry and will likely spur further research in this area.

### Weaknesses
1. The paper explains that N canonical Gaussians are down-sampled to M graph nodes (M << N) and that the GNN deforms these M nodes. However, it never explains how the deformation of these M nodes is propagated back to the full N Gaussians for the final rendering. Is each Gaussian rigidly attached to the nearest graph node? Is there an interpolation scheme (e.g., barycentric, LBS-like)? This is a critical, missing link in the pipeline.
2. The title is "Dynamic Texture Modeling...". However, the paper's novelty and focus are on dynamic geometry and motion. The authors model the deformation of 3D Gaussians (position, covariance) autoregressively. While color is predicted by a decoder (Eq. 8), there is no discussion of modeling dynamic texture (e.g., time-varying BRDFs, wrinkle maps, or view-dependent shading effects). The dynamic appearance is a result of the dynamic geometry, but the title suggests the texture itself is being modeled dynamically, which does not appear to be the case.

### Questions
1. The authors describe down-sampling N Gaussians to M graph nodes for the GNN. How are the deformations computed on these M nodes transferred back to the full set of N Gaussians for rendering?
2. Could the authors clarify the "Dynamic Texture" aspect of the title?
3. Section 4.3 and Table 2 state the best velocity window is T_v = 15, but Appendix D.1 says T_v = 11 yielded the highest performance. Could the authors clarify the difference in interpretation of these in selecting the final model?

### Soundness
3

### Presentation
4

### Contribution
3
