# ReLi3D: Relightable Multi-view 3D Reconstruction with Disentangled Illumination

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Reconstructing 3D assets from images has long required separate pipelines for geometry reconstruction, material estimation, and illumination recovery, each with distinct limitations and computational overhead. We present MIDR-3D, the first unified end-to-end pipeline that simultaneously reconstructs complete 3D geometry, spatially-varying physically-based materials, and environment illumination
from sparse multi-view images in under one second. Our key insight is that multi-view constraints can dramatically improve material and illumination disentanglement, a problem that remains fundamentally ill-posed for single image methods. Key to our approach is the fusion of the multi-view input via a transformer cross-conditioning architecture, followed by a novel unified two path prediction strategy. The first path predicts the object’s structure and appearance, while the second path predicts the environment illumination from image background or object reflections. This combined with a differentiable Monte Carlo multiple importance sampling renderer, creates an optimal illumination disentanglement training pipeline. Further with our mixed-domain training protocol, combining synthetic PBR datasets with real-world RGB captures, we establish generalizable results across geometry, material accuracy, and illumination quality. By unifying previously separate reconstruction tasks into a single feed-forward pass, we enable near-instantaneous generation of complete, relightable 3D assets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
ReLi3D is a feed-forward framework that takes a set of variably posed images as input and outputs a textured 3D mesh with PBR materials and an environment map. The model adopts a two-path architecture that separately learns geometry–material and environment–lighting representations, which are then fused through cross-attention layers. A differentiable renderer jointly renders outputs from both paths, enforcing material–illumination disentanglement and enabling image-space self-supervision. Extensive experiments demonstrate the superior performance of the proposed method.

### Strengths
1. Fast relightable 3D generation. The method can produce relightable 3D assets in under a second, which significantly enhances the practicality and usability of 3D generation systems.

2. Self-supervised learning on real data. The two-path architecture, combined with a differentiable renderer, allows the model to be trained on real-world images via self-supervision—an important advancement over prior works (e.g., Hunyuan, Trellis) that rely mainly on synthetic datasets.

3. Strong quantitative results. The approach achieves large improvements over baseline methods across multiple benchmarks.

### Weaknesses
1. No comprehensive visual results are provided for real-world data beyond the single example shown in the teaser. Also, including quantitative material evaluations on Stanford ORB would help substantiate the method’s real-world performance.

2. In the real-world example (the backpack in the teaser), noticeable artifacts appear in roughness and metallic predictions. Similarly, in Figure 2 (lamp case), the predicted high-frequency details appear blurry.

3. The presented examples mostly involve objects with simple geometry and uniform materials. It would strengthen the paper to include results on more complex, multi-material objects with intricate geometry.

I am happy to increase my score if the authors address these concerns.

### Questions
How does the choice of the hero view $h$ affect the final reconstruction quality? What is the selection strategy for $h$ ?

In Table 2, why doesn’t the performance consistently improve as the number of input views increases, given that additional views should generally reduce the task difficulty?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a method to reconstruct 3D assets with PBR materials + HDR envmaps from posed 2D images.

Main components of the method include:
- Efficient fusion of input views with cross-view feature fusion
- spatially-varying material prediction - a common approach to predict materials with MLP, given triplane features (inspired by LGM)
- separate branches for svBRDF (mesh + PBR) and HDR prediction
- Training on mix of gt PBR materials and envmaps (where available) and in "self-supervised" manner with MC renderer and 2D supervision

### Strengths
To my knowledge, the suggested approach is the first which jointly reconstructs mesh, PBR, **and** environment (HDR), all in a feedforward manner and at impressive speeds. To me, this constitutes a significant contribution.
Additionally, the paper contains novel ideas (more on that below) and is clear and well-written.

1. The idea of fusing arbitrary number of views with one "hero" view and other views with latent mixing (what the authors call "cross-view feature fusion") is novel and insightful.
2. Authors' choice of decoding svBRDF with an MLP -> Flexicubes from triplane features is very common in industry, which I think is a well-suited and efficient selection for 3D representation.
3. Both illumination reconstruction (Fig 4), PBR & relighting metrics (Table 1.) look convincing.

### Weaknesses
1. The most important ablation is missing on whether to use multiple paths (one for geometry & appearance, another for illumination path) or compute them all in a single path. 
2. While the method was trained and inferred either on real-world or synthetic data, it would be valuable to see how it generalizes to generated (e.g. with diffusion / flow models) images. This might improve practicality of this approach.
3. While 3D+Image metrics (Table 2) look convincing at first, qualitative results in Fig. 6 are hard to inspect.

### Questions
1. "Although rare, failure cases occur where the decomposition fails to disentangle lighting and materials, resulting in baked-in lighting affecting the material maps." - please provide visuals with failure cases
2. See weakness 1 - please provide ablation on using multiple paths vs single path.
3. Please clarify how the hero view is selected.
4. L292 "Critically, out training employs... background masking" - please provide more details on this approach. Theoretically, this should work similarly to taking masked / not masked render with some probability, not sure why custom occlusion logic is needed here.
5. See weakness 2 - although it's not critical please discuss or show at least a few 3D samples generated from images from diffusion / flow models.
6. See weakness 3 - please consider improving the visuals in Table 2. to better evaluate the quality of this method.
7. On choice of HDR prior (RENI++) - do you think the same framework could plug in alternative HDR priors (spherical harmonics / gaussians), did you experiment with any?

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
ReLi3D is the first unified end-to-end system that reconstructs complete 3D geometry, spatially-varying materials, and environment illumination from sparse multi-view images in under one second. By leveraging a transformer-based multi-view fusion and dual-path prediction architecture with differentiable rendering, it achieves fast, generalizable, and high-quality relightable 3D asset reconstruction.

### Strengths
1.The writing is clear and easy to follow.
2.The proposed pipeline is new and makes a meaningful contribution to the field.

### Weaknesses
please see weakness for detail

### Questions
1. Regarding mesh reconstruction alignment:

Based on my understanding, the meshes reconstructed by Huanyuan3D are represented in their own coordinate systems. Consequently, it may not be straightforward to obtain a direct correspondence with the input view — that is, the pose in which the input RGB view can be rendered. Under such a situation, it is unclear how the authors performed the image-based metric comparison in Table 2. The authors should clarify how they ensured pose alignment or view consistency before evaluating these metrics.

2. Regarding the training protocol (Appendix B.2):

According to Section B.2, the training process is divided into several stages. However, the current version of the paper does not clearly explain the details of each stage — for instance, which losses are used in each stage, which network components are trained or frozen. These details are crucial for reproducibility but are currently missing from both the main paper and the supplementary material. The authors should elaborate on these aspects and consider including a concise description of each training stage (objectives, losses, and training parts) in the main paper to facilitate re-implementation by readers.

3. Regarding ablation studies:

The current version lacks sufficient ablation experiments. At present, only a single experiment in the supplementary material ablates the MC-Render component. A more comprehensive ablation study is needed to strengthen the paper. Specifically, the authors are encouraged to include:

Performance comparison after each training stage;

Ablation of the main architectural modules;

Analysis of how performance changes if geometry&materials and environment illumination are predicted using two independent networks rather than a unified one.

If these necessary ablation results can be provided in the rebuttal, I would be inclined to reconsider and potentially raise my rating.

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
ReLi3D presents a feed-forward framework that reconstructs 3D geometry, spatially varying PBR materials, and HDR illumination from sparse multi-view images. It employs a two-path transformer architecture for joint material–lighting disentanglement, guided by a differentiable Monte Carlo renderer. Trained on mixed synthetic and real data, the model achieves fast (<1 s) and high-quality relightable 3D reconstruction.

### Strengths
1. The two-path feed-forward framework jointly reconstructs geometry, materials, and illumination under multi-view constraints, showing a clear and coherent design.

2. Uses Monte Carlo integration with MIS for training supervision, leading to more consistent and realistic reconstructions.

3. The model runs efficiently and shows some degree of cross-domain generalization with mixed synthetic–real training.

### Weaknesses
1. Limited evaluation diversity. The test data mostly covers diffuse or moderately lit objects. The paper lacks challenging cases such as metallic, transparent materials, or strong HDR illumination, where disentanglement performance would be most critical.

2. Lack of illumination disentanglement evaluation. The paper does not provide quantitative evaluation of the predicted lighting quality (e.g., comparison against SPAR3D or DiffusionLight) or at least sufficient qualitative examples demonstrating the accuracy of recovered illumination.

3. Potential dataset-specific bias. Since the real-world training data (UCO3D) includes only RGB supervision, the model might have learned dataset-specific entangled appearance cues rather than true material–lighting disentanglement on real data. More real-scene examples and an ablation showing the individual contribution of synthetic vs. real data would strengthen the claim of cross-domain generalization.

4. Missing ablation for cross-view fusion. Cross-view fusion is one of the key claimed contributions, but there is no explicit ablation showing how it improves view consistency compared to models without this fusion module.

### Questions
1. How is the hero view (line 231) chosen in multi-view fusion? Is it random or learned, and how sensitive is the model to this choice?

2. Have the authors tested failure cases such as baked-in lighting or specular highlight misinterpretation? Could you show such examples or briefly discuss possible solutions and limitations?

### Soundness
3

### Presentation
4

### Contribution
3
