# Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
We propose Light–Geometry Interaction (LGI) maps, a novel representation that encodes light-aware occlusion from monocular depth. Unlike ray tracing, which requires full 3D reconstruction, LGI captures essential light–shadow interactions reliably and accurately, computed from off-the-shelf 2.5D depth map predictions. LGI explicitly ties illumination direction to geometry, providing a physics-inspired prior that constrains generative models. Without such prior,  these models often produce floating shadows, inconsistent illumination, and implausible shadow geometry. Building on this representation, we propose a unified pipeline for joint shadow generation and relighting-unlike prior methods that treat them as disjoint tasks-capturing the intrinsic coupling of illumination and shadowing essential for modeling indirect effects. By embedding LGI into a bridge-matching generative backbone, we reduce ambiguity and enforce physically consistent light–shadow reasoning. To enable effective training, we curated the first large-scale benchmark dataset for joint shadow and relighting, covering reflections, transparency, and complex interreflections. Experiments show significant gains in realism and consistency across synthetic and real images. LGI thus bridges geometry-inspired rendering with generative modeling, enabling efficient, physically consistent shadow generation and relighting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework to generate shadow maps according to the specified lighting condition and supports object-background composition with plausible shadows and relighting. The paper leverages latent bridge mapping to generate the results, and condition the model by introducing Light–Geometry Interaction (LGI) maps. The LGI map leverages information from the 2.5D estimated depth map and captures light-geometry interaction by the elevation angle of the reprojected points from the depth map. The LGI provides light-aware occlusion cues and controls the diffusion model to generate plausible results. Experiments show plausible object insertion results with realistic shadow effects.

### Strengths
1. Reasonable design. The idea of using 2.5D information from depth maps to obtain occlusion cues is reasonable. The proposed LGI method leverages ray sampling and reprojection, which constitute a physically informed algorithm.
2. Dataset contribution. The paper proposes a large-scale dataset, ShadRel, providing synthetic data for lighting and shadow effects. The research community can benefit from the proposed dataset for training and benchmarking.

### Weaknesses
## Major
1. Limited novelty. This paper is built upon the existing LBM framework with a new LGI representation, which is the paper's main contribution, as a control signal providing occlusion cues. However, the idea of ray sampling in LGI is not new -- similar to screen-space raytracing (SSRT) in computer graphics and existing works [1,2] already use this for object insertion, cast shadows and relighting.
2. Simple setting. Most experiments use the setting of inserting the object into a planar background with point light sources, which is a simple setting and not very interesting. The image harmonization setting is more interesting -- but the paper only provides limited results in Fig. 7.
3. Missing baselines. Although this paper is more focused on shadow generation, general-purpose image-based relighting/harmonization methods such as IC-Light [3] are still important baselines since they also handle shadows, especially for the image harmonization setting.

## Minor
1. Low image quality in the figures. The paper uses bitmap images for Figs. 2, 6, 7, causing sawtooth artifacts in fonts when zooming in. In addition, the image resolution in Fig. 7 is low. I may expect a higher-quality result for image harmonization.
2. Missing citations. Lots of relevant papers, such as SSRT-related graphics papers [1,2] or image-based lighting papers [3] are not cited.

## References
- [1] Griffiths, David, Tobias Ritschel, and Julien Philip. "OutCast: Outdoor Single‐image Relighting with Cast Shadows." Computer Graphics Forum. Vol. 41. No. 2. 2022.
- [2] Zhu, Jingsen, et al. "Learning-based inverse rendering of complex indoor scenes with differentiable monte carlo raytracing." SIGGRAPH Asia 2022 Conference Papers. 2022.
- [3] Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport." The Thirteenth International Conference on Learning Representations. 2025.

### Questions
1. Can the authors clarify the difference between the proposed LGI and SSRT in the rebuttal and summarize their contributions in this design?
2. Can the proposed method work well on object insertion into more complex background environments, rather than a simple planar background with point light sources?
3. The example illustrated in Fig. 3 is interesting, but I don't find how the proposed method handles the ambiguous case in Fig. 3(d). How will the LGI encode the "not sure" information in such case?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel shadow generation and relighting framework, which introduces LGI maps to provide lighting and shadow cue for the model. The proposed method derives the LGI maps from the estimated 2.5 D depth map and lighting conditions, then a LGI maps and lighting parameter conditioned bridge matching model generates the relit images .  Experiment results are provided to demonstrate the effectiveness of the framework, highlighting the good performance and efficiency of the proposed method.

### Strengths
1. This paper proposes a novel shadow generation and relighting framework. The LGI map can model the light-geometry interactions from the depth map which encodes useful shadow and illumination information, enabling the model to generate reasonable relighting results.
2. Extensive experiments show that the method achieve better performance than baselines.

### Weaknesses
1. I have some doubts about the proposed LGI map and whether it can accurately represent shadow and lighting information. Consider the point $p$ in fig.3 (a), and for this point, $e^d_n$ is always negative, which means that the elevation of light is always higher than the horizon, so this point can see the light source and its $c^m_1$ is negative. But for the point $p$ in fig.3 (c), this point is in the shadow, but its $c^m_1$ is also negative. So why the $c^m_1$ indicates the potential start of occlusion. And why is $c^m_1$ darker in the shadow area of fig.4(b)? Logically, in areas under a light source, $c^m_1$ should always be negative, meaning that the illuminated region should have a darker $c^m_1$.
2. The derivation of LGI maps and shadow mask does not consider the radius or the area of the light. When the light source is a single point, then we will obtain the hard shadow. But when the light source is not a single point, then we should consider soft shadow, including umbra and penumbra. It seems that the mixture of $c^m_1$, $c^m_2$ and $c^m_3$ can handle the soft shadow, but i am not very sure.
3. The cases in the dataset may be too simple, only a floor and a wall. Could the proposed method handle background with more complex geometry and texture?
4. The baseline used is not strong enough. Could the authors use more recent generative relighting methods, such as DiffusionRenderer and DiLightNet?

Liang, Ruofan, et al. "Diffusion Renderer: Neural Inverse and Forward Rendering with Video Diffusion Models." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

Zeng, Chong, et al. "DiLightNet: Fine-grained lighting control for diffusion-based image generation." *ACM SIGGRAPH 2024 Conference Papers*. 2024.

### Questions
See weakness

### Soundness
2

### Presentation
2

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
This work presents a new 2.5D representation from RGB images, depth information, and lighting parameters, termed as Light-Geometry Interaction maps (LGI), to boost the joint shadow generation and object relighting. As claimed in the paper, this LGI representation provides a physics-inspired prior to reduce floating and implausible shadow effects and empowers a neat pipeline to tackle direct illumination, secondary reflections, and inter-reflections. Besides, to enhance their pipeline training and evaluation, a synthetic dataset with 800K samples is constructed. Their experimental results illustrate better quantitative and qualitative performance compared to prior works.

### Strengths
1. LGI provides an intuitive utilization of the depth prior and suits the pipeline objective as joint shadow generation and object relighting.
2. The proposed pipeline integrates the modern large model backbones, such as depth estimation models and latent bridge matching models.
3. Dataset contribution: The provided dataset, ShadRel, provides a valuable contribution to this community.

### Weaknesses
1. Dependence on monocular depth information. A significant improvement of the proposed pipeline is introducing the depth information into the LGI representation, which also raises concerns. Monocular depth is an ill-posed estimation, and it frequently fails to provide accurate estimations, thus leading to malicious priors in the generation pipeline. This requires further clarification. At the same time, although LGI provides better performance compared to previous baselines, the depth information constitutes an extra input requirement not shared by several baselines (e.g., LBM, CSG), which only rely on RGB and light parameters.
2. The experimental results are weakened by the tight relationships between the proposed method and the ShadRel dataset. Although the authors have retrained baseline methods on the ShadRel dataset, this does not fully address the domain bias, since the ShadRel is synthetic and specifically tailored for the method’s assumptions and requirements (such as a single object, known light direction, and planar contact). This raises concerns that the illustrated improvements may primarily reflect the dataset alignment instead of model effectiveness.

### Questions
1. The potential impact of inaccurate depth estimation should be justified.
2. Enhanced ablation studies are recommended, which illustrate other ways to incorporate the depth information or inject this information into baseline methods. These studies are beneficial for validating the paper’s core design LGI. A fairer comparison would require facilitating baselines with similar depth guidance or reporting ablations isolating the effect of added depth input.
3. To validate model generality, it is recommended to evaluate models on real-image benchmarks using identical training data or provide cross-dataset results demonstrating robustness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a unified pipeline that jointly generates shadows and performs object relighting, achieving physics-based control. It also introduces the ShadRel dataset for coupled light transport. The experimental setup is comprehensive, and the visual results are excellent for both synthetic and real-world cases.

### Strengths
- This paper presents the first generative method for the joint synthesis of shadows and object relighting from a single 2D RGB image. A key innovation is the introduction of a control mechanism conditioned on monocular depth, which enables the generation of continuous and coherent shadow and relighting effects in direct response to continuous variations in illumination.
- This paper introduces ShadRel, a large-scale synthetic dataset developed for the tasks of shadow generation and object relighting. The dataset is notable for its extensive volume and the inclusion of diverse and complex lighting scenarios, rendering it a valuable asset for training and evaluating models in these domains.
- This paper introduces a novel light-aware occlusion representation, LGI map. The methodology for acquiring this map is clearly articulated, and both theoretical analysis and experimental results compellingly demonstrate that the LGI map enables the generation of results with physically-based shadow and illumination constraints. This representation effectively encodes light-aware occlusion from monocular depth, providing a physics-inspired prior that constrains generative models to produce more realistic and consistent outputs.
- The paper is well-supported by thorough experimentation, clear theoretical explanations, and compelling visual results.

### Weaknesses
The visual results presented in the paper are compelling. However, the experiments primarily showcase objects with relatively simple geometric structures. To more rigorously assess the robustness and generalization capabilities of the proposed method, I would encourage the authors to include results on more structurally complex objects. For instance, objects with fine-grained details, intricate parts, or significant self-occlusion (e.g., a bicycle, a detailed sculpture, or a potted plant) would serve as more challenging test cases. Demonstrating high-fidelity shadow and relighting effects on such objects would provide stronger evidence of the model's effectiveness and further strengthen the paper's contributions.

### Questions
Please refer to the weakness.

### Soundness
4

### Presentation
3

### Contribution
3
