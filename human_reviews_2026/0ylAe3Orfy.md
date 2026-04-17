# Multi-Object System Identification from Videos

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
We introduce the challenging problem of multi-object system identification from videos, for which prior methods are ill-suited due to their focus on single-object scenes or discrete material classification with a fixed set of material prototypes. To address this, we propose MOSIV, a new framework that directly optimizes for continuous, per-object material parameters using a differentiable simulator guided by geometric objectives derived from video. We also present a new synthetic benchmark with contact-rich, multi-object interactions to facilitate evaluation. On this benchmark, MOSIV substantially improves grounding accuracy and long-horizon simulation fidelity over adapted baselines, establishing it as a strong baseline for this new task. Our analysis shows that object-level fine-grained supervision and geometry-aligned objectives are critical for stable optimization in these complex, multi-object settings. The source code and dataset will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a synthetic video dataset that presents contact and interaction between different materials. A baseline framework that identifies the continuous, object-specific physical properties is also proposed, with geometric-driven supervision and object-aware dynamic component.

### Strengths
- I found this paper reused and properly directed to external sources for development that helps the readers from diverse backgrounds.
- A few features I see interesting to have together in this framework: simulation-ready continuum for multiple objects across all time, permaterial parameters to predict future motions.

### Weaknesses
My background is not in physics simulation, so I look at the paper from the perspective of a computer vision researcher.
- It is unclear how long the predicted horizon is. The example videos appear quite short—only a few seconds in duration, and seem not to show motion prediction (which I find different from trajectory prediction. Trajectory prediction is accumulating past locations, while motion prediction is to predict future locations). Please clarify the temporal length of the predictions.
- How many object categories are supported in the dataset? A summary table presenting dataset statistics would be helpful for readers. Also, I find that only two object interactions are limited compared to existing datasets such as CLEVRER.
- Additionally, please discuss or evaluate the model's ability to generalize to out-of-distribution in both object categories and interactions. This would also help to strengthen section subsection 3.7.

### Questions
See weaknesses.

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
The paper presents MOSIV (Multi-Object System Identification from Videos), a novel framework designed to infer continuous physical constitutive parameters for multiple interacting objects directly from multi-view video observations. By employing per-object geometric reconstruction followed by the system identification of continuous parameters, MOSIV moves beyond prior work that relies solely on selecting from a fixed, categorical library of expert constitutive models. This advancement significantly enhances the fidelity, precision, and physical plausibility of subsequent dynamic simulations.

### Strengths
+ Pioneering Continuous Parameter Identification: The primary strength is the innovative shift from categorical material classification to the estimation of continuous constitutive parameters. This allows for highly granular, object-specific physical calibration, overcoming a major limitation of current video-to-physics pipelines.
+ Effective Multi-Object Handling: The framework successfully tackles the complexity of simultaneous system identification across multiple objects that are undergoing complex interactions (e.g., contact, collision) within the same observed scene, which is a critical capability for real-world robotics and simulation.
+ Robust Framework Integration: The successful integration of geometric reconstruction and a differentiable physics pipeline suggests a robust, end-to-end optimization strategy capable of generating strongly calibrated and faithful physical models from raw visual data.

### Weaknesses
- Overall, this paper presents a clear goal and a coherent overall narrative. However, the main issue lies in the lack of clear motivation behind many of the detailed design choices. In several instances, important implementation details are missing, which can lead to reader confusion and make the paper difficult to follow.
- According to the description in subsection 3.3, objects are first represented as Gaussians, then converted into voxels, and finally transformed into particles. However, the whole process is quite unclear given the current simple explanation. Moreover, what parameters define these particles? Are they simply points? If so, why not use point clouds as the initial representation instead?
- In subsection 3.5, why are silhouettes and surfaces used as the objective? Is this a common approach for such tasks? If so, please provide appropriate references; if not, please clarify the rationale behind this choice.
- In Figure 4, different methods appear to have different observed frames. Does this imply that they were simulated using different parameters or settings? Shouldn’t the observed frames be consistent across methods to ensure a fair comparison?
- Given that the method involves per-object geometric reconstruction and the iterative optimization of continuous parameters, the computational expense is likely very high. Could the authors provide a rigorous analysis of the computational complexity (e.g., model size, training time, and inference time) and a comparison with other methods?
- How robust is MOSIV to different levels of reconstruction uncertainty? Did the authors conduct a sensitivity analysis showing how small errors in the initial reconstruction propagate into the estimated values of the constitutive parameters?
- While the parameters are continuous, the underlying physics model must still be chosen. Does MOSIV include a mechanism for a priori selecting the correct base constitutive model type, or is that externally provided?

### Questions
See weaknesses

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MOSIV, a new framework created to solve the problem of identifying the physical properties of multiple interacting objects simultaneously from a video. MOSIV works by using a differentiable simulator. It directly optimizes the specific, continuous material parameters for each object by trying to match the geometry observed in the video. MOSIV also presents a new synthetic benchmark with contact-rich, multi-object interactions.

### Strengths
- the vision of learning physical properties and its interactions directly from videos seems appealing
- The overall approach is logical and builds upon several state-of-the-art components, including the dynamic Gaussian Splatting for reconstruction and a differentiable MPM for physics-based parameter identification
- the author introduces a new multi-object dataset with diverse geometry, materials properties and physical motions, that could be used from the community

### Weaknesses
- the paper studies multi-object system interaction, but the proposed dataset only contains two-object interactions, also 30 frames of interactions seems quite short for evaluation, given the authors claim that the calibrated models generalizes to "long-horizon predictions of complex multi-object dynamics"

### Questions
- The method appears to be computationally expensive? It involves (1) optimizing a 4D Gaussian scene and assigning instance partitions, (2) converting object's reconstruction into simulation-ready continuum (3) running a differentiable MPM simulation to optimize per-object parameter vectors, and (4) optimizing this entire unrolled simulation. What are the typical run time and memory footprint? how does it compare with baseline methods?
- what are the novel interactions in section 3.7? if my understanding is correctly, the physical motions are kept the same, you only change the materials?
- what consitutive models have been used? how are they chosen, can you provide more details on that?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a framework to recover material properties from multi-view videos of multi-object scenes. The task setting is similar to PAC-NeRF except that multiple objects exist in the same scene. In additional to pixel supervision, the method also reconstructs particle trajectories using a 4D Gaussian framework, so that 3D surface loss can also be used as supervision.

### Strengths
- The experiments are comprehensive and promising.

### Weaknesses
- The contribution is limited. The following two works are not cited. Justification of contributions needs revision.
    - The task of multi-object system identification from videos is not novel: [1] also tackles multi-object system ID based on NeRF.
    - Geometry-driven supervision is not novel: [2] uses 4D Gaussian to reconstruction mesh sequence from multiview videos and use 3D loss to tune physical parameters of cloth.

[1] Li, J., Gao, Y., Song, W., Li, Y., Li, S., Hao, A. and Qin, H., 2024, October. CoupNeRF: Property‐aware Neural Radiance Fields for Multi‐Material Coupled Scenario Reconstruction. In Computer Graphics Forum (Vol. 43, No. 7, p. e15208).

[2] Zheng, Y., Zhao, Q., Yang, G., Yifan, W., Xiang, D., Dubost, F., Lagun, D., Beeler, T., Tombari, F., Guibas, L. and Wetzstein, G., 2024, September. Physavatar: Learning the physics of dressed 3d avatars from visual observations. In European Conference on Computer Vision (pp. 262-284). Cham: Springer Nature Switzerland.

- Some key descriptions of the proposed method are missing: 
    - How discrete constitutive models of each objects are set? If the types are given, "Predicted Categorical Distribution" in Fig 1 is very misleading. And [1] already handled this case. If the types are predicted, the paper does not mention the procedure at all in method section.
    - How silhouettes are rendered? Are they rasterized directly from MPM point clouds?
    - How are "extracted surfaces" (Line 256) extracted in detail?

- No real-world examples are provided.

### Questions
Key questions that needs justifications or clarifications are mentioned above.

### Soundness
2

### Presentation
2

### Contribution
1
