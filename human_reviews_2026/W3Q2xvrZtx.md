# PD$^{2}$GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Articulated objects are ubiquitous and important in robotics, AR/VR, and digital twins. Most self-supervised methods for articulated object modeling reconstruct discrete interaction states and relate them via cross-state geometric consistency, yielding representational fragmentation and drift that hinder smooth control of articulated configurations. We introduce PD$^{2}$GS, a novel framework that learns a shared canonical Gaussian field and models the arbitrary interaction state as its continuous deformation, jointly encoding geometry and kinematics. By associating each interaction state with a latent code and refining part boundaries using generic vision priors, PD$^{2}$GS enables accurate and reliable part-level decoupling while enforcing mutual exclusivity between parts and preserving scene-level coherence. This unified formulation supports part-aware reconstruction, fine-grained continuous control, and accurate kinematic modeling, all without manual supervision. To assess realism and generalization, we release RS-Art, a real-to-sim RGB-D dataset aligned with reverse-engineered 3D models, supporting real-world evaluation. Extensive experiments demonstrate that PD$^{2}$GS surpasses prior methods in geometric and kinematic accuracy, and in consistency under continuous control, both on synthetic and real data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PD²GS, a novel framework for reconstructing and modeling articulated objects from multi-view images without manual supervision. Its core idea is to represent an object's various interaction states as continuous deformations of a single, shared canonical 3D Gaussian field, enabling smooth control and interpolation. Key contributions include a coarse-to-fine segmentation that automatically discovers rigid parts by clustering motion trajectories and refining boundaries with SAM, and the release of the RS-Art dataset for real-world evaluation.

### Strengths
The core idea of modeling all interaction states as continuous deformations of a single, shared canonical 3D Gaussian field is both simple and powerful. This elegantly sidesteps the "representational fragmentation" of prior two-state methods, enabling smooth, continuous control and interpolation of articulated poses, which is a major step towards high-fidelity digital twins. The framework automatically infers the number and boundaries of rigid parts without manual supervision. It achieves this through a clever coarse-to-fine process that first clusters Gaussians by their motion trajectories (using a VLM for part counting) and then refines boundaries using SAM, making it highly applicable to real-world objects with unknown kinematics.

### Weaknesses
The empirical evaluation lacks comparison to foundational dynamic scene representation works like D²NeRF (Dynamically Deformable NeRF) or Gao et al.'s deformable 3DGS, which also model scenes via a canonical field and latent-code-driven deformation. This omission makes it difficult to assess the true novelty and contribution of the deformation modeling component beyond the specific task of articulation.
The method is explicitly noted to assume "accurate camera poses," and its robustness to pose estimation noise, a common issue in real-world applications, remains entirely unvalidated. This is a significant practical limitation that is not addressed through ablations or sensitivity analysis, casting doubt on the method's real-world readiness. While tested on objects with up to three parts, there is no evidence provided for the method's performance on objects with a higher number of articulated parts (e.g., >5). The clustering and segmentation pipeline may face challenges with increasing complexity, and its scalability remains an open and significant question.

### Questions
1.Dynamic Scene Baselines: Why were foundational dynamic scene representations like D²NeRF or other deformable 3DGS methods not included as baselines? A comparison would help clarify whether the performance gains are specific to the articulated object modeling pipeline or also represent a general advance in deformation field modeling.

2.Camera Pose Robustness: The paper states an assumption of accurate camera poses. Could you provide an ablation or sensitivity analysis on the robustness of PD²GS to noisy camera poses, which are common in real-world SfM pipelines? This would significantly strengthen the claim of real-world applicability.

3.Handling Occlusion: The limitation of being unable to reconstruct unobserved geometry is acknowledged. Have the authors considered or experimented with incorporating learned or data-driven priors (e.g., diffusion models, symmetry) to plausibly complete the occluded parts of an object, especially around joints?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the problem of reconstructing articulated objects from multi-view, multi-state observations. The approach first learns a smooth deformation of a shared canonical field for each interaction state, and then uses the resulting deformation trajectories to guide a progressive coarse-to-fine part segmentation. The segmentation is further refined using SAM-based cues and boundary-aware Gaussian splitting. The method then estimates per-part meshes as well as joint types and parameters. In addition, the paper introduces a new dataset, RS-Art, containing a large number of real-world captures of articulated objects.

### Strengths
1. The newly proposed dataset RS-Art should be useful for further research work if made public, especially those real-world captures.
2. The paper seems to achieve SOTA performance than baselines with multi-state multi-view images in most cases.
3. The authors conducted extensive experiments on different datasets.

### Weaknesses
1. The whole systems seem to compose of numerous parts, which may be a little complicate and hard to extend.
2. Some visualizations on the newly-proposed dataset, including the data itself and the reconstructed results in videos would help readers grasp the new dataset.
3. The proposed method seem to be a little incremental though it achieves the best performance in most cases. It didn't deal with physical plausibility like 3D penetration. Its setting is also not unique as the main difference with previous method is changing from two-state to multi-state. The authors may elaborate on what's new that we are learning when building articulated objects.

Though I still have the mentioned concerns, I currently vote for borderline accept due to the extensive experiments and the SOTA performance.

### Questions
See above.

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
The paper proposes PD²GS, a self-supervised framework for articulated object modeling using 3D Gaussian Splatting. It learns a shared canonical Gaussian field and represents each interaction state as a continuous deformation via latent codes. A coarse-to-fine segmentation clusters Gaussian primitives by deformation trajectories and refines part boundaries using SAM-guided splitting, enabling part-level reconstruction and motion estimation. The authors also introduce RS-Art, a real-to-sim RGB-D dataset for evaluating generalization. Experiments show strong improvements over prior work on both synthetic and real objects.

### Strengths
- Technical contribution: the paper proposes a conceptually elegant unification of geometry and kinematics via continuous deformation of a canonical Gaussian field. Coarse-to-fine segmentation combining motion trajectories with SAM-driven boundary refinement is both novel and effective.
    
- RS-Art dataset is a meaningful contribution, bridging synthetic–real gaps with paired RGB-D data and 3D models.
    
- Comprehensive experiments on an expanded PartNet-Mobility split and the new dataset demonstrate strong performance and generalization.

### Weaknesses
- Pipeline is complex and involves many heuristic components, which limited the scalability of the method.
- The method proposed in the paper seems to require  multiple states, which puts forward more requirements for the data curation.  Furthermore, ensuring that the camera coordinate systems of all states are aligned is a challenge.  Outside the laboratory environment, such as in simple home scenarios, it is difficult for us to obtain states with multiple coordinate systems aligned, and the errors caused by coordinate misalignment are very likely to lead to failure.

### Questions
see weakness

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
This work presents PD$^2$GS, a framework for modeling articulated objects that overcomes the fragmentation and drift issues in existing self-supervised methods. It learns a shared canonical Gaussian field and represents arbitrary states as continuous deformations, jointly encoding geometry and kinematics. By associating each state with a latent code and using vision priors for part boundary refinement, PD$^2$GS enables accurate part-level decoupling while maintaining coherence. The method supports part-aware reconstruction, continuous control, and kinematic modeling without manual supervision.

### Strengths
1. The paper introduces a unified framework that models articulated objects through continuous deformations of a shared canonical Gaussian field, effectively addressing the fragmentation and drift issues inherent in previous discrete-state reconstruction methods. 

2. The method achieves part-level decoupling without manual supervision by leveraging generic vision priors and latent code associations, enabling fine-grained continuous control over articulated configurations. 

3. The paper contributes RS-Art, a valuable real-to-sim RGB-D dataset with reverse-engineered 3D models, facilitating rigorous evaluation on real-world data.

### Weaknesses
1. The reconstruction results exhibit excessive noise, particularly evident in the real-world examples shown in Figure 13, which raises concerns about the method's robustness in practical scenarios.

2. In Section 3.2 on deformable Gaussian splatting, the methodology bears strong similarity to existing 4DGS works such as [a], yet these related approaches are not cited or discussed.

3. The paper does not provide information about inference time per sample, which would be valuable for understanding the practical applicability of the method.

4. There are some related works that are missing in the paper: [b][c][d][e]

[a] 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering;

[b] SINGAPO: Single Image Controlled Generation of Articulated Parts in Objects;

[c] Part2GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting;

[d] REACTO: Reconstructing Articulated Objects from a Single Video;

[e] NAP: Neural 3D Articulation Prior.

### Questions
Please see the weaknesses. I am hesitant about the rating primarily due to the reconstruction quality. Since this is fundamentally a reconstruction task, the results appear too coarse and do not meet the expected level of fidelity for such work.

### Soundness
2

### Presentation
2

### Contribution
2
