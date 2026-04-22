# DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 8

## Abstract
Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio–temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind–object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio–temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enable new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind–object interaction modeling. The project page is available at: [https://zju3dv.github.io/DiffWind/](https://zju3dv.github.io/DiffWind/).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces **DiffWind**, a physics-informed generative framework that reconstructs and simulates **hidden wind fields** from **multi-view videos** of wind-driven objects.  
The core idea is to jointly optimize the **latent wind field** and the **object motion** under physical constraints.  
DiffWind represents wind using an **Eulerian grid** and objects using **Lagrangian particles**, coupled via the **Material Point Method (MPM)** for differentiable wind–object interaction.  
The method further enforces **fluid dynamics consistency** through a loss derived from the **Lattice Boltzmann Method (LBM)**, ensuring physically plausible flow.  

The authors also construct a new dataset (**WD-Objects**) containing both synthetic and real scenes of deformable objects driven by wind.  
Experiments show that DiffWind achieves high-quality 3D reconstruction, realistic forward simulation, and plausible “wind relocation” (transferring estimated wind fields to new objects or scenes).  

Overall, this is a technically strong paper with detailed derivations — good job by the authors.

### Strengths
- **Novel problem formulation:** One of the first attempts to jointly reconstruct *hidden wind fields* and *object dynamics* from visual observations, bridging 3D reconstruction, differentiable physics, and generative modeling.  
- **Strong technical depth:** The coupling of the Eulerian grid (wind) and Lagrangian particles (objects) via MPM is elegant and physically grounded. The inclusion of an LBM-based regularization further enforces physical realism.  
- **Thorough experiments:** Evaluated on both **synthetic and real** WD-Objects datasets with multiple categories (cloth, flags, plants, etc.), demonstrating reconstruction quality (PSNR, SSIM, LPIPS) and perceptual realism via user studies.  
- **Demonstrated versatility:** The model supports *forward simulation* and *wind relocation* tasks, showing potential for cross-scene generalization.

### Weaknesses
**W1. Dependence on ideal inputs:** The method requires multi-view, calibrated video and accurate segmentation, which limits its applicability to real-world, in-the-wild scenarios.  

**W2. Computational cost:** The framework integrates MPM, LBM, and differentiable rendering, but training time and memory requirements are not reported. Practical efficiency and scalability remain unclear.

### Questions
**Q1.** How robust is DiffWind to **incomplete or single-view** input? Could it generalize if only a subset of views is available?  

**Q2.** I am curious about the **training data setting**. The quantitative metrics in Table 1 are extremely high — for instance, a PSNR of 52.5 dB suggests a very dense camera setup. How many camera views were used during training?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a method for wind-driven object dynamics reconstruction. It leverages a combination of physics-based simulations and machine learning techniques to accurately model the behavior of objects subjected to wind forces. Given observational data, the method learns to infer the underlying physical parameters used in the Material Point Method (MPM) simulations, enabling realistic reconstructions of object dynamics in windy environments. A physics-aware optimization strategy is employed to improve the fidelity of the reconstructions.

### Strengths
1. The coupling of differentiable physics (LBM + MPM) with 3DGS for realistic wind–object interaction modeling is novel. 
2. Experiments on synthetic and real-world datasets demonstrate clear performance gains over state-of-the-art methods. 
3. The introduction of WD-Objects and the novel “wind retargeting” task broaden research potential.

### Weaknesses
1. This work can be viewed as exploration in generative simulation. However, the experimental results in the paper are still toy examples. The author should justify the practicality of the proposed method in real-world applications. What can this method be used for in practice? 
2. The novelty of the proposed method should be further emphasized. How does it compare to existing approaches in the literature? Based on PhysGaussian, the authors should discuss more about the differences and improvements over prior optimization-based methods for physical parameter estimation, such as PhysDreamer, Physics3D, DreamPhysics, OmniPhysGS, etc. 
3. In the supplementary material, the authors only provide backward reconstruction results for clean-background cases. It seems that the method can only handle simple scenarios, which limits its applicability. The authors should provide more analysis and discussion on the limitations of the proposed method.

### Questions
1. What's the computational cost and runtime of the proposed method compared to baselines?
2. How does the method initialize the physical parameters for optimization? A comparison between the initialization and the final learned parameters would be insightful.

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
This paper proposes a physics-informed, differentiable modeling framework designed to simulate and reconstruct wind-driven object dynamics from videos. 
It models wind as a grid-based physical field and objects as particle systems, and uses the MPM for object-wind interaction. DiffWind optimizes both wind forces and object dynamics via differentiable rendering and simulation with LBM, ensuring physical consistency.
The objective experimental results are quite impressive.

### Strengths
1. The framework allows joint optimization of wind forces and object dynamics and leverages differentiable physics simulation for accurate reconstruction. The use of LBM ensures the wind dynamics adhere to fluid mechanics laws. I think this should be effective and novel.
2. The method outperforms state-of-the-art dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity.
3. It introduces wind retargeting to enable wind dynamics to be applied to novel objects. This expands the range of its use in simulations and visual effects.
4. It introduces a dataset WD-Objects for modeling wind-driven object dynamics.

### Weaknesses
1. As the authors stated, the current implementation focuses on modeling object-level dynamics without accounting for interactions between multiple objects.
2. This method requires accurate segmentation for optimal performance. This may limit its application in less controlled environments and practical scenarios.
3. This paper focuses on continuum objects. What will happen when the method is extended to simulate behaviors in other types of objects?

### Questions
1. I wonder whether it becomes computationally intensive for large-scale or complex simulations with LBM and MPM?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes DiffWind, a physics-informed, differentiable framework that jointly reconstructs an invisible, time-varying wind field and the visible, deformable object motion from sparse-view RGB videos. The authors also introduce WD-Objects (synthetic + real scenes) and report improved novel-view rendering and physically plausible simulations, including wind retargeting and forward simulation under new wind conditions.

### Strengths
1. The motivation is quite clear. The paper is well written and easy to follow.
2. The creative combination of LBM + MPM + 3DGS removes limitations of prior work that either modeled only visible dynamics or only simple forces.
3. The wind retargeting demonstrates a strong capability to generalize the wind to other objects.
4. On both synthetic and real data, DiffWind outperforms state-of-the-art dynamic 3DGS baselines on novel view synthesis.

### Weaknesses
1. The method explicitly optimizes only the wind force field while keeping material parameters fixed after MLLM “physical agent” reasoning; this makes it hard to disentangle whether observed motion comes from wind magnitude or material stiffness/damping.
2. Evaluation on real data relies on image metrics (PSNR/SSIM/LPIPS) and a user study, but no direct wind-field measurements are reported. Given that there are no public datasets for wind-driven dynamics, it would strengthen claims to instrument selected scenes with anemometers or PIV.

### Questions
1. What are the failure modes when the wind is highly turbulent or when Reynolds numbers push LBM discretization limits at the chosen grid resolution?
2. How does performance scale with grid size (e.g., $128^3$) vs. runtime, and is there an adaptive meshing strategy planned?

### Soundness
4

### Presentation
4

### Contribution
4
