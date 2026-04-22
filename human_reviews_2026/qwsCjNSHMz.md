# Learning 3D-Gaussian Simulators from RGB Videos

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Realistic simulation is critical for applications ranging from robotics to animation. 
Video generation models have emerged as a way to capture real-world physics from data, but they often face challenges in maintaining spatial consistency and object permanence, relying on memory mechanisms to compensate. 
As a complementary direction, we present 3DGSim, a learned 3D simulator that directly learns physical interactions from multi-view RGB videos.
3DGSim adopts MVSplat to learn a latent particle-based representation of 3D scenes, a Point Transformer for the particle dynamics, a Temporal Merging module for consistent temporal aggregation, and Gaussian Splatting to produce novel view renderings.
By jointly training inverse rendering and dynamics forecasting, 3DGSim embeds physical properties into point-wise latent features. This enables the model to capture diverse behaviors, from rigid and elastic to cloth-like dynamics and boundary conditions (e.g., fixed cloth corners), while producing realistic lighting effects. We show that 3DGSim can generate physically plausible results even in out of distribution cases, e.g. ground removal or multi-object interactions, despite being trained only on single-body collisions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces 3DGSim, an end-to-end differentiable framework that learns 3D physical simulation directly from multi-view RGB videos. 3DGSim combines a feed-forward inverse renderer based on MVSplat for reconstructing 3D Gaussian particles, a transformer-only dynamics model leveraging space-filling curves and temporal merging for efficient spatiotemporal reasoning, and a Gaussian splatting renderer for differentiable image supervision. Trained solely with image reconstruction loss, the model learns latent visuo-physical particle features that capture diverse dynamics without explicit physics priors. Experiments demonstrate that 3DGSim achieves high-fidelity, physically plausible long-horizon predictions, robust generalization to unseen scenarios such as ground removal or multi-object interactions.

### Strengths
-  The model generalizes to unseen scenarios such as ground removal and multi-object interactions, demonstrating robust physical understanding and scene editability.
-  Extensive experiments across rigid, elastic, and cloth datasets show physically plausible long-horizon rollouts with strong quantitative and qualitative performance over 2D baselines like Cosmos.

### Weaknesses
- The paper does not provide quantitative comparisons with existing 3D physical simulators. Although the authors mention that code and data for these baselines are unavailable, the absence of such comparisons weakens the rigor and positioning of the work.
- Several implementation and presentation details are omitted or unclear, which may cause confusion for readers:
  - The variable $p$ in Equation (2) is not explicitly defined.
  - The process of extracting the feature $f$ is not clearly explained.
  - The paper does not specify what dataset(s) were used for training or how many data samples are included.
  - The meaning of “Groundtruth” in Figure 7 is unclear.
  - It is not explained how the ground plane is represented or modeled in the method.
- All experiments are on synthetic datasets. The model's performance on real-world videos remains to be verified.
- It would be better if the author could add some ablation study of key design parts.
- Missing related work: The proposed idea is also related to [1].

[1] Latent Intuitive Physics: Learning to Transfer Hidden Physics from A 3D Video. ICLR 2024.

### Questions
- Part of the questions are listed in the weakness part.
- In Figure 12, what exactly do “input” and “reconstruction” views refer to? Is the “12 (4+5)” notation a typo or intentional (since 4+5 ≠ 12)?
- In Figure 14, both “inference speed” and “cloth/rigid data FPS” are reported. What is the difference between inference FPS and dataset FPS? Is there any difference in inference FPS between the two dataset?

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
This paper introduces 3DGSim, a fully differentiable framework that learns 3D particle-based simulators from multi-view RGB videos. The method integrates inverse rendering, transformer-based dynamics, and Gaussian splatting for novel-view synthesis. The authors demonstrate the model's ability to simulate rigid, elastic, and cloth-like dynamics and show generalization to out-of-distribution scenarios such as ground removal and multi-object interactions. However, the experimental evaluation lacks comprehensive comparisons with relevant 3D baselines, and the presented results appear limited in complexity and realism.

### Strengths
1. The proposed framework is fully differentiable and learns directly from multi-view RGB videos, eliminating the need for privileged information such as depth or object tracks.
2. The method introduces a transformer-only dynamics model that avoids hand-crafted graph structures and uses space-filling curves for efficient spatiotemporal reasoning.
3. The paper is overall well-written and should be easy to follow.

### Weaknesses
1. There is a lack of quantitative and qualitative comparisons with relevant 3D baselines (e.g., PhysGaussian, or other particle-based simulators). The evaluation is primarily limited to a comparison with Cosmos, a 2D video generation model. While this highlights differences in representation (2D vs. 3D), it does not adequately demonstrate the advantages of the proposed method over existing 3D approaches that tackle similar tasks.
2. The presented examples appear simplistic and lack realism. It is unclear how well the method generalizes to real-world data. Evaluation on established real-world benchmarks (e.g., PhysGaussian’s dataset) would strengthen the claims.
3. The experiments are limited primarily to falling objects. It is important to demonstrate the method’s performance on a wider variety of dynamic scenes (e.g., complex collisions, articulated objects or robot manipulation).

### Questions
The paper template appears inconsistent: the title should be left-aligned, and the header should include "Under review as a conference paper at ICLR 2026".

### Soundness
2

### Presentation
3

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
The paper proposes 3DGSim, an end-to-end differentiable pipeline that learns a particle-based 3D simulator directly from multi-view RGB videos. Experiments on rigid, elastoplastic, and cloth regimes compare against strong 2D video predictors and include ablations over state parameterization, camera count, and masking, alongside a speed study indicating near real-time rollouts on an H100.

### Strengths
1. The implementation details are extensive, promoting its reproducibility.
2. The three-module architecture is laid out crisply, the motivation for replacing kNN with serialization is well argued, and figures for temporal merging/serialization make the mechanism concrete.
3. The paper is well written and easy to follow.

### Weaknesses
1. The comparison of some related open-sourced works is missing:

  [1] NeuMA: Neural Material Adaptor for Visual Grounding of Intrinsic Dynamics

  [2] PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification

  [3]  Vid2Sim: Generalizable, Video-based Reconstruction of Appearance, Geometry and Physics for Mesh-free Simulation

2. Physical fidelity is assessed primarily with image metrics (PSNR/SSIM/LPIPS) derived from reconstruction loss, so it remains unclear how well the model captures contact timing, energy dissipation, or material parameters.
3. The datasets are fully simulated, and although the authors plan real-world validation, current claims about real-scene deployment are necessarily tentative.

### Questions
1. To help the paper reach its stated goals, I would encourage the authors to add at least a small set of physics-grounded metrics—contact event timing, restitution estimates, mass/Young’s modulus recovery, or energy drift—alongside PSNR/LPIPS/SSIM, which would substantiate the “learned physics” claim beyond photometric fidelity.
2. On modeling and analysis, I’m curious whether the latent features correlate with interpretable physical parameters. A simple probing experiment (or a lightweight regressor trained post-hoc) could test identifiability for density or stiffness using the simulated ground truth.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a 3DGS-based physical simulation framework that accurately models really visual-physical processes. It integrates an inverse renderer built on MVSplat, a Transformer-only dynamics module without explicit neighborhood graphs, and a differentiable 3D Gaussian Splatting renderer. Trained with a joint reconstruction-and-prediction objective, the system learns geometry, appearance, and dynamics simultaneously. The authors validate long-horizon predictions and novel-view consistency under OOD settings, such as terrain changes, multi-body interactions, and lighting across rigid, elastoplastic, and cloth scenarios, and show stronger visual metrics and near real-time inference compared to 2D video baselines. Overall, the framework unifies physically consistent 3D generation with interpretable particle states in a single framework, addressing inherent limitations of 2D generation and laying a potential foundation for downstream applications.

### Strengths
1.	The framework maintains stable, physically consistent rollouts under diverse conditions and generalizes well: even when trained only on "single-object-ground contact," it produces plausible multi-body interactions and reproduces visual phenomena such as shadows.
2.	By leveraging PTv3, it avoids the overhead of kNN graph construction and distance computation, enabling near real-time inference.
3.	It is validated on multiple datasets and can effectively learn material dynamics even from limited samples.

### Weaknesses
I have some concerns about 3DGSim:
1.	The experiments lack ablations on the TEM mechanism. If TEM is replaced with conventional temporal stacking/causal attention, how much does 3DGSim's performance degrade?
2.	Without explicit physical constraints in optimizing the dynamics model, how is to prevent interpenetration between particles?
3.	Although 3DGSim performs strongly on synthetic datasets, there is no validation on real data. Can it retain its performance under lighting changes and real-world scenarios? In addition, what are the computational and data requirements (for both training and inference) that constitute the practical threshold for deployment?

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3
