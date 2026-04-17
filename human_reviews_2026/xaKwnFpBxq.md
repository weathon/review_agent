# Generalized Representation for Generalized Dynamics Generation

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Digital twin worlds with realistic interactive dynamics presents a new opportunity to develop generalist embodied agents in scannable environments with complex physical behaviors. To this end, we present PepGen (Potential Energy Perspective for Generalized dynamics Generation), a framework that seamlessly integrates rigid body, articulated body, and soft body dynamics into a unified, geometry-5
agnostic system. PepGen operates from the governing principle that the potential energy for any stable physical system should be low. This fresh perspective allows us to treat the world as one holistic entity and infer underlying physical properties from simple motion observations. We extend classic elastodynamics by introducing directional stiffness to capture a broad spectrum of physical behaviors, covering soft elastic, articulated, and rigid body systems. We propose a specialized network to model the extended material property and employ a neural field to represent deformation in a geometry-agnostic manner. Extensive experiments demonstrate that PepGen robustly unifies diverse simulation paradigms, offering a versatile foundation for creating interactive virtual environments and training robotic agents in complex, dynamically rich scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Gen-3, a unified, geometry-agnostic dynamics framework that extends classical elastodynamics with directional (anisotropic) stiffness, aiming to cover soft, rigid, articulated, and discontinuous behaviors within one potential-energy–based formulation. It learns a deformation field (reduced eigenmodes) and a per-point material field, then simulates under new forces. Experiments show reconstruction and short-horizon prediction across heterogeneous geometry types. Key limitations are the lack of direct comparisons to articulated-body reconstruction methods and the evaluation’s reliance on geometry metrics.

### Strengths
The directional Young’s moduli extend Neo-Hookean energy and enable a single system to mimic rigid/articulated/soft behaviors.

Works across meshes/point clouds/3DGS; reduced eigenmodes + neural material field is elegant and practical.

The energy-minimization with contrastive negatives is well-motivated and leads to robust reconstructions.

### Weaknesses
The paper convincingly demonstrates generalization across diverse object types; however, it lacks direct comparisons to specialized articulated reconstruction methods. I recommend adding a comparison with ”ArtGS:3D Gaussian Splatting for Interactive Visual-Physical Modeling and Manipulation of Articulated Objects”(whose examples indicate it can handle rigid/multi-part articulated motion) to more robustly support the claims on articulated scenarios.

### Questions
It would further strengthen the paper to include a comparative discussion with “Stable Constrained Dynamics”, since both works aim to build a unified simulator for elastic and rigid behaviors.

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
This paper introduces Gen-3, a framework that unifies rigid body, articulated body, and soft body dynamics through a generalized representation based on potential energy minimization. The core technical contribution lies in extending classical elastodynamics with directional stiffness via anisotropic Young's modulus, enabling the modeling of diverse physical behaviors within a single framework. Experiments demonstrate the framework's ability to handle different geometry representations and simulate various dynamics types.

### Strengths
- I believe this paper tackles an important problem: unifying disparate simulation paradigms (soft, rigid, articulated) within a single framework. This addresses a fundamental challenge in physical simulation with direct applications to robotics and virtual environments.

- Extending elastodynamics with directional Young's modulus may be an elegant mechanism for capturing diverse physical behaviors.

- The governing principle that stable physical systems maintain low potential energy states provides a solid theoretical foundation.

### Weaknesses
- There are very few visual results and no videos. Although the paper claims an anonymous page, I cannot find it.
There is no real example. 

- All test examples are synthetic. It would be much more convincing to have results on real data, such as the dataset introduced in SpringGaus (and its follow-ups). Therefore the results are not convincing to computer vision people.

Minor: A related work is WonderPlay (https://kyleleey.github.io/WonderPlay/) which also tackles the problem of simulating diverse types of dynamics within a single framework, although its problem setting is single image-based and thus different from this work. The related work section may benefit from discussing the relation to it.

### Questions
See the major weaknesses above. I'm confused that the paper says there is a link to video results, but I cannot find it. I cannot find any video in the supplementary zip, either. Without videos, there is no way to judge if the simulation looks good or not.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets dynamics generation for objects across diverse material types. The method learns dynamics via neural deformation eigenmodes coupled with transformation handles, while a learned material field captures material-specific attributes. Experiments show superior performance across multiple domains.

### Strengths
* The method is evaluated against multiple baselines and demonstrates superior performance.
* The introduced directional stiffness parameters enable modeling of a wide range of materials.

### Weaknesses
1. Clarify $W_{total}$ in Eq. 10. The definition of $W_{total}$ is unclear. Does it denote the sum of energies over all frames/observations, or an energy for only one frame?
2. Generalization beyond observed time. The method reads primarily as reconstruction from observed data. How does the model predict dynamics beyond the observation window? From Eq. (2), only the transformation handlers appear explicitly time-related. Please explain:
    1. How handlers reconstructed from observations extrapolate to unseen dynamics;
    2. Whether there is an explicit dynamic prior or evolution rule for handlers/eigenmodes;
    3. Any rollout strategy or error-accumulation analysis for long-horizon predictions;
    4. An efficiency analysis of inference speed.
3. The paper should include the discussion of [1,2] in related work, which are also related to 4D generation and dynamic simulations.

[1]. Cao, et al. Neural Material Adaptor for Visual Grounding of Intrinsic Dynamics. NeurIPS 2024
[2]. Shao, et al. GausSim: Foreseeing Reality by Gaussian Simulator for Elastic Objects. ICCV 2025

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes e a unified framework that integrates different physical systems, including rigid body, articulated body, and soft body systems.
The framework takes the potential energy perspective that the potential energy for any stable physical system should be low. With this perspective the problem can be cast into elastodynamics, using the elasticity energy function to enable interaction.

The framework contains two learnable networks: theta_W which learns the motion eigenmode weights, and theta_E which learns Young’s Moduli E. The learned components from the networks can then be used to simulate motion dynamics. For training, in addition to the reconstruction loss between predicted and observed outputs, the paper introduces orthogonality regularization (for theta_W) and the strain energy (for theta_E) in the loss term.

Experiments include evaluations on soft-body dynamics, articulated motion, and multi-body discontinuum systems. The proposed method is compared against several methods, including PhysGaussian, Simplicits, the differentiable MPM method, and SpringGauss. Results show that the proposed approach achieves lower reconstruction errors than the baselines and demonstrates strong ability in long-term dynamics prediction.

### Strengths
* The proposed framework can handle a wide range of physical systems (soft, rigid, and articulated), whereas many existing methods are limited to one or a few specific types.

* The experiments cover multiple diverse 3D scenes to better demonstrate the ability of the proposed method to handle different physical systems.

### Weaknesses
I have some additional questions listed in the section below.

### Questions
- The paper mentions “More visual results on our anonymous page”, but I was not able to find a link to the anonymous page from the paper and the appendix… 
- Can the model generalize to unseen scenarios? For instance, objects with the same material properties but different shapes and/or different numbers of input nodes, or the same learned objects but interacting with different fixed boundaries?
- It might be helpful to also demonstrate how each component of the loss function contributes to training performance, such as,, the impact of incorporating potential energy in the loss, and the effect of introducing negative transformation handles in the energy loss.

### Soundness
3

### Presentation
3

### Contribution
3
