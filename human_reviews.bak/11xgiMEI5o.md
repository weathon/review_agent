# OmniRe: Omni Urban Scene Reconstruction

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
We introduce OmniRe, a comprehensive system for efficiently creating high-fidelity digital twins of dynamic real-world scenes from on-device logs. Recent methods using neural fields or Gaussian Splatting primarily focus on vehicles, hindering a holistic framework for all dynamic foregrounds demanded by downstream applications, e.g., the simulation of human behavior. OmniRe extends beyond vehicle modeling to enable accurate, full-length reconstruction of diverse dynamic objects in urban scenes. Our approach builds scene graphs on 3DGS and constructs multiple Gaussian representations in canonical spaces that model various dynamic actors, including vehicles, pedestrians, cyclists, and others. OmniRe allows holistically reconstructing any dynamic object in the scene, enabling advanced simulations (~60 Hz) that include human-participated scenarios, such as pedestrian behavior simulation and human-vehicle interaction. This comprehensive simulation capability is unmatched by existing methods. Extensive evaluations on the Waymo dataset show that our approach outperforms prior state-of-the-art methods quantitatively and qualitatively by a large margin. We further extend our results to 5 additional popular driving datasets to demonstrate its generalizability on common urban scenes. Code and results are available at [omnire](https://ziyc.github.io/omnire/).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces OmniRe, a comprehensive framework for dynamic urban scene reconstruction. It leverages neural scene graphs with Gaussian representations to unify the reconstruction of static backgrounds, moving vehicles, and non-rigidly dynamic actors. Additionally, it incorporates specialized designs for human modeling. The effectiveness of the approach is demonstrated across multiple datasets, showcasing superior performance in both reconstruction quality and novel view synthesis.

### Strengths
1. The paper is clearly written, with illustrative figures that are easy to understand. The experiments are comprehensive.
2. Modeling dynamic objects and simulating interactive behaviors are essential for closed-loop simulation in autonomous driving systems.
3. This work is highly engineering-oriented and demonstrates impressive results. Additionally, the authors have committed to open-sourcing the code, which will have significant value in advancing autonomous driving simulation in the future.

### Weaknesses
As mentioned by the authors in the limitations section, there are still two key shortcomings: 1. The lack of lighting modeling results in unnatural object insertions. 2. The synthesis of new viewpoints is constrained to the original trajectory, limiting the approach from achieving fully free-trajectory digital reconstruction.

### Questions
1. GS-based methods generally perform well in scenarios with static environments or low vehicle speeds, as demonstrated by most of the demos on the project page. However, I am curious about the reconstruction performance of this approach in situations where the ego vehicle is moving at higher speeds.
2. I wonder about the computational cost of reconstructing a complete segment in the Waymo dataset, as the entire pipeline seems a bit complex.
3. Why does it seem that the reconstruction quality of NuPlan is significantly worse than that of other datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces OmniRe, a novel approach for urban scene reconstruction that focuses on dynamic actors, including vehicles, pedestrians, and cyclists. OmniRe employs a Gaussian Scene Graph-based framework to model both static and dynamic objects. To address the limitations of previous methods in reconstructing non-rigid human models, OmniRe integrates SMPL for in-the-wild human representation, allowing for joint-level control. Extensive evaluations across several driving datasets demonstrate OmniRe's superior performance compared to baseline methods.

### Strengths
1. The paper is well-organized and easy to follow.
2. The proposed method for in-the-wild human representation is straightforward yet crucial for driving scene reconstruction.
3. Both quantitative and qualitative experiments effectively support the claims made in the introduction, with OmniRe achieving state-of-the-art results across various experimental settings.

### Weaknesses
1. Handling Occlusions and Complex Dynamics: OmniRe addresses in-the-wild challenges, yet the performance might be limited by severe occlusions and overlapping actors in complex urban scenes. Further refinement or integration of advanced occlusion handling techniques could enhance reconstruction fidelity.
2. Performance in Specific Urban Scenes: For specialized scenarios, such as highways (with fast-moving vehicles), nighttime environments, and adverse weather conditions, does OmniRe maintain high reconstruction quality under these challenging conditions?

### Questions
Please refer to the Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
OmniRe is a framework designed to create high-fidelity digital twins of dynamic urban scenes for simulations, particularly for applications in autonomous driving. OmniRe goes beyond vehicle modeling to support diverse dynamic actors like pedestrians and cyclists, enabling complex simulations that reflect real-world scenarios. It utilizes Gaussian Scene Graphs with multiple representations, allowing detailed and editable scene reconstructions with both rigid (e.g., vehicles) and non-rigid (e.g., pedestrians) actors.

### Strengths
Comprehensive Dynamic Modeling: OmniRe can handle various actors in urban settings, unlike most previous methods that focus mainly on vehicles.

Scene Graphs and Gaussian Splatting: The system uses 3D Gaussian splatting for detailed scene and object rendering, including control over each object.

Human Behavior Simulation: Through SMPL modeling, OmniRe accurately reconstructs human motions, even in cluttered environments, enabling simulations of interactions between pedestrians and vehicles.

State-of-the-Art Performance: Extensive testing on datasets like Waymo and others show OmniRe significantly outperforms existing methods in terms of visual fidelity and reconstruction accuracy.

### Weaknesses
There are several limitations in OmniRe approach, which are correctly identified in the paper too.

Lighting Effects: OmniRe doesn’t model lighting variations explicitly. This can lead to visual inconsistencies when combining scene elements with differing lighting conditions, which may reduce realism in certain simulations. Addressing this would require additional modeling of lighting dynamics.

Novel View Synthesis Limitations: OmniRe’s per-scene optimization approach struggles to generate satisfactory results when the camera view deviates significantly from the training trajectories. This could be a limitation for scenarios requiring a wide range of viewing angles, such as free navigation through the reconstructed scenes. The authors suggest incorporating data-driven priors or generative models as future work to address this.

Computational Complexity: While the method achieves high-quality reconstructions, the complexity of the Gaussian Scene Graph and the joint optimization of multiple parameters (pose, appearance, etc.) require substantial computational resources. Training time per scene, though optimized for an RTX 4090 GPU, could still pose scalability issues for large datasets or continuous real-time simulation needs.

Challenges with Real-Time Adaptability: The method’s reliance on SMPL modeling for human actors and per-node deformation fields, though effective, might introduce delays in real-time applications, particularly if scenes are highly dynamic or involve many non-rigid actors.

### Questions
I wonder for items like causality and new synthesis if an approach more configurable could take place now that they have separated the pedestrians from the road. 

Thinking of something like this 
Wang, Cheng Yao, et al. "CityLifeSim: A High-Fidelity Pedestrian and Vehicle Simulation with Complex Behaviors." 2022 IEEE 2nd International Conference on Intelligent Reality (ICIR). IEEE, 2022.

Where the data is later attached to an engine like Carla or airsim

### Soundness
3

### Presentation
3

### Contribution
3
