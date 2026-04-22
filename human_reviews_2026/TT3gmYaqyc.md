# CameraNoise: Learning Precise Camera Control with Video Diffusion in Noise Space

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
Controlling camera pose in video diffusion models is essential for generating realistic videos, yet existing approaches struggle to achieve precise control. Methods that directly inject numerical camera parameters into the diffusion backbone often fail to capture subtle viewpoint variations and lead to structural distortions or visual artifacts. To overcome these limitations, we propose CameraNoise, a temporally coherent stochastic representation warped from camera intrinsic and extrinsic parameters. Unlike conventional approaches, CameraNoise embeds camera poses directly into the noise space. This makes our approach independent of scene appearance while faithfully encoding camera motion. Specifically, we introduce a novel Geometry-guided Reprojection Flow along with a CameraNoise warping algorithm, which jointly preserves the Gaussian prior of diffusion and ensures consistent noise propagation under camera transformations. By integrating CameraNoise into the diffusion process, our framework delivers stable and high-quality videos with precise camera control. Extensive experiments on the RealEstate10K benchmark demonstrate that our approach significantly outperforms prior methods in both fidelity and controllability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes CameraNoise, a method for camera-controlled video diffusion. CameraNoise embeds camera poses directly into the diffusion noise instead of using a conditioning signal within the model blocks, e.g., based on Plucker coordinates. For this, the paper proposes Geometry-guided Reprojection Flow (GRFlow), an alternative to using optical flow for noise warping, where the main motivation is to disentangle camera motion from visual content.

### Strengths
- Disentangled camera motion and visual content: Constructing warped noise where the camera motion is disentangled from the visual content is sound and a good idea compared to previous optical flow-based methods.

### Weaknesses
- Missing comparisons: Previous works such as Go-with-the-Flow [1] also use noise warping for camera-controlled video modelling. However, there are no comparisons with that line of work. Moreover, there is no comparison with GEN3C [2], a method that uses depth-based warping for 3D inductive bias.
- Lack of video results: The supplementary website is good but it is missing some more videos. First, it would be great to show generalization beyond RealEstate10K. So ideally the paper can show multiple OOD videos with the same trajectory to show that the model is stable across scenes and consistent w.r.t. camera control precision. Moreover, side-by-side comparisons with baselines would be good. Lastly, it would be great to show if noise from an input video can be transferred to other video generations.

[1] Burgert et al., Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise, CVPR 2025 \
[2] Ren et al., GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, CVPR 2025

### Questions
The paper seems sounds and I like the method. However, there is some lack of comparisons and visual results.
I would like authors to address following questions:
- How does the method compare to recent methods such as Go-with-the-Flow or GEN3C?
- How about some additional visual results such as results on scenes with dynamic objects, and comparisons with previous works?

I currently rate the work below the acceptance threshold but would be happy to consider my rating depending on the rebuttal.

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
3

### Summary
This paper introduces CameraNoise, a novel approach for camera-controllable video generation that embeds camera pose information directly into the noise space of diffusion models. The key contributions include: (1) a Geometry-guided Reprojection Flow (GRFlow) that captures camera motion independently of scene appearance, (2) a PDE-based warping algorithm that preserves Gaussian priors while propagating temporal correlations, and (3) a dynamic scaling training strategy to improve robustness. Experiments on RealEstate10K demonstrate improvements in both generation quality and camera control precision compared to existing methods.

### Strengths
1. Novel formulation and theoretical motivation: Embedding camera control in noise space is a conceptually clean departure from feature injection methods, with theoretical motivation and corresponding formulations.
2. Appearance-agnostic design: GRFlow successfully decouples camera motion from scene appearance (Figure 3), addressing a key limitation of optical flow-based approaches.
3. Comprehensive validation: Ablation studies (Tables 2-4) systematically justify design choices, and the method shows consistent improvements across multiple metrics.

### Weaknesses
1. Single-dataset evaluation: All experiments use only RealEstate10K (indoor scenes). Performance on diverse scenarios (outdoor, dynamic objects, varying depth) is unknown, limiting generalizability claims.
2. Missing computational analysis: Total training/inference time and overhead from PDE solving are not reported, making practical feasibility unclear compared to baselines.

### Questions
1. Failure cases. Under what conditions does the method fail? Can you provide examples where CameraNoise does not improve over baselines? 
2. Scale to longer videos. Table entries show 49-frame videos. How does temporal coherence degrade for longer sequences (100+ frames)?
3. Performance on dynamic scenes. The current evaluation focuses on static scenes where pixel displacement is primarily caused by camera motion. How does the method handle videos with significant object motion (e.g., people walking, cars moving)? Can you provide more generated samples under such challenging scenario?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CameraNoise, a way to encode camera pose control directly in the noise space of a video diffusion model. Instead of injecting camera parameters as features into the backbone, this paper computes an appearance-agnostic Geometry‑guided Reprojection Flow (GRFlow), which relies solely on camera parameters to characterize pixel displacements across frames. Then this paper 
formulates noise warping as a partial differential equation problem and solve it via a bipartite graph. To further enhance inference robustness, this paper introduces dynamic perturbations to camera extrinsics during training. On RealEstate10K, the method reportedly improves camera controllability (lower TransErr/RotErr) and overall video quality (lower FVD), with extensive ablations on GRFlow smoothing (α), λ, and DST.

### Strengths
1. Conditioning on noise rather than intermediate activations is well motivated and addresses the issue when directly injecting numerical camera parameters into the diffusion backbone fails to capture subtle viewpoint variations and leads to structural distortions or
visual artifacts.
2. The design of formulating noise propagation as the discrete solution of advection Partial Differential Equations (PDEs) is well derived and discussed. 
3. Strong empirical results on camera-control benchmarks on RealEstate 10k. 
4. Detailed ablations demonstrating the impact of each hyperparameter in the modeling process.

### Weaknesses
1. Experiments are confined to RealEstate10K, which contains mostly indoor, quasi‑static scenes with modest motion. The method’s generalization to outdoor/large‑scale scenes, heavy roll/pitch/yaw, or dynamic objects remains unclear.
2. Besides comparing to MotionCtrl, CameraCtrl, and AC3D, maybe also compare with other optical-flow derived noise warping in video diffusion (e.g., Go‑with‑the‑Flow: Motion‑Controllable Video Diffusion Models Using Real‑Time Warped Noise)

### Questions
N/A

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
The paper proposes CameraNoise, camera pose control framework for video diffusion models, addressing the structural distortions caused by directly injecting numerical parameters. CameraNoise embeds camera poses into the noise space using a temporally coherent stochastic representation, which is achieved via a Geometry-guided Reprojection Flow and a novel warping algorithm. This approach ensures consistent noise propagation while preserving the diffusion model's Gaussian prior, resulting in stable, high-quality videos with superior fidelity and controllability over prior methods on benchmarks like RealEstate10K.

### Strengths
- It's a good paper. Specifically, the motivation is clear and solution is intuitive and reasonable. 

- Unlike previous camera controllable video generation methods, the paper tackles initial noise representation that contains camera poses while keeping gaussianity. Although the method has similar philosophy as previous two works (How I warepd your nosie and Go-with-the-flow), CameraNoise does not require a source video or a reference video which is a strong merit.

- The experiments are comprehensive and presented clearly.

### Weaknesses
- In L193-194, how is the pseudo-depth $d$ computed? The pseudo-depth $d$ requires more description since if it's estimated using RGB pixels, the GRFlow can't be characterized as 'appearance-agnostic'. If the depth $d$ deivates too much from the g.t. depth, it would rather incur structural or camera errors.

- Another major weakness is that the method is experimented on RE10K only. Experiments on more benchmark dataset would add value to the paper. Moreover, is GRFlow and CameraNoise framework applicable to dynamic scenes (i.e., can it generate videos with objects with dynamic motion and also dynamic camera pose)?

- Can the authors clairfy how the proposed warping mechanism differs from Go-With-The-Flow warping and HIWYN?

- What is the computation complexity for the GRFlow construction and warping process, respectively?

- What is the base video model (t2v, i2v) for the training?

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
