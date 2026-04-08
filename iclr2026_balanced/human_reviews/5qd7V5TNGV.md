## Human Reviewer 1

### Summary
The CP4D framework addresses limitations in existing 4D scene generation by ensuring faithful adherence to complex physical dynamics. It uses a compositional paradigm that integrates static 3D environments with physically grounded dynamic objects.

### Strengths
CP4D presents a novel compositional framework for photorealistic 4D scene generation, emphasizing faithful adherence to complex physical dynamics. The method integrates static 3D environments with physically grounded dynamic objects. Contributions include a hybrid motion synthesis strategy combining physical simulators and video diffusion priors for plausible trajectories and realistic interactions, and an automated composition mechanism that seamlessly fuses scene elements.

### Weaknesses
The primary limitation of the CP4D framework is the relatively long runtimes required to generate a complete physically realistic 4D scene. This inefficiency is due to the adoption of a stage-wise optimization strategy. Furthermore, the complexity arises because initial physical parameters estimated by Vision-Language Models (VLMs) often lack numerical accuracy. The approach must also address physics solvers' reliance on coarse grid approximations, which can lead to perceptually implausible outcomes such as "spurious collisions" or "phantom contacts"

### Questions
Could the authors elaborate on the composition and key features of this planned dataset?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces a compositional framework for 4D scene generation focused on physical plausibility by decoupling scenes into static 3D backgrounds and dynamic, physically-grounded 3D foreground objects. The three-stage pipeline first synthesizes 3D assets using a cascade of pre-trained models (LLM, T2I, Image-to-3D). Next, a hybrid motion strategy generates coarse trajectories using physical simulators (MPM, PBD) and then refines them using priors from video diffusion models (SDS loss). Finally, the framework automatically composes the scene, using monocular depth estimation and optimization to set the scale and position of foreground objects.

### Strengths
1. The task is worse investigating.

### Weaknesses
1. The dynamic view demos contains only zoom-in zoom-out motions, no camera pose/view angle change, the Static novel-view generation results seems just a crop from the original view. And yet the task is called 4D scene generation.

2. The foreground and background are almost completely irrelavent in the final results, the authors only did a ground/plane estimation to put the foreground to the corresponding postions but no interactions with the background.

3. Comparisons are not convincing, the sora and wan results are too bad compare to my experiences, looks like a reverse cherry picking. And other physic aware 3D/video generation methods provides much better visual qualities in their demos, I highly doubt the fidelity of the Vbench and worldscore results in Table1 and Table 2.

4. The proposed method is a highly complex cascade of numerous expert models (LLM, T2I, Image Edit, SAM, I23D (Trellis, Viewcrafter), VLM, Depth Estimator, and Video Diffusion). This pipeline-of-pipelines has many potential points of failure. 

5.  In section 4.2, the author admitted that the VLM predicted parameteres are not accurate and need refinements from video diffusion model, which is counter-intuitive to the core idea of the framework the author proposed in the first place.

### Questions
1. Since the task is text guided and the foreground objects are generated, why bother use a VLM to determine parameters like Young’s modulus, Poisson’s ratio µ, and density from your rendered generated 3D objects multi-view? The poor textures in the examples might introduce more noises.

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
This paper introduces a physical grounded 4D generation pipeline. Given a text prompt, it can synthesize dynamic scenes composed of static background and foreground object with physically plausible motion. The proposed pipeline comprises three stages: In the first stage, it generates separate foreground and background 3D representation using existing image generation/editing/segmentation, and 3D reconstruction/generation models. In the second stage, it initializes the physical parameters and external force via VLM, then simulate motions for diverse object with heterogeneous physical solvers and refine the estimated parameters through a SDS loss. In the last stage, the relative scale and translation of foreground and background objects are determined via depth-aware heuristics and photometric optimization for adequate composition.

### Strengths
1.	The paper is clearly written and easy to follow.
2.	Evaluation on 17 representative prompts demonstrates that the proposed framework can generate physically plausible 4D scenes containing foreground objects with diverse materials.

### Weaknesses
1.	The main concern is about the limited technical novelty. The proposed framework is basically the direct combination of existing components without touching their own limitations. While it attempts to integrate multiple solvers for different materials, the integration is largely naïve and lacks rigor evaluation for specific materials. For example, the simulation for fluid objects is only shown in the fourth video of the anonymous webpage with two raindrops, while they exhibit strange elastic behavior (bounce off the ground) due to the naïve boundary constraint handling.
2.	Despite being able to generate plausible motion, the proposed framework does not consider complex lighting interaction in the composition, so describing it photorealistic or visually realistic is somewhat inappropriate. Actually, this has already explored in previous physically grounded generation works such as PhysGen3D. The proposed “automated composition” mainly consists of some engineering heuristics—given $I_{b,f}$ and trusting their monocular depth, this task seems relatively trivial.
3.	This framework still relies on simplified assumptions of uniform material, limiting its scalability beyond simple objects and motions, and making the comparison with general-purpose video generation models somewhat unfair.
4.	The input of noise estimator in Equation (1) and (4) should be related to $\epsilon$。
5.	L093 claims that the proposed framework can avoids “realistic environments juxtaposed with cartoon-like objects” compared to text-to-3D alternatives. But the adopted text-to-image-to-3D approach can only mildly constrain the style of single input view. The cartoon style of generated assets largely stems from the training data distribution of 3D generation models. 
6.	No substantial novel-view rendering is provided, making it difficult to assess the 3D consistency.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper proposes CP4D, a compositional, physics-aware framework for 4D scene generation. The core idea is to decouple a static 3D background from physically grounded dynamic foreground objects, following a three-stage pipeline: (1) generate high-fidelity 3D representations for background and foreground with pre-trained expert models; (2) synthesize motion via heterogeneous physics simulators (MPM for elastic/flexible, rigid-body, and PBD for fluids) and then refine with video-diffusion SDS; (3) automatically compose foreground into background using monocular-depth–based position initialization and a camera-frustum–based scale heuristic, followed by optimization.

Experiments compare CP4D with physics-driven simulators, conditional video generators, and text-to-4D baselines, using VBench/WorldScore and an LLM-assisted evaluation; ablations indicate material and position optimization both contribute.

### Strengths
The pipeline of CP4D is clear and easy to follow. And for method, combining VLM-assisted physical initialization with SDS refinement is an interesting hybridization.

### Weaknesses
1. The motivation of CP4D is dividing static background and dynamic foreground, which is not novel in 4D generation area, such as [1,2,3].
2. The biggest concern of the pipeline is robustness. The pipeline depends on many off-the-shell models, especially the monocular depth  and a frustum heuristic part. The authors should give more examples to support the robustness of CP4D, while now only 17 simple prompts are listed.
3. The video results in supplementary material are not convincing, all cases just include very shot clip, and multiview examples do not show big camera motion exchange.
4. There is a typo in line75, where foreground is spelled as "foareground".

[1]. Comp4d: Llm-guided compositional 4d scene generation. Arxiv 24.03. Xu et al.
[2]. Compositional 3d-aware video generation with llm director. Nips2025. Zhu et al.
[3]. DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing. CVPR2024. Liu et al.

### Questions
Please see weakness.

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
4

### Confidence
4