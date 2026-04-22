# GaussGym: An open-source real-to-sim framework for learning locomotion from pixels

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
We present a photorealistic robot simulator that integrates 3D Gaussian Splatting as a drop-in renderer within vectorized physics simulators such as IsaacGym. This enables unprecedented speed—exceeding 100,000 steps per second on consumer GPUs—while maintaining high visual fidelity, which we showcase across diverse tasks. We additionally demonstrate its applicability in a sim-to-real robotics setting. Beyond depth-based sensing, our results highlight how rich visual semantics improve navigation and decision-making, such as avoiding undesirable regions. We further showcase the ease of incorporating thousands of environments from iPhone scans, large-scale scene datasets (e.g., GrandTour, ARKit), and outputs from generative video models like Veo, enabling rapid creation of realistic training worlds. This work bridges high-throughput simulation and high-fidelity perception, advancing scalable and generalizable robot learning, and allowing researchers to benchmark their visual locomotion algorithms. All code and data will be open-sourced for the community to build upon. Videos, code, and data are available on the project website: https://gauss-gym.com

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents GaussGym, an open-source framework for photorealistic robot simulation that integrates 3D Gaussian Splatting (3DGS) as a drop-in renderer within the vectorized physics simulator IsaacGym. GaussGym can create diverse training worlds by ingesting data from various sources: smartphone scans, large-scale scene datasets (e.g., GrandTour), and outputs from generative video models like Veo, followed by a standard reconstruction pipeline with GSplat. The authors demonstrate the framework's utility by training visual locomotion and navigation policies for humanoid and quadrupedal robots using RL directly from RGB observations.

### Strengths
1. The stated throughput of 100,000 steps per second with $640 \times 480$ resolution RGB/Depth rendering across 4,096 parallel environments is a state-of-the-art result for a photorealistic simulator.

2. Ingesting data from various data sources into the simulation environment is a good idea.

3. The open-sourced simulation framework can benefit the community.

4. The paper is generally well-structured and clearly presented.

### Weaknesses
1.  **Overclaim and Lack of Novelty.** The most significant weakness lies in the limited novelty compared to prior works [1][2]. The claim in L097 — “a first step toward closing the visual sim-to-real gap” — appears overstated, as previous research [2][3] has already demonstrated that visual RL policy training in simulation can help bridge this gap. Furthermore, the proposed “splat-integrated simulator for evaluating locomotion policies” concept is closely related to [2], which already explored real-to-sim-to-real frameworks for visual locomotion and navigation. 

2. **Limited Technical Contribution.** The proposed framework primarily integrates reconstructed 3DGS)scenes from diverse sources into a simulator, achieving 4,096 FPS at 640×480 resolution on a single RTX 4090 GPU. However, the high rendering speed advantage stems directly from the Gsplat renderer’s inherent parallelization, rather than from novel system design. Additionally, the overall visual policy training pipeline follows a design highly similar to [2], limiting the perceived technical innovation.

3. **Lack of Comparion for Visual Locomotion**.  Although the paper presents ablation studies demonstrating the benefits of the voxel grid head and DINO encoder, it lacks direct quantitative comparisons with state-of-the-art depth- or geometry-based locomotion policies (e.g., ANymal Parkour, Miki et al.). Metrics such as success rate or velocity tracking error on common benchmark terrains (e.g., stair climbing) are missing. The main quantitative table (Table 2) focuses solely on internal ablations and a “blind” baseline, without showing performance against geometric SOTA methods. Moreover, there are no comparisons to prior visual locomotion baselines [2][3], further weakening the experimental validation.

4. **Questionable Experimental Design for Visual Locomotion and Navigation.** The visual locomotion experiments, such as the stair climbing task, do not convincingly justify the need for an RGB-based policy, since depth-based parkour policies are also capable of handling similar tasks. While the authors claim that RGB-based policies capture semantic cues, the goal-tracking navigation task does not effectively demonstrate this advantage. To substantiate the argument, the paper should include higher-level semantic tasks (e.g., obstacle-type recognition or affordance-based navigation) that genuinely highlight the semantic reasoning benefits of RGB perception.

**References:**

[1] Xie, Ziyang, et al. "Vid2sim: Realistic and interactive simulation from video for urban navigation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Zhu, Shaoting, et al. "Vr-robo: A real-to-sim-to-real framework for visual robot navigation and locomotion." IEEE Robotics and Automation Letters (2025).

[3] Yu, Alan, et al. "Learning visual parkour from generated images." 8th Annual Conference on Robot Learning. 2024.

### Questions
1. The paper adopts VGGT for camera pose calibration and coarse point cloud extraction. While VGGT offers fast inference, its pose accuracy is typically lower than SfM and BA-based approaches such as COLMAP or GLOMAP. How does this reduced accuracy affect the downstream policy learning or simulation fidelity in your framework?

2. When deploying the trained visual locomotion policies on real robots, what kind of computational hardware is used? The DINO encoder appears relatively heavy for onboard inference — have the authors evaluated its runtime performance and feasibility for real-time deployment on embedded platforms?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes GaussGym, a simulation platform which uses Gaussian Splatting to render realistic settings, while collisions and dynamics are simulated by IsaacGym (or other such simulators). GaussGym provides a pipeline to convert data from iPhone scans, scene datasets, or generative video models into  realistic Gaussian splats for rendering and meshes for simulation. As a result, GaussGym is able to train locomotion policies in realistic environments with easy-to-capture data, all while maintaining a high 100k FPS. The realism of GaussGym's renderer is validated via sim2real locomotion experiments, where RGB sensor information allows for tasks which require color information unavailable in standard depth+proprioception observations.

### Strengths
- GaussGym's pipeline for generating trainable environments from easy-to-collect data (e.g. iPhone scans) is unique and provides a direction for simulating under diverse, but realistic scenes
- Additional considerations (e.g. motion blur) improve rendering realism and training speed (e.g. updating renders at real camera fps)
- GaussGym maintains high simulation speed across multiple scenes, necessary for legged locomotion research/tasks, which require many samples and fast training speed (for high training/testing iteration speed)
- Sim2real experiments validate realism while demonstrating benefits of RGB observations over standard depth+proprioception (e.g. avoiding colored penalty areas)

### Weaknesses
- The authors note diminished performance (e.g. foot placement) when transferring to the real world, however it is unclear to what extent this is due to issues with reward tuning/dynamics randomization vs the rendering quality of the simulator
- Currently, the environments lack some sim2real features like image latency, and the 3DGS rendering setup doesn't directly support some common simulation tools for visual domain randomization (e.g. texture and lighting randomization)

### Questions
- Works like [1] are able to achieve RGB sim2real while using pretrained visual encoders (and some visual domain randomization) despite low-quality rendering; in other words, the use of pretrained encoders makes it difficult to determine how much the successful sim-to-real transfer is caused by photorealistic rendering vs the pretrained visual representations. Have the authors tried training a sim2real policy off of direct RGB observations, without pretrained encoders? If so, were these policies successful?
- How straightforwardly can can GaussGym's simulation backend be swapped out to a sim other than IsaacGym (e.g. if new simulators are released with improved performance)?


[1] Ruihan Yang, Yejin Kim, Rose Hendrix, Aniruddha Kembhavi, Xiaolong Wang, Kiana Ehsani:
Harmonic Mobile Manipulation. IROS 2024: 3658-3665

### Soundness
3

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
2

### Summary
This paper uses gaussian splatting to bring realistic visuals into fast physics simulators. The authors help with adoption of their work by providing many environments from a variety of sources. It demonstrates that this new framework is beneficial through sim-to-real experiments in stair climbing (locomotion) and goal reaching (navigation).

### Strengths
- focuses on the important problem of bringing real world visuals to simulators to improve sim-to-real
- provides helpful visuals to understand the impact of the work
- provides datasets for potential adoption

### Weaknesses
- limited experimental results and settings
- existing experimental settings seem relatively simple, and may not necessarily require the high quality visuals from 3dgs in simulation 
- lack of ablations or baselines to show downstream benefits of improved simulation visuals
- paper highlights the potential benefits of pairing GaussGym with video generation models, but does not run experiments proving this

### Questions
My main concerns are with the experimental results, which are mentioned in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a real-to-sim approach to learning locomotion and navigation from pixels. The paper scales up real2sim visual data for training autonomous agents with a mix of data sources including iphone captures, video generative model outputs as well as existing multi-view datasets like arkit. The paper utilizes of off-the-shelf tools like VGGT and NeRFstudio to create 3D Gaussian Splats of these scenes which act as drop in renders for physics sim such as Isaac-gym. Experiments demonstrate sim2real transfer of locomtion policies. Other experiments demonstrate navigation can also be achieved with a similar method.

### Strengths
In my opinion, following are the strengths of the paper:

1. Scaling up photorealistic real2sim data for policy learning is a major advantage. While still in static scenes, it shows the utility of current zero-shot foundation models can be robustly used to scale up visual data to train policies. 

2. Zero-shot sim2real deployment is a nice result demonstrating the method is able to get good performance with democratized data collection such as with iphone cameras or existing captures. 

3. The paper is nicely written and the visuals/diagrams support and complement the text very well. 

4. A vectorized rendering support for existing physics sim is a great feature to have in existing physics simulators to scale up real2sim learning, albeit with an initial capturing overhead.

### Weaknesses
In my opinion, below are the weaknesses of the method:

1. Lack of comparisons to existing similar works in this space: I think the paper is lacking comparisons or discussions to existing closely related works [1,2,3]. A comparison interms of visual realism as well as PSNR as well as efficiency would be great for the community to understand which approach is the most useful in terms of ease of acquiring real2sim vs accuracy axes. 

2. The evaluation setting for the single-task multi-environment real2sim policy that the paper trains for their main results is unclear. Can the authors show results on a common benchmark, perhaps something similar to EmbodiedSplat [2] where it is clear what the training data distribution is and if there are any gains from training a large policy on all the real2sim collected data on unseen enviornments i.e. whether the approach is able to quickly finetune etc.?

3. With a high throughput, did the authors also experiment with real-world RL?

4. Are the physics parameter tuned to achieve sim2real transfer? Details of these are missing in the paper. 

[Minor]

4. I am curious does the same results hold for manipulation as well where other factors can come into play i.e. occlusions, visual fidelity of the embodiment itself i.e. hands considering the current embodiment is a synthetic one. 

[1] Xie et al. Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation, CVP 2025
[2] Yu et al. Real2Render2Real Scaling Robotic Manipulation Data Without Dynamics Simulation or Robot Hardware, CORL 2025 Oral
[3] Chablani et al. EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device, ICCV 2025

### Questions
See questions and comments in the weakness section. I am looking forward to author's responses in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
2
