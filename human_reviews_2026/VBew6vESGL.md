# Light-X: Generative 4D Video Rendering with Camera and Illumination Control

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Recent advances in illumination control extend image-based methods to video, yet still facing a trade-off between lighting fidelity and temporal consistency. Moving beyond relighting, a key step toward generative modeling of real-world scenes is the joint control of camera trajectory and illumination, since visual dynamics are inherently shaped by both geometry and lighting. To this end, we present Light-X, a video generation framework that enables controllable rendering from monocular videos with both viewpoint and illumination control. 1) We propose a disentangled design that decouples geometry and lighting signals: geometry and motion are captured via dynamic point clouds projected along user-defined camera trajectories, while illumination cues are provided by a relit frame consistently projected into the same geometry. These explicit, fine-grained cues enable effective disentanglement and guide high-quality illumination. 2) To address the lack of paired multi-view and multi-illumination videos, we introduce Light-Syn, a degradation-based pipeline with inverse-mapping that synthesizes training pairs from in-the-wild monocular footage. This strategy yields a dataset covering static, dynamic, and AI-generated scenes, ensuring robust training.
Extensive experiments show that Light-X outperforms baseline methods in joint camera-illumination control and surpasses prior video relighting methods under both text- and background-conditioned settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript presents Light-X, a framework achieving end-to-end relighting and redirection given a monocular video. Following TrajectoryCrafter, the proposed architecture retains channel-wise concatenated point-cloud rendering on desired novel camera trajectories and Ref-DiT blocks that incorporates the raw reference video via cross-attention. For illumination control, a frame from the input video is first relighted using IC-Light to obtain a sparse relit video, which is then compressed into a set of illumination tokens via a Q-Former and consumed by a Light-DiT block. The reprojected sparse relit video is also concatenated with the input noise. Additionally, this paper proposes a degradation-based data curation pipeline, Light-Syn, to create training data from in-the-wild videos. Experimental results show that the proposed end-to-end approach outperforms two-stage baselines composed of existing relighting and redirection method on the joint camera–illumination control task, while maintaining flexibility for decoupled control and diverse lighting conditions.

### Strengths
1.	The paper is well-structured and easy to follow.
2.	It defines a new task, joint camera-illumination control, which has not been explored by previous works, and designs an end-to-end solution achieving impressive results far superior to naïve combinations of existing methods.
3.	The effort on data curation is commendable.

### Weaknesses
1.	Although the method demonstrates clear superiority that direct combination of existing redirection and relighting methods, it is insufficient to convincingly justify the necessity of their end-to-end design. As mentioned in L427, the bottleneck of the TC + LAV baseline lies in the weak relighting capability of the latter. Could it be mitigated by a stronger video-relighting model? Moreover, since the Light-Syn also adopted the TC+LAV approach to produce the source video as shown in Figure C, would it be possible to replace it with the trained Light-X model to generate higher-quality data and then improve the training of Light-X in turn?
2.	The proposed data curation pipeline only relies on the degradation of in-the-wild videos,  potentially introducing a train–inference gap in the distribution of source video and relationship between source and relit videos. For example, the geometry between source and relit videos should be identical in the ideal training data, but TrajectoryCrafter may introduce some deviations. Why not incorporate some accurate synthetic data from graphic engines like IC-Light or RelightVid?

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
3

### Summary
The paper proposes Light-X, a video diffusion framework that enables joint control of camera trajectory and illumination from a monocular input video. The method explicitly decouples geometry/motion from lighting: dynamic point clouds projected along user-specified camera paths provide geometric cues, while a single relit frame is re-projected through the same geometry to supply illumination cues. To obtain training pairs without multi-view or multi-illumination captures, the authors introduce Light-Syn, a degradation pipeline that turns in-the-wild videos into paired inputs/targets and aligned conditioning renders/masks. Experiments show improvements over composed baselines for the new joint camera+illumination task and competitive results for video relighting under text and background conditions, with extensive ablations.

### Strengths
- The proposed factorization method supplies the model with fine-grained, geometry-aligned cues (projected source views/masks and projected relit views/masks) and complementary global illumination tokens (Q-Former), which are technically sound and easy to reason about.
- For the new joint control task (no direct prior), the paper composes reasonable baselines from camera-control and relighting methods, and also introduces a tailored training-free baseline with documented adaptations.
- The evaluation method and metrics seem solid. The authors fix random seeds over prompts, light directions, and trajectories across methods; metrics cover fidelity (FID and Aesthetic Preference) and temporal aspects (CLIP similarity across frames, Motion Preservation via RAFT). A 57-person user study evaluates Relighting Quality, Video Smoothness, ID Preservation, and 4D consistency.
- Ablations show the benefits of different design choices proposed in the paper.

### Weaknesses
- For joint control, FID is computed against IC-Light-relit outputs on a TrajectoryCrafter sequence. That anchors “ground truth look” to IC-Light’s aesthetics and may bias the metric toward that method’s style.
- The related work mentions several recent camera/lighting control or camera-controlled generators (e.g., VidCraft3, ReCamMaster, CAMI2V, Free4D/VD3D) that appear not to be included in quantitative comparisons. A brief justification or attempts to compare to the most recent DiT-based camera-control systems (or to lighting-control contemporaries) would strengthen claims of SOTA.
- Light-Syn mixes static, dynamic, and AI-generated sources. While this increases diversity, it risks domain confounds (e.g., the model learning priors from GenAI models). Cross-domain breakdowns (real-only vs AI-only) would clarify generalization.

### Questions
In addition to the points listed in the weakness section, I have some additional questions:

- The method relights only one frame to build lighting cues, then relies on Light-DiT for global consistency. It would help to quantify how performance drops as temporal distance grows, or when the relit frame has occlusions/poor depth.
- How robust is Light-X to camera intrinsics errors and depth noise? A small controlled study would strengthen the geometric claims.
- What motivates using the first frame as the relighting frame? Is there a better way of choosing a reference frame from a video, as sometimes the first frame might not be the most informative?

### Soundness
2

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
This paper adds joint control of camera trajectory and illumination to render from monocular videos. They use a disentangled design to decouple geometry and lighting. They also use a degradation-based pipeline with inverse-mapping to synthesizes training pairs.
a clear understanding of the limitations of prior approaches, a well-defined methodology with ablations for every aspect, and valid baselines for evaluation. Starting with an input video, two sets of inputs for diffusion are generated: Camera control and Illumination control. These priors are used as conditions for video diffusion, passed through DiT blocks for denoising. They also introduce a Light-DiT layer for global illumination control as they observe that illumination strength diminishes as frames move further away from the relit frame. Experiments are sound and the results are very good.

### Strengths
- First paper to tackle the novel problem of video generation with joint camera and illumination control for monocular videos by providing conditioning to a diffusion transformer.
- The Light-Syn pipeline uses an effective degradation idea for training data creation. The data sources comprise static scenes, dynamic scenes, and AI-generated videos with ablations justifying the significance of each source.
- Light-DiT layer allows global illumination control by using a Q-Former to prevent diminishing illumination strength for frames that are further away from the relit frame.
- Paper is well written and easy to follow.

### Weaknesses
No major weakness.

Minor Weakness: The evaluation metric using FID between the output image and IC-Light would have a potential evaluation bias, as the model is judged on its ability to mimic the behavior of a component (IC-Light) used in its own conditioning scheme.

### Questions
How would the model perform for camera trajectories significantly different from the source? The DiT blocks would need to in-paint a large area with little conditioning, and would errors in this propagate to the next frames?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work presents a method, Light-X, for generative 4D video rendering, enabling the disentanglement of scene geometry and illumination in dynamic scenes. Light-X allows for controllable video relighting and camera trajectory redirection from monocular video inputs. Experiments show that the introduced model successfully achieves visually impressive relighting and redirection results across various tests.

### Strengths
* Light-X aims to achieve the very challenging task of disentangling geometry and illumination for dynamic scenes, yet it manages to achieve quite impressive visual results, as shown in the provided video.

* This work also introduces an easy-to-setup data curation pipeline for creating paired original and relighted videos of geometrically coherent dynamic scenes.

* The manuscript is well-structured and easy to follow.

### Weaknesses
* **Limited technical contribution within a complex framework.** While the controlled and reasonable generation results are impressive, the overall system appears to be a loose combination of prior works like TrajectoryCrafter and IC-Light. The authors should better detail why jointly controlling both camera and illumination is an important task and which specific module Light-X introduces to better fit this combined task.

* **Need for direct geometry comparison and evaluation.** The paper claims that the relighted video "remains geometrically coherent," but no direct experiments are provided to support this crucial claim. A stronger evaluation would involve running the input and relighted videos through methods like VGGT (for static scenes) or CUT3R (for dynamic scenes) to obtain corresponding point clouds. The authors could then calculate the Chamfer distance between these paired point clouds and provide visualizations for better qualitative assessment.

* **Deeper analysis required for the global illumination control module.** The global illumination control module is introduced to prevent the diminishment of illumination effects. To illustrate its contribution more clearly, a visual comparison from an ablation study of this module should be provided. Furthermore, the authors are suggested to analyze the module's effect on regions with non-Lambertian surfaces. I am concerned that enforcing the same global light effects might negatively impact performance in regions that naturally reflect differently under various lighting conditions.

### Questions
Kindly refer to the [Weaknesses] section.

### Soundness
3

### Presentation
3

### Contribution
2
