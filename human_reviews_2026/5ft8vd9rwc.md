# LightCtrl: Training-free Controllable Video Relighting

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Recent diffusion models have achieved remarkable success in image relighting, and this success has quickly been reproduced in video relighting. Although these methods can relight videos under various conditions, their ability to explicitly control the illumination in the relighted video remains limited. Therefore, we present \name, the first controllable video relighting method that offers explicit control over the video illumination through a user-supplied light trajectory in a training-free manner. This is essentially achieved by leveraging a hybrid approach that combines pre-trained diffusion models: a pre-trained image relighting diffusion model is used to relight each frame individually, followed by a video diffusion prior that enhances the temporal consistency of the relighted sequence. In particular, to enable explicit control over dynamically varying lighting in the relighted video, we introduce two key components. 
First, the Light Map Injection module samples light trajectory-specific noise and injects it into the latent representation of the source video, significantly enhancing illumination coherence with respect to the conditional light trajectory. 
Second, the Geometry-Aware Relighting module dynamically combines RGB and normal map latents in the frequency domain to suppress the influence of the original lighting in the input video, thereby further improving the relighted video's adherence to the input light trajectory. 
Our experiments demonstrate that \name can generate high-quality video results with diverse illumination changes closely following the light trajectory condition, indicating improved controllability over baseline methods. The code will be released at: https://github.com/GVCLab/LightCtrl.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-free method for light trajectory editing in videos. The approach extends the single-image relighting model, IC-Light, to the video domain by incorporating priors from video diffusion models to ensure temporal consistency. The core technical contributions include a light map injection module, which introduces trajectory-aware noise into the VDM's latent space, and a geometry-aware relighting module. This second module processes both RGB frames and corresponding normal maps estimated via StableNormal to guide the relighting process. The results are visually compelling and demonstrate good adherence to the specified lighting trajectories.

### Strengths
1. The paper tackles a relatively unexplored task of lighting control in video diffusion models rather than just global style. The direction, ideas, and results are promising and the problem is quite interesting.
2. By building on existing, well-known T2I and T2V models, the method is accessible and its components are easier to understand and potentially replicate.
3. The strategy of injecting trajectory-aware noise into the VDM's initial latent space appears effective for guiding the lighting. This concept may be adaptable to other conditional video generation tasks.
4. The design of the geometry-aware relighting module, which dynamically blends RGB and normal map information, is technically sound. It correctly reflects that surface geometry should remain invariant during a relighting task.

### Weaknesses
1. The use of PSNR_y against a pure white reference is not convincing. Relighting is a complex task involving light, geometry, and material, and does not necessarily mean "the brighter the better". This metric mainly captures pixel-wise brightness in 2D space and fails to account for the geometric and directional correctness of the illumination. It cannot distinguish excessive lighting and ignores geometry and material reflectance.
2. The evaluation is somewhat limited. The dataset contains only 50 videos. This small scale and likely limited diversity are insufficient to robustly validate the method's generalizability. The paper would significantly benefit from showcasing more results across a wider variety of video content and lighting conditions.

Minor issues:

There are plenty of papers about generative portrait relighting. Though they deal with portrait videos, I think they are still related to this topic. It is better to cite and discuss these papers. Following are some examples:

+ Lumos: Learning to Relight Portrait Images via a Virtual Light Stage and Synthetic-to-Real Adaptation

+ Neural Video Portrait Relighting in Real-time via Consistency Modeling

+ Real-time 3D-aware Portrait Video Relighting

### Questions
1. In Eq. (3), what exact $\omega$ values are used across videos and trajectories? Is it fixed or tuned per video?
2. Could you elaborate on the temporal stability of the StableNormal predictions? Are the normal maps computed independently per frame and fed directly into the video VAE, or is some form of temporal smoothing or other consistency enforcement applied?

### Soundness
3

### Presentation
4

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
This paper introduces LightCtrl, a training-free controllable video relighting framework that enables explicit, fine-grained control of video illumination via a user-specified light trajectory. The method combines two pre-trained diffusion models an image relighting model (IC-Light) and a video diffusion model (VDM, e.g., AnimateDiff) to achieve temporally coherent and controllable lighting effects without retraining.

The method achieves strong quantitative and qualitative improvements over baselines such as IC-Light, SDEdit, and Light-A-Video, showing superior controllability, temporal coherence, and visual quality.

### Strengths
1. Originality: The proposed combination of LMI and GAR modules represents a creative hybridization of diffusion-based image and video models, enabling spatio-temporal control without additional data or model fine-tuning. 

2. Technical Quality. The methodology is clearly formulated. The Light Map Injection method is grounded in the diffusion process and introduces a principled way to guide illumination through noise manipulation. The Geometry-Aware Relighting component is well-justified. Leveraging normal maps via frequency-domain fusion to mitigate original light leakage is both technically sound and physically motivated.
Experiments are comprehensive and clear.

3. Clarity. The paper is well-written and visually rich, with clear figures illustrating the pipeline and ablation effects.

4. Significance. This paper advances controllable generation from static to dynamic illumination domains, bridging image and video diffusion paradigms.

### Weaknesses
1. Physical Realism and 3D Awareness. The paper acknowledges limited 3D understanding of illumination. The method cannot simulate light scattering, occlusion, or volumetric effects, which restricts realism when the light trajectory crosses 3D geometry.

2. Geometry-Aware Relighting (GAR) module. Although the GAR module suppresses source illumination, it fuses normal and RGB latents via a frequency-domain filter with dynamically decreasing cutoff. But there is no quantitative evidence that this schedule balances structure preservation and lighting suppression optimally.

3. Limited Analysis of Control Robustness. The method only test on linear, circular and top–bottom light trajectory. More trajectorys should be tested.


.

### Questions
1. GAR module. Empirically, the low-frequency fusion could still introduce temporal blurring or artifacts around illumination edges. A systematic perceptual analysis or patch-level FVD comparison would strengthen confidence in this design

2. Light trajactory. Can this method be applied to : 1. Abrupt trajectory changes 2.Multiple moving light sources 3. Unaligned control inputs? Such tests would reveal whether LightCtrl’s latent-space conditioning is robust to domain shifts or natural illumination variance.

3. Transferability across video diffusion backbones: Have you tested the method on alternative video diffusion priors (e.g., CogVideoX, VideoCrafter2)? Does controllability degrade or improve when switching to models with different motion representations or schedulers?

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
This paper proposes a new method for video relighting that is training-free and allows users to provide a light trajectory as lighting conditions for the video. The authors make changes to the existing video diffusion framework as well as the image relighting framework (i.e., IC-Light) to allow them to use the user-provided light trajectory. The quality of the video relighting is evaluated with comparison to IC-Light and Light-A-Video. A user study is also conducted with 40 volunteers on video smoothness, lighting controllability, lighting quality, and alignment between lighting and text aspects of the relighting videos.

### Strengths
- The idea of designing a training-free method to control the lighting of a video generation model is interesting.

- It is known that the common metrics for relighting might not faithfully reflect the visual quality of the relighted results. The authors design and conduct a user study with 40 volunteers, making the evaluation more rigorous.

### Weaknesses
- It is unclear how significant it is to have a user-provided light trajectory for relighting. The authors explain their motivation for using manually labeled light trajectories in lines 069-072, but there is no supporting evidence to indicate that this is a desired or much-needed feature for users. 

- The user study is very important in evaluating the performance of the relighting methods. But many important details are not there in the current draft: how are those volunteers selected? Do they understand the relighting task? How are they trained on the provided metrics (e.g., lighting controllability)? How reliable are their answers? Is the result statistically significant? 

- The proposed method is designed to handle light trajectories in the relighting video task and is evaluated with the same type of light trajectory data (lines 315-318). This testing seems to be limited. There are many other ways to control lighting in relighting methods (e.g., environment maps or added light sources at certain locations in the image). It is currently unclear how this method compares to those more general cases of lighting conditions. 

- The writing of the paper could be improved. There are typos and formatting issues in the paper. In particular, in line 208 it says the initial noisy latent is $\hat{z}_m$, but line 243 says the initial noisy latent is $z_m$.

- Finally, being training-free is of course attractive, but the current design mostly uses existing video diffusion model and IC-light to do the heavy-lifting while the new parts are preparing the trajectory to the video diffusion model. The difference between the variants of LightCtrl (Figure 5, last three rows) is not that substantial. This makes it unclear about the significance of the proposed method.

### Questions
1. What is the supporting evidence to indicate that having a user-provided light trajectory is a desired feature for relighting tasks? 

2. Can the authors provide more details about the user study? Please see the weaknesses above for questions. 

3. Can the authors discuss more about the significance of the approach, given the ablation study? If the authors can provide more rigorous ablation with more insight, it will be better.

### Soundness
3

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
4

### Summary
This paper tackles the controllable video relighting task in a training-free manner. For this, LightCtrl builds on a pre-trained single-image relighting model, i.e., IC-Light, and tailors it to video relighting with several designs. First, it proposes to conduct frame-wise relighting on the video diffusion model's paired VAE decoder. Besides, it proposes to progressively fuse the source video and relit video during the diffusion process to maintain temporal coherence. Further, a user-defined light trajectory injection module is applied. Finally, they use off-the-shelf normal estimation to provide the video's normal maps to enable a geometry-aware relighting module. Experiments demonstrate the effectiveness of the proposed approach.

### Strengths
- originality-wise: the idea of using a single-image relighting model to enable video relighting is interesting.
- quality-wise: the qualitative and quantitative results are promising.
- clarity-wise: the presentation is good.
- significance-wise: the video relighting is important for a lot of downstream tasks, e.g., content creation.

### Weaknesses
I feel the ablations in the paper are quite inadequate. Even though there are some ablations in the appendix that are good, they are not the core part of the model design.

For example, can authors provide both **quantitative and qualitative** results to show the gradual improvement from the pre-trained IC-Light? From my understanding, there are several enhancements, but I have no clue which one contributes. Feel free to add things that I miss.
- decoded frames + residual instead of raw frames (L216)
- progressive fusion in Eq. (2)
- geometry-aware feature in latent space
- geometry-aware in frequency space.

I am actually confused why not directly use the raw frames in the source video? Since the authors add back the difference between the raw videos and the decoded videos (L216), isn't this the same as just using the original video?

### Questions
See "weakness"

### Soundness
2

### Presentation
3

### Contribution
2
