# ShapeGen4D: Towards High Quality 4D Shape Generation from Videos

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Video-conditioned 4D shape generation aims to recover time-varying 3D geometry and view-consistent appearance directly from an input video. 
In this work, we introduce a native video-to-4D shape generation 
framework that synthesizes a single dynamic 3D representation end-to-end from the video.
Our framework introduces three key components based on large-scale pre-trained 3D models:  (i) a temporal attention that conditions generation on all frames while producing a time-indexed dynamic representation; 
(ii) a time-aware point sampling and 4D latent anchoring that promote temporally consistent geometry and texture; 
and (iii) noise sharing across frames to enhance temporal stability. 
Our method accurately captures non-rigid motion, volume changes, and even topological transitions without per-frame optimization. Across diverse in-the-wild videos, our method improves robustness and perceptual fidelity and reduces failure modes compared with the baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ShapeGen4D, a feedforward framework for high-quality 4D shape generation from monocular videos. The method builds upon a pre-trained 3D generative model (Step1X-3D) and adapts it to the dynamic setting through three key innovations:

1. Spatiotemporal attention layers to capture cross-frame dependencies;
2. Temporally-aligned latent encoding to reduce jitter and improve consistency;
3. Shared noise across frames to enhance temporal stability.

ShapeGen4D directly outputs a sequence of 3D meshes, supporting non-rigid motion, volume changes, and topological transitions without per-frame optimization.

### Strengths
1. End-to-end 4d generation framework. The paper presents a direct, feedforward approach for generating 4D mesh sequences from a single video, which is a simplification over more complex optimization-based or multi-stage pipelines.
2. Handling of complex dynamics. The framework demonstrates the capability to generate a range of dynamic phenomena, including non-rigid motion and topological changes, which are challenging for methods restricted to simpler deformations.
3. Clear written. The paper is clearly written and provides a well-organized summary of related work, helping to situate the contributions within the field of 4D generation.

### Weaknesses
1. Inherent Limitations in Temporal Geometry Consistency. While the proposed techniques of latent alignment and noise sharing effectively reduce temporal jitter, they do not establish an explicit, parametric model of motion (e.g., a deformation field). The framework still generates each frame's mesh independently from a sequence of latents. This inherently discrete representation may struggle to guarantee as-rigid-as-possible or physically plausible transitions over time, potentially leading to subtle topological inconsistencies or unnatural deformations that are not explicitly regularized. This limitation is indirectly acknowledged by the authors, who note that "local temporal jitter remains visible in some results."
2. Limited Scalability to Long Video Sequences. The model is designed to generate a fixed-length sequence (e.g., 16 frames) in a single forward pass, constrained by the memory and architecture of the underlying diffusion transformer. This fixed-horizon generation prevents the method from processing arbitrarily long videos, a common requirement in real-world applications. The paper does not explore mechanisms for temporal auto-regressive generation or sliding-window inference, which would be necessary to scale to longer durations, potentially at the cost of error accumulation across segments.

### Questions
1. Since the meshes for consecutive timesteps are generated independently from discrete latents, the resulting 4D sequence lacks an explicit, continuous deformation field. This may lead to non-smooth interpolations and visually incoherent dynamics when rendered at frame rates higher than the generation rate. Do you have plans to incorporate a post-processing step or an intermediate representation (e.g., a neural deformation field) to enable truly continuous, smooth morphing between the generated key meshes? How might this be integrated into your current pipeline?
2. The current framework generates a fixed number of frames (e.g., 16) in a single feedforward pass. What is the potential of extending ShapeGen4D to handle arbitrarily long videos? For instance, could an autoregressive approach be adopted, where the generation of a subsequent clip is conditioned on the final frames of the previous clip, similar to the strategy employed by L4GM? If so, what would be the main technical challenges, such as error accumulation or maintaining global consistency across segments?

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
4

### Summary
The paper introduces ShapeGen4D, a direct, feedforward framework for producing 4D shape sequences from monocular video input. The method extends an existing 3D shape diffusion model into the temporal domain through several architectural modifications. Firstly, the authors obtain temporally aligned shape latents through a time-aware point sampling strategy. Then, the pretrained shape DiT is finetuned with incorporation of spatiotemporal attention. During training, all shape frames share same noise to enhance temporal stability. Extensive experiments on diverse benchmarks demonstrate the strong qualitative and quantitative results.

### Strengths
* Conceptual originality: the first to adapt pretrained 3D shape diffusion generator for 4D shape sequence generation.
* End-to-end pipeline: the proposed method produces temporally coherent mesh sequences directly from video, avoiding costly optimization.
* Simple but effective modification: the model builds on a well-known 3D backbone with several straightforward and effective architectural modifications, achieving quantitative and qualitative improvements compared to baseline methods.

### Weaknesses
* Underlying gap with respect to pretrained prior: the queries of latents are sampled from non-watertight mesh, while the base 3D backbone is pretrained on watertight queries. Although the authors have explained the reason to do so (to avoid costly mesh registration), the potential performance drop still exists due to this gap.
* No explicit motion modeling: the generated mesh under each frame is independent with each other. The lack of explicit motion constraint cannot guarantee physically plausible, leading to minor jitter and limiting the application to continuous motion interpolation.
* Technically lack of novelty: the main components of this framework are mostly common practices in this field.

### Questions
* Current design only handles short clips (e.g., 16 frames), have the authors considered scaling the model to longer videos?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper works on 4d shape generation from monocular video. This is a new task and the authors aim to generate per-frame yet temporally coherence mesh instead of static mesh and deformation field to accommodates variantions in topology and relaxe constrains on the type of possible animations. Specifically, they construct the 4d shape generation model on a 3d shape generatoin work, and add temporal attention layers to ensure temporal consistency. Besides, they use global pose registration and global texturation as post-processing steps to better present the generated results and evaluation. Experiments on public and collected datasets validate the performance of the proposed model.

### Strengths
1) the generated shape seems to be of high-quality in terms of single-frame results, probably benifiting from high-quality pretrained 3D generation model weight

2) the paper is well-written and present its contribution in a clear way

3 )the authors work an new task, and propose a new pipeline of 4D shape generation

### Weaknesses
1) The generated results show noticeable flickering artifacts in both geometry and texture, where the texture flickering may stem from the instability in geometry.
WPOwpo
2)  While the authors claim that per-frame mesh generation is intended to capture variations in topology and enable a wider range of animations, the paper provides only a single example illustrating this capability (the BANG case in the supplementary material). The remaining examples appear to be rendered from skeleton-based animation models without topology changes and with a limited diversity of animation types. The authors are encouraged to present examples of topology-changing animations include: object shattering, characters growing extra limbs, soft-body fusion or splitting, cloth tearing, and morphing into an entirely different mesh structure.

3) Lack of comparison with dynamic mesh generation method, for example DreamMesh4D and DriveAnyMesh, both are video-4d mesh generation method. Besides, the authors are encouraged to compare the generation result with general video-4d method beyond L4GM.

### Questions
It is unclear why the proposed method does not incorporate data augmentation using geometric or spatial transformations during training, which might have alleviated or removed the requirement for global registration.
If my concerns are solved, I'll raise the recommendation.

### Soundness
2

### Presentation
2

### Contribution
2
