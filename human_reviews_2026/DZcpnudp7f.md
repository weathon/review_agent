# MoCa: Modeling Object Consistency for 3D Camera Control in Video Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Camera control is important in text-to-video generation for achieving realistic scene navigation and view synthesis. 
This control is defined by parameters that describe movement through 3D space, thereby introducing 3D consistency into the generation process.
A core challenge for existing methods is achieving 3D consistency within the 2D pixel domain.  Strategies that directly integrate camera conditions into text-to-video models often produce artifacts, while those relying on explicit 3D supervision face challenges with generalization. 
Both limitations originate from the gap between the 2D pixel space and the underlying 3D world.
The key insight is that the projection of a smooth 3D camera movement produces consistency in object view, appearance, and motion across 2D frames. Inspired by this insight, we propose MoCa, a dual-branch framework that bridges this gap by modeling object consistency to implicitly learn 3D relationships between the camera and the scene.
To ensure view consistency, we design a Spatial-Temporal Camera Encoder with Plücker embedding, which encodes camera trajectories into a geometrically grounded latent representation. For appearance consistency, we introduce a semantic guidance strategy that leverages persistent vision-language features to maintain object identity and texture across frames. To address motion consistency, we propose an object-aware motion disentanglement mechanism that separates object dynamics from global camera movement, ensuring precise camera control and natural object motion.
Experiments show that MoCa achieves accurate camera control while preserving video quality, offering a practical and effective solution for camera-controllable video generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MoCa, a dual-branch diffusion-transformer framework for camera-controllable text-to-video generation. The framework targets to maintain consistency in terms of view, appearance and motion. Concretely: (1) a Spatial-Temporal Camera Encoder (ST-Encoder) encodes per-pixel camera rays using Plücker coordinates and fuses them via cross-attention into the DiT to improve view consistency; (2) a semantic guidance path (ReferenceNet) injects frozen vision-language features to stabilize appearance; (3) an object-aware motion disentanglement uses a high-frequency (2D-DWT) mask over VL features to separate local object motion from global camera motion for motion consistency. The model is fine-tuned from CogVideoX on RealEstate10K and evaluated on RealEstate10K (mostly static) and VidGen (dynamic): MoCa shows better or competitive camera controllability and object/background consistency (e.g., RotErr/OC/BC), and ablations support each component.

### Strengths
+Clear factorization of “consistency": Framing camera control around view/appearance/motion consistency is intuitive and aligns well with qualitative failures of prior work 

+Appearance stabilization via VL features. The ReferenceNet “semantic guidance” improves object identity/texture stability;

+The high-frequency, object-aware mask improves OC/BC and motion plausibility when the camera is moving;

+Details are provided for reproducing the method: The paper lists training details and links to code/resources.

### Weaknesses
-Novelty vs. prior camera-conditioning is incremental.

The paper adopts Plücker coordinates for camera rays (as in CameraCtrl) and fuses conditions into DiT blocks (as in AC3D/VD3D). What is new in the ST-Encoder beyond adding spatial/temporal convs before cross-attention, and how does it differ from CameraCtrl’s/AC3D’s conditioning path are not addressed.

-Missing evaluation for proposed component for view consistency

It’s plausible that per-pixel rays + temporal convs should help view (and maybe motion) consistency, but the paper doesn’t directly measure “consistency” improvements from the ST-Encoder alone (e.g., removing temporal convs, or comparing addition vs. attention per consistency metric). 

-Object-aware mask lacks important details/evaluation.

The mask comes from a 2D-DWT over foundation features, but key questions remain:
* How about multiple moving objects? How are several instance regions separated if VL features are class-level and not instance-level? 

* Static vs moving distinction. How does the method decide that an object is static vs moving. Are static objects still routed through the disentangling branch; if so, any degradation?

* User-specified object motion. Can users specify independent object motion (e.g., “a bear walks left-to-right while the camera dollies forward”), or is motion always unsupervised/implicit? Motion-control scope and ability of the proposed disentanglement method are needed, given that it is one of the major contribution. 

-Complexity/overhead not fully discussed.
The paper should report latency overhead from dual-branch fusion, and the cost/benefit of 2D-DWT during inference. Training uses 16×H200; inference speed vs. baselines would be helpful for practical adoption.

### Questions
Please see the Weakness section above.

### Soundness
3

### Presentation
2

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
The paper tackles camera-controllable text-to-video (T2V) generation. The core claim is that smooth camera motion in 3D should manifest as consistent object view, appearance, and motion in 2D frames. MoCa proposes a dual-branch framework with:
(1) a Spatial-Temporal Camera Encoder using Plücker ray embeddings to inject geometry-aware camera signals for view consistency
(2) a semantic guidance path that feeds vision-language features from ReferenceNet into the denoiser to stabilize appearance
(3) an object-aware motion disentanglementto separate object dynamics from global camera motion for motion consistency

Experiments on RealEstate10K (static) and VidGen (dynamic) show improved camera control and object stability versus MotionCtrl, CameraCtrl, and AC3D, with ablations for each design choice.

### Strengths
- Paper is clearly written and easy to understand
- Motion disentanglement through DWT is neat. Additional discussions in the appendix help strengthen the authors' claims.
- Under a uniform 16-frame protocol, the method achieves top-rank or second-best mixes on RealEstate10K, and on VidGen improves key control/consistency metrics (RotErr, OC, CLIPSIM).
- Ablations explicitly test fusion choice (cross-attention vs addition) and discuss alternatives in the appendix, helping attribute gains to the proposed architecture rather than incidental training details. This strengthens causal claims about the design.

### Weaknesses
- Coverage of recent SOTA baselines is limited. While AC3D is included, other very recent transformer-based or geometry-aware camera-control methods cited in the text (e.g., VD3D, CamCo, ViewCrafter, CameraCtrl II) are not in the main comparison tables; this weakens claims of state-of-the-art across the latest literature.
- Camera accuracy evaluation relies on Mega-SAM reconstructions. It would help to quantify Mega-SAM failure rates or uncertainty propagation.
- The 16-frame evaluation setting seems to be a bit short in 2025. Recent models have shown to handle longer clips well; reporting an additional 32 to 48 frame setting (even on a subset) would better reflect practical camera control usage.

### Questions
In addition to the weakness section above:

- How sensitive is MoCa to camera path magnitude and frame count?

- What fraction of scenes yield Mega-SAM tracking failures, and how do you handle them? 

- What is the runtime speed compared to the baselines?

### Soundness
3

### Presentation
3

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
This paper introduces MoCa, a framework for camera-controlled video generation that improves object stability by focusing on maintaining consistency in view, appearance, and motion. The model uses a dual-branch architecture with semantic guidance to preserve object identity and a disentanglement mechanism to separate object dynamics from camera movement.

### Strengths
1. The video examples shown in the paper supp are strong. The generated videos look much more stable and realistic than the comparison methods.
2.  The main idea of framing the problem around "object consistency" (view, appearance, and motion) is a smart way to tackle the challenge. It breaks down a complex 3D problem into more manageable 2D properties that we can observe in the final video. 
3.  The method for separating object motion from camera motion is clever. Using a 2D Discrete Wavelet Transform (2D-DWT) to create a "high-frequency object-aware mask" is an interesting technical contribution.

### Weaknesses
1. The paper relies on Object Consistency (OC) and Background Consistency (BC) scores from VBench to prove its main contribution. However, as the VBench paper itself explains, these metrics just measure feature similarity (using DINO and CLIP) across frames. This means they mainly check if an object is consistently present, not if its motion is natural or if it's free from distortion. A video with a "frozen" object sliding unnaturally across the screen could still get a high OC score, which doesn't really support the claim of improved motion consistency.
2. The method is built by fine-tuning CogVideoX, a very large (~5B) foundation model. While this helps achieve impressive results, it makes it hard to judge how much of the performance comes from the new MoCa architecture versus the power of the base model, especially considering the nature of randomness in the generation.
3. The paper could use another round of proofreading. There are several places where the citation formatting is incorrect (e.g., in Section 4.1, should be a proper \citep command). These small errors, along with some sections that are a bit dense, can make the paper harder to read and feel less polished, such the following part:

   - The semantic guidance strategy (Section 3.2) uses a ReferenceNet to maintain object identity. The paper says it uses "reference video frames" (line 228), but it's not clear what this means in practice. Is the input just the first frame of the video, or a specific set of keyframes? This detail is crucial for understanding how the model gets its "identity guidance."
   - The high-frequency object-aware mask is a key part of the motion disentanglement. Could you provide more detail on how this mask is actually used in the "Hybrid Condition Fusion" step (line 263)? For example, is it used as a soft attention map to guide the DenoisingNet, or is it combined with other features in a different way? A clearer explanation of this mechanism would be very helpful.

### Questions
- The semantic guidance strategy (Section 3.2) uses a ReferenceNet to maintain object identity. The paper says it uses "reference video frames" (line 228), but it's not clear what this means in practice. Is the input just the first frame of the video, or a specific set of keyframes? This detail is crucial for understanding how the model gets its "identity guidance."
   - The high-frequency object-aware mask is a key part of the motion disentanglement. Could you provide more detail on how this mask is actually used in the "Hybrid Condition Fusion" step (line 263)? For example, is it used as a soft attention map to guide the DenoisingNet, or is it combined with other features in a different way? A clearer explanation of this mechanism would be very helpful.

### Soundness
3

### Presentation
3

### Contribution
3
