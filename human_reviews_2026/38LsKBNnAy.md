# BlenderFusion: 3D-Grounded Visual Editing and Generative Compositing

- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
We present BlenderFusion, a generative visual compositing framework that recomposes objects, camera, and background to synthesize new scenes. It follows a layering-editing-compositing pipeline that (i) segments and converts visual inputs into editable 3D entities (layering), (ii) edits them in Blender with 3D-grounded control (editing), and (iii) fuses them into a coherent scene using a generative compositor (compositing).
The generative compositor extends a pre-trained diffusion model to process both the original (source) and edited (target) scenes in parallel, and is fine-tuned on video frames with two important training strategies: (i) source masking, enabling flexible modifications like background replacement; (ii) simulated object jittering, facilitating disentangled control over the objects and camera.
Extensive experiments on synthetic and real-world datasets show that BlenderFusion significantly outperforms prior methods in precise 3D-aware control and complex compositional scene editing. The framework also generalizes to unseen data and fine-grained editing operations beyond the training distribution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BlenderFusion, a framework for 3D-aware multi-object image generation. From the perspective of controllable generation, this work enables controlling the 3D pose and appearance of objects in an image. From the perspective of tool use in foundation models, this work leverages Blender as a physical engine to generate 3D consistent images. Compared to prior works like Object 3DIT and Neural Assets (NA), BlenderFusion explicitly reconstructs the 3D model of an object, transforms it in the 3D space, and then renders a photorealistic image by fine-tuning Stable Diffusion. By training on paired images or video frames, it is able to disentangle the pose and appearance of objects, enabling various editing applications. Both qualitative and quantitative results on three datasets show that BlenderFusion clearly outperforms baselines.

### Strengths
- The paper is well-motivated. It largely resembles the traditional workflow of computer graphics -- first create 3D assets (including geometry and appearance), then port them into software like Blender, and finally render to images.
- The results on this specific task are clearly better than baselines, especially the performance under extreme conditions like composing a large number of objects or large rotation and translation.

### Weaknesses
1. The novelty seems limited. Nothing is surprising to me after reading the paper:
- The object-centric modeling approach is similar to Neural Assets, which represents an object with 3D pose and appearance;
- Using explicit 3D representation or tool use to improve visual generative models has been explored extensively [1, 2].

2. My biggest concern is that this approach is likely not scalable. It reminds me of a line of work in neural-symbolic learning which converts objects to abstract representations, and then runs external tools like code or physical engine to simulate the world [3, 4, 5]. These methods often have a strong assumption about the perception models, e.g., they should be robust enough to reconstruct the 3D geometry of objects from in-the-wild images.
- Though object reconstruction from object-centric images is to some extent solved, objects in real-world images are often not well-defined. For example, the part-whole hierarchy often causes ambiguity. Severe object occlusion will also degrade the perception model's performance.

3. In some result videos on the attached website, it seems that background pixels also change when foreground objects are edited.
4. What's the training cost and inference speed of BlenderFusion compared to baselines like NA?

[1] Hu, Ziniu, et al. "Scenecraft: An llm agent for synthesizing 3d scenes as blender code." Forty-first International Conference on Machine Learning. 2024.

[2] Ren, Xuanchi, et al. "Gen3c: 3d-informed world-consistent video generation with precise camera control." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[3] Ding, Mingyu, et al. "Dynamic visual reasoning by learning differentiable physics models from video and language." Advances in Neural Information Processing Systems 34 (2021): 887-899.

[4] Rubanova, Yulia, et al. "Learning rigid-body simulators over implicit shapes for large-scale scenes and vision." Advances in Neural Information Processing Systems 37 (2024): 125809-125838.

[5] Tang, Hao, Darren Key, and Kevin Ellis. "Worldcoder, a model-based llm agent: Building world models by writing code and interacting with the environment." Advances in Neural Information Processing Systems 37 (2024): 70148-70212.

### Questions
1. I wonder how important Blender is in this 3D-aware editing task. A prior work Diffusion Handles [1] uses depth to warp diffusion features of an object to target location, and also fine-tunes a diffusion model to render images. Can you compare with one such baseline to show the advantage of Blender?
2. I'm a bit confused by the "Simulated Object Jittering" technique. In this case, the source image is replaced with the target image as the model input. Why won't the model just use its information to denoise the target image? How can the model learn anything in this case?

[1] Pandey, Karran, et al. "Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents BlenderFusion, a generative visual compositing framework that recomposes objects, camera, and background to synthesize novel scenes. The framework consists of three key stages: (i) segmentation and conversion of visual inputs into editable 3D entities (layering), (ii) manipulation in Blender with 3D-grounded control (editing), and (iii) fusion into a coherent scene using a generative compositor (compositing). Overall, the proposed method is well-motivated and the experimental results effectively support the claims.

### Strengths
•	The paper is well-written with a logical structure that makes the technical contributions easy to follow.
•	The proposed  framework is reasonable and well-justified. The experimental results convincingly demonstrate the effectiveness of the approach across various compositing scenarios.
•	Excellent supplementary materials: The demo videos and project page significantly aid in understanding the core concepts and practical applications of the method.

### Weaknesses
•	Recent works have explored 3D scene reconstruction and composition capabilities. A more thorough comparison and discussion of the relationship between BlenderFusion and these methods would strengthen the paper. For example:
•	CAST [1] performs component-aligned 3D scene reconstruction from a single RGB image. How does BlenderFusion's layering approach compare to CAST's decomposition strategy?
•	What are the trade-offs between the generative compositing approach and traditional 3D reconstruction-based composition methods?
•	Could you clarify when your method is preferable over existing 3D composition techniques?

Reference
[1] CAST: Component-Aligned 3D Scene Reconstruction From an RGB Image

### Questions
Could you pleasse examples about more complicated scenarios?

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
3

### Summary
This paper introduces BlenderFusion, a novel framework for 3D-grounded visual editing and generative compositing. The core idea is to integrate the precise, 3D-aware control of a graphics engine like Blender with the powerful synthesis capabilities of diffusion models.

### Strengths
1. Clear Motivation and Strong Problem Formulation: The paper clearly identifies a significant and practical limitation in current generative AI: the lack of precise, 3D-aware control for complex, multi-object scene compositing. It effectively positions its contribution against existing methods (Table 1), clearly highlighting the gap it aims to fill.

2. Novel and Elegant Framework Design: The primary strength of this work lies in its core idea of decoupling 3D control from generative synthesis. By leveraging a mature graphics engine (Blender) for precise geometric manipulation and a diffusion model for photorealistic rendering, the framework provides an intelligent and practical solution.

3. Effective Training Strategies: The two proposed training strategies, "Source Masking" and "Simulated Object Jittering," are simple, well-motivated, and clever solutions to concrete problems. They effectively address the challenges of large-scale edits (e.g., object removal) and the entangled nature of object/camera motion in video datasets, which is critical for achieving robust, disentangled control.

### Weaknesses
1. Insufficient Detail on the Core Technical Novelty (Sec. 3.2): The paper's primary methodological contribution, the "Dual-stream Diffusion Compositor" in Section 3.2, is not described with sufficient clarity. The architecture is presented as a high-level black box, and the paper fails to provide a detailed diagram or explanation of the crucial "cross-stream interaction" mechanism. It is strongly recommended that the authors add a dedicated figure and more detailed text to fully articulate this component.

2. Engineering-Heavy Framework with Brittleness (Sec. 3.1): While Section 3.1 serves as an adequate description of the overall framework, it is fundamentally an engineering pipeline that chains together multiple existing, off-the-shelf models (e.g., Grounding DINO, SAM2, Depth Pro). This practical approach introduces significant brittleness, as a failure in any upstream component can compromise the entire process. Furthermore, the lack of novel components in this section concentrates the paper's entire scientific contribution into the insufficiently explained compositor, weakening the overall methodological robustness. A discussion on these limitations should be included.

3. Inconsistent Object Appearance Preservation: The qualitative results, while strong, reveal issues with preserving fine-grained object appearance and texture. For example, in Figure 5, the details on the yellow car change, and in Figure 10, the "Frosted Flakes" cereal box texture is visibly altered. This is likely an inherent limitation of "re-synthesizing" the object rather than copying pixels, perhaps due to information loss in the VAE or imprecise 2.5D geometry. The authors should acknowledge this limitation in their analysis and discuss its potential causes.

4. Missing Runtime and Scalability Analysis: The paper lacks any discussion of the framework's computational cost. The full pipeline involves multiple large models plus Blender rendering, suggesting it may be too slow for interactive use. Furthermore, the quality of the single-view 2.5D reconstruction is a critical bottleneck for complex or self-occluded objects. The authors should provide a runtime analysis (e.g., time per edit) in the appendix and discuss the scalability and quality trade-offs of their approach, including the mentioned option of using more costly image-to-3D models.

### Questions
Same as weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a Blender-guided scene editing pipeline: segment and lift objects; perform multi-object edits, layout changes, and camera moves in 3D; then use a diffusion model to blend and harmonize the edited render with the original photo.

### Strengths
Clear, production-like workflow; easy to implement.

Results suggest better local control on some inserts.

### Weaknesses
**Key Baselines Omitted:**

- ZeroComp[1]: composites intrinsic layers (depth/normal/albedo/shading) and lets diffusion render the final image. Similar goal but without using Blender directly. But they use a rendering engine to give approx 3D compositing. 
- DiffusionRenderer [2]: turns G-buffers into photoreal images/videos; direct alternative to “Blender render to diffusion fix.”
- 2D diffusion compositors: ObjectStitch [3], Thinking Outside the BBox [4], ControlCom [5], IMPRINT [6]: Generative Object Compositing by Learning Identity-Preserving already handle harmonization, identity preservation, and shadows/reflections without 3D renders. I just added a few of them, and there are plenty more papers on this theme. 
- Multi-object, layout-controlled compositing: Multitwine [7] --  multi-object edits with text + layout control and interaction handling (occlusions, relations). This directly tests beyond single-object.
- 3D Copy-Paste [8]: physically plausible insertion (placement/collision/lighting) for real scenes; overlaps the paper’s claims on scene consistency.

Each of these baselines addresses challenges such as harmonization, identity preservation, and scene consistency using distinct strategies often without requiring full 3D modeling. Benchmarking against or discussing these alternatives in more depth would clarify how the proposed method advances or complements the current state of the art.

**Evaluation gaps for compositing**

The paper does not quantitatively evaluate several aspects: (1) background fidelity outside the edit mask, (2) identity preservation for inserted objects, (3) realism of contact shadows and reflections, and (4) multi-view consistency under small camera perturbations. These can be measured in controlled settings, e.g., using synthetic data from Blender.

**Robustness not tested:** The approach is not evaluated for common failure modes for this system's type of pipelines when relying on several components to get the final rendering. These include noisy masks, moderate pose or scale inaccuracies, challenging scene cases (e.g., clutter, high reflectance/specularities), and the domain gap from synthetic to real data.

**Ablations and cost:** It is unclear whether improvements are primarily from the 3D editing stage or the generative compositor. There is also an absence of runtime analysis -- how long does it take for the edit to be complete, etc, or failure case breakdown, leaves the practical utility and scalability unsure.

**References:**

[1] Zhang et al., ZeroComp: Zero-Shot Object Compositing from Image Intrinsics via Diffusion, WACV 2025. 

[2] Liang et al., DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models, CVPR 2025.

[3] Song et al., ObjectStitch: Object Compositing with Diffusion Model, CVPR 2023. 

[4] Canet Tarres et al., Thinking Outside the BBox: Unconstrained Generative Object Compositing, ECCV 2024. 

[5] Zhang et al., ControlCom: Controllable Image Composition using Diffusion Model, arXiv 2308.10040. 

[6] Song et al., IMPRINT: Generative Object Compositing by Learning Identity-Preserving, CVPR 2024. 

[7] Canet Tarres et al., Multitwine: Multi-Object Compositing with Text and Layout Control, CVPR 2025.

[8] Ge et al., 3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection, NeurIPS 2023.

### Questions
- Can you report SSIM/LPIPS outside the edit mask for all methods (yours and baselines)?
- Can you report DINO/CLIP similarity (inserted asset vs final composite) and show failure cases?
- How do you quantify contact-shadow and reflection realism? Will you provide a metric (e.g., overlap with a rendered reference) or a small user study, and add complex objects like glossy/metal scenes?
- Can you add: (a) Blender-only (no diffusion), (b) diffusion-only (no Blender edits), (c) replace Blender RGB with intrinsic layers (depth/normal/albedo/shading) to isolate what the compositor needs, (d) remove object-jitter and masking?
- Please provide a wall-clock breakdown (segmentation/lifting, rendering, diffusion), GPU hours for training/inference, memory, and throughput vs baselines.

### Soundness
3

### Presentation
3

### Contribution
1
