# NeoWorld: Neural Simulation of Explorable Virtual Worlds via Progressive 3D Unfolding

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
We introduce NeoWorld, a deep learning framework for generating interactive 3D virtual worlds from a single input image. Inspired by the on-demand worldbuilding concept in the science fiction novel Simulacron-3 (1964), our system constructs expansive environments where only the regions actively explored by the user are rendered with high visual realism through object-centric 3D representations. Unlike previous approaches that rely on global world generation or 2D hallucination, NeoWorld models key foreground objects in full 3D, while synthesizing backgrounds and non-interacted regions in 2D to ensure efficiency. This hybrid scene structure, implemented with cutting-edge representation learning and object-to-3D techniques, enables flexible viewpoint manipulation and physically plausible scene animation, allowing users to control object appearance and dynamics using natural language commands. As users interact with the environment, the virtual world progressively unfolds with increasing 3D detail, delivering a dynamic, immersive, and visually coherent exploration experience. NeoWorld significantly outperforms existing 2D and depth-layered 2.5D methods on the WorldScore benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposed NeoWorld, a framework designed for generating interactive 3D virtual worlds from a single input image. It introduces a hybrid object-centric scene representation that progressively transforms a 2D scene into high-fidelity 3D models, allowing dynamic world exploration. The approach utilizes cutting-edge representation learning, object-centric techniques, and language models for flexible viewpoint manipulation and controlling object appearance or dynamics. The latter experiment demonstrate its computational efficiency with high-level realism.

### Strengths
1. The idea of modeling virtual worlds interactively, where only the explored regions are rendered in detail, is quite compelling. The paper's motivation of merging object-centric 3D rendering with computational efficiency is both interesting and valuable for creating immersive environments.
2. The system's ability to support language-driven object manipulation, along with physics-based animation. The demonstration of the object dynamic interactions (e.g., chair collision and boat bouncing) shows the potential of this system to create physically plausible simulations. This feature is particularly impressive for an interactive virtual world generation framework.
3. NeoWorld combines several advanced techniques, including object-centric neural scene representations, 2.5D-to-3D unfolding, and user–scene interaction module. These innovations empower the interactive world generation.

### Weaknesses
1. Although the paper mentions the ability for users to manipulate objects within the scene, the example provided in the paper is more like edited-related cases; it is unclear if the framework could achieve dynamic object manipulation while the camera navigates with a trajectory. If it only supports object manipulation in a static camera pose, the contribution might be overclaimed.
2. The author provides only a single demonstration of multi-object manipulation case in the appendix (two statues melting), the performance degradation analysis when the number of objects increases will be helpful to support the "object-centric" contribution.

### Questions
Refer to my weakness paragraph.

### Soundness
3

### Presentation
3

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
This work presents NeoWorld, a system for interactive 3D world generation from a single image. It progressively expands 3D scenes during user exploration, combining efficiency with realism by using a hybrid structure—2D backgrounds and fully 3D foreground objects. NeoWorld integrates differentiable rendering and image-to-3D reconstruction to enable 3D-consistent exploration and physics-based interaction, surpassing prior 2D and 2.5D methods.

### Strengths
1. Comparing to prior works, it now support interaction with foreground objects.
2. It doesn't sacrifice too much time cost for bringing only the foreground objects to 3D.
3. It outperform prior works in terms of PromptAlign and some other criteria.

### Weaknesses
1. Some distortions remain visible in the qualitative results.
2. Object movement is still limited to basic, language-driven control. In realistic settings, objects move only through physical interactions, which may explain why current video generation models struggle with such behavior.
3. The reconstruction quality of the object is low. Is there any way to improve that?

### Questions
Could the generated motions could be more complex and realistic? For instance, the car in Figure 3 could move naturally along the road.

### Soundness
2

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
3

### Summary
This paper, NeoWorld: Neural Simulation of Explorable Virtual Worlds via Progressive 3D Unfolding, presents a system that builds interactive, explorable 3D worlds from a single input image. The method begins with an object-centric 2.5D Gaussian representation (with collapsed depth per primitive), progressively converting selected regions into full 3D reconstructions as the user navigates or issues natural-language commands. The system integrates several pretrained components — panoptic segmentation (OneFormer), depth/normal estimation (Marigold), image inpainting (Stable Diffusion v2.0 / distilled SD-XL), image-to-3D reconstruction (Amodal3R), and an LLM (Gemini-2.5pro) for object selection and physics parameter inference.

NeoWorld aims to balance efficiency and realism by unfolding only the necessary parts of a scene into 3D, enabling interactive world expansion and simple physics-based object manipulation (MPM). Evaluation is performed on the WorldScore and WonderWorld benchmarks.

### Strengths
Ambitious problem scope: Tackles interactive, incremental world generation — a compelling and forward-looking problem bridging image-to-3D, scene synthesis, and agentic interaction.

Conceptual novelty: The idea of progressive 2.5D → 3D unfolding is elegant and well-motivated. It provides a practical trade-off between full 3D reconstruction and efficient view synthesis.

System design: The combination of object-centric Gaussian layers, selective 3D conversion, and LLM-based interaction is well engineered and demonstrates clear qualitative improvements over prior 2D/2.5D methods (e.g., WonderWorld, WonderJourney, Matrix-Game2). in particular, having the LLM predict object parameters from an MPM solver, then rendering the result, is elegant and works better than baselines. 

Comprehensive qualitative results: The paper covers a wide range of baselines (Wan 2.1-I2V, Matrix-Game2, WonderJourney, WonderWorld) and provides illustrative visual and interactive demos in the supplement.

Human evaluation and benchmarks: Employs WorldScore and multiple human-study-based metrics (3D-Const, SceneQuality, PromptAlign), showing meaningful gains in perceived realism and interactivity.

Clear presentation: The architecture and pipeline are easy to follow, and figures effectively convey the unfolding process.

### Weaknesses
1. System-level composition, limited algorithmic novelty.
The method primarily integrates existing models (OneFormer, Marigold, Amodal3R, SD-XL, Gemini LLM) with minimal new learning formulations. The core contributions are system-level engineering rather than new representation learning or theory.

Ablations missing for key design choices e.g. why MPM, why a particular depth estimator was chosen.
No evaluation of alternative inpainting models (e.g., Flux, SD-3.5) or justification for choosing SD-2.0/distilled SD-XL.

Missing ablations for the pose-token alignment module, LLM-based object selection, and codebook design (size, per-scene vs global, dimensionality).

2. Unclear details and justifications.
Line 192: unclear whether the codebook is optimized per scene or shared globally.
Line 280: rationale for model choices (e.g., older SD version, specific normal/depth backbones) is not explained.
LLM choice (Gemini-2.5pro) may give the system an advantage unrelated to algorithmic contribution.

3. Evaluation and benchmarking gaps.

WorldScore and WonderWorld metrics are new and not widely validated; results should be complemented with standard metrics (FID, LPIPS, depth/normal consistency, temporal smoothness, etc.).

Baselines appear visually weaker than their official results; comparisons should use the same scenes, prompts, and camera trajectories for fairness if possible.

4. Quantitative analysis missing.
No per-module timing/memory breakdown 
No scaling study of runtime or memory vs. world size or number of unfolded objects.
No quantitative failure attribution across submodules (segmentation, depth, 3D reconstruction, LLM).
Missing physics/animation metrics — dynamics and prompt following are only shown qualitatively.

### Questions
1. In the move object examples in the supplemental video, the synthesized results don’t appear to harmonize well with the rest of the scene e.g plausible shadows underneath the boat or chairs. Why is this?

2. The presented samples in Fig. 3 clearly favor this method, but there appears to be a large quality gap between presented results of WonderWorld and what is shown on their project page. Why is this? Could the authors compare on the same scenes that they use in their paper for apples to apples?

3. Does the method work on test images with variation in style, domain, and resolution (e.g., indoor/outdoor, toon vs. photorealistic)?

4. Figure 11: there are several papers that can do inpainting. The authors present a distilled SDXL to do this. How does the distilled model compare with benchmarks in performance and quality?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a system for interactive 3D world generation from a single image. It introduces an object-centric representation that starts with a layered foreground and background separation and progressively reconstructs objects as 3D Gaussians. The system supports text-driven exploration and manipulation of scene context via LLMs. The paper creates a benchmark dataset based on WonderWorld, WorldScore and WonderJourney, and shows better performance based on human evaluation on CLIP-based metrics.

### Strengths
1.Object-centric representation provides greater flexibility in terms of interactive editing and semantic manipulation.
2. The progressive unfolding strategy allows lightweight scene initialization and efficient expansion based on user interaction.

### Weaknesses
1. The pipeline depends heavily on pretrained models and heuristics, such as, object alignment, fallback strategy for 3D reconstruction, without any unifying learning based approach.
2. While text-guided object control is well integrated, prior works such as Instruct-NeRF2NeRF, SceneTeller, etc. show similar capability.
3. The paper does not include qualitative video results which makes it difficult to judge the quality of the progressive expansion.
4. There are no ablation experiments or robustness analysis of different pretrained model choices.

### Questions
1. How does performance degrade with different model choices, e.g., with lighter models?
2. How robust is the text-based control to ambiguous instructions?

### Soundness
2

### Presentation
2

### Contribution
2
