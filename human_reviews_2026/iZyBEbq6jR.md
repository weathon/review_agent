# PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Generating a complete and explorable 360-degree visual world enables a wide range of downstream applications. While prior works have advanced the field, they remain constrained by either narrow field-of-view limitations, which hinder the synthesis of continuous and holistic scenes, or insufficient controllability that restricts free exploration by users or autonomous agents. To address this, we propose PanoWorld-X, a novel framework for high-fidelity and controllable panoramic video generation with diverse camera trajectories. 
 First, we propose a novel pipeline for synthesizing panoramic video-trajectory dataset pairs in virtual 3D environments via Unreal Engine. This pipeline consists of four main steps and enables the collection of a large-scale dataset with rich scene diversity and accurate trajectory annotations.
 To achieve precise panoramic video generation, we identify that the bottleneck arises from the misalignment between the spherical geometry of panoramic data and the inductive priors of conventional video diffusion models. To address this, we leverage the spherical connectivity characteristics of panorama data, and propose a Sphere-Aware Diffusion Transformer that reprojects equirectangular features onto the spherical surface, thereby capturing geometric adjacency in the latent space. This design significantly improves both visual fidelity and spatiotemporal continuity.
   Extensive experiments demonstrate that our PanoWorld-X achieves superior performance in various aspects, including motion range, control precision, and visual quality, underscoring its potential for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents PanoWorld-X, a framework for generating controllable 360-degree panoramic videos with trajectory-guided exploration. The main contributions are: (1) PanoExplorer dataset with 116,759 panoramic video-trajectory pairs generated via Unreal Engine, (2) Explorable Sphere-Aware DiT architecture featuring Exploration-Aware Attention for trajectory control and Sphere-Aware Attention that leverages spherical geometry to improve consistency, and (3) experiments showing improvements over existing methods in visual quality, motion range, and control precision. The work enables immersive world generation for VR, embodied AI, and autonomous driving applications.

### Strengths
1. The PanoExplorer dataset addresses a critical gap in panoramic video research with 116,759 high-quality samples featuring diverse scenes (10 categories), extensive motion patterns, and precise trajectory annotations. The data curation pipeline is well-designed with collision detection and multi-stage quality filtering.
2. The sphere-aware attention mechanism is theoretically well-motivated, properly accounting for the spherical topology of panoramic data. 
3.  The paper provides extensive comparisons with both panoramic generation methods (360DVD, Imagine360, GenEX) and camera-controllable methods (CameraCtrl, AC3D), demonstrating clear improvements across multiple metrics. Ablation studies systematically validate each component.

### Weaknesses
1. The entire dataset is synthetic from Unreal Engine. While Section D.1 shows some zero-shot generalization to in-the-wild images, more rigorous evaluation on real panoramic videos would strengthen the claims about practical applicability. The domain gap could limit real-world deployment.
2. There lacks some analyze regarding the proposed datasets. For example, this paper does not give the camera trajectory distribution (move forward, move forward and turn left / right, etc), which is important to understand the proposed datasets. And keeping a balanced camera trajectory is important for learning a model with good trajectory generalization ability.
3. The comparison between PanoWorld-X and other methods is unfair, since other methods are not trained using the proposed datasets, while the evaluation dataset comes from the proposed dataset.

### Questions
1. How to define the yaw, pitch, and roll angles (Line 252)?
2. Can the author give more detailed description regarding the scale normalization strategy? (Line 263)
3. Can the author compare PanoWorld-X with other baseline methods using the datasets comes from other sources?

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
4

### Summary
The paper proposes PanoWorldX, a novel framework for high-fidelity and controllable panoramic video generation with diverse camera trajectories. PanoWorldX consists of a novel pipeline using the Unreal Engine to procedurally generate a large, diverse dataset of panoramic video-trajectory pairs, and a Sphere-Aware Diffusion Transformer that reprojects equirectangular features onto the spherical surface to accurately model geometric adjacency in the latent space. By specifically addressing the geometric misalignment between the spherical geometry of panoramic data and conventional diffusion model priors, PanoWorldX demonstrates significant improvements over existing methods in terms of motion range, control precision, and visual quality.

### Strengths
- The panorama data synthesis pipeline for large-scale and accurately labeled panoramic video-trajectory pairs is a valuable contribution, as the scarcity of data has limited the development of this task.
- The Sphere-Aware Attention and Exploration-Aware Attention are theoretically sound and practical for integration into existing DiT architectures. Explicitly modeling for the spherical topology is effective for high-fidelity panoramic generation,
- Experimental results demonstrate that PanoWorldX outperforms baseline methods and achieves good panoramic video generation performance.

### Weaknesses
- While I agree that synthesized panoramic data is necessary, the domain gap between synthetic and real-world data remains significant. Complex real-world scenarios, such as variations in lighting, non-rigid deformation, and motion blur and noise discrepancies that occur during shooting, remain challenging to represent in synthetic data. This paper's analysis of the data lacks a detailed discussion of these cases, which may be beneficial for future high-quality panoramic data synthesis.
- PanoWorldX focuses heavily on static scene generation and camera movement. It is unclear how PanoWorld-X handles highly dynamic, non-structural elements, such as moving characters, weather changes, or complex large motions. Further discussion or ablation studies on the temporal stability of non-rigid objects would enhance the review of temporal coherence.

### Questions
- Modifying the standard DiT architecture with additional attention branches (Sphere-Aware and Exploration-Aware) inherently increases computational load during training and inference. What is the latency or memory overhead compared to the baseline DiT model?
- While trajectory complexity was discussed, can the model handle cases like abrupt changes in elevation, sharp turns in confined spaces, or rapid changes in camera velocity without noticeable artifacts?
- What is the individual contribution of the Exploration-Aware Attention in the ablation study?

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
4

### Summary
1. The paper introduces PanoWorld-X, a framework for generating controllable 360-degree panoramic videos with diverse exploration paths. 2. The paper proposes the PanoExplorer dataset, a large-scale synthetic dataset of panoramic video-trajectory pairs.
3. The paper proposes sphere-aware DiT block. This block consists of explorable-aware attention to handle trajectory control and a sphere-aware attention to improve spatiotemporal consistency.
4. The experimental results demonstrate that PanoWorld-X achieves superior performance compared to previous panoramic video generation and trajectory-based methods.

### Strengths
1. The paper introduces PanoWorld-X, which can generate high-fidelity, explorable panoramic videos controlled by trajectories.
2. The proposed Sphere-Aware DiT is the main technical contribution. It is well-motivated by the need to address the geometric misalignment between spherical panorama data and standard diffusion models, explicitly capturing spherical connectivity.
3. The PanoExplorer dataset is a valuable contribution to the field, which contains 110k panoramic videos paired with trajectory annotations from Unreal Engine.

### Weaknesses
1. The architectural contribution of the trajectory control mechanism(ControlNet) appears limited and there are many previous works about it in the field of general video generation. Another novelty is sphere-aware attention, where the similar spherical representation usage is discussed in previous panoramic image generation works likes PanFusion and Text2Light.
2. The explanation of the Exploration Route Representation in Section 3.3 lacks clarity. The paper introduces a 6-DoF signal $Er_i$​ but then discusses converting it into extrinsic matrices and using Plücker embeddings. The precise relationship between this initial 6-DoF vector and the final $R^{F×H×W×6}$ embedding tensor depicted in Figure 2 is ambiguous and needs to be clarified.
3. The practical utility of the framework is severely constrained by its low output resolution of 480×960. For panoramic content, this resolution is insufficient for reality usage, especially given the base model supports higher resolutions. This issue is acknowledged in the supplementary materials, where perspective crops are described as being "less than 50 pixels".
4. While quantitative metrics on the evaluation of trajectory adherence are provided, the paper do not offer side-by-side visualization of the desired camera path overlaid on the generated video to demonstrate the model's trajectory-conditioned effect on the generated video.

I will consider adjusting my score if my concerns are well-resolved.

### Questions
1. Given the final generated resolution is 480×960, what is the native resolution of the PanoExplorer dataset? It connects with the utility of the dataset for future high-resolution panoramic research.
2. How capable is the model of inferring an exploration path purely based on text instructions?
3. The current evaluation focuses heavily on static, synthetic scenes. How does PanoWorld-X handle out-of-distribution real-world panoramic images even with moving objects?

### Soundness
3

### Presentation
2

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
The paper proposes PanoWorld-X, a framework for generating high-quality, controllable 360-degree panoramic videos to address the narrow field-of-view and limited user control in existing models. The authors first created a large-scale synthetic dataset of panoramic videos with corresponding 3D trajectories using Unreal Engine. The core of the method is a novel Sphere-Aware Diffusion Transformer, an architecture specifically designed to handle the spherical geometry of panoramic data, which improves visual consistency. This model allows for precise control over camera movement and, according to the paper's experiments, outperforms previous methods in generation quality and control accuracy.

### Strengths
1. The paper proposes PanoWorld-X, a new framework that generates high-fidelity, controllable panoramic videos, effectively overcoming the key limitations of narrow field-of-view and lack of interactivity found in prior models.
2. The authors propose a novel pipeline for creating a large-scale panoramic video dataset with accurate trajectory data using Unreal Engine. This addresses the critical lack of high-quality, annotated data needed to train such models.
3. The paper introduces a sphere-aware architecture that adapts pre-trained diffusion model to the spherical geometry of panoramic data, leading to significant improvements in visual quality and spatiotemporal consistency.

### Weaknesses
1. The PanoWorld-X framework is fundamentally restricted to generating short video clips due to its reliance on the CogVideoX-5B backbone. This curtails the potential for creating truly long-form, explorable experiences.
2. The model struggles to generate dynamic content, particularly human figures, because its training dataset is composed of static scenes. This limits its ability to create lively and realistic virtual worlds.

### Questions
1. The current trajectory sampling method may create a bias towards simple, linear paths in certain environments. Have the authors analyzed the distribution of trajectory complexity, and have more naturalistic sampling strategies been considered to ensure diversity?
2. The Sphere-Aware Attention relies on a fixed, binary distance threshold. Could the authors elaborate on the design rationale for this specific choice? I wonder if the authors have compared their design against other potential strategies, such as using a soft, distance-weighted mask.
3. The introduction of parallel attention branches likely increases computational load. Could the authors provide a quantitative analysis of this overhead?

### Soundness
4

### Presentation
3

### Contribution
3
