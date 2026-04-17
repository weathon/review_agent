# PI-Light: Physics-Inspired Diffusion for Full-Image Relighting

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Full-image relighting remains a challenging problem due to the difficulty of collecting large-scale structured paired data, the difficulty of maintaining physical plausibility, and the limited generalizability imposed by data-driven priors. Existing attempts to bridge the synthetic-to-real gap for full-scene relighting remain suboptimal. To tackle these challenges, we introduce **P**hysics-**I**nspired diffusion for full-image re**Light** ($\pi$-Light, or PI-Light), a two-stage framework that leverages physics-inspired diffusion models. Our design incorporates (i) batch-aware attention, which improves the consistency of intrinsic predictions across a collection of images, (ii) a physics-guided neural rendering module that enforces physically plausible light transport, (iii) physics-inspired losses that regularize training dynamics toward a physically meaningful landscape, thereby enhancing generalizability to real-world image editing, and (iv) a carefully curated dataset of diverse objects and scenes captured under controlled lighting conditions. Together, these components enable efficient finetuning of pretrained diffusion models while also providing a solid benchmark for downstream evaluation. Experiments demonstrate that $\pi$-Light synthesizes specular highlights and diffuse reflections across a wide variety of materials, achieving superior generalization to real-world scenes compared with prior approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a physics-inspired diffusion for full-image relighting that integrates light transport priors into neural forward rendering to achieve more precise relighting. Additionally, the authors introduce a new dataset comprising diverse objects and scenes.

### Strengths
+ Experimental performance: The proposed method demonstrates superior results compared to existing methods in terms of consistency, maintaining, and rendering quality.
+ Clarity and readability: The manuscript is well-structured and clearly written, making the methodology and contributions easy to follow. 
+ Attribution: This paper provides a physical perspective to solve the relighting problem, and a meaningful dataset is collected.

### Weaknesses
- The authors claim that existing methods struggle to generalize to out-of-distribution data (Line 51). It remains unclear how the proposed approach addresses this challenge. The authors are encouraged to elaborate on the mechanisms or design choices that contribute to improved generalization beyond the training distribution.
- As shown in Fig. 2, the proposed method appears to misinterpret the light reflections of “moonlight scattered on the sea surface” in the background painting and treats it as the light source. The authors may improve albedo estimation accuracy by incorporating a deeper understanding of scene semantics under such challenging conditions.
- The paper should provide more details regarding the trainable components of the proposed network. In particular, it remains unclear which layers (e.g., convolutional layers in Fig. 3) are trainable and what their specific architectures or configurations are.
- Comparison with $RGB \leftrightarrow X$: Since the $RGB \leftrightarrow X$ was trained only on indoor scenes, it results in limited generalization performance. A performance comparison with a version of $RGB \leftrightarrow X$ trained on the same datasets as the proposed method is needed. This would offer a fairer evaluation of the proposed approach’s advantages.

### Questions
- About the batch-aware attention: Lines 237-238 state that “the standard self-attention layers are extended to be global-aware, enabling communication across batches. This design improves both efficiency and consistency among predicted intrinsics.” Can you provide a more intuitive explanation of why and how the self-attention can lead to such improvements in consistency and efficiency? And why can the observed performance gains be attributed to self-attention layers? Please provide more details on the implementation of batch-aware attention and how information is exchanged across batches.
- The paper mentions that normal and roughness are fed into the frozen VAE encoder pretrained on RGB images to extract latent features. Given that the encoder was trained on RGB data, have the authors considered the potential domain gap between the intrinsic maps (e.g., normal and roughness) and the RGB domain? How can you ensure the extracted features are practical?

### Soundness
2

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
5

### Summary
This paper tackles the problem of full-image relighting. To enable realistic relighting results, the authors propose to first estimate intrinsic properties, e.g., albedo, normal, and roughness, based on a diffusion model. Later, these intrinsic properties are fed into another diffusion model that produces a relit image as well as diffuse and specular maps. The inputs for these outputs are inspired by physically-based rendering. Specifically, one takes the source image and albedo; one takes normal, lighting, and mask; the other uses normal, lighting, metallic, roughness, and mask. Further, with a simplified physically based rendering equation, i.e., albedo $\cdot$ diffuse + specular, the directly-produced relit image is compared against the physically-rendering-composed image. Experiments on various datasets verify the effectiveness.

### Strengths
- originality-wise: the idea of closely following physically-based rendering to construct the input for the diffusion model is interesting.
- quality-wise: the qualitative and quantitative results are promising.
- clarity-wise: the presentation is good in general but can be further improved.
- significance-wise: the realistic relighting is important for various downstream applications, e.g., AR/VR.

### Weaknesses
## 1. Clarification on image intrinsics estimation

Can authors clarify why the model in the paper works better than baselines, e.g., IntrinsicAnything in Tab.1? Is it because of carefully curated data or different model designs?

## 2. About efficiency

**For training**. As mentioned in L302:
> Since physical laws lose their meaning when computed in the latent space, our physics-based losses for the relighting model are applied entirely in the RGB space, i.e., it is computed after the VAE decoder reconstructs the output

This makes the training extremely costly as each data in the training set needs to go through the full diffusion process to obtain the final latent. How to make the training affordable?

**For inference**, can the author provide some runtime analysis for the inference?

## 3. Presentation

1. There is no lighting condition in Figure 3's "Stage 2"

2. For Eq. (6) and (8), the input and output orders are not aligned. Either the authors can have separate equations, or the authors can make the order aligned. Currently, it isn't very clear. I do not know which input corresponds to each output.

## 4. References

Missing related works for generative relighting

[a] Ginter et al., A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis. EGSR 2024.

[b] Zhao et al., IllumiNeRF 3D Relighting without Inverse Rendering. NeurIPS 2024.

### Questions
See "Weakness"

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes PI-Light, a framework to relight images via the extraction of intrinsic properties of a scene and diffusion-based rendering for the final result: in the inverse rendering stage, surface normal and materials are estimated from a customized diffusion model; in the neural forward rendering stage, another diffusion model is used to synthesize the image under target lighting conditions. To facilitate training of the models, the authors constructed a new dataset featuring rendered data using objects from Objaverse. In the experiment section, the authors compare PI-Light with both intrinsic decomposition models and relighting methods. The proposed method shows reasonable performance.

### Strengths
- The motivation for having physics-inspired scene properties in the relighting pipeline is solid. With more and more frameworks becoming end-to-end, it is good to have a variety of approaches with competitive performance. In particular, this paper shows that it is possible to combine intrinsic decomposition with diffusion-based relighting for a good result. 

- The paper focuses on full image relighting, which is an under-explored direction in image relighting, although currently both the dataset and the experiment in this paper are still mainly object-centric. 

- The paper is easy to follow, and the design of the framework is well described. 

- The proposed dataset could be valuable to researchers studying both image relighting and intrinsic decomposition tasks.

### Weaknesses
- Intrinsic decomposition using neural networks is a well-studied problem. There is little new insight from this paper (Section 4.2). 

- It seems the relighting results shown in the paper are mostly directional lighting. In other relighting work, such as Neural Gaffer, HDR image-based environment maps are used as lighting conditions which provide more natural and complex lighting conditions. Since the proposed method is targeting full image relighting, single directional lighting could be a limitation. 

- Evaluation is done on 50 objects and 20 scenes. It might be better to try with more objects and scenes, as well as on other relighting datasets.

### Questions
- Why is the full model performance in Table 3 (last row) different from Table 2 (last row)? If I understand it correctly, they should be the same model and tested on the same Object50 test set.

- How does the proposed method work with HDR image-based environment maps? It seems by design the pipeline should not be limited to directional lights.

### Soundness
3

### Presentation
3

### Contribution
3
