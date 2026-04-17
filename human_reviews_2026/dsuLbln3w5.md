# Neural USD: An object-centric framework for iterative editing and control

- Decision: Reject
- Scores: 2, 8, 6, 4

## Abstract
Amazing progress has been made in controllable generative modeling, especially over the last few years. However, some challenges remain. One of them is precise and iterative object editing. In many of the current methods, trying to edit the generated image (for example, changing the color of a particular object in the scene or changing the background while keeping other elements unchanged) by changing the conditioning signals often leads to unintended global changes in the scene. In this work, we take the first steps to address the above challenges. 

Taking inspiration from the Universal Scene Descriptor (USD) standard developed in the computer graphics community, we introduce the “Neural Universal Scene Descriptor” or Neural USD. In this framework, we represent scenes and objects in a structured, hierarchical manner. This accommodates diverse signals, minimizes model-specific constraints, and enables per-object control over appearance, geometry, and pose. We further apply a fine-tuning approach which ensures that the above control signals are disentangled from one another. We evaluate several design considerations for our framework, demonstrating how Neural USD enables iterative and incremental workflows.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces "Neural USD," an object-centric framework designed to enable precise and iterative editing of generative models. Taking inspiration from the Universal Scene Descriptor (USD) standard in computer graphics, the authors propose representing a scene's components (objects, background) as assets with distinct, hierarchical attributes: appearance, geometry, and pose. The core technical contribution is a fine-tuning strategy that uses paired images from video sequences to train a generative model to disentangle these control signals. The paper demonstrates that this framework allows for object-level manipulations, such as changing pose, appearance, or geometry, and replacing objects or backgrounds, while aiming to keep other scene elements consistent.

### Strengths
Clever Conceptual Bridging: The core idea of adapting the structured, hierarchical USD standard from computer graphics to the conditioning of diffusion models is both novel and elegant. It provides a principled-sounding approach to a problem often tackled with less structured methods.

Core Training Strategy: The method of using paired images ($I_{src}$ for appearance/geometry, $I_{tgt}$ for pose) to force the model to learn disentangled representations is a key insight and a strong methodological contribution.

### Weaknesses
The paper's primary qualitative example, Figure 1, seems to undermine its central claim of disentangled control. In Fig 1(b), the stated operation is a "Pose" change. However, the background has clearly changed in appearance compared to Fig 1(a). Furthermore, the object's own appearance (the orange chair) also appears to have different lighting/shading in 1(b).

Missing Critical Ablation Studies (Especially on Geometry): The framework's complexity (requiring pose, appearance, and geometry) is not sufficiently justified. The authors have not provided a crucial ablation study to demonstrate the necessity of the geometry (e.g., depth map) signal. How does the model perform with only Pose + Appearance conditioning? As user paste the warped cropped region into background, then inpaint the image, it seems to get the similar result? Without this ablation, it is impossible to assess the contribution of the geometry component. This is a significant gap in the experimental validation.

(Minor Weakness): As noted, the main paper is surprisingly sparse on qualitative results, relegating most examples to the appendix.  Given that the paper does not fill the 9-page, key supporting results (especially those that successfully demonstrate the claims from Fig 1) should have been included in the main body.

### Questions
Same as weakenss.

### Soundness
2

### Presentation
2

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
This paper proposes Neural USD. In general, it can be seen as a ControlNet with a finer grid for object control. The paper generates a dataset with detailed annotations such as depth, boxes, and many others, then uses this information to fine-tune a pre-trained image model. The fine-tuned model shows great results in image control capability.

### Strengths
1. The paper targets a great problem for finer control of the objects in an image. The proposed method, although simple, is pretty straightforward and effective.

2. The datasets in the paper can benefit future research.

3. The demonstrated results are good.

### Weaknesses
1. Since the model is still a learning-based image-to-image model, keeping other objects unchanged is not guaranteed. I can see obvious background change in Figure 1.

### Questions
I do not see major weaknesses or questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a new object-centric generative framework inspired by USD standard, aiming to unify representation, control, and editing in image and 3D content generation. The core idea of this method is to tokenize each object in a scene, capturing its geometry, appearance, pose, and material attributes, into a structured latent representation that can condition diffusion or transformer-based generative models.

### Strengths
1. The idea is interesting to me, that is a unified, structured conditioning standard inspired by USD, enabling disentangled control over object appearance, geometry, and pose in generative models.
2. The ability to perform multi-step, fine-grained, object-level edits without unintended global changes, a clear improvement over existing conditioning methods (e.g., ControlNet, InstructPix2Pix).
3. The format is architecture-agnostic and supports diffusion, DiT, and transformer models through tokenized conditioning, improving portability and generalizability.

### Weaknesses
1. Global scene changes still occur, as I observed from the visual results. It is somewhat overclaimed, as the abstract and introduction sections stated.
2. It remains unclear whether the fusion happens at the feature level (joint embedding) or via concatenated conditioning channels.
3. Uses Stable Diffusion v2.1 as backbone, leading to lower image quality than state-of-the-art diffusion models; I still believe it should be adopted in more powerful backbones (e.g., Flux).

### Questions
Please refer to the weakness part

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work enables precise control over an object’s appearance, geometry, and pose. By encoding geometry, depth maps, and bounding boxes into the image generation model, it can control the placement of specific objects within specific scenes, as well as edit their pose/appearance, background, and viewpoint. Moreover, it supports an iterative editing workflow, allowing continuous refinement of scene composition and object attributes.

### Strengths
1.	Achieves disentangled control over an object’s appearance and geometry in a simple and straightforward manner.
2.	Enables more precise control of objects, with quantitative metrics outperforming previous methods.
3.	Designs a stable iterative 3D editing workflow, allowing sequential replacement of pose, appearance, object, and background, while preserving the results of previous edits during new editing steps.

### Weaknesses
1.	The paper repeatedly emphasizes that it can edit an object’s pose while keeping other attributes of the source image unchanged (as stated in the abstract, section 4.2). However, in practice, the camera pose and object pose are not successfully disentangled — pose editing often causes a change in the viewing angle instead of the object moving relative to the scene (see Fig. 9).
2.	This work follows a technical route very similar to Neural Assets, with comparable metrics (see Figs. 7 and 8), yet lacks corresponding visual comparisons.
3.	The background after editing differs noticeably from the original image (Figs. 5 and 6), and the object texture and appearance exhibit clear artifacts (e.g., Fig. 6 (c)). The authors attribute this to the base image model not being state-of-the-art, but this explanation remains unverified. The presence of such visible artifacts after even a single edit undermines the benefit of an iterative workflow, since preventing cumulative errors across edits is one of its key goals.
4.	The contributions are relatively limited.
Contribution 1 claims that the model learns disentangled control signals from video image pairs, but this training paradigm was already introduced by Neural Assets. The main difference lies in the additional encoding of geometry, enabling finer-grained disentanglement.
Discussion of Contribution 2 is provided in the previous point 3.
Contribution 3 claims improved accuracy based on Fig. 8, but the figure shows no significant improvement over Neural Assets.
5.	In Figs. 5 and 6, the 3D bounding boxes before and after editing should be marked.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
2
