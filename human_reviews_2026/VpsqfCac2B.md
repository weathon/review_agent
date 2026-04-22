# SpatialHand: Generative Object Manipulation from 3D Prespective

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
We introduce SpatialHand, a novel framework for generative object insertion with precise 3D control. Current generative object manipulation methods primarily operate within the 2D image plane, but often fail to grasp 3D scene complexities, leading to ambiguities in an object's 3D position, orientation, and occlusion relations. SpatialHand addresses this by conceptualizing object insertion from a true ``3D perspective," enabling manipulation with a complete 6 Degrees-of-Freedom (6DoF) controllability. Specifically, our solution naturally and implicitly encodes the 6DoF pose condition by decomposing it into 2D location (via masked image), depth (via composited depth map), and 3D orientation (embedded into latent features). To overcome the scarcity of paired training data, we develop an automated data construction pipeline using synthetic 3D assets, rendering, and subject-driven generation, complemented by visual foundation models for pose estimation. We further design a multi-stage training scheme to progressively drive SpatialHand to robustly follow multiple complex conditions. Extensive experiments reveal our approach's superiority over existing alternatives and its great potential for enabling more versatile and intuitive AR/VR-like object manipulation within images.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a method for image manipulation to control the 6DoF of objects appearing in it. To do so, a transformer diffusion model is modified to take a combination of depth and orientation control signals. First, the method extracts the scene depth using DepthAnything, and computes the occlusion on the desired edited region, determining the location of the 3D object. The object orientation is described with three parameters (azimuth, elevation, in-plane rotation), and added to the object reference condition. To train the transformer, the paper proposes to rely on 3D generative AI data, render the assets in images and generate captions, and then rely on rendering engines' simulations and generative AI placement to synthesize realistic scenes. Final 3D information (depth of generated scene, 2D masks, and positions) is obtained by piping ad-hoc foundational models. Experiments are conducted on 1600 test samples in total, and evaluated using AI-based metrics (e.g., DINO and CLIP similarity) as well as a human user study, surpassing the competitors.

### Strengths
- The paper provides a careful combination of different foundational models to generate data, showing impressive results in terms of consistency.
- The paper is well presented and clear
- The data, benchmark, and method could be useful for several downstream tasks and of interest for general ICLR audience

### Weaknesses
- The paper relies on an intricate combination of different tools, and not all the details are reported. I believe replicating the results could be particularly challenging, and code release is not mentioned in the paper. The method also requires a non-negligible computational effort for training, as it uses 8xA100 GPUs

- The papers rely on a test set that also uses generated background, DINO features, and other foundational models as judges to assess the consistency of the provided generation. However, since the method itself heavily relies on generative AI, I would argue that an evaluation using more grounded geometrical data would be informative. For example, relying on synthetic rendering or lightstage captured objects could provide an exact ground truth on how the edited image and the rotated object should look, which would also be a nice qualitative result for the reader to inspect the result of the method. On a similar note, it would be interesting to use the method to perform 3D reconstruction of image objects by generating controlled multi-views of it, and then compare the obtained result with the GT Objaverse asset.

### Questions
- Will the code be released?
- How long does it take for the training in wall-clock time?
- Could you provide further details on the training, e.g., what are the losses used for? How are the different stages balanced?
- Is the material of the assets considered in the training set creation through Blender? How does the method perform for objects with high reflectance/specularity?

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
This paper presents a method that enable object insertion with 3D control.
The main contribution from my point of view is the dataset curation pipeline and the ability to achieve better result of 3D control.

### Strengths
- The dataset generation process is reasonable and serve the purpose of the proposed method well.
- The proposed method is straightforward and easy to understand.
- The demonstrated results seem better than other existing methods.

### Weaknesses
- The need of additional training is a bit complicated compared to recent training-free method.
- The computational overhead compared to original base model (e.g., FLUX) is not discussed properly.
- The composited objects sometimes have weird projections or appearance and did not match the scene well, e.g., the airplane in Fig. 5. Regarding that case, I feel the composited results generated by GPT-4o is more convincing.

### Questions
- I think overall, composition is a difficult task with a long history in computer graphics and vision. For inserting an object, while the pose is vital, the appearance is also important. Have the authors considered how to better improve the composited appearances? For example, by inputting additional conditions? I think this is not critical but worth discussing in the paper.
- I am curious about how different datasets affect the results? In the paper, there are both simulated placement and generative placement datasets. But it is not clear shown and discussed why both datasets are needed.

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
SpatialHand proposes a practical, single-stage diffusion-transformer framework for 3D-aware object insertion and movement in images by decomposing an object’s 6DoF pose into three conditions: (1) 2D location via a geometry-aware masked scene, (2) target depth via a composited depth map, and (3) 3D orientation embedded as latent tokens (azimuth/elevation/in-plane rotation). They add additional tokens to a pretrained FLUX model, using LoRA to adapt to additional conditions. The authors build a large paired training corpus via (a) synthetic 3D assets + renderer and (b) generative placement using subject-driven models, then use visual foundation models to estimate pose for supervision. They train progressively (identity pretrain → novel-view fine-tune → insertion fine-tune) and show quantitative and qualitative gains for both insertion and movement tasks versus several baselines.

### Strengths
* The method presented is elegant and solves critical problems for object manipulation: 3D translation and rotation + in-context generation. 
* The scalable data generation pipeline using pretrained 3D model generators + traditional rendering is clearly effective for generating varied data

### Weaknesses
* Missing evaluation: I believe the authors should evaluate against LooseControl (SIGGRAPH 2024) and Build-a-Scene (ICLR 2025) which also can move/rotate objects in 3D space. 
* I believe the DragDiffusion line of work [CVPR 2024], more recently Dragin3D [CVPR 2025] can also rotate objects and should be evaluated. 
* This one is less important, but a recent frontier model Reve.ai can also move objects by placing 2D boxes and could be part of the evaluation. 
* The method can handle simple objects, but not humans or stylized artwork (according to the results and nature of the training data). This point should be discussed in the limitation. 
* Basic shadows of the rendered object look good, but the object tends to stick out/not always harmonize lighting-wise. Could natural lighting harmonization be evaluated? Could results be presented on diverse/colored lighting conditions? The current scenes all have daytime lighting that is similar to the object’s studio lighting.
* Could practical runtime speed/memory footprint be reported?

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
