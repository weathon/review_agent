# SpatialComposer: 3D Spatial Object Insertion via Image Gaussian Composition

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
With the rapid advancement of open-world image generation models in recent years, a series of image editing tasks have achieved excellent performance. However, considering object insertion as a representative example, this task still presents three primary challenges. First, the inserted object should maintain identity consistency with the reference object while preserving the original scene in non-edited regions. Second, the spatial position and scale of the inserted object should be reasonable and align with user expectations. Third, the inserted object should harmonize with other image components, typically involving object style and surface illumination harmonization. To address these challenges, we propose SpatialComposer, which leverages depth-aware image Gaussians to construct a spatially-structured scene representation from a single scene image and models object insertion as Gaussian composition, thereby achieving effective preservation of scene and object identity while enabling precise control over the scale and 3D spatial position of the inserted object. Subsequently, based on pre-trained diffusion generative models, we introduce a simple yet effective refinement method for the object harmonization process. By designating only the Gaussian components corresponding to the inserted object as trainable parameters, SpatialComposer avoids unintended modifications to other regions while simultaneously addressing both object-scene integration and scene detail preservation. Furthermore, recognizing that current object insertion benchmarks lack consideration for depth-aware position control, we construct a specialized benchmark featuring high-resolution scene images with substantial depth complexity.  Comprehensive evaluations demonstrate that SpatialComposer achieves comparable or superior performance over state-of-the-art object insertion approaches across all three aforementioned challenges.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles the object insertion task—adding a reference object into an existing scene image while preserving (1) object identity, (2) spatial realism (position, scale, depth), and (3) visual coherence with the scene (lighting, color, style).
The authors propose SpatialComposer, a method that constructs a depth-aware image Gaussian representation of the scene and formulates object insertion as Gaussian composition in 3D space. Only the Gaussian components corresponding to the inserted object are optimized (a “training-free” refinement step) on top of a pretrained diffusion model to ensure seamless blending.
A new benchmark dataset featuring scenes with higher depth complexity and image resolution is also introduced. Experiments demonstrate that SpatialComposer achieves competitive or superior results compared to existing approaches in terms of identity preservation, spatial control, and visual blending.

### Strengths
Clear problem definition (identity + spatial + style coherence). Methodologically coherent combination of Gaussian representation and diffusion-based refinement. Practical relevance for image editing, AR, and creative applications. New benchmark and comprehensive experiments add value.

### Weaknesses
1. Limited analysis of robustness and generalization to challenging real-world scenes (occlusion, reflection, lighting).
2. High pipeline complexity (depth estimation + Gaussian modeling + diffusion blending).
3. Dataset coverage may be narrow; unclear whether it generalizes to real “in-the-wild” data.
4. Innovation level is moderate; mainly a well-executed combination of existing ideas.
5. Missing deeper discussion on limitations and sensitivity analysis.
6. Some of the images in the paper do not use vector graphics and are very blurry when enlarged.

### Questions
1. How computationally expensive is the full pipeline? What is the average runtime or GPU memory usage for a single insertion?
2. Is the method practical for interactive or real-time use?
3. How sensitive is the method to errors in depth estimation or Gaussian fitting?

### Soundness
2

### Presentation
1

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
This paper introduces "Spatial Composer," a novel framework for object insertion. The core idea is to represent both the scene and the reference object as "Depth-Aware Image Gaussians (DA-ImgGS)," a tailored variant of 3D Gaussian Splatting (3DGS) optimized for single-image reconstruction. Object insertion is then modeled as a composition of these Gaussian sets in 3D space. This approach elegantly resolves occlusion by leveraging the inherent depth-sorted rendering mechanism of 3DGS. To ensure seamless integration, the framework employs a refinement pipeline based on pre-trained diffusion and illumination models, which cleverly updates only the object's Gaussian parameters to guarantee perfect preservation of the background.

### Strengths
True 3D Spatial Controllability and Inherent Occlusion Handling: The paper's primary strength lies in elevating object insertion from a 2D manipulation task to a 3D composition problem. This allows for precise, intuitive control over an object's placement and scale within the scene's depth. Crucially, complex occlusion relationships are handled naturally and correctly by the depth-aware rendering process, a fundamental challenge that most prior works struggle with.

Intelligent and Novel Adaptation of 3D Gaussian Splatting: The authors do not merely apply 3DGS but adapt it effectively for the ill-posed problem of single-view reconstruction. The proposed DA-ImgGS simplifies the representation for this specific task , and the back-projection initialization strategy provides a robust spatial prior from monocular depth estimation, which is critical for achieving a stable and meaningful 3D structure.

### Weaknesses
Insufficient Quantitative and Qualitative Results in the Main Paper: Te experimental validation in the main body of the paper is sparse. The quantitative evaluation relies heavily on a user study, and the newly proposed automatic harmony metrics are not benchmarked against prior work, with their definitions relegated to the appendix. Qualitatively, Figure 2 provides only a limited set of examples, forcing the reader to consult the appendix to assess the method's general performance.

Lack of a Defined Interaction Model: The paper shows final results but fails to address how a user would interact with the system to create them. Manually defining 3D coordinates and scaling factors is not a user-friendly process. The work currently stands as a proof-of-concept, but its practical utility is questionable without a well-defined and intuitive user interface for object manipulation in 3D space.

### Questions
Same as weakness.

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
This paper introduces SpatialComposer, a novel, training-free method for 3D-aware object insertion into images. The core idea is to represent both the scene and the object as "Depth-Aware Image Gaussians" (DA-ImgGS), which are simplified 3D Gaussian Splatting representations initialized using monocular depth estimates. Object insertion is then performed by composing these Gaussian representations—translating and scaling the object's Gaussians within the scene's spatial structure. Finally, a refinement stage uses pre-trained diffusion models to harmonize the inserted object's style and lighting with the scene, while only optimizing the object's Gaussian parameters to preserve the original scene.
The authors identify three key challenges in object insertion: preserving object/scene identity, controlling 3D position/scale, and achieving visual harmony. SpatialComposer directly addresses all three. To evaluate their method, they construct a new, high-resolution benchmark dataset (DAOI) featuring complex spatial structures, which they argue is more suitable than existing benchmarks like TF-ICON. Experiments show that SpatialComposer outperforms several state-of-the-art baselines in user studies and achieves competitive results on quantitative metrics.

### Strengths
1.	The paper is exceptionally clear. The problem is well-defined, the method is explained step-by-step, and the results are presented comprehensively with both quantitative tables and qualitative figures.
2.	The back-projection initialization strategy for Gaussians is clever and efficient, ensuring spatial coherence without requiring multi-view inputs or extensive training, while maintaining high-quality reconstruction.
3.	Introduction of the DAOI dataset is a valuable contribution, providing a more challenging benchmark with diverse, high-res scenes and objects, which better tests depth-aware capabilities compared to existing low-res datasets like TF-ICON.

### Weaknesses
1.	The refinement step is a practical application of existing models (FLUX-Fill-dev, LBM) but does not introduce a novel algorithm. The latent overwriting mechanism is a known technique in diffusion-based editing. The idea of making only the object Gaussians trainable is a good design choice to preserve the scene, but it is more of an engineering decision within an existing framework rather than a core technical contribution.
2.	While training-free, the refinement step depends on pre-trained models like LBM, which could inherit biases or fail in out-of-distribution styles/artistic domains, despite claimed robustness.

### Questions
1.	The authors didn't mention the details of compositing the scene gs and object gs.  Does this happen in a 3dgs editing tool like SuperSplat?  If that's the case, the work feels disjointed as the user will need to reconstruct the gs model, manually merge and place the 2 models, and finally run the refinement process. 

  The final step includes using two existing models to refine an image, which does not add much novelty to the work.  The second composition step also does not include technical contribution.  Thus the overall technical contribution is quite limited, though the authors deliver a great solution for object insertion task. 

2.	If the inserted object takes up a large part of the original image, the refinement process may affect the background part (non-targeted regions) due to the generative capabilities as also mentioned by the authors.  Do you have any idea to address such problems?

### Soundness
2

### Presentation
3

### Contribution
2
