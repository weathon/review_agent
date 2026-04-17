# LaRI: Layered Ray Intersections for Single-view 3D Geometric Reasoning

- Decision: Reject
- Scores: 8, 4, 6

## Abstract
We present Layered Ray Intersections (LaRI), a fully supervised method for occluded geometry reasoning from a single image. Unlike conventional depth estimation, which is limited to visible surfaces, LaRI predicts multiple surfaces intersected by the camera rays using layered point maps. Compared to the existing approaches that leverage neural implicit representations or iterative refinement, LaRI achieves complete scene reconstruction in one feed-forward pass, enabling efficient and view-aligned geometric reasoning to underpin both object-level and scene-level tasks. We further propose to predict the ray stopping index, which identifies valid intersecting pixels and layers from LaRI’s output. To better underpin and evaluate this task, we build an annotation pipeline using rendering engines, construct annotations for five public datasets, including synthetic and real-world data covering 3D objects and scenes. As a generic method, LaRI’s performance is validated in object-level and scene-level reconstruction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces LaRI,  a simple supervised method for estimated, potentially occluded. 3D geometry from a single image. Unlike traditional depth estimation, which only predicts the first visible surface along each camera ray, LaRI predicts multiple surfaces intersected by each ray, using layered point maps. This enables the reconstruction of both visible and unseen geometry in a single feed-forward pass, supporting both object-level and scene-level tasks.

Paper key innovations are:
1) simple architecture including a multi-layered points maps output and a ray stopping index to predicts multiples surfaces intersected by a ray.
2) new training data, annotating both visible and occluded ray intersection.

Author evaluated their approach for both scene-level and object-level 3D reconstruction and show that LaRI achieves comparable or better accuracy than state-of-the-art models (TRELLIS, SPAR3D.. ) with fewer parameters and faster inference.

### Strengths
- Unified Framework: LaRI provides a simple efficient framework for both object-level and scene-level 3D reconstruction, bridging a gap between methods that typically focus on one or the other.

- Occlusion Reasoning: The method explicitly models occluded surfaces enabling more complete scene understanding.

- Efficiency: LaRI achieves competitive results with significantly fewer parameters and faster inference compared to large generative models (e.g., TRELLIS, SPAR3D).  All ray intersections are predicted in one forward pass, avoiding iterative or per-point querying common in NeRF-based or grid-based methods.

- Ablation Studies: The paper provides thorough ablations on layer number, pre-training, and mask prediction, supporting the design choices.

### Weaknesses
- As a deterministic approach, LaRI may struggle with highly occluded regions where input context is limited, leading to inaccurate modeling.
-  While LaRI matches or exceeds most methods, it is slightly outperformed by TRELLIS in canonical ground truth object-level evaluation, though TRELLIS uses with far more parameters.

### Questions
- What is the impact of the scale and shift parameter?
- How does LaRI generalizes to more complex input, beyond indoor or synthetic data ?

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
5

### Summary
This work targets to the task of reconstructing 3D from single-view images, in both object and scene senarios. For object-level, there are many image-to-3D large models which can output a complete model from a single image. And for scene-level, there are also many works that can estimate geometry for visible surfaces, such as the popular depth anything, vggt etc. 

This work argues that exsiting works are usually not able to reason the invisible structure together with visible area in parallel. It presented a novel 3D representaion - LaRI - which is actually a multiple pointmaps for multiple layers which covers both visible and invisible regions in 3D space. Then, an end-to-end model is trained to map the input image into those pointmaps. Experiments verifies the ability of the proposed model to reason 3D especially for invisible parts.

### Strengths
- The proposed representation is novel for single-view reconstruction tasks.
- The model designs are reasonable.

### Weaknesses
- For the evaluation of object-level reconstruction, I don't think the current evaluation is fair. The paper aims to single-view reconstruction, however, it cannot support the mesh output while many existing works can do mesh reconstruction, such as Trellis. Only point cloud output is not valuable for production. This makes me doubt the practical usage of LaRI representation. 
- Actually, many objects have interial content. For example, a car contains seats in it. From this sense, I think LaRI may be useful for reconstructing 3D interial structures from images. Thus, I will appreciate if the authors can change the aim to interial reconstruction not only the outer surface reconstruction. 
- For scene-level reconstruction, the current results show that the proposed method can acheive sota but only for invisble regions. This is also one of my major concerns. I think a good model should be the best consistently for both visible and invisible regions, otherwise, we still require another model to pick the best model for different regions.

### Questions
No more.

### Soundness
2

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
4

### Summary
This paper presents a method that regresses 3D point maps in layers, representing both visible and occluded surfaces. As with other point map regression methods, the output is inherently in camera coordinates.
Given a single RGB image of size (H, W, 3) as input, it predicts stacked point maps of size (L=5, H, W, 3) and an integer map (H, W) representing the number of valid intersections, using two separately trained networks. The results show advantage in being able to predict invisible surfaces that most other methods cannot, and demonstrates results comparable to other regression methods.

### Strengths
Straightforward, minimal (clean) regression setup in which the only input is an RGB image.

The data annotation pipeline used to generate ground truth was the main factor that made this research possible where others couldn't before due to the lack of available data. The authors built the labels across five datasets with careful coordinate handling (I agree this is non-negligible engineering effort (A.1)).

### Weaknesses
The idea itself can be seen as incremental. Storing multiple per-ray samples is a known representation. It would have been stronger if the experiments helped other researchers learn more about the problem of predicting invisible surfaces and what such regression models actually learn. But considering typical standards for accepted papers at similar venues, this does not seem to be a major weakness. However, I do have some concerns about the experimental setup and validation.

Evaluation target: If I understand correctly, everything except the canonical object‑level evaluation (Table 2, left) uses the same layered camera‑aligned representation that the model was trained with (presumably with the same number of layers?). I think both global and camera coordinate evaluation should use a more canonical representation (e.g. point-mesh evaluation, points sampled on external surfaces of mesh, etc) more independent of the specific representation the model is trained with. Were the layers enough to cover all surfaces of the object/scene? If not, methods that predict more complete shapes might be disadvantaged (as with the object-level methods used for comparison). It might be good to see some coverage statistics (e.g. for each dataset, plot "mesh surface coverage vs. number of layers")

Appendix (under Objaverse annotation) says synthetic objects with problematic internal artifacts are filtered, but how was this done exactly? Also what about for datasets other than Objaverse? I would imagine a lot of meshes contain surfaces that cannot be seen from outside (which would be impossible for methods like CUT3R to predict), and if we don't get rid of them completely and canonically, the learned structure will be dataset specific. One could try making the meshes watertight, and then treat it as a solid and remove all structures in the internal volume. 

L317 says ScanNet++ was not used for quantitative evaluation due to potentially incomplete unseen regions. But I think it would still be good to see this evaluation. Incomplete unseen regions are common in non-synthetic datasets. How problematic are they for evaluation?

Any comparison with non-depth/pointmap based prediction in the scene-level experiments? Including at least one such representative method would strengthen the positioning (e.g. methods that predict meshes or multiple objects in camera coordinates).

### Questions
Training architecture: My understanding is that two separate networks are used for layered points and stop index. Would joint training with a shared encoder help or hurt?

Focal‑length sensitivity: L361 says depth-based methods used "predicted focal lengths" for point cloud conversion, but I wonder how sensitive the results are to focal length error. Most depth prediction methods aren't meant to be evaluated in 3D space. How much would they improve if they used ground truth focal length or the focal length estimated from LaRI's 3D point frustum?

Multi‑task potential: Could joint learning with readily available labels (object categories, instances) help? The datasets used to render supervision often include such annotations, for "free".

I would be interested in seeing:
- Percentage of seen vs unseen surfaces in the ground truth dataset.
- Prediction accuracy per object category.

Table 6 lists 314M parameters as "Ours." Does this include both point map and index predictors? What were their respective model sizes?

### Soundness
2

### Presentation
3

### Contribution
2
