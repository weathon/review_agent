# Beyond 'Templates': Category-Agnostic Object Pose, Size, and Shape Estimation from a Single View

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 2, 4, 10

## Abstract
Estimating an object’s 6D pose, size, and shape from visual input is a fundamental problem in computer vision, with critical applications in robotic grasping and manipulation. Existing methods either rely on object-specific priors such as CAD models or templates, or suffer from limited generalization across categories due to pose–shape entanglement and multi-stage pipelines. In this work, we propose a unified, category-agnostic framework that simultaneously predicts 6D pose, size, and dense shape from a single RGB-D image, without requiring templates, CAD models, or category labels at test time. Our model fuses dense 2D features from vision foundation models with partial 3D point clouds using a Transformer encoder enhanced by a Mixture-of-Experts, and employs parallel decoders for pose–size estimation and shape reconstruction, achieving real-time inference at 28 FPS. Trained solely on synthetic data from 149 categories in the SOPE dataset, our framework is evaluated on five diverse benchmarks SOPE, ROPE, ObjaversePose, HANDAL and HouseCat6D, spanning 300+ categories. It achieves state-of-the-art accuracy on seen categories while demonstrating remarkably strong zero-shot generalization to unseen real-world objects, establishing a new standard for open-set 6D understanding in robotics and embodied AI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified, category-agnostic framework for estimating an object's 6D pose, 3D size, and dense shape from a single RGB-D image, without requiring CAD models, templates, or category labels at test time. The key idea is to fuse dense 2D features from a vision foundation model (RADIOv2.5) with a partial 3D point cloud using a Transformer encoder enhanced by a Mixture-of-Experts (MoE) for scalable specialization.

### Strengths
This paper tackles the problem of category-agnostic 6D object pose, size, and shape estimation from a single RGB-D image, eliminating the need for CAD models or category labels at test time. The core idea is a unified architecture that fuses dense 2D features from a vision foundation model (RADIOv2.5) with a partial 3D point cloud using a Transformer encoder enhanced by a Mixture-of-Experts (MoE). This design enables the model to simultaneously regress 6D pose and size while performing coarse-to-fine shape reconstruction in a single, real-time forward pass at 28 FPS.

### Weaknesses
The claim of "remarkably strong zero-shot generalization" to unseen real-world objects is not fully substantiated, as the primary real-world benchmarks (ROPE and HANDAL) may still share underlying geometric or semantic commonalities with the synthetic training categories from SOPE, leaving true generalization to entirely novel, long-tail object concepts unproven. The evaluation of shape reconstruction is limited to the synthetic SOPE dataset, failing to demonstrate that the predicted dense shapes are of sufficient quality and accuracy to be useful for real-world robotic applications like grasping, where geometric fidelity is critical. The heavy reliance on synthetic data for training, without an ablation on real data or a thorough analysis of the sim-to-real transfer gap for the shape reconstruction task, raises questions about the method's practical deployment in diverse, unstructured environments. Furthermore, the proposed Mixture-of-Experts (MoE) component, a key architectural novelty, shows only marginal gains in the ablation study, suggesting its contribution may not be as significant as claimed compared to the core fusion and multi-task learning design.

### Questions
1.Provide a more detailed analysis of the semantic or geometric overlap between the "unseen" categories in your zero-shot tests (e.g., in ObjaversePose and HANDAL) and the 149 training categories from SOPE. This would help clarify the true extent of model's generalization beyond the training distribution.

2.The shape reconstruction quality is only quantitatively evaluated on the synthetic SOPE dataset. Can you provide the Chamfer Distance metrics for the real-world ROPE or HANDAL benchmarks to demonstrate that the reconstructed shapes are accurate and useful outside of the synthetic domain?

3.The ablation study shows only a modest performance drop when removing the MoE component. Can you provide further analysis or visualization (e.g., expert routing patterns) to more concretely demonstrate the MoE's role in handling diverse shape distributions, justifying its inclusion?

4.Given the model is trained purely on synthetic data, have you observed any specific failure modes or a significant performance drop on real-world objects with challenging materials (e.g., transparent, specular) for the shape reconstruction task, similar to the issues mentioned for pose estimation on ROPE?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes a 6D pose estimation method with three key designs to achieve category-agnostic pose estimation: 1. Integrate 2D foundation model radio2.5; 2. MoE design; 3.   integrating multi-task in one network including 6D pose, size, and shape. 
This work conducts comprehensive experiements to demonstrate its effectiveness.

### Strengths
This work proposes a 6D pose estimation method with three key designs to achieve category-agnostic pose estimation: 1. Integrate 2D foundation model radio2.5; 2. MoE design; 3.   integrating multi-task in one network including 6D pose, size, and shape. 
This work conducts comprehensive experiements to demonstrate its effectiveness.

### Weaknesses
There is a slight overclaim in this work. As cited in this paper, Any6D (CVPR 2025) has already achieved a zero-reference setup, yet this work still claims "beyond templates" and "category-agnostic" in its title.  How do you achieve better reconstruction performance than the 3D generation model used by Any6D, particularly in occluded regions or on the back of objects?

### Questions
1. The details of the MoE should be provided to ensure the reproducibility of this work. While Figure 2 contains many details, the main content is missing:
   - Do you have shared experts? 
   - What is the routing algorithm?  
   - Do you have a strategy to avoid unbalanced expert load?
   - ...
2. Table content is not aligned. 
   - what about the SGPA and GenPose performance in table.2?  
   - any6D performance in table.1 and table.2, as the any6d can also be seen as a reference free method. 
   - foundation pose performance  for HANDAL in table.3 ?
3. what is the effectiveness of  expert number ?  
4. ObjaversePose is listed as one of the main contribution. But, there is not motivation about 'ObjaversePose', why do you introduce this dataset?   there seems missing something about this new dataset in main content.
5. The explanation of d and n is a little bit far from its first appearance in L194.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed a template-free, category-agnostic single-view RGB-D framework that jointly infers 6D pose, object size, and full 3D shape; the method used a unified architecture and reports real-time inference with cross-category results across multiple benchmarks.

### Strengths
+ The method jointly predicts 6D pose, object size, and full 3D shape in a single forward pass without templates or class labels at test time, promoting geometric consistency across tasks and simpler inference.

### Weaknesses
+ The method relies heavily on “SE(3)-consistent” 2D semantic features (DINO) but does not enforce any explicit geometric equivariance or consistency constraints in its own architecture; as a result, robustness under strong viewpoint, lighting, or material shifts is assumed rather than guaranteed.
+ The proposed MoE replaces the FFN  and is argued to “specialize across diverse object types,” but there is no clear methodological support for this claim. 
+ The multi-task objective simply sums all losses with equal weight for pose, size, shape, which raises the risk that one head dominates optimization.
+ The experimental setting is underspecified.

### Questions
+ Line 81 says “trained purely on synthetic data from 149 SOPE”; was any additional training or fine-tuning done on other datasets?
+ The “category-agnostic” claim is limited to tabletop objects with substantial overlap with SOPE, and the differences from SOPE are not clearly described.
+ FPS reporting is unclear, what parts of the pipeline are included, and is the setup consistent with other methods?
+ The reconstruction comparison switches to a different set of methods, why exclude pose-estimation methods that also perform reconstruction?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper proposes a unified, category-agnostic framework for estimating an object’s 6D pose, size, and dense shape from a single RGB-D image. Unlike prior works that rely on object-specific templates or CAD models, the method performs reference-free inference by fusing dense 2D features from the RADIOv2.5 with partial 3D point clouds. The fused representation is processed by a Transformer encoder enhanced with Mixture-of-Experts (MoE) layers and decoded through two parallel heads for pose–size regression and shape reconstruction. The model achieves state-of-the-art results on seen categories, demonstrates strong zero-shot generalization to unseen objects, and operates in real time. The paper also introduces ObjaversePose, a new photorealistic dataset for open-set 6D understanding.

### Strengths
* The paper demonstrates a carefully reasoned combination of complementary modeling paradigms: foundation-model visual features (RADIOv2.5) for semantic generalization, DGCNN-based local geometric encoding for structure preservation, and Transformer-based global reasoning enhanced by a Mixture-of-Experts (MoE) mechanism for scalable specialization. The design is methodologically sound and practically effective
* The experimental validation is extensive and convincing. Evaluations span four benchmarks (SOPE, ROPE, ObjaversePose, HANDAL) covering both synthetic and real domains, as well as seen and unseen categories. The model consistently outperforms both category-level and reference-based baselines, even under severe occlusion and cross-domain shifts.
* The model achieves 28 FPS on commodity GPUs, a significant improvement over diffusion-based or multi-stage approaches such as GenPose++, which benefits the downstream tasks like robotic manipulation and embodied AI.
* The introduction of the ObjaversePose dataset substantially enriches the evaluation landscape for category-agnostic 6D perception.

### Weaknesses
* Although cross-domain generalization results are strong, the method remains trained entirely on synthetic data. It is unclear how performance scales to visually diverse, texture-rich, or long-tail real-world categories not represented in the synthetic domain.
* While success cases are well illustrated, the paper provides little insight into failure patterns (e.g., reflective surfaces, severe occlusion, or category ambiguity).

### Questions
* Quantitative evaluation under controlled photometric or geometric perturbations (e.g., sensor noise, lighting variation) would strengthen claims about deployment robustness.
* The RADIOv2.5 backbone is frozen during training. Have the authors investigated partial fine-tuning (e.g., LoRA or adapter layers) to improve transfer to 6D pose estimation task

### Soundness
4

### Presentation
4

### Contribution
3
