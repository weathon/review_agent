# GOLDILOCS: GENERAL OBJECT-LEVEL DETECTION AND LABELING OF CHANGES IN SCENES

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
We propose GOLDILOCS: a novel zero-shot, pose-agnostic method for object-level semantic change detection in the wild. While supervised Scene Change Detection (SCD) methods achieve impressive results on curated datasets, these models do not generalize and performance drops on out-of-domain data. Recent Zero-Shot SCD methods introduced a more robust approach with foundational models as backbone, yet they neglect the 3D aspect of the task and remain constrained to the image-pair setting. Conversely, 3D-centric SCD methods based on 3D Gaussian Splatting (3DGS) or NeRFs require multi-view inputs, but cannot operate on an image pair. Our key insight is that SCD can be reformulated as a 3D reconstruction problem over time, where geometric inconsistencies naturally indicate change. Although previous work considered viewpoint difference a challenge, we recognize the additional geometric information as an advantage. GOLDILOCS uses dense stereo reconstruction to estimate camera parameters and generate a pointmap of the commonalities between input images by filtering geometric inconsistencies. Rendering the canonical scene representation from multiple viewpoints yields reference images that exclude changed or occluded content. Rigid object changes are then detected through mask tracking, while nonrigid transformations are identified using SSIM heatmaps. We evaluate our method on a variety of datasets, covering both pairwise and multi-view cases in binary and multi-class settings, and demonstrate superior performance over prior work, including supervised methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GOLDILOCS, a zero-shot, 3D-aware framework for object-level scene change detection (SCD) from image pairs or multi-view image sets. The method reformulates SCD as a combination of stereo 3D reconstruction, semantic segmentation, mask tracking, and non-rigid change detection via SSIM-based comparison. It leverages foundational models (MASt3R for 3D reconstruction and SAM2 for segmentation) without additional training and introduces a standardized object-level change taxonomy including Removed, Added, Moved, and Warped. Experiments on both synthetic and real datasets (ChangeSim, VL-CMU-CD, 3DGS-CD, NeRFCD) demonstrate strong zero-shot performance, especially in IoU and F1 metrics, while avoiding expensive multi-view reconstruction at test time.

### Strengths
1. Zero-shot object-level change detection with 3D geometric reasoning.

2. Modular pipeline leveraging stereo reconstruction and segmentation models.

3. Introduces a clear object-level change taxonomy (Removed, Added, Moved, Warped).

4. Faster inference compared to traditional 3D reconstruction pipelines.

5. Comprehensive experiments across multiple datasets.

### Weaknesses
Ⅰ. Parameter Sensitivity:

SSIM Thresholds: The thresholds used for non-rigid change detection significantly affect results. A threshold set too high may miss subtle deformations, while a threshold too low may falsely classify static objects as changed.

Label Priority Ordering: The per-pixel prediction relies on the priority order (Warped > Moved > Removed > Added). This can affect nested or overlapping object changes; for example, small objects inside a larger moved object may be suppressed and misclassified.

Ⅱ. Limited Ablation:

Ablation experiments mainly focus on stereo reconstruction and novel view synthesis, but the contribution of non-rigid change detection and mask propagation is not thoroughly analyzed.

The effects of individual components such as depth filtering, cross-view voting, and mask propagation on each change type (added, removed, moved, warped) are not quantified.

Ⅲ. Data Limitations:

The evaluation datasets lack extreme real-world scenarios, including large occlusions, fast-moving objects, or highly dynamic scenes, which may cause 3D reconstruction or mask tracking failures.

### Questions
1. How sensitive is GOLDILOCS to SSIM thresholds and mask tracking parameters across datasets?

2. How does it handle partially occluded or highly dynamic objects?

3. Could hierarchical segmentation improve detection for nested objects?

4. Would longer temporal sequences improve detection robustness?

5. Any plans for evaluation in real-world dynamic scenes?

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
The paper introduces GOLDILOCS, a zero-shot and pose-agnostic approach for object-level semantic change detection (SCD). The key idea is utilizing  3D geometric inconsistencies to detect changes. With the estimated pointmaps by dense stereo reconstruction models , the method generates reference images of clean point clouds. Changes are then identified via mask tracking for rigid objects and SSIM heatmaps for non-rigid ones. Evaluations on various datasets show better performance over baseline methods.

### Strengths
- With the help of SAM2, the proposed method is zero-shot.
- The idea to utilize SAM and MASt3R to SCD problem is practical.
- The proposed method achieved best overall performance in most settings.

### Weaknesses
- The proposed method depends on two large foundation models, which demands large computations and may limit its applications.
- Lack of implementation details/runtime comparisons to show the extra cost of the proposed pipeline.
- The peromance improvement is marginal (e.g, resutls in Tab.4.)

### Questions
- Is there analysis of how the performance/generality of 3D reconstruction model affect the result of SCD results? Are there any visualization of reconstructed results?
- The “conflict resolution” in Sec. 4.2 is confusing, what does it mean?
- Is the method end-to-end? Or they are rule-based after getting the segmentation/reconstruction by existing models.
- Citation of 3DGS-CD of Tab.1 differs from that of Tab. 2.

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
The authors introduce GOLDILOCS, a framework that establishes a standardized taxonomy for object-level change labeling in 3D scenes. The approach is grounded in geometric reconstruction and visibility reasoning, where an object is defined as any visually identifiable entity ranging from volumetric elements like boxes or furniture to planar textures such as ink stains on paper.

GOLDILOCS reconstructs the 3D geometry of a scene and employs novel-view rendering, segmentation, and temporal tracking of object masks to detect both rigid and  non-rigid transformations over time. The framework is evaluated across multiple datasets, demonstrating consistent and satisfactory improvements over state-of-the-art methods in object-level change detection and labeling.

### Strengths
The paper introduces a well-defined and systematic taxonomy for object-level change labeling in 3D scenes, addressing the lack of standardization in existing approaches.

The authors adopt an inclusive definition of object, covering both volumetric entities (e.g., furniture) and planar or texture-based elements (e.g., ink blotches), which enhances the framework’s generality and applicability.

The integration of 3D geometric reconstruction and visibility-based reasoning allows for more precise understanding of scene changes compared to purely 2D or image-based methods.

The use of segmentation and mask tracking across time ensures consistent identification of objects and their transformations, including non-rigid changes.

The pipeline’s inclusion of novel-view rendering improves the robustness of change detection under different viewpoints and occlusions.

The model has been tested on multiple datasets, demonstrating strong generalization and satisfactory improvements over state-of-the-art methods.

By combining geometry, visibility, and temporal reasoning, GOLDILOCS can be extended to practical domains like robotics, AR/VR scene updates, and autonomous navigation.

### Weaknesses
The proposed framework is computationally heavy, as it integrates conventional segmentation and 3D stereo reconstruction modules, making it less efficient for real-time or large-scale applications.
The 3D stereo reconstruction component appears to rely on existing methods with minimal innovation, reducing the novelty of that part of the pipeline.
The paper lacks clear technical details about the underlying 3D reconstruction model its architecture, parameters, and optimization strategy are not adequately discussed.
The motivation for selecting specific existing models for segmentation and reconstruction is not well justified. The rationale behind these design choices should have been elaborated to strengthen the methodological clarity.
The computational complexity, including FLOPs, memory requirements, and inference time, is not reported. Such analysis would be valuable for early reference and comparative evaluation.
Given its reliance on multiple integrated modules, the pipeline may face scalability and deployment challenges in dynamic or resource-limited environments.

### Questions
The proposed framework is computationally heavy, as it integrates conventional segmentation and 3D stereo reconstruction modules, making it less efficient for real-time or large-scale applications.
The 3D stereo reconstruction component appears to rely on existing methods with minimal innovation, reducing the novelty of that part of the pipeline.
The paper lacks clear technical details about the underlying 3D reconstruction model its architecture, parameters, and optimization strategy are not adequately discussed.
The motivation for selecting specific existing models for segmentation and reconstruction is not well justified. The rationale behind these design choices should have been elaborated to strengthen the methodological clarity.
The computational complexity, including FLOPs, memory requirements, and inference time, is not reported. Such analysis would be valuable for early reference and comparative evaluation.
Given its reliance on multiple integrated modules, the pipeline may face scalability and deployment challenges in dynamic or resource-limited environments.

### Soundness
2

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
4

### Summary
The paper presents a new method for scene change detection. The new method, entitled GODILOCS, offers an alternative for zero-shot, class-agnostic object-level change detection, leveraging recent advances in foundation models. The method is 3D geometry aware, firstly by calculating the 3D geometry of the scene using Mast3R. The created point-maps are filtered for inconsistent regions (likely corresponding to changes in the scene) and then images are rendered using the same viewpoints such that the images can be compared. Discrepancies between the rendered and initial images help identify object-level changes, with the help of object-level masks created by SAM2. A foundation model for tracking is the afterwards deployed to differentiate between rigid and non-rigid changes. The presented method achieves state-of-the-art results compared with a set of relevant baselines both on synthetic and real-world dataset.

### Strengths
The paper addresses an interesting topic: scene change detection. Especially, it also helps sub-categorize the change into added, removed, or moved objects (rigid and non-rigid changes). Scene change detection is an understudied and interesting topic. 

The paper leverages the latest trends in foundation models, offering a class-agnostic, zero-shot method for scene change detection. More specifically, using Mast3R, the proposed method creates a 3D reconstruction of the scene to have a 3D geometry-aware 3D change detection, and then uses a class-agnostic instance segmentation model (SAM2) to identify objects in the scene and track them using DEVA].

The paper achieves SoTA results, compared to other baselines, validating the motivation of having a 3D geometry aware change detection methods when just images of a scene are given.

The paper is well-motivated, and well-written. The methodology is thoroughly explained.

### Weaknesses
The novelty of the paper is limited. The paper is mainly extending the work of [1] that uses Mast3R towards creating a geometry aware-scene change detection method. Mast3r offers the 3D reconstruction of the scene, along with the poses and the calibration matrices of the views. Through that, the proposed method can then compare depth maps and render novel views, using the same viewpoints. On the rendered views, the methodologies applied in [1] are then deployed.

More baselines should be included, including CYWS-3D (Sachdeva & Zisserman, 2023b). Even though the paper offers bounding boxes as changing regions, SAM could be deployed to get the mask for the bounding boxes. Moreover, since the method does not categorize into the different types of changes, the authors could present comparative metrics on all changes and not specific categories.

Most importantly, methods that reason in 3D such as (Taneja et al., 2011), [Adam et al., Objects can move: 3d change detection by geometric transformation consistency, ECCV 2022], [Palazzolo and Stachniss, Fast image-based geometric change detection given a
3d model, ICRA 2018] are not included in the experimental evaluation and are mostly not discussed in the related work. Such methods could be easily adopted to the given use case by using Mast3r to obtain the 3D models they reason on, which could result in strong baselines. Right now, the presented baselines do not integrate any kind of knowledge about the 3D scene.

[1] Kannan, Shyam Sundar, and Byung-Cheol Min. "Zeroscd: Zero-shot street scene change detection." 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025.

### Questions
Is the integration of Mast3R really necessary? Given that posed, calibrated images were used, using a simpler and more lightweight SfM method, wouldn’t the traditional render and compare lead to similar results? That would be an interesting experiment that would explore the trade-off between computational resource needed for the method and the success of the method.

Since the proposed method reasons in 3D, why are its results not also evaluated in 3D? Using an appropriate dataset, e.g., 3RScan [1] and extending the method by back-projecting the highlighted changes into 3D would also give interesting insights on the success of the method in the 3D environment.

[1] Wald, Johanna, et al. "Rio: 3d object instance re-localization in changing indoor environments." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

### Soundness
3

### Presentation
3

### Contribution
2
