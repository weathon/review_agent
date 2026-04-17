# Rosetum3D: A Large-Scale 3D Vision Dataset from Preharvest Roses

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
The global rose cultivation industry has experienced continued expansion driven by rising demand for cut flowers and botanical extracts. However, automated pre-harvest quality grading faces significant challenges due to occlusion-heavy canopy environments and morphological variations across growth stages. While 3D computer vision offers solutions for localization tasks, progress has been hindered by the lack of large-scale datasets with phenotypic keypoint annotations that capture plant architecture. To bridge this gap, we introduce Rosetum3D, constructed via occlusion-robust multi-view RGB-D capture protocols in commercial greenhouses. We provide fine-grained 2D localization using bounding boxes and botanically defined keypoints with 3D structures recovered through depth backprojection. Models trained on Rosetum3D have achieved 2D/3D rose localization, a crucial step for automated pre-harvest quality grading and growth monitoring. Beyond localization, Rosetum3D serves as a benchmark for agricultural vision tasks, including 2D rose object detection, local feature matching, and depth estimation. By enabling data-driven precision agriculture, Rosetum3D paves the way for robotic harvesting systems and AI-driven yield prediction in protected cultivation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Rosetum3D, a dataset for 3D vision in rose cultivation captured using RGB-D cameras in commercial greenhouses. The dataset includes multi-view sequences with 2D bounding boxes and botanically-defined keypoints, which are backprojected to 3D using depth maps.

### Strengths
- Addresses an underserved domain (agricultural 3D vision) with clear practical applications for robotic harvesting and yield prediction
- Comprehensive multi-view RGB-D capture with synchronized depth, enabling multiple downstream tasks
- Thoughtful annotation protocol using 2D labeling with depth backprojection, avoiding fragile 3D reconstruction in occluded environments
- Rigorous validation methodology (ruler-based measurement with 0.14 cm MAE)

### Weaknesses
- The scientific contribution is limited to data collection. While valuable, the paper doesn't introduce novel methods, algorithms, or theoretical insights. For a top-tier venue like ICLR, datasets typically need to enable fundamentally new research directions or demonstrate surprising findings.

- The scale claim ("large-scale") is somewhat overstated. With 20k+ images, Rosetum3D is substantial but modest compared to datasets like ScanNet (2.5M views) or nuScenes (1.4M frames) mentioned in comparisons. The number of annotated keypoints or 3D structures isn't clearly stated.

- Limited analysis of what makes rose cultivation uniquely challenging. The paper mentions occlusions and morphological variation but doesn't quantify these challenges or demonstrate that existing methods fail specifically due to domain characteristics rather than lack of training data.

- The multi-task benchmarking is comprehensive but shallow. Baseline experiments mostly apply off-the-shelf methods without domain adaptation or analysis of failure modes. What vision problems are specific to roses versus other plants or cluttered scenes?

### Questions
1. Can you provide more detailed statistics: total number of annotated keypoints, distribution across growth stages and varieties, occlusion severity metrics?

2. What is the inter-annotator agreement for 2D keypoint annotations? Have you measured this systematically?

3. How does depth quality vary with distance from camera? Are there systematic errors in backprojection at different depths or viewing angles?

4. You mention "botanically defined" keypoints—can you provide botanical references or expert validation for these choices? How do they map to actual quality grading criteria used in industry?

5. Have you compared RGB-D backprojection with alternative 3D supervision strategies (e.g., multi-view triangulation, SfM)? Why is backprojection definitively better for this domain?

6. Will you release code for the annotation pipeline and evaluation protocols? What about pretrained models?

7. The ruler validation is helpful, but have you validated against other ground truth (e.g., manual measurements of actual stem lengths)?

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
3

### Summary
This paper introduces Rosetum3D, a new large-scale RGB-D dataset collected in commercial rose greenhouses for 3D computer vision applications in floriculture. The data is captured across diverse rose varieties and growth stages, aiming to capture the structural variability and heavy occlusions typical of real cultivation environments. It contains synchronised RGB-D sequences that are carefully annotated with 2D bounding boxes and botanically defined 3D keypoints, which contain geometric and morphological details of rose plants. 

The authors validate their annotation accuracy through quantitative measurements and propose two new evaluation metrics, LKS and LKS3D, designed for the linear structure of rose stems. They benchmark several standard tasks, including object detection, keypoint localisation, depth estimation, and local feature matching, using both CNN and transformer-based models. Overall, Rosetum3D is intended to serve as a foundational resource for research on rose phenotyping, growth monitoring, and robotic harvesting.

### Strengths
This paper tackles a clear and underexplored problem in floriculture vision, focusing on pre-harvest rose monitoring where dense occlusions and fine structural details make visual perception particularly challenging.

1. The dataset collection and annotation process are carefully designed, featuring botanically defined 3D keypoints and a quantitative validation study that demonstrates strong geometric accuracy (0.14 cm MAE).

2. The dataset is well constructed and diverse, covering multiple rose varieties and growth stages, and it supports several relevant vision tasks, including detection, keypoint localisation, and depth estimation.

3. The authors also make improvements to the OKS-based evaluation metric, introduce LKS-based evaluation metric, and establish baseline benchmarks using both CNN and transformer-based architectures.

### Weaknesses
1. While the newly proposed dataset and evaluation design are valuable, the paper does not introduce a new learning method or theoretical result. As a result, the novelty lies more in the resources than in the methodology, which may fall short of ICLR’s expectations for algorithmic innovation.

2. Scope is narrower than comparable plant datasets; for example, the Crops3D (Zhu et al., 2024) dataset spans multiple species and field conditions, while PlantGaussian (Shen et al., 2025) models plants across time and scenes. In contrast, Rosetum3D’s rose-only focus limits generalisation. A small cross-species transfer experiment or adding a subset from another flower or crop would strengthen the case for broader impact.

3. The benchmarks provide broad task coverage detection, 2D/3D keypoints, depth estimation, and feature matching, but the analysis stays mostly at a summary level. It’s not very clear, why models succeed or fail?. For instance, while 2Dkeypoint AP and LKS/LKS3D results are reported, there’s no breakdown showing which keypoints perform poorly (e.g., corolla centre vs. stem base) or how performance varies with occlusion, scale, or viewpoint, even though the dataset distinguishes between visible and invisible keypoints. Similarly, there’s no quantitative discussion of scene or occlusion characteristics, and no cross-domain or transfer experiments to assess generalisation. The LKS and LKS3D metrics appear reasonable for stem structures, but without validation against human evaluation or downstream task relevance, their practical significance remains uncertain.

4. Although Rosetum3D contains around 21K RGB-D images, these were collected from only 20 greenhouse scenes (17 for training and 3 for testing). The overall image count is reasonable for a specialised domain, but the small number of distinct capture environments limits the dataset’s spatial and contextual diversity.

5. The zero-shot depth models pretrained on NYU show a clear drop in performance on Rosetum3D, which is expected, but the paper does not investigate the underlying reasons. Factors such as sensor differences from the structured-light RGB-D setup, the optical properties of rose petals and leaves, the mixed greenhouse lighting, and the short-range, occlusion-heavy geometry could all contribute to this gap. However, the paper provides no breakdown of errors by range, occlusion level, viewpoint, or material type, nor does it include any ablation studies or adaptation experiments. As a result, we see that performance decreases, but the work offers little understanding of why this happens or how it might be addressed.

### Questions
1. Analysis of failure cases: The benchmarks span several tasks, but the analysis remains limited. Could you provide more insight into where the models fail, for example, which keypoints are most affected by occlusion, lighting variations, or viewpoint changes?

2. Validation of new metrics: The new LKS and LKS3D metrics seem reasonable, but have you validated that they correspond to perceived or practical improvements, for example, through expert evaluation or correlation with 3D reconstruction accuracy?

3. Depth generalisation: The zero-shot depth models pretrained on NYU perform noticeably worse on Rosetum3D. Can you analyse the main factors contributing to this gap --  such as sensor characteristics, material properties, or short-range geometry -- and discuss potential adaptation strategies?

4. Cross-scene and cross-species generalisation: Have you tested how models trained on Rosetum3D generalise across different greenhouse scenes or to other flower or crop datasets? Such an experiment could clarify whether the dataset encourages transferable representations beyond roses.

Overall, the paper presents a well-executed, clearly documented dataset that addresses a meaningful gap in floriculture research. However, it suffers from a limited methodological innovation and relatively shallow analysis.

### Soundness
3

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
This paper introduces Rosetum3D, a multi-view RGB-D dataset captured in operational greenhouses, enabling 2D/3D localization and phenotyping of pre-harvest roses. The authors annotate each rose with a bounding box and nine botanically meaningful keypoints (five primary + four interpolated), then back-project these to 3D via synchronized depth, avoiding brittle full 3D reconstruction in dense, occlusion-heavy canopies. The capture setup uses Orbbec structured-light stereo sensors (1280×800 @ 30 Hz) with a prescribed walking protocol and calibration pipeline; annotation fidelity is validated with ruler measurements showing 0.14 cm MAE across 137 cases. The dataset supports benchmarks for detection, 2D/3D keypoint localization, monocular depth estimation, and local feature matching (a Rosetum3D-1800 split), and proposes new line-aware metrics (LKS/LKS3D) better aligned with stem-like morphology. Baselines indicate moderate detection (e.g., YOLOv5 AP≈35) and that top-down keypoint methods excel when perfect boxes are provided; fine-tuned depth models substantially outperform zero-shot ones on this domain. If released, Rosetum3D would fill a notable gap in floriculture vision and provide a concrete testbed for pre-harvest grading and robotic harvesting research.

### Strengths
- Well-motivated, carefully engineered data pipeline for a genuinely under-served application domain (pre-harvest floriculture), with practical choices (structured-light RGB-D, greenhouse-robust protocol) and thorough calibration details. 
- Annotation and metric design grounded in plant morphology, including five biologically defined keypoints plus inter-points, and the LKS/LKS3D metrics that emphasize line orientation/endpoints, appropriate for stem-like structures. 
- Comprehensive benchmarks spanning detection, 2D/3D keypoints, depth, and feature matching (Rosetum3D-1800), with both zero-shot and fine-tuned evaluations and clear takeaways (detector-limited top-down pipelines; fine-tuned depth wins).

### Weaknesses
- Inconsistent dataset statistics: the text states 66,848 annotated rose objects, while Table 1 totals 46,848—a substantial discrepancy that needs reconciliation and suggests QA issues in reporting. 
- Scale and diversity may be limited relative to the “large-scale” claim, as only 20 scenes (~21k images ) from two camera models, one capture protocol (single walking trajectory and tilt), and one crop in one environment type are used, which may constrain generalization. Clarify sites, varieties, growers, and capture days/seasons. 
- Metric clarity and robustness: LKS primarily uses the first/last keypoints, along with an angle threshold. However, results for some models hinge on endpoint prediction, raising concerns about metric sensitivity and potential gaming. LKS3D introduces normalized endpoint error with an unexplained constant (K=2) and threshold choices (e.g., “ACC@1&15°”), but units/interpretation are underspecified. Provide ablations and rationale. 
- Missing or underspecified release details: no license, access policy, privacy/compliance statement with the commercial greenhouses, or a validation split (train/test only), which hinders reproducibility and fair model selection.

### Questions
- Which figure is correct for the total number of annotated objects: 46,848 (Table 1) or 66,848 (main text)? Please provide a cleaned statistics table (scenes, images, objects, keypoints, and invisibility rates) for training, validation, and testing. 
- For LKS/LKS3D, could the authors justify the endpoint focus, angle thresholds, and the constant (K)? Have the authors evaluated robustness to endpoint mis-annotation or to varying which keypoints define the “line”?

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
The paper introduces Rosetum3D, a large-scale RGB-D dataset for 3D vision in rose cultivation, targeting pre-harvest quality grading and growth monitoring in greenhouses. Data are collected with structured-light RGB-D cameras (Orbbec Gemini 335L/336L) under realistic greenhouse conditions, producing 21k synchronized RGB-D frames with nearly 47k annotated rose instances. Each instance includes 2D bounding boxes and 9 biologically defined keypoints, which are back-projected into 3D using the corresponding depth maps. The paper also proposes two new evaluation metrics, LKS (Line-based Keypoint Similarity) and LKS3D, designed for linear plant structures. Rosetum3D serves as a benchmark for multiple tasks, including object detection, keypoint localization (2D/3D), monocular/multi-view depth estimation and feature matching. Evaluation is performed using standard models such as YOLOv5, ViTPose and BinsFormer. Annotation accuracy is validated using calibrated rulers inserted in the scene, showing a mean absolute 3D error of 0.14 cm.

### Strengths
1. The paper fills a gap in agricultural 3D vision by introducing a novel floriculture-oriented RGB-D dataset with dense 2D/3D annotations.
2. Data collection, calibration and annotation processes are described in detail and validated quantitatively.
3. The paper introduces some metrics inspired by human-pose estimation to assess the quality of 2D/3D keypoint localization considering the linear plant geometry. These metrics may also be considered in other keypoint localization tasks concerning plants.
4. The dataset is contributing to applications concerning robotic harvesting, phenotyping, and precision agriculture, bridging vision research with real-world greenhouse applications.

### Weaknesses
1. The methodological novelty introduced by this work is limited. The main contribution lies in the dataset itself and in the introduction of the pose-inspired metrics. It would also be important to introduce a baseline method as a reference for the tasks targeted by the dataset introduced. 
2. The dataset focuses solely on roses. It would be interesting to discuss whether the acquisition method, the LKS/LKS3D and the conclusions generalize to other floriculture or crop species.
3. The LKS/LKS3D metrics introduced in this work should be further analyzed. For instance, a more detailed discussion on the justification of the angle-based metrics would be valuable, as well as a discussion on the comparison with the OKS/MPJPE metrics on which they are based. The latter would help to assess how the new metrics contribute to the assessment of the localization quality.
4. Although robotic harvesting and yield prediction are motivating applications, the paper does not show any relevant use case.
5. Some complementary related work as [R1], [R2], and [R3] can be considered to provide better context.

### Minor comments
- Section 3.3 title contains a typo: “Staistics”

[R1] Mertoğlu et al., “PLANesT-3D: A New Annotated Data Set of 3D Color Point Clouds of Plants”. In Signal Processing and Communications Applications Conference (SIU) (pp. 1-4), 2023

[R2] Dutagaci et al., “ROSE-X: an annotated data set for evaluation of 3D plant organ segmentation methods”. Plant methods, 16(1), 28, 2020

[R3] Franchetti et al., “Vision based modeling of plants phenotyping in vertical farming under artificial lighting”. Sensors, 19(20), 4378. 2019

### Questions
1. Have the authors evaluated the performance of models trained on ScanNet or Crops3D on Rosetum3D?
2. How does LKS/LKS3D correlate with standard metrics like OKS or MPJPE? Are the rankings consistent?
3. Is there any temporal or multi-view consistency annotation to enable sequence-based learning?

### Soundness
3

### Presentation
3

### Contribution
2
