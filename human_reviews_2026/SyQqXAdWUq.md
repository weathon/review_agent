# InclusiveVidPose: Bridging the Pose Estimation Gap for Individuals with Limb Deficiencies in Video-Based Motion

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
Approximately 445.2 million individuals worldwide are living with traumatic amputations, and an estimated 31.64 million children aged 0–14 have congenital limb differences, yet they remain largely underrepresented in human pose estimation (HPE) research. Accurate HPE could significantly benefit this population in applications, such as rehabilitation monitoring and health assessment. However, the existing HPE datasets and methods assume that humans possess a full complement of upper and lower extremities and fail to model missing or altered limbs. As a result, people with limb deficiencies remain largely underrepresented, and current models cannot generalize to their unique anatomies or predict absent joints. To bridge this gap, we introduce InclusiveVidPose Dataset, the first video-based large-scale HPE dataset specific for individuals with limb deficiencies. We collect 313 videos, totaling 327k frames, and covering nearly 400 individuals with amputations, congenital limb differences, and prosthetic limbs. We adopt 8 extra keypoints at each residual limb end to capture individual anatomical variations. Under the guidance of an internationally accredited para-athletics classifier, we annotate each frame with pose keypoints, segmentation masks, bounding boxes, tracking IDs, and per-limb prosthesis status. Experiments on InclusiveVidPose highlight the limitations of the existing HPE models for individuals with limb deficiencies. We introduce a new evaluation metric, Limb-specific Confidence Consistency (LiCC), which assesses the consistency of pose estimations between residual and intact limb keypoints. We also provide a rigorous benchmark for evaluating inclusive and robust pose estimation algorithms, demonstrating that our dataset poses significant challenges. We hope InclusiveVidPose spur research toward methods that fairly and accurately serve all body types. The project website is available at: [InclusiveVidPose](https://anonymous-accept.github.io/inclusivevidpose/).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
InclusiveVidPose introduces the first large scale video based human pose estimation dataset focused on individuals with limb deficiencies, addressing the lack of representation in existing benchmarks which assume anatomically intact bodies . The dataset contains 313 videos totaling 327k frames featuring nearly 400 participants with amputations, congenital limb differences, and prostheses, and it adds eight residual limb end keypoints to accurately model altered anatomy . Each frame includes pose keypoints, segmentation masks, bounding boxes, subject tracking, and per limb prosthesis status collected with expert guidance. Experiments show that state of the art human pose estimation models frequently predict anatomically impossible joints in the presence of missing or prosthetic limbs, motivating a new evaluation metric called Limb specific Confidence Consistency (LiCC) to measure confidence calibration under anatomical absence. InclusiveVidPose provides a challenging benchmark intended to drive research toward inclusive and robust pose estimation for rehabilitation, health assessment, and assistive technologies.

### Strengths
- The paper presents a dataset of approximately 327,000 video frames featuring various types of limb deficiencies. It also introduces a new metric, called Limb-specific Confidence Consistency (LiCC), designed to help evaluate whether pose-estimation models can tell the difference between intact limbs and those that are residual or absent.
- As shown in Appendix D, the AP scores on residual endpoints are lower than on standard joints. Moreover, while adding COCO during training enhances performance on standard joints, it deteriorates performance on residual limbs. This highlights a research gap in how to effectively combine the two without harming performance on either group, suggesting a valuable direction for future work.
- The proposed dataset is in a video modality, which is crucial for individuals with amputations because per-frame information can help disambiguate occluded limbs from truly absent ones. Current 2D pose baselines rely on image-based datasets and often perform poorly on metrics such as LiCC. This suggests that models incorporating video modality are needed to reduce such ambiguity.
- A comprehensive evaluation of the dataset is conducted using recent 2D pose estimators, including YOLOX-Pose, DEKR, ViPNAS, Swin Transformer, RTMPose, and ViTPose, to analyze their behavior on the proposed dataset using COCO metrics in addition to the newly introduced LiCC metric. Furthermore, parallel experiments are performed to examine how incorporating COCO into training affects performance on both the COCO and InclusiveVidPose datasets.

### Weaknesses
- Although the paper highlights the ambiguity between occluded and absent limbs when annotating data, where annotators use temporal cues to disambiguate, the baseline comparisons are all image based. It would have been beneficial if a simple video baseline had been introduced to demonstrate how temporal cues can help the model resolve this ambiguity and how future work could further improve video based methods.

### Questions
- At line 309, the “annotation team training” field is empty. Is there any missing information?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents InclusiveVidPose, the first large-scale video-based human pose estimation (HPE) dataset designed specifically for individuals with limb deficiencies. The dataset contains 313 videos (≈327k frames) featuring 398 participants with various amputation and congenital limb-difference types. It introduces eight new “residual-limb end” keypoints extending the COCO 17-keypoint schema to better model missing or prosthetic limbs. Each frame is annotated with segmentation masks, bounding boxes, tracking IDs, and per-limb prosthesis status, verified under expert para-athletic supervision. The paper also proposes a new evaluation metric, Limb-specific Confidence Consistency (LiCC), which measures whether a model maintains coherent confidence calibration between existing and absent limbs. Extensive benchmarks with 12 existing pose models (e.g., ViTPose, Swin, HRNet, YOLOPose) reveal that current systems generalize poorly to limb-difference anatomies, often predicting anatomically implausible joints or over-confident scores on missing limbs.

### Strengths
1) The dataset is well-curated and technically rigorous: (i) Video-based rather than static image design allows temporal disambiguation between occluded and absent limbs. (ii) Multi-modal annotation (pose, mask, prosthesis status, tracking) ensures reusability across research domains (pose, segmentation, tracking, rehabilitation). It will be a great contribution to the community. 

2) LiCC is a conceptually sound and practically useful metric to quantify anatomical plausibility and confidence consistency — an important step toward calibrated pose estimation models. It provides a tangible way to evaluate inclusiveness beyond accuracy. 

3) In general, the paper addresses a genuine fairness gap in pose estimation research, highlighting the underrepresentation of hundreds of millions of people with limb differences.

### Weaknesses
1) While the dataset and metric are valuable and they are good contributions, the methodological side (pose estimation models) remains entirely benchmark-based. The paper would benefit from proposing a baseline model explicitly optimized for inclusive anatomy (e.g., deformable skeleton graph, conditional keypoint masking). 

2) Despite the dataset being video-based, all experiments are single-frame. The paper defers temporal modeling to “future work,” missing an opportunity to demonstrate how temporal priors improve residual-limb estimation.

3) Given the high variability in anatomy and camera conditions, annotation uncertainty modeling (e.g., probabilistic keypoints or confidence intervals) could be beneficial. The paper treats annotations as deterministic. That is potentially a good future work worth exploring further.

### Questions
1) Since the dataset is video-based, have the authors tested temporal models like PoseTrack transformers or 3D CNNs? How does motion continuity affect missing-limb recognition?

2) What happens if models are pre-trained on COCO and fine-tuned on InclusiveVidPose? Is catastrophic forgetting an issue for intact-body keypoints? Just curious about the outcome. 

3) Just wondering- could segmentation masks be used to guide pose estimation, e.g., via conditional keypoint visibility modeling?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces InclusiveVidPose, a large-scale video-based human pose estimation dataset focused on individuals with limb deficiencies. The paper argues that existing models are trained only on able-bodied individuals with full keypoints, which results in failing to generalize to this population. To address this, the authors have annotated a large amount of data with bounding boxes, segmentation masks, and keypoints. Additionally, the paper introduces an extended 25-keypoint schema for people with limb deficiencies, building on the 17 COCO standard. Lastly, the paper proposes LiCC, a percentage-based metric to evaluate how confident a model predicts limb deficiencies between mutually exclusive joints'. The paper presents a comprehensive set of experiments on recent HPE models and highlights the research gap this dataset addresses.

### Strengths
1. The problem is very well-motivated, and the paper addresses a clear gap in research. The paper presents all the tools necessary for future research in this domain, including a high-quality dataset, annotations, evaluation tools, and baselines.
2. The dataset statistics show a well-balanced distribution of the population.
3. The experimental evaluations are very comprehensive. The low LiCC score also supports the paper's claim about the necessity of this research.
4. The paper is very well-written and uses appropriate words for its arguments and to report its findings.

### Weaknesses
1. The results show that by training on the introduced dataset, the performance of some of the models drops. The paper notes that this is due to the size of the models, larger output space, and their architectural optimization for the COCO dataset. However, this can also occur due to annotation inconsistencies across datasets. However, the paper lacks any quantitative evaluation of the annotation consistency. While the qualitative reports and supplementary materials include high-quality labels, additional consistency analysis would benefit the paper.
2. There's a typo on line 515, where "sp lits" should be "splits"
3. The lack of annotation for limb prosthesis end-joint coordinates is a minor weakness in this paper, as many HPE applications rely on how humans interact with their surroundings (e.g., walking, grabbing objects). Marking these end nodes as invisible in the annotations could hinder their performance.
4. While the dataset labels the per-limb prosthesis status, it does not use it in the models. Still, this is not a significant weakness, and future work can build on it.

### Questions
1. As I mentioned above, some analysis of consistency/annotation quality seems to be necessary. Table 2 already provides an exhaustive overview of the results. However, adding a "COCO -> InclusiveVidPose" column and comparing it with "COCO -> COCO" can reveal the distribution shift between the two datasets. This analysis can also be broken down to limb deficiency types for more insight.

### Soundness
4

### Presentation
4

### Contribution
4
