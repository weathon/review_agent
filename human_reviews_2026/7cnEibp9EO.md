# triCAM: A Real Monocular Multi-Modal Event-based Pedestrian Dataset

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Event-based visions offer key advantages, such as low latency, high dynamic range, and microsecond temporal resolution. These strengths have motivated extensive research into their complementarity with other modalities, which led to the creation of several multi-modal event-based datasets. However, most of these datasets are designed for automotive or robotic domains, with limited attention to human-centered perception in everyday settings. In this paper, we introduce triCAM, a real-world monocular multi-modal event-based pedestrian dataset. triCAM integrates event streams, RGB images, depth images, IMU data, and pedestrian bounding box annotations. This dataset contains 20 sequences, each recorded in two different restaurants in both static and dynamic camera motions. By providing a rich dataset on pedestrian activities in socially interactive environments, triCAM contributes to the advancement of research in robust perception and human interaction understanding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces triCAM, an event-based, real-world monocular multimodal pedestrian dataset. The dataset integrates event streams, RGB images, depth images, IMU data, and pedestrian bounding box annotations recorded in indoor and outdoor restaurant environments under both static and dynamic camera motions.

### Strengths
This type of dataset seems uncommon in the event camera field.

### Weaknesses
1. Does PEDESTRIAN only cover people in restaurants? According to the KITTI rankings, PEDESTRIAN should also cover people in open environments like roads.

2. Is this dataset's task just pedestrian detection? I'd prefer to see results on depth estimation, event camera SLAM, or motion prediction. This results in the experimental section being too sparse, with tables and tables spanning less than a page. I'd prefer to see a more comprehensive, multi-task benchmark.

3. The TRICAM dataset is too small. The 20 sequences total only 60 minutes of data, which is far from sufficient for a portable dataset that doesn't require any motion capture. Furthermore, it was captured in just two restaurants, so the environment and lighting diversity are limited compared to large-scale benchmarks.

3. I have serious questions about the calibration results for the RGB and event cameras. In particular, I don't think the reconstructed checkerboard is good in Figure 3. I hope the authors can provide a visualization of the overlap between the event and RGB synchronized frames.

4. I'm very surprised that this dataset collects such sensitive personal data without any privacy protection. This clearly violates ethical review rules.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper introduces triCAM, a real-world monocular multi-modal dataset targeting pedestrian scenes in restaurant environments. triCAM contains synchronized streams from a Prophesee Gen3 event camera, an Intel RealSense D435i (RGB + depth + IMU), and an additional WitMotion IMU. The dataset comprises ~20 sequences recorded in two restaurants under both static and handheld (dynamic) camera motions, and includes calibration parameters and bounding-box annotations for both RGB and event-derived image representations. The authors describe the hardware/software setup, spatial/temporal synchronization pipeline (including E2VID reconstruction of events for calibration/annotation), dataset format (ROS bag + supplementary files), and present a baseline pedestrian detection evaluation using YOLOv8x on RGB-only, Event-only, and late-fusion Event+RGB setups. The manuscript positions triCAM as filling a gap in human-centric, socially interactive event datasets.

### Strengths
**Originality**: Focus on indoor/outdoor restaurant pedestrian scenes (socially interactive, cluttered) is novel among event multi-modal datasets — most prior work targets automotive or robotics navigation. 

**Quality**: Use of a modern Gen3 event camera plus RealSense D435i and separate IMU provides complementary modalities; distribution in ROS bag format and inclusion of intrinsic/extrinsic calibration parameters increases usability.

**Clarity**: Paper is structured clearly with useful tables/figures (sensor specs, sequence statistics, baseline results) that help readers understand dataset composition.

**Significance**: Enables research directions (multi-modal pedestrian detection, event-guided pose/depth estimation, perception in socially interactive settings) that are underexplored in event vision.

### Weaknesses
1. Figure/table layout and information density are insufficient. The paper’s sequence overviews and calibration/annotation examples (Figures 2, 3, 4) present a formal view but lack more details, so readers cannot reliably assess annotation or reconstruction quality.
2. Dataset statistics and sample-level examples are incomplete. Although Table 3 lists per-sequence duration, number of people, and total events, the manuscript misses key sample-level statistics (e.g., per-frame or per-sequence bounding-box count distributions, frame counts per distance bin, quantitative occlusion tiers) and lacks representative “difficult sample” parallel visualizations that would let readers judge diversity and difficulty.
3. Lacks clear problem definition and theoretical explanation. While the introduction and contribution sections state triCAM’s purpose and novelty, the manuscript does not formalize the tasks or evaluation objectives (for example, concrete definitions of the “multimodal fusion” problem being solved and the evaluation criteria).
4. Experimental reporting is insufficient (narrow baselines, missing ablations and per-condition breakdowns). The evaluation uses YOLOv8x variants (Event-only, RGB-only) and a simple late-fusion, reporting only overall mAP50/Precision/Recall (Table 4). There are no per-condition results (static vs. dynamic, occlusion levels, distance bands), nor ablations on key hyperparameters (e.g., the 33.33 ms event aggregation window).

### Questions
1. Improve the layout of Figures 2–5 to make the information more readable (e.g., clearer captions)
2. Consider presenting diversity metrics, for example: variation in lighting, motion, and occlusion; number of unique subjects; and environmental diversity (indoor/outdoor).
3. Formally define the tasks that triCAM aims to support (e.g., “multimodal monocular detection,” “event–RGB fusion tracking,” etc.)
4. Add ablation studies on key parameters such as the event aggregation window (33.33 ms).
5. Provide per-condition analyses, such as static vs. dynamic scenes, different occlusion levels, lighting conditions, and distance ranges.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents triCAM, a monocular multi-modal pedestrian dataset collected at two restaurant venues, combining event streams, RGB, depth, IMU, pedestrian boxes, and calibration. Sequences include static and dynamic camera motions. Events are binned at 33 ms to align with ~30 FPS depth frames, and the spatial/temporal synchronization pipeline is described in detail. The authors report YOLOv8x baselines for Event-only, RGB-only, and late fusion (NMS), with fusion outperforming single modalities in mAP metric. Despite the clear sync/calibration procedures and usable baselines, the paper has core issues: the motivation for the restaurant setting is weak, scale/diversity are limited, and there is no condition-wise analysis showing where events help (e.g., low light, fast motion, occlusion). The baselines and evaluation are insufficient (limited metrics/fusion strategies, no cross-scene generalization/transfer), and the overall presentation reads more like a careful engineering report than a dataset paper with compelling novelty. Therefore, I recommend rejection.

### Strengths
1. **Comprehensive multimodal setup with clear sync/calibration.** The platform jointly captures events, RGB, depth, and IMU, enabling research on cross-modal fusion and alignment. The paper details event-to-frame temporal alignment and spatial calibration procedures that are reasonably reproducible, lowering the barrier to use.

2. **Practical data organization.** The release includes pedestrian bounding boxes and starter training/evaluation scripts (e.g., YOLO), making it friendly for baseline reproduction and quick experimentation.

3. **Empirical indication of modality complementarity.** Although the analysis is not yet thorough, initial results show Event + RGB fusion outperforming single modalities, suggesting genuine potential for multimodal gains.

### Weaknesses
1. **Motivation for restaurant events is under-argued.** The paper cites event cameras’ advantages for high-speed and robust perception, but the collected restaurant scenes are not clearly high-speed. With only two venues, it is hard to claim representativeness for “socially interactive environments.” The paper does not concretely show which sub-conditions (e.g., strong backlight, low light, rapid motion, heavy occlusion, hand-held shake) in the dataset require events beyond what RGB can handle. 

2. **Scale and diversity are limited.** The dataset contains ~20 sequences across two restaurants, while per-sequence stats are given, there is no explicit partitioning by motion speed or illumination, and no targeted “event-advantage” splits (e.g., low light, motion blur, fast actions).  Table 4 reports improvements on the entire test set only. Condition-wise statistics (e.g., low light, fast motion, dynamic camera, high occlusion) are necessary to determine where events are most beneficial, rather than relying on a single overall number. 

3. **Label scope is narrow compared to stated goals.** Only bounding boxes for pedestrians are provided, tasks central to human-centered interaction (segmentation, pose/landmarks, ReID, trajectories/MOT) are absent, limiting the dataset’s relevance to the broader HRI/behavior understanding vision it outlines. 

4. **Baselines and evaluation are thin.** Metrics stop at mAP@50, COCO mAP@[.5:.95] is not reported. Fusion is late-fusion NMS only, no early/mid-level or joint training fusion. No cross-scene generalization, and no evidence that models trained on triCAM can generalize better to other datasets/scenes. 

5. **“Monocular” emphasis lacks compelling evidence.** The paper claims to be the first publicly available monocular multi-modal pedestrian dataset and contrasts with stereo-heavy rigs in related work, but it does not quantify the practical advantages of monocular over (i) using a single lens from a stereo rig or (ii) small-baseline stereo under the same setup (e.g., differences in calibration/sync complexity, cost, power, drift, failure cases).

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a monocular multi-modal pedestrian dataset, triCAM. The dataset is captured with RGB, depth, event, and IMU sensors in two different restaurants. The images are further manually annotated by humans. In the experimental results, the mAP
performance using YOLOv8 is provided. The hardware, software setup, and calibration process are also introduced.

### Strengths
- The collected triCAM provides a new multi-modality benchmark to the pedestrian detection community.
- The collecting and post-processing pipeline is introduced clearly.

### Weaknesses
- The major concern is about privacy protection. As a pedestrian dataset, it is not mentioned whether permission was obtained from the subjects, or whether the dataset will be made public, or under what license.
- The dataset used manual annotations. But the annotation compensation was not mentioned, so I also added a corresponding flag for the ethics review.
- This dataset is limited to two specific restaurant scenarios. Therefore, its general applicability is limited. The existing datasets mentioned in Table 1 cover more general situations than this dataset.
- The paper only reported the performance on a specific object detector, which fails to reflect the challenge of this new dataset and the necessity of releasing such a dataset.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
