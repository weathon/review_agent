# FastTracker: Real-Time and Accurate Visual Tracking

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Conventional multi-object tracking (MOT) systems are predominantly designed
for pedestrian tracking and often exhibit limited generalization to other object
categories. This paper presents a generalized tracking framework capable of
handling multiple object types, with a particular emphasis on vehicle tracking in
complex traffic scenes. The proposed method incorporates two key components: (i)
an occlusion-aware re-identification mechanism that enhances identity preservation
for heavily occluded objects, and (ii) a road-structure-aware tracklet refinement
strategy that utilizes semantic scene priors—such as lane directions, crosswalks,
and road boundaries—to improve trajectory continuity and accuracy. In addition,
we introduce a new benchmark dataset comprising diverse vehicle classes with
frame-level tracking annotations, specifically curated to support evaluation of
vehicle-focused tracking methods. Extensive experimental results demonstrate that
the proposed approach achieves robust performance on both the newly introduced
dataset and several public benchmarks, highlighting its effectiveness in general-
purpose object tracking. While our framework is designed for generalized multi-
class tracking, it also achieves strong performance on conventional pedestrian
benchmarks, with HOTA scores of 66.4 on MOT17 and 65.7 on MOT20 test sets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes FastTracker, a lightweight, real-time multi-object tracking framework. Its core contributions include: occlusion handling without ReID via DampenVelocity and EnlargeBox，and trajectory regularization using environment priors defined by a direction cone and a quadrilateral ROI. In addition, the authors introduce FastTrack, an internal CCTV multi-class traffic benchmark.

### Strengths
1、Originality: Explicitly encode the scene prior with ProjectToCone to correct the prediction before association. Use center proximity for occlusion detection and stabilize trajectories with DampenVelocity and EnlargeBox.
2、Quality: The method is well designed, the modules are interrelated, and the experimental pipeline is clear, with ablation and comparative experiments provided to demonstrate the framework’s performance.
3、Clarity: The flowcharts, visual examples, and algorithmic steps are easy to follow, and the experimental results are comprehensive.
4、Significance: By modularizing the various components, the paper provides a reusable, comparable baseline for subsequent Kalman-based trackers, and provides a dataset for complex scenarios that meaningfully advances engineering evaluation in MOT.

### Weaknesses
1、The association backbone follows ByteTrack’s two-stage matching scheme; the method’s novelty lies primarily in the pre-association constraints and occlusion heuristics rather than in the association paradigm itself.
2、Dependence on priors: The ROI/direction constraints must be predefined manually, and the ROI is restricted to a quadrilateral. This strong reliance on scene priors limits deployment and generalization.
3、Implementation details: Although the paper states that a Kalman filter is used, it does not fully specify the motion model, including the definition of the state vector and the settings for process noise and measurement noise.

### Questions
1、Will the FastTrack dataset be made publicly available? Please describe the privacy handling, data collection, and the compliance policies and procedures for external release.
2、How robust is the method when scene priors are missing or mis-registered? Does performance degrade substantially in such cases?
3、Please provide the complete mathematical specification of the NSA Kalman filter and the default parameter settings for categories such as pedestrians and bicycles.

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
FastTracker proposes a lightweight, general multi-object tracking (MOT) framework for vehicle-rich traffic scenes. Building on ByteTrack’s two-stage association, it replaces deep re-identification with motion and spatial cues. Two key modules are introduced: (1) an occlusion-aware re-ID using geometric coverage and velocity dampening for ID continuity, and (2) a road-structure-aware refinement leveraging manually defined ROIs and lane-direction constraints to enforce plausible trajectories. The authors also release the FastTrack dataset (9 classes, 12 scenes, 800K annotations) for vehicle-centric tracking. It achieves state-of-the-art results on MOT17 (HOTA 66.4), MOT20 (65.7), DanceTrack, and the new dataset, with notably fewer ID switches. Ablation studies show consistent gains from each module.

### Strengths
- Comprehensive Experiments: The authors evaluate on a wide range of benchmarks (MOT16/17/20, DanceTrack, and FastTrack) with detailed ablation studies (Tables 2–5) verifying each module’s impact. The improvements in key metrics (MOTA, HOTA, IDF1) and especially the reduction in ID switches (e.g. lowest IDs on MOT17/20) are convincingly shown.

- Strong Empirical Results: FastTracker achieves state-of-the-art or competitive performance on multiple datasets. For instance, it reaches HOTA 66.4 on MOT17 and 65.7 on MOT20 (test sets), and outperforms all baselines on DanceTrack (MOTA 93.4, HOTA 65.9). These results indicate a high practical impact.

- New Benchmark Dataset: The introduction of the FastTrack dataset (800K frames, 9 classes, diverse traffic scenes) fills a notable gap. The paper provides dataset statistics and sample frames, and demonstrates that existing trackers perform significantly worse on it, highlighting its challenge. Making this dataset available would be a great resource for the community.

### Weaknesses
- Manual ROI/Direction Constraints: A key limitation is the reliance on manually defined polygonal ROIs and fixed cone directions for scene priors. As acknowledged by the authors, this is labor-intensive and may not generalize to complex or evolving environments (e.g. intersections, roundabouts). The current system only supports quadrilateral regions, limiting flexibility. This reliance diminishes the novelty and practicality of the road-structure module.

- Lack of Runtime Analysis: The claim of real-time operation is not quantitatively supported. The paper reports performance with lighter detectors but does not give actual frame rates or hardware details. It is unclear whether the occlusion and ROI modules introduce any latency, or how the system performs on typical edge devices.

- Many parts of the system (velocity dampening factor, coverage thresholds, direction-angle limits) are hand-designed. While ablations show they work, it is not clear how sensitive the system is to these hyperparameters. In some cases, learned re-ID features might be more robust to varied conditions than the proposed heuristics.

### Questions
- ROI/Direction Robustness: How sensitive is FastTracker to inaccuracies in the manually defined ROI or direction cones? If the annotated road boundaries are slightly off, does performance degrade significantly?

- Automatic Priors: Do the authors plan to incorporate automatic scene understanding (e.g. segmentation of roads/lanes) to generate the ROI and direction constraints? Have any preliminary tests been done in this direction?

- Runtime Performance: Can the authors provide empirical runtime measurements (e.g. frames per second) for FastTracker on a typical hardware setup, both with the heavy (YOLOX-L) and lightweight detectors (YOLOX-Nano)?

- Failure Cases: Are there particular scenarios where FastTracker fails (e.g. very long occlusions, heavy clutter)? Can you provide qualitative examples of its limitations?

- Dataset Release: Will the FastTrack benchmark and annotations be made publicly available, and under what license?

### Soundness
3

### Presentation
2

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
The paper proposes FastTracker, a lightweight, motion-centric online MOT framework designed for multi-class tracking in complex urban scenes, with particular emphasis on vehicles. The method builds upon a two-stage association strategy (high- then low-confidence detections) and augments it with: (i) an occlusion-aware module that stabilizes track states without CNN-based ReID by damping velocity and enlarging boxes during occlusions, and (ii) environment-aware constraints based on road geometry and directional priors (ProjectToCone, ClampToROI). The authors also introduce a CCTV-based benchmark (FastTrack) with diverse traffic scenarios and claim state-of-the-art results on MOT16/17/20 and DanceTrack while remaining real-time and resource efficient.

### Strengths
1. Clear problem motivation: addresses generalization beyond pedestrian tracking and the need for multi-class vehicle-centric tracking under occlusions and complex layouts.
2. Practical, lightweight design: avoids deep appearance models in the online pipeline; relies on motion, geometry, and simple heuristics that are attractive for real-time deployments.
3. Environment-aware modeling: novel use of region semantics and directional constraints to limit drift and enforce plausible motion without heavy learning modules.

### Weaknesses
1. Clarity and correctness of some definitions:
The “center-proximity score CP” is described as computed via IoU, which is conceptually inconsistent (center-proximity is not IoU). A precise definition is missing.
2. Occlusion handling design details:
Marking occlusion based on overlap with other active tracklets via a single threshold may conflate crowding with occlusion and induce false occlusion states.
3. Dataset details and release:
The FastTrack dataset has only 12 videos (albeit very dense). More details are needed: annotation protocol, quality control, train/val/test splits, licensing, and release plan. Without public release, the dataset’s impact is limited.

### Questions
1. Please precisely define CP(t, t′) and how occlusion is decided, including edge cases in crowded scenes. Is CP IoU, or a center-distance metric?
2. How are occluded tracks re-associated upon reappearance if they are excluded from association? What gating, timing, and matching logic ensure ID continuity?
3. Do you output predicted boxes during occlusion? If yes, how do enlarged boxes affect FP/FN and HOTA?

### Soundness
2

### Presentation
3

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
This work proposes a new method for multi-object tracking. It focuses on vehicle tracking on complex traffic scenes. It proposes a mechanism to handle occlusions by moderating Kalman Filter and enlarging the candidate box. It also introduces a tracklet refinement strategy, which uses scene information to improve trajectory continuity and accuracy. A new traffic benchmark is collected as well. Experiments on public benchmarks and the new one demonstrate the effectiveness of the proposed method.

### Strengths
1. The idea of handing occlusion situations and utilizing scene information makes sense.
2. The implementation of the occlusion handling and the scene prior constraints is reasonable.
3. The collected dataset can be helpful for the community.
4. The proposed method is effective on public benchmarks and the new one.

### Weaknesses
1. Experiments on efficiency are lacking. The paper claims that the tracker is real-time, but related experiments, like running speed, computational burden, etc., are missing. These experiments are needed to support the claim.
2. Methodology contribution is limited. This work focuses on the Kalman Filter-based data association process, and improves the process from multiple aspects. However, these improvements are more like engineering optimization, instead of methodology contribution from my view. It has practical value, but the methodology contribution is not enough.

### Questions
How's the method's performance on BDD100k? BDD100k[1] is a popular MOT dataset with multiple classes on traffic scenes. It should be included for dataset and method comparison.

Ref.

[1] Yu, Fisher, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. "Bdd100k: A diverse driving dataset for heterogeneous multitask learning." CVPR 2020.

### Soundness
3

### Presentation
3

### Contribution
2
