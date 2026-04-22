# MeMoSORT: Memory-Assisted Filtering and Motion-Adaptive Association Metric for Multi-Person Tracking

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Multi-object tracking (MOT) in human-dominant scenarios, which involves continuously tracking multiple people within video sequences, remains a significant challenge in computer vision due to targets' complex motion and severe occlusions. Conventional tracking-by-detection methods are fundamentally limited by their reliance on Kalman filter (KF) and rigid Intersection over Union (IoU)-based association. The motion model in KF often mismatches real-world object dynamics, causing filtering errors, while rigid association struggles under occlusions, leading to identity switches or target loss. To address these issues, we propose MeMoSORT, a simple, online, and real-time MOT algorithm with two key innovations. At first, the Memory-assisted Kalman filter (MeKF) uses memory-augmented neural networks to compensate for mismatches between assumed and actual object motion. Secondly, the Motion-adaptive IoU (Mo-IoU) adaptively expands the matching region and incorporates height similarity to reduce mis-associations, while remaining lightweight. Experiments show that MeMoSORT achieves state-of-the-art performance, with HOTA scores of 67.9\% and 82.1\% on DanceTrack and SportsMOT, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces MeMoSORT, a simple, online, and real-time Multi-Object Tracking (MOT) algorithm designed specifically for human-dominant scenarios characterized by complex motion and severe occlusions. The method addresses two fundamental limitations of conventional tracking-by-detection (TBD) methods: the motion mismatch inherent in the Kalman Filter (KF) and the rigidity of standard Intersection over Union (IoU) association. MeMoSORT proposes two primary innovations: the Memory-assisted Kalman Filter (MeKF), which integrates memory-augmented neural networks to explicitly compensate for the discrepancy between assumed linear dynamics and actual non-Markovian object motion and the Motion-adaptive IoU (Mo-IoU), a spatial association metric that adaptively expands the matching region and incorporates height similarity based on the target’s normalized speeds.

### Strengths
- MeMoSORT achieves SOTA performance across multiple metrics, notably HOTA, on DanceTrack and SportsMOT.
- Mo-IoU provides a significant advantage in handling severe occlusions. It jointly controls the expansion scale (EIoU) and height weighting (HIoU) using the Motion-Adaptive Technique (MAT), which uses normalized speeds to adapt parameters discretely. This adaptive parameter selection ensures more robust and accurate tracking than existing fixed-parameter IoU variants.
- The overall presentation of the paper is clear and easy to follow

### Weaknesses
- The entire motivation for the Memory-assisted Kalman Filter (MeKF) rests on overcoming the "fundamental limitations of the linear, first-order Markovian motion model". The authors specifically implement the standard KF using a constant velocity model for the state transition matrix F. While the paper visually demonstrates that complex human movements (e.g., phased switching, predictable back-and-forth patterns) violate the Markovian assumption, the justification for choosing the most simplistic linear prior as the failure point is weak.
- The Motion-Adaptive Technique (MAT) employs a discrete piecewise design to set Mo-IoU's parameters ($p_t, q_t$) for efficiency, switching based on predefined, fixed velocity thresholds ($\Theta_{\text{center}}$, $\Theta_{\text{height}}$). Relying on this abrupt, binary switching rather than continuous tuning may introduce instability or non-smooth changes in the association cost function when normalized speeds fluctuate near these critical boundary thresholds, potentially destabilizing tracking in the fast, variable motion characteristic of DanceTrack and SportsMOT.

### Questions
1. The MeKF training relies on a specific dataset generation process using matched YOLOX detections and ground truth. Could the authors provide a sensitivity analysis or further discussion on how the performance of MeKF might change if a lower-performing or different detector were used to generate the training dataset?
2. The MeMoSORT framework is specifically tailored for human-dominant scenarios characterized by complex, non-Markovian motion and severe occlusions (DanceTrack, SportsMOT). While the cross-dataset evaluation between DanceTrack and SportsMOT showed only a "slight degradation" in performance, validating the learning of transferable motion patterns, have the authors tested MeMoSORT's generalization capability on more conventional Multi-Object Tracking (MOT) benchmarks, such as **MOT17** or **MOT20**? 
3. The MeKF is a hybrid filter designed to retain the stability of the classic Bayesian structure by explicitly compensating the physics-based prior (using $\Delta\hat{F}_t$, $\Delta\hat{H}_t$) rather than fully replacing it. The authors claim this approach "robustly ensures the stability of the state estimation" and provides a "failsafe" compared to purely data-driven methods (like DiffMOT or Mamba-based trackers). Can the authors elaborate on the theoretical or empirical mechanisms that prevent potential instability when the NN components, such as the covariance compensation terms $P_t^F$ and $P_t^H$, are trained and integrated into the Gaussian approximation framework, particularly when compared to other hybrid methods that discard the physics prior entirely?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MeMoSORT, a simple, online　MOT algorithm. First, the Memory-Assisted Kalman Filter (MeKF) uses LSTMs to correct discrepancies between the expected and actual object motion. Motion-adaptive IoU (Mo-IoU) reduces mismatches through an ad-hoc process that adaptively expands matching regions and incorporates height similarity. Several experiments demonstrate its effectiveness.

### Strengths
- This paper is easy to understand.
- The proposed method is easy to understand as it combines elementary techniques.

### Weaknesses
- The design of the proposed method is ad hoc
  - The design of the Expansion IoU and Height IoU techniques is ad hoc.
  - Furthermore, there is no hyperparameter study for the relevant parameters (M, N).

- Insufficient experiments
  - Generally, in tracking methods, comprehensive comparisons using multiple metrics like IDF1 and MOTA are common.
    - Specifically, metrics include IDF1, IDP, IDR, Recall, Precision, FP, FN, IDs, FM, MOTA, IDt, IDa, IDm, etc. This paper evaluates only a very limited subset of these metrics.
  - Furthermore, it does not compare with transformer-based methods like MOTR, MOTRv2, or their variants. It remains unclear whether the proposed method outperforms approaches like MOTRv2, especially on datasets such as DanceTrack.
  - Furthermore, the proposed method has not been compared against more general methods like MOT20 or MOT17. 
  - Consequently, it is difficult to judge the effectiveness of the proposed method.

- Processing Speed
  - A strength of Detection by Track is speed. Does the proposed method have an advantage in computational speed compared to representative methods like Bytetrack?

- Concerns Regarding Domain Shift and Sparse Data
  - Since machine learning is used, performance degradation due to domain gaps between training and test data is a concern. This is thought to occur not only due to differences between datasets like dance track and sport mot, but also due to factors like differences in frame rate. Also, when training data is scarce, does the proposed method become inferior compared to the comparison methods?

### Questions
- Why wasn't it compared against more common methods like MOTA or IDF1?
- Why weren't more sophisticated transformer-based methods like MOTR or MOTRv2 used for comparison?
- Why weren't more common methods like MOT20 or MOT17 used for comparison?
- Does it inherit the weaknesses of machine learning compared to the comparison methods?
- Regarding computational speed, can it claim an advantage over Bytetrack?

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
5

### Summary
This paper introduces MeMoSORT, a method designed to address the problem of multiple-object tracking (MOT). It presents two main innovations: (a) a Memory-assisted Kalman Filter (MeKF), which employs a memory-augmented neural network (LSTM-based) to bridge the gap between assumed and actual motion patterns; and (b) a Motion-adaptive IoU (MoIoU), which dynamically expands the matching region and integrates height similarity to reduce association errors.

### Strengths
This paper introduces MeMoSORT, a method designed to address the problem of multiple-object tracking (MOT). It presents two main innovations: (a) a Memory-assisted Kalman Filter (MeKF), which employs a memory-augmented neural network (LSTM-based) to bridge the gap between assumed and actual motion patterns; and (b) a Motion-adaptive IoU (MoIoU), which dynamically expands the matching region and integrates height similarity to reduce association errors.  It shows SoTA results in  the  DanceTrack dataset.

### Weaknesses
There are several concerns regarding both novelty and practicality.  First, the MeKF component is computationally expensive and difficult to train due to its reliance on LSTMs. Consequently, the paper omits evaluation on more challenging datasets such as MOT20, which significantly limits the strength of the experimental validation. Second, using height as a discriminative feature for association is questionable. Estimating reliable person height from uncalibrated cameras is inherently difficult, and individuals with similar heights cannot be easily distinguished. Therefore, this feature does not effectively address occlusion or identity confusion. Overall, the contribution is limited and primarily experimental. I therefore recommend rejection of this paper in its current form.

### Questions
Major Concerns: 
1. Frame Rate Comparison: FPS is compared with other state-of-the-art (SoTA) methods, but the results should also include performance on edge devices to demonstrate real-time feasibility.
2.  Outdated Baselines: Many compared SoTA methods are outdated. Recent trackers such as SMILETrack (AAAI 2024) and methods (MeMoTTR and MOTIP ) reported in CVPR 2023–2025 should be included for a fair comparison.
3. Efficiency Drop with ReID: As shown in Table 3, adding the ReID module significantly decreases efficiency, indicating scalability issues.
4.  Computation Overhead of MeKF: The MeKF component notably reduces performance from 74.5 FPS to 60.8 FPS, highlighting its inefficiency.
5.  Lack of Real-time Capability: The overall system is not real-time, especially when deployed on edge devices such as Jetson Nano or NX.
6.  Incomplete FPS Reporting: In Table 2, FPS results are missing for several compared methods. These should be provided for a complete and fair analysis or reported in another table.
7.  Missing Benchmark Evaluations: The paper does not evaluate on MOT17 or MOT20, which are standard benchmarks for MOT. This omission weakens the validity of the results and comparability with existing work.

### Soundness
2

### Presentation
2

### Contribution
2
