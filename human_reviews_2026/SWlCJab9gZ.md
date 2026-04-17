# Detecting Temporal Misalignment Attacks in Multimodal Fusion for Autonomous Driving

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Multimodal fusion (MMF) is crucial for autonomous driving perception, combining camera and LiDAR streams for reliable scene understanding. However, its reliance on precise temporal synchronization introduces a vulnerability: adversaries can exploit network-induced delays to subtly misalign sensor streams, degrading MMF performance. To address this, we propose AION, a lightweight, plug-in defense tailored for the autonomous driving scenario. AION integrates continuity-aware contrastive learning to learn smooth multimodal representations and a DTW-based detection mechanism to trace temporal alignment paths and generate misalignment scores. 
AION demonstrates strong and consistent robustness against a wide range of temporal misalignment attacks on KITTI and nuScenes, achieving high average AUROC for camera-only (0.9493) and LiDAR-only (0.9495) attacks, while sustaining robust performance under joint cross-modal attacks (0.9195 on most attacks) with low false-positive rates across fusion backbones. Code is available at: https://github.com/shahriar0651/AION.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper targets temporal misalignment attacks (TMA) in multimodal fusion for autonomous driving, where an attacker perturbs timestamps to make camera/LiDAR frames desync without touching sensor content. It proposes AION, a lightweight, plug-in “detection patch” that:
(1) learns a shared multimodal representation via continuity-aware contrastive learning (CACL) that down-weights negatives that are temporally near using a tanh(|i−j|/τ) weighting, and
(2) runs Dynamic Time Warping (DTW) over a rolling window of cross-modal similarity matrices to score alignment; deviations from a diagonal path lower the DTW “reward,” signaling misalignment. Architecture overview in Fig. 1 (p.4); attack types in Table 1 (p.6); similarity matrices and anomaly curves in Figs. 2–3 (p.8); ROC curves in Fig. 4 (p.9); DTW algorithm in Algorithm 1 (p.12). Reported AUROC on KITTI and nuScenes is 0.92–0.98 across several delay scenarios with small model overhead (~1.97M params, window w=5).

### Strengths
It offers a few contributions:
1. Timestamp-level, network-induced TMA is realistic and distinct from content spoofing; the threat model is clearly articulated (Sec. 2). 

2. AION is architecturally simple (shared encoder + DTW) and plausibly deployable as a non-intrusive patch; runtime cost is argued to be low (Sec. 5, “Scalability”). 

3. CACL’s graded negative weighting (λij=tanh(|i−j|/τ)) is intuitive for video-like sequences and is nicely visualized in Fig. 5 (p.12).

### Weaknesses
I would like to hear from the authors to address the following issues:
1. Attacks are synthetic (constant/random delays) and limited; no drift, burst, schedule-dependent, or closed-loop attacks; no real middleware latency manipulation or ROS2-in-the-loop study. (Table 1 + Sec. 4.3.)

2. Results are ROC/AUROC-only; no calibration/threshold selection, no reporting at practical FPRs (e.g., 0.1%–1%), and no end-to-end latency or throughput measurements in a real AD stack. (Sec. 5 focuses on AUROC.)

3. Sec. 3.3.2 informally treats Sij≈1 iff i=j to justify reward drops; in practice, cross-modal similarity is noisy and scene-dependent, so this theoretical justification is fragile without empirical sensitivity checks to motion, density, lighting, etc.

4. It’s unclear if the MRE is trained per dataset/backbone and how it transfers (train on KITTI, test on nuScenes?) or across different fusion designs; “task-agnostic plug-in” claim would be stronger with cross-architecture/domain transfer results. (Sec. 4.2 describes per-dataset setups.)

### Questions
1. Can you include discussion of difference of the work to "Fusion is not enough: Single modal attacks on fusion models for 3D object detection" ICLR.

2. Why not compare to (i) timestamp-sanity models with probabilistic jitter, (ii) cross-modal tracking consistency checks (adapted to timing), (iii) simple sliding-window correlation or CUSUM-style detectors?

3. Can a single MRE trained on one dataset/backbone generalize to another without retraining? Any train-on-KITTI, test-on-nuScenes results?

4. What FPR/TPR do you get at thresholds suitable for deployment (e.g., FPR ≤ 0.5%)? How are thresholds set (per-scene, per-vehicle, global)?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a method called AION to detect temporal misalignment attacks in multimodal fusion for autonomous driving. The main idea is to use continuity-aware contrastive learning to learn smooth multimodal representations and a Dynamic Time Warping (DTW)-based detection mechanism to trace temporal alignment paths and generate misalignment scores. The proposed method is evaluated on the KITTI and nuScenes datasets, achieving high AUROC values with low false-positive rates across fusion backbones. However, the realism of the threat model for tampering reported time stamps is questionable, and the actual handling of timestamp inconsistencies in real-world scenarios is not addressed. Additionally, the practicality of the proposed method is limited due to the lack of experiments on real vehicles.

### Strengths
1. The problem statement is clear.
2. The write-up is easy to follow.

### Weaknesses
1. The realism of the threat model for tampering with reported time stamps is questionable. If the attacker has the right access to the captured data, why is it a challenge for him to further tamper with the raw sensor data?

2. The actual handling of timestamp inconsistencies in real-world scenarios is not addressed, and it is unclear how the proposed method would perform in such situations.

3. The practicality of the proposed method is limited due to the lack of experiments on real vehicles, which could reveal potential issues or limitations in real-world deployment.

4. The experimental evidence is not enough; more evaluation (including the computation costs are needed).

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
AION is a lightweight, plug-in defense that detects temporal misalignment attacks (TMA) in camera–LiDAR fusion by (i) learning continuity-aware multimodal embeddings (CACL) and (ii) using Dynamic Time Warping (DTW) on a rolling similarity matrix to spot off-diagonal alignment paths indicative of delays/drift/jitter. On KITTI and nuScenes, it reports AUROC 0.92–0.98 across constant and random delay scenarios with low FP rates, using a ~1.97M-param module and small DTW window (w=5) for real-time viability.

### Strengths
Clear threat model & practical attack surface: focuses on timestamp manipulation in ROS2/DDS synchronizers within tolerance windows—no need to alter sensor payloads.

Continuity-aware contrastive learning: grades negatives by temporal distance to capture subtle misalignments, improving sensitivity over rigid positive/negative setups.

General plug-in design: shared multimodal encoder + DTW operates task/backbone-agnostically; demonstrated on KITTI encoders and BEVFusion for nuScenes.

### Weaknesses
Hard case acknowledged: equal constant delay on both modalities (or stationary scenes) can mimic benign diagonals; detecting this requires extra sensors (IMU/CAN) beyond the current scope.

Synthetic attack generation: evaluations inject delays at fixed intervals and windows; robustness to real network behaviors (bursts, variable τ, dropped frames) isn’t deeply profiled here.

Thresholding & ops details light: deployment describes reward-based anomaly scoring but gives limited guidance on threshold calibration and sensitivity to window size w beyond w=5 choice. 

The authors also needs to consider robustness to adaptive attacks.

### Questions
See in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper looks at a very concrete hole in current multimodal AD pipelines: if an attacker only tampers with timestamps (not pixels/points) to force the ROS2/DDS synchronizer to pair camera frame t with LiDAR frame t−k, fusion degrades badly, but most existing defenses don’t look at the temporal axis. The authors propose AION, a plug-in module that (1) learns a shared camera–LiDAR representation with a continuity-aware contrastive loss so temporally close pairs stay close and far pairs stay apart, and then (2) at run time builds a cross-modal similarity matrix over a short window and runs DTW to see whether the optimal path still follows the diagonal. On KITTI and nuScenes, under synthetic constant / random delays, AION gets AUROC around 0.92–0.98 while keeping the module lightweight.

### Strengths
a) Fits AD architecture. The threat model (attacker sits on ROS2 / in-vehicle network and rewrites timestamps) matches how actual AD stacks glue sensors together.

b) Problem is real. Recent works have shown that temporal desync alone can tank fusion AP, but most defenses assume honest timestamps and focus on spoofing / spatial inconsistency / context violations. AION directly targets the temporal gap, which is a good angle. 

c) CACL is a reasonable tweak. Using relaxed/graded negatives based on temporal distance is a sensible extension of ReCo-style contrastive learning to sequential sensor data, not just random pairs.

### Weaknesses
a) Attack model is fairly simple. All attacks are synthetic constant/random delays with a fixed window and fixed injection interval. Real attackers could do pattern-shaped delays, scene-aware delays, or imitate low-speed / stationary AD scenarios the paper itself says are hard to detect; for these, DTW over a short window may not separate benign vs. attack that cleanly. 

b) Stationary / low-dynamics scenes remain hard. The paper admits that if both modalities are delayed by the same amount or the scene stays visually/geometrically similar, the similarity map still looks diagonal, so AION can’t tell benign from malicious. This is exactly the case an attacker in city driving would want to exploit. This limitation should be foregrounded more.

### Questions
a) In deployment, is the threshold global (one value for all scenes / weather / motion) or per-sequence? How sensitive was AUROC to the threshold estimated on “benign” data?
﻿
b) For nuScenes where you aggregate 6 cameras -> BEV -> shared space, do you observe different DTW patterns than in KITTI (single front camera)? In other words, is AION implicitly assuming similar FoV on both modalities?

### Soundness
4

### Presentation
3

### Contribution
3
