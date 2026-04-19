# PEPNet: A Lightweight Point-based Event Camera 6-DOFs Pose Relocalization Network

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3

## Abstract
Event cameras exhibit remarkable attributes such as high dynamic range, asynchronicity, and low latency, making them highly suitable for vision tasks that involve high-speed motion in challenging lighting conditions. These cameras inherently capture movement and depth information in events, making them appealing sensors for Camera Pose Relocalization (CPR) tasks. Nevertheless, existing CPR networks based on events neglect the pivotal fine-grained temporal information in events, resulting in unsatisfactory performance. Moreover, the energy-efficient features are further compromised by the use of excessively complex models, hindering efficient deployment on edge devices. In this paper, we introduce PEPNet, a lightweight point-based network designed to regress six degrees of freedom (6-DOFs) event camera poses. We rethink the relationship between the event camera and CPR tasks, leveraging the raw point cloud directly as network input to harness the high-temporal resolution and inherent sparsity of events. PEPNet is adept at abstracting the spatial and implicit temporal features through hierarchical structure and explicit temporal features by Attentive Bi-directional Long Short-Term Memory (A-Bi-LSTM). By employing a carefully crafted lightweight design, PEPNet delivers state-of-the-art (SOTA) performance on public datasets with meager computational resources. Specifically, PEPNet attains a significant 38\% performance improvement on the random split DAVIS 240C CPR Dataset, utilizing merely 6\% of the parameters compared to traditional frame-based approaches. Moreover, PEPNet$_{tiny}$ accomplishes results comparable to the SOTA while employing a mere 0.5\% of the parameters.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a lightweight point-based end-to-end network (PEPNet) for camera relocalization from event cameras.  The network extracts spatial and implicit temporal features through a hierarchical structure and then explicitly attains temporal information via an attentive bi-directional LSTM (A-Bi-LSTM).  Experimental results on the DAVIS 240C CPR dataset show its good performance as well as the effacacy.

### Strengths
1) The way to treat the time dimension for event cameras in a point-based network is insightful.

2) The resultant network is lightweight yet performs well on the benchmarking dataset.

### Weaknesses
-- The symbol for concatenation operation is ambiguous. The same symbol is also used in Fig. 3 (Extractor), but it seems it does not mean concatenation there.

-- Fig. 4 can be replaced with qualitative comparison to the baselines rather than just some existing data samples from the dataset.

-- Some citations for the baselines are missing in Section 4.2.

### Questions
1) Section 3.2.2: What are the dimensions of PG and PS? How is the substraction done? Could the authors elaborate this in detail?

2) Section 3.2.3: Does f correspond with the ReLu operator in Fig. 3? It seems there is no ReLu before the summation operator in Fig. 3, but there is an f in Eq. (9). Why are they different?

3) Are the random data train/test splits the same as the baselines? "We randomly select" sounds like the authors use a different random split although following the same strategy. I wonder whether the comparison on this setting is really fair or not.

4) How is PEPNet_{tiny} obtained exactly?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the regression task of six degrees of freedom event camera poses. Considering that the existing methods neglect fine-grained temporal information in events, the proposed PEPNet directly deal with the raw point cloud to use the high-temporal resolution and inherent sparsity of events.

### Strengths
The proposed PEPNet outperforms other methods in both performance and running speed.

### Weaknesses
Weakness:

1.	The event camera data in the method is actually more like video data (not 3-D spatial data as point clouds), probably other video-based feature extraction methods could be utilized for the task.

2.	The proposed hierarchy structure, Bi-LSTM, and attention mechanisms are very common modules in the area of CNNs models for processing video data. This reduces the contributions of the method a little. However, the method does achieve better performance than previous methods.

3.	Is there any ablation study on the loss term weight?

4.	Are there intermediate visualization results of the attention mechanisms to show what has been learned in the module?

5.	Some typos: Second line of P6.

### Questions
As above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a lightweight DNN for pose estimation using event-based camera data. The evaluation is done on a DAVIS-240C camera dataset that contains a variety of camera motions and egomotion ground truth.

### Strengths
The paper is clearly written and illustrations support the text well. If the code for this work is released, this would be a notable contribution, as a lightweight method could be readily used in robotic applications.

### Weaknesses
1) The literature overview does not include a few works on event-based pose estimation - there are methods based on 3D pointcloud analysis (https://openaccess.thecvf.com/content_CVPR_2020/papers/Mitrokhin_Learning_Visual_Motion_Segmentation_Using_Event_Surfaces_CVPR_2020_paper.pdf), and self-supervised methods (https://arxiv.org/pdf/1903.07520.pdf) - and more, all of which were evaluated on more complex datasets than in this paper. How would these approaches (to DNN, loss functions, event encoding) compare with this work?

2) CPR Dataset evaluated in the paper contains mostly planar scenes or one-dimensional motion, making it easier for the network to overfit. It is also hard to gauge the full 6dof performance of the method, when using simplistic data. I would recommend evaluating on MVSEC (https://daniilidis-group.github.io/mvsec/) and/or EV-IMO (https://better-flow.github.io/evimo/download_evimo_2.html) - both datasets have pose ground truth.

3) Since the problem of motion estimation is geometric, it would benefit the method if the loss function incorporated some geometry constraints. In classic vision, sota egomotion pipelines leverage this successfully, and there were prior works on event cameras doing the same.

4) On motion estimation problem, with a random split, it is highly likely the network overfitted.

### Questions
1) Minor, in abstract: "These cameras inherently capture movement and depth information in events" - I would argue that event cameras are similar to classic ones in captureing depth. They provide continuous 'tracking', but the depth is not directly measured. At the very least, the advantage is not obvious.

2) In event-based processing it is specifically important to understand how exactly the events are fed into the DNN, and what are the implications of the approach / which other similar methods exist in literature. The Algorithm 1 answers this to a degree, but I am not sure I understand if the temporal window is always fixed, and if the number of the events within this window is subsampled to a constant value. What would happen if there are fewer events than Np (1024) due to the lack of motion?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
