# Offline Tracking with Object Permanence

- Decision: Reject
- Scores: 3, 3, 5, 5

## Abstract
To reduce the expensive labor cost for manual labeling autonomous driving datasets, an alternative is to automatically label the datasets using an offline perception system. However, objects might be temporally occluded. Such occlusion scenarios in the datasets are common yet underexplored in offline autolabeling. In this work, we propose an offline tracking model that focuses on occluded object tracks. It leverages the concept of object permanence which means objects continue to exist even if they are not observed anymore. The model contains three parts: a standard online tracker, a re-identification (Re-ID) module that associates tracklets before and after occlusion, and a track completion module that completes the fragmented tracks. The Re-ID module and the track completion module use the vectorized high-definition map (HD map) as one of the inputs to refine the tracking results with occlusion. The model can effectively recover the occluded object trajectories. It achieves state-of-the-art performance in 3D multi-object tracking (MOT) by improving over the original online tracking result by 45% IDS and 2% AMOTA on the vehicle tracks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces an offline tracking method utilizing point clouds, specifically designed to handle challenging heavy occlusions on vehicles. This method comprises three key components: a tracker, a Re-ID module for linking trackless segments before and after occlusions, and a track completion module that interpolates missing tracks caused by occlusions. The study showcases the method's efficacy in tracking objects even under conditions of occlusion.

### Strengths
The paper proposes a method to handle the occluded object tracking.

### Weaknesses
The paper has limited novelty. The tracker, Re-ID, and track completion components all employ established techniques, resulting in a relatively straightforward solution.

### Questions
1. In tracklet association, there appears to be a division between the utilization of map and motion data. However, it's worth considering their complementary contributions. For instance, map information could enhance the accuracy of motion association. Separating these aspects might lead to a loss of crucial information.
2. Appearance features could serve as a pivotal factor in tracklet association. Even when an object becomes temporarily occluded and untrackable, its re-emergence could still benefit from utilizing appearance data for accurate association.
3. Considering the use of map information, it's important to assess its impact on association when an object undergoes occlusion and a simultaneous lane change. This scenario introduces an additional layer of complexity that warrants thorough investigation.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a non-causal MOT technique for labelling large datasets in autonomous driving without human intervention. The technique reconsiders three well-known steps: 1) finding tracklets using causal MOT, 2) associating those tracklets by Re-ID and 3) trajectory completion compensating occlusions. The authors propose novel neural models for each step, including bipartite matching for association. The paper is well structured and readable; the literature research at least shows recent work to compare with.

### Strengths
The paper shows an elaborated approach to noncausal MOT. The neural models used seem innovative and novel and the attempt to combine tracklet association with a priori knowledge of lane maps in one end-to-end framework is promising.

### Weaknesses
Unfortunately, the paper is difficult to understand for the reader. Many details are presented in a course to a more detailed manner.
After reading the paper, it is not clear what exactly the contribution is. The idea of using object permanence for tracking has been introduced previously (Tokmakov et al.). The three steps of MOT are not new (Zhang, Li, Nevatia, Global Data Association for Multi-Object Tracking Using Network Flows, 2012).
The neural models seem novel. However, the claim in the abstract that the new models improve IDS by 45% is not confirmed by the experimental results. Table 2 shows the causal MOT of Wang et al., 2021 with 109 IDS better than the proposed method with 147+ depending on the version. The performance measures do not show significant evidence for improvement over the causal MOT approach, neither for table 1 nor for table 3.

### Questions
Considering the result of your experiments, as shown in table 1 - 3, why is the proposed method superior to the compared causal trackers?
Why is the proposed method better suited for offline labelling than the online trackers?
Does the proposed methods with approx. AMOTP 0.603 and IDS 145 allow the labelling of datasets?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
To track occluded objects, this paper proposes an offline tracking framework, including an online tracker to generate initial tracklets, a reid module to associate tracklets, and a track completion module to complete the fragmented tracks. Through aggregating and decoding outputs from several different encoders, the track completion model will output the final refined trajectory. Experiments are performed on nuScenes dataset with different evaluation setups.

### Strengths
1. The proposed framework is novel, which embeds both the motion and lane map to obtain the final matching matrix, and also fuses the time query embeddings to implement the trajectory regression.
2. This paper evaluates the proposed method under different evaluation setups, which demonstrate the effectiveness of the method more clearly in addressing occlusion situations.

### Weaknesses
1. Experiments are only performed on the validation split of nuScene dataset, and compared with few SOTA methods, which lack evidence of its effectiveness to some extent.

2. Based on Table1&2, the effect of the motion embedding doesn't seem obvious. Besides, there are no more ablation studies to show the effectiveness of other designs, like the impact of training with augmented GT data instead of tracker outputs, the impact of embedding time query, and so on.

3. The three modules in the proposed framework are separate from each other, rather than a unified end-to-end model, which makes the proposed framework a bit complicated.

### Questions
The issues I am concerned about are listed in order in the above "Weaknesses". I'll change my rating if the authors explain the first two issues well.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an offline tracking framework with object permanence to reduce the expensive labor cost in labeling large-scale autonomous driving datasets. The proposed approach can be briefly summarised as several steps: 1) applying the off-the-shell detector and tracker to generate initial tracklets; 2) using the Re-ID module for tracklet association; 3) employing the track completion module for trajectory completion. Specifically, the effectiveness of the model is validated on the nuScenes validation split.

### Strengths
- This paper aims to solve an essential problem in autonomous driving dataset labelling.
- Quantitative and qualitative results show some superiority of the proposed approach over the compared approaches.

### Weaknesses
- The technical contribution is limited. The proposed approach heavily relies on off-the-shell detectors/tracker, and are inspired from existing approaches a lot (especially for the track completion module), which seems not significant enough as the main contributions by considering the object permanence conception had already been proposed in previous works [1].
- Missing details about the training hyper-parameters for reproduction.
- The proposed approach does not show significant improvements over the compared Immortal tracker (See Tables 1-3). Is there any specifically design in Immortal tracker making the comparison unfair? Otherwise, it cannot effectively show the superiority of the proposed approach in this paper.

[1] Pavel Tokmakov, Jie Li, Wolfram Burgard, and Adrien Gaidon. Learning to track with object permanence. In ICCV, pp. 10860–10869, 2021.

### Questions
- The technical contribution is limited. The proposed approach heavily relies on off-the-shell detectors/tracker, and are inspired from existing approaches a lot (especially for the track completion module), which seems not significant enough as the main contributions by considering the  object permanence conception had already been proposed in previous works [1].
- Missing details about the training hyper-parameters for reproduction.
- The proposed approach does not show significant improvements over the compared Immortal tracker (See Tables 1-3). Is there any specifically design in Immortal tracker making the comparison unfair? Otherwise, it cannot effectively show the superiority of the proposed approach in this paper.

[1] Pavel Tokmakov, Jie Li, Wolfram Burgard, and Adrien Gaidon. Learning to track with object permanence. In ICCV, pp. 10860–10869, 2021.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
