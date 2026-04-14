# MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion

- Decision: Accept (Spotlight)
- Scores: 6, 6, 8, 8, 8

## Abstract
Estimating geometry from dynamic scenes, where objects move and deform over time, remains a core challenge in computer vision. Current approaches often rely on multi-stage pipelines or global optimizations that decompose the problem into subtasks, like depth and flow, leading to complex systems prone to errors. In this paper, we present Motion DUSt3R (MonST3R), a novel geometry-first approach
that directly estimates per-timestep geometry from dynamic scenes. Our key insight is that by simply estimating a pointmap for each timestep, we can effectively adapt DUSt3R’s representation, previously only used for static scenes, to dynamic scenes. However, this approach presents a significant challenge: the scarcity of suitable training data, namely dynamic, posed videos with depth labels. Despite this, we show that by posing the problem as a fine-tuning task, identifying several suitable datasets, and strategically training the model on this limited data, we can surprisingly enable the model to handle dynamics, even without an explicit motion representation. Based on this, we introduce new optimizations for several downstream video-specific tasks and demonstrate strong performance on video depth and camera pose estimation, outperforming prior work in terms of robustness and efficiency. Moreover, MonST3R shows promising results for primarily feed-forward 4D reconstruction. Interactive 4D results, source code, and trained models are available at: https://monst3r-project.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces MonST3R, a novel geometry-first approach for estimating 3D structure from dynamic scenes in videos. Unlike existing methods that rely on multi-stage pipelines prone to errors, MonST3R adapts the DUSt3R pointmap representation, originally designed for static scenes, to handle moving objects by estimating pointmaps at each time step. This adaptation is achieved through fine-tuning on carefully selected synthetic and real-world datasets that provide depth and camera pose annotations, despite the challenge of limited data availability. The method demonstrates significant improvements in video depth and camera pose estimation, surpassing traditional optimization-based methods in robustness and efficiency. MonST3R is capable of generating reliable 4D reconstructions in a predominantly feed-forward manner, enhancing performance in various video-specific tasks.

### Strengths
1. The proposed problem of reconstructing static scenes and estimating per-timestep geometry of dynamic objects is compelling and relevant.
2. The paper presents a reasonable and well-structured solution to the defined problem.
3. The method achieves state-of-the-art performance, demonstrating its effectiveness.

### Weaknesses
1. The complete optimization in Eq. 6 minimizes the loss to obtain optimal values for X, P_W, and \sigma. However, it seems that the smoothness loss in Eq. 4 is minimized specifically for R and T. The input notation L_{smooth}(X) might be misleading—should it be L_{smooth}(R,T) instead?
2. The notation for L_{flow}(X) also seems ambiguous. Would L_{flow}(F) or L_{flow}(F,S) be more appropriate to reflect the actual inputs?
3. There is no explanation provided for the notation global. Clarifying what the global parameters are and how S^{global} is derived would be helpful. Specifically, is S^{global} in Eq. 5 an extension of Eq. 2 that uses F_{cam}^{global,t→t′} and F_{est}^{global,t→t′}
4. The proposed network is trained on diverse datasets, including PointOdyssey, TartanAir, Spring, and Waymo Perception, but the authors rely on a pretrained optical flow model. This could introduce a domain gap between the depth network and the optical flow network. Were there any issues related to domain gaps during the experiments?
5. It would be beneficial if the authors provided depth evaluation results for static scenes and dynamic objects separately. This would clarify whether the performance improvement is due to coherent depth estimation on static scenes or the enhanced depth quality of dynamic objects.
6. Do the authors use zero-shot models like Marigold and Depth-Anything? The use of the Waymo dataset (similar to KITTI) and the PointOdyssey dataset (with properties akin to Sintel) might explain why the proposed method outperforms these models. Can the authors specify which components of their approach are most critical for achieving this performance boost?
7. Overall, the technical parts is not clearly written.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MonST3R, a method for estimating geometry in dynamic scenes. It adapts the DUSt3R's pointmap representation to handle moving objects. Key points include: using suitable synthetic and real datasets for training despite data scarcity, introducing new optimization methods for downstream tasks like depth and pose estimation, demonstrating strong performance compared to prior works, and achieving good results in 4D reconstruction.

### Strengths
- MonST3R successfully adapts the DUSt3R's pointmap representation to dynamic scenes. By estimating a pointmap for each timestep and representing them in the same camera coordinate frame, it can handle the movement of objects.
- Data Efficiency: Trains well with limited data via strategies.
- Good Performance: Strong in video depth and pose estimation.

### Weaknesses
- There are also some high-quality synthetic datasets in the video field, such as DynamicStereo and IRSDataset. The author could consider expanding the scale of the training data to explore better model performance.
- There are doubts about the metrics used by the author when comparing video depth estimation. When the delta metric is below 0.7 or lower, it is usually considered that the prediction has completely collapsed. However, as far as the reviewer knows, the video depth estimation methods compared by the author have good performance and can provide basically correct depth estimation results at least on the Sintel dataset. Some metrics, such as NVDS, seem to be inconsistent with the report in the paper.
- It is necessary to explore the robustness of DUSt3R, that is, the failure cases when the scene is too dynamic.

### Questions
See weakness

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
This paper approaches the task of obtaining depth and camera pose given dynamic video. To do so, it begins from the DUSt3R architecture and pretraining. The paper argues optimizing dynamic scenes requires good estimation of foreground depth, and alignment based on dynamic objects; both of which DUSt3R struggles on due to static training data of mostly backgrounds. To this end, the paper fine-tunes on dynamic videos, as well as proposing slightly different optimization criteria. It shows state-of-the-art results on depth and camera pose estimation on several standard dynamic video benchmarks.

### Strengths
Clearly identifies and addresses key shortcomings of DUSt3R when applied to dynamic scenes
- This work identifies non-obvious weaknesses of DUSt3R in foreground depth prediction, in addition to more predictable poor dynamic correspondence performance. Experiments show clear improvement accordingly
- Getting MonST3R to work with dynamic points is nontrivial. Careful finetuning choices in data and network, along with regularizations to handle a point representation taking into account time must have been necessary to achieve such strong performance
- Comprehensive ablations clearly show impact of several contributions in data, training and inference. Notably, PointOdyssey (dynamic) is essential, and fine-tuning specifically the decoder and head is critical.

Effective and efficient performance on an important task
- SOTA performance on depth and camera pose estimation across standard dynamic video datasets: Sintel, TUM-Dynamics, Bonn, KITTI, NYU-v2.
- Impressive the method is still competitive on static scenes ScanNet, showing this fine-tuning does not sacrifice original DUSt3R results.

My rating of 8 (Accept) is because of strong experiment results (SOTA Depth and Pose), resulting from clearly identified weaknesses (static and distant training data), from a strong starting point (DUSt3R); weaknesses are minor.


***Post-discussion Update*** I leave this paper at an 8. The rebuttal well addresses weaknesses in this paper. I do not feel it is worthy of a 10 because the contributions are not exceptional, i.e. it consists mostly of fine-tuning on synthetic data for a sizeable but not massive gain; but results, writing and method are still very strong and the paper should be accepted.

### Weaknesses
(Minor) Camera Pose Estimation suite could be more comprehensive
- E.g. COLMAP or masked COLMAP baseline would be good benchmarks.
- Why are DROID-SLAM, DPVO and ParticleSfM not run on TUM, and DROID-SLAM and DPVO not on ScanNet? 
- LEAP-VO is highly competitive with the proposed method. This is a minor weakness given this method also estimates depth and improves on DUSt3R, but should be considered when analyzing e.g. if this paper should be considered for strong accept.

### Questions
See Weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper extends DUSt3R, originally a feed-forward method for static scene reconstruction, to handle dynamic scenes by reconstructing scene geometry at each timestep. The initial performance of DUSt3R was poor due to its training on static datasets only. To improve results, this paper identifies and fine-tunes DUSt3R on several dynamic datasets. Consequently, the proposed method delivers competitive results in video depth estimation, camera pose estimation, and dense reconstruction. The approach is efficient and demonstrates results that rival those of specialized techniques.

### Strengths
Clarity and Readability: The paper is well-written, with a clear statement of the motivation, methodology, and challenges, making it easy to follow.

Effective Demonstration: The video demonstration is impressive, showcasing efficient 4D reconstruction with significant potential for downstream applications.

Sufficient Novelty: While the proposed approach builds on DUSt3R, extending it to dynamic scenes is non-trivial. The authors clearly outline the challenges and propose effective solutions to address them.

Convincing Evaluation: The evaluation is thorough, comparing the method to different categories of existing techniques, with both quantitative and qualitative results presented.

### Weaknesses
1. Mention of Related Work: For video depth estimation, self-supervised monocular depth methods such as Manydepth [a] and SC-DepthV3 [b] should be mentioned. SC-DepthV3, in particular, provides video depth estimation results on the Bonn dataset, which would make for a relevant comparison.

[a] The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth, CVPR 2021.

[b] SC-DepthV3: Robust Self-Supervised Monocular Depth Estimation for Dynamic Scenes, TPAMI 2024.

2. Robustness in Dynamic Regions: The statement “For most scenes, where most pixels are static, random samples of points will place more emphasis on the static elements” is generally true, but there are edge cases that require more robust approaches. For example, in the Bonn dataset, moving objects such as humans can occupy over 50% of the image. Potential solutions could include masking out dynamic regions using segmentation masks or re-running relative pose estimation after identifying dynamic regions.

### Questions
See weakness

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper first identifies the limitations in joint pose and geometry estimation methods on dynamic scenes, and proposes to finetune based on the recent geometric learning model DUSt3R, representing the dynamic scenes with per-timestamp pointmaps. During global alignment for the pairwise inference results, trajectory smoothness loss and flow projection loss are added to the alignment term to adapt the optimization on sequential dynamic inputs. Experimental results show that the proposed method achieves state-of-the-art performance on the video depth estimation task and remains competitive in camera pose estimation, while jointly estimating both.

### Strengths
1. The method motivates the problem well on joint pose and geometry estimation on dynamic scenes by identifying the limitations in the current geometric learning method DUSt3R which is trained on static scenes.
2. With carefully selected datasets for dynamic scenes, together with a window-based training strategy, the scarcity of dynamic training data for finetuning is dealt with.
3. Finetuning results validate the proposed idea, with only the decoder and prediction head tuned, the model is capable of predicting pointmaps for the dynamic regions, showing sharping improvement compared to the DUSt3R with mask baseline.
4. The method achieves SoTA results on video depth estimation task, and competitive results on video pose estimation. The selected baselines are thorough, including baselines specialized in each task.

### Weaknesses
1. The loss used in fine-tuning is not detailed in the methods section. In line 294, "static mask is both a potential output and will be used in the later global pose optimization", the word "potential" here is confusing, is the static mask also used during finetuning?
2. Although a sliding window technique is used, the runtime reported is 30s for pairwise inference and 1min for global optimization given the 60 frames (around 2.5s) video input, which is slow.

### Questions
1. In the depth comparison in figure A1, the depth colormaps effectively hides the error, it's better to also add error colormaps. Additionally, in all the visualized frames, the model all predicts closer depth compared to ground truth on the dynamic parts. Is this specific to this dataset or is it a general bias of the model? What could be the possible reasons?
2. In lines 261-269, to calculate induced flow from camera motions, the method needs to estimate the intrinsics separately from the pointmaps. I wonder if using dataset intrinsics would degrade the performance?
3. Another related question on intrinsics, in lines 456-458, the relax of requiring intrinsics can be seen as a strength, however DUSt3R is not able to include known intrinsics. Is it considered to include the input intrinsics during the method development?
4. As we see from DUSt3R to MASt3R, the pointmaps serve as general and powerful representation for static geometry, as multiple downstream task can be estimated based on them. However, it's questionable whether it fit best for dynamic scenes. Here, an off-the-shelf flow estimator is needed to help obtain the static/dynamic mask, are flow fields themselves better suited for representing dynamics?

### Soundness
3

### Presentation
3

### Contribution
3
