# 3DPoV: Improving 3D understanding via Patch Ordering on Videos

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Visual foundation models have achieved remarkable progress in scale and versatility, yet understanding the 3D world remains a fundamental challenge. While 2D images contain cues about 3D structure that humans readily interpret, deep models often fail to exploit them, underperforming on tasks such as multiview semantic consistency--crucial for applications including robotics and autonomous driving. We propose a self-supervised approach to enhance the 3D understanding of vision foundation models by (i) introducing a temporal nearest-neighbor consistency loss that finds corresponding points across video frames and enforces consistency between their nearest neighbors, (ii) incorporating reference-guided ordering that requires patch-level features to be not only expressive but also consistently aligned, and (iii) constructing a mixture of video datasets tailored to these objectives, thereby leveraging rich 3D information. Our method, 3DPoV, achieves state-of-the-art performance in keypoint matching under viewpoint variation, as well as in depth and surface normal estimation, and consistently improves a diverse set of backbones, including DINOv3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work develops a training framework to enhance the feature representations of self-supervised backbones (DINO, DINOv2) on 3D awareness tasks, such as multi-view correspondences, surface normal estimation and depth estimation. The framework trains on videos and uses point tracking as an additional cue for constructing the training objective. Relying on the student-teacher framework, the approach creates the training signal by computing a soft permutation matrix from the teacher and minimising the difference w.r.t. the corresponding matrix derived from the student network. The permutation matrix encodes the ordering of feature similarities: a query point, tracked in time, is compared to the “reference” points extracted from other videos and frames. The query points in the teacher and the student network may come from different frames in the video, but related to the same physical point, as predicted by a point tracker.

The approach trains only the final layer of the backbone network. The empirical results show benefit of the proposed training methodology across three tasks (depth, surface normals and multi-view consistency).

### Strengths
This is an interesting work with a few strong points to highlight:
* The work explores an exciting combination of point tracking and representation learning for 3D awareness.
* The proposed training strategy is new and interesting, yielding consistent benefits across the evaluated tasks. The results are particularly strong on keypoint matching.
* The evaluation included two backbone networks and explors a combination of three pre-training datasets.

### Weaknesses
* The empirical results are a bit mixed. While the approach does show a significant improvement on the SPair-71k dataset for multi-view correspondence, the benefit on depth and normal estimation is very marginal. The ablation study reveals further that point tracking yields marginal improvements compared to the use of optical flow. The experiment “Number of frames” (c.f. Tab. 5b) also undermines the need for point-tracking: considering only two frames already leads to a boost in downstream accuracy, on par with the setting using a larger temporal context. Overall, the experiments do not yet compellingly show a benefit of a longer temporal context and point tracking.
* The framework design is non-trivial, but the ablation study does not elaborate on key design choices. For example, differentiable sorting is central to the training objective, but there is no empirical study to show it is more effective than simpler alternatives (e.g. contrastive loss, Sinkhorn-Knopp). Another example is the choice of the student and teacher features: one is taken from the future frames, while the other is from t = 0. A third example are the reference crops: how sensitive is the approach to the number of the reference crops?
* The evaluation setting is somewhat unclear. Probe3D trains a DPT decoder for surface normal and depth estimation. The DPT decoder has skip connections to indermediate layers of the encoder. If the DPT is used here, this implies that only a single skip connection is affected, since the work trains only the final layer of the base model. This creates a strong constraint on the potential improvement, and is unlikely to affect the downstream accuracy in a significant way.

### Questions
* What is the difference between 3DPoV/ViT-B16 in the tables (e.g. 6th and 12th row in Tab. 1)? Is it a typo and one of them should be ViT-S16?
* How many parameters are finetuned in the base network? What happens if we increase the number of the finetuned layers? 
* Please clarify the evaluation protocol (see the note on the DPT decoder in “Weaknesses”).

### Soundness
2

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
The paper proposes 3DPoV, a self supervised post training objective that improves 3D awareness in vision backbones by aligning the ranking of patch similarities across video frames using differentiable sorting. A teacher student design with tracked points supplies temporally consistent and viewpoint tolerant signals without 3D labels. The method reports consistent gains on Probe3D keypoint matching under viewpoint changes and on linear probes for depth and normals.

### Strengths
- Originality: Ranking based temporal supervision on patch similarities is a fresh alternative to direct feature matching and leverages point tracks effectively.

- Quality: Results are consistent across multiple backbones with focused ablations on frames, datasets, and tracker choice.

- Clarity: The pipeline and loss are clearly specified with equations and a reasonable visibility treatment.

- Significance: Consistent gains on viewpoint sensitive benchmarks indicate that 3DPoV injects geometry aware structure into widely used backbones, which is practically valuable.

### Weaknesses
- **Sensitivity to tracking and visibility**: The approach depends on point tracks and learned visibility; robustness under drift, rapid motion, or occlusion could be analyzed more systematically.
- **Scope beyond Probe3D**: The results emphasize Probe3D; adding small but representative downstream tasks such as camera pose estimation or robotic correspondence would strengthen external validity.
- **Missing baselines and related works**: How does the method compared with "Multiview Equivariance Improves 3D Correspondence Understanding with Minimal Feature Finetuning" and "Omniview-tuning: Boosting viewpoint invariance of vision-language pre-training models", which also tried to improve the 3D understanding of DINO.
- **Results are a bit marginal**: Though there are consistent improvements over different datasets, they are a bit marginal and the method does not seem to gain much with a heavy video guided correspondence finetuning task.

### Questions
- How sensitive is performance to tracker noise and missed correspondences. Can you report results with synthetic noise or alternative trackers and include failure cases?
- Have you evaluated on additional multiview or other tasks to demonstrate transfer beyond Probe3D?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes 3DPOV, a self-supervised, post-training method to improve the 3D spatial understanding and viewpoint invariance of visual foundation models. The core problem it addresses is that models like DINO, while strong in 2D, fail to maintain consistency when camera poses change significantly.

The key idea of 3DPOV is to leverage video to enforce temporal consistency of patch-level feature ordering. Instead of forcing feature descriptors to be identical across time, it trains the model to maintain a consistent relative ranking of patch similarities when compared against a shared set of reference features. The method uses a point tracker (CoTrackerV3) to find patch correspondences across frames, a teacher-student (EMA) framework for stable targets, and differentiable sorting to create a trainable loss objective.

The method is trained on a mixture of video datasets (CO3D, DL3DV, YouTube-VOS) and evaluated on the Probe3D benchmark, where it demonstrates consistent improvements over strong baselines (DINOv2, DINOv3) in keypoint matching, depth estimation, and surface normal estimation, particularly under large viewpoint variations.

### Strengths
**Sound Objective:** The core contribution, supervising the relative ordering of patch similarities rather than their absolute values, is a clever learning objective. This relative supervision is probably less sensitive to small appearance variations (e.g., lighting) yet still captures the core geometric/semantic structure, promoting a more consistent similarity space.

**Good Empirical Results:** The method shows clear, consistent gains on the Probe3D benchmark. It critically outperforms baselines (like NeCo) in "medium" and "large" viewpoint shift categories, validating the paper's claim of improving robustness to viewpoint changes.

**Efficiency:** The method is presented as a lightweight post-training (fine-tuning) step, making it a practical way to enhance existing, pre-trained foundation models without the need for full retraining.

### Weaknesses
1. Justification of Design Choices: Some key design choices are not well-motivated.

- Inter-Video References: The use of reference patches from other videos in the batch is not well-explained. It's unclear why forcing a patch's similarity ranking relative to irrelevant patches from a different scene would help learn viewpoint-consistent features for the current object. This needs a clearer explanation or ablation.

- Final-Layer-Only Tuning: The paper states that only the final layer is fine-tuned. This is a highly constrained approach for a task that seems to require learning new, fundamental spatio-temporal invariances. It's surprising that this is sufficient, and this choice warrants a stronger justification.

2. Clarity in Presentation: The paper suffers from some minor but important clarity issues.

- Typo in Loss Function: Equation 4 appears to have a typo in its subscripts; the student and teacher permutation matrices are swapped.

- Ambiguous Table Labels: According to L#274 “Unless other stated”, in Table1, 3DPoV under Dino block and 3DPoV under Dino V3 block are same one? Then why there performance is different, or it’s based on different Dino variants?

### Questions
Please refer to the questions in the Weaknesses section. If these are addressed satisfactorily, I will consider raising my score.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes 3DPoV, a self-supervised post-training method that improves multiview/3D consistency of vision foundation models by (i) a temporal permutation loss over patch similarities across frames, (ii) a teacher–student scheme with a reference pool for robust supervision, and (iii) a mixture of video datasets (CO3D, DL3DV, YouTube-VOS). The target is to enhance viewpoint-robust features for Probe3D tasks: keypoint matching, depth, and surface normals.

### Strengths
1. The teacher–student + differentiable sorting pipeline is clear and easy to integrate with existing ViT backbones. The visibility-weighted loss is a neat, principled way to de-emphasize occluded/out-of-track patches.
2. Consistent improvements across regimes and backbones. The paper claims improvements without trading off small-viewpoint for large-viewpoint performance, and could be helpful for downstream tasks. 
3. The method requires minimal computational resources compared to Dinov3 pre-training (20 GPU-hours vs. weeks)

### Weaknesses
1. Limited Technical Novelty: While the paper presents a functioning system with promising empirical results, the core technical contribution primarily consists of combining well-established components from existing literature. Specifically, the method integrates: (i) standard point tracking via CoTrackerV3 (Karaev et al., 2024) without modification; (ii) the differentiable sorting mechanism from NeCo (Pariza et al., 2025) and Petersen et al. (2022), simply adapted from spatial to temporal domains; (iii) a conventional teacher-student framework with EMA updates following DINO (Caron et al., 2021); and (iv) temporal consistency objectives previously explored in TimeTuning (Salehi et al., 2023) and MoSiC (Salehi et al., 2024). While this specific combination yields positive results, the paper would be strengthened by either introducing more distinctive technical innovations or providing deeper theoretical insights explaining why this particular configuration enhances 3D understanding beyond empirical observation.
2. Clarity Issue (Minor): Table 2 contains two entries labeled "3DPoV ViT-B16" (rows 6 and 12) without clear differentiation. Based on the table structure and placement after the DINOv3 baseline, the second entry appears to use DINOv3 as the backbone, but this distinction is only clarified in Appendix E rather than the main text. Please consider explicitly labeling these variants in the table itself for clarity.
3. Diminishing Returns on Strong Baselines: More concerning is that when applied to state-of-the-art models, the method shows minimal improvement. Specifically, the gain over DINOv3 in Table 2 (94.40→94.47 on Navi θ₀¹⁵) represents only a 0.07% increase, suggesting the approach provides negligible benefits when applied to modern foundation models.

### Questions
Check the weakness above.
1. Why does patch ordering improve surface normal estimation? The connection between similarity rankings and learning surface orientation isn’t clear. Could you elaborate?
2. Which scenes or motions cause 3DPoV to fail? It would be more insightful to include failure cases as well.

### Soundness
3

### Presentation
3

### Contribution
2
