# PDTrack: Progressive Dense 3D Tracking

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
We propose a novel algorithm for accelerating dense long-term 3D point tracking
in videos. Through analysis of existing state-of-the-art methods, we identify two
major computational bottlenecks. First, transformer-based iterative tracking be-
comes expensive when handling a large number of trajectories. To address this,
we introduce a coarse-to-fine strategy that begins tracking with a small subset of
points and progressively expands the set of tracked trajectories. The newly added
trajectories are initialized using a learnable interpolation module, which is trained
end-to-end alongside the tracking network. Second, we adopt a lightweight im-
plementation for the 4D correlation block that reduces its computational cost on
common GPU backends. Together, these improvements lead to a 5–100× speedup
over existing approaches while maintaining state-of-the-art tracking accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a framework that employs a coarse-to-fine strategy together with a learnable interpolator to gradually increase the number of tracked pixels. This approach significantly reduces overall memory consumption and speeds up the original baseline.

### Strengths
Strength:

Novelty: A well-designed combination of iterative refinement and coarse-to-fine strategy that substantially improves tracking speed while maintaining the baseline accuracy.

Quality: The paper includes extensive experiments and ablation studies demonstrating the importance of the proposed module. The writing is clear and the methodology is easy to follow.

### Weaknesses
1) Is the learnable interpolator’s weight shared across iterations?

2) Table 7: The performance (OA & APD) on PSTUDIO lags behind BootsTAPIR and CoTracker2, which are 2D methods. Could the authors provide intuition or an explanation for this behavior?

3) Table 12: Tables (a) and (b) present the interpolator accuracy before and after upsampling. However, why does the accuracy decrease as the number of iterations increases? Why is the “learnable” interpolator performing worse than “nearest” and “bilinear”? I would expect the learnable version to achieve the best results—or do these numbers represent error rather than accuracy? 

4) Additionally, I could not find any discussion of Table 12 in the paper.

5) Comparison with CoTracker3, TAPIP3D.

### Questions
Please check the weaknesses.

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
5

### Summary
This paper presents PDTrack, a dense 3D point tracking framework that aims to improve efficiency over DELTA.
The method introduces three changes: (i) Coarse-to-fine trajectory subsampling, which tracks a sparse subset of points first and progressively densifies them; (ii) learnable interpolation module that predicts motion for untracked pixels based on nearby tracked ones; (iii) an optimized 4D correlation implementation for faster GPU execution. PDTrack reports up to 5-100× speedup compared to DELTA while maintaining similar accuracy on Kubric3D and TAP-Vid3D. The method is evaluated on both dense 3D and long-range 2D tracking benchmarks, with ablations on runtime and sampling strategies.

### Strengths
- The paper clearly identifies bottlenecks in dense tracking pipelines and provides quantitative analysis of each.
- Extensive runtime and accuracy trade-off studies justify the design choices, such as subsampling strategies and interpolation variants.
- The authors provide speed-accuracy curves, per-module breakdowns, and ablations on multiple datasets.
- Paper presentation is clean and easy to follow.

### Weaknesses
## Limited novelty
The coarse-to-fine scheme itself is an engineering improvement, not a new learning paradigm. Consequently, PDTrack reads as an efficiency-oriented extension of DELTA rather than a novel research contribution. **While runtime improvements are valuable and appreciated,** the paper lacks a deeper insight that would justify a full ICLR publication.

## Connection to prior work
- The interpolation mechanism is very similar to TAPTRv2 [1]’s iterative feature-weighted update rule, raising concerns about novelty.
- The coarse-to-fine densification mirrors strategies used in optical-flow pipelines, such as FlowFormer [2], but applied here at token level.

## Evaluation and completeness
- The motivation emphasizes efficiency, but Appendix B6 shows 40.1 GB GPU memory usage, indicating the method is still heavy.
- Table 4 (TAP-Vid3D) omits key recent 2D trackers with depth lifting such as CoTracker3 [3], Track-On [4], and BootsTAPNext [5], which are necessary to contextualize the performance advantage of full 3D tracking.

## Overall
While speedups are significant, the research insight is incremental. The method neither alters the representation of motion nor introduces a new way to reason about geometry. This makes the work better suited as an engineering-oriented extension of DELTA, not a standalone contribution.

---

## References  
[1] Li et al., TAPTRv2: Attention-based Position Update Improves Tracking Any Point, NeurIPS 2024

[2] Huang et al., FlowFormer: A Transformer Architecture for Optical Flow, ECCV 2022

[3] Karaev et al., CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos, ICCV 2025

[4] Aydemir et al., Track-On: Transformer-based Online Point Tracking with Memory, ICLR 2025

[5] Zholus et al., TAPNext: Tracking Any Point as Next-Token Prediction, ICCV 2025

### Questions
- Is the depth estimation time included in the reported run-time calculations?
- What explains the greater robustness on extended optical-flow sequences (Table 2) despite minimal architectural changes from DELTA?

### Soundness
2

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
This paper presents PDTrack, a progressive dense association framework designed to accelerate 3D point cloud single-object tracking. The core idea is to replace complex matching processes in existing trackers with a lightweight, progressively refined association scheme that improves tracking efficiency without substantially sacrificing accuracy. Experiments are conducted on several LiDAR-based tracking benchmarks to validate the method’s speed and accuracy balance.

### Strengths
1. The proposed progressive dense association structure is well-engineered and appears to provide tangible runtime benefits for 3D tracking, which is valuable for real-time applications.
2. Experimental results demonstrate moderate improvements in speed compared to existing baselines, suggesting that the method may serve as an efficient alternative in resource-limited scenarios.

### Weaknesses
1. The contributions are mostly engineering-oriented, improving efficiency through architectural simplifications rather than introducing new algorithmic or theoretical ideas. 
2. The progressive dense association concept feels like an incremental adaptation of existing dense matching or coarse-to-fine refinement techniques already explored in 3D tracking literature.
3. The motivation is not clearly articulated. Why progressive dense association is fundamentally better than sparse matching or correlation-based tracking is not convincingly justified.
4. The method’s acceleration effect is not comprehensively validated across different 3D tracking architectures. Since this is a general acceleration technique, it should ideally be tested on multiple 3D point tracking methods.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PDTrack, a fast and dense long-term 3D pixel tracking method for RGB-D videos. It addresses two key bottlenecks in DELTA and proposes three novel modules: a coarse-to-fine trajectory densification strategy, a learnable attention-based interpolation module, and a kernel-efficient 4D correlation block. PDTrack achieves comparable or slightly better accuracy than DELTA on dense 3D tracking and long-range optical flow tasks, while delivering a 5–100× speedup over existing methods.

### Strengths
1. This paper proposes a coarse-to-fine tracking algorithm with learnable attention-based interpolation that starts with sparsely sampled trajectories and densifies them over iterations
2. The author designed an simple accelerated 4D correlation implementation that improves kernel efficiency on common GPU backends.
3.The paper identifies the runtime bottlenecks in existing tracking methods and proposes a simple yet effective solution, supported by extensive experiments demonstrating its efficiency and effectiveness.

### Weaknesses
1. The paper evaluates the accelerated 4D correlation implementation only on the A100 GPU, without assessing its performance on other GPU architectures.
2. The interpolator employs four nearest neighbors, but the impact of using different numbers of neighbors is not analyzed.
3. The interpolator itself is not novel, as it is adapted from the DELTA framework.

### Questions
1. How do you apply your model to sparse tracking? Were any modifications made?
2. Could you conduct an ablation study to evaluate the effect of different numbers of nearest neighbors used in the interpolator?
3. Have you tried using other depth estimation models, such as UniDepthv2 [1]? It would be interesting to see how a more accurate depth estimator could improve the overall performance.

[1] Piccinelli, Luigi, et al. "Unidepthv2: Universal monocular metric depth estimation made simpler." arXiv preprint arXiv:2502.20110 (2025).

### Soundness
3

### Presentation
3

### Contribution
3
