# UniTrack: Differentiable Graph Representation Learning for Multi-Object Tracking

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
We present UniTrack, a plug-and-play graph-theoretic loss function designed to significantly enhance multi-object tracking (MOT) performance by directly optimizing tracking-specific objectives through unified differentiable learning. Unlike prior graph-based MOT methods that redesign tracking architectures, UniTrack provides a universal training objective that integrates detection accuracy, identity preservation, and spatiotemporal consistency into a single end-to-end trainable loss function, enabling seamless integration with existing MOT systems without architectural modifications. Through differentiable graph representation learning, UniTrack enables networks to learn holistic representations of motion continuity and identity relationships across frames. We validate UniTrack across diverse tracking models and multiple challenging benchmarks, demonstrating consistent improvements across all tested architectures and datasets including Trackformer, MOTR, FairMOT, ByteTrack, GTR,  and MOTE. Extensive evaluations show up to 53\% reduction in identity switches and 12\% IDF1 improvements across challenging benchmarks, with GTR achieving peak performance gains of 9.7\% MOTA on SportsMOT. Code and additional resources are available at https://github.com/ostadabbas/UniTrack and https://ostadabbas.github.io/unitrack.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces UniTrack, a differentiable graph representation learning framework designed to unify detection accuracy, identity preservation, and spatio-temporal consistency for multi-object tracking (MOT). Instead of relying on separate detection and association modules, UniTrack formulates tracking as a graph flow optimization problem and introduces a unified loss composed of three parts—flow, spatial, and temporal—optimized in an end-to-end differentiable manner. The framework includes an adaptive Laplacian-based weighting mechanism and can be integrated into existing trackers (e.g., MOTR, TrackFormer, ByteTrack, FairMOT, GTR) without architectural modifications. Experiments across MOT17, MOT20, SportsMOT, and DanceTrack show consistent gains (up to +9.7% MOTA, +12.3% IDF1) and reduced identity switches, highlighting the effectiveness and generality of the proposed framework.

### Strengths
1. UniTrack combines detection, identity, and temporal consistency into a single differentiable loss.
2. UniTrack can be applied to tracking architectures without network modification, demonstrating practical flexibility.
3. Experiments across MOT17, MOT20, SportsMOT, and DanceTrack show consistent gains (up to +9.7% MOTA, +12.3% IDF1) and reduced identity switches, highlighting the effectiveness and generality of the proposed framework.

### Weaknesses
1. The method introduces more training complexity and memory overhead; scalability to large scenes or dense MOT scenarios remains a concern.
2. The authors are suggested to add more ablation studies on the adaptive Laplacian weighting.
3. The ablation section focuses mainly on component removal but could include more fine-grained analysis of hyperparameters (e.g., λs, λt updates, thresholding strategies).
4. It is recommended that the authors compare UniTrack with more recent MOT approaches, especially those that incorporate memory-based mechanisms such as MOTRv2, MeMOTR, and MOTIP, to better show the advantages of the proposed framework.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a new loss for multi-object tracking. This loss is based on graph theory, which integrates detection accuracy, identity preservation, and spatiotemporal consistency. Convergence and consistency of the loss are theoretically validated. Experiments on multiple trackers and multiple benchmarks demonstrate the effectiveness of the proposed loss.

### Strengths
1. The idea of using a plug-and-ply graph-based loss makes sense.
2. The implementation of the graph-based loss is suitable for the MOT task.
3. The analysis of the loss is reasonable.
4. The loss is effective with different trackers on multiple benchmarks, which shows the universality of the proposed loss.

Overall, this work develops a loss which has both solid theoretic foundation and obvious improvement in practice. I believe this work will benefit the community.

### Weaknesses
The weights of spatial and temporal loss are adaptive to the graph connectivity. It is encouraged to compare with other solutions, like adaptive parameters directly learned by the network, and fixed parameters.

### Questions
Will the code be made publicly available to help re-implement and apply this work? It would further benefit people in this field.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
UniTrack introduces a training-only loss that couples three differentiable terms: flow, spatial, and temporal inside a graph, over a sliding-window graph with flow conservation and adaptive 𝜆. λ chosen by Laplacian connectivity; it is added to TrackFormer/MOTR/FairMOT/ByteTrack/GTR/MOTE without inference overhead. Reported gains across several mot benchmarks including mot 17/20.

### Strengths
1. One of the biggest advantages is that the proposed method is architecture‑agnostic: it demonstrates improvements when plugged into diverse families (end-to-end transformers, joint detection-tracking, tracking-by-detection, global transformers).

2. The proposed unified objective makes sense as it merges detection quality and identity preservation, and benefit the end-to-end MOT training.

3. The authors show Clear ablation on error types (Table 3), clearly presenting which term combats which failure mode; qualitative figures are convincing.

### Weaknesses
1. The details of the differentiability of the flow term are not clearly conveyed. The loss scales by factors that depend on false positives/false negatives, but the paper does not define a differentiable surrogate for those counts. As far as i understand that derivation treats the FP/FN counts inside the loss as if they were constants and never explains how those counts are made differentiable with respect to the model outputs. In practice, FP/FN are discrete functions of predictions (they jump when a score crosses a threshold.

2. Inconsistent definitions: the paper has defined λs and λt ((Eq. 8), as they are deterministic functions of graph connectivity (via Laplacian eigenvalues). Then, in eq.10, the λs and λt are treated as learnable parameters that can be updated by backprop. 

3. The paper includes frame-rate analysis and normalizes by the frame interval, which is good. Still, some ablations suggest removing the temporal term can improve certain metrics. eg MOTA by 2.1.

4. Prior work already models inter-object relations and global data association with differentiable mechanisms [1]

[1] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
