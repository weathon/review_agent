# FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models

- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Lane segment topology reasoning provides comprehensive bird's-eye view (BEV) road scene understanding, which can serve as a key perception module in planning-oriented end-to-end autonomous driving systems. Current approaches prioritize graph modeling, endpoint alignment, and multi-attribute learning, yet they often neglect temporal modeling. This leads to inconsistent inter-frame detection within scene flows and motivates our focus on temporal propagation for lane segments. Recently, stream-based methods have shown promising outcomes by integrating temporal cues at both the query and BEV levels. However, it remains limited by over-reliance on historical queries, vulnerability to pose estimation failures, and insufficient temporal propagation. To overcome these limitations, we propose FASTopoWM, a novel fast-slow lane segment topology reasoning framework augmented with latent world models. To reduce the impact of pose estimation failures, this unified framework enables parallel supervision of both historical and newly initialized queries, facilitating mutual reinforcement between the fast and slow systems. Furthermore, we introduce latent query and BEV world models conditioned on the action latent to propagate the state representations from past observations to the current timestep. This design substantially improves the performance of temporal perception within the slow pipeline. Extensive experiments on the OpenLane-V2 benchmark demonstrate that FASTopoWM outperforms state-of-the-art methods in both lane segment detection and centerline perception. Our code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes FASTopoWM, a novel fast-slow lane segment topology reasoning framework augmented with latent world models, which decouples the network into dual pathways: a slow pipeline that leverages temporal information to enhance detection performance and a fast pipeline that performs single-frame perception for system robustness. The unified architecture enables parallel supervision of both historical and newly initialized queries, facilitating mutual reinforcement between the two systems, and introduces latent query and BEV world models conditioned on ego-motion to effectively capture temporal dynamics and enable robust state propagation. Extensive experiments on the OpenLane-V2 benchmark demonstrate that FASTopoWM achieves state-of-the-art performance in both lane segment detection and centerline perception.

### Strengths
1. **Novelty**: This paper is the first to introduce a fast–slow system into the field of lane-segment topology reasoning, leveraging latent world models to address the limitations of previous methods—specifically, their vulnerability to pose estimation failures and insufficient temporal propagation.

2. **quality**: The method and experiments are well-designed and complete. Comprehensive visualizations and released code make the paper more solid and reproducible.
 
3. **Experiment**: The proposed method achieves state-of-the-art performance on the OpenLane-V2 benchmark.

### Weaknesses
1. **About the novelty claim**: This paper appears to propose a method that improves upon existing stream-based approaches, as illustrated in Figure 1. However, the lane-segment topology reasoning task is not limited to stream-based methods. Therefore, Figure 1, the abstract, and the introduction should be revised to more clearly highlight the distinctions and advantages of this work compared with previous methods in the field.

2. **About the visualization**: The current visualizations only present BEV (bird’s-eye view) results. Showing the lane-segment predictions in the front-view images would better demonstrate the effectiveness of the proposed method. It is recommended to include visualizations from the surrounding-view images for a more comprehensive presentation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper focuses on some issues arising from temporal bev-feature and query propagation in the lane prediction task. In order to alleviate the mentioned problems, the authors propose a framework that can leverage parallel supervision to historical BEV features and instance queries for mutual reinforcement, mainly by sharing the whole model and using the Latent World model to reconstruct the pseudo current features from past features. The extensive experiments demonstrate the effectiveness of the proposed components.

### Strengths
1. The parallel supervision for historical and current features via the latent world model to achieve better feature learning is novel and interesting. 
2. The individual designs in the whole framework are reasonable to achieve the goal. And there are corresponding ablations to validate the effectiveness.

### Weaknesses
1.    Experiments on three claims in the Introduction are missing: (1) over-reliance on historical queries (2) vulnerability to pose estimation failures and (3) Weak temporal propagation. I would like to see some experiments, an analysis or even some cues on how the author identifies these problems. 
2.    Likewise to problem 1, it’s still unclear to me why the “slow-fast system” can solve these problems fundamentally. Some evidence or statistics should be given.
3.    The definition of the Slow & Fast system in this paper may be confusing; originally, the Slow & Fast system on VLA is to address the asynchrony between fast model inference and slow decision processes in real-world action execution. While in this paper, ‘slow system’ and ‘fast system’ are more like a history feature aggregator and a current frame detector.
4.    Why, generally speaking, methods with Temporal feature aggregation perform worse than those without that, as shown on Table1, and 2?
5.    On Table 3, the author presents ablation studies on QWM and FWM, I would like to see the result that retains the QWM and FWM structure but removes the latent supervision (eq(3) and eq(4)).

### Questions
The questions have been listed on the points of weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
FASTopoWM introduces a fast–slow framework with latent world models for temporal lane segment topology reasoning in BEV. The slow pipeline leverages temporal cues by transforming historical queries and BEV features into current “stream” representations via transformer-based query and BEV world models conditioned on relative pose (action latent), while the fast pipeline performs single-frame perception. A unified decoder with shared weights enables parallel supervision of stream and newly initialized queries, mitigating over-reliance on history and providing a robust fallback when pose estimates are unreliable. The BEV world model is trained self-supervised with MSE on adjacent-frame BEV features; the query world model uses transformation losses on coordinates, classes, and masks. On OpenLane‑V2, FASTopoWM reports state-of-the-art results: 37.4% mAP for lane segments and 46.3% OLS for centerlines, outperforming temporal and single-frame baselines.

### Strengths
- The paper is well written and clearly organized, and figures are effectively created to support the content.
- Experimental results verify the effectiveness of the proposed approach against several baselines.

### Weaknesses
**Major Weaknesses**

1. Substantial overlap with TopoStreamer[1] in problem framing and technical pipeline, without the necessary systematic comparison or discussion. TopoStreamer also targets temporal lane segment topology reasoning and essentially adopts the same or highly similar streaming mechanisms and supervisory objectives. Yet the manuscript provides no systematic analysis of differences or pros/cons. This is a serious issue by ICLR standards: when facing the most directly related, recent work on the same task, the authors neither include experimental comparisons nor articulate methodological distinctions and trade-offs. In fact, the two papers share highly consistent setups, so adding a direct comparison should be straightforward.
2. The main text lacks detailed quantitative and qualitative experiments to substantiate the claimed “three critical limitations in stream-based frameworks,” providing only final accuracy comparisons. This obscures the paper’s contribution and makes it difficult to assess the effectiveness of the proposed approach.
3. Ablations are insufficient: there is no systematic cross-study of K values, memory length, or training strategies (e.g., pure world model vs. conventional Warp+Fuse). Meanwhile, compared with other baseline frameworks, the proposed method exhibits a noticeable increase in computational overhead. More compute metrics, including FLOPs and memory usage, should be reported to provide a more comprehensive evaluation.

**Minor Weaknesses**

1. Figure 4 caption: “The results of TopStreamer are shown on the top, and the results of temporal baseline are shown on the bottom.” This appears to have mistakenly included another method’s name in this paper’s visualization caption.
2. Figure 1 caption: “Comparsion” is misspelled.
3. Several implementation details remain unclear (e.g., the exact cost functions and weights used for Hungarian matching when the fast and slow branches run in parallel, and whether sharing weights between the two branches during training exacerbates mode collapse).

[1] Yang Y, Luo Y, He B, et al. TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving[J]. arXiv preprint arXiv:2507.00709, 2025.

### Questions
Please see major weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the authors introduce a fast-slow dual system named FASTopoWM to solve the current models’ inability to effectively leverage temporal information. Although some stream-based temporal propagation methods are demonstrated to be effective, they still depend too much on historical queries, are vulnerable to pose estimation failures, and have insufficient temporal propagation. FASTopoWM solves these limitations by introducing the dual-system framework. It can simultaneously supervise both historical and new queries and prompt mutual benefits. Besides, they also introduce latent representations to further improve the slow system’s performance. FASTopoWM is verified on OpenLane-V2, and achieves state-of-the-art performance on both lane-segment detection and centerline detection.

### Strengths
- This work provides a much better way to incorporate historical information to facilitate lane topology reasoning, compared to StreamMapNet.
- FASTopoWM achieves SOTA performance on OpenLane-V2 with a 37.4 mAP (+3.8 compared to the previous SOTA baseline Topo2Seq) while maintaining a certainly acceptable latency (11.4)
- Ablations shown in Table 3 demonstrated various parts’ functions in FASTopoWM. The experiments are overall comprehensive.

### Weaknesses
- The paper's primary weakness is a disconnect between its terminology and the methods described. Certain concepts, like "world models" and the "fast-slow system," feel overstated.
- For example, the "world model" is a two-layer transformer that predicts features for the next timestep. This is a much simpler implementation than what is typically understood by the term.
- Similarly, the proposed "fast-slow system" primarily relies on the slow system's output. The fast system only acts as a fallback when "reliable pose information is missing or inaccurate" (Line 196), which diminishes its role compared to conventional dual-system architectures.

### Questions
This leads to a key question: What is the precise trigger for the fast system?
- How is "missing or inaccurate" pose information defined or measured?
- Is this fallback mechanism only used during initialization, or can it be triggered at any point during operation?

### Soundness
2

### Presentation
2

### Contribution
2
