# Online 3D Instance Segmentation at task-oriented granularity with Unposed Monocular Video

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
We present a real-time, task-oriented 3D instance segmentation framework for unposed monocular video, enabling embodied agents to task-adaptively perceive and interact with objects in open-world scenes. Unlike most previous bottom-up segmentation paradigm that segment before recognition, we adopt a task-oriented segmentation approach. Specifically, objects are decoupled within each frame using an open-vocabulary detector combined with a prompt-based 2D segmentation model, while the 3D underlying geometry of the scene is simultaneously being reconstructed using a modern dense SLAM system. Guided by the SLAM-derived pose graph, we selectively associate multi-view masks and reuse the dense correspondences provided by the SLAM system, incrementally converting them into geometric association scores with minimal additional computation. By incorporating semantic similarity and mutual exclusivity metrics, we design a priority-ordered mask clustering algorithm for efficient online multi-view mask matching and merging. Evaluations on open-vocabulary 3D instance segmentation benchmarks show that our method effectively mitigates the performance degradation of existing approaches when using dense SLAM reconstructions instead of depth-sensor point clouds. On the Replica dataset, using only unposed images, it even achieves results comparable to methods leveraging ground-truth depth and poses. Codes will be released upon acceptance of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a real-time framework for task-oriented 3D instance segmentation from unposed monocular video. Departing from traditional bottom-up segmentation, the method first detects task-relevant objects in 2D using an open-vocabulary detector and a prompt-based segmenter. These 2D masks are then efficiently fused into 3D instances by leveraging the pose graph and dense point correspondences from a modern monocular SLAM system (MASt3R-SLAM). A novel greedy clustering algorithm incorporates semantic and geometric cues for online multi-view mask association. The approach effectively mitigates the performance drop typically seen when using SLAM reconstructions instead of sensor depth, achieving competitive results on benchmarks and enabling open-vocabulary 3D segmentation in real-time.

### Strengths
1. The proposed method only requires unposed RGB stream as input, which is a practical setting.

2. The experiments are clear and sufficient.

### Weaknesses
1. The major concern of this paper is about efficiency. Inference latency should be reported for main experiments, which is essential for real-world SLAM and perception application.

2. Although this method can directly take in unposed RGB stream, it is more like a simple concatenation with Mast3R-SLAM and an online perception pipeline. The perception part cannot affect the SLAM process. It is expected that the semantic and instance information can help to refine the SLAM results.

### Questions
The inference speed should be reported in the experiments. Based on the rebuttal, I will consider to raise my score.

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
3

### Summary
This paper proposes an online, task-oriented 3D instance segmentation framework for pose-free monocular videos. It leverages MASt3R-SLAM for dense reconstruction and pixel-level correspondence, while the front-end uses YOLO-World + SAM2 to generate task-relevant instance masks in 2D. And greedy clustering, supplemented by cluster-level 3D BBox-IoU merging. The authors report open-vocabulary/class-agnostic results on ScanNet200 and Replica, and analyze the performance degradation of existing methods when using SLAM reconstruction input.

### Strengths
1.Online scenarios and task-oriented settings have application value.

2.The results and failure reasons of ScanNet200 and Replica are analyzed and discussed.

### Weaknesses
1.Some hyperparameters were fixed at ϵ=0.2, GAM=0.25, SSM=0.85, and inter-cluster IoU=0.1, but no discussion or experimental analysis was performed.

2.Table 1 and Table 2 do not have any FPS comparison with the baseline, which cannot prove that the method is faster with the same accuracy, faster but with acceptable accuracy, or faster and accurate.

3.Table 1 and Table 2 do not compare Pose & Depth for GT.

### Questions
1.Section3.2  “given a task-relevant open-category set C” and “non-overlapping with small-mask retention” how this part is handled is not introduced in the paper.

2.The paper’s overall pipeline—2D mask cross-frame aggregation → 3D instance construction → text-conditioned selection is highly consistent with the core idea in Open3DIS, while its online, incremental cross-view mask integration overlaps with the OnlineAnySeg paradigm. The only substantive architectural deviation seems to be that mask association is restricted to SLAM pose-graph edges and that a greedy 3D IoU merge is used merely as a fallback. Could you analyze in more detail how this edge-restricted association differs from the multi-view matching strategy in Open3DIS and from the streaming merging in OnlineAnySeg, and in which scenarios this design may actually help?

3.The main architectural deviation seems to be that association is restricted to SLAM pose-graph edges and that a greedy 3D IoU merge is used as a post-hoc repair step. Could you clarify in more detail:

(a) how this SLAM-edge-restricted association differs, in practice, from the multi-view matching in Open3DIS and the streaming mask integration in OnlineAnySeg;

(b) whether this restriction could cause missed matches for long-range or sparsely observed objects; and

(c) whether the greedy 3D IoU backup can consistently recover such missed associations, or if there are scenarios where it fails?

[1] P. Nguyen, T. D. Ngo, E. Kalogerakis, C. Gan, A. Tran, C. Pham, and K. Nguyen, “Open3DIS: Open-vocabulary 3D instance segmentation with 2D mask guidance,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 4018–4028.

[2] Y. Tang, J. Zhang, Y. Lan, Y. Guo, D. Dong, C. Zhu, and K. Xu, “OnlineAnySeg: Online zero-shot 3D segmentation by visual foundation model guided 2D mask merging,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 3676–3685.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a real-time, task-oriented 3D instance segmentation framework for unposed monocular video, designed to help embodied agents perceive and interact with objects in open-world environments. Unlike conventional bottom-up pipelines that segment before recognition, the method performs task-adaptive segmentation by combining an open-vocabulary detector with a prompt-based 2D segmenter, while reconstructing the scene’s 3D geometry online through a dense SLAM system. Leveraging the SLAM-derived pose graph, it efficiently associates and merges multi-view masks using geometric, semantic, and exclusivity cues through a greedy clustering algorithm. Experiments on open-vocabulary 3D instance segmentation benchmarks demonstrate strong performance, particularly on the Replica dataset, where the approach using only unposed images rivals methods that rely on ground-truth depth and poses.

### Strengths
1.This paper integrates a dense SLAM system with an online open-vocabulary instance segmentation framework, enabling 3D instance segmentation from only image sequences without requiring camera poses.

2. The method fuses and disentangles instance features across multiple frames, producing more stable and discriminative instance representations that lead to improved segmentation performance. 

3.Extensive experiments on multiple datasets demonstrate the effectiveness of the proposed framework.

### Weaknesses
1.Figure 2 is difficult to interpret. The example only shows a single instance mask, and the components Mask Association Criteria and Online Clustering merely display matrices and instances without clearly illustrating the underlying mechanisms. Moreover, the depiction of the Pose Graph is overly abstract and does not intuitively convey its relation to camera poses. The final stage of the figure is also unclear—how the framework simultaneously produces both instance and semantic outputs remains unexplained.

2.Figure 2 claims that the method can output semantic segmentation results, yet the experiments do not include evaluations on standard semantic segmentation benchmarks.

3.In Equation (1), the “valid constraint”  V is not sufficiently explained. It would be helpful to clarify how the valid pixel correspondences are determined or obtained.

4.In Table 1, the comparison lacks fairness and completeness. The method SAM3D has an online version, and several other online approaches—such as EmbodiedSAM—are not included in the comparison.

5.Line 295 (page 6) states that the access and update operations on the matrix have an algorithmic complexity of O(1). A more detailed analysis or justification of this claim would make the explanation more convincing.

### Questions
1.In the ablation study, the MEM module appears to have the most significant impact. The paper mentions that “without MEM, inevitable point‑matching noise in the SLAM system” leads to performance degradation. Since the SLAM component is based on an existing network such as MASt3R‑SLAM, it would be valuable to clarify whether this issue originates from limitations inherent to the SLAM method itself or from the integration with the proposed framework.

2.It would be helpful to discuss the modularity of the proposed system. Specifically, can each component be replaced with functionally equivalent models—such as substituting VGGT‑SLAM for the SLAM backbone or using YOLO‑UniOW instead of YOLO‑World—without substantial adaptation?

3.The method relies on 2D models such as YOLO‑World and SAM, without additional training on 3D data or geometric structures. It would therefore be useful to elaborate on whether the approach demonstrates strong zero‑shot generalization capabilities in 3D perception tasks.

### Soundness
3

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
4

### Summary
This paper introduces a real-time, task-oriented 3D instance segmentation system for unposed monocular videos. The methodology departs from conventional bottom-up segmentation, instead coupling an open-vocabulary 2D segmentation (YOLO-World + SAM2) with a dense SLAM system (MASt3R-SLAM) for online 3D reconstruction. The core contribution is an online, greedy clustering algorithm that reuses the SLAM's internal dense correspondences (via GAM) and combines them with semantic similarity (SSM) and mutual exclusivity (MEM) metrics for multi-view mask merging.Evaluations on the ScanNet200 and Replica datasets show that the method is more robust to SLAM-induced artifacts compared to existing baselines, and maintains competitive performance without ground-truth poses or depth.

### Strengths
1. **Overall Strength**: This paper introduces a real-time, task-oriented 3D instance segmentation system for unposed monocular videos. The methodology departs from conventional bottom-up segmentation, instead coupling an open-vocabulary 2D segmentation (YOLO-World + SAM2) with a dense SLAM system (MASt3R-SLAM) for online 3D reconstruction. The core contribution is an online, greedy clustering algorithm that reuses the SLAM's internal dense correspondences (via GAM) and combines them with semantic similarity (SSM) and mutual exclusivity (MEM) metrics for multi-view mask merging. Evaluations on the ScanNet200 and Replica datasets show that the method is more robust to SLAM-induced artifacts compared to existing baselines, and maintains competitive performance without ground-truth poses or depth.
2. **Novel and clever Methodological Design**: The core technical idea is novel and clever. Reusing the internal, pose-independent dense correspondences from the SLAM system for mask association—rather than relying on the final, noisy 3D point cloud—is a sound strategy to bypass output-level noise.
3. **Well-Motivated Components**: The system's design is well-supported by empirical evidence. The ablation study (Table 3) provides clear justification for the three-metric association (GAM, SSM, MEM) and demonstrates that each component is necessary for the final performance.

### Weaknesses
1. **Fragile Pipeline and Critical Dependency on Upstream Modules**: The method entirely rely on MASt3R-SLAM for 3D reconstruction, and the segmentation performance heavily depend on the perfect functioning of its upstream components (MASt3R-SLAM, YOLO-World and SAM2). The paper fails to analyze how the system performs when the SLAM itself tracking is lost or provides highly noisy correspondences (e.g., in low-texture areas or during fast motion).The paper The paper fails to analyze the system's performance when either of these upstream modules fails or provides noisy input (e.g., SLAM tracking loss, low-texture scenes, 2D detection failures).
2. **Low Absolute Performance**: While the method is relatively more robust than failing baselines, its absolute performance (e.g., 6.5 AP50 on ScanNet200) is extremely low. This indicates that the proposed solution is only marginally functional and far from practical.
3. **Lack of Error Propagation Analysis**: The method is greedy and online. The paper does not discuss or analyze error accumulation. If an incorrect merge is made in an early frame, it's unclear if this error permanently poisons the instance cluster or if the system has any mechanism to recover.

### Questions
1. **Error Propagation**: Can your system recover from an incorrect merge made in an early frame (e.g., due to a 2D segmentation error), or do these errors propagate permanently?
2. **Failure Analysis**: The absolute performance is relatively low. Can you provide a more detailed failure analysis? For instance, what is the AP for small vs. large objects, or for head vs. tail classes? This would help clarify if the low score is due to SLAM/segmentation noise (as hypothesized) or systemic merging failures.
3. **Missing Demo Video**: Appendix A.2 (Reproducibility Statement) mentions a demo video available at an anonymous URL. Upon visiting this URL, I was only able to find a Readme file, and the demo video appears to be missing. Could the authors please verify the link and make the video available for review?
4. **Missing LLM Usage Statement**: ICLR submissions require a statement regarding the use of Large Language Models (LLMs). This paper appears to be missing this statement. Could the authors please clarify if LLMs were used in any part of the research or writing process?
5. **Minor Typo**: A minor point on presentation: In Equation (3), the formatting of the second norm $||s_m^j||$ appears to be malformed in the text.

I am open to raising my score if my concerns can be well addressed in the authors' rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2
