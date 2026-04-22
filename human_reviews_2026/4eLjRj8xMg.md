# TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 8

## Abstract
Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates considerable improvements over state-of-the-art methods, achieving substantial performance gains of +3.0 mAP in lane segment perception and +1.7 OLS in centerline perception tasks. Our code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
1. This paper designs a comprehensive loss function that imposes consistency constraints on the temporal propagation of lane centerline and boundary line coordinates as well as their classification information, ensuring the stability of multidimensional attributes.
2 .DBPE: In the layer-by-layer forward propagation of the decoder, it resolves the conflict between the static nature of traditional positional encoding and the dynamic updating of reference points, significantly improving localization accuracy.

### Strengths
1. Introducing temporaling modeling into the lane segment perception task is valuable for stable map construction.
2. Experimental results show that on the OpenLane-V2 dataset, TopoStreamer achieves a 3.0% mAP improvement in lane segment perception and a 1.7% OLS improvement in centerline perception compared to the previous state-of-the-art (SOTA)

### Weaknesses
1. Many parts of this work claimed as core contributions have similar spirits to other autonomous driving works like StreamMapNet, and etc. Overall, this paper mainly solves several problems when such temporal modeling is applied to this specific lane segmentation prediction task.

2. Streaming Attribute Constraints seems to be an auxiliary loss that adds direct supervision to the results of stream queries. Compared to the previous transformation loss, it seems to have a similar effect but is implemented differently. It seems to be a trick.

3. The position of the ) in equation 4 seems to be wrong; inside LA it should be query, F_bev, and R.

4. The symbols in the paper are difficult to understand. Some symbols in the paper, from lines 167-181, should be clearly explained. Some symbol definitions are inconsistent; for example, some symbols are missing \mathbf{}.

5. The details of stream memory are not well described.

6. Some of the proposed methods, such as Streaming Attribute Constraints, DBPE, seem to be optimization tricks based on SDQ-MapNet, MapQR, and LaneSegNet. Could you emphasize the main motivation and contribution of this paper compared to previous methods?

6. There are too many hyperparameters related to the loss in the method. Although the ablation experiments and supplementary materials demonstrate effectiveness, it is challenging to fully ensure that each loss is valid and the hyperparameters are appropriately set.

### Questions
The questions have been included in the points of weakness.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes TopoStreamer, a temporal perception framework for lane segment topology reasoning in autonomous driving. It introduces three components: streaming attribute constraints to enforce temporal consistency, dynamic lane boundary positional encoding (PE) for improving spatial localization, and lane segment denoising to handle diverse temporal patterns. The method is evaluated on the OpenLane-V2 benchmark and reports +3.0% mAP improvement in lane segment perception and +1.7% OLS in centerline perception over previous methods.

### Strengths
- The paper proposes a temporal perception model that explicitly addresses temporal consistency and positional embedding issues in lane topology reasoning.
- The paper includes comparisons with multiple strong baselines and extensive quantitative ablations, demonstrating solid engineering effort. The proposed method achieves state-of-the-art (SOTA) results in both lane segment and centerline perception tasks.

### Weaknesses
- The main contributions — temporal propagation, positional encoding refinement, and query denoising — are all direct extensions of existing methods.The proposed modules are minor architectural tweaks rather than conceptual advances.
- The paper lacks a detailed analysis of computational complexity (e.g., FLOPs, parameter count) and its comparison with baseline methods.

### Questions
- In Table 1, why does TOPlsls of TopoStreamer remain lower than TopoLogic, despite your method being temporal and supposedly more consistent? Would integrating TopoLogic’s topology reasoning potentially improve your model’s results?
- TopoStreamer achieves only higher FPS than TopoNet, and is slower than other recent baselines. Does this imply that the proposed model trades off speed for accuracy?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the task of topological reasoning for autonomous driving and proposes a new model called TopoStreamer. The authors identify two key issues in current streaming-based learning: (1) consistent positional embedding and (2) temporal multiple attribute learning for lane segments. To tackle these challenges, they introduce dynamic explicit positional encoding, multiple streaming attribute constraints, and a lane segment denoising module.

Experimental results show that TopoStreamer significantly improves lane segment detection by better leveraging temporal information and also achieves notably higher accuracy in topology prediction. The authors also provide detailed visualizations to support their analysis. Overall, this is a solid and well-executed paper.

### Strengths
1. The authors rightly choose to incorporate temporal information into the lane segment topology reasoning task, as it helps address missed detections caused by occlusions and high-speed motion—a critical issue in topology reasoning.

2. The two challenges highlighted by the authors, consistent positional embedding and temporal multiple attribute learning for lane segments, are well-motivated and meaningful. Moreover, their proposal of a new metric to evaluate lane boundary classification accuracy is a valuable addition that further enhances the OpenLaneV2 benchmark.

3. The denoising module introduced in this work is also interesting and could serve as a useful foundation for future research.

### Weaknesses
1. I think the authors lack a detailed description of Section 3.4, LANE SEGMENT DENOISING. Why is denoising needed for predicting the topological relationships of these fine-grained lane segments? I hope the authors can further explain the motivation behind denoising, preferably with some simple examples.

2. Since introducing temporal information incurs significant computation, I hope the authors can provide the training and inference times.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3
