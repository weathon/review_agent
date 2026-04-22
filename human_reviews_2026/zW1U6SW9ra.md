# Compressed Map Priors for 3D Perception

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Human drivers rarely travel where no person has gone before.
After all, thousands of drivers use busy city roads every day, and only one can claim to be the first.
The same holds for autonomous computer vision systems.
The vast majority of the deployment area of an autonomous vision system will have been visited before.
Yet, most autonomous vehicle vision systems act as if they are encountering each location for the first time.
In this work, we present Compressed Map Priors, a simple but effective framework to learn spatial priors from historic traversals.
The map priors use a binarized hashmap that requires only 32 KB/sq km, a 20x reduction compared to storing features densely.
Compressed Map Priors easily integrate into leading 3D perception systems at little to no extra computational costs, and lead to a significant and consistent improvement in 3D object detection on the nuScenes dataset across several architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Compressed Map Priors, a framework that integrates spatial priors from historical traversals into multi-view 3D object detection for autonomous driving. CMP uses binary-quantized hash encoding to achieve extreme memory efficiency. This framework is compatible with both dense grid-based and transformer-based architectures. Experiments on nuScenes show consistent gains, outperforming traditional map priors and learned priors.

### Strengths
1. CMP’s binary hash embeddings achieve 32 KB/km² storage, 20× better than dense/GT Map, addressing on-board memory constraints. Its <2% runtime overhead also ensures real-time feasibility.
2. CMP adapts to both dense and transformer architectures with minimal modifications.

### Weaknesses
1. It is not a reasonable configuration to use learned prior map for 3D object detection. If a parked vehicle appears in both the training set and test set, it will be recorded in the prior map during training, making it easier to detect in the testing stage. However, this is meaningless for real-world autonomous driving, as such a vehicle may drive away at any time. Previous methods have not adopted a similar configuration. BEVMap utilizes the map annotations that do not contain object information to improve the accuracy of object detection. NMP leverages learned map priors, but these priors are used for the task of map segmentation.
2. The authors should provide more comparisons between CMP, NMP, and SOTA map segmentation methods in terms of map segmentation accuracy. The accuracy of "Val-only" and "Both" scenes can be presented separately.

### Questions
1. Why does the memory usage of GT maps much larger than that of CMP in Table 2? Is it possible to compress the GT map in the way of CMP?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Compressed Map Priors (CMP)—a multi-resolution spatial hash embedding that stores binarized per-cell features as a persistent prior, fused into standard camera-only 3D detectors. CMP is trained end-to-end (STE for binarization), adds negligible latency (~3%), and claims ~32 KB/km² storage (≈20× smaller than a dense alternative). On nuScenes, CMP consistently improves NDS/mAP across BEVDet, BEVFormer, PETR, and outperforms classical rasterized GT-map priors under a far smaller memory budget.

### Strengths
Pros:
1. Consistent accuracy gains on nuScenes across three diverse baselines; largest relative lift on BEV-style models.
2. Simple, detector-agnostic add-on: clean fusion blocks for BEV and transformer stacks (concat+Conv vs. cross-attention). 
3. Thoughtful ablations: traversal count sensitivity and distance-band analysis support the “priors help when signal is weak/far”.

### Weaknesses
Cons:
1. While the method is shown across multiple camera-only 3D detectors, several stronger, recent baselines (e.g., StreamPETR, BEVNext) are missing. Without results on higher baselines, it’s hard to judge headroom and true practical impact.
2. The approach assumes AVs mostly drive in previously seen areas where priors exist; the conclusion also notes retraining/retuning is needed for new environments, which reduces universality.
3. No stress tests for prior dropout or corruption. In deployment, parts of the map prior can disappear or become stale. It’s unclear whether the system gracefully falls back to a normal 3D detector.
4. Train–val spatial leakage risk: Although Appendix details a 50 m overlap rule and a “val-only/both” partition, the main-paper results don’t report metrics conditioned on overlap. If CMP’s benefit concentrates where training traversals exist, the headline numbers could overstate generalization.
Ask to add: (a) per-split results: val-only vs. both; (b) performance vs. exact nearest-traversal distance (continuous, not bins).
5. Stale priors / change management :CMP encourages persistence. Lane closures, construction, snow, cones, parked trucks introduce map drift. Random patch masking helps, but it doesn’t simulate systematic stale bias.

### Questions
1. I am curious how you gate trust in the prior when sensor evidence contradicts it (e.g., temporary barriers). Is there a learned reliability head or confidence calibration for X_prior?
2. What is the failure mode when the car localizes off by one grid cell at the finest 1 m resolution? Any qualitative examples? 
3. I am curious whether cross-attention fusion could over-attend to priors in sparse-query detectors. Did you observe reduced reliance when priors are masked at test time?
4. How much of the total memory footprint (per km²) includes the MLP projector and positional embeddings vs. the hash tables alone? Current reporting appears to count only the tables.
5. What is the performance about city-transfer: train priors in Boston only and evaluate on Singapore val-only segments; do CMP gains persist? 
6. For fairness, BEVMap uses GT annotations (oracle). Why does CMP beat it on detection? Is it because CMP captures non-semantic cues (texture, curb geometry) that the 6-class raster misses? Could a richer GT (curbs, stop lines, sidewalks) close the gap?

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
The paper introduces Compressed Map Priors (CMP), a data-driven approach for incorporating spatial priors into 3D perception models for autonomous driving. CMP leverages a multi-resolution hash-based embedding scheme with binary quantization to efficiently encode prior knowledge from repeated traversals of the same environment. The compressed prior is fused with standard 3D detection backbones and trained end-to-end with downstream detection losses. Extensive experiments on the nuScenes dataset show that CMP provides consistent detection gains, substantial memory savings (20x reduction), and minimal computational overhead, outperforming both traditional map priors and recent learned priors.

### Strengths
1. The idea of efficiently leveraging historical traversals to inform 3D perception systems addresses a fundamental inefficiency in current approaches, which often treat every scene as novel despite repeated exposure.
2. The architecture is modular and demonstrated to work across several leading baselines (BEVDet, BEVFormer, PETR), with minimal intrusion.
Thorough experiments: CMP is compared quantitatively against strong baselines, including modern learned and traditional map priors, across multiple architectures and uses appropriate metrics .

### Weaknesses
1. The method is explicitly described as being beneficial in well-traversed environments (Section 6), but its limitations in places with limited or no prior coverage are only superficially addressed via random patch masking.  No rigorous experiments or quantitative breakdowns for novel/unseen areas are provided, raising concerns for real-world deployment.
2. Though BEVFormer, PETR, and BEVDet are credible representatives, modern BEV occupancy grid predictors (such as OccFeat or PointBeV, referenced in the related web context) and object-centric methods (e.g., OC-SOP) are not included as comparative baselines in either main results or ablations, potentially missing stronger competitors or alternative design philosophies.

### Questions
1. Can the authors provide more rigorous or quantitative evaluations of CMP when applied to environments with entirely novel scenes (not traversed during training), explicitly reporting both absolute and relative performance drops?
2. Please clarify the impact of the random patch masking augmentation.  Have ablation studies been performed where this is turned off?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a framework for incorporating historical context into perception models with Compressed Map Priors, which employs a multi-resolution hash-based spatial encoding with binary quantization to efficiently store and retrieve prior spatial information.

### Strengths
1. The map prior encoding method proposed in this paper offers storage advantages compared to previous approaches.
2. The proposed method enables end-to-end optimization of map priors and perception tasks.

### Weaknesses
1. The proposed method was only tested on a single dataset and lacks validation on other mainstream datasets, such as KITTI, Waymo, Argoverse2, etc.
2. The experimental section compares against outdated methods and lacks comparisons with the latest state-of-the-art approaches.
3. The proposed method is limited to datasets where the training and testing data have overlapping areas on the map, making it difficult to apply in real-world open scenarios.
4. The proposed model contains numerous hyperparameters, and ablation studies are lacking for some of them.

### Questions
1. Please refer to the Weaknesses section.
2. Is the proposed method applicable to other 3D perception tasks such as 3D segmentation and tracking?

### Soundness
2

### Presentation
2

### Contribution
2
