# Online Navigation Refinement: Achieving Lane-Level Guidance by Associating Standard-Definition and Online Perception Maps

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Lane-level navigation is critical for geographic information systems and navigation-based tasks, offering finer-grained guidance than road-level navigation by standard definition (SD) maps. However, it currently relies on expansive global HD maps that cannot adapt to dynamic road conditions. Recently, online perception (OP) maps have become research hotspots, providing real-time geometry as an alternative, but lack the global topology needed for navigation. To address these issues, Online Navigation Refinement (ONR), a new mission is introduced that refines SD-map-based road-level routes into accurate lane-level navigation by associating SD maps with OP maps. The map-to-map association to handle many-to-one lane-to-road mappings under two key challenges: (1) no public dataset provides lane-to-road correspondences; (2) severe misalignment from spatial fluctuations, semantic disparities, and OP map noise invalidates traditional map matching. For these challenges, We contribute: (1) Online map association dataset (OMA), the first ONR benchmark with 30K scenarios and 2.6M annotated lane vectors; (2) MAT, a transformer with path-aware attention to aligns topology despite spatial fluctuations and semantic disparities and spatial attention for integrates noisy OP features via global context; and (3) NR P-R, a metric evaluating geometric and semantic alignment. Experiments show that MAT outperforms existing methods at 34 ms latency, enabling low-cost and up-to-date lane-level navigation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new and important task for Online Perception Maps and SD map association. It first introduces Navigation Refinement P-R to evaluate the geometric and association accuracy, and then proposes a model named MAT to take centerlines, SDMap roads and boundary lines as input to predict road-lane association. 
In the MAT model, two modules - PA and SA are carefully designed to consider topological and spatial features learning. 
The paper also proposes the online map association dataset (OMA). Overall, this work focuses on a new problem of map association for low-cost and up-to-date lane-level navigation.

### Strengths
1. This paper proposes a new and important task for the online perception maps association, which is valuable for real-world applications in autonomous driving. Assigning prediction lanes to SD road elements is essential for building topological relationships between lanes and roads, further benefiting navigation and planning. Building a corresponding benchmark and evaluation metric fills the gap in the field of online map construction, as far as I know.
2. The evaluation metrics are carefully designed to cover various cases, which reflect the distance matching, connection assignment, and length accuracy ratio.
3. The contrastive experiments, ablation study and analysis are thorough, which demonstrates the effectiveness of each part of the model and post-process algorithm.

### Weaknesses
1. For the subfigure Fig 2.(3), there are two junction curve lines connecting two crossroads. The mapping of such transition
 lines is ambiguous due to their connection attributes. How to solve this problem when constructing the ground truth assignment labels?
2. As this paper also studies the noise of road elements and assignment with predicted lanes, a line of related works regarding the use of SDMaps and the noise problem for map perception should be included and discussed, such as SMERF [1],  TopoSD [2], P-MapNet [3] and etc.
3. We know there are inevitably some cases with the absence of the SDMap road elements or shifting road geometry,  caused by the errors of GPS localization or map construction.  We would like to know how these factors affect the association process and how the proposed metrics maintain robustness when dealing with imperfect SDMaps. In particular, this concern applies to cases where the generated online maps are geometrically or topologically inaccurate, even though they are otherwise precise.

[1] Luo K Z, Weng X, Wang Y, et al. Augmenting lane perception and topology understanding with standard definition navigation maps[C]//2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024: 4029-4035.
[2] Yang S, Jiang M, Fan Z, et al. Toposd: Topology-enhanced lane segment perception with sdmap prior[J]. arXiv preprint arXiv:2411.14751, 2024.
[3] Jiang Z, Zhu Z, Li P, et al. P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors[J]. arXiv preprint arXiv:2403.10521, 2024.

### Questions
1. In Figure 4, the caption states that “Label overlap in both paths is 67%. TP for T=50% but FP for T=95%.” I am wondering how the value of Label overlap is computed — is it defined as the ratio of correctly matched (or overlapping) edges to the total number of edges? How is an edge seen as the correct one?
2. Why does the order of road tokens or centerline tokens matter in the MAT attention modules?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Targeting the pain points regarding cost and real-time performance in lane-level navigation, this paper proposes a new task of Online Navigation Refinement (ONR). By associating Standard-Definition Maps (SD) with Online Perception Maps (OP), it upgrades road-level navigation to lane-level navigation, while simultaneously addressing the core issues that existing solutions rely on expensive HD maps and OP maps lack global topology.

### Strengths
1. It directly tackles the core pain points of existing lane-level navigation solutions. The proposed ONR task bridges SD and OP maps to achieve low-cost, up-to-date lane-level guidance, which is highly relevant to real-world needs in GIS and autonomous driving.
2. The paper provides a "dataset-model-metric" trinity solution to fill research gaps:OMA Dataset, MAT Model and NR P-R Metric.
3. The paper provided solid experimental validation.

### Weaknesses
1. The test set's OP map noise only comes from MapTRv2.  It does not evaluate MAT’s performance under other common OP noise types (e.g., severe lane occlusion by vehicles, sensor failure in heavy rain/fog), I think it should be evaluated to improve the model's robustness.
2. The test set uses OP maps generated solely by MapTRv2. If other OP map generators produce different noise patterns, whether MAT’s performance will degrade or not remains untested, i think the cross-generator generalization ability should be evaluated.

### Questions
1. The OMA dataset’s annotations are manually completed. Is there a plan to introduce semi-supervised/unsupervised annotation tools to reduce costs for future dataset expansion?
2. The metric aggregates results across 15 length intervals to avoid short-path bias. However, it does not consider road complexity (e.g., straight roads and curved roads). Could we adding a "complexity weight" to improve the metric’s fairness in evaluating model performance on diverse road types?

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
This work focuses on the task of Online Navigation Refinement, which aims to transform the road-level navigation derived from SD maps into precise lane-level navigation aligned with the online perception maps. Specifically, the authors introduce the Online Map Association Dataset (OMA), which is developed from nuScenes, and a corresponding transformer-based model for the real-time map association task. In addition, to measure the alignment of the paths and the precision of correspondence, the authors also proposed a new metric named Navigation Refinement P-R (NR P-R). MAT is verified on the OMA dataset and achieved rather low latency (34ms) while maintaining comparable performance improvements compared to previous methods.

### Strengths
- Although existing autonomous driving datasets, such as nuScenes and OpenLane-V2, provide fine-grained local lane geometry annotations, and OpenStreetMap provides road-level SD Map annotations, the mappings between them are less explored. The OMA dataset proposed by this work greatly mitigates this data gap.
- Path-aware attention is introduced to align the topologies under the distractions introduced by spatial fluctuations and semantic disparities. By forcing each token can appear only once in the path, cycles are prevented from happening and interactions are ensured to occur only between tokens within the same path.
- The authors also proposed a Spatial Attention module that enhances feature interactions at the instance level across a wider spatial scope.

### Weaknesses
- Figure 5 is not self-contained. The concepts of path-aware and spatial attention are hard to grasp from the visual alone, as the figure lacks sufficient illustrative descriptions or a detailed caption.
- Abbreviations are sometimes unclear or non-standard. For example, "Para." in Table 1 is ambiguous and should be explicitly defined (e.g., as "Parameters").
- The section on "attention with vector serialization" is underdeveloped. The explanation is too brief, lacking the detail needed for the reader to fully understand the proposed mechanism.

### Questions
- What’s the difference between MAT-T and MAT-L?

### Soundness
2

### Presentation
2

### Contribution
2
