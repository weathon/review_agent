# Query-aware Hub Prototype Learning for Few-Shot 3D Point Cloud Semantic Segmentation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Few-shot 3D point cloud semantic segmentation (FS-3DSeg) aims to segment novel classes with only a few labeled samples. However, existing metric-based prototype learning methods generate prototypes solely from the support set. This often results in prototype bias, where prototypes overfit support-specific characteristics and fail to generalize to the query distribution, especially in the presence of distribution shifts, which leads to degraded segmentation performance. Although some works make efforts to refine or align these prototypes with queries, prototype bias remains poorly addressed, as the initial prototypes have already deviated significantly from the queries. To address this issue, we propose a novel Query-aware Hub Prototype (QHP) learning method that explicitly models semantic correlations between support and query sets. Specifically, we propose a Hub Prototype Generation (HPG) module that constructs a bipartite graph connecting query and support points, identifies frequently linked support hubs, and generates query-relevant prototypes that better capture cross-set semantics. To further mitigate the influence of bad hubs and ambiguous prototypes near class boundaries, we introduce a Prototype Distribution Optimization (PDO) module, which employs a purity-reweighted contrastive loss to refine prototype representations by pulling bad hubs and outlier prototypes closer to their corresponding class centers.  Extensive experiments on S3DIS and ScanNet demonstrate that QHP achieves substantial performance gains over state-of-the-art methods, effectively narrowing the semantic gap between prototypes and query sets in FS-3DSeg.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes Query-aware Hub Prototype learning for few-shot 3D point cloud semantic segmentation (FS-3DSeg) to address prototype bias. The framework includes two main components: (1) a Hub Prototype Generation (HPG) module that constructs a bipartite graph between support and query points, identifies frequently linked hub points, and clusters them into query-relevant prototypes; and (2) a Prototype Distribution Optimization module that refines bad hubs with a purity-reweighted contrastive loss. Experiments on S3DIS and ScanNet show consistent improvements over other baselines.

### Strengths
1. The paper is clearly written and easy to follow, with strong motivation and clear figures to explain the technical details.
2. Introducing query-aware hub prototypes is a novel perspective that leverages the hubness phenomenon to address the prototype bias for FS-3DSeg.
3. The method achieves superior performance over other baseline methods, with ablation studies validating each module’s contribution.

### Weaknesses
1. In Hub Point Mining (line 233-235), query points are used as centers, but in Bad Hub Selection (line 254-256), query and support points are merged. The rationale for this inconsistency is unclear.
2. For the Purity-reweighted contrastive loss (Eq. 7), the weighting scheme suggests the weight for bad hubs is smaller than the fixed weight 1 for foreground prototypes while line 274-277 writes the bad hubs need stronger guidance for the alignment. The smaller weights do not align with the motivations for “stronger guidance” for bad hubs.
3. Missing related work. Recent works on 3D few-shot learning (e.g., Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model, CVPR 2025) are highly relevant and should be cited.
4. Some grammar errors (e.g., “After that, we identify all potential bad hubs within H, which with stronger connections to center points belonging to different classes,” at line 257).

### Questions
Please refer to the weakness part and address the concerns there.

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
4

### Summary
The paper proposes a Query-aware Hub Prototype (QHP) approach for few-shot 3D point-cloud segmentation. It introduces (i) Hub Prototype Generation (HPG), selecting “hub” support points via neighborhood popularity, then performing local clustering to form multiple prototypes per class while interacting with query features; and (ii) Prototype Distribution Optimization (PDO), identifying “bad hubs” by purity and applying a purity-weighted contrastive loss to pull them toward class centers.

### Strengths
The motivation for addressing prototype bias from using only support features is interesting. 

The experimental comparison is sufficient. 

Stepwise ablations (Baseline → +HPG → +PDO) indicate incremental improvements

### Weaknesses
1. The abstract and introduction state that “existing metric-based prototype learning methods generate prototypes solely from the support set, without considering their relevance to query data.” This is not generally accurate. A substantial body of transductive / query-guided work explicitly leverages unlabeled query features to refine or align prototypes with the query distribution, e.g., support–query alignment and query-aware refinement [1, 2, 3]. Because this statement underpins the paper’s motivation and novelty positioning, it weakens the contribution as written.

2. The claimed benefits of hub selection and purity-weighted correction are motivated empirically. There is no analysis of hub statistics, misalignment probability, or even simple bounds showing when purity-weighted attraction is preferable to standard contrastive objectives.

3. Why do you prioritize pulling low-purity (“bad-hub”) / outlier prototypes toward the class center instead of the complementary strategy of reinforcing high-purity (“good-hub”) prototypes toward positives? What theoretical or empirical evidence shows that focusing on repairing bad hubs yields better decision boundaries than amplifying good hubs?

4. The improvements of most settings in Tables 1 and 2 are quite marginal. In Table 6, the authors perform a comparison between PC loss and contrastive loss. Do you use the optimal hyperparameter (e.g., $\lambda$) for the standard contrastive loss? The performance difference between PC loss and contrastive loss is marginal, and your Fig. 5 (c) shows that different $\lambda$ have a critical influence on the performance. Will using a better hyperparameter for contrastive loss lead to better performance? 

5. Figure 5 shows that the performance of the proposed method is highly sensitive to the hyperparameters, $k$, $\eta$, $\lambda$, and $\gamma$. It’s unclear how robust the method is for a new dataset without re-tuning.

6. What is the performance when using more prototypes than 100? 



[1]. Wang K, Liew J H, Zou Y, et al. Panet: Few-shot image semantic segmentation with prototype alignment[C]//proceedings of the IEEE/CVF international conference on computer vision. 2019: 9197-9206.

[2]. D. Hu, S. Chen, H. Yang and G. Wang, "Query-Guided Support Prototypes for Few-Shot 3D Indoor Segmentation," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 6, pp. 4202-4213, June 2024. 

[3]. Ning Z, Tian Z, Lu G, et al. Boosting few-shot 3d point cloud segmentation via query-guided enhancement[C]//Proceedings of the 31st ACM international conference on multimedia. 2023: 1895-1904.

### Questions
see my weakness.

### Soundness
2

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
This paper introduces a novel method for few-shot 3D point cloud semantic segmentation (FS-3DSeg) by proposing a Query-aware Hub Prototype (QHP) learning framework. The core idea is to mitigate prototype bias in metric-based few-shot methods by selecting query-relevant hub points from the support set. Two main modules are proposed: (1) Hub Prototype Generation (HPG), which constructs a bipartite graph between support and query sets to identify frequently-linked support “hub” points and generate prototypes via local clustering; and (2) Prototype Distribution Optimization (PDO), which uses a purity-reweighted contrastive loss to refine bad prototypes and enhance class compactness. Extensive experiments on S3DIS and ScanNet show improved performance over several baselines, supported by ablation studies and qualitative comparisons.

### Strengths
- The introduction of “hubness” for prototype generation in FS-3DSeg is novel and provides a new perspective on addressing support-query misalignment.
- Extensive experiments: Evaluations on S3DIS and ScanNet across 1-shot and 5-shot settings are comprehensive. Ablation studies and parameter sensitivity analyses support the effectiveness of the proposed modules.
- Clear writing and figures: The paper is well written, and figures (e.g., Figure 1, Figure 2, Figure 3) effectively illustrate the key concepts.

### Weaknesses
- Limited comparisons to recent or stronger baselines: The paper lacks comparisons with more recent state-of-the-art methods beyond COSeg and QGE/QGPA. Recent approaches that incorporate transformer-based meta learners, distillation, or prompt-based adaptation are not discussed or evaluated.
- Unclear generalization and robustness of hub mining: The “hubness” concept is somewhat heuristic and heavily relies on k-NN and purity thresholds. There is limited discussion or analysis on whether hub mining is robust under significant domain shift, class imbalance, or in real-world open-set scenarios.
- Efficiency claims are underexplored: While FLOPs and inference time are briefly compared, the computational cost of bipartite graph construction and clustering in large-scale 5-way settings is not analyzed. It is unclear whether the method truly scales well.
- No failure cases shown: All qualitative visualizations emphasize improvement. It would be helpful to include cases where the method fails or introduces errors to better understand its limitations.
- Motivation example could be more intuitive: Although prototype bias is well described in text, there is no concrete visual example showing how traditional support-only prototypes fail on specific query samples.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses prototype bias in few-shot 3D point cloud semantic segmentation by introducing a Query-aware Hub Prototype (QHP) learning framework. QHP constructs class prototypes using high-frequency hub points that exhibit the highest similarity to the query point clouds. To further enhance prototype quality, the proposed Prototype Distribution Optimization (PDO) module identifies potential bad hubs, and then adopt a Purity-reweighted Contrastive (PC) loss to suppress these bad hubs and optimize the prototype distribution.

### Strengths
1. It is interesting to introduce the concept of hubs into few-shot 3D point cloud semantic segmentation. By selecting the parts of the support point cloud closest to the query to generate prototypes, the resulting prototypes better align with the distribution of the query point cloud.

2. The experimental evaluation is sufficiently comprehensive and convincingly supports the effectiveness of the proposed QHP.

### Weaknesses
1. The experiment in Table 5 suggests that QHP is relatively sensitive to hyperparameter choices, raising concerns about its generalization across different datasets. This sensitivity may also explain why the performance gains on ScanNet are less pronounced than those on S3DIS. To address this issue, please verify whether substantially different hyperparameters are required for ScanNet.

2. The temperature parameter in Equation 8 has not been subjected to ablation analysis, despite being a critical factor influencing the effectiveness of contrastive learning.

### Questions
I have concerns regarding the weakness, particularly the generalization of QHP. If these concerns are adequately addressed, I would be happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3
