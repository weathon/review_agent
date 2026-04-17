# History-Aware Transformation of ReID Features for Multiple Object Tracking

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
In Multiple Object Tracking (MOT), Re-identification (ReID) features are widely employed as a powerful cue for object association. 
However, they are often wielded as a one-size-fits-all hammer, applied uniformly across all videos through simple similarity metrics. We argue that this overlooks a fundamental truth: MOT is not a general retrieval problem, but a context-specific task of discriminating targets within a single video. To this end, we advocate for the adjustment of visual features based on the context specific to each video sequence for better adaptation. In this paper, we propose a history-aware feature transformation method that dynamically crafts a more discriminative subspace tailored to each video's unique sample distribution. Specifically, we treat the historical features of established trajectories as context and employ a tailored Fisher Linear Discriminant (FLD) to project the raw ReID features into a sequence-specific representation space. Extensive experiments demonstrate that our training-free method dramatically enhances the discriminative power of features from diverse ReID backbones, resulting in marked and consistent gains in tracking accuracy. Our findings provide compelling evidence that MOT inherently favors context-specific representation over the direct application of generic ReID features. We hope our work inspires the community to move beyond the naive application of ReID features and towards a deeper exploration of their purposeful customization for MOT. Our code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To achieve more robust ReID-based matching in MOT, the authors propose HAT to make ReID features more discriminative. Specifically, FLD transformation is applied to shrink intra-trajectory feature distance and enlarge inter-trajectory feature distance. Meanwhile, feature centroid update strategy is designed to prioritize features from recent observations.

### Strengths
1) Application of FLD transformation for ReID feature can be directly applied to other TBD trackers, and inspires more in-depth studies on other transformation techniques to improve ReID feature discriminativeness.
2) The proposed feature centroid update strategy provides a alternative for the widely-used EMA strategy.
3) the manuscript is well-organized with clear expressions.

### Weaknesses
1) Although the authors mention the proposed tracker achieves 22.7 FPS on DanceTrack, the impact on computational speed from the proposed modules remains unclear. It's recommended to include speed comparison with the baseline method.
2) The proposed ReID feature transformation method is the major contribution of the paper. As far as I'm concerned, there are other works also making such efforts, e.g. TOPICTrack. Comparisons or discussions on such related work are encouraged, if possible.
3) Exploiting ReID cues is not new for MOT, this paper essentially seems incremental work, so the noveltiy is somewhat limited. If a well enough ReID model emerges, there is no need to optimize and improve it.
4) The performance imprvements are limited on MOT17. Additionally, What's about its results on MOT20?
5) The introduced extra hyper-parameters brings the difficulty for generalization.
6) All experiments are conducted for SORT-like methods, Can the proposed HAT benefical for E2E MOT?

### Questions
See Weaknesses

### Soundness
3

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
4

### Summary
this paper investigates the visual similarity measurements for multi-object tracking (MOT). specifically, the authors argue that the default approach of re-ID feature distance might be sub-optimal for the tracking task, since re-ID aims to distinguish all potential targets, whereas tracking aims to differentiate similar targets within the same video sequence. as a solution, the paper treats the historical trajectory features as conditions and adopts Fisher Linear Discriminant (FLD) to project the original re-ID feature into a different feature space. on multiple benchmarks, the proposed feature achieves improvements over the baseline.

### Strengths
+ the observation that the re-ID feature might be sub-optimal for tracking is very interesting. 
+ the feature dimension reduction method is easy to implement and introduces an additional 'degree-of-freedom' for test-time optimization to the feature-distance-based visual similarity calculation. 
+ consistent improvements over the baseline across multiple benchmarks.

### Weaknesses
- no new insights & high similarity to undiscussed existing work [r1]. [r1] also points out that re-ID feature might not be the optimal choice for tracking, and conducted experiments to verify the mismatch. given the specific points listed below, it is very concerning that this paper does not compare against it.
  - motivation - reID feature & matching scope in tracking: Fig. 3 in [r1] vs. Fig. 1 in this paper
  - issue in existing methods - similarity estimation & mismatch between reID and tracking: section III in [r1] vs. section 2.1-2.2 in this paper 
  - preliminary verification - matching performance improvements from the proposed metric: Fig. 4 in [r1] vs. Fig. 2b in this paper 
- ablation & variant study.
  - in Table 5, the main ablation for the proposed dimension reduction with FLD should be a variant where no feature dimension reduction is conducted. is it the $d=d'$ variant? are variants #1 and #5 the same as the baseline? or do they still include the Temporal-Shifted Trajectory Centroid?
  - in Table 6, the main ablation should be the setting where no Temporal-Shifted Trajectory Centroid is incorporated, not the baseline tracker. is it the $T=inf$ line?
  - in Table 7, the main ablation should be the version where only the traditional re-ID distance (NO projection) is considered, which should be $\alpha=0$. the reviewer cannot find this variant. 
- this method is not applicable to end-to-end tracking approaches
- no comparison with SOTA methods such as MOTRv2 (CVPR 2023) and ColTrack (ICCV 2023)

[R1]. Hou, Yunzhong, Zhongdao Wang, Shengjin Wang, and Liang Zheng. "Adaptive affinity for associations in multi-target multi-camera tracking." IEEE Transactions on Image Processing 31 (2021): 612-622.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a history-aware feature transformation method that dynamically creates more discriminative subspaces using Fisher Linear Discrimination (FLD), tailored to each video's unique sample distribution.
The proposed method treats the history features of established trajectories as context and projects adjusted raw ReID features into a video-specific representation space.
The effectiveness of the proposed method is discussed by comparing it with several tracking methods using multiple datasets.
The paper states that the code will be made publicly available.

### Strengths
- The paper is easy to understand.
- The proposed method is a combination of elementary techniques, making it straightforward to comprehend.

### Weaknesses
- Insufficient experiments
  - Generally, tracking methods are comprehensively compared using multiple metrics like IDF1 and MOTA.
  - Specifically, these include IDF1, IDP, IDR, Recall, Precision, FP, FN, IDs, FM, MOTA, IDt, IDa, IDm, etc. This paper evaluates only a very limited subset of these metrics.  

  - Furthermore, comparisons with transformer-based methods like MOTR, MOTRv2, and their variants are absent. It remains unclear whether the proposed method outperforms approaches like MOTRv2, especially on datasets such as DanceTrack.
  
  - Furthermore, the proposed method is not compared against more general methods like MOT20 or MOT17. 
  - Consequently, it is difficult to assess the effectiveness of the proposed method.

- While the proposed method states it “requires no training,” FLD itself is a form of machine learning. Moreover, the proposed method is a combination of rudimentary techniques, lacking technical depth.

### Questions
- Why was it not compared against more general methods like MOTA or IDF1?
- Why was it not compared against more sophisticated transformer-based methods like MOTR or MOTRv2?
- Why was it not compared against more general methods like MOT20 or MOT17?

### Soundness
3

### Presentation
2

### Contribution
2
