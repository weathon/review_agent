# TAPTRv3: Spatial and Temporal Context Foster Robust Tracking of Any Point in Long Video

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
In this paper, built upon TAPTRv2, we present TAPTRv3. TAPTRv3 improves TAPTRv2 by addressing its shortage in querying high quality features from long videos, where the target tracking points normally undergo increasing variation over time. In TAPTRv3, we propose to utilize both spatial and temporal context to bring better feature querying along the spatial and temporal dimensions for more robust tracking in long videos. For better spatial feature querying, we identify that off-the-shelf attention mechanisms struggle with point-level tasks and present Context-aware Cross-Attention (CCA). CCA introduces spatial context into the attention mechanism to enhance the quality of attention scores when querying image features. For better temporal feature querying, we introduce Visibility-aware Long-Temporal Attention (VLTA), which conducts temporal attention over all past frames while considering their corresponding visibilities. This effectively addresses the feature drifting problem in TAPTRv2 caused by its RNN-like long-term modeling. TAPTRv3 surpasses TAPTRv2 by a large margin on most of the challenging datasets and obtains state-of-the-art performance. Even when compared with methods trained on large-scale extra internal data, TAPTRv3 still demonstrates superiority.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an enhanced version of TAPTRv2, a DETR-like framework for point tracking. The authors aim to mitigate the feature drifting issue that occurs in long video sequences by introducing several key modifications: (i) a patch-level similarity computation for refining attention weights, (ii) feature anchoring based on the initial frames, and (iii) aggregation of historical feature changes. Experimental results demonstrate consistent and substantial performance gains over the baseline.

### Strengths
The paper provides an updated and improved version of a well-recognized community benchmark in point tracking. The proposed enhancements, such as patch-level similarity computation in attention and initial feature anchoring, are methodically designed and empirically validated, leading to stable and meaningful improvements across benchmarks.

### Weaknesses
While the proposed techniques yield consistent quantitative improvements, the overall methodological novelty appears limited. Patch-level feature similarity computation has been explored in prior works such as Context-PIPs and Track-On. Similarly, leveraging initial-frame features and historical feature fusion are well-established practices in tracking literature, making it difficult to identify a distinct conceptual innovation in the paper.

Minor Issues:
- Figure 3 is not particularly illustrative. I recommend replacing it with a better qualitative example to better demonstrate the improvements.
- A visualization of point trajectories for multiple tracked points using a well-received benchmark video would be highly valuable for evaluating qualitative performance and motion consistency.

### Questions
Please see weakness

### Soundness
4

### Presentation
3

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
The authors build upon point tracking framework of TAPTRv2, where they propose an improved version called TAPTRv3 that achieves better performance on long-term scenarios. In the proposed tracking method, they introduce two main contributions where (1) context-aware cross-attention (CCA), and (2) visibility-aware long-temporal attention (VLTA) are the key components of this paper. To validate this, experimental evaluations including large-scale benchmarks of TAP-Vid and RoboTAP where the proposed algorithm shows improvements over the previous methods.

### Strengths
- The formulation for the proposed method is simple and reasonable, and contributions claimed by the authors are straightforward and clear.

- The motivation of the proposed method is clear, where failure cases of TAPTRv2 including scenarios with long-term temporal drift and scene cuts are well-addressed and quantified.

- The authors performed ablation experiments that validate the effectiveness of each proposed component. The results in Table 2 show that each proposed module (VLTA, CCA, etc.) adds to the gains in empirical performance.

### Weaknesses
- The main experimental comparisons are performed on the subsets of TAP-Vid benchmark, and albeit them being oriented for long-term tasks compared to previous benchmarks such as DAVIS, they seem still short in terms of temporal length. Datasets such as PointOddyssey [a] contain longer sequences and is more oriented for evaluating the proposed method. 

  - [a] Yang et al., PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking., ICCV 2023.

- Although the proposed framework somehow works by detecting scene cuts through the external library, it is not a trainable module with questionable generalizability issues. Since the main objective of the proposed framework is to deal with long-term videos, the overall framework should systematically include such algorithm in a integrated and optimized manner. 

- Although the main objective of the proposed method is to tackle long-term scenarios well, Table 13 in the supplementary material shows slightly lower performance on short-term scenarios compared to other methods. Does this imply that the proposed CCA and VLTA modules hinder precise localization in shorter videos? Are there experimental analysis for the reason for the performance degradations?

### Questions
Please refer to the weaknesses section. Although the proposed scheme seems effective, limitations in the evaluations and performance degradations should be addressed.

### Soundness
3

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
4

### Summary
This paper proposes TAPTRv2, a point tracking method. The core of this paper lies in presenting two corresponding operations: Context-aware Cross-Attention (CCA) and Visibility-aware Long-Temporal Attention (VLTA), which are designed to improve the quality of spatial cross-attention and the effectiveness of long-term feature updates. Meanwhile, to address scene switches, this paper proposes an auto-triggered global matching mechanism, which enhances tracking stability when scene switches occur.

### Strengths
- This paper focuses on improvements such as enhancing context-aware capabilities and modifying attention weights using visibility, which are highly intuitive and also suitable for video tasks.
- This paper achieves good performance on multiple datasets.
- The ablation experiments for the newly proposed modules in this paper are conducted in detail.

### Weaknesses
- The key improvement of this paper lies in proposing two types of attention mechanisms to replace previous RNN-like methods. However, there are now many improved RNN-like neural networks, such as readily applicable Mamba and RWKV. The paper lacks sufficient explanation for these methods; could improved recurrent structures also alleviate the challenges in long-context modeling?
- In visibility-aware attention, the authors modify attention weights using predicted visibility. While this operation is shown to be beneficial in results, another scenario is that errors in visibility prediction may in turn lead to performance degradation. The authors still need to explain why this operation can be guaranteed to have a positive effect.
- The implementation of CCA shares the same core idea as deformable attention.  And the paper in my opinion is more about combining and improving existing technologies.
- There are some ambiguities in the description of the method, such as the implementation of APU and the rationale behind such operations.
- The main reason for the acceleration in this paper is the use of a smaller backbone. However, could it be that the backbone inherently has little impact on this task? The authors still need to rule out this impact.

### Questions
- There are now many improved approaches for RNN-like neural networks, such as Mamba and RWKV. Why is it necessary to return to using attention? Is this truly a better choice compared to other structures?
- In visibility-aware attention, the authors modify attention weights using predicted visibility. While this operation is shown to be beneficial in results, are there cases where errors in predicting visibility lead to performance degradation instead? Why can it be guaranteed to have a positive effect?
- There are some ambiguities in the description of the method, such as the implementation of APU. What is the rationale behind such an operation? Is it merely due to the addition of parameters?
- Are point tracking methods sensitive to the backbone network?

### Soundness
3

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
This work presents TAPTRv3, which aims to leverage both spatial and temporal context to for robust tracking in long videos. It introduces two key operations, i.e. Context-aware Cross- Attention and Visibility-aware Long-Temporal Attention, which can improve the quality of spatial cross attention and long-term feature updating. An auto-triggered global matching mechanism is further triggered when a scene cut is detected. The tracker has been evaluated on public benchmarks.

### Strengths
1. The introduction of Context-aware Cross-Attention (CCA) and Visibility-aware Long-Temporal Attention (VLTA) seems reasonable.
2. The auto-triggered global matching mechanism is easy to follow.
3. This paper provides extensive quantitative results on the public tracking benchmarks, demonstrating the method's robustness, and efficiency.

### Weaknesses
1. The overall framework of TAPTRv3 is highly similar to TAPTRv2. In TAPTRv2, it also introduces Attention-based Position Update and visibility classifier to maintain the temporal consistency. Despite the implementation variations, the key idea of this work is not very novel.
2. The performance gains compared to CoTracker3 and Track-On in Table 1 are marginal.
3. The auto-triggered global matching is simply a global redetection mechanism, which has been widely explored in other tracking frameworks.

### Questions
1. What are the major insights of this work compared to TAPTRv2?
2. What’s the key difference between the proposed auto-triggered global matching and other redetection approaches?

### Soundness
2

### Presentation
3

### Contribution
2
