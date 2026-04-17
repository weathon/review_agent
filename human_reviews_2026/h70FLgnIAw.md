# TeFlow:  Enabling Multi-frame Supervision for Feed-forward Scene Flow Estimation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6

## Abstract
Self-supervised feed-forward methods for scene flow estimation offer real-time efficiency, but their supervision from two-frame point correspondences is unreliable and often breaks down under occlusions. Multi-frame supervision has the potential to provide more stable guidance by incorporating motion cues from past frames, yet naive extensions of two-frame objectives are ineffective because point correspondences vary abruptly across frames, producing inconsistent signals.
In the paper, we present TeFlow, enabling multi-frame supervision for feed-forward models by mining temporally consistent supervision. 
TeFlow introduces a temporal ensembling strategy that forms reliable supervisory signals by aggregating the most temporally consistent motion cues from a candidate pool built across multiple frames.
Extensive evaluations demonstrate that TeFlow establishes a new state-of-the-art for self-supervised feed-forward methods, achieving performance gains of **up to 33\%** on the challenging Argoverse 2 and nuScenes datasets. Our method performs on par with leading optimization-based methods, yet speeds up **150** times. The source code and model weights will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates self-supervised scene flow estimation from multi-frame point clouds. It introduces a self-supervised framework that segments the scene into static and dynamic regions, and leverages temporal ensembling and voting to obtain supervision signals for the dynamic parts. Experimental results on the Argoverse 2 and nuScenes datasets show that the proposed approach achieves competitive performance with low computational cost.

### Strengths
- The proposed approach demonstrates competitive performance compared to other feed-forward methods on the Argoverse 2 and nuScenes datasets.
- The experimental evaluation is comprehensive.

### Weaknesses
1. The writing should be improved.

- Figure 2 needs improvement.
  As the key figure illustrating the overall framework, Figure 2 does not effectively help readers understand the temporal ensembling and voting algorithms. In particular, the meanings of the different colors and arrows in the motion candidate pool are not explained, making it difficult to interpret the figure.

- The writing of Section 4.1 should be improved.
  In Line 213, the paper states that "we establish correspondences by finding, for each point p_i, its nearest neighbor in P," whereas in Eq. (3), the nearest-neighbor search is performed between p_k and P. This makes the process of motion candidate generation hard to follow.

2. The rationale of Motion Candidate Generation needs to be further clarified.

    When generating the supervisory signal from previous frames, the method directly finds the nearest neighbor of the current points in the previous frame, without warping the current points according to the (predicted) motion between the two frames. By ignoring the inter-frame motion, performing nearest-neighbor search without such warping or motion compensation becomes inappropriate for establishing accurate correspondences. It is worth noting that in self-supervised scene flow estimation, almost all self-supervised loss functions (e.g., Chamfer loss) warp the source points toward the target frame to find correspondences and thereby generate the supervision signal.

3. In Eq. (5), the authors use the motion direction (i.e., cosine similarity) to measure the consistency between two flow candidates. It would be helpful to explain why the end point error (EPE) is not used. Since cosine similarity only accounts for the direction of motion while ignoring its magnitude, the consistency evaluation may be incomplete or potentially misleading.

### Questions
1. Please explain the detailed process of Motion Candidate Generation, especially Line 213 and Eq. (3).
2. Please clarify the rationale behind the design of Motion Candidate Generation.
3. Why is cosine similarity used instead of the EPE for measuring the consistency? Is there any experimental evidence supporting this design choice?

### Soundness
2

### Presentation
2

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
The paper presents a feed-forward network that learns how to solve scene flow using temporal ensembling strategy. The results are strong and on part of the dataset examines showed significant improvement over STOA. The used technique involves adding temporal data and a joined cost function over points and blocks.

### Strengths
The primary strength of the TeFlow method  is its introduction of cluster loss that enables balanced multi-frame supervision. While prior feed-forward methods rely on two-frame correspondence losses, TeFlow first aggregates a highly stable and temporally consistent motion target for each dynamic object cluster through a temporal ensembling strategy. This cluster-level averaging prevents the loss from being dominated by larger objects with more points, ensuring that smaller dynamic objects, such as pedestrians, receive fair and effective supervision.

### Weaknesses
The ideas presented in this paper are not new but their combination provides strong outcome. Specifically, clustering of object-level loss enforcement was already published (and cited by the authors), as well as temporal constraints (more than two frames). Hence, while they provided solution is worthy and achieve STOA in some cases, it is an incremental improvement over known methods.

### Questions
Please elaborate on the contribution of each item already used and known in literature over the provided solution.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a supervised multi-frame scene flow prediction framework. To address issues such as occlusion and multi-frame temporal expansion, the proposed TeFlow presents an effective temporal aggregation strategy, according to the authors, which has significant speed improvements and performance advantages.

### Strengths
1. Good presentation and clear writng, which makes it easy to read.

2. Effective method design and good performance.

3. Comprehensive Experimental Validation. The study includes rigorous evaluations on two large-scale autonomous driving datasets, with detailed ablation studies on input frame count, loss components, and hyperparameters.

### Weaknesses
1. Although the method is leading in many metrics, it can be learned that TeFlow has room for improvement on some indicators, which are areas that can be done better.

2. Line 464, inconsistent capitalization.

3. Has the speed of this method been averaged from multiple measurements? Specifically, how many times?

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
