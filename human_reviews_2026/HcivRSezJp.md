# Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Segment Anything Model 2 (SAM2) shows excellent performance in video object segmentation tasks; however, the heavy computational burden hinders its application in real-time video processing.
Although there have been efforts to improve the efficiency of SAM2, most of them focus on retraining a lightweight backbone, with little exploration into post-training acceleration.
In this paper, we observe that SAM2 exhibits sparse perception pattern as biological vision, which provides opportunities for eliminating redundant computation and acceleration:
i) In mask decoder, the attention primarily focuses on the foreground objects, whereas the image encoder in the earlier stage exhibits a broad attention span, which results in unnecessary computation to background regions.
ii) In memory bank, only a small subset of tokens in each frame contribute significantly to memory attention, and the salient regions exhibit temporal consistency, making full-token computation redundant.
With these insights, we propose Efficient-SAM2, which promotes SAM2 to adaptively focus on object regions while eliminating task-irrelevant computations, thereby significantly improving inference efficiency.
Specifically, for image encoder, we propose object-aware Sparse Window Routing (SWR), a window-level computation allocation mechanism that leverages the consistency and saliency cues from the previous-frame decoder to route background regions into a lightweight shortcut branch.
Moreover, for memory attention, we propose object-aware Sparse Memory Retrieval (SMR), which allows only the salient memory tokens in each frame to participate in computation, 
with the saliency pattern reused from their first recollection.
With negligible additional parameters and minimal training overhead, Efficient-SAM2 delivers 1.68$\times$ speedup on SAM2.1-L model with only 1.0\% accuracy drop on SA-V test set, where SWR and SMR provide 1.83$\times$ and 1.78$\times$ speedups, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the key motivation of resolving SAM2’s high computational burden that limits its real-time application in video object segmentation (VOS). The  proposed Efficient-SAM2, a post-training acceleration framework, has two core components: 1) Object-Aware Sparse Window Routing (SWR) for the image encoder, which leverages spatial-temporal consistency and perceptual saliency from the previous frame’s mask decoder to route background windows to a lightweight shortcut branch  and preserve full computation for object-relevant windows,2) Object-Aware Sparse Memory Retrieval (SMR) for memory attention, which caches each memory frame’s saliency pattern during its first recollection and reuses it in subsequent frames to only involve salient tokens in computation. Experimentally, Efficient-SAM2 achieves a 1.68× end-to-end speedup on the SAM2.1-L model with only a 1.0% accuracy drop on the SA-V test set.

### Strengths
1.  Efficient-SAM2 avoids expensive full-model retraining. It adds negligible parameters  and low training overhead, making it flexible for low computational deployment.
2.  By aligning with SAM2’s natural sparse perception, it eliminates redundancy without compromising core functionality—unlike generic token-merging methods (e.g., ToMe) that cause severe accuracy drops.
3.  SWR (image encoder) and SMR (memory attention) are independent modules, allowing separate optimization or integration with other SAM2 variants. Their design (window-level routing, cached saliency patterns) is intuitive and supported by qualitative analysis (e.g., Figure 6 shows SWR preserves object attention).
4.  It maintains strong accuracy across diverse VOS datasets (SA-V, DAVIS 2017, MOSE) and scales to larger models (SAM2.1-L).

### Weaknesses
1. The claimed contribution 1 should be merge with contribution 2 as whole one.


2. SWR relies on hyperparameters like the prediction confidence threshold (θₒᵦⱼ=0.5) and saliency threshold (τ=0.7), while SMR depends on the sparsity ratio (s=0.95). The paper does not explore how these parameters generalize to edge cases (e.g., highly cluttered scenes, fast-moving objects) or different datasets.


3. The variable symbols cause confusions, especially for different A.


4. The codes released as supplementary material fail to run according to SAM-2 environment. The clarity of the code is also limited.

### Questions
1.What's the actual theory that the equation (11) can reflect the saliency of a window? It should be explained.
   
 2.How can the object-aware router adapt to the fast moving object since all the information of prediction and saliency are from the preceding frame?
   
 3.Mask decoder module seems to disappear from the figure 3. How is the segmentation mask produced?
   
 4.What is the alignment of  background feature processed by lightweight shortcut branch?
   
 5.How would Efficient-SAM2 adapt to dynamic prompts (e.g., user-added points in middle frames) that change the focus of attention?
   
 6.Can you provide statistical data (e.g., average CS across all frames in SA-V/DAVIS) to quantify how often saliency patterns remain consistent? What is the impact of inconsistent patterns on SMR’s accuracy?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Efficient-SAM2, a post-training acceleration framework to address the significant computational bottleneck of SAM2 in real-time video object segmentation. The authors identify a mismatch between SAM2's dense computation and its inherently sparse perception, highlighting redundancy in the image encoder's background processing and in full-token memory retrieval. To exploit this, the framework introduces two key components: object-aware Sparse Window Routing (SWR) and object-aware Sparse Memory Retrieval (SMR). SWR dynamically routes irrelevant background windows in the encoder to a lightweight shortcut branch, guided by saliency and consistency cues from the previous frame's decoder. SMR leverages temporal consistency by identifying a sparse set of salient memory tokens during their first recollection and reusing this pattern for subsequent frames, drastically reducing memory attention computations.

### Strengths
1. The motivation for reducing SAM2's computational overhead is well-grounded and intuitive for video object segmentation
2. The post-training approach is practical, enabling efficient adaptation by leveraging the generalized parameters of the pre-trained SAM2.
3. The method achieves a good speed-performance trade-off, delivering a speedup of nearly 2x while incurring only a minimal and acceptable performance degradation of approximately 1%.

### Weaknesses
1. The SWR component is heavily dependent on the previous frame's prediction and salient mask. I think this may cause challenges in some cases, such as rapid motion, abrupt scene cuts, or severe occlusions, where this temporal assumption would be violated.
2. For a video domain paper, the qualitative results with static images are insufficient. Supplemental videos would be significantly stronger to properly demonstrate temporal consistency, failure modes (especially in scenarios mentioned in point 1), and the practical impact of the optimizations.
3. I think an evaluation is needed to determine if the SMR module, which uses a cached saliency pattern, maintains its effectiveness in long video scenarios where significant appearance and context drift are likely.
4. The paper lacks a dedicated discussion of its limitations. This omission leaves the impression that the proposed efficiencies might be confined to easy (or trained) scenarios.

### Questions
While the paper presents a valuable approach to making SAM2 more efficient, the design of the SWR and SMR modules appears highly heuristic and is tightly coupled to assumptions about temporal continuity. This raises a significant question: Is SAM2's strong, general-purpose segmentation performance fully preserved? The evaluation is currently confined to standard VOS benchmarks. To truly validate that these heuristics do not compromise the model's robustness, I would ask the authors to provide evaluations on more diverse datasets, particularly on challenging "in-the-wild" videos, which would better test the limits of these heuristic assumptions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper aims to reduce the high inference cost of SAM2 in video segmentation. Specifically, they propose two modules: Object-aware Sparse Window Routing (SWR), which skips background windows in the image encoder based on object masks and saliency, and Object-aware Sparse Memory Retrieval (SMR), which selects only salient memory tokens and reuses their mask across frames. Together, these modules accelerate SAM2 by up to 1.75× with minimal accuracy loss on benchmarks such as SA-V, DAVIS, and MOSE.

### Strengths
1. The paper makes a solid technical contribution by streamlining the SAM2 model. The object-aware pruning of the image encoder and the introduction of a background shortcut for non-foreground patches are both clever ideas that substantially reduce computation.

2. The ablation study is comprehensive. It not only analyzes the proposed components in isolation but also integrates other efficient methods (e.g., ToME) into their framework for comparison, which provides valuable insights.

### Weaknesses
1. The proposed routing mechanism heavily relies on the assumption of temporal consistency in video streams, meaning that no significant camera shaking or viewpoint shift occurs. This limits the method’s applicability in real-world scenarios with dynamic motion. It would be interesting to see comparisons with SAM2 on datasets such as MOSEv2[1] and SeCVOS[2], which feature frequent viewpoint transitions.




2. Performing grid search on only one benchmark is not sufficient to demonstrate robustness. It would strengthen the paper to include additional grid search curves across multiple benchmarks (in Figure 5).







[1] MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex Scenes
[2] SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction

### Questions
1. What is the specific layer index after which the router divides foreground and background tokens? How many subsequent modules benefit from reduced computation? It would be helpful to include an ablation varying the routing layer index to assess its impact.

2. How is speedup defined in this paper? It would be clearer to disclose it in two aspects—FLOPs reduction and throughput improvement—to give a more complete view of efficiency.

3. What is the intuition behind using two different temporal intervals (∆t =1,5) in the experiments? It seems the performance difference might largely stem from the frame rate (FPS) of the original benchmark. For high-FPS datasets like SA-V, increasing the interval could naturally yield greater gains since the memory bank contains more diverse frames.

4. For SMR, have you evaluated a variant that selects memory frames only when object presence is confident (akin to SAM2Long’s strategy)? It would be informative to report the gain from this filtering.

### Soundness
3

### Presentation
3

### Contribution
3
