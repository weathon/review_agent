# Multi-Modal Point Cloud Completion with Intra- and Inter-Graph Transformer

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Multi-modal point cloud completion aims to leverage complementary image information to assist point cloud completion. Existing multi-modal approaches predominantly employ Transformers to facilitate interactions between different modalities. However, fully-connected attention-based Transformers lead to high computational cost and redundancy, and often fail to fully capture the complex relations between these modalities. To address these issues, we propose the Intra- and Inter-Graph Transformer (I$^{2}$GraphFormer), which leverages sparse graph connections to restrict attention to neighboring nodes both within and across modalities. I$^{2}$GraphFormer enhances interactions in terms of efficiency and expressiveness. Specifically, we model relations from both intra-graph and inter-graph perspectives, obtaining more expressive representations and producing higher-quality completion results. Extensive quantitative and qualitative experiments demonstrate that I$^{2}$GraphFormer outperforms state-of-the-art multi-modal approaches across various evaluation scenarios with low complexity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes I² GraphFormer, a multi-modal point cloud completion framework leveraging sparse intra- and inter-graph attention in Transformer-based architectures. The approach encodes both images and point clouds and constrains attention to local neighborhoods within and across modalities, aiming for improved efficiency and interaction expressiveness. Experiments on the ShapeNet-ViPC benchmark and KITTI dataset demonstrate competitive quantitative and qualitative results, with additional ablation studies highlighting the roles of each module in the architecture.

### Strengths
- The proposed method significantly outperforms existing methods on widely-used datasets.
- Overall, the manuscript is well-written.

### Weaknesses
- Using cross attention for modality fusion has been widely explored and proved effective. The motivation for using graphs in I2GraphFormer, as opposed to using traditional Transformers, is not fully clear in the paper. 
- Using feature distance to build graphs can be computationally expensive.
- There is no fundamental novelty regarding the cross-modal graph fusion.
- Recently, a trend of zero-shot point completion using multi-modal generative models has emerged (e.g., SDSComplete—NeurIPS 2024, ComPC—ICLR 2025, GenPC—CVPR 2025). These methods should at least be discussed in the related work section.
- The manuscript lacks an analysis of failure cases and limitations.

### Questions
Please see the weaknesses.

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
3

### Summary
This paper introduces an efficient multi-modal point cloud completion framework. By incorporating an Intra- and Inter-Graph Transformer structure with sparse graph connections, the method enables effective information interaction between point cloud and image modalities, capturing complex geometric and semantic relationships. Furthermore, a dual-view guided upsampling module is proposed to refine the reconstruction process using both geometric and image cues, resulting in finer and more accurate point clouds. Extensive experiments on synthetic and real-world datasets demonstrate that the proposed model outperforms existing multi-modal approaches across various evaluation scenarios.

### Strengths
- The paper is easy to follow.
- The proposed method demonstrates excellent performance on the cross-modal point cloud completion task.
- The authors identify the efficiency limitations of existing Transformer-based point cloud completion methods and explore an effective and efficient solution to address this issue.

### Weaknesses
**Major Concerns**
- The paper briefly mentions in line 63 that Transformers incur high computational cost, which is understandable. However, it is unclear why the Transformer "fails to adequately capture the complex relations between modalities." This claim lacks sufficient justification and should be further elaborated.
- The proposed intra-graph and inter-graph structures do not seem to follow conventional graph formulations. Each node is encoded by three distinct MLP layers, resulting in three feature representations. The method computes similarities between nodes from Q and K, selects the top-k similar nodes to construct adjacency relationships between Q and K, and then aggregates corresponding V nodes based on this relation. However, this adjacency only reflects the relationship between Q and K, seemingly unrelated to V.
From a graph-based perspective, this makes the structure somewhat unconventional. While QKV MLPs remain essential to Transformer architectures, the proposed design appears to be a more efficient variant of a Transformer block rather than a novel graph model. Despite its strong empirical performance, the paper lacks a clear explanation of why this structure leads to such improvements. The authors are encouraged to provide deeper analysis or intuition behind the observed performance gains.
- Several critical settings are missing. For instance:
  - What is the value of k in the top-k selection?
  - How many layers are used for the intra-graph Transformer?
  - Is the inter-graph Transformer a single layer?
  - What are the specific details of the geometry-guided part in the Dual-View Guided Upsampling Module? According to the ablation study, removing this module causes a significant performance drop, suggesting it plays a more important role than the intra- and inter-graph components highlighted as the paper's main contribution. However, this module is insufficiently described and not emphasized in the introduction. The authors should clarify its design and significance.
- It would be valuable to analyze the effect of different k values on performance and to include a comparison without top-k selection (i.e., using all tokens) to better understand its impact.

**Minor Concerns**
- For computational efficiency analysis, it is recommended to include MACs and FLOPs metrics.
- Figure 1 is not referenced in the main text and should be properly cited.

### Questions
Please refer to the Weaknesses part.

### Soundness
2

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
This paper introduces I2GraphFormer, a novel method for multi-modal point cloud completion that leverages complementary image information. The key contribution is a graph-based Transformer architecture designed to address the high computational cost and limited expressiveness of fully-connected attention in existing methods. The model operates by first encoding point clouds and images into tokens, then processing them through two main stages: (1) Intra-Graph Transformers that capture internal structures within each modality using sparse, top-K attention, and (2) Inter-Graph Transformers that facilitate cross-modal information exchange via a bipartite graph structure. Finally, a Dual-View Guided Upsampling Module reconstructs the complete point cloud using guidance from both geometric and semantic perspectives. The authors demonstrate state-of-the-art performance on the ShapeNet-ViPC dataset and the real-world KITTI benchmark, while also achieving lower model complexity than competitors.

### Strengths
1. Multi-modal completion is a task of considerable practical importance autonomous driving. By offering a more efficient and expressive model for fusing point cloud and image data, this work represents a meaningful advancement. The demonstrated performance on real-world KITTI data underscores its potential for practical application.
2. The core idea of replacing fully-connected cross-modal attention with a sparse graph-based paradigm is well-motivated. The explicit separation into intra-graph and inter-graph learning provides a structured and interpretable framework for modality interaction that differs significantly from prior work.
3. The paper is technically sound. The experimental evaluation is thorough, encompassing standard benchmarks (ShapeNet-ViPC), generalization to unseen categories, and real-world data (KITTI). The results show good performance in both completion quality and model efficiency.

### Weaknesses
1. Scalability.  While the top-K graph attention reduces complexity, its scalability to very large token sets or higher-resolution inputs is not thoroughly analyzed. The computational cost of the pairwise cosine similarity calculation and top-K selection for all queries, though less than full attention, could still become a bottleneck. An analysis of how the method's time/memory consumption scales with the number of tokens (N, M) and the choice of K would strengthen the paper.

2. Insight of token fusion. The paper shows that the model works well, but could provide more insight into what is being transferred between modalities. A qualitative analysis or visualization of which image tokens attend to which point cloud tokens (and vice-versa) in the inter-graph transformer would be very valuable. For instance, does the image primarily provide high-level semantic context, or does it also help infer fine-grained geometric details? This would deepen the understanding of the cross-modal fusion process.
3. The contribution of the image. The paper clearly demonstrates that the full multi-modal model performs well. However, it lacks an ablation study that quantifies the specific contribution of the single-view image to the final completion result. This would clearly isolate the performance gain attributable to the image information, both in known and unknown categories, and strengthen the claim that the image provides essential complementary guidance.
4. Robustness. The method assumes well-aligned image and point cloud pairs. A key limitation in real-world systems is imperfect calibration or temporal misalignment between sensors. The paper should evaluate the robustness of I2GraphFormer to such misalignments.

### Questions
Scalability: Could the authors provide a more detailed complexity analysis (e.g., FLOPs or memory usage vs. token count) for the graph construction and attention steps compared to a standard transformer? Have you experimented with larger token sizes or values of K, and what were the observed trade-offs?

Interpretation: Can you provide a qualitative visualization or case study showing examples of the top-K connections formed in the inter-graph transformer? For example, when a point cloud token from a missing car part queries the image, which image regions are most frequently attended to? This would help clarify the nature of the cross-modal complementarity.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes I2GraphFormer, an Intra- and Inter-Graph Transformer for multi-modal point cloud completion, which aims to leverage complementary image information to enhance 3D completion quality.

### Strengths
+ The proposed method has a smaller model size compared to existing models.
+ The claim of outperforming SOTA methods with lower complexity suggests empirical validation.

### Weaknesses
- Unclear novelty. The paper does not clearly differentiate I2GraphFormer from prior graph-based or cross-modal attention methods (e.g., Graphormer, CrossGraphFusion). The novelty appears incremental unless stronger distinctions or innovations are articulated.

- Insufficient motivation and justification. The rationale for adopting the intra- and inter-graph design is not well explained. It remains unclear why this design is superior to existing approaches. Moreover, while the sparse graph mechanism is intuitive, the paper lacks theoretical analysis or ablation studies demonstrating why inter-graph connections contribute to better completion quality.

- Weak support for the “low complexity” claim. The paper repeatedly emphasizes its low complexity, but the evaluation only compares model size without providing concrete analyses such as computational complexity (e.g., FLOPs) or actual runtime measurements.

- Unclear motivation for the Dual-View Guided Upsampling Module. The purpose and necessity of this module are not well justified. The paper does not explain why this specific design is required。

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
