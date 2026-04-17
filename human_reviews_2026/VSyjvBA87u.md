# RHGCL: Representation-Driven Hierarchical Graph Contrastive Learning for User-Item Recommendation

- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Graph Contrastive Learning (GCL), which fuses graph neural networks with contrastive learning, has evolved as a pivotal tool in user-item recommendations. While promising, existing GCL methods often lack explicit modeling of hierarchical item structures, which represent item similarities across varying resolutions. Such hierarchical item structures are ubiquitous in various items (e.g., online products and local businesses), and reflect their inherent organizational properties that serve as critical signals for enhancing recommendation accuracy. In this paper, we propose Representation-driven Hierarchical Graph Contrastive Learning (RHGCL), a novel GCL method that incorporates hierarchical item structures from learned representations for recommendations. First, RHGCL pre-trains a GCL module using cross-layer contrastive learning to obtain user and item representations. Second, RHGCL employs a representation compression and clustering method to construct a two-hierarchy user-item bipartite graph. Ultimately, RHGCL fine-tunes user and item representations by learning on the hierarchical graph, and then provides recommendations based on user-item interaction scores. Experiments on three widely adopted benchmark datasets ranging from 70K to 382K nodes confirm the superior performance of RHGCL over existing baseline models, highlighting the contribution of representation-driven hierarchical item structures in enhancing GCL methods for recommendation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes RHGCL, a hierarchical graph contrastive learning method for user-item recommendation that incorporates hierarchical item structures. The approach consists of three stages: (1) pre-training using XSimGCL to obtain user and item representations, (2) clustering items in a 2D space using t-SNE and geometric partitioning, and (3) fine-tuning on both the original user-item graph and a user-clustered item graph. Experiments on three datasets show improvements over baseline methods.

### Strengths
1. The paper identifies a relevant gap in existing GCL methods regarding hierarchical item structures and provides concrete examples of why this matters for recommendation systems.  

2. The method demonstrates performance gains across all three datasets and metrics, suggesting the approach has merit.  

3. The paper includes appropriate baselines, multiple datasets of varying sizes, and sensitivity analyses for key hyperparameters.

### Weaknesses
1. The core contribution appears to be applying t-SNE dimensionality reduction followed by geometric clustering (radial/angular divisions) to create hierarchical structures. This is a relatively straightforward extension of XSimGCL without significant methodological innovation.

2. The paper does not provide theoretical or empirical justification for why t-SNE is optimal for capturing hierarchical item relationships over alternatives like UMAP, spectral embedding, or learned hierarchical representations.

3. The deterministic polar coordinate partitioning seems arbitrary and overly simplistic. Why should semantic item relationships align with geometric sectors in a 2D embedding space? This lacks theoretical grounding.  

4. No comparison with methods that learn hierarchical structures (e.g., differentiable pooling variations adapted for bipartite graphs).  

5. The related work section (Appendix A) lists several hierarchical GCL methods for recommendations (HKGCL, HGCL, ReHCL, HEK-CL, HNECL), but none are included in the experimental comparison. While the authors claim these methods require knowledge graphs or review data, this justification is insufficient: At minimum, methods that only require interaction data (like HNECL) should be compared.


6. The improvements over XSimGCL are very modest.

### Questions
1. Can you provide theoretical or empirical justification for why t-SNE followed by geometric clustering is superior to alternative approaches for capturing hierarchical item structures?  

2. Why not compare with other hierarchical recommendation methods, particularly those that don't require external knowledge graphs (e.g., HNECL)?  

3. Can you provide statistical significance tests demonstrating that the observed improvements are not due to random variation?  

4. How do the learned clusters correspond to actual item categories or semantic relationships? Can you provide qualitative analysis?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a novel Graph Contrastive Learning (GCL) method for user-item recommendation that incorporates hierarchical item structures. It pre-trains a GCL module, constructs a two-hierarchy user-item bipartite graph via representation compression and clustering, and then fine-tunes representations. Experiments show RHGCL outperforms existing models by enhancing GCL with representation-driven hierarchical item structures for recommendation tasks

### Strengths
1. an effective method with neat writing. 
2. The model consistently outperforms strong baselines (e.g., XSimGCL, SimGCL, LightGCN) across three benchmark datasets (Yelp2018, Amazon-Kindle, Alibaba-iFashion), with meaningful improvements in Recall@20 and NDCG@20
3. The paper presents a complete pipeline—pretraining, hierarchical clustering (via t-SNE), and fine-tuning on dual-level graphs—illustrating conceptual clarity and reproducibility.

### Weaknesses
1. As mentioned by the authors, incorporating hierarchical information into a recommendation system is not new. The novelty is my main concern.
2. Adding hierarchical information will incur more computation; it is better to provide some analysis.
3.  Using t-SNE followed by radial–angular partitioning introduces hyperparameters  that may not generalise well. The clustering might be unstable or computationally expensive for large graphs.

### Questions
The questions are related to the weakness.
1. what's the core novelty compares to other works except combining different small pieces?
2. computational anlaysis?
3. ablation study on clustering?

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
This paper proposes Representation-driven Hierarchical Graph Contrastive Learning (RHGCL), which enhances user-item recommendation by explicitly modeling hierarchical item structures within a graph contrastive learning framework. Unlike existing GCL methods that overlook multi-level item similarities, RHGCL first pre-trains user and item representations using cross-layer contrastive learning, then clusters items into hierarchical groups via representation compression, and finally fine-tunes embeddings on a two-hierarchy bipartite graph combining both original and clustered interactions. By leveraging both fine-grained and coarse-grained item relationships, RHGCL improves recommendation accuracy, achieving superior performance on benchmark datasets.

### Strengths
1. RHGCL learns item hierarchies directly from data-driven representations, enabling multi-resolution modeling without relying on external graphs or attribute labels.

2. The proposed experiments across three benchmark datasets seem to demonstrate the effectiveness of the proposed method

### Weaknesses
1. The paper presents item clustering as a key innovation, but similar representation-driven clustering ideas (e.g., neighborhood-enhanced clustering in NCL [1]) have already appeared in recent recommendation literature; the proposed method is not compared with such closely-related works, weakening the novelty claim. Could the authors add comparisons with such methods to further highlight the unique contribution of the proposed clustering module?

2. All contrastive-learning baselines compared are from 2023 or earlier; several stronger 2024/2025 GCL recommenders are ignored, so the performance gap may be smaller than reported.

3. The paper offers no case study or concrete examples to illustrate why the hierarchical module matters, leaving readers without intuitive evidence of its practical value.

[1] "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning" Lin et al. WWW 2022

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2
