# A Consensus Anchor-Guided Hypergraph Framework for Incomplete Multi-View Clustering

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
As a significant task within the field of unsupervised learning, Incomplete Multi-View Clustering (IMVC) faces considerable challenges in scenarios involving large-scale datasets, heterogeneous data, and missing views. Existing anchor-based clustering approaches primarily reduce computational and storage overhead by introducing anchors, yet they often focus on binary sample-anchor relationships. These methods lack robust learning of consensus anchors under missing conditions and fail to effectively model high-order relationships among samples. Furthermore, systematic discussions regarding implementation details and robustness mechanisms remain insufficient. To address this, this paper proposes a Missing-aware Consensus  Anchor-guided Hypergraph Clustering (MCAHC) framework. This method constructs hypergraph through sample-anchor connections and anchor guidance to capture high-order relationships among samples, effectively mitigating view-missing and noise interference. Concurrently, it designs sample-level and view-level reweighting mechanisms to suppress inter-view imbalance and promote cross-view consistency, while explicitly down-weighting severely incomplete samples to prevent them from biasing anchor selection. Experimental results demonstrate that MCAHC provides an efficient and robust solution for multi-view clustering in large-scale and high-missing-value scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an incomplete multi-view clustering framework named MCAHC. It captures the high-order relationships among samples by introducing a consensus anchor-guided hypergraph structure and designs sample-level and view-level reweighting mechanisms to address the view missing problem. Combining hypergraph Laplacian regularization and a missing-aware mechanism, this method achieves efficient and robust clustering performance on multiple datasets and in scenarios with high missing rates.

### Strengths
1. Introduction of Hypergraph Structure: It extends the traditional binary anchor-sample relationship to a high-order relationship, enhancing the model’s ability to capture complex structures.

2. Missing-Aware Mechanism: Through sample-level and view-level reweighting, it effectively mitigates the impact of view imbalance and missing data on clustering.

3. Efficient Optimization Algorithm: It adopts an alternating optimization strategy, featuring good convergence and computational efficiency, and is suitable for large-scale data.

4. Strong Robustness and Generalization: It performs excellently on multiple datasets and under high missing rates (up to 70%), outperforming existing baseline methods.

### Weaknesses
1. Hyperparameter Sensitivity: The model contains multiple hyperparameters (e.g., $\lambda_1$, $\lambda_2$, $T$), which require careful tuning, and no adaptive selection strategy is provided.

2. Dependence on Anchor Quality: The generation and selection of anchors significantly affect the results, yet the paper does not fully discuss their robustness to missing data.

3. Insufficient Innovation in the proposed objective function: It merely incorporates a regularization term into a common framework, lacking sufficient innovation.

4. Weak Interpretability: Its core components are weak in terms of intuitive understanding and interpretability. It is difficult to clearly explain to domain experts why a sample is assigned to a specific cluster and what kind of semantic group a hyperedge specifically represents.

### Questions
1. How to adaptively select hyperparameters to avoid reliance on grid search?

2. Can the hypergraph structure be further integrated with semantic information or prior knowledge to enhance the interpretability of clustering?

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
4

### Summary
This paper addresses the Incomplete Multi-View Clustering (IMVC) problem by proposing a novel framework named MCAHC. The core innovation of the method lies in extending the traditional anchor-based bipartite graph model into an anchor-guided hypergraph model, which effectively captures higher-order relationships among samples and anchors. The paper provides a detailed formulation of the objective function, derivations of the optimization algorithm (an alternating optimization procedure involving SVD and FISTA), and a complete algorithmic workflow. The experimental section, validated across multiple datasets and varying missing rates, demonstrates that MCAHC outperforms a range of baseline methods in both clustering performance and computational efficiency. Ablation studies, convergence analysis, and parameter sensitivity analysis further substantiate the effectiveness of its individual components and the overall robustness of the model.

### Strengths
1.The integration of a hypergraph into the anchor-guided IMVC framework, combined with sample and view level reweighting mechanisms, represents a clear and significant innovation. This approach effectively mitigates view imbalance and missingness while suppressing the influence of incomplete data on anchor selection. It enhances cross-view structural consistency, and its efficacy is convincingly validated through experiments.

2.The paper provides a very complete and systematic description of the proposed method. This includes the motivation, model formulation, detailed optimization derivations, and algorithm , which together form a systematic and cohesive whole. This thoroughness significantly enhances the paper's reproducibility and represents a valuable contribution to the community.

### Weaknesses
1.The paper details the iterative optimization process for the core objective function but provides little discussion on the initialization strategy for the model parameters (e.g., the consensus anchor matrix A). The quality of initializations can influence the optimization trajectory and final results. It would strengthen the paper to include a brief discussion on this aspect in a revised version.

2.The experimental section includes a dedicated discussion on the impact of hyperparameters λ₁ and λ₂. However, it lacks a detailed analysis on the selection or sensitivity of T (the number of anchors each sample connects to in the hypergraph). This parameter crucially influences the hypergraph's topology. The paper would benefit from discussing how T was set and its potential impact on performance.

 3.On Page 1, the phrase "an contrastive learning framework" contains a grammatical error.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

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
This paper aims to address the incomplete MVC task. Traditional methods frequently use anchor learning to avoid large computations, but lack robust learning of anchors and fail to extract high-order correlations. The authors propose MCAHC to overcome these drawbacks and propose an alternating optimization process.

### Strengths
1.The code is provided in the appendix, ensuring its reproducibility.

2.The extraction of high-order correlations between anchors and samples is interesting.

3.The experimental results demonstrate the method's superiority.

### Weaknesses
1.The novelty appears limited. Anchor learning is widespread in IMVC, and the hypergraph concept has been proposed for IMVC and anchor-based MVC, as seen in [1-2].

2.Most of the compared algorithms are designed for complete MVC. It should be clarified how these methods were extended to the IMVC setting. Furthermore, the paper should justify why more state-of-the-art IMVC methods were not included in the comparison.

3.The authors claim that “These methods lack robust learning of consensus anchors”. However, this claim lacks sufficient theoretical or experimental support.

[1]Chen J, Xu H, Xue J, et al. Incomplete multi-view clustering based on hypergraph[J]. Information Fusion, 2025, 117: 102804.

[2]Zeng Y, Song P, Yang B, et al. Hypergraph Regularization-Based Anchor Learning for Multi-View Clustering[J]. Pattern Recognition, 2025: 112465.

### Questions
1.A  analysis of the space and time complexity is needed.

2.An analysis of the model's sensitivity to the number of anchors should be included.

### Soundness
3

### Presentation
3

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
This paper proposes a Missing-aware Consensus Anchor-guided Hypergraph Clustering (MCAHC) method to address two limitations of existing anchor-based multi-view clustering approaches: inability to robustly learn consensus anchors under missing data, and inadequacy in capturing high-order sample relationships. MCAHC first introduces a missing-aware mechanism to mitigate distribution shift under incomplete conditions, then designs an anchor-sample hypergraph structure for high-order relation capture. Notably, its anchor-guided hypergraph framework, which is used to construct the anchor-sample hypergraph, is one of the significant contributions of this paper. This framework builds the hypergraph by measuring sample-anchor similarity and applies graph Laplacian regularization to enforce cross-view structural consistency while retaining the scalability advantages of anchors.

### Strengths
1. The manuscript is well-written and logically rigorous, moving seamlessly from problem analysis through method design to experimental verification, ensuring high readability. 
2. Its motivation is compelling: it tackles distribution drift and lost higher-order relations under missing data by integrating a missing-data-aware mechanism with an anchor-sample hypergraph framework, yielding a clearly stated, innovative contribution.

### Weaknesses
1. The METHODOLOGY section is insufficient and requires a more detailed description.
2. Most of the baseline methods in the paper are from more than 2024 years ago, and it may be necessary to add relevant work from 2025 years ago.
3. The experimental environment and computational complexity analysis of the algorithm are lacking.

### Questions
1. The METHODOLOGY section is insufficient and requires a more detailed description.
2. The resolution of Figure 1 may need to be increased; it does not appear to be a vector graphic.
3. Most baseline methods selected were published before 2024, yet anchor-based clustering has seen extensive recent attention, including numerous 2025 works. Therefore, if conditions permit, it is recommended to select 1-2 of the latest 2025 methods for comparison.
4. The anchor-sample hypergraph structure is a crucial component of MCAHC and a core contribution of the paper. Thus, visualizing this anchor-sample hypergraph would significantly facilitate readers’ understanding of the work.
5. This paper involves a comparison of the running time of different algorithms, so it may be necessary to provide the specific running environment of these models.
6. Since the model's runtime is affected by many factors, it is necessary to add a computational complexity analysis of MCAHC and compare its computational complexity with other baseline methods.

### Soundness
3

### Presentation
3

### Contribution
2
