# Multi-Domain Riemannian Graph Gluing for Building Graph Foundation Models

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 4

## Abstract
Multi-domain graph pre-training integrates knowledge from diverse domains to enhance performance in the target domains, which is crucial for building graph foundation models. Despite initial success, existing solutions often fall short of answering a fundamental question: how is knowledge integrated or transferred across domains? This theoretical limitation motivates us to rethink the consistency and transferability between the pre-trained model and target domains. In this paper, we propose a fresh differential geometry perspective, whose core idea is to merge any graph dataset into a unified, smooth Riemannian manifold, enabling a systematic understanding of knowledge integration and transfer. To achieve this, our key contribution is the theoretical establishment of neural manifold gluing,
which first characterizes local geometry using an adaptive orthogonal frame and then “glues” the local pieces together into a coherent whole. Building on this theory, we present the GraphGlue framework, which supports batched pre-training with EMA prototyping and provides a transferability measure based on geometric consistence. Extensive experiments demonstrate its superior performance across diverse graph domains. Moreover, we empirically validated GraphGlue’s geometric scaling law, showing that larger quantities of datasets improve model transferability by producing a smoother manifold.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a differential-geometry view of multi-domain graph pre-training: neural manifold gluing that merges graphs from diverse domains into a unified smooth Riemannian manifold. Locally, geometry is estimated via a (k, M)-sparse perturbation and an Adaptive Orthogonal Frame (AOF) to form tangent-space bases and local metrics. Globally, pieces are “glued” by edge tangent translation (isometries) with holonomy (triangle) triviality and further curvature-based smoothing to promote a smooth global metric. Experiments over six domains show strong few-shot transfer and a geometric scaling law: adding more pre-training datasets yields smoother manifolds and better transfer.

### Strengths
1. Conceptual originality & unification: Frames multi-domain GFM pre-training/adaptation in a single geometric framework with local-to-global gluing, tying transferability to metric compatibility, holonomy, and Ricci-related smoothness—offering interpretable levers rather than only heuristics.
2. Experiments span diverse graph domains and demonstrate solid performance and scalability. The observed geometric scaling law is an insightful and interpretable empirical finding that supports the theoretical motivation.
3. The proposed framework is well structured, moving clearly from local metric learning (AOF) to global manifold construction (holonomy and curvature smoothing) and then to practical pre-training and adaptation.

### Weaknesses
1. Although the method is conceptually rich, the explanation is dense, with long mathematical expressions and limited intuition. The main framework diagram is visually cluttered and could better emphasize the data flow between modules. Simplifying the narrative and improving the visuals would greatly enhance clarity.
2. The paper lacks a clear analysis of computational cost and training scalability — given that manifold operations such as matrix logarithm and curvature regularization can be expensive, a runtime or memory comparison would help readers assess practicality for large-scale deployment.
3. The method relies on geometric assumptions such as triangle-based holonomy and curvature smoothness, but many real graphs are sparse or irregular, with very few closed loops. It remains unclear how well the model behaves when these geometric constraints cannot be fully satisfied. Adding experiments or visualizations on sparse or highly heterogeneous graphs would make the theory–practice connection more convincing.

### Questions
1. Triangle coverage & sparsity: How does performance degrade when the underlying graph (or inter-domain scaffold) has few triangles, so triangle holonomy regularization is weak or inapplicable? Do you back off to cycle-based approximations or add synthetic motifs? (Additionally, some theorems are based on the assumption that every edge belongs to at least one triangle, e.g., Theorem 4.8)
2. How sensitive is the learned manifold geometry to the hyperparameters of the AOF module (e.g., k and perturbation scale)?
3. Could you provide more intuition or examples to help readers understand how the “manifold gluing” process works in practice?
4. From which previously introduced equations or lemmas is **equation (30)** derived?
5. See weaknesses.

### Soundness
3

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
This paper presents a novel differential-geometry view of multi-domain graph pretraining and domain adaptation, which the authors call a neural manifold gluing framework. Locally the model learns tangent-space bases and local metrics via an Adaptive Orthogonal Frame (AOF) and a (k, M)-sparse perturbation; globally the framework “glues” local pieces using edge tangent translation and holonomy/curvature-based smoothing to induce a smooth Riemannian metric across domains. The method includes practical engineering (EMA-based prototyping and a Riemannian MoE) to scale to large multi-graph settings. Experiments on multiple datasets and leave-one-domain-out few-shot transfer show consistent improvements over baselines and an empirical “geometric scaling law.”

### Strengths
1. Strong theoretical foundation and unification. The paper provides a mathematically solid perspective, casting multi-domain pretraining and domain adaptation as a manifold-gluing problem; this is novel and helps unify several previously disparate ideas about metric compatibility, holonomy, and transferability. Theorem-level results and carefully defined operators (e.g., edge tangent translation) make the theoretical contribution convincing.

2. Practical scaling via an EMA strategy. The use of EMA prototypes and the design choices to make the framework operate on large graphs are valuable. These engineering choices materially improve the applicability of multi-domain graph learning to realistic, large-scale settings.

3. Comprehensive experiments and diverse datasets. The authors evaluate across a variety of datasets and tasks (leave-one-domain-out few-shot settings, ablations, scaling-law experiments). The empirical section is thorough and supports the main claims.

### Weaknesses
1. The idea of “manifold gluing” may depend on the number of domains and on how many pairwise/collective gluing operations are needed. The paper briefly remarks that a QR-based subroutine reduces complexity, but it lacks a clear, end-to-end complexity and empirical runtime/memory analysis showing how cost scales with the number of domains, graph size, and manifold dimension.
2. The paper states (line ~1532) “For pretraining, we extract the 2-hop ego-graph with 10 neighbors each hop for single graph datasets and adopt a 2-layer GCN...” Please clarify: how are those 10 neighbors chosen? Are they random uniform samples among neighbors, the top-10 by degree/score, or chosen by some importance sampling / structural heuristic? This will inform whether the pretraining pipeline’s effectiveness depends on a particular sampling heuristic.
3. Many of the node-classification datasets used appear to be homophilic. The manifold gluing assumptions may rely implicitly on local coherence of labels/structure; it is unclear whether the proposed framework adapts to heterophilic graphs. We suggest that the authors either include experiments on heterophilic benchmarks (or a small toy study) or provide a discussion of expected behavior if heterophilic graphs are included in pretraining.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces GraphGlue, a novel framework for multi-domain graph pre-training that leverages principles from differential geometry to address the long-standing challenge of knowledge transfer across diverse graph domains. By conceptualizing graphs as pieces of a unified Riemannian manifold, the authors provide a systematic approach to ensure that these domains are seamlessly “glued” together, preserving both local and global geometric consistency. The framework employs an Adaptive Orthogonal Frame (AOF) to model local graph structure, then uses holonomy-based regularization to smoothly align these pieces into a coherent whole.

One of the key contributions of this work is the Geometric Transfer Metric (GTM), which quantifies the transfer difficulty between pre-trained models and target domains. The experiments show that GraphGlue achieves superior transferability and scalability across multiple graph domains, outperforming existing methods. The paper also introduces a new empirical insight: geometric scaling law, which suggests that increasing the number of domains during pre-training leads to a smoother manifold and better generalization.

### Strengths
1. The paper introduces a novel approach to multi-domain graph pre-training by treating the graphs as local pieces of a larger, unified Riemannian manifold. This fresh perspective allows for a more systematic understanding of knowledge transfer across domains, which is a critical issue in graph foundation models.
2. The concept of “neural manifold gluing” is well-formulated, using differential geometry to tie together multiple domains. The method employs an Adaptive Orthogonal Frame (AOF) to model local geometry and uses holonomy and curvature constraints to ensure smooth global alignment. This provides both theoretical depth and practical utility.
3. The experiments cover a range of graph domains and demonstrate that the proposed method outperforms previous models in terms of transferability. The geometric scaling law showing that increasing the number of domains improves transferability is both a novel and insightful contribution to the field.
4. Convincing theoretical analyses.

### Weaknesses
1. The method heavily relies on triangle-based holonomy for graph gluing. However, in sparse graphs or those with few cycles, this assumption may not hold, limiting the approach's applicability. Further analysis of the method’s behavior on sparse or acyclic graphs is needed.
2. The paper introduces the AOF for local geometry estimation, but does not provide sufficient analysis on how sensitive the results are to the choice of hyperparameters like perturbation strength and neighborhood size.
3. Some parts of the paper writing needs to be polished.

### Questions
1. See weaknesses.
2. Could you provide a clear logical chain explaining the connections and roles of the numerous theorems and lemmas? **This is my main concern**, and I hope the authors can primarily address this question within a limited number of characters.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on how knowledge is theoretically integrated and transferred during multi-domain graph pre-training, which is a fundamental and underexplored problem in the field of graph foundation models (GFMs). Authors propose a novel perspective from differential geometry, with the core idea of gluing diverse graph datasets into a unified Riemannian manifold. Moreover, authors empirically validate GRAPHGLUE’s geometric scaling law and show that larger quantities of datasets could improve the transferability of models by producing a smoother manifold.

### Strengths
- This paper is good in originality and introduces a principled and powerful theoretical framework from differential geometry. This Neural Manifold Gluing concept provides a new, systematic, and theoretically sound language to model knowledge integration in graphs, which is a conceptual advance for the GFM field.
- The experiments provide strong support for the theory. The Geometric Scaling Law experiment shows that 1-shot accuracy improves and transfer loss decreases as more datasets are added, which is a powerful validation of the smoother manifold hypothesis. Furthermore, the case study in Figure 5 shows that GRAPHGLUE benefits from adding semantically distinct domains in mitigating negative transfer is a strong practical result.

### Weaknesses
- The proposed theory assumes that all source and target domains can be glued into a single smooth manifold. However, it is unclear how the model would perform if a new domain is fundamentally geometrically incompatible. For example, what if a new domain possesses a markedly different intrinsic dimensionality? Therefore, further discussion on the limitations and potential failure cases would be valuable for practical use. 
- Several key design choices in the GRAPHGLUE framework are not ablated. In particular, what is the impact of the EMA prototyping and the L_proto loss? How critical is the Riemannian MoE during adaptation compared to a simpler prompting scheme? I expect a more comprehensive ablation that provides deeper comprehension regarding the impact of individual components within the full framework. 
- The proposed framework involves several non-trivial operations. For instance, (k, M)-sparse perturbation, maintaining an Adaptive Orthogonal Frame (AOF), and calculating gluing losses (L_holo, L_curv). I am curious about the computational cost and memory complexity of the overall pre-training and adaptation process.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
3
