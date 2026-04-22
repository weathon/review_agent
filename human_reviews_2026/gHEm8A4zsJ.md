# GAT++: Adaptive Relation-Aware Graph Attention Networks

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Leveraging high-order structural semantics in knowledge graphs (KGs) is critical for modeling complex user preferences in recommendations. However, during multi-hop propagation, semantic noise arising from heterogeneous relation distributions obscures meaningful preferences, making it challenging to learn robust user-item representations. To address this challenge, we propose GAT++, a novel graph convolutional network that integrates relation-aware attention mechanisms with contrastive denoising regularization to learn robust and expressive user-item representations. At its core, GAT++ introduces an adaptive attention module that captures multiple semantic relation spaces by projecting entities into relation-specific subspaces and learning distinct relation weight distributions. To further suppress noise from high-order message passing, we introduce a contrastive regularizer that leverages multi-relation subgraph variants to enforce consistency across augmented views. Moreover, we develop a personalized denoising encoder that dynamically refines user-item representations end-to-end, removing the need for external data generation modules. We evaluate GAT++ on extensive real-world datasets across music, literature, and food domains. GAT++ achieves up to 34.81% improvement in Recall@N over strong baselines, demonstrating its effectiveness and generalizability across diverse recommendation scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper propose a graph attention framework designed to improve knowledge-graph-based recommender systems by addressing high-order semantic noise. The method combines:
	1. Relation-aware multi-space attention, projecting entities into relation-specific subspaces,
	2. Contrastive denoising regularization, aligning subgraph variants,
	3. A personalized denoising encoder, a Transformer-style module that refines user-item representations.
Experiments on three public datasets (Last.FM, Book-Crossing, Dianping-Food) show large gains over baselines such as KGAT, CKAN, and DR4SR+.

### Strengths
1. The paper addresses an important problem, semantic noise and heterogeneity in multi-relational KGs for recommendation.
2. The overall architecture is coherent and modular, combining relational attention, contrastive regularization, and sequence-level denoising.
3. The empirical results are strong and consistent across multiple datasets and metrics (AUC, F1, Recall@K, NDCG).
4. The ablation studies are well-structured and demonstrate that each proposed component contributes positively.

### Weaknesses
1. **Limited Novelty.**  
  The main contributions appear to be incremental combinations of existing approaches.  
  Relation-specific projections or similar concepts have been already explored in *CompGCN*, *KGIN*, and *KGAT*.
  Contrastive denoising follows the paradigm of *SGL (SIGIR 2021)* and *KGCL (SIGIR 2022)*.
  and the personalized denoising encoder resembles *BERT4Rec* or *DIEN*.  
  The claim of being the “first to introduce relation attention weight distributions” is not substantiated.

2. **Methodological Vagueness.**  
  Core equations (1)–(4) lack precision.  
  It is unclear how projection matrices \(M_r\) interact with entity embeddings (additive or multiplicative).  
  The "adaptive saliency mechanism" is not formally defined, and the sampling of "subgraph variants from salient relations"
  lacks reproducible specification.  
  The role and placement of the denoising encoder within the GAT layers remain ambiguous.

3. **Implausible Empirical Gains.**  
  Reported improvements (e.g., +424% Recall@N over GAT) are unusually large for this domain.  
  No variance, seed averaging, or significance testing details are provided despite repeated "p < 0.01" claims.  
  Runtime and scalability analyses are missing even though the model adds computationally heavy components.

4. **Incomplete Baselines and Fairness.**  
  The paper omits strong recent baselines such as *LightGCN*, *SimGCL*, *NCL*, and *KGCL*.  
  *DR4SR+* is not a directly comparable baseline as it targets sequential recommendation rather than KG reasoning.  
  Hyperparameter tuning fairness is not discussed.

5. **Clarity and Presentation Issues.**  
  The text overuses vague terms such as "fine-grained semantics" and "semantic robustness" without clear definitions.  
  Notation inconsistencies (e.g., \(L_{Noise}\) vs. \(L_{Noize}\), \(e_u^{R0}\) vs. \(e_u^{r_n}\)) hinder readability.  
  Figures are schematic but fail to illustrate the precise data flow or architectural layering.

6. **Insufficient Analysis and Interpretation.**  
  The paper lacks visualization or interpretation of learned attention weights, qualitative case studies, or  
  error analyses that could clarify why the model performs better.

7. **No Theoretical Justification.**  
  Claims such as "contrastive regularization maximizes mutual information" are not derived or proven.  
  The discussion of "robustness to high-order noise" remains purely empirical and lacks formal analysis.

### Questions
1. How are the "relation-specific subspaces" and "salient relations" concretely defined and implemented?  
2. How many contrastive views per node are sampled, and what is the computational cost of this procedure?
3. Were improvements verified over multiple random seeds?  Please include standard deviations in all tables.
4. How do you prevent potential data leakage between KG triples and user–item interactions?
5. Have you compared with modern contrastive recommendation baselines such as *SimGCL*, *NCL*, or *KGCL*?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes GAT++, a relation-aware graph attention framework for knowledge-enhanced recommendation. It projects entities into relation-specific subspaces with adaptive attention, adds a contrastive denoising regularizer built from salient relation-subgraph variants, and introduces a personalized denoising encoder trained end-to-end. Experiments on Last.FM, Book-Crossing, and Dianping-Food report consistent gains, including a cold-start setting.

### Strengths
The encoder is described with a Transformer formulation, eliminating external generators and aligning with task objectives.

### Weaknesses
Mathematical/notation consistency issues: Mixed “Lnoize” (Eq. (5)) vs “LNoise” (Fig. 2), and ambiguous indexing in Eq. (4); 

Outdated baseline coverage (2019–2020 only): As reported gains are only established over pre-2021 baselines, the experimental section currently provides limited evidence for contemporary competitiveness; stronger conclusions would require including recent SOTA baselines or justifying their omission.

### Questions
Could you precisely specify how the “relation-subgraph views” are constructed and sampled? In particular, (a) how are “relation variants” defined per layer and what saliency score selects the “top two” variants; (b) what is the negative-sample distribution (in-batch vs. memory queue) and the value/sensitivity range of the temperature τ

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GAT++, a graph attention network for knowledge graph-enhanced recommendation systems that addresses semantic noise during multi-hop propagation. The method features three components: (1) adaptive relation-aware attention with relation-specific projections, (2) contrastive denoising regularization using multi-relation subgraphs, and (3) a personalized denoising encoder. Experiments on three datasets show up to 34.81% improvement in Recall@N over baselines.

### Strengths
1. The paper introduces a novel relation-aware attention mechanism that explicitly models multiple semantic relation spaces through learnable projection matrices, enabling fine-grained discrimination among heterogeneous relational dependencies in knowledge graphs.

2. Extensive experiments have been conducted across three diverse datasets from different domains (music, literature, and food), with comprehensive comparisons against multiple state-of-the-art baseline methods and thorough ablation studies demonstrating statistical significance.

3. The effectiveness of individual components has been well validated through systematic ablation studies, showing that each proposed module (adaptive attention, contrastive denoising, and personalized encoder) contributes meaningfully to the overall performance improvements.

### Weaknesses
1. The paper's title, abstract, and introduction fail to clearly specify that this is recommendation system research, creating significant confusion for readers. While the methodology section and experiments clearly focus on user-item recommendation tasks, the early sections present the work as general graph neural network research, which is misleading given the task-specific nature of the proposed solutions.

2. The paper proposes modifications to GAT architecture by introducing relation-specific projections and multi-space attention mechanisms. However, GAT has been extensively studied for many years with numerous architectural variants proposed. It is unclear how this paper makes a significant contribution to the already rich literature of GAT architecture designs, particularly given that the core innovation appears to be relatively incremental adaptations for recommendation scenarios.

### Questions
1. Why doesn't the paper clearly identify itself as recommendation system research in the title and early sections?

2. What are the fundamental technical contributions beyond existing GAT architectural variants that justify publication in a top-tier venue?

### Soundness
2

### Presentation
3

### Contribution
1
