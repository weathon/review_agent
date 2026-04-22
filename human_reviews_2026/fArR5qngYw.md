# From Moments to Models: Graphon Mixture-Aware Mixup and Contrastive Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Real-world graph datasets often consist of mixtures of populations, where graphs are generated from multiple distinct underlying distributions. However, modern representation learning approaches, such as graph contrastive learning (GCL) and augmentation methods like Mixup, typically overlook this mixture structure. In this work, we propose a unified framework that explicitly models data as a mixture of underlying probabilistic graph generative models represented by graphons. To characterize these graphons, we leverage graph moments (motif densities) to cluster graphs arising from the same model. This enables us to disentangle the mixture components and identify their distinct generative mechanisms. This model-aware partitioning benefits two key graph learning tasks: 1) It enables a graphon-mixture-aware mixup (GMAM), a data augmentation technique that interpolates in a semantically valid space guided by the estimated graphons, instead of assuming a single graphon per class. 2) For GCL, it enables model-adaptive and principled augmentations. Additionally, by introducing a new model-aware objective, our proposed approach (termed MGCL) improves negative sampling by restricting negatives to graphs from other models. We establish a key theoretical guarantee: a novel, tighter bound showing that graphs sampled from graphons with small cut distance will have similar motif densities with high probability. Extensive experiments on benchmark datasets demonstrate strong empirical performance. In unsupervised learning, MGCL achieves state-of-the-art results, obtaining the top average rank across eight datasets. In supervised learning, GMAM consistently outperforms existing strategies, achieving new state-of-the-art accuracy in 6 out of 7 datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a unified framework for inferring multiple underlying generative models (i.e., graphon mixtures) from observed graph data and leverages this structure to enhance downstream tasks such as graph mixup augmentation and graph contrastive learning.

### Strengths
The work elevates "graph augmentation" from the observation space to the generative space, which is logically self-consistent. Once the graphon estimation is completed, the per-unit training cost is weakly coupled with K, making the computational overhead appear manageable and facilitating easy integration into existing contrastive learning pipelines.

### Weaknesses
1. A core idea of this paper is modeling dataset heterogeneity via multiple latent generative factors, which closely resembles the concept of latent factors in disentangled graph representation learning [1, 2]. However, the article lacks comparisons with baselines from this related line of work.

 2. The paper claims to obtain a more disentangled representation but lacks corresponding visualizations or experiments using quantitative disentanglement metrics. For example, visualizations like feature correlation matrices or comparative analyses are missing.

 3. Several choices in the pre-modeling stage (e.g., the selection of K, potential bias in graphon estimation) likely influence the results, yet the paper lacks ablation studies examining these aspects.


[1] Disentangled Graph Contrastive Learning. NeurIPS 2021


[2] Disentangled Graph Convolution Networks. ICML 2019

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
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a framework for graph representation learning that models datasets as mixtures of underlying generative processes represented by graphons. The key idea is to represent each latent generative mechanism by a graphon, a continuous function that defines connection probabilities between nodes. To uncover these mechanisms, the authors propose to characterize graphs using motif densities (graph moments), which serve as structural fingerprints. Graphs with similar motif statistics are clustered together, and a distinct graphon is estimated for each cluster. Building on this mixture model, the authors propose two applications: Graphon Mixture-Aware Mixup (GMAM) for semantically consistent data augmentation, and Model-aware Graph Contrastive Learning (MGCL) for reducing false negatives in unsupervised learning. The approach is supported by theoretical analysis and achieves competitive results across several benchmark datasets.

### Strengths
**Conceptual novelty** Clearly identifies and formalizes the overlooked “mixture of graphons” problem, which challenges the single-distribution assumption in existing graph learning frameworks.

**Strong theoretical contribution** Introduces a novel, tighter motif concentration bound and provides complete proofs.

**Empirical validation** Demonstrates improvements on both synthetic and real datasets, with extensive ablation and visualization.

**Interpretability** Motif-based clustering yields interpretable “graph fingerprints” and meaningful estimated graphons.

**Clarity and reproducibility** The presentation is very clear, and the appendices provide all implementation details.

### Weaknesses
W1: The framework is only evaluated within two settings — Mixup augmentation and contrastive learning. There is no discussion or experiment on extending the mixture-aware framework to other learning paradigms, such as semi-supervised node classification, which essentially corresponds to a subgraph classification task over ego-networks across different hops.

W2: While the proposed methods achieve the best overall results, the performance gains over strong baselines are small, often below 1%, raising concerns about the practical significance of the improvement.

W3：There are minor typos, such as “Equation equation 10” in line 208.

W4: Experiments are confined to small- and medium-scale TUDatasets. It remains unclear how the proposed methods perform on large graphs.

W5: The paper sets the number of mixture components as log of the number of graphs, but the ablation in Appendix shows that performance is quite sensitive to the choice of $K$, suggesting that this prior strategy requires further investigation. In addition, no such ablation is reported for the Mixup setting, where similar sensitivity may arise.

W6: While Appendix presents an ablation on the number of motifs, the paper does not explore how different combinations or types of motifs affect clustering or downstream performance. This leaves open whether the proposed results are robust to motif choice.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This submission presents a framework for graph representation learning that models data as a mixture of graphons, using motif density-based clustering to disentangle generative models. It introduces two methods: GMAM (for supervised mixup augmentation) and MGCL (for unsupervised contrastive learning with model-aware sampling). 
A theoretical result provides a bound linking the cut distance between graphons and differences in empirical motif densities. Experiments show improved performance over existing mixup and contrastive learning methods.

### Strengths
1. The motivation to address graph heterogeneity via graphon mixtures is reasonable and intuitively appealing.
2. The paper is clearly written and well-organized, with good visual aids (e.g., Figure 1) explaining the workflow.
3. Empirical results are generally positive, demonstrating improvements on standard benchmark datasets.

### Weaknesses
1. The theoretical component (Theorem 1) is incremental and largely reuses existing concepts from graph theory (e.g., motif density concentration). 
The bound provided, although claimed to be tighter, does not appear to yield any substantial new theoretical insight or algorithmic design.
2. Both GMAM and MGCL are relatively straightforward extensions of existing approaches such as G-Mixup, SIGL, and GraphCL. 
The modifications mainly add a clustering step based on motif statistics, followed by standard mixup or contrastive loss. 
This design is incremental and lacks conceptual depth.
3. The paper only briefly mentions computational complexity in Appendix A.1, without any comparison to baselines or quantitative analysis (e.g., runtime, GPU hours, or scaling with graph size). 
Since the proposed methods require motif counting and multiple graphon estimations, the computational overhead is likely significant.
Without this analysis, it is unclear whether the performance gains stem from higher computational cost rather than algorithmic improvement.
4. Further experimental evaluations are needed. 1) No ablation on the number of mixture components or motif types. 2) No sensitivity study to clustering quality or graphon estimation accuracy. 3) The datasets used are relatively small and may not sufficiently stress-test scalability. 4) Missing discussion on training efficiency and memory requirements.

### Questions
Could the authors provide an ablation study for GMAM that compares it against a baseline that uses SIGL to estimate a single graphon per class (instead of a mixture)? This would help quantify the specific contribution of the mixture model idea.

### Soundness
2

### Presentation
3

### Contribution
2
