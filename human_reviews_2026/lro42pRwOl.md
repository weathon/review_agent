# Subgraph Generation for Generalizing on Out-of-Distribution Links

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Graphs Neural Networks (GNNs) demonstrate high-performance on link prediction
(LP) datasets, especially when the distribution of testing samples falls within the
dataset’s training distribution. However, GNNs suffer decreased performance
when evaluated on samples from outside their training distribution. In addition,
graph generative models (GGMs) show a pronounced ability to generate novel
output graphs. Despite this, the application of GGMs remains largely limited to
domain-specific tasks. To bridge this gap, we propose leveraging GGMs to produce
synthetic samples which extrapolate between training and testing distributions.
These synthetic samples are then used for fine-tuning GNNs to improve link
prediction performance in out-of-distribution (OOD) scenarios. We introduce a
theoretical perspective on this phenomena which is further verified empirically via
increased performance across synthetic and real-world OOD settings. We conduct
further analysis to investigate how inducing structural change within training
samples improves OOD performance, indicating promising new developments in
graph data augmentation on link structures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles OOD generalization for link prediction. The core idea is to augment training with counterfactual subgraphs synthesized by a generative model so that a pre‑trained link predictor becomes robust to structural shifts. The proposed framework, FLEX, first pretrains a GNN and a semi‑implicit variational autoencoder on k‑hop subgraphs extracted with the labeling trick, then co‑trains them in an adversarial way (much like GAN). Counterfactuality is encouraged by maximizing KL divergence between the posterior and prior. On the LPShift benchmark, the method improves over baselines in most cases. Efficiency analyses show FLEX training/inference overheads are modest compared to a non‑parametric counterfactual baseline like CFLP.

### Strengths
1. It has a well‑motivated problem and clear intuition.

2. FLEX has simple, modular setup. It can be used with any pre‑trained GNN. and the $$/gamma$$ threshold tackle the tendency of generators to over‑densify.

3. The empirical results over LPShift show the effectiveness of the FLEX.

### Weaknesses
1. Limited backbone and generator diversity. Results are only on GCN and NCN. Modern link‑predictors (e.g., NBFNet, Neo‑GNN, LPFormer) are referenced but not evaluated; likewise, the framework is described as agnostic to the generator, but only SIG‑VAE is used. A “FLEX‑with‑VGAE/diffusion” variant or at least a plug‑in ablation would isolate how much the semi‑implicit choice matters.

2. I am concerned about the efficiency. The recent trend of modern LP models has been shifted from subgraph-level prediction (SEAL) to more efficient node-level encoding (BUDDY, NCN). This shift makes the method efficient and applicable to the real-world use case with large-scale graphs. However, FLEX still operates at subgraph-level, meaning that it will struggle to scale to large graphs. For example, OGBL-Citation2 can be a good test bed to evaluate FLEX on large-scale dataset.

### Questions
1. During inference time, is the cotrained GNN predictor being used for prediction or a new GNN being trained from scratch on the original graph+FLEX-generated graphs? If the former, will FLEX, as a data augmentation method, generalize across different GNN backbones? In other words, if the FLEX-generated graphs can improve performance of any LP methods (including both train-from-scratch GNN or even just heuristics like Common Neighbor), it will have much broader use case.


2. Can FLEX improve the model performance not only on the distribution shift dataset but also general ones (which already has some degree of distribution shift)?

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
5

### Summary
This paper introduces FLEX, a framework designed to improve OOD generalization in GNNs for the link prediction task. The core idea is to jointly train a GGM and a GNN to generate counterfactual subgraphs that expand the structural support of the training distribution. The framework demonstrates considerable innovation and empirical effectiveness, achieving efficient subgraph-level generation through a well-designed training mechanism. By reformulating link prediction in terms of structural feature distributions (e.g., Common Neighbors), the paper provides a principled theoretical foundation for understanding the limitations of traditional link predictors under distribution shift. Extensive experiments on multiple benchmark datasets demonstrate that FLEX outperforms both traditional OOD baselines and graph-specific methods in robustness and generalization performance.

### Strengths
By reformulating the task in terms of structural feature distributions (e.g., Common Neighbors), the paper provides a principled explanation for why traditional link predictors underperform under distribution shifts. The proposed set-theoretic and ELBO-based analysis forms the unified theoretical perspectives on OOD generalization for link prediction.

The empirical evaluation spans multiple benchmark datasets and diverse graph structures, consistently demonstrating the robustness and OOD generalization ability of FLEX on link prediction tasks.

FLEX performs counterfactual generation at the subgraph level, which is an efficient and scalable design choice. This approach reduces unnecessary graph-wide computation and provides a practical path toward efficient OOD generalization.

### Weaknesses
Although the appendix provides a set-theoretic argument that counterfactual subgraph generation can enlarge the overlap between training and testing distributions, the analysis remains qualitative. It lacks a quantitative derivation of generalization bounds, risk functions, or error guarantees. The theoretical foundation that KL divergence regularization and structural diversity objectives necessarily improve OOD generalization remains insufficient.

The paper emphasizes generating “structurally different” counterfactual subgraphs, yet it does not explain how these generated subgraphs maintain semantic or structural validity. Furthermore, no visualization or statistical characterization of the generated samples is provided.

All experiments are conducted on homogeneous graphs for the link prediction task, its applicability to heterogeneous or more complex graph structures remains unverified.

The paper does not sufficiently isolate the contribution of each component within FLEX, as it lacks sensitivity analysis on the number and perturbation strength of generated subgraphs, ablation results for removing the KL constraint, and comparison between joint training with GNNs and independent optimization, making it difficult to determine whether the observed improvements truly originate from the proposed mechanism rather than from generic data augmentation or increased model capacity.

The paper claims efficiency through subgraph-level generation, yet several experiments report OOM or training exceeding 24 hours. There is no analysis of computational complexity, runtime, or hardware requirements.

Key implementation details such as generator architecture, sampling strategy, and loss coefficients are missing. 

The presentation is weak, with numerous grammatical, stylistic, and typographical errors throughout the manuscript. For example, line 178 “it’s”; line 240 “said features”; and line 340 “we an input links…”.

### Questions
This paper provides a set-theoretic argument that counterfactual subgraph generation can enlarge the overlap between training and testing distributions. However, this analysis is mostly qualitative and lacks quantitative characterization of the generalization error or risk bounds. Could the authors provide a more rigorous theoretical or empirical analysis to substantiate the claimed generalization improvement?

Could the authors provide visualization or statistical characterization of the generated subgraphs to demonstrate their structural diversity and semantic consistency?

All experiments are conducted on homogeneous, static graphs for the link prediction task. Have the authors evaluated FLEX on more complex graph settings such as heterogeneous graphs (e.g., MAG [1], DBLP [2])? If not, could they discuss the applicability or potential limitations of the proposed framework under such conditions?

The paper lacks a thorough ablation study to isolate the contribution of each FLEX component. Could the authors supplement experiments that (1) analyze the sensitivity to the number and perturbation strength of generated subgraphs, (2) evaluate performance without the KL-divergence constraint, and (3) compare joint versus independent training of FLEX and GNNs?

Several experiments are marked as “OOM” or “>24h,” but no analysis of runtime, complexity, or hardware requirements is provided. Could the authors include a detailed analysis of computational complexity, training time, memory consumption, and hardware setup to clarify FLEX’s scalability and practical feasibility?

The paper omits important implementation details such as the generator architecture, sampling strategy, and loss coefficients. To enhance reproducibility, could the authors release the implementation and provide detailed hyperparameter settings and training configurations?

[1] Hu, et al. OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs. NeurIPS 2021.
[2] Zhang, et al. Oag-bench: a human-curated benchmark for academic graph mining. KDD 2024.

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
4

### Summary
This paper introduces FLEX, a subgraph generation framework designed to enhance the generalization capability of graph neural networks in out-of-distribution link prediction tasks. Its core idea is to utilize a graph generation model trained collaboratively with the GNN to generate counterfactual subgraphs that differ structurally from training samples but share consistent node features, thereby enabling GNN fine-tuning. The method encourages structural diversity by maximizing the KL divergence between the posterior and prior distributions (with quadratic penalty), while preserving semantic relevance. The authors validate FLEX's effectiveness across multiple synthetic (LPShift) and real-world (ogbl-collab, Amazon Cross-Domain) OOD settings, conducting ablation studies, hyperparameter sensitivity analyses, and structural alignment evaluations.

### Strengths
1. OOD link prediction is a critical bottleneck for GNN deployment, yet existing work predominantly focuses on node/graph classification. This paper explicitly demonstrates that standard OOD generalization methods (e.g., IRM, CORAL) exhibit limited effectiveness in LP tasks (from Table 1), providing empirical evidence and establishing a robust problem motivation.


2. The paper employs k-hop subgraph generation instead of full-graph generation, introduces a labeling trick to enable GNNs to perceive target edges, and utilizes a semi-implicit VAE to balance generation quality and scalability.

3. Formally define meaningful structural differences through counterfactuals and feature condition equivalence, and prove in Appendix B that generated samples can scale training support sets to cover the test distribution from a set-theoretic perspective.

4. The experimental results showed improvement.

### Weaknesses
1. The paper repeatedly cites Pearl's causal framework, yet the FLEX generation process does not model structural equation models or interventions. Instead, it achieves structural differences solely through KL divergence maximization. This approach aligns more closely with diversity sampling in data augmentation than with counterfactuals in a strict causal sense.
2. Gamma significantly impacts performance, but selection relies on grid search. In real-world out-of-distribution scenarios where the test distribution is unseen, how can gamma be adaptively set. If gamma is set too high, resulting in overly sparse graphs, critical structural information may be lost.
3. FLEX has not been directly compared with recent OOD learning methods on LP tasks. While it is noted that these methods do not directly optimize LP generalization, experimental evidence is required to demonstrate that FLEX outperforms these generative OOD approaches.
4. Appendix B's Theorem 1 assumes that the generated sample set S satisfies  \mu(S\cap(U\T))>0. However, in practice, the Lebesgue measure of discrete graph spaces is zero. This assumption holds under continuous approximation but does not address discretization error.

### Questions
please see weakness.

### Soundness
2

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
3

### Summary
This paper proposes FLEX, a framework for improving out-of-distribution (OOD) link prediction in graphs. It co-trains a graph neural network with a semi-implicit variational autoencoder (SIG-VAE) that generates link-conditioned subgraphs—synthetic, counterfactual examples meant to expand the structural diversity of training data. By training on both real and generated subgraphs, the model aims to generalize better to unseen graph structures. Experiments on LPShift and real datasets are conducted to validate the proposed method.

### Strengths
1. The motivation is clear and intuitive.

2. The evaluation is conducted on four datasets with diverse shift schemes, and an ablation study is also provided.

### Weaknesses
1. I’m not fully convinced by the performance improvements shown in Tables 1 and 3. AUC scores below 0.5–0.6 are essentially trivial, indicating near-random predictions. Although FLEX often yields statistically significant gains, improvements such as 50% → 52% provide limited practical utility. The per-dataset breakdown in Table 3 further shows that GCN+FLEX frequently increases AUC while remaining in the trivial range. In some cases (e.g., Backward–PA), FLEX even degrades GCN’s AUC from 73% to 59%, effectively turning a non-trivial score into a trivial one. In addition, the “average gain” reported for NCN+FLEX is somewhat misleading—it should be compared against its backbone (NCN), not against GCN.

2. Table 5 reveals a substantial computational burden introduced by FLEX. The preprocessing and co-training steps are notably expensive, raising concerns about scalability to larger graphs or real-time applications. It is surprising that preprocessing even a small dataset like CiteSeer takes more than six hours.

3. It would strengthen the empirical validation to include results on a more diverse set of GNN backbones, such as GAT or GIN, to test the general applicability of FLEX.

4. It is not obvious how standard OOD generalization methods (e.g., IRM, VREx, GroupDRO) are adapted for link prediction tasks in this work. Providing implementation details or specific design choices for these adaptations would help the reader better assess fairness and reproducibility.

Minor issue:
All parentheses are printed incorrectly.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
