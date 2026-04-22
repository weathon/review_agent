# Directed Semi-Simplicial Learning with Applications to Brain Activity Decoding

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Graph Neural Networks (GNNs) excel at learning from pairwise interactions but often overlook multi-way and hierarchical relationships. Topological Deep Learning (TDL) addresses this limitation by leveraging combinatorial topological spaces, such as simplicial or cell complexes. However, existing TDL models are restricted to undirected settings and fail to capture the higher-order directed patterns prevalent in many complex systems, e.g., brain networks, where such interactions are both abundant and functionally significant. To fill this gap, we introduce Semi-Simplicial Neural Networks (SSNs), a principled class of TDL models that operate on semi-simplicial sets---combinatorial structures that encode directed higher-order motifs and their directional relationships. To enhance scalability, we propose Routing-SSNs, which dynamically select the most informative relations in a learnable manner. We theoretically characterize SSNs by proving they are strictly more expressive than standard graph and TDL models, and they are able to recover several topological descriptors. Building on previous evidence that such descriptors are critical for characterizing brain activity, we then introduce a new principled framework for brain dynamics representation learning centered on SSNs. Empirically, we test SSNs on 4 distinct tasks across 13 datasets, spanning from brain dynamics to node classification, showing competitive performance. Notably, SSNs consistently achieve state-of-the-art performance on brain dynamics classification tasks, outperforming the second-best model by up to 27\%, and message passing GNNs by up to 50\% in accuracy.  Our results highlight the potential of topological models for learning from structured brain data, establishing a unique real-world case study for TDL. Code and data are uploaded as supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper introduces Semi-Simplicial Neural Networks (SSNs), which learn on directed, higher-order structures (semi-simplicial sets) rather than just pairwise edges. A scalable variant (Routing-SSN) learns which relations to use. The authors prove SSNs are more expressive than standard GNNs and existing topological models and can recover key topological descriptors of brain activity. On simulated neocortical stimulus-decoding tasks, SSNs achieve state-of-the-art results and sizable gains over baselines.

### Strengths
- The paper has a well-motivated focus on directionality and multi-way interactions, important for brain data.
- General framework that subsumes GNNs, directed GNNs, and simplicial networks.
- Theoretical results on WL-expressivity and recovery of neuro-topological invariants.
- Strong empirical gains on brain-dynamics classification.

Overall, I find the paper to be strong. It introduces a novel framework that is well-grounded in theory, effectively subsuming several pioneering baselines. The experimental evaluation also appears comprehensive, covering a wide range of datasets, tasks, and key baselines, which collectively serve as an implicit ablation study for the proposed architecture.

That said, I am not deeply familiar with the field of brain activity modeling, and from my view it seems that the baselines considered are primarily taken from [4] (e.g., GNNs, DeepSets, MPSNN, etc.), potentially missing other relevant "high order message passing approaches" approaches such as [1, 2, 3]. Given my limited expertise in this domain of brain activity—particularly regarding datasets and evaluation protocols—I would prefer to calibrate my final assessment after reviewing feedback from other reviewers who may have more specialized knowledge in this area.

**References:** 

[1] Cin++: Enhancing topological message passing. Giusti et.al. 2023

[2] Cycle invariant positional encoding for graph representation learning. Yan et al. 2024

[3] Attending to topological spaces: The cellular transformer. Ballester et. al 2024

[4] Position: Graph learning will lose relevance due to poor benchmarks Bechler-Speicher et al., 2025

### Weaknesses
- There is no comparison with other topological or higher-order message-passing baselines (e.g., [1, 2, 3]), if they are applicable on these tasks. If these methods are indeed not relevant or comparable in this context, it would be helpful for the authors to clarify why that is the case.
- Minor weakness: the paper lacks an ablation study showing which types of relations contribute most to performance or how sensitive results are to the choice of relations.

**References:**

[1] Cin++: Enhancing topological message passing. Giusti et.al. 2023

[2] Cycle invariant positional encoding for graph representation learning. Yan et al. 2024

[3] Attending to topological spaces: The cellular transformer. Ballester et. al 2024

### Questions
1. How should practitioners pick the top-k in Routing-SSN beyond grid search?
2. Can routing scores or other signals identify which directed motifs drove a prediction?

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
3

### Summary
The paper proposes Semi-Simplicial Neural Networks (SSNs), a TDL architecture that operates on semi-simplicial sets rather than classical simplicial complexes, enabling joint modeling of directionality and higher-order structure. It further introduces Dynamical Activity Complexes (DACs) to represent time-varying neuronal co-activation and provides formal results showing SSNs can recover a broad class of topological invariants. Experiments span brain-dynamics classification, traffic edge-regression, and node classification.

### Strengths
- The paper gives clean algebraic definitions, distinguishes orientation vs directionality, and proves that SSNs strictly contain the expressive power of GNNs/Dir-GNNs/MPSNNs while being able to recover key invariants.

- Using semi-simplicial sets allows the model to treat different vertex orderings as distinct simplices, preserving directionality information that is lost in traditional TDL architectures. 

- The experiments span three application domains: brain dynamics, traffic flow regression, and node classification. The results are consistent with the theoretical claims, showing that the advantage of SSNs increases with the strength of directionality in the data.

### Weaknesses
1. Although the paper presents DACs as a novel construct, similar formulations of dynamic higher-order connectivity already exist in prior work such as [1]. The paper should clarify what is genuinely new.

2. The semi-simplicial formalism is conceptually close to combinatorial complexes [2] and several Combinatorial-Complex Neural Networks (CCNNs) already exist [3], [4]. The paper lacks a conceptual and empirical comparison, leaving ambiguity as to whether SSNs represent a subset, superset, or merely a re-parameterization of these frameworks.

3. The latest higher-order model compared is MPSNN (2021). More recent approaches such as combinatorial-complex networks, cell-complex GNNs, or hypergraph GNNs are not included.

4. In the brain-related experiments, 2-simplices (triangles) are included, but in edge regression and node classification, only 1-simplices are used. The absence of experiments with higher-order (>2) simplices weakens the empirical evidence supporting the claimed expressivity advantages.

[1] Higher-order connectomics of human brain function reveals local topological signatures of task decoding, individual identification, and behavior (Nature Communications, 2024)

[2] Combinatorial Complexes: Bridging the Gap Between Cell Complexes and Hypergraphs

[3]Topological Deep Learning: Going Beyond Graph Data

[4] TopoTune: A Framework for Generalized Combinatorial Complex Neural Networks

### Questions
Can the proposed SSNs be generalized to combinatorial complexes with cells of dimension greater than two?

Can DACs be extended from binary activation to real-valued activity?

### Soundness
3

### Presentation
2

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
This paper introduces Semi-Simplicial Neural Networks (SSNs), a new class of topological deep learning models for directed higher-order structures. SSNs operate on semi-simplicial sets using face-map–induced relations to propagate messages between simplices, extending both GNNs and simplicial neural networks. A scalable variant, Routing-SSN, dynamically selects relevant relations.
The authors prove that SSNs are strictly more expressive than directed GNNs and simplicial networks under the WL hierarchy and can recover a richer family of topological invariants relevant to brain activity. Experiments on 13 datasets show large gains over prior models (up to 50% over GNNs and 27% over other TDL baselines).

### Strengths
- The paper is well written and easy to follow.

- The paper provides strong theoretical results on WL-expressivity, permutation equivariance, and invariant recovery, establishing clear theoretical advantages over prior models. 

- The integration with brain dynamics modeling via Dynamical Activity Complexes (DACs) connects deep learning with neurotopology in a rigorous, data-driven way, replacing handcrafted invariants. 

- SSN is able to achieve large accuracy gains (up to 50% over GNNs) on challenging stimulus classification tasks, validating the theory with well-controlled baselines and consistent robustness. 

- The Routing-SSN mechanism provides a practical path to scaling directed topological models while retaining expressivity and efficiency.

### Weaknesses
- Recent literature has shown that transformers can be considered message passing neural networks [1]. However, results using multi-head attention over node features are missing. Since SSNs generalize message passing to relation-aware updates, such attention-based comparisons would clarify whether the proposed relational framework yields benefits beyond architectural scaling, or the transformer model is able to learn relations.

- The ablation on relation classes (e.g., face-map–induced vs. interdimensional vs. intradimensional) is not reported. Understanding which classes contribute most to performance would help assess the inductive bias of the relational algebra and the necessity of routing.

- Several of the paper’s most important theoretical results, including the corollaries, are deferred to long appendices. Condensing or summarizing them in the main text would improve accessibility and emphasize the technical depth.

- On standard node classification benchmarks (e.g., cora), SSNs do not outperform existing baselines, and these graphs are relatively small. This weakens the general claim that SSNs are broadly superior rather than domain-specific to brain networks. Results on datasets like OGB would emphasize the scalability and generalizability of the model.

[1] Joshi, Chaitanya K. "Transformers are graph neural networks." arXiv preprint arXiv:2506.22084 (2025).

### Questions
- Which face-map–induced relation classes (e.g., interdimensional, intradimensional, directed adjacency) contribute most to performance?

- The framework mainly uses binary activations for brain data. How would SSNs handle continuous-valued features or multimodal signals?

- How will the routing behave under noise or shuffled connectivity?

### Soundness
3

### Presentation
3

### Contribution
3
