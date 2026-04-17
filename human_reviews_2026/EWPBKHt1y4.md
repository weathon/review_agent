# Hierarchical Molecular Representation Learning via Fragment-Based Self-Supervised Embedding Prediction

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Graph self-supervised learning (GSSL) has demonstrated strong potential for generating expressive graph embeddings without the need for human annotations, making it particularly valuable in domains with high labeling costs such as molecular graph analysis. However, existing GSSL methods mostly focus on node- or edge-level information, often ignoring chemically relevant substructures which strongly influence molecular properties. In this work, we propose Graph Semantic Predictive Network (GraSPNet), a hierarchical architecture that predicts both node and semantically meaningful fragments of a graph in the embedding space. GraSPNet decomposes molecular graphs into meaningful fragments without relying on predefined chemical vocabulary and learns graph representations through message-passing graph neural networks. It further captures fragment-level semantics by encoding fragment information and modeling interactions through node-fragment and fragment-fragment message passing. By performing masked prediction of node and fragment features in semantic space, GraSPNet captures structural information at multiple resolutions. Experiments show that GraSPNet is both expressive and generalizable, outperforming existing state-of-the-art methods on multiple molecular property prediction benchmarks in transfer learning settings. The code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present GraSPNet (Graph Semantic Predictive Network), a hierarchical SSL framework for molecular representation learning, modeling semantically meaningful molecular fragments, allowing the network to capture hierarchical and chemically relevant substructures without relying on a predefined vocabulary.
It first decomposes molecular graphs into fragments, encodes them using message-passing GNNs, and models interactions between nodes and fragments through node-fragment and fragment-fragment message passing. GraSPNet employs masked embedding prediction at both node and fragment levels to jointly capture fine-grained and high-level semantic dependencies.

### Strengths
- The introduction of both node- and fragment-level prediction enables multi-resolution representation learning, which is biologically and chemically intuitive.

- To avoid predefined chemical substructures makes the approach flexible and domain-agnostic, Self-Supervised and Vocabulary-Free allow it to generalize across molecular datasets.

### Weaknesses
- The idea of leveraging molecular fragments for hierarchical or semantic representation learning is not entirely new. Recent studies, such as GraphFG, S-CGIB, and other fragment- or motif-based molecular pretraining methods, have already explored similar directions.

- While the empirical results are strong, there is no theoretical discussion about the expressive power of the proposed hierarchical GNN relative to existing architectures (e.g., WL hierarchy).

- Both node- and fragment-level masked prediction could lead to overlapping learning signals; an ablation study is needed to disentangle their respective contributions.

### Questions
- From a theoretical perspective, does GraSPNet offer higher expressive power than 1-WL GNNs? For instance, can fragment-level reasoning distinguish certain non-isomorphic molecular structures that node-level GNNs cannot?

- How GraSPNet captures chemically meaningful substructures?

- To me, the prediction task is somehow not related to the molecular graph structure preservation, as they are from different perspectives. It is difficult to say that the model could transfer the correct patterns well to downstream datasets. Is there any theoretical analysis or proof that a prediction task is sufficient for an SSL task (in both structural and semantic preservation)?

### Soundness
3

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
This paper introduces GraSPNet (Graph Semantic Predictive Network), a framework that addresses the neglect of chemical substructures in existing graph self-supervised learning methods. It uses a fragmentation technique to decompose molecules into rings, paths, and articulation points, constructing a multi-level graph structure. GraSPNet employs a self-supervised task similar to MAE, masking nodes and fragments, and predicts their embeddings using a context encoder.

### Strengths
1. The model achieves state-of-the-art or near-state-of-the-art performance on several challenging molecular property prediction benchmarks, particularly in transfer learning settings, demonstrating the effectiveness of its pretraining strategy and strong generalization ability.

2. The model explicitly models information transfer between atom-fragment and fragment-fragment, enabling it to capture higher-level chemical semantics that standard GNNs may overlook.

### Weaknesses
1. The fragmentation strategy is a fixed decomposition method based on heuristic rules. It remains unclear whether this decomposition approach is optimal for all downstream tasks. For example, some tasks may require substructure partitions with different granularities or types. Simply applying this method of partitioning could potentially disrupt the information carried within the molecular graph.

2. The "fragment-based" approach is not novel; utilizing substructures or motifs to enhance molecular graph representation learning has long been an established research direction in cheminformatics and graph machine learning.

3. Masking data at the fragment level directly could potentially disrupt the semantics represented by the data. After all, the premise of contrastive learning is to ensure that the semantics to be learned in the positive samples remain unchanged.

### Questions
1. Could you explain whether the fragmentation approach used is reasonable and preserves the molecular property features?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents GraSPNet, a hierarchical self-supervised framework for molecular graphs. It decomposes molecules into rings, paths, and articulation points as semantic fragments and jointly predicts node- and fragment-level embeddings through dual-channel message passing (node→fragment and fragment→fragment). Experiments on MoleculeNet benchmarks show consistent gains over prior self-supervised methods.

### Strengths
1. **Programmatic Fragmentation Strategy:** The model partitions molecules into structural subgraphs (rings, paths, articulation points) through a deterministic graph algorithm, ensuring reproducibility without predefined vocabularies.

2. **Hierarchical Message Passing:** The dual-channel design captures both local atomic and global fragment semantics.

3. **Comprehensive Experiments:** Evaluated on 8 classification and 3 regression benchmarks, showing consistent performance gains over GraphCL, GraphMAE, and MGSSL.

### Weaknesses
1. **Limited Novelty:** The core idea of hierarchical molecular representation has been explored in [1][2][3], and similar fragment-based or hierarchical pretraining exists. GraSPNet mainly integrates known techniques (fragment-level modeling + masked prediction + hierarchical GNN).

2. **Heuristic Fragment Extraction:** Although the method avoids chemical vocabularies, it still depends on hand-crafted structural heuristics (rings, paths, articulation points). A comparison with functional groups [4] or principal subgraph mining [5] is missing.

3. **Insufficient Analysis of Chemical Validity:** There is no visualization or quantitative evidence showing that extracted fragments correspond to meaningful chemical motifs.

References

[1] Li, Yuquan. Learning Hierarchical Interaction for Accurate Molecular Property Prediction. (2025).
[2] Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. Hierarchical Generation of Molecular Graphs Using Structural Motifs. ICML, 2020.
[3] Luong, Kha-Dinh, and Ambuj K. Singh. Fragment-Based Pretraining and Finetuning on Molecular Graphs. NeurIPS 36 (2023): 17584–17601.
[4] Chen, Fangying, Junyoung Park, and Jinkyoo Park. A Molecular Hyper-Message Passing Network with Functional Group Information. arXiv:2106.01028 (2021).
[5] Kong, Xiangzhe, et al. Molecule Generation by Principal Subgraph Mining and Assembling. NeurIPS 35 (2022): 2550–2563.
[6] Zhang, Yikun, et al. Atomas: Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Generation. ICLR 2025.

### Questions
1. Could the authors integrate a **learned principal-subgraph** extraction mechanism (as in PS-VAE [5]) instead of fixed heuristics to enhance adaptability?

2. How does GraSPNet’s hierarchy differ from Atomas [6], which also performs automatic atom→fragment→molecule decomposition?

3. Have the authors analyzed the **distribution or diversity** of extracted fragments to ensure chemical representativeness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GraSPNet, a novel self-supervised learning framework for molecular graphs that learns hierarchical representations by jointly predicting node and fragment-level embeddings. The goal is to improve molecular property prediction (for downstream tasks). This is done by capturing semantically rich pre-defined substructures (e.g., rings, paths, articulation points) during pretraining. Authors have shown that their proposed fragmentation strategy can effectively capture richer semantics, which will later be used for training their model.

### Strengths
1-	The paper proposed a novel approach to capture both node and fragment-level semantics. The proposed GraSPNet architecture introduces a dual-level semantic prediction mechanism, which is underexplored in graph self-supervised learning (GSSL).
2-	The proposed fragmentation strategy looks promising. Moreover, the WL-test example in Figure 2 clearly demonstrates how fragment-level abstraction helps distinguish structurally similar but semantically distinct molecules. This is a strong theoretical motivation.
3-	Authors have conducted an inclusive ablation study with respect to fragmentation. They have tested with and without fragmentation to demonstrate the effect of their proposed fragmentation strategy. Moreover, they have evaluated different fragmentation strategies to study the effectiveness of their proposed method compared to MGSSL, S-CGIB, and HiMOL methods.

### Weaknesses
1-	The baselines are outdated. Especially the graph contrastive learning methods. Here a list of GCL methods that have been published more recently and outperform current baselines:
GRACE: Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S., & Wang, L. (2020). Deep graph contrastive representation learning. arXiv preprint arXiv:2006.04131.
GCA: Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S., & Wang, L. (2021, April). Graph contrastive learning with adaptive augmentation. In Proceedings of the web conference 2021 (pp. 2069-2080).
GREET: Liu, Y., Zheng, Y., Zhang, D., Lee, V. C., & Pan, S. (2023, June). Beyond smoothing: Unsupervised graph representation learning with edge heterophily discriminating. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 4, pp. 4516-4524).
EPAGCL: Xu, Y., Huang, S., Zhang, H., & Li, X. (2025, April). Why does dropping edges usually outperform adding edges in graph contrastive learning?. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 20, pp. 21824-21832).
2-	The authors have not included any analysis of training or inference cost as the graph size increases. Fragment graphs can become large and dense, but memory/runtime implications are not discussed.
3-	The paper needs an additional round of proofreading. There are several grammatical errors:
a.	Line 069: “at three semantic levels—node (atoms), fragment (e.g., functional groups)” -> “at three semantic levels: node (atoms), fragment (e.g., functional groups)”
b.	Line 182: GNNS -> GNNs
c.	Line 187: “can be more powerful than 2-WL test in distinguish graph isomorphic." -> “can be more powerful than the 2-WL test in distinguishing graph isomorphisms.”
d.	Ling 265: “to each nodes and fragments” -> “to each node and fragment”
e.	Line 811: “The code will be release upon acception.” -> “The code will be released upon acceptance.”

### Questions
a.	How do different fragmentation rules contribute to performance? It would be interesting to have an ablation study comparing different fragmentation schemes (e.g., rings-only, no articulation).

### Soundness
3

### Presentation
2

### Contribution
2
