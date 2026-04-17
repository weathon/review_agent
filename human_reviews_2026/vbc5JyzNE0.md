# Geometric Self-Supervised Pretraining on 3D Protein Structures using Subgraphs

- Decision: Reject
- Scores: 2, 2, 2, 4, 4

## Abstract
Protein representation learning aims to learn informative protein embeddings capable of addressing crucial biological questions, such as protein function prediction. Although sequence-based transformer models have shown promising results by leveraging the vast amount of protein sequence data in a self-supervised way, there is still a gap in exploiting the available 3D protein structures. In this work, we propose a pre-training scheme going beyond trivial masking methods leveraging 3D and hierarchical structures of proteins. We propose a novel self-supervised method to pretrain 3D graph neural networks on 3D protein structures, by predicting the distances between local geometric centroids of protein subgraphs and the global geometric centroid of the protein. By considering subgraphs and their relationships to the global protein structure, our model can better learn the geometric properties of the protein structure. We experimentally show that our proposed pretaining strategy leads to significant improvements up to 6%, in the performance of 3D GNNs in various protein classification tasks. Our work opens new possibilities in unsupervised learning for protein graph models while eliminating the need for multiple views, augmentations, or masking strategies that have been used so far.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a self-supervised pretraining method for 3D graph neural networks on protein structures. The method trains a model to predict the distance between subgraph centroids and the global centroid of a protein, where distances are discretized into bins, turning the task into a classification problem. The motivation is to capture hierarchical geometric information from 3D protein data without requiring augmentations or masking strategies. The authors evaluate this pretraining on standard protein tasks (fold classification and enzyme reaction prediction) using models including ProNet, SchNet, and GCN, and report modest improvements compared to baselines like edge-distance prediction and InfoGraph.

### Strengths
1. This work addresses the under-explored area of geometric self-supervision for protein representation learning.
2. The method is simple and broadly applicable to any 3D GNN model.
3. Comprehensive experiments on multiple baselines and datasets are conducted. And the ablation studies are clear to demonstrate the impact of different design choices such as subgraph extraction and centroid computation.

### Weaknesses
1. Task Formulation and Theoretical Motivation. The decision to discretize distance prediction into a classification problem is not sufficiently justified. While the authors argue that classification offers robustness and smoother gradients, this design introduces artificial discontinuities near bin boundaries. For samples whose true distances fall close to bin edges, small measurement noise can result in different classification labels, contradicting the stated robustness goal. Moreover, the regression tasks should be more accurate than classification tasks, which should be more suitable for the statement 'In the context of proteins, however, even a minor change in an amino acid can have a substantial impact on protein function.'

2. Computational Efficiency vs. Practicality. The proposed pretraining uses over 500k AlphaFold structures, yet the paper dismisses the computational cost as “low overhead.” In practice, the subgraph extraction and pretraining phases dominate runtime. If the downstream training (e.g., fold classification) takes much less time compared with pre-training, the claimed efficiency advantage becomes negligible. Therefore, the benefits of the proposed pre-training should be evaluted more comprehensively.
3. Questionable Experimental Consistency. The fold classification results reported for ProNet[1] show noticeable deviation from the original ProNet paper, both for GCN and ProNet itself. For instance, on ProNet, the Fold split accuracy in Table 1 (≈50.1%) diverges significantly from the reference implementation’s reported accuracy (~57–60%). This inconsistency casts doubt on the reproducibility and fairness of comparison. The claim of up to 6% improvement must therefore be interpreted cautiously, as it may arise from different preprocessing or training setups.
4. The concept of using centroid distances or local-global geometric relations is not entirely new. Prior works on 3D GNNs and structural pretraining (e.g., GearNet[2]) have explored geometric distance prediction in related forms. The methodological distinction here. The authors should discuss it and consider it in the baselines.

[1]Wang, Limei, et al. "Learning hierarchical protein representations via complete 3d graph networks." arXiv preprint arXiv:2207.12600 (2022).

[2]Zhang, Zuobai, et al. "Protein representation learning by geometric structure pretraining." arXiv preprint arXiv:2203.06125 (2022).

### Questions
See weeknesses.

I am willing to modify my score after further revision and discussion.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the application of self-supervised pre-training techniques to learn improved representations for 3D protein structures. The authors propose a novel pre-training scheme that leverages the geometric and hierarchical nature of protein graphs by utilizing subgraph-based objectives. The goal is to move beyond simple node or atom masking methods to capture more complex structural relationships, with performance evaluated on several downstream protein classification tasks.

### Strengths
The focus on leveraging 3D geometric information for self-supervised learning in protein representation is a highly promising and critical research area. It correctly identifies the gap left by purely sequence-based models. The proposed pre-training protocol, which aims to capture complex structure beyond local features, is conceptually sound and represents a valuable attempt to advance geometric self-supervised learning (SSL) for biomolecules.

### Weaknesses
1. The paper fails to acknowledge and discuss highly relevant and established work in this domain. Specifically, the omission of GearNet—a powerful and widely recognized protein structure encoder combined with various self-supervised tasks—is a major oversight. Furthermore, the claim that the most recent related work cited is from 2023 suggests a lack of current literature review, making it difficult to assess the paper's position within the current landscape of protein SSL research.

2. The core algorithm, utilizing subgraph-based pre-training, appears conceptually simple and lacks sufficient technical depth or novelty to distinguish it significantly from prior geometric or graph-based SSL methods. The authors must clearly describe the specific technical breakthroughs that differentiate this work from existing frameworks.

3. The empirical validation is severely lacking. The comparison is restricted to only a few, older pre-training baselines like Distance Prediction and InfoGraph. Crucially, the absence of a comparison against strong contemporary models, such as GearNet and other leading methods from 2024/2025 literature, makes it impossible to judge the true effectiveness and competitive performance of the proposed approach.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposed a pretext task for self-supervised learning of protein structure representations using graph neural networks. Namely, the task consist in predicting the distance of the centroid of a subgraph to the centroid of the whole protein, given the embeddings of both the full graph and the subgraph as input. the authors assess the performance of this representation learning strategy on reaction and fold classification.

### Strengths
- The proposed scheme is simple.
- The authors perform some comparison with different pretext tasks, node featurisaton, and architectures, and some ablation study, giving some sense of the potential of the proposed strategy and of the influence of external parameters.

### Weaknesses
- The distance to the protein centroid is not a stable measure. It may be strongly influenced by the addition or removal of a domain, or by conformational changes, in particular rigid transformations of domains with respect to one another. It is not clear to me what the rationale or the intuition behind this choice. Why would that be a good measure of protein geometry? Why would 2 surface residues, located in similar 3D local environments have the same distance to the protein centroid?

- The results are not convincing. More specifically, the gain in performance brought by the proposed pre-text task is not substantial compared to predicting edge distance for instance. What is more, the manuscript does not present comparison with state-of-the-art baselines. A number of prior works have proposed self-supervised pre-training using masking tasks for 3D environments and it is not clear how this work positions itself with respect to them, espcecially in terms of predictive performance. Overall, the experiments look more like a benchmarking of a selection architectures and pre-text tasks than a demonstration of the usefulness of the proposed scheme.

### Questions
- How big are the subgraphs defined by the 2-hop neighbourhood? Could the authors give an idea of the maximal distance (in A) between any pair of nodes in the subgraph?

- How good is the network at performing the pretext task? What is the unit for the y-axis in Figure 2?

- How robust are the conclusions with respect to noisy inputs? How does the model behaves with intrinsically disordered regions?

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
4

### Summary
This paper deals with protein representation learning. Since proteins are a sequence of amino acids that folds to a 3-d structure, prior work have explored representation learning in both the sequence space and the structure space. Sequence based methods like ESM have used transformer based models in a self supervised way, where as structure based methods have used masking on 3D structures. In this work, the authors propose a novel self supervised method to pretrain 3D graph neural networks.  Each amino acid is a node in the graph and the position of the $C_{alpha}$ atom is used as the position of the amino acid. A subgraph is created with 16 nearest neighbors. The residue type is used as node features and the distances are used as edge features.
The self supervised learning task is to predicting the distances between local geometric centroids of the protein subgraphs and the global geometric centroid of the protein. The subgraph computation step is used as a pretraining step and the overhead is incurred only once.

### Strengths
- The method of pretraining that looks into predicting geometric distances between subgraphs is novel and potentially has applications beyond proteins
- The pretraining is done on a large dataset (Alphafold DB)
- They do a reasonable abalation, with different subgraph extraction methods, classification vs regression as a pretraining objecting and different methods to compute the centroids of the graph.
- Full source code has been released to easily replicate the work.
- The paper is well written and clear

### Weaknesses
- The biggest drawback of the paper is that it is evaluated only on a limited set of tasks (Reaction classification and Fold classification). There are several other tasks that can be used to benchmark this method (For example, GO prediction), which are not explored.
- The comparison is only done with GCN and SchNet with different pretraining and featurization methods. There is no comparison with other methods. For example, how does the method perform with sequence only pretraining?

The paper has weak evaluation which makes it hard to judge the significance of the paper. The authors should perform a more thorough evaluation. For example, see Zhang. et al, PROTEIN REPRESENTATION LEARNING BY
GEOMETRIC STRUCTURE PRETRAINING, ICLR 2023 and Yang et al. Convolutions are competitive with transformers for protein sequence pretraining, Cell Systems, for some more baselines.

### Questions
- Can this method be used for protein fitness prediction tasks from the proteingym benchmark? If yes, why was this not evaluated?
- Please also address the concerns in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new self-supervised learning approach designed specifically for learning from 3D protein structures using graph neural networks (GNNs). Rather than relying on standard pretraining techniques like node or edge masking or contrastive learning with data augmentations, the authors propose a more geometrically informed task, which is predicting the Euclidean distance between the centroid of a local protein subgraph and the centroid of the full protein. The idea is rooted in the observation that protein function is deeply connected to 3D geometry, and learning geometric relationships among substructures may offer richer representations than traditional pretraining signals. This work presents a compelling direction for pretraining in structural biology.

### Strengths
1-	Authors have addressed an interesting challenge that can be beneficial not only for protein classification, but also for other domains where 3D graphs are used. Moreover, it is model agnostic and is applicable to various 3D GNNs
2-	Authors have used a list of ablation studies to evaluate the impact of different components, i.e., subgraph extraction method, distance prediction, and centroid computation methods.
3-	The method is integrated into the ProteinWorkshop library, fostering reproducibility and extension.

### Weaknesses
1-	The complexity section does not really explain the time complexity of the proposed pre-training method. Sounds more like a justification that it is “feasible”. A detailed, precise analysis is expected.
2-	Authors have reported some specific architecture hyperparameters in the 3.1 notation and architecture sections without properly explaining the rationale. For instance, the k is set to 16 for nearest neighbours, yet this is not explained. It is better if authors only mention the parameter k in methodology and report what value they set in experiments, and explain how they selected that. Moreover, they have applied a sum pool layer at the very end and just mentioned that they have used it without explanation. If authors are following practices that the original architecture has used, they should cite it properly and back it up with reasoning; otherwise, this may look like arbitrary modifications.
3-	I recommend that authors have another round of proofreading. There are some minor issues:
Line 190: “we use also use a” > “we also use a…”
Line 374: “Compared to models without pretraining and those using edge distance pretraining, the subgraph distance method consistently yields higher performance.” -> This is not true, In table 1, Fold task, family, Edge distance pretraining is performing best using ProNet. Using the SchNet model, on fold and super-family, Edge distance is the best (except for fold accuracy).

### Questions
1-	Is k=16 for nearest neighbours advised by the original architectures you are building, based on, or a hyperparameter? If it is a hyperparameter, a sensitivity analysis can demonstrate how the value can affect the performance and robustness of the model with respect to it.

### Soundness
3

### Presentation
2

### Contribution
3
