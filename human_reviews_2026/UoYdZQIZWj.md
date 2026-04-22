# KGOT: Unified Knowledge Graph and Optimal Transport Pseudo-Labeling for Molecule-Protein Interaction Prediction

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 2

## Abstract
Predicting molecule-protein interactions (MPIs) is a fundamental task in computational biology, with crucial applications in drug discovery and molecular function annotation. However, existing MPI models face two major challenges. First, the scarcity of labeled molecule-protein pairs significantly limits model performance, as available datasets capture only a small fraction of biological relevant interactions.
Second, most methods rely solely on molecular and protein features, ignoring broader biological context—such as genes, metabolic pathways, and functional annotations—that could provide essential complementary information. To address these limitations, our framework first aggregates diverse biological datasets, including molecular, protein, genes and pathway-level interactions, and then develop an optimal transport-based approach to generate high-quality pseudo-labels for unlabeled molecule-protein pairs, leveraging the
underlying distribution of known interactions to guide label assignment. By treating pseudo-labeling as a mechanism for bridging disparate biological modalities, our approach enables the effective use of heterogeneous data to enhance MPI prediction. We evaluate our framework on multiple MPI datasets including virtual screening tasks and protein retrieval tasks, demonstrating substantial improvements over state-of-the-art methods in prediction accuracies and zero shot ability across unseen interactions. Beyond MPI prediction, our approach provides a new paradigm for leveraging diverse biological data sources to tackle problems traditionally constrained by single or bi-modal learning, paving the way for future advances in computational biology and drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In the paper, the authors studied methods for predicting ligand-target binding affinity as link prediction problems. In order to resolve the issue of lacking large-scale training data. The authors proposed methods for augmentation of labelled data with useful information from multi-modal knowledge graphs. 

The proposed method following 4 different steps:

+ use a small labelled datasets to train a predictive model that predicts the probability of binding between any pair of molecule and protein.

+ use the given model to create a pseudo-label data on a larger unlabelled graph. The pseudo labels are enforced to give overall consistency prediction on the entire unlabelled graph by Optimal Transport and the regularization of the similarity between nodes on the latent spaces.

+ the pseudo-labelled graph together augmented with the edges of the large multimodal knowledge graph is then used to train a final link prediction model

In order to evaluate the proposed approach the author compared there KGOT method with baseline methods that do not use external information on the DUDe and  LIT-PCBA with leakage removal from the training graphs. Their proposed approach yields better results then the baseline methods.

In another experiments for link prediction, KGOT was used in addition to other graph embedding methods. the experimental results show that the data augmentation procedure in KGOT via OT helps simple graph embedding methods improve linking prediction accuracy.

### Strengths
An interesting idea was proposed regarding using OT to control the consistency of pseudo-labels overall.

The paper was well written and easy to follow.

### Weaknesses
The idea of using multimodal knowledge graphs to augment training data for ligand-target affinity prediction is not a new idea. The given idea has been proposed and studied in the following works:

+ N. Zhang, Z. Bi, X. Liang, S. Cheng, H. Hong, S. Deng, Q. Zhang, J. Lian, and H. Chen. Ontoprotein: Protein
pretraining with gene ontology embedding. In International Conference on Learning Representations, 2022.

+ H.-Y. Zhou, Y. Fu, Z. Zhang, B. Cheng, and Y. Yu. Protein representation learning via knowledge enhanced
primary structure reasoning. In The Eleventh International Conference on Learning Representations, 2023.

+ Lam H. T., Sbodio M. L., Martínez Galindo M., et al. “Otter-Knowledge: benchmarks of multimodal knowledge graph representation learning from different sources for drug discovery.” (2023).

+ Ye Q., Hsieh C-Y., Yang Z., et al. “A unified drug–target interaction prediction framework based on knowledge graph and recommendation system.” Nature Communications 12:1 (2021).

I think the authors have not discussed those related work carefully and compare to those approaches.


The results on DUDe and LIT-PCBA are interesting but I think the authors should compare to the above approaches on the following standard benchmarks:

+ TDC DTI https://tdcommons.ai/benchmark/dti_dg_group/overview/

+ DAVIS

+ KIBA

### Questions
Could you please compare your approach with the given related works listed in the weakness section?

 Could you please do more experiments on the datasets listed in the weakness section?

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
3

### Summary
This study presents a new method for predicting protein-molecule interactions. The method relies on extracting the knowledge from a multimodal knowledge graph. The construction and composition of the multimodal knowledge graph is not novel. The two main novel contributions are: using optimal transport-based pseudo-labeling strategy to leverage a large unlabelled dataset and augment the original knowledge graph. The method shows robust improvement over the SOTA.

### Strengths
1. The optimal transport-pseudo labelling strategy seems a good contribution to the overall field and opens the door to creating more sophisticated augmented knowledge graphs.
2. The results are shown with some measure of the performance dispersion, it is unclear which or where it is derived from, but it allows for some determination of the statistical significance of the results.
3. The ablation study is comprehensive and convincingly demonstrates that the different components of the method improve prediction accuracy.

### Weaknesses
1. It is unclear what the error represents in Tables 1 and 2.

### Questions
1. Why did you not use a pre-made KG like PrimeKG, for example?

### Soundness
3

### Presentation
3

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
This paper proposed a unified framework KGOT that integrates knowledge graph and optimal transport to generate high-quality pseudo-labels for unlabeled molecule-protein pairs in molecule-protein interactions (MPIs) prediction tasks.

### Strengths
S1: This paper is to leverage large-scale multimodal knowledge graphs and propose an optimal transport-based pseudo-labeling strategy for the MPIs prediction.

S2: In experiment part, the proposed KGOT outperforms existing MPIs prediction methods in terms of AUROC, early recognition metrics, and generalization to unseen interactions.

### Weaknesses
W1: A primary weakness of KGOT is its reliance on the critical yet potentially unstable step of pseudo-label generation via optimal transport. The quality of the entire approach hinges on the assumption that the optimal transport mechanism can accurately infer the underlying distribution of known interactions to assign reliable labels to unknown molecule-protein pairs.

W2: Aggregating molecular, protein, gene, and pathway-level information requires sophisticated fusion techniques to handle the disparate scales, formats, and sparsity levels of each modality. However, the integration of diverse biological data introduces significant complexity and potential challenges in data harmonization and model interpretation.

W3: Without validation on truly de novo targets or wet-lab confirmation, the practical utility and robustness of the framework for unseen interactions, iterative drug discovery pipeline remain unproven.

### Questions
- What is the specific cost function used in the Optimal Transport plan? Is it based on the model’s predicted scores, molecular/protein embeddings, or a combination?
- How is the marginal distribution for the OT problem defined? Are uniform distributions assumed for molecules and proteins, or is there a prior (e.g., based on node degree in the KG) that biases the label assignment?
- How is the entropy regularization parameter chosen and tuned? This parameter critically balances between fitting the data and the smoothness of the transport plan, directly impacting pseudo-label quality.
- What is the architecture of the initial scoring model (Step 2)? Is it a simple neural network, or does it already incorporate some KG information?
- How is the knowledge graph itself encoded in Step 4? Are you using TransE, ComplEx, a Graph Neural Network (GNN), or another Knowledge Graph Embedding (KGE) method? The choice here significantly affects the model’s ability to capture complex relational paths.
- The mutual retrieval objective suggests a dual-encoder architecture. How are the molecule and protein encoders designed and aligned? Is the contrastive loss used, and if so, what is the strategy for mining hard negatives?
- How do you prevent data leakage between the small labeled dataset (Step 2) and the large unlabeled dataset (Step 3)? If proteins/molecules from the labeled set appear in the unlabeled KG, it could artificially inflate performance.
- What is the criteria for a high-quality pseudo-label? Is there a threshold on the OT-assigned probability, and how is this threshold determined?
- Is the proposed KGOT process iterative? In other words, is the KG-augmented model (Step 4) used to re-score pairs and generate new, improved pseudo-labels in a self-training loop?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a framework to predict molecule–protein interactions. It generates high-quality pseudo-labels to leverage diverse biological modalities.

### Strengths
The paper proposes a unified framework that aims to integrate biological entities such as pathways and genes.

### Weaknesses
(1) The proposed framework mainly combines existing methods without introducing any new modeling components or theoretical insights.

(2) The performance is not compared with highly relevant works such as KG-MTL and BioKDN.

### Questions
Could the authors compare the performance and clarify the advantages over relevant KG-based methods such as KG-MTL and BioKDN?

### Soundness
1

### Presentation
1

### Contribution
1
