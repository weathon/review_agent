# DHENN: A Deeper Hybrid End-to-end Neural Network for Highly Accurate Drug-Drug Interaction Events Prediction

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
Accurate prediction of drug-drug interactions (DDIs) is crucial for therapeutic safety yet poses a substantial challenge due to complex pharmacodynamics. Traditional DDI prediction methods often falter for three reasons. First, they simplify dependency structures among entities (e.g., drugs, targets, enzymes, and transporters) in bipartite networks, falling short in modeling their high-order interactions. Second, the over-smoothing effects constrain the depth of the adopted neural networks, thereby limiting their learning capacity. Third, they mainly decouple the stages of representation and prediction, leading to suboptimal solutions that overlook the potential for ground-truth DDIs to refine embedding generation. In response, this paper proposes Deeper Hybrid End-to-end Neural Network (DHENN), which integrates a Multimodal Knowledge Graph (MKG) with a Prediction-Enhanced Cascading Network (PECN) in an end-to-end learning manner. Specifically, MKG captures higher-order relationships across entities, offering a holistic view of DDIs. PECN mitigates over-smoothing by incorporating shallow embeddings into deeper layers, preserving node-level diversity. The endto- end learning manner guarantees that the representation learning and predictive modeling of MKG and PECN are formulated into a unified learning objective. Extensive experiments substantiate that DHENN outperforms thirteen competitors on two real-world DDI datasets.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes an end-to-end DDI prediction model to enhance the high-order interactions between biomedical entities and the learning capacity of adopted neural networks. The DDI prediction task is interesting. However, the motivations are convincing enough. The experimental results look positive, but the key KG-based prediction methods are missing.

### Strengths
The paper is well-structured.

### Weaknesses
Novelty:
The novelty is limited and this is an incremental work based on MKG-FENN [1]. The authors should clarify the key differences between them.

Motivation:
This paper proposes three motivations for the traditional DDI methods' failing short in the high-order interactions between biomedical entities, the limited learning capacity, and the decoupled learning framework. For the first motivation, this paper considers the relational semantics with topological structures as high-order interactions, however, some works have considered this information[2-5]. For the second motivation, I suggest that the authors elaborate on what the advantages of deep neural networks are for DDI prediction, and from the experimental results, there is no huge advantage of using deep neural networks over the previous methods. For the third motivation, these works [2-6] are all end-to-end methods for DDI prediction. I suggest the authors provide more evidence or analysis to demonstrate their strengths.

Experiment:
For the baseline methods, many KG-based works [1-5] are missing. So the experimental results are not convincing. The authors can further discuss these methods and explain how your approach differs from or improves upon these methods.

Some minor suggestions, the authors can provide more discussion on these.
1. In the Method section, Equation (5) lacks sufficient explanation of the symbol, as the symbol σ is used without specifying the type of activation function it represents.
2. The example "986" is used as a SMILES encoding for a drug, but it is not a typical SMILES sequence. A more representative SMILES sequence is recommended.
3. The Related Work section mentions four categories of methods, but the main experiment’s baseline selection does not include models from each of these four categories.
4. The main experiment analysis is insufficient, only presenting results without in-depth discussion on the reasons for performance improvements or why it outperforms certain types of methods. There is also no analysis of the strengths and weaknesses of models other than the proposed one. The authors can provide more analysis.
5. The descriptions of Tasks 2 and 3 in the experiment section are not detailed enough, and the overview and differences between the two tasks are unclear.
6. Table 4 lacks an explanation, and there is no analysis of the experimental results for different combinations of the four entity types.

[1] Wu, D., Sun, W., He, Y., Chen, Z., & Luo, X. (2024, March). MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate Drug–Drug Interaction Prediction. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 9, pp. 10216-10224).

[2] Chen, Y., Ma, T., Yang, X., Wang, J., Song, B., & Zeng, X. (2021). MUFFIN: multi-scale feature fusion for drug–drug interaction prediction. Bioinformatics, 37(17), 2651-2658.

[3] Yu, Y., Huang, K., Zhang, C., Glass, L. M., Sun, J., & Xiao, C. (2021). SumGNN: multi-typed drug interaction prediction via efficient knowledge graph summarization. Bioinformatics, 37(18), 2988-2995.

[4] Wang, Y., Yang, Z., & Yao, Q. (2024). Accurate and interpretable drug-drug interaction prediction enabled by knowledge subgraph learning. Communications Medicine, 4(1), 59.

[5] Zhang, Y., Yao, Q., Yue, L., Wu, X., Zhang, Z., Lin, Z., & Zheng, Y. (2023). Emerging drug interaction prediction enabled by a flow-based graph neural network with biomedical network. Nature Computational Science, 3(12), 1023-1033.

[6] Lin, X., Quan, Z., Wang, Z. J., Ma, T., & Zeng, X. (2020, July). KGNN: Knowledge Graph Neural Network for Drug-Drug Interaction Prediction. In IJCAI (Vol. 380, pp. 2739-2745).

### Questions
1. In the Method section, it is noted that MKG includes various entities and relationships. Does the model have a mechanism to adaptively assign different weights to these relationships?
2. Can the model handle cases where data is incomplete or certain modality information is missing?
3. The paper mentions using a fixed neighborhood size for sampling. Has dynamic neighborhood sampling been considered, where sampling size could be adjusted based on node importance or graph structure? Would this improve model performance on DDI networks with varying structural complexities?

### Soundness
2

### Presentation
2

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
This paper presents DHENN, a deep learning framework for predicting drug-drug interactions (DDIs). DHENN tries to address three main limitations current DDI suffer from:
1) Simplification of dependency structures, failing to model high-order interactions. They solve this by building a multimodal knowledge graph, from where complex interactions can be learned.
2) over-smoothing effects of neural network approaches. They solve this by implementing a cascading network that can leverage shallow embeddings and prevent over-smoothing.
3) decoupling of representation and prediction, yielding suboptimal solutions. They adress this by training both representation and prediction modules in an end-to-end manner.
The authors perform several experiments across various DDI datasets and state-of-the-art prediction models, showing improved performance in many occasions. The paper is well written and presented.

### Strengths
1- The manuscript is clear and well explained and the techniques behind the developed model present novelty.

2- The performed analysis is extensive, across several state-of-the-art methodologies and datasets.

3- The presented methodology proofs to over-perform, in a statistically significant manner, many state-of-the-art DDI approaches.

4- They present an extensive ablation study tackling several important concepts of the model, such as the over-smoothing effect or the multimodal graph importance, which adds robustness to the work.

### Weaknesses
1- The paper would benefit from an analysis of the model’s computational performance, such as as training time and memory/GPU usage.

2- It is unclear how different drugs perform based on the connectivity on the multimodal graph. Can this model be applied to new drugs that may not have connections within the multimodal graph?

3 - Although the authors present an analysis based on new drugs, the paper could benefit from a further strict evaluation on generalization capabilities. This could encompass train/test folds to be different datasets i.e. multimodal networks, with low intersection among them. Maybe network A can be drugbank and network B can be a smaller network coming from a different database.

### Questions
1 - Regarding the comparison based on new drugs, are the authors reporting the 5th part, i.e., the one containing the new drugs? Are these parts equally-sized? I would appreciate further clarification.

2 - How is the non-end-to-end implemented? is it a sequential two-phases training? I would appreciate if the authors can provide a detailed description of the implementation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This study presents a drug-drug interaction (DDI) prediction method, DHENN, designed to overcome the limitations of previous approaches, including the neglect of high-order information in drug knowledge graphs (KGs), oversmoothing issues, and non-end-to-end frameworks. The authors introduce a message-passing process for drug representation learning and a cascading layer called PECN for DDI prediction. DHENN's performance is evaluated against 12 baseline models across three tasks and two benchmark datasets, complemented by an ablation study and hyperparameter sensitivity analysis.

### Strengths
- The model architecture and computational processes are well-articulated.
- The inclusion of multiple competitive baselines offers a comprehensive assessment of DHENN’s performance and competitiveness.
- The ablation study is designed to directly address and evaluate the core limitations identified by the authors.

### Weaknesses
-  Regarding limitation 1 (High-order interactions): DHENN utilizes a single-layer GNN, which only captures first-order relationships for representation learning. This raises the question of **whether it is genuinely more effective at capturing high-order information compared to other models**. Moreover, referring to "high-order information" as a significant limitation seems overly generalized, as this issue could be easily addressed by stacking multiple GNN layers to propagate higher-order interactions throughout the graph.
-  Regarding limitation 2 (Oversmoothing): Given that DHENN only employs a one-layer GNN, **it is unclear how effectively the model addresses the oversmoothing issue**. A more convincing approach would be to experimentally demonstrate that oversmoothing is a problem for other baseline models (e.g., by comparing node similarity across embeddings from different layers) rather than making direct assertions. Furthermore, a similar similarity analysis should be conducted for DHENN to substantiate this claim.
-  Regarding limitation 3 (Non end-to-end framework): Many existing methods, whether based on single-graph or knowledge graph approaches, are already end-to-end. Thus, it is **difficult to convincingly argue that this is a major limitation** in the current landscape of DDI prediction research.

### Questions
- Could you please define what constitutes a "hybrid method" in the context of your study and elaborate on how it differs from KG-based methods? Typically, KG-based methods also employ end-to-end node representation learning or feature extraction.
- What specific modalities are used in DHENN to establish a multi-modal framework? It appears that molecular structures are not used directly but are instead converted into molecular fingerprints.
- How are substructures extracted and selected from drug SMILES?
- MACCS are typically 166-dimensional. Why does DHENN only utilize 13 MACCS features?
- The necessity for neighborhood sampling in DHENN is unclear. In general, hetero-GNNs, similar to the one used in your model, often consider all neighbors to preserve the complete network topology.
- Could you clarify the difference between Task 2 and Task 3?
- The experimental settings for the non-end-to-end version of DHENN lack. Specifically, what was the optimization objective for the feature extraction component in this configuration?

### Soundness
3

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
5

### Summary
In this paper, the authors proposed a new framework DHENN which integrates the different entities related to molecule, and perform drug-drug interaction events prediction. In order to utilize these entites information, the authors propsed multimodal knowledge graph (MKG) and prediction enhanced cascading network (PECN). In two datasets and different tasks, DHENN outperforms other methods.

### Strengths
1. The experiments are thorough. The authors performed the experiments on both DDIMDL's dataset and a recently-collected dataset.
2. I doubted the architecture of the prediction enhanced cascading network (PECN) at first, but the authors used ablation study to prove the effectiveness of the PECN, which is impressing.

### Weaknesses
The quality of presentation can still be improved, there are many problems such as:
1. For Figure 1, target, transporter and enzyme. They are all proteins. But why did the authors use atomic nucleus icon to represent target? And for the enzyme, I believed the icon is bacteria.
2. Also in Figure 1, please use molecule SMILES to represent SMILES. For example, C1=CC=CC=C1 for Benzene. And I don't know what does the authors want to do by putting two molecule graphs into drug SMILES and repeate it again. Is it representing $drug_i$ and $drug_j$? 
3. In the drug-substructures, it said the drug has a SMILES attribute “986”, what does it mean? That's so confusing. And also, the authors didn't introduce SMILES at all.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
3
