# Certifying Graph Neural Networks Against Label and Structure Poisoning

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Robust machine learning for graph-structured data has made significant progress against test-time attacks, yet certified robustness to poisoning – where adversaries manipulate the training data – remains largely underexplored. For image data, state-of-the-art poisoning certificates rely on partitioning-and-aggregation schemes. However, we show that these methods fail when applied in the graph domain due to the inherent label and structure sparsity found in common graph datasets, making effective graph-partitioning difficult. To address this challenge, we propose a novel semi-supervised learning framework called deep Self-Training Graph Partition Aggregation (ST-GPA), which enriches each graph partition with informative pseudo-labels and synthetic edges, enabling effective certification against node-label and graph-structure poisoning under sparse conditions. Our method is architecture-agnostic, scales to large numbers of partitions, and consistently and significantly improves robustness guarantees against both label and structure poisoning across multiple benchmarks, while maintaining strong clean accuracy. Overall, our results establish a promising direction for certifiably robust learning on graph-structured data against poisoning under sparse conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Partition-based certification methods struggle to provide meaningful poisoning guarantees for GNNs because partitioning graph data often creates overly sparse subgraphs for training base models. To overcome the limitations, this paper introduces ST-GPA, using semi-supervised learning and link prediction to generate pseudo-labels and synthetic edges to densify each partitioned subgraphs. Experimental results show that ST-GPA consistently improves robustness guarantees against both label and structure poisoning across multiple dataset.

### Strengths
* The writing is well-structured and easy to follow.
* Addresses an important problem in guaranteeing robustness against poisoning attacks
* Their method certifies defense against both node-level and structure-level attacks.

### Weaknesses
* The authors state that their method maintains strong clean accuracy. However, their figures do not provide any comparison against clean accuracy of vanilla GNNs without any partitioned training. The authors should integrate the details from Table 2 (appendix) in the main paper for better clarity.

* The author only evaluate under citation-based dataset with the largest dataset being medium sized (Pubmed). I highly encourage the authors to provide evaluations results on large scale dataset such as arXiv and non-citation dataset like WikiCS.

* The core certification guarantee (Theorem 2) is a minor generalization of the existing DPA theorem and simplifies (d_h=1) in the paper's specific application, making the resulting condition formally analogous to the original DPA framework.

* While the reasons for omission of certain baselines are described in section 7, [1] should still be included as a baseline. Although [1] doesn't report certificates against label flipping, its underlying partition-based methodology is general and applicable.

### Questions
* In the proof for Theorem 2, shouldn't the minus be a plus in equation 22?

* Clarification is needed regarding the claim that the structural certificates from [1] fall below MLP performance. There appears to be no direct evidence supporting this assertion presented either in the current paper or in the cited work [1]

\
[1] Deterministic certification of graph neural networks against graph poisoning attacks with arbitrary perturbations.

### Soundness
2

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
3

### Summary
The paper tackles the problem of certifying robustness of Graph Neural Networks (GNNs) against training-time poisoning attacks on node labels and graph structure. It extends the Deep Partition Aggregation (DPA) framework to non-i.i.d. graph data and introduces Self-Training Graph Partition Aggregation (ST-GPA), which enriches each sparse partition with pseudo-labels and pseudo-edges via semi-supervised learning. Experiments on multiple benchmarks show that ST-GPA substantially improves both clean accuracy and certified robustness, establishing an effective certification framework for label- and structure-poisoned GNNs.

### Strengths
1. The paper targets training-time attacks and provides a clear certified condition based on the robust margin.
2. The work extends Deep Partition Aggregation (DPA) from i.i.d image data to non-i.i.d. graph data.
3. Evaluations across multiple benchmarks (Cora, CiteSeer, PubMed, Cora-ML) and architectures (GCN, GAT, APPNP) show consistent certified accuracy improvements.

### Weaknesses
1. Self-training has already been shown to significantly improve adversarial robustness in GNNs[1,2]. Therefore, I strongly suspect that the robustness improvement observed in this paper may mainly stem from the self-training component, rather than from the proposed Graph Partition Aggregation mechanism itself.
2. The paper lacks comparisons with strong existing certified-robustness baselines, such as NTK-based or MILP-based exact certificates[3].

[1] Li, Kuan, et al. "Revisiting graph adversarial attack and defense from a data distribution perspective." The Eleventh International Conference on Learning Representations. 2023.

[2] Chowdhury, Subhajit Dutta, et al. "Unveiling Adversarially Robust Graph Lottery Tickets." Transactions on Machine Learning Research.

[3] Gosch et al., 2024 — Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks

### Questions
1. Can ST-GPA apply to larger graphs (e.g., ogbn-arxiv, ogbn-products, ogbn-papers100M) and how's the certified accuracy?
2. It's always worth to see a defense method's robustness under adaptive attacks: Consider scenarios where an attacker knows the pseudo-labeling mechanism.

### Soundness
2

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
3

### Summary
This paper proposes a model that achieves robustness for Graph Neural Networks (GNNs) against label and structure poisoning. It effectively describes the challenges of applying certification methods to graphs, which differ significantly from the image domain. The proposed model utilizes a self-training-based ensemble of classifiers to make robust predictions and demonstrates consistent performance improvements across various datasets.

### Strengths
S1. The paper addresses the vulnerability of GNNs to adversarial poisoning attacks, which is a critical and still largely unresolved problem in the field.

S2. It clearly demonstrates the limitations of the naive partitioning-and-aggregation method (which works for images) when applied to graphs, supporting this analysis with experimental results (e.g., Figures 2 and 3).

S3. The proposed method, ST-GPA, shows consistent and significant performance improvements over the baselines on multiple datasets and GNN architectures.

### Weaknesses
W1. The motivation for the work could be strengthened and more clearly articulated in the introduction, compared to the existing work. In my opinion, the discussion on self-training for GNN robustness could be more comprehensive. There are several relevant studies, such as Good-at [1] and GPR-GAE [2], that are not discussed. A discussion of how the proposed method relates to these and, if possible, an experimental comparison would be beneficial.

W2. The evaluation of the adversarial setting should be extended to more realistic and challenging attack scenarios, such as those generated by optimization-based algorithms (e.g., PR-BCD, LR-BCD). The current evaluation assumes a simpler threat model, which may limit the perceived benefits and generalizability of the proposed method.

W3. The paper claims that the guarantee from Theorem 1 (DPA) "does not depend on the i.i.d. nature of the data." in Lines 140-141. This claim needs further clarification. How is the i.i.d. assumption, which is fundamental to the original DPA proof (as poisoning one data point only affects one partition), properly bypassed in the generalized non-i.i.d. graph setting?

W4. A significant limitation is that the pseudo-edge generation technique (Section 5.2) is explicitly "guided by the homophily assumption." This approach is likely to be ineffective or even detrimental on heterophilous graphs, where connected nodes often have different labels. This limitation should be acknowledged and discussed.

W5. While the paper is generally well-written, there are areas for improvement in its presentation. For example,

- Some notations are unclear (e.g., the variable '$r$' in Theorem 1 is used without a clear definition in that immediate context).

- The font size in Figure 1 is too small, which hinders readability.


References

[1] Boosting the adversarial robustness of graph neural networks: An ood perspective, ICLR 2024

[2] Self-supervised Adversarial Purification for Graph Neural Networks, ICML 2025

### Questions
Q1. Could you strengthen the motivation by differentiating more clearly from existing work? 

Q2. The current evaluation uses a basic threat model. To what extent would the certified guarantees hold against more realistic, optimization-based attacks like PR-BCD or LR-BCD?

Q3. Could you please clarify the claim in Lines 140-141? How is the i.i.d. assumption from the original DPA proof (where one poisoned point affects one partition) provably bypassed in your generalized non-i.i.d. graph setting?

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
This paper addresses certified robustness of GNNs against training-time poisoning attacks. The authors identify that existing partition-and-aggregate approaches used in image domain fail on graph data due to label and structure sparsity. They propose self training graph partition aggregation (ST-GPA), which augments partitions with pseudo-labels and synthetic edges through semi-supervised learning to enable effective certification.

### Strengths
1. Certified robustness against poisoning for GNNs is understudied compared to test-time attacks. 
2. Using semi-supervised learning (label propagation, self-training, link prediction) to enrich sparse partitions seems to work.

### Weaknesses
1. The entire defense hinges on the self-training (label propagation, label self training, and link prediction self training) stage. However, these SSL methods are themselves run on the poisoned training data before certification. Could an attacker poison labels/edges in such a way that the label propagation step generates incorrect pseudo-labels that reinforce the attack?
2. The experiments do not address the scalability of this approach to larger datasets like OGB. How does the proposed idea scale with the number of nodes, edges, and partitions k?
3. How does the proposed work apply to non-homophilic graphs datasets? All the experiments are for homophilic graphs. 
4. Some grammatical issues are there. Please work on addressing them.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
