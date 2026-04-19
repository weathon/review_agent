# Contextualized Messages Boost Graph Representations

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 3

## Abstract
Graph neural networks (GNNs) have gained significant attention in recent years for their ability to process data that may be represented as graphs. This has prompted several studies to explore their representational capability based on the graph isomorphism task. These works inherently assume a countable node feature representation, potentially limiting their applicability. Interestingly, only a few study GNNs with uncountable node feature representation. In the paper, a novel perspective on the representational capability of GNNs is investigated across all levels—node-level, neighborhood-level, and graph-level—when the space of node feature representation is uncountable. More specifically, the strict injective and metric requirements are *softly* relaxed by employing a *pseudometric* distance on the space of input to create a *soft-injective* function such that distinct inputs may produce *similar* outputs if and only if the *pseudometric* deems the inputs to be sufficiently *similar* on some representation. As a consequence, a simple and computationally efficient *soft-isomorphic* relational graph convolution network (SIR-GCN) that emphasizes the contextualized transformation of neighborhood feature representations via *anisotropic* and *dynamic* message functions is proposed. A mathematical discussion on the relationship between SIR-GCN and widely used GNNs is then laid out to put the contribution into context, establishing SIR-GCN as a generalization of classical GNN methodologies. Experiments on synthetic and benchmark datasets then demonstrate the relative superiority of SIR-GCN, outperforming comparable models in node and graph property prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new perspective on the representational capability of GNNs by presenting a soft-injective function using a pseudometric distance. It then proposes a new message-passing scheme that performs competitively across various datasets.

### Strengths
1. The paper presents an interesting approach, offering clear insights into relaxing the injective constraint in message-passing processes by defining a pseudometric distance, which captures differences within data effectively.

2. The theoretical basis is solid, providing a coherent and accessible framework that helps explain the introduced ideas.

3. Although the proposed method is limited by the 1-WL test, it shows strong representational capabilities. Its effectiveness is supported by experiments conducted on both synthetic and real-world datasets, confirming its practical usefulness.

### Weaknesses
1. The experimental section is somewhat unclear, especially regarding the number of parameters in Table 4 compared to Table 3. It is confusing why the same model appears to have twice the number of parameters. The authors emphasize "a single layer" in line 507 but do not mention this detail in Table 4, which could lead to potential inconsistencies. To improve clarity and fairness, the authors should add explanations of any differences in the model architecture between datasets. This would help readers better understand the experimental setup and assess the fairness of the comparisons.


2. Although the authors claim to achieve "a balance between computational complexity and model expressivity" (line 820), the experiments related to computational efficiency (Tables 6,7) are not convincing enough. Adding runtime analysis on larger-scale datasets, such as ogbg-molhiv/ogbn-arxiv, would support these claims and better demonstrate the method's scalability.

### Questions
See Weaknesses

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
This paper investigates the representational capacity of Graph Neural Networks (GNNs) with uncountable node features and introduces an MLP-based MPNN named SIR-GCN, which generalizes to several popular GNN architectures. The authors demonstrate that for uncountable node features, it is possible to identify a soft-injective function corresponding to a specific pseudometric that quantifies dissimilarity in the node feature space. They then model this soft injective function using an MLP and design an architecture that maintains both anisotropic and isotropic properties. Experimental results highlight the model’s superiority across various scenarios, including cases with countable and uncountable node features, as well as graphs exhibiting heterophily.

### Strengths
1.	It is interesting to see how the author tackles the problem of uncountable node features using pseudo metric and soft-injective functions.
2.	The designed model has certain flexibility in terms of anisotropic and isotropic properties. The model architecture also generalizes easily to some popular GNNs.
3.	The experiment shows the SIR-GCN performance well on the Dictionary Lookup task and against other baseline models during the benchmarking test.

### Weaknesses
1.	Since the SIR-GCN can generalize to other GNNs, it would be better if the author can explain why it is hard for other GNNs to handle the problem of uncountable node features.
2.	It would be nice if the author can explain why there are some missing values in Table.
3.	For the graph heterophily experiment, it would be better if the author can use some real world datasets with different degrees of heterophily[1].

[1] Mao H, Chen Z, Jin W, Han H, Ma Y, Zhao T, Shah N, Tang J. Demystifying structural disparity in graph neural networks: Can one size fit all?. Advances in neural information processing systems, 2024.

### Questions
1.	According to the description of GAT, does GAT also preserve both anisotropic and isotropic? 
2.	For Table 4, Can GAT or GraphSAGE achieve similar performance with more parameters?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper extends PNA, which considers uncountable node features, by incorporating anisotropic and dynamic. The difference between PNA and the proposed SIR-GCN is the nonlinear mapping outside the message creation. To motivate this method, the authors introduce the concept of soft-injective function. This paper shows some existing methods are variants of the proposed SIR-GCN. Evaluations on synthetic and real datasets demonstrate its effectiveness.

### Strengths
- The uncountable feature is an important topic in GNN. 
- The writing and organization are good to follow. 
- The insight of existing methods under the framework of SIR-GCN is interesting.

### Weaknesses
- The novelty seems weak. Both the anisotropic and dynamic are not novel.  This paper can be seen as a combination of PNA and GATv2. 
- The motivation and the proposed SIR-GCN are not closely connected. It is not clear the connection between the soft-injective function, dynamic transformation, and anisotropic message. 
- The description of the GraphHeterophily is not clear. Thus,  it is not obvious why the proposed SIR-GCN significantly outperforms existing ones. 
- The derivation from Eq. 15 to Eq. 16 seems incorrect. First, the definition of $A$ is not given. Secondly, the anisotropic of GAT is on the edge weight, while that of Eq. 16 is on the message. It is not obvious.
- Figure 2 is not described clearly. What is the meaning of the horizontal and vertical coordinates? Why the contour of MLP is as in Figure 2(c) and 2(d).
- The evaluations are not convincing. Firstly, the ablation study and illustrative examples are not given. So, the effect of the proposed SIR-GCN is not justified. Secondly, it is not knowns whether the proposed SIR-GCN can be applied to complex models, whose performance is higher.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2
