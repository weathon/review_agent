# Federated Graph-Level Clustering Network with Dual Knowledge Separation

- Decision: Accept (Poster)
- Scores: 8, 2, 8, 6

## Abstract
Federated Graph-level Clustering (FGC) offers a promising framework for analyzing distributed graph data while ensuring privacy protection.
However, existing methods fail to simultaneously consider knowledge heterogeneity across intra- and inter-client, and still attempt to share as much knowledge as possible, resulting in consensus failure in the server.
To solve these issues, we propose a novel **F**ederated **G**raph-level **C**lustering **N**etwork with **D**ual **K**nowledge **S**eparation (FGCN-DKS). 
The core idea is to decouple differentiated subgraph patterns and optimize them separately on the client, and then leverage cluster-oriented patterns to guide personalized knowledge aggregation on the server.
Specifically, on the client, we separate personalized subgraphs and cluster-oriented subgraphs for each graph. Then the former are retained locally for further refinement of the clustering process, while pattern digests are extracted from the latter for uploading to the server.
On the server, we calculate the relation of inter-cluster patterns to adaptively aggregate cluster-oriented prototypes and parameters. Finally, the server generates personalized guidance signals for each cluster of clients, which are then fed back to local clients to enhance overall clustering performance.
Extensive experiments on multiple graph benchmark datasets have proven the superiority of the proposed FGCN-DKS over the SOTA methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new method for federated graph-level clustering. Compared with existing federated graph-level clustering methods, this method solves the heterogeneity problem of multi-source data through dual knowledge separation. The article is clear in expression, motivation is clear, and the experimental design is complete.

### Strengths
1. The experimental setup is comprehensive, with comparisons made against both supervised and unsupervised methods. Ablation analysis demonstrates the superiority of the proposed approach.
2. As the novel method for federated graph-level clustering, this approach significantly enhances performance by separating different subgraphs and utilizing cluster-guided prototype aggregation.
3. Research on federated graph-level clustering remains scarce, and this study effectively fills that gap.

### Weaknesses
1. The authors should further elaborated The non-IID setting in the appendix for additional clarification.
2. The authors should include experiments on federated communication costs.
3. Some minor errors should be further corrected, such as changing "leverage" to "leverages."

### Questions
1. The paper mentions using a graph kernel method to measure network heterogeneity. To facilitate understanding and replication, could the authors please specify which specific graph kernel or types of kernels they used? Also, could they explain the rationale for choosing this particular kernel function to quantify heterogeneity?
2. The authors need to clarify how and to what extent label information is used in the unsupervised training phase and the downstream task evaluation phase.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenging issue of Federated Graph-level Clustering, where heterogeneity across clients, both intra- and inter-client, can cause difficulties in achieving a meaningful consensus on the server side. The authors propose a novel framework called FGCN-DKS. The core idea involves two levels of decoupling: 1) On the client side, each graph is split into an "invariant" subgraph and a "variant" subgraph. Only the digest of the invariant part is shared. 2) On the server side, a Common Knowledge Sharing Strategy is employed to perform a personalized aggregation of client models, guided by the similarity between clients' cluster pattern digests. A two-stage clustering process on the client leverages both shared and local representations.

### Strengths
1.The paper effectively identifies the key issue of heterogeneity in federated graph clustering, distinguishing it from the more common challenges seen in node-level federated learning.

2.The comprehensive experiments across multiple datasets and non-IID settings lend strong support to the method's robustness and generalizability.

### Weaknesses
1.The framework introduces multiple new components and hyperparameters, making it complex and difficult to analyze or tune effectively. It is unclear which specific component contributes most to the performance gains, making it challenging to pinpoint areas for improvement or fine-tuning.

2.The many moving parts in the system make it difficult to attribute performance improvements to specific components. 

3.While the dual knowledge separation and two-stage clustering approach is novel, the system’s complexity may limit its practicality and ease of adoption. Simplifying some aspects could make it more accessible.

### Questions
1.Given the complexity of the framework, could the authors provide a more detailed ablation study to isolate the contribution of each key component? 

2.Could the authors quantify this communication cost and compare it to standard federated learning approaches like FedAvg, which transmit model parameters?

3.How can the "quality" of this separation be directly evaluated, beyond just its impact on downstream clustering accuracy? Is there a way to ensure that the invariant subgraphs capture truly common, cross-client patterns?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This manuscript introduces a novel framework for federated clustering, centered around the idea of decoupling knowledge both within and across clients to enhance clustering performance. This manuscript is well-written, logically structured, clearly articulated, and experimentally robust, addressing an innovative problem. The experiments demonstrate significant improvements compared to advanced methods.

### Strengths
1. The proposed method is novel, as it introduces for the first time the dual decomposition of knowledge into cluster-oriented and client-oriented dimensions.
2. The experimental setup is robust and reliable and thoroughly demonstrates the superiority of the proposed framework.
3. The authors cleverly integrate invariant learning into the federated graph-level clustering framework, which I find very insightful. It suggests that a similar strategy could potentially be used to improve federated graph classification as well.

### Weaknesses
1. The adaptation of other methods, such as federated graph learning and federated anomaly detection, to federated graph clustering should be further explained.
2. Communication overhead experiments should be included to further validate the superiority of the method.
3. The fonts in the architecture diagram are consistent. It is recommended to further unify them to New Times Roman or Microsoft YaHei.

### Questions
1. The calculation of the heterogeneity of intra-client graphs is unclear. Is the author referring to the heterogeneity of all graphs within the client, or the heterogeneity of graphs within different clusters on the client?
2. Does this method have the capability to address the issue of inconsistent numbers of clusters across different clients?
3. Is this federated client configuration self-defined, or does it follow the setup used in prior work?
4. What exactly does the prototype refer to? I hope the author can explain it further.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel federated graph-level clustering algorithm. The algorithm decouple knowledge at both the local and server levels, thereby improving the quality of global consensus. The motivation is clearly presented, the overall presentation is of high quality, and the experimental results demonstrate superior performance compared with existing methods.

### Strengths
S1. Unlike traditional approaches that enhance consensus by increasing shared data, this framework adopts an alternative strategy by reducing the amount of data involved in consensus, which not only facilitates more efficient agreement but also contributes to better privacy protection.

S2. The article demonstrates high quality in writing, presentation, and overall organization, and it is easy to follow and understand.

S3. Compared with other methods, this approach is better suited for real-world federated scenarios. Theoretical analysis is further provided to demonstrate the convergence of the proposed method.

### Weaknesses
W1. The convergence analysis should include the loss variation during training.

W2. Performance comparisons should be performed across all clients in a non-iid setting to demonstrate that the federated approach can improve overall performance without sacrificing the performance of individual clients.

W3. Comparing it to supervised methods seems less meaningful; replacing it with the experiments described above better demonstrates the advantages of the method.

### Questions
Q1. The convergence analysis mentions the FedPKA method, but it is missing from the comparative experiments. Although this method was not published in a top-tier conference, could it be that the authors accidentally omitted it?

Q2. Why was FedSage compared in the experiment? This method seems to be applicable to node-based tasks rather than graph-level tasks?

### Soundness
4

### Presentation
4

### Contribution
3
