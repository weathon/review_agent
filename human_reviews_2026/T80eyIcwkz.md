# Who Owns This Sample: Cross-Client Membership Inference Attack in Federated Graph Neural Networks

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Graph Neural Networks (GNNs) are increasingly integrated with federated learning (FL) to protect data locality in domains such as social networks, finance, and biology. While membership inference attacks (MIAs) have been widely studied in centralized GNNs, their scope in federated settings remains underexplored. We present CC-MIA, a framework that reformulates membership inference in federated GNNs into a cross-client attribution problem, where an adversarial client aims to determine not only whether a node was part of training but also which client owns it. CC-MIA operates under a realistic threat model: the adversary is a legitimate participant who observes global updates and can eavesdrop on other clients’ gradients, a well-studied vulnerability in recent gradient inversion attacks. To approximate the target data distribution, CC-MIA leverages publicly available shadow datasets from the same domain, consistent with established MIA practice. The attack combines shadow-based training for membership inference, gradient inversion to reconstruct client subgraphs, and prototype-based matching to assign nodes to clients. Experiments on six benchmark datasets and five federated schemes show that CC-MIA consistently outperforms strong MIA baselines, achieving up to 72.16\% improvement in inference accuracy. These results highlight that membership inference in federated GNNs naturally extends to client attribution, underscoring the need for defenses robust to gradient-level and client-level leakage. Codes are available at https://anonymous.4open.science/r/CC-MIA-54C3.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CC-MIA, a framework for cross-client membership inference attacks (MIAs) in federated graph neural networks (FedGNNs). It reformulates traditional MIAs to not only determine if a node was part of the training set but also attribute it to the specific client that owns it. The attack leverages a threat model where an adversarial client observes global model updates and eavesdrops on other clients' gradients. Using publicly available shadow datasets from the same domain, CC-MIA combines shadow-based training for membership inference, gradient inversion to reconstruct client subgraphs, and prototype-based matching for client attribution. Experiments are conducted across six benchmark datasets and five federated learning schemes, demonstrating the effectiveness of the proposed method.

### Strengths
+ The research question is reasonable, considering that the client-wise ownership is important in FL MIA.
+ Various experimental evaluation claimed on multiple datasets (six benchmarks) and federated schemes (five), with quantitative improvements over baselines.

### Weaknesses
- The threat model is too strong, especially for gradient eavesdropping. It may not be hold in some FL settings, e.g., secure aggregation, cross-device FL (the number client is large). Besides, I can not get how the malicious client could possibly another client's gradients in practice.
- The assumption of shadow dataset requires more evaluations. It is important in FL, considering the client-side datasets are normally private, and it may be hard to find the in-domain public data.
- It would be better for the authors to provide more details on the related works. There have been many works on MIA in GNN, the authors should make a detailed comparison to present the novelty of designs. Besides, the authors should highlight the differences between centralized and FL MIA.
- Assuming the clients share gradients instead of weight updates is limited. For practical FL, the clients do not directly share gradients, they normally share weight update after multiple rounds of mini-batch local updates. This is also important for the performance gradient inversion attacks.
- The authors should explicitly make evaluations on non-iid settings, and it is important to FL and MIA (e.g., transferability).
- The performance of client-wise ownership inference is not that effective. Though it is higher that the random-guess, there is (possibly) still a very high TPR for the attacker, which will comprise the confidence.

### Questions
1. Why the assumption on gradient eavesdropping is reasonable?
2. More evaluations on shadow dataset and non-iid settings are needed.
3. The novelty and differences between the proposed MIA and previous MIAs on GNN.
4. Is the attack only effective against raw gradients instead of multiple rounds of mini-batch local updates?
5. Report the TPR of client-wise ownership inference.

### Soundness
2

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
The paper proposes a method for membership inference attacks in GNNs (Graph Neural Networks) under the Federated Learning (FL) setup. Specifically, it assumes that one client in the FL setup is adversarial, and tries to infer two types of information: first, whether a given node was part of the training graph (membership inference), and second, which client owns the node (cross-client attribution). To solve for these, the paper proposes a framework called CC-MIA (Cross-Client Membership Inference Attack). While membership inference attacks have been shown for GNNs and for FL in the past, the setup explored in this paper is novel.

### Strengths
1. The paper explores a novel attack vector; membership attacks on GNNs in FL is a previously unexplored area. 
2. The attack model used in the paper is based on commonly accepted assumptions used in the FL and MIA literature, such as access to a shadow dataset, gradient eavesdropping capabilities, etc. 
3. The paper includes comprehensive experimentation including different types of GNNs, FL algorithms, and graph datasets.

### Weaknesses
1. The first part of the attack, which is membership inference, is conceptually not much different from a membership inference attack on a centralized GNN, such as Olatunji et al. [a]. The paper could benefit from explicitly stating the challenges that FL brings into the picture. 
2. The second part of the attack uses a pseudo-graph (as mentioned in line 256), based on which the optimization in eq(12) is carried out. However, it is unclear how this pseudograph is obtained or generated.
3. At several points in the paper, the underlying setup and assumptions have not been clearly stated, rendering it difficult to understand the work. For instance, 
    - In Section 3.1, the Federated GNN setup could be explained better: Is this a horizontal FL setup where all clients have unique nodes? Do clients' subgraphs have connecting edges?
    - The objective function in eq(12) does not mention the optimization variable. 
    - Certain variables have not been defined, for e.g., $D$ in line 217, $c$ in line 226, $x_t^a$ and $x_t^c$ in line 240.
    - Value of hyperparamter $\gamma$ in line 236 has not been stated.
    - There is an overuse of notation at several places, e.g. $x_t^a$ in line 240, $D$ as diagonal matrix in line 266, $D$ in line 217, 

[a] Iyiola E Olatunji, Wolfgang Nejdl, and Megha Khosla. Membership inference attack on graph neural networks. In 2021 Third IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS-ISA), pp. 11–20. IEEE, 2021a.

### Questions
1. How does the proposed membership inference attack differ from that on a centralized GNN, in terms of unique challenges encountered and addressed?
2. What are the assumptions for the shadow dataset? Does it also have subgraphs like the original training dataset? Also, is it possible for the adversarial client to train the shadow network on its own subgraph dataset, instead of the shadow dataset?
3. How frequently (for how many FL rounds) is the gradient inversion step performed?
4. How is the proxy term $\zeta_t$ estimated?
5. How does the value of the hyperparameter $\gamma$ affect gradient inversion?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a membership inference attack method for federated graph learning. Specifically, the proposed method successfully achieves two types of membership inference: training set membership inference and client-level membership inference. Technically, the authors leverage gradient inversion technology to implement the attack. Extensive experiments conducted in this work demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper pioneers the proposal of a membership inference attack method tailored for federated graph learning, filling the research gap in this specific domain.
2. For the client-data identification problem, the authors design a gradient inversion reconstruction method exclusively for graph data. This method effectively reconstructs the original graph structure from gradient information, showing strong targeted performance for graph-specific characteristics.
3. Compared with the contrastive experiments conducted under the centralized setting, the attack method proposed in this paper demonstrates a significant advantage in success rate, fully verifying its superiority in federated scenarios.

### Weaknesses
1. The method for the membership inference attack part lacks special design targeted at graph data which may limit the method’s adaptability and effectiveness in graph-specific scenarios.
2. The paper does not clearly explain on the differences between the method used in the membership inference attack part and the methods applied in the centralized setting. Key distinctions in aspects such as data access constraints, gradient utilization patterns, and attack optimization objectives remain unaddressed, making it difficult for readers to understand the method’s novelty in federated scenarios.
3. The client-data identification part utilizes virtual graph data, but the generation process of this virtual graph is not explained in detail.

### Questions
1. Are the node features of Cora and DBLP (shadow dataset for Cora) consistent? Why can DBLP be used for Cora’s membership inference?
2. Do category prototypes from the shared global GNN have spatial consistency? Is using them for client-level data inference reasonable?
3. What differences exist between the proposed membership inference method and traditional centralized ones? What unique advantages does it have?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Membership inference in FedGNNs naturally extends to client attribution. CC-MIA, fusing shadow training, gradient inversion, and prototype matching, outperforms baselines. It confirms gradient transmissions leak client-specific structural info, demanding defenses against gradient/client-level leakage.

### Strengths
• Judges node training membership and client ownership (vs. single MIA by others).
• Strong generalizability，Works for FedAvg/FedProx/SCAFFOLD/FedDF/FedNova and GCN/GAT/GraphSAGE.
• Fits threat models (adversaries as legitimate participants with gradient/global update access).

### Weaknesses
• Scalability: Gradient inversion quality drops with more clients, limiting large-scale FedGNN use.
• While the paper emphasizes the risks associated with CC-MIA, it offers limited practical insights or proposals for defending against such attacks.

### Questions
Relies on class-consistent gradients and separable prototypes, what if it fails in real non-IID data?

### Soundness
3

### Presentation
2

### Contribution
2
