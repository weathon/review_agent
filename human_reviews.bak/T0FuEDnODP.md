# Cooperative Graph Neural Networks

- Decision: Reject
- Scores: 6, 3, 8, 5

## Abstract
Graph neural networks are popular architectures for graph machine learning, based on iterative computation of node representations of an input graph through a series of invariant transformations. A large class of graph neural networks follow a standard message-passing paradigm: at every layer, each node state is updated based on an aggregate of messages from its neighborhood. In this work, we propose a novel framework for training graph neural networks, where every node is viewed as a player that can choose to either listen, broadcast, listen and broadcast, or to isolate. The standard message propagation scheme can then be viewed as a special case of this framework where every node `listens and broadcasts' to all neighbors. Our approach offers a more flexible and dynamic message-passing paradigm, where each node can determine its own strategy based on their state,  effectively exploring the graph topology while learning. We provide a theoretical analysis of the new message-passing scheme which is further supported by an extensive empirical analysis on a synthetic dataset and on real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel message-passing paradigm in which nodes have different states and can decide to receive, send, and ignore messages based on the states. The states are determined by the output of a Gumbel-softmax estimator trained along with the model. Some theoretical analysis is provided to show that the proposed method alleviates the over-squashing issue that regular GNNs have.

### Strengths
The paper is generally well-written and easy to follow. The motivation for using such a model to prevent over-squashing is explained well, both theoretically and intuitively. The proposed model effectively reduces the total number of message passing, which potentially improves efficiency. However, the actual improvement is unclear due to the non-deterministic nature of the method. The analysis of the kept edges across layers provides good insights into the underlying mechanism of the technique.

### Weaknesses
My primary concern lies in that this work appears very similar to Agent-based Graph Neural Networks [1]. An agent that decides to move towards neighboring nodes can be seen as listen/broadcast states in CoGNN, and an agent that decides to stay in the current node can be seen as the isolate state in CoGNN. In such a case, will CoGNN be identical to Agent-based Graph Neural Network?

The graph classification datasets being evaluated are small compared to larger datasets such as QM9 and ZINC.

Node classification evaluation is only conducted on heterophilic datasets. While cora is evaluated in the appendix, showing CoGNN generally works on homophilic graphs is also essential.

[1] Martinkus, Karolis, et al. "Agent-based graph neural networks." arXiv preprint arXiv:2206.11010 (2022).

### Questions
- While it makes sense that for homophilic graphs, less directed edges in the last layers help filter distant nodes' information, it does not make much sense to me why fewer directed edges in the first layers help heterophilic graphs learning. It is essentially suppressing the first few layers, but the last layers are still promoting homophilic learning. Can you explain this further?

- Can you explain the difference between CoGNN and agent-based GNN?

- How is gradient propagated to the Gumbel-softmax estimator?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors of this paper propose a new message-passing scheme where each node can lister or broadcast messages, and therefore can alter the graph topology by removing or making some edges directed.

### Strengths
1. The proposed model is flexible in adapting the graph topology to specific tasks. By allowing nodes to choose their message-passing actions based on their state, the model can optimize information flow according to the task at hand. This task-specific adaptation is a valuable feature, as it enables better exploration of graph topologies.

2. The paper is well-written and the idea is communicated very clearly.

### Weaknesses
1. Novelty and Clarifications: The proposed approach may lack technical novelty as it primarily involves altering the direction of edges or discarding them altogether.  Additionally, the paper does not provide a clear explanation of how the proposed approach can be adapted for use in directed graphs, where edge direction is a critical factor, leaving a gap in its applicability.
Furthermore, the authors claim that the model can efficiently filter out irrelevant information by focusing on the shortest path connecting two nodes to maximize information flow to the target node. However, due to the node-based nature of the actions, there is a significant risk that a node, for example, u_1, may be irrelevant for one node, u_2, but important for another node, u_3. With the current framework, if node u_1 chooses to listen and not broadcast, it won't be able to transmit information to u_3, creating a potential limitation in information flow within the model.
In contrast, Graph Attention Networks (GATs) can address similar challenges by focusing on edge-based attention scores, allowing for fine-grained control over information transmission. GATs can effectively filter out irrelevant information by learning attention scores close to zero, similar to the "listen and not broadcasting" actions mentioned in the proposed framework. A thorough discussion of the similarities and differences between the proposed approach and GATs, as well as other relevant models, would enhance the paper's clarity and its position in the context of existing graph neural network research.

2. Expressivity: Since the sampling process can introduce variability, it's possible for two identical graphs to obtain different representations due to the differences in sampled action. While this non-deterministic behavior enhances expressiveness, it's essential to be aware of it when working with CO-GNNs, as it may introduce variability in model outcomes and representations, even for isomorphic graphs. A proper discussion of this limitation is currently missing.

3. Experiments: The experimental evaluation is weak. Several strong baselines are missing (see below for some examples [1,2,3] ) and some well-known graph and node classification datasets are missing as well (REDDIT-BINARY, REDDIT-MULTI, Cora, Citeseer, Texas, Wisconsin, Cornell etc). Moreover,  the proposed approach does not significantly outperform the simple GIN model in most cases.  

References:

[1] Zhang, Muhan, and Pan Li. "Nested graph neural networks." Advances in Neural Information Processing Systems 34 (2021): 15734-15747.

[2] Pasa, L., Navarin, N. & Sperduti, A. Polynomial-based graph convolutional neural networks for graph classification. Mach Learn 111, 1205–1237 (2022). https://doi.org/10.1007/s10994-021-06098-0

[3] Nikolentzos, Giannis, Michail Chatzianastasis, and Michalis Vazirgiannis. "Weisfeiler and Leman go Hyperbolic: Learning Distance Preserving Node Representations." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

### Questions
1. Can the authors elaborate on how the proposed approach can be adapted for use in directed graphs? Edge direction is crucial in various real-world applications, and it would be valuable to understand how the model deals with it.

2. The paper claims that the model can efficiently filter out irrelevant information by focusing on the shortest path. However, it's not clear how this works when information from one node is important for some nodes and irrelevant for others (see weaknesses above). Could the authors provide more clarity on this aspect?

3. The paper introduces an innovative approach, but it's important to discuss its similarities and differences with existing models, such as Graph Attention Networks (GATs) (see weaknesses above). A thorough discussion would help readers understand the model's unique contributions. Can the authors provide insights into how their approach compares to GATs or other models in terms of filtering out irrelevant information and optimizing information flow within a graph?

4. It's mentioned that CO-GNNs introduce variability due to the non-deterministic nature of the sampling process. Could the authors discuss the potential limitations and implications of this variability, especially when working with real-world data or applications where consistency in representations is critical?

5. The experimental evaluation lacks some strong baselines and omits some well-known graph and node classification datasets. Could the authors explain the rationale for the choice of datasets and provide justification for not including certain well-known benchmarks like REDDIT-BINARY, REDDIT-MULTI, Cora, Citeseer, Texas, Wisconsin, Cornell, etc.?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the author proposed a more flexible and dynamic message-passing paradigm. In this paradigm, each node is viewed as a player with actions of either 'listen', 'broadcast', 'listen and broadcast' or 'isolate'. Based on this paradigm, the author proposed a new GNN, called cooperative GNN (Co-GNNs). Experimental results on regression, node classification, graph classification have showed better or competitive performance than SOTA works.

### Strengths
I have enjoyed reading the manuscript because it proposed a novel and solid message passing operation. The introduction is written very well, that is easy to make readers easily immersed in this work. The author has done detailed theoretical analysis and experimental analysis to prove the superiority of the proposed Co-GNN framework. I believe this work can inspire many subsequent works, making potential contributions to the GNN field.

### Weaknesses
My only concerns is if it is possible to shows some visualization results for this new message passing operation on the node- and graph-level task,e.g., For a right prediction, how each nodes behaves in the Co-GNN framework.  I think that will make readers more appreciate this nice work, and will encourage following works along this line.

### Questions
Please see the above

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers GNNs that are typically based on a nearest neighbor message passing protocol, and specializes to include learning how nodes should communicate.  This is set up as an action space with 4 possibilities, and the action space is co-learned for each node along with the associated GNN, and refer to this a cooperative-GNNs (Co-GNNs) approach.  The Gumbel-softmax is used to train the action network as part of the co-learning.  The idea could in principle be applied to many GNNs to co-learn.  Some experiments show that the Co-GNN idea may have merit in some scenarios, although the results are mixed.

### Strengths
The basic idea is interesting, and potentially provides a useful GNN tool for cases where graph isomorphism needs to be characterized, e.g., graph classifiers. 

The method has the potential to reduce inference computational complexity.

The method may reveal long range dependencies that aren’t easily described with conventional GNNs. 

The co-learning appears to be a straightforward addition to existing GNN architectures, so this opens the idea of how to harness this for different scenarios, some of which are explored in the paper.

### Weaknesses
The paper seems to be preliminary; interesting but with too many loose ends.  The experiments are interesting, but not quite fully understood yet.  

The actions lead to a time-varying topology, but the results are not connected to the many graph theoretical works on random graphs, or shortest-path routing protocols in communications networks. The Appendix B results are interesting, and clearly suggest an analysis based on information flows within the graph.
 
Theorem 5.2 is interesting, but ultimately seems only to say that there will (or can) be a path learned between arbitrary nodes for them to transfer information.

The method seems likely to be brittle over topology or some erroneous training data, and it isn’t clear what kind of generality is possible. 

The synthetic example in section 6.1 obviously favors the proposed method and the problem is very artificial.  A useful solution would be model based.

Ablation studies are needed to better understand the benefits and issues.

*Revised Review*

The authors have addressed some of my questions and issues, although there remains a considerable focus on discussion with mixed results, and the synthetic example is just that.

### Questions
Section 5.3: Is there a claim that the method will always learn a shortest-path route?  For example, it isn’t clear what this means of information is acted upon along the way.

As the paper notes in Section 5.4, the actions lead to time-varying graph topology, at least in the sense of turning edges on and off in an overall fixed topology?

In the simulations there are claims about “using relatively simple architectures”, but what are the comparisons made to?

Figure 7(b).  Does the graph have specific bottlenecks or other features that restrict the flow at early layers?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
