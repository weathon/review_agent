# LoRE: Robust and Adaptive Graph Embeddings via Local Self-Reconstruction Mechanisms

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Graph embeddings aim to project nodes into numeric vector spaces that capture structural and semantic regularities, enabling their use as general-purpose representations for a broad range of downstream applications. However, existing embedding methods distort local geometry through negative sampling, fail to enforce semantic consistency, and require expensive retraining when graphs evolve. Therefore, we introduce LoRE, a versatile graph embedding framework based on attention-driven self-reconstruction mechanisms and a perspective-preserving training procedure. Built on a generalized formulation, LoRE can be applied to a wide range of graph types, from undirected graphs to relational knowledge graphs and even attributed node sets without inherent topologies. It enforces identical embeddings for structurally equivalent nodes, respects local context during training, and reduces the likelihood of violations of the open-world assumption. Unlike traditional methods, LoRE supports efficient on-the-fly adaptation: embeddings can be updated in real time as graphs change, without full retraining. Its reconstruction mechanism acts as a self-supervised training signal that improves embedding robustness, yielding improved performance compared to existing approaches. Extensive experiments demonstrate that LoRE consistently matches or outperforms baseline results while maintaining stability under dynamic conditions. Qualitative analyses further show that LoRE produces more separable and compact clusters in embedding spaces. Together, the results underscore its enhanced generalizability and practical value as a global, task-agnostic embedding method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LORE, a self-supervised graph embedding framework. The authors design it to simultaneously address four key challenges in real-world graph applications: neighborhood-invariance (P1), obedience to the open-world assumption (P2), locality-awareness (P3), and adaptability to graph dynamics (P4). The core mechanism is a model that reconstructs a node's "independent" embedding ($\psi$) using only the base embeddings ($\phi$) of its first-order neighbors. This reconstruction is paired with a node-level contrastive loss, which is designed to respect the OWA under the assumption that it is less likely for two distinct nodes to share an entire neighborhood than a yet unobserved edge. The method's effectiveness is demonstrated on node classification tasks for both simple graphs and knowledge graphs, as well as in a dynamic graph setting.

### Strengths
The primary strength is its conceptual design, which directly confronts several practical challenges of graph embedding. More specifically, 

1. The reconstruction-based mechanism is inherently inductive, providing an "on-the-fly" method for generating embeddings for new nodes (P4), which is a key requirement for dynamic graphs.
2. By design, the framework enforces neighborhood-invariance (P1), ensuring that structurally identical nodes produce identical embeddings.
3. The shift from traditional edge-level negative sampling to a node-level contrastive loss on full neighborhood reconstructions is a well-reasoned and strong attempt to better respect the OWA (P2).

### Weaknesses
W1
The paper would be stronger if it explicitly stated the key assumptions underlying its 1-hop reconstruction mechanism. The method succeeds only if the first-order neighborhood contains sufficient information to define a node's embedding. This is a strong assumption that may not always hold.

W2
I appreciate that the authors attempted to design a loss function that preserves local geometry (P3). However, the justification for the novel and complex 'perspective-preserving' loss ($\delta_{LORE}$) is not fully justified. It is unclear why this anchor-based loss is essential. The paper would benefit significantly from an ablation study that compares $\delta_{LORE}$ directly against more standard contrastive loss functions (e.g., a margin-based cosine similarity or InfoNCE). Without this, it is unclear if the added complexity is necessary or if a simpler, more common objective would achieve similar results.

W3
The empirical results show a very marginal performance gain over the best-performing baselines (e.g., GAT/RGAT). Given the added complexity of the model (e.g., two sets of embeddings per node, a novel loss function), this small improvement in accuracy appears to be insufficient. I am guessing that the small margin is attributed to the easiness of the benchmarks, namely that the baseline models could already achieve near optimal performance and the small improvement by the propose method might be significant but appear to be marginal. I suggest the authors to find challenging benchmarks where baseline models fails but the proposed model works. This reveals more clearly the cases in which the proposed model stands out.

W4
The term "independent node embedding" ($\psi$) is confusing to me. The paper states it is "independent" of the node's own parameters, but it is not independent precisely, i.e., it depends on the base embeddings of its neighbors, and the entire training objective is to make it coherent (i.e., dependent) with the focal node's own base embedding ($\phi(v)$). This choice of terminology makes it unclear, rather than clarifies, the conceptual design of the model.

### Questions
Q1: What are the key underlying assumptions about the data for the model to be successful?
Q2: Do we really need every single component to achieve this performance? Is the local geometry preservation really essential to improve the task performances?
Q3: Can you showcase more clearly the cases in which the proposed model stands out?
Q4: Is "independent node embedding" independent? If so, in what sense precisely in math?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes LoRE, an attention-based “local self-reconstruction” graph embedding framework. Its core idea is to perform self-reconstruction on randomly initialized base embeddings using only the target node's neighborhood and relation types, thereby forming independent embeddings that satisfy neighborhood invariance (P1). Simultaneously, it mitigates violations of the open-world assumption (OWA) (P2) through node-level negative samples and an angle loss preserving perspective, preserves local geometric relationships during training (P3), and enables online adaptation via forward updates without full retraining when graphs are updated (P4). Experiments on simple graphs and knowledge graphs demonstrate that LoRE achieves classification accuracy comparable to or surpassing strong baselines while showcasing its dynamic adaptation capabilities.

### Strengths
1. This paper is well-written, with clear motivations and solutions.
2. It achieves accuracy comparable to or better than strong baselines on SG/KG benchmarks, while demonstrating advantages in parameter efficiency and training time across multi-relational scenarios.
3. Dynamic update experiments show that LoRE improves downstream classification with edge updates without requiring full retraining, demonstrating practical applicability.

### Weaknesses
1. Dynamic capabilities currently only evaluate “delete-and-reinsert” scenarios; they could be extended to cover additional scenarios such as adding new nodes and removing nodes.
2. Adding references when introducing baselines in Section 5.2 is essential. I am unclear why the authors omitted this, as it prevents readers from determining when these methods were proposed and hinders quick access to their detailed descriptions.
3. The table design lacks clarity, making it difficult to identify the top-ranked and second-ranked methods for each task.

### Questions
See Weekness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a graph embedding framework called LORE (LOcally Reconstructed Embeddings). The experimental results demonstrate the effectiveness of LORE.

### Strengths
S1. This paper claims that LORE enforces neighborhood-invariance, enables learning under incomplete graphs, preserves locality-awareness, and integrates graph updates on the fly.

S2. The experiments show that LORE achieves competitive or superior accuracies compared to some baselines.

### Weaknesses
W1. The theoretical part of the paper is not easy to follow. For example, as shown in Definition 5, A= V \times M. Later, A= V \times O. It is better to clearly present the mathematical symbols.

W2. The organization of the paper should be improved. For instance, the section of Related Work in placed on Section 3. Maybe, it will be better to move this section to Section 2.

W3. The experimental results are not convincing. For example, "community detection, node classification, and link prediction" are classical downstream tasks for graph embedding as mentioned in the Abstract. Then, it is better to demonstrate the performance of the proposed methods on these tasks to demonstrate the advantages of the proposed methods. For another example, this paper states that "Incremental updates are simulated by randomly removing roughly half of each test node's edges before training, while ensuring that at least one edge remains per node." in Section 5.3. The authors should clearly prove the correctness of this kind of simulation. Otherwise, it is difficult to understand the results presented in Table 3.

W4. Some statements in this paper are not reasonable. For example, the authors claim that "Simple undirected graphs and KGs are considered as representative graph types." Why these kinds of graph types are considered as representatives?

W5. The baselines should be improved. For example, as shown in Table 1, HOPE (published in 2016), Node2Vec (2016), GCN (2017), and GAT (2018). Why not include more recent work, such as [1]? The authors should give the reasons for the baseline selection rather than just saying that "Baselines include HOPE (Ou et al., 2016), Node2Vec, GCN, and GAT".

[1] Powerful Graph Convolutional Networks with Adaptive Propagation Mechanism for Homophily and Heterophily. AAAI 2022.

### Questions
W1-W5

### Soundness
2

### Presentation
2

### Contribution
2
