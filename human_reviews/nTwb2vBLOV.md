# Rethinking the Power of Graph Canonization in Graph Representation Learning with Stability

- Decision: Accept (poster)
- Scores: 6, 6

## Abstract
The expressivity of Graph Neural Networks (GNNs) has been studied broadly in recent years to reveal the design principles for more powerful GNNs. Graph canonization is known as a typical approach to distinguish non-isomorphic graphs, yet rarely adopted when developing expressive GNNs. This paper proposes to maximize the expressivity of GNNs by graph canonization, then the power of such GNNs is studies from the perspective of model stability. A stable GNN will map similar graphs to close graph representations in the vectorial space, and the stability of GNNs is critical to generalize their performance to unseen graphs. We theoretically reveal the trade-off of expressivity and stability in graph-canonization-enhanced GNNs. Then we introduce a notion of universal graph canonization as the general solution to address the trade-off and characterize a widely applicable sufficient condition to solve the universal graph canonization. A comprehensive set of experiments demonstrates the effectiveness of the proposed method. In many popular graph benchmark datasets, graph canonization successfully enhances GNNs and provides highly competitive performance, indicating the capability and great potential of proposed method in general graph representation learning. In graph datasets where the sufficient condition holds, GNNs enhanced by universal graph canonization consistently outperform GNN baselines and successfully improve the SOTA performance up to $31$%, providing the optimal solution to numerous challenging real-world graph analytical tasks like gene network representation learning in bioinformatics.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work focuses on the problems of the expressive power of graph neural networks (GNNs) where they aim to enhance the expressive power of GNNs by graph canonization. To this end, the paper first reveals the trade-off between the expressivity and stability in GNNs enhanced by graph canonization, showing that graph-canonization-enhanced GNNs maximize their expressive power at the cost of model stability. The paper further proposes a universal graph canonization to tackle this trade-off under a sufficient condition. The advantages of the proposed models against some GNN baselines are validated by their experimental results on both synthetic datasets and several benchmark datasets.

### Strengths
1. The paper is clear and well-structured.

 2. The theoretical results that reveal the trade-off between expressivity in distinguishing non-isomorphic graphs and the stability in GNNs enhanced by graph canonization are interesting. The theoretical findings have shown the limitations of applying graph canonization to enhance the expressive power of GNNs. The proposed universal graph canonization to tackle the trade-off between expressivity and stability is novel. 

 3. The experimental results of the proposed model, UGC-GNN, on several bioinformatical graph datasets are impressive and consistently outperformed some well-known GNN baselines.

### Weaknesses
1. Lack of experiments to compare the expressive power of the proposed models with high-order GNNs [1, 2] and subgraph-based GNNs [3, 4] in distinguishing non-isomorphic graphs. Since high-order GNNs also enhance the expressive power of GNNs, it would be helpful to conduct experiments to compare the expressive power of the proposed models and high-order GNNs. Are graph-canonization-enhanced GNNs more powerful than high-order GNNs and k-WL test algorithms?

 2. Lack of computation cost comparison with k-WL-GNNs. High-order GNNs and subgraph-based GNNs may suffer from high computational cost when applied to large-scale graphs. The paper asserts that the proposed models are more efficient with a significantly low space and computation cost. It would be nice to provide empirical evidence to support the claim.

 3. As shown in Table 7, the GC-GNN doesn't consistently enhance the performance of the GNN backbone. In some instances, the performance even diminishes. It would be beneficial to delve deeper into which graph properties might influence this fluctuation in performance when integrating the GNN backbone with GC-GNN or UGC-GNN.

 4. The universal graph canonization problem is NP-hard. While the paper has provided a sufficient condition to compute the discrete coloring and discussed the applicability in several application scenarios, finding an injective function $l(v|\mathbb{G})$ might be elusive or challenging for general graph learning contexts. Furthermore, if there could be multiple choices of $l(v|\mathbb{G})$, it is unclear what the impact of applying different choices of $l(v|\mathbb{G})$ is on the performance of the proposed model.

Overall, the paper is well-written and easy to follow. The theoretical findings on the trade-off between expressive power and stability in graph-canonization-enhanced GNNs are interesting and the proposed universal graph canonization to tackle the trade-off is novel. Despite the potential high computational cost induced by the graph canonization algorithms and the effectiveness of the proposed models that might not be generalized to general graph datasets, the paper has provided some insights into the expressive power of GNNs enhanced by graph canonization methods. Therefore, the reviewer is inclined to accept the paper.

Reference

 [1] Weisfeiler and leman go neural: Higher-order graph neural networks, AAAI 2019.

 [2] Provably powerful graph networks, NeurIPS 2019.

 [3] Nested graph neural networks, NeurIPS 2021.

 [4] Equivariant Subgraph Aggregation Networks, ICLR 2022.

### Questions
Please refer to weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Authors propose to tackle GNN expressivity issues by graph canonization. Either using ordering of nodes implied by the referene data distribution (e.g. image or voxel datasets that can give canonical ordering of nodes) or by running existing fast canonization algorithms. Inclusion of this ordering is shown to improve GNN expressive power and performance on real-world datasets.

### Strengths
I find the canonization of graphs to be a very intriguing avenue of making GNNs better or to automatically tailoring generic GNNs for a given data distribution if the canonization is specific for that distribution.

The authors do show that including such canonical order does improve the performance of backbone GNN architectures.

### Weaknesses
In general I find the paper quite problematic. Mainly due to the total disregard for related work. Let me split this in three parts:

1) Including ordering of nodes as features is very close to other feature augmentation techniques, widely used for GNN expressive power improvements, such as adding random walk embeddings (https://arxiv.org/pdf/2110.07875.pdf), graph laplacian embeddings (https://arxiv.org/abs/2202.13013 https://arxiv.org/pdf/2201.13410.pdf) or even just random features (https://www.ijcai.org/proceedings/2021/0291.pdf). These are not mentioned among the types of expressive GNN architectures (only higher order and subgraph GNNs are mentioned), while conceptually they are very similar (add pre-computed features to increase power). Neither any of the experiments compare to these existing models. 

2) The selection of discussed subgraph-based GNNs is quite peculiar, as it skips in my opinion some of the most popular works in this context (such as DropGNN https://arxiv.org/pdf/2111.06283.pdf and ESAN https://arxiv.org/pdf/2110.02910.pdf as well as various follow up works to these of which there has been many). They also say that subgraph GNNs are not more expressive than 2-WL, which in general is definitely not true, for example see theorem 6.5 in https://proceedings.mlr.press/v162/papp22a/papp22a.pdf where the marking GNN is very closely related to the DropGNN and one of the ESAN variants mentioned above. Also as shown in Table 1 here https://openreview.net/pdf?id=8WTAh0tj2jC experimentally many of these expresive GNNs such as ESAN or DropGNN distinguish the usual 2-WL counter example graphs. (Note that here by 2-WL following the notation in those papers I mean the folklore WL hierarchy, where 2-WL is equal to the 3-WL in the usual WL hierarchy).

3) The known results for the power of assigning unique identifiers to nodes, such as the seminal work of https://arxiv.org/pdf/1907.03199.pdf which shows that GNNs with unique IDs are universal, although generally we don't have a way to assign such IDs in a stable manner. In my eyes the UG-GNN in Tables 4 and 5 essentially just a direct application of this in cases where the data comes with a good ID assignment due to the nature of data distribution. 

Note that many more of works in all 3 categories exist, I just cited a few important ones that came to my mind as an example.

Now moving on to the experiments, its again very strange that none of the aforementioned baselines, especially the pre-computed feature ones are included. And generally almost. no expressive GNNs from the past couple of years are included in the benchmarks (e.g. ESAN, CIN, etc.). Not even giving nodes random noise/random IDs is compared against, which is a very usual expressive GNN baseline. Actually it is interesting that in Table 1 the GC-GNN does not achieve 100% test accuracy in any of the cases, usually expressive GNNs manage to hit 100% or at least 99%, with GNN augmented with random features also managing >90% test set accuracy. Random features are usually underperforming other expressive GNNs due to the poor stability of feature (or random ID) assignment, but as the authors point out their GC-GNN can also suffer from poor ordering stability.  

So now, for me its completely unclear if this proposed expresivity enhancement is any better than existing, even simplest ones.

While OGB benchmarks are mentioned, the model is not evaluated on any of them? Why is that? It is now the main standard GNN benchmark as the TU-Datasets usually have quite few graphs in them, which can induce high variance of the results.

### Questions
Please address the weaknesses above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
