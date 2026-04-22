# TREEGEAR: Learning Graph Edit Distance with Zero Ground-truth Labels

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Graph Edit Distance (GED) is a fundamental measure for assessing similarity between graphs, with broad applications across domains such as bioinformatics, cheminformatics, and social network analysis. Unfortunately, computing exact
GED is NP-hard. Besides a number of approximation algorithms, neural methods have emerged as a promising solution to this challenge. However, the training of these neural models requires a large number of ground-truth labels, which
is computationally expensive to obtain due to the NP-hardness, thereby hindering their scalability. In this work, we introduce a novel framework, TREEGEAR for learning GED without the need of ground-truth GED labels. Our approach
uses structural supervision from tree edit distances (TED), which can be computed in polynomial time, enabling the model to learn meaningful representations from approximate signals. Unlike existing approaches that directly regress to
GED, TREEGEAR learns pairwise node mappability scores through node embeddings, on which, we apply a neighbor-biased mapper to derive the best possible edit paths between two graphs. This novel reformulation enables strong out-of-distribution generalization, interpretability, and better alignment with the properties of the true GED. Extensive experiments across GED benchmarks demonstrate
that TREEGEAR achieves state-of-the-art results, beating both non-neural and neural baselines that are trained on 100% ground-truth GED. Moreover, TREEGEAR is architecture-agnostic and generalizes effectively to unseen graphs, making it suitable for real-world deployment across diverse graph domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper provides a novel method to approximate the graph edit distance between two graphs. The proposed method uses tree edit distance as a proxy, making it scalable and more efficient compared to related methods.

### Strengths
The idea of training GNN on TED is interesting, allowing one to leverage the difficulties of supervised training of the GNN. The paper is relatively well written.

### Weaknesses
The underlying idea of training the GNN using the tree edit distance is interesting. This strategy shifts the problem to the selection of the trees from the graphs under study. This is a difficult task and requires some heuristics in order to properly select a pair of trees that cover the graph regions where structural differences are most likely to occur.
Such heuristics could be questionable, even though there are some proposed hypotheses.

The proposed strategy of approximating the GED through tree representations is not new. There are many related methods that are cited in the submitted paper, such as:
- Bause, F., Permann, C., & Kriege, N. M. (2024, August). Approximating the graph edit distance with compact neighborhood representations. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 300-318). Cham: Springer Nature Switzerland.
- Xu, M., & Chang, L. (2025). Graph Edit Distance Estimation: A New Heuristic and A Holistic Evaluation of Learning-based Methods. Proceedings of the ACM on Management of Data, 3(3), 1-24.

It is not clear why in the proposed formulation, the graph neural network does not directly predict the graph edit distance, but its output needs to be transformed with another function. It would have been more relevant to provide an end-to-end architecture.
 
The experimental analysis is in favor of the proposed method. However, the authors have chosen not to present all the comparable methods in all the tables. For instance, Table 2 providing the EMR compared the proposed method with only 3 neural methods, while Table 1 includes more competitive neural methods, such as H2MN and GraphEdx. Furthermore, it is not clear why the authors did not compare to non-neural methods, such as the 7 methods given in Table 1.
Moreover, it would have been relevant to consider the same datasets in all the tables.
At the end, it seems cherrypicking the best results for the proposed method.

### Questions
No further comments.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework to learn graph edit distance (GED) with graph neural networks (GNN). There are two novel contributions: (1) using polynomial-time tree edit distance (TED) for weak supervision instead of NP-hard GED, and (2) a _neighbor-biased mapper (NBM)_ to compute edit paths using TED-trained GNN to score pairs of nodes. Experimental results show superior performance than baselines trained with direct GED supervision. Further analysis includes out-of-distribution generalization, size generalization, robustness and ablation studies.

### Strengths
The paper presents two novel and clever ideas:
(1) weaker supervision from TED is enough to achieve accurate GED predictions,
providing an alternative to exact GED computations for training data generation, and,
(2) the NBM algorithm to produce node matchings using TED-trained GNN embeddings,
and computing GED from that (this second part is well known).

TreeGear achieves much better results than the baselines.

Ablations establish the importance of the NBM component and the HybridTree construction used therein.

### Weaknesses
My main concern with this paper is that it does not come full circle
in experimentally demonstrating the central motivation for using TED supervision
instead of GED.
Lower cost of training data generation would enable training on larger graphs
than existing methods,
but no experiment setup considers the graph-size scalability of TreeGear.

From Table 3, major gains are from NBM.
TED is worse than GED, which seems to be reasonable to compute for all experimental settings
considered in this paper.
Hence GED + NBM seems the more useful contribution.
Even for larger graphs approx GED with timeout + NBM might work better than TED.

The paper is imprecise and unpolished. Examples:
* Important parts such as hybrid tree construction were hard to understand.
* I found some examples of bad citations which do not fit well with the sentence. e.g. L040, L212
* Lots of citations are missing parentheses.

There are other works which compute GED with a node matching step [1, 2, 3, 4] which could be
baselines or at least should be discussed as related work,
but are missing (including from discussion, e.g. in L068-071) .

[1] MATA*: Combining learnable node matching with A* algorithm for approximate graph edit distance computation. In CIKM, 2023.

[2] Computing approximate graph edit distance via optimal transport. Proceedings of the ACM on Management of Data, 3(1):1–26, 2025.

[3] NOAH: Neural-optimized A* search algorithm for graph edit distance computation. In ICDE, 2021.

[4] GRAIL: Graph edit distance and node alignment using llm-generated code. In ICML, 2025.

### Questions
1. Can you show some results for larger graphs where exact GED computation is prohibitive?

2. How does TED compare with exact GED computation but with a reasonable timeout?

3. How does weakly supervised TreeGear compare with unsupervised GED methods, e.g. [1]?
(not published at a conference, so wouldn't change my opinion of this paper)

4. On L268, what is a "chordless cycle"? Please define for readers.

5. In Def. 5 what if the root node is deleted?

6. In Fig. 4 the input graph is symmetric wrt nodes 2 and 3. How does the hybrid tree construction break the symmetry to give an output graph which is asymmetric wrt nodes 2 and 3?

7. In Fig. 2, "trained GNN" would be better than "pretrained GNN".


[1] EUGENE: Explainable Structure-aware Graph Edit Distance Estimation with Generalized Edit Costs

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
3

### Summary
This paper proposes a framework for learning the graph edit distance (GED) and alleviates computational complexity by learning the tree edit distance (TED) (i.e., a weakly supervised GED). Specifically, to leverage TED, the authors propose a method to transform graph structures into tree structures in the following order: 1) root selection, 2) hybrid tree construction, and 3) a Neighbor-Biased Mapper (NBM). GED is then computed by aligning the mapped nodes using four operators: node substitution, edge deletion, node insertion, and edge insertion. A GNN is trained to predict the edit cost after selecting the final matching path. The proposed approach is efficient in terms of computational complexity and exhibits strong generalization across diverse graph domains. Experiments on graph edit distance prediction tasks show that the proposed method outperforms prior approaches.

### Strengths
* The idea of weakly supervised GED leveraging TED is interesting and demonstrates its efficiency.
* The ablation study is well-designed to demonstrate the effectiveness of each component in the proposed method.
* The proposed method can generalize to diverse domains and offers interpretability.

### Weaknesses
* My primary concern is that the proposed method is evaluated only on graph edit distance prediction tasks, which limits its overall contribution and scope of application. I recommend conducting experiments on a wider range of tasks and comparing it against other representation learning methods.
* The performance heavily depends on the quality and diversity of the node embeddings fed into the NBM, preventing the NBM from fully leveraging its advantage by selecting the minimum edit path.

### Questions
* The performance tendency two molecular datasets are different (i.e., ADIS and molhiv). Could you explain why this difference occurs?
* Can learning the graph edit distance mitigate the oversmoothing problem?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TREEGEAR, a framework for learning to approximate the Graph Edit Distance (GED) without relying on ground-truth GED supervision, which is costly to compute due to the NP-hard nature of GED. The key idea is to extract (so called hybrid) rooted trees from graphs and use tree edit distance (TED)—which is polynomial-time computable—as a proxy supervision signal. A GNN is trained on pairs of such trees to produce node-level embeddings that capture structural similarity. At inference time, node embeddings are used with a neighbor-biased mapping (NBM) procedure to produce a node correspondence, from which an edit path and corresponding GED upper-bound can be derived.

### Strengths
The motivation is well-founded: current neural GED approximators rely on expensive ground-truth GED labels, restricting scalability and practical applicability. The idea of replacing GED supervision with a relatively easily available tree distance based proxy is conceptually appealing.

### Weaknesses
As far as I understand, the method uses tree edit distance as a proxy signal to train node embeddings, and then applies an injective node mapping strategy (NBM) to derive a node correspondence and subsequently an edit path. While the choice of tree representation (e.g., hybrid trees vs. BFS trees) may affect performance to some extent, the overall inference procedure, node matching based on pairwise node affinity (resulting in an interpretable edit path),  is already well-established. Thus, beyond the use of TED-supervised training, I do not see anything new here.

Now, my primary concern is the role and necessity of the TED-supervised training process itself. If the final node alignment depends only on the relative ordering of node similarity scores (as is the case with NBM), then it is unclear what benefit the neural training stage provides. In fact, untrained GNNs with distance based downweighting can already capture neighborhood-aware structural information. Then what is the model actually learning when trained to regress tree edit distance? 

Lastly,  AIDS and LINUX benchmarks used in the experiments are known to have structural leakage [1], meaning that graphs in the test split are often isomorphic to the training graphs. As such any evaluation currently done on these datasets is unreliable. 


[1] Position: Graph Matching Systems Deserve Better Benchmarks. In Forty-second International Conference on Machine Learning Position Paper Track.

### Questions
Quesitons are mostly based off the mentioned weaknesses. 
- The NBM-based inference relies primarily on the relative ordering of node similarity scores. Given this, what is the justification for training the GNN to regress absolute TED values, rather than using an untrained or self-supervised embedding scheme?

- Beyond substituting GED labels with TED supervision, are there additional conceptual or methodological contributions that I may be overlooking? 

- Several of the evaluation datasets (e.g., AIDS, LINUX) are known to exhibit structural leakage, which can inflate performance metrics. How do the authors interpret their results in light of this?

### Soundness
1

### Presentation
2

### Contribution
2
