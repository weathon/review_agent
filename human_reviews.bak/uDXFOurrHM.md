# Learning Partial Graph Matching via Optimal Partial Transport

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Partial graph matching extends traditional graph matching by allowing some nodes to remain unmatched, enabling applications in more complex scenarios. However, this flexibility introduces additional complexity, as both the subset of nodes to match and the optimal mapping must be determined. While recent studies have explored deep learning techniques for partial graph matching, a significant limitation remains: the absence of an optimization objective that fully captures the problem’s intrinsic nature while enabling efficient solutions. In this paper, we propose a novel optimization framework for partial graph matching, inspired by optimal partial transport. Our approach formulates an objective that enables partial assignments while incorporating matching biases, using weighted total variation as the divergence function to guarantee optimal partial assignments. Our method can achieve efficient, exact solutions within cubic worst case time complexity. Our contributions are threefold: (i) we introduce a novel optimization objective that balances matched and unmatched nodes; (ii) we establish a connection between partial graph matching and linear sum assignment problem, enabling efficient solutions; (iii) we propose a deep graph matching architecture with a novel partial matching loss, providing an end-to-end solution. The empirical evaluations on standard graph matching benchmarks demonstrate the efficacy of the proposed approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper formulated the partial graph matching problem as an optimal partial transport problem, enabling partial assignments (not have to be a bijective mapping) between a source graph and the target graph.The linear solver Hungarian algorithm is adopted to the LAP objective.

### Strengths
1. Formulating partial graph matching problem into an optimal transport problem is natural. 
2. The objective is a linear assignment problem (LAP) and easy to be optimized.
3.  Both qualitative and quantitative results on several commonly used datasets valid the effectiveness of the proposed formulation.

### Weaknesses
1. The main contribution of the paper is to formulate the partial graph matching problem into an optimal partial transport problem, which is not new [1,2] and is a common relaxation of the original quadratic objective of graph matching involving both vertices and edges constraints (into a linear assignment problem).
2. Line 065 is not 100% accurate since there are some efficient algorithms (e.g. [3] is quite intuitive and easy to implement as the k-LAP is transformed into a standard LAP that can be efficiently solved by the Hungarian algorithm) targeting towards k-assignment problem. As such, manually filtering out infeasible assignments with a hard threshold in Theorem 5.1 is not necessary.
3. Some important references are missing [4,5].
4. The objective of Eq(9) is almost equivalent to the first order term of Koopmans-Beckmann’s QAP with partial matching constraints in Eq(10), and the network design is the same as GCAN, together with the common linear solver Hungarian algorithm making the contributions both network design and optimization insight a bit marginal. In fact, one promising and more insightful way for the paper to stand out among others is to explicit formulate graph edges into optimal transport objective or constraints rather only implicitly involved into feature aggregation by GCN.

To conclude, the paper proposed an optimal transport objective with partial matching constraints for partial graph matching problem. However, I haven't seen any special design for the partial graph matching nor optimal transport.



[1] Pan et al. Subgraph Matching via Partial Optimal Transport. arxiv, 2024
[2] Xu et al. Gromov-Wasserstein Learning for Graph Matching and Node Embedding. ICML, 2019.
[3] Volgenant et al. Solving the k-cardinality assignment problem by transformation. European Journal of Operational Research, 2004.
[4] Fey et al. Deep Graph Matching Consensus. ICLR, 2020.
[5] Gao et al. Deep graph matching under quadratic constraint. CVPR, 2021.

### Questions
Theorem 5.1 about filtering infeasible assignments with a hard threshold seems to be not necessary even without a k-LAP algorithm. Since the proposed method will output an optimal partial matching in anyway, why it needs to filter out some infeasible assignments at the very beginning?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces OPGM, which addresses the partial graph matching problem. It begins by establishing theoretical connections between optimal partial transport and the partial matching problem. The paper then presents a method that leverages this formulation to learn matching bias and a cost affinity matrix, improving the outcomes of partial graph matching. Experimental results on various datasets demonstrate that this method is comparable or better than existing approaches.

### Strengths
The motivation behind the study is clear, and the results presented looks convincing.

### Weaknesses
1. The connection between optimal transport and the graph matching problem is already well-established, making it seem redundant to re-establish this connection in the partial setting.

2. The use of the term “optimal” in the title may be misleading. A significant challenge in partial matching is setting thresholds ($\rho$, $\alpha$, $\beta$ in this paper) to eliminate redundant points. The paper assumes these thresholds are predefined, which undermines the rigor of claiming their results as “optimal”.

3. PascalVOC is still the most important benchmark for the graph matching community, including its result only in the appendix does not seem appropriate, especially when this method does not do better on this benchmark.

4. The paper omits two critical references [1,2], both of which have achieved better results on PascalVOC.

[1] Nurlanov et al., "Universe Points Representation Learning for Partial Multi-Graph Matching"

[2] Jiang et al., "Learning Universe Model for Partial Matching Networks over Multiple Graphs"

### Questions
1. How effective is the applied bias term in identifying non-matching points, and what are the false positive and negative rates? It would be important to quantify the effectiveness of the proposed scheme.

2. What would the performance of other methods look like if they simply applied a thresholding scheme on their cost matrix similar to this study? An ablation study would be appreciated.

3. How is the parameter $\rho$ chosen? Is it chosen by some heuristics? Fig. 2 (middle) suggests that $\rho$ values supporting better performance are quite narrow (0.35-0.4), and these values vary across different tasks. Determining this parameter is one of the most challenging aspects of partial matching, so it is essential to understand how to set $\rho$ without extensive trial and error.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new objective for partial graph matching based on total variation and partial optimal transport. The paper shows that this formulation leads to a specialized version of optimal transport, and one optimal solution exists in the unit-weight setting. 
Then, the paper proposes to solve the problem by adapting the Hungarian algorithm and training a neural network to learn two components: the matching cost matrix and the biases to represent the matched and unmatched nodes.

The experiments cover protein-protein interaction networks and image key-point matching, showing results under different noise levels and analyzing the method's efficiency.

### Strengths
1) The paper addresses a challenging and fundamental problem, with an impact on several down stream domains. The derivation lead to a formulation includes also a computational complexity statement for the problem, which seems useful for future works.
2) The general formulation is interesting and leads to a concrete network formulation. The network seems practical, and the authors have attached the code, so I am confident the work can be reproduced.
3) Although the performance does not push the state of the art further, the method seems particularly more computationally efficient.

### Weaknesses
1) The method requires setting a series of hyperparameters tailored for the domain, which might be difficult in practice. For example, by my understanding, the parameter \rho seems quite important and encodes the amount of partiality. This is quite an assumption, and it would be nice to find some heuristics to fix it automatically (e.g., the ratio of the number of nodes or the graph radius, etc.). Could you offer a comment on this, and in case a possible heuristic? Is it possible to explicitly relate \rho to the amount of partiality?
2) Since this work is a bit outside my direct expertise, I had a hard time grasping all the details, especially building an intuition. The paper itself does not show visual results, and then it is not straightforward to understand how concretely the proposed methodology differentiates in terms of artifacts (e.g., is the error caused by wrong matching bias and partiality identification, or more about mismatching?) would be interesting to add some qualitative examples, especially for keypoint, that is particularly easy to interpret. Did you observe any specific failure mode or particularly challenging setting? Could you provide a visualization to support the intuition about the main source of the error?

### Questions
Please, refer to the weaknesses section comments for the major questions I would see addressed in the rebuttal.

Minor:
1) line 075: multiple missing spaces: "transportidentifies", "transportcan", "transportdoes"
2) The recent work "Spectral Maps for Learning on Subgraphs", Pegoraro et al., NeuReps 2023, seems relevant for the paper. In particular, the paper suggest to use spectral properties to infer properties on subgraphs, and also correspondence. Similarly "A functional representation for graph matching", Wang et al., PAMI 2019, propose to use spectral properties to compute matching on (full) graphs. I believe a discussion on this line of works would be interesting.
3) How does the method perform in case of rewiring of the partiality?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new optimization framework for partial graph matching, a variant of traditional graph matching that allows for some nodes to remain unmatched. This flexibility accommodates complex real-world scenarios, where not all entities have a direct counterpart. The authors introduce an optimization objective that balances matched and unmatched nodes using concepts from optimal partial transport, providing an efficient, cubic-time complexity solution via the Hungarian algorithm.

### Strengths
- The paper applies concepts from optimal partial transport to graph matching, resulting in a framework that handles partial assignments rigorously. By connecting the problem to the linear sum assignment, the authors leverage the Hungarian algorithm, achieving cubic time complexity—an important advancement for handling larger datasets.

- Experimental results on multiple datasets illustrate the proposed method’s robustness and competitive edge over established baselines, particularly in terms of matching F1-scores and runtime efficiency.

- The end-to-end architecture, with its unique loss function, offers practical advantages for data-driven applications, adapting dynamically to different matching scenarios.

### Weaknesses
- The paper shows that the proposed model’s performance decreases dramatically with increasing noise level, with the PPI dataset. Although the proposed method performs better than other methods generally. The performance decrease rate is similar or enven larger than other methods, as shown in Fig 2 (left). 
- It seems the approach’s performance is sensitive to the unbalancedness parameter ( \rho ), as shown in sensitivity analyses.
- For applications where annotations are inconsistent or sparse, such as biological or social network datasets, the model’s dependence on well-annotated data could limit its generalizability.

### Questions
please address the points mentioned as weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
