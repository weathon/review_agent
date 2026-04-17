# Bootstrap Learning for Combinatorial Graph Alignment with Sequential GNNs

- Decision: Reject
- Scores: 6, 4, 8, 2

## Abstract
Graph neural networks (GNNs) have struggled to outperform traditional optimization methods on combinatorial problems, limiting their practical impact. We address this gap by introducing a novel chaining procedure for the graph alignment problem—a fundamental NP-hard task of finding optimal node correspondences between unlabeled graphs using only structural information.

Our method trains a sequence of GNNs where each network learns to iteratively refine similarity matrices produced by previous networks. During inference, this creates a bootstrap effect: each GNN improves upon partial solutions by incorporating discrete ranking information about node alignment quality from prior iterations. We combine this with a powerful architecture that operates on node pairs rather than individual nodes, capturing global structural patterns essential for alignment that standard message-passing networks cannot represent.

Extensive experiments on synthetic benchmarks demonstrate substantial improvements: our chained GNNs achieve over 3× better accuracy than existing methods on challenging instances, and uniquely solve regular graphs where all competing approaches fail completely. When combined with traditional optimization as post-processing, our method substantially outperforms state-of-the-art solvers on the graph alignment benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel "chaining" method for the combinatorial Graph Alignment Problem (GAP), which involves finding the optimal node correspondence between two unlabeled graphs using only structural information. The core idea is to train a sequence of Graph Neural Networks (GNNs) where each subsequent GNN learns to refine the similarity matrix produced by its predecessor, leveraging discrete ranking information about node alignment quality. This bootstrap process, combined with a powerful Folklore-type GNN architecture that operates on node pairs, allows the model to iteratively improve its solutions. The method is shown to significantly outperform state-of-the-art optimization-based solvers like FAQ, especially on challenging instances such as regular graphs where traditional methods fail. Furthermore, the chained GNNs can be effectively combined with traditional solvers (e.g., as an initialization for FAQ) to create hybrid models that achieve superior performance.

### Strengths
High Novelty and Impact: The "chaining" idea is innovative and represents a significant shift from typical end-to-end learning approaches for CO problems. Successfully outperforming a strong traditional solver like FAQ on its home turf (synthetic benchmarks) is a notable achievement that could influence the field.

Compelling Empirical Results: The experimental evaluation is extensive and rigorous. The results are clear and convincing, showing dramatic improvements over baselines, particularly in high-noise regimes and on regular graphs. The use of synthetic data allows for controlled and unambiguous benchmarking.

Effective Combination of Paradigms: The paper excellently demonstrates how machine learning and traditional optimization can be combined, rather than positioned as rivals. Using the GNN to produce a high-quality initialization for FAQ is a practical and powerful insight.

Strong Ablation Studies: The paper includes valuable analyses on the effect of chain length, the "looping" technique, and the importance of training noise level, providing deep insights into the method's behavior.

Well-Identified Limitations of Prior Work: The introduction and related work sections do a good job of contextualizing the paper by clearly stating the limited success of prior learning-based methods in surpassing traditional solvers for pure CO problems.

### Weaknesses
1, Scalability and Computational Cost: The proposed method is computationally intensive. The Folklore-type GNN architecture operates on node pairs, leading to \(O(n^2)\) memory complexity, and the chaining procedure requires sequential training and inference of multiple such networks. While the linear time complexity of inference is mentioned, the practical wall-clock time and memory footprint for large graphs (e.g., n > 10,000) remain a significant concern and are not thoroughly discussed.
2, Limited Theoretical Analysis:  While the method is empirically powerful, it lacks a solid theoretical foundation. The paper does not provide theoretical guarantees on convergence or approximation ratios. Why and how the chaining procedure leads to iterative improvement is explained intuitively but not formally characterized. 
3, Evaluation on Real-World Data: The exclusive use of synthetic benchmarks, while methodologically sound for a CO paper, leaves open the question of performance on real-world graphs. Real-world graphs often have features, community structure, and noise patterns that differ from the synthetic models used here. Demonstrating effectiveness on even a few real-world datasets would greatly strengthen the paper's practical claims.
4, Clarity of the "Looping" Mechanism: The "looping" technique (reusing the final GNN \(g^{(L)}\) multiple times) is an interesting empirical finding, but its rationale is somewhat unclear. Why does the final GNN generalize to its own outputs in subsequent iterations? A deeper investigation or discussion of this phenomenon would be beneficial.

### Questions
Scalability: What is the largest graph size (n) you can feasibly handle with your current implementation, and what are the primary bottlenecks (memory, training time, inference time)? Are there strategies to make the chaining procedure or the Folklore-GNN more scalable?

Generalization: This method is trained on a specific graph model (e.g., ER) and noise level. How would you expect it to perform on a graph from a completely different distribution (e.g., a small-world or scale-free network)? Have you conducted any out-of-distribution tests?

Theoretical Underpinnings: Can you provide any theoretical intuition or analysis for why the chaining procedure works? For instance, can you frame it as a form of fixed-point iteration or relate it to a known optimization algorithm? 

Ablation on Architecture: How critical is the expressive Folklore-GNN architecture to the success of chaining? Could a less expressive but more scalable MPNN be used in the chain if given more steps, or is the high-quality initial output from the Folklore-GNN indispensable for bootstrapping?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Combinatorial graph alignment aims to find the optimal permutation that best represents the node correspondence between two graphs. The authors proposed a new iterative learning framework using a sequence of graph neural networks (GNNs). Each GNN refined the solution predicted by the previous GNN by additionally considering the alignment quality of the prior prediction. The process began with an initial permutation prediction, followed by the computation of a ranking score for each node that reflects its alignment quality. This ranking was then incorporated as additional input--alongside the adjacency matrix--to the next GNN, which generates a refined prediction. Furthermore, the authors designed a new GNN architecture, FGNN, which exhibits improved expressivity. Experimental results demonstrated that both FGNN and the proposed iterative learning framework improve alignment accuracy.

### Strengths
(S1) The authors introduced a novel learning methodology for combinatorial graph alignment.

(S2) The explanation of the learning framework is clear and easy to follow.

### Weaknesses
(W1) Organization of writing:
While the experimental results showed that both the model architecture and the chaining learning mechanism improve alignment accuracy, the architectural details of FGNN are only described in the appendix. This reviewer believes that at least one paragraph describing the model architecture should be included in the main text.

(W2) Lack of runtime comparison:
Runtime is a critical metric for combinatorial optimization problems. Although the authors mentioned that runtime comparison is challenging, they do not provide sufficient justification for this difficulty. According to the appendix, FGNN was inspired by Folklore-type GNNs, which offer higher expressivity at the cost of scalability. Therefore, the use of FGNN may increase computational time, and this aspect should be discussed in more detail.

(W3) Limited baselines:
The authors only compared their approach with non-neural-based optimization methods. To better assess the proposed framework, comparisons with existing GNN-based methods and combinations of GNNs with their iterative learning mechanism should be included. Also, the baseline should include various graph alignment (graph matching) methods in the literature.

(W4) Lack of motivation
The paper did not clearly explain why solving the combinatorial graph alignment problem is important, what broader implications it has, or how plausible the problem setting is where only the adjacency matrix is utilized. A more concrete motivation should be presented in the introduction.

### Questions
The authors used ranking-based evaluation to measure alignment quality, which prevents gradient flow. Instead of using ranking, how about introducing a differentiable metric--such as one based on the difference between the softmax of the similarity matrix and the ground-truth permutation matrix?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors present a method for the graph alignment problem that uses several chained GNNs. At each step, a similarity matrix is computed, and then a ranking score for the nodes is obtained according to the permutation resulting from this similarity matrix. These rankings are used as features in the following step. The results are promising.

### Strengths
The paper is generally well written, and it includes references to several works that employ different methodologies. The idea is simple (and I consider this a strength), and the results are very good.

### Weaknesses
I would have liked to see at least one example involving real graphs, as well as a runtime comparison (during inference).

### Questions
The accuracy (eq (3)) measures the coincidence between the "optimal" solution $\pi^{A\to B}$ and the permutation $\pi$. However, given two graphs, there may exist multiple permutations that perfectly align them. This is particularly true for certain regular graphs, and, more generally, for graphs that are not “friendly” (see "On convex relaxation of graph isomorphism", PNAS, Aflalo, Bronstein, and Kimmel), or that do not satisfy other more general conditions (see "On spectral properties for graph matching and graph isomorphism problems", Information and Inference: A Journal of the IMA, by Fiori and Sapiro).
In any case, this accuracy should not be used to evaluate the performance of graph alignment methods. I know that asymptotically most graphs under the ER model have trivial automorphism groups and therefore this issue does not arise, but still, I would remove this evaluation metric from the paper, or at least put a comment on that.

In the description of the FAQ method (lines 166-172), it seem like it solves (6), but the solution of (6) is actually a doubly stochastic matrix, and in fact the last step of the FAQ method consists on a projection step with the Hungarian algorithm, just like Proj.

On line 377: for regular graphs, the convex relaxation may return any doubly stochastic matrix in the convex set that contains (among others) the true permutation matrix/matrices and the barycenter. In this case, the convex problem does not have a unique minimizer, and thus the obtained solution depends on the algorithm used.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a learning-based method for unlabeled graph alignment exploiting a bootstrap chaining effect over sequential GNNs that operate on node pairs. Each GNN refines a similarity matrix produced by its predecessors over iterations. The central claim of the paper is that it is, after all, possible for a GNN-based method to outperform a traditional optimization method on the combinatorial optimization problem of graph alignment. The claim is meant to be corroboreated by a comparison to FAQ, a method for graph alignment introduced in 2015. All work on the graph alignment problem since 2015 is ignored. Experimentation with synthetic data, including a combination with traditional solvers as post-processing, shows an improvement over other GNN-based methods and FAQ.

### Strengths
S1. Novel proposal of bootstrapping GNNs.
S2. Novel suggestion of working on node pairs.
S3. Introduction of challenging benchmark of synthetic regular graphs.

### Weaknesses
W1. Ignores all work on graph alignment since 2015; recent advances include SGWL [NeurIPS 2019] and FUGAL [NeurIPS 2024], which outperform the FAQ method considered to be state-of-the-art in this paper.
W2. Experiments limited to synthetic data; real-world data pose unique challenges, as recent work has shown.
W3. Post-processing limited to FAQ and a straightforwards projection by the Hungarian algorithm.

### Questions
How does the proposal perform compare to state of the art methods?

### Soundness
1

### Presentation
3

### Contribution
2
