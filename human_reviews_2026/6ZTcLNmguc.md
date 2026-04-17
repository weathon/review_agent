# Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
The graph edit distance (GED) is a widely used graph dissimilarity measure that quantifies the minimum cost of the edit operations required to transform one graph into another. Computing it, however, involves solving the associated NP-hard graph matching problem. Indeed, exact solvers already struggle to handle graphs with more than 20 nodes and classical heuristics frequently produce suboptimal solutions. This motivates the development of machine-learning methods that exploit recurring patterns in problem instances to produce high-quality approximate solutions. In this work, we introduce Gelato, a graph neural network model that constructs GED solutions incrementally by predicting a pair of nodes to be matched at each step. By conditioning each prediction autoregressively on the previous choices, it is able to capture complex structural dependencies. Empirically, Gelato achieves state-of-the-art results, even when generalizing to graphs larger than the ones seen during training, and runs orders of magnitude faster than competing ML-based methods. Moreover, it remains effective even under limited or noisy supervision, alleviating the demand for costly ground-truth generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a novel method to approximate the graph edit distance ( GED). The major contribution is the use of a neural network to solve the GED sequentially.

### Strengths
The proposed method is interesting as it allows to construct the edit paths.

### Weaknesses
The designation of “autoregressive” seems not to be correct, or at best not well justified. The proposed method can be clearly defined as sequential. At the end, it uses roughly a search-based strategy in the same spirit as A* and related methods.

A major issue is that there are no guarantees of optimality for the proposed method. Moreover, it is well known that sequential optimization is non-optimal in general.

While the authors provide some theoretical results, they cannot be explored in practice. For instance, Lemma 1, which motivates the greedy selection, cannot be exploited because the optimal function cannot be computed in practice, making these results intractable.

It would be relevant to provide a comprehensive ablation study beyond Section 5.3. For instance, it is not clear if the batch normalization and the residual connections are beneficial or not. Moreover, the paper does not justify the choice of the values of most hyperparameters, such as the number of layers of the GIN set to 5, embedding dimension, number of randomly selected pairs, resembling with k=32...

The computational complexity needs to be studied in depth. The authors mainly present the inference runtime, but not the training runtime.

The expression “Any state is also a terminal state” is misleading.

There are several spelling and grammatical errors that can be easily identified and corrected, such as “We define a the set…”, as well as GINE.

### Questions
No further comments.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
GELATO solves the graph edit distance (GED) problem
by reformulating it as a sequential decision making process.
The graph neural network (GNN) is trained to predict one matching node pair at a time,
given a partially matched pair of graphs,
with efficiency and soundness considerations such as
overlapping state space and automorphisms,
allowing for autoregressive inference to generate the full matching
to compute the GED.
Comprehensive experiments show that GELATO achieves high solution quality compared to baselines,
while taking less time, especially compared to approaches with an non-neural matching/search component.
Further analyses include generalization on larger graphs, robustness under limited supervision and ablations.

### Strengths
The paper proposes a novel method, with well-motivated and carefully-considered components.

The experiments are comprehensive and the results are strong.

The experiments support the main motivations for this research,
e.g. generalization to larger graphs and faster inference times.

### Weaknesses
While the paper is well-written overall,
some more-involved parts of the method could be better explained,
especially the state-space reduction and automorphism considerations.

### Questions
1. What is the basis for selection of 3 baselines out of 8 in Section 5.2 / Figure 5?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the problem of Graph Edit Distance (GED) estimation by framing it as a sequential decision-making task. The authors propose GELATO, an autoregressive framework built on a GNN backbone, in which a graph matching solution is incrementally constructed step-by-step. In line with classical search-based algorithms for GED, the method introduces a dynamic programming–style decomposition of the problem, where partial matchings and their induced subproblems are reduced while preserving equivalence to the optimal solution set. The model is trained using node-level matching supervision and evaluated on several benchmark datasets, with experiments demonstrating improved generalization to larger graph sizes compared to some previous learning-based GED predictors.

### Strengths
**Strengths**

- GED prediction often requires resolving symmetries and tie-breaking, further exacerbated by expressivity limitations of standard GNNs. This make the sequential decision framework, in line with  traditional heuristic search methods for GED, well-motivated. I also like the analysis showing that the reduced subproblem maintains equivalence of optimal solutions. 

-  Despite relying on explicit node-level matching supervision, which would usually count as a disadvantage, the paper shows robust generalization to larger graphs, which confirms practical utility of the approach.

- By generating incremental matchings, the method offers interpretability advantages compared to one-shot predictors.

### Weaknesses
**Weaknesses**

1.  The statement of Theorem 1 differs between the main paper and the appendix, and the notation in the main paper version is unclear. Since the theorem plays a central conceptual role, the paper would benefit from, aligning the statements across sections, clarifying notation, and adding a brief intuitive explanation (2–3 sentences) to the main body to highlight why equivalence is preserved.

2. The architecture section does not clearly specify how partial matchingsare represented internally.  Do cross-graph edges introduced by a partial matching participate in message passing? If so, the graph structure is dynamically modified, which may have unintended representational implications.  If not, how are embeddings of partially matched nodes coupled or tied together?  Similarly, the implementation of the reduce operation in the neural pipeline is not clearly described.

3. The choice of a 128-dimensional embedding for comparatively small graphs is not well justified, since the node embedding matrix is larger than the adjacency matrix!

### Questions
Please refer to the weaknesses for some questions.  


In addition,  I would like to better understand the practical utility and trade-offs of using an autoregressive approach as opposed to existing one-shot GED predictors. In general, one would expect autoregressive inference to incur higher computational cost, due to stepwise decoding and possible sensitivity to early decisions that may require techniques such as parallel beam search. While the ability to produce interpretable edit paths is a meaningful benefit, the paper does not clearly analyze what is gained and what is sacrificed in moving from one-shot estimation to autoregressive decoding. 

In the reported timing comparisons, the authors attribute improved runtime to the fact that GELATO is “GPU-friendly,” whereas several baselines are not. This makes it difficult to disentangle architectural advantages of the autoregressive formulation from implementation differences. It is desirable to have a more direct study of these trade-offs including inference latency, resource usage, prediction accuracy improvements, etc. 

Also, is there any reason why GraphEDX (Jain et al., 2024) has been omitted as a baseline. I would  consider it a decent exemplar of GPU-frienly one-shot predictor for comparison.  I suspect  GraphEDX does not scale well to larger graphs. However, a comparison on in-distribution graphs, highlighting accuracy vs. interpretability vs. inference costm, would be intersting.

Minor comment: The authors highlight Jain et al., 2024 in the dataset isomorphism issue, but perhaps [1] would be a better reference in this regard. 

[1] Position: Graph Matching Systems Deserve Better Benchmarks. In Forty-second International Conference on Machine Learning Position Paper Track.



Overall, I am positive on the motivation and approach of this paper, but have questions about the neural architecture design and trade-offs around the autoregressive design.

### Soundness
2

### Presentation
2

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
In this paper, the authors present Gelato, an RL agent framework for solving graph edit distance. The authors first formulate the editing problem as a matching problem, and then train an RL agent to assign matched node pairs sequentially. The experimental evaluation shows that the proposed RL framework outperforms existing non-RL learning methods.

### Strengths
* This paper is well written with a self-contained story.
* The evaluation on various datasets demonstrates the utility of the proposed RL method, and the empirical improvements over existing methods are impressive.

### Weaknesses
* This manuscript overlooks an important prior work, [Liu et al: Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching](https://openreview.net/pdf?id=QjQibO3scV_). Liu et al presented an RL framework for the QAP form of graph matching, which is also used in Gelato. Though the underlying application and datasets are different, the methodologies of both papers seem relevant. I will be more convinced of the contribution and novelty of this paper if the authors can show the technical differences, unique innovations, and insights compared to Liu et al.

### Questions
* I will be happy to reconsider this manuscript if the authors could provide more discussion with the relevant work, [Liu et al: Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching](https://openreview.net/pdf?id=QjQibO3scV_)
* What will the performance look like if the methodology in Liu et al. is directly applied to GED learning?
* If I understand it correctly, different models are trained on different datasets. What will the accuracy look like if a model is trained on, say, the AIDS dataset and tested on other datasets? Is it possible to plot a confusion matrix to report the generalization ability? This question is important because the agent is learning the QAP, and we should expect it to generalize with such a general form.
* Is it possible to train a general agent by mixing multiple training datasets?

### Soundness
3

### Presentation
4

### Contribution
2
