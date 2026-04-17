# Building the dual graph of the activation regions in a deep neural network: what it means for interpretability

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Understanding the geometric representations of deep neural networks (DNNs) which employ a piecewise linear activation function has become a popular research direction for model explainability. 
A complete geometric picture of the representations of a DNN would include both the polytope regions formed by the network partitions and the set of neighboring regions, i.e., a dual graph.
Prior work has resulted in algorithms which enumerate all of the activation regions formed by a network, but no algorithms have been proposed for constructing the dual graph in its entirety. 
This gap may stem from the naive assumption that because identifying neighboring regions is trivial in shallow networks, it is also trivial in deep networks.
In this work, we demonstrate that this assumption is false; finding neighboring regions in a deep network is in fact a difficult problem due to the conditional nature of the partitions in the deep layers. 
We introduce a method to solve the difficult problem of neighbor finding in DNNs.
We implement this algorithm along with region enumeration, which together constructs the dual graph.
Further, we demonstrate the usefulness of the graph in the context of generalization. 
We show that test data that are near training data, as measured by path length along the graph, tend to yield the best generalization results.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the geometric representations of deep neural networks by introducing a method to compute the dual graph of the input space partition induced by a continuous piecewise affine deep network on its input space. The authors then implement this method to compute the dual graphs of deep neural networks trained on the Extended MNIST dataset. They find that measuring the paths being training samples along the dual graph provides insights into the generalisation of the deep neural network.

### Strengths
1. Well Motivated: The challenge of computing the dual graph of an input space partition is well motivated, and the proposed method is outlined clearly.
2. Novel Approach that Builds Upon Existing Work: The authors effectively leverage prior results/algorithms where possible, but then derive a novel neighbour finding algorithm that extends existing work.

### Weaknesses
1. Correctness of Algorithm: The proposed method is described; however, no formal statement or validation is provided on a toy example.
2. Small Experiments and Practical Scalability: As deep networks become large, the number of activation regions grows exponentially; therefore, computing the dual graph would seem intractable. Furthermore, supposing obtaining a dual graph is feasible, algorithms operating on these graphs are likely to be computationally expensive. Perhaps evidenced by the fact that only small-scale experiments are considered. 
3. Overstating the Importance and Applicability to Interpretability: Using the term 'interpretability' is somewhat misleading. Interpretability is typically used in reference to understanding the learned features of a deep network. Although this application is mentioned in the conclusion, it is not the main focus or application considered in the paper.

### Questions
1. What is the computational complexity of the proposed method for finding the dual graph?
2. How would you consider adapting the algorithm for deep networks that are not continuous piecewise affine?
3. How does the utilisation of this exact algorithm compare to just sampling points along a linear interpolation of input points and using the number of unique input-output Jacobians as a proxy for the shortest path?
4. Is there a property of a deep neural network that you foresee we can only explain using the dual graph, rather than just using the input space partition?

### Soundness
2

### Presentation
3

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
This paper introduces the first exact algorithm for constructing the dual graph of activation regions in deep neural networks (DNNs) with piecewise linear activations (e.g., ReLU). Unlike prior work that focuses solely on enumerating activation regions, this work identifies neighboring regions—a much harder problem due to conditional layer dependencies. The authors leverage results from computational geometry (e.g., Sleumer 2000, Rada & Černý 2018) to design an output-polynomial algorithm that enumerates regions and discovers neighbors via "tight hyperplane" detection. They then demonstrate the interpretive utility of the dual graph by correlating graph-based path distances between training and test regions with generalization performance on EMNIST.

### Strengths
* The proposed work has its novelty in that the authors are the first to explicitly formalize and solve the neighbor-finding problem for DNN activation regions. Prior work cited in Related Work (p. 2 L107–L141) enumerates cells but not adjacency—this paper extends the geometric picture to the full dual graph.

* The construction of the network hyperplane arrangement and the use of linear programs for interior-point search are mathematicallly consistent with computational geometry practice.

### Weaknesses
* The paper claims the algorithm is output-polynomial (pg. 6) but gives no empirical runtime or complexity scaling. 
* Experiments seem to be confined to 2-layer MLPs with width = 11. There is no timing or memory analysis across larger architectures.
* Adding the comparative evaluation based on some of the related works (e.g. Balestriero & LeCun (2024)) would strengthen the work. Without such comparisons, it is unclear if the new algorithm improves over existing enumeration methods
* The observed correlation between path length and accuracy (Figure 6) is very interesting but not anlayzed in depth. The discussion section uses strong claims ("operationally meaningful definition of generalization") without rigorous justification.

The dual-graph formalism represents a meaningful conceptual and algorithmic advance over prior geometric analyses of neural networks.
However, the work remains computationally limited and empirically under-demonstrated (Sections 5–6, p. 6–8).
If scalability evidence and stronger experimental validation are added, I would be happy to raise the score. 

** Some minor points ** 
* Sometimes the input is denoted as $\theta$ but sometimes it's $\mathbf{x}$
* Minor typographical inconsistencies (e.g., “tesselation” (pg. 8) vs. “tessellation” (pg. 6)) and overuse of the term “network sign vector” without shorthand notation make some passages verbose.
* The “bounded domain” discussion (Appendix D) could be integrated earlier for completeness.
* References to recent theoretical works on piecewise-linear region geometry (e.g., Raghu et al. 2017, Poole et al. 2016) are missing.

### Questions
* Please provide empirical scaling results (runtime vs. number of neurons/layers) and compare them with prior region enumeration methods.
* Approximation or Sampling: Can the proposed algorithm be adapted into an approximate dual-graph construction for larger models? If so, outline how accuracy vs. computational cost would trade off.
* Beyond the EMNIST correlation, can the dual graph capture interpretable clusters (e.g., digits with similar morphology) or identify adversarial transitions across edges?
* Does the “output-polynomial” complexity claim hold under degenerate network configurations where many neurons are inactive or redundant?
* How sensitive is the neighbor-finding step to numerical precision (e.g., floating-point stability in LP solvers)?

### Soundness
2

### Presentation
3

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
The paper studies the full geometric representation of piecewise-linear deep neural networks by building dual graphs. The authors show that finding neighbors in deep networks is nontrivial because of the conditional, layer-wise partitioning that makes adjacency harder to detect than in single-layer hyperplane arrangements. They present what they claim is the first exact algorithm to construct the dual graph for such DNNs, and combine it with a region-enumeration procedure to recover the complete graph. Using this graph, they define a path-length distance on the manifold of activation regions and empirically show that test points that are close to training points under this metric tend to generalize better.

### Strengths
1. The paper identifies the dual graph of activation regions, moving beyond mere enumeration of regions to their adjacency structure.

2. It provides an exact algorithm for neighbor finding in deep networks, which is harder than commonly assumed.

3. The dual-graph viewpoint is intuitive and positions path length as a natural, geometry-aware metric that can capture relationships missed by Euclidean distances in input space.

### Weaknesses
1. A few sentences are slightly awkward and could be tightened for clarity: “This may be due to a naive assumption that because identifying neighboring regions is trivial in shallow networks, it too is trivial in deep networks.” reads as clunky.

2. Experiments are restricted to relatively simple, shallow architectures and MNIST. It is unclear how the enumeration + neighbor-finding pipeline scales to practical modern networks (e.g., deeper convnets, transformers, large input dimensions) or to richer datasets; the paper should either provide evidence of scalability or clearly delimit the claimed scope.

### Questions
1. I am not a specialist in every low-level technical aspect of region enumeration algorithms, so I cannot fully vouch for implementation subtleties or edge cases. In particular, I would like the authors to explain the broader significance of studying the dual graph for DNN theory and practice, given their focus on simple networks. Why should practitioners or theorists prioritize this object, and what concrete insights does it unlock beyond what regions already provide?

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
2

### Summary
This paper focuses on the interpretability approach of activation polytopes induced by the ReLU activations within a network, and describes a method for not only finding such regions, but for constructing a graph to describe the full set of regions and their connectivity to one another.

### Strengths
- Lots of technical detail

### Weaknesses
- The paper didn't do a good job of clearly articulating why and how activation polytopes could be used for interpretability 
- The paper focused too much on minutia that is confusing to a non-expert, and not enough on the ways that such a technique could be used

### Questions
- How well does this method scale to larger networks?

### Soundness
3

### Presentation
2

### Contribution
3
