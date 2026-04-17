# Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Long-range dependencies are critical for effective graph representation learning, yet most existing datasets focus on small graphs tailored to inductive tasks, offering limited insight into long-range interactions. Current evaluations primarily compare models employing global attention (e.g., graph transformers) with those using local neighborhood aggregation (e.g., message-passing neural networks) without a direct measurement of long-range dependency. In this work, we introduce $\texttt{City-Networks}$, a novel large-scale transductive learning dataset derived from real-world city road networks. This dataset features graphs with over $10^5$ nodes and significantly larger diameters than those in existing benchmarks, naturally embodying long-range information. We annotate the graphs based on local node eccentricities, ensuring that the classification task inherently requires information from distant nodes. Furthermore, we propose a generic measurement based on the Jacobians of neighbors from distant hops, offering a principled quantification of long-range dependencies. Finally, we provide theoretical justifications for both our dataset design and the proposed measurement—particularly by focusing on over-smoothing and influence score dilution—which establishes a robust foundation for further exploration of long-range interactions in graph neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper suggests a new graph dataset that is clearly based long-range dependencies. The graphs are based on city road networks of four major cities with the target to compute the furthest distance one can travel passing 16 junctions. In addition, they provide a measurement that computes the influence of far-away nodes, highlighting that indeed the constructed graphs contain long-range dependencies.

### Strengths
The paper addresses an important open problem in graph learning as currently good benchmarks for long-range dependencies are missing even though we know (or rather assume) that long-range dependencies exist in many real-world tasks. The construction of the dataset is very clear and the provided metric improves upon the main existing alternative (Bamberger et al 2025) in terms of speed and possibly accuracy. The theoretical justification is non-trivial and makes a sound impression. The paper is well-written and easy to follow.

### Weaknesses
Not exactly strong weaknesses, but rather points that I would have liked:
- a more detailed comparison to the metric by Bamberger et al which I did expect to be mentioned in the introduction as well (e.g. in 107 I did expect that reference)
- a more concrete statement in 125ff that the appendix contains the exact list of features that made it into the dataset. The description in the main paper is a little to vague here for my personal taste.
- the conclusion states that LRGB's claim for long-range is solely based on the performance gap while the paper also talks about larger (even though not large) graphs and larger diameter.

### Questions
see weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a real-world, attributed, transductive benchmark consisting of four city networks, where the task is to classify each junction’s "urban accessibility", defined via a local eccentricity measure with radius 16. This, they argue, creates non-trivial long-range task dependencies. They also explore other radii values: smaller ones make the task too easy, while larger ones suffer from uninformativeness of the graph structure. They compare several network metrics with existing graph datasets and show that their city networks have larger diameters but are sparser and more grid-like; this mitigates over-smoothing and makes them better suited for benchmarking long ranges. They benchmark both GNNs and graph transformers and find that, on their dataset, performance consistently improves with depth, whereas on common benchmarks it typically plateaus or drops. Finally, they introduce a measure of long-range dependency that examines the layer-wise and cumulative influence of distant nodes on predictions. On their dataset, this influence is stronger and decays much more slowly than on the others.

### Strengths
- The paper introduces a transductive benchmark built from real road networks, and moves beyond the small citation-like graphs that dominate current GNN evaluations of long range tasks.
- On their benchmark, all models improve with depth, which is not the case on most standard datasets, and is thus a good indicator that the task needs long-range interactions. They furthermore support it with theory on over-smoothing, showing why this is possible on this kind of structure.
- Road networks are an important application area of graph learning, so a benchmark based on them is very welcome.

### Weaknesses
1. I am not fully convinced by the benefits of the proposed long-range influence metric. It does not seem strictly model-agnostic, since the influence values still depend on the trained parameters and even change across models (cf. Table 2). It mostly answers "did this model use long-range information?" rather than "is this dataset inherently long-range?"
2. I think the limitation of this metric on dense graphs is too important to be buried in the appendix; it should be brought into the main text and discussed more lengthily and explicitly.
3. While it is a good point to test graph transformers on large graphs for scalability, I am not sure this setting is the most revealing for getting insights. I.e. failure cases on smaller graphs are often more informative because there are fewer confounding factors. This benchmark awkwardly sits between being realistic and being revealing.
4. The defined task is not necessarily informing about the accessibility of a city area. For instance, in practice longer and faster highways can make areas more accessible and not less. The task is still a synthetic one defined on top of a real graph structure.
5. Fixing the ground-truth radius at 16 and mostly benchmarking models up to that depth is not entirely fair. The other tasks can make models suffer from over-smoothing from having more layers than required. Apart from a single point in Fig. 8, I would like to see more results with deeper models than the target radius.

### Questions
1. It would help to see how the influence measurement behaves on synthetic long-range tasks (1). In a setting without real-world noise and with fully controlled dependencies, does the influence become more stable across models, or do we still observe the same variability?
2. The paper states that "graphs with long-range dependencies are expected to have higher proportional influence between more distant nodes", but this assumes a monotone growth of influence with distance. In practice a task could depend on radius 0 and on exactly radius R, and your global metric would average it out. Can this happen for your task, and could models overfit to intermediate radii in a way that even distorts the per-hop measurement?
3. How does the theoretical argument about sparsity and slower over-smoothing relate to phenomena like the Braess paradox, where removing edges can increase leading eigenvalues, and to empirical findings that sparsity can reduce measured over-smoothing even when the leading eigenvalue grows (2)?

(1) GLoRa: A Benchmark to Evaluate the Ability to Learn Long-Range Dependencies in Graphs. Dongzhuoran Zhou et al., ICLR 2025.

(2) Spectral Graph Pruning Against Over-Squashing and Over-Smoothing. Adarsh Jamadandi et al., NeurIPS 2024.

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
2

### Summary
This paper introduces City-Networks, a new benchmark dataset designed to evaluate long-range dependency handling in graph learning models. It consists of large real-world road networks (100k–569k nodes) from four global cities and labels nodes by computing local eccentricity over large hop distances (k=16 layers). The task is transductive node classification; models must incorporate information from far-away neighborhoods to succeed.

The authors also propose a model-agnostic Jacobian-based influence measure that quantifies how much distant nodes contribute to predictions. They show deeper GNNs and graph transformers consistently improve on these datasets and provide theoretical justification linking dataset topology to reduced over-smoothing and emphasizing influence dilution in grid-like graphs.

### Strengths
1. **Novel Benchmark Contribution**: Introduces a real-world long-range benchmark on large graphs.

2. **Task Design**:
- Long-range target signal (local eccentricity) is explicitly tied to graph distance, not just node features.
- Sensible justification for choosing k=16 to require long-range aggregation.

3. **Good Empirical Study**: Systematic layer-depth experiments show deeper message passing helps.

4. **Theoretical Support**: Spectral argument connecting large diameter & low degree to slower over-smoothing.

### Weaknesses
1. **More Clarity Needed on Task Setup**:
- Distribution of quantile labels — class imbalance?
- Exact splits and sampling details

2. **Transductive-only Setting**: Dataset is transductive; how will the ideas generalize to inductive settings?

3. **Label Leakage and Spatial Bias Concerns**: Although the authors argue against pure geographical dependence, node features include latitude & longitude and spatially-derived attributes. It is not fully demonstrated that models can't rely largely on spatial features alone. Stronger evidence is needed to show that spatial coordinates alone cannot solve most of the signal.

### Questions
1. Can you please defend and answer the questions or concerns raise in **Weaknesses**?
2. How sensitive is performance to the inclusion of geographic features (lat/long)? Can you report results where positional coordinates are removed?
3. Can models exploit spatial coordinates alone (that is consider separately MLP with coordinates only and GNNs with coordinates masked)?
4. What is the runtime overhead of Jacobian measurement on a single city graph?
5. Can you please provide a Table with accuracy results you presented in Figure 3 for k=16? It will be more easier to compare the results.
6. How would your model perform on heterephilic datasets (Texas, Wisconsin, Cornell, Roman-Empire, etc)

### Soundness
4

### Presentation
4

### Contribution
3
