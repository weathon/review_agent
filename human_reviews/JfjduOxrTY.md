# Understanding Graph Transformers by Generalized Propagation

- Decision: Reject
- Scores: 3, 6, 5, 3

## Abstract
Graph Transformers (GTs) have recently shown stellar performance on various
graph learning benchmarks, which is typically attributed to their underlying global
self-attention mechanism. In this paper, we use generalized propagation graphs,
constructed through two abstract configurable functions and offering a unified
view across various GNN models used in the literature. We show that by con-
figuring the two abstract functions governing the generation of propagation graph,
one could recover the most popular GNN models including graph Transformers,
message-passing neural networks (MPNNs), as well as various forms of graph
rewiring. We show that the expressivity of the instances of our framework depends
on one of the governing functions (the adjacency function). Empirical results con-
firm our theory: by keeping the adjacency function while removing self-attention,
the state-of-the-art GT maintains its performance. In other words, by designing
appropriate adjacency functions, one could construct novel GNN models with di-
verse expressive power. We also study the geometric properties of the propagation
graphs across a wide range of models, using a novel extension the Ollivier-Ricci
curvature to weighted digraphs.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes GPNNs, a framework unifying graph transformers, graph rewiring, and MPNNs. It shows the expressivity of GPMM equals to a color-refinement algorithm with adjacency function. Furthermore, it proposes a continuous extension of the Ollivier-Ricc curvature for analyzing the information propagation.

### Strengths
1. A unfied framework for MPNN, graph rewiring, and graph transformer.

### Weaknesses
1. The equivalence between Graph Transformer and the color refinement algorithm in proposition 3.1 are proved in previous work [1].

2. Section 3.3 looks incomplete. At end it mentions various GNN expressivity hierarchy without clarifying GPNN's connection with them.

3. Figures are not mentioned in the maintext.

4. The connection between CURC and GPNN seems fully intuitive. More formal connection can make me understand better.

5. The definition and properties of CURC seem quite straighforward. However, therefore meaning for GNN are not quite clear to me. How to use these theories to guide the design of GNN?

[1] Wenhao Zhu, Tianyu Wen, Guojie Song, Liang Wang, Bo Zheng, On Structural Expressive Power of Graph Transformers. KDD 2023. (released on Arxiv in May)

### Questions
1. The formal connection between CURC and GPNN.

2. More clear and detailed implication of CURC properties.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a unified view by defining the generalized propagation graph which is a weighted directed graph constructed from the input graph. By configuring the adjacency function f and entry-wise function \pi, various GNNs (MPNNs, graph rewiring and GTs) can be unified into a general framework, generalized propagation networks (GPNNs). And the authors show that the expressiveness of models within GPNN framework sorely depends on the adjacency function f. Therefore, novel GNN models with diverse expressive power can be constructed by designing appropriate adjacency functions. Extensive experiments are conducted on several public datasets to verify the effectiveness of the proposed model.

### Strengths
1.	The idea of this paper is interesting and clear. .
2.	The proposed method seems sensible, and perform well on several benchmarks.
3.	The paper provides a lot of theoretical analysis to support their claims.

### Weaknesses
1.	Generally, the paper is not very friendly to readers and need to be polished, especially section 3 and section 4. 
2.	Deeper analysis about the experimental results in Table 2 and Table 3 are missed.
3.	More experiments need to be conducted on larger graph benchmarks (cora, Citeseer, Pubmed and OGB datasets) based on different GTs (except GRIT) to validate the effectiveness and scalability of the proposed method on large graphs. 
4.	The paper is not very easy to follow and some notations are confusing. And some formulas are not numbered. Specifically, the format of Table 3 seems need to be reorganized.

### Questions
Please refer to the weaknesses part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors introduce a general framework for representing Graph Neural Networks (GNNs), Message Passing Neural Network (MPNNs), Graph Transformers (GTs) and graph rewirings. They call it Generalized Propagation Neural Network (GPNN) and it can be specialized/instantiated by defining a function of the adjacency matrix of the graph to learn and an entry-wise function for pairs of graph nodes. They show that expressiveness of the model depends only on the function of its adjacency and they empirically test this by removing self-attention (i.e. the entry-wise function) from one of the GT models  and still retaining its performance (prior to its modification). The authors also extend Ollivier-Ricci (OR) curvature to weighted directed graphs, thus defining Continuous Unified Ricci Curvature (CURC). They study the theoretical properties of CURC and leverage this in analyzing the shifts in curvature distribution for propagation graphs after training, for a number of graph learning models that can be cast as GPNNs.

### Strengths
- This is a rich/extensive and ambitious work (both in theory and experiments - particularly in theoretical developments), around two key notions: GPNN and CURC. These notions are novel and serve as interesting additions to the expanding neural graph learning literature.

- Structure and high-level flow is smooth; content is reasonably split across the main manuscript text and its appendix. This is particularly useful in cases like this when a broad set of definitions and theorems must be combined.

### Weaknesses
- The presentation of GPNN can be considerably simplified/clarified. In Section 3.2 in particular: (a) using fewer symbols (are both $P$ and $\pi$ absolutely necessary?), (b) providing standard names to symbols so that referring to them is straightforward (e.g. $\rho()$ is referred to as both "normalized function" and "normalized propagation" which may be confusing, $\phi()$ does not have a descriptive name), (c) consider using a simple model as GNN as an example of the choices for the functions "embedded" within the text (rather than only as part of Table 1).

- CURC could be illustrated (and  ontrasted to OR) using a very small, simple example graph. The reader would then get an intuitive understanding of why CURC is strictly necessary in this context and also be able to better understand its algebraic and geometric properties (Section 4.3)

- It is not clear how the connection of the expressiveness of the model to (only) the function of its adjacency matrix could drive the choice of particular forms for such appropriate functions (which would certainly be a highly practical implication of this work).

### Questions
- Challenging the benefits of global self-attention in graph transformers (GTs) is a very strong statement. How could this reconcile with the reported elevated efficacy of GTs (i.e. with global self-attention) in various graph learning tasks in the literature (relative to GNNs, which do not have global self-attention)? The reader would be interested to know of any other potential empirical factors that could account for this conclusion.

Minor typos
- Page 3: entries-wise -> entry wise
- Page 4: identical to 2 -> identical to Equation (2)
- Page 4: every p layer -> every p layers
- Page 9: optical CURC distribution -> ?(optical)

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a general definition of generalized propagation graphs, which covers various graph neural networks based on different constructions of the adjacency function (see 3.2). The paper explored the expressiveness of GPNN, and an upper bound was proved. Plus, the paper defined CURC (continuous unified Ricci curvature) and utilized it to explore expressive power. At the end, the paper designed experiments to empirically prove their theory. By dropping the irrelevant part,  it shows graph network retaining the performance as original model, which in turn serves as an example that the graph network is primarily dependent on the adjacent function

### Strengths
1. Understanding a neutral network with strict mathematical theory is challenging. The paper proposed a general framework and some intuitive definitions to provide a general theoretical exploration of the graph neural network.

2. The paper designed experiments to validate its theoretical observation.

3. The paper provides proof for propositions and theorems it claims.

### Weaknesses
1. Since it is a theoretical-style paper, the paper needs to improve its notation and clarification, and it's better to treat these parts like math.
For the definition of adjacency function f, In section 3.2, 'it is a mapping from A to... ', so it is a mapping from R^{n^2} to R^{n^2 \times d}. But when referring to Appendix A1, it mentions some function that maps over some tensor power space from R^{n^p \times d} to R^{n^k \times d}. It seems that the adjacency function is a function on input with features, i.e., R^{n^2 \times d}. I am confused about which one should be.

2. The paper would be improved if it talked about why we chose Ricci curvature (generalized or not) as a measurement to explore the graph. In geometry, Ricci curvature is a degrade of Riemann curvature, while it is enough to characterize two-dimensional manifolds.

3. (Section 4.2, definition 4.2, d^{\epsilon} and (8)) The definition of CURC is questionable, which lies in the so-call asymmetric metric function. As a curvature to measure the curvedness of a manifold (in our case, a graph), its definition should be based on distance (or, say, metric) satisfying positivity, symmetry and triangle inequality. But CURC is based on an asymmetric metric function (which is not a metric), which means the metric could be different from u to v or v to u, which in turn means that the curvature from u to v could be more curved or less curved than the reverse. But CURC is defined as a scale, which contradicts the above. 

4. The paper would be improved if it elaborated more on the implications of CURC, KR duality in section 4.3. It is hard to follow the terminology and abstract indications.

5. The definition of a generalized propagation network is more of a rewrite of a graph network with an adjacency function that incorporates propagation and feature mapping (there is still some ambiguity, as I mentioned in the first point). It is more on the conveniences to bring out the notion of curvature.

### Questions
1. (7) and lemma B7 in appendix: what does it mean m(x) divided by m(y)? as the definition of m just above, they are vectors.
2. CURC measuring GPNN (the paragraph above section 4.3). 'Furthermore, the incorporation of the Perron measure as .....'. How they relates to bottlenecking within weighted-directed graphs? any section it refers to?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
