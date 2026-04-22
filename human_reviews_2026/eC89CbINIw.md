# Differentiable Lifting for Topological Neural Networks

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Topological neural networks (TNNs) enable leveraging higher-order structures on graphs (e.g., cycles and cliques) to boost the expressive power of message-passing neural networks. In turn, however, these structures are typically identified a priori through an unsupervised graph lifting operation. Notwithstanding, this choice is crucial and may have a drastic impact on a TNN's performance on downstream tasks. To circumvent this issue, we propose 
 ∂lift (DiffLift), a general framework for learning graph liftings to hypergraphs and cellular, simplicial, and combinatorial complexes in an end-to-end fashion. In particular, our approach leverages learned vertex-level latent representations to identify and parameterize distributions over candidate higher-order cells for inclusion. This results in a scalable model which can be readily integrated into any TNN. Our experiments show that  ∂lift outperforms existing lifting methods on multiple benchmarks for graph and node classification across different TNN architectures, with TNN+ ∂lift combinations surpassing standard GNN baselines. Notably, our approach leads to gains of up to 45% over static liftings, including both connectivity- and feature-based ones.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of learning liftings, i.e., functions that map graphs to higher-order domains for use in Topological Neural Networks (TNNs). While most existing approaches rely on handcrafted, "static" lifting procedures, this work explores the possibility of learning liftings directly from data in an end-to-end manner. The authors introduce DiffLift, a general framework that can learn to lift attributed graphs to various higher-order topological structures.

The method runs steps:
(i) Generating node embeddings using a standard GNN.
(ii) Constructing candidate higher-order cells or hyperedges.
(iii) Assigning scores to each candidate via a learnable module.
(iv) Sampling candidates using a Bernoulli distribution based on the scores above, using Straight Through Estimators to ensure differentiability.

Two main instantiations of steps (i,ii) are described:
- Graph to Hypergraph lifting: Candidate hyperedges are formed via k-NN search on node embeddings. Their cardinalities are sampled, and scores are computed using a permutation-invariant multiset function.
- Graph to Cell complex lifting: Candidates are derived from cycles in a chosen cycle basis, and scored using a DeepSets-based model.

Experiments are conducted on both graph-level and node-level prediction tasks. When compared with static lifting approaches, DiffLift generally achieves significant improvements. However, the benefits are less consistent on node-wise classification in heterophilic datasets. An additional experiment shows that the choice of GNN backbone (e.g., GPS vs. GIN) in the first stage can noticeably affect performance.

### Strengths
[S1] The paper is easy to understand and the narrative quite straightforward.

[S2] The experiments are comprehensive, covering a broad set of use-cases.

### Weaknesses
[W1] The relationship between this work and the DCM method is not clearly articulated. The authors cite it, but the concrete technical advantages over DCM are to be discussed. What limitations of DCM are addressed? How? If DiffLift claims more generality across topological domains, the authors should explain why DCM cannot be extended similarly and highlight key methodological differences.

[W2] The proposed learnable lifting mechanism resembles differentiable pooling approaches as well as methods that *learn* to wire virtual nodes in graphs. This paper, e.g., is quite relevant (https://proceedings.neurips.cc/paper_files/paper/2024/file/31df6a082046111e605abfec26ef5ccc-Paper-Conference.pdf). The authors should position their method relative to these lines of work and provide a minimum experimental comparison, in particular, w.r.t. pure hypergraph lifting, where no cycle-like inductive bias is leveraged.

[W3] The requirement to restrict candidate cells to cycles within a chosen basis seems somewhat arbitrary and potentially limiting. If only a subset of those cycles is ultimately selected, it is unclear why a cycle basis is necessary in the first place. Previous works also consider lifting based on any induced or non-induced cycles (see e.g. Bodnar et al., 2021). Is this primarily a computational convenience? A clearer rationale or explanation is needed.

[W4] On graph property prediction benchmarks, the TNN-based baselines very often do not outperform standard GNNs, which raises concerns. Is this expected? It is unclear whether the experimental setup or hyperparameter tuning might be limiting performance.

[W5] For node classification tasks, TNN models are not compared against vanilla GNN baselines. Some lifting methods (e.g., “kernel” on CiteSeer) perform suspiciously poorly without explanation.

### Questions
[Q1] What is the difference between using a DeepSet model in the case of Hypergraphs and a Set Function in the case of Cell Complexes in Figure 4?

[Q2] Line 314 – Hajij et al, characterised various lifting together, but other previous uncited works introduced them first?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a differentiable lifting to create an end-to-end learnable
pipeline for the lifting of graphs into higher order structures to enhance
their expressivity. Their pipeline consists of an initial embedding layer that
maps a combinatorial complex (specialized to graphs in the experiments) into a
point cloud. A cell sampler subsequently samples rank-k cells from this point
cloud and learns whether or not the k-cell should be included in the final
lifted cell complex. Hitherto the standard procedure for this lifting has been
deterministic, based on either cliques or cycles, for instance. The author
argue that the optimal lifting depends heavily on the dataset, motivating the
need for a *learned* lifting in order to mitigate this need for  pre-defined
choices. An experimental suite shows that this method works rather well on a
wide range of standard benchmarks for both node classification as well graph
classification.

### Strengths
Overall the paper is well-written and the idea is interesting. In particular
eliminating the choice of lifting provides a more general approach for TNNs to
work on a wider set of graphs. This is an important fact, as multiple works
have already shown that various graph datasets have very different properties
and that it is not always clear what type of networks operates best on each
particular dataset. Hence overcoming this challenge is a good contribution to
the TDL community. The code to reproduce the experiments are provided and is
well-documented, which is great to see.

### Weaknesses
While the reviewer views the experiments section good it is a little surprising
that the methods only compare to other TNN methods, whereas other comparison
partners could be interesting as well, such as DiffPool or other methods
proposed by the TDA community and graph learning community in general. As this
method seems to have a higher complexity that other methods, it is in the
opinion important to discuss what may persuade a practitioner to use this
method over say a simpler method if it performs more or less equal.

### Questions
- Is there a particular reason the method is restricted to graphs? I could
imagine that, since the idea of TNNs is to operate on higher dimensional
complexes as well, higher dimensional simplicial data also to be interesting. 
- Have you considered applying the method to point clouds as well? 
- One limitation is that to construct the cell complexes, on has to loop over
the full power set, which get quite intense. It is understandable that one
would only consider 1- and 2-cells, but even that would become rather
prohibitive. E.g. A "tiny" point cloud or mesh of an object quickly contains up
to 2-4k points and this is where your method already could become quite
expensive since even constructing the 1-cells (graph) for a single point clouds
would already have to consider $2048 choose 2 = 2,096,128$ or $4096 choose 2 =
8,386,560$ possibilities and for 2-cells one would have to consider billions of
possibilities. Some remarks could shed some light into these considerations.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DiffLift, a differentiable framework for end-to-end learnable graph lifting to higher-order topological domains, including hypergraphs and cell/simplicial complexes. Instead of relying on static, task-agnostic liftings (e.g., cycles, k-hop neighborhoods), DiffLift learns probabilistic distributions over candidate higher-order cells using node embeddings from a backbone GNN. The method integrates with multiple TNN architectures and supports hierarchical sampling for cellular complexes. Experiments across 12 benchmarks show improvements over static liftings.

### Strengths
1. The paper proposes a general formulation for differentiable lifting applicable across multiple domains (hypergraphs, cell complexes, simplicial complexes), which enables task-aware structure learning.

2. Experimental validation across multiple datasets and TNN backbones shows performance gains.

3. The proposed framework is robust enough to be integrated with existing TNN frameworks.

### Weaknesses
1. Computational bottlenecks remain for cell complexes, especially in cycle-basis computation (cubic in nodes), limiting scalability to larger graphs.

2. The gains on node classification are less uniform, particularly for homophilic datasets with hypergraph models.

3. The proposed method is Dependent on NN-based k-selection, and probability estimators may introduce additional hyperparameter complexity.

4. There are no explicit theoretical guarantees on representation expressiveness or stability under stochastic lifting.

### Questions
1. How does Difflift behave on very large graphs (e.g., OGB-LSC scale), where cycle-basis computation and candidate sampling may be expensive?

2. Could the authors elaborate on how sensitive Difflift is to stochastic sampling noise, the backbone GNN used for embedding generation, and the choice of aggregation function? Did you evaluate alternative backbone models or pooling operators, and do you observe increased variance or instability when weaker embeddings or different set functions are used? Are variance-reduction techniques (e.g., Gumbel-softmax, annealing, deterministic thresholds) helpful?

3. Does Difflift mitigate known issues in deep TNNs, such as over-smoothing or over-squashing, more effectively than static lifting?

4. Can the authors report runtime and memory overhead relative to common static lifting pipelines, particularly in high-dimensional molecular tasks?

### Soundness
3

### Presentation
3

### Contribution
3
