# Torch Geometric Pool: the Pytorch library for pooling in Graph Neural Networks

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
We introduce Torch Geometric Pool (tgp), a library for hierarchical pooling in Graph Neural Networks. Built upon PyTorch Geometric, tgp provides a wide variety of pooling operators, unified under a consistent API and a modular design based on the Select-Reduce-Connect-Lift framework. The library emphasizes usability and extensibility, and includes features like precomputed pooling, which significantly accelerate training for deterministic operators. In this paper, we present tgp's architecture and systematically compare the performance of the implemented poolers in different downstream tasks. The results, showing that the choice of the optimal pooling operator depends on tasks and data at hand, support the need for a library that enables fast prototyping.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents Torch Geometric Pool (tgp), a modular library for hierarchical pooling in Graph Neural Networks, built on top of PyTorch Geometric. The library unifies diverse pooling operators under a consistent Select-Reduce-Connect-Lift framework, emphasizing usability, extensibility, and computational efficiency through features like precomputed pooling. Comprehensive experiments across various downstream tasks demonstrate that the optimal pooling method is task-dependent, highlighting the importance of a flexible library for rapid prototyping and evaluation.

### Strengths
- The writing is good and easy to read.
- A good contribution to the GNN community.

### Weaknesses
- The contribution is primarily engineering-oriented and lacks sufficient methodological innovation.
- The current implementation only supports homophilic graphs, which limits its applicability to broader graph types.
- The experimental design is not fully consistent with the stated goal of evaluating hierarchical pooling approaches.

### Questions
- The package is titled “Torch Geometric Pool: the PyTorch library for pooling in Graph Neural Networks”, yet it focuses solely on hierarchical pooling and omits global pooling methods. The title may therefore overstate the scope of the library.
- The in-memory caching mechanism is designed for single-graph tasks but may restrict scalability for large graphs. How does the framework handle such scalability concerns?
- The experimental evaluation is not comprehensive, several included pooling operators are not compared across all relevant tasks.
- In the clustering experiments, dense pooling methods produce fully connected weighted graphs. How is the threshold determined for forming clusters?
- Metrics such as ACC, ARI and F1-score should be included for clustering evaluation.
- This paper targets hierarchical pooling, but the downstream GNN architectures for clustering, node classification, and graph classification use only a single pooling layer, which does not sufficiently demonstrate the hierarchical nature of these compared pooling methods.
- Given the engineering focus and practical utility, this work may be more suitable for a workshop submission, similar to prior PyG 1.0 and 2.0 releases. Alternatively, the authors could consider contributing it as a pull request to the PyG project, which would enhance accessibility and impact within the community.
- Other graph hierarchical pooling works: 
	* HGP-SL. Zhang Z, Bu J, Ester M, et al. Hierarchical multi-view graph pooling with structure learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2021, 35(1): 545-559.
	* GSAPool. Zhang L, Wang X, Li H, et al. Structure-feature based graph self-adaptive pooling[C]//Proceedings of the web conference 2020. 2020: 3098-3104.
	* SEP. Wu J, Chen X, Xu K, et al. Structural entropy guided graph hierarchical pooling[C]//International conference on machine learning. PMLR, 2022: 24017-24030.
	* GPN. Song Y, Huang S, Wang X, et al. Graph Parsing Networks[C]//The Twelfth International Conference on Learning Representations.
	* CGIPool. Pang Y, Zhao Y, Li D. Graph pooling via coarsened graph infomax[C]//Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021: 2177-2181.

### Soundness
3

### Presentation
3

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
The paper presents a PyTorch Geometric–based library that consolidates a broad set of graph pooling operators behind a consistent interface, organized by a select–reduce–connect–lift abstraction. It standardizes batching, readout, and auxiliary-loss handling; adds engineering optimizations such as caching and dataset-level pre-coarsening to speed up deterministic operators; and offers a systematic, within-library benchmark across tasks to illuminate trade-offs among sparse vs. dense and trainable vs. non-trainable pooling methods.

### Strengths
1. **Well-scoped unification that improves day-to-day usability.** Despite focusing on a single GNN subcomponent, the library makes swapping and configuring pooling operators straightforward via a single, coherent API; the standardized handling of batching, readout, and auxiliary losses reduces integration overhead and enables more reliable ablations without chasing disparate third-party repositories.
    
2. **A clear, modular taxonomy that clarifies the design space.** By structuring pooling into select–reduce–connect–lift stages, the work offers both a conceptual lens and code-level modules that help practitioners reason about alternatives and assemble hybrids with minimal boilerplate, which in turn encourages systematic exploration rather than ad-hoc, one-off implementations.
    
3. **Practical efficiency and reproducibility gains within a unified stack.** The inclusion of caching and pre-coarsening accelerations for deterministic operators, together with standardized datasets and scripts, supports faster iteration and more consistent, apples-to-apples comparisons inside the same framework, aiding reproducibility and lowering the barrier to controlled empirical studies.

### Weaknesses
1. **The scope and community impact appear limited for a venue like ICLR.** While the library offers a tidy unification of graph pooling under a common API, it targets a single GNN subcomponent rather than a broadly enabling platform; in contrast, end-to-end frameworks (e.g., general-purpose GNN libraries) typically shift community practice more substantially. Consequently, the manuscript’s contribution feels incremental at the level of research impact, even if it may be practically helpful for users who frequently switch among pooling operators.
    
2. **The work reads primarily as curation and repackaging rather than methodological or systems innovation.** Most pooling operators already have public implementations scattered across original repositories or mainstream GNN stacks, and practitioners can usually incorporate them with moderate effort; therefore, the added value here seems to lie in consolidation, consistent interfaces, and engineering polish rather than in new algorithms, theory, or systems primitives. Although such consolidation is useful, the paper should better articulate what is fundamentally new beyond harmonization—e.g., what capabilities would be difficult or cumbersome to realize without this library.
    
3. **Head-to-head comparisons against existing community implementations are missing where they matter most.** The manuscript would benefit from systematic evaluations of API ergonomics (e.g., installation friction, lines of code to swap methods, configuration clarity) and runtime characteristics (e.g., throughput and memory) against widely used baselines such as native modules in established libraries and the original authors’ code; importantly, while the paper appears to benchmark multiple pooling operators within its own framework, this is not the same as demonstrating external parity or superiority, and a fair, apples-to-apples comparison outside the proposed stack is necessary to substantiate the claimed practical value.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Torch Geometric Pool (tgp), a library built on PyTorch Geometric (PyG) that unifies various graph pooling methods under a consistent API. By using the Select-Reduce-Connect-Lift (SRC(L)) framework, tgp provides a modular design for both trainable and non-trainable, sparse and dense poolers. The work implements 17 pooling methods spanning trainable/non-trainable and sparse/dense paradigms, provides caching and precoarsening mechanisms for efficiency, and includes systematic benchmarks across unsupervised clustering, node classification, and graph-level tasks on 20+ datasets. 

Overall, tgp aims to simplify experimentation, enhance the usability of graph pooling, and facilitate rapid prototyping in GNN research.

### Strengths
(1)Addresses a real need. The fragmentation of pooling implementations across frameworks is a genuine pain point. The paper's motivation—enabling fair comparison and rapid prototyping—is compelling and well-articulated.

(2)Comprehensive graph pooling benchmark: The paper presents a thorough benchmarking of multiple pooling operators across several tasks, highlighting the trade-offs and aiding researchers in selecting the best pooling method for their specific application.

(3)Novel efficiency mechanisms. The caching and precoarsening features are practical innovations that meaningfully advance the usability of deterministic poolers.

### Weaknesses
(1) Limited contribution. The SRC(L) framework itself is not a novel contribution; it is attributed to Grattarola et al. (2022). Consequently, the primary contribution of this paper lies in the engineering effort of building a comprehensive, well-designed software library around the existing framework, rather than in proposing new architectural principles or algorithmic insights for graph pooling itself.
[Reference] Grattarola, Daniele, et al. "Understanding pooling in graph neural networks." IEEE transactions on neural networks and learning systems 35.2 (2022): 2708-2718.

(2) Missing comparisons to recent unified frameworks: While the paper addresses the fragmentation of graph pooling implementations, it does not provide direct comparisons to other recent unified frameworks.

(3) Lack of concrete evidence for extensibility. The paper emphasizes SRC(L) modularity as a design strength, claiming users can "combine new and existing modules" to create novel poolers. However, no experimental ablations validate this claim. It would strengthen the paper to include practical examples or experiments that demonstrate how easy it is for users to create new poolers by combining different modules from the library.

### Questions
(1) The paper addresses the fragmentation of pooling implementations, but it does not directly compare tgp to other recent unified frameworks like Spektral or Deep Graph Library (DGL). Could you provide a comparison of tgp with these other libraries, especially in terms of their pooling capabilities, ease of use, and performance?

(2)Beyond engineering, does the library introduce conceptual innovations? The SRC framework is from Grattarola et al. (2022), and all 17 operators are existing methods. Does tgp propose new pooling paradigms, theoretical analysis, or novel inductive biases?

(3)In the unsupervised node clustering experiments, the paper uses default hyperparameters for each pooling method. However, as you mention, hyperparameter tuning can significantly impact the performance of different poolers. Could you provide more details on how you plan to address the tuning for each method? Specifically, how does the performance change with tuned hyperparameters, and could a controlled tuning experiment be conducted to demonstrate fair comparison across the poolers?

### Soundness
3

### Presentation
3

### Contribution
3
