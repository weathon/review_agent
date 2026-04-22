# GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
A fundamental challenge in graph learning is understanding how models generalize to new, unseen graphs. While synthetic benchmarks offer controlled settings for analysis, existing approaches are confined to single-graph, transductive settings where models train and test on the same graph structure. Addressing this gap, we introduce GraphUniverse, a framework for generating entire families of graphs to enable the first systematic evaluation of inductive generalization at scale. Our core innovation is the generation of graphs with persistent semantic communities, ensuring conceptual consistency while allowing fine-grained control over structural properties like homophily and degree distributions. This enables crucial but underexplored robustness tests, such as performance under controlled distribution shifts. Benchmarking a wide range of architectures—from GNNs to graph transformers and topological architectures—reveals that strong transductive performance is a poor predictor of inductive generalization. Furthermore, we find that robustness to distribution shift is highly sensitive not only to model architecture choice but also to the initial graph regime (e.g., high vs. low homophily). Beyond benchmarking, GraphUniverse’s flexibility and scalability can facilitate the development of robust and truly generalizable architectures. An interactive demo is available at https://graphuniverse.streamlit.app.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
GraphUniverse is a hierarchical generator for families of graphs designed to study inductive generalization across unseen graphs. At the “universe” level it fixes persistent communities with heterogeneous edge propensities, degree profiles, and feature centroids; “family” ranges then constrain homophily, average degree, size, and degree separation; graph instances sample from these ranges and realize edges via a Bernoulli reformulation of DC‑SBM with community‑coupled degree factors and feature Gaussians. Validation correlates generation parameters with graph properties, signal strength, and cross‑graph consistency. Benchmarking compares inductive vs. transductive performance and robustness under controlled distribution shifts and size scaling for several architectures.

### Strengths
1. A very timely and important direction for the whole graph community
2. the demo website is well designed and validates the reproducibility
3. comprehensive experiments

### Weaknesses
1. the biggest problem of this framework is that I don't see its clear advantage over other generative frameworks like GraphWorld, CGT (Graph Generative Model for Benchmarking Graph Neural Networks, icml 2023). And it doesn't mention a very relevant work, A Metadata-Driven Approach to Understand Graph Neural Networks (nips 2023). Although authors conduct comprehensive experiments on using this framework to evaluate different kinds of GML architectures, two experiments are missing: (1) the horizontal comparison between this framework against other generative pipelines, this makes the value of this work unclear; (2) This paper presents train on syn, test on real-world experiments, but there's no train on syn, test on real-world ones. This limits the usage of these synthetic datasets. I also doubt whether these purely synthetic graphs can really be used to pressure test GNN models. I personally think the way CGT is adopting (start from real-world ones and do generation) is more promising. 
2. In the very beginning, authors talk about the problem of graph foundation model. However, i don't see this work able to address these relevant problems. 
3. Statements about enabling “next‑generation graph foundation models” are aspirational without pre‑training results (Abstract; Sec. 1, 6). No direct evidence found in the manuscript.
4. Semantic “consistency” is validated via statistical proxies, not via downstream cross‑family transfer beyond the reported tasks (Fig. 2; Sec. 4.2–5).
5. Why are these factors considered? There's no validation. You can see (Metadata-Driven Approach to Understand Graph Neural Networks (nips 2023)) conducts a statistical test, which is much more promising and scientific. 
6. Table 1 reports only two sizes (100 vs. 1000 nodes) and single‑thread CPU throughput; linearity claims rest on a very small sample (Sec. 4.3; Table 1)—limited generality.
7. Minor notation/typo inconsistencies (e.g., “DCCC‑SBM true true” in Table 4)
8. Only one transformer variant (GPS) is included (Sec. 5.1), narrowing architectural coverage. GraphGPS also heavily relies on GNN.

### Questions
Please see weakness

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
4

### Summary
This paper presents a framework to generate graphs to evaluate graph learning models on transductive and inductive tasks. Furthermore, the authors use their tool to benchmark a variety of GNNs and achieve results about the impact of data on model generalization (e.g. the impact of distribution shifts).

### Strengths
- **(S1) Originality + Significance:** Current benchmarks of GNNs are severely lacking (as pointed out by _[Position: Graph learning will lose relevance due to poor benchmarks](https://arxiv.org/abs/2502.14546)_). GraphUniverse comes timely to help with this issue. It seems to be easy to use (please contrast with W3) and can be used to create a diverse array of graphs. This could lead the community to a more principled evaluation of GNN architectures and thus to a better understanding of their capabilities.
- **(S2) Clarity:** The paper is well written and easy to understand. (Please contrast with W2)
- **(S3) Quality:** The authors use GraphUniverse to benchmark a variety of different GNNs. I am impressed by the variety of benchmarked GNNs and find the results interesting:
	- GIN struggles in the transductive setting (but not the inductive) whereas the graph transformer performs similarly in both settings. This is interesting to me, as I would have expected GIN to also perform well in the inductive case. Furthermore, while the explanation _... indicating its success may stem from memorizing a single graph’s structure_ makes intuitive sense, it goes against my experience of GIN performing well on inductive molecular tasks.
	- Furthermore, the benchmarks show that models are less robust to distribution shifts than thought. This opens the research direction of developing models that perform better under distribution shifts.

### Weaknesses
I like the contributions of this work and my weaknesses all center around the presentation.

- **(W1) Citations:** This paper often cites the arXiv version of papers instead of the published version. While not only formally wrong, this makes it difficult to evaluate whether this paper builds upon established research or on irrelevant work.
- **(W2) Presentation of experimental results:** Figure 3 is the cornerstone of the experimental section. It contains almost all results and most research questions reference it. Despite this, Figure 3 is extremely unclear making it difficult to verify the claims of the authors. Below are some pointers for improvement that could serve as inspiration to make the figure more legible:
	- The first line reads "Transductive" and the second "Inductive". However, in the legend you have a separate style for inductive and transductive results and according to this the first result in the transductive line is inductive.
	- The split into (A) and (B) is not that clear. It would help if you could explicitly mention in the figure (not in the caption!) which research question (A) and (B) answer.
	- Explicit color coding and re-use: When your main text refers to the result of a model in Figure 3 re-use the color of the model in the main text. This makes it easier to find that information in the figure. Furthermore, you could explicitly color code types of models (which I think you attempted but it is not clear enough).
	- Why do some results have N/A?
	- Alternatively, you could simplify this figure by showing the results of fewer models (and putting the more extensive figure in the appendix). As far as I can see, you are not discussing the results of GCN, GAT and TopoTune in the main text anyway.
- **(W3) No code:** While the authors pledge to make the code public there is no code for the reviewers to check. It is unclear how easy to use GraphUniverse is. Furthermore, I wanted to check the code for the following. You write:
>Graph-level tasks (triangle counting) initially show only GPS, NSD, and GIN effectively solving the task, but while GPS and NSD maintain performance when scaling to larger graphs, GIN fails to generalize, suggesting traditional MPNNs overfit to training graph sizes.


- **(W4) Figures of graphs:** Fundamentally, this paper is about generating graph data. Thus, the  paper should include some figures of generated graphs. While the interactive demo is great, it is not guaranteed to be online or work in the future.

Is it possible that these models differ in how they aggregate local embeddings to the global level (e.g. sum vs mean pooling)?

### Questions
- See weaknesses


**Overall,** I think this is a good paper with clear flaws in its presentation. Thus, I vote for (6 - marginally above acceptance threshold).

### Miscellaneous
- Section 4.1.1 should in my opinion be a subsection instead of the only sub-subsection in the main paper.
- No information about the pooling type for some architectures.

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
4

### Summary
This paper addresses the challenge of evaluating the true generalization capability of graph learning models by proposing the GraphUniverse framework. This framework systematically assesses model performance on unseen graph data by generating controllable graph families with semantically consistent communities. Experiments demonstrate that strong performance on known graphs does not guarantee generalization ability, and a model's robustness to distribution shifts heavily depends on the structural characteristics of the training graphs (such as the level of homophily in node connections), thus challenging the conventional practice of judging model quality solely based on transductive performance

### Strengths
1. The work directly confronts the "flawed benchmarking culture" in graph learning—an over-reliance on a few static benchmarks and transductive evaluation, which is considered a major bottleneck hindering progress in the field.

2. GraphUniverse is the first framework to enable large-scale, systematic evaluation of inductive generalization. Its paradigm of generating graph families with persistent semantics, rather than single graphs, is a significant advancement over existing tools like GraphWorld.

3. The experiments compellingly demonstrate the gap between transductive and inductive performance and reveal that model robustness is context-dependent. not universal. These findings have important implications for model evaluation and development in the field.

4. The paper is accompanied by an interactive demo and a promise to open-source the package. This is a valuable public contribution that will greatly facilitate future research.

### Weaknesses
1. The entire framework is built upon the Degree-Corrected Stochastic Block Model (DC-SBM), which is inherently biased toward generating graphs with strong community structures. This limits the applicability of the framework's conclusions to other common graph topologies that lack clear communities (e.g., geometric graphs, some biological networks), thus constraining its universality.

2. The experimental evaluation focuses primarily on node-level community detection and a simple graph-level task (triangle counting). It notably omits other critical graph learning tasks, especially Link Prediction, which is itself a core research area for inductive learning. This narrow task scope weakens the generality of the paper's main conclusions.

3. In light of the two points above, the paper's claim of enabling "systematic evaluation of inductive generalization" may be too broad. The findings and conclusions are largely about generalization for "node classification tasks" on "graphs with community structure," a scope limitation that should be more explicitly stated in the paper.

### Questions
1. How can the authors ensure that the paper's central finding—the gap between transductive and inductive performance—is not an artifact of the DC-SBM generator itself? Could models not designed for community-structured data be unfairly penalized by this evaluation framework?

2. Link prediction is a key inductive graph learning task. Why was it omitted from the benchmark? Can the GraphUniverse framework be extended or adapted to support a systematic evaluation of inductive link prediction?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GraphUniverse, a framework for generating families of synthetic graphs to systematically study inductive generalization in graph learning. By extending the Degree-Corrected Stochastic Block Model into a hierarchical generator, GraphUniverse produces semantically consistent graphs with controllable properties such as homophily, degree, and feature variance. Experiments show that model rankings differ substantially between transductive and inductive settings, revealing key insights into robustness and generalization behavior across diverse graph architectures.

### Strengths
* Fixes a fundamental limitation of GraphWorld and other synthetic graph generators by enabling multi-graph, inductive evaluation, moving beyond the conventional single-graph, transductive paradigm.   
* Demonstrates that transductive performance is not predictive of inductive generalization and reveals architecture-specific robustness patterns under controlled distribution shifts. The analyses provide clear empirical evidence for a well-known but rarely quantified observation in the community—that graph models are often dataset- and domain-specific, performing well only in certain structural regimes. GraphUniverse’s controlled setup brings this behavior to light in a direct and systematic way.  
* The inclusion of an interactive demo application is a valuable and user-friendly addition, enhancing reproducibility and making the framework easy to explore and adopt for future research.

### Weaknesses
* While the proposed framework convincingly demonstrates inductive generalization on synthetic graph families, it remains unclear whether these findings extend to real-world datasets. The paper does not map real datasets into the GraphUniverse parameter space or provide empirical validation that inductive robustness measured on synthetic graphs predicts robustness in real settings. As a result, it is difficult to determine whether the generated families capture realistic structural regimes or how the observed trends translate to practical graph learning scenarios.  
* The claim that GraphUniverse can facilitate or benefit graph foundation model pretraining feels premature given the absence of supporting experiments or evidence. While this is not the paper’s central claim, the authors should at least include a more substantive discussion clarifying how GraphUniverse-generated data could be integrated into large-scale pretraining pipelines or what aspects of the framework make it suitable for such use.  
* The underlying DC-SBM formulation assumes discrete, non-overlapping communities, whereas many real-world graphs contain overlapping structures and higher-order motifs that this model cannot capture. 

My main concern is whether the insights obtained from GraphUniverse will transfer to real-world data. I suggest that the authors evaluate whether model rankings observed in the synthetic setting align with those on real benchmarks. For example, testing models on both heterophilic and homophilic real-world datasets could reveal whether performance trends from GraphUniverse are predictive of behavior in practical scenarios. It would be even more compelling if the authors also computed model rankings using GraphWorld and demonstrated that the rankings from GraphUniverse align more closely with those from real data. This would provide strong evidence for the external validity and practical relevance of the proposed framework. If the authors can demonstrate this, I am willing to increase my score.

### Questions
* Can the authors include a heterophily-specialized model (e.g., FAGCN, GPR-GNN, or H2GCN) in their comparisons, specifically for experiments varying homophily? This addition could better highlight how such models are specialized toward particular regimes by showing opposite performance trends across low- and high-homophily settings.  
* Could the authors include a discussion of recent graph foundation model efforts that have demonstrated the benefits of using synthetic data, such as GraphFM \[1\], KumoRFM \[2\], and OpenGraph \[3\]? Including a brief overview of these works, along with parallels to how synthetic data has driven major advances in other domains, would strengthen the paper’s positioning and broaden its appeal to the wider graph learning and foundation model community.  
* Can the authors add an analysis similar to Figures 1 and 2 from GraphWorld \[4\] to illustrate where real-world graph datasets lie within the GraphUniverse parameter space?

\[1\] Lachi, Divyansha, et al. "Graphfm: A scalable framework for multi-graph pretraining." *arXiv preprint arXiv:2407.11907* (2024).  
\[2\] Fey, M., Kocijan, V., Lopez, F., Lenssen, J. E., & Leskovec, J. KumoRFM: A Foundation Model for In-Context Learning on Relational Data.  
\[4\] Xia, L., Kao, B., & Huang, C. (2024). Opengraph: Towards open graph foundation models. *arXiv preprint arXiv:2403.01121*.

### Soundness
2

### Presentation
3

### Contribution
2
