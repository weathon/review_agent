# TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Well-designed open-source software drives progress in Machine Learning (ML) research. While static graph ML enjoys mature frameworks like PyTorch Geometric and DGL, ML for temporal graphs (TG), networks that evolve over time, lacks comparable infrastructure. Existing TG libraries are often tailored to specific architectures, hindering support for diverse models in this rapidly evolving field. Additionally, the divide between continuous- and discrete-time dynamic graph methods (CTDG and DTDG) limits direct comparisons and idea transfer. To address these gaps, we introduce Temporal Graph Modelling (TGM), a research-oriented library for ML on temporal graphs, the first to unify CTDG and DTDG approaches. TGM offers first-class support for dynamic node features, time-granularity conversions, and native handling of link-, node-, and graph-level tasks. Empirically, TGM achieves an average 7.8× speedup across multiple models, datasets, and tasks compared to the widely used DyGLib, and an average 175× speedup on graph discretization relative to available implementations. Beyond efficiency, we show in our experiments how TGM unlocks entirely new research possibilities by enabling dynamic graph property prediction and time-driven training paradigms, opening the door to questions previously impractical to study.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new library for temporal graph learning that unifies CTDG and DTDG, supporting a wide range of tasks and methods efficiently. The work includes several novel components, notably the unification of CTDG and DTDG and the hook management mechanism for obtaining training features. Its modular design is well-conceived and will allow researchers to easily develop, extend, and evaluate new methods.

### Strengths
The proposed library is timely and represents a valuable contribution to the field of temporal graph learning. It offers a convenient experimental infrastructure and a fair evaluation platform that can accelerate research progress. Compared with existing libraries, it appears more general and efficient, and it provides a diverse collection of datasets and methods to facilitate easy and consistent comparison across studies.

### Weaknesses
W1. My primary concern lies in the writing. First, while the authors emphasize the efficiency of TGM, the paper lacks a detailed explanation or analysis clarifying why TGM is more efficient than existing methods. Second, Section 3 spans nearly two pages but presents multiple items in a disconnected manner, making it difficult to follow the logical flow of ideas.

W2. I also noticed that a prior study, TGB-Seq, introduced several new datasets and integrated them into DyGLib. It would be better if the authors could incorporate the TGB-Seq datasets into TGM as well, since these datasets have been widely adopted in recent research.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TGM, a research-oriented library for Temporal Graph Learning (TGL) 
that supports both continuous-time (CTDG) and discrete-time (DTDG) dynamic graphs within a single modular system. 
The library proposes a formal unification of both paradigms via the notion of graph discretization and time granularity. 
TGM supports link-, node-, and graph-level tasks and demostrantes significant efficiency gains over prior libraries such as DyGLib and UTG.

### Strengths
- The paper addresses a timely and relevant problem in dynamic graph learning, where current frameworks lag behind static counterparts (e.g., PyG, DGL) in terms of flexibility, modularity, and computational efficiency.
- The manuscript is well written and clearly structured, providing sufficient context and motivation for the proposed library.
- If properly released and maintained as open-source, TGM could serve as a standard reference library for temporal graph learning research.
- The framework demonstrates substantial efficiency gains across multiple models and tasks, indicating careful engineering and system design.

### Weaknesses
- The empirical section primarily focuses on efficiency metrics and new results on CTDG tasks solved using DTDG-based models. However, these results offer limited analysis of performance in comparison with prior libraries and existing literature.
- Some aspects are insufficiently detailed:
    1) It is not clear whether experiments across libraries use identical data splits, hyperparameters, preprocessing, etc.
    2) The paper does not clearly explain how batches are processed, specifically, whether events or snapshots within a batch are handled in parallel or sequentially, and if parallel processing is used, if temporal causality is preserved.
    3) The treatment of deletion events (e.g., node removal) is not detailed in the paper.
    4) It is unclear if irregularly sampled snapshot-based tasks (e.g., as in [4,5]) can be processed within TGM.
- TGM currently implements tasks for CTDGs and enables DTDG models to operate on these tasks. However, the TGL community would greatly benefit if TGM also supported tasks specifically designed for DTDGs, such as Metr-LA and Pems-Bay [6], making it a truly unified library that facilitates research across both CTDG and DTDG paradigms.
- When evaluating DTDG-based models, the authors should include comparisons against dedicated DTDG frameworks, such as Torch Spatiotemporal or PyTorch Geometric Temporal, to reduce the large number of unsupported baselines reported in Tables 3, 4, and 9.
- The unified conceptual view connecting CTDGs and DTDGs is interesting; however, the theoretical relationship between these two temporal graph paradigms has already been discussed in prior works, including [1, 2, 3].
  Although not implemented in a general-purpose library, these contributions should be properly acknowledged in the manuscript.
- Given the growing interest on long-range information propagation and oversquashing in temporal GNNs [7], the authors should consider including CTAN [8], a model specifically designed to address this problem, within the TGM. This addition would further broaden the TGM's coverage and enhance its utility for research on novel temporal GNN architectures.

-----

[1] [Representation Learning for Dynamic Graphs: A Survey. JMLR 2020](https://jmlr.csail.mit.edu/papers/volume21/19-447/19-447.pdf)

[2] [Deep learning for dynamic graphs: models and benchmarks. IEEE TNNLS 2024](https://ieeexplore.ieee.org/document/10490120)

[3] [Graph neural networks for temporal graphs: State of the art, open challenges, and opportunities. TMLR 2023](https://openreview.net/pdf?id=pHCdMat0gI)

[4] [Graph Neural Controlled Differential Equations for Traffic Forecasting. AAAI 2022](https://cdn.aaai.org/ojs/20587/20587-13-24600-1-2-20220628.pdf)

[5] [Temporal Graph ODEs for Irregularly-Sampled Time Series. IJCAI 2024](https://www.ijcai.org/proceedings/2024/0445.pdf)

[6] [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR 2018](https://arxiv.org/abs/1707.01926)

[7] [Over-squashing in Spatiotemporal Graph Neural Networks. 2025](https://arxiv.org/pdf/2506.15507)

[8] [Long Range Propagation on Continuous-Time Dynamic Graphs. ICML 2024](https://openreview.net/pdf?id=gVg8V9isul)

### Questions
Refer to the paper’s weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a research‑oriented software library that unifies continuous‑time and discrete‑time learning on temporal graphs under one formal and practical framework. The authors introduce a typed hook abstraction for composing common temporal operations, and they formalize time‑granularity conversion via a discretization operator, enabling iteration either by events or by time. The system follows a three‑layer architecture, supports node and edge events, and implements representative models spanning message‑passing, transformer‑based, and snapshot methods. Empirically, TGM reports stronger efficiency  than DyGLib and UTG. It also enables research case studies on dynamic graph properties, snapshot granularity, and batching effects.

### Strengths
S1. This paper is generally well-writen and easy to follow.

S2. This paper proposes a clean theoretical unification of CTDG and DTDG via the notion of a native time granularity and a principled discretization operator.

S3. This paper introduces a typed hook formalism with explicit dependency contracts, which makes complex temporal pipelines composable and easier to reason about and test. 

S4. This paper provides a well‑architected system that separates data storage, execution, and model layers. the proposed framework demonstrates broad coverage of tasks and models.

### Weaknesses
W1. This paper’s primary contribution is a software framework. While the system is valuable, the methodological novelty is limited relative to prior unification attempts (e.g., UTG) and existing libraries (e.g., DyGLib), raising questions about the conceptual contributions beyond engineering improvements.

W2. This paper claims to unify CTDG and DTDG approaches, and it indeed provides a conceptual bridge through an event-sequence formulation and discretization operator. However, the empirical section does not yet demonstrate unification beyond efficiency. For example, there is no study comparing a CTDG model and a DTDG model on exactly the same data, tasks, and evaluation protocol.

W3. The coverage of DTDG methods in the library and experiments is limited. The implemented and evaluated snapshot-based models are mainly GCN, GCLSTM, and T-GCN, while stronger or more representative DTDG baselines such as EvolveGCN[1] and DySAT[2] are neither implemented nor evaluated.

W4. The experimental section focuses heavily on efficiency, but it does not verify correctness against the original authors’ official implementations or reported numbers across multiple metrics. As a result, the validity of the re-implemented models cannot be fully confirmed.

W5. This paper lacks empirical comparisons with PyG Temporal[3] and Torch Spatiotemporal[4], two DTDG-oriented libraries that are discussed in the related-work section but not benchmarked in Section 5.

Minor Typos. 

+ "up to 246× than DyGLib" -> "up to 246× faster than DyGLib". 

+ The spelling of behavior and behaviour should be made consistent.

+ Table 8 "first and second are highlighted" -> "First and second best are highlighted"



Reference

[1] EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs

[2] Dynamic Graph Representation Learning via Self-Attention Networks

[3] PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models

[4] https://github.com/TorchSpatiotemporal/tsl

### Questions
1. Will the framework be adapted to text-attributed dynamic graphs[1]?

[1] DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TGM, a modular and efficient open-source library for temporal graph learning. 
It unifies continuous-time and discrete-time dynamic graph paradigms within a single framework, providing native support for time operations, event-driven iteration, and node/edge dynamics. 
TGM uses a _hook_ abstraction to implement flexible composition of temporal graph operations and efficient workflows. 
The experiments show convincing speedups on a broad set of models.

### Strengths
- Unification: Comprehensive integration of CTDG and DTDG models under a single abstraction.

- Technical design and efficiency: The hook-based modularity and vectorized implementation lead to significant computational gains.

- Empirical validation and usability: Extensive benchmarks on multiple datasets and models.

### Weaknesses
-Limited novelty beyond software engineering: While the framework is well-engineered, the conceptual contribution (e.g., the hook abstraction) is mostly organizational rather than methodological.

- Evaluation scope: The Experiments focus mainly on efficiency and reproducibility; some examples showing directions of possible novel research would strengthen the claim of scientific impact.

### Questions
Apart from the points discussed above, there are the following minor points:

- Could the authors clarify how hooks differ from standard PyTorch data transformations or DGL message-passing pipelines conceptually?

- The paper mentions “dynamic graph property prediction” as a novel task; could more examples or datasets be shown?

- How does discretization handle overlapping time intervals or missing timestamps in real-world data? Or edges appearing and disappearing within the same time interval.

- Are there guidelines for selecting optimal time granularities, given their strong impact on performance?

- Representing Continuous-Time and Discrete-Time Graphs: A similar idea has been used already in [1], Definition 2 and 3.

[1] A. Longa et al., Graph Neural Networks for Temporal Graphs: State of the Art, Open Challenges, and Opportunities, TMLR (2023)

### Soundness
4

### Presentation
4

### Contribution
3
