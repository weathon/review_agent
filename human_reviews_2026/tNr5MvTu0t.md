# PLANETALIGN: A Comprehensive Python Library for Benchmarking Network Alignment

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Network alignment (NA) aims to identify node correspondence across different networks and serves as a critical cornerstone behind various downstream multi-network learning tasks. Despite growing research in NA, there lacks a comprehensive library that facilitates the systematic development and benchmarking of NA methods. In this work, we introduce PLANETALIGN, a comprehensive Python library for network alignment that features a rich collection of built-in datasets, methods, and evaluation pipelines with easy-to-use APIs. Specifically, PLANETALIGN integrates 18 datasets and 14 NA methods with extensible APIs for easy use and development of NA methods. Our standardized evaluation pipeline encompasses a wide range of metrics, enabling a systematic assessment of the effectiveness, scalability, and robustness of NA methods. Through extensive comparative studies, we reveal practical insights into the strengths and limitations of existing NA methods. We hope that PLANETALIGN can foster a deeper understanding of the NA problem and facilitate the development and benchmarking of more effective, scalable, and robust methods in the future. The source code of PLANETALIGN is available at https://github.com/yq-leo/PlanetAlign

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new python library, PLANETALIGN, for the evaluation of network alignment algorithms. This new framework includes 18 datasets from various domains, like social, biological, knowledge graph, and publication. It supports 14 methods which covers major categories of NA algorithms. 
This new framework provides unified APIs, and standardized evaluation metrics for model implementation and evaluation. The author conducted experiments to compare the performance of 14 algorithms across all datasets on effectiveness, efficiency, robustness and sensitivity. 
Compare with the existing algorithms, the new framework achieves similar accuracy with 3x faster on efficiency.

### Strengths
This new python library integrates a wide range of datasets and algorithms. It provides a useful and fair benchmark for NA researches. It provides a consistent API and unified evaluation pipeline, which makes the method comparisons easier, more consistent and more reliable. 
The implementation is efficient and well optimized. The runtime is significantly less the the original baseline. 
The experiments shows meaningful empirical findings.  
The documentation is clear and easy to follow. Users can ramp up quickly with the tutorial and detailed references.

### Weaknesses
The main limitation is lack of extensibility on dataset, functions and customized use cases. This library is mainly focus on algorithm comparisons rather than support the downstream researches. 
Through this python library is well designed and easy to use, it is not very clear how will it keep up with the fast iterating NA research field. New datasets and new algorithms emerge quickly. 
Currently this framework mainly serves as a benchmark tool with fixed datasets. The paper does not mention whether there is a maintenance plan to expand to new datasets, or support for customized datasets. 
This work can be stronger and makes larger impact if the library could be more extensible and more research orientated, supporting NA related researches rather than simply method comparison. For example, add APIs for downstream tasks, customized loss functions, multitask learning, etc. Such improvements will certainly contribute more to the acceleration of new scientific discoveries.

### Questions
My questions are mainly related to the limitations I mentioned above.
1. The paper mentioned that the multi-network alignment is not explored in this study. I was wondering, with the current architecture, how difficult would it be to extend PLANETALIGN to multi-graph or cross domain alignment? 
2. Are there future plans to extend the support of new datasets or new graph modalities, like heterogeneous graph?
3. If future versions is improved to support new dataset, especially larger datasets, will it include distributed computation or other scalability improvements to handle large graphs?

### Soundness
3

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
This paper introduces PLANETALIGN, a comprehensive Python library for benchmarking Network Alignment (NA) algorithms across multiple domains. The framework integrates 18 datasets spanning six application areas (social, biological, academic, knowledge graph, infrastructure, and communication networks) and implements 14 representative algorithms, including consistency-based, embedding-based, and optimal transport (OT)-based methods. PLANETALIGN provides a unified PyTorch-style API, standardized evaluation metrics, and extensive experiments assessing algorithms on four dimensions—effectiveness, scalability, robustness, and supervision sensitivity. The work fills an important gap in the NA community by establishing a reproducible, standardized evaluation foundation, though it introduces no new alignment algorithm or theoretical framework.

### Strengths
(1) Comprehensive benchmark coverage: PLANETALIGN integrates 18 datasets from six domains and 14 representative algorithms, providing the most complete evaluation platform to date for network alignment research.
(2) Systematic multi-dimensional evaluation: The framework assesses algorithms along four complementary dimensions—effectiveness, scalability, robustness, and supervision sensitivity—offering a more holistic perspective than prior work.
(3) High implementation quality and reproducibility: The library provides a clean, modular PyTorch-style API, reproducible experiment settings, and optimized code achieving up to 3× faster runtime compared to original implementations.
(4) Insightful empirical analysis: Experiments yield practical insights, such as the superior stability of OT-based methods and the efficiency-memory trade-offs in embedding-based approaches.
(5) Community impact: By releasing a well-documented, open-source library, the work provides an infrastructure that can serve as a long-term standard for the NA community.

### Weaknesses
(1) Limited algorithmic novelty: The paper does not introduce new alignment methods or theoretical developments. Its contribution is primarily infrastructural, which may limit interest for readers expecting methodological innovation.
(2) Incomplete scalability validation: Experiments are limited to graphs with fewer than 100k nodes. Given the scalability claims, evaluation on larger-scale graphs (e.g., >1M nodes) would better support the authors’ argument.
(3) Shallow theoretical motivation: The paper lacks deeper theoretical analysis or justification for why OT-based approaches perform better, relying mainly on empirical evidence.
(4) Restricted scope: The current implementation only supports pairwise alignment. Extending the framework to multi-network or dynamic alignment scenarios could significantly broaden its applicability.

### Questions
(1) How does PLANETALIGN ensure fairness when comparing algorithms that rely on different supervision levels (e.g., supervised vs. semi-supervised vs. unsupervised settings)? Are hyperparameter budgets standardized across methods?
(2) The paper claims scalability and efficiency advantages. Could the authors include additional results on large-scale graphs (e.g., >1M nodes) to substantiate this claim?
(3) The framework currently focuses on pairwise alignment. Are there plans or design provisions to extend it toward multi-network or temporal (dynamic) alignment scenarios?
(4) Since the results show that OT-based methods consistently outperform others, can the authors provide more theoretical intuition or ablation evidence to explain why OT-based models are so robust across domains?

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
2

### Summary
This paper introduces PLANETALIGN, an open-source Python library aimed at providing a standardized platform for evaluating and developing network alignment (NA) methods. Network alignment is essential for identifying node correspondences across multiple networks, a task central to applications in various domains, including social network analysis, bioinformatics, and knowledge graph fusion. Despite the importance of this task, there has been a lack of a comprehensive benchmarking tool for NA methods. PLANETALIGN addresses this gap by providing a unified library that integrates 18 datasets from 6 domains, supports 14 NA methods, and offers a rich set of evaluation metrics. The library features easy-to-use APIs and a flexible design, allowing users to benchmark and develop new methods for network alignment. The authors claim that their library outperforms previous solutions in terms of speed and provides in-depth comparative studies that highlight the strengths and limitations of existing NA methods.

### Strengths
1. The paper offers a large collection of 18 datasets from diverse domains and 14 NA methods, providing a solid foundation for comparison across different scenarios. 

2. PLANETALIGN emphasizes reproducibility by using standardized evaluation metrics (Hits@K, MRR) and consistent dataset splits. The inclusion of time and memory usage metrics makes it easier to assess the scalability and efficiency of algorithms.

3. PLANETALIGN has demonstrated significant improvements in execution time, with some methods being up to 3 times faster than their official counterparts.

### Weaknesses
1. Many datasets are synthetically generated or permuted from a single network, which may not fully capture the heterogeneity and noise of real-world multi-network environments.

2. The evaluation focuses heavily on quantitative metrics, but does not address the interpretability or explainability of the alignment results—a critical aspect in domains like bioinformatics or fraud detection.

3. The paper emphasizes the library’s ease of use and API design but provides no user study or external feedback to validate these claims.

### Questions
1. While PLANETALIGN supports a few OT-based methods, could you elaborate on potential future plans to incorporate more OT methods, especially those that handle non-convex optimizations better?

2. How do the methods in PLANETALIGN perform when applied to networks with millions of nodes and edges? Have you tested the library's performance on very large-scale graphs?

3. Could you discuss potential plans for updating the library, such as adding new NA methods, datasets, or advanced evaluation techniques in future releases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
PlanetAlign collects and organizes under clean and standardized APIs, datasets, methods and benchmarks for the network alignment problem: 18 datasets, 14 methods, standard evaluation metrics (Hits@k, Mean Reciprocal Rank (MRR)),  diverse benchmarking settings,  emphasizing effectiveness (e.g. supervision regimes), scalability (e.g. logging), robustness (e.g. controlled noise injection). PlanetAlign is put into a comprehensive test: empirically evaluating all methods over collected datasets (semi-supervised setting), reporting on metrics and drawing insights on what type of methods work better than others or their particular requirements).

### Strengths
- This is an excellently organized manuscript, describing the design and practical usage of a solid, software engineering work.

- The engineering contribution is substantial and noteworthy: links to related, mature artifacts (code repository, documentation) are conveniently provided. Abstractions identified are well thought-of and disciplined. 

- It includes reports on extensive empirical studies: it would be hard to imagine comparing so "easily", so many and diverse methods for network alignment over multiple (and multiple-type) networks prior to this work. PlanetAlign makes a Herculean task, manageable: it certainly accelerates "network alignment" research. Figure 3 provides a compelling summary of the development simplicity made possible.

- Nicely reuses and integrates organization ideas and abstractions from other, highly-popular, open-source projects (e.g. pytorch-geometric).

- Impressive performance comparison agreement of PlanetAlign and "official" implementations of methods.

### Weaknesses
Relevance to ICLR audience is the major weakness of this effort:
- PlanetAlign is an (excellently engineered) collection of existing artifacts for network alignment methods, so  - although by "design" - the novelty dimension is limited, in particular for ML audience expecting emphasis on deep/representation learning (although there is an embeddings-related component part).
- This work would be potentially a best fit for a venue focusing on networks.

### Questions
- Are there provisions/facilities for the user community to easily "extend" PlanetAlign (i.e. integrated methods/datasets into it) - rather than the authors expanding it as in Appendix E?
- Any thoughts for a kind of leaderboard for the network alignment community?

### Soundness
4

### Presentation
4

### Contribution
3
