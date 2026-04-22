# GALAX: Graph-Augmented Language Model for Explainable Reinforcement-Guided Subgraph Reasoning in Precision Medicine

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
In precision medicine, quantitative multi-omic features, topological context, and textual biological knowledge play vital roles in identifying disease-critical signaling pathways and targets, guiding the discovery of novel therapeutics and effective treatment strategies. Existing pipelines capture only one or two of these—numerical omics ignore topological context, text-centric LLMs lack quantitative grounded reasoning, and graph-only models underuse rich node semantics and the generalization power of LLMs—thereby limiting mechanistic interpretability. Although Process Reward Models (PRMs) aim to guide reasoning in LLMs, they remain limited by coarse step definitions, unreliable intermediate evaluation, and vulnerability to reward hacking with added computational cost. These gaps motivate jointly integrating quantitative multi-omic signals, topological structure with node annotations, and literature-scale text via LLMs, using subgraph reasoning as the principle bridge linking numeric evidence, topological knowledge and language context. To resolve this challenge, we propose GALAX (Graph Augmented LAnguage model with eXplainability), an innovative framework that integrates pretrained Graph Neural Networks (GNNs) into Large Language Models (LLMs) via reinforcement learning guided by a Graph Process Reward Model (GPRM), which generates disease-relevant subgraphs in a step-wise manner initiated by an LLM and iteratively evaluated by a pretrained GNN and schema-based rule check, enabling process-level supervision without explicit labels. As an application, we also introduced Target-QA, a benchmark combining CRISPR-identified targets, multi-omic profiles, and biomedical graph knowledge across diverse cancer cell lines, which enables GNN pretraining for supervising step-wise graph construction and supports long-context reasoning over text-numeric graphs (TNGs), providing a scalable and biologically grounded framework for explainable, reinforcement-guided subgraph reasoning toward reliable and interpretable target and pathway discovery in precision medicine.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Precision medicine relies on the use of multi-omic features, structural information as context and textual biological knowledge. Each of these aspects however, have their own shortcomings. The paper suggests leveraging their strengths by proposing a unified framework of subgraph reasoning. The paper proposes GALAX, a method that combines pretrained language models with Graph Neural Networks (GNNs) for reasoning using reinforcement learning. A process reward model over graphs guides the subgraph generation process pertaining to disease information. The final reasoning traces then leverage subgraphs to generate query responses. Authors further propose Target-QA, a QA benchmark for precision medicine, developed using the DepMap for evaluating GALAX. Empirical evaluations demonstrate that GALAX improves over considered baselines and its components are found to be performant.

### Strengths
* The paper is combined an innovative set of ideas.
* The paper is well organized and comprehensive.

### Weaknesses
* **Motivation:** The central motivation and idea behind the paper remains unclear. The paper states that it aims to integrate multi-omics, topological structure and textual information in order to curb their limitations. However, it remains unclear how the paper carries this out as combining these aspects leads to combining both their strengths and weaknesses. The paper does not mention on how multi-omics and structure interact with each other and how language plays an important role in representing topological structure. Furthermore, authors claim that the combination of these aspects makes reasoning explainable/interpretable. However, this claim is not empirically studied or validated as the paper does not mechanistically study interpretability of the reasoning process. It would be worthwhile to refine the central motivation and scope of the work.

* **Dataset:** Authors curate and propose the Target-QA dataset. However, it remains unclear as to why the dataset is needed and how it serves evaluation. Does the dataset balance between quality and diversity of medicinal queries? Why curate a dataset from an established dataset for evaluation? Were other QA datasets in phenomics, transcriptomics and biological summarization considered? It also remains unclear on how effective the dataset is evaluating GALAX. The paper states that Target-QA consists of 363 QA pairs which are significantly low for a textual QA dataset. 

* **Experiments & Ablations:** My main concern is that the paper does not validate its central contribution. Experiments currently compare GALAX to different settings of data aggregation and training. However, the main subgraph generation scheme using process reward model is not empirically studied. How does RL help in the generation process? How do process rewards contribute to guidance? How does RL compare to SFT? What happens if we ablate dense rewards with sparse rewards? How does the model behave when we replace subgraphs with the full knowledge graph? These questions remain unexplored corresponding to the central theme of the work.

* **Baselines:** Experimental setting currently considers different training and dataset modality regimes. However, these baselines do not put the method and its central contribution of subgraphs in perspective. It would be worthwhile to consider LLM with KGs for process rewards. Furthermore, what happens if we assess generation across different cell lines or heldout structures? Baseline ablations could help answer these questions.

* **Presentation:** Overall the presentation of the paper is dense. The paper presents complex terminology which is often overloaded and makes the text harder to understand. Section 4 is presented in a vague form. Various details on the graph generation process and training could be placed in the appendix for improving readability.

#### Minors
* line 185: has has $\rightarrow$ has
* Table 1: what is LUAD and BRCA?

### Questions
Refer to weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces GALAX, a pipeline for QA in precision medicine using a joint LLM-GNN model using subgraphs generated via a graph process reward model as explainable artifacts for reasoning. The authors also introduce a new benchmark, Target-QA, focused on biomedical graph knowledge, multi-omic profiles, and CRISPR targets.

### Strengths
- The interpretability aspect of the method is strong and well-reasoned. Penalizing schema violations in the reward function and rewarding path-like structures is good.
- Experiments show GALAX consistently outperforms baselines. The performance is very good.
- Target-QA is an important contribution to the community for working with multi-omic features.

### Weaknesses
- There doesn't seem to be studies on the performance of the method when the KG is incomplete or noisy/not well aligned, which is almost always the case in real-world KGs.
- Most evaluation is done on Target-QA. Validation on external datasets is not emphasized, making it more difficult to assess the impact of this work.
- The authors mention reward hacking as a problem for previous PRMs. However, how the method combats this is not clear beyond the schema relation penalty. There is no analysis on how gameable the proposed graph PRM is.

### Questions
- How robust is the GPRM to shifts in important graph characteristics such as graph density or KG incompleteness?
- How sensitive is performance w.r.t. the backbone LLM?
- What are the failure modes of GALAX?  Are there particular paths/subgraphs that make any systematic failures evident?
- How robust is the NER function? It seems this is critical for seeding a good graph.

### Soundness
4

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
This paper tackles target discovery in precision medicine by integrating multi-omic features, biomedical graphs, and literature via GALAX, which couples an LLM with a pretrained GNN and a Graph Process Reward Model to build explainable, RL-guided subgraphs. On the new Target-QA benchmark, GALAX consistently outperforms strong language-only and graph-augmented baselines on precision/recall and Hit@5/10, with qualitative pathway enrichment analyses supporting biological plausibility.

### Strengths
- The paper introduces a clear process-level supervision scheme (GPRM) that links subgraph construction to a graph foundation model, yielding interpretable rationales rather than only outcome metrics.

- The Target-QA benchmark and comprehensive comparisons/ablations (including complexity analysis and enrichment studies) make the evaluation credible and reproducible.

### Weaknesses
- Please streamline the problem formulation: many set-indexed variables (e.g., X1, X2, X3,...) could be replaced with compact matrix/tensor notation, and a schematic showing each modality and how it connects to the graph would improve readability.

- The framework depends on multiple data sources (multi-omics, biomedical graph, literature); it would be helpful to analyze performance in low-resource scenarios (e.g., patients lacking certain omic assays) to gauge robustness

### Questions
- There are minor typos: two consecutive “and” on p.2, line 64, and duplicated “has” on p.4, line 186.

- The symbol 𝜀DTI appears to be used inconsistently: in the formulation it denotes disease–protein interactions, while 4.1 uses it for drug–target interactions. Please unify the definition throughout.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The manuscript introduces GALAX (Graph Augmented LAnguage model with eXplainability), a framework that integrates pretrained Graph Neural Networks (GNNs) with Large Language Models (LLMs) through reinforcement learning guided by a Graph Process Reward Model (GPRM). The authors also introduce Target-QA, a benchmark combining CRISPR-validated targets, multi-omic features, and biomedical knowledge graphs across cancer cell lines, supporting both GNN pretraining and long-context text-numeric graph reasoning. The work targets explainable, reinforcement-guided subgraph reasoning for interpretable pathway and target discovery in precision oncology.

### Strengths
1.  Bridging numerical multi-omics, graph structure, and textual knowledge for mechanistic interpretability in precision medicine.

2.  Target-QA dataset/benchmark combining CRISPR screens, multi-omics, and biomedical KG, enabling both supervision and evaluation.

### Weaknesses
- Need stronger baselines (e.g., state-of-the-art KGQA, graph-augmented LLMs)

- It’s unclear how well GALAX transfers across cell lines, cancer types, and unseen targets, especially under shifts in omic distributions or KG incompleteness.

- Human-in-the-loop or expert-curated assessments of subgraph plausibility are missing.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
2
