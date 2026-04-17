# Global Meta-path-level Counterfactual Explanation for Heterogeneous Graph Neural Networks by Path Exclusion

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Heterogeneous graph neural networks (HGNNs) capture rich structural and semantic information in real-world data, but their decision processes are often opaque. While recent work in graph counterfactual explanations perturbs nodes, edges, or subgraphs to identify influential components, such approaches are too coarse for heterogeneous graphs: a single perturbation can disrupt many meta paths, obscuring which relations truly drive model behavior. Since meta paths are fundamental units of semantic information in HGNNs, explanations at the meta-path level are both natural and necessary.
We introduce meta path exclusion, a framework for global counterfactual explanations that directly perturbs specific meta paths. To achieve this, we propose the spare path algorithm, which modifies the forward pass of HGNNs to exclude target meta paths while preserving the rest of the graph, enabling precise attribution of model performance to individual paths.
Experiments on four benchmark datasets (DBLP, OGB-MAG, IMDB, and MovieLens) show that excluding a small set of meta paths can significantly reduce accuracy, revealing their critical role. Moreover, when these identified paths are reused in meta-path-dependent HGNNs, their removal consistently degrades accuracy by 5–12\%, confirming their general importance. Our results establish meta paths as atomic units for fine-grained, faithful counterfactual explanations in heterogeneous graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors introduce a method to generate global counterfactual explanations (CFE) of heterogeneous graph neural networks. They argue that perturbing nodes, edges, or other structural features cannot satisfactorily give high quality CFE because each perturbation of this nature can fail to identify the phenomenon that actually dictates the model behavior. They propose a framework that is a post-hoc explainer that perturbs meta paths instead to generate CFE for HGNN. They use their spare path algorithm to exclude the computation of the leaf part of the excluding path in the HGNN on the forward pass. After perturbed path embeddings are computed the framework optimizes to find the optimal meta path (which acts as a CFE). They conduct experiments with relevant baselines on fidelity, distance, and sparsity. Their initial experiments suggest improvements when using their framework over existing

### Strengths
S1. The paper is well written, simple, logical and provides a novel angle to solving a niche problem; global CFE on HGNNs.


S2. Methodology is reasonable and initial implementation decisions make sense. 


S3. The results in some experiments suggest that the method is more effective than existing global graph CFE methods particularly when applied to meta paths explanations.

### Weaknesses
W1. The work provides a reasonable solution to an overlooked problem. Although the experiments suggest their methodology can exhibit superior performance the results are by no means definitive. They need more experiments to supplement their claims.
W2. The authors leave a lot of technical details out of the main body of the paper that are necessary. For instance, even though the optimization procedure is not novel to the work the authors fully leave out how they optimize the best perturbed meta path. They state this is put in the appendix but there is a serious lack of technical details in the paper that are quite necessary.

### Questions
1) I believe as per W1 more experiments are needed to justify the superiority of this method. If you can include some additional experimental datasets (this is necessary to bolster claims) and possibly some additional baselines (if needed).
2) The paper should clearly indicate the technical details of their methodology. These details should not be delegated to the appendix. It should be clear how the framework works in all of its technical details.

### Soundness
3

### Presentation
2

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
This paper presents a novel framework for global meta-path-level counterfactual explanation in heterogeneous graph neural networks. The authors propose a spare path algorithm that directly excludes specific meta-paths during forward propagation to evaluate their global impact. The method is implemented as a wrapper over HGT-like architectures and evaluated on three benchmark datasets (DBLP, OGB-MAG, IMDB). Results demonstrate that the approach effectively identifies semantically meaningful meta-paths and achieves competitive fidelity–sparsity trade-offs.

### Strengths
1.	Conceptual novelty: Introducing meta-path-level counterfactual analysis fills a gap between edge-level and subgraph-level explainers, offering a new abstraction for heterogeneous graphs.
2.	Practical implementation: The proposed spare path mechanism is elegant, simple to integrate into existing GNNs, and supported by clear pseudo-code.
3.	Empirical support: Comprehensive experiments on multiple datasets demonstrate the method’s interpretability and scalability within realistic heterogeneous settings.
4.	The framework could benefit practical applications (e.g., academic network analysis, recommender reasoning, and knowledge graph inspection) where meta-path semantics are essential.

### Weaknesses
1.	Results (fidelity, sparsity, accuracy) are reported as single values without variance or confidence intervals. Including results over multiple random seeds would make the conclusions statistically stronger. 
2.	The current set of baselines omits some recent counterfactual explanation approaches such as CFExplainer [1] and GCFExplainer [2].

[1] Graph neural networks for vulnerability detection: A counterfactual explanation. 2024

[2] Global counterfactual explainer for graph neural networks. 2023

3.	The adaptation of local explainers (e.g., CF-GNNExplainer, PGExplainer) to the global setting lacks detailed parameter descriptions, which slightly affects fairness assessment.
4.	The interpretation of results could be more analytical. For example, why certain meta-paths cause stronger performance drops or how redundancy between paths affects importance ranking.

### Questions
Please address the weaknesses.

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
3

### Summary
Counterfactual explainability in heterogeneous graphs is a challenging problem. The paper introduces the Spare Path algorithm, which removes meta-paths while preserving the remainder of the heterogeneous graph. Evaluated on three datasets, the method demonstrates strong effectiveness.

### Strengths
- Addresses a hard, real‑world problem with clear practical relevance.
- Code is available to support reproducibility, though some components appear incomplete.
- Results are promising but limited in scope and breadth.
- Method is simple yet effective, offering a straightforward baseline with strong initial performance.

### Weaknesses
- The paper is poorly written; several sections are difficult to follow and transitions between sentences are weak.
- In Table 2, meta-path exclusion or usage has little impact on IMDB but shows some effect on DBLP.
- The experimental evaluation is underpowered, with too few datasets.
- There is no ablation study and no concrete procedure for meta-path selection.

### Questions
- Can you validate the Table 2 findings on additional datasets to test whether the meta-path effect generalizes?
- Please expand the experimental suite with more heterogeneous graph datasets to better demonstrate the model’s effectiveness.
- How many independent runs did you perform? Provide full experimental settings (seeds, splits, hardware, hyperparameters) and report mean -+ standard deviation across runs. If results are from a single run, please rerun multiple times and add variance metrics and statistical significance.
- Is there a principled way to select meta-paths a priori; without inspecting outcomes from the Spare Path algorithm (e.g., via heuristics, validation criteria)?

### Soundness
2

### Presentation
1

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
This paper proposes a framework for global meta-path-level counterfactual explanations in heterogeneous graph neural networks (HGNNs). The main contribution is an algorithm that excludes specific meta paths during the forward pass to estimate their influence on model predictions. The goal is to achieve finer-grained and semantically meaningful explanations compared to traditional node- or edge-level perturbations. Experiments on DBLP, OGB-MAG, and IMDB show that removing a small number of meta paths can significantly degrade accuracy, suggesting that these paths are crucial for model reasoning.

The topic is timely, and the conceptual novelty is evident. However, the paper in its current form suffers from critical flaws in its evaluation methodology, including inconsistent metric definitions and unclear experimental presentation, which undermine the validity of its conclusions.

This paper should be rejected because (1) the definitions and application of core evaluation metrics are flawed, reversing the established meanings of fidelity and sparsity from cited work without justification, (2) the reported fidelity and sparsity values contradict their theoretical range, which undermines the credibility of the quantitative results, (3) the presentation of experiments is difficult to understand, particularly in Table 1, which lacks the necessary clarity for interpretation, and (4) the analysis of results is superficial, failing to provide sufficient interpretive depth to convincingly support the paper's conclusions.

# Overall Evaluation

This is a creative and promising submission introducing a new perspective on HGNN explainability. However, the work is undermined by inconsistent and non-standard metric definitions, unclear reporting in tables, and a discussion that lacks interpretive depth. With clear terminology, explicit and correct metric definitions, improved table formatting, and a deeper analytical discussion, the paper could become a valuable contribution in future iterations. In its current state, the foundational issues with the evaluation prevent a confident assessment of the method's effectiveness.

### Strengths
The idea of counterfactual reasoning at the meta-path level is original and well-motivated. The implementation of the proposed path exclusion algorithm appears technically sound and is clearly explained. The approach has the potential to open a promising new direction for HGNN interpretability once the significant issues with evaluation and presentation are addressed.

### Weaknesses
A critical flaw lies in the definition, interpretation, and reporting of the evaluation metrics. The paper cites fidelity and sparsity from prior work (CF-GNNExplainer and Yuan et al., 2022), but inverts their standard interpretations: fidelity is treated as higher-is-better and sparsity as lower-is-better, which is the opposite of the cited definitions. This fundamental inversion is not justified or even acknowledged in the main text, causing significant confusion and making it difficult to compare results against the established literature.

Compounding this issue, both reported fidelity and sparsity values frequently exceed 1 (e.g., values like 44.43 for fidelity and 81.64 for sparsity in Table 1), contradicting the theoretical [0, 1] range for such metrics. These appear to be percentages, but this is never explicitly stated. Even if they are percentages, this does not resolve the core problem of the inverted definitions and makes the quantitative results difficult to trust. A rigorous and consistent evaluation framework is essential for the paper's claims to be verifiable.

Table 1 is particularly unclear and requires substantial revision. To make it interpretable:

Add arrows in the column headers (Fidelity~↑, Sparsity~↓) to explicitly indicate the optimization direction you are assuming.
Use boldface to highlight the best-performing result within each dataset and metric column.
Clarify in the caption that the Accuracy row at the top of each dataset refers to the baseline performance of the HGNN before any meta-path exclusion and is not tied to a specific path explanation.
Include a short note or footnote specifying that Fidelity and Sparsity values are reported as percentages and clarify their definitions.

The discussion of results is largely descriptive, with limited interpretation of why certain meta paths are more influential than others. Captions merely restate what figures depict instead of summarizing insights. As a result, the empirical analysis does not convincingly support the conclusions drawn in the text.

### Questions
# Clarifications on Current Presentation
1. Terminology: Please clarify the precise meaning of “spare path” and how it differs conceptually and computationally from a more straightforward term like “excluded path”.
2. Metrics: Why do fidelity and sparsity values exceed 1 in Table 1? Please confirm if they are expressed as percentages and, more importantly, justify the inversion of their standard definitions from the literature. Furthermore, the evaluation would be strengthened by including the performance of the comparison models on the actual paths. Please provide these results or explain their omission.
3. Table 1 Clarity: Could you expand the description of Table 1 to explicitly link the “Accuracy” row to the baseline model and to specify which metrics are improved by higher or lower values?


# Suggestions for Additional Experiments

1. Sparsity Correlation: To better ground your sparsity metric, could you provide analyses connecting it to simpler graph perturbation statistics? For instance, how does the meta-path sparsity of an explanation correlate with the total number of unique edges removed or the average change in node degree across the graph? This would help validate the claim of achieving "fine-grained" explanations.
2. Fidelity-Sparsity Trade-off: The trade-off between explanation quality (fidelity) and cost (sparsity) is central to your work. Could you provide AUROC curves or correlation plots for fidelity versus sparsity? This would offer a more standardized and quantitative way to compare your method against baselines than the current plots in Figure 4.
3. Qualitative Case Study: To address the lack of interpretive depth, could you include a qualitative case study? For example, for the most important meta-path found (e.g., A-P-V-P-A), show a concrete subgraph where removing its instances flips a node's prediction and provide a domain-specific reason why this occurs.
4. Model Robustness: The experiments are performed on HGT. How robust are the identified important meta-paths to the choice of the underlying HGNN model? Showing that similar meta-paths are identified as important using a different architecture (e.g., HAN) would significantly strengthen the claim that your method uncovers fundamental data patterns rather than model-specific artifacts.

# Minor comments

In Fig 3, there appears to be a typo in the label of the third figure. Furthermore, there are two em dashes in the first paragraph, probably mistyped between two dashes.

### Soundness
2

### Presentation
3

### Contribution
2
