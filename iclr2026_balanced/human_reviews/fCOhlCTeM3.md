## Human Reviewer 1

### Summary
**B-XAIC** is a benchmark for evaluating **GNN explainability** on real chemical datasets. It provides ~50k small molecules across **7 tasks** with **ground-truth node/edge rationales**, enabling both *null-explanation* (no subgraph should be highlighted) and *subgraph-explanation* (specific atoms/bonds matter) settings.

**What’s inside**
- **Data & Tasks:** Real-world molecule graphs with diverse labels (e.g., functional groups, PAINS, ring counting).
- **Ground Truth:** Node/edge-level rationales for each task.
- **Models & Explainers:** Benchmarks multiple GNNs (GCN/GAT/GIN/Proto-based) and a wide range of explainers (gradient-, mask-, perturbation-based).

**Evaluation Protocol**
- **NE (Null Explanation):** Measures uniformity (no spurious highlights) using dispersion-based statistics.
- **SE (Subgraph Explanation):** Measures how well explanations recover the true subgraph using ranking metrics (e.g., AUROC).

**Key Findings**
- Gradient-style explainers often excel at subgraph recovery (SE) but can produce spurious highlights on NE tasks.
- Mask/perturbation methods tend to be more conservative (better NE uniformity), with mixed results on SE.
- GIN models achieve strong predictive performance, providing a stress-test for explainer fidelity.

### Strengths
## Strengths

1) **Real-world scale with ground-truth rationales**
   - ~50k molecules across 7 chemically meaningful tasks, each with node/edge-level rationales. This moves beyond synthetic motifs and enables fine-grained, objective scoring of explanations.

2) **Two-regime evaluation that targets common XAI failure modes**
   - Separates **Null Explanations (NE)** (nothing should be highlighted) from **Subgraph Explanations (SE)** (specific atoms/bonds matter), encouraging both *specificity* (SE) and *restraint* (NE).

3) **Broad, standardized explainer sweep**
   - Benchmarks gradient-, mask-, and perturbation-based methods on multiple backbones (GCN/GAT/GIN/prototype variants), giving practitioners apples-to-apples comparisons.

4) **Chemistry-aligned task design**
   - Tasks span functional groups, PAINS filters, and ring counting—grounded in domain knowledge—so explanation scores reflect chemically meaningful reasoning rather than dataset quirks.

5) **Open, reproducible setup**
   - Public dataset and code with a clear protocol make it straightforward to add new explainers, models, or tasks and track progress consistently.

### Weaknesses
1) **F1 definition is ambiguous**
   - The paper reports “F1” but does not specify **micro vs. macro** (or weighted) averaging.
   - **Why it matters:** Micro-F1 can be dominated by frequent classes, while Macro-F1 reflects per-class balance; conclusions about model/explainer performance can flip depending on this choice.
   - **Fix:** Report both Micro- and Macro-F1 (plus class-wise F1), or clearly state and justify the chosen averaging scheme.

2) **Missing presentation of extracted subgraphs**
   - The work evaluates explainers but does **not show qualitative subgraph extractions** (e.g., top-k atom/bond rationales overlaid on molecules) in the main text or appendix.
   - **Why it matters:** Visual sanity checks are essential to verify that highlighted structures correspond to chemically meaningful motifs rather than artifacts.
   - **Fix:** Add exemplar visualizations per task: input molecule, ground-truth rationale, and each explainer’s top-k subgraph (node/edge masks), with brief commentary.

3) **Explanation stability not analyzed (only performance over seeds)**
   - While predictive performance is averaged across seeds, there is **no stability analysis of the explanations** themselves.
   - **Why it matters:** Practitioners need to know if an explainer returns **consistent rationales** across random seeds/splits/augmentations.
   - **Fix:** For each explainer, report explanation-stability metrics across seeds: rank correlation (Spearman/Kendall) of node/edge importance; Jaccard/IoU overlap of top-k subgraphs; variance bars on AUROC/AP computed per-seed explanations.

### Questions
Beside the weakness, I want to know why you include the ProtGNN. The performances are not best and it is claimed to be interpretable, then it is somehow strange to evaluate the explanation methods over an interpretable GNN.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper develops a benchmark called B-XAIC (Benchmark for eXplainable Artificial Intelligence in Chemistry) for explainability. The dataset contains 50K small molecules and seven detection tasks: specific atoms, fused bicyclic structures, multiple rings, large rings, and complex patterns as defined by PAINS. For each task, ground truth explanations are categorized as null explanations (NE) meaning that no substructure is more important than others for the specific task or subgraph explanations (SE) meaning that a specific substructure is important for the task.

The paper evaluates different GNN architectures such as GCN, GAT, GIN and adopts the GIN as the base architecture. Then on this architecture, it evaluates different explanation methods for the proposed tasks using the metrics of IQR for NE tasks and AUROC for SE tasks.

### Strengths
Explainability is a difficult problem and comparing different explainability methods is even harder. Benchmarks that may offer an unbiased comparison can be useful.

The benchmark is a useful characterization of the common tasks in the chemical space.

### Weaknesses
There are a number of recent explainability methods that the paper could have included. For example, those in Fig 1 from https://arxiv.org/pdf/2306.01958 or from Fig 1 in https://arxiv.org/pdf/2310.01794. These methods include GSAT, DIR, SubgraphX, VGIB, GIB and others. Also, this work on substructure masking: https://www.nature.com/articles/s41467-023-38192-3.

The paper discusses factual as well as counterfactual explainers. It is unclear if the proposed benchmarks will work for counterfactual explainers.

There are other benchmarks that have been proposed in the literature:
https://www.sciencedirect.com/science/article/pii/S2666389922002604
The paper does not discuss them. A comparative discussion would have been useful.

### Questions
Please take a look at the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
The presented paper aims to address the ongoing evaluation crisis in the domain of explainable AI. The authors identify the crucial need for novel methods to evaluate the quality of explanation methods - including the graph processing domain and the explainability of graph neural networks respectively. In this context, the paper proposes the B-XAIC dataset consisting of 7 molecular property prediction tasks on 50k molecular graphs with known ground truth explanations. A special emphasis is placed on distinguishing between null explanations - cases where no elements of the graph should be contained in an explanation - and subgraph explanations - for which only certain subgraph structures constitute a correct explanations. The authors apply several popular explainers from the xAI literature on the proposed benchmark dataset and find large discrepencies in their performances - especially regarding the null explanations - despite being based on the same underlying and high-performing graph neural network model. With this finding, the paper motivates the need for more robust explanation method and emphasises the need for xAI evaluations.

### Strengths
The distinction between the subgraph explanations and the null explanations adds an interesting nuance to xAI evaluations. The authors correctly point out that sometimes it is equally important to quantify if the correct explanation has been found as it is to make sure that no incorrect explanations are generated if they are not warranted.

 The paper presents a comprehensive empirical evaluation of various common explainers as well as various common graph neural network architectures.

### Weaknesses
- The same general idea was presented 5 years ago by Sanchez-Lengeling in their work on "Evaluating Attribution for Graph Neural Networks". In their work, Sanchez-Lengeling et al. also propose various subgraph detection-based molecular classification tasks with the explicit goal of benchmarking graph explainability methods regarding the known subgraph masks. In addition, they also take the possibility of a regression task into consideration.

- Since this work presents essentially the same schema, it is subject to the same limitations - primarily that only attributional explanation methods are being evaluated. However, the landscape of XAI literature in recent times has seen an increased attention to non-attributional explanation modalities such as concept-based explanations, prototype-based explanations, and counterfactuals. In light of these developments, an important contribution would have been to present an evaluation framework that goes beyond traditional attributional explanations.

- A core pillar of the paper's motivation is calling into question the existing regime of using synthetic datasets for the evaluation due to their usual lack of complexity. However, 6 out of the 7 tasks presented in this dataset are arguably highly trivial. Two tasks are pure node detection tasks (1, 2) for which a graph neural network wouldn't be necessary at all. Another 3 tasks are simple subgraph classification tasks (3, 5, 6), which are equally simple as tasks 1 and 2. With the exclusion of task 4 (PAINS), all remaining tasks essentially are equivalent to commonly used synthetic tasks, with the only difference being that they are defined on molecular graph structures instead of color graphs, for instance. This means that a good performance on those tasks does not generalize to a good performance and actual "real-world" tasks, where explanations are unknown and certainly differ from simple node detection or subgraph classification tasks.

- All presented tasks are "simple", indicated by the extremely high accuracy of the models in Table 2. In most real-world cases, it is relevant to benchmark and quantify explanations even if models only have accuracies of 0.7 or 0.8 (or r2 scores in regression tasks, which are omitted here, see next comment). 

- All the presented tasks are exclusively classification tasks. Especially when talking about the chemistry domain, molecular regression tasks play an important role as well. To that extent, a rigorous evaluation should contain a balance not only of the task complexity but also a mix of classification and regression tasks.

- All presented graph explanation methods (apart from FlowX) are more than 4 years old. Recent self-explaining graph neural networks are not benchmarked or mentioned at all. All presented methods are post-hoc approaches applied to the GIN model (and other backbones in the appendix), even though the introduction claims that "B-XAIC enables a direct and fair comparison of various factual XAI approaches, both post-hoc explainers and inherently self-explainable models."

### Questions
- Regarding the distinction between the null explanations and the subgraph explanations: Does that mean that the to metrics (AUROC, IQR) are only evaluated on the respective subsets of the elements that represent the SE and NE instances within a certain task? Or are both metrics computed across all the elements of each task/dataset?

- It is mentioned that weighted sampling is used to avoid a huge class imbalance. Does that mean the final datasets are perfectly balanced with regard to the classes of all the tasks, or has the imbalance merely been reduced?

### Soundness
3

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper introduces B-XAIC, a new benchmark dataset aimed at evaluating explainability methods for Graph Neural Networks (GNNs) in the molecular domain. The dataset consists of 50,000 real molecules derived from ChEMBL, annotated with seven tasks and corresponding ground-truth explanations at both atom and bond levels. The benchmark distinguishes between “null” and “subgraph” explanations and evaluates various post-hoc explainers (e.g., GNNExplainer, PGExplainer, FlowX, etc.) using AUROC and IQR metrics.

### Strengths
* **Timely contribution:** The paper tackles a real gap in XAI for GNNs by providing a real-world benchmark instead of relying on synthetic datasets. I am actually currently reviewing a work at ICLR 2026 that uses the B-XAIC dataset.
* **Well-structured dataset:** The tasks are clearly motivated, ranging from simple atom detection (e.g., boron, phosphorus) to complex pattern recognition (PAINS, ring counting).
* **Clear presentation**

### Weaknesses
* **Benchmark incompleteness:** The benchmarking omits several self-explainable GNNs that are now standard for comparison, such as PxGNN (Dai et al., 2025), PGIB (Seo et al., 2023), PiGNN (Ragno et al., 2021), GIP (Wang et al., 2024), KerGNNs (Feng et al., 2022), and IMPO (Ragno et al., 2025).  The comparison also lacks newer explanation techniques such as SubGraphX (Yuan et al., 2021), GStarX (Zhang et al., 2022), and EdgeShaper (Mastropietro et al., 2022).
* **Evaluation limitation:** The evaluation only checks alignment with ground-truth substructures but does not verify faithfulness with respect to the model’s decision process. An explanation should first be evaluated for the faithfulness using standard metrics (Fidelity, Inv Fidelity): if a model learns a spurious bias, the correct explanation should reflect that bias. Only after confirming model alignment, comparisons to chemical ground truth are meaningful. This hinders the findings of the paper for instance when the authors claim that "despite near-perfect predictive performance, most explainers fail to recover relevant chemical structures". Let me cite Agarwal et al, 2023: " trained GNN model may only capture one or an entirely different rationale. In such cases, evaluating the explanation output by a state-of-the-art method using the ground-truth explanation is incorrect because the underlying GNN model does not rely on that ground-truth explanation. In addition, even if a unique ground-truth explanation generates the correct class label, the GNN model trained on the data could be a weak predictor using an entirely different rationale for prediction. Therefore, the ground-truth explanation cannot be used to assess post hoc explanations of such models."
* **Incorrect assumptions of null explanations**: The assumption that "if atom B is not present in the graph, then no specific substructure is considered relevant" is factually incorrect. The model could be using a negative reasoning and therefore checking whether all other methods are present instead. Or additionally there could be some atoms that are actually important, for instance the nitrogen, that can accept 3 bonds exactly as the boron, but has different charge/weight.

### Questions
1. How does B-XAIC handle cases where model explanations diverge from ground truth but are internally consistent (i.e., correct with respect to the model but not the chemistry)?
2. Can the authors justify why certain recent explainable GNNs (e.g., KerGNNs, PiGNN, PxGNN) were omitted, given their relevance to interpretability evaluation?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
2

### Confidence
5