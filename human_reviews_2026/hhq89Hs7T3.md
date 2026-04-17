# VCWorld: A Biological World Model for Virtual Cell Simulation

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Virtual cell modeling aims to predict cellular responses to perturbations. Existing virtual cell models rely heavily on large-scale single-cell datasets, learning explicit mappings between gene expression and perturbations. Although recent models attempt to incorporate multi-source biological information, their generalization remains constrained by data quality, coverage, and batch effects. More critically, these models often function as black boxes, offering predictions without interpretability or consistency with biological principles, which undermines their credibility in scientific research. To address these challenges, we present VCWorld, a cell-level white-box simulator that integrates structured biological knowledge with the iterative reasoning capabilities of large language models to instantiate a biological world model. VCWorld operates in a data-efficient manner to reproduce perturbation-induced signaling cascades and generates interpretable, stepwise predictions alongside explicit mechanistic hypotheses.  In drug perturbation benchmarks, VCWorld achieves state-of-the-art predictive performance, and the inferred mechanistic pathways are consistent with publicly available biological evidence. Our code is publicly available at https://anonymous.4open.science/r/VCWorld-B970.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
VCWorld is a framework for simulating cellular perturbation responses, integrating a structured biological knowledge graph with the reasoning of Large Language Models (LLMs). Using a Chain-of-Thought (CoT) process, it generates step-by-step, interpretable predictions. The paper also introduces GeneTAK, a benchmark curated from the Tahoe-100M dataset, and claims state-of-the-art, interpretable performance.

### Strengths
The design, which combines a knowledge graph with LLM reasoning, seems a novel approach that improves LLM performance on perturbation prediction.

### Weaknesses
- STATE has a particularly low performance, which contradicts the reported performance in [1]. The reason remains unclear and can arise from suboptimal implementation or insufficient training data. This should be better clarified in the work.
- The contribution is unclear. GeneTAK should not be emphasized as a contribution since it is just reorganizing an existing database. Current claims exaggerate the contribution of the work.
- It is unclear how the results compare to well-established benchmarks in [1-2].
- Importantly, additional results with modern reasoning language models should be presented, such as the results with more widely used LLMs. From the current results, the real usefulness of VCWorld remains unclear; maybe simply using the latest generation of LLMs will remove the need of the knowledge base.

[1] Adduri, Abhinav K., et al. "Predicting cellular responses to perturbation across diverse contexts with State." bioRxiv(2025): 2025-06.
[2] Ahlmann-Eltze, Constantin, Wolfgang Huber, and Simon Anders. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." Nature Methods (2025): 1-5.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VCWorld, a system for predicting gene expression changes following drug perturbations in cells. The authors reformulate the traditional cell-level regression problem into a gene-centric binary classification framework addressing two tasks: differential expression detection (DE) and directional change prediction (DIR). VCWorld integrates a biological knowledge graph built from seven databases with LLM-based retrieval and chain-of-thought (CoT) reasoning using Gemini 2.5 Flash to produce mechanistic explanations. A new benchmark, GeneTAK, derived from Tahoe-100M, includes five cell lines and 348 drug perturbations with a 30:70 train-test split by perturbation. VCWorld achieves an average accuracy of 0.68 on both tasks, outperforming existing deep learning baselines: STATE (0.30–0.51), scVI (0.42–0.61), and CPA (0.23–0.44).

The experimental methodology shows both strengths and weaknesses. Positively, the work uses appropriate statistical testing (Wilcoxon signed-rank), clear dataset construction, extensive ablations, and transparent reporting. However, several issues limit validity. The term “world model” is used incorrectly; in established RL literature (Ha & Schmidhuber 2018; Hafner et al. 2020), world models learn temporal dynamics, which VCWorld does not. Key baselines are missing: no graph neural networks (GCN, GAT, GraphSAGE) on the same knowledge graph, no classical ML using graph features, no simpler retrieval pipeline without LLM reasoning. Consequently, it is unclear whether performance gains stem from graph structure, LLM reasoning, or both. While the ablations are informative, the large performance gap between Gemini 2.5 Flash and Llama3-8B (0.68 to 0.37 DE accuracy) without cost or compute analysis raises reproducibility concerns. Finally, the “white-box” interpretability claim is unvalidated; the paper includes no quantitative audit verifying that reasoning traces correspond to real biological mechanisms.

The paper is well-organized and clearly written, with strong visual presentation (pipeline diagram and tables). The GeneTAK dataset and labeling procedures are described in adequate detail, and the appendices provide useful prompt examples. Nonetheless, the “world model” framing is misleading and should be replaced with a more accurate term such as knowledge-grounded retrieval-augmented classification system. The novelty is somewhat overstated: VCWorld is essentially retrieval-augmented generation (RAG) applied to biological perturbation prediction, combining known components (knowledge graph integration, hybrid retrieval, CoT reasoning) rather than introducing a new algorithm. Discussion of existing RAG literature could be expanded, and the paper should include clearer descriptions of knowledge-graph maintenance, conflict resolution, and update policies.

Empirically, the paper contributes valuable resources and results; methodologically, it is modest. Strengths include the GeneTAK benchmark, the gene-centric reformulation improving data efficiency, strong empirical performance demonstrating that knowledge-grounded reasoning outperforms purely data-driven models, and interpretable natural-language reasoning outputs. However, the architecture primarily reuses established components: graph construction, hybrid semantic-structural retrieval, contrastive case retrieval, and CoT prompting (Wei et al. 2022). There are no fundamentally new algorithms or theoretical insights. The system’s dependence on proprietary Gemini 2.5 Flash, with much weaker results for Llama3-8B (0.37 DE, 0.56 DIR) limits reproducibility and accessibility. Without graph-based or classical ML baselines, it remains unclear whether LLM reasoning is necessary or whether the knowledge graph alone drives performance improvements.

### Strengths
1.	Addresses an important problem in computational biology with strong empirical performance: VCWorld reaches 0.68 accuracy on both DE and DIR, while existing models perform near chance on DIR.

2.	GeneTAK provides a useful community benchmark with rigorous construction and consistent evaluation protocols.

3.	Gene-centric formulation effectively mitigates data sparsity and improves interpretability.

4.	Provides mechanistic textual explanations alongside predictions, improving transparency over black-box deep models.

5.	Thorough ablation studies demonstrate component importance: removing BioContext drops accuracy to ~0.51, removing CoT to ~0.59, and replacing Gemini 2.5 with Llama3-8B to ~0.37 (DE).

6.	Integrates structured biological knowledge from seven authoritative databases in a coherent, data-efficient system.

### Weaknesses
1.	VCWorld does not learn temporal dynamics or simulate cellular trajectories and this misframing creates false expectations about the system's capabilities. Please revise throughout.

2.	Missing critical baselines: no GNNs, traditional ML with graph features, or retrieval-only ablations.

3.	Overstated methodological novelty: essentially standard RAG adapted to a biological context.

4.	Interpretability claims unvalidated: no systematic check that explanations align with known biology.

5.	Missing analyses: computational cost, statistical significance, error analysis by perturbation type, and retrieval-parameter sensitivity.

6.	Unusual 30:70 train-test split is only lightly justified.

7.	Heavy dependence on Gemini 2.5 Flash without open alternatives demonstrated limits reproducibility and accessibility for most researchers.

### Questions
1.	Can you provide GNN baselines (GCN, GAT, GraphSAGE) using your knowledge graph?
2.	How does performance compare when using smaller fine-tuned models?
3.	Have you validated mechanistic reasoning against literature, what fraction of explanations are biologically correct?
4.	What are the computational and API costs per query relative to baselines?
5.	Can you ablate the LLM entirely (graph-based k-NN or scoring) to test the necessity of language reasoning?
6.	How does accuracy vary with retrieval-set size or when retrieving only positive/negative examples?
7.	Provide failure analysis by perturbation and gene coverage; how are out-of-graph entities handled?
8.	Replace the “world model” terminology with a more precise descriptor. Major issue which needs to be addressed for acceptance.
This paper offers solid empirical advances, especially the GeneTAK benchmark and strong results showing knowledge-grounded reasoning can outperform data-hungry deep learning. However, several weaknesses must be addressed: misleading terminology, incomplete baselines (no GNNs or retrieval-only models), overstated novelty, lack of cost analysis, and unvalidated interpretability claims. With revisions adding graph-based and classical baselines, testing open LLMs, validating reasoning quality, reporting costs, and reframing the contribution as a knowledge-grounded retrieval-augmented classifier, the work would merit acceptance.
I will consider raising it to 5 if authors revise the misleading world model framing and perform GNN baseline tests. I would further raise it to 6 (accept) if authors perform cost analysis and address interpretability claims.

### Soundness
2

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
3

### Summary
This paper introduces VCWorld, a novel computational framework for predicting cellular responses to drug perturbations. First, VCWorld reframes the prediction task, moving from a high-dimensional regression problem (predicting exact gene levels) to a gene-level binary classification task. To generate a prediction, the model retrieves relevant information from the constructed KG, along with relevant experiments from its training data, and then feeds the information to the LLM, which generates a step-by-step CoT to support its final answer.

### Strengths
1. The idea of generating a CoT rationale to provides a human-readable, verifiable hypothesis for the model prediction is original;
2. The paper introduces GeneTAK, a novel benchmark that simplifies the prediction proble;
3. By leveraging a pre-existing KG, it can reason about new drugs or genes as long as they are present in its knowledge base.

### Weaknesses
1. The gene-centric, one-at-a-time prediction model raises scalability concerns. This formulation also ignores the cell's pre-perturbation state and treats genes as independent entities, which is not biologically accurate.
2. The KG's construction is described at a high level. It is not explained if the resulting graph consists of a single connected network or many disconnected pieces, which impacts the "graph-based structural similarity" metric.
3. The model works by retrieving similar past experiments to guide its reasoning. It is not explained why this was this chosen over a more direct Graph-RAG approach that would retrieve causal paths directly from the KG to explain the (drug, gene) relationship.
4. Finally, the paper presents its generated reasoning (the CoT) as evidence of interpretability. However, a generated explanation is not a guaranteed, faithful transcript of the model's actual internal process. Therefore, **reasoning and interpretability should not be treated as interchangeable**.

### Questions
1. I think state does not use only 5 cell lines in tahoe, but it is trained instead on entire 50. Please clarify the setup to ensure fairness of comparison.

2.The model uses a hybrid similarity (part text, part structure) to find similar experiments. Why is this complex metric necessary, and what is its impact compared to a simpler, purely semantic retrieval?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles virtual cell modeling: predicting how single cells change gene expression under perturbations (primarily small‑molecule drugs). The authors argue that current end‑to‑end neural approaches are data‑hungry, generalize poorly to unseen perturbations, and provide limited mechanistic interpretability. CWorld is positioned as a white‑box “biological world model” that integrates (i) a curated, heterogeneous biological knowledge graph (KG) built from sources such as PubChem, DrugBank, UniProt, GO, Reactome, STRING, and CORUM; (ii) LLM‑generated, textual node features from local KG neighborhoods; (iii) a hybrid retrieval scheme that scores semantic similarity (cosine on LLM descriptions) and structural similarity (KG path‑based) to construct analogue/contrast evidence sets; and (iv) Chain‑of‑Thought (CoT) prompting to synthesize a stepwise explanation and a binary prediction (DE: differentially expressed or not; DIR: up vs down for DE genes).

### Strengths
1. The paper is well‑motivated by the need for mechanistic, inspectable predictions. The pipeline provides explicit reasoning steps and an evidence set (analogue vs contrast) that biologists can audit

2. Gene‑centric reformulation. Moving from whole‑profile regression to (c,p,g) classification reduces sparsity and makes evaluation at the gene level more straightforward. The label generation process is transparent.

3. Combining LLM‑semantic similarity with KG path‑based similarity is sensible and well aligned with biological priors.

4. Quantitative improvements beyond accuracy. The analysis of precision/recall/F1 and predicted DEG counts shows that VCWorld is less over‑ or under‑confident than generative baselines that either inflate or deflate DEGs.

5. The rule‑based verbalization is a thoughtful way to reduce hallucinations and standardize facts presented to the LLM.

### Weaknesses
1. Converting continuous predictions (STATE/scVI/CPA) to DE/DIR via Wilcoxon on predicted profiles is one way to compare, but it can bias against models trained/optimized for regression‑style metrics. Please add threshold‑free metrics (per‑gene AUROC/AP, balanced accuracy) and calibration to avoid dependence on one downstream test. (Table 1/2 focus on Accuracy and F1 for DE only; DIR lacks P/R/F1.)  

2. Reproducibility and dependence on a proprietary LLM. The core gains hinge on Gemini 2.5‑Flash; the Llama‑3 variant performs much worse. There is no open‑weights replication (e.g., Qwen, DeepSeek‑R1) to show that the method (KG+retrieval+CoT) is the driver rather than the closed model. Also, hyperparameters for retrieval (α, k_a, k_c), prompt templates, temperature/seeds, and cost/latency are not fully specified. 

3. Accuracy can be misleading given heavy class imbalance (Table 4); per‑drug/per‑gene AUROC, AUPRC, macro/micro averages, and confusion‑matrix analyses would strengthen claims. DIR evaluation is especially thin beyond Accuracy.  

4. Retrieval is said to use only the training set, but the analogue/contrast selection and KG‑based similarity could still exploit signals close to test perturbations if drug families or near‑duplicates exist; an explicit “zero‑shot by drug” and “zero‑shot by target/pathway” protocol would clarify generalization.

### Questions
1. What is the exact structural similarity function and neighborhood depth for KG paths? How sensitive are results to these choices?  

2. For DIR labels, what thresholds define up vs down and how are ties handled? Please report DIR P/R/F1.

### Soundness
3

### Presentation
3

### Contribution
3
