# Distilling and Adapting:  A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Multiplex Biological Networks (MBNs), which represent multiple interaction types between entities, are crucial for understanding complex biological systems. Yet, existing methods often inadequately model multiplexity, struggle to integrate structural and sequence information, and face difficulties in zero-shot prediction for unseen entities with no prior neighbourhood information.  To address these limitations, we propose a novel framework for zero-shot interaction prediction in MBNs by leveraging context-aware representation learning and knowledge distillation. Our approach leverages domain-specific foundation models to generate enriched embeddings, introduces a topology-aware graph tokenizer to capture multiplexity and higher-order connectivity, and employs contrastive learning to align embeddings across modalities. A teacher–student distillation strategy further enables robust zero-shot generalization. Experimental results demonstrate that our framework outperforms state-of-the-art methods in interaction prediction for MBNs, providing a powerful tool for exploring various biological interactions and advancing personalized therapeutics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this study, the authors proposed a CAZI-MBN (Context-Aware and Zero-shot Interaction prediction in Multiplex Biological Networks) model framework for multiplex biological interaction prediction. The evaluation results showed improved performance.

### Strengths
The CAZI-MBN (Context-Aware and Zero-shot Interaction prediction in Multiplex Biological Networks) model framework for multiplex biological interaction prediction is kind of new. 
The evaluation results showed improved performance.

### Weaknesses
For the interactome data, there are much more negative data (no interactions). It is unclear how was the negative data selected and evaluated using the proposed model.

### Questions
It is unclear how many negative data was selected for the model evaluation?
For the zero-shot setting, will the performance heavliy based on the density of the available positive controls?
In the datasets, it is unclear how many interactions (in addition to the no. of nodes)

### Soundness
3

### Presentation
3

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
This paper tackles zero-shot interaction prediction in multiplex biological networks by combining domain-specific sequence embeddings with a topology-aware graph tokenizer, a context-aware enhancement module, and teacher–student distillation. Across five curated multiplex benchmarks and both transductive and zero-shot settings, the method reports consistent gains in AUROC/AUPRC over single-graph, multiplex, and domain-specific baselines, with ablations attributing the largest lift to LLM embeddings.

### Strengths
- The authors evaluate on multiple benchmarks spanning different biological domains, suggesting improvements that generalize beyond a single dataset.

- The training/inference workflow is clearly presented and makes the role of each component (UGT, CAE, MoE, KD) easy to follow.

### Weaknesses
- Figure 3’s dataflow appears inconsistent with the algorithm: pooling is depicted in parallel with the graph transformer, whereas the text implies a post-transformer operation.

- Several attention mechanisms are introduced without a clear design rationale; reporting layer-wise attention weights (e.g., across multiplex layers) would help justify the need for attention aggregation.

- Some modules resemble standard variants (e.g., the “node-level context-aware attention” aligns with common attention aggregation); please delineate what is novel vs. adapted.

- All benchmarks are curated by the authors, yet basic dataset statistics and curation decisions (interaction counts by type, inclusion/exclusion filters, handling of homogeneous edges within MBNs) are missing and should be documented.

- For ChEMBL, specify the exact assay criteria/IDs used to build the antibiotic response benchmark; the current size seems very small relative to the database.

- Several baselines may be underpowered: e.g., MLP/XGB should also use the same sequence LLM embeddings to provide a fairer comparison (especially in zero-shot). Also clarify any hyperparameter search to avoid bias.

### Questions
- Please specify the exact model versions/checkpoints for the sequence LLMs (e.g., ESM-2 model size).

- How is edge/relational heterogeneity handled in the graph transformer—via relation-specific parameters, type encodings, or layer separation?

- How are edge perturbations performed?

- Detail the MoE design (experts, gating, placement); is it a Transformer-MoE or a lightweight MLP-expert setup?

- Describe the zero-shot split protocol.

- Where are the MolTrans results referenced as a domain baseline?

- How are bacterial entities embedded (which LLM/preprocessing)?

- Please add error bars to Figure 4 to reflect variability across interaction types.

- Clarify how graph baselines are adapted to zero-shot.

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
This paper proposes CAZI-MBN, a topology-aware teacher–student framework for interaction prediction on multiplex biological networks (MBNs), targeting in particular the zero-shot case where new entities have no observed neighborhood (e.g., new drugs, uncharacterized genes). The teacher model combines (i) domain LLM–based sequence embeddings (ChemBERTa-2, DNABERT-2, ESM-2), (ii) topology embeddings from a Unified Graph Tokenizer over the supra-adjacency, and (iii) a Context-Aware Enhancement (CAE) module with graph transformers and contrastive/consensus objectives. A Mixture-of-Experts head is used for multi-label interaction prediction across relation types. A topology-agnostic student is then distilled to match the teacher’s latent representations and task outputs, so that it can make zero-shot predictions from sequence only. The authors also curate five multiplex benchmarks (DGIdb, ChEMBL, PINNACLE, MetaConserve, TRRUST) to standardize evaluation in this setting.

### Strengths
1. The teacher-student distillation idea is well matched to the zero-shot MBN problem: topology is used only when available, but its signal is transferred to a sequence-only student so that unseen entities can still be scored. 

2. The architecture is thoughtfully composed (LLM sequences + UGT topology + CAE alignment + MoE for multi-label), and the ablations suggest that CAE and LLM features contribute the most.

3. The IBD case study shows that the method can surface biologically plausible candidates, not just improve AUROC.

### Weaknesses
1. The full teacher pipeline is heavy: it requires multiple domain LLMs, SVD/factorization on a supra-adjacency matrix for UGT, and graph transformers within CAE. The paper does not report runtime / GPU hours / UGT rank on the largest MBN, so it is hard to assess scalability to very large networks.

2. On smaller / skewed datasets such as TRRUST, the model improves AUROC but underperforms some baselines on Hamming Score / Subset Accuracy, suggesting that label imbalance, MoE calibration, or the thresholding strategy may affect multi-label metrics. A short analysis would make the empirical section tighter.

3. Several baselines are adapted to the multiplex setting by training a separate classifier per interaction type. This is a reasonable but relatively weak adaptation. Adding at least one heterogeneous GNN (e.g., an R-GCN-style model or a multi-head GAT over relation types) would strengthen the comparison.

### Questions
1. Please report GPU hours, parameter counts (teacher vs. student), UGT rank, and runtime per epoch on the largest dataset.

2. Can you add or report a native multiplex/heterogeneous GNN baseline to show that CAZI-MBN is competitive even against relation-aware models?

3. For datasets where CAZI-MBN lags on HS/SA, how were decision thresholds chosen (global vs. per-label vs. per-dataset)? Would per-label calibration close the gap?

### Soundness
2

### Presentation
2

### Contribution
3
