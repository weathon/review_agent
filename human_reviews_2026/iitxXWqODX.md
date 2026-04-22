# ALMEA: Active Learning-Enhanced Multimodal Entity Alignment with Semantic Modality Imputation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Multimodal knowledge graphs (MMKGs) offer enriched knowledge representation by integrating structural, visual, and textual information from heterogeneous sources. However, existing multimodal entity alignment (MMEA) approaches face significant challenges due to missing modalities and semantic inconsistencies across sources. These limitations compromise alignment robustness, especially in low-resource scenarios with limited seed pairs (i.e., manually annotated aligned entities as supervision).

To bridge the gap, we propose **Active Learning for Multimodal Entity Alignment with Semantic Imputation (ALMEA)**, a MMEA framework that integrates semantic calibration and active learning to improve alignment. Specifically, ALMEA synthesizes embeddings for missing modalities and refines semantic representations to address inconsistencies across MMKGs. This approach iteratively selects optimal candidate pairs within the learnable budget through active learning strategies, thereby acquiring richer modal information in low-resource scenarios.

On the benchmark MMKG dataset, experimental results indicate that ALMEA consistently outperforms state-of-the-art baseline models under the low-resource scenario, achieving average improvements of **5.16% in Mean Reciprocal Rank (MRR)** and **5.57% in Hits at Top-1 (Hits@1)**.

Our anonymized code is available at [github.com/RTX4090123/ALMEA](https://github.com/RTX4090123/ALMEA).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ALMEA, a novel framework for Multimodal Entity Alignment under low-resource settings. ALMEA integrates semantic imputation of missing modalities with active learning to improve alignment robustness. The authors evaluate ALMEA on two datasets (FB15K-DB15K and FB15K-YAGO15K) under varying supervision budgets, and the results show consistent improvements over state-of-the-art baselines.

### Strengths
This paper addresses two critical challenges in MMEA, missing modalities and limited seed alignments, both of which are common in real-world KGs but often overlooked in prior work.

The experiments cover multiple settings, including ablation studies, statistical significance tests, and qualitative case analyses.

The modular structure (LSL, LSC, ACS) enhances interpretability and facilitates component-wise analysis.

### Weaknesses
Leveraging VAE to generate the missing modalities has already been explored by GEEA and UMAEA [1]. The latter also proposes a new multi-modal OpenEA benchmark. The authors currently evaluate ALMEA only on two single-lingual datasets. It would be better to validate the effectiveness of the proposed method across multiple benchmarks.

This paper uses active learning for annotation, which is analogous to the bootstrapping or iterative algorithms used in existing EA and MMEA methods. However, the authors only present the non-iterative results of baselines. The performance advantage of the proposed method isn't that large.

The baselines (e.g., MEAFormer) are marked with "*" indicating reproduction. However, their results are significantly lower than those in their original paper. This is perhaps because the authors use a constant embedding dimension setting for all methods, but why?

The writing and presentation also need improvement to meet the standard of ICLR. For example, Figure 2 introduces too many terms that are not discussed (as well as Algorithm 1), and tables contain several typos.

[1] Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment, ISWC 2023.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
1

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
This paper proposes a novel framework called ALMEA for MMEA, which addresses the problem of aligning entities across MMKGs with missing modalities and semantic inconsistencies. The proposed approach integrates semantic calibration and active learning to improve alignment in low-resource scenarios, especially where manually annotated seed pairs are scarce. The ALMEA framework includes three core modules: Latent Semantic Learning, Latent Semantic Calibration, and Active Candidate Selection. Experimental results show that ALMEA outperforms existing baseline methods, demonstrating an improvement in alignment accuracy in various low-resource scenarios.

### Strengths
1. LSL effectively synthesizes embeddings for missing modalities, which ensures that the missing modality information can still contribute to alignment tasks.
2. The use of active learning to select the most informative entity pairs for annotation is a key strength, helping to alleviate the low-resource challenge by iteratively improving the alignment model with minimal manual intervention.
3. The paper provides strong experimental results across two benchmark datasets. ALMEA consistently outperforms the state-of-the-art baselines, especially in low-resource settings.

### Weaknesses
1. About novelty. One concern is that active learning has been used for weakly supervised EA, which is your mentioned low-resource scenarios, for a long time. It seems ALMEA is similar to these active learning-based methods [1,2]. Another concern is that some methods have similar performance to ALMEA in the same settings, like DESAlign [3], which is also designed for tackling the semantic consistency. So why don`t you compare and analyze them? Or maybe there are some other different settings I have ignored.
2. While the paper provides strong qualitative results, there is limited statistical validation to support the claim that ALMEA’s improvements are significant and not due to random variation. Error bars or significance tests like paired t-tests could provide more confidence in the reported improvements.
3. While the paper claims that global weighting is more robust, a more detailed analysis of how these different weighting strategies perform across various datasets would be beneficial. A more detailed comparison of the global weighting vs. dynamic weighting could further clarify the advantages of ALMEA's approach in different settings.


[1].Berrendorf, Max, Evgeniy Faerman, and Volker Tresp. "Active learning for entity alignment." European Conference on Information Retrieval. Cham: Springer International Publishing, 2021.

[2].Liu, Bing, et al. "ActiveEA: Active Learning for Neural Entity Alignment." 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021. Association for Computational Linguistics (ACL), 2021.

[3].Wang, Yuanyi, et al. "Towards semantic consistency: Dirichlet energy driven robust multi-modal entity alignment." 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 2024.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

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
This paper proposes a new MMEA framework called ALMEA to employ active learning for maintaining semantic consistency across different KGs and addressing the low-resource scenario. ALMEA consists of three designs: Latent Semantic Learning (LSL), atent Semantic Calibration (LSC), and Active Candidate Selection (ACS). Experiments are conducted on FB15K-DB15K and FB15K-YAGO15K.

### Strengths
- The active learning framework is firstly introduced in the MMEA field. Therefore, the design of the overall framework is novel.
- Low-resource scenario is an important topic for MMEA research.

### Weaknesses
- The citation format in this paper is wrong, which should be revised in the rebuttal. \cite --> \citep
- The paper shows that ALMEA has lower performance gain when the data is sufficient and it mainly works for low-resource scenario.
- The unsupervised MMEA setting is not explored in the experiments. Besides, the datasets used in the experiments are mainly FB15K-DB15K and FB15K-YAGO15K, which are monolingual (English) datasets. The multilingual datasets are not explored in the main experiments.
- The presemtation of Table 1 can be further optimized to make it more clear.

### Questions
- What about ALMEA's performance under unsupervised MMEA? Can active learning still work? I hope that you can add more experiments on this setting to show whether active learning works for it.
- In the datasets you used, the entity amounts are not significantly different. Can you consider other scenario that the entity amounts of the two KGs are with a significant order-of-magnitude gap?

### Soundness
3

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
This paper proposes ALMEA, a novel framework for Multimodal Entity Alignment (MMEA) that combines semantic imputation and active learning to improve robustness under missing-modality and low-resource conditions. The method consists of three main modules:
(1) Latent Semantic Learning (LSL) uses a VAE-based generative model to synthesize embeddings for missing modalities;
(2) Latent Semantic Calibration (LSC) aligns cross-graph semantic distributions via KL divergence to mitigate semantic inconsistency;
(3) Active Candidate Selection (ACS) employs a diversity-regularized subset selection strategy to efficiently choose representative entity pairs under a limited labeling budget.
Experiments on FB15K–DB15K and FB15K–YAGO15K show consistent improvements over state-of-the-art baselines (up to +5.16% MRR and +5.57% Hits@1), especially in low-resource scenarios.

### Strengths
1. Novel integration of semantic imputation and active learning for MMEA, addressing both missing modalities and sparse supervision.

2. Comprehensive evaluation on two benchmark MMKG datasets with multiple baselines, demonstrating consistent gains in both low- and high-resource settings. And Clear modular design (LSL, LSC, ACS) that enables ablation and interpretability.

3. Strong robustness analysis, showing resilience to missing modalities and diverse candidate selection strategies.

### Weaknesses
1. The connection between the active learning optimization and overall alignment objective lacks formal analysis or proof of convergence.

2. Training involves multiple components (VAE, calibration, optimization with ADMM), which may increase computational cost and make real-world deployment difficult.

2. While quantitative results are strong, more case studies or visualization of latent semantics could better illustrate the improvements brought by LSC and ACS.

### Questions
1. Could the active learning module be combined with uncertainty-based criteria (e.g., entropy or margin sampling) to further improve sample efficiency?

2. What is the computational overhead compared to MEAformer or SimDiff, and how does ALMEA scale with large-scale MMKGs?

### Soundness
3

### Presentation
3

### Contribution
3
