# Topology over biology: network representation improves multi-omics models without need for prior knowledge

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Cancer is a heterogeneous and complex disease with substantial variation in patient outcomes. Multi-omics data (including mRNA expression, DNA methylation and micro-RNA expression) capture transcriptional and post-transcriptional regulation of gene expression within the tumor microenvironment, with the potential to reveal mechanisms responsible for different patient outcomes. However, multi-omics data are complex and high dimensional, and extracting meaningful features through machine learning is a challenging task. Current SOTA techniques involve GNNs based on correlation networks built using omics data, and more recent models introduce improvements by augmenting these correlation networks with known biological interactions and pathways. However, this approach relies on the experimental characterization of biological interactions, which requires significant resources. In this work, we take a different approach by enhancing the representation of the correlation networks using topological tools: the Mapper algorithm for pooling nodes, and topological deep learning to represent higher order interactions. Our novel biology-agnostic models M-SAN and M-HGAT outperform both the naive correlation network approach, and models augmented with prior knowledge, in survival prediction across six cancer types (breast cancer, colon cancer, kidney cancer, melanoma, lung cancer and ovarian cancer) with sample sizes between 149 and 333. Additionally, by examining the most important feature interactions within our models, we find that they have learned gene interactions corresponding to biological processes relevant to cancer proliferation and metastasis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes using topological data analysis tools (Mapper algorithm and topological deep learning) to improve graph neural network-based multi-omics models for cancer survival prediction. The authors introduce two novel architectures: M-SAN (Mapper with Simplicial Attention Network) and M-HGAT (Mapper with Hypergraph Attention Network). The key claim is that these topology-based approaches outperform both baseline correlation networks and models augmented with biological prior knowledge (protein-protein interactions, gene pathways) across six cancer types. The paper demonstrates that their models learn biologically meaningful gene interactions despite having no access to prior biological knowledge.

### Strengths
1. **Comprehensive evaluation**: Testing across 6 cancer types with multiple architectures shows thoroughness.

2. **Practical approach**: Avoiding dependence on curated biological networks is pragmatically valuable given their incompleteness and tissue-specificity issues.

3. **Reproducibility effort**: Code availability promise and detailed hyperparameters support reproducibility.

4. **Interesting negative results**: The analysis of when Mapper helps (Section 4.2, Tables 9-10) provides useful insights about method applicability.

5. **Clear motivation**: The paper articulates the limitations of current approaches well.

### Weaknesses
1. **Statistical validation**: 
   - Only 5 runs per experiment is insufficient
   - No significance testing between methods
   - Standard errors often overlap between methods
   - No correction for multiple comparisons across 6 cancer types

2. **Experimental design flaws**:
   - Different batch sizes between patient similarity (no batching) and biological interaction models (batched) confounds architectural comparisons
   - Vastly different parameter counts (Table 8: 5M to 414M) make comparison unfair
   - Different training times and convergence criteria may favor different models

3. **Baseline issues**:
   - GGNN comparison lacks experimental parity
   - No comparison with other recent multi-omics methods (only cite Leng et al. 2022 comparison but don't include those methods)
   - Missing comparison with simple baselines (e.g., concatenated features + MLP)

4. **Method clarity**:
   - Novel SAN message passing scheme needs clearer mathematical formulation
   - "Gradient of feature matrix" initialization is undefined
   - Predictive Mapper details missing
   - How are outliers in test set handled exactly?

5. **Biological validation concerns**:
   - Single simplex/hyperedge analyzed per cancer (no robustness check)
   - GO enrichment with Bonferroni correction on single test is underpowered
   - Generic cancer processes found, not novel biology
   - No validation that learned interactions are more meaningful than random feature sets of same size

6. **Overfitting risks**:
   - Small sample sizes (149-333)
   - High-dimensional input (up to 18,790 features)
   - Extensive hyperparameter search
   - Small validation sets for hyperparameter selection
   - Many degrees of freedom (network construction, architecture choices)

7. **Generalization claims unsupported**:
   - Only cancer survival prediction tested
   - All data from single source (TCGA)
   - Claims about other domains (line 60) are speculative

### Questions
1. What are the actual p-values from paired statistical tests comparing your methods to baselines within each cancer type?

2. Can you provide ablation studies separating Mapper contribution from TDL contribution?

3. Why do M-SAN and M-HGAT sometimes perform worse than M-GNN (e.g., Melanoma, Lung in Table 2)? What does this tell us about when TDL helps?

4. How sensitive are results to:
   - Number of PCA components in filter function?
   - DBSCAN parameters?
   - Distance correlation threshold selection?
   - Number of Mapper intervals?

5. For biological validation:
   - What GO enrichment do you get for randomly selected feature sets of the same size?
   - Are the identified simplices/hyperedges consistent across the 5 runs?
   - How many simplices/hyperedges show significant enrichment, not just the top one?

6. Can you clarify the "gradient of feature matrix" initialization mathematically?

7. Why are parameter counts so different across models (Table 8)? Shouldn't fair comparison use similar model capacities?

8. Have you tested on any non-cancer datasets or survival prediction tasks outside TCGA?

9. What is the actual implementation of predictive Mapper for test set patients, especially outlier handling?

10. Can you provide learning curves showing validation performance during hyperparameter tuning to assess overfitting?

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
3

### Summary
In this study, the authors proposed to make use Mapper algorithms and topological deep learning (TDL) to improve the sample classification/prediction using multi-omics datasets. The evaluation results showed the improved performance on cancer patient survival dataset.

### Strengths
Make use Mapper algorithms and topological deep learning (TDL) to improve the sample classification/prediction using multi-omics datasets. 
The evaluation results showed improved performance on cancer patient survival dataset.

### Weaknesses
In multi-omics data analysis, the prediction of cancer patients' survival data is not much meaningful (also need to consider the confoundation factors, like age, gender, stage of individual patients). 
Rather, the discovery of novel multi-omic features/biomarkers that can explain the mechanisms of patients' survial or drug response are more important. However, the evaluation of the important targets and mechanisms (espeically the multi-omic interactions are not well presented).

### Questions
Put more efforts on the identification of multi-omic signaling interactions that are correlated or associated with patients' survival and drug responses are important.
Also considering the confounding factors in the prediction model in addition to multi-omics data.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This manuscript addresses the task of multi-omics modeling, specifically focusing on leveraging GNN-based approaches to capture correlations in networks derived from multiple omics layers. Unlike prior methods that incorporate external knowledge such as protein-protein interaction networks or gene pathway information, this work aims to enhance modeling directly from the topological structure of the data. The proposed models, M-SAN and M-HGAT, demonstrate improved performance over selected baselines on the cancer survival prediction task.

### Strengths
- The manuscript is well-written and well-argued. It clearly explains the limitations of existing multi-omics GNN models and proposes the motivation to use topological tools to improve network representations.

- This paper introduces a framework that does not require prior knowledge and utilizes the Mapper algorithm and topological deep learning (including a novel simplex attention network) to capture high-order interactions and reduce oversmoothing in related networks. Technically speaking, this makes sense.

- This paper evaluates the M-SAN and M-HGAT models in survival prediction for six cancer types, demonstrating superior performance compared to state-of-the-art GNN models and knowledge augmentation methods. Furthermore, the analysis reveals biologically significant feature interactions.

### Weaknesses
- This paper has limited innovation because it focuses on specific tasks in the multi-omics field, with the main goal of improving graph neural network (GNN) models using the Mapper algorithm and topological deep learning. While technically interesting, this approach is incremental and may face scalability challenges on large-scale datasets, and its applicability in other domains or graph-based tasks has not been fully demonstrated.

- The experimental setup is not fully convincing, as only GGNN (Zhu et al., 2023) is used as a baseline. Including additional baselines, particularly non-GNN methods or other multi-omics survival models, would provide a more comprehensive evaluation and strengthen the claims of improvement.

### Questions
Please see the weaknesses section for further discussion.

### Soundness
2

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
This paper proposes two novel GNN-based architectures, Mapper SAN (M-SAN) and Mapper HyperGAT (M-HGAT), that enhance cancer survival prediction from multi-omics data by leveraging topological tools rather than biological priors. The authors apply the Mapper algorithm for unsupervised node pooling to address oversmoothing in correlation networks, and introduce topological deep learning architectures (simplicial attention networks and hypergraph attention networks) to capture higher-order interactions. In the evaluation presented, the proposed methods consistently outperform both naive correlation network approaches and models augmented with protein-protein interaction or gene pathway information.

### Strengths
1. The application of topological data analysis (Mapper algorithm) and topological deep learning to address fundamental issues in GNN-based multi-omics modeling is creative and well-justified. The paper clearly articulates the problems of oversmoothing due to dense connectivity and the limitation of pairwise interactions, and proposes targeted solutions that directly address these issues rather than applying generic improvements.
2. The proposed methods outperform GGNN (which leverages explicit biological knowledge including protein-protein interactions and pathway data) across multiple cancer types and metrics. The investigation of when Mapper helps versus when it doesn't (distinguishing "Mapper group" vs. "Benchmark group" cancers) and the mechanistic analysis through GNNExplainer application demonstrates scientific rigor.
3. The post-hoc analysis using GNNExplainer to validate learned interactions is helpful.

### Weaknesses
1. With sample sizes ranging from 149-333 and modest performance improvements in several cases, the statistical significance of reported gains is questionable. Although error bars are provided, no formal significance testing is reported.
2. The Mapper algorithm and TDL models introduce numerous hyperparameters (filter intervals: 5-20, overlap fractions: 0.1-0.0001, distance correlation thresholds, max_rank values), each tuned separately per cancer type. Table 7 shows dramatic variation across cancers (e.g., mRNA intervals from 0.1 to 6.0), suggesting either the method is highly sensitive to these choices or the tuning process introduces implicit cancer-specific adaptation that could inflate performance estimates. The paper lacks ablation studies on sensitivity to these hyperparameters or justification for the tuning strategy. This could lead to overfitting to the specific cancer types in the dataset.

### Questions
1. Given the small sample sizes and multiple comparisons across six cancer types, could you provide formally tested statistical significance (with appropriate correction for multiple testing) for the key performance differences? 
2. The initialization of higher-rank features as "gradients" and the behavior of the predictive Mapper algorithm for outliers need clarification. How exactly are these implemented? Does outlier handling in the test set affect performance?
3. How do your methods compare to simpler alternatives like dimensionality reduction followed by standard classifiers?
4. Can you provide negative control analyses (e.g., enrichment p-values for randomly selected gene sets from your learned features)? Which identified interactions are truly novel versus confirmatory of known biology? Have you validated any predicted interactions experimentally or against curated databases? (searching in the databases will be a quick thing, and provide a strong evidence)

### Soundness
3

### Presentation
3

### Contribution
2
