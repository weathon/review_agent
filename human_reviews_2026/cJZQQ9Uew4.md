# Brain PathoGraph Learning

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Brain graph learning has demonstrated significant achievements in the fields of neuroscience and artificial intelligence. However, existing methods struggle to selectively learn disease-related knowledge, leading to heavy parameters and computational costs. This challenge diminishes their efficiency, as well as limits their practicality for real-world clinical applications. To this end, we propose a lightweight Brain PathoGraph Learning (BrainPoG) model that enables efficient brain graph learning by pathological pattern filtering and pathological feature distillation. Specifically, BrainPoG first contains a filter to extract the pathological pattern formulated by highly disease-relevant subgraphs, achieving graph pruning and lesion localization. A PathoGraph is therefore constructed by dropping less disease-relevant subgraphs from the whole brain graph. Afterwards, a pathological feature refinement module is designed to reduce disease-irrelevant noise features and enhance pathological features of each node in the PathoGraph. BrainPoG can exclusively learn informative disease-related knowledge while avoiding less relevant information, achieving efficient brain graph learning. Extensive experiments on four benchmark datasets demonstrate that BrainPoG exhibits superiority in both model performance and computational efficiency across various brain disease detection tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a lightweight brain graph learning framework, named Brain PathoGraph Learning (BrainPoG), aiming to improve the efficiency and interpretability of brain disease analysis. The authors argue that existing brain graph methods often fail to focus on disease-related patterns, resulting in high computational costs and redundant parameters. To address this, BrainPoG introduces mechanisms for pathological pattern filtering and pathological feature distillation, enabling selective learning of disease-relevant representations. The paper further provides analyses of disease-related regions to support the interpretability of the extracted features.

### Strengths
### **The paper makes a meaningful attempt to extract disease-specific brain features.**
By focusing on identifying brain patterns that are specific to different neurological or psychiatric disorders, the work contributes to improving the interpretability of AI-based brain disease prediction models. This direction is important not only for enhancing diagnostic accuracy but also for advancing our understanding of the underlying neurobiological mechanisms.
### **The introduction of PathoGraph as a disease-specific subgraph representation is conceptually innovative.**
Instead of modeling the whole-brain connectome, the proposed method isolates pathological subgraphs that capture disease-relevant interactions. This approach offers a perspective on selective brain network learning and represents a creative step toward localizing disease-related brain regions in graph-based analysis.
### **The visualization and clinical interpretation of disease-specific brain regions enhance the study’s significance.**
The authors visualize the brain regions identified by their model, providing an intuitive understanding of the disease-specific activation patterns. They further relate these findings to previously reported results in the literature, offering supporting evidence that reinforces the validity and interpretability of the conclusions.

### Weaknesses
### **The motivation is not sufficiently convincing; the necessity of model lightweighting needs re-examination.**
The core motivation of reducing model parameters and computational costs seems weak, as most brain network models are already relatively small and can be easily handled with current computational resources. There is currently little demand for deployment in low-resource environments. To make the motivation more persuasive, the authors are encouraged to clearly define a realistic low-resource or deployment scenario (e.g., edge devices, privacy-preserving federated settings, or multi-site low-resource hospitals) where lightweight modeling is genuinely beneficial.
### **The paper relies on a strong and controversial assumption that brain diseases correspond to specific lesion regions or circuits.**
This assumption is debatable, especially for neurodegenerative and psychiatric disorders, which often show distributed and network-level dysfunction rather than focal lesions. Theoretical justification is weak—the introduction cites only two references (Mahesh & Tasneem, 2014; Le et al., 2025), one lacking impact and the other a preprint, which are insufficient to support such a strong claim. The authors could strengthen this part by:
1. expanding the literature review to distinguish between diseases with localized lesions (e.g., structural or tumor-related) and those with diffuse, network-wide alterations (e.g., schizophrenia, depression), and clarifying BrainPoG’s applicability in both contexts;
2. validating the model separately on datasets representing these two categories to assess robustness to different pathophysiological assumptions; and
3. adding hypothesis tests (e.g., random-label or spatial permutation tests) to verify that the extracted “PathoGraphs” capture meaningful disease patterns beyond random or confounding effects.
### **The originality claim is limited; the idea of subgraph extraction has precedents, and related work is not sufficiently compared.**
The proposed PathoGraph shares conceptual similarities with previous work (e.g., Le et al., 2025) that already introduced disease-specific subgraph extraction, and with hypergraph-based approaches for modeling brain networks (e.g., Multiview hyperedge-aware hypergraph embedding learning for multisite, multiatlas fMRI-based functional connectivity network analysis). The lack of direct comparison with these studies weakens the novelty. The authors should:
1. explicitly clarify the methodological and conceptual differences from prior subgraph or hypergraph-based models;
2. include at least one or two of these methods as baselines for quantitative comparison; and
3. provide theoretical or empirical analysis explaining how BrainPoG improves over prior work (e.g., through architectural simplification, interpretability, or robustness).
### **Serious experimental concerns: potential label leakage and over-claimed results undermine credibility.**
The reported over 90% binary classification accuracy on ABIDE and ADHD datasets is surprisingly high and difficult to believe. From Table 3, it appears that the performance gain primarily comes from the Feature Distillation (FD) module, while PathoGraph itself may even reduce performance in some settings. As described, FD uses the entire dataset and label information, introducing a major label leakage risk that compromises result validity. The authors must carefully recheck their data pipeline and address the following issues:
1. Eliminate leakage: Re-implement FD using only the training set and ensure no test data or labels are accessed during distillation or feature extraction.
2. Clarify data splits: Clearly report how data were divided (train/validation/test), whether site stratification was used, and whether any cross-site evaluation was conducted. Ideally, introduce a completely held-out external test set for unbiased evaluation.
3. Adopt stricter evaluation: Use nested cross-validation or fixed train/test splits with confidence intervals and statistical tests (e.g., bootstrapping or random resampling) to verify the significance of performance gains.
4. Report ablations rigorously: Present results for models without FD, without PathoGraph, and with standard baselines. If FD accounts for most improvements, describe its setup in detail (data used, supervision level, teacher model, loss function) and confirm the benefit persists under strict no-leakage conditions.
5. Validate robustness: Conduct stability tests (different random seeds, sub-sampling) and permutation tests to confirm that the detected regions are not artifacts of data bias or preprocessing.

**Addressing these concerns would substantially strengthen the paper. I will reconsider my overall score if the authors can adequately address the concerns raised above.**

### Questions
1. Why did the authors use different brain atlases across datasets instead of adopting a consistent atlas such as AAL for both ABIDE and ADHD?
2. Given that the Schaefer atlas provides well-defined 7/17 functional subnetworks, why was it not used for subnetwork-based analysis?
3. How exactly does the SVM accept graph or subgraph representations as input, and what feature transformation or embedding process is applied?
4. If disease-specific feature extraction is beneficial, why were pretrained models that learn generalizable representations not included for comparison?
5. Considering that SVM shows limited predictive performance in the results, why was it chosen as the subgraph selection evaluator?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes BrainPoG, a lightweight brain graph learning framework designed to improve computational efficiency and disease-specific interpretability in fMRI-based brain network analysis. The method consists of two core modules: a pathological pattern filter that constructs a PathoGraph by retaining only highly disease-relevant subgraphs using SVM-based patho-scores, and a pathological feature distillation module that removes noise features and enhances discriminative pathological features via SVD and group-specific masking. Extensive experiments on four public neuroimaging datasets (ADNI, PPMI, ABIDE, ADHD-200) demonstrate consistent improvements in both accuracy and efficiency over state-of-the-art baselines.

While BrainPoG presents a promising and well-motivated framework with strong empirical results, the paper suffers from significant technical ambiguities, methodological gaps, and reproducibility issues. The conceptual innovation is limited, and key components lack rigorous justification. The evaluation, though extensive, would benefit from stronger baselines, statistical validation, and clearer fairness in comparisons.

### Strengths
1. The paper convincingly identifies the limitations of existing brain graph models, high computational cost and lack of pathological focus, and proposes a lightweight, pathology-aware alternative that is highly relevant for real-world clinical deployment.

2. The two-stage design (subgraph filtering → feature distillation) is intuitive and well-motivated. 

3. Covers four diverse datasets spanning both binary and multi-class classification tasks.

### Weaknesses
1. The framework integrates existing ideas (subgraph pruning, SVD-based feature selection) without introducing a fundamentally new graph learning principle. The contribution is more combinatorial than foundational.

2. The term “distillation” is misleading, as it implies a teacher-student framework, which is not used. A more precise term such as “feature refinement” or “extraction” is recommended.

3. The use of X∈RN×N derived directly from the adjacency matrix is non-standard and leads to dimensional inconsistencies. It is unclear whether these are node features or the adjacency matrix itself.

4. The SVM-based patho-score lacks statistical rigor. The use of classification accuracy as a relevance metric is problematic without confidence intervals or statistical testing.

5. The claim that top-k and bottom-k SVD components correspond to “communal” and “non-informative” features is intuitive but lacks theoretical or empirical justification. Proposition 1 is descriptive rather than formally proven.

6. Equations (2)–(5) are poorly motivated. The Bernoulli-sampling-based masking introduces unnecessary stochasticity without clear benefits over deterministic masking.

7. Missing comparisons with recent efficient GNNs and attention-based models that would help isolate the benefits of hard vs. soft subgraph selection. No statistical significance tests (e.g., t-tests) to validate performance improvements.

8. ABIDE and ADHD-200 are multi-site datasets, but no site-wise splitting or harmonization strategy is mentioned, raising concerns about data leakage.

9. Key implementation details are missing (e.g., SVM hyperparameters, spectral clustering settings). The computational cost of the SVM-based subgraph scoring is not included in the efficiency analysis, potentially skewing comparisons.

10. Although some hyperparameters (k, ρ) are studied, others (e.g., patho-score threshold αα, community number) are set without sensitivity analysis.

11. Figure 3 uses similar colors for distinct brain networks (e.g., Limbic vs. Cognitive Control), reducing clarity.

12. The flow between “Division” and “Cluster” in Figure 1 is ambiguous and should be clarified to avoid misinterpretation.

### Questions
Please see weakness first.

1. The patho-score is derived from SVM accuracy differences. What is the theoretical or neurobiological basis for equating this empirical score with true pathological relevance? Have you considered alternative relevance metrics (e.g., mutual information, gradient-based attributions)?

2. Why was SVM chosen over other classifiers (e.g., Random Forests) for subgraph scoring? Was its computational cost included in the efficiency analysis? Similarly, why was a linear model (GCN) chosen for the final classification?

3. Were all baseline models trained on the same preprocessed input features and given comparable hyperparameter tuning? Can you confirm that the reported improvements are not due to uneven optimization?

4. Why are both top-kk and bottom-kk features dropped with the same kk? Is there empirical or theoretical evidence that both sets are disease-irrelevant?

5. Have you considered comparing against an attention-based model (e.g., GAT or Transformer) to evaluate whether the benefits come from hard subgraph selection or softer, learned weighting?

6. The paper claims “lesion localization,” but this is only qualitatively visualized. Have you quantitatively compared the identified regions against established clinical biomarkers or atlas-based priors?

7. Will you release fully documented code, including all preprocessing steps, hyperparameter configurations, and data splits?

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
The paper proposes Brain PathoGraph Learning (BrainPoG), a lightweight framework for efficient brain graph learning. It introduces two novel components: (1) a pathological pattern filter that identifies disease-relevant subgraphs using an SVM-based patho-score, and (2) a pathological feature distillation module that drops noise features via SVD and enhances disease-specific ones. Experiments on four benchmark fMRI datasets (ADNI, PPMI, ABIDE, ADHD-200) show superior accuracy and computational efficiency compared to several strong baselines.
Overall, the work is technically sound and presents an interesting idea that bridges neuroscience and graph learning.

### Strengths
(1) Novel two-stage design (filter + distillation) effectively combines graph pruning and feature refinement.
(2) The concept of PathoGraph is intuitive and biologically interpretable.
(3) Extensive experiments on multiple benchmark datasets, including both binary and multi-class settings.
(4) Ablation studies, visualization, and hyperparameter analysis demonstrate robustness and design justification.

### Weaknesses
(1) The effectiveness of the pathological pattern filter depends heavily on the chosen subgraph partitioning method (e.g., atlas-based division vs. spectral clustering). This reliance may introduce variability in results and raises concerns about reproducibility when applied to different brain atlases or parcellation schemes
(2) While BrainPoG achieves strong performance on benchmark datasets (ADNI, PPMI, ABIDE, ADHD-200), it has not been validated on real-world clinical workflows. The lack of external, clinically diverse datasets and longitudinal validation limits its immediate applicability in healthcare
(3) Although the proposed framework integrates subgraph filtering and feature distillation, the degree of novelty compared with previous efficient brain graph learning methods (e.g., IGS, BrainGNN, ALTER) is not fully emphasized.
(4) Missing statistical significance. Improvements over baselines are reported without hypothesis testing or confidence intervals. The ablation study merges multiple submodules; more granular experiments (e.g., only dropping augmentation) could strengthen causal conclusions.

### Questions
(1) Why is SVM chosen for patho-score computation instead of a neural or information-theoretic relevance model? Please add more comparison analysis.
(2) Could the model handle multimodal inputs (e.g., structural + functional connectivity)?
(3) How robust is the proposed method to noisy or missing fMRI connections, which are common in real datasets?
(4) Are the efficiency metrics (runtime, memory) measured for training only or include feature preprocessing and SVM scoring? 
(5) Are the “disease-relevant” subgraphs consistent across different random seeds or dataset splits? Please explain it.
(6) See more on Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Brain PathoGraph Learning (BrainPoG), a lightweight brain graph learning framework for disease detection from fMRI data. The key idea is to selectively learn pathological patterns and pathological features while filtering out irrelevant substructures to reduce computation and improve interpretability.

### Strengths
1. The authors address a well-known bottleneck in brain graph learning—high model cost and noise from disease-irrelevant regions—while emphasizing clinical deployability.
2. The decomposition into “pattern filter” and “feature distillation” modules is conceptually intuitive and computationally efficient. Both steps are interpretable, offering potential for lesion localization and neuroscientific insight.

### Weaknesses
1. While simple, using SVM accuracy as a proxy for disease relevance lacks statistical rigor. It would strengthen the argument to justify why SVM (and not, e.g., permutation-based relevance testing or mutual information) is ideal for computing patho-scores.
2. Dropping top-k and bottom-k features based purely on singular vectors assumes linear separability and might ignore non-linear feature interactions.
3. Although the model is “lightweight,” the SVD operations across nodes and subjects could still be heavy for large-scale datasets.
4. Some phrases (“disease-irrelevant noise features,” “communal features,” “group-specific feature augmentation”) are vague without formal definitions in the main text.

### Questions
1. How sensitive are the patho-scores to the SVM hyperparameters or random initialization? Could small changes in data split alter which subgraphs are pruned?
2. Can BrainPoG trained on one dataset (e.g., ADNI) transfer to a different disease cohort (e.g., PPMI) without retraining the filter?
3. Beyond qualitative lesion maps, did you consult domain experts or quantitative metrics (e.g., overlap with known ROI masks) to validate localization accuracy?

### Soundness
3

### Presentation
3

### Contribution
3
