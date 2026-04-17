# M$^2$DDI: A Unified Framework for Dynamic Multimodal Fusion in Drug-Drug Interaction Prediction

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Drug-drug interaction (DDI) prediction is critical for ensuring patient safety and optimizing therapeutic outcomes. Existing computational approaches are limited by their inability to jointly model the heterogeneous mechanisms underlying DDIs, which span molecular structure, pharmacodynamic function, and network-mediated relations. To address this limitation, we introduce \texttt{M$^{2}$DDI}, a unified framework for dynamic multimodal fusion in DDI prediction. \texttt{M$^{2}$DDI} utilizes a Mixture-of-Experts architecture, with each expert dedicated to a distinct pharmacological modality. A novel prior-enhanced dual-path gating strategy adaptively selects relevant experts for each drug pair by integrating mechanism-matched feature queries and ATC-based biomedical priors, thereby aligning expert selection with underlying pharmacological mechanisms and addressing the challenge of data incompleteness. Empirical evaluation on benchmark datasets demonstrates that \texttt{M$^{2}$DDI} achieves state-of-the-art performance, particularly in new drug scenarios. Additional robustness experiments show that \texttt{M$^{2}$DDI} maintains high predictive accuracy even when modality-specific information is partially missing, outperforming existing methods under similar conditions. Analysis of expert selection patterns further confirms alignment with established pharmacological mechanisms. These results establish \texttt{M$^{2}$DDI} as an effective and mechanism-aware solution for comprehensive DDI prediction. The code are available at \url{https://anonymous.4open.science/r/M2DDI-AECB}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To jointly integrate molecular structure, pharmacodynamic function, and network-mediated relations in DDI prediction, this paper utilizes a Mixture-of-Experts (MoE) architecture, with each expert dedicated to a distinct pharmacological modality. A novel prior-enhanced dual-path gating strategy is proposed to adaptively select relevant experts for each drug pair by integrating mechanism-matched feature queries and Anatomical Therapeutic Chemical (ATC)-based biomedical priors, thereby aligning expert selection with underlying pharmacological mechanisms and addressing the challenge of data incompleteness. The proposed M^2DDI method was compared with seven state-of-the-art DDI methods on two widely used DDI datasets DrugBank and TWOSIDES.

### Strengths
S1: This work is to develop a unified framework that integrates different modalities to enhance the DDI prediction, outperforming single modality-based approaches.
S2: To address modality heterogeneity, dynamic relevance, and data incompleteness, a new multi-modal mixture-of-experts DDI prediction method is proposed.
S3: A novel dual-path gating strategy is embedded in M^2DDI by combining mechanism-matched feature queries and biomedical prior for dynamic expert selection.

### Weaknesses
W1: How to incorporate biomedical knowledge prior to guiding expert selection when data are incomplete or missing is not well elaborated. If the modality incompleteness is a major obstacle, this work should compare the proposed strategy with state-of-the-art solutions, which was mostly neglected by the current version.
W2: What is the difference and connection between the proposed M^2DDI and multi-head fusion attention strategy? The proposed strategy is to synergize a heterogeneous expert pool and a prior-enhanced dual-path gating strategy for expert selection, which is similar to multi-head fusion attention strategy.
W3: A prior-enhanced dual-path gating mechanism is designed to activate the top-K most relevant experts from a heterogeneous pool, and to dynamically decide which experts are relevant for a given input and how to weight their contributions. In the experiments, four structural experts, four functional experts, and one relational expert were configured while K was set-up to five for each prediction. Obviously, the number of relational experts insufficient and the rationale to determine K=5 is not discussed.

### Questions
- The motivation of addressing modality incompleteness or missing is lacking. Incorporating the discussion of dynamic relevance and data incompleteness regarding background and challenges in DDI would improve the paper.
- What is the difference and connection between the proposed M^2DDI and multi-head fusion attention strategy? The proposed strategy is to synergize a heterogeneous expert pool and a prior-enhanced dual-path gating strategy for expert selection, which is similar to multi-head cross-attention mechanism where each head could be considered an expert attending to different aspects of the drug pair.
- A prior-enhanced dual-path gating mechanism is designed to activate the top-K most relevant experts from a heterogeneous pool, and to dynamically decide which experts are relevant for a given input and how to weight their contributions. In the experiments, four structural experts, four functional experts, and one relational expert were configured while K was set-up to five for each prediction. Obviously, the number of relational experts insufficient and the rationale to determine K=5 is not discussed. Was this value empirically validated? An ablation study showing performance vs. different K values is essential.
- The paper claims to align expert selection with underlying pharmacological mechanisms. How are these mechanisms formally defined and mapped to the proposed modalities (structural, functional, relational)? Is there a many-to-many relationship between mechanisms and modalities?
- This paper would be strengthed by comparing graph-based fusion strategies instead of early vs. late fusion. The claim of dynamic multimodal fusion needs a more precise definition in the context of prior work.
- What are the two distinct paths in the gating network? Is one path processing the raw drug-pair features and the other processing the ATC-code priors? How are their outputs combined (e.g., additive, multiplicative, concatenated) to produce the final gating weights? How is the ATC-based prior knowledge represented and integrated? Is it a simple embedding lookup, or is there a more complex encoding process?
- Does the performance of M^2DDI degrade gracefully, and is the gating network's behavior interpretable when a key modality is missing (e.g., does it successfully re-weight its reliance to the available experts)?
- The M^2DDI method proposed in this paper overlaps to some extent with the Multimodal Feature Optimal Fusion Drug Interaction Prediction (MOF-DDI) method [1], which integrates features from multiple data sources to predict drug interactions. Specifically, the MOF-DDI model comprehensively considers literature descriptions of drug interactions, biomedical knowledge graphs, and drug molecular structures to predict drug-drug interactions.

[1] Wen, Q., Li, J., Zhang, C. and Ye, Y., 2023, October. A multi-modality framework for drug-drug interaction prediction by harnessing multi-source data. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (pp. 2696-2705).

Minor errors:
In page 2, Prior-Enhanced Dual-Path Gating: M2DDI employs a Mixture-of-Experts architecture that includes a novel dual-path gating strategy, combining mechanism-matched feature queries and biomedical priors for dynamic expert selection, directly addressing
https://github.com/mechanism heterogeneity and data incompleteness. "https://github.com/mechanism" should be removed.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper accurately identifies a core challenge in DDI prediction: existing methods struggle to jointly model the heterogeneous pharmacological mechanisms underlying DDIs. To address this, the authors propose M2DDI, an innovative Mixture-of-Experts (MoE) framework. This framework assigns specialized experts to each pharmacological modality, offering a logical and elegant architectural solution to this fundamental problem.

### Strengths
The paper accurately identifies a core challenge in DDI prediction: existing methods struggle to jointly model the heterogeneous pharmacological mechanisms underlying DDIs. To address this, the authors propose M2DDI, an innovative Mixture-of-Experts (MoE) framework. This framework assigns specialized experts to each pharmacological modality, offering a logical and elegant architectural solution to this fundamental problem.

### Weaknesses
1.	The evaluation metrics are insufficient. In multi-class DDI studies, it is standard to report six core metrics: Accuracy (ACC), AUC, AUPR, F1, Recall, and Precision.
2.	The model's strong generalization for new drugs (S1/S2 scenarios) is a key advantage. The problem, however, is that this ability is heavily reliant on ATC prior knowledge. According to the experiments in Figure 7, the model's performance in S1 and S2 scenarios drops "markedly" without this ATC path. This indicates its generalization isn't truly learned but rather "borrowed" from an external knowledge base. This creates a risk: if faced with a brand-new compound that lacks an ATC code, the model's predictive performance could plummet.
3.	The baseline comparison is incomplete: the model is not evaluated against multimodal DDI methods published in 2024–2025.
4.	Within the multimodal fusion framework, the paper lacks cross-modal interpretability.
5.	In a realistic novel-drug setting, a new drug typically comes with only partial information. However, if “new drugs” are selected directly from an existing knowledge graph, those attributes are often already fully recorded in the graph—so they are not truly novel or missing. This choice can overstate the information available to the model and deviate from real-world deployment conditions.

### Questions
1.	The evaluation metrics are insufficient. In multi-class DDI studies, it is standard to report six core metrics: Accuracy (ACC), AUC, AUPR, F1, Recall, and Precision.
2.	The model's strong generalization for new drugs (S1/S2 scenarios) is a key advantage. The problem, however, is that this ability is heavily reliant on ATC prior knowledge. According to the experiments in Figure 7, the model's performance in S1 and S2 scenarios drops "markedly" without this ATC path. This indicates its generalization isn't truly learned but rather "borrowed" from an external knowledge base. This creates a risk: if faced with a brand-new compound that lacks an ATC code, the model's predictive performance could plummet.
3.	The baseline comparison is incomplete: the model is not evaluated against multimodal DDI methods published in 2024–2025.
4.	Within the multimodal fusion framework, the paper lacks cross-modal interpretability.
5.	In a realistic novel-drug setting, a new drug typically comes with only partial information. However, if “new drugs” are selected directly from an existing knowledge graph, those attributes are often already fully recorded in the graph—so they are not truly novel or missing. This choice can overstate the information available to the model and deviate from real-world deployment conditions.

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
3

### Summary
This paper proposes a unified multimodal framework for DDI prediction based on MoE architecture. Each expert corresponds to a distinct pharmacological modality and a dual-path gating mechanism dynamically selects the most relevant experts for each drug pair. The gating combines feature-based similarity and prior biomedical knowledge via ATC embeddings, aligning expert selection with underlying pharmacological mechanisms. Experiments show SOTA performance, particularly in new drug scenarios, and robustness against incomplete modality information.

### Strengths
* This paper presents a novel unified multimodal framework that fuses structural, functional, and relational modalities.

* The dual-path gating that integrates data-driven features with pharmacological priors is elegant and biologically meaningful.

* The $M^2DDI$ achieves excellent performance in unseen-drug scenarios.

* The alignment between expert activation and known pharmacological mechanisms provides a mechanistic understanding of DDI.

### Weaknesses
* The overall technical components are standard, and the innovation lies primarily in their combination rather than in new algorithmic formulations.

* The experiments lack direct comparison with recent multimodal foundation models.

### Questions
* How does this method perform compared to recent large pretrained multimodal models (e.g., DDI-GPT [1]) on the same datasets?

* What is the training/inference cost compared to early/late fusion models?

* How sensitive is the model to the depth or incompleteness of ATC hierarchies, especially for rare drugs?

[1] Xu, Chengqi, et al. "DDI-GPT: Explainable Prediction of Drug-Drug Interactions using Large Language Models enhanced with Knowledge Graphs." BioRxiv (2024): 2024-12.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes M²DDI, a mixture-of-experts framework for drug-drug interaction prediction that combines molecular structure, textual information, and knowledge graph modalities through a dual-path gating mechanism.

### Strengths
This paper has a clear motivation, is well-written, and explores the impact of modality loss on DDI tasks.
- Clear problem formulation addressing modality heterogeneity, dynamic relevance, and data incompleteness
- Novel ATC-based prior path for handling missing modalities in new drugs

### Weaknesses
Although the architecture of this paper is well designed, it lacks novelty and theoretical justification. Moreover, issues such as insufficient experimental design and incomplete experiments weaken its contribution.

1. The paper compares Static Fusion and Ensemble as multimodal fusion approaches, but why not consider a more direct and computationally cheaper method? Specifically, obtaining embeddings from three modalities, learning a weight for each modality through an MLP, and then fusing them directly for prediction? This is important because M2DDI's improvement over Ensemble appears quite limited (S0: Ensemble 97.58 → 97.82; S2: Ensemble 46.45 → 46.49).

2. In the ablation study shown in Figure 3, can the authors provide the Accuracy results? From Table 2, the improvements in Accuracy are marginal (S0: EmerGNN 97.42, Ensemble 97.58 → 97.82; S2: EmerGNN 46.34, Ensemble 46.45 → 46.49).

3. Figure 4 aims to examine the impact of missing functional and relational modalities while always retaining structural information. However, why does the structure-based baseline "MLP" exhibit the steepest performance degradation? Moreover, it would be more convincing to include additional graph-based baselines in this robustness evaluation.

4. It remains unclear why the authors did not include more recent baselines, such as: [1] CARMEN: Context-Aware Safe Medication Recommendations with Molecular Graph and DDI Graph Embedding (AAAI 2023); [2] SSF-DDI: a deep learning method utilizing drug sequence and substructure features for drug-drug interaction prediction (2024); [3] DSN-DDI: an accurate and generalized framework for drug-drug interaction prediction by dual-view representation learning (2023); [4] Learning motif-based graphs for drug-drug interaction prediction via local-global self-attention.
 
5. Figure 6 demonstrates that the optimal configuration is Nr = 1, Ns + Nf = 8, K = 5, but can the authors provide an explanation for why this specific setting works best? More importantly, I am interested in understanding the expert selection dynamics during training. Are the scores of different experts often close to each other? Does the model always tend to select the same fixed set of k (5) experts, or does the selection vary significantly across different drug pairs? Can the authors clarify this point, perhaps by providing visualizations of expert utilization statistics or the distribution of gating scores across the dataset?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
