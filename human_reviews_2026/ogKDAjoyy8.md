# Unsupervised Dynamic Graph Multi-Model Representation Learning for Temporal Patterns Discovery: Uncovering Parkinson’s Disease Stages Using Cerebrospinal Fluid Longitudinal Profiles

- Decision: Reject
- Scores: 2, 4, 0, 6

## Abstract
Dynamic graph learning methods typically capture local structural information
and short-range temporal dependencies at each time step. In this work, we introduce a dynamic graph learning architecture that generates time-step embeddings capturing both local structural context and progression-trajectory patterns for each node across an entire longitudinal sequence. Unlike existing approaches, our framework clusters fused embeddings that integrate (i) the global temporal trajectory of each node and (ii) its local spatial context at every graph snapshot to discover meaningful temporal patterns in longitudinal datasets. We
evaluate the proposed model in the context of Parkinson’s disease (PD) progression using six years of longitudinal cerebrospinal fluid (CSF) profiles from 24 patients. Visit-based graphs were constructed by representing patients as nodes enriched with peptide-abundance features, and by connecting patients with similar features profiles. A Graph Convolutional Network (GCN) captures visit-specific spatial relationships, while a sequential model learns global temporal representations. A fusion module integrates both sources of information to produce enriched node embeddings that reflect inter- and intra-patient molecular dynamics.
Clustering the learned embeddings reveals four distinct PD progression stages, supported by strong validity indices (Davies–Bouldin: 0.169; Calinski–Harabasz: 1264.24). Significant differences in motor severity (UPDRS 2 and UPDRS 3; p < 0.05) were observed across clusters, whereas non-motor scores showed a more diffuse pattern (p = 0.11). Compared with PCA, autoencoders, GCN, T-GCN, and GC-LSTM, the proposed architecture yields more clinically discriminative representations of disease severity. These findings demonstrate the potential of the proposed dynamic graph learning for data-driven disease staging and offer a generalizable framework for uncovering latent temporal patterns in longitudinal datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a novel unsupervised dynamic graph multi-model framework for discovering temporal disease patterns from longitudinal biomedical data, with application to Parkinson’s disease (PD). The authors integrate a Graph Convolutional Network (GCN) for learning spatial dependencies among patients at each clinical visit with a Gated Recurrent Unit (GRU) for modeling temporal progression across visits. The model fuses spatial and temporal embeddings to produce comprehensive node representations, which are then clustered to reveal disease stages. Evaluation on a longitudinal cerebrospinal fluid (CSF) dataset of 24 PD patients demonstrates that the proposed model identifies four distinct disease stages, with significant correspondence to clinical UPDRS motor scores.

### Strengths
1. The combination of GCN (for intra-timepoint spatial modeling) and GRU (for temporal modeling) is well-motivated and implemented in a consistent framework.

2. The linkage between discovered clusters and clinically meaningful UPDRS scores provides valuable interpretability and supports the validity of the results.

### Weaknesses
- The manuscript is lengthy and sometimes reads as a descriptive technical report rather than a concise scientific contribution; key design motivations and insights are not emphasized. For instance, the abstract focuses too much on procedural details rather than emphasizing the core contributions, **resulting in an overly lengthy summary that obscures the main point.**
- Only 24 patients were included in the final analysis, which significantly limits the generalizability and robustness of the conclusions.
- The proposed architecture largely combines established modules (GCN and GRU) with standard fusion operations; no novel learning mechanism or theoretical contribution is introduced.
- The impact of each model component (GCN, GRU, fusion layers) is not quantitatively assessed. **The ablation analysis is necessary.**
- The paper does not provide a clear theoretical justification or complexity analysis explaining why the fusion of spatial and temporal embeddings improves representation quality.
- While ICLR permits supplementary material for code submission, the authors provided a GitHub link (seems not fully anonymized) pointing to their repository, yet no implementation code could be found there.

### Questions
Please refer to the above **Weaknesses**.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an unsupervised dynamic graph learning framework designed to discover temporal disease stages from longitudinal biomedical data. The authors propose a multi-model architecture that integrates a single-layer GCN, to capture spatial inter-patient similarity at each time step with a GRU to capture temporal dependencies across visits. The fused embeddings are clustered to identify Parkinson’s Disease stages using longitudinal cerebrospinal fluid peptide profiles. The method is evaluated against standard baselines using clustering validity metrics and non-parametric statistical tests. Results show that the model identifies four interpretable PD stages that correlate with UPDRS motor scores.

### Strengths
1. Using dynamic graph learning for unsupervised disease stage discovery is original, particularly in modeling temporal patient trajectories via age-based graphs.

2. The combination of GCN and GRU with a fusion module is conceptually coherent and technically straightforward. The reasoning behind using shallow layers to avoid over-smoothing is well-motivated.

3. The use of clustering validity indices and statistical significance testing (Kruskal–Wallis, Dunn) to validate discovered disease stages adds credibility to the biological interpretation.

### Weaknesses
1. The “multi-model” framework is effectively a shallow GCN + GRU fusion, a design already seen in many T-GCN and GC-LSTM variants. The contribution lies more in the application and evaluation context than in model innovation.
2. Only 24 patients are used after filtering, which raises concerns about overfitting and the reliability of clustering outcomes. The reported high Calinski–Harabasz and low Davies–Bouldin scores may not generalize.
3. The significance tests (e.g., Kruskal–Wallis, Dunn) are performed on very small sample sizes (some clusters appear to have few patients). The power of these tests and their reliability for clinical interpretation are questionable.
4. Presentation issues and overclaiming. 
(a) The text occasionally conflates Parkinson’s and Huntington’s disease (see Section 2–3 confusion line 210–213), which undermines clarity. (b) Figures are referenced but not well-integrated in the discussion (e.g., Figure 2 architecture schematic is described verbosely but visually contributes little). (c) Claims of “first dynamic graph model for neural disorder diseases” seem overstated, given existing work in dynamic GNNs for biomedical longitudinal analysis.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes an unsupervised graph-based learning framework for longitudinal disease data analysis. Each node represents a single patient, and edges encode pairwise similarity between nodes. A single graph is constructed based on the same aged patients, and a set of such graphs represents the temporal evolution of the cohort. The model integrates GCN and GRU to capture spatio-temporal dependencies across these age-based graphs, and a downstream clustering algorithm is applied to identify disease stages. The clustering quality is evaluated using three metrics, demonstrating that the proposed framework generally outperforms the baseline GCN.

### Strengths
1)	The motivation of the study, i.e., learning longitudinal patterns of neurodegenerative diseases, is clear and intuitive. 
2)	The research includes statistical analyses of results.

### Weaknesses
1)	There are lack of technical novelty and significant lack of comparisons with existing works. The proposed method does not introduce a fundamentally new learning mechanism. The model is simply built upon a simple GCN with an additional GRU layer and a downstream clustering phase, offering incremental changes. Moreover, existing spatio-temporal graph models were not sufficiently examined; among the adopted baselines, only T-GCN is a spatio-temporal graph model. Additional recent baselines [1-4] are encouraged to be added. Given the growing number of graph studies, I believe that the authors could find more spatio-temporal graph learning methods with open source codes to strengthen the contribution of the proposed method.

[1] Cini et al., “Scalable Spatiotemporal Graph Neural Networks”, AAAI 2023

[2] Tang et al., “Predicting 30-Day All-Cause Hospital Readmission Using Multimodal Spatiotemporal Graph Neural Networks”, IEEE Journal of Biomedical and Health Informatics, 2023

[3] Cho et al., “Mixing Temporal Graphs with MLP for Longitudinal Brain Connectome Analysis”, MICCAI 2023

[4] Pareja et al., “EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs”, AAAI 2020

2)	Ablation studies of the model components and ranges of hyperparameter tuning (for both the proposed method and the baselines) are not provided. Including these details would improve the transparency and reproducibility of the experiments.
3)	Overall, the presentation of the paper has lots of room for improvement, including the writing, figures, and formulas. For example, the abstract contains too much details of experimental settings. The flow of the Introduction section is disorganized and mixed with background, motivation, and methodological contributions without a clear transition. In the method section, no loss function is given; instead, dataset description and model training configurations are presented, which would be more appropriately placed in the experiment section. Figure 2 is overly complex and difficult to interpret; it would benefit from a clearer depiction of the model architecture that highlights the key components and their interactions, rather than presenting every detail from all $G_t$. Furthermore, Fig. 3 does not provide exact numerical values of the performance metrics, making it difficult to compare the models quantitatively. Presenting these results in a table format would make the comparison clearer and more informative. The paper lacks an introduction of the evaluation metrics (e.g., UPDRS), which makes it difficult for readers outside the clinical domain to interpret the reported results.

### Questions
1)	Why is the method described as a multi-‘MODEL’ framework? The proposed method combines a GCN and a GRU, but GRU (Gated Recurrent Unit) itself is a unit/layer, rather than an independent model architecture. Moreover, although the loss function of the proposed method is not explicitly stated in the paper, it appears that the method only utilizes a single loss function, which makes it difficult to call it ‘multi-model’ framework.

2)	Why were edges constructed based on small nodal similarities rather than high similarities? I think the term ‘similarity’ might have been misused and should be replaced with ‘distance’, since Euclidean distance was employed as the measurement.

3)	Compared to recent spatio-temporal graph studies, what is the concrete technical advantage of the proposed method, and how do each model component meaningfully contribute to model performance or interpretability?

4)	In line 127, it is stated that T-GCN and GC-LSTM are limited to capturing only short-term temporal graph features. However, the rationale behind this claim is not clearly supported. It would be helpful if the authors could provide either a theoretical justification or quantitative comparison results demonstrating how the proposed method captures longer temporal dependencies compared to existing approaches.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new unsupervised dynamic graph representation learning framework for longitudinal biomedical data applied to a Parkinson's disease progression dataset. It combines per-visit graph neural network embeddings based on patient similarity edges, yielding integrated patient representations that capture both within-visit context and the entire across-visits temporal trajectory. It is claimed that this method outperforms methods like PCA and autoencoders, among others, while also claiming that the clusters discovered are meaningful.

### Strengths
1. It tackles the unsupervised representation learning problem for temporal graph-structured data in healthcare, an area that has been underexplored.
2. The architecture seems to be a clever/innovative way to put together previous well-established GNN methods for the specific challenge at hand.
3. The model naturally handles missing visits by simply omitting nodes for those missing visits (i.e. if a patient has no record at a given timepoint, they are not included in that snapshot graph). This approach to incomplete longitudinal data is practical and avoids complex imputation, while I also think it's a clever way to handle missing visits that I have not seen before.
4. The resulting clusters appear reasonable. 
5. The method seems to achieve the highest clustering scores across many different baselines
6. The paper reports that the clusters identified by the so-called Multi-Model are more stable than the GCN baseline, which is important.

### Weaknesses
1. The experiments are run on a dataset with only 24 patients which is quite limited and thus raises concerns about robustness. The cluster findings might not generalise well for broader PD populations.
2. It's unclear if this model would scale to larger patient cohorts, as training multiple GCNs with so many nodes and a sequence model could be computationally heavy and not handle hundred of patients as seen in other datasets without significant modification or risks of overfitting.
3. Some aspects of the writing are confusing or inconsistent, which impacts clarity. For example, the authors describe constructing “age-based” graphs, when in fact each graph corresponds instead to a specific visit time (a better -and correct- term in my opinion would be visit-based graphs). The term “multi-model” is overused without a clear definition, and in my perspective only one model is actually being presented in this work, rather than many. Additionally, section 3.1 wrongly refers to "both datasets" even though only one dataset is used, suggesting a leftover from a previous draft. Finally, I believe "Huntington's disease" at the end of Related Work is a typo.
4. The inclusion of t-SNE as a baseline for the embedding/clustering comparison is questionable. t-SNE is a stochastic visualisation algorithm, not a fixed representation learning method, so its results can vary run to run and it's not typically used for clustering performance benchmarks.
5. The work lacks any evaluation on external datasets or discussion of generalisability. All results are on a single small PD cohort, and there's no evidence the learned representations would transfer to a different patient population or data source. This absence is a major concern for real-world application, since clinical tools typically require validation on independent cohorts.
6. The paper does not clearly explain how the model was trained and tuned, which affects reproducibility. It's unclear how the train/validation split was handled given the unsupervised nature of the model. It is mentioned that hyperparameters were adjusted, but I don't know how (ie, was there a held-out validation set, or was the entire dataset used for training/clustering?). I'm left unsure whether the reported performance might be inflated by overfitting or trial-and-error hyperparameter tuning on the test set
7. The authors do not provide any error analysis of potential misclustered cases. For a clinical application, understanding whether certain patients were borderline or inconsistently clustered (and why) would be not only important but also interesting. The absence of such analysis means we don't know how robust each assignment is, especially given the small sample size.

### Questions
1. What exactly does "multi-" in "multi-model" mean? The term is used repeatedly (and even marked with a special icon in Figure 2), but it isn't explicitly defined.
2. How were hyperparameters tuned and what was the training/validation splits used for model selection? Since this is an unsupervised task, did the authors use a portion of data to validate the representation quality when tuning parameters, or was the entire dataset used for both training and evaluation?
3. What was the rationale for using t-SNE as one of the baseline embedding methods for clustering? t-SNE is stochastic and typically intended for visualization rather than as a fixed embedding for clustering as absolute Euclidean distances could vary across runs, so this was a bit unexpected to me.
4. Given the unsupervised framework and how the graphs are created, it seems to me that this model cannot be directly applied to new patients. So, how were the authors thinking about using this in a clinical setting when applied to new people?
5. Did the authors examine any cases of misclustered or borderline patients, or otherwise measure uncertainty in the cluster assignments? For example, were there patients who repeatedly switched clusters across different runs, or whose embeddings were near cluster boundaries?

### Soundness
2

### Presentation
2

### Contribution
2
