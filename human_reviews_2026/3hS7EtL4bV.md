# PGRF-Net: A Prototype-Guided Relational Fusion Network for Diagnostic Multivariate Time-Series Anomaly Detection

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Multivariate time-series anomaly detection (MTSAD) faces a critical trade-off between detection performance and model transparency. We propose PGRF-Net, a novel architecture designed to achieve competitive performance while providing structured evidence to support diagnostic insights. At its core, PGRF-Net uses a Multi-Faceted Evidence Extractor that combines prototype learning with the discovery of dynamic relational structures between variables. This extractor generates four distinct types of anomaly evidence: predictive deviation, structural changes in learned variable dependencies, contextual deviation from normal-behavior prototypes, and the magnitude of localized spike events. This evidence is then processed by a Gated Evidence Fusion Network, which learns to weigh each source via data-driven gating. PGRF-Net is trained via a two-stage unsupervised strategy for robust extractor learning and subsequent fusion tuning. Extensive experiments on five public MTSAD benchmarks demonstrate its competitive or superior detection performance. Importantly, by decomposing the final anomaly score into these four evidence types, our model facilitates diagnostic analysis, offering a practical step towards more interpretable, evidence-based MTSAD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PGRF-Net, a prototype-guided relational fusion network for unsupervised multivariate time-series anomaly detection (MTSAD). The stated objective is to close the gap between high detection performance and diagnostic transparency by decomposing an anomaly score into four interpretable “evidence” channels: (i) predictive deviation, (ii) structural deviation in learned inter-variable dependencies, (iii) contextual deviation from normal-behavior prototypes, and (iv) localized spike magnitude. Methodologically, the model has two stages: a Multi-Faceted Evidence Extractor and a Gated Evidence Fusion Network that learns instance-dependent weights for the non-predictive channels and combines them with prediction error. Training is fully unsupervised, with Stage-1 losses for prediction and prototype/graph regularization, and Stage-2 losses for gate suppression on pseudo-normal samples plus entropy sharpening. Experiments on five public benchmarks (MSL, SMAP, PSM, SMD, SWaT) report state-of-the-art average F1 and competitive AUC-ROC/PR, along with qualitative “five-panel” diagnostic visualizations linking anomaly attributions to changes in relational masks.

### Strengths
- A clear diagnostic framing of MTSAD that operationalizes interpretability via four disjoint evidence channels and a gated fusion mechanism. This goes beyond post-hoc explanations by making attribution a first-class training target.

- The two-stage training is well-motivated: learn robust features and prototypes first; then calibrate evidence attribution with gate suppression on pseudo-normal samples and entropy sharpening for selectivity.

- Sensible regularization of the structural channel (sparsity, acyclicity, temporal stability vs. a baseline mask) reduces degenerate graphs and contributes to stable, interpretable dependency patterns.

- The paper asserts “competitive or superior” performance; averaged F1 and AUC-PR are high, but per-dataset AUC-ROC/PR tables show mixed margins and sometimes narrow differences. Statistical significance (e.g., paired tests across seeds) and variance bars are not reported; claims of superiority would be stronger with formal significance tests.

### Weaknesses
- The structural deviation relies on a learned baseline mask (M_base) defined via Early-training averages on presumed normal data. If the normal set contains regime shifts or drifting dependencies, M_base may encode mixture regimes; deviations may then over- or under-fire. The paper does not quantify how sensitive S_struct is to mis-specified normal baselines, distribution shift, or domain drift within normal periods.

- The acyclicity constraint is imposed on structural prototypes so that convex combinations remain DAG-like. However, instantaneous correlations in cyber-physical systems can be cyclic (feedback loops). For domains with known cycles, forcing DAGs may bias the learned mask and redirect anomaly mass to other channels, potentially mis-attributing structural anomalies as contextual or predictive deviations.

- The use of point adjustment and best-F1 thresholding is conventional, but can inflate event-level detection and mask temporal localization quality. While AUCs are reported, there is no comparison on strict point-wise metrics.

- The interpretability story is qualitative. The paper would benefit from quantitative interpretability metrics (fidelity, stability, human-utility studies) or counterfactual tests (e.g., perturb relationships to see whether S_struct changes in expected directions, with calibration curves).

- Figure 1 could be improved; excessive whitespace and mixed concepts dilute key information, with small labels.

### Questions
- DAG Constraint vs. Cyclic Dependencies: Many industrial/biological systems have feedback loops. By imposing acyclicity on structural prototypes, are you baking in a bias that pushes cyclic phenomena into other channels (e.g., S_ctx or Spred)? What breaks if the true dependency graph is nearly cyclic (or dense), and how would relaxing DAG constraints (e.g., allowing small cycles with penalties) affect S_struct fidelity?

- Baseline Mask (M_base) Stability: M_base is averaged from early-training masks. If normal data itself is multi-modal or drifting, the mean mask may be unrepresentative. Why not prototype-ize M_base itself (i.e., multiple baseline masks with selection), or define baselines by time-clustered regimes? What empirical evidence shows M_base is stable and representative, and how sensitive is S_struct to the time window used to compute it?

- Stage-2 suppresses explanatory gates on pseudo-normal samples identified by low Spred. This equates “predictable” with “normal.” How does the method behave when contextual drifts are predictable (low Spred) yet abnormal relative to prototypes (high S_ctx)? Could gate suppression systematically down-weight contextual evidence in such cases, and have you measured the false-negative impact under controllable scenarios?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes PGRF-Net, an unsupervised framework for multivariate time-series anomaly detection that reframes detection as a diagnostic process. Rather than outputting a single anomaly score, PGRF-Net decomposes anomalies into four types: predictive deviation, structural changes in inter-variable dependencies, contextual deviation from learned normal-behavior prototypes, and localized spike events. These are generated via a Multi-Faceted Evidence Extractor and adaptively fused using a Gated Evidence Fusion Network trained in a two-stage unsupervised manner. The method is evaluated on five standard MTSAD benchmarks, where it achieves competitive or state-of-the-art performance while providing decomposed, human-interpretable explanations for detected anomalies.

### Strengths
1.	The work considers a critical gap in MTSAD—namely, the lack of diagnostic interpretability in deep models. By explicitly modeling multiple anomaly modalities and attributing scores to distinct evidence types, PGRF-Net offers actionable insights for domain experts, which enhances trust and operational utility.
2.	The architecture is thoughtfully designed. The use of frequency decomposition, conformer encoders, and learnable prototype banks for structural, contextual, and spike patterns can grasp both time-series modeling and interpretable representation learning. The two-stage training strategy effectively separates representation learning from evidence fusion.

### Weaknesses
1.	The four evidence types are presented as diagnostic, but the model provides associative, not causal, explanations. For instance, a high structural deviation score indicates a shift in learned dependency patterns—but it does not establish whether this shift caused the anomaly or is a consequence of it.
2.	All benchmarks provide only binary anomaly labels, not fine-grained labels indicating anomaly types (e.g., spike vs. contextual shift). Consequently, the paper cannot verify whether the decomposed evidence correctly identifies the underlying anomaly mechanism.
3.	The structural prototype bank assumes acyclic dependency graphs (via DAG constraints), which may not hold in many real-world feedback systems (e.g., control loops in industrial plants). The paper does not discuss this limitation or evaluate scenarios where cyclic dependencies are essential.

### Questions
The paper claims the four evidence types provide diagnostic insights, but all benchmarks only have binary anomaly labels. How do the authors validate that each evidence type correctly reflects the true anomaly mechanism without ground-truth type annotations?

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
This paper proposes PGRF-Net for multivariate time series anomaly detection and interpretation. PGRF-Net decomposes anomaly evidence into multiple types and introduces prototype banks and a gated fusion mechanism to learn diagnostic representations. Experiments across several benchmark datasets are presented to demonstrate its performance advantage and diagnostic interpretability.

### Strengths
* The paper presents a novel and well-motivated model to unify detection and interpretability.
* The work provides comprehensive ablation studies that evaluate several architectural components, demonstrating the contribution of the proposed modules to detection performance.
* The visualization and synthetic case studies offer qualitative interpretability.

### Weaknesses
* Robustness under anomaly contamination is not well explored.
* Limited benchmark results.

Please find the detailed comments in the following section.

### Questions
* How do prototype formation and gate suppression behave when the training data are contaminated with anomalies or noisy labels?
* How are the scores in Table 4 aggregated? Have the authors compared the proposed gated fusion mechanism against simple aggregation strategies such as averaging, max pooling, or naive ensemble fusion?
* How does end-to-end joint training compare against the proposed two-stage training in terms of performance, convergence, and stability?
* Point-adjustment techniques have been shown to overestimate model performance, and pointwise measures such as AUC-PR are sensitive to anomaly ratio and temporal noise. The authors are encouraged to consider time-series–aware measures such as Range-F1 [1] and VUS-PR [2] and evaluate on a more comprehensive benchmark such as TSB-AD [3].
* The paper lacks an analysis of runtime scalability with respect to both sequence length and the number of variates

[1] Tatbul N, Lee TJ, Zdonik S, Alam M, Gottschlich J. Precision and recall for time series. Advances in neural information processing systems. 2018;31.

[2] Paparrizos J, Boniol P, Palpanas T, Tsay RS, Elmore A, Franklin MJ. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proceedings of the VLDB Endowment. 2022 Jul 1;15(11):2774-87.

[3] Liu Q, Paparrizos J. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. Advances in Neural Information Processing Systems. 2024 Dec 16;37:108231-61.

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
4

### Summary
This paper proposes PGRF-Net, a prototype-guided relational fusion network for multivariate time-series anomaly detection. The method aims to bridge the gap between performance and interpretability by decomposing anomalies into four evidence types (predictive, structural, contextual, and spike deviations) and integrating them via a Gated Evidence Fusion Network trained in a two-stage unsupervised framework. Experiments across five benchmarks (MSL, SMAP, PSM, SMD, SWaT) show competitive or superior results to state-of-the-art models while maintaining computational efficiency. The paper claims that the design not only enhances accuracy but also offers diagnostic interpretability, enabling users to understand why an anomaly occurs, not just that it does.

### Strengths
- **S1.** The paper presents a well-motivated and coherent framework combining multiple sources of anomaly evidence through a gated fusion mechanism.
- **S2.** The two-stage unsupervised training is thoughtfully designed and appears to improve robustness and fusion stability.

- **S3.** The diagnostic decomposition (predictive, structural, contextual, spike) is conceptually valuable and well illustrated in case studies.

### Weaknesses
- **W1.** The evaluation lacks methodological rigor: the paper relies on benchmark protocols with known inflation issues and does not provide sensitivity tests on the point-adjusted evaluation method. (See Section 6.2.5 in [a])

- **W2.** The interpretability claim is unsubstantiated quantitatively. It seems that no metrics, user studies, or human-in-the-loop validations are provided. This omission undermines the central thesis of diagnostic transparency. There is no analysis of failure cases or trade-offs, which would be crucial for a model claiming diagnostic transparency.

- **W3.** The novelty, though meaningful, builds incrementally upon prior prototype-based and dependency-aware methods rather than fundamentally redefining the paradigm.


[a] Trirat, P., Shin, Y., Kang, J., Nam, Y., Na, J., Bae, M., ... & Lee, J. G. (2024). Universal time-series representation learning: A survey. arXiv preprint arXiv:2401.03717.

### Questions
- **Q1.** Can the authors provide a user-centered or human evaluation of interpretability (e.g., do practitioners find the decomposed evidence helpful)?

- **Q2.** How robust are the results when evaluated with non point-adjusted metrics, or under stricter event-level protocols?

- **Q3.** Could the authors clarify how the fusion gates' learned weights correspond to anomaly semantics in unseen domains?

- **Q4.** How does the proposed diagnostic interpretability compare quantitatively to existing interpretable baselines like InterFusion [b]?

[b] Li, Zhihan, et al. "Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding." Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

### Soundness
2

### Presentation
2

### Contribution
3
