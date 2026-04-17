# Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research

- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Epilepsy affects over 50 million people worldwide, and one-third of patients suffer drug-resistant seizures where surgery offers the best chance of seizure freedom. Accurate localization of the epileptogenic zone (EZ) relies on intracranial EEG (iEEG). Clinical workflows, however, remain constrained by labor-intensive manual review. At the same time, existing data-driven approaches are typically developed on single-center datasets that are inconsistent in format and metadata, lack standardized benchmarks, and rarely release pathological event annotations, creating barriers to reproducibility, cross-center validation, and clinical relevance. With extensive efforts to reconcile heterogeneous iEEG formats, metadata, and recordings across publicly available sources,
we present $\textbf{Omni-iEEG}$, a large-scale, pre-surgical iEEG resource comprising $\textbf{302 patients}$ and $\textbf{178 hours}$ of high-resolution recordings. The dataset includes harmonized clinical metadata such as seizure onset zones, resections, and surgical outcomes, all validated by board-certified epileptologists. In addition, Omni-iEEG provides over 36K expert-validated annotations of pathological events, enabling robust biomarker studies. Omni-iEEG serves as a bridge between machine learning and epilepsy research. It defines clinically meaningful tasks with unified evaluation metrics grounded in clinical priors, enabling systematic evaluation of models in clinically relevant settings. Beyond benchmarking, we demonstrate the potential of end-to-end modeling on long iEEG segments and highlight the transferability of representations pretrained on non-neurophysiological domains. Together, these contributions establish Omni-iEEG as a foundation for reproducible, generalizable, and clinically translatable epilepsy research. The project page with dataset and code links is available at $\url{https://omni-ieeg.github.io/omni-ieeg/}$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Omni-iEEG, a large-scale dataset and benchmark for epilepsy research, comprising 302 patients and 178 hours of high-resolution intracranial EEG (iEEG) recordings from eight leading epilepsy centers. It addresses challenges in clinical EZ localization by providing harmonized data formats, metadata, and over 36K expert-validated annotations of pathological events (e.g., spkHFOs). The dataset supports reproducible research with standardized tasks—Pathological Event Classification and Pathological Brain Region Identification—plus exploratory tasks, demonstrating potential for end-to-end modeling and transfer learning from non-neurophysiological domains.

### Strengths
- **Originality**: First large-scale, multi-center iEEG dataset with harmonized annotations and clinical metadata.
- **Quality**: Expert-validated data and diverse tasks ensure robustness and clinical relevance.
- **Clarity**: Clear task definitions and dataset structure, supported by visuals.
- **Significance**: Bridges ML and epilepsy research, enhancing reproducibility and translatability.

### Weaknesses
- **Methodological Flaws**: Inter-rater reliability for 36K annotations is not quantified, risking bias. HFO detection algorithm selection lacks justification.
- **Experimental Gaps**: No baseline model performance or cross-validation results are provided for benchmark tasks. Transfer learning potential is theoretical without empirical support.
- **Oversight**: Data privacy protocols beyond de-identification are unclear. Scalability of annotation processes for future expansions is unaddressed.
- **Validation**: Claims of clinical translatability lack pilot study results or online validation.

### Questions
1. Can the authors provide inter-rater reliability metrics (e.g., Cohen’s kappa) for the 36K annotations to ensure consistency?
2. What criteria were used to select and tune the HFO detection algorithms (e.g., Navarrete et al., 2016), and how were artifacts filtered?
3. Can baseline performance metrics (e.g., AUC, F1) be provided for the benchmark tasks on Omni-iEEG?
4. How was transfer learning from non-neurophysiological domains tested, and what specific performance gains were observed?
5. What additional privacy measures were implemented beyond de-identification, and how will annotation scalability be managed?

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
The core contribution of this paper is the construction of a large-scale, comprehensive iEEG dataset—Omni-iEEG, along with standardized benchmark tests for the dataset. The dataset includes patient data from multiple epilepsy centers and ensures data quality through expert annotations. The authors also conducted a series of machine learning model tests and evaluations on this dataset, particularly for clinical tasks such as epilepsy event classification and focal region identification.

1. The paper seems to primarily focus on the construction of the dataset, while ICLR typically places more emphasis on algorithmic, model, or methodological innovations. How does this dataset significantly differ from existing EEG/SEEG datasets? If it is simply an extension of existing datasets, can the authors clarify its novelty more specifically?
2. The paper mentions that the dataset will be made publicly available. Could the authors further clarify the dataset's release method and the specific platform for access?
3. While the paper mentions the potential applications of the dataset, there is no concrete benchmark testing or application scenario presented. Can the authors provide one or two actual use cases or examples to show how this dataset can advance epilepsy detection and prediction models?
4. The quality of the dataset is crucial for its application. While the paper mentions the annotation process and methods, it does not provide detailed information about the accuracy and consistency validation of these annotations. Could the authors elaborate on this aspect?
5. Does the dataset include a diverse range of epilepsy patients to ensure its representativeness? Can the authors provide detailed statistics on the dataset, such as patient age, gender, and epilepsy type?
6. Details regarding data preprocessing and collection methods could be further refined. Adding easy-to-understand flowcharts or tables would help readers better understand the dataset construction process.

### Strengths
1.The Omni-iEEG dataset contains data from multiple epilepsy centers, offering a large sample size and diversity, which better represents different populations and pathological types.
2.The dataset's annotations are performed by experts, ensuring high quality and accuracy, making it suitable for machine learning model training and clinical applications.
3.The dataset provides standardized benchmark tests, enabling other researchers to evaluate the dataset easily, promoting model and method comparisons and validations in future studies.
4.The dataset has wide applications in epilepsy detection, prediction, and seizure foci identification, with great potential to advance research in related fields.
5.The dataset will be made publicly available, promoting wide usage in both academia and industry, encouraging data sharing and collaboration.

### Weaknesses
1.The paper focuses mainly on dataset construction, while ICLR typically emphasizes innovation in algorithms, models, or methods. The dataset's contribution does not highlight any novelty in algorithms or methods, which may not meet ICLR's review standards.
2.No concrete application scenarios presented: While the paper mentions the potential applications of the dataset, it does not demonstrate its actual effectiveness or value through specific benchmark tests or application scenarios, lacking real-world examples to support its claims.
3.Insufficient detail on annotation accuracy and consistency: Although the paper mentions the annotation process, it does not provide a detailed description of how the accuracy and consistency of annotations are verified, leaving uncertainty about the reliability of the annotations.
4.Lack of diversity and representativeness of the dataset: Although the dataset includes data from multiple centers, it does not adequately address whether it includes patients with various types of epilepsy. Detailed statistics about patients’ age, gender, epilepsy types, etc., are lacking.
5.Insufficient detail on data preprocessing and collection methods: The paper does not provide enough detailed descriptions of the data preprocessing and collection methods, making it difficult for readers to fully understand the dataset construction process. There is a lack of clear flowcharts or tables to support this.
6.Unclear data release plan: While the paper mentions that the dataset will be made publicly available, it does not specify the exact release methods and platforms, lacking transparency and a clear plan for making the dataset accessible.

### Questions
1. The paper seems to primarily focus on the construction of the dataset, while ICLR typically places more emphasis on algorithmic, model, or methodological innovations. How does this dataset significantly differ from existing EEG/SEEG datasets? If it is simply an extension of existing datasets, can the authors clarify its novelty more specifically?
2. The paper mentions that the dataset will be made publicly available. Could the authors further clarify the dataset's release method and the specific platform for access?
3. While the paper mentions the potential applications of the dataset, there is no concrete benchmark testing or application scenario presented. Can the authors provide one or two actual use cases or examples to show how this dataset can advance epilepsy detection and prediction models?
4. The quality of the dataset is crucial for its application. While the paper mentions the annotation process and methods, it does not provide detailed information about the accuracy and consistency validation of these annotations. Could the authors elaborate on this aspect?
5. Does the dataset include a diverse range of epilepsy patients to ensure its representativeness? Can the authors provide detailed statistics on the dataset, such as patient age, gender, and epilepsy type?
6. Details regarding data preprocessing and collection methods could be further refined. Adding easy-to-understand flowcharts or tables would help readers better understand the dataset construction process.

### Soundness
2

### Presentation
2

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
The paper assembles a multi-center presurgical iEEG benchmark by merging several public cohorts, harmonizing metadata/recordings, and adding expert-validated event labels (spike-associated HFOs). It defines clinically motivated tasks (event classification; pathological channel/patient-level analyses tied to SOZ/resection/outcome), proposes subject-level splits, and reports baselines spanning biomarker-centric pipelines and long-context end-to-end models. The authors state that data, code, and checkpoints will be released.

### Strengths
1. Consolidates fragmented iEEG datasets into a unified benchmark with consistent structure and task definitions, which could materially improve comparability in the area.

2. Tasks and evaluation targets are tied to familiar clinical surrogates, increasing practical relevance.

3. Adds a sizable layer of expert-validated event annotations (spkHFOs) with a described protocol and agreement checks.

4. Includes both biomarker-driven and long-context end-to-end baselines, highlighting trade-offs and opening room for future work.

5. If released with strong artifacts (schema, loaders, splits, checkpoints), the resource can become a de-facto standard.

### Weaknesses
1. It’s hard to separate what is newly curated/validated post-merge (re-labeling, unified clinical ontology, normalized resection masks, QC decisions) from what is simply inherited. Please enumerate concrete new artifacts.

2. Pooled or random subject splits are insufficient for a multi-center resource. The paper needs leave-one-center-out/per-center reporting for the primary tasks, not only a subset, to demonstrate robustness to site effects.

3. “Harmonized” is described procedurally (e.g., resampling, montage, channel cleaning), but there’s little quantitative evidence that results are stable to these choices (referencing/resampling/artifact policy). Short ablations would increase trust.

4. SOZ/resection/outcome fields appear inherited rather than re-adjudicated. Without normalization across centers, surrogates may encode site-specific conventions; sensitivity analyses to alternative definitions would help.

5. The set omits a canonical EEG CNN (e.g., EEGNet/DeepConvNet-class) or a clear rationale for excluding it; a simple linear baseline on strong features would also anchor expectations.

6. Code (preprocessing, splits, evaluation) and exact split files/checkpoints are not available during review; data access is deferred. For a benchmark paper, this materially limits verifiability

### Questions
1. Provide a bullet list of post-merge artifacts created by the authors (new labels, unified ontologies, resection mask normalization, QC) versus fields inherited unchanged.

2. Report leave-one-center-out and per-center results for all primary tasks, with thresholds fixed on training centers and calibration (e.g., reliability) reported.

3. Add brief ablations for referencing, resampling rate, and artifact policy to show conclusions are not artifacts of these choices.

4. Summarize the inter-rater protocol for event labels and how detector-seeded candidates avoid biasing the class distribution; include agreement statistics and adjudication steps.

5. Share (or commit to camera-ready) the repository with preprocessing/evaluation code, exact split files (including center IDs), model checkpoints, and prediction files used to compute tables; include a datasheet/dataset card and a concrete data availability statement (host, license/access path, date).

6. Either add a canonical EEG CNN (or justify its omission) and a simple linear baseline on robust features, or clearly explain why they are not applicable here.

### Soundness
3

### Presentation
3

### Contribution
2
