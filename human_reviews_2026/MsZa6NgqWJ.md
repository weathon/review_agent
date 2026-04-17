# Comprehensive Benchmark for Tailored Small Molecule-Binding Aptamer Design

- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Despite their growing role as recognition elements in diagnostics, therapeutics, and biosensing, aptamers remain overlooked by computational design tools compared to antibodies and protein binders. Current pipelines are fragmented and predominantly protein-focused, leaving small-molecule aptamer discovery underexplored. A key bottleneck has been the absence of a unified benchmark dataset that would allow systematic evaluation of predictive and generative models. To address this gap, we introduce the first comprehensive benchmark for aptamer–small molecule interactions, integrating eight curated sources into 6,413 annotated pairs covering 1,686 unique aptamers (DNA and RNA) and 1,041 chemically diverse ligands. More than 30\% of the entries include quantitative binding affinities, enabling not only binary classification but also regression. To demonstrate the utility of this resource, we establish baseline results across shallow and deep learning baseline models under multiple splitting protocols. Our analysis reveals two central findings. First, the diversity and coverage of aptamer sequences are sufficiently broad to support robust modeling, indicating that limitations in current approaches do not stem from the receptor side. Second, the primary bottleneck emerges from the ligand space: a comparatively small number of molecules spans a highly heterogeneous chemical landscape, substantially constraining model transferability to unseen targets. Since practical aptamer design ultimately requires generalization to novel small molecules, these observations underscore a fundamental representation challenge on the ligand side. By providing a standardized corpus, rigorous evaluation protocols, and reproducible baselines, our benchmark establishes a foundation for systematic progress in aptamer-small molecule prediction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces what I think is one of the moore valuable benchmark efforts in biomolecular machine learning to date — a comprehensive, well-curated dataset for small-molecule aptamer prediction and design (a really cool but understudied problem). The authors integrate seven previously fragmented databases into a unified corpus of 2,001 annotated aptamer-ligand pairs covering 1,309 unique aptamers and 479 distinct ligands. About half of the pairs include quantitative affinities, enabling both regression and classification tasks. The benchmark defines three clear evaluation protocols (stratified, aptamer-disjoint, molecule-disjoint) and establishes baseline models ranging from LightGBM and Random Forests to deep learning architectures built from GENA-LM (aptamer encoder) and ChemBERTa (ligand encoder). The results are strong and well contextualized. The authors show that aptamer-side representations are largely sufficient, while the main bottleneck lies in ligand diversity and representation learning. I think this work fills a critical gap and sets the stage for meaningful and reproducible model development in aptamer–small molecule modeling.

### Strengths
- This paper fills a long-standing void in the field by creating a unified benchmark for small-molecule aptamer modeling, which is understudied but very important in drug development.  

- The integration effort across seven sources is handled with impressive care: deduplication, cleaning, balanced coverage of DNA and RNA aptamers, and inclusion of both binary and quantitative data.  

- The authors also actually define meaningful split protocols (very helpful) that reflect realistic scientific use cases, such as predicting for unseen aptamers or entirely new ligands.
  
- I really appreciate how they contextualize the ligand bottleneck and back it with quantitative evidence, rather than hand-waving about model underperformance.  

- The baseline analyses are thorough and transparent, showing that even simple LightGBM models outperform deep learning approaches under realistic molecule-disjoint splits, which I find refreshingly honest.  

- The writing and referencing are strong, and I do believe this paper will serve as a foundation for future predictive and generative work in this area.

### Weaknesses
- Accessibility could be improved. The dataset, while well described, should be hosted in a public, easy-to-use format (like HuggingFace Datasets) with aptamer sequences, SMILES/SELFIES representations of ligands, and both RNA secondary and tertiary structures to support broader use and model training. If the authors can commit to this, I will be supportive of acceptance.

- The negative sampling approach feels heuristic. It would be helpful if they could better justify or statistically evaluate the assumption that cross-paired negatives represent true non-binders. The authors should incorporate true negatives from experimental works as a "gold standard" hold-out evaluation set.

- I’d like to see the authors plan for versioned updates as new aptamer–ligand data become available, since this benchmark could easily become the go-to reference for the community and will need maintenance. Again, the HuggingFace database strategy would be perfect here.

- The manuscript could clarify small details around reproducibility (e.g., fixed seeds, data split scripts, preprocessing code) to make replication more straightforward.  

- I do think that including a few structure-aware baselines (like, graph neural network encoders for ligands or RNA folding-based embeddings) could make the benchmark even more useful for multimodal learning. Though, I do agree that the more sequence-based approaches will be more robust.

### Questions
1. The authors should release the dataset on HuggingFace with both sequences and ligand structures (SMILES/SELFIES) to facilitate training across modalities. 

2. The authors should also add predicted RNA secondary and tertiary structures, perhaps using tools like RNAfold or Rosetta, to make the dataset more suitable for structure-informed modeling.

3. The handling of false negatives when generating synthetic negative pairs should be clarified. I am hesitant to recommend acceptance without stronger guarantees on the negative pairs.

4. The authors should include generative benchmarks (e.g., inverse design or sequence-conditioned binding optimization) in future versions of this resource.
 
5. Is there a plan for continuous updates or community-driven extensions as new aptamer–ligand pairs are experimentally validated? The authors should discuss this and have a solid plan in place for updates.

If the authors can satisfactorily answer these questions, I will raise my score to an 8.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a benchmark for modeling aptamer–small-molecule interactions, combining seven publicly available databases into a unified corpus of 2,001 aptamer–ligand pairs (1,309 unique aptamers and 479 ligands). Approximately half of these entries include quantitative binding affinities, enabling both classification and regression tasks.
Three evaluation protocols—stratified, aptamer-disjoint, and molecule-disjoint—are defined to simulate different discovery scenarios. The authors evaluate tabular machine-learning models alongside pretrained deep encoders and conclude that traditional feature-based methods outperform end-to-end deep architectures. The stated goal is to provide a reproducible benchmark to facilitate systematic progress in aptamer–ligand modeling.

### Strengths
- Addresses an unexplored yet meaningful problem domain where no standardized dataset currently exists.

- Draws attention to an important insight—the ligand-side representation is likely the dominant limitation in current aptamer-ligand prediction models.

### Weaknesses
1. The negative sampling approach is conceptually inconsistent with the task definition. Labeling untested aptamer–molecule pairs as negatives imposes a prior assumption that contradicts the goal of predicting binding for unobserved pairs. This procedure effectively predefines the outcome the model is meant to infer, introducing systematic bias into the training process, even if the assumption is limited to model training rather than evaluation.


2. With respect to line 054, while I acknowledge that the authors position aptamer design as the reverse of structure-based small-molecule design and recognize the associated challenges, the assertion that this task is necessarily more difficult is not sufficiently supported. The chemical search space of drug-like molecules, estimated at approximately 10⁶⁰ candidates, already poses severe scalability constraints. Furthermore, the paper’s results suggest that a dataset of only about 2 k samples is “sufficient” to support aptamer modeling, which appears inconsistent with the earlier argument emphasizing the intrinsic difficulty and vast search space of the task.


3. The integration process appears to be a mere merging of existing datasets with minimal quality control or validation.


4. Minor: "Binding strength" should be replaced with "binding affinity"; several abbreviations are undefined on first use; and Figure 3 is visually unclear, edges overlap with nodes, and it is unclear whether certain lines represent meaningful interactions or merely graphical artifacts.

### Questions
1. Could you add a comprehensive overview figure (e.g., as Figure 1) summarizing the benchmark construction pipeline, data sources, and biological context of aptamers, to help readers from machine-learning backgrounds quickly understand the workflow?

2. What criteria were used for test set selection and construction? How were the training and validation sets deduplicated, and were scaffold-based deduplication for ligands or sequence-similarity-based deduplication for aptamers applied, as is standard in DTI/DTA benchmark design—particularly given that the paper emphasizes the analysis of “generalization”?

3. How are the terms “sufficient” and “robust modeling”, as used in the abstract, formally defined? Please clarify the criteria or empirical evidence supporting these claims of modeling adequacy.

4. Please report the detailed t-SNE parameters (e.g., number of iterations, perplexity, learning rate) and explain how the visualizations support the claim of limited generalization, given that the displayed distributions appear uniform and well mixed.

5. Why were state-of-the-art deep-learning baselines (cited in Related Work), especially for the regression task, not included in the evaluation?

6. How were binding-affinity measurements harmonized across databases? Were there normalization or outlier-removal steps applied to ensure consistency?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Aptamers are short single-stranded DNA or RNA sequences capable of binding to small molecules, proteins, and other targets with high specificity and affinity. They have broad applications in diagnostics, therapeutics, and biosensing. However, compared to protein-targeting aptamers, small-molecule-binding aptamers have been largely overlooked in computational design. Existing datasets are fragmented, inconsistently annotated, and lack a unified benchmark, which hinders the development of machine learning models in this direction. This paper introduces the benchmark dataset for small molecule–aptamer interactions, supporting both classification and regression tasks. It aims to evaluate various shallow and deep learning models, identify current modeling bottlenecks.

### Strengths
1. Provide a standardized, curated dataset with consistent annotations, enabling fair comparison and reproducible research.
2. Introduces aptamer-disjoint and molecule-disjoint splits to assess model generalization to new aptamers and new molecules, respectively, reflecting real-world design challenges.
3. Compares shallow and deep models across tasks and splits, showing that simple models outperform deep learning due to the poor transferability of molecular embeddings.

### Weaknesses
1. Despite high chemical diversity, the small molecule set is still too small for training or evaluating large-scale or pre-trained models effectively (479 Unique Molecules).
2.  Current DL models perform worse than LightGBM on molecule-disjoint splits, but the paper does not deeply analyze why.
3. Does not include graph neural networks, 3D-aware models, or contrastively pre-trained molecular encoders, which may offer better generalization.
4. Negative samples are generated by cross-pairing unobserved aptamer–molecule pairs, which may not truly be non-binding, potentially introducing label noise.

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the first comprehensive benchmark for aptamer-small molecule interactions, integrating seven data sources into 2,001 annotated pairs covering 1,309 unique aptamers and 479 ligands. Approximately 58% of entries include quantitative binding affinities, enabling both classification and regression tasks. The authors establish baseline results using shallow ML (LightGBM, MLP, RF) and deep learning models under three splitting protocols (stratified, aptamer-disjoint, molecule-disjoint). Their analysis reveals that aptamer sequence coverage is sufficient for robust modeling, while the main bottleneck arises from ligand space, where a small number of structurally diverse molecules limits model transferability. While this benchmark addresses an important gap and provides a valuable resource, significant limitations around dataset size, label quality, and modest performance raise questions about its utility for driving meaningful progress.

### Strengths
The paper addresses an important gap in computational aptamer design by creating the first standardized benchmark for aptamer-small molecule interactions. The dataset integration from seven diverse sources with both 2D sequences, 3D conformations, and functional labels represents substantial curation effort. The three splitting protocols (stratified, aptamer-disjoint, molecule-disjoint) are well-designed to evaluate different real-world scenarios, and the rigorous evaluation methodology includes multiple folds, proper error bars, and appropriate metrics (ROC-AUC, MCC) for imbalanced data.
The baseline evaluation is thorough, systematically comparing multiple aptamer representations (k-mers, one-hot, pretrained GENA-LM embeddings) and molecular descriptors (Morgan fingerprints, MACCS keys, RDKit descriptors, ChemBERTa) across both shallow and deep learning architectures. The analysis clearly identifies that ligand representation, not aptamer representation, is the primary limiting factor, a valuable insight for future method development. The inclusion of both classification and regression tasks, along with detailed appendices documenting architectures and hyperparameters, supports reproducibility.

### Weaknesses
The dataset contains only 479 unique small molecules, which is extremely small compared to drug discovery benchmarks (MoleculeNet has 100,000+ molecules). This is compounded by extreme class imbalance (1,842 positives vs 159 negatives = 92% positive). While synthetic negative sampling addresses training balance by cross-pairing aptamers with unrelated molecules, this introduces substantial label noise that is neither quantified nor addressed. The paper acknowledges these synthetic negatives "may still bind but remain uncharacterized," but provides no estimate of false negative rates or validation of the assumption that arbitrary pairs don't bind. With t-SNE analysis showing ligands are "dispersed and lack tight clustering" combined with only 479 molecules, the dataset may be fundamentally too small and diverse to support robust generalization.

The best LGBM model achieves MCC ≈ 0.70 under grouped CV but drops dramatically to 0.34-0.38 under molecule-disjoint splits a 50% performance degradation. Deep learning performs even worse (MCC ≈ 0.41 grouped, 0.18 molecule-disjoint). Since the stated goal is "designing aptamers for previously unseen molecules," these molecule-disjoint results are quite discouraging. The paper correctly identifies ligand representation as the bottleneck but offers only vague suggestions ("graph neural network encoders") without testing them. If the best baseline achieves MCC 0.34-0.38 on molecule-disjoint splits, this benchmark may have limited utility for driving progress unless the authors can demonstrate that better methods exist or provide clearer guidance on representation improvements.

For a benchmark paper, there is surprisingly little analysis of what chemical features correlate with binding. Feature importance analysis shows "specific k-mer indices and several Morgan bits" but no chemical interpretation. No analysis of which functional groups, charges, sizes, or structural motifs predict binding. No investigation of why Morgan fingerprints match or outperform ChemBERTa embeddings under molecule-disjoint splits (Table 1: 0.384 vs 0.342 MCC), which suggests pretrained molecular embeddings don't capture binding-relevant features. This limits the benchmark's utility for guiding future method development—researchers need to know what kinds of representations and architectures are likely to improve performance.

While 58% of data has quantitative Kd values, regression receives minimal attention. Results appear in only one table (Table 3) with limited discussion. RMSE of 2.42 pKd units is quite large, spanning ~2-3 orders of magnitude in actual Kd values. No analysis of whether regression performance varies by molecule type, aptamer length, binding affinity range, or data source. No comparison to structure-based methods (molecular docking) for the subset with structural data. No investigation of whether classification and regression tasks could be jointly optimized or whether regression could provide auxiliary supervision.

The paper briefly mentions that "docking and molecular dynamics simulations...remain computationally prohibitive for large-scale aptamer screening" but provides no actual comparison. For molecules and aptamers with available structural data, how does the best ML model compare to molecular docking in terms of both accuracy and computational cost? This would help establish whether ML is competitive, complementary, or still inferior to physics-based methods. Given that DEL-Dock was evaluated in the previous paper I reviewed, there should be comparable aptamer docking tools available.

The dramatic performance drop from grouped CV (MCC 0.70) to molecule-disjoint (MCC 0.34) deserves deeper investigation. Is this gap due to: (1) insufficient molecular diversity in training, (2) poor ligand featurization, (3) overfitting to training molecules, or (4) fundamental limitations of the approach? Learning curves showing performance vs number of unique training molecules would help diagnose the issue. Analysis of which molecule types are hardest to generalize to would guide data collection efforts.

### Questions
This paper makes a valuable contribution by creating the first standardized benchmark for aptamer-small molecule interactions, which addresses an important gap in the field. The dataset curation represents substantial effort, the evaluation protocols are well-designed, and the identification of ligand representation as the primary bottleneck is a useful insight. 

However, several critical limitations prevent a stronger recommendation. The dataset is small (479 molecules) with extreme class imbalance (92% positive), and the synthetic negative sampling strategy introduces unquantified label noise. More concerning, the best methods achieve only modest molecule-disjoint performance (MCC 0.34-0.38), and the paper provides limited guidance on how to improve beyond vague suggestions. The deep learning architectures lack interaction modeling despite binding being fundamentally an interaction problem. The lack of comparison to structure-based methods and limited chemical interpretation of results further limit the benchmark's utility.
The paper would be strengthened by: (1) validation or bounding of false negative rates in synthetic negatives, (2) testing interaction-aware DL architectures, (3) deeper analysis of what chemical features predict binding, (4) comparison to molecular docking, and (5) clearer guidance on promising directions for improvement.

### Soundness
3

### Presentation
3

### Contribution
3
