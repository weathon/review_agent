# DockedAC: A Dataset with Comprehensive 3D Protein-ligand Complexes for Activity Cliff Analysis

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Artificial intelligence has become a crucial tool in drug discovery, excelling in tasks such as molecular property prediction. However, an activity cliff---a phenomenon where a minor structural modification to a molecule leads to a significant change in its biological activity---poses a challenge in predictive modeling. The activity cliff depends on the interaction between the target and the ligand, which is largely overlooked by previous ligand-centric studies. However, the limited availability of activity cliff data for target-ligand 3D complexes constrains the predictive power of modern deep learning models. In this paper, we introduce DockedAC, a new dataset incorporating the protein target and 3D complex structure information for studying the problem of activity cliffs. By matching protein binding information and ligand bioactivity, we employ molecular docking to generate the complex structure for each activity value. The DockedAC dataset contains 82,836 activity data across 52 protein targets annotated with activity cliff information. This dataset represents a significant step toward large-scale activity cliff research using 3D complex structures. We benchmark the dataset with traditional machine learning and deep learning approaches. Our data and benchmark platform are publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DockedAC, a novel dataset designed for activity cliff (AC) prediction that incorporates 3D protein-ligand complex structures generated via molecular docking. The dataset is substantial, comprising over 82,000 ligands across 52 protein targets. The authors provide a comprehensive benchmark comparing traditional machine learning methods, 2D/3D deep learning models, and analyze key factors influencing AC prediction, such as target dependence and the proportion of ACs in the training data. The primary motivation is to address the data scarcity in structure-based AC modeling, moving beyond the traditional ligand-centric view.

### Strengths
Novel and Valuable Contribution: The creation of a large-scale, publicly available dataset for structure-based AC prediction is a significant contribution to the field. The provided benchmark is extensive and will be a valuable resource for the community.

Rigorous Dataset Construction: The process for building the dataset is well-documented and involves careful steps for data cleaning, AC identification, target-structure mapping, and complex generation. The use of multiple similarity measures and a strict binding site verification process is commendable.

Interesting Empirical Findings: The paper presents compelling results, such as the superior performance of 3D GNNs (like IGN) on AC prediction and the critical influence of the AC ratio in the training set, which highlights the unique challenges of this task compared to general bioactivity prediction.

### Weaknesses
1. Lack of Justification vs. Existing Docking Datasets (Major): The paper does not adequately justify why a new docking dataset was necessary, given the existence of several recent, large-scale protein-ligand docking datasets. For example:

    A High-Quality Data Set of Protein-Ligand Binding Interactions Via Comparative Complex Structure Modeling,  DOI: 10.1021/acs.jcim.3c01170

    SIU: A Million-Scale Structural Small Molecule-Protein Interaction Dataset for Unbiased Bioactivity Prediction, https://doi.org/10.48550/arXiv.2406.08961

    SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset, https://doi.org/10.1101/2025.06.17.660168

    The unique value of DockedAC lies in its explicit focus on curated Activity Cliffs, but this contribution can be built upon these other resources. 

2. Questionable Data Homogeneity (Major): A critical issue is the assumption that bioactivity data collected from different sources and assays across ChEMBL can be averaged and are directly comparable. Biochemical assays can vary significantly in conditions, leading to systematic biases. Aggregating this data without accounting for assay-type or laboratory-of-origin can introduce noise and confound the model's ability to learn true structure-activity relationships. The authors should at a minimum discuss this limitation and consider if assay-aware splitting or normalization could be applied.

3. Incomplete and Outdated Baselines (Major): The benchmark, while comprehensive in some aspects, omits several state-of-the-art (SOTA) methods that are highly relevant. For example:

    Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction, doi: https://doi.org/10.1101/2025.06.14.659707

    LigUnity: A foundation model for protein-ligand affinity prediction through jointly optimizing virtual screening and hit-to-lead optimization, doi: https://doi.org/10.1101/2025.02.17.638554

    The inclusion of these strong baselines is essential to properly position the performance of the tested models and to demonstrate the current challenge level posed by the DockedAC dataset.

4. Clarity of Visualizations: Figure 1, as an example, the protein residues in the 3D structure diagram (Figure 1b) should have atoms (C, N, O) colored differently (e.g., gray, blue, red) to allow readers to unambiguously identify the hydrogen bond donors and acceptors. The current coloring makes it difficult to verify the claimed hydrogen bond interactions.

### Questions
1. How does DockedAC docking process specifically complement or surpass existing large-scale docking datasets like BindingNet, SIU or SAIR. Why not generate activity cliff pairs on these existing datasets?

2. Given the known variability in biochemical assays, have you considered an assay-centric data split to evaluate the impact of assay bias?

3. Why was the decision made to use a 0.99 similarity threshold for PDB template matching, rather than a lower threshold (like 0.9) that would include structurally similar ligands and increase dataset coverage?

4. What was the rationale for not including recent SOTA baselines like Boltz-2 or LigUnity in your benchmark? Would you be open to adding them in a future version?

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
This paper introduces DockedAC, a new dataset designed to tackle the activity cliff prediction problem, which is a major challenge in molecular machine learning. The dataset provides 3D docked poses of protein–ligand complexes for over 17000 activity cliff pairs across 58 protein targets. It combines structure-based information via docking with traditional ligand-based representations. The authors evaluate multiple baseline models, including deep learning and fingerprint-based methods, and benchmark their performance primarily using RMSE. The results highlight the difficulty of predicting activity cliffs.

### Strengths
1. The motivation of the introduced dataset is reasonable as it addresses a meaningful challenge in drug discovery, activity cliffs, which are often overlooked in traditional benchmarks.
2. Some baseline methods are tested and some insights are discussed

### Weaknesses
1. Model variety is limited. It would be beneficial to include models with stronger inductive biases, such as equivariant models like different EGNN, as well as various pretrained models.
2. Evaluation metrics are insufficient. Relying solely on RMSE is not ideal. Additional metrics such as pairwise ranking accuracy and correlation coefficients should be included to provide a more comprehensive evaluation.
3. Fingerprint-based models outperforming deep learning models is concerning. This suggests that the dataset may allow models to overfit to individual molecular features rather than learning true interaction patterns. As a result, the structural information introduced by the dataset might not be contributing as intended, which weakens the overall contribution of the work.
4. Dataset splitting strategy is unclear. The paper does not explain how the dataset was split, which is critical for assessing the validity of the reported results.
5. There is no quality control for the quality of the docking poses.

### Questions
see weaknesses

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
5

### Summary
The work introduces a new dataset and benchmark called DockedAC. DockedAC comprises over 80,000 docked protein-ligand complexes with activity cliff annotations. Feature-based traditional machine learning approaches and 1D, 2D, 3D based deep learning methods are compared. Further analysis is performed following the prior work by van Tilborg et al. (MoleculeACE). 

I recommend the rejection of the paper in its current form, for the following reasons:
1. There is no evident contribution by DockedAC.
2. Dataset quality is ambiguous
3. The discussion of the results needs to be more balanced and in-depth.

### Strengths
1. A wide range of models are tested on the dataset, including traditional ML and various DL approaches on 1D, 2D, and 3D representations.
2. The data curation pipeline is described in detail and appears sound. 
3. The writing is clear.
4. The final dataset is of a considerable size to train structure-based binding affinity prediction models.

### Weaknesses
1. The performance of the docking procedure is not evaluated. Without this information, it is impossible to assess the quality of the dataset, as well as why the 3D models perform worse than simpler ML models. I strongly recommend including these aspects, e.g., via enrichment analysis.
2. The analysis in the paper closely resembles the MoleculeACE paper (e.g., delta RMSE, impact of AC percentage, dataset set size, protein family analysis), and confirms its findings. Analysis from new perspectives, or discovering new insights, is needed to introduce added value.
3. The text overstates the contribution of the work. For example, the initial analysis begins with a comparison of 3D methods against only 2D models, emphasizing the importance of 3D information. However, traditional ML approaches using ECFP fingerprints outperform all 2D and 3D models, as shown and briefly discussed in Figure 6. This is not a surprising finding (as written in L431) and reported in the MoleculeACE paper and other literature already. The authors should clearly discuss when 3D models underperform simpler models, and analyze the reasons behind this. The analysis should also start from a comparison against the strongest baselines, not the weakest ones.
4. Activity cliffs might not be the most relevant topic for ICLR researchers, as it is a rather specific cheminformatics concept. A technical journal might be a more suitable venue for this work.

### Questions
1. What is the correlation between AC performance and overall performance per architecture and target? Is a separate AC analysis still needed if the correlation is very high? 
2. What is the quality of the generated complexes? Please provide an analysis of the docking performance.
3. What is the added value introduced by DockedAC dataset? All contributions written in the introduction are already covered by previous work.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DockedAC, a new dataset for studying activity cliffs using 3D protein–ligand complex structures. The dataset integrates over 82,000 activity measurements across 52 protein targets, with corresponding docked complexes generated using molecular docking tools. Activity cliffs are identified through combined molecular similarity criteria (substructure, scaffold, and SMILES similarity) and bioactivity differences greater than one order of magnitude. The paper benchmarks a range of classic machine learning and deep learning methods—including traditional machine learning models using molecular descriptors, 2D GNNs, and 3D structure-based GNNs—to evaluate their performance on AC-specific prediction tasks.

### Strengths
- The data curation pipeline---from ChEMBL data extraction and activity-cliff identification to docking-based complex generation---is clearly presented and thoughtfully designed. 

- The study addresses the underexplored challenge of modeling activity cliffs, which represents an important problem in drug discovery. 

- The inclusion of both RMSE and RMSE$_{cliff}$ as evaluation metrics provides valuable insights into model performance specifically on activity-cliff compounds.

### Weaknesses
1. The double-stratified sampling strategy (in Section 3.5) used for dataset splitting can lead to information leakage, as structurally similar ligands may appear in both the training and test sets. This setup prevents a fair evaluation of the model’s generalization ability on unseen small molecules.

2. One of the paper’s main contribution is the construction of protein–ligand complexes. However, the experimental results show that ligand-only models, which do not use these 3D complexes at all, perform better. This discrepancy weakens the logical connection between the paper’s motivation and its findings. The authors should clarify why the 3D complexes fail to improve performance, or provide analyses showing where 3D information offers advantages.

3. Minor: IC$_{50}$, K$_i$, and K$_d$ should be written in upright (non-italic) font; Database names like "ChEMBL" should maintain consistent capitalization throughout the manuscript.

### Questions
1. DTA/DTI tasks inherently involve two domains: the small-molecule side and the target protein side. The current work focuses primarily on generalization across proteins but pays limited attention to ligand-level de-duplication and generalization. Since activity cliffs mainly concern fine-grained distinctions between structurally similar ligands against the same target, ligand-side analysis should be emphasized more strongly.

2. The baseline models are relatively outdated. In addition to classical machine-learning and basic GNN models, recent pretrained molecular representation models (e.g., Uni-Mol) should be included for a fair comparison. Similarly, more recent DTA/DTI models could serve as relevant structure-based baselines.

3. How do the authors justify the conclusion that “current deep learning methods underperform compared to traditional machine learning approaches using molecular fingerprints, underscoring the need for next-generation 3D-QSAR algorithms”? There seems to be a substantial gap between the observed results and this claim. For instance, do deep learning models—particularly GNN-based ones—require larger data volumes to outperform traditional models? Is per-target QSAR modeling inherently more effective than cross-target modeling? Or do handcrafted descriptors such as ECFP offer unique advantages for activity-cliff prediction? A more systematic analysis would be needed to substantiate this conclusion.

### Soundness
1

### Presentation
3

### Contribution
2
