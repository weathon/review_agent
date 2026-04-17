# MolPILE - large-scale, diverse dataset for molecular representation learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
The size, diversity, and quality of pretraining datasets critically determine the generalization ability of foundation models. Despite their growing importance in chemoinformatics, the effectiveness of molecular representation learning has been hindered by limitations in existing  small molecule datasets. To address this gap, we present MolPILE, large-scale, diverse, and rigorously curated collection of 222 million compounds, constructed from 6 large-scale databases using an automated curation pipeline. We present a comprehensive analysis of current pretraining datasets, highlighting considerable shortcomings for training ML models, and demonstrate how retraining existing models on MolPILE yields improvements in generalization performance. This work provides a standardized resource for model training, addressing the pressing need for an ImageNet-like dataset in molecular chemistry.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents MolPILE, a large-scale dataset of 222 million compounds curated from six public chemical databases. The authors aim to provide a foundation dataset for molecular representation learning. The dataset construction pipeline includes preprocessing, standardization with a QSAR workflow, and physicochemical filtering, followed by evaluation through retraining Mol2vec and ChemBERTa-2 on MoleculeNet and other benchmarks.

### Strengths
- The paper is clearly written and well-organized, with comprehensive background research and thorough comparison to prior datasets.

- The analysis of elemental composition, molecular filters, and property distributions is detailed and informative.

- The motivation—to create a large unified dataset for molecular pretraining—is relevant and valuable for the field.

### Weaknesses
1. **Limited experimental scope:** The evaluation includes only two outdated SMILES-based baselines (Mol2vec and ChemBERTa-2). The absence of modern graph- or 3D-based models (e.g., GROVER, GraphMAE, UniMol) substantially weakens the claim that “MolPILE benefits molecular representation learning”, as the comparison lacks breadth and contemporary relevance.

2. **Minor presentation issues:** Figure 1, while conceptually clear, appears visually simplistic; incorporating schematic icons or stylized elements could enhance interpretability and visual appeal. In addition, Table 1 employs inconsistent numerical units (M/k), which should be standardized for clarity and professional presentation.

### Questions
1. **Feasibility filters appear overly permissive:**

The chosen thresholds (MW ≤ 2500, RB ≤ 60, logP ∈ [−10, 25]) seem inconsistent with the stated  focus on “small molecules”, as they may still include peptides, nucleic acids, polysaccharides, and other large or complex entities. The authors could provide quantitative justification—such as the percentage of entries containing peptide bonds, nucleic acid motifs, or polysaccharide fragments—to demonstrate exclusion.

Additionally, note that logP values in extreme ranges are themselves model-predicted and unreliable; the range [−10, 25] may not be chemically meaningful, as such out-of-distribution (OOD) predictions are typically inaccurate.

2. **Rationale for mixing heterogeneous data sources:**

The dataset combines large experimental databases,  screening catalogs, and natural product databases, which differ significantly in their chemical-space distributions and functional purposes. What is the motivation for merging these domains?

Why did the authors choose to operate at the level of “synthesizable compounds” rather than, for instance, integrating synthetic reaction–based datasets or restricting their scope to drug-like small molecules? Whereas many previous studies focus exclusively on natural products to maintain distributional consistency, what advantages or empirical evidence support the authors’ decision to adopt this scope?

3. **The reported experimental results raise multiple questions:**

    - The absolute performance of the models falls short of current state-of-the-art results.

    - In Table 26, 5 of 8 tasks show negative gains, with overall improvement seemingly driven mainly by ClinTox. Additionally, simply averaging the AUROC gains across different MoleculeNet tasks may not have clear statistical meaning.

    - Some “original” ChemBERTa scores (e.g., 98.36 AUROC on ClinTox in Table 33) diverge sharply from those in the original literature[1]. Could the authors clarify the evaluation setup, number of random seeds, and whether these discrepancies stem from preprocessing, data splits, or metric computation?

[1] Ahmad, Walid, et al. “ChemBERTa-2: Towards Chemical Foundation Models.” arXiv preprint arXiv:2209.01712, 2022

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
This paper presents MolPILE, a new large-scale dataset for molecular representation learning. The authors identify limitations in existing datasets, such as insufficient size, poor quality control, and reduced structural diversity due to excessive filtering. To address these issues, MolPILE compiles 220 million compounds through an automated pipeline, ensuring scale, diversity, and quality. Retraining existing models on MolPILE demonstrates improved generalization, highlighting the dataset's value.

### Strengths
- Clear Motivation: The paper convincingly outlines the limitations of current pretraining datasets and the need for a unified, large-scale chemical dataset. Constructing high-quality datasets is a fundamental step for broadening its application across diverse chemical domains.
- Systematic Data Construction: The multi-stage pipeline—preprocessing, standardization, and filtering—is applied. Employing only minimal and chemically valid filtering enables the construction of a dataset that is both larger in scale and structurally more diverse.

### Weaknesses
- Insufficient Validation of Dataset Quality: Although the paper provides various analyses on size, diversity, and synthesizability, these remain descriptive rather than evidential. It is unclear whether the reported statistics truly demonstrate that MolPILE is a high-quality dataset that improves model learning or generalization.
- Limited Evidence for Generalization: The evaluation covers only a narrow set of models and tasks, leaving it unclear whether MolPILE truly enhances transferability across different molecular learning frameworks.
- Experimental Design Concerns: Mol2vec and ChemBERTa embeddings were used as feature extractors and then fed into a random forest classifier for prediction. The evaluation setup does not clearly reflect common practices and may not accurately capture the true capability of the pretrained models.
- Missing Analysis on Dataset Scaling: MOLFILE provides subsets of different sizes, such as 1M, 5M, and 10M, to support scalable experiments. If the dataset is truly high-quality, it is necessary to verify whether performance improvements are consistently observed across different dataset sizes compared to existing datasets of similar scale.

### Questions
- Generalization Across Methods: Could the authors test MolPILE’s generalization effect by applying it to other molecular learning methods, using each method’s original experimental setup, to verify whether similar improvements are consistently observed?
- Element Group Analysis: Figure 2 shows only marginal differences in element distributions across datasets. Does this actually lead to broader chemical coverage or measurable performance gains?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MolPILE, a large-scale dataset of approximately 222M experimentally validated small molecules designed for molecular representation learning. The authors describe a standardized multi-stage curation pipeline for data cleaning, structure normalization, and feasibility filtering. They analyze MolPILE in terms of size, chemical diversity, and synthesizability, and demonstrate that retraining existing models (Mol2vec and ChemBERTa) on MolPILE leads to improved performance across a wide range of molecular property prediction benchmarks.

### Strengths
1. The work targets an important problem in molecular ML by proposing a standardized large-scale dataset.

2. The curation pipeline is clearly defined, reproducible, and appropriate for large-scale preprocessing.

3. Extensive dataset quality and diversity analyses are provided.

### Weaknesses
1. Effect of Data Scale vs. Data Curation is Not Clearly Disentangled.

The performance gains reported for Mol2vec and ChemBERTa may be influenced heavily by the scale difference rather than the data quality. The paper does not control for dataset size when comparing pretraining sources. For instance, recent versions of ZINC and PubChem offer significantly larger collections than the proposed dataset, yet the authors do not include experiments where MolPILE and alternative datasets are sampled to matched scales. As a result, it remains unclear whether the improvements stem from the dataset’s curation pipeline (diversity, structure normalization, feasibility filtering) or simply from training on more data.

2. Limited Model Coverage Restricts the Strength of Claims.

The experiments retrain only two relatively small and older architectures (Mol2vec and ChemBERTa). These models do not represent the current state of the field, which includes large graph transformers, equivariant 3D neural networks, etc. Without evaluating at least one contemporary large model, it is difficult to determine how useful MolPILE will be for modern molecular ML and foundation model scaling.

3. Vocabulary and Tokenization Limitations Not Investigated.

ChemBERTa trained on MolPILE shows degraded performance on ApisTox, attributed to tokenization limits (vocabulary not covering rare structural patterns). However, the authors do not evaluate alternative tokenization strategies (e.g., SMILES BPE scaling, SELFIES, fragment tokenization), nor do they quantify how vocabulary coverage correlates with downstream performance. This leaves unclear how well MolPILE supports SMILES language modeling when chemical space becomes highly diverse.

4. No Ablation on Filtering Pipeline.

While the dataset construction pipeline includes multiple important steps (standardization, feasibility filtering, deduplication), the paper does not study how each step contributes to data quality or downstream model performance. An ablation analysis could reveal, for example, whether feasibility filtering meaningfully improves representation learning, or whether diversity selection is the primary factor.

### Questions
1. Can the authors perform controlled experiments where MolPILE is downsampled to the same size as baseline datasets (e.g., ZINC subsets), in order to demonstrate that performance improvements arise from the curation pipeline rather than dataset scale alone?

2. How would MolPILE perform when evaluating MolPILE-pretrained models on 3D-aware benchmarks or reaction prediction tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MolPILE, a large-scale, diverse, and rigorously curated dataset for molecular representation learning, developed to overcome the size, diversity, and quality limitations of existing small-molecule datasets.

### Strengths
1. With 222 million compounds, MolPILE achieves a far larger scale than commonly used datasets like ChEMBL (2.4M) and ZINC (12.7M), providing sufficient data volume for training large-scale foundation models
2. Second, it exhibits superior structural and molecular type coverage—covering halogens, metalloids, metals (unlike ZINC and GDB-17, which lack metals/metalloids), over 3.6 million unique Bemis-Murcko scaffolds, and 1.09 million salts, effectively expanding the chemical space for model pretraining 
3. The authors validate the dataset’s value by retraining 1D-based models (Mol2vec and ChemBERTa) on MolPILE, showing consistent performance gains across 48 benchmarks, demonstrating its effectiveness for 1D molecular representation learning

### Weaknesses
1. MolPILE only provides 2D molecular representations (SMILES/InChI) without 3D conformations, restricting its applicability to 3D-based pretraining methods or multimodal fusion approaches; the validation is solely focused on 1D models, failing to support broader molecular learning paradigms.
 
2. While the dataset claims high diversity (e.g., metal-containing compounds, improved structural diversity with ~3% normalized #Circles), there is a lack of targeted validation on downstream tasks specific to these diverse subsets—for example, no dedicated experiments on agrochemistry tasks (despite covering metals) or natural product property prediction (despite including SuperNatural3/COCONUT data) to confirm if the enhanced diversity translates to task-specific improvements. 

3. Although scale is a core highlight, the paper lacks detailed ablation studies on molecular data scaling laws. It does not systematically demonstrate whether increasing data volume leads to stepwise improvements in molecular property prediction accuracy for classical pretraining methods, which is essential to justify the necessity of such a large-scale dataset.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
