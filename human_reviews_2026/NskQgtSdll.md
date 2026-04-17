# PepBenchmark: A Standardized Benchmark for Peptide Machine Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Peptide therapeutics are widely regarded as the “third generation” of drugs, yet progress in peptide Machine Learning (ML) are hindered by the absence of standardized benchmarks. Here we present \textbf{PepBenchmark}, which unifies datasets, preprocessing, and evaluation protocols for peptide drug discovery. PepBenchmark comprises three components: (1) \textbf{PepBenchData}, a well-curated collection comprising 29 canonical-peptide and 6 non-canonical-peptide datasets across 7 groups, systematically covering key aspects of peptide drug development—representing, to the best of our knowledge, the most comprehensive AI-ready dataset resource to date; (2) \textbf{PepBenchPipeline}, a standardized preprocessing pipeline that ensures consistent dataset cleaning, construction, splitting, and feature transformation, mitigating quality issues common in ad hoc pipelines; and (3) \textbf{PepBenchLeaderboard}, a unified evaluation protocol and leaderboard with strong baselines across 4 major methodological families: Fingerprint-based, GNN-based, PLM-based, and SMILES-based models. Together, PepBenchmark provides the first standardized and comparable foundation for peptide drug discovery, facilitating methodological advances and translation into real-world applications. The data and code are publicly available at \href{https://github.com/ZGCI-AI4S-Pep/PepBenchmark/}{\texttt{https://github.com/ZGCI-AI4S-Pep/PepBenchmark/}}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper provides a collection of standard benchmarks and leaderboards for peptide representation learning. The authors collected and clean a large set of peptides including both canonical and non-canonical peptides. Negative sampling was proposed to extend the peptide datasets for machine learning.  Different splitting strategies were considered, including spliting by kmers and then spltting by sequence similarity using clustering methods. The authors consider different representations of the peptides including simple fingerprints, pretrained smiles representation, pretrained PLM representation (for canonical sequences) and difefrent learning approaches including tree-based ML methods like RF and LightGBM, graph neural networks and finetuning pretrained smiles or PLM models. The datasets, the splits and the benchmark results are reported on the rich datasets. Some conclusions were drawn from the experimental results based on the benchmark experiments.

### Strengths
In peptide representation, there is lack of standard leaderboards with fixed splits so that research in the field can rely on whenever there is a requirement for comparison between different methods. The proposed benchmark and frameworks are useful for that purpose.

The dataset cover broad types of tasks, even the datasets were collected elsewhere, the collection and bringing them to the same place will trigger standard benchmarking for the research field.

The conclusion and benchmarking results are interesting even it is known in the field that current not only limitted to non-canonical peptide in general for small molecules, fingerprint-based approaches are the best representation. For canonical representation PLM is the best. This is not new as previous work that the authors cited also had such similar results.

### Weaknesses
Although the datasets and the benchmark is useful, I think the technical contribution of the work is limited for the machine learning community. The paper may fits better a specific dataset and benchmark tracks rather a research main track.

Regarding the methods for collecting negative samples, the authors criticizes related work regarding false negatives possibility but their own method neither resolve that issue, I don't see how the proposed approach in the paper can help resolving the false negative may happen in the collected data.

### Questions
Could you please explain how the proposed methods of negative sampling would be better in related approach in resolving the false negative issues in the benchmark data?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents PepBenchmark, a standardized benchmark for peptide machine learning. It integrates three components: PepBenchData, a curated collection of 35 datasets (29 canonical, 6 non-canonical) across 7 pharmacological tasks; PepBenchPipeline, a preprocessing framework featuring biologically informed negative sampling and hybrid data splitting to prevent leakage; and PepBenchLeaderboard, a unified evaluation across four model families: fingerprint, GNN, SMILES, and PLM. Experiments show that Protein Language Models (PLMs) outperform others, with peptide-specific finetuning and scaling improving performance. Fingerprint models remain strong for small and non-canonical datasets, while GNN and SMILES models excel in peptide–protein interaction tasks. PepBenchmark provides the first reproducible foundation for systematic peptide ML evaluation and model comparison.

### Strengths
1. Comprehensive scale: 35 datasets covering over 78k sequences across canonical, non-canonical, and interaction tasks enable unified evaluation.

2. Rigorous preprocessing: New negative sampling and hybrid-split methods reduce leakage and overestimation of model performance.

3. Empirical insight: Systematic benchmarking across 4 model types and 30 tasks reveals clear scaling laws and practical model guidance for peptide ML.

### Weaknesses
1. Authors might want to inlcude these works in the related works section
- https://www.nature.com/articles/s42256-023-00691-9
- https://www.nature.com/articles/s42004-025-01601-3
- https://www.nature.com/articles/s41587-025-02761-2
- https://arxiv.org/abs/2410.19222?

2. For data splitting, the authors use sequence-level splits. However, since some evaluated methods are graph-based or structure-based, a more rigorous approach might consider structural or graph similarity during splitting. At minimum, the authors could quantify the graph or structural similarity within and across splits or clusters to assess potential data leakage.

### Questions
1. In Lines 185–186, the authors state that all non-canonical peptides are converted to SMILES. How exactly was this conversion performed, and does it result in any **loss of structural or chemical information**?

2. In Lines 234–245, regarding **MMSeq**, what **exact command** and **sensitivity parameter** were used during clustering or filtering?

3. For **“Task correlation filtering,”** the paper mentions using expert prior knowledge and statistical analysis to estimate correlations among tasks and exclude closely related ones. Could the authors clarify **what specific expert knowledge** and **which statistical methods** were applied in this process?

4. The paper notes that the model takes a canonical peptide as input and generates a chemically modified non-canonical peptide, allowing transformation of a canonical negative set into a non-canonical one. Please provide **more details about this generative model**, including its architecture, training data, and how the modifications are controlled or validated.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces PepBenchmark, a standardized benchmark for peptide machine learning that unifies diverse datasets(PepBenchData), standardized preprocessing(PepBenchPipeline), and unified evaluation protocols(PepBenchLeaderboard) across diverse peptide-related tasks.

### Strengths
- The paper presents the largest AI-ready peptide database, integrating 23 datasets across 7 pharmacological tasks, which supports a wide range of predictive applications in peptide therapeutics.
- The biologically-informed, distribution-controlled negative sampling strategy ensures that decoy peptides are generated in a realistic and un-biased manner.
- The benchmark systematically evaluates multiple model families and provide detailed insights into how each type of model performs across different peptide-related tasks.
- The benchmark explicitly addresses k-mer leakage and sequence redundancy

### Weaknesses
1. The benchmark is sequence-centric, which can understate GNN baselines that need reliable 3D/contact graphs.
2. More strong PLM embedders should be benchmarked (for example, ESM-2, ProtT5, and MSA Transformer), since they output fixed embeddings for prediction tasks. [1]
3. Add language models that can process non-canonical peptides (for example, GPepT) to cover tasks with modified residues. [2]
4. An ablation of the biologically informed, distribution-controlled negative sampling would show how much this strategy drives the gains.


[1] Zhang, R., et al. “Evaluating the advancements in protein language models for encoding protein sequences.” *Frontiers in Bioengineering and Biotechnology*, 2025. 

[2] Oikawa, Y., et al. “GPepT: A Foundation Language Model for Peptidomimetics Incorporating Non-canonical Amino Acids.” *ACS Medicinal Chemistry Letters*, 2025.

### Questions
1. How are graphs built for the GNN-based models?
2. How does distribution-controlled sampling change training and test results? Does matching five key properties between positive and negative sets (length, charge, hydrophobicity, molecular weight, isoelectric point) improve early enrichment and calibration?

### Soundness
4

### Presentation
4

### Contribution
3
