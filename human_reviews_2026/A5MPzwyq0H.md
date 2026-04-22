# Pep2Prob Benchmark: Predicting Fragment Ion Probability for MS$^2$-based Proteomics

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Proteins perform nearly all cellular functions and constitute most drug targets, making their analysis fundamental to understanding human biology in health and disease. Tandem mass spectrometry (MS$^2$) is the major analytical technique in proteomics that identifies peptides by ionizing them, fragmenting them, and using the resulting mass spectra to identify and quantify proteins in biological samples. In MS$^2$ analysis, peptide fragment ion probability prediction plays a critical role, enhancing the accuracy of peptide identification from MS$^2$ spectra as a complement to the intensity information. Current approaches rely on global statistics of fragmentation, which assumes that a fragment's probability is uniform across all peptides. Nevertheless, this assumption is oversimplified from a biochemical principle point of view and limits accurate prediction. To address this gap, we present **Pep2Prob**, the first comprehensive dataset and benchmark designed for peptide-specific fragment ion probability prediction. The proposed dataset contains fragment ion probability statistics for 608,780 unique precursors (each precursor is a pair of peptide sequence and charge state), summarized from more than 183 million high-quality, high-resolution, HCD MS$^2$ spectra with validated peptide assignments and fragmentation annotations. We establish baseline performance using simple statistical rules and learning-based methods, and find that models leveraging peptide-specific information significantly outperform previous methods using only global fragmentation statistics. Furthermore, performance across benchmark models with increasing capacities suggests that the peptide-fragmentation relationship exhibits complex nonlinearities requiring sophisticated machine learning approaches. Pep2Prob provides a standardized evaluation framework that will accelerate algorithmic innovation in computational proteomics while introducing a biologically significant prediction task to the machine learning community.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Pep2Prob, the first comprehensive dataset and benchmark specifically designed for peptide-specific fragment ion probability prediction. It collects a vast amount of mass spectrum data and selects 600k data points as the standard set. Additionally, benchmark are conducted on several baselines and metrics;

### Strengths
1. Novelty: The task of peptide-specific fragment ion probability prediction is sufficiently novel, as there is currently no corresponding large-scale research, comprehensive datasets, or benchmarks in the AI community;

2. Writing: Section 2 of the article explains the task in a step-by-step manner, making it easy-to-follow

### Weaknesses
1. The field is very narrow and not suitable for the AI community: AI for MS2 proteomics is a relatively less-focused area (at least within the AI community), and the proposed task of peptide fragment ion probability prediction in this paper is a very niche one in AI for MS2 proteomics field. To my knowledge, there have been no technical papers on this topic in recent years at top-tier ML/AI conferences (such as ICLR, Neurips, etc.); this makes this paper unsuitable for appearance at ICLR, a top AI conference. It is not a matter of concern to the AI community, and it may be more appropriate to publish it in a biological journal;

2. The significance is unclear; this article fails to clearly explain the significance of the task of peptide fragment ion probability prediction, merely making a general statement that this task is helpful in the fields of "peptide identification, post-translational modification (PTM) localization, and protein quantification.";

3. The baseline is outdated and lacks relevant work support; the baselines tested in this paper's benchmark, including LR, ResCNN, ResFFNN, and vanilla transformer, are all very early models in the AI community and appear outdated today (2025). Moreover, these models were first used by the author of this paper to complete the peptide fragment ion probability prediction task; does this indicate that the AI community did not prioritize this task in the past, resulting in a lack of representative ML methods in this field?

### Questions
in 'Weaknesses'


In summary, this paper proposes a benchmark and dataset within a narrow and specialized field of biology, which is sufficiently novel and paper is well-written. However, it does not address issues and directions of interest to the AI community, and there have been no ML-based works related to this direction proposed in the AI community in recent years. Additionally, the significance it proposes is not explained clearly enough.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Pep2Prob, a comprehensive benchmark for peptide-specific fragment ion probability prediction. By combining over 183 million spectra across 600,000 precursors, the study shows that integrating peptide sequence information consistently improves prediction accuracy as model capacity increases.

### Strengths
1. The benchmark is carefully designed and fills an important gap in computational proteomics by focusing on peptide-specific fragment ion probability prediction.

2. The dataset is large-scale and rigorously constructed from over 183 million spectra.

3. The analysis demonstrates clear performance gains when incorporating peptide-specific information, highlighting the importance of modeling sequence-dependent fragmentation.

### Weaknesses
1. The benchmark is technically sound but lacks a clear connection between improved model performance and its biological or practical significance. The main finding appears to be that “after including peptide-specific information, prediction accuracy continuously improves with increasing model capacity.” However, for readers without a strong proteomics background like me, it remains unclear how this improvement in prediction accuracy directly translates to biological insights or downstream applications. 

This makes me a bit unsure about the magnitude of the biological advance of this benchmark, and I will raise my score if the authors can briefly clarify.

2. Can the authors provide a rationale for why they decided to exclude PTMs in this study? This choice seems important, as many downstream tasks papers(eg. de novo peptide sequencing) treat PTM handling as a key evaluation aspect (eg. NovoBench, Zhou et al.).

### Questions
see weakness

### Soundness
3

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
This paper introduces Pep2Prob, a novel large-scale dataset and benchmark for peptide-specific fragment ion probability prediction. The dataset is constructed from 183 million high-quality HCD MS² spectra, covering 608,780 unique precursors (peptide sequence and charge state combinations). The authors propose a similarity-based data partitioning strategy and establish multiple baseline models ranging from global statistical approaches to Transformer-based architectures. The experiments demonstrate that incorporating peptide sequence information and increasing model capacity can enhance prediction performance.

### Strengths
1. The paper is well-organized and clearly written.

2. Dataset Scale and Quality: Pep2Prob demonstrates significant advantages in data volume, source diversity, and annotation quality. Built upon authentic, high-quality HCD spectra, it exhibits good representativeness and practical utility.

3. Data Splitting Strategy: The graph-based modeling of peptide sequence similarity effectively prevents information leakage between training and test sets, thereby enhancing the robustness of evaluation.

4. Benchmark Evaluation: The study establishes multiple baselines spanning from simple statistical methods to modern deep learning models, employing a variety of evaluation metrics for comprehensive assessment.

### Weaknesses
1. Lack of In-depth Insights: The paper merely demonstrates that larger models yield better performance but fails to provide deeper explanations regarding why this occurs or which specific aspects of the fragmentation process these models capture. In particular, while the improvement is attributed to "complex nonlinear relationships" in the data, no further in-depth discussion or analysis is provided to substantiate this claim.

2. Outdated and Incomplete Baselines: The baseline models employed in the study do not reflect the current state-of-the-art methodologies. Moreover, the paper fails to include a direct comparison with advanced intensity prediction models, such as Prosit, as core baselines for probability prediction. This omission obscures the unique value and performance boundaries of "probability prediction" relative to "intensity prediction."

3. Insufficient Demonstration of Practical Utility: Although the complementary value of probability prediction is emphasized, the study does not provide direct evidence of its practical benefits through performance improvements in downstream tasks, such as peptide identification or database search. The lack of end-to-end validation undermines the claimed utility of the proposed method.

4. Limited Data Diversity: The exclusive use of HCD spectra, the absence of post-translationally modified peptides, and the restriction to Orbitrap instruments significantly limit the model's generalizability and p

### Questions
1. When trained exclusively on HCD data, can the model generalize effectively to other fragmentation techniques (e.g., CID or ETD)?

2. Why not directly derive probability estimates from the outputs of intensity prediction models like Prosit? Are there fundamental differences in the modeling objectives or output distributions between these two tasks?

3. Are there plans to integrate Pep2Prob predictions into peptide identification pipelines (e.g., MSGF+ or MaxQuant) to validate improvements in metrics such as false discovery rate (FDR) or peptide identification counts?

4. While the Transformer model achieves the best performance, what is its computational cost? Could you provide a computational efficiency analysis comparing different models?

5. Does the dataset exhibit overfitting to specific peptide sequences or charge states? Has any analysis been conducted to examine systematic biases in predictions for longer peptides or high-charge precursors?

6. The paper attributes "complex nonlinearities" to peptide-fragmentation relationships based primarily on the observation that larger models yield better performance. Could you provide deeper analysis to reveal concrete manifestations of these nonlinearities and connect them to biochemical principles, thereby offering more interpretable insights beyond this general claim?

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
2

### Summary
This paper introduces Pep2Prob, the first comprehensive dataset and benchmark designed to predict peptide-specific fragment ion probabilities in \(MS^2\)-based proteomics. Current methods rely on oversimplified global fragmentation statistics (assuming uniform fragment probabilities across all peptides), which fail to account for sequence-dependent biochemical factors (e.g., amino acid neighbors, bond stability). Pep2Prob addresses this gap by curating 608,780 unique precursors (peptide sequence + charge state) derived from over 183 million high-resolution HCD \(MS^2\) spectra with validated annotations.

### Strengths
1. Strict filtering (peptide length 7–40, ≥10 spectra per precursor, no modifications) and precise annotation (0.05 Th mass tolerance, binary ion masks for valid fragments) ensure high reliability of the dataset.
2.The authors test both statistical rules (Global, BoF) and state-of-the-art ML models (Transformer), enabling a clear performance hierarchy. This breadth validates that peptide-specificity and model capacity drive improvements.
3. Improved fragment ion probability predictions directly enhance downstream tasks (peptide identification, PTM localization, biomarker discovery) by better distinguishing signal from noise in complex spectra.

### Weaknesses
1. The dataset only includes HCD spectra from Orbitrap instruments, excluding other common fragmentation techniques (e.g., ETD, CID) and instrument platforms (e.g., timsTOF). 
2. PTMs (e.g., phosphorylation, acetylation) are critical for protein function and drastically alter fragmentation patterns, but Pep2Prob excludes modified peptides due to "challenges in confident identification at scale."

### Questions
The dataset is human-only—how would models trained on Pep2Prob perform on non-human organisms (e.g., yeast, mouse) with distinct amino acid usage patterns?

### Soundness
3

### Presentation
3

### Contribution
3
