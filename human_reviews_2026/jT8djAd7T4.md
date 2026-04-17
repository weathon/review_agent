# BioSASANet: Self-Interpretable Shapley Attribution for Deep Genomics Sequence Modeling

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Deep learning models have significantly advanced genomics sequence modeling by uncovering complex patterns, but their black-box nature limits the biological insights that can be derived from their predictions. While recent efforts have employed post-hoc methods to identify important nucleotides or motifs, these approaches are decoupled from the prediction process and often suffer from accuracy limitations. In this paper, we introduce the Shapley Additive Self-Attribution (SASA) framework to genomics sequence modeling and propose BioSASANet. BioSASANet integrates a marginal contribution-based sequential module that captures long-range nucleotide interactions, and a positional Shapley value module that explicitly models the reverse complementarity (RC) property of genomic sequences, enabling position-aware, biologically grounded self-attribution. Experiments on two genomics tasks show that BioSASANet achieves accurate predictive performance while providing more faithful nucleotide-level attributions. Additionally, its flexible design allows integration with state-of-the-art DNA language model backbones, enabling advanced models with Shapley value-based self-interpretation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes BioSASANet, applying the Shapley Additive Self-Attribution (SASA) framework to genomic sequence modeling. The model integrates a marginal contribution module and a positional Shapley value module for end-to-end interpretable predictions, while incorporating MambaDNA to model DNA's reverse complementarity property. Experiments on histone modification and plant promoter prediction tasks show that BioSASANet provides nucleotide-level attributions while maintaining competitive predictive performance. However, the paper suffers from limited theoretical novelty, insufficient validation of key design choices, and unclear experimental setup.

### Strengths
1. Model interpretability in genomics is crucial for scientific discovery and biological validation, making this an important problem to address.

2. The paper includes comprehensive experiments on multiple datasets (histone marks, plant promoters) across different species with proper evaluation metrics (MP@k), and demonstrates framework flexibility through compatibility with different backbone models (Evo2, HyenaDNA).

### Weaknesses
1. The manuscript contains numerous formatting issues indicating insufficient proofreading: 
- Line 042 has citation "Chen et al." isolated by periods; 
- inconsistent use of \citet vs \citep throughout; 
- Section 2 improperly has only subsection 2.1; 
- Line 151 states "The structure overview of BioSASANet can be found in figure" without providing a figure number or cross-reference; more critically, this architecture overview figure does not exist anywhere in the manuscript.
- Line 248 presents Theorem 4.1 which should be cited from Caduceus [3] rather than restated as original contribution.

2. The literature review is insufficient. The introduction only mentions dated/limited-impact works (DeepBind, DeepSEA) while missing critical recent works like Enformer [1], Borzoi [2], and other state-of-the-art genomics models. Section 2 ("Related Work") mostly introduces basic genomics background rather than positioning against relevant literature.

3. The theoretical contribution is limited. Despite claiming "BioSASANet", the work essentially applies existing SASANet to genomics 
without novel insights or designs. The only genomics-specific addition is RC-equivariance from Caduceus [3], but no ablation study validates its benefits, and Caduceus itself showed limited gains from RC in some tasks. Missing are experiments on sequence-pair prediction consistency and controlled comparisons.

4. The baselines are weak and unclear. Only basic post-hoc methods (DeepLIFT, GradientSHAP, LIME) are compared, missing domain-specific methods like TF-MoDISco. Critically, the paper does not specify which model the baseline methods are applied to, nor reports baseline models' predictive performance for fair comparison. State-of-the-art genomics models [1][2][4] routinely perform interpretability analysis but are not compared.

5. The evaluation scope is limited. Only 2 task types are tested (histone marks, promoters) versus NT benchmark [4] with 18+ tasks. Only 4 histone marks in K562 cell line are evaluated, missing standard benchmarks like GenomicBenchmarks. The MP@k metric only evaluates recovery of known annotations, not discovery of novel motifs.

[1] Avsec, Žiga, et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature methods 18.10 (2021): 1196-1203.

[2] Linder, Johannes, et al. "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation." Nature Genetics 57.4 (2025): 949-961.

[3] Schiff, Yair, et al. "Caduceus: Bi-directional equivariant long-range dna sequence modeling." Proceedings of machine learning research 235 (2024): 43632.

[4] Dalla-Torre, Hugo, et al. "Nucleotide Transformer: building and evaluating robust foundation models for human genomics." Nature Methods 22.2 (2025): 287-297.

### Questions
1. On which model are the baseline attribution methods (DeepLIFT, SHAP, LIME) applied? If on BioSASANet itself, is this a fair comparison? If on a separate baseline model, what is its architecture and predictive performance?

### Soundness
2

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
This paper introduces BioSASANet, a self-interpretable neural network that applies SASA principles to genomic sequence modeling.

### Strengths
- The BioSASANet framework consistently outperforms other attribution methods that have been used in the genomics domain. 
- The design of the framework enables it to be utilized on a variety of DNA models types, thus it is a flexible tool for interpretability.

### Weaknesses
- Since a core claim of the paper is BioSASANet’s adaptability, it would be valuable to include experiments using a broader range of DNA model backbones to assess how the framework performs across diverse architectural families.
- No ablations are included that highlight the relative contribution of each of the architectural modules that are utilized in the approach. Including such ablations would be helpful in understanding the relative utility of each approach under different domain-specific modeling settings. 
- Using multiple metrics (not just mean precision at top-k) could be helpful in further interpreting the results. 
- The case studies require further explanation and motivation - some of these results are unclear. 


Minor comments:
- Some appendix sections (A.6–A.8) appear to be misaligned or improperly referenced. Additionally, having a written explanation for the appendix tables would be helpful. 
- The sentence “The structure overview of BioSASANet can be found in figure .” is missing a figure reference. There are also several minor typographical and formatting errors throughout the manuscript that should be carefully proofread and corrected.

### Questions
- How computationally costly is BioSASANet? Given the complexity of the approach, it would be useful to include metrics that report on the efficiency of the approach, especially in comparison to other similar approaches.
- How does the underlying DNA model backbone affect the quality (and by extension interpretability) of the attributions?

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
The paper proposes BioSASANet, a self-interpretable approach for genomic sequence modeling that extends the Shapley Additive Self-Attribution (SASANet) framework by incorporating Mamba/MambaDNA modules to capture long-range nucleotide interactions and reverse-complementary structure, achieving competitive predictive accuracy and more biologically faithful, nucleotide-level attributions on histone-mark and plant promoter datasets.

### Strengths
- Well-grounded theoretical foundation:
Builds directly on the Shapley Additive Self-Attribution (SASA) framework, preserving provable convergence to true Shapley values while adapting it to biological sequence data.While SASANet (Sun et al., 2025) introduced the general Shapley Additive Self-Attribution framework, BioSASANet extends it into a biologically specialized regime. It integrates reverse complementarity (RC)-aware modeling, Mamba/MambaDNA-based long-range sequence encoding, and nucleotide-level Shapley convergence within DNA sequences. 

- Biologically informed architecture:
Incorporates the NMCSM and MambaDNA modules, enabling modeling of long-range dependencies and bidirectional RC equivariance specific to DNA. This directly addresses the limitations of SASANet’s generic sequential encoder and grounds the interpretability mechanism in actual biological constraints.

- Faithful and interpretable results:
Consistently surpasses post-hoc baselines (DeepLIFT, GradientSHAP, LIME) on MP@k metrics and visually aligns high-attribution positions with true promoter or histone-mark regions.

- Architectural flexibility:
Demonstrates compatibility with multiple backbone DNA LMs, showing potential generality across future genomics models.

- Clarity and presentation quality:
The paper maintains a clear narrative flow from theoretical background to biological application, supported by equations, figures, pseudocode, parameter tables.

- Significance and potential impact:
The work aims to connect model interpretability with biological understanding.

### Weaknesses
- Limited in evaluation:
Current experiments focus only on histone-mark and promoter tasks. Extended evaluation to additional genomic contexts/tasks )(e.g., enhancer prediction, variant effect modeling, cross-species generalization etc.) would demonstrate robustness and transferability.

- Limitation in the attribution validation approach:
MP@k captures overlap with known motifs but does not directly ensure causality (biological correctness). One possible way to  to test attribution stability would be to conduct in silico mutagenesis. This is quite easy to conduct with even with limited comptational budget.

- Ablation missing:
While loss balancing hyperparameteres $\lambda_s$ and $\lambda_v$ are mentioned, their individual effects are not clear. I would suggest authors add ablations isolating each component to clarify their contributions

- Efficiency and scalability claims not empirically verified:
The paper cites Mamba’s linear complexity, however actual runtime and memory analysis (especially in genomic context) is missing.

- Figure + reference missing (line 151).

### Questions
- Can the authors test BioSASANet on additional genomics tasks or use in silico mutagenesis to validate attribution causality beyond MP@k?

- Can the authors provide ablations for $\lambda_s$ and $\lambda_v$, or provide discussion on this?

- Can the authors add runtime and/or memory comparisons to support efficiency claims?

### Soundness
2

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
The manuscript applies a recently developed framework Shapley Additive Self-Attribution
(SASA), to genomics sequence modeling. Technically this involved acknowledging reverse complementarity by combining the results of the relevant module (Mamba) from both DNA strands

### Strengths
- Potentially useful idea and framework 
- good job acknowledging reverse complementarity

### Weaknesses
1. Somewhat of a naive attempt for out-of-the-box application of a method from outside biology, with obviously limited biological understanding
2. Experiments compare against outdated methods on outdated data that is not at the single-nucleotide resolution that Shapley-value-based interpretation is supposed to be foucsed on

### Questions
1. Can you provide favorable comparisons vs. advanced feature-determinaiton methods on single-nucleotide-sensitive features (binding sites)? Figures of merit should involved clinical meaning or changes in downstream expression levels.

### Soundness
2

### Presentation
2

### Contribution
2
