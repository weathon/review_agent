# HAIPR: A High-Throughput Affinity Prediction Framework

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Accurate prediction of protein binding affinity is key for drug discovery and protein engineering, but commonly used evaluation protocols like Random Cross-Validation (RandomCV) can misrepresent true model generalization. We present HAIPR, a unified, open-source framework that streamlines the full machine learning pipeline for affinity prediction from training and optimization to inference, with curated benchmark datasets and robust, biologically meaningful evaluation protocols. By extending the BindingGYM benchmark and introducing realistic data splits, HAIPR reveals that RandomCV substantially overestimates model performance on out-of-distribution tasks. We systematically compare Support Vector Regression (SVR) using protein language model (pLM) embeddings to parameter-efficient fine-tuning (PEFT) of pLMs. SVR shows competitive results and increased stability in data-scarce scenarios, while PEFT excels as datasets grow larger and tasks become more complex. Analysis of model input setups shows that incorporating structural information does not always improve, and may sometimes hinder, performance for practical affinity prediction. Finally, we determine the lower limits of data required for reliable prediction, finding that even compact models can achieve performance close to the reproducibility limit of state-of-the-art assays, a practical ceiling for computational prediction. Code and pre-computed embeddings are publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces HAIPR, a unified framework for training, evaluating and performing inference for single‑complex binding‑affinity prediction using deep mutational scanning (DMS) data. The authors argue that standard random cross‑validation (RandomCV) inflates performance estimates because train and test variants are drawn from similar distributions. To address this, they extend the BindingGYM benchmark with five combinatorial DMS datasets and propose two evaluation splits that preserve all available data but better mimic real‑world generalisation: Leave‑One‑Mutation‑Out (LoMo), where all variants containing a particular mutation are withheld, and Out‑of‑Distribution (OOD) splits, where affinity labels are binned and one bin is held out for testing. The HAIPR pipeline includes data preprocessing, configurable split generation, hyper‑parameter optimisation, and a simple Predictor/Generator interface to support arbitrary models and sequence generators.

### Strengths
- Predicting how mutations affect binding affinity is a key challenge in drug discovery and protein engineering. The paper clearly shows that RandomCV can overestimate model performance because many mutations appear in both train and test sets. The authors make a strong case that better splits are needed to reflect real-world generalisation, where the goal is to predict unseen mutations.

- The proposed LoMo and OOD splits address this issue effectively. LoMo tests generalisation to unseen mutation sites, while OOD splits hold out affinity ranges. Both maintain full dataset size, unlike single-mutant filtering. Figures 4 and 5 show these splits reveal much lower performance than RandomCV, highlighting the gap between conventional benchmarks and realistic use cases.

- The authors evaluate SVR and PEFT models across multiple ESM sizes (8 million to 15 billion parameters) on 21 datasets.

- The authors implement a genetic algorithm that uses an ensemble of fine‑tuned ESMC‑300M models to explore sequence space and fold promising candidates with BOLTZ‑2.

- The paper is generally well‑written and includes informative diagrams.

### Weaknesses
- The paper does not clearly explain how OOD bins are defined. It's unclear if they are based on equal width, equal counts, or another rule. The LoMo split assumes that each mutation can be held out without hurting diversity in the training set. It would help to report how many variants are used in train and test per LoMo split, and how sensitive the results are to the binning method. Even a small table in the appendix with sample counts per split would improve clarity and reproducibility.

- Figure 2-B’s caption needs more explanation. As it stands, the figure is hard to understand without extra effort.

- Section 3.3 should describe what the model inputs and outputs are. Explaining this clearly would help readers in machine learning better understand the setup. Skipping these details makes the method harder to follow.

- The models considered are a standard SVR with radial‑basis kernel and a PEFT variant of pLM fine‑tuning. No new architectures or task‑specific losses are proposed, and no structural models or graph neural networks are evaluated, despite recent successes in geometry‑aware affinity prediction. The demonstration that SVR can be competitive in low‑data regimes is interesting but incremental; similar conclusions can be noted in the LoMo and OOD splitting strategy. As such, the methodological contribution lies mainly in designing the evaluation pipeline rather than new methods.

- The models used are standard SVR with RBF kernel and a PEFT-based fine-tuning of pLMs. There are no new architectures, no custom loss functions, and no comparison with structure-based models like GNNs, which are increasingly common in affinity prediction. While it's useful to show that SVR works well in small data settings, this is not a new insight. Similar conclusions can be noted in the LoMo and OOD splitting. The main contribution lies in the evaluation pipeline, not in model design.

- The PEFT experiments use fixed DoRA rank and dropout values, without tuning. The paper notes that PEFT sometimes collapses, causing missing results.

- Only Spearman correlation is reported. While useful, practical applications often care about identifying top binders or predicting absolute affinities. Including other metrics like mean squared error or top-k classification accuracy would give a fuller picture.

### Questions
I appreciate the effort and care in this work. In my view, it is better suited for a bioinformatics workshop at ICLR than for the main track. The paper does not introduce a novel idea or method that would justify main-track acceptance. But I understand, the binding affinity problem is important. A practical pipeline for realistic evaluation is valuable for the community. If the authors address the weaknesses listed above or clarify points I may have misunderstood, I am willing to raise my score to 6.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes HAIPR, a framework that standardizes training, evaluation, and inference for predicting binding-affinity changes of protein–protein complex (PPC) variants. This paper states that conventional RandomCV overestimates generalization performance. To address this issue, they proposed a dataset and introduced a novel splitting method, OOD and LoMo, to evaluate models under screening scenarios that better reflect practice. They also compare SVR and parameter-efficient fine-tuning (PEFT; DoRA) on top of pLM embeddings to analyze trade-offs across dataset size and task difficulty. Additionally, they examine the dependency of sample size to both RandomCV and proposed splitting method. They also analyzed the effect of focus-on/off input settings, and they demonstrate a high-throughput design pipeline that combines a genetic algorithm with structure prediction (BOLTZ-2).

### Strengths
- This paper proposes an evaluation protocol that addresses RandomCV’s tendency of overestimating model performance by introducing OOD and LoMo splits.
- This paper also integrates a GA and BOLTZ-2-based sequence generation method, demonstrating practical downstream utility.

### Weaknesses
- The evaluation relies heavily on Spearman correlation; therefore, it would be better to include other evaluating metrics other than Spearman correlation.
- The proposed split is not compared against Contig or Modulo splits under identical conditions. It would be better to show the comparison result using quantitative metric.
- There was no comparison of proposed sequence generation algorithm and conventional optimizers (e.g., greedy, random mutational search) under the same settings.
- LoMo split might be inefficient for context-aware models. When a model leverages adjacent residues to predict binding affinity, omitting the mutated residue has minimal impact, which might undermine the purpose of the split. 
- The title can be read as a general molecular evaluation framework; it should explicitly state its PPI/PPC focus to avoid ambiguity and enhance title clarity. 
- OOD split should consider distance between embeddings, which was not considered in this paper. It would be better if showing distances between embeddings.

### Questions
- Is there a reason for choosing SVR as the machine-learning baseline rather than models such as Random Forest or XGBoost? 
- The explanation of LoMo split is unclear. Since there are combinatorial library dataset containing multiple mutations, did you mean excluding all the specific mutations at specific site or at all sites?

### Soundness
2

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
The paper introduces HAIPR, a unified framework for high-throughput affinity prediction on protein–protein complexes (PPCs) using DMS data, which is particularly helpful improving the given protein's affinity. Its main technical pieces are: (i) standardized evaluation splits—Leave-one-Mutation-out (LoMo) and label-binned OOD—intended to avoid RandomCV optimism; (ii) a curated benchmark (extending BindingGYM with five combinatorial datasets to 21 PPCs); and (iii) baselines comparing SVR on ESM embeddings with PEFT (DoRA) fine-tuning, plus a genetic-algorithm demo with Boltz-2 folding checks. The results reinforce that RandomCV overestimates performance, LoMo/OOD are more realistic, focus-on vs. focus-off context yields only small differences for sequence PLMs, and PEFT can outperform SVR but is training-sensitive and collapses more often.

### Strengths
- **Evaluation realism.** Defines LoMo (hold out all samples containing a mutation token) and label-binned OOD (hold out an affinity bin), addressing RandomCV optimism and avoiding single-mutant filtering losses seen in contig/modulo.
- **Benchmark expansion.** Expands BindingGYM with five combinatorial datasets (total 21 PPCs) and documents their characteristics for split design.

### Weaknesses
- **Limited novelty.** The core contribution is split design and packaging; modeling components (ESM embeddings+SVR, DoRA-PEFT) and DMS curation build on existing lines (ProteinGym/BindingGYM). Scientific novelty is modest for a top-tier venue.
- **Framework description is high-level.** Interfaces (Predictor/Generator), dataset registry, and exact split manifests are not specified in enough detail to guarantee drop-in applicability although the authors choose this as their main contribution. If authors can give an example of using new model with their framework in the level of the code, it would be more helpful to understand the paper's strength.
- **Underpowered design-loop evidence for a "unified" framework.** Although HAIPR is motivated by accelerating enrichment for protein design, the paper demonstrates only one generator (a genetic algorithm) on a single dataset, without head-to-head enrichment against diverse generators (e.g., RFdiffusion/ProteinMPNN-based loops) or standardized enrichment metrics (e.g., top-k hit rate per iteration, best/median affinity gain, sample complexity under LoMo/OOD). As a result, the practical acceleration claim remains weak.
- **Multi-chain input treatment is ad-hoc.** Focus-off concatenates chains with a separator token (not native to ESM training), yet shows little benefit; the work does not compare against chain-wise embedding+fusion or structural models in a controlled way. This weakens conclusions about "structural context".

### Questions
1. **Scatter vs. density panels.** In Fig. 4, are the prediction scatter plots drawn on train+test or test-only data? A legend/footnote clarifying this would help interpret calibration and the apparent distribution mismatch.
2. **PEFT collapse diagnostics.** What proportion of runs collapsed per model? Why does the model collapse frequently? Are there any expected reasons for the issue, such as noisy data labels?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes HAIPR, a high-throughput framework for protein–protein binding affinity prediction. It extends the BindingGYM benchmark by adding new combinatorial deep mutational scanning datasets and provides standardized evaluation protocols for assessing model generalization. The authors argue that common practices such as Random Cross-Validation can lead to overly optimistic results. To address this, they propose two alternative evaluation schemes, Leave-One-Mutation-Out and Out-of-Distribution splits. The framework supports both classical machine learning methods, such as Support Vector Regression, and parameter-efficient fine-tuning of protein language models. The authors further analyze the effects of sample size and evaluate the feasibility of in silico screening for variant design.

### Strengths
The paper addresses a relevant issue in computational biology for fair and reproducible evaluation of protein affinity prediction models. The motivation is clear, and the idea of systematically comparing data splits to assess generalization is useful. The experiments are extensive, and the results effectively illustrate the overestimation caused by Random Cross-Validation. The inclusion of minimal data size analysis and the availability of a unified interface for benchmarking may benefit future research in this area.

### Weaknesses
The novelty of the work is limited. The framework mainly combines existing datasets, standard evaluation strategies, and previously available models into one framework. The proposed data splits are incremental extensions rather than fundamentally new evaluation concepts. The analysis provides limited mechanistic or theoretical insight into model behavior, and the experimental findings are largely confirmatory rather than revealing new patterns.

### Questions
Besides using SVR and PEFT, have the authors considered freezing all parameters of the pLMs and training only a lightweight MLP head on top of the frozen embeddings?

### Soundness
2

### Presentation
3

### Contribution
2
