=== CALIBRATION EXAMPLE 10 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is informal and catchy ("Palooza", "Odyssey"), which is fine stylistically, but the abstract is imprecise on claims. The abstract states the benchmark achieves "superior diversity compared to existing alternatives," but diversity here is defined operationally through reconstruction error—a proxy measure that conflates *coverage* of existing ranking patterns with diversity. This conflation should be addressed upfront. The abstract also does not mention key limitations (e.g., the ≤50,000 sample, ≤10 class restrictions) that substantially narrow the benchmark's scope.

---

### Introduction & Motivation

The motivation is genuine and well-articulated: inconsistent model rankings across benchmarks is a recognized and practically important problem. However, the contributions as stated are unsatisfyingly vague:

- "We propose a pipeline to assess the diversity of tabular benchmarks" — but the pipeline is largely a nearest-neighbor reconstruction procedure operating in meta-feature space. Calling this a "diversity assessment" overstates its breadth.
- The framing of diversity vs. efficiency as "often competing" is asserted without justification. Why must they be in tension? More datasets → more compute, yes, but the trade-off structure is not formally established.

---

### Methodology

**Section 3.1.1 – Diversity Assessment Protocol:**  
This is the most critical component of the paper, and it has serious underspecification and conceptual problems.

1. **Incomplete definition of p_k^n**: The text mentions that p̄_k is estimated from the source benchmark S, but *how* p_k^n (the per-dataset rank estimate) is computed is never stated in Section 3.1. The definition only appears implicitly in Section 3.1.5 (nearest-neighbor matching). The reader cannot reconstruct the main metric from Section 3.1.1 alone.

2. **Circular definition of diversity**: The paper equates a lower reconstruction error (TabPalooza approximates other benchmarks' rankings well) with *higher diversity*. But this conflates diversity with coverage. A benchmark that is specifically optimized to approximate existing benchmarks could achieve low dr simply by being a "union" of existing benchmark datasets—not because it is more diverse. Conversely, a benchmark of truly novel, OOD datasets would have high dr (can't approximate existing benchmarks) but might be the most scientifically interesting. The paper never justifies why coverage-of-existing-benchmarks implies genuine diversity.

3. **Symmetry issue**: The metric dr is claimed to be asymmetric—TabPalooza achieves lower dr when used as source, and higher dr when used as target. This asymmetry is presented as evidence of diversity but is actually expected by construction (TabPalooza was *built* to approximate others). This is not a property that can be interpreted as diversity without a cleaner formulation.

**Section 3.1.4 – Meta-Feature Evaluation:**  
The rank predictor model—which maps (meta-features, model_index) → rank—is never specified. What model class is used? The feature encoding uses the raw integer index of the model (`si`), which is highly unusual. Integer indices impose an ordinal structure on models with no semantic basis. One-hot encoding or model-specific descriptors would be more principled. The omission of the predictor's model class is a reproducibility gap.

**Section 3.1.5 – Benchmark Alignment:**  
The nearest-neighbor matching over meta-features uses Euclidean distance, but there is no mention of feature normalization/standardization. With 112 meta-features on heterogeneous scales, raw Euclidean distance is dominated by high-variance features. Additionally, the greedy procedure may select the same source dataset multiple times (if it is the nearest neighbor of multiple target datasets). Is deduplication enforced? This is not stated.

**Section 3.2 – Efficiency Guarantee:**  
Hierarchical clustering with *random selection* of one dataset per cluster introduces stochasticity. Are results averaged over multiple realizations? How sensitive are the conclusions to a particular random draw? This is not addressed, which undermines the reliability of any specific TabPalooza instance.

---

### Experiments & Results

**Section 4.1 – Dataset Curation:**  
Using two LLMs to parse and validate Kaggle datasets is creative, but the reliability of this process is uncertain. The paper only retains cases where DeepSeek-r1 and Qwen3 *agree*, but there is no ground-truth validation of LLM-identified targets. Downstream errors (wrong target column identification) would silently corrupt the benchmark.

The solvability threshold (AUC > 0.55 for classification, R² > 0.2 for regression using XGBoost) is conceptually problematic. This filters based on one model's performance, which means the benchmark is biased toward datasets where XGBoost has predictive signal—potentially skewing comparisons between tree-based and neural models in favor of the former. This should at minimum be discussed as a limitation.

The exclusion of datasets with >10 classes—to meet ICL-based model limitations—is explicitly design-driven rather than data-driven. This excludes large families of real-world tasks. The decision is acknowledged but its impact on generalizability is not analyzed.

**Section 4 header artifact – stray text:**  
Section 4 opens with "deep tabular prediction, machine learning"—this appears to be leftover metadata or keywords. While this is likely a PDF parsing artifact, it is worth noting that the paper's Section 4 has no introductory paragraph of its own.

**Section 4.2 – Meta-Feature Evaluation:**  
Tables 2 and 3 show that the rank predictor outperforms the baseline in most cases, but with notable exceptions: for TabArena, the predictor is *worse* than the constant baseline for both ACC and F1 (delta = −0.040 and −0.164). The paper dismisses this in one sentence ("exceptions are observed"). No explanation or analysis is provided. Why does meta-feature-based prediction fail for TabArena specifically? Is it that TabArena has a distinctive dataset distribution? This deserves investigation.

More fundamentally: what model is being used as the rank predictor? Without knowing this, the delta values have no interpretive context.

**Section 4.3 – Benchmark Size Selection:**  
The size selection via an inflection point on a dr-vs-size curve is subjective. The paper selects 100 (classification) and 140 (regression), but without a more principled criterion (e.g., a formal elbow detection or diminishing-returns threshold), the choice is open to question. More importantly, these two sizes are very different (100 vs. 140) despite classification having many more total datasets (501 vs. 335). This asymmetry is unexplained.

**Section 4.4 – Benchmark Alignment:**  
The dr values in Tables 4 and 5 show TabPalooza achieves lower reconstruction error, but there is no baseline comparison here. What does a *random subset* of the same size achieve? Without this null model, it is impossible to assess whether the hierarchical clustering selection provides any benefit over simple random sampling. This is a serious gap.

**Section 4.5 – Comparison with Other Benchmarks:**  
The cross-reconstruction matrices in Figures 2 and 3 are central claims, but the figures are not readable in the parsed text. From the prose, the claim is that TabPalooza achieves lower dr as a source benchmark. However, because TabPalooza was explicitly optimized to minimize this very quantity, this comparison is not a fair external validation. A holdout evaluation on truly unseen future benchmarks would be more convincing.

**Missing experiments of critical importance:**
- There are no actual model ranking results reported on TabPalooza. For a benchmark paper, this is a substantial gap—readers need to know what conclusions TabPalooza would support (e.g., do tree-based models still dominate? Does TabPFN-v2 lead?).
- No statistical significance testing on dr differences between benchmarks.
- No ablation on the contribution of the Kaggle datasets to diversity.
- No analysis of dataset overlap between TabPalooza and the benchmarks it claims to outperform.

---

### Writing & Clarity

The paper's core methodological pipeline (how p_k^n is estimated in the diversity protocol) is split across Sections 3.1.1 and 3.1.5 in a way that makes it hard to understand the method as a whole. The logical flow of Section 3 needs restructuring. The distinction between *meta-feature evaluation* (Section 3.1.4, which is about validating that meta-features predict ranks) and *benchmark alignment* (Section 3.1.5, which is the actual diversity assessment procedure) is insufficiently signposted, making it unclear which component drives the main claims.

---

### Limitations & Broader Impact

The paper lacks a dedicated limitations section. Key concerns not acknowledged:
- The benchmark excludes large-scale datasets (>50K samples), which are increasingly common in practice and specifically where ICL models may behave differently.
- Fixing the benchmark to existing task taxonomies (classification/regression only, ≤10 classes) means it does not cover multi-output, multi-label, or ordinal regression settings.
- The meta-features used are from PyMFE, adapted for regression via ad hoc binning. This adaptation is not validated, and its impact on meta-feature reliability is ignored.
- The stochasticity in dataset selection (random sampling from clusters) means the benchmark is not uniquely defined, raising reproducibility concerns.

---

### Overall Assessment

TabPalooza addresses a real and important problem—the lack of an agreed-upon, diverse benchmark for tabular ML evaluation. The idea of using meta-feature diversity and cross-benchmark reconstruction as evaluation criteria is interesting and novel. However, the paper has several critical methodological weaknesses that prevent acceptance in its current form. The core diversity metric conflates coverage of existing benchmarks with genuine diversity; the rank predictor—which underpins the validity of the meta-feature analysis—is never specified; the main comparison against random-subset baselines is absent; and the paper contains no actual model performance rankings on TabPalooza, which would be the most basic deliverable of a benchmark contribution. For ICLR, where methodological rigor and empirical validation are held to a high standard, these gaps are disqualifying. The dataset collection effort is commendable, but the scientific framing and experimental validation need substantial revision. The work would benefit substantially from (1) formally separating diversity from coverage, (2) comparing against random-subset baselines, (3) reporting full model rankings on TabPalooza, and (4) complete specification of the rank predictor used throughout.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces TabPalooza, a new tabular classification and regression benchmark explicitly optimized to balance dataset diversity with evaluation efficiency. The authors propose a quantitative pipeline that assesses benchmark diversity via a rank-based reconstruction error (`d_r`) and leverages meta-feature correlations with hierarchical clustering to select a representative subset of datasets. Empirical analysis demonstrates that TabPalooza consistently achieves lower reconstruction errors when aligned against existing benchmarks, suggesting it provides more generalized and reliable coverage for evaluating modern tabular models, particularly in-context learning approaches.

### Strengths
1. **Well-Motivated Benchmark Design:** The explicit trade-off between diversity and efficiency directly addresses a recognized bottleneck in tabular ML. Using hierarchical clustering on a Universal Dataset Pool (UDP) of 501 CLS and 335 REG datasets provides a principled mechanism to minimize redundancy while keeping computational costs tractable.
2. **Quantitative Diversity Metric:** The reconstruction error pipeline (`d_r`) moves benchmark evaluation beyond subjective claims. By measuring how well a source benchmark's ranking predictions approximate those of target benchmarks, the authors establish a replicable framework for comparing benchmark representativeness.
3. **Comprehensive Baseline Model Suite:** The evaluation spans 11 models covering tree-based, neural network, Auto-ML, and state-of-the-art ICL architectures (TabPFN-v2, TabICL, LimiX, Mitra). This ensures the benchmark reflects current modeling paradigms rather than relying on outdated baselines.
4. **Rigorous Data Curation Pipeline:** The multi-stage filtering process (size/feature limits, ICL compatibility constraints, and LLM-assisted Kaggle validation with XGBoost solvability thresholds) results in a clean, high-quality UDP. Excluding datasets with trivial performance floors (ROC AUC < 0.55, R² < 0.2) improves benchmark signal-to-noise ratio.

### Weaknesses
1. **Baseline Formulation & Statistical Evaluation:** The comparison baseline for `d_r` (training set empirical mean rank) is weak and does not adequately demonstrate the added value of the meta-feature predictor. Furthermore, the results are presented as point estimates without variance quantification. Given the stochastic elements (e.g., random dataset selection per cluster, LLM validation), bootstrapped confidence intervals or statistical significance tests are missing.
2. **Underspecified Rank Prediction Architecture:** Section 3.1.4 states the goal is to learn `f: R^(z+1) -> R`, but the specific regression model, architecture, hyperparameters, and training protocol used to map meta-features to ranks are omitted. Without these details, the meta-feature evaluation pipeline cannot be independently reproduced.
3. **LLM Validation Opacity:** Relying on DeepSeek-r1:32b and Qwen3:32b to parse Kaggle dataset structures and targets is innovative but introduces non-determinism. The exact prompts, temperature settings, schema constraints, and disambiguation logic for multi-target or poorly documented datasets are not provided, creating reproducibility risks.
4. **Narrative & Referencing Inconsistencies:** The manuscript contains cross-reference mismatches (e.g., Section 4.4 cites Section 4.1 for size selection, though it is actually Section 4.3). While equation rendering artifacts are noted as parser issues, the textual references to metrics (alternating between `d_r` and `diff ~~r~~ ank`) require editorial cleanup for professional publication standards.

### Novelty & Significance
- **Novelty:** Moderate to High. While benchmark curation via clustering and meta-feature analysis exists in isolation, the paper's primary novelty lies in the *alignment pipeline* that quantifies benchmark diversity through inter-benchmark rank reconstruction error. The formalization of `d_r` as a proxy for evaluative coverage offers a fresh methodological lens for comparing benchmark suites.
- **Clarity:** Moderate. The conceptual flow from dataset pooling to clustering to alignment evaluation is logical. However, the mathematical notation in Sections 3.1.1 and 3.1.4 suffers from garbled formatting and missing variable definitions, which temporarily obscures the technical pipeline. Section 4 requires careful cross-referencing corrections.
- **Reproducibility:** Moderate to Good. The authors commit to releasing TabPalooza on HuggingFace and specify libraries (PyMFE, Optuna, TALENT, Ollama). The pipeline is conceptually reproducible, but exact hyperparameter search spaces, the regression model used for rank prediction, and LLM prompt templates must be disclosed in supplementary materials to meet ICLR's rigorous reproducibility standards.
- **Significance:** High. The tabular ML community is currently fragmented by inconsistent evaluation suites, and the emergence of ICL-based tabular foundation models amplifies the need for standardized, efficient, and diverse benchmarks. TabPalooza's data-driven selection protocol and quantitative diversity metric provide a practical template that can directly improve fairness in model comparison and accelerate iterative research in structured data learning.

### Suggestions for Improvement
1. **Strengthen Baselines & Add Uncertainty Quantification:** Compare the meta-feature rank predictor against stronger null models (e.g., random rank permutation, global median rank, or leave-one-out cross-validation baselines). Report bootstrapped standard errors or confidence intervals for `d_r` across multiple clustering initializations to demonstrate statistical robustness.
2. **Fully Specify the Meta-Feature to Rank Model:** Explicitly detail the machine learning algorithm (e.g., Random Forest, linear model, or neural net), feature encoding, training/validation splits, and hyperparameter tuning procedure used in Section 3.1.4. Consider moving detailed configuration to an appendix if space is constrained.
3. **Document LLM Parsing Protocol:** In an appendix, provide the exact system/user prompts, output schema, temperature/top_p parameters, and conflict-resolution logic used by DeepSeek-r1 and Qwen3. Include a small manually audited subset to quantify the LLM's parsing accuracy and failure modes.
4. **Standardize Notation & Fix Cross-References:** Ensure consistent use of `d_r` throughout the manuscript. Correct the Section 4.3/4.4 reference mismatch. Provide cleanly rendered, self-contained equations for the rank estimation pipeline so that all terms (`p̄_k`, `r̄_k^T`, `S`, `T`, `M`) are explicitly defined at first use.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Downstream Model Selection Utility:** Demonstrate that selecting models based on TabPalooza rankings leads to better generalization on holdout real-world tasks compared to selection based on TabZilla or TabArena. Without this, the claim that TabPalooza is a "better" benchmark is unsubstantiated.
2. **Kaggle Label Verification:** Conduct manual verification on a statistically significant subset of the Kaggle datasets where LLMs identified the target column. Relying solely on model consensus for ground truth identification introduces unacceptable noise for a benchmark paper.
3. **Meta-Feature Ablation:** Perform an ablation study removing categories of meta-features (e.g., Info-theory vs. Statistical) to show which features drive the dataset selection. Without this, the claim that "rich meta-features" improve diversity is opaque.
4. **Compute Budget Reporting:** Explicitly report the total GPU/CPU hours required to generate the benchmark results. The claim of "Efficiency" is meaningless without quantifying the computational cost of the evaluation pipeline itself.

### Deeper Analysis Needed (top 3-5 only)
1. **Circularity of Diversity Metric:** Analyze whether minimizing reconstruction error ($d_r$) against existing benchmarks simply averages their biases rather than capturing true diversity. If existing benchmarks are biased, aligning to them propagates those biases rather than solving them.
2. **Statistical Significance of $d_r$:** Provide confidence intervals or significance tests for the differences in reconstruction error between TabPalooza and baselines. The reported decimal differences (e.g., 0.043 vs 0.056) may be within the variance of the evaluation pipeline.
3. **Sensitivity to Clustering Thresholds:** Analyze how sensitive the final dataset selection is to the hierarchical clustering parameters. If small changes in thresholds drastically alter the benchmark composition, the method lacks robustness.

### Visualizations & Case Studies
1. **Meta-Feature Space Coverage:** Provide a PCA or t-SNE plot showing the coverage of TabPalooza datasets versus existing benchmarks in the 112-dimensional meta-feature space. This would visually verify the claim of superior diversity.
2. **Rank Correlation Heatmaps:** Replace reconstruction error heatmaps with Spearman rank correlation matrices between benchmarks. This provides a more standard and interpretable view of benchmark agreement.
3. **Failure Case Analysis:** Include case studies of specific datasets where TabPalooza's rank prediction significantly deviates from the actual performance. This exposes the limits of the meta-feature modeling.

### Obvious Next Steps
1. **Human-in-the-Loop Validation:** Implement a human verification step for the Kaggle dataset curation pipeline before release to ensure target columns are correctly identified.
2. **Release Evaluation Code:** Publish the complete code for the diversity assessment pipeline, not just the dataset, to allow others to validate the $d_r$ metric and reproduce the selection process.
3. **Baseline Availability Check:** Ensure all cited baseline models (especially 2025-dated models like TabPFN-v2) have public code and weights available at the time of publication to guarantee reproducibility.

# Final Consolidated Review
## Summary

TabPalooza introduces a new benchmark for tabular classification and regression, explicitly designed to balance dataset diversity with evaluation efficiency. The authors construct a Universal Dataset Pool (UDP) of 501 classification and 335 regression datasets, select representative subsets via hierarchical clustering on meta-features, and propose a reconstruction error metric (d_r) to quantify how well one benchmark's model rankings approximate another's. The paper demonstrates that TabPalooza achieves lower reconstruction error when used as a source benchmark compared to existing alternatives.

## Strengths

- **Addresses a recognized gap with a clear problem formulation:** The paper correctly identifies that inconsistent model rankings across existing tabular benchmarks hinder reliable model evaluation, and proposes an explicit diversity–efficiency trade-off as a guiding principle. The motivation is genuine and well-articulated.

- **Quantitative benchmark evaluation framework:** The reconstruction error (d_r) pipeline provides a concrete, reproducible method for comparing benchmark representativeness, moving beyond qualitative claims of "diversity." This formalization is a genuine methodological contribution.

- **Comprehensive model coverage:** The evaluation spans 11 models including modern in-context learning approaches (TabPFN-v2, TabICL, LimiX, Mitra), tree-based methods (XGBoost, CatBoost, Random Forest), neural architectures (ExcelFormer, ResNet, MLP), and AutoGluon. This ensures the benchmark reflects current modeling paradigms.

- **Substantial data curation effort:** The multi-stage pipeline—aggregating datasets from TALENT, OpenML-CC18, TabZilla, TabArena, PFN benchmarks, and Kaggle, with filtering for size constraints, ICL compatibility, and solvability thresholds—represents a significant resource contribution.

## Weaknesses

- **Underspecified rank predictor model:** Section 3.1.4 states the goal of learning a function f: R^(z+1) → R that maps (meta-features, model_index) to predicted ranks, but the model class, architecture, hyperparameters, and training procedure are never specified. Without this, the meta-feature evaluation results cannot be independently reproduced or interpreted.

- **Missing baseline for dataset selection:** The paper demonstrates that TabPalooza achieves lower d_r than existing benchmarks when used as a source, but does not compare against a random subset of the same size drawn from the UDP. Without this null model, it is unclear whether hierarchical clustering on meta-features provides any benefit over simple random sampling—the central efficiency claim remains unsubstantiated.

- **No model ranking results reported on TabPalooza itself:** For a benchmark paper, it is notable that no actual model performance rankings on the final TabPalooza benchmark are presented. Readers cannot assess what conclusions TabPalooza would support (e.g., do tree-based models still dominate? How do ICL methods compare?), which limits the paper's immediate practical utility.

- **Incomplete definition of per-dataset rank estimation:** The notation p_k[n] (rank estimate for model g_k on dataset t_n) is introduced in Section 3.1.1 but defined only implicitly via nearest-neighbor matching in Section 3.1.5. The methodological pipeline is split across sections in a way that impairs comprehension.

- **Unexplained negative results for TabArena:** Table 2 shows that for TabArena classification, the meta-feature rank predictor performs worse than the trivial baseline for ACC (delta = −0.040) and F1 (delta = −0.164). The paper dismisses this in a single clause ("although exceptions are observed") without analysis. Why does meta-feature prediction fail for TabArena specifically?

- **LLM-based dataset validation lacks ground-truth verification:** The Kaggle dataset curation relies on consensus between DeepSeek-r1 and Qwen3 to identify target columns, with no manual validation or external ground truth. Systematic target misidentification would silently corrupt the benchmark's integrity.

- **Stochasticity in dataset selection unaddressed:** The selection of one random dataset per cluster introduces variance, but results are presented as point estimates without confidence intervals or sensitivity analysis across multiple realizations. The benchmark is not uniquely defined.

## Nice-to-Haves

- **Downstream model selection utility:** It would strengthen the paper to demonstrate that model rankings derived from TabPalooza generalize better to holdout real-world tasks compared to rankings from existing benchmarks, but this is beyond the stated scope of demonstrating diversity and coverage.

- **Compute cost transparency:** The paper claims efficiency as a design principle but does not report the computational cost of the evaluation pipeline itself.

- **Meta-feature ablation:** An ablation study on which meta-feature categories drive dataset selection would provide insight into the diversity characterization, but is not essential to the main claims.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Title informality:** Criticizing the title style ("Palooza", "Odyssey") is a formatting preference, not a substantive issue.

- **"deep tabular prediction, machine learning" artifact in Section 4:** This is acknowledged as a PDF parsing artifact in the review instructions, not an author error.

- **Claims about missing related works:** Per the instructions, claims about missing references cannot be verified without external sources and should not be included.

- **Scope creep criticisms:** Criticisms that the benchmark excludes large-scale datasets (>50K samples), multi-label tasks, or ordinal regression are outside the paper's stated scope (ICL-compatible classification/regression). The paper explicitly acknowledges the ≤10 class constraint for ICL compatibility.

- **Solvability threshold bias toward XGBoost:** Using XGBoost performance to filter solvable datasets is a design choice that may slightly favor tree-based methods, but the paper explicitly states this criterion. The impact is indirect and not clearly biasing conclusions.

- **Claims about unreleased models or unavailable benchmarks:** Per instructions, citations to papers/models should be assumed to exist unless proven otherwise. The reviewer cannot verify release status of cited ICL models.

## Novel Insights

The paper's formalization of benchmark diversity as cross-benchmark rank reconstruction error is methodologically interesting. If accepted, this framing could shift how benchmark suites are evaluated—not just by their size or domain coverage, but by their ability to predict model rankings on external benchmarks. However, the current approach risks circularity: a benchmark optimized to reproduce rankings of existing benchmarks may simply average their biases rather than capturing genuinely novel evaluation scenarios. A stronger paper would distinguish between coverage of known ranking patterns and exploration of truly out-of-distribution tasks.

## Suggestions

- Add a random-subset baseline: compare d_r for TabPalooza against d_r for multiple random subsets of the same size drawn from the UDP. This is essential to justify the hierarchical clustering approach.

- Fully specify the rank predictor: in the paper or appendix, state the model class (linear, tree-based, neural?), feature preprocessing, training/validation split strategy, and hyperparameters.

- Report model rankings on TabPalooza: provide a table showing mean ranks and standard deviations for all 11 baseline models across the final benchmark, at minimum for one metric (e.g., AUC for classification, R² for regression).

- Provide confidence intervals for d_r: bootstrap the dataset selection and/or alignment procedure to show that reported improvements are statistically meaningful rather than point estimates from a single stochastic realization.

- Analyze the TabArena failure case: explain why meta-features fail to predict rankings on TabArena specifically, and discuss what this reveals about the relationship between meta-features and model performance.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0, 2.0]
Average score: 1.2
Binary outcome: Reject
