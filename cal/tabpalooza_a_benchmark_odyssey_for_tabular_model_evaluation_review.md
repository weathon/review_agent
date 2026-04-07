=== CALIBRATION EXAMPLE 9 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is catchy but vague — "A Benchmark Odyssey" communicates nothing technical. More importantly, the abstract makes strong claims ("superior diversity," "significantly lower reconstruction error") that rest on a circular methodological foundation, as elaborated below. The abstract also conflates two distinct properties: *diversity of datasets in meta-feature space* and *ability to reconstruct model rankings on other benchmarks*. These are not the same thing and treating them as equivalent is a core conceptual problem that haunts the entire paper.

---

### Introduction & Motivation

The motivation is reasonable — inconsistent model rankings across benchmarks are a genuine problem in the tabular ML community. However, the framing of "diversity" and "efficiency" as the two competing axes is introduced without formal definitions. The claim that "most existing benchmarks are limited to fewer than five attributes" (line 66–67) is stated without a systematic comparison to prior work. The contributions, while adequately listed, are modest for an ICLR submission: (1) a new benchmark, (2) a diversity assessment pipeline, and (3) empirical validation of that pipeline. There is no novel methodological insight beyond engineering choices.

---

### Methodology (Section 3)

**3.1.1 — The Core Diversity Metric Has a Circular Design Flaw**

The diversity of TabPalooza is measured by its ability to *reconstruct model rankings on other benchmarks T using its datasets S*. The pipeline (Section 3.1.5) selects, for each dataset in T, the nearest neighbor in S by Euclidean distance in meta-feature space, and uses the matched S-dataset's model rankings as a proxy. A lower mean rank difference (d_r) is claimed to indicate higher diversity.

However, this is measuring *coverage and representativeness*, not diversity per se. A benchmark that exhaustively covers all regions of meta-feature space will score well by this metric, but TabPalooza is constructed to do exactly that (by maximizing coverage via hierarchical clustering over the UDP, which is itself a superset of the target benchmarks). The circularity: TabPalooza was derived from the same pool of datasets as the target benchmarks, so of course it can find close neighbors. This is not evidence of "superior diversity" — it is evidence of good recall from a shared pool.

**3.1.2 — Baseline Model Selection**

The claim that three representatives were chosen from tree-based and NN-based models "with diverse ranking performances across different benchmarks" lacks specificity. Why these three and not others? How was diversity in ranking performances operationalized?

**3.1.5 — Benchmark Alignment Pipeline**

The nearest-neighbor assignment allows duplicate selections: the same source dataset could be selected multiple times if it is closest to multiple target datasets. There is no discussion of handling ties or degenerate cases. Furthermore, the Euclidean distance in raw meta-feature space (112 features of wildly different scales and distributions) is not justified — distance in this space is sensitive to feature scaling and may not reflect semantic similarity between datasets.

**3.2 — Efficiency Guarantee**

This section is essentially two sentences. "Efficiency" is not formally guaranteed — selecting one dataset per cluster does not bound the computational cost in any rigorous sense. The predefined "acceptable range" for mean rank difference is never quantified. This section is incomplete.

---

### Experiments & Results (Section 4)

**4.1 — Dataset Curation**

*Using LLMs for dataset validation* (DeepSeek-r1 and Qwen3 to identify target variables in Kaggle datasets) is pragmatic but introduces systematic bias. There is no error analysis: how often do the LLMs disagree? What fraction of 2,273+1,381 Kaggle datasets were ultimately retained? The "solvability" threshold (ROC AUC > 0.55 or R² > 0.2 using XGBoost without tuning) is arbitrary and will exclude genuinely difficult datasets, which could be the most important ones for discriminating models.

*The constraint excluding datasets with >50,000 samples or >10,000 features* is a major limitation that is not adequately discussed. This excludes most large-scale real-world problems and specifically excludes datasets where gradient boosted trees most clearly dominate neural methods. The resulting benchmark may not generalize to the regime most practically relevant for industry.

*Deduplication by name + size matching* is fragile. The same dataset from different sources may differ in column encoding or preprocessing, and different datasets may share a name or size by coincidence.

**4.2 — Meta-Feature Evaluation (Tables 2 and 3)**

The paper claims meta-features "significantly improve rank estimation across most benchmarks." However, for TabArena, the ACC delta is −0.040 and the F1 delta is −0.164, meaning meta-feature-based prediction is *worse than a naive mean-rank baseline*. The paper acknowledges "exceptions" but does not explain them. This is a meaningful failure — TabArena is a recent, large, and high-quality benchmark, and the failure to predict rankings there undermines the claim that meta-features reliably capture model performance differences.

Furthermore, the dr values in Table 2 remain large throughout (0.3–0.6 range). No context is given for interpretation: what is d_r for a random predictor? What does 0.5 mean in practice? Without bounds or reference points, it is impossible to determine whether meta-features are actually useful.

**4.3 — Benchmark Size Selection**

The inflection point method for choosing 100 (classification) and 140 (regression) datasets is subjective: "where the curve becomes relatively flat" is not a principled criterion. No statistical test, no sensitivity analysis. Why not 90 or 120? Why does the regression benchmark end up 40% larger than the classification benchmark, even though the regression UDP (335) is smaller than the classification UDP (501)? This asymmetry is not explained.

**4.4 — Benchmark Alignment Evaluation (Tables 4 and 5)**

The d_r values in Table 4 for classification are in the range 0.29–0.53. These are not obviously "good" without comparison. The paper does not provide the d_r for a random subset of 100 datasets, which is the most natural baseline. If a random subset achieves similar d_r, then the elaborate hierarchical clustering + meta-feature selection pipeline adds no value.

The paper uses a single configuration (Kendall+0.07 for classification, dCor+0.12 for regression) but provides no justification for choosing this over other configurations, beyond calling it "best-performing." If configuration was selected based on the same evaluation data used to report results, this is overfitting.

**4.5 — Comparison with Other Benchmarks (Figures 2 and 3)**

This section makes the central empirical claim: TabPalooza has lower d_r when used as a source (i.e., it can reconstruct others) and higher d_r when used as a target (i.e., others cannot reconstruct it). The second part of this claim is presented as evidence of diversity but is equally consistent with TabPalooza being an *outlier* — containing unusual datasets that other benchmarks simply don't sample. The paper does not distinguish between these interpretations.

Critically, the heat maps in Figures 2 and 3 are not reproduced in the extracted text. The paper's central comparative claim rests on these figures, which are unavailable in this form. This is a parser issue, but it underscores that the paper's argumentation is heavily figure-dependent, and the quantitative tables provided elsewhere are insufficient to stand alone.

**Missing Experiments — Critical Omissions**

1. **No model ranking table on TabPalooza itself.** A benchmark paper's primary deliverable is enabling model comparison. There is no table showing how the 11 evaluated models rank on TabPalooza. Without this, the paper doesn't demonstrate its benchmark's utility.

2. **No ablation of the dataset selection pipeline.** How does hierarchical clustering compare to random sampling, k-medoids, or greedy maximum diversity? Without this, it's unclear whether the pipeline adds value.

3. **No statistical significance testing.** All d_r differences are point estimates. There are no confidence intervals, bootstrapped error bars, or significance tests.

4. **No contamination analysis.** TabPFN-v2, TabICL, and other ICL models may have been trained on datasets that overlap with TabPalooza (drawn from OpenML, Kaggle, and existing benchmarks). Pre-training data contamination would invalidate comparisons involving these models.

---

### Writing & Clarity

Section 4 opens with "deep tabular prediction, machine learning" — these appear to be orphaned keyword tags, suggesting incomplete manuscript preparation. Section 3.2 (Efficiency Guarantee) is too brief to constitute a proper section. The equation for d_r appears mid-paragraph split across the page (lines 193–204), disrupting reading flow significantly. These issues go beyond formatting artifacts and reflect rushed preparation.

---

### Limitations & Broader Impact

The paper has no limitations section. Key unaddressed limitations include:

- **Size restriction**: Capping at 50k samples creates a benchmark that may not reflect large-scale ML practice.
- **Classification cap of 10 classes**: Excludes fine-grained classification tasks, skewing toward binary/few-class problems.
- **ICL model constraint driving the design**: The 10-class restriction is explicitly motivated by ICL model constraints, meaning the benchmark is designed around *current model limitations* rather than the space of real-world problems.
- **Kaggle dataset quality**: Kaggle datasets are often leaky, poorly described, or contain competition-specific artifacts. The LLM-based validation pipeline may not catch all such issues.
- **Temporal bias**: No discussion of whether datasets are time-sensitive or whether training/test splits respect temporal ordering where applicable.

---

### Overall Assessment

TabPalooza addresses a real and important problem — the proliferation of inconsistent tabular ML benchmarks — but its contributions do not meet the bar for ICLR. The central methodological claim (that lower reconstruction error implies higher diversity) is logically circular: TabPalooza is constructed from the same dataset pool as the target benchmarks and selects via nearest-neighbor matching, so naturally it scores well on the reconstruction metric. This conflation of *coverage* with *diversity* is never resolved. The experiments are incomplete: the most important result — model rankings on TabPalooza — is absent entirely, and critical baselines (random subsets, alternative selection strategies) are missing. The efficiency guarantee is unsubstantiated. The meta-feature analysis shows that meta-features sometimes *hurt* rank prediction on high-quality target benchmarks (TabArena), which contradicts the paper's narrative without explanation. The manuscript also shows signs of incomplete preparation (orphaned keywords, two-sentence methodology sections). While the engineering effort in curating ~800 datasets is considerable and the dataset release is a worthwhile contribution to the community, the paper as written does not provide sufficient methodological rigor or empirical completeness to justify acceptance at ICLR. A substantial revision addressing the circularity of the diversity metric, adding model comparison results, and providing meaningful baselines would be required.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **TabPalooza**, a new benchmark for tabular data evaluation designed to maximize dataset diversity while maintaining computational efficiency. The authors introduce a quantitative pipeline utilizing rank reconstruction error ($d_r$) to assess and select datasets that best align with existing standards. Empirical comparisons indicate that TabPalooza achieves lower reconstruction error than competing benchmarks while evaluating a wide spectrum of models, from tree-based to recent in-context learning architectures.

### Strengths
1. **Robust Model Coverage:** The evaluation includes a diverse lineup of 11 baseline models spanning tree-based (XGBoost, CatBoost), neural network (ExcelFormer, ResNet), and emerging In-Context Learning (TabPFN, TabICL, LimiX) approaches (Section 3.1.2).
2. **Methodological Contribution to Benchmarking:** The introduction of benchmark diversity assessment via reconstruction error ($d_r$) offers a novel, quantitative metric beyond simple meta-feature overlap, allowing for the empirical comparison of benchmark quality (Section 3.1.1).
3. **Comprehensive Dataset Curation:** The effort to aggregate and curate over 800 datasets from multiple sources (TALENT, OpenML-CC18, Kaggle, etc.), including rigorous filtering for solvability, significantly expands the pool of available data (Section 4.1).
4. **Practical Efficiency:** The proposed selection pipeline uses hierarchical clustering to distill a large universal dataset pool into a manageable size (100/140 datasets) without losing alignment with the full benchmark suite (Section 4.3-4.4).

### Weaknesses
1. **Circular Validation of Diversity:** The primary metric for benchmark diversity ($d_r$) relies on the rank reconstruction of the specific baseline models selected to construct the benchmark. This risks circular reasoning where benchmarks optimized for these models simply reinforce their own rankings (Section 4.5).
2. **Opacity in LLM-Assisted Curation:** The use of proprietary LLMs (DeepSeek, Qwen) to automate dataset parsing for Kaggle introduces potential non-determinism and bias that is not fully transparent due to the "black box" nature of the inference API (Section 4.1).
3. **Inconsistent Metric Stability:** The diversity metric depends heavily on the stability of performance rankings, which can fluctuate due to hyperparameters (e.g., AutoGluon has fixed constraints, others use Optuna), potentially skewing the assessment of benchmark alignment (Section 3.1.2).
4. **Questionable Future Dated Citations:** Several references cite papers from 2025 (e.g., TabPFN-v2, LimiX) which, if not preprints, may indicate versioning confusion that undermines the clarity of the related work landscape (References section).

### Novelty & Significance
The **novelty** lies primarily in the methodology for evaluating benchmark quality rather than the dataset collection itself. Defining "diversity" through model rank reconstruction error is a theoretically interesting departure from standard meta-feature clustering. However, the **significance** is contingent on the metric's ability to generalize to future, unseen algorithms. **Clarity** is generally high, despite some necessary inference of garbled equations due to extraction issues. **Reproducibility** is supported by the public HuggingFace release, though the LLM filtering steps require specific prompt documentation to be fully reproducible. Overall, the work makes a meaningful contribution to the community resource of tabular ML, provided the evaluation metrics are refined.

### Suggestions for Improvement
1. **External Validation:** Validate the benchmark diversity metric using a hold-out set of models not used in the dataset selection or reconstruction process to prevent circular bias.
2. **Document LLM Prompts:** Provide the specific prompts and logic used by the DeepSeek/Qwen models to parse Kaggle datasets in the supplementary materials to ensure reproducibility.
3. **Standardize Evaluation:** Clarify the hyperparameter search settings for baselines to ensure the "ranking" metric isn't artificially influenced by varying search constraints (e.g., 5-fold vs 6-fold, Optuna space vs AutoGluon defaults).
4. **Refine Title:** Consider removing "Odyssey" from the title to adhere to stricter academic conventions typical of ICLR submissions.
5. **Clarify Citations:** Ensure all citations (especially those dated 2025) are clearly labeled as preprints or arXiv versions to maintain bibliographic accuracy.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Holdout Generalization Correlation:** Evaluate whether model rankings on TabPalooza correlate more strongly with rankings on the full Universal Dataset Pool (UDP) than competing benchmarks do. Without this, the claim that TabPalooza offers "superior diversity" is unsubstantiated, as alignment error ($d_r$) does not guarantee better generalization to unseen data.
2. **Computational Efficiency Measurement:** Report actual wall-clock time and GPU hours required to evaluate all baselines on TabPalooza versus competitors. Claiming "Efficiency" based solely on dataset count is misleading if the selected datasets are significantly larger or more complex than those in existing benchmarks.
3. **Selection Bias Ablation:** Re-evaluate benchmark diversity without the XGBoost AUC > 0.55 filtering criterion used for Kaggle datasets. This filter likely excludes difficult datasets where neural networks might outperform trees, directly biasing the benchmark's conclusions regarding model superiority.

### Deeper Analysis Needed (top 3-5 only)
1. **Metric Validity Justification:** Provide evidence that lower reconstruction error ($d_r$) actually translates to more reliable model selection in downstream practical tasks. Currently, $d_r$ is an abstract proxy with no proven link to the ultimate goal of identifying the best model for a new user dataset.
2. **Meta-Feature Predictive Stability:** Analyze the variance of meta-feature correlations across different model families (e.g., Trees vs. NNs). If meta-features predict tree performance well but fail for NNs, the diversity metric is inherently skewed towards tree-based evaluation, undermining the benchmark's neutrality.
3. **Statistical Significance Testing:** Perform significance testing (e.g., Wilcoxon signed-rank) on the rank differences between TabPalooza and other benchmarks. The reported improvements in Tables 4-5 may be within the noise margin, and without significance tests, the claimed superiority is statistically weak.

### Visualizations & Case Studies
1. **Ranking Correlation Scatter:** Plot model rankings on TabPalooza vs. the full UDP for all benchmarks side-by-side. This would visually expose whether TabPalooza truly preserves the global ranking structure better than alternatives or merely approximates them.
2. **Dataset Difficulty Distribution:** Visualize the distribution of dataset difficulty (e.g., best possible accuracy or entropy) across TabPalooza vs. competitors. This would reveal if the XGBoost filtering created a ceiling effect that hides performance differences on hard tasks.
3. **Failure Case Heatmap:** Show specific datasets where TabPalooza's ranking prediction fails significantly compared to other benchmarks. This identifies blind spots in the meta-feature selection and clustering process that the aggregate metrics hide.

### Obvious Next Steps
1. **Reproducibility of Baselines:** Replace or provide verified code for the cited 2025 models (TabPFN-v2, Mitra, LimiX) which are currently unavailable. A benchmark relying on unpublished or inaccessible models cannot be adopted by the community for immediate evaluation.
2. **Pipeline Code Release:** Release the exact scripts for meta-feature extraction and dataset clustering, as minor variations in preprocessing drastically alter meta-feature values. The provided HuggingFace link appears to be a placeholder, preventing immediate verification of the dataset composition.
3. **Formalize Metric Definition:** Clarify the mathematical definition of reconstruction error ($d_r$) which is currently ambiguous in the text. Precise formalization is required to prevent independent implementations from diverging due to interpretation differences.

# Final Consolidated Review
## Summary

TabPalooza proposes a new benchmark for tabular data classification and regression, designed around two principles: diversity and efficiency. The authors construct a Universal Dataset Pool (UDP) from existing benchmarks and Kaggle, introduce a reconstruction error metric (d_r) to assess benchmark quality, and use hierarchical clustering to select a compact representative subset. The paper claims superior diversity compared to existing benchmarks based on lower reconstruction error.

## Strengths

- **Comprehensive dataset curation effort**: The paper aggregates 501 classification and 335 regression datasets from multiple sources (TALENT, OpenML-CC18, PFN, TabZilla, TabArena, Kaggle), applies systematic filtering for solvability and deduplication, and releases this collection publicly. This represents substantial engineering effort that benefits the community.
- **Novel evaluation methodology for benchmarks**: The reconstruction error metric (d_r) provides a quantitative framework for comparing benchmark quality, moving beyond simple dataset counts or meta-feature overlap. This offers a principled way to assess how well a benchmark captures the ranking behavior of models on a larger pool.
- **Modern model coverage**: The evaluation includes 11 baseline models across four categories: tree-based (XGBoost, CatBoost, Random Forest), neural networks (ExcelFormer, MLP, ResNet), in-context learning models (TabPFN-v2, TabICL, LimiX, Mitra), and AutoML (AutoGluon). Including recent ICL-based models makes the benchmark forward-looking.

## Weaknesses

- **No comparison to random subset selection**: The most critical methodological gap is the absence of a baseline comparing the hierarchical clustering selection to simple random sampling. If a random subset of 100/140 datasets achieves similar reconstruction error to the carefully curated TabPalooza, then the elaborate selection pipeline adds no demonstrable value. This comparison is essential to validate the proposed methodology.

- **No model rankings shown for TabPalooza itself**: A benchmark paper's primary utility is enabling model comparison, yet the paper contains no table showing how the 11 evaluated models rank on TabPalooza. Without this, readers cannot assess what conclusions the benchmark supports or how rankings compare to other benchmarks.

- **Conceptual conflation of coverage with diversity**: The paper frames its metric as measuring "diversity," but d_r measures *reconstruction ability*—how well rankings on TabPalooza predict rankings on other benchmarks. A benchmark could achieve low reconstruction error by simply containing datasets very similar to those in target benchmarks (good coverage of shared regions), not necessarily by being diverse in meta-feature space. The distinction between coverage and diversity is not addressed, and the hierarchical clustering procedure maximizes spread in meta-feature space but the validation metric measures something different.

- **Dataset filtering biases toward tree-favorable regimes**: The paper excludes classification datasets where XGBoost achieves AUC < 0.55 and regression datasets where R² < 0.2 (Section 4.1). This filtering excludes genuinely difficult datasets where different model families might show more variance—precisely the datasets most valuable for discriminating model capabilities. The 10-class maximum restriction, explicitly motivated by ICL model constraints, further shapes the benchmark around current model limitations rather than the space of real-world problems.

- **Negative meta-feature prediction results unexplained**: In Table 2, meta-feature-based rank prediction performs *worse* than the baseline for TabArena on ACC (delta = -0.040) and F1 (delta = -0.164). The paper acknowledges "exceptions" but offers no explanation for why meta-features hurt prediction on this recent, large benchmark. This failure case deserves analysis.

- **No statistical significance testing**: All reported d_r values are point estimates without confidence intervals or significance tests. Without uncertainty quantification, it's unclear whether observed differences between benchmarks are meaningful or within noise margins.

- **No ablation of the selection pipeline**: The paper doesn't examine how sensitive the results are to the choice of correlation method (Kendall vs. Pearson vs. dCor) or threshold values. The selection of Kendall+0.07 for classification and dCor+0.12 for regression is described as "best-performing" but without systematic comparison to alternatives.

## Nice-to-Haves

- **Holdout generalization validation**: Evaluating whether model rankings on TabPalooza correlate with rankings on held-out datasets (not in UDP) would strengthen claims about the benchmark's predictive validity for new problems.
- **Computational efficiency measurement**: The paper claims "efficiency" based solely on dataset count, but wall-clock time and computational cost would provide practical guidance for users.
- **Contamination analysis for ICL models**: TabPFN-v2, TabICL, and other ICL models may have been pre-trained on datasets that overlap with TabPalooza (drawn from OpenML, Kaggle, existing benchmarks). Analysis of potential contamination would strengthen the benchmark's validity for these models.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **"Benchmark Odyssey" title criticism**: This is a style preference, not a substantive critique.
- **"Questionable future-dated citations"**: The 2025 citations (TabPFN-v2, LimiX, Mitra, etc.) appear to be arXiv preprints, which is standard practice. As per review guidelines, assume cited works exist.
- **"Models unavailable for reproducibility"**: Same as above—if papers are cited and linked, assume they exist.
- **"Orphaned keywords in Section 4"**: This is a formatting artifact from extraction, noted in the paper header as a parser issue.
- **"Section 3.2 too brief"**: While accurate, this is a stylistic complaint about section length, not substance.
- **"Deduplication by name+size is fragile"**: The paper describes matching by both dataset names AND sizes. Different datasets coincidentally sharing both name and size is rare. This criticism overstates the issue.

## Novel Insights

The most notable insight is the tension between **benchmark validity and benchmark utility**. TabPalooza optimizes for reconstruction of *existing* benchmarks' ranking behavior—but the entire motivation is that existing benchmarks give inconsistent rankings. If existing benchmarks disagree with each other (the stated problem), then optimizing to match them precisely may perpetuate their inconsistencies rather than resolving them. The paper would benefit from explicitly addressing whether matching existing benchmarks' rankings is the right goal, or whether the benchmark should instead identify datasets that *maximize* model performance variance to better discriminate between approaches.

## Suggestions

1. **Add a random subset baseline**: Compare TabPalooza to uniformly random subsets of equal size from UDP. This directly tests whether the selection pipeline adds value.
2. **Include model ranking tables**: Show the actual performance rankings of all 11 models on TabPalooza across metrics. This is the minimum expected content for a benchmark paper.
3. **Analyze the TabArena failure case**: Explain why meta-features hurt rank prediction on TabArena for ACC and F1 metrics. Understanding this boundary case would strengthen the methodology.
4. **Add confidence intervals**: Report bootstrap confidence intervals for d_r values to establish whether differences are statistically meaningful.
5. **Document the LLM prompts**: For reproducibility, include the exact prompts provided to DeepSeek-r1 and Qwen3 for Kaggle dataset parsing in an appendix or supplementary materials.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0, 2.0]
Average score: 1.2
Binary outcome: Reject
