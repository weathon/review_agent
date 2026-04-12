=== CALIBRATION EXAMPLE 18 ===

# Final Consolidated Review
## Summary
This paper introduces **TabPalooza**, a new benchmark for tabular classification and regression built from a large universal pool of datasets, together with a pipeline for selecting a compact subset intended to balance **coverage of tabular problem regimes** and **evaluation efficiency**. The key technical idea is to assess a benchmark by how well model-rank behavior on other benchmarks can be reconstructed from it using dataset meta-features, and then use clustering in meta-feature space to select representative datasets.

The paper is ambitious and potentially useful to the tabular-learning community, especially because it attempts to move benchmark design from ad hoc curation toward a quantitative procedure. However, the central claim is currently overstated: the proposed metric primarily measures **rank-representativeness / alignment**, not diversity in the broader sense the paper repeatedly claims, and the evaluation protocol appears to tune TabPalooza using the same target benchmarks on which it is later declared superior.

## Strengths
- **The paper tackles a real and specific problem in tabular ML benchmarking:** model rankings vary substantially across existing benchmark suites, and the paper directly targets this instability by trying to build a benchmark that better covers multiple evaluation regimes rather than optimizing for a single benchmark lineage.
- **The benchmark construction starts from a substantially broader pool than typical tabular benchmarks.** The paper assembles a Universal Dataset Pool of **501 classification** and **335 regression** datasets after curation and deduplication, which is materially larger than the individual benchmark suites it compares against.
- **The paper contributes a concrete benchmark-selection pipeline rather than only a new dataset list.** In particular, it combines (i) extraction of **112 meta-features**, (ii) benchmark alignment via nearest-neighbor matching in meta-feature space, and (iii) hierarchical clustering to produce a compact subset under a size budget.
- **The evaluation spans multiple model families, including newer tabular foundation / ICL-style methods.** The baseline suite includes tree models, neural nets, AutoML, and ICL-based approaches, which makes the resulting ranking-based analysis more relevant than benchmarks restricted to only GBDTs vs MLPs.
- **An interesting empirical signal does emerge from the results:** benchmark choice materially changes model-rank behavior, and a benchmark selected from a larger heterogeneous pool can better reconstruct the rank behavior of multiple existing suites than those suites reconstruct one another. That is a useful observation even if the paper’s “diversity” terminology is too strong.

## Weaknesses

###: Fatal
- **The paper’s central evaluation target is mischaracterized: \(d_r\) does not directly measure “diversity” as claimed, but rather cross-benchmark rank representativeness/alignment.**  
  The paper explicitly states that “**Diversity is assessed**” via reconstruction error and that “**A smaller value of \(d_r\) indicates that the source benchmark better preserves the ranking behavior of the target benchmark**.” This is a coherent measure of whether a benchmark covers the *performance regimes* present in another benchmark. But that is not the same as diversity in the ordinary sense of breadth, novelty, heterogeneity, or support over dataset characteristics. A benchmark could achieve low \(d_r\) by being highly representative of common ranking regimes without necessarily containing unusual or edge-case datasets.  
  This matters because the headline claim—“superior diversity compared to existing alternatives”—is stronger than what the metric justifies. The paper has a valid story around **coverage / representativeness under a budget**, but its core framing overclaims.

- **There is a serious risk of evaluation leakage / circularity in how TabPalooza is configured and then evaluated.**  
  The paper says in Section 4.4: “**For classification tasks, we construct the TabPalooza using the setting Kendall+0.07**” and “**For regression tasks, we construct TabPalooza using the configuration dCor+0.12, which was identified as the best-performing setting in Section 4.1.**” It also states in Section 4.3 that size is chosen by varying benchmark size and selecting the “**inflection point**” of average rank difference.  
  As written, both the benchmark size and the meta-feature-selection configuration are chosen using the same benchmark family later used for evaluation. That means the paper is not cleanly separating **construction/tuning** from **held-out assessment**. The resulting superiority claims in Section 4.5 therefore cannot be interpreted as unbiased evidence that the method generalizes; they may partly reflect optimization to the evaluation targets.

### Major:
- **The benchmark-selection method lacks proper subset-selection baselines, so it is unclear how much benefit comes from the proposed pipeline versus simply selecting any moderately sized subset from a large pool.**  
  The paper compares TabPalooza mainly to existing benchmark suites, not to alternative ways of building a 100/140-dataset subset from the same UDP. Missing baselines include simple random sampling, stratified sampling by coarse meta-features, k-means / medoid selection, or farthest-point/max-coverage heuristics. Without those controls, the evidence does not isolate the value of the proposed clustering-and-alignment procedure.

- **The choice of final benchmark size is weakly justified.**  
  Section 4.3 says the authors “**select the inflection point where the curve becomes relatively flat**” and therefore set TabPalooza to **100 classification** and **140 regression** datasets. This is heuristic and not operationalized. There is no quantitative criterion for the elbow, no uncertainty estimate, and no comparison against nearby sizes. Since efficiency is one of the paper’s two main design goals, the size-budget decision should be more rigorously defended.

- **The nearest-neighbor alignment and clustering depend on Euclidean distances over heterogeneous meta-features, but the paper does not specify how these features are normalized before distance computation.**  
  The extracted features include counts, ratios, statistical summaries, information-theoretic quantities, landmarking features, and model-based features. Section 3.1.5 states that Euclidean distance is computed “**based on a selected set of meta-features**,” but the paper does not explain scaling/standardization. If raw Euclidean distance is used, the geometry of the benchmark construction could be dominated by a subset of high-variance dimensions. This is a substantive methodological omission because distance drives both alignment and clustering.

- **The evidence for “efficiency” is underdeveloped.**  
  In practice, the paper defines efficiency almost entirely by limiting dataset count. That is reasonable as a proxy, but the claimed efficiency benefit would be much stronger if accompanied by actual evaluation-cost measurements (e.g., total training/evaluation time across baselines or normalized compute saved relative to larger suites). As written, the efficiency claim is plausible but only partially demonstrated.

- **The data curation/filtering choices are consequential but insufficiently justified.**  
  The paper filters out datasets above 50k training samples or 10k features, excludes classification datasets with more than 10 classes, and also removes Kaggle datasets if XGBoost underperforms thresholds (AUC < 0.55, \(R^2 < 0.2\)). These choices substantially shape the benchmark and may bias it toward moderate-scale, relatively solvable problems. Some of these filters are understandable for including ICL baselines, but the resulting scope limitation should be made explicit and its effect analyzed more carefully.

### Minor
- **The regression meta-feature pipeline relies on discretizing continuous targets into 10 equal-frequency bins for PyMFE-derived features, but this adaptation is not validated.**  
  The paper acknowledges PyMFE is classification-oriented and adapts it by target discretization. This is a reasonable expedient, but for a benchmark paper, some evidence is needed that this surrogate representation does not distort regression dataset similarity too severely.

- **The meta-feature predictor results are mixed in places, and the text slightly oversells their consistency.**  
  The paper claims meta-features significantly improve rank estimation “across most benchmarks,” which is mostly true, but there are explicit exceptions in Table 2, including negative deltas for TabArena on ACC and F1. This does not invalidate the section, but the narrative should emphasize the heterogeneity more carefully.

- **The final benchmark composition appears partly stochastic.**  
  Section 3.2 states that after hierarchical clustering, “**one dataset is randomly selected**” from each cluster. If that is the final procedure, benchmark identity may vary with the random seed. For a benchmark paper, the exact released subset is what ultimately matters, so the paper should be explicit about determinism, seeds, and whether the released benchmark is a single fixed draw or a canonical medoid-like selection.

- **The paper would benefit from clearer decomposition by model family.**  
  Since \(d_r\) aggregates over all baselines, it is hard to tell whether the observed alignment is uniformly good across tree, NN, ICL, and AutoML methods, or driven mainly by one family.

### Trivial
- **Tie handling in rank computation is not specified.**  
  Since the benchmark metric is based on mean model ranks, the handling of ties can affect \(d_r\), especially when several methods perform similarly.

## Nice-to-Haves
- Add subset-selection baselines using the same UDP and the same size budget: random sampling, stratified random sampling, k-medoids / k-means representatives, farthest-point sampling.
- Evaluate a genuinely held-out benchmark-construction setting: choose correlation method, threshold, and size on one subset of benchmarks and report final \(d_r\) on unseen benchmark suites or withheld dataset domains.
- Report uncertainty for \(d_r\): bootstrap confidence intervals over datasets and/or multiple random seeds for the cluster sampling step.
- Provide per-family and per-model \(d_r\) breakdowns to check whether TabPalooza is equally representative for tree models, neural models, and ICL models.
- Show 2D projections or coverage plots of UDP vs TabPalooza vs existing benchmarks in meta-feature space to support the intended “coverage” intuition.
- Quantify efficiency with actual wall-clock or compute-budget comparisons, not just dataset count.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concern that cited models/tools/benchmarks may not exist or were unavailable.**  
  Removed per instruction. The paper cites them, so existence/release status should not be questioned.

- **Complaint about reproducibility due to not publishing all seeds/indices/prompts/hyperparameter trajectories.**  
  Partially removed as a strict weakness. Exact benchmark release details would improve reproducibility, but lack of every implementation artifact is not itself a core flaw for this submission.

- **Criticism that AutoGluon’s 600-second budget makes the baseline comparison unfair.**  
  The paper is not making a direct claim that all baselines were given identical compute budgets; rather, it uses a mixed baseline suite to induce model rankings. Since any asymmetry here does not obviously favor the proposed benchmark construction in a way that invalidates the main claim, this is not a central weakness.

- **Claim that LLM-based Kaggle parsing is inherently unreliable and invalidates the benchmark.**  
  This is too strong based on the paper alone. The paper does at least apply two models and retains only consistent outputs. It remains fair to ask for more validation, but not to treat this as a decisive flaw absent evidence of actual curation errors.

- **Formatting/style issues and parser artifacts.**  
  Removed.

## Novel Insights
The most useful way to reinterpret this paper is not as a benchmark proving “greater diversity,” but as an attempt to construct a **small benchmark whose induced model ranking is maximally representative of a larger universe of tabular evaluation regimes**. Under that interpretation, the paper contains a potentially valuable contribution: it suggests benchmark design can be framed as a **compression problem over rank behavior**, using meta-features as the bridge between datasets. This reframing is stronger and more defensible than the current diversity-centric language, and it could become genuinely influential if evaluated with a leakage-free protocol and stronger subset-selection baselines.

## Suggestions
- Reframe the core claim from **diversity** to **rank-representativeness / coverage under a budget**, unless you add an independent diversity analysis that truly measures support over dataset characteristics.
- Introduce a clean train/validation/test protocol for benchmark construction: use one split of benchmarks or datasets to choose meta-feature filter, threshold, and size; report final results on held-out benchmarks not used for selection.
- Add direct subset-selection baselines on the same UDP and same budget to isolate the value of your pipeline.
- Specify and justify the preprocessing for meta-features before Euclidean distance computation; if features are standardized, say so explicitly.
- Replace the heuristic “inflection point” size choice with a quantitative criterion and sensitivity analysis around nearby sizes.
- Validate the regression meta-feature adaptation or use a regression-native meta-feature extraction strategy.
- Report uncertainty estimates and, if random cluster sampling remains part of the method, average over multiple seeds or define a deterministic representative-selection rule.
- Clarify the intended scope of the benchmark given the filtering choices (dataset size, feature count, number of classes, solvability thresholds), and discuss which important tabular regimes are excluded.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0, 2.0]
Average score: 1.2
Binary outcome: Reject
