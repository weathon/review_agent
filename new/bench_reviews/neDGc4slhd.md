The paper proposes using topological data analysis—specifically persistent homology and Betti curve similarity (BCS)—to analyze the global functional structure of deep neural networks. It presents an empirical study training CNNs on 30 disjoint ImageNet subsets, extracting activations, reducing them via k-means++, computing persistent homology, and comparing Betti curves across models and epochs. Results suggest BCS captures training dynamics and distinguishes models on datasets with different class characteristics, potentially revealing representation differences beyond accuracy metrics.

## Strengths

- **Novel application of topological summaries**: BCS is applied, to the authors’ knowledge for the first time, to compare DNNs across disparate datasets and across training epochs (Section 2.5, Eq. 7), opening a new perspective on network analysis.
- **Large-scale empirical setup**: The study covers 30 disjoint ImageNet subsets, four CNN architectures, and seven checkpointed epochs, providing a substantial dataset for topological comparison.
- **Clear pedagogical introduction to PH**: Section 2.5 offers an accessible tutorial on persistent homology and Betti curves, lowering the barrier for non-experts.
- **Reproducibility infrastructure**: Code, random seeds, hyperparameters, and hardware details are disclosed, and a GitHub repository is referenced.
- **Informative visualizations**: The flowchart (Fig. 1), persistence diagrams/Betti curves (Fig. 3), and heatmaps of BCS (Figs. 4–9) effectively communicate the analysis pipeline and main patterns.

## Weaknesses

### Fatal
None. The exploratory nature means reported patterns are not claimed as universal laws; however, serious methodological concerns limit confidence in the findings (see Major).

### Major

1. **Faulty metric space foundation**: The distance function \(d_\rho = \sqrt{1 - |\rho|}\) is explicitly acknowledged to fail positivity (Sec. 2.4), yet the authors characterize the activation space as a “finite metric space” and do not discuss consequences for Vietoris–Rips complex construction or PH interpretation. This conceptual mismatch undermines theoretical rigor.
2. **Unsupported dimensionality reduction**: Activations are reduced from thousands to 1000 points via k-means++ with silhouette scores indicating poorly separated clusters. The claim that this “captures the global structure in a non-linear way” (Sec. 2.3) is unsupported by ablation; sensitivity to \(k\), algorithm choice, or random seed is unexplored, risking topological artifacts.
3. **Ad hoc and uninterpretable similarity measure**: BCS is defined as the \(L^\infty\) distance between Betti curves without theoretical justification. This discards temporal structure of the filtration; the authors neither compare to standard PH distances (bottleneck, Wasserstein) nor provide a rationale for why \(L^\infty\) meaningfully captures network similarity. The measure is therefore uninterpretable.
4. **Missing baseline comparisons**: No comparison to established network similarity metrics (CKA, SVCCA, RSA, weight-space MMD) is made. Without baselines, it is unclear whether BCS offers new insights or merely recapitulates known phenomena.
5. **Absence of statistical validation**: Trend claims (e.g., increasing self-similarity over epochs, low BCS on distinct subsets) rest on visual inspection of figures. No confidence intervals, multiple seeds, or hypothesis tests are reported, making the evidence anecdotal.

### Minor

1. **Terminology and labeling inconsistencies**: Text and figure captions interchangeably use “distance” and “similarity” for BCS (e.g., Fig. 4–6), and color scales are not consistently described, potentially confusing readers.
2. **Extended LeNet architecture**: The extended LeNet adds two linear layers, increasing capacity relative to the original. This modification is not discussed in relation to topological results, though it may affect comparisons.
3. **Insufficient motivation for PH**: The paper does not argue why persistent homology should capture learning dynamics more effectively than simpler spectral or covariance-based analyses.
4. **No layer-wise analysis**: Activations from all layers are pooled; analyzing layers separately could reveal which stages drive BCS and strengthen conclusions.

### Trivial
None beyond minor presentation issues already noted.

## Nice-to-Haves

- Ablation over the number of clusters \(k\) and alternative reduction methods (random sampling, PCA) to verify robustness of topological patterns.
- Correlation analysis between BCS and performance metrics (accuracy, generalization gap) to interpret what topological similarity reflects.
- Layer-specific or block-specific BCS to pinpoint where network representations differ.
- Comparison of BCS with CKA/SVCCA across the same models and subsets to contextualize findings.
- Experiments on non-CNN architectures (e.g., Transformers) and datasets beyond ImageNet subsets.
- Theoretical discussion linking Betti numbers to network properties such as capacity, overfitting, or robustness.

## Removed Points

These points are flagged to be removed; they either misread the paper or are not substantive weaknesses.

- *Disjoint ImageNet subsets are “highly fragmented” and limit generalization.* The authors explicitly state that 30 disjoint subsets are used to achieve statistical significance without excessive computation; this is a valid design choice, not a flaw.
- *The paper “confuses whether points are individual neuron activations (scalars) or activation vectors.”* The methodology clearly aggregates activations from all layers into an \(M\times N\) matrix, treating each row as a high-dimensional vector; the criticism arises from a misreading.
- *The term “non-linear” is misapplied to k-means++.* While k-means is not a nonlinear manifold method, the authors cite prior work supporting its use for PH; this is a minor terminology preference, not a substantive weakness.
- *Any formatting inconsistencies attributed to parser errors.* The original submission is assumed correctly formatted; such artifacts are not author errors.

## Novel Insights

The reviews collectively highlight that applying topological data analysis to neural networks demands rigorous validation of preprocessing steps (distance function, dimensionality reduction) and careful interpretation of summary statistics. A key insight is that a novel similarity measure must be benchmarked against established methods and shown robust to implementation choices; otherwise observed patterns may be artifacts. Moreover, the tension between preserving global topology and achieving compact representation via k-means++ reveals a general challenge in scaling PH to high-dimensional activation data.

## Suggestions

1. **Clarify the metric foundation**: Either adopt a proper metric (e.g., \(1-|\rho|\) with careful handling of zeros) or explicitly frame the analysis in terms of pseudometric spaces and discuss implications for Vietoris–Rips construction.
2. **Validate the reduction pipeline**: Run PH on the full activation set for a subset of data to confirm that k-means++ preserves key topological features; perform ablations on \(k\) and alternative reduction strategies.
3. **Ground BCS with comparisons**: Compute standard network similarity measures (CKA, SVCCA) on the same data and correlate with BCS; also experiment with bottleneck or Wasserstein distances on persistence diagrams.
4. **Add statistical rigor**: Repeat experiments with multiple random seeds, report means and standard deviations for BCS, and apply statistical tests to support trend claims.
5. **Improve presentation**: Standardize terminology (use “distance” or “similarity” consistently), clearly explain figure color scales, and discuss the extended LeNet’s impact.

## Calibration Anchors

I compared the paper against the following human-reviewed anchors to calibrate the score:

- `/home/wg25r/review_agent/human_reviews/EzjsoomYEb.md` (avg 8.0, Accept Oral): Novel expressivity analysis of topological deep learning with strong empirical gains and new benchmarks. My paper shares topical novelty but lacks rigorous validation and baselines.
- `/home/wg25r/review_agent/human_reviews/0JsRZEGZ7L.md` (avg 8.0, Accept Poster): Introduces a differentiable cell complex module with strong experiments on diverse datasets. My paper’s empirical design is broad but not as convincingly connected to improvements.
- `/home/wg25r/review_agent/human_reviews/X6y5CC44HM.md` (avg 5.75, Accept Poster): MANTRA benchmark for topological deep learning; accepted as a resource paper with systematic evaluation. My paper is not a benchmark but an application; its evaluation is weaker.
- `/home/wg25r/review_agent/human_reviews/R4gqcDRJ9l.md` (avg 5.75, Reject): TopoFR with SOTA face recognition results but rejected due to limited novelty and methodological concerns. My paper similarly lacks baseline comparisons and suffers from similar validation issues, supporting a sub‑6 score.
- `/home/wg25r/review_agent/human_reviews/upoxXRRTQ2.md` (avg 5.0): Empirical study combining theory and experiments on subset learning. My paper lacks theoretical grounding and strong empirical validation, placing it below this anchor.
- `/home/wg25r/review_agent/human_reviews/ZHTYtXijEn.md` (avg 2.33, Reject): Criticized for poor baselines and inadequate experiments. My paper shares missing baselines and no statistical tests, though it is better organized and on a larger scale, preventing a score as low as 2.33.

Relative to these anchors, the paper’s strengths (novel application, extensive data collection) are offset by major methodological and evaluative shortcomings, positioning it in the below‑average to borderline range.

## Score and Decision

The core idea is promising, but execution lacks sufficient validation through baselines, statistical testing, and sensitivity analysis. The theoretical foundation is shaky due to the non‑metric distance treatment. Major revisions would be required before consideration for publication.

**Final score: 3.5**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>