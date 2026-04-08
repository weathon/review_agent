=== CALIBRATION EXAMPLE 36 ===

# Harsh Critic Review
## Section-by-Section Critical Review of DISCO: Diversifying Sample Condensation for Efficient Model Evaluation

---

### Title & Abstract

The title is apt. "Diversifying Sample Condensation" accurately conveys the central idea of selecting samples that maximise model response diversity. The abstract correctly captures the two-part contribution (sample selection via disagreement; performance prediction via model signatures) and makes concrete numerical claims (99.3% cost reduction, 1.07%p MAE on MMLU) that are supported in the body of the paper.

One minor tension: the abstract says "promoting diversity among samples is not essential; what matters is diversity in model responses." This framing is strong—arguably, high-disagreement samples *are* diverse in the traditional sense (they span the decision boundary between model families). The contrast could be drawn more carefully.

---

### Introduction & Motivation

The motivation is strong and well-evidenced with concrete GPU-hour figures (1400h for LMMs-Eval, 4000h for HELM). The two claimed improvements—simpler selection via disagreement and simpler prediction via model signatures—are clearly stated and directly tested.

One important caveat: the PDS metric (Rubinstein et al., 2024) used as the central selection criterion was introduced by the same first author. The introduction would benefit from being more explicit that the key selection primitive is being *repurposed* from prior OOD-detection work rather than invented here. This affects how the novelty is perceived.

---

### Method / Approach

**Proposition 1 (theoretical justification for JSD-based selection).** The proof in Appendix G is mathematically correct: under a uniform model prior (A1) and deterministic predictions (A2), MI(S(m); ŷ_i) = JSD of the per-model predictive distributions. This is a clean result.

However, there is a significant logical gap between Proposition 1 and the actual top-K selection procedure:

- Proposition 1 establishes the informativeness of a **single** sample in isolation. Maximising JSD sample-by-sample is greedy; no submodularity or independence argument is provided to show that the top-K highest-JSD samples are jointly optimal. High-JSD samples may be highly correlated (e.g., all from the same narrow difficulty regime where models disagree), leaving other informative aspects of the benchmark uncovered.
- The injectivity assumption on S(m) (e.g., accuracy) is non-trivial: two distinct models may share identical benchmark accuracy. The authors note this but do not discuss how it affects the tightness of the MI = JSD equality.
- Assumption A1 (uniform model prior) is argued to be without loss of generality via replication, but in practice DISCO uses a subset of 100 models out of 382 (buried in Appendix I). This reweighting is never discussed in relation to Proposition 1.

The theory therefore provides motivation rather than a formal guarantee. The paper could present it more modestly.

**Proposition 2 (JSD–PDS sandwich bounds).** The proof in Appendices H.1–H.3 is technically careful and the result is sound: JSD is bounded quadratically below and linearly above by (PDS − 1). This justifies PDS as a proxy for JSD. The construction of the proof via total variation intermediary is elegant.

**Model signature approach.** Replacing scalar accuracy with a high-dimensional vector of raw outputs as input to the performance predictor is intuitive and empirically effective. However:
- PCA to 256 dimensions is selected as a hyperparameter jointly with the RF hyperparameters (Appendix I). Table 2(d) shows dimensionality matters significantly (no-reduction: .918 rank, PCA-256: .987). The cross-validation procedure for selecting this hyperparameter is not clearly described—it presumably uses the source model split, but if not done carefully it risks leaking information about the difficulty of specific benchmarks.
- The dimensionality of the signature (K × C where C is the number of classes) grows with K and C. For ImageNet (1000 classes), even 100 samples yield a 100,000-dimensional signature before PCA. The storage and compute implications of this are acknowledged in Appendix B but not discussed as a scalability concern.

**Source model subsetting for PDS computation.** Appendix I reveals that for MMLU, using all 382 source models to compute PDS actually *hurts* performance; only 100 are used (selected randomly, with M tuned as a hyperparameter). This is a non-trivial implementation detail that has direct bearing on the reproducibility of the main results and should be foregrounded, not buried. It also slightly undermines the "simpler than prior methods" narrative.

---

### Experiments & Results

**Comparison with Metabench (Table 1).** The comparison is acknowledged as "not directly comparable" (different K: 100 for DISCO vs. 150/450/200 for Metabench). Presenting these in the same table—even with a footnote—invites unfair visual comparison, especially since MAE and Rank are the primary metrics without normalising for sample count. The correct comparison would either fix K or explicitly adjust for the budget difference.

**Separation of contributions.** Table 1 shows that "Random + Sig.+RF" already achieves 1.81%p MAE and .933 rank on MMLU—already at or above the prior SOTA from Anchor-corr + gp-IRT (2.08%p, .927). Adding PDS selection then improves to 1.07%p and .987. This is important: the model signature approach alone drives a large fraction of the gain, somewhat independent of the PDS selection. The narrative emphasises PDS as the key contribution, but the decomposition suggests the performance predictor redesign is equally or more important. The paper should be more explicit about this.

**Statistical significance.** Confidence intervals are provided in Table 7 (Appendix D), which shows DISCO's variance is low. This is reassuring, but the main text (Table 1) contains no uncertainty estimates and presents point estimates as definitive. For a venue like ICLR, this is a notable omission from the main table.

**Chronological split.** The chronological split is well-motivated and more realistic than IID. However, Table 10 reveals that DISCO degrades significantly in performance-gap scenarios (89.2 rank vs. 87.4 for direct eval—effectively no advantage). The authors dismiss this scenario as "not realistic," but frontier model evaluation (where target models substantially outperform the source pool) is precisely a common practical use case. This failure mode deserves a more careful discussion and quantification in the main paper.

**Vision experiments.** DISCO is tested on ImageNet with 400 timm models—a sensible domain transfer test. However, the baselines from Lifelong Bench. and SSEPY were computed by the authors themselves (Appendix J.2), since those papers do not contain ImageNet results. Reproducing these methods introduces a risk of implementation discrepancies; more validation of the baseline implementations would strengthen this section.

**Missing baseline.** Li et al. (2025), "Active Evaluation Acquisition," is cited in related work but not compared against. This appears to be a close competitor (anchor-point based, but adaptive) and its omission from Table 1 is unexplained. Zhang et al. (2025), who introduce a benchmark for comparing efficient evaluation methods and apparently show that methods "miss the mark" for out-of-distribution models, is cited but not used as an evaluation testbed. Given that Zhang et al.'s work directly challenges the validity of this line of research, engaging with it more substantively is important.

**Offline cost amortisation.** The break-even analysis (Appendix B.3) concludes DISCO is worthwhile after ~389 evaluations. This analysis correctly accounts for shared offline costs, but the 3,284 GPU-hours offline figure represents evaluating 385 large models on the full 14k-sample MMLU. Not every practitioner has access to this pool of pre-evaluated models. The paper mentions that outputs can be "downloaded from open-llm-leaderboard" but does not acknowledge that this constrains applicability to benchmarks covered by that leaderboard.

---

### Limitations & Broader Impact

The limitations section is honest about distribution shift and open-ended generation incompatibility. However, several important limitations are understated:

1. **Dependence on large model pools.** DISCO requires hundreds of source models pre-evaluated on the *full* benchmark. This is a substantial barrier for benchmarks not on leaderboards, for proprietary models, or for newly created benchmarks.

2. **Calibration sensitivity.** Appendix E shows a Pearson correlation of 0.49 between MAE and ECE—poorly calibrated models are predicted worse. The primary driver is overall confidence level (correlation −0.47 with mean confidence). This means that for models with systematic over/underconfidence (e.g., post-RLHF models), DISCO's signature is less informative. This is a practically important limitation for evaluating RLHF-tuned chat models.

3. **Multiple-choice constraint.** The limitation is correctly identified but understated in severity. Most modern LLM evaluation is increasingly moving toward open-ended or generative tasks. The method is explicitly inapplicable to these settings.

4. **Greedy vs. joint optimality.** As discussed in the method section, the theoretical guarantee only covers one-sample selection, not the top-K greedy strategy.

---

### Writing & Clarity

The paper is generally well-written and flows logically. The main clarity concern is that several critical implementation details (M=100 source models for MMLU PDS, PCA dimension tuning as a hyperparameter, kNN k=1) are scattered across Appendix I rather than consolidated in a reproducibility-focused section. Given that these choices affect the main results, readers attempting replication would benefit from a cleaner implementation section.

The factor analysis (Section 5.4) is presented as running text with references to Table 2, but Table 2 itself appears to be in the appendix region—making it hard to follow in the main paper.

---

### Overall Assessment

DISCO makes a genuine and practical contribution to efficient model evaluation: using model disagreement (JSD/PDS) for sample selection combined with full-signature metamodeling achieves measurable improvements over TinyBenchmarks and Anchor-corr across four standard LLM benchmarks, and generalises to ImageNet. The method is simpler than IRT-based alternatives and the theoretical motivation (Proposition 1) is well-executed, even if it falls short of justifying top-K greedy selection formally.

The main concerns that need to be addressed before this reaches its full potential as an ICLR contribution are: (1) the unfair comparison with Metabench due to different sample budgets; (2) the absence of comparison with Li et al. (2025) and engagement with Zhang et al. (2025)'s critique; (3) the critical implementation detail (subsampling source models for PDS) buried in appendices; (4) the gap between the single-sample theoretical optimality result and the top-K greedy procedure used in practice; (5) the failure mode under a performance gap, which is more realistic for frontier model evaluation than the authors acknowledge. Despite these issues, the empirical results are convincing, the method is simple and reproducible, and the work addresses a problem of growing practical importance. With revisions addressing the comparison fairness and theoretical scope, this paper is above the ICLR acceptance threshold.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces DISCO, an efficient evaluation framework that selects a small, highly informative subset of benchmark examples by maximizing inter-model response disagreement (via PDS or JSD) and predicts full-dataset accuracy using lightweight predictors trained on concatenated "model signatures." Across multiple language and vision benchmarks, DISCO consistently outperforms clustering-based anchor selection and psychometric baselines, achieving >99% reduction in evaluation cost with ~1% absolute error while preserving near-perfect model rank correlation.

### Strengths
1. **Rigorous & Realistic Experimental Design:** The use of a chronological train/test split for models (training on pre-2024 models, testing on newer ones) realistically simulates practical evaluation pipelines and mitigates optimistic leakage. DISCO maintains robust performance under this split (Table 1, Sec 5.4).
2. **Clear Practical Value & Compute Transparency:** The authors provide a thorough offline/online cost breakdown (Appendix B), including GPU-hours, storage requirements, and a precise break-even analysis (389 models). Demonstrating that the meta-predictor trains in <1 minute and yields massive inference savings makes the method highly actionable.
3. **Strong Empirical Performance Across Domains:** DISCO achieves state-of-the-art compression on MMLU, HellaSwag, Winogrande, ARC, and ImageNet. The systematic ablations (Table 2) effectively isolate the contribution of each component (source model count, PCA dimensionality, predictor choice), showing stability across extreme compression rates (Fig 5).
4. **Conceptual Simplicity with Theoretical Motivation:** Replacing complex IRT latent-variable modeling and global clustering with greedy disagreement sampling and direct signature regression simplifies the pipeline. Proposition 1 provides a clean information-theoretic justification (linking mutual information between model identity and accuracy to JSD) for why disagreement is an optimal proxy for sample informativeness.

### Weaknesses
1. **High Upfront Offline Compute Barrier:** The offline stage requires evaluating ~385 source models on the full dataset, costing ~3,284 GPU-hours on an H100 (Appendix B.2/B.3). While amortized, this substantial initial investment limits accessibility for academic labs and forces reliance on pre-computed leaderboard logits, which may become stale or unavailable.
2. **Restricted to Fixed-Choice/Multiple-Choice Tasks:** The method explicitly requires predictive probability distributions over predefined answer choices to compute PDS/JSD (Sec 6, Limitations). This excludes modern open-ended generative evaluations (e.g., free-form reasoning, stepwise judged benchmarks, or instruction-following tasks), significantly narrowing its applicability in current LLM assessment.
3. **Sensitivity to Model Calibration & Distribution Shifts:** Appendix E demonstrates a Pearson correlation of 0.49 between target model ECE and prediction MAE, indicating that poor calibration degrades estimator accuracy. Additionally, when target models significantly outperform the source pool, rank correlation drops sharply (Sec F), revealing a reliance on overlapping performance distributions that may not hold during rapid model advancement.
4. **Theoretical Contribution is Incremental:** Propositions 1 & 2 correctly establish the MI-JSD relationship and bound JSD with PDS using standard inequalities. While these results justify using disagreement as a selection signal, they do not fundamentally constrain the algorithm or provide novel learning guarantees, serving primarily as mathematical intuition rather than a core algorithmic driver.

### Novelty & Significance
- **Novelty:** Moderate. The synthesis of disagreement-based greedy selection with high-dimensional signature regression is a novel and effective engineering contribution, but it extends rather than reinvents the anchor-point/meta-evaluation paradigm. The core shift from sample representativeness to response diversity is conceptually straightforward.
- **Significance:** High. Reducing benchmark evaluation costs by >99% while maintaining sub-1.1% MAE addresses a pressing bottleneck in LLM/VLM development. The method enables frequent performance monitoring, cheaper hyperparameter tuning, and more inclusive evaluation cycles, aligning strongly with current needs in scalable ML.
- **Clarity:** High. The two-stage pipeline is clearly articulated, mathematical notation is consistent, and experimental reporting (including confidence intervals, detailed baselines, and factor analysis) meets top-tier conference standards. (Parser formatting artifacts do not impact the logical flow.)
- **Reproducibility:** High. The authors release code and project links, report exact hyperparameters (Appendix I), detail data/model sourcing (Sec 5.2/J.1), and provide exhaustive compute/storage metrics. The evaluation protocol is unambiguously defined, enabling independent replication.

### Suggestions for Improvement
1. **Adaptation to Open-Ended Evaluation:** Propose or experiment with a proxy for disagreement on free-form generation tasks (e.g., verifier-score variance, embedding-space disagreement, or log-likelihood sampling over constrained continuations) to broaden applicability beyond multiple-choice benchmarks.
2. **Theoretical Tightening or Complexity Bounds:** Augment the information-theoretic motivation with a sample-complexity or generalization bound for greedy JSD/PDS selection. Analyzing how subset size $K$ relates to prediction error variance would strengthen the theoretical claim that disagreement maximizes information content.
3. **Mitigation for Calibration/Shift Sensitivity:** Introduce a lightweight calibration-aware reweighting scheme or discuss adaptive strategies (e.g., periodic lightweight offline updates, domain adaptation of signatures) to maintain accuracy when target models diverge significantly in confidence scaling or capability distribution from the source pool.
4. **Expanded Baseline Context & Metric Alignment:** Explicitly compare against recent concurrent methods like SMART filtering (Gupta et al., 2025) and Bento (Zhao et al., 2024) in a dedicated table. Clarify whether all methods use identical predictor families when reporting MAE, ensuring that performance gains are attributable to sample selection rather than hidden regressor advantages.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Source Pool Size Ablation:** Evaluate performance with significantly fewer source models (e.g., 10, 50) to test practicality. Without this, the claim of "efficient evaluation" is misleading for users who cannot afford the 3284 GPU-hour offline cost.
2. **Cross-Architecture Generalization:** Test target models with fundamentally different architectures (e.g., Mamba, MoE) not represented in the source pool. If performance drops, the claim that model signatures generalize across architecture families is unsupported.
3. **Data Contamination Check:** Verify results on a strictly held-out benchmark (e.g., MMLU-Pro) where source models are less likely to be contaminated. Current results may reflect memorization patterns rather than genuine reasoning capability estimation.
4. **Fair Baseline Re-evaluation:** Re-run Metabench and TinyBenchmarks with the exact same K=100 constraint without noting them as "not directly comparable." SOTA claims are undermined if baseline comparisons differ in sample budget.
5. **Open-Ended Task Proxy:** Implement a judge-model layer to convert open-ended generations into probabilities for DISCO selection. Excluding this ignores the dominant trend in LLM evaluation, limiting the method's relevance.

### Deeper Analysis Needed (top 3-5 only)
1. **Offline Cost Utility Curve:** Plot performance gain vs. number of target models evaluated to show where DISCO actually becomes cost-effective. The current break-even point (389 models) suggests the method is inefficient for individual researchers, contradicting the core motivation.
2. **Source Model Homogeneity Sensitivity:** Analyze how performance degrades if the source pool lacks diversity (e.g., only LLaMA models). Proposition 1 relies on heterogeneity; without this analysis, the theoretical guarantee is unverified in realistic settings.
3. **Temporal Decay Analysis:** Show error growth as the target model's release date moves further from the source cutoff. This validates whether the selected anchors remain informative over time or require frequent refreshing.
4. **Calibration vs. Accuracy Disentanglement:** Quantify how much prediction error is driven by model miscalibration versus actual accuracy differences. If DISCO conflates the two, it fails as a pure performance estimator.
5. **Sample Difficulty Profile:** Compare the difficulty distribution of DISCO-selected samples against random and IRT-selected samples. This is necessary to prove the claim that "diversity matters more than representativeness."

### Visualizations & Case Studies
1. **High-Error Failure Cases:** Scatter plot target models where prediction error exceeds 5%p, annotated with architecture and size. This reveals specific regimes where the disagreement signal fails to predict performance.
2. **Signature Space Clustering:** t-SNE visualization of model signatures colored by true accuracy. This directly validates the core hypothesis that signature similarity implies performance similarity.
3. **Architecture Bias Heatmap:** Show prediction error broken down by model family (e.g., LLaMA vs. GPT vs. BERT). This exposes whether the method systematically biases against certain model types.
4. **Anchor Sample Inspection:** Display examples of top-ranked DISCO samples vs. random samples. Visual inspection is needed to confirm these samples are indeed "diverse" rather than just "hard."

### Obvious Next Steps
1. **Public Source Model Repository:** Release the pre-computed source model outputs as a community resource. Without this, the high offline cost prevents reproduction and adoption, weakening the paper's impact.
2. **Dynamic Anchor Updating Mechanism:** Propose and test a strategy to incrementally update anchors as new model families emerge. This addresses the acknowledged limitation regarding distribution shifts in the model population.
3. **Judge-Model Integration for Generative Tasks:** Include a pilot experiment applying DISCO to generative tasks via an LLM judge. This is required to claim the method is viable for modern evaluation beyond multiple-choice.
4. **Theoretical Relaxation of Uniform Prior:** Extend Proposition 1 to non-uniform model priors. Real-world model populations are not uniform, so the current theory does not fully support the practical application.

# Final Consolidated Review
## Summary
DISCO proposes an efficient evaluation framework that selects informative benchmark samples by maximizing inter-model disagreement (via PDS or JSD) and predicts full-dataset accuracy using "model signatures"—the concatenation of raw model outputs on selected samples. The method achieves state-of-the-art performance prediction on MMLU, HellaSwag, Winogrande, ARC, and ImageNet while reducing evaluation cost by >99%.

## Strengths
- **Strong empirical results with realistic evaluation protocol:** The chronological train/test split (training on pre-2024 models, testing on newer ones) provides a rigorous test of generalization. DISCO achieves 1.07%p MAE and 0.987 rank correlation on MMLU at 100 samples, substantially outperforming TinyBenchmarks (2.08%p MAE, 0.927 rank).
- **Conceptual simplicity with theoretical grounding:** Proposition 1 correctly establishes that mutual information between model identity and benchmark accuracy equals Jensen-Shannon divergence of predictive distributions. This provides clean information-theoretic justification for selecting samples that maximize model disagreement. Proposition 2 bounds JSD between quadratic and linear functions of PDS, validating PDS as a tractable proxy.
- **Clear practical value with transparent cost analysis:** Appendix B provides concrete GPU-hour breakdowns (offline: 3,284 GPU-hours; online: 0.07 GPU-hours) and break-even analysis (389 model evaluations). The method enables efficient evaluation for users with access to pre-computed model outputs.

## Weaknesses
- **Unfair comparison with Metabench due to unequal sample budgets:** Table 1 presents DISCO (K=100) and Metabench (K=150–450 depending on benchmark) together, but the methods use different numbers of samples. Metabench requires 50–350% more samples to converge. While disclosed in a footnote, this presentation invites unfair visual comparison. A controlled comparison at fixed K, or explicit cost-adjusted metrics, would be more appropriate.
- **Critical implementation detail buried in appendix:** The fact that DISCO uses only M=100 of the 382 available source models for PDS computation (Appendix I) materially affects reproducibility and the "simpler than prior methods" narrative. This choice is motivated but never discussed in the main text, where the method is presented as straightforward greedy selection on all source models.
- **Gap between theoretical guarantee and practical algorithm:** Proposition 1 establishes optimality for selecting a single sample, but the paper uses greedy top-K selection without proving submodularity or any joint optimality property. High-JSD samples may be correlated, concentrating coverage in narrow difficulty regimes. The theory motivates but does not formally justify the algorithm.
- **Failure mode under large performance gaps understated:** When target models substantially outperform source models (Table 10), DISCO's advantage over direct evaluation collapses (89.2 vs. 87.4 rank—essentially no benefit). The authors dismiss this as "not realistic," but frontier model evaluation where new models outperform the training pool is precisely a common use case. This limitation deserves more prominence.
- **Applicability restricted to multiple-choice formats:** The method requires predictive probabilities over predefined answer choices to compute PDS/JSD. Modern LLM evaluation increasingly uses open-ended generation (reasoning, instruction-following, summarization). The limitation is acknowledged but the severity is understated.

## Nice-to-Haves
- Comparison with Li et al. (2025) Active Evaluation Acquisition, cited in related work but not compared experimentally
- Extension to open-ended tasks via judge-model probability elicitation
- Cross-architecture generalization analysis (e.g., testing on Mamba or MoE architectures not in source pool)

## Removed Points
- "Li et al. (2025) omitted from comparison": The paper does cite and discuss this work in Related Work. While experimental comparison would strengthen the paper, omission is not a fundamental flaw given the methodological differences.
- "Zhang et al. (2025) critique not engaged": The paper cites Zhang et al. and discusses their observation about performance gaps in Section 5.4, using it to motivate the chronological split.
- "Statistical significance missing from main table": Confidence intervals appear in Table 7 (Appendix D). This is standard practice; the point is procedural rather than substantive.
- "Source pool size ablation missing": Table 2(c) explicitly shows ablation over number of source models (100, 200, 300, 382). This experiment is already present.
- "PDS repurposed from prior work by first author without disclosure": The paper states "disagreement is measured by predictive diversity scoring (PDS, Rubinstein et al. (2024)), originally proposed for out-of-distribution detection." The connection is disclosed.

## Novel Insights
The key insight that "model response diversity" rather than "sample space diversity" drives efficient evaluation is both principled and practical. The theoretical result that maximizing JSD is equivalent to maximizing mutual information between sample predictions and model identity (under injective statistics) provides a clean justification for greedy disagreement sampling. However, this remains a single-sample result—whether the top-K greedy selection preserves optimality or merely approximates it is an open theoretical question that the paper does not resolve.

## Suggestions
- Present a controlled comparison at fixed sample budgets for all baselines, or normalize MAE/Rank by computational cost.
- Move the M=100 source model subsetting detail from Appendix I to the main method section; justify whether this is a robust hyperparameter or a dataset-specific artifact.
- Add a subsection on the performance gap failure mode with quantitative analysis of when DISCO breaks down (e.g., what accuracy gap triggers >50% rank correlation loss).
- Consider adding a correlation analysis between sample JSD and calibration, which would clarify whether disagreement-rich samples help or hinder on poorly calibrated models.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
