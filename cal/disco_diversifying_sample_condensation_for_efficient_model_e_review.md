=== CALIBRATION EXAMPLE 68 ===

# Harsh Critic Review
Now I have enough material for a thorough review. Let me write it up.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "DISCO: Diversifying Sample Condensation for Efficient Model Evaluation" is appropriate and memorable. The abstract accurately describes the core idea and the two claimed contributions—disagreement-based selection and model-signature-based prediction. However, the abstract states DISCO achieves "state-of-the-art results" across all four benchmarks, which is slightly overstated: as Table 1 reveals, Metabench beats DISCO on HellaSwag (0.80 vs. 0.86 MAE) and ARC (1.14 vs. 1.47 MAE) — though Metabench is noted in a table footnote as using more samples. This nuance should be surfaced more honestly in the abstract.

---

### Introduction & Motivation

The motivation is well-stated: the escalating GPU cost of full benchmark evaluation is a genuine pain point, well-documented with concrete figures (HELM at 4000 GPU-hours, LMMs-Eval at 30–1400 GPU-hours). The two claimed contributions are clearly stated: (1) disagreement-based selection over clustering-based representativeness; (2) model-signature–based prediction as a simpler alternative to IRT-based methods.

**Concern 1 — Novelty framing.** The claim that "promoting diversity among samples is not essential; what matters is diversity in model responses" is slightly misleading. The prior clustering methods (Anchor-corr, Anchor-IRT) already cluster *by model response patterns*, not by raw feature diversity. The real distinction is between *greedy per-sample scoring* and *global clustering* for subset selection. This subtler framing would be more accurate and more intellectually honest.

**Concern 2 — Contribution separation.** The introduction bundles two contributions (new selection criterion, new prediction method) without giving the reader intuition about which matters more. From Table 1, swapping only the predictor to model signatures (Random + Sig + RF) already achieves 1.81%p MAE vs. 2.08%p for Anchor-corr + gp-IRT — a competitive jump even without PDS selection. This suggests the prediction innovation may be doing more of the work than the selection innovation, but this isn't explicitly acknowledged upfront.

---

### Problem Formulation (Section 3)

The formal setup is clean and well-defined. The split into *source models* (known performance) and *target models* (to be estimated) is standard in the literature and appropriately laid out. The definition of *model performance* SD[f] is sufficiently general.

One minor but important conceptual point: the injective assumption on S(m) in Proposition 1 (that accuracy distinguishes models) is almost certainly violated in practice when many models have nearly identical accuracy scores. While the authors acknowledge this is a simplification, the practical implications for highly correlated model pools (common on leaderboards) deserve at least a brief comment.

---

### Theoretical Analysis (Section 4.1.2, Appendices G & H)

**Proposition 1 (Appendix G as "Proposition 3"):** The core theoretical claim — that MI(S(m); ŷ_i) = JSD over the per-model predictions — is a clean and valid result. The proof is correct: it follows directly from the definition of MI under the uniform model prior (A1) and deterministic predictions (A2). The result elegantly connects sample informativeness for ranking models to the JSD of their predictive distributions.

**Concern 3 — Greedy ≠ Optimal subset.** Proposition 1 establishes that the *single* most informative sample is the one with maximum JSD. The paper then extends this to justify *greedy top-k selection*: take the K samples with the highest individual JSD scores. However, this greedy procedure does not account for *redundancy between selected samples*. Two samples can individually have high JSD (both elicit strong model disagreement) but be highly correlated (the same subset of models always disagree on both). An optimal subset selection would account for this joint redundancy, similar to submodular optimization in feature selection. The paper presents greedy top-k as if it follows from Proposition 1, but that claim requires additional justification. The authors should either acknowledge this gap explicitly or show empirically that their top-k selection produces a diverse set of disagreements.

**Concern 4 — Assumption A1 (Uniform prior).** The uniformity assumption on the model index is used to convert conditional entropy into a simple average. In practice, source model distributions are highly non-uniform: there are many more fine-tuned variants of LLaMA-7B than there are 70B frontier models. The authors give a replication argument ("if the prior is non-uniform, replicate models proportionally"), but this presupposes knowledge of the true prior over models—which is unavailable and arguably not meaningful in this context. The theoretical guarantees are thus only applicable in a setting that may not match the experimental one.

**Proposition 2 (JSD–PDS bounds, Appendix H):** The sandwich inequality between JSD and PDS-1 is mathematically sound. However, the bounds are quite loose (quadratic below, linear above), which means high PDS does not tightly imply high JSD and vice versa. The authors use PDS in practice (not JSD) and claim it works empirically. In Table 1, JSD sometimes beats PDS (HellaSwag MAE: 0.86 for JSD vs. 1.01 for PDS) and sometimes loses (MMLU MAE: 1.30 vs. 1.07). The theory does not explain which criterion should be preferred in which setting; the choice seems to be post-hoc empirical tuning.

---

### Method (Section 4.2 — Model Signatures & Prediction)

The model-signature prediction is simple and effective. The idea of using a high-dimensional concatenation of raw softmax outputs as input to a Random Forest is somewhat unusual but pragmatically well-motivated.

**Concern 5 — Dimensionality and PCA sensitivity.** The signature dimensionality before PCA is up to 3100 (100 samples × ~31 classes for MMLU). The authors report that PCA at 256 dimensions improves rank correlation from 0.918 to 0.987 (Table 2d), but only a small number of PCA dimensions are tested. The choice of 256 is not cross-validated in a principled way, and the sensitivity to this hyperparameter is not fully explored. For different benchmarks (e.g., WinoGrande with 2 choices vs. MMLU with 4) the effective dimensionality is very different; how is this hyperparameter set across datasets?

**Concern 6 — Requirement for predictive probabilities.** The method requires *calibrated probability outputs* over answer choices. This is only possible for models providing logit-level access. The paper acknowledges in the Limitations section that DISCO does not apply to open-ended generation, but the deeper concern is that many production APIs (GPT-4, Claude, Gemini) do not expose calibrated per-class probabilities for arbitrary MCQs. This limits applicability to the open-weight ecosystem, which narrows the real-world use case considerably.

---

### Experiments (Section 5)

**Section 5.3 — Main Results:**

The main results in Table 1 are strong, particularly for MMLU and HellaSwag. However, there are several important concerns:

**Concern 7 — Apples-to-oranges comparison with Metabench.** The comparison with Metabench is explicitly flagged in a table footnote as "not directly comparable" because Metabench requires more samples (150 for MMLU, 450 for HellaSwag, 200 for WinoGrande vs. 100 for DISCO). Crucially, on Metabench's two strongest benchmarks, it *outperforms* DISCO's best variant: HellaSwag MAE 0.80 (Metabench) vs. 0.86 (DISCO best), ARC MAE 1.14 (Metabench) vs. 1.47 (DISCO best). The main text states "DISCO achieves a stronger efficiency-precision trade-off" and the conclusion claims "state-of-the-art," but DISCO is strictly dominated by Metabench on two of the four benchmarks—at the cost of using 50-350% more data points. The paper should engage with this more honestly: is DISCO better at extreme compression (100 samples) while Metabench wins at higher budgets?

**Concern 8 — Missing ablation: DISCO selection + prior predictor.** To disentangle the contributions of the two DISCO components (disagreement-based selection vs. model-signature prediction), the key missing ablation is: **High PDS + gp-IRT** (DISCO selection, prior predictor). From Table 1, we can see that switching only the predictor (Random selection + Sig + RF gives 1.81 MAE) already substantially improves over Anchor-corr + gp-IRT (2.08 MAE). This suggests the model-signature predictor contributes significantly, possibly more than the PDS selection step. Without the PDS + gp-IRT entry, the relative contribution of each component cannot be cleanly assessed.

**Concern 9 — Ablation on only MMLU (Section 5.4).** The factor analysis in Table 2 is only conducted on MMLU. Findings about PCA dimension choice, stratification, and prediction model may not generalize. At minimum, the ablations should be spot-checked on HellaSwag and WinoGrande, which have different sizes and task structures.

**Section 5.4 — Model Split:**

The chronological split is a welcome and realistic evaluation choice. The result that DISCO is robust to splitting strategy (0.987 for chronological vs. 0.986 for IID, Table 2a) is reassuring. The performance-gap experiment in Appendix F is important: when there is an 8.18%p accuracy gap between source and target models, DISCO's advantage over direct evaluation collapses to only 1.8 percentage points in rank correlation (89.2 vs. 87.4). The authors argue this scenario "does not happen in practice," but they do not support this empirically and it represents a real failure mode for practitioners who deploy DISCO and then a qualitatively better model appears (e.g., a new GPT or Gemini-level model). This dismissal is too quick.

**Section 5.5 — Vision Domain:**

The extension to ImageNet with 400 pre-trained vision models is valuable. Results show DISCO (0.63 MAE, 0.969 rank) substantially outperforms Lifelong Bench (2.06 MAE, 0.838 rank) and SSEPY (3.05 MAE, 0.762 rank). However, the vision baselines are weaker than the language baselines — there is no vision equivalent of the Anchor-IRT or Anchor-corr methods. It is unclear whether this is because they genuinely cannot be applied to ImageNet classification, or because the authors did not implement them. If the latter, this should be stated clearly.

---

### Writing & Clarity

The paper is well-written and logically organized. The problem formulation section is precise. One structural issue: Proposition 1 in the main body of Section 4.1.2 corresponds to "Proposition 3" in Appendix G, suggesting a renumbering discrepancy between propositions in the main text and appendix. This is disorienting and should be reconciled.

The break-even analysis (Appendix B.3) is a useful addition for practitioners: DISCO's offline cost of 3284 GPU-hours amortizes to savings after evaluating ~389 models. This is a concrete and helpful framing. However, the comparison of online costs (219 GPU-seconds for tinyBenchmarks vs. 218 GPU-seconds for DISCO, Table 6) shows effectively zero online cost difference between methods, which somewhat undermines the framing in the abstract that DISCO is "conceptually simpler" — the main savings are in evaluation quality, not unique compute efficiency.

---

### Limitations & Broader Impact

The authors identify two main limitations: (1) distribution shift in model population, and (2) inapplicability to open-ended generation tasks. These are genuine limitations, and the authors handle them fairly.

**Concern 10 — Calibration dependence is underappreciated.** Appendix E reveals a Pearson correlation of 0.49 between model calibration error (ECE) and DISCO's MAE, and a correlation of -0.47 between mean model confidence and MAE. This means DISCO's predictions are systematically better for well-calibrated, confident models and worse for miscalibrated ones. Since recent instruction-tuned and RLHF-trained models often exhibit unusual calibration (e.g., overconfident refusals), this is an important failure mode for precisely the class of models practitioners care most about. This should be elevated from an appendix observation to a main-text limitation.

**Concern 11 — Source model homogeneity.** The entire approach relies on the representativeness of the source model pool to learn the predictive mapping. The current pool (424 models from HuggingFace Open LLM Leaderboard) is dominated by LLaMA fine-tuned variants. If the method is applied to a new class of models (e.g., mixture-of-experts, multimodal, chain-of-thought reasoning specialists), the source pool may not represent the response distribution of target models well. This is related to the distribution shift limitation, but deserves specific mention.

---

### Overall Assessment

DISCO is a well-executed paper with a clear central idea: use inter-model disagreement (JSD/PDS) as a sample selection criterion and learn a simple mapping from model signatures to benchmark performance. The information-theoretic motivation is elegant, the empirical results are generally strong, and the chronological split addresses a genuine methodological weakness in prior work. The method is simpler than IRT-based alternatives and achieves competitive or superior performance at 100 samples.

However, several concerns limit my enthusiasm for acceptance at the current state. The theoretical claim that greedy top-k selection by JSD is "information-theoretically optimal" is overstated — Proposition 1 justifies selecting the *single* most informative sample but does not address joint redundancy in a subset. The comparison with Metabench is unfair (DISCO loses on HellaSwag and ARC when samples are matched), yet the paper claims state-of-the-art across all benchmarks. The key ablation — DISCO selection + prior predictor — is missing, making it impossible to cleanly attribute credit between the two contributions. The calibration sensitivity (Appendix E) is an important practical failure mode that is underappreciated. These are non-trivial issues, but most are addressable with clarified theoretical claims, an additional ablation row, and a more careful characterization of when DISCO wins and loses. I would recommend **weak rejection** with encouragement to revise, as the core contribution is sound and the empirical foundation is solid.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes **DISCO**, a method for efficient model evaluation that compresses benchmarking costs by selecting a small subset of informative samples based on model disagreement (diversity in model responses) rather than sample representativeness. It further simplifies performance prediction by using model signatures (outputs on the subset) with simple regressors instead of complex psychometric models. Extensive experiments across language (MMLU, etc.) and vision (ImageNet) domains demonstrate state-of-the-art accuracy prediction with ~99% cost reduction compared to full evaluation.

### Strengths
1.  **Strong Empirical Performance:** The method achieves state-of-the-art performance in efficiency-precision trade-offs. On MMLU with only 100 samples, DISCO reaches a Spearman rank correlation of 0.987 and MAE of 1.07%, significantly outperforming baselines like tinyBenchmarks and Metabench (Table 1, Section 5.3).
2.  **Clear Theoretical Justification:** The authors provide an information-theoretic argument (Proposition 1) linking sample informativeness for performance estimation to Jensen-Shannon Divergence (disagreement), supported by a proof in the appendix (Appendix G). This grounds the heuristic of selecting "disagreement" mathematically.
3.  **Robust Evaluation Protocol:** The use of a **chronological split** (training on models before Jan 2024, testing on newer ones) addresses the common criticism of meta-learning approaches relying on static model pools, ensuring results are more relevant to real-world deployment scenarios (Section 5.2, Section 5.4).
4.  **Domain Agnostic:** The approach is validated on both NLP benchmarks (MMLU, HellaSwag, etc.) and Computer Vision (ImageNet), demonstrating generalizability beyond language models (Section 5.5, Table 3).
5.  **Simplicity in Prediction:** Moving away from latent parameter estimation (IRT-based methods) to direct mapping of model signatures simplifies the pipeline and reduces potential overfitting risks associated with complex parameter estimation (Section 4.2.2).

### Weaknesses
1.  **High Offline Initialization Cost:** While online inference cost is negligible, the **offline stage** requires evaluating ~400 source models on the *full* dataset, costing approximately 3,284 GPU-hours (Table 4, Appendix B.2). This is a significant barrier for smaller institutions, contradicting the "efficiency" message for the initial setup phase.
2.  **Dependency on Source Models:** The method strictly requires a pre-existing pool of source models with known full-evaluation performance to function (Section 5.2). This creates a dependency on public leaderboards or large-scale consortium evaluations, which may not exist for niche or rapidly evolving architectures.
3.  **Applicability Limitations:** The method is restricted to tasks with predefined answer classes (multiple-choice), explicitly excluding open-ended generation tasks like summarization or translation (Section 6, Limitations). While acknowledged, this limits the scope of "model evaluation" the paper claims to address.
4.  **Assumption Sensitivity:** The theoretical Proposition 1 relies on assumptions like injectivity of the performance function and uniform prior over models (Appendix G). If the model space in practice deviates significantly from these assumptions (e.g., a specific new architecture behaves erratically), the information-theoretic guarantee weakens (Section 5.4 Factor Analysis acknowledges distribution shifts).
5.  **Baseline Comparisons:** Some prior work (e.g., Metabench) uses more samples per benchmark to converge (Note below Table 1), making direct comparisons at fixed sample sizes (100) potentially less favorable to baselines that might benefit from slightly larger subsets.

### Novelty & Significance
**Novelty:** The paper introduces a distinct shift in efficient evaluation from **representativeness** (clustering samples) to **diversification** (maximizing model disagreement). While active learning uses uncertainty/diversity, applying this specifically to *test-time evaluation condensation* via model signatures rather than IRT parameter estimation is a novel contribution.
**Significance:** Given the escalating cost of LLM evaluation (thousands of GPU hours per benchmark), DISCO offers a practical pathway for frequent monitoring of model checkpoints and accessibility for smaller research groups. The demonstration of high rank correlation with minimal data could change how companies evaluate internal model progress.

### Suggestions for Improvement
1.  **Quantify Offline Cost in Main Text:** The high offline cost (3,284 GPU-hours) should be more prominent in the introduction and conclusion. A more detailed discussion on strategies to lower this initial barrier (e.g., using fewer source models, leveraging public API outputs from existing leaderboards as mentioned in Section B.3) would strengthen the practical impact argument.
2.  **Address the "Cold Start" Problem:** Explicitly discuss or experiment with how DISCO performs when the pool of source models is small (e.g., < 50 models) or lacks architectural diversity. Section 5.4 analyzes source model count but only on a large set; extrapolating to smaller sets would clarify the method's scalability.
3.  **Clarify Dimensionality Reduction:** The use of PCA for model signatures (Section 4.2.2) is practical but heuristic. Provide a discussion or ablation on how much information is lost compared to using raw high-dimensional signatures, or discuss alternatives like hashing if PCA is the bottleneck for storage/computation.
4.  **Expand Limitations Section:** While the limitation regarding open-ended tasks is noted, elaborate on whether any preprocessing strategies (e.g., grading with an LLM) could be integrated to extend DISCO to generative tasks without violating the "few-shot/efficient" promise.
5.  **Refine Break-Even Analysis:** Clarify the context of the 389 evaluations required to break even (Appendix B.3). Ensure readers understand that this assumes a continuous series of model evaluations during training; single-model evaluations do not benefit from the method.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Total Compute Budget Comparison:** Compare DISCO (Offline + Online costs) against Random Sampling for a fixed total GPU-hour budget. Without this, the "efficiency" claim is misleading for users who cannot amortize the high offline cost (~389 model evaluations) over many target models.
2. **Compute-Matched Online Inference:** Compare DISCO (100 samples) against Random Sampling (200 samples). Both incur similar online inference cost; if Random wins, the complex selection method adds no practical value over simply evaluating more data.
3. **Architectural Distribution Shift:** Evaluate target models with architectures absent from the source pool (e.g., SSMs or hybrids vs. pure Transformers). Chronological splits do not guarantee architectural diversity, risking failure on novel model families.
4. **Source Pool Diversity Ablation:** Systematically vary the heterogeneity of the source model pool (e.g., single family vs. mixed architectures). The method relies on disagreement; homogeneous sources should theoretically break the signal, but this is not quantified.
5. **Generative Benchmark Evaluation:** Test on open-ended tasks (e.g., summarization) using semantic similarity for disagreement. The current multiple-choice restriction limits the method's applicability to modern LLM evaluation, and this limitation needs empirical validation.

### Deeper Analysis Needed (top 3-5 only)
1. **PCA Information Retention:** Quantify variance retained after dimensionality reduction of model signatures. If significant signal is lost, the "model signature" advantage over scalar accuracy is questionable.
2. **Selection Stability:** Measure Jaccard similarity of selected subsets across different source model seeds. Unstable selection implies the condensed benchmark is not reproducible or static.
3. **Calibration-Induced Bias:** Analyze if DISCO systematically underestimates miscalibrated models (Appendix E shows correlation). This would bias the evaluation metric against certain model types, undermining fairness.
4. **Task-Specific Performance Breakdown:** Report MAE per MMLU sub-category (e.g., STEM vs. Humanities). Global averages may hide failure modes on specific capabilities, masking where the method fails.
5. **Negative Transfer Analysis:** Investigate why adding more source models sometimes degrades performance (Appendix I). This contradicts the intuition that more data improves the predictor and suggests a flaw in the source pooling strategy.

### Visualizations & Case Studies
1. **Signature Space Clustering:** t-SNE of model signatures colored by ground-truth accuracy. If high-performing models do not cluster, the kNN/RF prediction premise is visually unconvincing.
2. **Disagreement Heatmap:** Visualize prediction entropy on DISCO samples vs. Random samples. This directly verifies the core claim that DISCO selects higher-diversity points.
3. **Rank Swap Case Studies:** Highlight specific model pairs where DISCO flips the ground-truth ranking. Explain *why* the signature misled the predictor to expose failure modes.
4. **Source Model Influence:** Heatmap showing which source models contribute most to the selection of specific anchor points. This reveals potential bias in the selection process driven by outlier source models.
5. **Error Distribution Histogram:** Plot the distribution of prediction errors (not just MAE). Heavy tails would indicate unreliable outliers despite low mean error, critical for trustworthiness.

### Obvious Next Steps
1. **Incremental Source Pool Updates:** Develop a method to update source statistics without re-evaluating the full dataset, reducing the prohibitive offline cost for dynamic benchmarking.
2. **Public Source Model Repository:** Release pre-computed source model outputs to eliminate the offline barrier for downstream users and ensure reproducibility.
3. **Generative Disagreement Metric:** Define a robust disagreement metric for open-ended generation (e.g., semantic embedding distance) to broaden applicability beyond multiple-choice.
4. **Principled Source Selection:** Analyze why subsampling source models improves performance (Appendix I) and propose a strategy to prune redundant sources automatically.
5. **Theoretical Bound Tightening:** Refine Proposition 2 to provide tighter bounds on JSD vs. PDS for practical sample size estimation, as current bounds are too loose for guidance.

# Final Consolidated Review
## Summary

DISCO proposes a method for efficient model evaluation by selecting a small subset of evaluation samples that maximize model disagreement (measured via Jensen-Shannon Divergence or Predictive Diversity Score), then predicting benchmark performance from "model signatures"—the concatenation of model outputs on this subset. The approach achieves ~99% reduction in evaluation cost while maintaining high accuracy (e.g., 1.07%p MAE on MMLU with 100 samples vs. full evaluation), outperforming prior methods like tinyBenchmarks across multiple NLP and vision benchmarks.

## Strengths

- **Strong empirical results with clear methodology.** On MMLU with 100 samples, DISCO achieves 1.07%p MAE and 0.987 Spearman rank correlation (Table 1), substantially outperforming tinyBenchmarks' Anchor-corr + gp-IRT (2.08%p MAE, 0.927 rank). The method shows consistent improvements across MMLU, HellaSwag, Winogrande, and ARC, with gains particularly pronounced at extreme compression rates (Figure 5 shows DISCO maintaining advantage even at 10 samples).

- **Principled theoretical motivation.** Proposition 1 establishes that mutual information between model identity and sample predictions equals Jensen-Shannon Divergence, providing an information-theoretic justification for selecting samples that elicit diverse model responses. The proof (Appendix G) is clean and correct under stated assumptions.

- **Realistic evaluation protocol.** The chronological split—training on models before January 2024, testing on newer models—addresses a genuine weakness in prior meta-learning approaches that use IID splits. The robustness analysis (Table 2a) shows nearly identical performance (0.987 vs 0.986 rank correlation) between chronological and uniform splits.

- **Simplicity of prediction pipeline.** Replacing IRT-based latent parameter estimation with direct regression on model signatures is conceptually simpler and avoids potential overfitting in parameter estimation. Table 1 shows Random Forest on signatures with random sampling already achieves competitive results (1.81%p MAE on MMLU).

- **Cross-domain validation.** Results extend to ImageNet (Table 3) with 400 vision models, achieving 0.63%p MAE and 0.969 rank correlation, demonstrating generalizability beyond language models.

## Weaknesses

- **Missing critical ablation separating selection from prediction contributions.** The paper bundles two innovations: PDS/JSD-based selection and model-signature prediction. From Table 1, Random + Sig + RF achieves 1.81%p MAE vs. Anchor-corr + gp-IRT's 2.08%p—a substantial jump from prediction alone. Without the ablation of PDS selection + gp-IRT prediction, readers cannot assess how much of DISCO's improvement comes from selection versus prediction. This is a significant gap in the experimental design.

- **Theoretical gap between single-sample optimality and greedy top-k selection.** Proposition 1 correctly establishes that the single most informative sample maximizes JSD, but the extension to greedy top-k selection ignores sample redundancy. Two samples with individually high JSD could be highly correlated (the same models disagree on both), yet DISCO would select both. Submodular optimization or explicit diversity constraints would address this theoretically; the paper should acknowledge this gap or show empirically that selected samples are diverse.

- **Calibration sensitivity is underappreciated as a practical limitation.** Appendix E shows Pearson correlation of 0.49 between model calibration error (ECE) and DISCO's prediction error—systematically worse predictions for miscalibrated models. Since instruction-tuned and RLHF models often exhibit unusual calibration (e.g., overconfident refusals), this failure mode affects precisely the models practitioners evaluate most. This belongs in the main limitations section, not an appendix.

- **Comparison with Metabench requires nuance.** The paper claims "state-of-the-art results" but Table 1 shows Metabench achieves lower MAE on HellaSwag (0.80 vs 0.86) and ARC (1.14 vs 1.47)—albeit using 50-350% more samples (footnote: 150-450 vs 100). The efficiency-precision trade-off framing is correct, but "state-of-the-art" without qualification overstates the case. The conclusion should specify that DISCO achieves SOTA *at 100 samples*.

- **Factor analysis limited to MMLU.** Design choices (PCA dimension, stratification, prediction model) are ablated only on MMLU (Table 2). HellaSwag has 10 classes vs. MMLU's 4, and Winogrande has 2 classes—different output dimensionalities that could affect optimal PCA settings. Cross-validating key findings on at least one other benchmark would strengthen generalization claims.

- **Substantial offline cost creates practical barrier.** The 3,284 GPU-hour offline stage (Table 4) is non-trivial for smaller institutions. While the break-even analysis (389 evaluations) is useful, the paper could more prominently discuss strategies to reduce this initial cost (e.g., using fewer source models—Table 2c shows reasonable performance with 100 models).

## Nice-to-Haves

- Analysis of prediction error distribution (not just MAE) to assess whether heavy tails exist that would undermine trust despite low mean error
- Architectural diversity experiment testing whether source pool dominated by one architecture family (e.g., LLaMA variants) generalizes poorly to novel architectures (SSMs, mixture-of-experts)
- Task-specific performance breakdown on MMLU sub-categories to identify capability-specific failure modes

## Removed Points

- **Claims that "diversity among samples is not essential" is misleading.** The paper correctly distinguishes between sample diversity and diversity in model responses—the former being prior work's clustering approach. The critic's concern that prior methods already cluster by model responses is accurate, but the paper's real contribution is greedy per-sample scoring versus global clustering, which is a genuine distinction.

- **Proposition numbering discrepancy (main text vs. appendix).** The harsh critic flagged this as disorienting. While true, this is a minor formatting issue that doesn't affect content quality.

- **Uniform prior assumption (A1) criticism.** The authors provide a valid replication argument: non-uniform priors can be simulated by proportional replication of models. This is a reasonable theoretical workaround that doesn't invalidate the framework.

- **Applicability to open-ended generation tasks.** The paper explicitly acknowledges this limitation in Section 6. Requesting experiments on generative benchmarks is scope creep beyond the paper's stated contributions.

- **Missing related work citations.** The AI reviewer cannot verify which references exist; citations should be assumed valid unless proven otherwise.

## Novel Insights

The calibration analysis (Appendix E) reveals an underappreciated tension: DISCO performs best on well-calibrated models, yet the models practitioners most want to evaluate efficiently (frontier instruction-tuned models) often exhibit calibration anomalies. This suggests a future direction: incorporating calibration-aware features into model signatures, or developing calibration-invariant selection criteria. Additionally, the finding that subsampling source models can improve performance (Appendix I notes using 100 models outperforms all 382 for MMLU disagreement scores) contradicts the intuition that more source data always helps—the heterogeneity of the source pool, not its size, may be the key factor.

## Suggestions

- Add the critical ablation: High PDS selection + gp-IRT prediction to cleanly separate selection versus prediction contributions
- Move the calibration sensitivity discussion from Appendix E to Section 6 (Limitations), explicitly warning that miscalibrated models may yield less reliable predictions
- Qualify the "state-of-the-art" claim to specify "at 100 samples" or similar fixed-budget framing, acknowledging Metabench's superior performance when sample budgets are flexible
- Add at least one factor analysis result on HellaSwag or Winogrande to validate that design choices generalize beyond MMLU

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
