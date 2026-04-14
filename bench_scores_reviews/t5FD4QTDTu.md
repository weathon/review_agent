## Summary

NoTS (Narratives of Time Series) proposes a novel autoregressive pretraining framework that treats time series as sequences of progressively degraded temporal functions rather than sequences of time-period chunks. Degradation operators of varying intensity (local averaging and global low-pass filtering) generate a coarse-to-fine sequence from each sample; a transformer is trained autoregressively to predict each successive, less-degraded function from all prior ones. A theoretical motivation grounded in universal approximation theory argues that function-space sequences bypass a continuity failure of sampled-domain representations for operators like differentiation. The method is evaluated through synthetic feature regression and across 22 real-world datasets covering classification, anomaly detection, and imputation, with a lightweight pre-trained model NoTS-lw that adapts with <1% of its parameters trained.

---

## Strengths

- **Conceptually differentiated pretraining objective.** Replacing next-period prediction with next-function prediction via a degradation curriculum is a genuinely new angle on time-series autoregressive pretraining. The coarse-to-fine degradation axis for the AR sequence—rather than the temporal axis—is distinct from both masked autoencoders and period-chunking AR models, and from multi-scale methods that still operate temporally.

- **Motivated synthetic experiments with non-trivial regression targets.** Using fBm and autocorrelated sinusoids as controlled testbeds, and targeting features like SSC, WAMP, Hurst index, and band power (features that are discontinuous or globally structured), directly connects the experiments to the theory's claim about approximating discontinuous operators. This is considerably more principled than reconstruction-based benchmarking alone, and Table 1 results are internally clean with three-run statistics reported.

- **Parameter-efficient cross-domain transfer.** The frozen-weight setting in Table 2 (first four rows) shows that NoTS-lw achieves 82% of full-fine-tuning performance while training only <1% parameters, via prompt tuning and channel/task adaptors. This is a concrete and measurable outcome, not just an aspiration.

- **Systematic ablation decomposing each component.** Table 3 isolates the contribution of the latent consistency term, autoregressive masking, connected degradation sequence, and kernel type. Each ablation variant has a clear motivation and the results directionally confirm the design choices. The comparison of Gaussian noise degradation to convolution-based smoothing is especially informative and connects naturally to the diffusion literature.

- **Pilot scaling study.** Figure 3(C) shows a power-law fit to reconstruction loss across four model sizes (127k to 2.1M parameters), providing empirical evidence of favorable scaling behavior consistent with AR frameworks in other domains.

---

## Weaknesses

### Fatal

**None clearly fatal**, but the issue described below under Major (1) borders on fatal for specific claims in the paper.

---

### Major

- **Table 2 "+NoTS" rows are numerically incoherent and undermine trust in those specific results.** For PatchTST+NoTS, classification UCR-9 is reported as **11.71** and UEA-5 as **11.65** — while the PatchTST baseline shows 83.57 and 63.31. For imputation, +NoTS shows **1.003** while PatchTST shows 0.181. These numbers cannot be reconciled as either accuracy or error rate in a consistent unit alongside the rest of the table: an "error rate" of 11.71% for classification would imply ~88% accuracy (plausible), yet 1.003 imputation MAE versus a baseline of 0.181 would indicate catastrophic degradation. The averaged error rate column shows 18.33 (better than PatchTST's 21.78), which cannot be derived from the listed per-task values under any consistent aggregation. Whether this is a subscript/rendering artifact of the PDF or a genuine reporting error, the PatchTST+NoTS and iTransformer+NoTS rows as they appear in the submitted paper are uninterpretable. The claim that "NoTS improves their performance without specific backbone or adaptors" rests entirely on these rows and is therefore unverifiable. These rows must be corrected and re-verified.

- **Training objective (Equation 3) contains a likely subscript error with substantive implications.** The text states "we minimize the differences between **S**_k and the reconstructed **S**'_{k+1} for every k < K." The masking equation shows that **R**'_{k+1} is the transformer's prediction of level k+1 given inputs at levels 1 through k. The reconstruction **S**'_{k+1} should therefore be compared to its ground truth **S**_{k+1}, not the strictly more degraded **S**_k. If this is not a typo — if the model is trained to match a *more degraded* target — then the training signal actively pushes predictions toward losing information, contradicting the goal. If it is a typo, correcting it would still change the loss function meaningfully, and all reported results should be verified against the corrected objective.

- **Pretraining is limited exclusively to synthetic data, which significantly limits the "foundation model" framing.** All NoTS pretraining uses synthetic fBm or sinusoid datasets. Transfer to real-world tasks is interesting, but it does not demonstrate that NoTS can learn from or generalize across *heterogeneous real-world* time series distributions. The claim of "viable alternative for building foundation models" remains unsubstantiated without at least one experiment pretraining on a real-world multi-domain corpus and transferring to unseen datasets.

- **No forecasting evaluation.** Forecasting on standard benchmarks (ETTh/m, Weather, Traffic, Electricity) is the dominant evaluation paradigm for time-series pretraining papers at this venue. The paper scopes itself to classification, anomaly detection, and imputation, which is a legitimate choice, but the "foundation model" framing implies broader applicability. The complete absence of forecasting results means it is unknown whether NoTS's functional-decomposition objective helps or hurts on the task most commonly associated with AR pretraining for time series.

---

### Minor

- **Proposition 1 provides weak support for the implemented method.** The two sufficient conditions — "there exists a continuous mapping from **S**_i to the target" or "there exists an expressive tokenizer" — are existential and high-level. They do not demonstrate that the specific local/global smoothing degradations used in NoTS satisfy these conditions more efficiently than alternatives, nor do they rule out that a period-chunk tokenizer with sufficient capacity would satisfy the same conditions. The section is titled "An Intuitive Example," which is honest, but the paper's introduction elevates it to a stronger analytical contribution than it delivers. Clarifying which condition the NoTS construction concretely satisfies, with reference to Appendix A.3, would help.

- **No uncertainty estimates for real-world results.** Table 2's first eight rows compare NoTS-lw to SimMTM, bioFAME, and next-period prediction without standard deviations. Many margins are small (e.g., MSL 84.28 vs 84.15; PSM 96.88 vs 96.88; ETTm2 0.116 vs 0.107). Without repeated runs or confidence intervals, it is unclear which individual dataset wins are robust. The synthetic Table 1 reports three-run statistics, and the same discipline should apply to Table 2.

- **Computational overhead not reported.** Constructing K degraded views and processing all of them jointly in a grouped AR sequence increases sequence length by a factor of K relative to single-signal methods. Training and inference wall-clock time, and memory usage relative to MAE and next-period prediction, are not discussed. This matters for reproducibility and for the scalability claims.

- **Aggregated "Avg. error rate" metric in Table 2 is not defined in the main text.** Classification accuracy, anomaly detection F1-score (or accuracy), and imputation MAE operate on entirely different scales. The normalization procedure that converts them into a common "error rate" for averaging is deferred to the appendix. Since the headline "up to 6% improvement" depends on this metric, its construction should be explained in the main paper.

- **"26% improvement" in the abstract is not immediately clear.** The statistic refers specifically to the fBm subset of synthetic experiments (average of 37.80%, 8.41%, 31.44% ≈ 25.9%), not to all synthetic experiments; the sinusoid improvements are much smaller (5.66%, 0.98%, 2.20%). The abstract should specify "on fBm" to avoid overstating breadth.

---

### Tiny

- The masking definition — mask[Ω_k] = 0 for ∪_{m=1}^k Ω_m, −∞ elsewhere — is written as a scalar condition over a set rather than a standard per-position-pair attention mask. The notation should be clarified to specify how the group-wise causal pattern maps to a standard attention matrix.

- The practical sinc filter requires truncation; finite-length truncation changes the frequency response meaningfully. The windowing/truncation details and their effect on the stated "frequency cutoff of 0.5p_k" should be specified in the appendix.

---

## Nice-to-Haves

- **Forecasting experiments on at least one standard benchmark.** Even a single ETTh1 or Weather forecasting result would help establish that the functional narrative objective does not harm the most common downstream evaluation.

- **Pretraining on a heterogeneous real-world time series corpus** followed by zero- or few-shot transfer to held-out datasets. This is the canonical experiment for a foundation-model pretraining claim and would substantially strengthen the paper.

- **Failure mode visualization.** A side-by-side case study where next-period prediction recovers only local structure while NoTS recovers a non-local feature (e.g., trend or periodicity) on a real signal would make the paper's central motivation visually concrete.

- **Per-dataset breakdown table.** Average error rates can be dominated by a few datasets. A per-dataset heatmap or breakdown would help assess whether gains are consistent or concentrated.

- **Comparison to parameter-efficient fine-tuning baselines** (LoRA, standard prompt tuning without NoTS pretraining) to isolate the contribution of the pretraining objective versus the adaptation mechanism in the <1% parameter efficiency result.

- **Analysis of sensitivity to degradation schedule hyperparameters** (number of levels K, spacing of {p_k}, local vs. global only). The entire method depends on this schedule, and its current selection criteria are not discussed in the main paper.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Criticism that specific cited models (Chronos, TimeGPT, Lag-Llama) cannot be compared to:** The paper correctly positions NoTS against pretraining *methods* (MAE, next-period prediction) using a shared architecture rather than against production-scale foundation models trained on orders-of-magnitude more data. Demanding parity comparisons with Chronos or MOIRAI is scope creep given the paper's stated experimental design. Removed.

- **"Foundation model" framing is premature:** The paper uses appropriately cautious language throughout — "viable alternative for building foundation models," "potentially demonstrating," "preliminary experimental exploration." The pilot scaling study provides initial evidence. The criticism over-penalizes modest language in a research paper. Weakened/removed; the pretraining-on-real-data gap is the substantive issue and is retained as a Major weakness.

- **Criticism of the narrative/language analogy as "forced branding":** The analogy is metaphorical but not technically misleading, and the paper does not overextend it mechanistically. This is a style preference. Removed.

- **Demanding confidence intervals / repeated runs on large-scale benchmarks is non-standard:** For multi-dataset evaluation settings of this kind, single-run results are the norm. The weakness is retained only for Table 2 (NoTS-lw vs baselines) where margins are small and runs are feasible; it is removed as a demand for inference about statistical significance on the full 22-dataset suite.

- **The narrative analogy as a substitute for mechanism:** While the introduction is occasionally more metaphorical than formal, this is a style observation, not a substantive flaw. Removed.

- **Claiming the tokenization step alone determines generalizability:** The harsh critic argues that pointing to pointwise tokenization as "less generalizable" is overstated because downstream layers add context. This is correct as a nuance but doesn't change the paper's main argument about the AR sequence structure. Not a paper flaw. Removed.

---

## Novel Insights

The most genuinely novel framing in this paper — underexplored even in the reviews — is the observation that the *axis of the AR sequence* (degradation level vs. temporal position) determines what class of operators the transformer can learn to approximate. Theorem 1 demonstrates that a sampled-domain transformer cannot approximate the differential operator due to phase sensitivity, while constructing inputs as degraded functional variants routes around this by making successive tokens differ in *spectral content* rather than *temporal position*. This suggests a broader principle: the choice of sequence axis in AR transformers is a design variable with functional-analytic consequences, not merely a representation choice. The connection between this principle and the empirically observed improvements on SSC/WAMP (which are globally thresholded, hence discontinuous) is one of the paper's most compelling threads and deserves more explicit development. The diffusion model connection is also underexplored — NoTS can be viewed as a deterministic "cold diffusion" process in the reverse direction (coarse-to-fine rather than fine-to-coarse), and the contrast between convolution-based and Gaussian degradation in Table 3 implicitly tests this hypothesis on time series.

---

## Suggestions

1. **Correct and re-verify Table 2's "+NoTS" rows.** Provide individual task metrics in a consistent unit (all accuracy or all error rate) and verify that the per-task numbers derive to the stated average error rate. This is the highest-priority fix.

2. **Correct or clarify Equation (3).** If the target in the AR loss should be **S**_{k+1} (not **S**_k), correct the subscript and verify all reported numbers against the corrected objective. If **S**_k is intentional, provide a clear rationale.

3. **Define the average error rate aggregation in the main text**, including how classification accuracy, anomaly detection accuracy/F1, and imputation MAE are normalized and combined. The headline quantitative comparison depends on this.

4. **Add at least one forecasting experiment** on a standard benchmark (e.g., ETTh1) to test whether the functional narrative objective extends naturally to the task most associated with AR pretraining in this domain.

5. **Add training time and memory comparison** to MAE and next-period prediction baselines, even in a table in the appendix, so readers can assess the computational tradeoff of generating K degraded views.

6. **Strengthen the theory-to-implementation bridge.** Appendix A.3 apparently provides an example solution for the differential operator under the NoTS construction. Summarizing this example in the main text (1–2 sentences) would make Proposition 1 actionable rather than purely existential.

7. **Show per-dataset results** for at least the classification and anomaly detection subsets to allow assessment of whether gains are consistent or concentrated in specific datasets.