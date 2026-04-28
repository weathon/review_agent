## Summary
This paper proposes TrajGPT, a Transformer architecture for irregularly-sampled time series that introduces Selective Recurrent Attention (SRA) with data-dependent decay and interprets the model as discretized ODEs to enable time-specific inference at arbitrary timesteps. The method is evaluated on large-scale EHR datasets (PopHR: 489K patients, eICU: 139K patients) for forecasting, drug usage prediction, and phenotype classification tasks.

## Strengths
- **Data-dependent decay mechanism is empirically validated**: Table 3 ablation shows removing decay gating reduces forecasting recall at K=10 from 71.7% to 70.3%, confirming the design choice adds value beyond fixed decay (TimelyGPT's approach).
- **Time-specific inference outperforms auto-regressive on irregular data**: Table 1 shows time-specific inference achieves 71.7% recall at K=10 versus 65.5% for auto-regressive inference on PopHR, with consistent gains across forecast windows (Section 5.2).
- **Large-scale empirical evaluation on real-world EHR data**: Experiments span 489K patients (PopHR) and 139K patients (eICU) across multiple tasks including zero-shot classification, demonstrating the method works at meaningful scale rather than toy datasets.
- **Learned embeddings show clinically meaningful structure**: Figure 3.a visualizes token embeddings clustering into disease categories (mental disorders near neurological, circulatory near endocrine), suggesting the representation space captures semantic relationships beyond co-occurrence statistics.

## Weaknesses

### Fatal
None

### Major
- **Time-specific inference mechanism is underspecified for forecasting gaps**: Section 3.2 states the model computes $S_{n'} = D_{\Delta_{t_{n'}, n}} S_n + K_{n'}^\top V_n$ for target timesteps, where $D_\Delta$ depends on cumulative decay ratios $b_n/b_m$. However, the paper does not clarify how decay is computed for unobserved intervals where no input $X$ exists to compute $\gamma_n = \text{Sigmoid}(X_n w_\gamma^T)$. If the model uses the last observed $\gamma$ or assumes default decay for gaps, this should be explicitly stated, as it affects the validity of the "data-dependent ODE" claim for interpolation/extrapolation scenarios. This underspecification makes the core theoretical contribution (Section 3.2's ODE interpretation) difficult to verify or reproduce.

### Minor
- **Pre-training objective asymmetry between TrajGPT and baselines**: Section 4.4 states TrajGPT uses next-token prediction (NTP) for pre-training while baselines "without an established pre-training paradigm" use masking-based methods (MLM). The forecasting evaluation task is inherently NTP (predicting next diagnostic codes). While this is common practice when comparing foundation models against models without established pre-training protocols, it introduces a confounder: NTP pre-training naturally aligns with the evaluation objective. A control experiment with baselines pre-trained using NTP would strengthen the claim that architectural differences (SRA vs. standard attention) drive the performance gap rather than pre-training task alignment.
- **Zero-shot classification protocol lacks mathematical precision**: Section 5.1 describes projecting sequence representations "onto the same scale as token embeddings" and notes "clear separation between groups" enables zero-shot classification, but does not specify the decision mechanism (e.g., cosine similarity threshold, nearest centroid, linear probe). Without a precise formulation, the reported AUPRC scores (67.2% for insulin, 72.8% for CHF) are difficult to reproduce or compare against future work.

### Trivial
- **Figure reference missing**: Section 5.1 mentions "We visualized the projected head-specific decay vectors $w_i^h$ in Eq. 4 using the UMAP techniques (Fig. ??)" with a missing figure number.

## Nice-to-Haves
- **Decay visualization over time gaps**: Plotting learned $\gamma_n$ values for patients with irregular visit intervals would help verify whether decay adapts to time gaps or only to content, strengthening the "time-sensitive" claim.
- **Continuous variable extension**: Section 6 acknowledges the method currently focuses on discrete codes; extending to continuous physiological measurements (vitals, lab values) would better validate the ODE interpretation since discrete codes are poor proxies for continuous dynamics.
- **Baseline sensitivity analysis**: Testing whether performance gaps persist when baselines are pre-trained with NTP instead of MLM would help isolate architectural contributions from pre-training alignment effects.

## Removed Points
These points are flagged to be removed, treat them with caution:

1. **Harsh Critic Claim 1 (contradiction in time-specific inference)**: Partially retained as "underspecification" in Major weaknesses, but the critic's framing as a "fundamental contradiction" is overstated. The model can use last-observed $\gamma$ or default decay for gaps—this is underspecified rather than mathematically impossible. Removed the claim that this "invalidates the central theoretical claim."

2. **Harsh Critic Claim 2 (unfair pre-training comparison)**: Weakened to Minor weakness. Per Hard Rules, criticisms about unfair comparison where asymmetry favors the author's method should be removed, but here the asymmetry is common practice in foundation model evaluation. Retained as a valid concern about experimental design that could be addressed with controls, but removed the claim that performance gains "cannot be attributed to the SRA architecture."

3. **Harsh Critic Claim 3 (zero-shot undefined)**: Retained as Minor weakness but removed the claim that results are "unverifiable and likely overestimate." The mechanism appears to be similarity-based clustering; the issue is lack of precision for reproducibility, not fundamental invalidity.

4. **Strength Finder Claim about "strong zero-shot transfer"**: Kept but noted the limitation that the protocol needs more specification. Removed any implication that this demonstrates true zero-shot learning in the strictest sense (no access to label distributions).

5. **Generic strengths removed**: "Problem Relevance" and "Architecture Design" from Strength Finder were too generic ("addresses high-impact problem," "logical architectural evolution") and moved to Removed Points. Only strengths with specific evidence (Table 3 ablation, Table 1 comparison, Figure 3 visualization) were retained.

## Novel Insights
The paper's core contribution—combining data-dependent decay (inspired by Mamba/RetNet) with time embeddings for irregular EHR data—is a logical architectural evolution rather than a fundamentally novel insight. The ODE interpretation follows established patterns from Neural ODE literature (Chen et al., 2018; Rubanova et al., 2019) and does not offer new theoretical understanding beyond what those works established. The empirical finding that time-specific inference outperforms auto-regressive on irregular data is useful but expected given the method's design. No genuinely novel insights emerge beyond the paper's own contributions.

## Suggestions
1. **Explicitly specify gap handling**: Add a paragraph in Section 3.2 clarifying how $D_\Delta$ is computed when forecasting through unobserved intervals. State whether the model uses the last observed $\gamma$, assumes a default decay rate, or interpolates $\gamma$ values—and discuss the implications for the ODE interpretation.

2. **Add pre-training control experiment**: Re-train at least one strong baseline (TimelyGPT or PatchTST) using the same NTP objective as TrajGPT. This would isolate whether performance gains come from the SRA architecture or pre-training task alignment.

3. **Formalize zero-shot classifier**: Add an equation in Section 5.1 specifying the classification rule (e.g., $P(y=1) = \sigma(\text{sim}(h_{seq}, c_{\text{endocrine}})/\tau)$ where $c$ is a cluster centroid). Include the threshold selection method.

4. **Fix missing figure reference**: Replace "Fig. ??" in Section 5.1 with the correct figure number.

## Calibration and Scoring
I retrieved the following calibration anchors:

**High-scoring (≥6.0):**
- `/home/wg25r/review_agent/human_reviews_2026/oZJFY2BQt2.md` (CoTAR, 6.0): Medical time series transformer with centralized attention, strong multi-dataset evaluation, clear contribution. TrajGPT has similar empirical scale but less methodological clarity.
- `/home/wg25r/review_agent/human_reviews_2026/pXw0uRTSKT.md` (Record2Vec, 6.0): Portable EHR representations with LLMs, rigorous cross-site transfer evaluation. TrajGPT lacks this level of transfer analysis.
- `/home/wg25r/review_agent/human_reviews_2026/VVJ6Ck9JBl.md` (Aurora, 6.0): Multimodal time series foundation model with comprehensive ablations. TrajGPT has weaker ablation coverage.

**Medium-scoring (4.5-5.5):**
- `/home/wg25r/review_agent/human_reviews_2026/pQzQfslqlD.md` (TimeAlign, 5.5): Time series forecasting with distribution alignment, some underspecification in gradient flow and hyperparameters. Very similar profile to TrajGPT—solid experiments but methodological details need clarification.
- `/home/wg25r/review_agent/human_reviews_2026/JEIDxiTWzB.md` (ReIMTS, 5.5): Irregular time series forecasting with recursive multi-scale modeling, good experiments but notation confusion and efficiency questions. Comparable to TrajGPT.
- `/home/wg25r/review_agent/human_reviews_2026/dw2vxWVrA9.md` (D-LinOSS, 5.0): State-space models with learnable damping, strong theory but limited experiments. TrajGPT has better experiments but weaker theory.

**Low-scoring (≤4.0):**
- `/home/wg25r/review_agent/human_reviews_2026/Z8Hu7CJfZy.md` (EHR pretraining, 4.0): Time-conditioned EHR foundation model with methodological flaws (time split issues, missing baselines). TrajGPT is stronger—larger datasets, fewer fundamental flaws.
- `/home/wg25r/review_agent/human_reviews_2026/mMLzMZrH5Y.md` (UniTSGAN, 2.0): Unfair baseline comparisons, missing ablations, critical inconsistencies. TrajGPT is significantly stronger.
- `/home/wg25r/review_agent/human_reviews_2026/biZBdFpOzu.md` (Neural CDE corrector, 4.0): Fairness concerns in comparisons, insufficient theoretical analysis. Similar level of concern to TrajGPT but TrajGPT has better empirical scale.

**Score reasoning**: TrajGPT is positioned between the medium anchors (TimeAlign 5.5, ReIMTS 5.5, D-LinOSS 5.0) and low anchors (EHR pretraining 4.0). It has stronger empirical scale than the 4.0-scoring papers but similar levels of underspecification to the 5.5-scoring papers. The time-specific inference underspecification is comparable to TimeAlign's gradient flow underspecification (both scored 5.5). The pre-training asymmetry is less severe than the 2.0-scoring UniTSGAN's unfair comparisons. I score this at **5.5**, aligned with TimeAlign and ReIMTS—solid borderline work with real contributions but needing clarification on key mechanisms.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>