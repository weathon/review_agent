=== CALIBRATION EXAMPLE 50 ===

# Harsh Critic Review
Now I have read the full paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is accurate and descriptive. The abstract's headline claim of "31.1% MSE reduction on covariate-aware forecasting" refers specifically to improvement over the supervised model TimeXer, not over prior adaptation methods—a distinction the abstract does not make clearly. This framing inflates the perceived contribution, since much of the gain likely comes from leveraging a powerful pre-trained backbone (Sundial) rather than from CoRA's adaptation design per se.

The abstract characterizes CoRA as maintaining "equivalence of initialization and parameter consistency," but this property is never formalized. The claim is essentially that zero-initialization recovers the original TSFM at the start of training—which is true but trivial given the architecture chosen, and the same property holds for LoRA and DiT's adaLN.

### Introduction & Motivation

The problem is well-motivated: most TSFMs are univariate, yet real-world forecasting almost always involves covariates across multiple modalities. The gap between existing adaptation methods (ChronosX, AdaPTS, UniCA) and a principled, zero-initialized, causality-aware adapter is legitimate.

One concern: the introduction claims a "date-dependent approach" to ground covariate-aware adaptation in Granger causality. However, the learned Causality Embedding **W**_CE ∈ ℝ^N is a single static vector, not window- or time-dependent. This is internally inconsistent with the framing and is never reconciled.

### Method / Approach

**Causality Embedding design (Section 3.2):** The key technical contribution is **W**_CE, a learned N-dimensional vector used to re-weight covariates via Softmax before the adaLN injection. This is essentially *learned soft attention over covariates*, not a causal mechanism. The paper makes a strong interpretability claim—that **W**_CE "automatically evaluates Granger causality"—but the mechanism is purely supervised gradient descent optimizing forecasting MSE. The empirical alignment with Granger-Geweke statistics (Section 4.2, Figure 7) is shown only on a single dataset (ETTh1) and only correlates the learned weights with Granger scores; it does not demonstrate that the causality-aware routing is *why* performance improves. An alternative explanation is that it simply performs importance weighting of covariates.

Using Softmax over **W**_CE means all N covariates always receive strictly positive weight. If some covariates are genuinely uninformative or harmful, CoRA cannot zero them out—it can only assign them low weight. This is a meaningful limitation not acknowledged.

**Zero-initialization (Section 3.2 / "Zero-Initialization"):** The claim that at initialization the adapted model is "identical to the pre-trained TSFM" is correct given the architecture (Eq. 7: with α=β=γ=0, the output reduces to TSFM-Head(**Ẽ**^target)). However, this is a standard property of adaLN (DiT, Peebles & Xie 2023) and the paper claims it as a differentiating contribution vs. prior methods without adequately showing the others cannot be similarly zero-initialized.

**Covariate aggregation (Eqs. 3–4):** The asymmetry in how TS covariates (last-step embedding) versus language/image covariates (temporal average) are aggregated is an architectural choice with significant implications for temporal alignment and information retention. No ablation is provided for this asymmetry. For future-known covariates vs. past-only covariates (mentioned in footnote 1), the aggregation strategy also differs but is not analyzed.

**adaLN injection (Eq. 7):** The formula `x̂ = (1+α)·TSFM-Head(γ + (1+β)·Ẽ^target)` applies covariate information by shifting and scaling the *input* to the TSFM head, not inside the backbone. This means covariates can only influence the final head computation, not any of the backbone's internal representations of the target series. This is a severe architectural bottleneck that limits expressivity—if the TSFM head is a single linear projection (as in most decoder-only models), the covariate information can only do an affine rescaling of the output. The paper does not discuss this limitation.

### Experiments & Results

**Most significant concern — confounded comparison:** The primary comparison in Tables 1 and 2 pits CoRA (and all adaptation methods) using Sundial (a large-scale pre-trained TSFM) against supervised deep models (TimeXer, PatchTST, iTransformer, etc.) trained from scratch. Sundial was pre-trained on massive proprietary data collections. The reported 31.1% MSE reduction over TimeXer therefore reflects the advantage of large-scale pre-training, not CoRA's adaptation strategy specifically. A fair ablation would be CoRA applied to a much smaller or weaker backbone, to isolate the contribution of the adaptation framework itself from the backbone's generalization power.

**Unfair lookback length:** All experiments in Tables 1, 4, and 8 fix the lookback length L=2880, which is an extremely long context. This number is tailored to Sundial's architecture and is far beyond what most supervised baselines (e.g., TimeXer, PatchTST) are designed or tuned for. Standard benchmarks use L=96–512. Using L=2880 systematically disadvantages supervised baselines and is not standard practice. This is a significant threat to validity.

**Traffic OOM for all competing adaptation methods (Table 1):** AdaPTS, ChronosX, and UniCA are all marked OOM for Traffic (862 covariates). CoRA somehow does not run OOM, despite also running multiple frozen foundation models over 862 covariates. No explanation is given for how CoRA avoids the memory issue. If CoRA simply subsamples or processes the 862 covariates differently, this must be disclosed.

**No statistical significance testing:** No variance, standard deviations, or confidence intervals are reported for any result across all tables. For differences as small as 1–2 MSE percentage points (e.g., Time-MMD improvements), whether results are statistically significant is unknown.

**Time-MMD improvements are modest and inconsistent:** The abstract/introduction emphasize multi-modal capabilities, but gains on Time-MMD over the best baseline (UniCA) are only 1.9% in MSE (average). In Table 9 (full results), CoRA is *not* the best on multiple subsets—e.g., Climate MSE, Traffic CRPS, Social Good MSE, and Environment CRPS. The paper reports "state-of-the-art" based on the average, but the per-domain picture is mixed.

**Ablation study (Table 5):** The ablation study is on the EPF task only. Performance differences between ablation variants are often small (e.g., w/o zero-init vs full CoRA differ by roughly 1–2% across datasets). Given the lack of significance testing, it is unclear whether these differences are meaningful. More importantly, there is no ablation on the backbone choice (Sundial vs others for the same dataset) in the main ablation table.

**Interpretability study (Section 4.2):** The Granger-Geweke correlation is shown only on ETTh1 with 1000 windows. No quantitative correlation coefficient with aggregate statistics is reported (only a histogram). The choice to use spectral Granger-Geweke (rather than classical Granger F-test) is made without justification. Since the learned **W**_CE is a single static vector but Granger causality is measured per-window, there is an inherent mismatch: the correlation plot (Figure 7) presumably compares per-window Granger scores against a fixed scalar weight—this methodological detail is not clearly explained.

**Covariate-aware vs. standard multivariate framing (Section 4.1.4):** When extended to multivariate forecasting (Table 4), CoRA uses channel independence to predict each target variate individually, with all other variates as covariates. The comparison with iTransformer and Timer-XL, which model all variates jointly, is not apples-to-apples. CoRA's advantage here partly comes from having access to all other variates as "covariates" in a channel-independent manner, while still benefiting from Sundial's backbone.

**Reference artifacts:** Multiple citations in the paper appear as "(**?**)" (e.g., RT-1 and Time-MMD references in Section 4.1.2, Table 3, and Section A.1). These are clearly incomplete references that should have been resolved prior to submission.

### Writing & Clarity

The paper is generally readable, but the placement of core equations is disjointed. Specifically, Eq. 7 (the adaLN injection formula) appears on page 5 in the middle of a section that nominally describes results (page layout artifact), while the zero-initialization description appears after the equation. The formal algorithm (Algorithm 1) is deferred to the appendix, making it difficult to fully understand the method from the main text alone.

### Limitations & Broader Impact

There is no dedicated limitations section, which is below ICLR norms. Key limitations not discussed include:

1. **Computational overhead:** Running multiple frozen pre-trained models (ViT-base, Qwen3-Embedding-0.6B, Sundial) at inference time multiplies compute and memory requirements substantially. No inference cost analysis is provided.
2. **Dependency on backbone quality:** CoRA is essentially an adapter on top of Sundial. If the underlying TSFM has poor zero-shot performance on a domain, CoRA provides only marginal improvement (e.g., 3.3% MSE on FlowState and TimesFM in Figure 6 vs. 14.2% on Sundial).
3. **Static covariate weighting:** The Causality Embedding is a fixed, task-specific global weight, meaning it cannot adapt to distributional shift or regime changes within a time series.
4. **Head-only injection:** As noted above, restricting covariate injection to the TSFM head may be fundamentally insufficient for complex covariate-target interactions.
5. **Scope of "causality" claims:** The use of the term "causality" in "Causality Embedding" is methodologically misleading, as the learned weights are driven by predictive loss minimization, not by any formal causal criterion.

---

### Overall Assessment

CoRA addresses a real and important problem—adapting univariate TSFMs to multi-modal covariate-aware forecasting—and the modular design (frozen backbones, adaLN injection, learned covariate weighting) is clean and practical. The empirical results are broadly positive and consistent across tasks and TSFM backbones. However, the paper has three serious issues that need to be addressed before publication at ICLR: **(1)** The primary comparison is confounded—supervised baselines are disadvantaged by a non-standard lookback length (L=2880) and trained without large-scale pre-training, making it impossible to attribute gains to the adaptation method rather than the backbone; **(2)** the "causality" framing overstates what is essentially learned covariate gating, and the interpretability claim rests on a single dataset without adequate methodology; **(3)** no statistical significance testing is reported anywhere, and several key design choices (aggregation asymmetry, head-only injection, softmax-forced positive weighting) are not ablated or even acknowledged as limitations. The contribution is meaningful but the empirical substantiation needs substantial strengthening to meet ICLR's bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes CoRA, a framework for adapting Time Series Foundation Models (TSFMs) to covariate-aware forecasting tasks by integrating multi-modal exogenous information (text, images, time series). The method freezes pre-trained backbone models, employs a learnable Causality Embedding to weight covariates based on their predictive utility, and utilizes zero-initialized condition injection to prevent catastrophic forgetting. Extensive experiments demonstrate that CoRA achieves state-of-the-art performance on various benchmarks, including uni-modal, multi-modal, and multivariate scenarios, while requiring fewer training samples than supervised alternatives.

### Strengths
1.  **Strong Empirical Performance:** CoRA consistently outperforms strong supervised baselines (e.g., TimeXer, PatchTST) and recent adaptation methods (UniCA, ChronosX). For instance, Table 1 shows a 31.1% MSE reduction compared to TimeXer on long-term uni-modal covariate-aware forecasting, and Table 3 on Time-MMD demonstrates superior results on multi-modal tasks with textual covariates.
2.  **Thoughtful Adaptation Strategy:** The decision to use zero-initialized condition injection (similar to LoRA principles) to maintain parameter equivalence with the pre-trained model is theoretically sound for TSFMs. This design choice addresses the instability often seen in fine-tuning foundation models with new modalities, as evidenced by the ablation study in Table 5 where removing zero-initialization leads to performance degradation.
3.  **Generalizability and Interpretability:** The framework is demonstrated to work across diverse backbone architectures (Sundial, TimesFM, Chronos-Bolt) as shown in the "Generality" section (Figure 6) and Table 11. Additionally, the Causality Embedding is shown to correlate with Granger-Geweke causality (Figure 7), offering a degree of interpretability into which covariates are prioritized during prediction.

### Weaknesses
1.  **Computational Overhead:** While the paper emphasizes efficiency in *training* samples, the inference cost is potentially high. Leveraging separate frozen backbones for diverse modalities (e.g., ViT for images, Qwen for text) alongside the main TSFM implies significant computational resources compared to simpler adaptation methods that might only project covariates. The paper does not provide a detailed analysis of inference latency or memory consumption compared to baselines like UniCA.
2.  **Causality vs. Statistical Correlation:** The claim of using "causality" is technically grounded in Granger causality, which measures predictability rather than structural causality. While Figure 7 shows a correlation with Granger-Geweke, the learned weights are optimized for the downstream forecasting loss, potentially conflating predictive correlation with the interpretability suggested by the framework's name. This distinction needs clearer elaboration to avoid overclaiming causal discovery capabilities.
3.  **Handling of Temporal Dynamics in Non-Time-Series Modalities:** The authors acknowledge in the Limitations section (Section D) that for text and image covariates, the model applies mean aggregation over the temporal dimension, discarding fine-grained dynamics. This simplification contradicts the goal of fully integrating multi-modal temporal information and represents a significant constraint on the model's effectiveness in scenarios where the *temporal pattern* of the covariate (e.g., a changing image in a sequence) is crucial.

### Novelty & Significance
**Novelty:** The paper presents a novel paradigm shift in applying TSFMs to multi-covariate settings. While prior works like UniCA and ChronosX focus on adapting TSFMs, CoRA distinguishes itself through the specific combination of frozen embedding extraction, a learnable causality-based gating mechanism, and strict zero-initialization preservation. This addresses a specific gap where foundation models are typically univariate but real-world data is multi-variate.

**Significance:** The work is highly significant for time series communities and practitioners. The ability to utilize TSFMs in covariate-aware scenarios without full retraining allows for more efficient deployment in domains like finance and energy where exogenous variables are critical. The demonstration of strong few-shot performance (Figure 5) is particularly impactful for applications with limited historical data.

**Clarity & Reproducibility:** The methodology is described clearly in Section 3, and Appendix A provides detailed implementation settings. The results are benchmarked against SOTA models using standard datasets. However, the reliance on external pre-trained encoders (e.g., Qwen3Embedding, ViT) requires users to set up additional inference pipelines, which adds complexity to reproducibility compared to end-to-end models.

### Suggestions for Improvement
1.  **Analyze Computational Cost:** Add a table in the Appendix or main text comparing the inference time, GPU memory usage, and FLOPs of CoRA against baselines. Given the multi-encoder architecture, this is a critical practical consideration.
2.  **Clarify Causality Claim:** Explicitly distinguish between the "learned importance weights" and true causal inference. Consider framing the interpretability results as "predictive relevance alignment with statistical causality metrics" rather than implying the model learns structural causality.
3.  **Address Temporal Aggregation Limitation:** The authors admit mean aggregation is a limitation for text/imaging. Future work suggestions should be concrete now; consider a pilot experiment showing a simple temporal attention mechanism or LSTM over the covariate embeddings to demonstrate how much performance improves, validating the need for the limitation to be overcome.
4.  **Statistical Significance:** For the claims of improvement over baselines (especially close ones like UniCA), consider providing confidence intervals or p-values for the improvements on key datasets to assert statistical significance rather than just raw metric reduction.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add full backbone fine-tuning baseline** — Without comparing against fine-tuning the entire TSFM backbone (not just LoRA/SFT), the claim that freezing backbones prevents catastrophic forgetting is unsupported and the adaptation strategy's necessity is unclear.

2. **Include statistical significance testing** — ICLR expects rigorous claims; the 31.1% MSE reduction and other improvements need p-values or confidence intervals across multiple seeds to establish they are not due to random variation.

3. **Add computational efficiency metrics** — Claims about "fewer training samples" must be accompanied by training time, inference latency, and memory overhead compared to baselines, since frozen backbones with multiple embedding extractors may incur hidden costs.

4. **Complete RT-1 multi-modal results** — The RT-1 table is garbled/incomplete; ICLR reviewers will reject papers with missing benchmark results, especially for the novel multi-modal contribution.

5. **Test on more diverse TSFM backbones in main results** — Only Sundial is used for main comparisons; claiming generality requires showing CoRA works on TimesFM, Chronos, etc. in the primary evaluation, not just Section 4.2.

### Deeper Analysis Needed (top 3-5 only)
1. **Validate the Causality Embedding actually learns causality** — The correlation with Granger-Geweke is shown but not analyzed; without showing *which* covariates get high weights and whether this matches domain knowledge, the "interpretable" claim is unsubstantiated.

2. **Analyze failure cases and limitations** — ICLR expects honest assessment; the paper must identify scenarios where CoRA underperforms (e.g., weak covariate-target relationships, distribution shift) rather than only showing successes.

3. **Ablate the zero-initialization claim empirically** — The assertion that zero-init prevents catastrophic forgetting needs direct evidence comparing forgetting metrics (e.g., zero-shot performance degradation on original tasks) with and without zero-init.

4. **Explain why frozen backbones outperform trainable ones** — The core design choice needs justification beyond empirical results; analysis of embedding space drift or feature quality would strengthen the methodological contribution.

5. **Address potential data leakage concerns** — The claim that test sets don't overlap with Sundial's training data needs verification details; ICLR reviewers are sensitive to foundation model evaluation contamination.

### Visualizations & Case Studies
1. **Show Causality Embedding weight distributions** — Visualize which covariates receive high/low weights across datasets to demonstrate the selection mechanism actually differentiates informative from noise covariates.

2. **Provide failure case visualizations** — Show examples where CoRA performs poorly alongside successful cases; this reveals whether the method's benefits are consistent or dataset-dependent.

3. **Visualize embedding space before/after adaptation** — Show whether frozen backbone embeddings remain stable during CoRA training to validate the "no catastrophic forgetting" claim directly.

### Obvious Next Steps
1. **Include wall-clock training and inference time** — This should have been in the main experiments table; efficiency is critical for adaptation methods and ICLR expects this baseline information.

2. **Clarify RT-1 dataset construction protocol** — The subset creation process (skill selection, sequence filtering, train/val/test split) needs explicit description for reproducibility.

3. **Add error bars or variance across runs** — Single-run results are insufficient for ICLR; at minimum 3 seeds with standard deviation should be reported for all main tables.

4. **Compare against more recent TSFM adaptation methods** — ChronosX, UniCA, and AdaPTS are all 2025 arXiv papers; reviewers will question whether baselines are properly selected or if this is moving-target evaluation.

# Final Consolidated Review
## Summary
CoRA proposes a framework for adapting univariate Time Series Foundation Models (TSFMs) to covariate-aware forecasting by freezing pre-trained backbones as embedding extractors, learning a Causality Embedding to weight covariates, and injecting this information via zero-initialized adaptive layer normalization. Experiments across uni-modal, multi-modal, and multivariate forecasting tasks demonstrate strong performance compared to both supervised baselines and prior adaptation methods.

## Strengths
- **Addresses a real gap:** Most TSFMs are pre-trained on univariate data, yet real-world forecasting involves multi-modal covariates. The adaptation framework is well-motivated and practically relevant.
- **Strong empirical results:** CoRA consistently outperforms supervised baselines (TimeXer, PatchTST) and adaptation methods (UniCA, ChronosX, AdaPTS) across long-term forecasting (Table 1), short-term EPF (Table 2), and multi-modal benchmarks (Table 3). The improvements are substantial (e.g., 31.1% MSE reduction vs. TimeXer averaged across Table 1 datasets).
- **Generalizes across TSFM backbones:** Figure 6 and Table 11 demonstrate CoRA improves multiple foundation models (Sundial, TimesFM, Chronos-Bolt, FlowState), with MSE reductions of 3.3–14.2% depending on the backbone.
- **Principled adaptation design:** The zero-initialized adaLN mechanism ensures the adapted model begins training from the pre-trained state, avoiding catastrophic forgetting. The ablation study (Table 5) confirms removing zero-initialization degrades performance.
- **Interpretability evidence:** Figure 7 shows correlation between learned Causality Embedding weights and Granger-Geweke causality scores, providing some validation that the learned weights capture meaningful covariate importance.

## Weaknesses
- **Non-standard experimental setup favors the proposed method:** All experiments use a lookback length L=2880 (Table 1 caption), far longer than standard benchmarks (typically 96–512). This long context length benefits large pre-trained models like Sundial but may disadvantage supervised baselines designed for shorter contexts. Additionally, adaptation methods are trained only on horizon 720 and truncated for shorter horizons, while supervised baselines are trained separately for each horizon—introducing an asymmetry that could inflate perceived gains.

- **No statistical significance testing:** None of the tables report variance, standard deviations, or confidence intervals across multiple runs. For small improvements (e.g., 1.9% MSE reduction on Time-MMD vs. UniCA), it is unclear whether these differences are statistically significant.

- **Unexplained memory disparity:** In Table 1, AdaPTS, ChronosX, and UniCA all fail with out-of-memory on the Traffic dataset (862 covariates), yet CoRA runs successfully. The paper does not explain how CoRA avoids this memory bottleneck when processing the same high-dimensional covariate set with the same frozen backbone.

- **"Causality" terminology overclaims:** The Causality Embedding is a single static weight vector learned via gradient descent on forecasting MSE—essentially learned attention over covariates. The correlation with Granger-Geweke statistics (Figure 7) is shown only on ETTh1 and demonstrates alignment, not that the mechanism discovers true causal relationships. The Introduction's reference to a "date-dependent approach" is inconsistent with the actual static design.

- **No computational efficiency analysis:** CoRA requires running multiple frozen foundation models (Sundial for time series, ViT for images, Qwen3-Embedding for text) at inference time. No analysis of latency, GPU memory, or FLOPs compared to baselines is provided, leaving a practical deployment question unanswered.

- **Head-only injection limits expressivity:** Covariate information influences only the TSFM head via shift-and-scale operations (Eq. 7), not the backbone's internal representations. If the TSFM head is a simple linear projection (common in decoder-only models), covariate influence is limited to affine rescaling of output predictions. This architectural constraint is not discussed as a limitation.

## Nice-to-Haves
- Analysis of failure cases or domains where CoRA provides minimal benefit
- Comparison with full fine-tuning of the TSFM backbone (not just adapter methods)
- Temporal aggregation ablation for non-time-series modalities (currently uses mean pooling)

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"Reference artifacts (**) — the paper does reference actual works (Brohan et al., 2022 for RT-1; Liu et al., 2024a for Time-MMD) in the references section.*

- **"Unfair comparison using L=2880" as invalidating results:** The concern is valid but not fatal—the paper achieves strong results across multiple datasets and settings. The lookback length concern affects interpretability of *margins*, not the validity of showing CoRA works.*

- **"Confounded comparison with supervised baselines":** The comparison is appropriate—CoRA's goal is to leverage pre-trained models, and showing it outperforms supervised methods trained from scratch is a valid contribution. The backbone advantage is part of the method's value proposition.*

- **"Causality Embedding cannot zero out covariates due to Softmax":** True, but the paper's ablation shows removing the selection mechanism degrades performance, indicating the weighting provides value even with positive weights.*

## Novel Insights
The zero-initialized adaLN mechanism for TSFM adaptation is a principled contribution that addresses a real architectural challenge. The empirical finding that frozen backbone embeddings combined with learned covariate gating can substantially outperform end-to-end supervised models suggests that pre-trained temporal representations carry transferable predictive structure—even for tasks with novel covariate configurations. The correlation between learned weights and Granger-Geweke scores (Figure 7) provides modest evidence that gradient descent discovers predictive importance aligned with classical causality notions, though the mechanism is ultimately supervised learning, not causal discovery.

## Suggestions
- **Report variance across multiple runs** with standard deviations or confidence intervals for all main results to establish statistical significance of improvements.
- **Analyze computational overhead** by reporting inference latency, GPU memory, and FLOPs for CoRA vs. baselines—especially important given multiple frozen encoder requirements.
- **Explain the Traffic OOM asymmetry** in the paper—either document the memory footprint of each method or explain how CoRA handles high-dimensional covariates more efficiently.
- **Use standard benchmark lookback lengths** (96–512) as an additional evaluation to show results hold under typical experimental settings.
- **Clarify the "date-dependent" causality claim** by either implementing time-varying weights or revising the introduction to match the actual static design.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 8.0, 4.0]
Average score: 4.5
Binary outcome: Reject
