=== CALIBRATION EXAMPLE 46 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Mostly yes. The title clearly signals the central idea: representing time series as videos and using a spectro-temporal generative diffusion model. The phrasing “GEN## ERATIVE” is clearly an extraction artifact and not a paper issue.
- **Does the abstract clearly state the problem, method, and key results?**  
  Yes, at a high level. It states the modeling challenges, the STFT-based video representation, the custom video diffusion model, and the claim of new SOTA on unconditional generation.
- **Are any claims in the abstract unsupported by the paper?**  
  The main concern is that the abstract makes broad claims about “extensive empirical evaluation” and “new state-of-the-art” without qualifying the limited set of baselines and metrics used. The paper does support strong results on the reported benchmarks, but “new SOTA” is only established relative to a fairly narrow baseline set and only for unconditional generation. The abstract also suggests generality “beyond unconditional time-series generation,” which is speculative and not demonstrated.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  The motivation is good: existing diffusion approaches either stay in the time domain or collapse time into a static image. That said, the paper somewhat overstates the novelty of preserving temporal structure, since STFT-based time-frequency representations are already standard in signal processing and audio generation, and the key gap is specifically the lack of a video-style generative treatment for general multivariate time series.
- **Are the contributions clearly stated and accurate?**  
  Yes, the three contributions are explicit. However, contribution 1 (“formalize the treatment of time series generation as a video task”) is more a framing than a formal theoretical contribution.
- **Does the introduction over-claim or under-sell?**  
  It slightly over-claims. The paper suggests the paradigm is “optimal” or “more natural,” but the evidence is empirical rather than principled. Also, the claim that prior image-based methods “preclude” spatiotemporal modeling is true by construction, but the paper does not sufficiently justify why a video formulation should be expected to outperform direct time-domain diffusion beyond an inductive-bias argument.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  The high-level pipeline is clear: EMA trend decomposition, STFT on residuals, pack into a 4D video tensor, diffuse in that space, invert with iSTFT and add trend back. But several details are under-specified for reproducibility:
  - The exact EMA parameterization is not given.
  - It is not fully clear how complex STFT coefficients are represented and normalized before diffusion.
  - The invertibility conditions for the chosen STFT settings are asserted but not spelled out for each dataset/configuration.
  - The architecture description is somewhat high-level; the exact block structure, tokenization, patching, and attention order would need more detail to replicate.
- **Are key assumptions stated and justified?**  
  Some are justified, especially the claim that residuals may be more quasi-stationary. But important assumptions are not fully examined:
  - Treating covariates as an unordered axis is reasonable, yet the model still introduces a learnable bias matrix initialized from correlations, which implicitly assumes stable cross-channel structure.
  - Using a single STFT design across heterogeneous datasets may be fragile; the paper partly addresses this with sequence-length-dependent settings, but not with dataset-specific signal characteristics.
- **Are there logical gaps in the derivation or reasoning?**  
  Yes. The paper argues that representing time series as “videos” enables video diffusion architectures, but the mapping is not a natural video in the visual sense: one “spatial” axis is frequency and the other is variables, while the video time axis is STFT frame index. This is plausible, but the paper does not justify why the particular factorization of attention across temporal/frequency/covariate axes is the right one beyond intuition.  
  Another important issue is the handling of complex-valued STFT outputs. The method stores real and imaginary parts as channels, but there is no discussion of whether diffusion in that representation preserves the Hermitian/synthesis constraints needed for exact inverse transforms in the general case.
- **Are there edge cases or failure modes not discussed?**  
  Yes:
  - Non-stationary signals with rapid regime changes may be poorly served by EMA detrending.
  - Very short sequences may yield unstable STFT representations.
  - For multivariate series with strong causal directionality or heterogeneous sampling, the “video” construction may be less meaningful.
  - The method may generate spectrally plausible but temporally inconsistent signals if the inverse transform and trend reconstruction do not preserve global alignment.
- **For theoretical claims: are proofs correct and complete?**  
  There are no formal theorems/proofs, so this is not a concern. But several claims are presented as if theoretically grounded—e.g., invertibility, preservation of temporal dynamics, and the suitability of the representation—without formal guarantees.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Partially. The experiments do test unconditional generation quality, spectral/temporal matching, and scalability to longer sequences. This aligns with the main claim. However, they do **not** test the broader claim that the time-series-as-video paradigm generalizes to forecasting, imputation, or anomaly detection, which are mentioned as future directions.
- **Are baselines appropriate and fairly compared?**  
  The baseline set is reasonable for the literature cited, but there are two concerns:
  1. Several baseline results are taken from original papers rather than re-run under the same codebase and settings. That weakens comparability, especially because preprocessing, metrics, and sequence slicing can materially affect time series generation scores.
  2. The paper compares against only four methods in Table 1, and not against newer or stronger diffusion variants if any exist beyond those cited. If ST-Diff is meant to be a state-of-the-art claim for ICLR, the baseline coverage should be especially convincing.
- **Are there missing ablations that would materially change conclusions?**  
  Yes, several:
  - **Representation ablation:** raw time-domain diffusion vs STFT-only vs STFT + trend decomposition vs the full video formulation.
  - **Architecture ablation:** standard 2D/3D attention vs the proposed tri-axial factorized attention.
  - **Bias ablation:** effect of covariance-initialized bias matrices \(B_C\) and \(B_F\).
  - **Loss ablation:** the cross-covariance loss is introduced, but the paper does not show its isolated contribution.
  - **STFT parameter ablation:** window size, hop length, and patch size matter a lot, yet their impact is not studied.
  These are important because the paper’s central claim is architectural/representational, not merely “a diffusion model trained carefully.”
- **Are error bars / statistical significance reported?**  
  Error bars are reported in the tables, which is good. But the paper does not discuss statistical significance or the number of random seeds used in a way that makes the results easy to interpret. Some reported variances are small, but others are not, and the paper does not analyze whether the improvements are consistently robust across seeds.
- **Do the results support the claims made, or are they cherry-picked?**  
  The reported results support the claim that ST-Diff performs very well on the chosen benchmarks. However, the narrative is somewhat overstated:
  - In Table 1, ST-Diff is not best on every metric/dataset combination; in some cases Diffusion-TS is slightly better, and in some cases the margins are small.
  - The text claims “21 out of 24 metric–dataset combinations,” which is plausible from the table but should be read carefully because some baselines lack reported entries.
  - The qualitative analyses are all favorable, but they are not diagnostic enough to rule out mode collapse or overfitting.
- **Are datasets and evaluation metrics appropriate?**  
  The datasets are standard and diverse, which is good. The metrics are common in prior synthetic time series work, but they have known limitations:
  - Discriminative score and TSTR-style predictive score can be gameable and depend strongly on the choice of classifier/forecaster.
  - Context-FID depends on the embedding model and may not fully capture temporal structure.
  - Correlation error is useful but insufficient for dynamic dependence structure.
  The paper does include ACF/PSD plots, which helps, but the overall evaluation remains somewhat conventional for ICLR standards and would be stronger with more task-driven or downstream evaluation.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  The method section is the main issue. It is understandable at a high level, but several key implementation details are not crisp enough for a reader to fully reconstruct the approach, especially around:
  - exact tensor layout,
  - how complex-valued inputs are normalized,
  - how the trend channel is broadcast/resampled,
  - how the frequency and temporal attention blocks are sequenced,
  - how the bias matrices are applied in the transformer.
  The paper also sometimes mixes intuitive explanation with assertions of superiority without enough intermediate justification.
- **Are figures and tables clear and informative?**  
  Conceptually, yes. Figure 1 and Figure 2 seem central and appropriate. Tables 1 and 2 are informative. However, the qualitative figures are too optimistic to be fully convincing on their own, and the paper could better explain what exactly is being shown in the t-SNE/KDE/ACF/PSD plots. Also, the paper would benefit from a clearer visual comparison of the representation pipeline and the model block design.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Some limitations are acknowledged in the conclusion: higher computational and memory cost, and possible future work on more efficient video diffusion. This is good.
- **Are there fundamental limitations they missed?**  
  Yes:
  - Dependence on STFT choices and the potential mismatch between signal characteristics and fixed-window spectral analysis.
  - The method’s reliance on invertible, evenly sampled sequences.
  - The limited scope to unconditional generation; it is not yet shown that the representation helps for forecasting or imputation, where the claimed paradigm may be less directly beneficial.
  - The fact that the representation may be redundant or suboptimal for signals whose important structure is not well captured in the time-frequency plane.
- **Are there failure modes or negative societal impacts not discussed?**  
  The broader impact is limited but should include a caution about synthetic data misuse: if the method is used for privacy-preserving medical or financial time series, there is a risk of overclaiming privacy or utility without formal guarantees. Also, synthetic data can obscure rare-event structure if it matches aggregate metrics but misses tail behavior. The paper does not discuss these concerns.

### Overall Assessment
ST-Diff is a promising and conceptually interesting paper that introduces a clear representational idea: treating multivariate time series as spectro-temporal videos and using a tailored video diffusion model. The results are strong on standard synthetic-data benchmarks, and the paper is aligned with ICLR’s interest in representation learning and generative modeling. That said, the current submission falls short of a fully convincing ICLR-level contribution because the core empirical claim is supported by a fairly conventional evaluation suite, the baseline comparison is not as strong as it should be for a “new SOTA” claim, and the paper is missing the ablations needed to isolate whether the gains come from the video formulation, the STFT representation, the trend decomposition, the cross-covariance loss, or the architecture itself. The contribution stands as a solid and interesting step, but the evidence is not yet sufficiently granular or exhaustive to fully establish that the proposed paradigm—not just a well-tuned model—drives the improvement.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes ST-Diff, a diffusion-based framework for unconditional multivariate time-series generation that first maps sequences into an STFT-derived “spectro-temporal video” and then applies a custom video diffusion transformer. The main claim is that preserving an explicit temporal axis while modeling the frequency structure as video-like spatial content yields better sample quality and temporal/spectral fidelity than prior time-domain and image-based generators. On standard benchmarks, the paper reports strong empirical results and claims new state-of-the-art performance on several metrics.

### Strengths
1. **Clear and interesting high-level idea with a plausible inductive bias.**  
   The “time series as videos” framing is conceptually appealing: STFT preserves temporal evolution of spectra, and video diffusion architectures naturally model dynamics across frames. This is a legitimate and potentially useful bridge between signal processing and generative modeling, and it is more principled than collapsing sequences into static images.

2. **The method is tailored to the data structure rather than reusing a generic video model unchanged.**  
   The paper describes anisotropic patching, separate attention over temporal/frequency/covariate axes, learnable priors for covariate and frequency dependencies, and trend-residual decomposition. These design choices show an effort to align the model with multivariate time-series structure rather than applying a vision transformer naively.

3. **Broad benchmark coverage.**  
   The evaluation spans six datasets with diverse characteristics (synthetic periodic, financial, sensor, simulator, and fMRI data) and also includes longer sequence lengths on ETTh. That breadth is aligned with ICLR expectations for a generative modeling paper and helps demonstrate the method is not narrowly tuned to one dataset.

4. **Multiple evaluation axes beyond a single metric.**  
   The paper reports discriminative, predictive, correlational, and Context-FID scores, plus qualitative t-SNE/KDE plots and ACF/PSD comparisons. This is better than relying on one proxy metric and at least attempts to assess both distributional realism and temporal structure.

5. **Claims of scalability are supported by longer-sequence experiments.**  
   The additional ETTh experiments at lengths 64/128/256 are useful, since sequence-length scaling is a real concern for time-series generation. The paper reports consistently strong performance there, which strengthens the case that the representation is not only effective on very short sequences.

### Weaknesses
1. **The empirical comparison is not fully convincing as presented.**  
   The paper states that ST-Diff is state of the art, but several baselines are taken “from the original publications,” which can be problematic because evaluation protocols, preprocessing, and implementation details may differ. For an ICLR standard, this weakens the strength of the claim unless all methods are re-run under the same pipeline or the paper provides strong evidence of protocol matching.

2. **There is insufficient evidence that the gains come from the “video” formulation specifically.**  
   The method combines several ingredients: STFT representation, trend-residual decomposition, cross-covariance loss, anisotropic patching, factorized attention, and learnable biases. Without careful ablations, it is hard to know whether the improvement is due to the video view itself, the custom architecture, or generic engineering improvements. This is a significant gap given the paper’s central claim of a new paradigm.

3. **The paper does not convincingly demonstrate superiority over closely related frequency-domain methods.**  
   The most relevant comparison is arguably with frequency-domain diffusion and image-transform methods, but the reported baseline set is limited and the discussion of differences is mostly conceptual. In particular, the paper cites a frequency-domain diffusion baseline, but the evidence presented does not isolate why joint time-frequency modeling is better than pure frequency modeling.

4. **Reproducibility is only moderate.**  
   Some implementation details are given, but key aspects remain underspecified for exact reproduction: dataset preprocessing pipelines, normalization, the precise form and weighting of the cross-covariance loss, training schedules per dataset, random seed handling, and whether the reported metrics are averaged over multiple runs with consistent splits. The paper also lacks a full ablation table in the provided text.

5. **The novelty is incremental rather than obviously transformative.**  
   Recasting time series into time-frequency representations is well established, and video-style spatiotemporal modeling is a natural extension once one chooses an STFT tensor representation. The contribution is best viewed as a thoughtful integration of known components rather than a fundamentally new generative principle. That may still be publishable at ICLR if the empirical gains are strong and the analysis is thorough, but the novelty bar is not clearly surpassed from the current evidence.

6. **Some claims appear overstated relative to the evidence.**  
   The paper suggests the approach “establishes a new state-of-the-art” broadly and has potential for many downstream tasks, but the experiments are limited to unconditional generation. Claims about forecasting, anomaly detection, and general sequence modeling are speculative and not substantiated experimentally.

### Novelty & Significance
**Novelty:** Moderate. The core idea—using STFT-based time-frequency structure for generation—is not new in itself, but the specific framing of multivariate time series as videos and the tailored spectro-temporal diffusion architecture is a reasonable and somewhat novel integration. The method feels more like a strong synthesis of existing ideas than a fundamentally new conceptual breakthrough.

**Clarity:** Moderately good at the high level. The motivation and pipeline are easy to follow, but the parser-extracted text suggests some parts of the exposition and formulas may need cleanup in the final version. More importantly, the technical description is not yet sufficiently precise in key places for full implementation.

**Reproducibility:** Fair, but not excellent. The paper provides a number of hyperparameters and training details, yet important methodological specifics and ablations are missing or unclear. Reproducing the exact gains would likely require additional clarification.

**Significance:** Moderate to potentially strong if the reported results hold under rigorous controlled comparisons. The idea is practically relevant for multivariate generative modeling, and the benchmark coverage is broad. However, at ICLR’s acceptance bar, the paper would benefit from stronger evidence that the gains are due to the proposed paradigm rather than a collection of model and preprocessing choices.

### Suggestions for Improvement
1. **Add a rigorous ablation study.**  
   Separately evaluate: STFT representation vs. raw time domain; video attention vs. standard 1D/2D attention; with vs. without trend-residual decomposition; with vs. without cross-covariance loss; with vs. without the frequency/covariate bias matrices. This is essential to support the central thesis.

2. **Re-run or standardize baselines under a unified protocol.**  
   For ICLR-level confidence, the strongest version of the paper should compare against baselines trained/evaluated using the same preprocessing, splits, metrics, and compute budget. If that is not possible, the paper should clearly document discrepancies and ideally include sensitivity checks.

3. **Clarify the exact training objective and architecture details.**  
   Specify the cross-covariance loss formula, its weighting relative to diffusion loss, how trend is broadcast/resampled, the patching/tokenization scheme, and the dimensionality flow through the model. These are critical for reproducibility.

4. **Include stronger analysis of why the method works.**  
   Add diagnostics showing when the time-frequency representation helps most, e.g., periodic vs. non-periodic datasets, low-dimensional vs. high-dimensional settings, or stationary vs. non-stationary signals. A deeper frequency-domain error analysis would strengthen the paper’s scientific contribution.

5. **Report statistical significance and variance more fully.**  
   Provide multiple-seed results, confidence intervals, and perhaps paired significance tests across datasets. This is especially important because some metric differences appear small and the baselines come from heterogeneous sources.

6. **Temper broader claims or validate them with additional tasks.**  
   If the paper wants to claim general utility beyond unconditional generation, it should include at least one conditional task such as forecasting or imputation. Otherwise, the conclusion should stay focused on unconditional generation.

7. **Discuss computational cost more concretely.**  
   Since the method uses a video diffusion transformer, it may be substantially heavier than time-domain methods. Reporting training/inference time, memory usage, and parameter counts would help readers judge whether the gains justify the overhead.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct ablation showing STFT/video representation vs. the same diffusion backbone in raw time domain, frequency-only, and static-image STFT settings. Without isolating the representation, the claim that “time series as videos” is the key contribution is not convincing at ICLR standards.

2. Add a full architecture ablation: remove tri-axial attention, remove anisotropic patching, remove covariance/frequency bias matrices, and remove trend-residual decomposition. The paper claims the custom video model matters, but currently it is unclear which component actually drives the gains.

3. Compare against stronger and more relevant baselines, especially the most recent diffusion-based time-series generators beyond Diffusion-TS and ImagenTime, and include a fair retrained comparison rather than only reporting numbers from prior papers. ICLR reviewers will question state-of-the-art claims if baseline training protocols are not matched.

4. Add compute/memory/runtime comparisons against baselines, especially since the method is a video transformer and the paper itself admits higher cost. Without efficiency results, the practical value of the proposed paradigm is unclear and the claimed scalability is incomplete.

5. Add robustness experiments over STFT settings and sequence lengths with identical model capacity control. The current results could be driven by a favorable choice of FFT size/hop length rather than the proposed framework itself.

### Deeper Analysis Needed (top 3-5 only)
1. Provide a principled analysis of why the STFT video representation is preferable to a direct spectrogram or frequency-domain model for multivariate data. Right now the key claim is conceptual, but there is no evidence that preserving the temporal axis in video form is better than simpler alternatives.

2. Analyze whether the model is actually learning long-range temporal structure or mainly matching short-window spectral statistics. The ACF/PSD plots are too limited to justify the stronger claim of faithful dynamics, especially for long-horizon generation.

3. Report statistical significance, variance across seeds, and confidence intervals for all main metrics. ICLR expects robust evidence; the current tables do not establish whether the gains are stable or within noise.

4. Analyze failure cases by dataset type and covariate dimensionality. The paper claims broad generality, but it is unclear where the approach helps most and where it breaks, especially on highly nonstationary or high-dimensional series.

5. Explain the role of the cross-covariance loss quantitatively. There is no analysis of whether it improves correlation metrics without harming diversity or whether it merely overfits the evaluation metric.

### Visualizations & Case Studies
1. Add per-sample reconstructions and error plots in time and frequency space for both successful and failed generations. This would expose whether iSTFT reconstruction and the video diffusion process preserve fine-grained structure or just global averages.

2. Show side-by-side comparisons of real vs. synthetic spectro-temporal tensors for representative channels. The central claim is about modeling evolving spectra, so visual evidence in the proposed representation is necessary.

3. Include failure-case visualizations on difficult datasets or long sequence lengths where the method degrades. This would reveal whether the apparent gains are uniform or driven by easy regimes.

4. Visualize attention maps or learned bias matrices to demonstrate that the model uses the proposed covariate/frequency structure meaningfully. Without this, the custom attention design reads as speculative.

5. Add conditional/generated examples with known periodicity shifts or abrupt regime changes. These case studies would show whether the method handles the nonstationarity it is designed to address.

### Obvious Next Steps
1. Extend the method to conditional tasks such as forecasting and imputation, and evaluate against established benchmarks. The paper itself suggests this, but without it the “general paradigm” claim remains untested.

2. Test on additional real-world domains where time-frequency structure is critical, such as EEG, audio, or seismic data. If the representation is truly general, the paper should demonstrate that beyond the current benchmark set.

3. Study latent or compressed video diffusion variants to reduce the heavy computational cost. The current approach appears expensive, which limits the strength of the contribution unless efficiency is addressed.

4. Perform a controlled comparison against alternative invertible transforms and multi-resolution time-frequency representations. This would clarify whether the improvement comes from video modeling or simply from using a better spectral decomposition.

# Final Consolidated Review
## Summary
This paper proposes ST-Diff, a diffusion-based framework for unconditional multivariate time-series generation that first converts each sequence into a spectro-temporal “video” via STFT, then applies a custom video diffusion transformer, and finally reconstructs the signal with iSTFT. The core claim is that preserving an explicit temporal axis while modeling spectral evolution yields better generative quality than raw time-domain diffusion or static image-based approaches, and the paper reports strong benchmark results on several standard datasets.

## Strengths
- The central representation is genuinely interesting and well-motivated: STFT preserves temporal evolution of frequency content, and the paper’s “time series as videos” framing is a plausible inductive bias for generation. This is not a trivial repackaging of existing image transforms, because it explicitly retains the temporal axis rather than collapsing it.
- The evaluation is broader than a single proxy metric, covering Context-FID, discriminative score, predictive score, correlation error, plus ACF/PSD and t-SNE/KDE visualizations across six datasets and longer ETTh horizons. The longer-sequence experiments in particular are useful, since scalability is a real stress test for time-series generators.

## Weaknesses
- The empirical case for the main claim is not yet convincing enough for an ICLR paper. The results are reported against a limited baseline set, and several baseline numbers are taken from original papers rather than re-run under a unified protocol. Given the strength of the “new SOTA” claim, this weakens comparability substantially.
- The paper does not isolate what actually drives the gains. The method combines STFT representation, trend-residual decomposition, cross-covariance loss, anisotropic patching, factorized tri-axial attention, and learned bias matrices, but there are no ablations separating these components. As a result, the central “time-series-as-video” thesis is not established; the improvement could be coming from one or two engineering choices rather than the paradigm itself.
- Reproducibility is only partial. The paper still leaves important details underspecified, including the exact EMA parameterization, the precise cross-covariance loss weighting, how complex STFT coefficients are normalized/handled during diffusion, and the full training/evaluation protocol across datasets. For a method that depends on several interacting design choices, this is a real limitation.
- The evaluation remains conventional and somewhat indirect. Discriminative and TSTR-style predictive scores are standard in this literature but are not fully trustworthy on their own, and the qualitative ACF/PSD plots are supportive rather than diagnostic. The paper does not provide stronger task-level evidence that the model captures long-range dynamics beyond matching short-window statistics.

## Nice-to-Haves
- A full ablation study over representation, architecture, loss, and STFT hyperparameters would make the paper much more credible, especially raw time-domain vs. STFT-only vs. STFT+trend vs. full video model.
- Reporting multi-seed confidence intervals, runtime/memory, and parameter counts would help readers judge whether the gains justify the added complexity of a video diffusion transformer.
- A controlled comparison against additional recent diffusion-based time-series generators would strengthen the state-of-the-art claim.

## Novel Insights
The most interesting insight is that the paper’s value is not really “using STFT,” which is standard, but treating the time-frequency transform as a genuine spatiotemporal object and then designing a diffusion backbone to respect three axes: temporal evolution, spectral structure, and inter-covariate dependencies. That is a coherent conceptual bridge between signal processing and video generation. However, the current evidence suggests the paper is still more of a promising synthesis than a fully proven new paradigm, because the experiments do not yet disentangle whether the gains come from the video framing itself or from a bundle of model-specific choices layered on top of it.

## Potentially Missed Related Work
- **Time series diffusion in the frequency domain (Crabbé et al., 2024)** — closely related because it also uses a frequency-domain generative formulation; relevant for clarifying what joint time-frequency modeling adds over pure frequency modeling.
- **ImagenTime (Naiman et al., 2024)** — already cited, but especially relevant as the nearest static-transform/image-based baseline for the representation claim.
- **TimeGrad / CSDI** — relevant as established diffusion-based time-series generation/imputation baselines, though the paper’s task is unconditional generation rather than conditional inference.

## Suggestions
- Add a clean ablation table with at least: raw time-domain diffusion, STFT-only static representation, STFT + trend decomposition, full ST-Diff, and component removals for tri-axial attention, covariance/frequency biases, and cross-covariance loss.
- Re-evaluate the strongest baselines under the same preprocessing, sequence slicing, and metric code to support the SOTA claim more rigorously.
- Include a compact appendix with all missing implementation details: exact STFT settings per dataset, EMA parameters, loss weights, normalization, and seed-averaged metrics.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 2.0, 4.0]
Average score: 3.5
Binary outcome: Reject
