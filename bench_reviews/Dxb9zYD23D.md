## Summary

The paper introduces Spectro-Temporal Diffusion (ST-Diff), a framework that reframes multivariate time series generation as a video generation task by transforming signals into spectro-temporal video tensors via the Short-Time Fourier Transform (STFT). The representation preserves the temporal evolution of spectral content across frames, enabling the use of spatiotemporal diffusion architectures. The authors propose a custom transformer with tri-axial factorized attention and learnable bias matrices initialized from empirical data statistics, demonstrating state-of-the-art performance on unconditional time series generation across six benchmark datasets.

## Strengths

- **Novel representation paradigm:** The time-series-as-video approach is conceptually sound—unlike static image transforms (e.g., ImagenTime) that collapse the temporal axis, this representation explicitly preserves spectral evolution over time, enabling architectures designed for spatiotemporal dynamics.

- **Principled architectural design:** The anisotropic patching strategy (aggregating along frequency while preserving unit granularity along covariates) correctly avoids imposing artificial spatial locality on unordered covariates. The learnable bias matrices B_C and B_F, initialized from empirical cross-correlation and spectral covariance, meaningfully encode domain priors.

- **Strong empirical performance:** ST-Diff achieves substantial improvements on most benchmarks, with particularly notable gains on high-dimensional datasets (Energy, fMRI, MuJoCo). The long-sequence experiments (Table 2) demonstrate that discriminative scores remain stable as sequence length increases (0.030 → 0.032 → 0.029), while Diffusion-TS degrades more significantly.

- **Comprehensive qualitative analysis:** The paper provides t-SNE visualizations, Kernel Density Estimations, and per-covariate ACF/PSD comparisons that support the quantitative findings and demonstrate preservation of both temporal and spectral characteristics.

## Weaknesses

- **No ablation study:** The paper introduces multiple non-trivial components—trend-residual decomposition, tri-axial factorized attention, learnable bias matrices, anisotropic patching, and a cross-covariance auxiliary loss—yet provides no ablations to isolate which components drive performance. This is a significant gap; for example, the cross-covariance loss is introduced in Section 5 (Implementation Details) without formal presentation in the Method section, and readers cannot assess its contribution.

- **Unexplained failure on Sines dataset:** On the Sines dataset—the simplest synthetic benchmark designed as a sanity check—ST-Diff's Predictive Score (0.186) is approximately double that of all baselines (~0.093). This regression on the easiest dataset warrants investigation and explanation, particularly whether it relates to STFT resolution for very short sequences (L=24, nfft=11 yields coarse frequency bins).

- **Missing results for most relevant baseline:** ImagenTime is the closest competitor (both approaches use STFT representations), yet Table 1 shows "–" for 16 of 24 metric-dataset combinations for ImagenTime. The authors state they report results from original publications, but incomplete comparison to the most architecturally similar method weakens the evaluation.

- **Trend channel consistency during generation:** During training, the trend component is computed deterministically as EMA(x). During generation, the model produces the trend channel jointly with STFT coefficients from noise, with no constraint ensuring the generated trend approximates EMA of the generated signal. This decoupling could produce inconsistent outputs, particularly for non-stationary signals.

- **No computational cost analysis:** The paper acknowledges higher computational and memory costs but provides no quantitative comparison of training time, inference latency, parameter counts, or GPU memory usage. Without this, readers cannot assess whether performance gains stem from architectural innovation or simply from a larger, more expensive model.

- **Scalability evaluated on single dataset:** Long-sequence experiments (L=64, 128, 256) are conducted only on ETTh, which has strong periodic structure that may particularly benefit from spectral representations. Testing at least one additional dataset with different characteristics would strengthen scalability claims.

## Nice-to-Haves

- **STFT hyperparameter sensitivity analysis:** The FFT size formula (nfft = seq_len/2 - 1) and hop length are set heuristically. Analysis of how performance varies with different window sizes and overlap ratios would demonstrate robustness.

- **Ablation of bias matrices B_C and B_F:** While the bias initialization from empirical statistics is well-motivated, empirical validation that these components matter beyond standard attention would strengthen the architectural contribution.

- **Connection to audio spectrogram diffusion:** The related work section underplays the substantial audio generation literature using spectrograms with diffusion models (e.g., DiffWave, AudioLDM). Explicitly distinguishing this work from audio diffusion—where temporal structure is fundamentally different from multivariate time series with unordered covariates—would clarify novelty.

- **Visualization of generated spectro-temporal tensors:** Showing the intermediate video representation before iSTFT reconstruction would provide insight into whether the model learns coherent spectral evolution.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Demand for formal statistical testing:** While the reviewer requested paired t-tests or Wilcoxon tests, reporting means with standard deviations is standard practice in ML benchmark papers. The margins on most metrics are substantial enough that formal testing is unnecessary.

- **Demand for related work on "missing" methods:** Reviewers requested inclusion of Crabbé et al. (2024) as a baseline, but without external verification of whether this method applies to the same task and datasets, such requests should not be included.

- **Demand for conditional task experiments:** The paper explicitly scopes its contribution to unconditional generation. Requesting experiments on forecasting, imputation, or anomaly detection is scope creep—the paper should be evaluated on whether it does its stated task well.

- **Baseline fairness concerns:** The critique that "reporting results from original publications" invalidates comparisons is overstated. This is standard practice in ML papers; while controlled re-implementation would be ideal, it is not a requirement for acceptance.

- **Generic "topic is important" strength:** Removed as strengths must identify something specific this paper does well.

## Novel Insights

The time-series-as-video paradigm offers a principled middle ground between two extremes: time-domain models that lack spectral inductive biases, and static image transforms that sacrifice temporal structure. The key insight is that preserving the explicit temporal axis in a spectro-temporal representation allows video diffusion architectures to learn *how frequency components evolve*, rather than just learning to match frequency marginals. This is particularly valuable for capturing phase relationships and harmonic structures in periodic or quasi-periodic signals. However, the approach introduces a fundamental tension: STFT trades off time and frequency resolution, and for short sequences (L=24), the spectral representation may offer limited advantage over time-domain approaches—potentially explaining the Sines anomaly.

## Suggestions

- Add at least a minimal ablation study isolating the contribution of the cross-covariance loss and the learnable bias matrices—these are the most distinctive architectural choices and their impact should be quantified.

- Investigate and discuss the Sines Predictive Score anomaly; at minimum, acknowledge it and hypothesize whether it stems from STFT resolution limits on short periodic sequences or the trend decomposition interfering with pure sinusoids.

- Report computational costs (parameter count, training time, inference latency, GPU memory) for ST-Diff and at least the strongest baseline to enable practical assessment.

- Provide ImagenTime results for the missing metric-dataset combinations, or clearly explain why they cannot be obtained if the original paper did not report them.

- Consider adding one more dataset to the long-sequence evaluation to strengthen scalability claims beyond a single strongly-periodic dataset.