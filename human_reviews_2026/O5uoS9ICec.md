# Adaptive Energy Amplification for Robust Time Series Forecasting

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Deep learning models for time series forecasting often exhibit a spectral bias, prioritizing high-energy, low-frequency components while underfitting predictive but low-energy, high-frequency signals. 
Existing efforts attempt to correct this by amplifying high-frequency components but suffer from indiscriminate amplification, enhancing both meaningful signals and task-irrelevant noise, which destabilizes training and impairs generalization. 
To address this, we propose AEA (Adaptive Energy Amplification), a novel framework that reframes the problem as one of adaptive signal enhancement. 
AEA introduces two synergistic innovations: (1) a Spectral Mirroring mechanism that constructs a phase-preserving, low-frequency surrogate to guide targeted, distortion-free amplification of high-frequency signals; 
and (2) a lightweight Differential Embedding module that operates in a latent space to adaptively suppress common-mode noise. 
By decoupling signal amplification from noise suppression, AEA selectively enhances only informative features. 
Extensive experiments show that our model-agnostic framework consistently improves the forecasting performance of state-of-the-art backbones in both long-term and short-term forecasting tasks, while significantly enhancing training stability and generalization.
The code repository is available at https://anonymous.4open.science/r/AEA-685E/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the pervasive issue of spectral bias in deep learning models for time series forecasting, where models tend to overfit low-frequency, high-energy signals and neglect potentially predictive yet low-energy high-frequency components. The authors propose Adaptive Energy Amplification (AEA), a model-agnostic frequency-domain framework that (1) employs spectral mirroring for phase-preserving, distortion-free amplification of high-frequency signals based on low-frequency surrogates, and (2) integrates a lightweight differential embedding designed to suppress common-mode noise in the latent space. Extensive experiments across eight real-world datasets and four prominent forecasting backbones demonstrate that AEA not only improves forecasting accuracy but also training stability and robustness to high-frequency noise.

### Strengths
1.	The paper rigorously identifies indiscriminate amplification as a key limitation of prior frequency-domain approaches and provides empirical evidence motivating selective, content-aware high-frequency enhancement. The proposed AEA module is formulated as a lightweight plug-in that can be broadly applied to existing forecasting architectures.
2.	AEA is presented with detailed mathematical exposition and clear component decomposition. Fig. 2 effectively illustrates the conceptual and operational flow among spectral mirroring, differential embedding, and spectral energy alignment, enabling readers to understand both the intuition and implementation pathway.

### Weaknesses
1.	The proposed Spectral Mirroring + Adaptive Scaling framework closely resembles Amplifier. The additional components—phase mixing, differential embedding, and non-stationarity regularization—appear incremental rather than conceptually groundbreaking.
2.	The paper claims that existing models neglect high-frequency components. However, Fig. 1(a) shows that masking high frequencies leads to only minor degradation (< 5%), whereas masking low frequencies causes > 100% loss. This observation undermines the stated motivation and weakens the justification for “energy amplification” of high-frequency bands.
3.	Average improvements across datasets are around 3–4% (sometimes < 1%) in MAE/MSE. The paper reports win ratios instead of statistical tests, making it difficult to judge whether these gains are statistically meaningful or merely within random variation.
4.	The reproduced Amplifier results are significantly higher than those originally reported. Given that AEA’s claimed improvements are largely measured against this weakened baseline, the validity of the comparative conclusions is questionable.
5.	Although AEA is claimed to be model-agnostic, experiments are limited to DLinear, PatchTST, TimesNet, and Amplifier. Comparisons with stronger spectral models (e.g., FEDformer, FITS, FreTS) and emerging LLM-based forecasting architectures (e.g., AutoTimes, Time-LLM) are missing, restricting the assessment of AEA’s competitiveness.
6.	The paper omits efficiency profiling in comparison with baselines. Since AEA adds rFFT/iFFT, phase mixing, differential embeddings, and an energy predictor, the end-to-end cost may be non-trivial—particularly for long horizons and high-dimensional multivariate series.
7.	The paper would benefit from qualitative visualizations (e.g., input/output spectra or time-domain signals) illustrating what high-frequency structures are amplified and what types of noise are suppressed, thereby enhancing interpretability.
8.	It remains unclear whether learning dataset- or task-specific mixing ratios (instead of a fixed $\alpha$) could yield more adaptive behavior and improved performance.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper claims that forecasters tend to overlook high-frequency features because they carry less energy and more noise. 
The authors propose AEA, a lightweight, model-agnostic module that (i) mirrors spectrum to lift highs without breaking phase, (ii) uses a differential embedding to suppress presumed common-mode noise, and (iii) applies an energy-alignment head to keep the spectrum in shape. 
Results across standard benchmarks and several backbones look consistently positive. Learning frequency-specific features without indiscriminate amplification is sensible.

### Strengths
- Clear, modular, and light-weight. AEA adds small spectral components around any backbone; easy to adopt.
- Right target. Moves beyond “boost everything” by pairing amplification with noise control and post-hoc spectral alignment.
- Empirically steady. Gains appear across multiple datasets and models without heavy engineering.

### Weaknesses
Scope of evidence. Most tests sit in regular, well-behaved setups. It would help to see robustness under varied window lengths, irregular sampling, or mild regime shifts.

Assumption clarity. The “common-mode” story is intuitive, but the paper does not show how to diagnose it or what happens when it only partly holds.

Attribution. With several moving parts, it is still a bit hard to tell which module is doing the decisive work in different regimes.

For the details please refer to the question part. 

--------

Another concern is that while the proposed method improves multiple baselines, most comparisons are made only against common time-domain models (e.g., Dlinear, PatchTST) in Table 1. There is a lack of frequency modeling baselines.
I understand several frequency modeling works are architecutre-specific, but since this paper focuses on the frequency modeling, including more direct frequency-aware baselines with similar motivations is needed.
Suggested additions include:
- FEDformer — Tian Zhou et al. “FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.” ICML 2022 (PMLR 162).
- FiLM — Tian Zhou et al. “FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting.” NeurIPS 2022.
FreTS / Frequency-domain MLPs — Kun Yi et al. “Frequency-domain MLPs are More Effective Learners in Time Series Forecasting.” NeurIPS 2023.
- Fredformer — Xihao Piao et al. “Fredformer: Frequency Debiased Transformer for Time Series Forecasting.” KDD 2024 (arXiv:2406.09009).
- TimeMixer++ — Shiyu Wang et al. “TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis.” ICLR 2025 (Oral).

### Questions
Overall, I think this is a good paper. The central point is important: we should learn features within each frequency band, but not at the cost of indiscriminate, disproportionate amplification. I also like the model-agnostic, lightweight design, and the comparison angle with Fredformer is interesting.

- Phase handling. Why is it important to keep phase fixed? I agree phase matters, but the Introduction doesn’t really set this up, while §4.2 spends a lot of space on it. A short paragraph up front motivating this choice would help.

- Common noise. What exactly do you mean here, and how should a practitioner tell when the common-mode assumption holds? A brief clarification or simple diagnostic would make this more concrete.

- Results and Fredformer. The results look fine, but I don’t see a direct comparison with Fredformer. Also, the paper repeatedly warns that indiscriminate amplification can hurt accuracy—could you provide a more intuitive piece of evidence (e.g., a small visual or per-band error plot)? In Fig. 1, Fredformer seems to hold up well, not obviously harmed by “amplify everything.” A short explanation would make things better.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the spectral bias problem in deep learning models for time series forecasting, where models prioritize high-energy, low-frequency components while underfitting predictive high-frequency signals. The authors identify that existing frequency enhancement methods suffer from "indiscriminate amplification," which amplifies both meaningful signals and task-irrelevant noise. To address this, they propose AEA (Adaptive Energy Amplification), a model-agnostic framework featuring two key components: (1) Spectral Mirroring that constructs a phase-preserving surrogate from low-frequency components to guide targeted amplification of high-frequency signals, and (2) Differential Embedding that operates in a latent space to suppress common-mode noise. The framework is evaluated on eight benchmark datasets with four state-of-the-art backbones, demonstrating consistent improvements.

### Strengths
1 - The paper identifies and clearly articulates a fundamental limitation of existing frequency-aware forecasting methods - the indiscriminate amplification problem - and provides compelling empirical evidence through Figure 1b showing how existing methods degrade significantly when noise is injected into high-frequency bands.

2 - The proposed AEA framework is model-agnostic and demonstrates consistent improvements across diverse forecasting paradigms, with comprehensive experiments on eight datasets showing improvements on MSE and MAE.

3 - The theoretical foundation is well-developed, particularly Proposition 4.1 which provides formal analysis of how the differential embedding mechanism achieves adaptive noise suppression through superior bias-variance trade-off, supported by detailed mathematical proofs in the appendix.

### Weaknesses
1 - The computational overhead analysis in Section 4.6 claims the complexity is "negligible" but lacks empirical runtime comparisons - while the theoretical complexity is O(T·C), the actual wall-clock time impact when integrated with different backbones is not quantified, which is crucial for practical deployment.

2 - The paper relies heavily on a single hyperparameter configuration across all experiments without demonstrating whether these values are universally optimal or if dataset-specific tuning could yield better results, potentially limiting the framework's adaptability.

3 - The amplitude mixing strategy using a fixed ratio 0.5 between original and mirrored spectra appears simplistic - the paper doesn't explore adaptive or learnable mixing strategies that could potentially better capture dataset-specific characteristics.

4 - The evaluation focuses exclusively on long-term forecasting without examining short-term forecasting scenarios where high-frequency components might play different roles, limiting the generalizability of the findings.

### Questions
The intuition of using low-frequency spectrum to guide the selection of high frequency is really not straightforward and making sense to me naturally. Can you explain why this should work according to any prior knowledge?

Why did you choose a fixed mixing ratio 0.5 rather than making it learnable or adaptive to the input characteristics?

How does AEA perform on datasets with inherently noisy high-frequency components, such as financial time series with market microstructure noise?

Can you provide actual runtime measurements comparing AEA-enhanced models against their vanilla counterparts across different sequence lengths?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper targets spectral bias in time-series forecasting—modern models overfit high-energy low-frequency components and underfit informative high-frequency signals—and argues that prior frequency-aware methods amplify high frequencies indiscriminately, boosting noise as well as signal. It proposes Adaptive Energy Amplification (AEA), a plug-and-play framework with three modules: Spectral Mirroring (phase-preserving surrogate from low frequencies to guide targeted high-frequency amplification), Differential Embedding with a non-stationarity regularizer to suppress common-mode noise, and an Energy Predictor that aligns the forecast’s spectrum with the original data before returning to the time domain. AEA consistently improves four backbones across eight benchmarks and enhances training stability and generalization. It is robust to injected high-frequency Gaussian noise, unlike vanilla amplification methods.

### Strengths
- Clear articulation of spectral bias and a modular, plug-and-play design (Spectral Mirroring, Differential Embedding, Energy Predictor).

- Differential Embedding provides an explicit mechanism for common-mode noise suppression.

- Lightweight to integrate (FFT + small linear maps); easy to wrap around existing forecasters.

### Weaknesses
- Core rationale is heuristic: No formal guarantee that reversing low-frequency spectra provides a meaningful template for true high-frequency content; the mirroring prior may misalign with real signal structure.

- Missing “when it helps vs. hurts” analysis: Lacks characterization of data regimes (e.g., cross-band correlation, SNR profiles) where mirroring is beneficial or counterproductive; few concrete failure cases.

- Theory coverage is uneven: Differential Embedding has an intuitive justification, but the Energy Predictor’s alignment properties (stability, bias, identifiability) are argued mostly empirically.

- Limited ablations: Component ablations are centered on a single backbone; unclear whether module importance/generalization carries over to non-linear models.

- Hyperparameter rigidity: Fixed amplitude blend (e.g., α=0.5) and limited sensitivity analyses; unclear if learning α or making it frequency-dependent would help/hurt.

- Limited experiments: This paper discusses two types of enhancement approaches and points out their shortcomings. Figure 1(b) suggests the effectiveness of the proposed method for each enhancement approach (Amplifier and Fredformer). However, the evaluation experiments only evaluate the Amplifier. In other words, the effectiveness of the proposed method for indirect enhancement approaches is not shown (Figure 1(b) suggests that Fredformer+AEA may have better accuracy than Amplifier+AEA).

### Questions
- Can you provide formal conditions (e.g., in terms of cross-spectral correlation or energy monotonicity) under which spectrum reversal yields a functional high-frequency template—and a counterexample where it fails?

- Please characterize data regimes (bandwise SNR, cross-band correlation, spectral flatness/roll-off) that predict positive or negative gains from AEA, and report at least a few concrete failure cases.

- Do you have stability/identifiability results or bounds showing that the frequency-space alignment does not reintroduce low-frequency dominance or amplify noise? Any convergence/generalization insights?

- Replicate the component ablations (removing each AEA module) on at least one non-linear backbone (e.g., PatchTST, TimesNet, or Fredformer). Do the module-importance rankings persist?

- Figure 1(b) suggests gains for both Amplifier and Fredformer, but the main experiments only evaluate Amplifier. Please add full, protocol-matched results for Fredformer±AEA and compare against Amplifier±AEA (with significance and robustness tests).

### Soundness
2

### Presentation
3

### Contribution
2
