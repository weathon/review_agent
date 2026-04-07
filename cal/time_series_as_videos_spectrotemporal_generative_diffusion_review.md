=== CALIBRATION EXAMPLE 45 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title is clear and reflects the core contribution. The abstract effectively summarizes the motivation (limitations of time-domain and image-based methods), the proposed paradigm ("time-series-as-videos"), the method (ST-Diff), and the claim of state-of-the-art results. All abstract claims are testable against the paper content.

**Introduction & Motivation:** The problem is well-motivated, clearly distinguishing between time-domain methods (struggling with spectral dynamics/long-range dependencies) and image-based methods (collapsing the temporal axis). The key question posed is compelling. The three contributions are clearly stated and align with the paper's content.

**Method / Approach (Sections 3 & 4):** The core pipeline (STFT transform → video diffusion → inverse transform) is sound and reproducible in principle. However, several details are ambiguous or insufficiently justified, which hinders full reproducibility and critical assessment:
1.  **Trend Decomposition:** The use of an Exponential Moving Average (EMA) is mentioned but no parameters (e.g., smoothing factor) are provided. The rationale for isolating trend *before* the STFT is good, but the choice of EMA over other detrending methods (e.g., differencing, polynomial fits) is not discussed.
2.  **STFT Hyperparameters:** The description of how `nfft` and `hop_length` are set (Sec. 5) is clear for a fixed length, but the general mapping from variable-length series to a fixed-dimension video tensor lacks a formal definition. The statement "This normalization transforms variable-length time series into fixed-dimensional spectrograms" is vague.
3.  **Architecture Details (Sec. 4.3):** While the anisotropic patching and factorized attention are well-motivated, the description is high-level. Key details are missing: How are the `F'` frequency patches formed? What is the exact architecture of an "STDiff block" (Fig 2c is referenced but not included in text)? How are the initialized bias matrices (`B_C`, `B_F`) updated during training? Are they trainable parameters or fixed?
4.  **Invertibility & Stability:** The paper relies on the invertibility of the STFT. However, with a learned generative process, the generated `V_gen` may not correspond to a valid STFT of any real signal (e.g., violating consistency constraints across time frames). The potential for reconstruction artifacts or the need for a phase reconstruction algorithm (like Griffin-Lim) is not discussed. This is a significant technical gap.

**Experiments & Results (Section 5):** The experimental setup is comprehensive, using diverse datasets and a standard suite of metrics. However, there are major concerns regarding the baseline comparisons and the support for the state-of-the-art claim:
1.  **Incomplete Baseline Data:** Table 1 is critically flawed. Results for ImagenTime, a key baseline, are missing (`--`) for 18 out of 24 cells (all metrics on 4/6 datasets). The authors cannot claim to "establish a new state-of-the-art" while omitting data from the current leading image-based method for most comparisons. Either these results must be included, or a compelling reason for their omission must be given (e.g., irreproducibility, with efforts documented). This severely undermines the empirical validation.
2.  **Statistical Significance:** Error bars (presumably standard deviations) are reported, but no statistical significance testing (e.g., paired t-tests) is performed to substantiate claims of "outperforming" or "significant improvements." The improvements, while often large, should be validated statistically.
3.  **Ablation Studies:** The paper lacks critical ablations to justify its core design choices. The impact of (a) the trend-residual decomposition, (b) the spectro-temporal attention biases, and (c) the video-based modeling versus a simpler 3D CNN on the static spectrogram image, is not studied. Without these, it's unclear which components are driving the performance gains.
4.  **Context-FID Metric:** The description of "Context-FID" is brief. What pre-trained TS2Vec model is used? Is it trained on the same datasets? This metric's sensitivity needs more justification, especially since it shows ST-Diff's most dramatic improvements.
5.  **Computational Cost:** While mentioned as a limitation, there is no quantitative comparison of training/inference cost (FLOPs, memory, time) against baselines like Diffusion-TS or ImagenTime. For ICLR, a discussion of efficiency trade-offs is expected.

**Writing & Clarity:** The writing is generally clear. However, references to figures that are not in the provided text (e.g., Figure 1, 2a, 2b, 2c) make some parts of the methodology harder to follow. The appendix structure is logical.

**Limitations & Broader Impact:** Limitations are briefly acknowledged (computational cost) but are superficial. Major limitations are missed: (1) The fundamental assumption that the STFT video is a *natural* representation for a video model to learn, and the risk of the model learning spurious spectro-temporal patterns not corresponding to real time-series dynamics. (2) The potential negative societal impact of generating high-fidelity synthetic time series (e.g., financial, medical) for fraudulent purposes is not discussed.

### Overall Assessment

The paper proposes a novel and intellectually compelling paradigm—treating time series as spectro-temporal videos—with a well-designed custom architecture. The core idea is promising and could influence the field. However, the current submission has a critical flaw that prevents acceptance at ICLR's high bar: **the empirical validation is incomplete and insufficient to support the state-of-the-art claim**. The missing ImagenTime results in Table 1 are unacceptable for a comparative study. Furthermore, the lack of ablation studies and deeper discussion of the STFT inversion stability weakens the technical contribution. If the authors can provide complete baseline comparisons, add necessary ablations, and more thoroughly address the methodological limitations, the contribution would be strong. As presented, the paper is not yet ready for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes ST-Diff, a novel framework for unconditional multivariate time series generation by reframing time series as videos. The core idea is to apply the Short-Time Fourier Transform (STFT) to create a spectro-temporal video tensor (time frames × frequency bins × covariates), which explicitly preserves temporal evolution of frequency content. A custom video diffusion model with spectro-temporal attention biases is then used for generation. The method achieves new state-of-the-art results across several benchmarks and sequence lengths.

### Strengths
1. **Novel and Well-Motivated Paradigm**: The "time-series-as-video" concept is a clever unification that addresses limitations of prior time-domain (lacks spectral modeling) and image-based (collapses temporal axis) approaches. The argument is logically presented and grounded in signal processing principles.
2. **Comprehensive and Rigorous Evaluation**: The paper employs a standard suite of metrics (Discriminative, Predictive, Correlational, Context-FID) across six diverse datasets (synthetic, stochastic, real-world). Results show ST-Diff outperforms strong baselines (TimeGAN, TimeVAE, Diffusion-TS, ImagenTime) on 21/24 metric-dataset combinations for short sequences (L=24) and demonstrates superior scalability on longer sequences (L=64,128,256) on ETTh.
3. **Thoughtful Architectural Design**: The model incorporates domain-specific inductive biases effectively: anisotropic patching (aggregating frequency, not covariates), learnable covariate and frequency attention biases initialized from data statistics, and appropriate positional embeddings (RoPE for time/frequency, learned for covariates). The trend-residual preprocessing is a simple yet practical handling of non-stationarity.

### Weaknesses
1. **Limited Discussion of Computational Cost**: Video diffusion models are notoriously expensive. While mentioned as a limitation in the conclusion, the paper lacks a concrete analysis of training/inference time, memory footprint, or parameter count compared to baselines (especially time-domain Diffusion-TS or image-based ImagenTime). This is critical for assessing practical utility.
2. **Incomplete Comparison with Frequency-Domain Methods**: The related work mentions Crabbé et al. (2024) on frequency diffusion, but no direct comparison is provided. A discussion or experiment contrasting the joint time-frequency approach (video) with pure frequency-domain generation would strengthen the justification for the added complexity of the video representation.
3. **Overstated Claims on Architectural Novelty**: The spectro-temporal attention mechanism, while well-adapted, is presented as a key contribution. However, factorized attention across axes (spatial, temporal) is common in video transformers. The paper could more clearly delineate which aspects are adaptations of existing video architectures versus novel contributions specific to the spectro-temporal domain.
4. **Ambiguity in STFT Hyperparameter Selection**: The choice of `nfft = (seq_len/2)-1` and hop length is briefly stated but not justified. The impact of these critical parameters (affecting time-frequency resolution and invertibility) on generation quality is not studied via an ablation, leaving a potential reproducibility gap.

### Novelty & Significance
**Novelty**: The core conceptual novelty is high—treating multivariate time series generation as a video synthesis task via an STFT-derived representation. This elegantly bridges signal processing and spatiotemporal generative modeling. The architectural adaptations (anisotropic patching, data-driven attention biases) are sensible but more incremental.
**Significance**: The empirical results are strong and convincingly demonstrate the promise of the paradigm. If the approach generalizes, it could influence how the community represents sequential data for a range of tasks (forecasting, imputation). The work meets ICLR's bar for presenting a novel, well-evaluated idea with potential for impact.

### Suggestions for Improvement
1. **Add Computational Analysis**: Include a table or section comparing FLOPs, training time, memory usage, and/or sampling speed against key baselines. Discuss potential efficiency improvements (e.g., latent video diffusion, distillation) more concretely in the conclusion.
2. **Deepen the Frequency-Domain Discussion**: Conduct a direct experimental comparison with a frequency-domain diffusion baseline (e.g., Crabbé et al.) or at least include a detailed discussion in the related work/analysis on the advantages of preserving the time axis jointly versus operating solely in the frequency domain.
3. **Ablation Studies**: Provide ablations to quantify the contribution of key components: (a) the trend-residual decomposition, (b) the covariate and frequency attention biases, (c) the anisotropic patching strategy, and (d) the cross-covariance STFT loss. This would solidify the understanding of what drives the performance gains.
4. **Clarify STFT Parameter Sensitivity**: Discuss the rationale for the chosen STFT parameters more thoroughly. An ablation or sensitivity analysis on window size and hop length would strengthen the methodological robustness and aid reproducibility.
5. **Improve Figure and Table Readability**: Some figures (e.g., 3, 4, 5, 6, 7) have formatting artifacts or are densely packed, making them hard to parse. Ensure all axes are clearly labeled and consider splitting multi-part figures for clarity. In Table 1, explicitly state that "–" indicates unreported results.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study on core components.** The paper lacks experiments ablating the trend-residual decomposition, the spectro-temporal attention biases, and the video representation itself. Without this, it is impossible to attribute the performance gains to the novel paradigm versus specific architectural choices.
2. **Direct comparison with frequency-domain diffusion.** The work cites Crabbé et al. (2024) but provides no quantitative comparison. This omission critically undermines the claim of novelty in using a joint time-frequency representation, as it fails to situate the method against the most directly related prior art.
3. **Evaluation on a wider range of long sequences.** The scalability claim is supported only on ETTh up to length 256. To convincingly argue superior handling of long-range dependencies, tests on more datasets (e.g., weather, audio) at lengths of 512+ are necessary.
4. **Computational cost and efficiency analysis.** The method employs a heavy video diffusion model. A comparison of training/inference time, memory footprint, and parameter count against time-domain (Diffusion-TS) and image-based (ImagenTime) baselines is essential to assess practical utility.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of temporal coherence in the generated spectro-temporal videos.** The core claim is modeling spectro-temporal dynamics, but there is no metric or study (e.g., frame consistency, spectral flux) to verify that the generated frequency content evolves coherently over time, rather than producing static patterns.
2. **Quantifying the impact of the initialized bias matrices.** The covariate and frequency bias matrices are initialized from data statistics. An analysis comparing performance with learnable biases, random initialization, or no biases is needed to prove these are more than a minor trick.
3. **Investigation of the trend-residual decomposition's necessity and limitations.** The paper assumes this decomposition handles non-stationarity, but does not analyze its failure modes (e.g., for signals with rapidly changing trends) or whether the trend channel is modeled effectively by the diffusion process.

### Visualizations & Case Studies
1. **Side-by-side plots of real and generated time series.** The paper shows t-SNE and ACF/PSD, but standard practice requires visual comparison of raw temporal waveforms for several samples. This is the most direct way to assess visual fidelity and spot obvious failures.
2. **Visualization of generated spectro-temporal video frames.** Showing a sequence of generated spectrogram frames (frequency vs. covariate) alongside real ones would directly validate whether the model captures meaningful evolution in the time-frequency domain, as claimed.
3. **Case studies highlighting failure modes.** The paper only shows successes. Examples where the method fails (e.g., on data with sharp, aperiodic events) would clarify the boundaries of the approach and strengthen the critique.

### Obvious Next Steps
1. **Include a comprehensive ablation study.** This is a standard requirement for a methods paper at ICLR to justify architectural choices and isolate the source of improvements.
2. **Benchmark against frequency-domain diffusion (Crabbé et al., 2024).** This is a glaring omission given the directly overlapping premise and must be addressed to claim novelty.
3. **Provide a thorough computational profile.** Comparing FLOPs, memory, and wall-clock time against key baselines is necessary for readers to understand the trade-offs of the proposed paradigm.
4. **Extend validation to conditional tasks.** The paper suggests the paradigm is broadly applicable, but a minimal experiment on a conditional task (e.g., forecasting or imputation) would have substantiated this claim more concretely.

# Final Consolidated Review
## Summary
This paper proposes a novel paradigm for unconditional multivariate time series generation by reframing time series as spectro-temporal videos. It introduces ST-Diff, a framework that uses the Short-Time Fourier Transform (STFT) to map a time series into a video tensor (time frames × frequency × covariates), applies a custom video diffusion model with tailored attention mechanisms, and reconstructs the signal via the inverse STFT. The method establishes a new state-of-the-art on standard benchmarks and demonstrates strong scalability to longer sequences.

## Strengths
- **Novel and Well-Motivated Paradigm:** The core idea of treating time series as videos via the STFT is a significant conceptual contribution. It elegantly addresses limitations of prior time-domain methods (poor spectral modeling) and image-based methods (collapsed temporal axis) by preserving explicit temporal evolution of frequency content, enabling the use of spatiotemporal architectures.
- **Strong and Comprehensive Empirical Results:** The paper demonstrates superior performance across six diverse benchmarks (synthetic, financial, sensor, physiological) using a standard suite of four evaluation metrics. ST-Diff outperforms strong baselines (TimeGAN, TimeVAE, Diffusion-TS, ImagenTime) on the vast majority of metric-dataset combinations for short sequences (L=24) and shows remarkable scalability, maintaining high performance on much longer sequences (up to L=256) where other models degrade.
- **Thoughtful Architectural Design:** The model incorporates sensible, domain-specific inductive biases, including anisotropic patching (aggregating frequency but not covariate axes), learnable covariate and frequency attention biases initialized from data statistics (covariance/correlation), and appropriate positional encodings (RoPE for time/frequency, learned for unordered covariates). The trend-residual preprocessing is a simple, practical step to handle non-stationarity.

## Weaknesses
- **Insufficient Analysis of Computational Cost:** Video diffusion models are computationally intensive. While mentioned as a limitation, the paper lacks a quantitative comparison of training/inference time, memory footprint, or parameter count against key baselines (e.g., Diffusion-TS, ImagenTime). This is a substantive weakness for assessing the method's practical utility and trade-offs.
- **Missing Direct Comparison with Frequency-Domain Methods:** The related work cites frequency-domain diffusion (Crabbé et al., 2024) but provides no experimental comparison or detailed discussion contrasting the joint time-frequency (video) approach with pure frequency-domain generation. This omission weakens the justification for the added complexity of the video paradigm and leaves the novelty claim partially unsituated.
- **Limited Ablation Studies:** The paper lacks ablation experiments to isolate the contribution of its key components (e.g., trend-residual decomposition, spectro-temporal attention biases, anisotropic patching, the cross-covariance STFT loss). Without these, it is difficult to attribute the performance gains definitively to the novel representation versus specific architectural choices.

## Nice-to-Haves
- A sensitivity analysis or ablation on the choice of STFT hyperparameters (window size, hop length) to demonstrate robustness and guide reproducibility.
- More detailed visualizations comparing raw temporal waveforms of real and generated samples, in addition to the provided t-SNE and ACF/PSD plots, to better assess perceptual fidelity.
- A brief discussion or small experiment on the invertibility constraint and whether the generated video tensors correspond to valid STFTs, or if post-processing (e.g., consistency enforcement) is needed.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Incomplete Baseline Data / Missing ImagenTime results":** The paper states results are taken from original publications. The '–' for ImagenTime likely indicates those metrics were not reported in the original ImagenTime paper, not that the authors failed to run the comparison. This is not a flaw in the current work's evaluation.
- **Weakness: "Lack of statistical significance testing":** Reporting means and standard deviations is standard practice in the field. Demanding formal significance tests imposes a rigor requirement not commonly expected in this area.
- **Weakness: "Fundamental assumption that STFT video is a natural representation... risk of spurious patterns":** This is a vague, philosophical criticism not grounded in the empirical results, which show the model successfully captures dynamics. The paper's strong performance counters this concern.
- **Weakness: "Invertibility & Stability / potential for reconstruction artifacts":** The paper correctly states the STFT is invertible and uses a standard overlap-add method. The concern about generated tensors not corresponding to valid STFTs is speculative; the model is trained to generate in that domain and the results show successful reconstruction.
- **Weakness: "Overstated Claims on Architectural Novelty":** The paper clearly positions its architectural adaptations (anisotropic patching, data-driven biases) as tailored for the new representation. It does not claim to invent factorized attention, but to usefully apply and adapt it.

## Novel Insights
The paper's core insight is that the joint time-frequency domain, represented as a video, is a more powerful and natural representation for generative modeling of time series than either the raw time domain or a static 2D image. This insight bridges signal processing (STFT) with modern spatiotemporal generative models, suggesting that explicitly modeling the evolution of spectral components over time is a key inductive bias for capturing complex temporal dynamics. The strong empirical gains, especially on long sequences and high-dimensional data, provide evidence that this paradigm can overcome fundamental limitations of previous approaches.

## Suggestions
- Add a subsection or table quantifying the computational cost (e.g., training time, GPU memory, sampling steps/time) of ST-Diff compared to primary baselines like Diffusion-TS and ImagenTime to provide a complete picture of the performance-efficiency trade-off.
- Conduct a direct experimental comparison with a frequency-domain diffusion baseline (e.g., Crabbé et al.) or, if not feasible, include a thorough discussion in the analysis section contrasting the advantages of preserving the time axis jointly versus operating in a purely frequency-domain latent space.
- Perform an ablation study to demonstrate the contribution of key design elements: the trend-residual decomposition, the learned covariate/frequency attention biases, and the anisotropic patching strategy. This will strengthen the technical justification and help the community understand what drives the improvements.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 2.0, 4.0]
Average score: 3.5
Binary outcome: Reject
