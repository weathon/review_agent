=== CALIBRATION EXAMPLE 47 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title "Time Series as Videos: Spectro-Temporal Generative Diffusion" effectively captures the paper's core conceptual contribution. The abstract clearly states the problem, the proposed solution (ST-Diff), and the claimed outcome (new SOTA). One point of concern is the strong claim of a "new state-of-the-art," which hinges entirely on the experimental results being robust and the baselines being fairly compared. The abstract also mentions potential for broader sequence modeling tasks, which is not substantiated in the main paper beyond a brief mention in the conclusion; this could be seen as slightly overreaching without accompanying evidence or discussion.

**Introduction & Motivation**
The introduction effectively motivates the problem, highlighting the limitations of existing time-domain and image-based approaches. The framing of the central question—creating a representation that preserves both explicit temporal evolution and frequency structure—is compelling and clearly sets the stage for the proposed "time-series-as-video" paradigm. The contributions are stated clearly. A minor point: the critique of Transformer-based approaches lacking an "ideal inductive bias" might be slightly overstated, given their success in many sequence tasks, but it serves to justify the search for alternative representations.

**Method / Approach**
This is the core of the paper and where the most significant questions arise. The overall pipeline (STFT -> Video Diffusion -> iSTFT) is conceptually sound and well-explained.
1.  **STFT Transformation & Hyperparameters:** The transformation of time series to a video tensor is clearly described. However, critical implementation choices are not justified or analyzed. The selection of STFT parameters (window size `N`, hop length `H`) is central to the time-frequency trade-off. The paper states these are scaled relative to sequence length (e.g., `nfft = (seq_len/2)-1`), but provides no rationale for this specific scaling or analysis of its impact. For a sequence length of 24, this results in only 11 frequency bins, which is a very coarse spectral representation. A sensitivity analysis or justification for these parameters is missing.
2.  **Trend-Residual Decomposition:** The use of an EMA to extract a trend is a practical heuristic. However, the choice of the EMA smoothing factor (alpha) is not discussed. This parameter controls what is considered "trend" vs. "residual," which could significantly affect the stationarity of the residual and the effectiveness of the STFT.
3.  **Model Architecture Details:**
    *   The anisotropic patching strategy is well-motivated (to avoid imposing spatial correlations on covariates). However, the choice of patch size (e.g., (2,1), (4,1)) in Table 4 is not justified or ablated.
    *   The learnable bias matrices **B_C** and **B_F** are an interesting idea. The paper states they are initialized from "empirical statistics of the data" (cross-correlation and covariance of log-magnitudes). This process needs more precise description for reproducibility: Are these matrices computed from the entire training set and then fixed, or are they learnable parameters initialized this way? How is the covariance of STFT log-magnitudes computed (averaged over time frames?)? Furthermore, the claim that these biases "encourage the model to respect domain-relevant structural and spectral relationships" is not validated by any ablation study. Their necessity and contribution to final performance are unclear.
    *   The tri-axial factorized attention mechanism is a major architectural component. While motivated, there is no ablation to show that this complex design is superior to a simpler, unified spatiotemporal attention mechanism. Given the computational cost mentioned in the conclusion, justifying this design choice is important.
4.  **Additional Loss Term:** In Section 5 (Implementation Details), a "cross-covariance loss applied directly to the Short-Time Fourier Transform (STFT) magnitudes" is mentioned. This is a significant training detail that is **not described in the Method section (Section 4)**. Its inclusion affects reproducibility and understanding of the final model's capabilities. Its purpose and formulation should be integrated into the methodological description.

**Experiments & Results**
The experimental setup is comprehensive, using multiple datasets and established metrics.
1.  **Baseline Comparison (Major Concern):** Table 1 is missing results for ImagenTime on several metrics (marked with '–'). For a paper claiming a new SOTA, a direct and complete comparison with the current strongest image-based baseline is crucial. The authors should either report the missing numbers from the ImagenTime paper or re-run the evaluation using their own implementation/metrics to ensure a fair comparison. This gap significantly weakens the claim of superior performance.
2.  **Long-Sequence Generation:** The scalability experiment on ETTh is positive. However, it only tests one dataset. To strongly claim that the method "overcomes a key limitation" of other approaches regarding long contexts, showing results on other datasets (e.g., MuJoCo, Energy) at longer lengths would be more convincing.
3.  **Qualitative Analysis:** The t-SNE and KDE plots (Fig. 3, 5) show good distributional alignment. However, t-SNE is known to be sensitive to hyperparameters and should be interpreted cautiously. The ACF and PSD plots (Fig. 4, 6, 7) are excellent for demonstrating temporal and spectral fidelity. The note in the caption of Fig. 4 about "some slight difference... on high-frequency ones" is honest and should perhaps be expanded into a brief discussion in the main text (e.g., is this a limitation of the coarse STFT resolution for short sequences?).
4.  **Ablation Studies:** There is a critical lack of ablation studies. Key design choices—such as the necessity of the video representation vs. a static spectrogram image, the effect of the trend-residual decomposition, the contribution of the specialized bias matrices, and the tri-axial attention—are not experimentally validated. For an ICLR submission advocating a new paradigm and a custom architecture, these ablations are essential to isolate the source of the gains.
5.  **Statistical Significance & Error Bars:** Error bars (presumably standard deviations over multiple runs) are reported, which is good practice. The improvements often appear substantial relative to the variance.

**Writing & Clarity**
The paper is generally well-written and clearly structured. The figures effectively illustrate the pipeline and architecture. The main issues with clarity are in the Method section, where key details (e.g., bias matrix initialization, cross-covariance loss) are either omitted or described elsewhere, hindering a self-contained understanding. The connection between Figures 1 and 2 could be clearer; Figure 1 is referenced for the pipeline, but the model details are in Figure 2, causing some back-and-forth for the reader.

**Limitations & Broader Impact**
The conclusion briefly mentions computational/memory costs and suggests future work (conditional tasks, anomaly detection). This is adequate. However, a more thorough discussion of limitations could strengthen the paper. For example: the sensitivity to STFT hyperparameters, the assumption that an EMA adequately captures non-stationarity for all datasets, potential boundary effects from the STFT windowing, or the challenge of generating very high-frequency components noted in the qualitative results. The broader impact statement is minimal and generic, which is acceptable for this type of technical work.

### Overall Assessment
This paper presents a novel and conceptually appealing idea: reframing time series generation as video generation in a spectro-temporal domain. The core premise is strong, and the proposed ST-Diff framework is a plausible instantiation of this idea. The experimental results, while showing impressive quantitative gains, are currently undermined by **incomplete baseline comparisons** (specifically with ImagenTime) and a **lack of critical ablation studies** to justify the architectural complexities. Furthermore, key methodological details regarding STFT parameter selection, bias matrix initialization, and an auxiliary loss term are either missing or inadequately described, affecting reproducibility. If these issues are addressed—by providing a complete comparison, adding ablations, and elaborating on methodological choices—the contribution could be significant and likely meet ICLR's bar. In its current form, however, the empirical validation and methodological clarity are not yet sufficient to fully support the claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a novel paradigm for multivariate time series generation by reframing it as a video generation task. The proposed Spectro-Temporal Diffusion (ST-Diff) framework uses the Short-Time Fourier Transform (STFT) to map a time series into a spectro-temporal video tensor (with frequency and covariate dimensions as spatial axes and STFT time frames as the temporal axis). A custom video diffusion model with domain-specific architectural biases (e.g., anisotropic patching, factorized attention with learnable covariate/frequency biases) is then applied to generate in this representation. The method claims new state-of-the-art performance on unconditional generation across several benchmarks.

### Strengths
1. **Novel and Well-Motivated Paradigm**: The core idea of treating time series as videos via STFT is innovative and effectively bridges signal processing, time series analysis, and video generation. It addresses a clear limitation of prior image-based methods (collapsing the temporal axis) and time-domain models (struggling with spectral dynamics). The motivation is compelling and well-articulated in the introduction and related work.
2. **Strong Empirical Performance**: The paper provides extensive experiments on six diverse datasets. Quantitative results (Table 1, 2) show ST-Diff outperforms strong baselines (TimeGAN, TimeVAE, Diffusion-TS, ImagenTime) across most metrics (Discriminative, Predictive, Correlational, Context-FID) and sequence lengths, often by significant margins. The scalability to longer sequences (Table 2) is particularly convincing.
3. **Thoughtful Architecture and Inductive Biases**: The model design incorporates sensible domain knowledge. The anisotropic patching (aggregating frequency but not covariate dimensions), factorized tri-axial attention, and the initialization of covariate and frequency bias matrices from empirical data statistics (e.g., cross-correlation, STFT log-magnitude covariance) are well-justified inductive biases for the structure of the spectro-temporal video.
4. **Comprehensive Evaluation**: The evaluation goes beyond standard metrics to include qualitative analyses like t-SNE/KDE plots (Fig. 3, 5) and temporal/spectral fidelity checks via ACF and PSD (Fig. 4, 6, 7), providing a holistic view of sample quality. The use of a "Context-FID" score based on TS2Vec embeddings is also a robust choice.

### Weaknesses
1. **Computational Cost and Efficiency Acknowledged but Underexplored**: The paper concedes that video diffusion models incur higher computational/memory costs but does not quantify this overhead compared to baselines (e.g., FLOPs, training/inference time, memory footprint). For ICLR, a more detailed efficiency analysis or discussion of trade-offs is expected.
2. **Limited Exploration of STFT Hyperparameters and Their Impact**: The transformation from time series to video is critical, yet the choice of STFT parameters (window size, hop length) is described with a fixed formula (`nfft = (seq_len/2)-1`, `hop = ceil(nfft/4)`). There is no ablation study on how these choices affect generative performance, invertibility quality, or model sensitivity. This is a gap in understanding the method's robustness.
3. **Simplistic Trend Handling**: The trend component is isolated using a simple Exponential Moving Average (EMA) and stored in a separate video channel. While pragmatic, this approach may be insufficient for complex, non-stationary trends. The paper does not discuss alternatives (e.g., more sophisticated detrending, differentiable decompositions) or ablate the importance of this step.
4. **Evaluation on Moderately Long Sequences**: Although scalability to L=256 is tested, many real-world time series are much longer (e.g., thousands of steps). The performance and computational feasibility for very long sequences remain an open question. The fixed STFT parameter scaling might become problematic for extremely long `seq_len`.
5. **Missing Ablation Studies on Architectural Choices**: The contribution of key architectural components (anisotropic patching, factorized attention, learned bias matrices) is asserted but not rigorously validated through controlled ablations. This makes it difficult to attribute performance gains specifically to the proposed design versus the general video diffusion framework.

### Novelty & Significance
**Novelty**: The work is highly novel. It is the first to systematically propose a "time-series-as-video" paradigm for generation, combining STFT-based representation with a tailored video diffusion model. It differentiates itself from frequency-domain diffusion (Crabbé et al., 2024) by operating in the joint time-frequency plane and from image-based methods (ImagenTime) by preserving the explicit temporal axis.
**Significance**: The significance is potentially high. The approach establishes a new SOTA for unconditional generation, and the paradigm could influence other time series tasks (as suggested). The paper is clearly written, and the method appears reproducible given the details in Sec. 4, 5, and Appendix B. The main limitation for practical impact is the implied computational cost of video diffusion models.

### Suggestions for Improvement
1. **Include a Computational Efficiency Analysis**: Add a subsection or table comparing training time, inference time, and GPU memory usage of ST-Diff against key baselines (especially Diffusion-TS and ImagenTime). Discuss potential avenues for efficiency improvements (e.g., latent video diffusion, distillation) more concretely.
2. **Conduct Ablation Studies**: Systematically ablate the core components: (a) the necessity of the trend-residual decomposition, (b) the impact of anisotropic vs. isotropic patching, (c) the contribution of the learned covariate (`B_C`) and frequency (`B_F`) bias matrices, and (d) the factorized attention design. This would solidify the claims about architectural contributions.
3. **Analyze Sensitivity to STFT Parameters**: Perform an ablation or sensitivity analysis on the key STFT hyperparameters (window size, hop length, window type). Show how variations affect reconstruction error (iSTFT), and subsequently, generative performance metrics. This would provide practical guidance for applying the method to new datasets.
4. **Explore More Complex Trend Modeling**: Experiment with or discuss more advanced methods for handling non-stationarity (e.g., learnable linear projections, variational trend decomposition) to assess if the simple EMA is a bottleneck for certain data types.
5. **Extend Evaluation to Longer Sequences**: Test on at least one benchmark with very long sequences (e.g., L > 1000) to better understand the method's scalability and limitations. Report on any changes needed in STFT parameterization or model architecture to handle such lengths.
6. **Clarify the Video Tensor Structure**: The description of the video tensor `V` (Sec. 4.1) could be enhanced with a small, concrete numerical example to make the dimensional transformation (`L x K` -> `T x 3 x F x K`) absolutely clear to readers unfamiliar with STFT.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Full comparison against the key image-based baseline (ImagenTime).** The results table (Table 1) reports ImagenTime's discriminative score for only 2 out of 6 datasets, and omits its scores for Context-FID, Correlational, and Predictive metrics entirely. Without this complete comparison, the claim that ST-Diff "establishes a new state-of-the-art" over all prior methods, especially the most relevant image-based paradigm, is not substantiated.
2. **Ablation study on core components.** The paper lacks an ablation isolating the contribution of the trend-residual decomposition, the anisotropic patching strategy, the spectro-temporal attention biases (B_C and B_F), and the STFT cross-covariance loss. Without this, it is impossible to attribute performance gains to the novel "video" paradigm versus these specific architectural and training choices.
3. **Evaluation on conditional generation tasks.** The paper claims the paradigm has "significant potential to advance a broad spectrum of sequence modeling tasks beyond unconditional generation," but provides zero experiments on conditional tasks like forecasting, imputation, or anomaly detection. This critically undermines the claimed generality of the approach.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the STFT transformation's impact and limitations.** There is no discussion of how the choice of STFT parameters (window size, hop length) affects generation quality, nor an analysis of the inherent time-frequency uncertainty principle's impact on modeling very rapid or very slow dynamics. This is essential to trust the method's robustness across diverse time series.
2. **Quantification of computational cost and scalability.** The paper mentions higher computational cost but provides no comparison of training/inference time, memory footprint, or parameter count against time-domain (Diffusion-TS) or image-based (ImagenTime) baselines. For a video-based model, this is a critical practical consideration for assessing its contribution.
3. **Failure mode analysis.** The paper shows only successful qualitative results. An analysis of where the model fails (e.g., for specific frequency bands, on sharp transitions, or on covariates with particular characteristics) is necessary to understand the method's limitations and boundary conditions.

### Visualizations & Case Studies
1. **Side-by-side visual comparisons of raw time series.** The provided t-SNE and ACF/PSD plots are aggregate statistics. To truly judge fidelity, the paper needs direct visual comparisons of several randomly sampled real and generated multivariate sequences (line plots) for the most complex datasets (e.g., fMRI, Energy). This would immediately reveal if the model produces plausible temporal trajectories or unnatural artifacts.
2. **Visualization of generated spectro-temporal video tensors.** The core intermediate representation is never shown. Visualizing frames of the generated `V_gen` tensor (e.g., as spectrograms for a few covariates) would reveal whether the model learns coherent evolution of frequency content or generates unstructured noise in this space.

### Obvious Next Steps
1. **Conduct the full ablation study.** This is a standard requirement for a method paper at ICLR to validate that each proposed component is necessary and contributes to the final performance.
2. **Benchmark against frequency-domain diffusion models.** The related work cites "frequency diffusion models" (Crabbe et al., 2024). A direct comparison on the same benchmarks is essential to position the contribution of the joint *spectro-temporal* modeling versus pure frequency-domain generation.
3. **Test on much longer sequences to stress scalability.** The "long-term" experiment only goes to L=256. To convincingly argue the video paradigm overcomes limitations of time-domain models with long contexts, testing on sequences of length 1000+ is a logical and necessary step that was omitted.

# Final Consolidated Review
## Summary
This paper introduces Spectro-Temporal Diffusion (ST-Diff), a novel framework that reframes multivariate time series generation as a video generation task. It maps time series to a spectro-temporal video tensor using the Short-Time Fourier Transform (STFT) and employs a custom video diffusion model with domain-specific architectural biases. The method claims to establish a new state-of-the-art for unconditional generation across several standard benchmarks.

## Strengths
- **Novel and well-motivated paradigm:** The core idea of treating time series as videos via STFT is innovative, directly addressing limitations of prior image-based methods (which collapse the temporal axis) and time-domain models (which may struggle with spectral dynamics). The motivation is clearly articulated.
- **Strong empirical performance:** Extensive experiments on six diverse datasets show ST-Diff outperforms strong baselines (TimeGAN, TimeVAE, Diffusion-TS) across most established metrics (Discriminative, Predictive, Correlational, Context-FID) and demonstrates compelling scalability to longer sequence lengths (up to L=256).
- **Thoughtful, domain-informed architecture:** The model design incorporates sensible inductive biases, such as anisotropic patching (to avoid imposing spurious correlations among covariates), factorized tri-axial attention, and the initialization of covariate and frequency bias matrices from empirical data statistics.

## Weaknesses
- **Incomplete comparison with a key baseline:** Table 1 is missing critical metrics (Context-FID, Correlational, Predictive scores) for the image-based baseline ImagenTime. Since the paper claims a new state-of-the-art, this incomplete comparison significantly undermines the claim of superior performance over all prior methods.
- **Lack of ablation studies:** The paper does not include ablation experiments to validate the contribution of its core components, such as the trend-residual decomposition, the anisotropic patching strategy, the learned bias matrices (`B_C`, `B_F`), and the factorized attention design. Without these, it is impossible to attribute performance gains to the novel video paradigm versus these specific architectural choices.
- **Unreported auxiliary loss:** The method section (Section 4) does not describe the "cross-covariance loss applied directly to the STFT magnitudes" mentioned in the implementation details (Section 5). This omission affects reproducibility and the understanding of the final model's training objective.
- **Unquantified computational cost:** While the conclusion mentions higher computational and memory costs, the paper provides no quantitative comparison of training/inference time, memory footprint, or parameter count against time-domain or image-based baselines. This is a critical practical consideration for a video-based model.
- **Overreaching claims without evidence:** The abstract and conclusion claim the paradigm has "significant potential to advance a broad spectrum of sequence modeling tasks," but the paper provides zero experiments on conditional tasks (e.g., forecasting, imputation). This claim is not substantiated within the work presented.

## Nice-to-Haves
- A sensitivity analysis or justification for the chosen STFT hyperparameters (window size, hop length) and their impact on generative performance.
- Testing scalability on sequences longer than L=256 to more thoroughly stress the method's ability to handle very long contexts.

## Novel Insights
The paper's primary novel insight is the "time-series-as-video" paradigm itself: using the STFT to create an explicit, invertible spectro-temporal video representation that preserves the temporal evolution of frequency content. This representation enables the direct application of spatiotemporal (video) models to time series generation, a unifying perspective that differentiates the work from both pure time-domain and static image-based approaches.

## Suggestions
- Run and report the full suite of evaluation metrics for the ImagenTime baseline to provide a complete and fair comparison.
- Add an ablation study section to quantify the contribution of the proposed architectural components (trend removal, anisotropic patching, bias matrices, factorized attention).
- Integrate the description of the STFT cross-covariance loss into the method section (Section 4) for clarity and reproducibility.
- Include a brief quantitative analysis (e.g., a table) comparing the computational cost (training time, inference time, GPU memory) of ST-Diff against its main baselines.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 2.0, 4.0]
Average score: 3.5
Binary outcome: Reject
