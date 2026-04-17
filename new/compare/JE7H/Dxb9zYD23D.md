---
job_id: 4cb72ff6-76f3-47ca-b81b-81a6ca1b6d4c
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Dxb9zYD23D.pdf
paper: Time Series as Videos: Spectro-Temporal Generative Diffusion
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a generative diffusion model for multivariate time series using a spectro-temporal (STFT-based) video representation, clearly within ICLR’s scope of generative models, representation learning, and spatiotemporal modeling.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present, the work is technically nontrivial, and the empirical evaluation is substantial. While there are weaknesses (missing ablations, some under-specified losses, incomplete related work), they do not rise to “fundamental flaw / no evidence” level.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any instructions targeting automated reviewers or hidden prompt-like content within the paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces Spectro-Temporal Diffusion (ST-Diff), a framework for unconditional generation of multivariate time series that first maps sequences to a time–frequency “video” using STFT, then applies a custom video diffusion transformer, and finally inverts via iSTFT to obtain time-domain samples. Each video frame has frequency and covariate as spatial axes and uses three channels (real/imag STFT coefficients plus broadcasted trend). The model employs anisotropic patching and factorized attention with frequency and covariate bias matrices, and is evaluated on six standard datasets where it typically outperforms strong baselines such as Diffusion-TS and ImagenTime on multiple metrics.

## Strengths

1. **Conceptual reframing: time series as videos, preserving temporal axis.**  
   The central idea of treating multivariate time series as 3D videos in the joint time–frequency–covariate space is well argued and addresses a real gap between prior time-domain diffusion models and image-based approaches that collapse time. Section 4.1’s construction of \(V \in \mathbb{R}^{T \times 3 \times F \times K}\) is conceptually clean and leverages an invertible STFT, unlike many heuristic transforms.

2. **Architectural design aligning with the representation.**  
   The spectro-temporal transformer in Section 4.3 and **Figure 2b–2c** is thoughtfully designed to match the data structure:  
   - Anisotropic patching aggregates only along frequency, avoiding artificial “2D locality” across covariates.  
   - Factorized attention over temporal, frequency, and covariate axes separates different interaction types.  
   - Bias matrices \(B_C\) and \(B_F\) (Page 5–6) inject data-informed priors about cross-covariate and cross-frequency relationships.  
   This goes beyond a naive application of off-the-shelf video transformers, and the inductive biases are well motivated by properties of time series and spectra.

3. **Strong empirical performance on standard benchmarks.**  
   **Table 1** (Page 7) shows that ST-Diff achieves the best Context-FID and Discriminative scores on 5–6 datasets and often materially improves over Diffusion-TS, especially for high-dimensional real data (e.g., Energy and fMRI). The Discriminative score reductions on Energy (0.009 vs 0.122) and fMRI (0.021 vs 0.167) are particularly notable, as they suggest the GRU discriminator cannot reliably distinguish ST-Diff samples from real sequences. Similarly, on ETTh (L=24) ST-Diff improves Context-FID from 0.116 to 0.050 and Discriminative from 0.061 to 0.005.

4. **Scalability to longer sequences with empirical evidence.**  
   The long-sequence results in **Table 2** (Page 9) are a real plus: for ETTh with \(L=64,128,256\), ST-Diff dominates or matches all baselines on all four metrics, and particularly crushes Context-FID at \(L=64\) (0.031 vs 0.631 for Diffusion-TS) and maintains low Discriminative scores (~0.03) across all lengths. This supports the claim that explicitly modeling spectro-temporal structure scales better than pure time-domain approaches for longer horizons.

5. **Qualitative analysis that directly probes temporal and spectral fidelity.**  
   The qualitative results are thoughtful rather than superficial. **Figure 3** (Page 8) and **Figure 5** (Appendix C) show t-SNE and KDE overlays of real vs generated series across datasets and lengths, with good overlap rather than obvious mode collapse or spurious clusters. **Figure 4** (Page 8) plus **Figure 6–7** (Appendix C) compare ACF and PSD for real vs generated series per covariate. The near overlap of low-lag ACF and low-frequency PSD is evidence that the model is not just matching marginals but genuinely capturing temporal and spectral structure, which is exactly what the method is supposed to emphasize.

6. **Methodologically sound use of invertible signal-processing backbone.**  
   The reliance on STFT/iSTFT in Section 3 and 4.1, with 75% overlap and nfft/hop choices summarized in **Table 4** (Appendix B), shows attention to invertibility and resolution trade-offs. This avoids the “throw-away phase” style commonly seen in image-based embeddings and provides a principled bridge between classical DSP and modern generative modeling.

7. **Clear overall structure and reasonably good writing.**  
   The paper is mostly well organized. The high-level pipeline is easy to follow from **Figure 1** and the main text in Section 4, and the experiments are reasonably described. The conceptual motivation of why “video” is preferable to “image” for time series is clearly articulated in the Introduction and Related Work.

## Weaknesses

1. **Limited ablations on key design choices and inductive biases.**  
   The model introduces several nontrivial design elements, but there is essentially no ablation study to validate their importance:
   - Trend–residual decomposition using EMA (Section 4.1). How much does this matter versus directly STFT’ing the raw series? Does using a more sophisticated de-trending method change results?  
   - Anisotropic vs isotropic patching, and factorized vs standard joint spatiotemporal attention.  
   - The covariate and frequency bias matrices \(B_C\) and \(B_F\) (Page 5–6), which are a central architectural contribution, are not ablated. We do not know whether they improve performance beyond standard attention.  
   - The cross-covariance STFT magnitude loss introduced in Implementation Details (Page 7) is nonstandard and likely important, but there is no ablation comparing vanilla diffusion loss vs with this extra term.  
   Without at least some quantitative ablations (e.g., variants in a new **Table**), it is hard to attribute the observed gains to the “time series as video” idea rather than to training tweaks or larger/stronger backbones.

2. **Underspecified and non-mathematical treatment of the extra STFT covariance loss.**  
   On Page 7, the authors state that they “introduce a cross-covariance loss applied directly to the STFT magnitudes” to align normalized covariance matrices of generated vs real data. However:
   - There is no explicit formula for the loss, nor clear definition of which random variables it is computed over (per batch vs dataset, per frame vs aggregated over time). For example, something like  
     \[
     \mathcal{L}_{\text{cov}} = \left\|\frac{\Sigma_{\text{gen}}}{\|\Sigma_{\text{gen}}\|_F}
      - \frac{\Sigma_{\text{real}}}{\|\Sigma_{\text{real}}\|_F} \right\|_F
     \]
     is never actually written.  
   - It is not specified how this loss is combined with the standard DDPM MSE loss; is there a weight \(\lambda\)? Is it annealed?  
   - It is not clear whether this uses the same STFT representation as the model input or a different resolution.  
   Since this term directly targets spectral structure and covariances, it is likely important to the results, and the lack of precise mathematical specification undermines reproducibility and makes it impossible to reason about its optimization properties.

3. **No empirical comparison to frequency-domain or alternative spectro-temporal diffusion models.**  
   In Related Work (Page 2–3) they cite Crabbé et al. (2024)’s “time series diffusion in the frequency domain”, but there is no empirical comparison despite Crabbé being a very close baseline: both use Fourier-domain representations and diffusion for time series. At minimum, one would expect:
   - A clear justification of why Crabbé cannot be run on these benchmarks, if so.  
   - Or a quantitative comparison on at least a subset of datasets (e.g., Sines/ETTh).  
   Similarly, given the heavy reliance on video diffusion architectures, there is no comparison to a simpler off-the-shelf video diffusion backbone (e.g., a plain space-time U-Net or a generic video diffusion transformer) instantiated on their STFT “videos”. Without such comparisons, it is hard to judge how much of the gain comes from the conceptual reframing versus specialized architecture and training tricks.

4. **Metrics and results discussion gloss over notable failures / regressions.**  
   While **Table 1** mostly favors ST-Diff, there are cases where it is worse or only marginal:
   - On Sines, the Predictive Score for ST-Diff is 0.186 ± 0.004, roughly 2× worse than all baselines (0.093). This is surprising given Sines is the easiest dataset and is exactly where spectral modeling should shine. The text on Page 7–8 claims “ST-Diff establishes a new state of the art across the majority of datasets and metrics” but does not acknowledge or analyze this regression.  
   - On fMRI Correlational Score, ST-Diff (1.661 ± 0.059) is slightly worse than Diffusion-TS (1.411 ± 0.042), but this is again not discussed.  
   - ImagenTime contextual metrics (Context-FID, Correlational) are shown as “-” in Table 1, so it is impossible to see whether ST-Diff’s main claimed advantages over the image-transform-based competitor hold on those metrics.  
   The narrative is somewhat over-optimistic relative to the numbers; a more balanced discussion that highlights where ST-Diff struggles (e.g., very simple periodic signals or fine-grained correlations) would strengthen scientific credibility.

5. **Methodological clarity issues around the STFT-based video mapping.**  
   Although Section 4.1 gives a high-level description, several technical details are missing or ambiguous:
   - The STFT formula in Section 3 uses \(X[m,k] = \sum_{n=0}^{L-1} x[n] w[n-mH] e^{-j 2\pi k n / L}\). For typical STFTs with window length \(N\), the sum is over a windowed range (e.g., \(0 \leq n < N\)), not over \(L\), and the window is centered around \(mH\); this conflation of indices suggests either a typo or a misunderstanding. At least bounds like \(n = mH, \dots, mH + N - 1\) should be written.  
   - The trend channel \(\mathbf{x}_{k,\text{trend}}\) is “broadcasted across the frequency dimension and resampled to match \(T\)” (Page 4). The resampling scheme (nearest neighbor, linear interpolation, etc.) is not specified; since this is one of the three channels passed to the transformer, different choices could matter.  
   - The covariate width \(K\) serves as the spatial dimension of the video frames. Handling variable \(K\) remains unclear; all datasets have fixed covariates, but the method’s applicability to changing dimensionality (e.g., different sensors) is not discussed.  
   These are not fatal, but they detract from rigor and reproducibility and weaken the claim of “near-perfect reconstruction”.

6. **No computational or efficiency analysis despite heavy video-transformer architecture.**  
   The conclusion briefly acknowledges higher computational and memory costs (Page 9), but the empirical section omits any runtime, parameter count, or GPU-memory comparisons. Given that ST-Diff is compared to relatively light models like TimeGAN/TimeVAE and to Diffusion-TS (which operates directly in time), it is important to quantify the overhead of:  
   - Computing STFT/iSTFT for all sequences with nfft scaling as \((L/2)-1\).  
   - Running a multi-block transformer with tri-axial attention on tensors \(T \times F' \times K\).  
   Without such data, the practical trade-off between performance gains and resource costs is obscured, which matters for real-world adoption.

7. **Positioning w.r.t. broader diffusion and vision-for-time-series literature is incomplete.**  
   The Related Work section covers TimeGAN, TimeVAE, Diffusion-TS, ImagenTime, and CSDI, but omits several directly relevant recent works:
   - Surveys on diffusion for time series and spatio-temporal data (e.g., Yang et al. 2024; Panagiotakopoulos et al. 2025), which would help situate ST-Diff within a broader taxonomy of time-series diffusion techniques.  
   - TIMED (EskandariNasab et al. 2025), which is a unified diffusion-based framework for time series generation and refinement, clearly in the same problem class.  
   - Surveys on vision models for time series (Ni et al. 2025), directly relevant to this paper’s “treat sequences as visual objects” theme.  
   - Modern video diffusion architectures such as Lumiere (Bar-Tal et al. 2024) or acceleration methods like BlockDance (Zhang et al. 2025) that are natural references for the chosen transformer backbone.  
   The omission does not invalidate the work, but it suggests that the positioning is somewhat narrow and misses relevant context.

8. **No indication of robustness to STFT hyperparameters or windowing choices.**  
   **Table 4** lists nfft, hop length, and patch sizes for each sequence length, but the paper provides no exploration of sensitivity to these values. Given that STFT resolution trade-offs are central (time vs frequency resolution; window overlap for invertibility), it is important to know if the reported results are robust or heavily tuned. For instance, how does performance change if hop length is doubled, or if nfft is reduced? This affects the practicality of the method for diverse applications where optimal STFT settings may differ.

9. **Unclear training protocol for the empirical bias matrices \(B_C\) and \(B_F\).**  
   Page 5–6 states that \(B_C\) and \(B_F\) are “learnable priors” and “initialized from empirical statistics” (cross-correlation and log-magnitude covariance). However, the paper never clarifies whether these matrices:
   - Remain fixed after initialization,  
   - Are updated via gradient descent,  
   - Are regularized (e.g., towards the empirical value or towards 0), or  
   - Are normalized in any way (e.g., scaled to a certain variance).  
   Since these matrices directly shift attention logits, their scale relative to \(QK^T/\sqrt{d_k}\) matters critically; without specifying a normalization or training scheme, it is hard to understand or reproduce their effect.

## Potentially Missing Related Work

1. **Yang, Y., Jin, M., Wen, H. (2024): “A Survey on Diffusion Models for Time Series and Spatio-Temporal Data.”**  
   This survey systematically categorizes diffusion approaches for time series and spatio-temporal signals. It is directly relevant to Section 2 (“Generative Models for Time Series”) and would help contextualize ST-Diff among other architectures operating in time, frequency, and hybrid domains.

2. **Li, Y., Lu, X., Wang, Y. (2023): “Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement.”**  
   Proposes a generative diffusion-based framework for time series forecasting with disentangled representations. While focused on forecasting rather than unconditional generation, it is methodologically close and should be discussed in Section 2 as part of the broader family of time-series diffusion methods.

3. **Bar-Tal, O., Chefer, H., Tov, O. (2024): “Lumiere: A Space-Time Diffusion Model for Video Generation.”**  
   Introduces a space–time U-Net for video diffusion. Given that ST-Diff builds a custom spatiotemporal architecture for “videos” derived from time series, this work should be cited in Section 2 (“Time-Frequency Representations and Video Generation”) and potentially compared as an alternative backbone in Section 4.3.

4. **EskandariNasab, M., Hamdi, S. M., Boubrahimi, S. F. (2025): “TIMED: Adversarial and Autoregressive Refinement of Diffusion-Based Time Series Generation.”**  
   TIMED is a diffusion-based framework explicitly targeting time series generation, making it a highly relevant baseline and conceptual comparator. It should be added in the “Generative Models for Time Series” subsection, and ideally considered as an additional baseline in **Table 1** if feasible.

5. **Zhang, H., Gao, T., Shao, J. (2025): “BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers.”**  
   Proposes acceleration strategies for diffusion transformers in video generation. Given ST-Diff’s acknowledged computational cost (Conclusion, Page 9), this paper is relevant for future work on efficient spectro-temporal diffusion and should be cited around Section 4.3 or in the conclusion’s discussion of efficiency.

6. **Gu, J., Shen, Y., Chen, T. (2025): “STARFlow-V: End-to-End Video Generative Modeling with Normalizing Flow.”**  
   Presents an alternative generative paradigm (flow-based) for videos. While not diffusion-based, it is relevant to the “time-series-as-video” idea and could be cited in Section 2.3 as part of the broader landscape of video generative models that could, in principle, be applied to STFT video tensors.

7. **Zhang, S., Luo, B., Wang, H. (2024): “Temporal Action Detection in Videos with Generative Denoising Diffusion.”**  
   Uses diffusion models over video to solve a temporal task. This is relevant as another example of modeling temporal video dynamics with diffusion, and could be mentioned briefly in the “Video Diffusion Models” subsection to reinforce the connection between temporal modeling in videos and time-series tasks.

8. **Panagiotakopoulos, T., Kotsiantis, S., Gkillas, A. (2025): “Conditional Diffusion Models: A Survey of Techniques, Applications and Challenges.”**  
   Although focused on conditional diffusion, many techniques are relevant for the future extensions to forecasting and imputation discussed in the Conclusion. It would fit well in Related Work or as part of the future-work discussion on conditional variants of ST-Diff.

9. **Ni, J., Zhao, Z., Shen, C. (2025): “Harnessing Vision Models for Time Series Analysis: A Survey.”**  
   Directly aligned with the paper’s theme of using vision models for time series (ImagenTime-style methods and beyond). It should be cited in Section 2 (“Time Series to Image Transformations”) to better situate the “time-series-as-video” paradigm within the larger vision-for-time-series literature.

## Questions

1. **Role and implementation of the STFT covariance loss.**  
   - Could you provide a precise mathematical definition of the cross-covariance loss on STFT magnitudes, including the random variables and dimensions over which covariance is computed and how it is normalized?  
   - How is this loss weighted relative to the standard DDPM MSE term, and did you tune this weight per dataset?  
   - Can you share an ablation (e.g., added to **Table 1** or in a new table) showing performance with and without this loss on at least ETTh and fMRI?

2. **Effect of trend–residual decomposition and third channel.**  
   - How much performance do you lose if you remove the trend channel and operate on complex STFT of the raw signal only?  
   - Did you experiment with alternative trend extraction (e.g., low-pass filtering, polynomial smoothing)? Any quantitative or qualitative evidence here would clarify whether EMA is critical or merely convenient.

3. **Training behavior and role of bias matrices \(B_C\) and \(B_F\).**  
   - Are \(B_C\) and \(B_F\) frozen after initialization or updated via backpropagation? If trainable, do they drift far from their empirical initialization? Some statistics or visualizations would be informative.  
   - Have you run ablations where these matrices are removed (set to zero) or replaced by simpler learned biases (e.g., learned scalar per head)? How does that affect the results in **Table 1** or **Table 2**?

4. **Why is predictive performance on Sines worse than baselines?**  
   - Can you explain or analyze why ST-Diff underperforms all baselines on the Predictive Score for Sines (Table 1), despite the dataset being simple and periodic? Is this due to model capacity, overfitting, STFT resolution, or some artifact in the TSTR evaluation?  
   - Would higher nfft or different hop length help for Sines, and did you try it?

5. **Practical computational costs.**  
   - Could you report approximate wall-clock training time, number of parameters, and peak GPU memory for ST-Diff versus Diffusion-TS (and ImagenTime if available) on ETTh with \(L=128\) or \(256\)?  
   - Are there any specific bottlenecks (e.g., STFT pre-processing vs transformer passes) and do you foresee straightforward ways to reduce them?

6. **Robustness to STFT hyperparameters.**  
   - Have you examined how sensitive performance is to nfft and hop configuration beyond those listed in **Table 4**? For example, what happens if you halve or double nfft or change overlap from 75% to 50%?  
   - Is there any qualitative degradation of ACF/PSD alignment as you vary these parameters?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core diffusion formulation and STFT-based representation are standard and technically sound, and the empirical evaluation is solid. However, some crucial components (STFT covariance loss, attention biases) are under-specified and not ablated, preventing a top soundness score.

## Presentation Rating

3: good.  
The paper is generally well written, with clear figures (**Figures 1–4, 5–7**) and tables (**Tables 1–4**). Nonetheless, some mathematical details and implementation aspects are missing or ambiguous, and the related work section could be broader.

## Contribution Rating

3: good.  
The “time series as videos” paradigm with an STFT video tensor plus customized spectro-temporal transformer is a meaningful and nontrivial contribution, and the empirical gains on multiple benchmarks are convincing. The lack of ablations and incomplete positioning keep it from an “excellent” rating.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a well-motivated and mostly well-executed approach that significantly improves unconditional time-series generation on several benchmarks by leveraging a spectro-temporal video representation and a tailored video diffusion architecture. The idea is interesting and the results are strong, but important architectural components and auxiliary losses are under-specified and unablated, and some related work is missing. With clearer methodological details and a more thorough empirical dissection of what drives the gains, this would be a clear accept; as is, I lean positive but see room for substantial improvement.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion models, time-series generative modeling, and video transformers, and I carefully inspected the mathematical and experimental sections. Some implementation details are missing, but they do not affect my overall judgment of the work’s strengths and limitations.