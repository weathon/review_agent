---
job_id: af0f61b2-890b-4135-be4f-74e1b8d73b03
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 0cbUKCyBsH.pdf
paper: Influence-Aware Forecasting: Breaking the Self-Stimulation Barrier in Time Series
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is about time series forecasting with external textual influences, proposes a new benchmark and model, and includes theoretical analysis of forecasting error bounds. This is clearly within ICLR’s scope (representation learning, multimodal TSF, learning theory, datasets/benchmarks).

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work in Appendix C, Methodology, Experiments, Results, Conclusion) are present and written in English. The theoretical parts are nontrivial and mostly sound, and the empirical evaluation is extensive with many baselines. While there are issues to critique, there is no fatal flaw that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts or attempts to manipulate automated reviewing systems. The only mention of LLMs is in the ethics / usage statement and in the method description, not as instructions to the reviewer.

---

# Expected Review Outcome:

## Summary

The paper argues that the standard “self-stimulation” setup in time series forecasting (TSF), which predicts future values using only past observations, induces an intrinsic error floor because it treats unobserved external influences as noise. Using a control-theoretic formulation, the authors derive lower bounds on the forecast error covariance for self-stimulated models and show that incorporating any measurable influence can provably reduce this bound.  

They introduce the Influence-Aware Time Series Forecasting (IATSF) paradigm, construct a temporally-synced, leak-aware multimodal benchmark with textual influences, and propose FIATS, a lightweight, LLM-free model that uses channel-aware attention mechanisms (CASM and CAPS) to integrate text embeddings with numerical time series. Experiments on synthetic, physics-based, traffic, electricity, and game-usage datasets show consistent gains over strong self-stimulated baselines and several time-series foundation models.

---

## Strengths

1. **Clear, principled framing of the “self-stimulation” limitation with explicit error bounds.**  
   Sections 2 and B give a clean dynamical-systems view: the true system is \(X_f = F(X_h, U)\), while self-stimulated models learn \(f(X_h)\) and thus converge to the conditional expectation \(F^*(X_h) = \mathbb{E}_U[F(X_h,U)]\). Proposition 2.1 and the extended Proposition B.1 rigorously formalize the induced irreducible error covariance \(\mathrm{Cov}(\epsilon)\succeq \mathbb{E}_{X_h}[\nabla_U F \, \Sigma \, (\nabla_U F)^\top]\), with the linear case simplifying to \(B\Sigma B^\top\). The derivations in B.1.1–B.1.3 are detailed and correct under the stated independence assumptions, and the decomposition in Eq. (23) nicely separates model mismatch and irreducible noise.

2. **Conceptually simple but powerful message: “influence-aware” TSF as a paradigm.**  
   The formulation in Eq. (5) and Proposition 3.1 shows that adding any subset of influences \(U^j_t\) reduces the error covariance by \(\Delta \mathrm{Cov}(\epsilon) = \nabla_{U_j}F\,\Sigma_j\,(\nabla_{U_j}F)^\top\) (Eq. (6)). This makes an intuitive but rarely formalized point: external information is not just “nice-to-have” but mathematically necessary to break the error floor. This is conceptually important for the TSF community, which has indeed been stuck on closed-loop benchmarks.

3. **New benchmark with reasonably careful leak analysis and realistic use of textual influences.**  
   Section 4 and Appendix O describe the IATSF benchmark comprising:  
   - A frequency-modulated toy system where influences control frequency with a theoretical zero-error bound, and Electricity Utility with holiday/weekday text.  
   - Atmospheric Physics with weather reports every 6 hours, whose design explicitly separates raw weather time series from textual reports from a different source (O.4.1) to reduce leakage.  
   - GAUD, daily active users per game with developer logs and release/update events.  
   The explicit discussion of leak-free design in §4.1 and the error propagation analysis with imperfect influence forecasters in B.3 is more thoughtful than what I usually see in multimodal TSF papers.

4. **FIATS architecture is aligned with the theory and yields strong empirical results.**  
   FIATS is built on a PatchTST-like encoder plus a text encoder with CASM (channel-aware adaptive sensitivity modeling) and a CAPS influence-modulated decoder. Figure 2 clearly shows this structure: time-series patches and text embeddings enter separate encoders, CASM computes channel-wise sensitivities via cross-attention with channel descriptions as queries and news embeddings as keys/values, and CAPS performs influence-modulated decoding. This architecture directly reflects the theoretical structure \(X_f = CAZ_h + CBU_f\), where channel-specific sensitivity is \(c^i B\).  

   On the FM Toy dataset in **Table 1**, FIATS achieves MSE 0.003–0.027 across horizons 14–120, essentially matching the theoretical bound, while all self-stimulated models (including big foundation models like Chronos-L, MOIRAI-L, Time-MoE-U, and TimeLLM) remain two orders of magnitude worse. That is very compelling evidence that the bottleneck is missing influence information rather than model scale.

5. **Extensive experimental comparison across several regimes, including channel-wise and ablation analyses.**  
   - **Table 1** shows results on Toy, Electricity, NYC Traffic, and two Atmospheric Physics splits, with clear and consistent gains for FIATS. The improvements on NYC Traffic (e.g., MSE 0.443 vs 0.858 for PatchTST at horizon 96) and Atmospheric Physics 2014–19 (0.182 vs 0.252 for PatchTST at horizon 96) are substantial.  
   - **Table 2** and **Table 5** give channel-wise MSE on Atmospheric Physics, revealing particularly dramatic improvements on harder, low-frequency channels like pressure \(p\) and VPdef (e.g., p: 0.136 vs 0.930 for PatchTST, 83% reduction).  
   - **Figure 3** visualizes three representative channels: for \(p\), PatchTST is essentially flat while FIATS tracks the slow trend; for sparse “raining (s)” PatchTST collapses to near-zero while FIATS predicts events conditioned on text; for SWDR, FIATS aligns the waveform’s phase and amplitude better. These plots concretely back up the claims that (1) traditional TSF collapses to conditional expectations and (2) influence-aware modeling restores sharp structure.  
   - **Table 3** ablation on Atmospheric Physics shows the effect of different text embeddings and zeroing news/desc; “Zero News” degrades performance to self-stimulated levels (e.g., 0.249 vs 0.182 at horizon 96), and “Zero Desc.” also hurts substantially, supporting the importance of both influence and channel descriptions for CASM.  
   - The GAUD analysis, especially **Figure 4** and Tables 8–9, provides a nice real-world story: FIATS_pretrain is best on 53 of 89 games, with large gains particularly for games released after 2021 where history is short.

6. **Interpretability via attention visualizations.**  
   The CASM attention maps in **Figure 5** and **Figure 10** are used reasonably well: layer 1 attends mostly to the first “datetime” sentence for all channels, layer 2 shows strong attention on the pressure-related sentence for pressure and related channels (rho, CO2), and layer 3 diversifies attention per channel. The modality-mixer attention in **Figure 11** highlights historical rainfall windows for the rain channels and clear periodic structure for SWDR/PAR, consistent with the qualitative behavior in Figure 3. This gives some evidence that CASM and CAPS are not just black boxes but actually learning interpretable channel–text and temporal alignments.

7. **Nontrivial discussion of structural error sources beyond unobserved influences.**  
   Appendices B.4 and B.5 analyze error introduced by channel-wise weight sharing and partial observability. For example, Eq. (73) quantifies the error floor due to using a shared decoding weight \(c\) instead of channel-specific \(C_i\), and Eq. (89) extends the influence-bound to include hidden-state uncertainty. These are useful, if somewhat tangential, insights for TSF practitioners.

---

## Weaknesses

1. **Theoretical novelty is somewhat overstated; many results are reformulations of standard omitted-variable / conditional-expectation arguments with strong assumptions.**  
   - Proposition 2.1 and Proposition B.1 assume \(U \perp X_h\) throughout (e.g., in Eq. (11), Eq. (23), and §B.1.3), which is explicitly called out: they repeatedly require that \(\mathbb{E}[(U-\mu) X_h^\top] = 0\) to make cross-terms vanish. In real time series, exogenous drivers (weather, holidays, macro variables) are typically *correlated* with the past state (e.g., weather has persistent autocorrelation, holidays influence past demand). Under dependence, the optimizer need not converge to simple conditional expectations \(AX_h+B\mu\), and the lower bounds \(\succeq B\Sigma B^\top\) or Eq. (32) may not hold in the same form. The paper never discusses how the results change when \(U\) and \(X_h\) are dependent, which is a serious conceptual omission given the centrality of Proposition 2.1 to the “hard barrier” narrative.  
   - The derivations in B.1.2 and B.1.3, while algebraically correct under the assumptions, are essentially re-deriving the classic fact that the Bayes-optimal regression function is the conditional expectation and that irreducible variance equals conditional variance. The use of \(\nabla_U F\) in Eq. (3) and Eq. (30) is effectively a first-order delta method. This is fine, but the paper positions it as if this were a new “control-theoretic” discovery. It would be more honest to frame it as a clear, domain-tailored restatement of standard bias–variance / omitted-variable analysis rather than a fundamentally new theory.  
   - Proposition 3.1 similarly reuses the linear decomposition \(U_t = \sum_i U_i^t\) and independence of components \(U_i^t\), then states that including a known \(U_j^t\) reduces the error covariance by \(B_j \Sigma_j B_j^\top\). Again, this is intuitive (conditioning reduces variance), but the independence assumption between influence components is strong and not justified in realistic settings (e.g., weather factors are heavily correlated).

2. **Influence modeling and benchmark design hinge on a “perfect influence forecaster” assumption that is somewhat unrealistic and sidelined to the appendix.**  
   - Section 4.1 states: “Since system responses to influences often occur much faster than the sampling interval, we assume influences take effect *instantaneously* and denote the up-to-date influence as \(U_f\).” Then, “In deployment, ground-truth future influences are unavailable, so our benchmark restricts inputs to… predictions of \(U_f\) from expert sources.” However, in practice their experiments clearly feed in *true* future weather reports, holiday labels, and developer logs aligned with the forecast horizon.  
   - Appendix B.3 derives the error decomposition when \(\hat U_f = U_f + \epsilon_f\) from a non-optimizable influence forecaster, showing \(\mathrm{Cov}(\epsilon_\text{test}) = \Sigma_w + B \Sigma_{\hat U} B^\top\). Then B.3.2 explicitly assumes a “perfect influence forecaster” (\(\Sigma_{\hat U} \approx 0\)) “for fairness.” This may be fine for a theoretical benchmark, but then the *practical* claims in the abstract/introduction (“primary path forward for meaningful progress”) are overstated; what is being demonstrated empirically is the performance under access to accurate future influences, not under realistic forecasted or noisy influences.  
   - Figure 6 shows some robustness to “noise levels” on the influences on Atmospheric Physics, but this setup is only very briefly described and the noise is artificially injected, not from real forecast errors. Important questions like: How quickly do gains disappear as weather forecasts become less accurate? Are there regimes where influence errors dominate and self-stimulated models are preferable? remain largely unanswered.

3. **The FIATS architecture, while principled, is not fully specified mathematically, and key design choices are under-justified.**  
   - Section 5 describes CASM in prose, but the core attention operation is only verbally mentioned: “Query as Channel-wise Sensitivity \(\tilde C = Desc \cdot W_Q\)… Key as influence Filter \(\widehat{B}_{U_f} = (News \cdot W_K)^\top\)… Value as influence Translator \(\tilde U_f = News \cdot W_V\).” The actual attention computation \(U_f^e = \mathrm{softmax}(QK^\top / \sqrt{d})V\) appears only as a vague reference in the caption of Figure 2 and once in the text (“Attention(Q = U_t^c, K, V = \hat Z)”). There is no explicit equation for the CASM block’s output, its dimensionalities, normalization, or how multiple blocks in residual are composed. For a method whose central selling point is “control-theoretic alignment,” the lack of a precise mathematical definition of CASM/CAPS is disappointing and complicates reproducibility.  
   - For CAPS, they mention “a channel-conditioned decoder… through cross-attention \(Attention(Q=U_t^c, K, V=\hat Z)\) to simulate such nonlinear projection. To avoid future information leakage, we apply causal attention mask here. We will omit the analysis.” Omitting both the mathematical details and the error analysis for CAPS weakens the claimed theoretical alignment between architecture and propositions.  
   - The weight-sharing analysis in B.4 is interesting, but the link from Eq. (73) to the concrete CAPS implementation is never quantitatively instantiated (e.g., how many parameters per channel, how much of \(\text{Cov}(\epsilon)\) is explained in practice). In effect, the theory suggests that full per-channel decoders are ideal, but FIATS implements a specific low-rank adapter via attention without any clear connection to the bounds.

4. **Comparisons to other text-informed TSF methods are incomplete, and some closely related recent work is missing.**  
   - In the main experiments, the only text-aware baseline is TimeLLM (Table 1). There is no empirical comparison to GPT4MTS (Jia et al., 2024), XForecast (Aksu et al., 2024), Time-MMD’s own baselines (Liu et al., 2024a), or other recent LLM- or text-based TSF models like LangTime (Niu et al., 2025) that are cited. This makes it hard to disentangle how much of FIATS’s gains are due to its paradigm vs. simply having any text in a well-engineered way.  
   - Beyond the cited works, there are several directly relevant recent papers not referenced at all (see “Potentially Missing Related Work” below), including methods that integrate exogenous variables and external text information with attention architectures. Given that a core selling point is “first principled influence-aware TSF,” the lack of discussion and comparison to these models weakens the novelty and positioning arguments.  
   - On datasets like Electricity and GAUD, it would be very informative to see a “text-agnostic FIATS” (i.e., same architecture but with News zeroed) compared to TimeLLM, GPT4MTS, XForecast etc., to disentangle the effect of architecture vs. influence-aware training objective.

5. **Benchmark realism and generalization claims are not fully convincing, in particular for GAUD and Time-MMD.**  
   - GAUD is an interesting dataset, but the evaluation uses de-normalized MAE where games differ wildly in scale, and **Tables 8–9** then average “IMP%” in a way that mixes tiny MAEs (tens) and huge ones (tens of thousands). This makes interpreting the overall improvement (12.6%) tricky: a few large games (e.g., ids 1085660, 1172470) can dominate the aggregate signal. There is no clear breakdown by game scale or by cold-start severity, despite repeated claims that FIATS is particularly helpful for games released after 2021.  
   - For Time-MMD, Appendix N appropriately criticizes the dataset’s flaws (placeholder text, information leakage). However, **Table 11** still reports FIATS’s “>50%” relative improvements over Time-MMD baselines in some domains, and this is mentioned as evidence of robustness. Given the severe leakage they document (e.g., text containing the exact target value), these numbers are not very meaningful scientifically and could be misinterpreted by readers. I would recommend de-emphasizing Time-MMD performance entirely or at least labeling it clearly as “illustrative only, not a trustworthy benchmark result.”

6. **Scalability and computational cost are not quantified despite claims of lightweight design.**  
   The abstract and introduction stress that FIATS is “lightweight” and avoids LLM overhead. However, there is no table or discussion of parameter counts, GPU hours, or inference latency compared to baselines like PatchTST, Chronos-L, or TimeLLM. CASM computes attention between \(C\) channels and \(M\) textual sentences (e.g., 21 × 7 for Atmospheric Physics), and CAPS attends over patch embeddings; this may be cheap on the presented datasets, but it is unclear how FIATS scales to hundreds of channels and long texts, or to high-frequency data with many patches. Without even rough complexity analysis, the “lightweight and practical” claim remains somewhat anecdotal.

7. **Some notation and exposition issues, especially around the task formulation and influence variables, reduce clarity.**  
   - The paper flips between \(U_t\), \(U_f\), and “News”/“influences” throughout. Eq. (1) uses \(U_t\), Eq. (5) also uses \(U_t\) in the IATSF objective, §4.1 suddenly introduces \(U_f\) as “up-to-date influence for the future segment,” and CASM/CAPS predominantly refer to \(U_f^e, U_f^c\). It is not completely clear whether in all experiments the model receives only influences within the look-back window, or also some forecast of influences spanning the prediction horizon. This matters for leak-freeness and for aligning the theory (which is single-step) with the multi-step horizon setting.  
   - In Table 4, both “\(X_f\)” and “\(\hat X_f\)” are labeled just “Xf,” which is confusing. Also, in Section 5, the notation “\(U_t^c\)” suddenly appears in the decoder attention without prior definition.  
   - The introduction of the benchmark is split between §4, Appendix O, and scattered comments (e.g., on Electrical Utility timestamps being mis-dated in O.3). For a reader who wants to reuse the benchmark, having a single concise, precise section in the main text with the key design decisions and statistics (Table 12 helps, but it arrives very late) would be beneficial.

---

## Potentially Missing Related Work

Below are directly related works that are not cited in the paper and should be integrated into the discussion:

1. **Huang et al., “Exploiting Language Power for Time Series Forecasting with Exogenous Variables,” 2025.**  
   This work explicitly studies using language models to represent exogenous variables and integrate them into TSF. It is very close in spirit to IATSF’s “language as influence modality” (§3.2). It should be cited in the related work on text-informed forecasting (Appendix C.2) and briefly compared in Section 1–2 when motivating influence-aware TSF and critiquing existing LLM-based multimodal approaches.

2. **Bi et al., “Spatiotemporal Learning With Decoupled Causal Attention for Multivariate Time Series,” 2025.**  
   They introduce decoupled causal attention to model inter-variable relations for multivariate TSF. This is related to FIATS’s CASM/CAPS mechanisms for channel-wise sensitivity and channel-aware decoding. It should be discussed in §5 and Appendix B.4 when talking about channel heterogeneity and causal attention, and ideally a baseline or at least conceptual comparison should be added.

3. **Ye et al., “Non-stationary Diffusion For Probabilistic Time Series Forecasting,” 2025.**  
   This paper proposes a diffusion-based framework to handle non-stationarity in TSF, particularly relevant when external influences induce regime changes. While it does not explicitly use text, it is directly relevant to the “barrier due to unmodeled influences” discussion in §2 and B.5. The authors should mention it in the introduction/related work as a complementary approach for non-stationary dynamics and justify how IATSF differs (explicitly modeling influences vs. modeling non-stationarity implicitly).

4. **Peng et al., “MSP-EDA: Multivariate Time Series Forecasting Based on Multiscale Patches and External Data Augmentation,” 2025.**  
   This method integrates multiscale patches with external data augmentation for TSF, clearly aligned with the idea of injecting external influences. It should be discussed in §2.2 and Appendix C.2 as a related approach for using external variables, and if feasible, MSP-EDA could be included as a baseline on at least one dataset where external data is available.

5. **Haselbeck & Grimm, “EVARS-GPR: EVent-Triggered Augmented Refitting of Gaussian Process Regression for Seasonal Data,” 2021.**  
   This work integrates external events to refit GP models to handle sudden shifts in seasonal data, similar in motivation to IATSF’s argument around sharp transitions and event-driven dynamics (Fig. 1, FM Toy). It would be useful to cite this in the motivation (Section 2.1–2.2) as prior work on event-triggered forecasting, and in the discussion of toy systems and Electricity Utility.

---

## Questions

1. **Dependence between \(U_t\) and \(X_h\).**  
   The core theoretical results assume \(U \perp X_h\). In realistic systems (e.g., weather variables and past state, economic indicators and past prices) this independence rarely holds. Can the authors clarify how the main claims (Proposition 2.1 and Proposition B.1) change when \(U\) and \(X_h\) are dependent? Is there a more general bound that holds under arbitrary joint distributions, perhaps expressed in terms of conditional covariance \(\mathrm{Cov}(U|X_h)\)?

2. **Influence availability and forecast errors.**  
   In the real-world datasets (Atmospheric Physics, NYC Traffic, Electricity, GAUD), are the influences provided at test time the true future influences, or forecasted ones? For weather, do you use actual ex-post weather reports for the forecast horizon or weather forecast products? Please be explicit in Section 4.2. Also, could you extend Figure 6 or add a table showing performance as a function of *realistic* forecast quality (e.g., using historical ECMWF forecasts, adding biases, or simulating mis-timed events) to demonstrate how robust FIATS is when influences are imperfect?

3. **Exact CASM and CAPS formulations.**  
   Can you provide explicit mathematical definitions of the CASM and CAPS blocks, including all projection matrices, shapes, and attention equations? For example, an equation like  
   \[
   U_f^{e} = \mathrm{softmax}\left(\frac{(Desc\,W_Q)(News\,W_K)^\top}{\sqrt{d_k}}\right)(News\,W_V)
   \]
   and the precise form of the residual stacking and layer norms. Similarly, for CAPS, how exactly are channel-conditioned queries derived and how is causal masking applied across patches?

4. **Scalability and computational cost.**  
   What are the parameter counts and training/inference time per epoch for FIATS vs. PatchTST, FITS, and TimeLLM on, say, the Atmospheric Physics 2014–19 dataset? How does memory/time scale with number of channels \(C\) and number of textual sentences \(M\) in CASM and CAPS? A small complexity analysis or benchmark would strengthen the “lightweight baseline” claim.

5. **Ablation on architecture vs. text usage.**  
   To better isolate the benefits of IATSF-style influence-aware learning from architectural choices, could you add an ablation where FIATS is trained *without* textual inputs but with the same encoder–decoder (e.g., “Zero News” in Table 3, but reported across all datasets and compared to PatchTST and FITS)? This would help assess whether part of the gains comes from CASM/CAPS acting as generic channel adapters even without text.

6. **Cold-start analysis on GAUD.**  
   The paper claims the gains are “most pronounced for games released after 2021” and that FIATS helps in cold-start scenarios. Can you provide a quantitative breakdown: for example, average improvement for games with history length < X days vs. longer histories, and perhaps a figure that zooms into the right-hand region of Figure 4 with confidence intervals?

Author responses addressing these points, particularly 1–3, could significantly increase my confidence in the generality and reproducibility of the work.

---

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The datasets are primarily physical measurements and anonymized game activity counts, and the authors discuss licensing and LLM usage for preprocessing. I do not see clear fairness, privacy, or safety red flags.

---

## Soundness Rating

3: good.  
The mathematical analysis is careful and correct under stated (though strong) assumptions, and the empirical evaluation is thorough with many baselines and ablations. However, the independence assumptions and perfect-influence setup limit the generality of the theoretical claims, and some architectural details (CASM/CAPS) are under-specified.

---

## Presentation Rating

3: good.  
The paper is generally well written, with helpful figures like **Figure 1** (conceptual error-floor illustration) and **Figure 2** (architecture), and extensive visualizations (Figures 3, 5, 10, 11, 14). There are some notation inconsistencies and important details (e.g., exact CASM equations, influence availability) are relegated to the appendix or not fully defined, which slightly hurts clarity.

---

## Contribution Rating

3: good.  
The contribution is a solid combination of: (1) a clear theoretical reframing of self-stimulated TSF under a dynamical-systems lens; (2) a new, leak-aware multimodal benchmark; and (3) a well-designed, interpretable architecture that achieves strong empirical gains over strong TSF baselines and foundation models. The theoretical novelty is moderate rather than deep, and comparisons to other text-informed TSF methods could be stronger, but overall the work is relevant and valuable to the TSF community.

---

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper makes a compelling case, both theoretically and empirically, that explicitly incorporating external influences (especially text) can break the performance plateau of self-stimulated TSF, and it backs this with a new benchmark and a principled architecture. At the same time, the strongest theoretical statements rely on restrictive assumptions, and some aspects of the method and evaluation (CASM/CAPS specification, influence forecasting realism, related work coverage) need tightening. With revisions addressing these points, this would be a strong ICLR contribution; as it stands, I lean positive but not enthusiastically.

---

## Reviewer Confidence

4: confident.  
I am familiar with time series forecasting, control/dynamical systems, and multimodal architectures, and I carefully checked the main derivations and experimental tables/figures. Some aspects (e.g., exact preprocessing pipelines for GAUD, detailed implementation of CASM/CAPS) remain somewhat opaque, but overall I am confident in this assessment.