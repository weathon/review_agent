---
job_id: f881283a-e1af-4b9c-907d-0a789998096c
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 3icvqeC1sA.pdf
paper: ChaosNexus: A Foundation Model for Universal Chaotic System Forecasting with Multi-Scale Representations
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a large neural foundation model (Transformer + MoE) for forecasting chaotic dynamical systems, with strong emphasis on representation learning, scaling laws, and applications to physical sciences and time series; this is well within ICLR’s scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present and reasonably complete. The work is technically detailed, the math is mostly consistent, and there is substantial experimental evidence. No obvious fatal methodological or statistical flaw is apparent from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts or attempts to influence automated reviewing beyond standard ethics and reproducibility statements.

---

# Expected Review Outcome:

## Summary

The paper introduces ChaosNexus, a pretrained “foundation” model for chaotic system forecasting. The core architecture, ScaleFormer, is a U‑Net‐like multi‑scale Transformer with dual axial attention, per‑block Mixture‑of‑Experts (MoE) layers, and a wavelet scattering–based “frequency fingerprint”; training uses an MSE objective augmented with MoE load balancing and an MMD distributional regularizer to better match attractor statistics. The model is pretrained on the Panda chaotic‑ODE corpus (~20K synthetic systems) and evaluated zero‑shot on 9.3K unseen chaotic ODEs, on a 5‑day global weather benchmark (WEATHER‑5K) in zero‑ and few‑shot regimes, and in scaling studies, showing strong attractor‑level metrics and competitive or better pointwise forecasts compared to Panda and other time‑series foundation models.

## Strengths

1. **Well‑designed multi‑scale architecture grounded in the problem**  
   The ScaleFormer design (Section 3.2, **Figure 1a–b**) is very thoughtfully aligned with multi‑scale chaotic dynamics. The encoder–decoder with patch merging/expansion along the temporal axis provides explicit coarse‑to‑fine hierarchies, while dual axial attention cleanly factors variable vs temporal interactions. The use of RoPE and RMSNorm is up to date. The ablation in **Table 1** shows that removing patch merging/expansion severely worsens both sMAPE@128 (+7.8%) and \(D_{\mathrm{frac}}\) (+21.7%), strongly supporting the claim that hierarchical temporal structure is not just cosmetic.

2. **System‑aware specialization through MoE and frequency fingerprints**  
   The MoE layers (Eq. (1)–(4)) and the wavelet scattering fingerprint (Section 3.3, Appendix C.3) together provide a compelling mechanism for cross‑system generalization. The MoE routing visualizations in **Figure 6** and **Figure 10**, plus the pruning experiment in **Table 4**, are convincing: systems derived from the same foundation ODE trigger similar expert patterns, and pruning a system’s top‑activated experts reliably degrades both sMAPE and attractor metrics. Likewise, **Table 7** shows that the wavelet scattering fingerprint clearly outperforms STFT and “learnable” Fourier filters on both sMAPE and attractor statistics, which nicely supports the choice of a fixed, mathematically grounded spectral representation.

3. **Strong and fairly comprehensive empirical evaluation on chaotic benchmarks**  
   The synthetic chaotic benchmark is large‑scale and rigorous. **Figure 2** (with numerical details in **Table 2**) shows that ChaosNexus has competitive point‑wise sMAPE@128 vs Panda (slightly better mean, similar dispersion) while substantially improving attractor‑based metrics, especially KL divergence between attractors \(D_{\text{step}}\) and the correlation‑dimension error. The authors also report Lyapunov‑exponent error and weighted spectral energy error (Table 1, Table 2, Appendix D.2), which is exactly the sort of evaluation one wants for chaotic systems; this is far more thorough than “just sMAPE”.

4. **Compelling zero‑shot and few‑shot results on real‑world weather data**  
   On WEATHER‑5K, **Figure 3** and the expanded plots in **Figures 19–23** and **24–28** show that ChaosNexus attains sub‑\(1^\circ\)C zero‑shot 5‑day temperature MAE and remains clearly better than strong system‑specific baselines (FEDformer, CrossFormer, PatchTST, Koopa) even when they are fine‑tuned on 0.1–0.5% of the training set. The latitude‑stratified analysis in **Figures 29–31** further demonstrates that this zero‑shot <1°C MAE holds across low, mid, and high latitudes, which is an unusually strong result for a model that has never seen the weather dataset during pretraining.

5. **Thoughtful scaling‑law analysis with clear take‑home message for foundation models**  
   The scaling plots in **Figure 4** are well executed and interpretable: (a) increasing parameters improves performance in a smooth, diminishing‑returns way; (b) simply giving more trajectories per system has marginal benefit; and (c) increasing the number of distinct systems yields strong gains across horizons. The comparison between **Figure 4b** and **4c** is particularly informative and provides a concrete design guideline: for this domain, corpus diversity (number of systems) is far more valuable than depth (trajectories per system).

6. **Good level of interpretability and analysis of internal mechanisms**  
   The multi‑scale attention visualizations in **Figure 5** and **Figure 8** are well argued: shallow encoder layers show Toeplitz‑like or block patterns depending on system regularity, while deep layers have globalized patterns; shallow decoder layers act as “future‑pattern selectors” and deep decoder layers focus on recent patches. The gating‑entropy study in **Figure 11** (high entropy in shallow layers, low at bottleneck, rising again in shallow decoder) and the MMD sensitivity experiments in **Table 5–6** and **Figure 12** add further insight that the architecture is doing something nontrivial rather than just memorizing.

7. **Reproducibility and detail**  
   The implementation is thoroughly specified: training protocol, hyperparameters (**Table 8**), dataset details (Appendix D and F), exact evaluation metrics (Appendix D.2) and even inference latency (**Table 3**) are documented. The open‑source code link is provided, and the description of Koopman‑style input features, wavelet scattering transform, and MMD implementation is meticulous.

## Weaknesses

1. **Conceptual novelty is more architectural composition than fundamentally new theory**  
   The main components of ChaosNexus are all established ideas: U‑Net‑style multi‑scale architecture, axial attention, sparse MoE, wavelet scattering fingerprints, and MMD regularization to match invariant measures. The contribution is in their integration and careful tuning for chaotic systems, rather than a significantly new representation principle. For instance, the multi‑scale encoder–decoder in **Figure 1b** is very close in spirit to multi‑scale CNN/Transformer time‑series models and multi‑scale precipitation‑nowcasting architectures, and the wavelet scattering module essentially plugs in standard scattering coefficients (Eq. (14)–(second‑order) in Appendix C.3) as side‑information. The paper would benefit from a sharper theoretical or algorithmic story about *why this particular combination* is fundamentally better suited to chaotic dynamics than, say, existing multi‑scale or Koopman‑inspired models beyond what is empirically shown.

2. **Some experimental comparisons are not fully apples‑to‑apples, especially on WEATHER‑5K**  
   In Section 4.2, ChaosNexus is pretrained on a massive synthetic chaotic corpus, then evaluated zero‑shot or few‑shot on WEATHER‑5K, whereas the main baselines (FEDformer, CrossFormer, PatchTST, Koopa in **Figure 3**) are trained *from scratch* on 0.1–0.5% of WEATHER‑5K and do not leverage any pretraining. This strongly favors ChaosNexus and conflates architecture with pretraining. There is a partial remedy in the appendices where pretrained chaotic‑domain foundation models (Panda, Chronos‑S‑SFT) are compared in **Figures 24–28** and **Table 9**, but that comparison is not highlighted in the main text. A fairer and clearer story would either (a) fine‑tune Panda and Chronos‑S‑SFT on the exact same WEATHER‑5K subsets and present those results alongside **Figure 3** in the main paper, or (b) explicitly state that WEATHER‑5K is primarily a transfer‑learning demonstration, not a strict architectural comparison to system‑specific models.

3. **MoE gating formulation and load balancing are under‑specified mathematically**  
   Equations (2)–(4) are somewhat ambiguous and deserve more precise treatment. In Eq. (3), \(\phi_{i,p}\) is set to \(s_{i,p}\) if \(s_{i,p}\) is in the TopK scores, zero otherwise, where \(s_{:,p} = \mathrm{Softmax}(W \bar{h}_p)\). However:
   - It is unclear whether the TopK scores are re‑normalized so that \(\sum_{i=1}^M \phi_{i,p} \le 1\) or whether the softmax probabilities for non‑selected experts are literally set to zero while keeping the others unchanged. This affects gradient behavior and the interpretation of the load‑balancing loss in Eq. (9).
   - The shared expert’s coefficient \(\phi_{M+1,p} = \sigma(W_{M+1}\bar{h}_p)\) is independent of \(s_{i,p}\); there is no guarantee or discussion that \(\phi_{M+1,p}\) is on a comparable scale to \(\phi_{i,p}\). In practice, this could lead to the shared expert dominating or being negligible, depending on initialization, and it is not clear how this interacts with the balancing loss \(M\sum_i f_i r_i\).
   - There is no explicit formula for \(f_i\) and \(r_i\) in terms of \(\phi_{i,p}\); the reader has to reverse‑engineer from Dai et al. (2024). Given that MoE behavior and the balancing loss are central to the paper, this is a nontrivial omission. A short derivation or at least explicit definitions would materially improve clarity.

4. **MMD regularization objective is heavy and somewhat opaque in practice**  
   The MMD loss in Eq. (10) is stated over batches of full trajectories with a mixture of rational quadratic kernels (Eq. (18) in Appendix C.4). However:
   - As written, \(\kappa(\hat{X}^i, \hat{X}^j)\) is defined on entire trajectories, but it is not clear what vectorization is used: is each trajectory flattened over time and variables, or are states subsampled? This matters because in the naive case, the cost of the Gram matrix is \(O(B^2 T^2 V^2)\). The paper claims low computational complexity for MMD as an IPM (Appendix C.4), but does not state how they control this in practice for \(T = 4096\) training trajectories.
   - The kernel scale set \(\sigma = \{0.2, 0.5, 0.9, 1.3\}\) is borrowed from prior work, but there is no justification that these scales are appropriate for the very heterogeneous distribution of chaotic systems considered here. The sensitivity study in **Table 6** is useful but quite coarse; it changes the kernel *family*, not the scale hyperparameters themselves.
   - Eq. (10) uses the biased‑style estimator with \(1/B^2\sum_{i,j}\), while Appendix C.4 labels Eq. (17) “unbiased empirical estimator” but uses the same style; for small batch sizes, the difference can be important. Clarifying which estimator is used and why would strengthen the theoretical consistency.
   While these are not fatal issues, they make it harder to reason about how much of the performance gain in **Table 1** (“w/o MMD”) is robust versus implementation‑specific.

5. **Multi‑scale readout discards temporal structure rather aggressively**  
   Section 3.3 states that each decoder output \(\bm{H}_{\text{dec}}^{(i)}\) is temporally mean‑pooled to produce \(\hat{\bm{H}}^{(i)}\), then concatenated and fused linearly to form \(\bm{H}_{\text{uni}}\). This is quite a crude aggregation: all phase information and temporal ordering within each scale are discarded before prediction, yet no justification or ablation is provided. Given the sensitivity of chaotic systems to phase and the emphasis on multi‑scale temporal structure, it is somewhat surprising that such a simple pooling suffices. An ablation contrasting mean pooling vs learned attention pooling or convolutional readouts across scales would be very informative and could reveal whether the model is leaving performance on the table, especially for longer horizons. Right now, the effectiveness of this joint‑scale readout rests only on aggregate metrics in **Table 1**, where other components are changed simultaneously.

6. **Zero‑shot synthetic results are more nuanced than the narrative suggests**  
   In **Figure 2** and **Table 2**, ChaosNexus clearly outperforms Panda and other foundation models on \(D_{\text{step}}\), weighted spectral error, and usually on \(D_{\mathrm{frac}}\). However, for sMAPE@512 Panda actually has a better mean (102.333 vs 108.293), and even for sMAPE@128 the gains are marginal (68.901 vs 69.567, well within the confidence intervals). The text on Page 7 is slightly one‑sided, emphasizing “competitive point‑wise accuracy” and “superior fidelity” in long‑term statistics; strictly speaking this is correct, but the strong improvement is on attractor metrics rather than pointwise forecasting. It would be more honest and informative to explicitly acknowledge that ChaosNexus trades a small amount of long‑horizon sMAPE for significantly better invariant‑measure fidelity.

7. **Weather evaluation omits comparison to specialized weather ML models and NWP**  
   While I do not expect full comparison to operational weather centers in an ICLR paper, the claim in the abstract that ChaosNexus achieves “competitive zero-shot mean error below 1°C in 5-day global weather forecasting” is somewhat unanchored. There is no comparison to modern ML weather models (e.g., graph‑based or transformer‑based global forecasters) or even to very simple climatology / persistence baselines on WEATHER‑5K. The current set of baselines in **Figure 3** are generic forecasting models, not weather models. At minimum, including a persistence baseline and commenting on how <1°C MAE compares to standard global weather evaluation practices would temper or contextualize the “competitive” claim.

8. **Notation and metric naming are at times confusing or inconsistent**  
   There are several small but cumulatively distracting inconsistencies:
   - **Table 2** labels the main forecasting metric as “eMAPE@128” and “eMAPE@512” whereas the text consistently uses sMAPE. There are also duplicated metric labels \(D_{exp}\) and ambiguous abbreviations \(D_{mn}\), \(\Delta f_{L_{km}}\) that do not match the definitions in Appendix D.2 (\(D_{\mathrm{frac}}, D_{\mathrm{step}}, \text{ME}_{\mathrm{LRw}}, D_{\mathrm{Lyap}}\)).
   - In **Table 1**, “Davg” is introduced without explanation in the main text; readers have to infer that it probably corresponds to \(D_{\mathrm{step}}\).
   - The MMD metric is sometimes referred to as \(D_{step}\), sometimes \(D_{stsp}\), etc.  
   None of these are critical errors, but they slow down comprehension and make cross‑referencing figure/table claims more laborious than necessary.

9. **Limited positioning vs broader multi‑scale time‑series and chaotic‑forecasting literature**  
   The related work section focuses heavily on reservoir computing and recent foundation models (Panda, TimeMoE, etc.), but omits several directly relevant works that also use multi‑scale architectures or foundation models for complex/chaotic systems (see “Potentially Missing Related Work” below). Citing and briefly discussing such work would help clarify what is genuinely new in ScaleFormer vs what is an adaptation of existing multi‑scale designs to this particular synthetic corpus.

## Potentially Missing Related Work

1. **Wang & Qin, “A TCN‑Linear Hybrid Model for Chaotic Time Series Forecasting”, 2024**  
   This paper introduces a hybrid architecture combining temporal convolutional networks (TCNs) with linear components aimed specifically at improving long‑term forecasting of chaotic time series. It is directly relevant to the stated goal of modeling long‑horizon chaotic dynamics and should be discussed in Section 2 as an example of non‑Transformer architectures tailored to chaotic systems, and ideally included as a baseline (at least on a subset of synthetic systems) to contextualize the benefits of the ScaleFormer backbone vs TCN‑style designs.

2. **Tan et al., “Deep learning model based on multi‑scale feature fusion for precipitation nowcasting”, 2024**  
   This work uses multi‑scale feature fusion for precipitation forecasting, a chaotic and multi‑scale physical process. It is conceptually similar to the U‑Net‑like multi‑scale design in ScaleFormer. A brief comparison in Section 2 or Section 3.2 would help differentiate what is unique about the proposed temporal patch merging/expansion and skip connections vs generic multi‑scale fusion in spatiotemporal forecasting.

3. **Jafari et al., “Time Series Foundation Models and Deep Learning Architectures for Earthquake Temporal and Spatial Nowcasting”, 2024**  
   This paper surveys and evaluates time‑series foundation models for earthquake nowcasting, another complex, partially chaotic system. It seems directly relevant to the framing of ChaosNexus as a scientific foundation model for chaotic phenomena. It should be cited in Section 2 when discussing foundation models for non‑PDE physical systems and could be mentioned in the discussion as an example of domain‑specific foundation modeling parallel to ChaosNexus.

4. **Perez‑Diaz et al., “Foundation Model Forecasts: Form and Function”, 2025**  
   This work analyzes the behavior and evaluation of time‑series foundation model forecasts. It is relevant to the authors’ emphasis on attractor‑level vs pointwise metrics and scaling laws. Incorporating this work in Section 2 and the scaling discussion (Section 4.3) would help position ChaosNexus within the emerging literature that critically examines what makes a foundation model “good” for scientific forecasting.

5. **Zhang & Gilpin, “Zero‑shot Forecasting for Chaotic Systems”, 2024**  
   While the paper does cite Zhang & Gilpin (2024) in the context of metrics, it does not clearly articulate how its foundation‑model approach relates to or improves on their zero‑shot chaotic forecasting framework. A more direct comparison in Section 2 and/or the experiments (zero‑shot synthetic benchmark) would help clarify the incremental contribution of ChaosNexus over earlier zero‑shot chaotic forecasting approaches.

## Questions

1. **Clarification of MoE gating and balancing**  
   Could the authors explicitly define \(f_i\) and \(r_i\) in Eq. (9) in terms of \(\phi_{i,p}\) and the routing decisions, and state whether the TopK scores in Eq. (3) are re‑normalized? Additionally, how is the scale of the shared expert’s coefficient \(\phi_{M+1,p}\) controlled relative to the specialist experts to avoid dominance or collapse?

2. **Implementation details of the MMD loss**  
   How exactly is \(\kappa(\hat{X}^i, \hat{X}^j)\) computed on trajectories in Eq. (10)? Are trajectories flattened over time and variables, or are they subsampled/embedded before kernel evaluation? What is the effective computational cost per batch, and how is memory handled for long trajectories?

3. **Role of temporal mean pooling in joint‑scale readout**  
   Did the authors try alternative readout mechanisms (e.g., attention pooling over time at each scale, or cross‑scale attention) in place of simple temporal averaging in Section 3.3? If so, how did they compare? If not, could the authors comment on why mean pooling was chosen given that it discards phase information?

4. **Weather baselines and pretraining**  
   For WEATHER‑5K, have the authors run experiments where Panda and Chronos‑S‑SFT (both pretrained on the chaotic corpus) are fine‑tuned on the same 0.1% and 0.5% subsets, and if so, what are the relative gains vs ChaosNexus? Bringing at least a subset of those results from Appendix A.6 into the main text would make the claims around “exceptional data efficiency” more concrete.

5. **Sensitivity to wavelet scattering hyperparameters**  
   The WST uses \(J=8, Q=8\) (Appendix B). How sensitive are the results in **Table 7** and **Table 1** to these choices? For example, does reducing \(J\) or \(Q\) noticeably degrade attractor metrics, or is there a fairly broad “good” range? This would help understand whether the WST fingerprint is a brittle hyperparameter or a robust prior.

6. **Extent of generalization beyond ODEs**  
   Section A.12 presents a single PDE experiment on VKVS with PCA projection and shows promising sMAPE curves in **Figure 13**. Could the authors elaborate on how robust this is to the choice of latent dimensionality (d = 16) and whether ChaosNexus has been tested on other PDE systems? Are there any failure modes where the PCA projection yields latent dynamics that differ qualitatively from the ODE training corpus, breaking the assumption that the model is “seeing something ODE‑like”?

Clarifications on these points could strengthen my confidence in the robustness and generality of the approach.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodology is coherent and mostly well justified, with extensive empirical validation on synthetic and real chaotic systems. Some mathematical and implementation details (MoE routing, MMD computation, pooling) are under‑specified, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is dense but generally clear, with strong figures (e.g., **Figures 1–5, 10–12**) and thorough appendices. However, notational inconsistencies (especially in **Tables 1–2**) and some missing explanations for metric abbreviations make parts of the evaluation harder to follow than necessary.

## Contribution Rating

3: good.  
The work advances the state of practice for foundation models in chaotic forecasting with a well‑engineered multi‑scale architecture, rigorous attractor‑level evaluation, and informative scaling laws. The conceptual novelty is more in the integration and domain focus than in fundamentally new theoretical ideas, but the empirical gains and analysis are substantive and of clear interest to the ICLR community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper presents a strong, carefully executed foundation model for chaotic systems, with convincing improvements in attractor‑level metrics and compelling zero‑/few‑shot performance on weather data. The architecture is largely a thoughtful combination of known ingredients, and some comparisons and mathematical details could be sharpened, but the empirical depth and analysis justify a positive recommendation.

## Reviewer Confidence

4: confident.  
I am familiar with chaotic dynamics, time‑series foundation models, and multi‑scale architectures, and I carefully examined the equations and experimental setup. Some implementation details (especially around MMD and MoE routing) are not fully specified, but they do not appear to overturn the main conclusions.