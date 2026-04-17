# Breaking the Limits of Autoregression! A Diffusion-Bridge with Mutual-Information for Time Series Forecasting

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Time series forecasting (TSF) is a fundamental task in many real-world applications, yet effectively modeling both global dependencies and local dynamics remains challenging. Existing diffusion-based approaches typically adopt a noise-driven paradigm, which disrupts temporal continuity and fails to leverage intermediate evolutionary states, thereby limiting forecasting accuracy and robustness. To overcome these limitations, we propose \textbf{AR-DBMI} (\textbf{A}uto-\textbf{R}egressive \textbf{D}iffusion \textbf{B}ridge with \textbf{M}utual-\textbf{I}nformation correction), a novel generative forecasting framework. AR-DBMI reformulates time series evolution as a ``future-to-history'' diffusion bridge, where intermediate states are deterministically generated through sliding operations to preserve transitional dynamics. To further enhance performance, we introduce a velocity-consistency constraint to capture first-order dynamics across windows, and a mutual-information alignment mechanism to ensure semantic consistency between predicted and ground-truth endpoints. In addition, dual-domain regularization combining time-domain anchoring and spectral consistency improves stability under non-stationary and noisy conditions. Extensive experiments conducted on seven widely used datasets demonstrate that our model achieves state-of-the-art performance, significantly outperforming existing diffusion-based TSF models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The AR-DBMI paper proposes a diffusion-based time series forecasting model that is based on the recent ARMD framework (Gao et al., AAAI 2025). The authors claim to "break the limits of autoregression" by reformulating time series evolution as a "future-to-history" diffusion bridge, where intermediate states are deterministically generated through sliding operations rather than noise injection. The core architecture inherits ARMD's sliding-based diffusion mechanism and augments it with three additional components: (1) a diffusion-bridge velocity consistency constraint that learns first-order dynamics between adjacent windows through an auxiliary head, (2) a mutual-information alignment mechanism using InfoNCE loss to align predicted and ground-truth endpoints in embedding space, and (3) dual-domain regularization combining time-domain anchoring and spectral consistency. The paper evaluates the method on seven benchmark datasets (ETT family, Solar Energy, Exchange, and Stock) and reports better performance. A notable claim is that AR-DBMI achieved a 77.9\% reduction in MSE on the Stock dataset compared to the next-best method. 

However, the paper suffers from several critical issues. First, the title's claim of "breaking limits of autoregression" is fundamentally misleading as the model is not autoregressive in the classical sense, but rather performs one-shot prediction of the entire forecast horizon using a non-autoregressive diffusion framework. Second, the novelty over ARMD appears incremental, primarily consisting of auxiliary loss terms (velocity consistency, InfoNCE, spectral matching) rather than fundamental architectural or algorithmic innovations. Third, the exceptional performance on the Stock dataset raises concerns about potential overfitting or evaluation inconsistencies, especially given the dataset's small size (length of 3,685). The ablation study is limited and does not decompose the individual contributions of each auxiliary component. Overall, while engineering is competent and results show improvements in most benchmarks, the conceptual contribution beyond ARMD and the validity of central claims require substantial clarification to be suitable to consider as a publication in ICLR.

### Strengths
1. The diffusion bridge formulation, velocity consistency constraint, and mutual information alignment are presented with correct mathematical notation and well-defined loss functions

2. The paper clearly acknowledges ARMD as the foundation and explicitly states which components are inherited versus novel

3. The combination of velocity consistency, mutual information alignment, and dual-domain (time + spectral) regularization provides complementary inductive biases

4. The use of DDIM with $\eta=0$ enables fast and deterministic inference without the velocity head, avoiding computational overhead at the test time

### Weaknesses
1. The claim of "breaking limits of autoregression" seems to be incorrect as the model performs non-autoregressive one-shot prediction of the entire horizon, not sequential timestep-by-timestep generation, which is typically expected when a term such as "autoregression" is used. Crucially, there are no architectural components supporting autoregressive generation.

2. There is limited novelty over ARMD (AAAI 2025). The core contribution reduces to adding three auxiliary loss terms $(L_{bridge}, L_{mi}, L_{spec})$ to ARMD's framework. The sliding-based diffusion, future-to-history paradigm, and deterministic intermediate states are all inherited from ARMD.

3. The "velocity consistency" component may be redundant. Since the sliding operation is deterministic with known ground truth $X^t$ and $X^{t+1}$, the target velocity $v^*_t = \gamma(X^{t+1} - X^t)$ is directly computable. It is unclear why an auxiliary head is needed to learn this already-determined quantity rather than having the main diffusion loss capture inter-window dynamics.

4. A crucial drawback is the insufficient ablation study presented. Table 4 only tests w/o MI, w/o Diffbri, and w/o both. Several ablations are missing as this is an empirical work. For example ablations on individual contributions of spectral consistency, $x_o$ anchoring, SNR weighting are missing. Which component contributes most? (See also questions below). Crucial for any empirical work is to include a hyperparameter sensitivity analysis, for example $(\lambda_{bridge}, \lambda_{mi}, \lambda_{x0}, \lambda_{spec})$. I am not listing more, but the direction should be clear. Without strong theoretical or empirical evidence, it is difficult to accept why so many loss terms and data are needed.

5. Eventhough the related work talks about models such as MrDiff, CSDI, SSSD etc, these are not included as baselines. Some of these missing baselines learn the prior (which is similar to the bridge concept) and others learn multi resolution aspects (similar to the spectral concept). I quickly checked the MrDiff paper and it seems they report better numbers, although I acknowledge that the experimental setup is not the same, which leads to the next weakness.

 6. The experimental setup is not consistent when compared to other TSF work in recent ICML/ICLR papers. There is also no evidence of autoregressive nature of TSF. Specifically, the look-back length and forecast horizon both fixed to 96 for all datasets. However, it is well known in the TSF community that different datasets may benefit from different window sizes. No experiments on varying horizons are shown (e.g., 192, 336, 720 as common in TSF literature). A reader would reasonably expect an autoregressive model to show autoregressive generation, but this is also missing.

7. All results are reported as point estimates without confidence intervals, standard deviations, or significance tests across multiple runs (for a few datasets in the appendix should have been fine). 
    
The key point is that the narrow margin with ARMD raises questions about whether the new loss functions are truly effective or if the performance enhancement is primarily due to extensive tuning of hyperparameters. Although engineering work is valuable, this aspect has not been explored in sufficient depth in this manuscript version for ICLR.

### Questions
There are several pressing questions about this paper that need carefully designed numerical experiments, ablation studies, sensitivity analyzes, computational analyzes, and/or theory development. This reviewer lists some important questions here, but the authors must ask and answer many more questions to make this paper a solid standalone contribution to ICLR.

1. Can the authors clarify how the model is "autoregressive" when it predicts the entire forecast horizon in one shot (as shown in Algorithm 2 and Equation 6)? Classical autoregressive models generate $y_t$ conditioned on $y_{t-1}$, $y_{t-2}$, etc. Please either provide architectural evidence of sequential timestep-by-timestep generation or revise the framing to accurately reflect the non-autoregressive nature of your approach.

2. Can the authors articulate what conceptual advance justifies a separate publication versus treating this as an extended version of ARMD?

3. The paper includes both anchoring and mutual information alignment. Both aim to align predictions with ground truth. Please explain: (a) why both are necessary, (b) whether they capture different aspects, and (c) provide results showing the interaction effect between these two losses. Some visualizations of the representation space could help in understanding.

4. Table 4 only shows three configurations. Please provide a complete ablation table with: (a) baseline (ARMD), (b) +$L_{bridge}$ only, (c) $+L_{mi}$ only, (d) $+L_{x0}$ only, (e) $+L_{spec}$ only, (f) two combinations (g) three combinations and (f) all combinations (that is your final model). Which component contributes the most to performance? What is the marginal contribution of each?

5. There are several hyperparameters. How sensitive are results to these choices? Please provide: (a) the values used for each dataset, (b) sensitivity curves showing performance vs. each $\lambda$, and (c) whether the same hyperparameters work across all datasets or require per-dataset tuning.

6. What is the computational overhead of your auxiliary components? Please report: (a) number of parameters for $f_\theta$, $h_\theta$, and $\phi$ separately, (b) training time per epoch compared to ARMD (or whichever is the second best model), (c) inference latency, and (d) memory consumption. Does the 3-10\% performance gain justify the added cost?

7. All results in Tables 2-4 are point estimates. Please provide: (a) mean $\pm$ standard deviation over at least 5 random seeds for all methods on all datasets, (b) statistical significance tests comparing AR-DBMI vs. ARMD, and (c) confidence intervals to determine whether observed improvements are statistically meaningful.

8. Can the authors add other baselines (MrDiff, CSDI, SSSD, CNDiff etc) in a similar experimental setup and do a thorough evaluation?

9. Beyond aggregate metrics, can the authors provide: (a) qualitative visualizations comparing AR-DBMI, ARMD, and ground truth on both successful cases (Stock) and failure cases (Solar), (b) error distribution analysis, and (c) characterization of when each auxiliary component helps versus hurts? The ARMD (AAAI paper) does not show good qualitative prediction especially for time series with several portions that are flat. Does adding the three losses mitigate this issue?

10. The 77.9\% MSE reduction in Stock (0.235 to 0.052) is dramatically larger than your 3-10\% improvements on other datasets. Can you provide: (a) results with error bars across multiple random seeds, (b) confirmation that the same hyperparameters were used for ARMD and AR-DBMI on Stock, (c) per-variable breakdown of errors, and (d) explanation for why this dataset shows such disproportionate gains?

11. It seems that the code lines 29-30 uses eq 3 and not eq 4 (sorry if I misunderstood). Why is that? Are there any numerical stability patterns that you observe on one vs the other?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **AR-DBMI**, an autoregressive generative framework that reinterprets diffusion models for time-series forecasting. Unlike conventional noise-driven diffusion models that disrupt temporal continuity, AR-DBMI formulates forecasting as a deterministic “future-to-history” diffusion bridge, where intermediate states are produced via sliding windows to preserve transitional dynamics. 
It integrates a velocity-consistency constraint to learn first-order temporal dynamics and a mutual-information alignment module that ensures semantic consistency between predicted and ground-truth sequences. 
Additionally, dual-domain regularization—combining time-domain anchoring and frequency-domain spectral consistency—enhances robustness under non-stationary and noisy conditions. 
Evaluated on seven benchmark datasets, AR-DBMI achieves SOTA performance, outperforming prior diffusion-based and transformer-based baselines. 
At the same time, ablation studies confirm the complementary benefits of the velocity and mutual-information components.

### Strengths
S1.
The proposed AR-DBMI framework reformulates diffusion processes as a deterministic “future-to-history” bridge, replacing stochastic noise injection with sliding-window evolution to preserve temporal continuity—an elegant conceptual shift that directly addresses one of the key mismatches between diffusion modeling and TSF.

S2.
The methodology is mathematically grounded, with detailed derivations, explicit algorithmic pseudocode, and ablation studies that isolate each component’s contribution.

S3.
The structure (from problem definition to methodology to evaluation) is logical, figures effectively illustrate the model’s intuition, and notation is consistent.

### Weaknesses
W1.
This paper’s core innovation—replacing stochastic diffusion with a deterministic sliding bridge—is intuitively appealing but lacks deeper theoretical analysis. There is no formal discussion of how this reformulation affects the probabilistic semantics of diffusion modeling or its connection to score-based generative processes.

W2.
Although this work benchmarks on seven datasets, these are all standard mid-scale TSF benchmarks (ETT, Solar, Exchange, Stock). The model’s scalability and generality remain unclear for long-horizon, high-dimensional, or irregularly sampled series—scenarios that diffusion models might struggle with.

W3.
The ablation study focuses only on removing the mutual-information and diffusion-bridge modules. Still, other crucial design choices—such as spectral consistency, SNR weighting, and DDIM determinism—are not analyzed. Including per-component performance deltas or visualization of how each regularizer affects spectral smoothness or stability would clarify where performance gains originate.

### Questions
Q1.
The paper redefines diffusion as a deterministic sliding process rather than a stochastic noise-driven one. Could the authors clarify how this formulation relates to conventional diffusion probabilistic models in terms of likelihood estimation or score matching?

Q2.
Would the authors provide a short theoretical discussion (or visualization) comparing the latent trajectories between stochastic and deterministic diffusion?

Q3.
Does the model require access to full sequences (past + future) during training only, or does this formulation affect the autoregressive inference direction at test time?

Q4.
Could the authors provide a quantitative or qualitative breakdown showing the contribution of each auxiliary component (velocity consistency, MI alignment, x₀ anchoring, and spectral regularization)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AR-DBMI, an autoregressive diffusion-bridge framework for time series forecasting (TSF) that reinterprets the diffusion process as a deterministic "future-to-history" evolution via sliding windows, rather than a traditional noise-injection process. The method introduces several novel components: a diffusion-bridge velocity consistency constraint to capture first-order dynamics between adjacent windows, a mutual-information alignment module to semantically align predicted and ground-truth endpoints, and dual-domain consistency regularization (time and frequency) to enhance stability and fidelity. The model is evaluated on seven public TSF benchmarks and demonstrates state-of-the-art performance against both diffusion-based and non-diffusion baselines, particularly under noisy and non-stationary conditions.

### Strengths
1. The idea of reframing diffusion as a deterministic sliding process from future to history is innovative and better aligned with the temporal structure of time series than conventional noise-based diffusion.

2. The integration of multiple components—velocity consistency, mutual-information alignment, and spectral consistency—provides a holistic approach to capturing both local dynamics and global dependencies, while also improving semantic and numerical fidelity.

3. The use of deterministic DDIM sampling, SNR-aware weighting, and train-only auxiliary heads (e.g., velocity head) shows thoughtful design choices that balance performance and inference efficiency.

### Weaknesses
1. While the model is designed to avoid inference overhead from auxiliary heads, no analysis or comparison of training or inference time is provided. This is important for assessing practical applicability, especially given the use of multiple loss terms and sliding operations.

2. The ablation study only removes two main components (MI and Bridge). It would be stronger if it also evaluated the impact of spectral consistency and time-domain anchoring independently.

3. The paper relies heavily on quantitative metrics. Visualizations of forecasted sequences (e.g., trend recovery, periodicity) would help illustrate the model’s advantages, especially for frequency-aware components.

4. While AR-DBMI outperforms many non-diffusion models, the gap is less pronounced than with diffusion-based models. A deeper discussion on when and why diffusion-based approaches are preferable would be useful.

### Questions
1. How does the model scale with longer forecast horizons (e.g., >96 steps)? Does the deterministic sliding mechanism remain effective, or does it suffer from error accumulation like other autoregressive models?
2. The velocity head is only used during training. Have you experimented with using it during inference to further improve temporal consistency? If not, why?
3. The mutual-info alignment uses a simple temporal average pooling. Did you explore more advanced pooling mechanisms (e.g., attention-based) to better capture temporal semantics?
4. The spectral loss uses the amplitude spectrum. Was phase information considered? If not, what is the rationale for omitting it?
5. The paper mentions a warm-up strategy for auxiliary loss weights. Could you share more details on the scheduling and its impact on convergence?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents AR-DBMI (Auto-Regressive Diffusion Bridge with Mutual-Information correction), an extension of ARMD for time series forecasting (TSF). The proposed framework aims to address limitations in existing diffusion-based approaches by integrating a velocity-consistency constraint and a mutual-information alignment mechanism. It further employs dual-domain regularization to enhance model stability under non-stationary and noisy conditions. The paper provides a detailed methodological description and conducts comprehensive experiments to validate its approach.

### Strengths
- The paper is very well written and clearly structured. The figures and illustrations effectively support the explanations.

- The proposed method shows noticeable improvements across benchmarks and considers a broad range of baselines for comparison.

- The combination of velocity consistency (capturing local dynamics) and mutual-information alignment (preserving semantic consistency) is a thoughtful design choice that tackles both local and global aspects of TSF in a coherent way.

### Weaknesses
- The experiments focus only on $H=96$. It would be helpful to assess whether the proposed method maintains robustness under longer forecasting horizons, which are common in industrial applications. In addition, it is unclear whether the choice of $H$ depends on the dataset’s temporal resolution (e.g., days, hours, minutes).

- The ablation study does not isolate the effects of time-domain anchoring and spectral consistency, making it difficult to evaluate their individual contributions.

### Questions
- What is the value of $k$ used in the experiments? How is $k$ adjusted across datasets with different temporal frequencies, and has its impact on performance been validated?
- In Figure 2, the trajectories do not appear as smooth as expected. Additional visualizations might help clarify this effect.
- In Section 3.2.3, the role of the additional velocity head $h_\theta$ could be better explained. Based on the code, it seems implemented as a simple linear layer---could the authors elaborate on whether this design is standard or specific to this work?

### Soundness
3

### Presentation
3

### Contribution
2
