# Neural Fourier Attention: A Framework for Learning Data-Adaptive Signal Bases

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
The standard approach in signal processing and deep learning is to use fixed, predefined basis functions. This includes classic methods like the discrete Fourier transform (DFT) and modern tools like convolutional kernels. Our work challenges this fundamental paradigm. We introduce Neural Fourier Attention (NFA; also referred to as a Neural Fourier Basis Generator, NFBG), a framework designed to learn a data-adaptive, non-stationary signal transform. The core idea is to replace projection onto a fixed basis with a neural controller that dynamically generates basis-function parameters for each input, constructing a bespoke representation tailored to the signal's characteristics. Importantly, our use of the term “Attention” does \emph{not} denote a query–key–value operator; it denotes a controller that attends to the input to \emph{generate the basis}. We provide a rigorous formulation within frame theory and propose an orthogonality regularizer that encourages near-tight frames for stability. Extensive experiments show state-of-the-art results on challenging benchmarks (notably on Weather) and strong average performance across the ETT family, with ablations validating the controller design and the benefits of controlled non-orthogonality. \textbf{When to use/when not to}: NFA targets signals with complex, non-stationary periodicities (e.g., Weather/ETT); for trend-dominant or near-random-walk series (e.g., Exchange), a hybrid pipeline that decomposes trend/seasonality first and applies NFA to residuals is recommended.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Neural Fourier Basis Generator (NFBG), a framework that learns a data-adaptive, non-stationary signal transform and is applicable to signals with complex, non-stationary periodicities.

### Strengths
S1. This paper provides a rigorous formulation within frame theory and propose an orthogonality regularizer that encourages near-tight frames for stability.  
S2. This paper provides a computational complexity analysis.

### Weaknesses
W1. The paper suffers from poor organization and unclear writing, which make it hard to follow. There are many inconsistent symbol usages that need to be carefully checked.  
W2. The motivation of this paper is not clearly articulated. It is unclear why the basis must be learned dynamically for each input, and what concrete drawbacks would arise if the underlying basis vectors were fixed or learned globally. The paper does not provide sufficiently strong references or deeper insights to justify this design choice. Moreover, prior work has achieved long-term forecasting with fixed basis functions, such as FBM [1]. A clearer articulation of the motivation and a more systematic discussion of related work are needed to highlight the novelty and distinct contribution of this approach.  
W3. While NFA is designed for signals exhibiting complex, non-stationary periodicities, the baseline setup omits representative non-stationary forecasting methods (e.g., FAN [2], DDN [3], DERITS [4]), which weakens the fairness and completeness of the comparison.

[1] Rethinking Fourier Transform from A Basis Functions Perspective for Long-term Time Series Forecasting. NeurIPS, 2024.  
[2] Frequency Adaptive Normalization For Non-stationary Time Series Forecasting. NeurIPS, 2024.  
[3] DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting. NeurIPS, 2024.  
[4] Deep Frequency Derivative Learning for Non-stationary Time Series Forecasting. IJCAI, 2024.

### Questions
pls refer to weakness.

### Soundness
2

### Presentation
1

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
This paper introduces Neural Fourier Attention (NFA), a framework that learns data-adaptive signal bases rather than using fixed, predefined basis functions. The key point is generating custom basis functions for each input signal, rather than just learning coefficients for a fixed basis. The controller network processes the input signal and generates parameters for the basis. Then the input is projected on to this specific basis that adapts to this specific input. There are experiments conducted to show tha this method have good performance over some datasets. On weather datasets the improvement is significant. Also on some dataset this method does not has much effect.

### Strengths
This paper provides a novel approach of learning adaptive fourier bases instead of fixed ones. The theoretical foundation is also solid, where in section 2.2 there are analysis of the learn transformation. The experiment analysis is comprehensive, that it reports the cases where the architecture both works well and not. And provides analysis about in what case can the model give better performance and cases where this model does not improve. On weather dataset produces significant improvement.

### Weaknesses
Line 44 says that what if the basis itself could be learned dynamically for each input. To validate this statement, there should be some experiments on fixed bases models. However in the experiment there are no such comparision.

### Questions
1. Experiment part does not has a baseline model that is fixed basis. Since this paper proposes a adaptive bases architecture, it is important to compare with fixed basis baseline to demonstrate how significant to use a model with adaptive bases. There should be validation of the claim "What if the basis itself could be learned dynamically for each input?" 

2. The controller network $F_ctrl$ tooks a input window, how is this chosen and what this affect?

3. Other than ETT and weather is there any dataset that the model can have improvement over existing methods.

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
3

### Summary
This paper presents Neural Fourier Attention (NFA), a method for learning data-dependent signal bases. Instead of relying on a fixed basis for all signals, the authors suggest learning an input-dependent basis made up of sine waves with adjustable frequencies, amplitudes, and phases. They then project the signal into this basis. A neural network controller is used to learn the basis. To ensure the learned basis performs well, the authors base their approach on frame theory and introduce a new orthogonality regularizer to maintain stability and expressiveness in the basis. This approach achieves top forecasting results on benchmarks with complex, non-stationary periodicities, particularly the Weather dataset. The authors also provide thorough ablation studies to support the design choices and present an efficient version, Harmonic-Constrained NFA (HC-NFA), which uses FFTs for quicker computation.

### Strengths
- To my knowledge, the idea of learning the basis instead of using a fixed one is novel. It is also well motivated for the problem of non-stationary signals, where patterns change over time.

- The approach is principled. The theory clearly explains why the orthogonality regularizer is helpful in creating a stable and expressive representation. Specifically, proposition 1 offers a proof sketch that their orthogonality regularizer ($||BB^T - I||_F^2$) shapes the learned basis into a "Parseval tight frame" on the subspace it covers. Additionally, the solid justification for the orthogonality regularization, supported by clear propositions and SVD-based derivations (see Section 2.2.3 and Appendix A.1), gives a strong theoretical basis.

- On datasets with complex, non-stationary periodicities (notably Weather and the averaged ETT family), NFA achieves top performance. Table 1 (Page 6) provides a clear, statistically robust comparison against competitive baselines (iTransformer, PatchTST, DLinear).

- The ablations are comprehensive. For example, Fig. 3 shows that a small, nonzero orthogonality penalty ($\lambda_{\text{ortho}}$) offers the best performance. Figure 2 illustrates the incremental advantages of different controller designs, showing that the adaptive basis mechanism is mainly responsible for the strong results.

- The main text and appendices offer solid implementation details that support reproducibility.

### Weaknesses
- Missing critical evaluation: The central claim, that adapttive, per-sample nature of the basis is key, is not validated empirically. A key missing cmoparison is a model with fixed, randomly initialized sinosoidal basis, on top of which samples are projeced. 

- Limited Evaluation on Other Domains/Modalities: While the main results focus on time series with periodicity, there is no attempt to apply the framework to domains other than time series (e.g., spatial signals, images).

- Limited evaluation to baselines and discussion in related work. While the approach provides a comparison to some baselines, many relevant baselines are missing. This makes both the related work and the comparison incomplete, and it is not possible to determine how well the approach works relative to current art. See, for example, [1-8] below. 

- Efficiency: While HC-NFA is provided as a computationally efficient alternative, the NFA’s full projection step is $O(NK)$, and the Gram matrix calculation can be expensive. Fig. 10 in the Appendix illustrates the trade-off versus the number of probes, further confirming meaningful overhead. This limits practical scalability in real-time settings. Further, a full cost analysis is not provided. For instance, it would be beneficial to understand the cost of all components, including the LSTM controller, as well as the projection and regularization steps. It's important to note a direct comparison of time and FLOPs against key competitors, such as iTransformer, under a matched parameter budget; however, this is currently missing.

- Results are not fully convincing. Tab. 1 results show that NFA performs very poorly on the Illness and Exchange Rate datasets. As such, the claim that the approach is a universally superior model seems like an overclaim. 

[1]. First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting (TDformer), 2022 
[2]. DESTformer: A Transformer Based on Explicit Seasonal–Trend Decomposition for Long-Term Series Forecasting, 2023 
[3]. DFCNformer: A Transformer Framework for Non-Stationary Time-Series Forecasting Based on De-Stationary Fourier and Coefficient Network, 2025 
[4] Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting (FreDF), 2024 
[5] ATFNet: Adaptive Time-Frequency Ensembled Network for Long-term Time Series Forecasting, 2024 
[6] Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting (FBM/FBM‑S), 2025 
[7] MFRS: A Multi-Frequency Reference Series Approach to Scalable and Accurate Time-Series Forecasting, 2025 
[8] Adaptive Temporal-Frequency Network for Time-Series Forecasting (ATFN), 2022

### Questions
Related to the above weaknesses:

- How well does the approach work related to the fixed, randomly initialised sinosoidal basis?
- How well does the approach work on other domains?
- How does the approach compare to other recent SOTA approaches? 
- What is the clock time of FLOPs comparison to key baselines that do not learn the basis?

### Soundness
2

### Presentation
3

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
The paper introduces Neural Fourier Attention (NFA), a forecasting block that, for each input window, uses a neural controller (typically an LSTM) to emit parameters of $K$ sinusoidal atoms $({a_k,\omega_k,\phi_k})$, builds a data-dependent basis $B\in\mathbb{R}^{K\times N}$, projects $c=Bx$, and predicts with a small MLP head (Algorithm 1). Here “attention” denotes a controller that attends to the input to generate basis parameters, not a QKV operator. The authors ground NFA in frame theory and regularize with a row-Gram penalty $\lVert BB^\top-I\rVert_F^2$, showing (via Proposition 1) that it drives the non-zero singular values of $B$ to $1$ (near-Parseval) on the data-dependent subspace $V(x)$; they note a global Parseval frame is impossible when $K>N$. They further propose HC-NFA, which constrains $\omega_k$ to the DFT grid so projection is done by FFT, reducing projection cost to $O(N\log N)$; the orthogonality penalty is estimated with a Hutchinson trick to avoid explicit $(K\times K)$ Gram materialization. Experiments on ETT (macro-average over four subsets), Weather, Illness, and Exchange follow a leakage-aware protocol with multi-seed reporting; headline gains are on Weather and ETT Avg; limitations on Exchange are discussed.

### Strengths
1. **Principled geometry control.** The row-Gram penalty’s effect on singular values is proved (A.1), matching the goal of near-Parseval frames.
2. **Clear pipeline & stability measures.** Controller-generated sinusoids, $c=Bx$, and parameter squashing choices are explicit and motivated by stability/Nyquist.
3. **Efficiency variant & timings.** HC-NFA’s FFT path and empirical timings vs. NFA/Transformer are reported.
4. **Transparent scope & protocol.** Multi-seed results, anti-leakage checklist, and explicit limitations on Exchange.

### Weaknesses
1. **Baseline traceability.** Table 1 aggregates external numbers (not re-tuned locally) and cites the iTransformer paper; per-cell provenance (paper vs. repo, URL/commit) is absent, limiting auditability of the headline gains.
2. **Ablation generality.** Most sensitivity plots (orthogonality weight, $K$, controller) are on ETTh1-96; transfer of optimal $\lambda_{\text{ortho}}/K$ to Weather or other horizons is not demonstrated.
3. **Statistical comparisons to baselines.** The paper reports CIs for NFA vs. HC-NFA (Weather-96) but not for external baselines in Table 1, so statistical significance vs. baselines cannot be assessed from the manuscript alone.
4. **Caption specificity.** Several figure captions do not restate dataset/horizon (e.g., Fig. 3), reducing stand-alone clarity.

### Questions
1. **Provenance table.** For every dataset–horizon cell in Table 1, can you list source (paper vs. official repo), URL/commit, and any preprocessing differences? This would materially strengthen reproducibility.
2. **Penalty variant.** Main results: do you use the raw $\lVert BB^\top-I\rVert_F^2$ (with per-dataset $\lambda_{\text{ortho}}$) or the normalized $K^{-2}$ variant? Please clarify and, if mixed, report both.
3. **Ablation transfer.** How do the ETTh1-96-optimal $\lambda_{\text{ortho}}$ and $K$ translate to Weather-192/336/720 and ETTm? A small grid on at least one other dataset/horizon would help.
4. **Controller parity.** Under matched FLOPs/params, how do a compact 1D-Conv and a tiny Transformer compare to LSTM as controllers? Fig. 2 suggests recurrence helps, but a controlled capacity match would isolate recurrence vs. capacity.
5. **Efficiency scaling.** Can you add wall-clock comparisons for larger $N$ (and varying $K$) for NFA vs. HC-NFA to complement Table 5’s ETTh1 setting?
6. **Phase parameterization.** Did you try predicting $(\sin\phi,\cos\phi)$ (angle-on-the-circle) to mitigate wrap-around, and if so did that change stability/convergence relative to the current sigmoid scheme?

### Soundness
3

### Presentation
3

### Contribution
3
