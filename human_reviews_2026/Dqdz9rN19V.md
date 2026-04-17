# RAYQUAZA : Input-Conditioned Radial Basis Decomposition for Efficient Univariate Time-Series Forecasting

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Time-series forecasting presents a persistent trade-off between simple, scalable linear models that struggle with complex dynamics and large neural architectures that offer high accuracy at a steep computational cost. We introduce RAYQUAZA, a parameter-efficient architecture designed to fill this gap. RAYQUAZA learns an adaptive basis decomposition of the signal into three complementary components: a smooth trend extractor, a residual correction branch, and a novel input-conditioned radial basis function layer. The iRBF module dynamically learns a compact set of localized Gaussian atoms for each input sequence, enabling it to model transient, non-stationary patterns like structural breaks and spikes that challenge simpler methods. With fewer than 0.12M parameters, RAYQUAZA achieves state-of-the-art accuracy on large-scale public benchmarks and demonstrates consistently strong performance across a diverse range of forecasting domains. Crucially, it outperforms lightweight linear baselines in the majority of long-horizon forecasting scenarios while remaining two to three orders of magnitude smaller than transformer-based models. These results establish RAYQUAZA as a practical, interpretable, and efficient model, proving that adaptive basis representations can deliver high accuracy without sacrificing efficiency

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel architecture, RAYQUAZA, which addresses a persistent problem in time-series forecasting: the gap between simple linear models (efficient but fail to model complex patterns) and large neural networks (accurate but computationally expensive).

The core of RAYQUAZA is to decompose the input signal into three complementary components:

1. ATE (Adaptive Trend Extractor): Smooth low-frequency trends and seasonality.
2. RCC (Residual Correction Convolutions): High-frequency residual signals.
3. iRBF (Input-Conditioned Radial Basis Function): (Core Innovation) A dynamically generated, "customized" set of localized Gaussian basis functions for each input sequence.

The iRBF module uses a hypernetwork (MLP) to predict the centers, spreads, and amplitudes of Gaussian 'atoms' specific to the input data. This allows it to effectively model transient and non-stationary patterns, such as spikes and structural breaks, which are challenging for existing models.

Experimental results show that with fewer than 0.12M parameters, RAYQUAZA achieves state-of-the-art (SOTA) accuracy on major benchmarks (M4, ETT, TFB), surpassing massive Transformer-based models. This demonstrates that it successfully overcomes the trade-off between model efficiency and expressive power.

### Strengths
1. This is the paper's strongest contribution. RAYQUAZA uses 100x to 1,000x (two to three orders of magnitude) fewer parameters than existing SOTA models (e.g., ModernTCN, Informer) while achieving better performance (Section 5.7, Appendix G). This makes it a highly practical solution for resource-constrained deployment environments.

2. The model is not just lightweight; it consistently demonstrates top-tier accuracy across diverse and challenging benchmarks, including short-term (M4), long-term (ETT), and large-scale (TFB) forecasting tasks (Section 5.1, 5.2, 5.3).

3. The core idea of the iRBF, 'per-instance basis generation,' is novel. The ablation study (Section 5.6, Fig 3) clearly shows this module is decisive in capturing spikes and abrupt changes, significantly outperforming fixed-basis methods (FFT, Global RBF).

4. The authors conducted extensive experiments with diverse benchmarks, strong baselines, and thorough ablation studies (module removal, basis comparison, normalization schemes) that robustly support the paper's claims.

### Weaknesses
1. Limitation of "Multivariate" Claim:  The paper claims SOTA on multivariate benchmarks (ETT, TFB), but the RAYQUAZA architecture is fundamentally Univariate. Multivariate data is handled using a 'Channel-Wise' strategy, forecasting each channel independently (Section 5.4, Section I). This approach does not model any cross-channel correlations. Therefore, its performance may degrade on problems where inter-variable interactions are critical, and it cannot be considered a true multivariate model.

2. Lack of Sensitivity Analysis for Hyperparameter 'K': The number of Gaussian basis functions (K) in the iRBF module—a key hyperparameter—is manually set differently based on the data's frequency (e.g., K=8, 12, 16) (Appendix A). This can be a significant tuning point that undermines the model's automaticity. The paper lacks a sensitivity analysis on the choice of K or how much performance degrades if K is set sub-optimally.

3. Overstated Interpretability Claim: The authors claim the iRBF is "inherently interpretable" (Abstract). However, unlike the simple synthetic data in Figure 4, intuitively 'interpreting' the result of 16 overlapping Gaussian functions on a complex, noisy real-world signal (like M4-Hourly) would be nearly impossible. While the module may be 'inspectable,' this is not the same as being 'interpretable.'

### Questions
1. (related to weakness 1) How does RAYQUAZA compare in a fair multivariate setting against SOTA multivariate models (e.g., SCINet, Crossformer) that explicitly learn cross-channel interactions, rather than using a channel-wise strategy?

2. (related to weakness 2) Could you provide a sensitivity analysis for the number of basis functions (K) in the iRBF module? For example, how much does the performance (OWA) degrade when applying K=16 to the M4-Yearly data (where K=8 was optimal), or K=8 to the M4-Hourly data (where K=16 was optimal)?

3. Could you please specify the detailed architecture of the MLP_iRBF hypernetwork that generates the iRBF parameters (e.g., the pooling method used to aggregate information from the input sequence x, and the MLP's depth/width)?

4. (related to weakness 3) Could you provide a visualization of the learned iRBF Gaussian functions for a sample from a complex, real-world dataset, such as M4-Hourly or ETTm1, instead of the synthetic data from Figure 4?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RAYQUAZA, a parameter-efficient architecture for time-series forecasting designed to bridge the gap between simple linear models and large neural networks. The core contribution is an input-conditioned RBF (iRBF) module, which uses a hypernetwork to dynamically generate a compact set of Gaussian atoms tailored to each input sequence. This allows the model to capture transient and non-stationary patterns effectively.

### Strengths
1.Parameter Efficiency: The model's primary strength.It achieves SOTA performance while being 2-3 orders of magnitude smaller (<0.12M params) than large transformer-based competitors.

2.Novelty: It is is a highly novel mechanism for creating adaptive, per-sample bases to model non-stationary patterns.

### Weaknesses
1.Limited Multivariate Modeling: The model relies on a channel-wise (univariate) strategy for multivariate datasets. This ignores cross-channel dependencies。

2.Strong Inductive Bias: The architecture assumes that transient, non-stationary behaviors are well-modeled by Gaussian atoms. This may not be universally true for all signal types and lacks theoretical grounding.

3.Simplistic Fusion : The model combines components via simple addition, assuming they are additively separable. This may be an oversimplification, as components could interact non-linearly in complex systems.

### Questions
1.Multivariate Extension: How could the RAYQUAZA architecture be extended to a true multivariate model that explicitly captures cross-channel dependencies, and how might this impact its signature parameter efficiency?

2.Have you experimented with other input-conditioned basis functions besides Gaussian atoms, particularly for signals with sharp, non-Gaussian transients?

3.Did you investigate more complex, non-linear fusion mechanisms in the FPL instead of simple addition?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces RAYQUAZA, a lightweight neural architecture for time-series forecasting that decomposes input signals into three components: a smooth trend via an Adaptive Trend Extractor, localized transients through an input-conditioned Radial Basis Function layer, and high-frequency residuals using Residual Correction Convolutions. The iRBF is the core innovation, employing a hypernetwork (MLP) to generate per-input Gaussian basis functions, enabling adaptive modeling of non-stationary patterns like spikes and shifts. The model is trained end-to-end with MSE loss and evaluated on benchmarks like M4 (short-term), ETT (long-horizon), and TFB (large-scale). It achieves SOTA or near-SOTA results with <120K parameters, outperforming linear baselines on complex tasks while being 100-1000x smaller than transformers.

### Strengths
S1: RAYQUAZA achieves a strong efficiency-interpretability balance, matching transformer-level accuracy with only 50–120K parameters (0.1% of ModernTCN) and 6-22% training cost, while offering rare transparency through visualizable Gaussian atoms that clearly reflect spikes and trends, providing frequency-like interpretability in a spatial RBF form.

S2: Unlike previous decomposition-based methods that primarily separate smooth trends and seasonal components, RAYQUAZA explicitly models spikes and transient irregularities through input-conditioned Gaussian bases, extending decomposition beyond stable structures to capture short-lived, non-smooth events.

### Weaknesses
W1: While effective, the univariate/channel-wise approach may overlook cross-variable interactions, a limitation in real-world scenarios.

W2: Because RAYQUAZA’s iRBF module builds localized Gaussian bases conditioned on each input, it likely inherits the classical noise sensitivity of radial basis function networks. In high-noise or low signal-to-noise settings, such bases may overfit spurious fluctuations, as each atom locally interpolates the input. Without explicit regularization or smoothing, the adaptive Gaussians could capture random noise rather than meaningful transients. The paper does not analyze robustness to noise, which would strengthen confidence in the model’s generalization under realistic conditions.

W3: While the paper positions RAYQUAZA as an advance in adaptive basis decomposition, it lacks direct quantitative comparison with recent basis-based forecasting architectures, such as BasisFormer[1], FBM[2].

[1] BasisFormer: Attention‑based Time Series Forecasting with Learnable and Interpretable Basis
[2] Rethinking Fourier Transform from A Basis Functions Perspective for Long-term Time Series Forecasting

### Questions
1. What is the motivation for adopting a univariate/channel-wise design? Was it chosen for efficiency, or do cross-variable dependencies offer limited gains?

2.  Since the iRBF module builds localized Gaussian bases per input, how robust is RAYQUAZA to high-noise or low-SNR time series? Could the authors share any sensitivity analysis or ablation (e.g., varying noise levels) to demonstrate the model’s stability compared to non-RBF baselines?

3. Could the authors compare the proposed iRBF decomposition with other basis formulations (e.g., Fourier, wavelet, or learned basis such as BasisFormer) to better demonstrate its advantages?

I would consider increasing my score if the authors respond convincingly to the raised concerns.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes RAYQUAZA, a tiny forecasting model that targets the gap between very light linear models and heavy neural/Transformer models. It claims strong accuracy with ~0.1M parameters via an adaptive basis decomposition. Parameter counts are orders of magnitude lower than big baselines; authors report large training-time savings. The architecture sums four modules: (i) an input-conditioned RBF (iRBF) that, per sequence, generates K Gaussian atoms (centers/widths/amplitudes) using a small MLP; (ii) an Adaptive Trend Extractor (ATE): softmax-normalized smoothing filters; (iii) Residual Correction Convolutions (RCC) for high-frequency remnants; and (iv) a Fusion Projection Layer (FPL) for the final forecast. Component ablations show the iRBF is most critical. A basis-family study (fixed/global RBF, FFT, wavelets) favors input-conditioned iRBF.

### Strengths
- Simple, modular inductive bias with clear roles. The trend (ATE), localized transients (iRBF), and residual refinement (RCC) are well motivated, delineated and easy to implement/ablate

- Broad benchmark coverage (for univariate / channel-wise). Results on M4, ETT, and TFB show robustness across domains/frequencies; channel-wise evaluation probes long horizons where linear models struggle

- Ablations support the core claim. Removing iRBF hurts most; replacing iRBF with classic bases shows large performance gaps, aligning with the "per-input localization" thesis

- Interpretability angle. The iRBF atoms are visualizable and linked to transients/spikes (with qualitative figures and a diversity analysis)

- Appendices provide comprehensive additional results, both qualitative and quantitative - this is appreciated

### Weaknesses
- The paper puts itself within hypernetwork class - the analogy I actually like a lot. However, there is no comprehensive literature review and positioning of this work within the the hypernetwork family

- Additional benchmarking results would help to strengthen empirical part 

- Missing comparison against other tiny models

- Novelty framing needs sharpening w.r.t. BasisFormer

### Questions
- Could you please provide literature review on hypernetwork family and provide the discussion of positioning within this class of models? See [1,2]

- The paper makes a link with NBEATS in the intro. However, comparison with NBEATS and ESRNN [3,4], the two models especially successful on M4 and other classical benchmarks is missing. Could please add OWA and sMAPE to tables? Additionally, putting the results in perspective by comparing to classical models presented in Tables in [3] would be very useful. Also, M3 and TOURISM datasets provide very valuable perspective on comparison against statistical and hand-crafted models (Table 1 in [3]). Could you please add these benchmarks and associated results?

- NBEATS and ESRNN are ensemble models. I wonder how well the proposed method responds to ensembling? Given the fast training of the model, it should be easy to train multiple models in parallel or use checkpoint averages such as in ESRNN to provide experimental results in this axis. If the model responds well to ensembling, this should further strengthen empirics. If not, I would challenge the usefullness of the model. BTW the ESRNN model is not very big either

- Paper makes a claim of compute efficiency. In this context, comparison against other recent tiny models from Koopman family such as SKOLR and Koopa [5,6] feels important. Could you please add this?

- Table 1 can be moved to Appendix. I think it would be more important to have Tables 7,8 in the main body as a main result

- Can you provide architecture implementation, training and testing code?

- What is model size relation to DLinear?

- An input-conditioned RBF layer (hypernetwork-generated atoms) is the central contribution. Prior basis-decomposition models and global-basis attention (e.g. BasisFormer [7]) are close neighbors. The paper should better isolate what is principally new beyond per-sequence parameterization and demonstrate why this is preferable to richer global bases plus gating.



[1] HyperNetworks https://arxiv.org/abs/1609.09106 \
[2] A Brief Review of Hypernetworks in Deep Learning https://arxiv.org/abs/2306.06955 \
[3] NBEATS https://arxiv.org/pdf/1905.10437 \
[4] ESRNN https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153 \
[5] SKOLR https://arxiv.org/pdf/2506.14113 \
[6] Koopa https://arxiv.org/abs/2305.18803 \
[7] BasisFormer https://arxiv.org/abs/2310.20496

### Soundness
3

### Presentation
3

### Contribution
3
