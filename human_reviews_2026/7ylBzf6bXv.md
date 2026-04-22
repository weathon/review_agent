# Effective Probabilistic Time Series Forecasting with Fourier Adaptive Noise-Separated Diffusion

- Avg Score: 4.00
- Decision: Reject
- Scores: 0, 8, 6, 2

## Abstract
Existing diffusion-based time series forecasting methods often target on mixed temporal patterns or undifferentiated residuals, limiting the potential of distinct temporal components. In this paper, we propose the Fourier Adaptive Lite Diffusion Architecture (FALDA), a novel probabilistic framework for time series forecasting. FALDA leverages Fourier-based decomposition to incorporate a component-specific architecture, enabling tailored modeling of individual temporal components. A conditional diffusion model is utilized to estimate the future noise term, while our proposed lightweight denoiser, DEMA (Decomposition MLP with AdaLN), conditions on the historical noise term to enhance denoising performance. Grounded in rigorous mathematical proof, we introduce the Diffusion Model for Residual Regression (DMRR), a framework which methodologically unifies diffusion-based probabilistic regression method and theoretically demonstrate that FALDA effectively reduces epistemic uncertainty, allowing probabilistic learning to primarily focus on aleatoric uncertainty through further probabilistic analysis. Experiments on six real-world benchmarks demonstrate that FALDA consistently outperforms existing probabilistic forecasting approaches across most datasets for long-term time series forecasting while achieving enhanced computational efficiency without compromising accuracy. Notably, FALDA also achieves superior overall performance compared to state-of-the-art (SOTA) point forecasting approaches, with improvements of up to 9\%. The code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes FALDA, a Fourier-based diffusion framework for probabilistic time series forecasting. By decomposing sequences into non-stationary, stationary, and noise components, FALDA improves accuracy and uncertainty modeling. Built on the DMRR framework, it unifies diffusion-based regression and achieves state-of-the-art results with better efficiency and uncertainty separation.

The authors’ signal decomposition method requires further explanation — why are the components with larger amplitudes in the frequency domain considered non-stationary signals, and how can it be justified that the smaller-amplitude frequency components are purely noise?

The authors’ definition and understanding of non-stationary signals seem to differ from the mainstream perspective in signal processing, which needs further clarification or analysis.

The proposed top-K frequency selection method for signal decomposition may distort the overall characteristics of the signal. In signal processing, even using a rectangular window causes sidelobe effects, let alone selecting only a few top-K frequency components.

Many recent works have combined decomposition-based approaches with diffusion models for time series analysis, and the authors should provide a more detailed discussion and comparison with these studies.

### Strengths
see summary

### Weaknesses
see summary

### Questions
see summary

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents FALDA, a probabilistic time series forecasting framework that leverages Fourier-based decomposition to facilitate the tailored modeling of individual temporal components. The authors also propose a theoretical framework, DMRR, which unifies several existing diffusion-based regression methods and establishes the mathematical foundation for FALDA. Experiments conducted on six real-world datasets demonstrate the competitive performance of FALDA in both point forecasting and probabilistic forecasting.

### Strengths
- **S1** The paper is well-motivated, with clear writing and presentation that makes the technical contributions accessible. 
- **S2** The paper propose a theoretical framework DMRR, which mathematically classifies the existing diffusion-based time-series forecasting methods.
- **S3** As a plug-and-play method, FALDA shows broad applicability and can enhance existing forecasting methods.

### Weaknesses
- **W1** FALDA's Fourier decomposition process involves two key hyperparameters, K1 and K2. It appears that the selection of these hyperparameters varies across different datasets, which could potentially increase the practical deployment difficulty of the model. 
- **W2** Limited complexity experiments. The authors demonstrate the training and inference efficiency of FALDA in Appendix F.6, and Figure 6 shows its fast convergence property. However, this experiment is conducted only on the small-scale Exchange dataset. In addition, this section lacks comparisons with other DMRR-based approaches, such as D3U.

### Questions
1. Could the authors provide additional evidence in the complexity and efficiency analysis section—for example, comparisons of training and inference efficiency on larger-scale datasets and against other DDRM-based methods?
2. How do simpler forecasting approaches, such as Seasonal Naïve, perform on the datasets used in this paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the FALDA framework for multivariate time series (MTS) probabilistic forecasting. FALDA proposes a diffusion model approach based on residual regression, leveraging Fourier-transform-based time series decomposition to break down input sequences into stationary, non-stationary, and noise components, thereby enabling customized modeling of these components. This design helps FALDA reduce the model's epistemic uncertainty. Experimental results on six real-world datasets demonstrate that FALDA not only endows point forecasting models with probabilistic forecasting capabilities but also achieves competitive point forecasting performance overall.

### Strengths
- **S1** The FALDA framework reduces epistemic uncertainty​ by explicitly modeling time series components through ​time-series-decomposition-based modeling.
- **S2** The authors theoretically prove the equivalence​ between the ​DMRR (Diffusion Model with Residual Regression) scheme and CARD, strengthening the methodological foundation.

### Weaknesses
- **W1** The Fourier-based decomposition introduces two hyperparameters ($K_1$ and $K_2$)​. It appears that the selection of these hyperparameters varies across different datasets, which could potentially increase the practical deployment difficulty of the model. ​In new scenarios, determining optimal values for these hyperparameters is non-trivial, potentially limiting the model’s deployability.
- **W2** It is difficult to ensure that the Fourier-based sequence decomposition can effectively extract the ideal noise components in all cases. While the separation of stationary and non-stationary components generally works well and yields noticeable performance improvements, the approach does not consistently provide advantages across all datasets. In particular, on high-dimensional datasets such as Traffic and Electricity, FALDA shows limited gains in probabilistic forecasting, suggesting that the proposed decomposition method may not always be able to isolate the desired noise components effectively.

### Questions
1. I suggest that the authors include in the main text a more detailed discussion of the selection strategies for the Fourier-based decomposition hyperparameters ($K_1$ and $K_2$) and their impact on model performance with different values.
2. Compared with D3U, FALDA employs a more manual approach to extracting non-stationary components in time series, relying on Fourier-based decomposition. Quantitative results confirm that FALDA outperforms D3U in most experimental settings; however, in a few cases—particularly in probabilistic forecasting tasks—D3U still shows certain advantages. This might be because the proposed Fourier-based decomposition is less effective at capturing the uncertainty components of time series in those scenarios. How do the authors interpret and explain this phenomenon?
3. What is the rationale for choosing iTransformer as the default backbone? Will the implementation code be released publicly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FALDA (Fourier Adaptive Lite Diffusion Architecture), a diffusion-based probabilistic time series forecasting framework built on the unified DMRR formulation. By using Fourier decomposition to decouple non-stationary, stationary, and noise components, FALDA effectively separates epistemic and aleatoric uncertainties. The lightweight denoiser DEMA enhances efficiency and accuracy. Experiments show that FALDA outperforms existing diffusion-based and deterministic models in both point estimation and probabilistic forecasting.

### Strengths
1.	The paper is easy to follow and well written.
2.	The visual examples are complete and clearly presented.

### Weaknesses
1. I don’t understand how the paper separates epistemic uncertainty and aleatoric uncertainty. Epistemic uncertainty comes from the model’s limited predictive capability, which should reside in the residuals component in line 250, but the authors do not explicitly model different types of uncertainty within the residuals.
2. To my knowledge, due to the inherent non-stationarity of time series data, the associated uncertainty often exhibits temporal shifts. However, the authors only use noise as the guidance information during the denoising process, which does not account for the non-stationary characteristics. Moreover, Table 4 does not include the critical probabilistic forecasting metric CRPS, which undermines the persuasiveness of the experimental results.
2. Limited novelty — the method for non-stationary decomposition has already been proposed in FAN[1], and the residual decomposition has been applied in D3U[2].
3. For the probabilistic forecasting metric CRPS, the proposed method shows no clear improvement over D3U on most datasets, except for the ILI and Exchange datasets.

[1] Frequency Adaptive Normalization For Non-stationary Time Series Forecasting

[2] Diffusion-Based Decoupled Deterministic and Uncertain Framework for Probabilistic Multivariate Time Series Forecasting

### Questions
see weakness

### Soundness
1

### Presentation
3

### Contribution
1
