# FITS: Conditional Diffusion Model for Irregular Time Series Forecasting with Pseudo-future Exogenous Covariates

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Irregular multivariate time series (IMTS) present unique challenges due to non-uniform intervals and different sampling rates. While existing methods struggle to capture both long-term dynamics and cross-channel dependencies under such irregularities, we tackle this by formulating time series forecasting as a conditional generation problem and introducing FITS, a conditional diffusion model for IMTS forecasting that leverages pseudo-future exogenous covariates. Our approach incorporates two key innovations. First, we propose a novel density-aware adaptive patching scheme that generates data-driven segments with dynamic boundaries determined by the information density. This scheme overcomes the limitations of traditional fixed-length or fixed-span segmentation in preserving continuous local semantics and modeling inter-time series correlations. Second, we develop a transformer-based prior knowledge extractor that captures forward-looking covariate dependencies via a novel cross-variate attention mechanism. The transformer structure is integrated into the conditional diffusion generative process as a unified framework, enabling precise distributional forecasting for IMTS. Extensive experiments on six datasets with four evaluation metrics validate the effectiveness of FITS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes FITS, a conditional diffusion framework for irregularly sampled time series forecasting. By combining neural ODEs with time-aware diffusion, FITS models non-uniform temporal dynamics without resampling and achieves state-of-the-art accuracy and consistency across benchmark datasets.

The task proposed by the authors is very meaningful and represents a valuable innovation.

The model itself is moderately innovative, but the overall performance of the paper is strong.

There are still some issues with formatting and citations, which should be further improved.

It is recommended to include comparisons with time series imputation methods for a more comprehensive evaluation.

### Strengths
see summary

### Weaknesses
see summary

### Questions
see summary

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
3

### Summary
This paper frames time series forecasting as a conditional generation problem, using a diffusion model that incorporates both historical observations and exogenous covariates as conditional inputs. The approach is particularly tailored for IMTS, where traditional methods struggle with non-uniform intervals and varying sampling rates. The paper demonstrates that FITS outperforms state-of-the-art diffusion models and other advanced forecasting methods on several benchmark datasets, especially in probabilistic forecasting tasks.​

### Strengths
1. The entropy-aware patching scheme dynamically adjusts segment boundaries based on information density, preserving local semantics and improving modeling of inter-series correlations.​
2. FITS uses a transformer with cross-variate attention to capture forward-looking dependencies from exogenous covariates, enhancing the model's ability to forecast under irregularities.​
3. The model achieves state-of-the-art performance in probabilistic forecasting, as evidenced by lower CRPS and QICE scores on multiple datasets.​

### Weaknesses
1. FITS does not consistently outperform simpler models (like TiDE and DLinear) on long-term forecasting tasks, especially when inter-variable dependencies are weak.​
2. The model’s performance seems to rely on the availability and quality of exogenous covariates. In domains lacking such informative covariates or with noisy external data, the predictive gains may diminish or even worsen due to noisy conditioning.
3. How sensitive is the model to the entropy computation parameters (embedding dimension, tolerance)?
4. How does FITS perform when exogenous covariates are not included with

### Questions
See Weaknesses

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
2

### Summary
The paper introduces FITS, a diffusion model for irregular multivariate time series forecasting. Unlike previous work, they include external covariates from future times where the target data is not observed. The authors construct patches based on the entropy to embed historical context. To parametrize the reverse diffusion process, a neural network consisting of different attention mechanisms operating on the exogenous and endogenous data is used. Empirically, the model outperforms previous diffusion and transformer baselines.

### Strengths
- The work focuses on irregular time series, which are common in real-world settings but often neglected in previous works.
- A new architecture is proposed that is able to handle endogenous and exogenous time series.
- The empirical results demonstrate strong performance, especially in probabilistic forecasting.

### Weaknesses
- Standard deviations are not reported.
- Novelty is limited, and many changes are architectural changes.
- The setting, if understood correctly, only focuses on forecasting a single target variable. It would be interesting to see whether this can be extended to an arbitrary number of variables.
- It is unclear where the model makes use of the irregularity of the time series.

Minors:

- L231: transformer -> a transformer

### Questions
See weaknesses and:

- Can the method be extended to multiple target variables?
- How are exogenous variables included in the baselines?
- Can the method change the forecasting times, i.e., forecast the same number of steps but at different times?

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
4

### Summary
This paper proposes FITS (Conditional Diffusion Model for Irregular Time Series Forecasting with Pseudo-Future Exogenous Covariates), which aims to address the challenge of capturing long-term dynamics and cross-channel dependencies in Irregular Multivariate Time Series (IMTS) caused by non-uniform intervals and different sampling rates. The model overcomes the limitations of existing methods through two core innovations: first, an entropy-aware adaptive patching scheme that quantifies information density based on Sample Entropy (SampEn) and generates segments with dynamic boundaries via a Boundary Network (BoundaryNet), avoiding information fragmentation from traditional fixed-length patching; second, a transformer-based prior knowledge extractor that combines intra-series self-attention (to capture temporal dependencies) and inter-series cross-attention (with the target's global token as the query, and covariates split into historical/pseudo-future segments as keys/values) to capture forward-looking covariate dependencies, which is then integrated into the conditional diffusion generation process.

### Strengths
1. Clear elaboration of problem and innovation: The authors clearly elaborate on the challenges faced by existing works and the model designed to address these challenges.
2. Comprehensive experimental validation: The authors conduct extensive experiments, including tabular results, visualizations, and ablation studies, to verify the effectiveness of the proposed model, and provide objective and fair analysis of the experimental results.
3. Clear and complete paper structure with good readability and few errors: The manuscript follows a logical flow, making technical content accessible, and contains minimal grammatical or spelling mistakes (with LLMs used to refine language as noted).

### Weaknesses
1. Compared with traditional fixed patching, the Entropy-aware adaptive patching proposed by FITS introduces a more complex process. The authors only analyze its effectiveness but not its efficiency. Although the authors mention that the Boundary Net is a lightweight MLP, this is merely a qualitative description, lacking quantitative experimental analysis. Readers cannot judge the "effectiveness-efficiency trade-off" of this design.
2. FITS is intended to solve problems in IMTS caused by non-uniform intervals and different sampling rates. However, the irregularity of IMTS seemingly includes not only "target sequence sparsity" but also "covariate sparsity". FITS only addresses the former, which may result in a gap in achieving its intended goal.

### Questions
In the entropy-aware patching, how is the window size for calculating Sample Entropy (SampEn) determined? The paper mentions that the initial number of patches $P$ is determined by $T/S_{init}$, but it does not explain the basis for setting $S_{init}$. Will different window sizes affect the accuracy of information density quantification and thus change the patching effect?

### Soundness
3

### Presentation
3

### Contribution
3
