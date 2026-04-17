# Numerion: A Multi-Hypercomplex Model for Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Many methods aim to enhance time series forecasting by decomposing the series through intricate model structures and prior knowledge, yet they are inevitably limited by computational complexity and the robustness of the assumptions. Our research uncovers that in the complex domain and higher-order hypercomplex spaces, the characteristic frequencies of time series naturally decrease. Leveraging this insight, we propose Numerion, a time series forecasting model based on multiple hypercomplex spaces. Specifically, grounded in theoretical support, we generalize linear layers and activation functions to hypercomplex spaces of arbitrary power-of-two dimensions and introduce a novel Real-Hypercomplex-Real Domain Multi-Layer Perceptron (RHR-MLP) architecture. Numerion utilizes multiple RHR-MLPs to map time series into hypercomplex spaces of varying dimensions, naturally decomposing and independently modeling the series, and adaptively fuses the latent patterns exhibited in different spaces through a dynamic fusion mechanism. Experiments validate the model’s performance, achieving state-of-the-art results on multiple public datasets. Visualizations and quantitative analyses comprehensively demonstrate the ability of multi-dimensional RHR-MLPs to naturally decompose time series and reveal the tendency of higher-dimensional hypercomplex spaces to capture lower-frequency features.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Numerion, a time-series forecaster that maps inputs into multiple hypercomplex spaces (complex, quaternion, octonion, sedenion) and learns with a unified Real–Hypercomplex–Real MLP (RHR-MLP). The key idea is that higher-dimensional hypercomplex spaces naturally bias toward lower-frequency components, enabling a simple, parallel decomposition and an adaptive fusion head to combine per-space predictions. Across standard long-term forecasting benchmarks, Numerion reports SOTA or competitive results, and provides visual/quantitative analyses indicating specialization of higher dimensions for trends and lower dimensions for fluctuations. The method generalizes linear layers and tanh to arbitrary power-of-two dimensions and supplies ablations, sensitivity, and efficiency studies.

### Strengths
- This paper extends linear/activation layers to arbitrary hypercomplex dimensions and composes them into a parallel multi-space forecaster with adaptive fusion—distinct from typical decomposition or spectral methods.
- Offers a simple, backbone-agnostic path to multi-frequency modeling without heavy architectural complexity; could inspire broader use of hypercomplex representations for TS.
- Comprehensive experiments across 10 datasets with consistent improvements and honest discussion where variable-fusion models can outperform on high-dimensional data; includes ablations, visualization, and sensitivity.

### Weaknesses
- This paper lacks comparisons against strong, explicit frequency or time–frequency baselines, such as FBM[1], to support the claimed advantage brought by operations in higher-dimensional hypercomplex spaces. 

- It remains unclear whether improvements stem from genuinely hypercomplex algebra or simply from vector factorization at dyadic sizes ($n \in$ \{$1,2,4,8,16$\}). The Cayley–Dickson construction does not mathematically guarantee a low-pass (frequency-selective) behavior; thus the observed “low-frequency preference” might arise from architecture/parameterization effects rather than algebraic properties. The authors may add controls that replace hypercomplex layers with *grouped real MLPs* (same splits $n $); also visualize learned frequency responses to disambiguate where the bias truly comes from.


[1] Yang, R., Cao, L., & YANG, J. (2024). Rethinking Fourier transform from a basis functions perspective for long-term time series forecasting. Advances in Neural Information Processing Systems, 37, 8515-8540.

### Questions
How sensitive is Numerion to patch levels/embedding dims under noisy domains (e.g., Exchange, where MAE improvement is only 0.005), beyond the Electricity study?

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
This paper introduces Numerion, a novel framework for long-term time series forecasting that leverages multi-hypercomplex algebraic representations to model complex inter-channel dependencies and cross-frequency interactions. The core idea is to embed different temporal and channel-wise features into multiple hypercomplex spaces, and use hypercomplex convolutions and mixing operations to enhance expressive power while maintaining compactness.

### Strengths
It is the first work to systematically apply multi-hypercomplex algebra to time series forecasting and provides a unified representation that captures both amplitude and phase interactions across channels.
It outperforms DLinear, TimesNet, PatchTST, and other baselines on multiple benchmarks.
This paper includes comparisons across different algebraic spaces, and studies of layer depth and hypercomplex dimensionality. The ablation results clearly isolate the contribution of each module.

### Weaknesses
How are the results of PatchMLP and TimeMixer in Table 1 obtained? According to the original papers of PatchMLP [1] and TimeMixer [2], there is an excessively large gap between their results and those presented in your paper. Additionally, Numerion's performance on some datasets is far from achieving the state-of-the-art performance when compared to PatchMLP. For example, on the Weather dataset, Numerion achieves an MSE of 0.246 and an MAE of 0.271, while PatchMLP yields an MSE of 0.231 and an MAE of 0.256. On the Solar-Energy dataset, your model’s results are an MSE of 0.252 and an MAE of 0.262, whereas PatchMLP achieves an MSE of 0.211 and an MAE of 0.261. Besides, Numerion also underperforms PatchMLP on the two datasets of ETTh2 and ETTm2.
Although the paper mentions the properties of partial hypercomplex multiplication in the appendix, it lacks a systematic theoretical analysis to explain why the multi-hypercomplex space can improve prediction performance.
While the model is parameter-efficient, hypercomplex multiplication scales quadratically with algebra dimension. Can the authors provide explicit complexity formulas?


[1] Tang P, Zhang W. Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(12): 12640-12648.
[2] Wang S, Wu H, Shi X, et al. Timemixer: Decomposable multiscale mixing for time series forecasting[J]. arXiv preprint arXiv:2405.14616, 2024.

### Questions
As in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Numerion, which achieves multi-frequency decomposition and prediction by mapping time series to multiple hypercomplex spaces. The core innovation lies in generalizing linear layers and Tanh activation functions to hypercomplex spaces of arbitrary power-of-two dimensions and designing the RHR-MLP architecture. The paper also provides extensive ablation experiments, visualizations, and frequency analyses to validate the effectiveness of the Numerion model.

### Strengths
1. The authors' idea of achieving decomposition by mapping time series data to hypercomplex spaces of different dimensions is very inspiring.
2. The authors provide a mathematical formulation of the Numerion framework based on the Cayley-Dickson algebra system. They use the Cayley-Dickson construction and random matrix theory principles of spectral bias as the theoretical foundation of the model. This is very novel and interesting.
3. The paper provides extensive comparative and ablation experiments, visualizations, and frequency-domain analyses to support their claims. The experimental design is relatively complete, and the results are clear.

### Weaknesses
1. There are multiple instances of missing spaces throughout the manuscript, such as after model names.
2. The mathematical formula formatting needs revision, as punctuation marks (commas or periods) are missing at the end of equations.
3. While the authors claim that experimental results are averaged over multiple runs, standard deviations are not reported.
4. The authors only analyze the training time of the model but lack analysis of time complexity.
5. It is suggested that the authors move critical theoretical analyses to the main text to improve readability.
6. The authors acknowledge that the hypercomplex linear layers do not satisfy the hypercomplex differentiability conditions, and the gradients are not strictly hypercomplex. Does this imply that the backpropagation process is actually performed in the real domain? What are the implications of this approach for model optimization and convergence?
7. The authors acknowledge that hypercomplex networks consume significantly more memory than real-valued layers. Have the authors considered approximation methods (e.g., low-rank decomposition, sparsification) to reduce computational costs?

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes $\text{Numerion}$, an $\text{MLP}$-based model for time series forecasting. Its core idea is that mapping time series into higher-order hypercomplex spaces ($\mathbb{C}$, $\mathbb{H}$, $\mathbb{O}$, $\mathbb{S}$) naturally decomposes the signal by frequency; higher dimensions capture lower frequencies. The method generalizes linear layers and $\tanh$ to create a $\text{Real-Hypercomplex-Real MLP}$ ($\text{RHR-MLP}$). The $\text{Numerion}$ model runs parallel $\text{RHR-MLPs}$ from $\mathbb{R}^{1}$ to $\mathbb{R}^{16}$ and fuses their outputs. The concept is highly novel but undermined by contradictory ablations and severe efficiency limitations. I recommend a Score 4 (Marginally Below Threshold).

### Strengths
S1. The core idea is using the algebraic hierarchy ($\mathbb{R} \to \mathbb{C} \to \mathbb{H} \to \mathbb{O} \to \mathbb{S}$) as an implicit frequency decomposition mechanism. This is a novel and elegant approach, distinct from traditional methods.

S2. The paper formally generalizes linear layers and activations to $n$-dimensional hypercomplex spaces. The $\text{Hypercomplex Linear Layer}$, $O^{(n)} = {W^{(n)}}^{T}X^{(n)} + b^{(n)}$, and the $\text{Hypercomplex Norm Tanh}$ ($\text{HNTanh}$) activation create a unified framework based on the $\text{Cayley-Dickson construction}$.

S3. The central hypothesis—higher dimensions capture lower frequencies—is well-supported by visualizations and quantitative analysis. Appendix L shows higher-dimensional $\text{RHR-MLPs}$ have lower $\text{Mean Absolute Frequency}$ ($\text{MAF}$) and higher $\text{Pearson correlation}$, indicating better trend-fitting, while the $\text{Real-MLP}$ handles high frequencies.

### Weaknesses
W1. Contradictory Ablation Study (Major Flaw)The ablation in Table 2 contradicts the paper's claims. On $\text{ETTh1}$, the full $\text{Numerion}$ model ($\text{MSE}$ $0.449$, $\text{MAE}$ $0.449$) performs worse than or equal to variants where components are removed (e.g., "w/o Real", "w/o Adaptive Fusion"). This contradicts the text stating these components are essential and suggests they are not synergistic, at least on this dataset. This is a major red flag.

W2. Severe Practical and Efficiency LimitationsThe paper admits to severe inefficiency. Lacking native $\text{PyTorch}$ support, operations are simulated. This causes extreme memory overhead (a sedenion layer is $16\text{x}$ larger), parameter inflation (5 layers become 63), and slow training/convergence. Efficiency tables confirm $\text{Numerion}$ is much slower and larger than $\text{iTransformer}$, $\text{PatchTST}$, or $\text{DLinear}$, making it impractical.

W3. Vague Justification for DecompositionThe paper shows decomposition happens but not why. The theory in Appendix M is a hypothesis linking $\text{Random Matrix Theory}$ ($\text{RMT}$) and stable rank reduction to a "low-pass bias". This is an insightful post-hoc explanation, not a formal proof. The causal link between $\text{Cayley-Dickson}$ rules and frequency filtering is not established.

### Questions
Q1. Please explain Table 2. Why does removing components like "w/o Real" or "w/o Adaptive Fusion" lead to equal or better performance on $\text{ETTh1}$? This suggests these parts are not contributing.

Q2. Given the $16\text{x}$ memory cost, parameter inflation, and slow speed, how is $\text{Numerion}$ practically useful compared to efficient $\text{SOTA}$ models like $\text{iTransformer}$ or $\text{DLinear}$?

Q3. The choice of $p=6$ for $\text{HNTanh}$ seems arbitrary. Table 10 shows $p=\text{inf}$ is best for $\text{ETTh1}$ and $p=7$ for $\text{ETTh2}$. How was $p=6$ chosen, and how sensitive is the model to this value?

Q4. Figure 4 shows $\text{MSE}$ on $\text{ETTm2}$ improves as layers increase from 1 to 5, suggesting underfitting. How does this align with the text claiming the model "prefers fewer layers... [to avoid] overfitting"?

### Soundness
2

### Presentation
3

### Contribution
3
