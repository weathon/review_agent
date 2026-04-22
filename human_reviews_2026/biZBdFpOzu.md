# Neural CDEs as Correctors for learned time series models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Learned time-series models, whether continuous- or discrete-time, are widely used to forecast the states of a dynamical system. Such models generate multi-step forecasts either directly, by predicting the full horizon at once, or iteratively, by feeding back their own predictions at each step. In both cases, the multi-step forecasts are prone to errors. To address this, we propose a Predictor-Corrector mechanism where the Predictor is any learned time-series model and the Corrector is a neural controlled differential equation. The Predictor forecasts, and the Corrector predicts the errors of the forecasts. Adding these errors to the forecasts improves forecast performance. The proposed Corrector works with irregularly sampled time series and continuous- and discrete-time Predictors. Additionally, we introduce two regularization strategies to improve the extrapolation performance of the Corrector with accelerated training. We evaluate our Corrector with diverse Predictors, e.g., neural ordinary differential equations, Contiformer, and DLinear, on synthetic, physics simulation, and real-world forecasting datasets. The experiments demonstrate that the Predictor-Corrector mechanism consistently improves the performance compared to Predictor alone.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a general predictor–corrector framework for enhancing the performance of learned time-series models. The corrector is implemented via a Neural Controlled Differential Equation that models the dynamics of prediction errors driven by the predictor’s forecast trajectories, augmented with two regularization strategies to improve extrapolation. Experiments across diverse datasets demonstrate that the proposed method consistently improves the performance of baseline predictors.

### Strengths
1.  The writing is clear.
2.  The proposed method is easy to understand and implement. It can be integrated with any time series models conveniently.

### Weaknesses
1. There are potential fairness concerns in the experimental comparisons, particularly in the LTSF section. The study primarily contrasts DLinear augmented with the proposed Corrector against various Transformer baselines, but does not evaluate the Corrector when attached to strong Transformer models. Transformer-based architectures such as iTransformer [1] and PatchTST [2] already outperform DLinear; consequently, it remains unclear whether the proposed method can still deliver gains when applied to competitive state-of-the-art models.
2. Since the additional NCDE is introduced as the corrector, it is important to compare against alternative baselines to substantiate the advantage of NCDE. For example, what performance can be achieved by augmenting ContiFormer with an additional linear head to model the residuals? Does NCDE consistently outperform such a simple baseline? Moreover, within the continuous-time paradigm, the paper does not compare against other plausible continuous-time correctors (e.g., using ODE-RNN [3], GRU-ODE [4], or Latent SDE [5] to model the error dynamics).
3. Insufficient theoretical analysis: While the motivation is intuitive, the paper lacks a formal treatment of the key assumption that “the error dynamics are sufficiently controlled by the forecast trajectories.” Can relevant literature be cited to support this premise? Why are forecast trajectories used as the sole control signal rather than augmenting the control with “forecasts + observations” or “forecasts + uncertainty estimates” (e.g., predictive variance)? Moreover, there is no analysis of stability, error propagation bounds, or conditions for extrapolative generalization. Could the authors provide sufficient conditions (e.g., under Lipschitz continuity or local linearization) ensuring convergence or boundedness of the error CDE, or derive upper bounds on the corrected forecast error?
4. If the predicted residual has the opposite sign to the true residual, the error may be further amplified. Are there regimes in which the Corrector systematically amplifies errors (e.g., specific phase regions of chaotic systems)? How pronounced is the impact of such cases on overall performance?

---
[1] Liu, Yong, et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." The Twelfth International Conference on Learning Representations.

[2] Nie, Yuqi, et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." The Eleventh International Conference on Learning Representations.

[3] Rubanova, Yulia, Ricky TQ Chen, and David K. Duvenaud. "Latent ordinary differential equations for irregularly-sampled time series." Advances in neural information processing systems 32 (2019).

[4] De Brouwer, Edward, et al. "GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series." Advances in neural information processing systems 32 (2019).

[5] Li, Xuechen, et al. "Scalable gradients for stochastic differential equations." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

### Questions
Pls refer to Weaknesses

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Predictor-Corrector mechanism to address the problem of error accumulation in multi-step time-series forecasting. The core idea is to treat any pre-trained time-series model as a Predictor and use its output forecasts to drive a Corrector model. This Corrector is implemented as a Neural Controlled Differential Equation (Neural CDE).

The mechanism works as follows:
1. The Predictor generates a forecast trajectory.
2. This forecast trajectory is used as the continuous control path for the Neural CDE.
3. The Neural CDE is trained to predict the error (the residual between the Predictor's forecast and the ground truth).
4. The final, corrected forecast is the sum of the original prediction and the CDE's predicted error.

The authors demonstrate that this framework can be applied to diverse Predictors, including continuous-time (NODE, ContiFormer) and discrete-time (DLinear) models, and works on both regular and irregular data. The method is validated on synthetic, physics simulation, and real-world LTSF datasets, showing consistent performance improvements over the Predictor-alone baselines. The paper also introduces two regularization strategies, "Variable-Length Control Paths" and "Sparse Control Paths," to improve the Corrector's extrapolation capabilities.

### Strengths
1. The primary strength of this work is its agnosticism to the Predictor model. The ability to use the Corrector as a plug-and-play module to improve the performance of existing, pre-trained models—ranging from NODE to ContiFormer and even the non-continuous DLinear—is a significant practical advantage.
2. The paper provides a comprehensive set of experiments across synthetic data (Lorenz, FHN) , high-dimensional physics simulations (MuJoCo) , and challenging real-world LTSF benchmarks (Weather, Exchange, etc.). The consistent improvement in MSE/MAE across all these settings (e.g., in Table 1, 2, and 4) makes a strong case for the method's effectiveness.
3. The introduction of Variable-Length and Sparse Control Paths  is a clever and well-reasoned contribution. These regularizers directly address a known weakness of neural differential equations (extrapolation) and are shown to improve generalization while also accelerating training by reducing NFEs (as shown in Figure 4).

### Weaknesses
1. Over-reliance on the Decoder's Expressiveness: The entire framework hinges on the assumption that the CDE's latent state $z(t)$, controlled by the forecast path $X(s)$, will capture the necessary information to predict the error $e(t)$. However, the paper provides no structural or theoretical guarantee for this. The connection between $z(t)$ and $e(t)$ is established solely by a very powerful 4-layer MLP decoder ($FC(400)_4$). This design feels contrived. It is difficult to ascertain whether the CDE is truly learning the "error dynamics" or if it is merely acting as a complex feature extractor, with the heavy-lifting (i.e., fitting the complex $z$-to-$e$ mapping) being done by the high-capacity MLP decoder.

2. Limited Methodological Novelty: The core idea of using continuous-time neural models (NODEs/CDEs) to address shortcomings in time-series forecasting is not entirely unique. For instance, the provided paper by Jhin et al. (2024, "CONTIME") also employs a continuous-time (NODE-based GRU) architecture to specifically solve the problem of prediction delays in time-series models. While this paper's application (post-hoc error magnitude correction) is different from CONTIME's (end-to-end delay correction), the foundational concept of using CDEs/NODEs to fix a known flaw in time-series prediction is very similar. This concurrent work suggests that this methodological tool is a "next logical step" in the field, which somewhat diminishes the paper's claim to high conceptual novelty.

### Questions
1. Following Weakness #1: Could the authors provide an ablation study on the complexity of the decoder ($\xi_{\varphi}$)? Specifically, how does the Corrector's performance change if the 4-layer $FC(400)_4$ decoder is replaced with a simpler linear layer, or a 1-layer MLP? This would be crucial to disentangle the contribution of the CDE's dynamic modeling from the powerful function approximation capabilities of the decoder.
2. Following Weakness #2: Given that other concurrent work (e.g., Jhin et al., 2024) also uses continuous-time models (NODEs) to address related problems like prediction delay, how do the authors position the conceptual novelty of their contribution? Is the novelty in the idea of using CDEs for time-series, or purely in the application as a post-hoc correction module?

### Soundness
2

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
The paper proposes a generic predictor–corrector scheme for time-series forecasting: a trained Predictor (e.g., NODE, ContiFormer, DLinear) produces a rollout, and a Neural CDE Corrector learns the residual trajectory and adds it back to improve forecasts. Research in Predictor corrector is important for many real world use cases. The authors claim the approach is model-agnostic (works for continuous- and discrete-time predictors, regular and irregular sampling) and introduce two “control-path” regularizers purported to improve extrapolation and reduce NFEs (function evaluations). Experiments span synthetic ODE systems, MuJoCo dynamics, and LTSF benchmarks; the paper reports consistent gains over the base predictors (with a few exceptions) and emphasizes irregular-sampling compatibility.

### Strengths
Clear, modular residual-learning formulation for error trajectories with a continuous-time model (Neural CDE) and a simple MSE loss; training/inference diagrams and algorithm are easy to follow.

Claimed agnosticism to predictor class and sampling regime is appealing for practitioners. 

Order-agnostic correction. By modeling residual dynamics with a latent first-order CDE driven by a rich control path, the corrector can, in principle, compensate errors from higher-order systems (e.g., 2nd–4th order) without changing the predictor, provided sufficient history/derivative channels and capacity.

The  regularization ideas (variable-length / sparse control paths) presented seems intuitive and deployment friendly.  

Broad empirical scope (synthetic, MuJoCo, and LTSF) demonstrates that the template can be dropped on several predictors.

### Weaknesses
Related Work / Positioning (major)

The paper omits central predictor–corrector lines that already cast learning as predict + update (observer/assimilation) or learned residual correction; directly overlapping the proposed idea. Some of the recent related works not cited/discusses/compared against:

PhyDNet (CVPR 2020): states that PhyCell is based on a prediction–correction paradigm inspired by data assimilation, disentangling a physics branch (predictor) from a residual branch (learned corrector) for video dynamics. 

 Neural Koopman Prior for Data Assimilation (TSP 2024): a learned Koopman dynamical prior inside a variational data-assimilation loop; a learned predictor with correction via assimilation. 

KalmanNet (IEEE TSP’22): NN-aided Kalman filtering that learns the gain/updates under partial/unknown dynamics; canonical “predict–update” with learning.

Semilinear Neural Operators: a unified recursive framework for prediction and data assimilation using observer design (ICLR 2024); explicitly combines prediction and correction operations for long-horizon PDEs with irregular/noisy measurements. 

Predictor-Corrector Enhanced Transformers with Exponential Moving Average Coefficient Learning (Neurips 2024).

Related-Work paragraph on “Correctors in Forecasting” instead cites older ARIMA+ML hybrids and domain-specific error modeling, then claims novelty for a general corrector (Neural CDE) that “works for irregularly sampled time series.” This fails to acknowledge the modern ML-DA, observer, PhyDNet, and Koopman-assimilation literatures that already instantiate predictor–corrector at scale and in precisely the regimes you claim.

Conceptual & Technical gaps : 

The choice to use forecasts as the sole control path is under-motivated. Why is this preferable to (i) augmenting with measurements/exogenous inputs, (ii) an observer-style gain (as in NODA), or (iii) a latent residual branch akin to PhyDNet? Provide analysis/theory for when a CDE-driven error process is identifiable/advantageous.

The paper asserts the corrector “works with irregularly sampled time series” but does not formalize stability or long-horizon behavior of the autoregressive correction (error accumulation remains a concern). A rigorous statement or bound (even empirical stability diagnostics) is missing.

The regularizers are framed as improving extrapolation and reducing NFEs, but this reads as empirical observation rather than a principle (no formal link between sparsity of the control path and generalization).

Weaknesses in experiments (Mostly Minor points)  :

Compute claims without measurements: You claim accelerated training from reduced NFEs; only qualitative NFE curves are shown. Provide wall-clock, solver tolerances, and NFEs for Predictor and Corrector (train & inference) against baselines.

Irregular sampling story is thin: While you claim compatibility with irregular sampling, the strongest empirical gains are on regular LTSF; the paper lacks strong irregular baselines (e.g., GRU-ODE-Bayes, ODE-RNN with assimilation updates) under identical missingness protocols.

Limited uncertainty reporting: Most tables lack std/CI across seeds; even the large LTSF table is point estimates only. ICLR standards call for robust seeds and uncertainty, especially with highly stochastic MuJoCo and long-horizon forecasts.

### Questions
Order-agnostic claim under-substantiated. The ability to correct higher-order dynamics hinges on observability (what signals the corrector sees), sampling rate, control-path engineering, and model capacity; the paper provides neither a formal analysis nor targeted experiments (e.g., explicit 4th-order regimes) to validate this claim.

How does a CDE-based corrector fundamentally differ from a more traditional observer-style update or kalman filter style correction? What advantages (expressivity, stability, identifiability) does control-path formulation have over these traditional approaches? Citing these works and discussing the contrasting advantage could enhance the positioning of the work.

From the looks of it the proposed method seems order agnostic for the underlying ODE. In my opinion a test that shows robustness/scalability to different order of ODEs will really help. 

Evaluate against strong irregular baselines (e.g., ODE-RNN/GRU-ODE-Bayes) under matched missingness; does the CDE-corrector still win?

Do long-horizon rollouts exhibit bounded error under iterated correction? Provide diagnostics or a proposition delineating conditions for stability.

Since you correct forecasts, do you model predictive uncertainty or calibration? If not, how does the corrector affect coverage under conformal/quantile evaluation?

Ad-hoc training horizon choices: For LTSF you vary the corrector training length (50/100/…/336) and then use intermediate windows because very short/very long horizons hurt performance. This looks like post-hoc tuning to the sweet spot and should be replaced with a fixed, pre-registered protocol plus sensitivity/ablation.

In summary if the authors can improve their related work section adding discussions wrt both recent and traditional literature on different predictor corrector frameworks, can actually show how their corrector model behaves for different order of ODEs and observability it could significantly improve the overall contribution of their paper positively. In contrast, without these the paper claims seems overstated and lacks proper literature placement.

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
4

### Summary
This paper introduces a **Predictor–Corrector** framework for time-series forecasting in which any learned model (continuous- or discrete-time; e.g., NODE, ContiFormer, DLinear) acts as the **Predictor**, and a **Neural Controlled Differential Equation (Neural CDE)** serves as the **Corrector** that learns and adds back the Predictor’s **error dynamics**. The Corrector treats the Predictor’s multi-step forecast trajectory as a control path; a decoder maps the Neural CDE hidden state to residuals that are added to forecasts. Two regularizations aim to improve extrapolation and reduce training cost: **Variable-Length Control Paths** (randomly truncating the control path) and **Sparse Control Paths** (subsampling knots). Evaluations span **synthetic ODE systems** (Lorenz, Lotka–Volterra, FHN, Glycolytic), **MuJoCo physics** (Hopper, Walker2D, Pen, Hammer), and **LTSF** datasets (Exchange, ETTm2, ETTh2, ILI, Weather). Results show consistent MSE reductions versus base Predictors in interpolation and sustained improvements up to long horizons in several settings (e.g., DLinear gains on Exchange; ContiFormer gains on Walker2D/Pen). The study reports ablations on κ/η and the number of function evaluations (NFEs), indicating accuracy–extrapolation–efficiency trade-offs.

### Strengths
- **Originality.** This paper reframes residual correction as **continuous-time error-dynamics learning** using Neural CDEs driven by the Predictor’s trajectory, generalizing classic hybrid error-modeling beyond discrete, regularly sampled data.
- **Technical quality.** This paper presents a clear formulation (spline-based control paths, adjoint training, simple decoder) and systematic ablations of the κ/η regularizers, exposing accuracy–extrapolation–efficiency trade-offs (e.g., NFEs).
- **Clarity & reproducibility.** This paper clearly separates interpolation vs. extrapolation settings, provides algorithmic details and hyperparameter tables, and outlines data generation and training choices for replication.
- **Significance.** This paper shows consistent gains across heterogeneous Predictors (NODE, ContiFormer, DLinear) and domains (synthetic, physics, LTSF), indicating a practical, plug-in corrector for long-horizon forecasting—especially under irregular sampling.

### Weaknesses
- **Baselines for discrete tasks.** LTSF comparisons omit strong *residual* correctors (e.g., MLP/TCN residual heads, diffusion/consistency-style calibrators), making it unclear when a CDE-based corrector is strictly preferable.
- **Stability theory.** This paper lacks a formal analysis of stability/consistency for the closed-loop Predictor–Corrector under long-horizon rollouts; success criteria for extrapolation are largely empirical.
- **Efficiency reporting.** Beyond NFEs, this paper does not report wall-clock time, memory usage, or parameter counts versus simpler correctors, limiting claims about training/serving efficiency.
- **Predictor breadth.** Results are limited to NODE/ContiFormer/DLinear; adding RNN/TCN/graph forecasters would better substantiate the architecture-agnostic claim.
- **Interpretability.** This paper provides a limited qualitative analysis of learned corrections (e.g., whether they address phase drift, bias, or scale errors).

### Questions
1. **When is a CDE needed?** Please compare against residual MLP/TCN heads and diffusion/consistency-style correctors on LTSF to delineate regimes where the CDE brings clear advantages.
2. **Stability & robustness.** Provide long-horizon stress tests and sensitivity to interpolation schemes and ODE solvers; consider theoretical bounds or empirical diagnostics for divergence/overshoot.
3. **Training paradigm.** Compare two-stage vs. joint end-to-end (or alternating) training; discuss bias–variance–stability trade-offs and whether joint optimization improves extrapolation.
4. **Cost metrics.** Report runtime, memory, and parameter counts; include κ/η vs. accuracy **Pareto curves** to substantiate efficiency claims for the proposed regularizers.
5. **Predictor diversity & non-stationarity.** Evaluate on RNN/TCN/GRU-ODE-Bayes/graph predictors and regime-shifted splits; consider uncertainty/coverage metrics.
6. **Qualitative insight.** Visualize per-dimension corrections on real datasets to reveal which systematic Predictor biases are being fixed.

### Soundness
3

### Presentation
3

### Contribution
3
