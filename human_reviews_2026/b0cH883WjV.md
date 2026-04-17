# Conditional Guided Flow Matching: Modeling Prediction Residuals for Enhanced Time Series Forecasting

- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Time series forecasting predominantly focuses on modeling the mapping between historical and future sequences, and existing improvements are often constrained to optimizing model architectures to better capture this relationship. This essentially reduces prediction residuals to mere optimization targets while overlooking their informative structures such as systematic biases or nontrivial distributions that could otherwise be exploited to directly reduce forecasting errors. Unfortunately, discriminative models struggle to capture the complete residual structure and its dynamic temporal dependencies when applied to residual learning. To fill this gap, we introduce Conditional Guided Flow Matching (CGFM), a novel framework built upon flow matching. CGFM innovatively leverages auxiliary predictions as the source distribution and constructs two-sided conditional paths to prevent path crossing, which enables the explicit learning of the full structure of prediction residuals and thereby theoretically guarantees superior performance over discriminative models. Extensive experiments show that CGFM enhances diverse forecasting models including
state-of-the-art ones and demonstrates its effectiveness and generality. Code link: \url{https://anonymous.4open.science/r/CGFM-31DB}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Conditional Guided Flow Matching, CGFM, a Flow Matching-based framework that enhances time series forecasting by explicitly modeling the full probabilistic structure of prediction residuals. It addresses the limitation of traditional methods, namely treating residuals as mere optimization targets, via two-sided conditional paths and affine parameterization that ensures non-crossing paths, achieving consistent improvements across diverse base models and datasets.

### Strengths
1. The paper is easy to understand, with Figure 1 particularly intuitively illustrating the method’s core principle by visualizing residual distribution changes before and after CGFM learning.

2. The work addressing an often-overlooked information source in time series forecasting: unlike traditional methods that treat prediction residuals as passive optimization targets, CGFM explicitly leverages residuals’ informative structures o refine forecasts, with Proposition 4.2 formally linking CGFM to residual distribution learning via Flow Matching and grounding this innovation in rigorous mathematics.

3. The framework exhibits strong model-agnostic flexibility and delivers convincing empirical results: it acts as a universal enhancement module that integrates seamlessly with diverse existing forecasting models. Table 1 validates this with consistent, significant improvements over strong baselines

4. The paper achieves high theoretical rigor and comprehensive experimental design: it provides thorough theoretical analysis, including well-structured propositions and proofs that validate key design choices, while conducting thorough experiments across 7 benchmark datasets using MSE, MAE, and CRPS metrics. Its comprehensive ablation studies and full discussion of computational overhead facilitate a clear understanding of each component’s role.

### Weaknesses
1. The paper only compares CGFM with a simple discriminative residual corrector (RLinear-based) and lacks comparisons with advanced state-of-the-art residual modeling methods. This makes it unclear whether CGFM’s advantages stem from its flow matching-based design or merely from the utilization of residual information, failing to isolate the unique value of its core architectural innovations.

2. It remains unknown whether CGFM can still effectively optimize residuals when the source distribution (auxiliary predictions) is drastically misaligned with the target distribution. This gap limits our understanding of the "lower bound" of CGFM’s applicability, especially in scenarios where high-quality auxiliary models (that generate source distributions close to the target) are unavailable.

3. Although the framework demonstrates model-agnostic flexibility, it lacks an in-depth investigation into the causes of performance improvement disparities across different base models.

### Questions
1. Could the authors supplement experiments comparing CGFM with advanced residual modeling baselines? Such comparisons would help verify whether CGFM’s performance gains are driven by its unique flow matching design rather than the general paradigm of residual utilization.

2. Could the authors design experiments where the auxiliary model generates severely misaligned source distributions (e.g., near-random predictions, predictions with persistent large biases)? Testing CGFM’s residual optimization ability in such scenarios would clarify its "lower bound" of applicability and practical robustness.

3. Could the authors conduct additional analyses linking base model architectural characteristics to residual properties?

### Soundness
4

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
This paper investigates a conditional flow matching framework for time series forecasting. To mitigate the biased results in previous forecasting methods, they design a post-hoc residual correction algorithm with flow matching. They empirically validated the strength of their approach across various settings, including different configurations and datasets.

### Strengths
- The paper investigates residual correction using flow matching.
- The authors also provide a theoretical framework in the appendix in detail.
- Their method performs well in various settings and against baselines.

### Weaknesses
Please refer to the Questions section.

### Questions
- When applying the suggested framework, training both the deterministic forecasting model and the flow matching model is needed. The authors need to provide a specific analysis of the computational burden.
- If the deterministic forecaster is designed to match the MSE (or maybe the first order, as the authors explained), why does the distribution of the residual in the deterministic forecaster not show an unbiased result in Figure 1 (right) or the real data in Figure 5?
- In Proposition 4.1, noise smoothing, the addition of Gaussian noise is for smoothness. The reviewer believes that the selection $\sigma$ is important for training. How do the authors select it? Is it data-dependent or prediction-dependent? Please provide an ablation study for.
- The reviewer believes that using flow matching for residual prediction in time series forecasting has not been well explored before. However, the idea of using a ‘generative model’ to adjust the guidance of deterministic neural networks is well-developed, both in time series and other domains. The authors should include these references (maybe find more). 

[1] CARD: Classification and Regression Diffusion Models, Neurips 22. 

[2] TimeBridge: Better Diffusion Prior Design with Bridge Models for Time Series Generation, https://arxiv.org/abs/2408.06672

- Why is the path linearity important for residual prediction?
- How did the authors design the architecture (A.15) and why?
- Still, the reviewer has concerns about the improvement; thus, I have two suggestions. First, show improvement over more recent methods (such as methods from recent AI conferences). Second, use a comparison like: BASE | BASE+Diffusion | BASE+Diffusion Bridge | BASE+Flow matching, to compare and demonstrate that the strength actually comes from flow matching.
- What does Figure 4 represent? Does it show the difference in the residual after using flow matching?

### Soundness
3

### Presentation
4

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
This paper proposes Conditional Guided Flow Matching (CGFM), a framework that applies flow matching to learn prediction residuals in time series forecasting. CGFM uses an auxiliary model's prediction distribution as the source distribution, instead of guassian distribution.

### Strengths
1.	Using the auxiliary model's predictions as the source distribution (rather than noise) is intuitive and well-motivated. 
2.	Provide different target parameterization and loss design, and through experiments, demonstrate the optimal design.

### Weaknesses
1. The novelty is not clearly presented. This paper applies flow matching to time series prediction, and the theorems and corollaries in the paper are very similar to flow matching, such as proposition 4.6. 
2. Proposition 4.2 claims are trivial.
The claim that CGFM "equivalently learns residual ε = x₁ - x₀'s probabilistic characteristics" is trivial. The proof (Appendix A.5) shows that X_t = (α_t + β_t)X₀ + α_t ε, which means the path involves the residual ε. However, this is trivially true for ANY flow matching between X₀ and X₁ = X₀ + ε. This is not a unique property of CGFM—it's just a restatement of the fact that you're transforming from predictions to targets. The "equivalence" adds no new insight beyond standard flow matching theory.
3. The presentation can be improved. 
 - The symbol F in Chapter 4.1 represents target future data, but it appears again in the superscript. 
 - Many symbols lack explanations. 
 - Fig. 1 shows the residual distribution changed after CGFM learning. Why has it changed? What do the icons in Fig. 4 represent?
4. The experimental evaluation can benefit from a comparison with other residual prediction models.

### Questions
1.	Can you provide a comparison with other residual prediction models?
2.	What are the technical contributions, except for adopting flow matching?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes the CGFM method, which aims to model the part of the prediction residuals. The authors also conduct partial validation on several methods in an attempt to demonstrate its effectiveness.

### Strengths
S1: The motivation of the paper is reasonable.

S2: The paper provides derivations and theoretical support.

### Weaknesses
W1：The literature review in this paper is seriously inadequate, as it omits several important baseline methods in its survey and comparisons, including linear model–based methods (e.g., TQNet [1], CycleNet [2]), TCN-based models (e.g., ModernTCN [3]), RNN-based methods (e.g., SegRNN [4], WITRAN [5], PGN [6]), and Transformer-based methods (e.g., Leddam [7]).

[1] Temporal Query Network for Efficient Multivariate Time Series Forecasting. In Forty-second International Conference on Machine Learning.

[2] CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

[3] ModernTCN: A modern pure convolution structure for general time series analysis. In The Twelfth International Conference on Learning Representations.

[4] SegRNN: Segment recurrent neural network for long-term time series forecasting.

[5] WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting. In Thirty-seventh Annual Conference on Neural Information Processing Systems.

[6] PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

[7] Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-series Dependencies and Intra-series Variations Modeling. In Proceedings of the 41st International Conference on Machine Learning.

W2: The experiments in the paper have several shortcomings: (a) CGFM is only validated on a subset of models, leaving its generalizability unclear and necessitating evaluation on a broader range of models; (b) the paper provides no information about the hyperparameter search space, making it unclear whether the authors tuned hyperparameters on the validation set before reporting results on the test set. Since incorporating CGFM alters the model architecture, the sensitivity of certain common hyperparameters (e.g., d_model, e_layers) may also change.

Moreover, even with identical parameters and random seeds, results can vary across different hardware platforms. To ensure fair and rigorous comparisons, all baseline methods should be run on the same platform, share a sufficiently wide hyperparameter search space, and select the best parameters on the validation set before evaluating on the test set. This approach eliminates the influence of platform and hyperparameter sensitivity on model performance, allowing a fair assessment of CGFM.

W3: The paper lacks a theoretical derivation of CGFM’s complexity. The efficiency experiments also omit detailed settings for shared hyperparameters, making the experimental setup unclear. To fairly compare model efficiency, the effect of hyperparameter scale on results must be eliminated; otherwise, selectively choosing very small parameters for certain tasks to claim high efficiency is highly unrigorous and unfair. The authors need to clarify this issue further.

### Questions
Please explain the model's generalization issues and provide details of the relevant experiments.

### Soundness
2

### Presentation
3

### Contribution
1
