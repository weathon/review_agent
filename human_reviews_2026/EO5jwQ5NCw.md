# GCGNet: Graph-Consistent Generative Network for Time Series Forecasting with Exogenous Variables

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Exogenous variables offer valuable supplementary information for predicting future endogenous variables. Forecasting with exogenous variables needs to consider both past-to-future dependencies (i.e., temporal correlations) and the influence of exogenous variables on endogenous variables (i.e., channel correlations). This is pivotal when future exogenous variables are available, because they may directly affect the future endogenous variables. Many methods have been proposed for time series forecasting with exogenous variables, focusing on modeling temporal and channel correlations. However, most of them use a two-step strategy, modeling temporal and channel correlations separately, which limits their ability to capture joint correlations across time and channels. Furthermore, in real-world scenarios, recorded time series are frequently affected by various forms of noises, underscoring the critical importance of robustness in such correlations modeling. To address these limitations, we propose GCGNet, a Graph-Consistent Generative Network for time series forecasting with exogenous variables. Specifically, GCGNet first employs a Variational Generator to produce coarse predictions. A Graph Structure Aligner then further guides it by evaluating the consistency between the generated and true correlations, where the correlations are represented as graphs, and are robust to noises. Finally, a Graph Refiner is proposed to refine the predictions to prevent degeneration and improve accuracy. Extensive experiments on 12 real-world datasets demonstrate that GCGNet outperforms state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces GCGNet, a Graph-Consistent Generative Network aimed at time series forecasting with exogenous variables. The framework combines a variational generator for coarse forecasting, a graph-based discriminator enforcing joint temporal and channel consistency, and a graph refiner that leverages learned adjacency structures to refine outputs. Empirical evaluation on 12 real-world datasets, including settings with missing values and ablation studies, demonstrates the proposed method’s improved robustness and competitiveness compared to a suite of baselines.

### Strengths
- This paper breaks the existing mindset in forecasting with exogenous variables, which typically follows a two-step modeling strategy inherited from traditional forecasting methods. Instead, it jointly models the time and channel, which represent a effective approach for forecasting with exogenous variables.

- The proposed Graph Discriminator in this paper is interesting, as it provides a novel way to align the predicted values with the underlying temporal and channel joint correlations.

- The experimental results demonstrate that the proposed method achieves strong performance.

- The code and datasets has been open, supporting reproducibility.

### Weaknesses
- The description of the Sparsify operation in Section 3.4 only briefly explains its purpose but doesn't  clearly describe how it actually works. More details are needed to help  readers understand what this step does and how it fits into the Graph  Refiner module.

- The Sparsify operation uses a hyperparameter k that controls how much sparsification is applied, and this likely has a big impact on how well the model performs. However, the paper doesn't  explain how the value of k was chosen, or test how different choices of k affect the results.

- Please carefully check the references. References to preprints, such as "Long-term forecasting with tide: Time-series dense encoder. " should be updated to their formally published versions.

### Questions
- Please provide a more detailed explanation of the Sparsify operation and conduct a sensitivity analysis on the parameter k  within Sparsify.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes GCGNet, a general and robust framework for time series forecasting that incorporates both historical and future exogenous variables. A key component is the Graph Discriminator, which utilizes graph structure consistency to guide the generator, ensuring that the generated forecasts align with temporal and inter-channel dependencies. Extensive experiments are conducted on 12 real-world datasets.

### Strengths
1. The paper is well structured, and it focuses on an important problem that could have real-world applications.

2. Nice figures help to understand the proposed method, and both the figures and tables are clear and well-presented.

3. The proposed method may address some limitations of existing approaches, such as separate modeling and limited robustness.

### Weaknesses
1. The current set of baselines includes many models that do not inherently accommodate exogenous variables; however, this paper focuses on forecasting with exogenous variables.

2. The adaptation of models that inherently lack support for exogenous variables, such as PatchTST, using a simple MLP fusion approach to incorporate future covariates raises concerns about its effectiveness. It remains questionable whether this method preserves the original modeling principles or potentially introduces degradation in performance.

3. The experimental results for the scenario where future exogenous variables are unavailable are only reported on a subset of datasets.  This is questionable, particularly since this scenario represents a more common and practical use case in real-world applications.

4. The naming of the key module as "Graph Discriminator" strongly suggests an adversarial training paradigm. However, its loss function L_disc functions more as an L1-based graph structure  reconstruction loss rather than a true discriminator performing binary  classification to distinguish "real" from "generated" graphs. This nomenclature is misleading.

### Questions
1. Since this paper focuses on forecasting with exogenous variables, why weren’t the baselines restricted to algorithms that natively support exogenous variables? 

2. Why not use the original model directly, and instead add an MLP fusion approach, since this module might actually degrade the model’s performance?

3. Why are the experimental results for the scenario without future exogenous variables reported only on a subset of datasets?

4. Why is the key module named "Graph Discriminator" when it is actually unrelated to an adversarial training paradigm?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In summary, this paper introduces GCGNet, a graph-consistent generative network for time series forecasting with exogenous variables. GCGNet first generates coarse predictions, then enforces structural alignment to capture joint temporal and channel correlations, and finally refines the predictions to prevent degeneration and improve accuracy. With these designs, GCGNet effectively models complex temporal–channel dependencies while maintaining robustness in real-world scenarios.

### Strengths
S1. Time series forecasting with exogenous variables is important to various domains.

S2: This paper proposes GCGNet to address the problem of forecasting with exogenous variables.

S3: This paper proposes Graph Discriminator module to ensuring the outputs align with the underlying temporal and channel joint correlation.

S4: The authors released an anonymous GitHub repository, which improves reproducibility.

### Weaknesses
W1: In Figure (c), does it correspond to the framework proposed in this paper? \(Y^{endo}\) represents future endogenous variables, so why is \(Y^{endo}\) assumed to be accessible in the framework? 

W2: In Figure (c), I noticed that the figure labels it as \(\tilde{Y}^{endo}\), but the figure does not provide any explanation of what \(\tilde{Y}^{endo}\) represents.

W3: The description in lines 240–244 is unclear. In Equation (7), the process of obtaining \(\hat{A}\) from \(A\) is too vague. Since \(A\) represents a graph, the authors should clarify how a graph is processed using the VAE, rather than simply stating that \(\hat{A}\) is obtained from \(A\).

W4: In lines 77–84, the authors claim that using a generative network helps improve robustness. However, they only provide the results in Table 3 to validate the overall framework, without isolating the contribution of the generative network itself. It would be more convincing if the authors could provide theoretical justification or additional experiments, such as an ablation study, to demonstrate that the generative network specifically contributes to the performance improvement.

W5: In lines 428–429, the authors mention that in Table 4 the "random" setting replaces masked values with random values sampled from the normal distribution N(0,1). This is unreasonable because the range and scale of the original features may differ significantly from N(0,1). Such noise does not reflect realistic scenarios, and therefore the "random" experiment is not meaningful.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper **proposes GCGNet**, a *Graph-Consistent Generative Network* that jointly models temporal and channel correlations for robust time series forecasting with exogenous variables. Through a variational generator, graph discriminator, and refiner modules, **GCGNet achieves state-of-the-art performance** across 12 real-world datasets, outperforming prior two-step modeling approaches by effectively capturing joint dependencies and handling noisy data.

### Strengths
**Strengths:**

1. The experimental results are significant.
2. The writing is clear and easy to follow.

### Weaknesses
**Weaknesses:**

1. **Lack of Novelty:** I fail to see significant advantages of this paper over related works. On one hand, graph-based methods for modeling the relationships between variables and the joint modeling of time and variables have been discussed in early literature, such as GWNe[1], DCRNN[2], and STEP[2]. On the other hand, the lack of joint modeling of temporal and channel correlations is not supported by quantitative case analyses. Furthermore, the authors argue that noise affects the model, which motivates the use of generative models, but this lacks persuasiveness. Even generative models are data-driven and cannot fully address the impact of data noise. Overall, the authors should provide more evidence to support their motivation, which is the foundation for the proposed methods and contributions.

2. **Lack of Technical Innovation:** The authors lack a thorough analysis of the challenges encountered while addressing issues in related works. Instead, they directly propose the graph-consistent generative module, including the Variational Generator, Graph Discriminator, Graph Variational Autoencoder, and Graph Refiner modules, without sufficiently justifying each design choice. The overall model design, as well as the design of each part, lacks enough motivation and insights, making the approach feel overly engineered.

3. **Writing Needs Improvement:** As mentioned above, the writing requires significant improvement. The authors should provide more evidence to validate their motivations, the significance of their contributions, and the insights behind the method design.

[1]. Graph WaveNet for Deep Spatial-Temporal Graph Modeling
[2]. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
[3]. Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
