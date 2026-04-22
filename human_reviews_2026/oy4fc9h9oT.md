# Signals, Concepts, and Laws: Toward Universal, Explainable Time-Series Forecasting

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Accurate, explainable and physically credible forecasting remains a persistent challenge for multivariate time-series with domain-varying statistical properties. 
We propose DORIC, a Domain-Universal, ODE-Regularized, Interpretable-Concept Transformer for Time-Series Forecasting that generates predictions through five self-supervised, domain-agnostic concepts while enforcing differentiable residuals grounded in 
first-principles constraints. The concepts are softly regressed toward analytic statistics of the raw signal, and a driven–damped ODE head couples these concepts to the forecast as a shared, mean-reverting dynamical template across datasets. Unlike prior efficiency-focused Transformers, such as Informer(sparse attention) or FEDformer(frequency priors), DORIC combines latent explainability with explicit scientific constraints, while preserving the attention mechanism’s capacity to model long-range dependencies.  
We evaluate DORIC on six publicly-available datasets and it achieves the lowest error in eight of twelve MSE/MAE metrics.  
Compared with TimeMixer, DORIC outperforms it on four datasets while maintaining strong interpretability. 
Interpretability analyses show that the learned concepts remain strongly aligned with their analytic targets, physics residuals stay relative to the signal scale, and the learned ODE coefficients follow domain-consistent patterns.
Ablation studies reveal complementary contributions: removing the physics residual increases average MSE from 0.328 to 0.547, eliminating concept alignment raises it to 0.698, and replacing the shared encoder with disjoint concept heads results in a 76% increase.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a deep forecaster, DORIC, that generates the coefficients of an ODE which resolution leads to a one-step forecast.
DORIC estimates in a self-supervised manner 5 quantities as parameters of the ODE, corresponding to 5 first-order notions (mean, velocity, power, dominant periodic amplitude, volatility), and monitors those 5 quantities to provide interpretability.

Experiments on 6 well known benchmark datasets are conducted, with predictive performance and ablation analysis.

### Strengths
Clarity:
1) The paper is well written, with particular attention to the 5 quantities, their interpretation and monitoring through residuals.

Quality:
1) Presented forecasting results are convincing, especially noting that 5 quantities are apparently sufficient to represent the next timestep.
2) Ablation analysis is well done, both in main paper and appendix. We can reasonably expect all components to be necessary and perform well together.
3) Assumption 1 and Theorem 2 show that under mild assumptions (except perhaps for L-lipschitzianity), the physics loss vanishes during training with regularization schedule.

Significance:
1) This is a physics enriched approach that is remarkably easy for a non-physicist layperson to interpret.

### Weaknesses
Quality:
1) While related work include PINN and PhysicsSolver, the paper would benefit from including other ODE-based DNN, as it is the central technique of the paper.
2) Result analysis only quantifies MSE and MAE. There seem to be no use of either the learned coefficients $c_i$, or the residuals. The interpretability side is lacking in the main paper, even though this should be equally important to predictive performance.
3) Some interpretability indicators have no matching table or figure: in D.1, in D.2 (777 & 788).
4) In figure 6, just the ci are insufficient to confirm the coherence of the plot. It would be better to add the ci* for comparison.

Clarity:
1) The final computation of the model is not presented, as the methodology ends with an undefined $\mathcal{L}_{data}$. How to obtain the prediction from the ODE is not explicited. This is important as it is necessary to precise how retropropagation works in this context. It should be at least mentionned in the main paper, even if it is present in appendix pseudocode.
2) The main paper explains how to obtain 1-step forecasts, hence, before reading the appendix, I believed that forecasting results used one-step forecasts, whereas the datasets used are typical to Long Term Forecasting problems. A precision on the transformation from 1-step to multi-step would be welcome in the main paper.
3) Figure 2 requires a better legend. Section 4.4 is not enough to describe the plots. Figure 3 also, I cannot understand the plot settings.

### Questions
No further questions than the in weaknesses. 

Overall, I lean toward an accept. My main concern is on the interpretability side. I believe too few quantitative elements are given to motivate the interpretability of the forecasting method, especially in main paper. More attention to the ci, ci* and residuals (alongside training, during inference, etc) should be given. Explicitely using the sanity checks metrics (788) in a figure would be welcome.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method aiming to achieve Domain-Universal, ODE-Regularized, Interpretable Time-Series Forecasting.

### Strengths
1. The motivation in this paper is great.
2. Physics constraints are important for time series forecasting.

### Weaknesses
1. The quality of the paper writing is weak. The subscripts and asterisk in Eqs. (10-14) are not common in time series publications, but without clear explanations.  For example, it would be great to explain 1t, ..., 5t.
2. In your third contribution, you mentioned adaptive positional encoding for a domain-universal architecture, but I did not find the details on that. Moreover, shared concept heads seem to be used to train the model with all time series together, which implies the universal feature is attributed to the training manner rather than your architecture.
3. The number of concepts in Eq. 11 is empirical. How can you guarantee how many statistical concepts you need? It should vary across different time series datasets.
4. Clear ablation studies definitely deserve to be included in the main paper, since they are helpful to understand how important each component is. The current results in Table 2 are too simple.
5. From my understanding, time series interpretability should focus on how time points in the original time series contribute to the final prediction, rather than projecting it to some statistical concepts.
6. The code is too limited. Hard to reproduce. At least, a demo should be provided.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces an approach called DORIC, a domain-universal, physics-regularized, interpretable-concept Transformer for multivariate time-series forecasting. The model integrates five self-supervised, domain-agnostic concepts as concept bottleneck (level, growth, periodicity, volatility, exogenous pressure) and a physics-informed residual loss enforcing first-principles consistency. Experiments across six public datasets show comparable performance to SOTA and certain robustness under noise.

### Strengths
-interpretability is done through five fundamental concepts: they can be computed directly from the data and used in a soft-supervised fashion, avoiding the need for labeled concept datasets.

- The model demonstrates cross-domain generalization across diverse datasets (electricity, traffic, weather, epidemiology, finance).

- There are theoretical results showing doric will eventually yield  good enough result (despite the interpretability), with physical plausibility.

### Weaknesses
-My main problem is that interpretability is claimed but not really studied or demonstrated. The paper does not present any empirical or qualitative evidence that the learned concepts are interpretable or used meaningfully by the model.  Overall it remains unclear how these scores are used by the model and how much the final forecast was determined  due to the concept scores.There are no visualisations, case studies to show, or attribution analyses showing how forecasts depend on concept activations at inference time (this is so to say, the work lack interpretability centric ablation study).


- Related to above, the concepts that are used are still very low level i.e., they are just signal descriptors, not sure it this at all  useful with interpretability at all, hence also not much lose in performance is not a big surprise.  Just as  we could also measure these concepts (or statistical features) for given time-series without any sophisticated ML techniques.  

-It looks like they use a vanilla Transformer (with probsparse attention) rather than stronger recent architectures they included e.g., autoformer, patchtst, or tmemixer. This makes it hard to attribute performance gains specifically to the concept-physics mechanism.  Why not? 

-The analysis of interpretability vs. performance trade-offs is missing. There is no discussion whatsoever whether the same concepts are equally meaningful across all domains.

-The combination of attention, differentiable ode constraints, and multi-loss optimization can make  harder to reproduce or tune than conventional transformer-based forecasting models.

-Not a major weakness, but just to report that 3 out of 6 dataset, Doric is outperformed.

### Questions
Please feel free to respond to any weaknesses I listed above. Additionally: 

Do the concept activations differ meaningfully across datasets, or are they simply re-parameterized statistical features?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces DORIC (Domain-Universal, ODE-Regularized, Interpretable-Concept Transformer), a novel framework for time-series forecasting that combines explainable concept bottlenecks with physics-informed regularization. The model routes multivariate input through five interpretable latent concepts before predicting via a driven–damped ODE head. Unlike prior Transformers that optimize efficiency or frequency decomposition, DORIC emphasizes scientific plausibility and explainability. Experiments across six benchmarks (Electricity, Traffic, Weather, Illness, Exchange Rate, and ETT) show DORIC achieves the lowest MSE/MAE in most metrics.

### Strengths
The paper’s main strength is its innovative integration of concept bottlenecks and physics-informed residuals within a Transformer, effectively combining interpretability with physical plausibility. It presents a comprehensive evaluation across six diverse datasets and strong baselines, supported by detailed ablation studies that clearly show each component’s contribution. The authors emphasize interpretability through five structured latent concepts and provide theoretical grounding via expressiveness and convergence analyses. Additionally, the appendix enhances reproducibility by including implementation details and pseudo-code.

### Weaknesses
1. The paper’s central claim of interpretability is not convincingly demonstrated. While the architecture enforces a five-concept bottleneck, there is no theoretical proof or empirical validation that these learned concepts retain their intended meanings. Evidence is limited to internal correlations, without visual, human, or domain-level verification of interpretability.
2. The paper introduces a driven–damped ODE as the core of its “physics-informed” design, but the justification is mostly heuristic. The ODE form is applied uniformly across unrelated domains without evidence that such dynamics are meaningful or empirically valid, and the learned coefficients are never analyzed. As a result, the physics component functions more as a generic smoothness prior than a genuinely grounded physical model.
3. Figures lack axis explanations such as Figure 2 and Figure 3, making it hard to interpret visual differences quantitatively.
4. There are some grammar errors:
Line 149-150: “time series data first enters …” → plural mismatch; should be “data first enter …” or “the time-series signal first enters …”.
Line 150-151: “prediction..” → double period; correct to “prediction.”
Line 372-373: “Quantitative performance :” → remove space before colon.
Line 323: “For he detailed theorem setting …” → missing “t”; should read “For the detailed theorem setting …”.

### Questions
1. Can the authors provide quantitative or qualitative evidence (e.g., visualizations, case studies, or human evaluations) showing that the learned latent concepts correspond to their intended meanings and remain interpretable after training?
2. Can the authors clarify why a driven–damped ODE was chosen as the universal physical prior across all datasets, and provide evidence—empirical or theoretical—that the learned ODE parameters correspond to meaningful dynamics rather than serving only as a generic regularizer?
3. How are the ODE coefficients ($\beta$, $\gamma$) initialized and constrained? Are they shared across datasets?
4. Can the method handle irregularly sampled or non-stationary time-series data without retraining?
5. Are there any computational trade-offs compared to other Transformers (e.g., inference latency)?

### Soundness
2

### Presentation
2

### Contribution
2
