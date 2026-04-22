# Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 4

## Abstract
We address multivariate time–series imputation by learning the velocity field of a
data-conditioned ordinary differential equation (ODE) via flow matching. Our
method, Time-Gated Multi-Scale Flow Matching (TG-MSFM), conditions the
flow on a structured endpoint comprising observed values, a per-time visibility
mask, and short left/right context, processed by a time-aware Transformer whose
self-attention is masked to aggregate only from observed timestamps. To recon-
cile global trends with local details along the trajectory, we introduce time-gated
multi-scale velocity heads on a fixed 1D pyramid and blend them through a time-
dependent gate; a mild anti-aliasing filter stabilizes the finest branch. At inference,
we use a second-order Heun integrator with a per-step data-consistency projection
that keeps observed coordinates exactly on the straight path from the initial noise
to the endpoint, reducing boundary artifacts and drift. Training adopts gap-only
supervision of the velocity on missing data coordinates, with small optional regu-
larizers for numerical stability. Across standard benchmarks, Time-Gated Multi-
Scale Flow Matching attains competitive or improved MSE/MAE with favorable
speed–quality trade-offs, and ablations isolate the contributions of the time-gated
multi-scale heads, masked attention, and the data-consistent ODE integration

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Time-Gated Multi-Scale Flow Matching (TG-MSFM), a deterministic framework for multivariate time-series imputation based on flow matching. TG-MSFM learns the velocity field of a data-conditioned ODE using gap-only supervision, which focuses training on the missing entries while enforcing data consistency for observed points during inference. The model features a time-aware Transformer backbone with visibility-masked attention and a multi-scale velocity decomposition modulated by a time-dependent gating mechanism, allowing coarse-to-fine refinement along the generative trajectory. During inference, TG-MSFM integrates the learned flow using the Heun solver combined with a per-step data-consistency projection, ensuring measurement preservation and stable trajectories. Extensive experiments across standard benchmarks demonstrate competitive performance with favorable speed-accuracy trade-offs.

### Strengths
- **Timely and relevant topic**: The paper addresses time-series imputation through flow matching, a rapidly emerging research direction that provides an efficient and deterministic alternative to diffusion-based models.

- **Innovative training strategy**: The use of gap-only supervision, where the model is trained exclusively on missing entries, is conceptually elegant and departs from the conventional full-sample training paradigm dating back to GAIN [1]. This design choice aligns the learning signal directly with the imputation target.

- **Strong empirical evidence**: The experimental section is comprehensive and well structured, covering ten widely used benchmarks and demonstrating consistently strong performance across varying missing ratios and datasets.

- **Clear exposition and illustrations**: The method is well presented, with clear mathematical formulation and figures that effectively convey the architecture and flow process, making the paper easy to follow despite its technical depth.

[1] Yoon, J., Jordon, J., & Schaar, M. (2018, July). Gain: Missing data imputation using generative adversarial nets. In International conference on machine learning (pp. 5689-5698). PMLR. https://arxiv.org/abs/1806.02920

### Weaknesses
- **Limited analysis of the gap-only supervision effect**: Although the empirical results are strong, the paper does not fully explain why training only on missing values leads to better performance. From a representation learning perspective, one might expect that also reconstructing observed entries, as in GAIN [1], helps maintain a richer compression of the overall time-series distribution. A deeper analysis or an ablation comparing gap-only versus full-sample supervision could strengthen the justification for this design choice.

- **Missing discussion on consistency models**: Since the paper draws a clear connection between flow matching and diffusion models, it would be valuable to include a brief discussion on consistency models, which can be viewed as a discrete, distilled, and consistency-enforced formulation of flow matching aimed at improving inference efficiency [2]. In the context of time-series imputation, the recently proposed CoSTI model [3] follows this direction: it can produce probabilistic imputations in a single step, while requiring multiple runs to obtain deterministic estimates such as the median. Adding a short discussion, or even a small efficiency comparison, would help clarify how TG-MSFM relates to this broader family of consistency-based approaches and where it stands in terms of inference trade-offs.

[2] Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models.https://arxiv.org/abs/2303.01469

[3] Javier Solís-García, Belén Vega-Márquez, Juan A. Nepomuceno, and Isabel A. Nepomuceno-Chamorro. Costi: Consistency models for (a faster) spatio-temporal imputation. Knowledge-Based Systems, 327:114117, 2025. https://arxiv.org/abs/2501.19364

### Questions
- Deterministic vs. probabilistic behavior: The paper states that TG-MSFM produces deterministic imputations, yet I found this somewhat unclear. Based on Section 3.2, since the process samples $z_0 \sim \mathcal{N}(0,I)$, wouldn’t the final imputation depend on this random initialization, effectively yielding stochastic (probabilistic) outputs rather than fully deterministic ones? Could the authors clarify whether the inference is performed with a fixed $z_0$ or if randomness is averaged out in some way?

- Model capacity: Could the authors provide the number of parameters in TG-MSFM (and optionally compare it with key baselines)?
This would help assess the method’s complexity and the fairness of computational comparisons

- Addressing reviewer concerns: I would be glad to revise my evaluation if the authors can improve upon some of the points mentioned above.

### Soundness
4

### Presentation
3

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
This paper proposes the TG-MSFM, a method for multivariate time-series imputation by flow matching (FM). To make the conditional generation wrok well in time series setting, they use time-aware transformer and time-gated multiscale velocity heads. At inferene stage, they use a Heun ODE integration with a per-step data-consistency projection, which keep observed coordinates exactly on the linear bridge. They then evaluate the TG-MSFM on 10 benchmars, and show it ttains competitive/ improved performance efficiently.

### Strengths
1. The paper is well written and idea is clear.
2. The transformer is carefully designed to handle properties of time series.
3. The experiements are extensive.

### Weaknesses
1. The work is mainly engineering (on transformer), and the insight it provided to the FM & TS community can be limited (don't take points off on this)
2. Linear bridge and global time warp may be suboptimal for long/heterogeneous gaps.

### Questions
How sensitive of the porposed methods to outliers?

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
3

### Summary
The paper introduces TG-MSFM for deterministic multivariate time-series imputation using a data-conditioned ODE learned via flow matching. It employs a time-aware Transformer with visibility-masked self-attention to aggregate only observed timestamps, and a time-gated multi-scale velocity decomposition on a fixed 1D pyramid, blended by a softmax gate for coarse-to-fine refinement, stabilized by an anti-aliasing filter. Inference uses a second-order Heun integrator with per-step data-consistency projection to preserve observed values. Training focuses on gap-only supervision, with optional regularizers. Tested on 10 benchmark models, TG-MSFM achieves competitive or better MSE/MAE, excels in long-gap scenarios, and offers favorable speed-quality trade-offs, validated by ablations.

### Strengths
1. The method innovatively integrates flow matching with time-series-specific adaptations, including visibility-masked self-attention and time-gated multi-scale velocity, providing a transparent and deterministic alternative to stochastic diffusion models.
2. The proposed model delivers superior or comparable performance to state-of-the-art methods across diverse benchmarks, particularly shining in long-gap cases, while maintaining efficient speed-accuracy trade-offs.
3. The approach shows consistent performance as gap lengths increase, with component benefits confirmed through ablations and fixed hyperparameters across datasets.

### Weaknesses
1. The reliance on fixed pyramid scales and a simple MLP gate may not optimally adapt to all data spectra, potentially underperforming against more dynamic multi-resolution techniques.
2. The paper focuses exclusively on block missing imputation. This narrow focus neglects point missing imputation, a common scenario in many time-series applications (e.g., sensor noise or sporadic data loss), rendering the method inadequate for datasets where isolated missing points predominate.

### Questions
See weaknesses.

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
This paper proposes Time-Gated Multi-Scale Flow Matching (TG-MSFM), an approach for multivariate time-series imputation. The method learns a velocity field of a data-conditioned ODE using flow matching, enabling the reconstruction of missing values in partially observed temporal data. The model incorporates (1) a time-aware Transformer with masked attention over observed timestamps, (2) time-gated multi-scale velocity heads operating over a fixed 1D pyramid to balance global trends and local detail, and (3) a data-consistent Heun integrator that projects observed dimensions back to their known values at each step to prevent drift. The approach is trained using gap-only supervision and demonstrates competitive or superior performance on standard benchmarks. However, the paper lacks a clear positioning with respect to system identification–based methods (concurrent learning, event-based learning) that deal with irregular data, and reservoir network–based systems that also learns a delay-embedding, deterministic high-dimensional vector fields, which could provide useful context regarding model interpretability and dynamical consistency.

### Strengths
1. The idea of learning the flow of an ODE for time-series imputation via flow matching is elegant and relatively underexplored compared to diffusion or autoregressive approaches.
2. The introduction of multi-scale time-gated velocity heads is a clever mechanism to blend local and global temporal patterns.
3. The time-aware Transformer with masked attention is well-motivated and aligns with the irregular sampling structure of real-world time-series data.
4. The method is described with sufficient detail, and the components (velocity heads, time gates, anti-aliasing) each have clear motivations.

### Weaknesses
1. While results are strong, it is unclear how TG-MSFM compares against the most recent latent ODE or diffusion-based imputation models that use similar continuous-time formulations.
2. The ablation study is mentioned but could be expanded to include the quantitative impact of each design choice (e.g., anti-aliasing, Heun vs. Euler integration, time-gate parametrization).
3. Some parts of the model (e.g., the 1D pyramid structure and time-dependent gating) may be difficult to follow without schematic diagrams.
4. The paper occasionally relies on terminology (“velocity heads,” “time-gated blending”) that could use more formal mathematical grounding.
5. Learning deterministic dynamic models for irregular time series is studied in dynamic systems theory, event-triggered learning, and more explicitly with reservoir models. The paper could benefit from reviewing and contextualizing the flow-matching-based methods with these theoretically grounded methods.

### Questions
1.	How does TG-MSFM perform under irregular and sparse sampling conditions compared to continuous-time diffusion models?
2.	Could the data-consistency projection be viewed as a constrained ODE solver? If so, how does it affect the learned velocity field during training?
3.	What is the computational complexity relative to baseline flow-matching or diffusion-based methods, and what are the benefits of using the proposed method versus traditional systems theoretic methods?
4.	Did you experiment with adaptive time-stepping for the Heun integrator to further improve efficiency or stability?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the TG-MSFM model, which introduces a data-conditioned ODE framework trained via flow matching using gap-only supervision of the velocity on missing data coordinates. This approach adopts a data-conditioned ODE through flow matching, leading to a deterministic imputation mechanism. The model further incorporates time-gated multi-scale velocity heads on a fixed 1D pyramid, blended through a time-dependent gating mechanism to reconcile global trends with local details. In addition, a second-order Heun integrator combined with a per-step data-consistency projection ensures recovery of the exact linear bridge across all coordinates. The paper presents several innovations, and experiments conducted on diverse datasets across multiple domains demonstrate the effectiveness of the proposed model.

### Strengths
1 The proposed model employs a data-conditioned ODE trained via flow matching as a deterministic framework, which enhances the reproducibility and stability of the imputation process.

2 The introduction of time-gated multi-scale velocity heads organized on a fixed 1D pyramid, blended through a time-dependent gating mechanism, effectively reconciles global trends with local details along the trajectory, achieving efficient fusion of global and local information.

3 The time-gated multi-scale velocity decomposition schedules coarse-to-fine refinement along the ODE trajectory and incorporates a light anti-aliasing filter to suppress high-frequency ringing artifacts.

4 The Heun+DC procedure successfully recovers the exact linear bridge across all coordinates, providing a robust accuracy–cost trade-off under appropriate hyperparameter settings.

5 The proposed model is easy to deploy and achieves competitive performance across diverse datasets.

### Weaknesses
1 Some abbreviations should be clearly defined upon first appearance to improve readability—for example, Data Consistency (DC) and Flow Matching (FM). In addition, several sentences could be refined for clarity. For instance, the sentence “At inference we integrate the learned velocity with the second-order Heun method (line 48)” could be revised to “At inference, we integrate the learned velocity using the second-order Heun method.” The visualization in Figure 1 could also be improved—for example, the two arrows under v_theta in the second block could be revised to make the diagram clearer.

2 The Diffusion-based probabilistic imputation section in Related Work requires improvement. Several relevant diffusion-based imputation models are missing, such as PriSTI, SSSD, and Frequency-aware Generative Models for Multivariate Time Series Imputation. In addition, recent works closely related to this paper should be discussed, including:


[1] Zhou, J., Li, J., Zheng, G., Wang, X., & Zhou, C. (2024, October). Mtsci: A conditional diffusion model for multivariate time series consistent imputation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (pp. 3474-3483).


[2] Yuan, X., & Qiao, Y. Diffusion-TS: Interpretable Diffusion for General Time Series Generation. In The Twelfth International Conference on Learning Representations.


[3] Zhang, H., Fang, L., Wu, Q., & Yu, P. S. (2025). Diffputer: Empowering diffusion models for missing data imputation. In The Thirteenth International Conference on Learning Representations.

3 The baseline models in the experimental section could be updated and compared against more recent and relevant approaches, such as the works listed below. Including stronger baselines would make the experimental validation more convincing.

[1] Zhou, J., Li, J., Zheng, G., Wang, X., & Zhou, C. (2024, October). Mtsci: A conditional diffusion model for multivariate time series consistent imputation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (pp. 3474-3483).

[2] Yuan, X., & Qiao, Y. Diffusion-TS: Interpretable Diffusion for General Time Series Generation. In The Twelfth International Conference on Learning Representations.

[3] Yang, X., Sun, Y., & Chen, X. (2024). Frequency-aware generative models for multivariate time series imputation. Advances in Neural Information Processing Systems, 37, 52595-52623.

4 One citation is not properly formatted: Freeformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting and should follow a formal citation style.

5 The motivation of the proposed approach could be better articulated. The paper emphasizes the differences between the proposed model and prior methods, but it remains unclear why a deterministic flow-matching framework is preferable to an SDE-based or neural ODE-based stochastic model. Is the main goal to improve reproducibility or reduce computational efficiency? If reproducibility is the motivation, increasing the sampling steps or applying ODE in the reverse process of diffusion model as probability flow could achieve similar improvements. If computational efficiency is the target, DDIM-based comparisons should be included. Moreover, generative stochastic models inherently provide insights into the underlying data generation process, offering better interpretability and uncertainty estimation—this trade-off deserves further discussion.

### Questions
1 Motivation: The authors state that the motivation for focusing on deterministic imputation is to improve reproducibility and ensure a unique reconstruction. Is this the main motivation for adopting a flow-matching framework and replacing the SDE formulation with a data-conditioned ODE as a discriminative model? Clarifying this reasoning would help readers better understand the core design choice.

2 In the Positioning against recent FM/continuous/diffusion and graph methods section, the paper emphasizes how the proposed model differs from existing approaches. However, from the architecture shown in Figure 1, the model appears to combine several design elements from prior works. Moreover, some baseline models with related components are missing in the experimental comparison. Could the authors more clearly articulate the essential differences and innovations of the proposed method, beyond assembling components from existing architectures?

3 In the Flow and Path Matching section, the paper states that “Compared with stochastic sampling, a learned ODE allows deterministic integration at test time and often reduces the number of function evaluations needed for a target quality.” This suggests that using an ODE instead of an SDE provides a favorable speed–quality trade-off. However, as shown in Song et al. (Section 4.3, Probability Flow and Connection to Neural ODEs), the reverse process of diffusion models can also use an ODE formulation, allowing an explicit trade-off between accuracy and efficiency. In this context, what are the specific advantages of the proposed deterministic ODE framework compared to the Probability Flow ODE derived from generative stochastic models?

[1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.

4 Regarding the time-gated multi-scale velocity mechanism for suppressing high-frequency ringing and addressing the data-consistency issue, several recent diffusion-based models have proposed similar solutions, Could the authors further clarify how TG-MSFM fundamentally differs from or improves upon these approaches?

[1] Zhou, J., Li, J., Zheng, G., Wang, X., & Zhou, C. (2024, October). Mtsci: A conditional diffusion model for multivariate time series consistent imputation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (pp. 3474-3483).

[2] Yuan, X., & Qiao, Y. Diffusion-TS: Interpretable Diffusion for General Time Series Generation. In The Twelfth International Conference on Learning Representations.

[3]  Yang, X., Sun, Y., & Chen, X. (2024). Frequency-aware generative models for multivariate time series imputation. Advances in Neural Information Processing Systems, 37, 52595-52623.

5 Some datasets used in the experiments, such as Traffic and PEMS03, exhibit strong spatio-temporal dependencies. Including GNN-based baselines for comparison could enhance the comprehensiveness and fairness of the evaluation. For example, the discriminative models such as Grin, Spin and ImputeFormeras follows:

[1] Cini, A., Marisca, I., & Alippi, C. Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks. In International Conference on Learning Representations, 2022.

[2] Marisca, I., Cini, A., & Alippi, C. (2022). Learning to reconstruct missing data from spatiotemporal graphs with sparse observations. Advances in neural information processing systems, 35, 32069-32082.

[3] Nie, T., Qin, G., Ma, W., Mei, Y., & Sun, J. (2024, August). ImputeFormer: Low rankness-induced transformers for generalizable spatiotemporal imputation. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining (pp. 2260-2271).

### Soundness
3

### Presentation
2

### Contribution
3
