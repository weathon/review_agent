# Temporal Alignment Guidance: On-manifold Sampling in Diffusion Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Diffusion models have achieved remarkable success as generative models. However, even a well-trained model can accumulate errors throughout the generation process. These errors become particularly problematic when arbitrary guidance is applied to steer samples toward desired properties, which often breaks sample fidelity. In this paper, we propose a general solution to address the off-manifold phenomenon observed in diffusion models. Our approach leverages a time predictor to estimate deviations from the desired data manifold at each timestep, identifying that a larger time gap is associated with reduced generation quality. We then design a novel guidance mechanism, `Temporal Alignment Guidance' (TAG), attracting the samples back to the desired manifold at every timestep during generation. Through extensive experiments, we demonstrate that TAG consistently produces samples closely aligned with the desired manifold at each timestep, leading to significant improvements in generation quality across various downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel method called Temporal Alignment Guidance (TAG), designed to address the off-manifold issue that arises when applying external guidance in diffusion models.
The approach introduces a Time Predictor that dynamically estimates the temporal deviation of samples from the true data manifold during generation, and employs a Time-Linked Score (TLS) to attract samples back toward high-density regions. TAG can be seamlessly integrated into existing diffusion models without retraining (training-free) and demonstrates significant improvements in both sample quality and stability across various domains, including image, molecular, and audio generation. Extensive experiments and detailed theoretical analyses show that TAG not only enhances generative fidelity, but also reduces temporal drift, suggesting its strong potential as a general-purpose guidance framework for diffusion models.

### Strengths
1. The paper introduces Temporal Alignment Guidance (TAG), which leverages a time predictor and time-linked score to effectively address the off-manifold issue in diffusion models under external guidance.
2. TAG operates in a training-free manner and can be seamlessly integrated as a plug-in into existing diffusion frameworks.
3. The authors provide theoretical analysis proving TAG’s ability to reduce generation error bounds, supported by extensive experiments across multiple domains that confirm its robustness and generality.

### Weaknesses
1. TAG employs a temporal gradient term to correct the sampling trajectory, but the paper does not provide an in-depth analysis of its error propagation or convergence boundaries within continuous-time dynamic systems.
2. The time guidance schedule $\omega$ in TAG relies on manually selected hyperparameters, without any adaptive update mechanism. As a result, the method still requires manual tuning across different noise levels, tasks, or model scales to achieve optimal performance.
3. In Section E.2, Table 10 shows that under a fixed noise schedule $\sigma=0.2$, as the guidance strength $\omega$ increases, both TG and FID values gradually decrease (ignoring the rebound of TG beyond $\omega$ = 40), while IS values steadily increase. (1) However, in Table 11 (lines 1997–1999), the parameter correspondence with Table 10 is unclear — in particular, for $\sigma=0.2$, the optimal parameter $\omega$ appears to be 45 instead of 4.5. The authors are advised to verify the correct parameter mapping. (2) In addition, since the results in Table 12 are almost identical to those in Table 11, consistency in numerical rounding and presentation should be ensured across both tables.
4. In Appendix G, the visual differences between methods are difficult to distinguish in several generated image comparisons (e.g., Figures 11–13 and 17). It is recommended that the authors include more representative samples or provide zoom-in crops to make the differences between TAG and other methods more visually discernible.
5. The paper does not include a formal analysis or quantitative discussion of the time complexity of TAG.
6. (1) In the Introduction, the sentence “This score approximation errors can accumulate over each timestep” contains a grammatical mismatch — the singular demonstrative “This” does not agree with the plural noun “errors.” (2) Lines 70–73 contain repetitive phrasing; removing redundancy would improve clarity and conciseness.

### Questions
Please see weakness.

### Soundness
3

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
3

### Summary
The paper introduces Temporal Alignment Guidance (TAG), a plug-in guidance term for diffusion sampling that aims to keep trajectories on-manifold when external guidance is applied. The key idea is to learn a lightweight time predictor that estimates the timestep posterior. The authors then add the gradient of the Time-Linked Score to the model score at every step, which encourages the sample to remain consistent with the intended time manifold so that guidance does not push it into low-density regions. The paper provides a theoretical rationale for this correction and reports improvements across several scenarios.

### Strengths
1. The proposed method offers a clear and broadly applicable mechanism that enhances existing guided diffusion frameworks.
2. The manuscript is well-structured and readable, making its contributions accessible.
3. The theoretical component provides sound motivations that align well with the empirical findings.
4. A wide set of experiments across different domains supports the practical relevance and robustness of the approach.

### Weaknesses
1. Since the paper employs a large number of symbols, it would greatly improve readability to include a notation table summarizing the meaning of key variables and subscripts. 
2. The paper does not state the relation between TAG and prior manifold-preserving guidance approaches [1–3].

[1] Yang, Lingxiao, et al. "Guidance with Spherical Gaussian Constraint for Conditional Diffusion." *International Conference on Machine Learning*. PMLR, 2024.

[2] He, Yutong, et al. "Manifold Preserving Guided Diffusion." *ICLR*. 2024.

[3] Chung, Hyungjin, et al. "CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models."  *ICLR*. 2025.

### Questions
1. How reliable is TAG when the time predictor $p_\phi(t\mid x)$ is imperfect? Does the method remain stable under strong external guidance or off-manifold drift?
2. Is the proposed Time-Gap metric quantitatively correlated with established quality measures such as FID, CLIP, or task-specific reward scores? Could the authors provide correlation analyses to demonstrate its validity as a general misalignment indicator?
3. What is the computational cost of training the time predictor and computing the additional gradient term $\nabla_x \log p_\phi(t\mid x)$ at each sampling step, especially in large latent models?

### Soundness
3

### Presentation
2

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
This paper seeks to address the off-manifold phenomenon observed in diffusion models. Defined as the issue where even well-trained models accumulate errors during generation, particularly when arbitrary guidance is applied to steer samples toward desired properties. This phenomenon causes generated samples to deviate from the desired data manifold, ultimately compromising sample fidelity. To tackle this, the authors first employed a time predictor to estimate deviations from the desired data manifold at each timestep, identifying that a larger time gap correlates with diminished generation quality. Subsequently, they developed a novel guidance mechanism termed Temporal Alignment Guidance (TAG), which pulls samples back to the desired manifold at every timestep throughout the generation process.

### Strengths
+ The proposed method is intuitive and effective in addressing the off-manifold phenomenon of diffusion models.

+ Extensive experiments and theoretical analysis are conducted to demonstrate the improvements brought by the proposed method.

+ The proposed method can be applied to various downstream tasks to enhance their performance.

### Weaknesses
- The proposed method necessitates the integration of a time predictor, which must be tailored to different downstream models. Notably, the performance of this time predictor exerts a substantial influence on the overall efficacy of the method.

- The method presented in this paper is developed primarily by analyzing and addressing challenges arising from the use of classifier guidance in diffusion-based generation. However, it would be valuable to further investigate two key points: first, whether the off-manifold phenomenon also occurs in classifier-free guidance—a technique widely adopted in diffusion-based generation, and second, if it does, whether a analogous solution can be effectively applied.

- In Equation (7), the Temporal Alignment Guidance (TAG) introduces a parameter $\omega$ to control its strength. In practical applications, how should the value of this parameter be determined? And will the optimal value of $\omega$ vary significantly across different downstream tasks?

### Questions
Please see weaknesses above.

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
3

### Summary
This paper addresses the problem in diffusion models where conditional gradients during sampling cause samples to gradually deviate from the true data manifold (referred to as the "off-manifold" phenomenon). It proposes an effective gradient correction strategy: Temporal Alignment Guidance (TAG). This method explicitly applies corrective gradients at each sampling timestep, pulling the samples back to high-probability regions of the original distribution. This suppresses deviations caused by unreasonable conditional guidance, ensuring the sampling trajectory remains aligned with the target manifold throughout the reverse diffusion process.

### Strengths
1. Controlling by external guidance, Multi-conditional guidance, Few-step generation, and Degradation of sample quality in low-density regions. The paper clearly explains the sources of the problem and validates them experimentally, which is highly convincing.

2. Introduces the Time-Linked Score (TLS) and provides corresponding probabilistic and energy-based theoretical explanations. Through theorems and propositions, it analyzes TAG’s convergence and anti-deviation properties, offering strong mathematical support for the method’s reliability.

3. Demonstrates improvements across tasks including image, audio, molecular, multi-conditional, and few-step generation, showing the method’s generality. Additionally, TAG is implemented as an external guidance module, making it lightweight and easy to deploy.

### Weaknesses
1. Lacks direct experimental verification of score approximation errors.

2. When combined with strong conditional guidance, TAG may conflict with the original guidance gradient, potentially causing samples to deviate from the target semantics or degrade generation quality. This scenario is not discussed in the paper.

3. The Time Predictor may introduce new error sources; if time classification is incorrect, its gradient might push samples in the wrong direction. The robustness of the model in this context is not discussed.

### Questions
1. In practical sampling, TAG applies gradients alongside the original conditional guidance. If their directions conflict, does TAG’s correction affect the effectiveness of conditional control in maintaining sample quality?

2. Could the authors provide a more intuitive quantification or visualization of score approximation error accumulation during sampling, and demonstrate TAG’s ability to reduce this error at each timestep, rather than relying solely on final sample metrics?

### Soundness
3

### Presentation
3

### Contribution
2
