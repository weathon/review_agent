# Measurement-Aligned Sampling for Inverse Problem

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Diffusion models provide a powerful way to incorporate complex prior information for solving inverse problems. However, existing methods struggle to correctly incorporate guidance from conflicting signals in the prior and measurement, and often failed to maximizing the consistency to the measurement, especially in the challenging setting of non-Gaussian or unknown noise. To address these issues, we propose Measurement-Aligned Sampling (MAS), a novel framework for linear inverse problem solving that flexibly balances prior and measurement information. MAS unifies and extends existing approaches such as DDNM, TMPD, while generalizing to handle both known Gaussian noise and unknown or non-Gaussian noise types. Extensive experiments demonstrate that MAS consistently outperforms state-of-the-art methods across a variety of tasks, while maintaining relatively low computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper generalizes diffusion-based inverse problem solvers into a latent optimization framework by introducing a weighted matrix that balances measurement fidelity and diffusion prior regularization through two hyperparameters. Through an analysis of the solver’s behavior with respect to each hyperparameter, the paper derives appropriate design choices and demonstrates robustness across various types of measurement noise, including Gaussian and non-Gaussian noise, as well as for non-differentiable forward operators.

### Strengths
- The framework of optimization problem with weighted matrix that reflects the geometry of the forward operator generalize existing method and eventually provide a better choice that improves the performance.
- The proposed method extends the ability of solving inverse problem with diffusion models to unknown measurement noise.
- The paper provides extensive comparison on various linear inverse problems and baselines.

### Weaknesses
- The writing can be improved. Especially, it is quite hard to figure out the most important factor among a lot of parameters such as $\eta_1, \eta_2, c_t, r_t, \sigma_y, \sigma_\epsilon$.
- Discussion on negative $\eta_1$ does not explain the reason why "overshoot" is better than other cases. Appendix B.2 provides a discussion about why "overshoot" does not cause a crucial problem with non-invertible W.

### Questions
- From the ablation study result in Figure 4, the performance is gradually better if we use more negative $\eta_1$. If we use smaller value for $\eta_1$ (e.g. $\eta_1=-1.0$), will the performance be continuously better?
- Also, in figure 4, the performance is the best when $\eta_2=0$, which is DDNM according to Remark 1. Is $\eta_2 = \sigma_r^2/r_t^2$ after the proposition 3.1 is better then this setting?

### Soundness
3

### Presentation
2

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
The paper proposes a new framework called Measurement-Aligned Sampling (MAS), a unified method for solving linear inverse problems using diffusion models that generalizes and improves upon previous approaches like DDNM and TMPD. This framework is designed to better balance prior knowledge with measurement data, and demonstrates the robustness to handle unknown, non-Gaussian, and non-differentiable noise without requiring prior knowledge of the noise structure. Extensive experiments validate the effectiveness of MAS and show that MAS consistently outperforms other state-of-the-art methods across various tasks.

### Strengths
1. The paper introduces effective techniques for handling both Gaussian noise and unknown noise sources. 

2. The proposed method enhances the robustness of diffusion models in inverse problems.

### Weaknesses
1. This paper proposes an adaptive parameter scheme with $\eta_2 = ka_t / c_t$ to generalize approaches for linear inverse problems with diffusion models. This scheme seems to be heuristic. According to Appendix B.3, this scheme is obtained based on informal principles such as "hoping $\epsilon_{intro} + \epsilon_{new}$ is as close to $\mathcal{N}(0, c_t^2 \mathbb{I})$ as possible". There is no rigorous theoretical justification on the reason why $\eta_2$ must be proportional to $a_t/c_t$.

2. The reason why a negative $\eta_1$ value leads to improvements is not well explained. It is not clear whether this is related to compensating for inherent biases in the pre-trained diffusion model or there is any geometric interpretation in the optimization objective (Equation (6)) that differs from the probabilistic model (e.g., as a super-linear interpolation between $m_{0|t}$ and the DDNM solution).

3. Review on related works is limited. The paper claims to generalize approaches for linear inverse problems with diffusion models, but neglects related works with similar claims such as [R1]-[R3]. Discussion on the difference from these works and necessary comparison are required to clarify the contributions claimed in this paper.

4. The caption of Table 4 explicitly states that the method requires manually setting different k values for different degradation tasks (e.g., k=1.0 for JPEG QF=5 and k=3.0 for QF=2). It appears to require the user to have prior knowledge of the degradation type to select an appropriate k. This undermines the ability to handle "unknown" noise as claimed in the paper.

[R1] Yismaw N, Kamilov U S, Asif M S. Gaussian is all you need: A unified framework for solving inverse problems via diffusion posterior sampling. IEEE Transactions on Computational Imaging, 2025.

[R2] Peng X, Zheng Z, Dai W, Xiao N, Li C, Zou J, Xiong H. Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance. International Conference on Machine Learning. 2024: 40347-40370.

[R3] Fei B, Lyu Z, Pan L, Zhang J, Yang W, Luo T, Zhang B, Dai B. Generative Diffusion Prior for Unified Image Restoration and Enhancement. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 9935-9946.

### Questions
Please refer to Weaknesses.

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
2

### Summary
This paper introduces Measurement-Aligned Sampling (MAS), a framework for solving linear inverse problems using diffusion models. The method optimizes a weighted objective balancing prior and measurement fidelity through parameters η₁ and η₂, with closed-form solutions via SVD. MAS generalizes DDNM and TMPD, introduces an overshooting technique, and proposes adaptive parameterization for unknown noise. Experiments show state-of-the-art performance on super-resolution, inpainting, deblurring, JPEG restoration, and quantization tasks.

### Strengths
1.  The paper provides both probabilistic (Bayesian linear regression) and optimization perspectives for a single weighted objective balancing prior and measurement fidelity, with efficient closed-form solutions via SVD decomposition.

2. MAS outperforms baselines in several tasks, while maintaining efficiency comparable to DDNM.

3. The adaptive parameterization enables effective restoration on real-world degradations like JPEG and quantization without requiring exact noise specifications, though limited to problems approximable as linear with unknown noise.

### Weaknesses
1. The k parameter in lacks principled selection criteria. Table 4 shows different values (k = 1.0, 3.0, 0.5) without rationale, which might require expensive grid searches for new tasks. This contradicts the claimed relatively low computational cost.

2.  The probabilistic foundation defines η₁ ≥ 0, yet optimal results (Figure 4) use negative η₁ = -0.45. While numerical stability is addressed, the paper lacks justification for why violating this constraint improves performance.

3.  While Table 3 provides quantitative evaluation for salt-and-pepper and periodic noise, Poisson noise appears only qualitatively in Figure 1. This leaves unclear whether the method quantitatively generalizes across diverse unknown noise types meaningfully.

### Questions
1. Can the authors provide principled guidelines for selecting k based on observable degradation characteristics? How sensitive is the method's performance to k values?

2. Can the authors provide an intuition for why negative η₁ empirically improves results despite violating the probabilistic interpretation?

### Soundness
3

### Presentation
3

### Contribution
2
