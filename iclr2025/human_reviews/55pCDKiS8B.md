## Human Reviewer 1

### Summary
Consistency (trajectory) distillation typically uses a network parameterized as $f(x, t, s) = \alpha_{t, s} F_\theta(x, t, s) + \beta_{t, s} x$, with specific expressions for the coefficient (i.e. preconditioners) $\alpha_{t, s}, \beta_{t, s}$ so that the boundary conditions are automatically satisfied. This paper proposes an efficient method to find alternative preconditioners that yield improved performance and faster training. They derive the expressions for the preconditioners by re-expressing the underlying ODE in terms using certain additional variables and propose simple objectives whose minimizers can be found analytically to set the values for these variables.

### Strengths
Consistency (trajectory) distillation represents an extremely popular approach for fast generation, solving the main drawback of diffusion models. Finding approaches to improve distillation, either the final performance achieved or its training cost, is extremely relevant to improve the practicality and applicability of large models currently being trained in multiple domains.

This paper tackles this problem by changing the preconditioning parameters used for the neural network. To my knowledge the procedure described in the paper is novel, and empirical results show that, for CTMs, the proposed approach yields a training speedup of 2x for multiple datasets.

The final proposed method is quite simple to implement, with closed-form expressions for the preconditioners. (However, these expressions involve nonlinear functions intractable expectations which are estimated with samples, meaning that the actual values obtained have both bias and variance.)

### Weaknesses
The paper deals with the consistency (trajectory) distillation case. In practice, a very relevant alternative that is also widely used (and also uses similar preconditioners for the network) is consistency training. That is, directly training fast samplers without distilling a base model. The paper does not address this problem at all. While I understand this is not necessary, and distillation in itself is an important task, it would be interesting to see whether certain ideas from this work can be extended to consistency training. (Most of the expressions for the preconditioners rely on having a pre-trained model, so unclear how to generalize this approach, if possible.)

The approach seems to help when using CTM with 2 steps or more. For single step generation performance and training curves pretty much overlap with and without the proposed approach. 1 step generation plays an important role for real time generation, and it would be very interesting to develop methods that can improve training and final performance in that setting as well. Additionally, the fact that the derived preconditioners are essentially the same as the ones naively used by consistency (trajectory) distillation raises the question of whether there are other preconditioners that can be used that might help in this case as well. Again, not exploring this does not reduce the paper’s merit, but I think it is an interesting question too.

The method does not yield any benefits when used in concert with a GAN auxiliary loss. Using this loss has been observed to lead to improved performance, and indeed, the best results reported in the paper are using the GAN loss, if I understand correctly. The proposed approach does not yield benefits (but at least does not hurt) when using this auxiliary loss.

### Questions
How does performance and the actual values for the preconditioners change as we change the number of samples used to estimate Eqs. 15 and 17 (expressions for the preconditioners)? Results in the paper are obtained estimating them for 120 different times, using 4096 samples to estimate expectations. This yields good results for 2 step sampling, at a modest computational cost. This is definitely not strictly necessary, but I think it could be interesting to see how the performance and preconditioner values change as we change the number of samples used. The estimators used have variance and bias, both of which decrease with the number of samples. How small can we make the number of samples used while still retaining good performance? Can we further boost performance using more samples?

This is all for distillation. Did you think about consistency training? These preconditioning parameters also show up in that case, but we don’t have access to the teacher, so it is unclear how to use the ideas in this work.

While multi-step consistency models have been observed to underperform CTMs, it would be nice to have values in some of the results reported. I understand training is exactly the same, so I’m not expecting the new preconditioners to help there. But it would be nice having plain multi-step CMs in some results.

What dataset is used for the GAN experiment?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper works on _Preconditioning_, a technique used in the consistency distillation of diffusion models to obtain consistency functions that directly satisfy the boundary conditions required by the problem. Preconditioning consists in linearly linking the input to the output of a network. In the literature, the choice of linear coefficients is based on intuition. The paper introduces instead a new analytical method, called _Analytic-Precond_, for setting the coefficients. The method consists in applying a parametric discretization of the probability flow ODE, and then optimizing the parameters by minimizing the gap between the optimal student and the teacher, while keeping the discretization as robust as possible. Finally, some numerical proofs show that the derived result leads to a speed-up in the inference of diffusion models.

### Strengths
The paper presents strong mathematical arguments to support the choice of coefficients, including an explanation for the CMT choices that is not just based on intuition as previous methods.
- The paper shows numerical proofs of the claims made, underlying when _Analytic-Precond_ offers no advantage (single step) and when it does (two or more steps).

### Weaknesses
- The paper might be a little hard to read for who is not familiar with distillation. I personally took a while to grasp the setting and all the notation. For example, $\phi$ is used many times before definition. It could be worth having a brief discussion about some nomenclature like _teacher_ & _student_.

### Questions
- The paper focuses on the case where $f=0, g=\sqrt{2t}$ because of the recent literature. Is the method still applicable for other choices of $f$ and $g$?
- As far as I understand, the whole discussion depends on finding a good discretization of ODE (2). Both (9) and (13) use first order (Euler) methods. Can we get more insight if we try to use a better integrator?

Minor:
- Why is $q_T$ on line 118 a 0 mean Gaussian? Do we have some condition on $\mathbb{E}[q_0]$?
- The $\lambda_t$ below equation (11) might be confused with $\lambda(t)$ in equations (5), (7).
- x$ and $x_t$ are used interchangeably in the RHS and LHS of the equations, better to be consistent. Example: line 182, line 276.
- Is the code of the experiments released?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
The paper titled "Elucidating the Preconditioning in Consistency Distillation" examines consistency distillation techniques for diffusion models, where a student model learns to follow the probability flow trajectory set by a teacher model. This distillation accelerates generation by reducing the inference steps. The paper specifically explores preconditioning, a method that combines input data with network outputs to improve stability during training. Traditionally, preconditioning has been handcrafted, but this paper introduces a theoretically optimized method named "Analytic-Precond." This new approach minimizes the gap between teacher and student denoisers, thereby improving training efficiency and trajectory alignment. Experimental results demonstrate that Analytic-Precond achieves up to 3x acceleration in training across various datasets, indicating its potential in enhancing consistency models for faster multi-step generation.

### Strengths
**Theoretical Innovation in Preconditioning**: The paper introduces "Analytic-Precond," a novel, analytically derived preconditioning method that theoretically optimizes the consistency distillation process. This goes beyond prior handcrafted preconditionings, offering a principled approach that minimizes the consistency gap between the teacher and student models. This theoretical grounding not only strengthens the methodology but also provides new insights into consistency distillation.

**Significant Training Acceleration**: Experimental results show that Analytic-Precond achieves 2-3x faster training in multi-step generation tasks on standard datasets. This improvement in speed is impactful, especially for resource-intensive applications of diffusion models, as it directly addresses the bottleneck of slow inference that has historically limited diffusion models.

### Weaknesses
- This paper does not provide whether BCM is better than CTM+ Analytic-Precond in terms of FID.
- Analytic-Precond does not perform better when GAN is incorporated into the CTM. Can the authors provide an explanation or intuition for this?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposed a general paradigm of preconditioning design in consistency distillation, which is a common technique used for accelerating the inference time of consistency models based on teacher-student training (i.e., knowledge distillation). Specifically, this paper focused on preconditioning, which is a vital technique for stabilizing consistency distillation. Compared to previous hand-crafted choices of preconditioning, this paper proposed a principled way called "Analytic-Precond" to analytically optimize the preconditioning based on the consistency gap associated with the teacher probability flow ODE. Numerical experiments on multiple datasets are included to justify the effectiveness of "Analytic-Precond".

### Strengths
1. Complete proofs are included for each proposition in the manuscript.
2. Extensive numerical experiments are provided to validate the effectiveness of the proposed methodology.

### Weaknesses
Presentation of the manuscript can be further improved by rewriting certain phrases and expanding on some technical details. For instance, the phrase "CMs aim to a consistency function" on line 134 might be better rephrased as "CMs aim to learn a consistency function". For possible ways of explaining technical details in a better way, one may refer to the "Questions" section below.

### Questions
From line 265-266 the authors mentioned that the parameter $l_t$ in "Analytic-Precond" is chosen to be the minimizer of the expected gradient norm $E_{q(x_t)}[\|\nabla_{x_t}g_{\phi}(x_t,t)\|_F]$ based on earlier work [1]. Would it be possible for the authors to further expand on why such choice ensures the robustness of the resulting ODE again errors in $x_t$? Which section/part of [1] discussed the reason behind such choice?

References:

[1] Zheng, Kaiwen, Cheng Lu, Jianfei Chen, and Jun Zhu. "Improved techniques for maximum likelihood estimation for diffusion odes." In International Conference on Machine Learning, pp. 42363-42389. PMLR, 2023.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3