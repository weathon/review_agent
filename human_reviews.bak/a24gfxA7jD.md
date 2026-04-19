# Physics Informed Distillation for Diffusion Models

- Decision: Reject
- Scores: 3, 5, 6, 6

## Abstract
Diffusion models have recently emerged as a potent tool in generative modeling, although their inherent iterative nature often results in sluggish image generation due to the requirement for multiple model evaluations. Recent progress has unveiled the intrinsic link between diffusion models and Probability Flow Ordinary Differential Equations (ODEs), thus enabling us to conceptualize diffusion models as ODE systems. Simultaneously, Physics Informed Neural Networks (PINNs) have substantiated their effectiveness in solving intricate differential equations through implicit modeling of their solutions. Building upon these foundational insights, we introduce Physics Informed Distillation (PID), a novel approach that employs a student model to represent the solution of the ODE system corresponding to the teacher diffusion model, akin to the principles employed in PINNs. Our approach demonstrates remarkable results, such as achieving an FID score of 3.92 on CIFAR-10 for single-step image generation. Additionally, we establish the stability of our method under conditions involving a sufficiently high discretization number, paralleling observations found in the PINN literature,  thus highlighting its potential as a streamlined single-step distillation approach without the need for additional methodology-specific hyperparameters. The code will be made available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper looks to propose a knowledge distillation method for the score-based generative models, using a PINN method for ODE systems. As indicated in the original score-based model paper, for each generative SDE, there is a corresponding probability flow ODE. While it is expensive and difficult to distill an SDE model, it is much easier to distill an ODE model. The author uses existing methods from the PINN literature to distill a probability flow trained with score matching. The empirical results are promising, but some theoretical assumptions, such as the score function is Lipschitz continuous, are a bit far-fetched.

### Strengths
The empirical results from this paper looks promising for a single step generation model. The presentation is clear.

### Weaknesses
The literature of PINNs on ODE systems is very mature, and the paper did not propose innovations on how to better perform PINNs on ODE systems. Rather, it is simply applying the PINN techniques to ODE systems, limiting the contribution to developing a technique for distilling score based generative models. 

This is fine if the paper achieves state of the art performance (meaning it beats all previous benchmarks), however, as shown in Table 1, 2, 3, and Figure 7, this is not the case. The existing methods consistency model and EDM (which is heavily referred to when developing this paper) perform much better than the proposed method. 

That being said, it would be great if the authors can demonstrate qualitatively and quantitively how the proposed method is better than EDM and/or consistency model (such as distillation efficiency, inference time, etc.).

### Questions
I am concerned with the validity of the assumption that the score model is Lipschitz continuous. How often is this the case? If we do enforce this property, how much will it hurt the performance?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose Physics Informed Distillation (PID), a method for distillation of a teacher diffusion model up to single-step image generation. Inspired by models from the Physics Informed Neural Network (PINN) architecture, the method trains a student model to approximately satisfy the probability flow ODE induced by the teacher diffusion model. To speed up the training process, the residual loss is approximated using a first-order Euler discretization step, instead of having to apply backpropagation. A theoretical analysis of the discretization error is also provided.

### Strengths
- The authors make an interesting connection between distillation of diffusion models and PINNs via enforcement of the probability flow ODE.
- The authors propose PID, a relatively simple method for distillation, which shows results comparable to state-of-the-art single-step image generation for CIFAR10 and ImageNet64.
- The paper is generally well-written and clear.
- The PID distillation method achieves results comparable to current state-of-the-art single-step image generation methods (1) using an arguably simpler method involving fewer hyperparameters/training tricks.

1. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models.

### Weaknesses
- The specific parameterization of the PID model (Eqn. 7) seems somewhat undermotivated to me. The authors mention they take inspiration from the two common approaches to enforcing boundary conditions with PINNs, soft and strict conditions. However, beyond this high-level explanation, the parameterization is not justified and no ablations are performed.
- Similarly, a first-order numerical approximation of the residual loss is proposed for the sake of efficient training, but no ablations are performed as to how much this discretization affects the performance.
- A bit more background about PINNs, especially the soft enforcement of boundary conditions, could be helpful to better motivate the authors' choice of model parameterization.

### Questions
- Is there a reason why a combination of soft and hard enforcement of boundary conditions is necessary?
- Once a first-order discretization scheme for the residual loss is chosen, the student model training looks very similar in form to existing distillation techniques (1, 2). How is PID related to these techniques, e.g. can they be described as a special case of PID given a specific choice of time discretization of the probability flow ODE?
- How does the first-order approximation of the residual error loss compare to training directly using the ODE (backpropagation), or with using a higher-order approximation?
- What is the benefit of enforcing the probability flow ODE in a PINN-inspired way, as opposed to an operator learning method such as (3)?

1. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models.
2. Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models.
3. Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposed a PINN-based distillation technique for single-step diffusion sampling. The output of the trained network is equivalent to the integral of the diffusion ODE, and thus the sampling procedure can be equivalently rewritten to an ODE-solving problem with PINN methods. Experiments show that the proposed method can achieve comparable sampling results to other distillation methods such as consistency distillation.

### Strengths
- The proposed method and the corresponding nemerical methods can achieve comparable results to other distillation methods such as consistency distillation.
- The presentation is easy to follow and the algorithms are quite neat.

### Weaknesses
- Major:
    - **Lack of an important related work: BOOT[1]**. The proposed method seems **almost exactly the same as BOOT**, because they both distill the integral from time $T$ to time $t$, with the same integral and numerical differential method. Please compare with BOOT in details and discuss more about the own contirbutions.
  
  - Minor:
    - The results in Table 1 is unfair. Some of the results are based on the checkpoint of the VPSDE in ScoreSDE[2], but some of the results are based on the checkpoint of EDM[3]. The authors should at least split them into two parts.
    - The results of DPM-Solver in Table 1 seems not the best results. e.g., please see Table 1 in [4], where it involves "dpm-solver-fast".
    - A small typo: please use $\mathcal{O}(\Delta t)$ instead of $\mathcal{O}\Delta t$.
 

[1] Gu, Jiatao, et al. "Boot: Data-free distillation of denoising diffusion models with bootstrapping." *ICML 2023 Workshop on Structured Probabilistic Inference {\&} Generative Modeling*. 2023.

[2] Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." *arXiv preprint arXiv:2011.13456* (2020).

[3] Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." *Advances in Neural Information Processing Systems* 35 (2022): 26565-26577.

[4] Song, Yang, et al. "Consistency models." (2023).

### Questions
Please discuss in details with BOOT and highlight the own contributions and differences.


========================

I've read the authors' rebuttal and I think the comparison between BOOT and the proposed method is fair. However, I still think there are many similarity between BOOT and the proposed work because BOOT is the first work which combines the PINN loss with diffusion distillation. So I raised my score to 5.

========================

Thanks for pointing out the definition of concurrent work in the review guide. I think although they can be considered as concurrent work, the two works are too similar, and the author needs to discuss them more in detail, instead of only comparing the FID results. But I respect and follow the review guide because it is not the reason for rejection. However, the final results of this work are still not promising. So I consider to raise my score to 6.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The proposed physics-informed distillation (PID) distills a student model from the teacher diffusion model for single-step image generation of diffusion models. PID aims to solve the probability flow ODE by reducing the equation residual loss, which is approximated by finite difference method.  The theoretical results show that PID matches the Euler method if it achieves zero loss. PID show competitive single-step image generation performance on CIFAR10 and ImageNet64.

### Strengths
1. The single step generation ability of PID is competitive on CIFAR10. 
2. The training cost per step of PID is smaller compared to PD and CD. 
3. The training of PID does require any extra data.

### Weaknesses
1. PID cannot further improve the sample quality by investigating more NFEs. It is limited to single-step generation, where the performance is not that impressive.  
2. Equation 9 and 8 are equivalent up to a scaling factor for L2 metric, but not for arbitrary distance metric such as LPIPS, which is used for the main results. Changing $L_{PINN}$ from equation 8 to 9 will change the loss landscape. However, this step is not justified or explained in the paper. Why not use the original PINN loss given by equation 8? 
3. The authors choose LPIPS metric in the paragraph before Theorem 1. However, Theorem 1 is fully based on L2 metric, which is a bit confusing. I do not see how Theorem 1 can extend to LPIPS metric.

### Questions
1. Theorem 1 shows that PID will be equivalent to the Euler method with the same number of discretization steps N if the PID loss is zero. Can you also report the FID of the corresponding Euler method with the same $N$? This may help us understand the underlying gap between the learned model and the actual Euler method. 
2. The central difference method is more accurate than the forward difference that is used in the paper. Would it be beneficial to use the central difference for PID?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
