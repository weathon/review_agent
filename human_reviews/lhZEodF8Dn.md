# Efficient Denoising Diffusion via Probabilistic Masking

- Avg Score: 7.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 8, 8, 6

## Abstract
Diffusion models have exhibited remarkable advancements in generating high-quality data. However, a critical drawback of these models is their computationally intensive inference process, which requires a large number of timesteps to generate a single sample. Existing methods address this challenge by decoupling the forward and reverse processes, and they rely on handcrafted rules (e.g., uniform skipping) for sampling acceleration, leading to the risk of discarding important steps and deviating from the optimal trajectory. In this paper, we propose an Efficient Denoising Diffusion method via Probabilistic Masking (EDDPM) that can identify and skip the redundant steps during training. To determine whether a timestep should be skipped or not, we employ probabilistic reparameterization to continualize the binary determination mask. The mask distribution parameters are learned jointly with the diffusion model weights. By incorporating a real-time sparse constraint, our method can effectively identify and eliminate unnecessary steps during the training iterations, thereby improving inference efficiency. Notably, as the model becomes fully trained, the random masks converge to a sparse and deterministic one, retaining only a small number of essential steps. Empirical results demonstrate the superiority of our proposed EDDPM over the state-of-the-art sampling acceleration methods across various domains. EDDPM can generate high-quality samples with only 20\% of the  steps for time series imputation and achieve 4.89 FID with 5 steps for CIFAR-10. Moreover, when starting from a pretrained model, our method efficiently identifies the most informative timesteps within a single epoch, which demonstrates the potential of EDDPM to be a practical tool to explore large diffusion models with limited resources.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduced an Efficient Denoising Diffusion method via Probabilistic Masking (EDDPM) to address the computational intensity issue in diffusion models. EDDPM utilizes probabilistic reparameterization to determine whether a time step should be skipped or not, thereby identifying and eliminating redundant steps during training. This approach, which jointly learns mask distribution parameters with model weights, includes a real-time sparse constraint to significantly enhance training efficiency. Remarkably, as the model reaches full proficiency, random masks converge to a sparse and deterministic form, retaining only crucial steps.

### Strengths
The paper introduces the concept of an Efficient Denoising Diffusion Model (EDDPM) to improve the sampling efficiency. This innovation addresses a significant challenge in diffusion models, which often require a large number of steps for generating a single sample. EDDPM effectively identifies and skips redundant steps, enhancing the sampling process. In my eyes, this idea is interesting.

### Weaknesses
This article seems to be insufficiently prepared, containing various typos and using somewhat inappropriate notation, which can be confusing for readers. Additionally, the paper lacks intuitive explanations for some of the conclusions, and I will detail these issues in the following question.

### Questions
1. As far as I know, there is typically a trade-off between sampling speed and sample quality. Having fewer sampling steps usually improves the sampling speed but often results in a decline in the quality of generated samples. This observation has been discussed in numerous studies focused on accelerating diffusion sampling. Why does Figure 1 indicate an enhancement in sample quality when the number of sampling steps is very low? Is this a consequence of randomness or an mean outcome? Is there a qualitative explanation?

2. If I didn't miss it, the article doesn't employ the L0 norm. If that's the case, I recommend not introducing L0 in the notation section. Additionally, in Equation 6 and subsequent formulas, if you are using the L2 norm, I suggest explicitly writing it as $||\cdot||_2$ not $||\cdot||$.
3. Under section 3, "In this section, for the convenience of presenting our method DDPM......", I think it should be EDDPM;
4. Regarding the mask variable, $\mathbf{m}_t$, I understand it to be the $t$-th entry of the vector $\mathbf{m}$. Following tradition, a single random variable should not be represented in bold form, and it is recommended to write as $m_t$.
5. Under Eq.(9), $\tilde{\boldsymbol{\mu}}(\mathbf{x}_t,\mathbf{x}_0)$, $\tilde{\beta}_t$ should be written as $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0,\mathbf{m})$, $\tilde{\beta}_t(\mathbf{m})$, because they depend on $\mathbf{m}$. 
6. Under Eq.(9), in the definition of $\tilde{\beta}_t$ it has an extra "t" in the lower right corner.
7. Due to the L1 constraint on $\mathbf{s}$, most entries in $\mathbf{s}$ will be pushed to 0. But why are the other entries pushed towards 1? Are there situations where some $m_t$ are around 0.5? In such cases, how should step $t$ be handled? Is there any explanation?
8. Under eq.11, it should be $\nabla_{\theta}\Phi(\theta,s)$ and $\nabla_{s}\Phi(\theta,s)$.
9. Equation 12 is confusing because $\gamma_e$ is not used in the algorithm. Is this a typo? Is it the case that $K=\gamma_e*T$, and $\gamma_e$ is expressed as in equation 12?
10. In the context of image synthesis, the baseline comparison is limited. There are many acceleration sampling algorithms proposed for image synthesis, such as DPM-solver (Lu et al., 2022). While these works are mentioned in the related work section, they are not compared in the experiments.
11. There are several issues with the citation format and some references are duplicated or inappropriately cited in arXiv format, even though they have been published.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new skipping scheme, EDDPM, for denoising diffusion models. EDDPM uses parameterized probabilistic masking to decide whether to skip a diffusion time step for faster sampling speed. The proposed method probabilistically reparameterizes the discrete masking selection problem into a tractable continuous optimization problem. The experiment shows significant improvement in generation time (without losing much quality), and the learned masking scheme will reduce to deterministic masking.

### Strengths
1. The proposed EDDPM method is new. The idea of adding learnable probabilistic masking to the diffusion model is reasonable and seems to be new in the literature.
2. The derivation of the method seems to be solid.
3. The performance of EDDPM is good.

### Weaknesses
1. There are some typos and citation errors to be fixed. For example: two periods appear at the end of the first paragraph, [Bao et al., 2022a] and [Bao et al., 2022b] are duplicates, etc.

### Questions
1. Is the sparse masking result (like Fig. 2 (b)) pervasive across different datasets? 
2. I am wondering how you deal with the $\ell$-1 norm constraint on $s$ during training. Is it through projection? Is the training stable with the stochastic gradient estimator in Eq. (11)?
3. Do you have plans to release the code for open access?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new method called EDDPM to improve the sampling efficiency of denoising diffusion models. The key idea is to assign a probabilistic binary mask to each diffusion timestep, indicating whether it should be skipped or kept. The mask probabilities are learned jointly with the diffusion model to identify and eliminate redundant steps.  EDDPM is evaluated on image synthesis and time series imputation tasks. It efficiently compresses diffusion models to 20% of steps yet achieves equal or better performance than baseline methods.

### Strengths
- The proposed work addresses the critical challenge of inefficient sampling in diffusion models via a very clever and novel probabilistic masking approach. The formulation and inference are elegant and impressive.

- The experiment settings and results are solid. It's impressive to see the proposed work can achieve state-of-the-art performance in time series imputation and image synthesis with 5-20% of steps, and enables efficient step-compression of large diffusion models through one-epoch.

The presentation is also clear and easy to follow.

### Weaknesses
- The policy-gradient-based update for the prob masking is more like reinforcement learning, rather than the Bayesian variational inference. As the classical VI-based update is also feasible for inferring the Bern distribution, more discussion is encouraged on why adopting the policy-gradient-based update.  



- For the constrain $K$ of the total step, can the "Gradually Increasing Masking Rate" trick guarantee theoretically that the final learned step is constrained by $K$? My understanding is it controls the prob $p$ of the Bern distribution and the final steps are based on the random samples. Also, it's not very clear to me why the learned $s$ is doomed to be almost 0 or 1 . The L1 constraint can do it for sure. However, it seems the training procedure doesn't include the L1 constraint explicitly, but uses the  "Gradually Increasing Masking Rate" trick. Clarification on these points is encouraged.     



- There are some typos and missing things that should be fixed. For example:
1. The statement under equation 11,  "we can estimate the gradients of $\nabla_{\mathbf{s}} \Phi(\theta, \theta)$"- should it be $\nabla_{\mathbf{s}} \Phi(\theta, s)$
2. The equation 12, what's the meaning of $e_1$?

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a novel accelerated diffusion model, called Efficient Denoising Diffusion (EDDPM). EDDPM eliminates the need for manually selecting denoising steps in previous sampling acceleration methods via probabilistic masking. The probabilistic masking is parameterized to be a Bernoulli random variable and thus can be efficiently learned jointly with the model parameters. After full training, most of the probabilistic masks converge to deterministic values of either 0 or 1, retaining only a small number of important steps. Empirical results demonstrated the sampling efficiency of EDDPM over state-of-the-art sampling acceleration methods on two tasks including time series imputation and image generation.

### Strengths
- This paper is well-structured.
- The probabilistic masking technique is interesting and novel to the best of my knowledge.
- Empirically, EDDPM is more sample efficient than other baselines.

### Weaknesses
- The paper writing is not good for some parts.
- The authors did not align their work correctly in the literature.
- Some parts of the method are not clear.

### Questions
- The motivation part of this paper is not persuasive. The authors said that prior works often involve manual selection or the use of handcrafted rules, such as uniform skipping, to determine denoising steps. It is not entirely true for all efficient sampling methods. The authors are referred to check this survey [r1]. The authors may need to narrow down the scope of comparison.
  - [r1] Yang, Ling, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. "Diffusion models: A comprehensive survey of methods and applications." ACM Computing Surveys (2022).
- The literature review part of “Acceleration of DPMs” is not well-written. The authors fail to position their work in the literature and summarize the issues of prior work because their chosen scope is too wide.
- Why is it true? “Due to the constraints on s, i.e., $\lVert s \rVert_1 \leq K$ and $s \in [0, 1]^T$, the optimal $s$ would be sparse and most of its components would be either 0 or 1.”
- Section 4.2: Should the masking rate decrease gradually instead? In Eqn. (12), $y_e$ also decreases as e increases from 1 to N. The smaller K is, the fewer the number of steps is. In addition, what is $e_1$?
- Algorithm 1. 
  - What is the initialization of $s$?
  - It is unclear how to use $y_e$.
  - It is also unclear how the sparse constraint is enforced during training.
- Baselines. How is DDPM with 10% or 20% denoising steps implemented?
- Table 1. Although the algorithm is unclear, should we still expect EDDPM to become DDPM when using all the denoising steps?

**Minors**:
- Some citations are weird.
  - “However, this decoupled approach can lead to suboptimal performance (Song et al., 2020; Bao et al., 2022c; Liu et al., 2022; Bao et al., 2022b).” → It is unclear what is the purpose of putting citations here.
  - “Sohl-Dickstein et al. (Sohl-Dickstein et al., 2015) firstly introduced diffusion probabilistic models (DPMs) that they can convert one distribution into a target distribution, in which each diffusion step is tractable.” and many more→ There is a rule to put citations at the beginning of a sentence. Please follow it.
- “The training of diffusion models involves a weighted variational bound derived from the connection between diffusion probabilistic models and denoising score matching with Langevin dynamics..” This sentence is not correct.
- “Bao et al. (Bao et al., 2022c;b) proposed to estimate the optimal covariance in each timestep of the reverse process”. This sentence has a loose connection with previous sentences. It is unclear what are the benefits of the development.
- “(Luhman & Luhman, 2021) compressed the diffusion process by combining the GANs and DPMs, and the proposed model only needs one sampling step for generation.” This paper uses knowledge distillation. There is no combination of GANs and DPMs.
- The word reduced variance variational bound is confusing as reduced variance refers to another concept.
- There are some typos and grammatical errors, please correct it. To name a few: 
  - our method DDPM → our method EDDPM
  - thorough → through
  - In Section 4.1: $\tilde{\beta}t = \frac{1 - \alpha{t-1}(m)}{1 - \alpha_t(m)} \beta_t m_t$
  - Line 10 in Algorithm 1: $\nabla_\theta \Phi(\theta, s) \to \nabla_s \Phi(\theta, s)$
- Eqn. (4) (and its related sentences) should be put above Eqn. (2).
- Bulleted listings should be avoided when writing. In Section 5, some parts bulleted listing should be converted to paragraphs.
- The caption of Table 1 lacks the notation of the second-best method.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
