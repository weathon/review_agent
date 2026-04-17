# CL-DPS: A Contrastive Learning Approach to Blind Nonlinear Inverse Problem Solving via Diffusion Posterior Sampling

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Diffusion models (DMs) have recently become powerful priors for solving inverse problems. However, most work focuses on non-blind settings with known measurement operators, and existing DM-based blind solvers largely assume linear measurements, which limits practical applicability where operators are frequently nonlinear. We introduce CL-DPS, a contrastively trained likelihood for diffusion posterior sampling that requires no knowledge of the operator parameters at inference. To the best of our knowledge, CL-DPS is the first DM-based framework capable of solving blind nonlinear inverse problems. Our key idea is to train an auxiliary encoder offline, using a MoCo-style contrastive objective over randomized measurement operators, to learn a surrogate for the conditional likelihood \$p(\boldsymbol{y} | \boldsymbol{x}\_t)\$. During sampling, we inject the surrogate's gradient as a guidance term along the reverse diffusion trajectory, which enables posterior sampling without estimating or inverting the forward operator. We further employ overlapping patch-wise inference to preserve fine structure and a lightweight color-consistency head to stabilize color statistics. The guidance is sampler-agnostic and pairs well with modern solvers (e.g., DPM-Solver++ (2M)). Extensive experiments show that CL-DPS effectively handles challenging nonlinear cases, such as rotational and zoom deblurring, where prior DM-based methods fail, while remaining competitive on standard linear benchmarks. Code: \url{https://anonymous.4open.science/r/CL-DPS-4F5D}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Posterior sampling with diffusion models requires computing the gradient term $\nabla_{x_t} \log p(y \mid x_t)$ when solving the reverse ODE. To estimate this term, the paper trains an encoder $f$ via contrastive learning, such that the log-likelihood gradient can be approximated by the gradient of a contrastive score - defined as the inner product between $f(x_t)$ and $f(y)$. By training the encoder on synthetic measurements, the method enables blind inverse problem solving for both linear and nonlinear forward operators. Extensive experiments on Gaussian, motion, rotational, and zoom deblurring tasks demonstrate that the proposed approach outperforms existing baseline models in reconstruction quality.

### Strengths
- The paper proposes a way to solve non-linear blind inverse problem, which is not actively explored with existing baseline method, yet.
- The paper provides various empirical analysis on effects of introduced components (e.g. patch-wise inference, regularization, dictionary size...) as well as theoretical analysis. 
- The proposed method is more effective than baseline methods.

### Weaknesses
- Problem is partially blind: encoders are trained for a family of measurement, but with limited number of factors (e.g. two known angles for rotation blur).
- Missing details for the forward operations. Especially, there is not sufficient information for implementing zoom and rotation blur.
- Missing related works [1], which is originally proposed for linear inverse problem, but could be used for non-linear inverse problem too.


Reference 

[1] Diffusion Prior-Based Amortized Variational Inference for Noisy Inverse Problems, ECCV2024

### Questions
How the authors applied BlindDPS for zoom and rotation blur? As diffusion prior for blur kernel is only trained for the gaussian and motion blur, it is not straight-forward to use it for those tasks.

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
This paper introduces a framework for solving blind nonlinear inverse problems where existing diffusion models fail. It works by training an auxiliary contrastive encoder offline to learn a likelihood surrogate, which is then used as gradient guidance during diffusion posterior sampling to restore images without knowing the specific degradation operator. The evaluations are comprehensive.

### Strengths
1. The paper manages to apply diffusion models to the blind nonlinear inverse problem, a domain where previous SOTA DM-based methods often fail.

2. The core idea of using an offline, MoCo-trained auxiliary encoder to learn an amortized likelihood surrogate over a distribution of operators is reasonable and novel.

3. The empirical results are comprehensive. CL-DPS produces high-quality restorations on challenging nonlinear tasks (rotational/zoom deblurring).

### Weaknesses
1. Inference Cost: The primary weakness is computational overhead. The method requires a forward and backward pass through the auxiliary encoder at every sampling step to compute the guidance gradient. This is a significant practical limitation. Can the cost be further reduced?

2. Reliance on Special Encoders: The best-performing model requires training a separate encoder for each family of operators. This assumes the operator class is known at test time, which is a strong assumption that weakens the "fully blind" claim. The fully blind UNI model shows a consistent performance drop. While Appendix H.1 shows that a larger ResNet-50 can close this gap, this trade-off between generality and performance/efficiency is a key weakness.

3. Mismatch in Guidance: The main algorithm uses a simplified "numerator-only" guidance gradient, which deviates from the full, theoretically-derived gradient. While Appendix E provides an empirical justification (showing the denominator term is small and its inclusion gives minimal benefit for high cost), this is a disconnect between the presented theory and the practical implementation.

4. How does the model perform on OOD parameters, such as a $40^{\circ}$ rotation blur? Does it degrade gracefully or fail abruptly?

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a diffusion-based framework for solving nonlinear blind inverse problems, a setting that remains largely underexplored in the current literature.
The authors introduce a MoCo-style auxiliary encoder, which is ingeniously integrated into the diffusion inference process to address the challenges brought by both nonlinearity and blindness in the observation model.
Experimental results demonstrate that the proposed method can effectively handle complex nonlinear blind inverse problems that existing diffusion-based approaches fail to address, while maintaining competitive performance on standard linear blind inverse benchmarks.

### Strengths
1.	The paper addresses a novel and underexplored problem—nonlinear blind inverse modeling—and presents, to the best of my knowledge, the first diffusion-based solution to this setting.
2.	The proposed approach is conceptually sound and empirically effective, as demonstrated through comprehensive experiments.
3.	The writing is clear and well-structured: the motivation, background, and related work are coherently presented, and the proposed method is easy to follow.
Importantly, the authors support their claims with both theoretical analysis and solid empirical validation.

### Weaknesses
1. The main contribution of this paper lies in the design of the auxiliary encoder, but some important details remain unclear. For example, how is the training dataset for this encoder constructed?
2. The proposed CL-DPS increases the runtime by about 20% compared to the baseline. However, considering that CL-DPS can handle nonlinear blind inverse problems that the baseline methods cannot, this increase is acceptable.
3. The setup of the baselines requires further clarification. Since methods such as BlindDPS rely on the assumption of a linear observation model, it is important to explain what modifications were made so that these baselines can handle nonlinear cases while ensuring a fair comparison.

### Questions
What is the generalization ability of CL-DPS? Does it require training a specific auxiliary encoder for each dataset, or can a single encoder generalize across different data domains?

I also have a question about the input used during the training of the auxiliary encoder.
The goal of the auxiliary encoder is to cluster samples generated by the same blur kernel, which is not entirely consistent with MoCo’s objective of grouping semantically similar samples.
For example, how does the method avoid clustering together samples generated from the same x_t but under different blur kernels, given that these samples may appear very similar in feature space?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CL-DPS, a new diffusion-model framework for blind nonlinear inverse problems, removing the need to know measurement operators at inference. The method uses a contrastive learning trained encoder to approximate the conditional likelihood on some synthetic dataset. At test-time, the encoder is used to approximate the likelihood $p(y |x_t)$ and guide the diffusion sampling without operator inversion. The experiments show CL-DPS outperforms existing methods on complex nonlinear tasks and is competitive on linear benchmarks.

### Strengths
1. The method uses a synthetic dataset to train the encoder making it still a blind method which is practical.

2. The method outperforms the compared methods on different tasks.

### Weaknesses
1. The method requires some training, which increases the computational cost compared to other zero-shot methods that are training-free.

2. I have concerns about the fairness of the evaluation. Basically you are training your contrastive encoder on some degradations and then testing on them while the other methods do not. How do you think that this evaluation is conclusive on the superiority of the method given that the other methods do not have this advantage. A comparison with methods that use a synthetic dataset for training is needed.

3. To me, to prove that the method is really interesting, it should be shown that it works better than the existing training methods in the out-of-distribution case. Otherwise, I do not see a big advantage compared to simply learning the degradation operator using the synthetic dataset.

### Questions
1. How well does the method generalize to real-world, non-synthetic degradations?

2. Could the method be applied to more complex degradations such as unstructured degradations like rain for example?

### Soundness
2

### Presentation
3

### Contribution
2
