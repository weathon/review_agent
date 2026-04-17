# Latent-Space Denoising for Causal Representation Learning via Free-Energy-Guided Wasserstein Particle Flows

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Learning from corrupted observations is ubiquitous in practice, yet standard training procedures often fail under unknown nonlinear mixing and realistic noise. In causal representation learning (CRL), estimates of latent factors and their causal structure are particularly brittle to such mixing effects. We address this by denoising in a learned latent space, where the corruption approximately follows an additive noise model realized via an embedding encoder. We recover the clean latent distribution by minimizing a free-energy objective function, which couples a Kullback–Leibler divergence between the convolved clean model and the observed embedding distribution with an entropy regularizer for stability. From this objective function, we  further compute the variational derivatives, derive a weighted Wasserstein gradient, and design an explicit particle flow algorithm to carry out the latent-space denoising. The resulting denoiser functions as a drop-in module for CRL and, across noisy real-world and simulated datasets, improves overall accuracy and structural recovery relative to standard CRL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a latent-space denoising method for learning from data corrupted by unknown nonlinear mixing and noise. It uses an encoder to map observations into a low-dimensional latent space, where the corruption approximately becomes additive noise. The clean latent distribution is recovered by minimizing a free-energy objective:
$E(\rho) = D_{\mathrm{KL}} \left(q_{\rho} || \nu\right) + \lambda \mathrm{Ent}(\rho)$,
where $q_{\rho} = \rho * \varphi_{H}$ and $\varphi_{H}$ is the centered Gaussian kernel.
The authors derive the Wasserstein gradient flow of this objective, leading to a particle flow algorithm that 
progressively denoises the latent representations. 

The experiments show that this latent space denoiser can be used as a plug-in module for causal representation learning (CRL), improving the accuracy and stability of latent factor and causal structure recovery under realistic noise.

### Strengths
The paper presents a well-developed theoretical framework supported by convincing experiments with better performance compared to the baselines.

### Weaknesses
I am not very familiar with this field, and I found the paper quite difficult to follow. I struggled to understand several of the theoretical details and proofs, and I am still not confident that I fully grasp the details. I strongly recommend that the authors provide clearer explanations and more accessible notation to make the paper easier to follow, especially for readers who are not experts in this area.

### Questions
1. I am curious whether the proposed method can be extended to the case of indirect and noisy 
observations of the clean signal $X$, in the context of ill-posed inverse problems. 
Specifically, can the framework handle models of the form  $Y = A(X) + \epsilon$
where $A$ denotes a (possibly ill-posed) forward operator? This formulation is more general than 
the additive-noise setup presented in the paper.

2. The choice of the latent dimension $d_u$ is not discussed in the experimental section. 
Could the authors clarify how $d_u$ was selected for different experiments and elaborate on how the performance depends on this parameter?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors attempt to recover latent variable from observation to obtain a causal representation learning framework. The main approach is to model $Y=f(Z) + \epsilon$, and the first part is to say that X=f(Z) is identifiable and then will use X to recover Z. Experiments on several datasets is ok.

However, if my understanding is correct, the proof of identifiability of X is wrong. The basic idea of the proof is to use the fact that the characteristic function of Y equals the multiplication of characteristic of X and $\epsilon$ and to say that the distribution of X is solely determined by the distribution of $Y$. However, first if we do not assumption the distribution of $\epsilon$, we cannot get the distribution of $X$. Secondly, the derivation in line 658 is highly like not correct. What do you mean by law of Y? If 658 holds it may be a contradiction with the fact that $X$ and $\epsilon$ are independent. 

Another problem is to recover Z from X, generally, without further assumptions, just with the condition that $f$ is injective, this would not be possible.

### Strengths
As the basic framework of the paper do have some technical issues, the only strength comes from the experimental part.

### Weaknesses
1. The proof if proposition is highly likely to be wrong.

2. Recover Z from X is in generally not possible.

### Questions
See my previous comments about the technique issues.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a technique to adapt existing CRL methods to real-world observations that are corrupted. They do this by latent space denoising, where the embeddings of corrupted data are mapped to behave approximately as additive noise. Then recover a clean latent distribution via free-energy minimisation, combining KL divergence and entropy regularisation for stability. Authors demonstrate that the resulting latent-space denoiser can be seamlessly integrated into existing CRL pipelines, yielding substantial gains in accuracy and causal structure recovery on both real-world and simulated noisy datasets.

### Strengths
- This paper deals with a practical problem of applying CRL methods to real-world noise data

- The idea of performing denoising in the latent space using free-energy minimisation and Wasserstein particle flows is novel

- The authors refer to many works in the field, in-depth literature review

### Weaknesses
- Hard to follow the work, the content is very dense in some parts of the paper; easing it out would help the readers

-  The motivation to link Wasserstein denoising and causal mechanism recovery is mostly hand-wavy 

- The nonvanishing characteristic functions argument for identifiability is interesting, while the paper doesn't discuss the limitations of estimation methods 

- In lines 193-200 authors discuss how a trained encoder will result in an additive noise scenario, but the constraints required to achieve that are not that clear

- It would be really helpful to provide intuition for your theorem

### Questions
See weakness section

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of causal representation learning (CRL) from observations that are corrupted by both unknown nonlinear mixing and additive noise. The authors propose a new method called Latent-Space Denoising, which acts as a "plug-and-play" module upstream of an existing CRL model. The core idea is to first learn an encoder $h$ that maps noisy observations $Y = X + \epsilon$ (where $X=f(Z)$ is the nonlinearly mixed clean data) into a latent space $U= h(Y)$. The authors justify that, via a first-order Taylor expansion, the latent representation of the noisy data can be approximated as the latent representation of the clean data plus an additive noise term: $h(Y) \approx h(X) + J_h(X)\epsilon$. Given this latent additive noise model, the paper proposes to recover the "clean" latent distribution $p(Z)$ from the "noisy" latent distribution $p(U)$ by minimizing a free-energy objective. This optimization is implemented using a free-energy-guided Wasserstein particle flow. The resulting denoised latent representations are then fed into a standard CRL method (in this case, CCRL) to identify the underlying causal system. Experiments on synthetic and semi-synthetic benchmarks show that adding the Latent-Space Denoising module improves the performance.

### Strengths
Problem Significance: The paper addresses the critical and practical limitation of CRL methods, which often assumes the observations are the noiseless mixing of latent causal variables.

Originality: The proposed  Latent-Space Denoising method, which combines a learned encoder with a free-energy-guided particle flow for denoising in the latent space to help causal identification, is a novel and interesting approach.

Methodological Novelty: The application of Wasserstein particle flows to recover a clean latent distribution for a downstream discriminative task (CRL) looks like a technically non-trivial contribution.

### Weaknesses
Weak Justification for Latent Additivity: The paper's primary assumption that additive noise in the observation space translates to an approximate additive noise model in the latent space based on a first-order Taylor approximation. This argument is not convincing to me, as it seems to rely on the component wise identifiability of the inverse of the ground truth mixing map. Even in the noise-free cases, oftentimes the component wise identifiability cannot be guaranteed (only group wise). Even under strong conditions where the component wise identifiability can be achieved, with the additional noise in the observation, the identifiability does not automatically hold. Plus, when the noise is large the first-order Taylor approximation may not perform well.  Thus, the validity of this approximation, which is essential to the method, is not sufficiently investigated and justified.

Identifiability from Noisy Data: The paper does not theoretically address how its two-stage process (denoising then identification) impacts the identifiability guarantees of the baseline CRL model. Standard CRL identifiability results are for noise-free data. The authors need to provide justification that their denoising step successfully recovers a representation from which the true causal factors are still identifiable.

Limited "Plug-and-play" Validation: The paper claims LSD is a "plug-and-play" module, but this is only demonstrated by integrating it with CCRL. To substantiate this claim, the authors should show results with other, diverse CRL methods (e.g., VAE-based or other contrastive methods) to prove its general applicability. Plus, it would help to also compare against the most obvious and direct baseline: denoising in the observation space and then applying the CRL algorithm.

### Questions
1. Could the authors provide more empirical and theoretical evidence for the validity of the first-order Taylor approximation? 

2. Why did the authors not compare against the more direct baseline of (1) training a denoising model (e.g., DAE) in the observation space and (2) applying the CCRL model?

3. How does the proposed method provably ensure that the introduced additional steps preserves the necessary identifiability conditions for a baseline CRL model (e.g. CCRL)?

4. To better support the "plug-and-play" claim, could the authors provide results from integrating the method with at least one other CRL baseline?

### Soundness
2

### Presentation
3

### Contribution
2
