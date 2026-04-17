# Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Scaling inference methods such as Markov chain Monte Carlo to high-dimensional models remains a central challenge in Bayesian deep learning. A promising recent proposal, microcanonical Langevin Monte Carlo, has shown state-of-the-art performance across a wide range of problems. However, its reliance on full-dataset gradients makes it prohibitively expensive for large-scale problems. This paper addresses a fundamental question: Can microcanonical dynamics effectively leverage mini-batch gradient noise? We provide the first systematic study of this problem, revealing two critical failure modes: a limitation due to anisotropic gradient noise and numerical instabilities in complex high-dimensional posteriors. We resolve both issues by proposing a principled gradient noise preconditioning scheme and developing a novel, energy-variance-based adaptive tuner that automates step size selection and informs dynamical numerical guardrails. The resulting algorithm is a robust and scalable microcanonical Monte Carlo sampler that consistently outperforms strong stochastic gradient MCMC baselines on challenging high-dimensional inference tasks like Bayesian neural networks. Combined with recent ensemble techniques, our work unlocks a new class of stochastic microcanonical Langevin ensemble (SMILE) samplers for large-scale Bayesian inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the performance of microcanonical Langevin dynamics, leading to a robust and scalable minibatch microcanonical Monte Carlo sampler.

### Strengths
The paper proposes a minibatch microcanonical Langevin sampler, which is robust and scalable,  with promising numerical experiments.

### Weaknesses
A key limitation of the paper is that the proposed algorithm currently lacks formal convergence guarantees.

### Questions
1. Algorithm 1: How is the INTEGRATORSTEP implemented? Please detail it. 

2. How does the proposed algorithm behave for multimodal distributions?  For example, a 10-dimensional mixture Gaussian distribution with well separated modes. 

3. Any theoretical justifications for the convergence of the proposed algorithm?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work considers a stochastic gradient variant of microcanonical Langevin Monte Carlo, proposing a preconditioning method to account for anisotropic mini-batch noise and an adaptive tuning scheme to mitigate sensitivity to the step size.

### Strengths
- Just as methods such as SGLD and pSGLD were developed from the LMC, it seems natural and well-motivated that a similar line of research would emerge for the MCLMC as well.

- Section 3 is easy to follow, and the overall procedure seems sound. The Gaussian assumption for gradient noise is quite common, and the diagonal preconditioning is also constructed in a typical moving-average fashion, as seen in many prior works. The small-scale experiments also provide adequate empirical support.

### Weaknesses
- In essence, the preconditioning introduced in Section 3 aims to ensure that the stationary distribution matches the target posterior under anisotropic mini-batch noise. However, if we proceed as described in Section 4, it seems that the stationary distribution may no longer coincide with the target posterior; the guardrail-induced forced reversion could potentially disrupt proper posterior sampling.

- Table 3 includes Laplace and IVON baselines. Are those with K=1, S=8? If SGHMC, SMILE, and pSMILE (8) are using K=8, S>1, then for a fair comparison, consider the K=8, S>1 setting, i.e., MultiLaplace (8) and MultiIVON (8).

- The proposed algorithm ultimately results in an adaptive step-size scheme. In Bayesian deep learning, Zhang et al. (2020) demonstrated that an explicit cyclical schedule can effectively balance exploration and exploitation in SGMCMC algorithms. It seems that this is only briefly mentioned in Appendix E.1, but a more direct empirical comparison appears to be missing.

### Questions
- Could you clarify how many Monte Carlo samples were used for the values reported in the tables? Specifically, what are the K and S settings?

- In Table 3, the performance of IVON seems quite low. Did you perform proper hyperparameter tuning? From my experience running IVON on ResNet with CIFAR, it tends to be quite sensitive to hyperparameter choices.

- It seems that the guardrail mainly serves to prevent occasional large updates. Have you considered using a simpler gradient clipping strategy instead? In IVON, for example, element-wise gradient clipping is applied to stabilize the updates.

- Looking at Table 4, it seems possible to compute various additional evaluation metrics beyond Accuracy and LPPD. Why not use them for comparison with the baselines, rather than only for ablation purposes?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Scaling inference methods to high-dimensional models remains a major challenge in Bayesian deep learning.
Microcanonical Langevin Monte Carlo (MCLMC) is a promising approach; however, it requires computing gradients over the entire dataset, which can be computationally expensive. This paper explores whether mini-batch approximations can address this issue.
The authors also investigate the ensemble version on MCLMC, MILE, which uses multiple chains from different starting locations.

Overall, the paper tackles an important and timely problem, but in my opinion, the presentation needs some improvement before publication.

### Strengths
The paper addresses an important problem.

•	The paper demonstrates that the naïve version of MCLMC, which uses stochastic gradients instead of full-batch gradients, fails to achieve good results. This variant is referred to as SMILE-naïve.

•	To address this issue, the authors propose a gradient noise preconditioning scheme.

•	Prior work by Robnik & Seljak (2024) has shown that the stationary distribution of MCLMC equals the target posterior 𝑝(theta|D)  for any injected isotropic noise. In Appendix A, the authors extend this result, showing that it also holds for the continuous-time SMILE-naïve formulation if the mini-batching noise can be modeled as an isotropic Wiener process in the limit. However, in practice, the noise from mini-batching often has a position-dependent covariance matrix V(theta). Although V(theta) is not known, the authors propose a method to estimate its diagonal elements, enabling preconditioning to standardize the mini-batching noise in the stochastic gradient updates.

•	While this gradient preconditioning step helps mitigate the isotropic noise issue, it still fails to produce satisfactory results for large neural networks. To overcome this, the authors introduce an adaptive tuning scheme.

•	I appreciate the detailed citations of related work provided in Section 2.

### Weaknesses
•	The first three pages consist mainly of the introduction and discussion of related work. While this section is useful and well written, unfortunately, there isn’t enough space left in the main paper to adequately present the details of the new contributions.

•	Perhaps because of the limited space in the main paper, I found the presentation a bit dense and at times difficult to follow.

Some examples of missing details:

•	Line 92: “theta(k,s), k \in [K], s \in [S]} from K independent chains”. These chains have not been defined. To make the paper more self-contained and accessible, this definition would be useful.

•	Details of experiments in Table 1 are missing. What was theta? What was P(D| theta)?

•	These details are missing in Table 2 as well. What was theta in these experiments? What were the dimensions of the inputs and outputs in these regression problems?

•	Line 358: In the modelling of the energy error section, the authors write: “we then dynamically fit a Gamma distribution using moment-matching based on the empirical parameters… Quantiles from this dynamically fitted distribution as a measure of extremity are then efficiently approximated using the Wilson-Hilferty transform”. The paper would be clearer and easier to follow if precise mathematical expressions were provided for all the steps.

•	Line 178: Similarly, when the authors write: “the second-order Minimal-Norm integrator” is used to numerically solve (3), but the details are missing, and I wish these details were at least in the Appendix.

•	It would have been nice to have a more complete theoretical analysis of the proposed method. It is motivated by theoretical arguments, but a complete theory is still missing. We don’t know how much we suffer from the curse of dimensionality; there are no convergence rates, etc.

### Questions
Line 119: This equation approximates the numerical error, as a change in total energy per step, but it is not clear if this error is big or small, or how big issues these errors can cause.

Line 206: Since V(theta) is not fully estimated, only its diagonal elements are estimated. Even after pre-conditioning with the estimated diagonal elements, the noise won’t become isotropic. It is not clear if this can still be an issue.

See also my comments above in the Weaknesses section.

### Soundness
3

### Presentation
2

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
This paper explores how microcanonical Langevin dynamics, an MCMC method that evolves samples on constant-energy surfaces, can be scaled to large-scale Bayesian inference using mini-batch gradients. The mini-batch version of microcanonical Langevin dynamics, called stochastic microcanonical Langevin dynamics, introduces noisy gradients, which prevents convergence to the target distribution when the noise is anisotropic, an inevitable effect of mini-batch randomness. To address this issue, the authors propose two strategies: (1) gradient-noise preconditioning and (2) energy-variance-based adaptive tuning. Together, these methods adaptively reduce the bias induced by anisotropic noise.

### Strengths
The techniques of the paper jointly resolve anisotropic noise bias and numerical instability, enabling scalable and robust Bayesian inference. The proposed SMILE and pSMILE methods show theoretical grounding and consistent empirical gains over SGHMC across Bayesian neural networks, image classification, and language modeling tasks.

### Weaknesses
1. Although the paper benchmarks against strong baselines such as scale-adapted SGHMC, it omits comparison with several strong adaptive SGMCMC methods, e.g., SGFS (Ahn et al., 2012), pSGLD (Li et al., 2016), MSGLD (Kim et al., 2020), cyclical SGMCMC (Zhang et al., 2020), MAMBA (Coullon et al., 2021), and PX-SGMCMC (Kim et al., 2025). Especially, cyclical SGMCMC is a strong baseline in this field. Including these would clarify whether the improvements stem from the microcanonical formulation itself or solely from application of noise conditioning and adaptive step-size tuning. This will emphasize the reason why we use pSMILES instead of adaptive SGMCMC methods.

2. The paper conducted comparisons using multiple-chain SGMCMC methods. However, to properly assess the mixing time of an algorithm, it should be evaluated in a single-chain setting. One of the key properties of SGMCMC methods is their ability to explore multi-modal posteriors within a single chain, and relying solely on a multi-chain strategy may overestimate this capability. Comparison with baseline methods in the practical setting, such as image classification, is necessary to be conducted using the single-chain SGMCMCs.

### Questions
1. How sensitive is performance to these parameters in practice when not tuned extensively?
2. Could the authors report or estimate the per-step computational overhead of SMILE/pSMILE relative to SGHMC, especially regarding the adaptive tuning and energy estimation steps?

### Soundness
3

### Presentation
2

### Contribution
2
