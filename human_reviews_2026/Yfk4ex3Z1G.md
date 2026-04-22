# Noise-Adaptive Diffusion Sampling for Inverse Problems Without Task-Specific Tuning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 2, 8

## Abstract
Diffusion models (DMs) have recently shown remarkable performance on inverse problems (IPs). Optimization-based methods can fast solve IPs using DMs as powerful regularizers, but they are susceptible to local minima and noise overfitting. Although DMs can provide strong priors for Bayesian approaches, enforcing measurement consistency during the denoising process leads to manifold infeasibility issues. We propose Noise-space Hamiltonian Monte Carlo (N-HMC), a posterior sampling method that treats reverse diffusion as a deterministic mapping from initial noise to clean images. N-HMC enables comprehensive exploration of the solution space, avoiding local optima. By moving inference entirely into the initial-noise space, N-HMC keeps proposals on the learned data manifold. We provide a comprehensive theoretical analysis of our approach and extend the framework to a noise-adaptive variant (NA-NHMC) that effectively handles IPs with unknown noise type and level. Extensive experiments across four linear and three nonlinear inverse problems demonstrate that NA-NHMC achieves superior reconstruction quality with robust performance across different hyperparameters and initializations, significantly outperforming recent state-of-the-art methods. The code is available at https://github.com/NA-HMC/NA-HMC.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Noise-space Hamiltonian Monte Carlo (N-HMC), which uses HMC to search for a good initial noise for solving inverse problems. The authors additionally propose a noise adaptive version of N-HMC, which adjusts the algorithm to work with unknown noise levels.

### Strengths
1. The performance seems to be good, outperforming several widely established baselines.

2. The method of using HMC for searching better noise initialization is new.

### Weaknesses
1. *What* N-HMC is solving is unclear. Is this doing posterior sampling? The mathematical statement should be precisely provided. Currently, the derivation starts with (7), which is ad-hoc. *Where* is the posterior scores used?

2. One of the motivations for this method is that the performance is inherently free form hyperparameter tuning, which does not seem to be the case. As the method is based on HMC, there are actually *more* hyperparameters that one can adjust, including how you would define the burn-in period. Reading the appendix, I am not convinced that the method requires less efforts for hparam tuning. It actually seems to require more effort, as opposed to methods such as DPS where one can just choose a step size.

3. In the experiments, two more metrics should be reported. PSNR/SSIM/LPIPS are all distortion metrics, and reporting them all does not give a more informed picture. 1) Report the FID values (perception metric) with more than 1k, 2) Report the computational cost. The computational cost is reported in the appendix, but it should be more accessible. 90 seconds is relatively slow, which is another drawback of the method.

4. The equality for $\nabla_{x_T} \log p(y|x_T)$ is, at best, an approximation. This holds across the entirety of the derivations.

5. The noise-adaptive part is confusing. $\sigma_y$ is undefined in the main text. It starts by stating that they model the noise variance with an inverse-gamma prior, which is arbitrary. How this leads to Alg. 3, is again, ambiguous. How is $m$ set?

6. Following 5, even for unknown noise levels, methods such as DPS are fine off with choosing a static step size (e.g. 1.0), which works well across all noise levels. If the authors were to truly argue that the noise adaptive part is important, then the experiments should be conducted on real-world degradations that are off the inverse crime setting.

### Questions
1. The authors assume that $\sigma_y$ follows an inverse-gamma prior, but later admits that they use an uniformative (i.e. uniform) prior. Any clarification on this?

2. Why is phase retrieval branded as a *multimodal IP*? All inverse problems are inherently multimodal, and this may confuse the readers.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To avoid local minima or noise overfitting problem in inverse problem solving with diffusion model, the paper proposed to search a good initial noise by Hamiltonian Monte Carlo (HMC), which leads to the following sampling through an ODE staying on the data manifold. For this, the proposed method repeat updating initial noise with sampling only with 2 denoising steps. The paper also introduces noise adaptive sampling which provides robustness on various measurement noises.

### Strengths
- The paper presents motivation and methods clearly.
- The paper considers various measurement noise including impulse and speckle noises, which increases the effectiveness of the proposed method in real world.
- Extensive experiments support the effectiveness of the proposed method and gives sufficient analysis on its behavior.

### Weaknesses
- The major difference from DAPS is twofolds: the paper uses HMC instead of Langevin dynamics, and the search space is changed from image to noise space. However, both changes seems to introduce additional computational cost, which results in slower sampling.
- Missing related work [1] that update the initial noise with data fidelity gradient after sampling.
- The performance reported in Table 1 has a huge gap from the original paper. For example, DAPS for Phase Retrieval originally achieves 30.63dB of PSNR with the same setting, but it is 18.52dB in this paper.


References

[1] Diffusion Image Prior, ICCV 2025

### Questions
- Could authors explain the reason of large gap of performance between the original baseline paper and this paper?
- Could authors provide runtime comparison by setting the same number of function evaluation? Or Could authors provide performance comparison by setting the same computational budget? 
- Is there any reason for empty boxes for ReSample in Figure 10 - 16?
- What if we use the GAN or consistency model instead of diffusion model? The reviewer cannot find a strong reason to use the diffusion model from the algorithm 1.

### Soundness
3

### Presentation
3

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
This paper proposes a noise-adaptive method for solving inverse problems using diffusion models as priors. The main goal of the paper is to develop methods that adapt to the manifold structure of data, hence obtaining better performance for inverse problems under noisy scenarios.

### Strengths
This paper addresses an important problem, that is the noise sensitivity and intractability of general methods used for solving inverse problems via diffusion priors.

### Weaknesses
The paper has a few issues that needs to be addressed comprehensively (see questions for details):

- the mathematical justification is thin and derivations are unclear
- experimental results are missing some baselines proposed to address the same issues, notably noise sensitivity of inverse problem solvers.
- the method is inherently expensive as DDIM mapping should be autodiffed - the paper avoids this by using a few steps, which is not very principled.

### Questions
- Why is the method called noise-space sampling? This is a bit confusing.

- In general, there is a lot of mentions of "manifold" but I found this non-rigorous. There's really no geometric insight in any of these comments. For example, the authors mention "manifold feasibility problem" as the main motivation of their work, but this is not defined or explained. Is there any theoretical result regarding this?

- Please clarify the equation in line 211. How does first equality work? Since $\mathcal{D}(x_T) \approx x_0$, I don't understand why the first equality in line 211 would work.

-  Please provide a remark after Proposition 1 that explains and clarifies the result.

- The paper misses a reference for comparison, which also addresses the noise-robustness issue of standard solvers by adopting a second-order view:

> *Boys, B., Girolami, M., Pidstrigach, J., Reich, S., Mosca, A., & Akyildiz, O. D. Tweedie Moment Projected Diffusions for Inverse Problems. Transactions on Machine Learning Research, 2024.*

Please add this benchmark to your comparisons in your experiments. 

- To see the introduced bias of the method in a simple setting, the paper would benefit from a simple experiment, see Figure 1 of the paper cited above. Please consider adding this.

style comment: I do not think using bold text in such frequency is appropriate -- in fact, I think standard academic writing only allows italics for emphasizing - no bolds please.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Noise-space Hamiltonian Monte Carlo (N-HMC) sampler, using a pretrained diffusion prior to solve general inverse problems. N-HMC directly samples from $p(x_T|y)$ with a few-step diffusion rollout estimating $\hat x_0^*$. While reminiscent of DMPlug, it formulates recovery as sampling rather than optimization, naturally accommodating measurement noise. The authors establish theoretical guarantees for N-HMC, showing its robustness to measurement noise under mild assumptions. Based upon N-HMC, NA-NHMC is proposed to adapt to the possibly unknown measurement noise level without hyperparameter tuning. Experiments on natural images show consistent gains over prior methods, with especially strong performance in noisy inverse-problem settings.

### Strengths
- This paper is overall well-written. It categorizes and clearly explains the strengths and weaknesses of existing methods, especially highlighting how and why existing methods are sensitive to measurement noise and rely on extensive hyperparameter tuning.
- The proposed N-HMC sampler is well justified by Proposition 1 that indicates its robustness to measurement noises.
- NA-NHMC extends N-HMC to a blind inverse problem setting where the noise level is unknown. NA-NHMC coincides with N-HMC with known noise level under inverse-gamma prior assumption of the noise level, which is demonstrated both in theory and in practice.
- Experimental results show clear advantage of NA-NHMC over existing diffusion posterior samplers on image restoration tasks, especially with varying measurement noise levels.

### Weaknesses
- I found no major weaknesses in this paper, but I believe some justifications are needed. See questions below.

### Questions
- The proposed N-HMC performs $p(x_T|y)$ sampling in the noisy space with the help of a few-step sampler that estimates $\hat x_0$. Similar sampling strategy is discussed in [1], which also samples in the noisy space but follows an noise annealing scheme as ReSample and DAPS. Can the authors comment on the differences between these methods? In particular, what are the pros and cons of sampling $p(x_T|y)$ vs. sampling $p(x_t|y)$ with an annealing noise schedule? Also, it seems an empirical comparison against SITCOM is necessary as it reported better results than DMPlug and DAPS in the considered experimental setups.
- What is the reason of choosing inverse-gamma prior for $\sigma_y$? Is it a specific trick to derive proposition 2?

[1] Alkhouri et al. "SITCOM: Step-wise Triple-Consistent Diffusion Sampling for Inverse Problems", ICML 2025.

### Soundness
3

### Presentation
3

### Contribution
3
