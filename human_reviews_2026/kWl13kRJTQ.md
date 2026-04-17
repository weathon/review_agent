# AC-Sampler: Accelerate and Correct Diffusion Sampling with Metropolis-Hastings Algorithm

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Diffusion-based generative models have recently achieved state-of-the-art performance in high-fidelity image synthesis. These models learn a sequence of denoising transition kernels that gradually transform a simple prior distribution into a complex data distribution. However, requiring many transitions not only slows down sampling but also accumulates approximation errors.
We introduce the Accelerator-Corrector Sampler (AC-Sampler), which accelerates and corrects diffusion sampling without fine-tuning. It generates samples directly from intermediate timesteps using the Metropolis–Hastings (MH) algorithm while correcting them to target the true data distribution. We derive a tractable density ratio for arbitrary timesteps with a discriminator, enabling computation of MH acceptance probabilities. Theoretically, our method yields samples better aligned with the true data distribution than the original model distribution. Empirically, AC-Sampler achieves FID 2.38 with only 15.8 NFEs, compared to the base sampler’s FID 3.23 with 17 NFEs on unconditional CIFAR-10. On CelebA-HQ 256×256, it attains FID 6.6 with 98.3 NFEs. AC-Sampler can be combined with existing acceleration and correction techniques, demonstrating its flexibility and broad applicability. Our code is available at \href{https://github.com/aailab-kaist/AC-Sampler}{https://github.com/aailab-kaist/AC-Sampler.}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a sampling algorithm called Accelerator-Corrector Sampler (AC Sampler) that both accelerates and corrects diffusion sampling. This sampler is based on Metropolis-Hastings (MH) algorithm. The acceleration is achieved via a “warm-start” by performing denoising from prior distribution to a target tilmestep $\tau$ which serves as the initial sample of MCMC. The MH acceptance probability is computed with the help of score function and also an additional time-dependent discriminator. This time-dependent discriminator is trained to predict the likelihood ratio of the unknown target data distribution and model’s marginal distribution at time $t$. The other terms in the expression for acceptance probability are Gaussian distributions and can be easily computed. 

The advantages of AC-Sampler have been shown on CIFAR-10, CelebA-HQ 256 and ImageNet- 64 and 256. AC-Sampler can also be composed with other SOTA samplers such as EDM’s Heun sampler, discriminator guidance, DPM-v3 etc. and it results in improved FID score. In many cases, the average NFEs are also comparable or less, which is ideal.

### Strengths
1. The proposed method is orthogonal to many existing samplers and can be combined with them as indicated in Table 1 and Table 2. Further, this composition results in improved FID in general.
2.  The proposed algorithm seems to have better mode coverage than  EDM as indicated by Recall metrics. This also has been qualitatively demonstrated against DDPM solver in Figure 6.
3. The paper provides theoretical proof (Theorem 4.3 and 4.4) which shows that the generated sample distribution with AC-Sampler is closer to the true data distribution in terms of KL divergence. The paper also provide results on the expected reduction in the number of NFEs.

### Weaknesses
1. The method requires training an additional time-dependent discriminator however it needs to be done only one time. There will  also be additional overhead from this discriminator during sampling. It is unclear if the forward pass through the discriminator is accounted for in NFEs. The discriminator also uses a pre-trained ADM classifier as a feature extractor which might not be readily available for all datasets. 
2. There is a potential mismatch between theory and practice. The practical implementation of MH in Algorithm 1 employs “propose-until-accept” design. This can introduce stationary bias as mentioned in Appendix E. This should be highlighted in the main paper. This also means that in this case, the chain wouldn’t converge to the desired target distribution $q_T$ but would rather converge to a different distribution due to the bias. 
3. Appendix B reports poor performance on CelebA-HQ 256x256 where the method in the main paper doesn’t scale to high dimensional data. Appendix B proposes to do MH algorithm in the joint space of time and data. This suggests that the primary algorithm from the main paper is not robust. It is also unclear if this method can be applied to Text-to-image models. 
4. Additional overhead in terms of wall clock time over many samplers such as DDIM, DPM-v3 etc. is unclear from the paper. There is only a  comparison against EDM’s Heun sampler in the paper.  In some cases, the NFE reduction is not very significant and therefore confidence intervals or the standard deviation needs to be reported.
5. The performance of the method is quite sensitive to the choice of $\tau$ as indicated in Table 9. In addition, there are many hyper parameters to tune such as burn-in length, number of steps to skip, number of parallel chains etc.

### Questions
1. This work  is probably relevant and could be included in related works:
Score-Based Metropolis-Hastings Algorithms, Ahmed Aloui, Ali Hasan, Juncheng Dong, Zihao Wu, Vahid Tarokh, 2024
3. What is the typical length of MCMC chain i.e. how main times do we need to repeat Algorithm 1 before generating the final sample?

### Soundness
3

### Presentation
3

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
The paper proposes AC-Sampler, a “accelerator-corrector” for diffusion models that jumps to an intermediate timestep (instead of starting at pure noise) and then applies Metropolis–Hastings (MH) with a MALA proposal built from the pretrained score network. This both shortens the reverse trajectory (speedup) and, via MH acceptance, corrects samples so their marginal at targets the true distribution. A time-dependent discriminator provides an estimate of the density ratio so the MH acceptance probability is tractable.

### Strengths
1. Clever decomposition of the acceptance ratio and use of a time-dependent discriminator make the MH step closed-form and cheap to evaluate。

2. Theoretical guarantees: Expected NFE reduction when the acceptance rate exceeds a mild threshold; KL to the data distribution does not worsen and improves with more MALA steps (under stated integrability conditions).

3. Designed to sit atop existing accelerators/correctors (e.g., DPM-v3, DG), often improving their FID at fewer steps.

### Weaknesses
1. The proposed method does not appear to provide acceleration when the batch size is 1; in this regime, the acceleration gain vanishes.

2. Hyperparameter sensitivity & chain design: Performance depends on the choice of target timestep 𝜏 and the proposal step size/SNR, as well as burn-in/chain length—these trade speed for acceptance/mixing.

### Questions
1. Is the proposed method compatible with classifier-free guidance (CFG)? While CFG has known theoretical issues, modern large-scale systems rely on it heavily.

2. Could the authors add a discussion in related work comparing their approach with SDE-based sampling methods [1–3] (which can be viewed as Langevin-type correctors) and with work that addresses training–inference mismatch [4,5]?


[1] Gotta go fast when generating data with score-based models.

[2] SA-solver: Stochastic adams solver for fast sampling of diffusion models.

[3] Seeds: Exponential sde solvers for fast high-quality sampling from diffusion models.

[4] Input perturbation reduces exposure bias in diffusion models.

[5] Improved Diffusion-based Generative Model with Better Adversarial Robustness.

### Soundness
4

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
3

### Summary
Dear authors, I am the AC. Since two reviewers ghosted the paper or wrote last-minute that they wouldn't be able to submit a review, and since I was unable to find emergency reviewers, I have now written an emergency review of the paper. 

The paper proposes a Metropolis-Hastings correction at an intermediate diffusion step \tau to accelerate the sampling process of diffusion models. The idea of integrating MCMC updates into the reverse process is conceptually interesting, and the theoretical analysis is clearly written. The experimental results show modest improvements in FID with fewer or comparable NFEs.

### Strengths
Bringing an explicit Metropolis-Hastings correction into the diffusion sampling loop is an interesting idea to integrate score-based generative modeling and classical MCMC. Prior papers have explored MH with diffusion in other contexts, such as MCMC correction for model composition and Metropolis sampling for constrained diffusion, but using an MH step specifically as a generic accelerator within the standard image-synthesis pipeline is still relatively underexplored.

The paper not only proposes a new method but also analyzes the expected NFE under acceptance/rejection dynamics (e.g., conditions under which truncating at \tau, followed by a short MH chain, reduces per-sample NFEs). 

The empirical results, while limited in their scope (see below), are promising and show consistent improvements in both terms of FID scores and efficiency.

### Weaknesses
While the MH-corrected acceleration mechanism is interesting, the claimed efficiency improvement is not convincing. The method truncates the reverse diffusion at an intermediate noise level \tau runs a short chain there, and then further denoises each accepted sample from  \tau -> 0. Thus, every accepted sample still requires a full denoising segment, meaning there is no intrinsic saving *per sample* unless multiple final samples share the same denoising sequence from T to \tau. Again, if the goal is to generate one image at inference time (in the realistic setting), there is no saving, as far as I understand. 

In the "amortized" setting, where one truncated trajectory is used for several final samples, the expected NFE per sample can indeed drop, as shown in Proposition 4.2. However, if only one sample is drawn per trajectory, the method would be slower, not faster, due to the added proposal evaluations. The paper would benefit from explicitly stating this amortization assumption or correcting my understanding of the paper. 

While the paper emphasizes improved efficiency and reduced NFEs, it does not adequately situate the proposed MH-based acceleration among existing approaches explicitly designed for fast diffusion sampling. In particular, recent methods such as Tong et al., “Learning to Discretize Denoising Diffusion ODEs (LD3)” (ICLR 2025) and the references mentioned herein (which also work with a trained diffusion model) should be compared to. Relating to my point above, the comparison should be done for generating one single sample for the same number of time steps (plus the overhead of your discriminator), not averaged over a generation of a batch of l images.

### Questions
Is my understanding correct that the improved efficiency is only observed for the (imo unrealistic assumption) that multiple images are generated for the same prior noise?

What is the exact goal of the theorems and how do they relate to practical settings? E.g., Prop 4.2: how are all the assumptions made implicitly here realistic?

Why are you not comparing to other methods for reducing the step size without having to train a new diffusion model?

Why did you not also make a comparison for generating a single (or a small number of) images?

### Soundness
3

### Presentation
2

### Contribution
2
