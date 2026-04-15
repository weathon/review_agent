# Heavy-Tailed Diffusion with Denoising Levy Probabilistic Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Investigating noise distributions beyond Gaussian in diffusion generative models remains an open challenge. The Gaussian case has been a large success experimentally and theoretically, admitting a unified stochastic differential equation (SDE) framework, encompassing score-based and denoising formulations. Recent studies have investigated the potential of \emph{heavy-tailed} noise distributions to mitigate mode collapse and effectively manage datasets exhibiting class imbalance, heavy tails, or prominent outliers. 
Very recently, Yoon et al.\ (NeurIPS 2023), presented the Levy-Ito model (LIM), directly extending the SDE-based framework to a class of heavy-tailed SDEs, where the injected noise followed an $\alpha$-stable distribution -- a rich class of heavy-tailed distributions. 
Despite its theoretical elegance and performance improvements, LIM relies on highly involved mathematical techniques, which may limit its accessibility and hinder its broader adoption and further development. 
In this study, we take a step back, and instead of starting from the SDE formulation, we extend the denoising diffusion probabilistic model (DDPM) by directly replacing the Gaussian noise with $\alpha$-stable noise. 
By using only elementary proof techniques, we show that the proposed approach, \emph{denoising L\'{e}vy probabilistic model} (DLPM) algorithmically boils down to running vanilla DDPM with minor modifications, hence allowing the use of existing implementations with minimal changes. 
Remarkably, as opposed to the Gaussian case, DLPM and LIM yield different training algorithms and different backward processes, leading to distinct sampling algorithms. 
This fundamental difference translates favorably for the performance of DLPM in various aspects: our experiments show that DLPM achieves better coverage of the tails of the data distribution, better generation of unbalanced datasets, and improved computation times requiring significantly smaller number of backward steps.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper leverage a intrigue design, decomposing a heavy-tailed r.v. into a Gaussian and a standard alpha-stable r.v. , to propose a new heavy-tail diffusion model with minor modifications to the DDPM. Experiments showed its superiority.

### Strengths
1. The math derivation of the alpha-stable dm loss is rigorous.
2. This way of building heavy-tail DM is easier to understand and implement, because DLPM only need sampling to the heavy-tail distribution, there is no need to combine the heavy-tail distribution, into the network.
3. It is novel to make alpha-stable DM solvable via a decomposition.

### Weaknesses
1. I have a doubt on the convergence property, for t>0, the marginal distribution of this new dynamic should have infinite variance. Then really it would converge to the data distribution (with finite distribution) at t=0? Can you show me more theoretical convergence proof?
2. The term 'Data Augmentation' typically means increase the size of the dataset by creating new data through modifications or transformations of the original data. I am quite confused why the technique in section 3.2.2 is called 'Data Augmentation'.

### Questions
see Weaknesses

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
This work studies the diffusion-based models with a heavy-tailed noise and achieves great performance under the imbalanced, heavy-tailed data distribution. Different from the previous LIM method, which starts from a continuous SDE perspective, this work follows the DDPM framework and proposes DLPM and DLIM models. For the proposed methods, this work also shows a suitable training objective to achieve stable training for the $\alpha$-stable distribution. The synthetic and real-world experiments also support their results.

### Strengths
1. The link between the $\alpha$-stable distribution and Gaussian noise is helpful since we can adapt the DDPM framework by a simple transformation.
2. The suitable training objective of DLPM bridges the gap between theory and application. On the contrary, the LIM needs to choose a $L_1$ loss to guarantee stable training.

### Weaknesses
It would be better to discuss the experiment results in detail. Please see the Question part to find the details.

### Questions
Q1: When considering CIFAT10_LT, LIM performs better than the DLPM method, which can not support the discussion of this work. Could the author discuss it in detail?

Q2: As shown in the theoretical part, if $\alpha=2$, the DLPM algorithm is equivalent to DDPM. However, the experiment results of DLPM ($\alpha=2$) are worse than DDPM. Could the author discuss this experiment phenomenon in detail?

Q3: Could the author discuss the reason for the slightly worse performance of $DLPM^{\text{in}}$?

### Soundness
3

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
3

### Summary
In the context of generating data from a heavy-tailed distributions, the authors introduce a new generative model termed the Denoising Levy Probabilistic Model (DLPM). Previous work by Yoon et al (2023) had introduced the Levy-Ito Model (LIM) which is an extension of the SDE from Score-Based Diffusion Models (SBDM) by Song et al (2021) to the case of heavy-tailed distributions, where the noise added at each step follows an $\alpha$-stable distribution. On the other hand, the authors extend the Denoising Diffusion Probabilistic Model (DDPM) from Ho et al (2020) to the case of heavy-tailed distributions. It is well known that DDPM is equivalent to a particular discretization of the Variance Preserving SDE from Song et al (2021). However, when extended to heavy-tailed distributions, LIM and DLPM are not equivalent. In fact, the framework of DLPM is simpler and more principled than the LIM and allows for reusing previous code implementations of DDPM. A deterministic sampler counterpart to DLPM is introduced and showed to outperform the deterministic counterpart of LIM when few sampling steps are considered in experiments. More experiments are provided comparing the different models with mixed success.

### Strengths
* The presentation is clear and the necessary background is succesfully introduced.  There is a good flow on building the DLPM as a natural extension of the DDPM.
* The provided framework for DLPM has the nice property that it has DDPM as special case. More precisely, DLPM at each noising step adds a sample from a $\alpha$-stable distribution. In the case of $\alpha=2$, DLPM reduces to DDPM (except for a square root in the loss.) This is mathematically interesting and puts DDPM into a more general and flexible framework. 
* The authors do a good job comparing DLPM with LIM. In particular, DLPM uses more elementary tools and a more clear presentation than LIM. Yoon et al (2023) develop the theory for the LIM framework using an $\ell_2$ loss while the experiments are done using an $\ell_1$ loss. In contrast, the model for DLPM that is theoreticallly studied is the same as the one used for the experiments.
* The reparameterization provided in assumption D2 ensures that the function that the neural network estimates does not depend explicitly on the $\alpha-$stable distribution which means that previous implementations for DDPM can be reused.

I find that their main contribution is to provide a better and simplified mathematical framework for diffusion when the noise added can have heavy-tails.

### Weaknesses
I do not find the claim that the authors make on the experiments as satisfactory as the theoretical counterpart. Especially in proposing a new framework (for a task that had an already existing framework, i.e., LIM) it is important to ask whether the new framework leads to better performance in a class of tasks. 

1) The experiment for CIFAR10_LT is not conclusive: we see in Table 3 that in terms of FID, LIM is no worse than DLPM and for most $\alpha$ LIM actually outperforms DLPM. On the other hand, DLIM does outperform LIM-ODE measured by FID, but only when taking 25 sampling steps. The plot in Figure 4 shows that taking 100 steps or more LIM-ODE is comparable or outperforms DLIM. The only claim that could be drawn from here is that DLIM (the deterministic sampler for DLPM) is better when constrained to a few number of steps than LIM-ODE (the deterministic sampler for LIM.) Even this is not supported by further experimental data provided in the appendix, Table 6, where it is shown that LIM-ODE has better $F_1^\text{pr}$ scores than DLIM for all $\alpha.$ A more fair claim is that the performance of DLPM/DLIM is comparable to that of LIM/LIM-ODE.

2) Also, in Table 4 we see that the FID for DLPM at $\alpha=2$ is 21.07 while the FID for DDPM is 19.05. This is a big difference and seems to weaken the point that DLPM at $\alpha=2$ is similar to DDPM. We know DDPM and DLPM at $\alpha=2$ are different since one has a square root in the definition of the loss.

3) It seems to me important to consider $\alpha=1.5$ for the experiments whose results are shown in Table 1 and Table 2. Indeed, Yoon et al (2023) work in the interval $\alpha \in [1.5, 2]$ and we further see that performance for LIM is clearly increasing as $\alpha$ decreases both for the results in Table 1 and Table 2.

4) It is claimed in the paragraph above Table 2 that DLPM5 has a better performance than DLPM over all the range of $\alpha$ for data coming from a Gaussian grid but a t-test shows that this difference is not statistically significant for $\alpha=1.7$ and $\alpha=1.8.$ Also, the results reported in Table 6 for $\alpha=\{1.7,1.8,1.9\}$ do not seem to be statistically significant for supporting the claim that DLPM outperforms LIM, however, standard deviations are not provided.

5) It also seems the experiments missed an important comparison with the ODE counterpart of DDPM, the DDIM from Song et al (2020). That is, even if DLIM outperforms LIM-ODE, it is still relevant to compare it with the non-heavy-tailed version of the ODE framework, which is DDIM.

If this concerns about the experiments are addressed, it is likely I will increase the score.

Smaller comments
* It is not clear why the results for the non-isotropic noise case are mentioned since they give consistently worse results. It may be better to just mention that non-isotropic noise does not provide better performance and push the results about it to the appendix. One advantage of DLPM mentioned is that it can handle the non-isotropic noise, but as said before, it does not seem to give better results.
* line 383 it says 'appart' it should say 'apart.'

### Questions
The fact that the neural network does not have the $\alpha-$stable variable as input seems a bit magical. I may be missing something, but it seems that we build all this framework considering the $\alpha-$stable noise but then it only appears implicitly through $Y_t$ and in the conditioning in the loss. I think the reader would benefit from a bit more explanation regarding where is the $\alpha-$stable noise appearing even though the neural network does not model it.

What is behind the empirically seemingly big difference between DDIM at $\alpha=2$ and DDPM mentioned in Weaknesses item (2)?

Could you provide the results for results for $\alpha=1.5$ and $\alpha=1.6$ available for the experiments with results depicted in Table 1 and Table 2? (c.f. the comment in Weaknesses item (3).)

### Soundness
2

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
DDPMs degrade images by adding Gaussian noise. This paper proposes substituting Gaussian noise with alpha-stable random variables and defines DLPMs, which are diffusion models with heavy-tailed noise. As the authors solely focus on the discrete-time setting, their exposition is simpler and easier to understand than a previous already existing extension of diffusion models to heavy-tailed noise.

### Strengths
The paper is clear in its exposition and makes a good effort to clarify its difference with closely related prior work (Yoon et al). The derivations to make the training loss implementable are interesting.

### Weaknesses
1. The training loss is unstable, poorer approximations perform better: The authors overcome the bottleneck of not having analytical KL divergences available through conditioning. Under this conditioning, the denoising network still has to learn the conditional expectation of a heavy-tailed distribution contrary to what is said in the paper. In practice, in order to be implementable, the conditioning requires a Monte Carlo approximation of an expectation. In the experiments, approximations of the loss with a single sample perform better than approximations with 25 samples. This coupled with the authors' observation that using with alpha-stable distributions with $\alpha < 0.5$ is "too unstable" to train underlines the fact that heavy-tailed distributions are not a practical tool for approximating distributions. 
2. Extension for extension's sake: The work to extend to alpha-stable noises is theoretically interesting but appears to be an extension purely done for extension's sake. Moreover, it follows prior work that has already extended SDE diffusion models to heavy-tailed noises. The authors can strengthen their work by showing why alpha-stable noises are an interesting extension through better experiments.  The authors could define better what it means to be heavy-tailed for bounded distributions like images. Are there real distributions (in the sense of natural signals) where heavy-tailed phenomena have been observed and could the authors show DLPMs are better suited there? For example in section 4.1 why isn't DDPM included in the comparisons? In section 4.2, the FIDs reported for DDPM appear to be a little too high. The FID on CIFAR10 for EDM, a simplified DDPM, (Karras et al 2022) is 1.97, so MNIST should not be harder. In my view, the current numbers, especially in Appendix G.4, do not exhibit a strong enough advantage for DLPMs.

### Questions
Proposition 9: If the network is initialized close to 0 and $\hat{\epsilon}^\theta \approx 0$, the training loss derived in proposition 9 becomes an expectation of $A_t$ an alpha-stable random variable with $\alpha < 1$. Could the authors explain why this loss is not also infinite like in the LIM setting?

### Soundness
3

### Presentation
3

### Contribution
2
