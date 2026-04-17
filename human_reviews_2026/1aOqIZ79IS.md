# White Gaussian Noise Constraints for Reward-Guided Generation

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We propose a constrained optimization framework that preserves white Gaussian noise characteristics during latent optimization for reward-guided generation. At its core is a novel constraint formulation that allows efficient projection while tightly characterizing white Gaussian noise. In deep generative models, supplying white Gaussian noise as input is essential for stable and realistic generation, but preserving its characteristics during optimization remains challenging. This challenge is amplified in reward-guided generation, where gradient-based updates can exploit the reward and produce unrealistic or low-quality outputs. Prior methods address this by introducing regularization terms that encourage certain white Gaussian noise properties, particularly in the spectral domain. However, regularization offers only soft penalties and cannot guarantee that the latent vector retains the white Gaussian noise characteristics throughout optimization. To overcome this, we propose a constrained optimization approach that directly projects the latent vector onto a feasible set. Leveraging a bijective mapping to a compact spectral domain, we define constraints that tightly characterize white Gaussian noise and induce a feasible set with a closed-form projection, enabling efficient updates through projected gradient ascent. In experiments on reward-guided text-to-image generation, our approach outperforms regularization-based baselines across four reward functions in terms of reward, sample quality, and maximization speed.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a constrained latent optimization framework for diffusion model reward maximization. Latents are projected to the intersection of L1 and L2 norm spheres that match the first and second moments of a gaussian distribution. They present empirical gains with respect to penalized and unconstrained approaches for aesthetic/preference rewards.

### Strengths
I think that both the use of constrained optimization tools for latent diffusion optimization is a valuable contribution that could inspire future work in this direction. I believe that the empirical evidence strongly supports the advantages of their methods with respect to baselines on terms of performance, and that their approach is computationally efficient, and thus widely applicable.

### Weaknesses
I think that the theoretical and algorithmic contributions of this paper need to be properly outlined and references to prior work have to be added.

The "construction of a compact spectral domain", i.e. using the one-sided DFT, is customary in spectral analysis of real valued signals. That the DFT is hermitian (Lemma 1) and the Fourier transform of a gaussian is a gaussian (Theorem 1) are standard results. I think it would be useful to point this out and at least include a reference to a standard signal processing textbook.

In the same vein, algorithms projecting into the intersection of the L1 and L2 spheres (the algorithm presented in section 4.3)  have been studied before, see e.g. [1]. No reference to prior works are included here.

I am not saying the use of these tools is not appropriate or that their application in this problem is not novel, just that their relation to prior work is missing altogether, and their presentation might mislead the reader into thinking these are contributions of this work.

[1] Liu, H., Wang, H., & Song, M. (2019). A unified approach for projections onto the intersection of $\ell_1$ and $\ell_2$ balls or spheres.

### Questions
Can you expand a bit more on why constraining the l1 and l2 norm to exactly match their expected values under gaussianity is beneficial?

Can you also discuss the use of blocks instead of constraints on the whole vector, and the impact of block size choice?

Can you plot the resulting L1 and L2 distributions of baseline latent optimization methods?

### Soundness
3

### Presentation
1

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
The paper introduces a constrained optimization framework for reward-guided generation that explicitly enforces Gaussianity in the latent space. Instead of using soft regularization (as in previous methods like PRNO or MPGR), the authors propose a closed-form projection that ensures the latent vector remains within a feasible set representing white Gaussian noise.

### Strengths
- The idea of replacing regularization with explicit Gaussian constraints and deriving a closed-form projection is original and mathematically elegant.
- The use of a compact spectral domain mapping (bijective and preserving Gaussianity) is both theoretically sound and practically efficient.

### Weaknesses
The experimental results are limited.
- The number of testing prompts is limited to just 60 prompts, which is a very small set of samples. For evaluation, a set of at least 1000 prompts is needed. 
- The testing dataset's domain is limited to animals. 
- The testing problem is limited to text-to-image generation. I'm curious about the model performance when being tested with different problems.

### Questions
Please see the above weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces a novel inference-time input noise optimization technique for reward-guided generation based on projected gradient ascent (with closed-form projection) on a feasible set, which characterizes *Gaussianity* via block-wise $l_1$ and $l_{2}$ norm constraints of half-spectrum representation. The proposed feasible set constraints are presented to enforce the spatial and spectral characteristics of white Gaussian noise. Experiments show that the suggested method substantially outperforms similar baselines on *one-step* text-to-image human preference reward-guided generation.

### Strengths
1. Reformulation of test-time reward-guided optimization of the input noise via projected gradient ascent is novel;
2. The motivation to use the first two moments of magnitude of half-spectrum components as characterizations of *Gaussianity* is supported by strong empirical results compared to the chosen baselines;
3. Overall, the empirical performance of the method according to the quantitative evaluation on human-preference benchmarks is superior to the other input noise optimization methods.

### Weaknesses
1. Despite utilizing the closed-form projection, the approach relies on many gradient steps per prompt. This results in a highly impractical method that performs hundreds of backward passes through both the reward model and the diffusion model to perform just *one*  reward-guided inference with *one-step* model. If the method could generalize on different prompts, it would potentially improve its applicability. In its current form, I would rather treat the method as a demonstration of the potential in optimizing the latents, than a practical option.
2. The paper repeatedly frames the method as *preserving Gaussianity*, yet the feasible set enforces just the equalities of block-wise $l_{1}$ and $l_{2}$ norms in half-spectrum. Performing projection on this feasible set is not equivalent to maintaining the original Gaussian measure. Conversely, the resulting distribution will be concentrated on a manifold and will not have Lebesgue density. Moreover, this projection does not even guarantee that the distribution after projection will be similar to the Gaussian distribution projected on the same feasible set (e.g. projection of the Gaussian distribution on an $l_2$ sphere is uniform, while here this is an almost arbitrary distribution on the feasible set). The terminology and positioning seem misleading and could be read as stronger than what the constraints actually ensure.
3. Some parts of the manuscript seem to have either too little or too much description of the underlying observations. Sections 4.1 – 4.2, for example, revisit such classic results as DFT symmetry or Gaussian concentration (Figure 1) in detail. At the same time, the intuition behind the main projection algorithm is almost non-explained and is largely deferred to the Appendix, making the paper's main result less comprehensible.

### Questions
1. Could you please tell, how sensitive are the results of the method to the block size $B$?
2. Figure 2 claims MPGR requires a *slow gradient-based projection*, implying a large complexity gap between methods and a 4000$\times$ runtime difference. My understanding is that MPGR jointly optimizes reward with both spectral regularization and moment matching in the spatial domain, resulting in almost the same complexity as the proposed method. Could you please provide details for the setting used in Fig. 2, and what *projection* you attribute to MPGR in that comparison?
3. The method differs from the prior works in two ways: it introduces *hard* constraints and combines $l_{1}$ with $l_{2}$. Could the authors explain which of the two components is responsible for the gains compared to the prior methods: the combination of $l_{1}$ and $l_{2}$, or the explicit projection onto the feasible set? What would happen if we optimise soft constraints $L_{\text{norm}}$ and $L_{\text{power}}$ jointly with gradient ascent without any projections?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies how to preserve Gaussianity in the latent optimization of reward-guided generation. The authors present a constrained optimization approach that directly imposes a Gaussianity constraint on the latent. To introduce project gradient ascent, the authors show an efficient projection update throughout a closed-form projection in spectral domain. Finally, the authors present several experiments to show the performance of the proposed method.

### Strengths
- The authors present a constrained optimization approach to impose Gaussianity on the latent prior as Gaussian noise constraints. This is a different perspective compared to previous regularization-based methods. 

- The authors analyze Gaussian noise constraints in the spectral domain by showing that the projection can be evaluated explicitly. This is important to ensuring computational efficiency. 

- Experiments show the effectiveness of the proposed method in generating realistic images.

### Weaknesses
- When maximizing the task-specific reward of latent generative models, it is unclear how preserving Gaussianity for the latent prior affects stable and realistic generation.

- The difference between constraint and regularization is not clearly discussed. For instance, we can always formulate constraints as indicator functions in regularization.

- For the proposed constrained optimization, the feasibility and optimality are not analyzed.

- For preserving Gaussianity, it does not necessarily require Gaussianity for all gradient ascent steps. For instance, we can always project the last step to be a Gaussian.

- Projected gradient ascent can be very expensive, due to the computational cost of projection step and gradient evaluation. Gradient evaluation is not always feasible, for instance reward is non-differentiable. 

- It is not discussed how does the inaccurate FFT affect the projection step, since FFT is often computed within some accuracy level.

### Questions
See comments in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
