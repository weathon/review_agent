# Continuous-Time Discrete Markov Bridge

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Discrete diffusion has recently emerged as a promising paradigm in discrete data modeling. However, existing methods typically rely on a fixed-rate transition matrix during training, which not only limits the expressiveness of latent representations—a fundamental strength of variational methods—but also constrains the overall design space. To address these limitations, we propose **Discrete Markov Bridge**, a novel framework specifically designed for discrete representation learning. Our approach is built upon two key components: *Matrix*-learning and *Score*-learning. We conduct a rigorous theoretical analysis, establishing formal performance guarantees for *Matrix*-learning and proving the convergence of the overall framework. Furthermore, we analyze the space complexity of our method, addressing practical constraints identified in prior studies. Extensive empirical evaluations validate the effectiveness of the proposed **Discrete Markov Bridge**, which achieves an Evidence Lower Bound (ELBO) of \textbf{1.38} on the Text8 dataset, outperforming established baselines. Moreover, the proposed model demonstrates competitive performance on the CIFAR-10 dataset, achieving results comparable to those obtained by image-specific generation approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper under consideration studies a class of generative model to sample from a discrete data distribution $p_0$, called Discrete Markov Bridge (DMB). In contrast with most concurring models DMB also optimizes over the choice of prior. This phase, called *Matrix Learning* complements the more classical *score learning* phase, in which score approximation is performed by minimizing the expected KL divergence between the distribution at time $T$ for a given choice of prior (noising process) and conditional distribution at time $T$ for the same choice of prior given the initial state.

### Strengths
The main strength of the paper is to propose a generative framework that allows for flexible prior distributions that can in principle adapt to the underlying data distribution and evolve along iterations. The effectiveness of this idea is corroborated by numerical experiments on CIFAR-10 and Text 8. I like this idea and I believe it deserves to be further explored. The numerical experiments are promising.

### Weaknesses
- The main problem is that theoretical guarantees are too weak and the numerical experiments alone, though promising, are not in my opinion strong enough to compensate for the lack of theoretical results. To be more precise

    - Proposition 4.1 appears to me as a generic statement about Markov chains that use nothing of the specifics of DMB
    - Proposition 4.2 is again a very general results saying that in principle the method can be used to sample from a given data    distribution $\mu$. Moreover, its statement is also incomplete as the most interesting part is the upper-triangular shape of the transition matrix Q, which is not mentioned.
   - Theorem 4.7 states under vague and imprecise assumptions that $D_{KL}(\mu|p_0^{(k)})$ is converging.  But this is not a guarantee of convergence. I would rather expect a statement like $\lim_kD_{KL}(\mu|p_0^{(k)})=0$, but I could not find this result in the paper, and after reading the proof of Theorem 4.7, I could only infer that the $D_{KL}(\mu|p_0^{(k)})$ is decreasing along iterations. 

In conclusion, the theoretical results do not provide with any convergence rates, do not take into account the various sources of error (time-discretization, score approximation…) and even under idealized assumptions do not seem to guarantee the convergence of the algorithm to the target distribution. I may have misunderstood something, in which case I am happy to review my assessment.

- Most of the key statements lack rigor or precision, there are many typos and missing details. To give an example, in the pseudo-code for Algorithm 2 on top of page 5, which basically summarizes the contribution of this work I encountered the following issues

   - The authors propose to update the prior transition matrix $Q_\alpha$ according to (5) and predict $p_T$ according to (4). But (5) is just a loss function. So how how is the update actually done? I don’t feel like equation (14) clarifies this well enough. 
    - In the same spirit, how is the update of the score estimator $s_{\theta}$ performed according to equation (8)? At first glance, it seems quite costly as for each iteration, the prior evolves. Therefore, one needs to approximate all transition rates $p_{t|0}$ at each iteration and generate forward many trajectories at each iteration. How is this actually done? 
    - The sampling algorithm used is described quite vaguely as some form of Euler scheme for ODEs. It needs more explanations and detail. Also, I expect probability ratios like $ p_t(y)/p_t(x) $ to take both very large and very small values. How are these problems handled?
    -  $ \mathcal{J}_{Q} $ is not defined before. I guess it should be $ J_Q $. The quantity \mathcal{L}_Q does not seem to be defined before, though it probably is again  J_Q  . Similarly,  \mathcal{J}_{score} is not defined before: I assume it coincides with J_score

### Questions
- Has the idea of using a flexible prior been exploited before in the context of continuous diffusion models based on the Ornstein Uhlenbeck process? If so, with what results and outcomes?

- I understand that the definition of J_score is taken from previous works. What is its interpretation? Does it carry a probabilistic meaning as some averaged relative entropy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the foundation of the diffusion model and integrates the variational inference into the framework. The authors introduce a learnable transition rate matrix for both the forward and reverse processes.

### Strengths
1. The paper offers rigorous derivations of the forward–reverse coupling and provides formal proofs of convergence.

2. The proposed method achieves competitive performance, which is on par with or better than previous discrete diffusion baselines.

3. The paper is generally well-organized, includes clear notation, and presentation.

### Weaknesses
1. Experiments are constrained to Text8 and CIFAR-10. No large-scale or multimodal datasets are tested.

2. The advance is related to the SEDD model (Lou et al., 2024) and Variational Diffusion Models (Kingma et al., 2023). The method remains structurally similar, with the main novelty being a learnable rate matrix. It would be better for the author to establish the connection and clarify the novel point more clearly.

3. The paper does not provide ablation studies for the effects of the learnable rate matrix, continuous-time parameterization, or transition efficiency. It is unclear how much each contributes to the improvements.

### Questions
Please see the Weaknesses.

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
The paper proposes a new framework for discrete generative modeling that combines diffusion and variational approaches. It claims to address the rigidity of existing discrete diffusion models, which rely on fixed transition matrices, by introducing a **learnable continuous-time transition matrix**. The model consists of two components: a **Matrix-learning stage** that estimates the forward transition dynamics, and a **Score-learning stage** that reconstructs the data distribution via an ELBO-based objective. Formal guarantees of validity, accessibility, and convergence for their algorithm are discussed and it is given an efficient matrix structure that allows tractable exponentiation. Empirically, DMB is reported to outperform previous discrete diffusion models on `Text8` and achieve competitive image generation results on `CIFAR-10`. The contribution is mainly conceptual, positioning DMB as a general bridge between variational inference and discrete diffusion modeling, though the work remains largely theoretical and limited in empirical depth.

### Strengths
1. **Novel conceptual framework:**
The paper proposes a new paradigm that combines variational inference and discrete diffusion through the formulation of a continuous-time discrete Markov process, which is an original direction in discrete generative modeling.
2. **Learnable transition dynamics:**
Unlike most prior discrete diffusion models using fixed transition matrices (Absorb, Uniform), DMB introduces a learned transition-rate matrix, increasing model flexibility and expressiveness.
3. **Clear algorithmic decomposition:**
The two-stage structure (Matrix-learning and Score-learning) provides a clean, interpretable separation between the forward and reverse processes, similar to continuous diffusion methods but adapted to discrete spaces.
4. **Computational insights:**
The proposed diagonalizable matrix form allows efficient matrix exponentiation and reduced space complexity, addressing a key bottleneck in discrete diffusion computation.
6. **Empirical competitiveness:**
Despite its generality, the model achieves state-of-the-art performance on `Text8` and competitive image generation results on `CIFAR-10`, showing that the method can scale across modalities.
7. **Potential generality:**
The framework can in principle encompass various discrete domains (e.g., text, symbolic, or categorical data) and may serve as a unified foundation for discrete representation learning.

### Weaknesses
While the paper introduces an interesting conceptual framework, it currently falls short of the expectations for an ICLR-level contribution in terms of novelty, theoretical depth, and empirical validation. The work reads more as a promising exploratory idea than as a mature and rigorously substantiated contribution supported by solid theoretical or experimental evidence.

---

Main Concerns
1. **Limited theoretical novelty**
The proposed _Discrete Markov Bridge (DMB)_ framework closely resembles existing discrete diffusion approaches (e.g., D3PM, SEDD), extending them into a slightly more general variational formulation. The main theoretical results (_validity, accessibility, convergence_) appear to be formal restatements of well-established properties of continuous-time Markov processes, rather than providing genuinely new insights into discrete generative learning.
2. **Weak empirical evaluation**
The experimental validation remains limited in scope. Only two datasets are considered (`Text8` and `CIFAR-10`), with relatively few baselines and no ablation or sensitivity analysis. The reported improvements (for instance, a gain of 0.1 BPC on `Text8`) are modest and likely fall within the variance of previously reported results. This makes it difficult to assess the claimed advantages of the proposed approach convincingly.
3. **Insufficient connection to recent literature**
The discussion of related work is incomplete and does not engage deeply with recent progress in discrete and score-based generative modeling (e.g., Lou et al., 2024; Meng et al., 2023; or more recent flow-matching approaches).
In particular, **estimating ratios in discrete settings is notoriously difficult**, and recent advances have proposed alternative formulations that bypass this issue by defining the score as an $L^2$ approximation rather than a direct ratio (see [1]).
Moreover, there is relevant ongoing work on **discrete simulation in hypercubes**, which provides mathematically sound convergence guarantees under minimal assumptions, as well as recent insights on potential quantum extensions of discrete score-based models ([2]).
A more substantial comparison with these directions would be essential for positioning the contribution within the current theoretical landscape.
4. **Overstated theoretical claims**
The theoretical results rest on strong and somewhat idealized assumptions—such as perfect optimization, linearity of the dynamics, and exact reversibility. As presented, it is **not straightforward to see why the convergence results in Theorems 4.7 and D.2 imply convergence to zero** in practice, as the current derivations do not seem to support this claim rigorously. The guarantees therefore appear more formal than actionable.
5. **Methodological originality and clarity**
The methodological distinction of DMB compared to prior frameworks is limited. The idea of coupling a forward and backward process via learned matrices has been explored in several diffusion and flow-based settings. Here, the main innovation—replacing fixed matrices with learnable ones—does not seem to bring a clearly demonstrated advantage, and the motivation for the additional complexity remains somewhat unclear.

---

**References**

[1] Pham, L.T.N., et al. _“Discrete Markov Probabilistic Models: An Improved Discrete Score-Based Framework with Sharp Convergence Bounds under Minimal Assumptions.”_ Forty-second International Conference on Machine Learning (ICML, 2025).

[2] Bach, Francis, and Saeed Saremi. _“Sampling Binary Data by Denoising through Score Functions.”_ arXiv preprint arXiv:2502.00557 (2025).

### Questions
See **Weaknesses** section

### Soundness
3

### Presentation
3

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
This paper proposes a Continuous-Time Discrete Diffusion Model (CTDDM) that unifies continuous and discrete generative processes under a single stochastic differential framework.
While standard diffusion models assume continuous-valued states and Gaussian noise, the authors design a hybrid diffusion process operating on discrete state spaces (e.g., categorical or binary variables) parameterized by continuous-time transition kernels.
A key contribution is deriving the Kolmogorov forward equation for discrete diffusion with a continuous-time clock, enabling tractable training and sampling via reparameterized discrete noise schedules.
The paper establishes theoretical guarantees for consistency with continuous diffusion limits, introduces a novel variational training objective, and empirically validates improvements on text and graph generation benchmarks.

### Strengths
- Unifying framework: Derives a continuous-time theory for discrete diffusion, bridging an important conceptual gap.
- Strong theory: Clear proofs of convergence from discrete to continuous dynamics and vice versa.
- General applicability: Applicable to discrete domains such as text, graphs, and symbolic reasoning.
- Empirical evidence: Shows improved stability and sample diversity over prior discrete diffusion baselines (D3PM, MaskGIT).
- Analytical insight: The generator-based view clarifies how score matching extends to discrete probability fluxes.

### Weaknesses
- Notation overload: Sections 3–4 are mathematically dense; many operators $(Q_t, G_t, L_t)$ appear with minimal intuition.
- Empirical scope: Experiments focus on small or synthetic datasets; large-scale benchmarks (e.g., LM1B or large graph datasets) are missing.
- Comparative analysis: Comparison with recent hybrid discrete-continuous works (e.g., SEDD, SMC-Diffusion) could be more explicit.
- Practical guidance: It remains unclear when CTDDM should be preferred over purely discrete models or continuous relaxations.
- Computational cost: Continuous-time simulation of discrete jumps may introduce inefficiency; wall-clock comparisons are not discussed.

### Questions
- Generator specification: Is the continuous generator $G_t$ assumed time-homogeneous or time-dependent? If time-varying, how is it parameterized in practice?
- Diffusion limit: In Theorem 3.2, what conditions guarantee convergence to a continuous diffusion as the discrete grid refines? Are there counterexamples when ergodicity fails?
- Training objective: How is the ELBO-like objective derived from the pathwise KL divergence? Could you provide a short derivation for clarity?
- Score estimation: How is the score (gradient of log-prob) represented in the discrete case—via logit differences, or by interpolation between categorical probabilities?
- Sampling complexity: Does simulating continuous-time discrete jumps require adaptive step sizes or event-based simulation (e.g., Gillespie-style)?
- Empirical fairness: Are baselines tuned with equivalent training budgets? Some prior discrete models depend strongly on temperature annealing.
- Variance reduction: Does the continuous-time formulation mitigate gradient variance compared to purely discrete noise schedules?
- Hybrid variables: Can CTDDM handle mixed discrete-continuous data (e.g., tabular or multimodal)?
- Graph generation: For graph tasks, are transitions applied to node/edge labels independently or via structured coupling?
- Broader implications: Could this framework unify masked-token diffusion and continuous score-based text generation (e.g., Diffusion-LM, MaskGIT)?

### Soundness
4

### Presentation
3

### Contribution
3
