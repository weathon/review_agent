# RNE: plug-and-play diffusion inference-time control and energy-based training

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Diffusion models generate data by removing noise gradually, which corresponds to the time-reversal of a noising process.
However, access to only the denoising kernels is often insufficient.
In many applications, we need the knowledge of the marginal densities along the generation trajectory, which enables tasks such as inference-time control.
To address this gap, in this paper, we introduce the Radon-Nikodym Estimator (RNE).
Based on the concept of the density ratio between path distributions, it reveals a fundamental connection between marginal densities and transition kernels, providing a flexible plug-and-play framework that unifies (1) diffusion density estimation, (2) inference-time control, and (3) energy-based diffusion training under a single perspective.
Experiments demonstrated that RNE delivers strong results in inference-time control applications, such as annealing and model composition, with promising inference-time scaling performance, and achieves simple yet efficient regularisation for training energy-based diffusion models.
Additionally, our proposed RNE is modality-agnostic and applicable not only to continuous diffusion models but also to their discrete diffusion counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel **Radon–Nikodym Estimator (RNE)** framework that unifies marginal density estimation, inference-time control, and energy-based training in diffusion models. By leveraging time-reversal theory, RNE connects marginal densities and transition kernels without explicitly solving Fokker–Planck equations. Building on this, the **Radon–Nikodym Corrector (RNC)** enables plug-and-play inference-time control—including annealing, reward-tilting, and model composition—via sequential Monte Carlo weighting. Additionally, RNE serves as a principled regularizer for energy-based diffusion training and generalizes beyond standard diffusion models to stochastic interpolants, bridge models, and continuous-time Markov chains.

### Strengths
The paper presents a theoretically elegant and practically flexible framework for diffusion-based sampling. By introducing the Radon–Nikodym Estimator (RNE) and Corrector (RNC), it unifies marginal densities, transition kernels, and various correction techniques under a single probabilistic formalism. This approach enables plug-and-play inference-time control, bridges diffusion and energy-based modeling, generalizes beyond Gaussian diffusions, and demonstrates clear empirical improvements on tasks such as annealing, ligand design, and free-energy estimation, all while maintaining mathematical rigor and conceptual coherence.

### Weaknesses
### About the bound of the importance weight

One potential weakness of the proposed method is that the guidance weight w(t) used in the conditional or composite sampling scheme is not explicitly constrained. When w(t) becomes too large or varies significantly over time, it can cause several issues:

1. **High variance in importance weights** – In the sequential importance sampling / SMC interpretation, unbounded w(t) directly amplifies the RN derivative, which may lead to weight degeneracy, where only a few particles dominate the distribution. This reduces effective sample size and lowers approximation quality.
2. **Numerical instability or reducing diversity** – Large w(t) can excessively scale the conditional drift/score term, potentially causing gradient explosion or unstable particle trajectories in continuous-time SDE sampling. And excessive weighting favors certain modes strongly, leading to collapse of particle diversity and biased approximation of the target marginal qt.

### About CFG

In your formulation, **CFG** is considered as
$$
d \tilde X_t=\left(f_t(X_t)-(1-\beta )\sigma^2\nabla\log p_t^{(1)}-\beta \sigma^2_t\nabla\log p_t^{(2)}\right)dt+\sigma d\bar W_t,
$$
however, in some practical or generalized settings, the CFG formulation takes the form:
$$
d \tilde X_t=\left(f_t(X_t)-\sigma^2\nabla\log p_t^{(1)}-\gamma \sigma^2_t\nabla\log p_t^{(2)}\right)dt+\sigma d\bar W_t,
$$
i.e., 
$$
\alpha \neq (1-\beta).
$$
Under this more general setting, certain derivations or propositions in the appendix (e.g., **Proposition H.4**) may not hold exactly.
 Could the authors clarify:

- whether their framework can accommodate the case 
  $$
  \alpha+\beta\neq 1;
  $$

- and if not, what theoretical or empirical assumptions justify enforcing 
  $$
  \alpha+\beta=1
  $$
  in CFG-like processes?

### Reading difficulty / dense derivations

Some sections, especially the appendix and derivations of the CFG and RNE formulas, are mathematically dense. Readers may find it challenging to follow all steps without carefully tracing the equations. Adding more intuitive explanations or diagrams could improve accessibility.

### Questions
### A little typo

I believe there is an issue with the use of Girsanov's theorem at line **1763**. The formulation of the Radon-Nikodym derivative seems to be incomplete, as it is missing the standard **exponential term**.

### Soundness
3

### Presentation
2

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
The paper introduces the Radon-Nikodym Estimator (RNE), a framework designed to unify various diffusion models by connecting marginal densities with transition kernels through the density ratio between time-reversal processes.  The paper also presents RNE as generalizing existing methods like FKC, TDS, and Itô density estimators, offering increased flexibility and computational efficiency.   Empirical results validate RNE across inference control, energy training, and density estimation on molecular dynamics, drug design, and Gaussian mixtures.

### Strengths
1. The RNE framework offers a conceptually appealing and potentially unifying perspective on several recent techniques for controlling and analyzing diffusion models.  

2.  The proposed RNC method appears to offer increased flexibility in designing the sampling ($a_t$) and target ($b_t$) processes compared to methods like FKC, which often have more constrained designs to avoid divergence terms.   

 
3.  The identified potential numerical instability and the introduced reference process are validated by empirical results.

### Weaknesses
1. The introduced reference process, while stabilizing the RND estimator's variance, appears to increase the computational cost.

2.  How reference process approach relates conceptually to methods that aim to directly minimize the conditional variance $Var(x_{t_i}|x_{t_{i+1}})$ or reconstruction error during the reverse process itself, rather than stabilizing the estimator of a ratio?  
 

2. While RNC offers flexibility via parameters like $(c_a, c_b)$, the practical advantage over established methods like FKC seems conditional on tuning these parameters.  

3. The paper proposes extending RNE to Continuous Time Markov Chains for discrete diffusion, but this extension is discussed briefly in the appendix (Appendix D) and lacks empirical validation in the experiments section.

### Questions
1. Does the performance gain from RNC's flexibility (tuning $c_a, c_b$) consistently outweigh the tuning effort compared to methods like FKC? How can these parameters be chosen efficiently?

2. How does the proposed method handle the numerical instability issue arising from the schedules in diffusion models as $\sigma_t$ approaches 0? 

3.  How sensitive is RNC's performance to score model inaccuracies in practice?  Are there theoretical principles governing this sensitivity or ensuring robustness, apart from the specific case noted for reward-tilting (Proposition 2.2)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Radon–Nikodym Estimator (RNE) to enhance inference-time control and energy-based training for diffusion models. RNE expresses marginal density ratios along the diffusion trajectory as products of ratios between forward and reverse transition kernels across time. The central idea is that a diffusion process and its time-reversal induce the same path measure, yielding a Radon–Nikodym derivative of one; this identity links marginal densities to transition kernels. For inference-time control, RNE supplies importance weights within Sequential Monte Carlo (SMC), providing a plug-and-play recipe that unifies several prior approaches. For energy-based diffusion, RNE serves as a lightweight regularizer that enforces consistency between model-implied marginals and transition kernels during training.

### Strengths
Strengths:
1. Generality and unification. The framework applies broadly across settings by leveraging Bayes’ rule and time-reversal: it covers diverse inference-time control tasks (e.g., annealing, reward tilting, model composition) and extends beyond SDEs to continuous-time Markov chains (CTMCs), demonstrating genuine plug-and-play utility.
2. Comprehensive empirical evaluation. The method is validated on inference-time annealing (ALDP, LJ-13), model product for multi-target SBDD, scaling with particle count, and energy-based diffusion training (2D and 100D GMM, ALDP). The breadth and relevance of experiments substantiate the approach’s practicality.

### Weaknesses
Weaknesses:
1. Limited algorithmic novelty for control design. While RNE offers a unifying lens over existing methods, the new algorithmic insights for inference-time control are modest. The experimental instantiations primarily focus on drift reweighting, leaving broader design spaces underexplored.
2. Insufficient non-asymptotic guidance. Much of the theory is asymptotic, abstracting away time discretization and finite-particle effects. Clear prescriptions for step-size selection, resampling schedules, and particle budgets (M) are limited, leaving a potential gap between theory and practice in terms of variance, bias, and weight stability.
3. Convergence guarantees are implicit. The convergence narrative relies on standard Feynman–Kac SMC theory and RNE discretization analysis, but the paper does not present a single, unified theorem asserting that, as M→∞ and Δt→0, the particle system converges to q_t and ultimately q_0 under the proposed RNE-corrected scheme. Stating assumptions and conclusions explicitly in the main text would improve readability and reduce ambiguity about the limiting operator and target-matching guarantees.

### Questions
Questions:
One confusing issues are in Section 2.1.1. To ensure the reverse-process marginal matches q_t, and if one interprets the importance weight as a rejection rate, why do the authors use self-normalized importance resampling (SMC) rather than enforcing exact q_t via rejection at each step under a suitable design of a_t and b_t (guarantee the rejection rate)? Under self-normalized SMC, with a finite number of particles, the empirical distribution of X_τ may deviate from q_τ, and this error can accumulate over time. How should M be chosen to control this error in practice?

### Soundness
3

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
2

### Summary
The paper introduces the Radon–Nikodym Estimator (RNE). The authors make the observation that in diffusion, the forward and backward processes induce the same probability measure over the paths, which translates to their Radon–Nikodym derivative or, equivalently, their ratio, being equal to one. This allows them to calculate marginal densities using just the individual transition kernels, usually learned with a neural network, and cheap to compute. Using this estimator, the authors present different applications in inference-time control and regularization for training energy-based diffusion models.

### Strengths
- The proposed estimator provides a clean and elegant way to translate transition kernel probabilities to marginal densities. The idea of exploiting the intrinsic properties of the diffusion process to derive the estimator is novel and does not require any additional simulations, having potentially better scaling properties than previous approaches.

- From my understanding, the advantage of the RNE over the Feynman-Kac Corrector (FKC) is that, compared to FKC, RNE provides an easier way to compute the importance weights for inference-time control. For FKC, the weights are accumulated over the backwards trajectory, which may be problematic, especially when introducing the necessary resampling in Sequential Monte Carlo.

- Since there is no simulation required, the RNE can be computed for two neighboring timesteps using the forward and backward kernels, allowing the authors to utilize it in training an energy-based diffusion model. This extends the applicability of the proposed estimator to more applications than inference-time control.

### Weaknesses
- The main comparison regarding inference-time control seems to be with the Feynman-Kac Corrector. Although it is established that using density-ratio estimation scales badly with the number of dimensions, the FKC paper does showcase some results on high-dimensional data (images). The proposed RNE is only applied to low-dimensional settings.

### Questions
- Does the RNE compute a different weighting term than the Feynman–Kac formula? When are the two equivalent?

- How does the number of steps affect the Radon-Nikodym estimator? If the sampling is performed with 10 or 20 steps, then the backwards and forwards kernels will be significantly different. If I understand this correctly, the estimator relies on the two kernels being 'similar enough'.

- Are the importance weights for RNE "better" than the weights computed with FKC? Is there any intuition of why one should be better than the other? Do better importance weights work with fewer particles during sampling?

### Soundness
3

### Presentation
3

### Contribution
3
