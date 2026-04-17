# Discrete Feynman-Kac Correctors

- Decision: Reject
- Scores: 4, 8, 4, 0

## Abstract
Discrete diffusion models have recently emerged as a promising alternative to the autoregressive approach for generating discrete sequences. Sample generation via gradual denoising or demasking processes allows them to capture hierarchical non-sequential interdependencies in the data. These custom processes, however, do not assume a flexible control over the distribution of generated samples. We propose Discrete Feynman-Kac Correctors, a framework that allows for controlling the generated distribution of discrete masked diffusion models at inference time. We derive Sequential Monte Carlo (SMC) algorithms that, given a trained discrete diffusion model, control the temperature of the sampled distribution (i.e. perform annealing), sample from the product of marginals of several diffusion processes (e.g. differently conditioned processes), and sample from the product of the marginal with an external reward function, producing likely samples from the target distribution that also have high reward. Notably, our framework does not require any training of additional models or fine-tuning of the original model. We illustrate the utility of our framework in several applications including: efficient sampling from the annealed Boltzmann distribution of the Ising model, improving the performance of language models for code generation and amortized learning, as well as reward-tilted protein sequence generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The article under consideration considers the problem of correcting the output of a pre-trained generative modeling. More specifically, assuming that $ p_0 $ is the output of the pre-trained model, the following tasks are considered

- *Annealing* One wants to sample from the law propositional to to $p_0^{\beta}$ where $\beta$ 
- *Product and geometric averaging* Here, one wants to sample from the product of two pretained models $p^1_0p^2_0$
- *Reward-tilting* One wants to sample from the law propositional  to $p_0\exp(r)$ where $r$ is a reward function.

For all these tasks, the authors derive a Fokker-Planck Equation (FPE) for the modified flows given by $p^\beta_t$,$p^1_tp^2_t,p_t\exp(r)$ up to normalizing constants. This FPE involves modified jump rates $B_t(i,j)$, whose expression is given in terms of the original rates and the probability rations $p_t(i)/p_t(j)$, and a non linear Feynman-Kac term $g_t$. The authors exploit the structure of this equation to apply  Sequential Monte Carlo algorithms (SMC) to sample from the target modified distribution. The paper is completed by numerical experiments, one per task considered: temperature annealing for the Ising model, text generation and protein guidance for reward tilting.

### Strengths
The paper proposes a seemingly new method to tackle the general problem of modifying the output of a pre-trained model. Remarkably, this method does not require any extra training step or fine-tuning. Numerical experiments are conducted across a quite large range of tasks and show promising results.

### Weaknesses
- From a mathematical standpoint, the main weakness is that there is no theoretical guarantee of convergence for the proposed methodology. I would like very much to see som e result in this direction. The derivation of the FPE for the modified flow is interesting, but follows from a rather standard calculation. This does not mean that is unimportant, of course. 

- From the methodological perspective, it appears to me that the main message is to use SMCs instead of other methods for reward-tilting and other tasks. The final methodology is then obtained assembling together two class of algorithms: SMCs and generative modeling, but I do not see new algorithmical ideas emerging from the paper. I may be wrong and I would be happy to review my assessment if the authors bring convincing evidence about the novelty of their methodology. I am not sure the experiments alone are strong enough to warrant publication. I am not the best person to assess their validity and I will abstain from judging them in detail.

### Questions
- I would expect to see the value of the parameters in the modified flows to vary over time. For example, I would expect to see the annealing temperature in Thm 3.1 to depend on $t$ so that $\beta_1=0$ and $\beta_0=\beta$, where . Is there a reason for not implementing this in practice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Discrete Feynman-Kac Correctors, a framework that enables flexible control over the generated distribution of pretrained discrete masked diffusion models at inference time. Using Sequential Monte Carlo (SMC) algorithms, the method allows for temperature annealing, sampling from products of marginals, and integrating external reward functions—all without additional training. The approach is demonstrated on the Ising model, language modeling, multi-constrained generation, and protein sequence generation, offering a versatile tool for controlled sampling.

### Strengths
- The paper presents a theoretically sound and novel approach.
- The method is rigorously evaluated across a broad spectrum of benchmarks, demonstrating its versatility and potential applicability.

### Weaknesses
- The benchmarks on the Ising model lack comparison to theoretically available ground truth solutions, which would strengthen the validation of the proposed method.
- The evaluation does not include the critical temperature regime of the Ising model, where sampling is known to be particularly challenging.
- The notation $g_t(i)$ appears in Line 122 but is not formally introduced or defined elsewhere in the text.
- With the exception of Figure 4b, the paper does not provide direct comparisons to alternative methods, limiting the ability to contextualize the performance of the proposed approach.

### Questions
- How does the method perform on the Ising model at the critical temperature? Would it be possible for the authors to compare their results to theoretically derived values for the Ising model with periodic boundary conditions, as outlined in [1]? The following publicly available script could be used to compute these values for comparison:
 [link to script](https://github.com/ml-jku/DiffUCO/blob/main/IsingTheoryBaselines/IsingTheory.py)

**Minor Comment:**
- The related work section could be enriched by discussing recent advances in discrete diffusion samplers [2] and discrete flow samplers [3, 4]. These works also focus on sampling from unnormalized target distributions, albeit through learned rather than guided approaches, and include evaluations on the Ising model.

**References:**
[1] Arthur E. Ferdinand and Michael E. Fisher. "Bounded and inhomogeneous Ising models. I. Specific-heat anomaly of a finite lattice." *Physical Review*, 185(2):832, 1969.

[2] Sanokowski, Sebastian, et al. "Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics."

[3] Holderrieth, Peter, Michael Samuel Albergo, and Tommi Jaakkola. "LEAPS: A discrete neural sampler via locally equivariant networks." *Forty-second International Conference on Machine Learning*.

[4] Ou, Zijing, Ruixiang Zhang, and Yingzhen Li. "Discrete Neural Flow Samplers with Locally Equivariant Transformer." *arXiv preprint arXiv:2505.17741* (2025).

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
4

### Summary
The paper introduces FKC to discrete diffusion models which are based on jump processes. This framework enables inference alignment of discrete diffusion models without retraining. The key contribution is the derivation of theoretical results showing how annealing, distribution product formation, and reward tilting can be achieved by reweighting and SMC methods. Empirical results demonstrate the applicability of DFKC across three domains: sampling from the Ising model, language modeling, and protein sequence generation.

### Strengths
- The manuscript is well-organized and clearly written. The exposition is concise yet thorough, facilitating a clear understanding of the core contributions and methodologies.
 - The paper presents theoretically rigorous and well-founded derivations. The mathematical treatment of annealing, distribution product formation, and reward tilting via reweighting and SMC methods is sound.

### Weaknesses
- Unclear motivation: Given the presented and evaluated inference alignment strategies, their usefulness is not convincingly demonstrated. While there is previous work showing that annealing can be beneficial when sampling from Boltzmann distributions, the current manuscript presents this possible advantage for using FKC only in a rather toy-like experiment. Potential benefits for language models are only shown for synthetic and toy tasks rather than real world language modeling at scale. Reward guidance for protein sequence generation is not thouroughly evaluated with additional metrics on quality, diversity, distributional similarity.
 - Limited novelty: FKC is adapted from diffusion to jump processes, the essence of FKC was already presented in Skreta et al. (2025). The re-weighting approach is immediate when transitioning from diffusion to jump processes. From this CTMC formulation the presented proofs are straight-forward derivations and rather mechanical.
 - Limited experimental depth: Despite the appealing breadth of evaluated domains, the depth and rigor of the experimental analysis are limited. Sampling from Boltzmann distributions should have been done for more challenging tasks such as Maximum Independent Set, Maximum Cut as done in previous work. "Armortized Learning" is a synthetic task and "Multi-constraint Story Generation" is a toy task that does not convincingly show real-world benefits in larger-scale language modeling tasks. The protein design experiments lack diverse metrics and thourough comparison with reward guidance based baselines. For all domains, a thorough comparison/discussion and cost/benefit tradeoff analysis of FKC (in particular SMC) compared to inference alignment baselines (Singhal et al., 2025; Nisonoff et al., 2024), reward fine-tuning (Rector-Brooks et al., 2024), and standard diffusion models (more diffusion/euler-maruyama steps, more capacity, longer training) is missing.
 - Datasplits and hyperparameter selection procedures for all compared methods are not described in detail.

### Questions
- What are the details on datasplits and hyperparameter selection procedures for all compared methods?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes **Discrete Feynman–Kac Correctors (DFKC)**, a general framework for **controlling discrete diffusion models at inference time** without retraining. It extends the continuous Feynman–Kac Correctors, previously applied to SDE-based diffusion models, to discrete-state continuous-time Markov chains (CTMCs) that underlie masked diffusion models. DFKC introduces a principled way to sample from modified distributions such as (i) temperature-annealed, (ii) product or geometric averages of multiple diffusion processes, and (iii) reward-tilted distributions incorporating external objectives. The approach leverages Sequential Monte Carlo (SMC) sampling to reweight and resample trajectories using quantities that can be computed directly from trained discrete diffusion models, requiring no fine-tuning or retraining.

Theoretically, the paper derives discrete analogues of the Feynman–Kac formula and forward Kolmogorov equations to justify these transformations. Empirically, DFKC is demonstrated on three domains: Ising model sampling, language modeling (amortized inference and multi-constraint text generation), and protein sequence design guided by reward functions. Across all tasks, DFKC improves controllability and sample quality compared to baselines like discrete diffusion models and existing guidance schemes. The work positions DFKC as a unifying inference-time control framework for discrete generative models, bridging probabilistic control theory and discrete diffusion processes.

### Strengths
1. **Clear and ambitious motivation:**
The paper tackles an important and current challenge in discrete generative modeling — how to control discrete diffusion models at inference time — an area that has received much less attention than its continuous counterpart.
2. **Elegant theoretical formulation:**
The authors successfully extend the Feynman–Kac Corrector framework from continuous stochastic differential equations to discrete-state continuous-time Markov chains (CTMCs), providing a clean mathematical generalization rooted in the Forward Kolmogorov Equation (FKE).
3. **No retraining required:**
A key practical advantage is that DFKC enables fine-grained control at inference time without any model retraining or fine-tuning, which is computationally attractive and broadly applicable.
4. **Algorithmic clarity and simplicity:**
The connection between Feynman–Kac theory and Sequential Monte Carlo (SMC) is presented clearly, leading to an implementable inference algorithm (Algorithm 1) that integrates reweighting and resampling seamlessly.
5. **Diverse and well-aligned experiments:**
The experiments are well-chosen to illustrate each theoretical component — annealing (Ising model), product-of-marginals (language model with multiple prompts), and reward-tilting (protein generation). The applications are thoughtfully aligned with the theoretical constructs.

### Weaknesses
1. **Lack of formal convergence guarantees:**
The paper derives correct and interpretable rate equations, but the convergence properties of the Sequential Monte Carlo (SMC) estimators in the discrete setting are not analyzed in depth. There are no explicit results on variance, bias, or sample complexity, which weakens the theoretical completeness of the 
2. **Assumption-heavy derivations:**
Several key results rely on strong idealizations, such as perfect knowledge of the marginal ratios or ergodic and reversible Markov chains. In practice, these assumptions are difficult to satisfy in large discrete models like language or protein diffusion.
3. **Weak ablation and sensitivity analysis:**
The experiments do not analyze the impact of key hyperparameters, such as the number of SMC particles, temperature schedules, or the shape of the reward function. Such ablations would clarify robustness and sensitivity of the approach.
4. **Connection to related theory could be deepened:**
The paper does not sufficiently discuss recent stochastic control–based approaches that share conceptual similarities, such as the discrete stochastic control formulations in Pham et al. (2025) [1] or reinforcement-style discrete guidance models. Drawing clearer distinctions or theoretical parallels would strengthen the framing.
5. **Interpretation and intuition:**
While the mathematics is rigorous, the exposition is sometimes technically dense and abstract. The paper could benefit from more intuitive explanations or illustrative visualizations to make the discrete Feynman–Kac concept more accessible to a wider audience.

[1] Pham, L.T.N., et al. _“Discrete Markov Probabilistic Models: An Improved Discrete Score-Based Framework with Sharp Convergence Bounds under Minimal Assumptions.”_ Forty-second International Conference on Machine Learning (ICML, 2025).

### Questions
- Can the authors provide any formal convergence or variance bounds for the SMC estimator in the discrete case?
- Is the convergence to the target distribution guaranteed under approximate marginal ratios, or does the method risk degeneracy in high-dimensional state spaces?
- Can the authors clarify whether DFKC can be interpreted as a discrete control problem where the weighting term acts as a control cost?
- Could DFKC be applied to hybrid continuous–discrete models (e.g., molecular graphs or structured data)?
- Is there a potential to integrate DFKC with learning-based control, where the corrector parameters are adapted during training?
- Some derivations (e.g., Theorems 3.3–3.5) are technically dense. Could the authors provide a high-level algorithmic summary or schematic showing how the discrete Feynman–Kac updates interact with the diffusion process?
- Could a brief comparison table between continuous FKC and Discrete FKC formulations help readers understand the correspondence?

### Soundness
3

### Presentation
3

### Contribution
1
