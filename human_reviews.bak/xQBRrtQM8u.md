# Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6, 8

## Abstract
Dynamical generative models that produce samples through an iterative process, such as Flow Matching and denoising diffusion models, have seen widespread use, but there have not been many theoretically-sound methods for improving these models with reward fine-tuning. In this work, we cast reward fine-tuning as stochastic optimal control (SOC). Critically, we prove that a very specific *memoryless* noise schedule must be enforced during fine-tuning, in order to account for the dependency between the noise variable and the generated samples. We also propose a new algorithm named *Adjoint Matching* which outperforms existing SOC algorithms, by casting SOC problems as a regression problem. We find that our approach significantly improves over existing methods for reward fine-tuning, achieving better consistency, realism, and generalization to unseen human preference reward models, while retaining sample diversity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a SOC formulation for fine-tuning flow-based generative models, demonstrating that naive approaches can introduce inherent bias. They introduce a memoryless noise schedule to ensure convergence to the tilted distribution and present the Adjoint Matching objective as a scalable training objective for SOC problems. Their comparisons show improved generalization, consistency, and diversity in the fine-tuned generative models.

### Strengths
The concept of using stochastic optimal control (SOC) to fine-tune diffusion-based generative models is not new, as [1] formulates fine-tuning for diffusion models (specifically forward-reverse models) through Doob's h-transform, which involves optimally controlled diffusion processes. The novelty lies in:

* Reformulating fine-tuning of flow-matching based generative models as an SOC problem, the authors introduce a suitable cost functional for the control policy parameterized by a training neural network. Additionally, they incorporate a memoryless noise schedule to ensure factorizability, thereby eliminating inherent bias.
* Proposing an adjoint-matching objective to solve the above SOC problem. It is well known that solving SOC problems is computationally challenging due to the need for continuous gradient graph caching. Extending the adjoint methods commonly used in dynamical learning as a dynamic solver for SOC objectives is, in my view, a significant contribution.


```
[1] Denker et al., Efficient Finetuning of Conditional Diffusion Models by Learning the Generalised h-transform.
```

### Weaknesses
* While I may have missed it in the appendix, it appears that the experiments in this paper primarily address the quality of fine-tuning, without providing quantitative results on aspects of the proposed adjoint objective, such as convergence plots or memory usage statistics comparing the adjoint matching loss to traditional SOC objectives, as shown in studies like [2, 3]. From a theoretical standpoint, although the author suggests that the approach could be sufficiently applied, it would strengthen the claim to include experiments validating the numerical effectiveness of the new SOC objective.


```
[2] Nüsken and Ritcher, Solving high-dimensional Hamilton-Jacobi-Bellman PDEs using neural networks: perspectives from the theory of controlled diffusions and measures on path space.
[3] Domingo-Enrich et al., Stochastic Optimal Control Matching.
```

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper aims to fine-tune a pre-trained diffusion model. The contribution of this paper is two-fold:

First, the paper explains the problem of overoptimization when fine-tuning diffusion models based on a learned reward function. To avoid the initial value function bias problem, the paper proposes using a memoryless noise schedule, which is defined as a noise schedule that ensures that the joint distribution over the initial and final states is independent.

Second, the paper introduces (lean) adjoint matching for stochastic optimal control methods which resolves memory constraints of the discrete adjoint method and makes the continuous adjoint method more scalable by making the simulation of the adjoint ODE cheaper to compute.

### Strengths
- The paper is well-written and structured.
- Strengths of memory-less noise schedule:
    - The proposed approach is a less complex and provides an arguably more elegant solution to the initial value function bias compared to the work by Uehara et. al. [1].
    - The proposed approach is compatible with both DDPM and Flow/Bridge-Matching models.
- Strengths of (lean) adjoint matching:
    - Simple regression-based objective with circumvents memory problems associated with the discrete adjoint method
    - Simulating the adjoint ODE does not require control evaluations, making it more scalable than the continuous adjoint method.
    - Compatible with general SOC problems.
- Numerical evaluations show that the proposed approach outperforms competing methods on a variety of evaluation criteria.

### Weaknesses
- Weaknesses of the memory-less noise schedule:
    - The statement “however, there does not yet exist a simple approach which actually provably generates from the tilted distribution” is, to the best of my knowledge not true: Take for example the approach in [2, 3] and use the base measure as reference process. Then, we have the terminal cost
    
    $$
    g(X_1) =p^{\text{base}}(X_1)/p_{\text{target}}(X_1) = p^{\text{base}}(X_1)/(p^{\text{base}}(X_1)\exp(r(X_1)) = 1/\exp(r(X_1)) 
    $$
    
    which es ensures that 
    
    $$
    p_{\text{target}}(X_1) = \exp(r(X_1)).
    $$
    
    The ‘memoryless’ property thus eludes to the fact, that we need a time-reversal of the base process. A more general discussion compared to [2, 3] can be found in [4, 5].
    
- Weaknesses of (lean) adjoint matching:
    - It is not clear (at least to me) which objective function lean adjoint matching is optimizing. 
    - Lean/Continous Adjoint-matching requires simulating another differential equation which may result in computational overhead.
- Code is not public

### Questions
- Did the authors compare the proposed approach to the work by Uehara et. al. [1]?
- Did the authors test how adjoint matching works with only a few diffusion steps?
- How did the authors select the time points to discretize? Uniform in the interval [0,1]?
- Do the authors have an intuition as to why the lean adjoint matching objective outperforms the discrete/continuous adjoint matching objective? I read the explanation that the continuous adjoint needs clipping and is unstable, but are there other reasons?
- Are the authors planning to make the code public?

---

- [1] Uehara et al. (2024). Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control. *arXiv preprint arXiv:2402.15194*
- [2] Vargas, F., Grathwohl, W., & Doucet, A. (2023). Denoising diffusion samplers.  *ICLR 2023*. 2024.
- [3] Zhang, Q., & Chen, Y. (2021). Path integral sampler: a stochastic control approach for sampling. *arXiv preprint arXiv:2111.15141*
- [4] Nusken, Nikolas, et al. "Transport meets variational inference: Controlled Monte Carlo diffusions." *The Twelfth International Conference on Learning Representations: ICLR 2024*. 2024.
- [5] Richter, L., & Berner, J. (2023). Improved sampling via learned diffusions. *ICLR 2024*. 2024.

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
This paper studies reward-based fine-tuning for diffusion models. The authors frame the reward fine-tuning problem as stochastic optimal control, and point out an “initial value function bias” problem that exists in previous RLHF fine-tuning approaches. The authors propose using a memoryless noise schedule for fine-tuning in order to turn the learned distribution into the desired reward-tilted distribution without bias. Furthermore, a novel algorithm named adjoint matching is proposed to solve the stochastic optimal control problem. Experimental results show that fine-tuning a flow matching base model with adjoint matching outperforms baselines such as DRaFT, ReFL, and DPO.

### Strengths
This paper provides a theoretically sound framework for the reward fine-tuning problem, viewing it as a stochastic optimal control problem. The observation of the value function bias problem in previous approaches and proposal of using “memoryless” noise schedules are based on this view.

The proposed Adjoint Matching algorithm for SOC, casting it as a least-squares regression problem, is novel and effective.

This paper is well-written, clearly structured and easy to follow.

### Weaknesses
The main paper presented experimental results on fine-tuning a Flow Matching model, and provided pseudo-code for fine-tuning denoising diffusion models; it would be more convincing if results on denoising diffusion models are provided.

The experiments with classifier-free guidance do not seem comprehensive. It would be better if there are similar quantitative comparisons with other baselines than selected DRaFT-1.

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides theorectical insights on why optimizing a KL-regularized reward objective (which is popular and dominant in RLHF for LLM) could lead to a bias in the optimal solution for diffusion models, and how to address this issue by a proper choice of noise schedule. The paper also provides solutions on how to solve the stochastic optimal control problem by adjoint methods, and provide a more efficient alternative than the classical methods and prove its equivalence. Empirical examples are further provided to show the algorithm effectiveness.

### Strengths
This paper is theorectically well written, with detailed introduction to the literature, explanation of motivations, several interesting theorems and propositions, and provide theorectically-driven approaches for diffusion models fine-tuning. Diffusion Models fine-tuning or RLHF for diffusion models are an important direction which already contributes to improving the performance to SOTA diffusion models. This paper indeed provides a novel model-based SOC method for diffusion models alignment, which is theorectically sound and also yields good performance for tuning Flow Matching based models.

### Weaknesses
The main weakness is that the paper can be benefited from more comparisons with baseline methods empirically, specifically, there lacks a baseline in experiments in directly optimizing objective (17) (though theorectically there is value bias as shown in the paper), using stochastic control methods like adjoint matching proposed in this paper. There is comparison for this in the synthetic examples in Figure 2, but more practical downstream tasks evaluations are needed to show that the noise schedule proposed in this paper can indeed yield better performance. 

**Minors**:

1）On line 229-230, the expression "Dividing (14) by (15)" is odd as (14) is not an equality, "Plug in normalization constant (15) to (14)" might be better.

2）The adjoint method in Section 4 needs more introduction or discussion, or an earlier pointer to the part in appendix, including more clarification on what is adjoint on line 350-352, what is this loss on Equation (21); The reference in Appendix should be referred earlier instead of being deferred to until line 422.

3）In proposition 2, it is better to first define earlier of the adjoint matching objective instead of combining the definition with the proposition.

### Questions
What is the complexity/computation cost of computing/sampling adjoints in Equation (26) and (27)? 

In addition, for Diffusion-DPO, how the preference pairs are sampled for further fine-tuning? Can authors explain why Diffusion-DPO tuned models yield a decrease in performance?

### Soundness
4

### Presentation
3

### Contribution
4
