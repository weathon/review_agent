# Fine-Tuning Diffusion Models via Intermediate Distribution Shaping

- Avg Score: 3.33
- Decision: Accept (Poster)
- Scores: 6, 2, 2

## Abstract
Diffusion models are widely used for generative tasks across domains. Given a pre-trained diffusion model, it is often desirable to fine-tune it further either to correct for errors in learning or to align with downstream applications. Towards this, we examine the effect of shaping the distribution at intermediate noise levels induced by diffusion models. First, we show that existing variants of Rejection sAmpling based Fine-Tuning (RAFT), which we unify as GRAFT, can implicitly perform KL regularized reward maximization with reshaped rewards. Motivated by this observation, we introduce P-GRAFT to shape distributions at intermediate noise levels and demonstrate empirically that this can lead to more effective fine-tuning. We mathematically explain this via a bias-variance tradeoff. Next, we look at correcting learning errors in pre-trained flow models based on the developed mathematical framework. In particular, we propose inverse noise correction, a novel algorithm to improve the quality of pre-trained flow models without explicit rewards. We empirically evaluate our methods on text-to-image(T2I) generation, layout generation, molecule generation and unconditional image generation. Notably, our framework, applied to Stable Diffusion v2, improves over policy gradient methods on popular T2I benchmarks in terms of VQAScore and shows an 8.81% relative improvement over the base model. For unconditional image generation, inverse noise correction improves FID of generated images at lower FLOPs/image.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces fine-tuning methods for diffusion models through intermediate distribution shaping. The authors first unify rejection sampling-based fine-tuning methods as GRAFT (Generalized Rejection sAmpling Fine-Tuning) and prove it implicitly performs PPO with reshaped rewards, enabling marginal KL regularization for diffusion models despite intractable likelihoods. They then propose P-GRAFT (Partial-GRAFT), which fine-tunes only until intermediate denoising timesteps by assigning final generation rewards to partial noisy states. This is justified through a bias-variance tradeoff: while variance of rewards conditioned on intermediate states increases with noise level (Lemma 4.3), the score function becomes easier to learn at higher noise levels (Theorem 4.4). Additionally, they introduce Inverse Noise Correction for flow models, which trains a model to correct the distribution shift in initial noise without explicit rewards. Empirically, P-GRAFT applied to Stable Diffusion v2 achieves 8.81% relative improvement in VQAScore over the base model on text-to-image benchmarks and outperforms policy gradient methods. For unconditional generation, Inverse Noise Correction improves FID at lower FLOPs/image.

### Strengths
* The unification of rejection sampling methods (RAFT, RSO, BoN) under the GRAFT framework with explicit connection to PPO is novel and theoretically grounded (Lemma 3.2)
* P-GRAFT represents a genuinely new approach to diffusion model fine-tuning by operating at intermediate noise levels rather than final outputs
* The bias-variance analysis providing theoretical justification for intermediate distribution shaping is insightful and well-executed
* Mathematical formulations are generally clear with good notation

### Weaknesses
1. **Limited theoretical analysis for P-GRAFT optimality**: There is no formal analysis of the optimal choice of $N_I$ or convergence guarantees. The paper relies heavily on empirical validation across different $N_I$ values, but theoretical guidance for selecting $N_I$ would strengthen the contribution.

2. **Assumption dependencies**: Lemma 5.1 requires $\eta L < 1$ for backward Euler, but the paper doesn't discuss how this constraint affects practical step size choices or what happens when the Lipschitz constant $L$ is large. The impact on computational cost is not analyzed.

3. **Gap between theory and practice**: The theoretical analysis assumes continuous-time SDEs, but experiments use discrete-time DDPM/DDIM schedulers. The discretization error and its interaction with P-GRAFT is not discussed.

### Questions
1. How does the optimal $N_I$ depend on task characteristics (reward smoothness, data modality, model capacity)? Can you provide theoretical or empirical guidance for selecting $N_I$?

2. Theorem 4.4 shows score functions at later times are exponentially closer to Gaussian. Have you verified this empirically by measuring $H_t^T$ at different timesteps?

3. The backward Euler method (Algorithm 6) requires solving a fixed-point equation. How many iterations $N_b$ are typically needed in practice, and what is the computational cost relative to forward sampling?

4. Can P-GRAFT be combined with classifier-free guidance training? How does the guidance scale interact with the intermediate timestep choice?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors investigate:
(1) Rejection sampling and it’s uses in sampling and fine-tuning diffusion models. The authors derive the sampling distribution for top-k sampling, with arbitrary reward functions and de-duplication criterion, for diffusion models. The authors then propose fine-tuning on the trajectories yielded by rejection sampling.

(2) An inverse noise correction algorithm that trains a flow model from N(0, 1) to the t=1 distribution of the flow model, and then sample from the flow model.

### Strengths
See questions

### Weaknesses
See questions

### Questions
Will the authors clarify what their contributions are exactly, as

1. The sampling distribution of top-k sampling is known:
    1. As the authors acknowledge, the derivations are known from Amini et al 2024. The derivations there are not specific to any particular kind of method of generation, it applies to any generative model, including both AR, diffusion models. See theorem 2 in Amini et al 2024. 
2. Top-k sampling for fine-tuning has been proposed and shown in RAFT, see Dong et al

Questions regarding fine-tuning with Top-k sampling:

1. What distribution does alg 2 (P-GRAFT inference) sample?  Alg 2 stiches together two different diffusion models, with the first one being fine-tuned and sampling from the reward tilted intermediate distribution, followed by the base model. 

Questions regarding the motivation for inverse-noise correction:

1. The motivation for the method is not clear from the text, either in the main paper or appendix. 

Experimental Clarifications: 

1. Can the authors clarify what objective was used to fine-tune the diffusion model in alg 1 (P-GRAFT train)
2. The GenEval scores reported in the paper for SDv2 and SDXL-base are higher than those reported in the GenEval paper. Can the authors clarify what prompts were used for producing the numbers reported in table 2. 

Minor clarification regarding using the term PPO distribution: PPO is a method for learning the reward tilted distribution, p(x) exp(r(x)). This distribution can be learned using methods other than PPO.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper recognizes the importance of reward-guided fine-tuning of diffusion models and the complexity of leveraging proximal PG methods due to intractable likelihoods. The authors introduce a unified rejection sampling framework, show that fine-tuning according to it leads to the same solution of proximal PG schemes, and introduce a method for controlling also the tilt of intermediate distributions of the diffusion process. They present a mathematical analysis for bias-variance trade-off, and propose a noise-correction scheme for reward-independent improvement of flow models. Ultimately, the present an experimental evaluation of the proposed contributions.

### Strengths
- The paper correctly identifies a limitation of RL-based schemes for fine-tuning of diffusion models (i.e., intractable log likelihoods), a timely task with high-relevance.

- The idea proposed within Sec. 5 seems interesting, but I could not fully understand its logic on an intuitive level.

- The experimental sections includes diverse dataset spanning images, layout gen., and molecular design.

### Weaknesses
The paper presents an excessive amount of ideas, each not sufficiently justified, motivated, or explained. As the paper structure is very complex due to the multitude contributions, I will discuss weaknesses of each contribution a-d as listed in Sec. 1.

**a)** 
1. The paper often seems to confuse a *problem*, in particular solving an entropy-regularized MDP, with an algorithm, i.e. PPO. I understand that the presented algorithm has solution corresponding to the optimal solution of entropy-regularized MDPs, which can be tackled by PPO, but what does 'GRAFT enables PPO' or 'implicitly perform PPO' even mean? This confusion seems repeated several times within the work. Concretely, I believe these statements are effectively wrong, as the propose method, while inducing the same solution, does *not* enable/perform PPO.
2. It is well-known that a general class of inference-time schemes for diffusion induces this solution class (see [1], Eq. 1). One such case is [2] (see Sec. 3.2). So it seems there is already a variety of inference-time schemes solving this problem via diverse techniques (c.f., [1]). Moreover, there exist fine-tuning control-theoretic schemes that can solve this entropy-reg. problem as well (e.g., [3]) without value bias problem, as also mentioned by the authors. Ultimately, leveraging inference-time schemes solving this problem to then fine-tune a model, which seems to me the core algorithmic idea here, is already presented in [1, Sec. 9] arguably in a more performative fashion than the one presented within this paper (i.e. , via policy distillation rather than plain training). As a consequence, it seems to me that there isn't significant novelty within the presented contribution.

**b)**
1. The KL is typically enforced on the data level distribution as a way to preserve the high-probability set learned by the pre-trained model. The justifications presented within the paper (both in theory 4.1 and experiments) do not seem convincing to me regarding why we should instead enforce the KL-reg. at another time-step. In particular, it seems to me that the theoretical investigation in Sec. 4.1 does not really provide a concrete answer to this question. In a sense, it shows that this problem might be easier, but this does not imply (practical or theoretical) relevance. Since the authors here are presenting a novel problem setting, it would be essential to motivate it clearly.

**c)**
- (writing) this section (i.e. Sec. 5) is not clearly written to the point that I could not fully grasp the presented idea. The problem tackled seems not particularly related with the problems treated in the rest of the paper, and is not sufficiently formalized to properly understand the gains of the proposed methodology. I would strongly suggest to also introduce algorithmic aspects with an intuitive presentation of their workings before/after presenting their implementation.


**Overall (writing/structure)**
The paper is poorly structured and lacks a solid narrative. Concretely, it presents multiple ideas without sufficient clarification of their motivation, and/or workings. The text often lacks conceptual explanations of new concepts/mechanisms and their implications. 


**References**:

[1] Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review, 2025

[2] Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding, 2024

[3] Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control, 2024

### Questions
- Did I misinterpret or misunderstand any of my points above within (a) or (b)?  
- What is the core algorithmic intuitive idea for the method introduced in Sec. 5?

### Soundness
3

### Presentation
1

### Contribution
2
