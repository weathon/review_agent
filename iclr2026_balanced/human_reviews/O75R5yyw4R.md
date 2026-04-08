## Human Reviewer 1

### Summary
The paper proposes IterRef (“Iterative Reward-Guided Refinement”), a new inference-time (test-time) scaling method for discrete diffusion models that aligns generated outputs with reward functions (e.g., safety, fluency, sentiment) without retraining. Unlike prior single-pass methods, IterRef performs iterative noising–denoising refinement of intermediate states using a Multiple-Try Metropolis (MTM) framework, theoretically ensuring convergence to a reward-aligned distribution.

Empirically, IterRef significantly improves reward-guided generation for both language (MDLM, LLaDA-8B) and image (MaskGIT) models across tasks like toxicity reduction, sentiment control, and CLIP alignment—achieving up to 8× efficiency gains at equal compute. The authors also find that, in discrete diffusion, applying refinement to later denoising steps is most effective, contrary to continuous diffusion models.

### Strengths
* Introduces a principled, MCMC-based (Multiple-Try Metropolis) test-time refinement specifically for discrete diffusion — a gap in prior work.
* Provides a convergence guarantee toward a reward-aligned distribution, not just a heuristic.
* Performance: Consistent, large empirical gains (text + image), especially under low compute budgets (up to 8× efficiency).
* Flexibility: Supports selective refinement timesteps and adjustable iteration/candidate trade-offs.
* Insight: Reveals that late denoising steps are most influential for discrete diffusion guidance.

### Weaknesses
Theory–practice gap: The practical MTM variant simplifies away exact detailed balance; convergence guarantees may not strictly hold.
Limited evaluation metrics: Focuses mainly on reward scores (toxicity, CLIPScore) without checking fluency/diversity side effects.
Baseline fairness: Competing methods may not be fully re-tuned for the discrete setting.
Compute realism: “Equal NFEs” ignores real wall-clock cost differences between reward and generative models.
Missing failure analysis: No qualitative examples of when IterRef fails or over-optimizes rewards.

### Questions
* How does the practical MTM variant (without backward proposals or with pool reuse) maintain or approximate detailed balance?
* Does local convergence at each timestep guarantee global alignment of the final output x0x_0x0​? Any empirical validation?
* How do you ensure IterRef doesn’t overfit the reward (e.g., loss of fluency or diversity)?
* Were baselines like FK, SoP, and SVDD re-tuned under identical compute budgets for discrete diffusion?
* Can you report real wall-clock or GPU-time comparisons, not just NFEs?
* How stable is performance when varying which timesteps are refined (set UUU)?
* Can you show failure or degenerate cases to better understand limits of IterRef?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes IterRef, an inference-time alignment method for discrete diffusion models. The method adapts multiple-try Metropolis-Hastings (MTM) for iterative refinement of partially unmasked sequences, with the motivation that this iterative remasking and unmasking procedure allows the model to “correct” errors. Specifically, a noising+denoising transition kernel generates $N$ candidates, one is selected at random, and then it is accepted/rejected based on an acceptance ratio. This process is repeated $k$ times. The authors provide some implementation techniques to reduce the computational cost, most notably by performing this refinement procedure at select timesteps $U$. Experiments on diffusion language models and discrete image models show strong results, and ablation studies help understand the role of key parameters, including $k, N$, and $U$. The authors also perform experiments on safety alignment to reduce the toxicity of the generated text.

### Strengths
- Clear motivation for iteratively refining partially unmasked sequences to correct for errors before transitioning to the next denoised state, which has backing from previous work [1]
- Well-defined and actionable knobs (refinement steps $k$, number of proposals $N$, and timestep set $U$) with ablation studies studying the effect of each of these parameters on downstream performance.
- The paper discusses some modifications to reduce the computational cost of IterRe,f which can help when using it with large models.
- Strong empirical results, especially on language tasks. The safety alignment experiment is interesting and quite promising.
- The ablations provide to some useful insights for diffusion language models: 1) increasing refinement steps seems to benefit more than increasing parallel particle count (which supports one of the motivations of this paper), and 2) later timesteps seem more important for guidance.

*[1] Wang, Guanghan, et al. "Remasking discrete diffusion models with inference-time scaling." arXiv preprint arXiv:2503.00307 (2025).*

### Weaknesses
My main concerns are with the significant changes between IterRef and MTM that require re-evaluating the theoretical result, some improvements to the experimental setup (fair calculation of NFEs, wall-clock time comparison, better metrics, using multiple seeds, and one important baseline), and clarifications on certain statements in the paper.

### Theoretical guarantee and design choices of MTM

The proposed algorithm differs from standard MTM in several important ways, and these choices are not reflected in the proof of Proposition 1.

MTM requires a specific structure in the choice of weighting function $w(x_t,x'_t)$ and the acceptance ratio $\beta$, which depend on the transition kernel $K(x_t,x'_t)$ and the balancing function $\lambda(x_t,x'_t)$. IterRef, however, (1) sets the selection weights to be uniform $1/N$ (while somewhat confusingly referring to it as “reward-weighted sampling” in line 243), (2) the balancing function uses the difference of rewards rather than the weighting function, and (3) does not draw backward proposals to reduce computational cost. It is not clear whether these changes still implement a chain that satisfies the same detailed-balance guarantee stated in Proposition 1, since the proof is for the *original* MTM setting. Please either (a) provide a derivation or a citation to an MTM variant that guarantees invariance with this modification, or (b) clearly label this as a practical heuristic distinct from the theoretical result.

### Experiments

- **Calculation of NFEs.** The authors make the somewhat atypical choice of counting both the diffusion model and reward model calls equally under the total NFEs. This is simple but can distort fairness when the main model is large (e.g., LLaDA-8B) and rewards are small classifiers (BERT-like models or GPT-2 scale). It would help to: (i) report model NFEs and reward calls separately, and (ii) include a wall-clock time analysis since the iterative refinement procedure of IterRef seems like it could be significantly slower than purely particle-based methods.
- **Choice of rewards/metrics.** The language tasks are based on proxy rewards, which are well-known to be poor metrics for language [2,3]. No human evaluation or LLM-as-judge is reported (which often agrees better with humans [4]). For *safety*, the detox curve is informative, but it would help to also report helpfulness retention (do safe outputs remain on-topic/answer the prompt?). For images, CLIPScore can be gamed; ImageReward [5] may be a complementary metric.
- **Variance across multiple seeds.** It seems all results are reported for a single seed. I *strongly* suggest the authors report results + variance for multiple seeds (and, where relevant, across prompts) to improve the reliability of the results.
- **Missing remasking baseline.** Because a key motivation is “fixing earlier mistakes by remasking,” it seems natural to include a *remasking-capable* discrete baseline (e.g., ReMDM [1]) to help isolate where IterRef’s gains come from.
- **Toxicity experiments.** The experiments include *increase* toxicity, which is a bit unusual from a safety perspective. I strongly suggest the authors include an ethics statement addressing the potential misuse of their method for such applications.


### Clarification on statements and positioning

- **Fair assessment of asymptotic results.** The paper notes that particle methods have asymptotic guarantees that may not hold under finite particle count $N$. That observation *equally applies* to IterRef’s Proposition 1, which is also asymptotic in the number of MTM iterations **$n \to \infty$.** Acknowledging this symmetry up front would avoid suggesting that IterRef has a stronger guarantee in finite compute.
- **Computational cost of SMC.** The paper states that SMC “requires weighted sampling … at every timestep … to maintain theoretical guarantees,” contrasting it with IterRef. Most SMC implementations [6,7] use adaptive or scheduled resampling (and the authors *themselves* use SMC baseline with resampling every 20 steps following [6]). The current phrasing seems to be inaccurate; I recommend removing/changing it to align with practice and your own setup.
- **Scope of prior work.** The abstract says test-time scaling for discrete diffusion is “largely unexplored”, while later sections cite a growing body of work (SVDD, FK/SMC, etc.). I’d suggest softening to “comparatively less explored”. The paper could also benefit from a discussion of some recent works on inference-time alignment of discrete diffusion models [1,7,8,9].
- **Contrast with SoP.** The high-level idea of SoP [10] is similar since it also performs a series of sequential noising + denoising steps at inference time. A discussion on the differences and the merits of IterRef over SoP will help strengthen claims of novelty.

### Minor points

- Figures and captions mix “NFE ×64,” “16T NFEs,” and simple integers (e.g., “2 NFEs”). A single convention (e.g., total NFEs across the full schedule, or NFEs per step) would prevent confusion. If “2 NFEs” really means $2 \times T$ calls, please say so explicitly.
- The paper overloads $k$ to refer to the noisy timestep in the kernel transitions, as well as the number of refinement iterations. Renaming one of them will improve clarity. Also, is the number of iterations $n$ referred to in Proposition 1 different from $k$?
- I was initially confused about how the reward is calculated on partially unmasked sequences $r(x_t)$. The appendix explains that the $x_0$ predictions are used to compute the reward. I suggest the authors explain this in the main paper.
- Section 3.1 introduces the kernel, the balancing function, acceptance ratio without explaining their meaning and role in MTM. A short introduction of the role of these functions in MTM would improve clarity.

*[1] Wang, Guanghan, et al. "Remasking discrete diffusion models with inference-time scaling." arXiv preprint arXiv:2503.00307 (2025).*

*[2] Holtzman, Ari, et al. "The Curious Case of Neural Text Degeneration." International Conference on Learning Representations, 2020.*

*[3] Hashimoto, Tatsunori B., Hugh Zhang, and Percy Liang. "Unifying Human and Statistical Evaluation for Natural Language Generation." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.*

*[4] Liu, Yang, et al. "G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment." Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023.*

*[5] Xu, Jiazheng, et al. "Imagereward: Learning and evaluating human preferences for text-to-image generation." Advances in Neural Information Processing Systems 36 (2023): 15903-15935.*

*[6] Singhal, Raghav, et al. "A General Framework for Inference-time Scaling and Steering of Diffusion Models." Forty-second International Conference on Machine Learning, 2025.*

*[7] Dang, Meihua, et al. "Inference-time scaling of diffusion language models with particle gibbs sampling." arXiv preprint arXiv:2507.08390 (2025).*

*[8] Jain, Vineet, et al. "Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models." The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.*

*[9] Li, Xiner, et al. "Dynamic Search for Inference-Time Alignment in Diffusion Models." arXiv preprint arXiv:2503.02039 (2025).*

*[10] Ma, Nanye, et al. "Inference-time scaling for diffusion models beyond scaling denoising steps." arXiv preprint arXiv:2501.09732 (2025).*

### Questions
- Unlike standard MTM, neither the weights nor the acceptance ratio appears to depend on the balancing function, so could the authors clarify what role $\lambda$ plays in IterRef as implemented?
- The proof of convergence hinges on a symmetric balancing function $\lambda$. The derivation includes a swap that is not valid for general $q$ and $p_\theta$. Could the authors restate the argument with explicit conditions on $q$ and $p_\theta$ under which the key swap is valid?
- Section 3.3 states that with an “appropriate” $\lambda$, acceptance can be computed *without* drawing backward proposals and the proposal pool can be reused upon rejection. The specific details are not provided - could the authors please explain how exactly this is achieved?
- The proof of Proposition 1 states that the transition kernel of IterRef is irreducible and aperiodic. As noted above, there seems to be a mismatch between the transition kernel used to provide theoretical guarantees and the actual design choices of IterRef. Could the authors elaborate on the irreducibility/aperiodicity of the transition kernel used for IterRef?
- Section 3.2 states that SMC can be used as the transition kernel for IterRef. I find this statement confusing since SMC is a framework to approximately sample from a sequence of target distributions, rather than a single-step kernel. Could the authors provide details on how SMC can be used as a transition kernel here?
- For the sake of clarity, could the authors state the per iteration NFE cost of IterRef? My understanding is that it should be $(k-t)$ diffusion model calls for each kernel call, times $N$ proposals, plus $N$ reward model calls for calculating the acceptance ratio.
- Most results in Sections 4.2-4.5 are obtained by generating 20 samples for each of the 15 prompts. Are the final reported numbers the mean score across the 20 samples, or are they “best-of” scores?
- The MDLM results on perplexity are very different between Figure 2 and Figure 4. Which setting of $k, N$ was used to obtain Figure 2?

### Soundness
2

### Presentation
1

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents Iterative Reward-Guided Refinement (IterRef), a test-time scaling method for discrete diffusion models. It uses a Multiple-Try Metropolis (MTM) framework with a noising-denoising transition kernel to iteratively refine misaligned intermediate states during generation, thereby aligning samples with a given reward function. The method provides a theoretical guarantee of convergence to the reward-aligned distribution and achieves promising empirical results across image and language tasks.

### Strengths
The overall idea is similar to the predictor-corrector framework. While this concept has been extensively explored in recent years, the use of the Multiple-Try Metropolis method as the corrector is novel, making the contribution distinctive. 

Moreover, the empirical results are promising, showing significant improvements over guidance-, SMC- and BoN-based methods, etc.

### Weaknesses
My main concern lies in the clarity and coherence of the paper’s narrative. The presentation of the proposed method and its connection to the underlying theory is difficult to follow. Key algorithmic details, such as the roles of the balancing function, importance weights, and acceptance rate, are either missing or insufficiently explained (see questions below), making it challenging to fully understand how the sampling procedure operates. As a result, while the empirical results appear promising, the exposition currently lacks the precision and transparency needed for readers to reproduce or rigorously evaluate the method.

### Questions
- The sampling algorithm is unclear:
    - What are the balancing function $\lambda$, importance weight and acceptance rate in equation 2? I can't find how they are used in Algorithms 1 and 2.
    - Why is the importance weight in Equation (2) equal to $N^{-1}$? Is $w_n$ the same as the weight of the multinomial distribution in Algorithm 1?
    - Is the acceptance rate $\beta$ the same as $\gamma$ in Algorithm 1?
    - In Step 3 of Algorithm 1, why is one sample $x’’$ drawn from $x_t$? Could you provide some intuition behind this choice?
- Both the target distribution and the transition kernel are intractable. How do you compute the importance weights and acceptance ratio in Algorithm 1?
- It seems that the proposed method could be directly applied to continuous diffusion models. Why do you focus only on the discrete cases? Are there particular challenges that make it applicable only to discrete settings?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4