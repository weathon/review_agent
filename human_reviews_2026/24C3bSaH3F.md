# Deep SPI: Safe Policy Improvement via World Models

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Safe policy improvement (SPI) offers theoretical control over policy updates, yet existing guarantees largely concern offline, tabular reinforcement learning (RL). We study SPI in general online settings, when combined with world model and representation learning. We develop a theoretical framework showing that restricting policy updates to a well-defined neighborhood of the current policy ensures monotonic improvement and convergence. This analysis links transition and reward prediction losses to representation quality, yielding online, ''deep'' analogues of classical SPI theorems from the offline RL literature. Building on these results, we introduce DeepSPI, a principled on-policy algorithm that couples local transition and reward losses with regularised policy updates. On the ALE-57 benchmark, DeepSPI matches or exceeds strong baselines, including PPO and DeepMDPs, while retaining theoretical guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents theoretical bounds on expected return degeneration (or values) as a consequence of representation learning composed with policy updates.
The authors theoretically show that making controlled (regularized) updates reduces out-of-trajectory misgeneralization during learning.
Secondly, they show that including the world model losses as part of the returns (aka an auxiliary reward/ objective) reduces the policy confounding problem.
Finally, an extension to PPO with auxiliary losses was implemented to validate the authors theory, which showed marginal performance increase on the ALE benchmark.

### Strengths
- Strong mathematical clarity, the background is compact yet comprehensive, and I could understand all nomenclature from this point on. The authors are precise without impeding reading flow. 

- Clear communication and examples of model learning issues and confounding in section 3.

- Overall excellent discussion of theory. Almost every theoretical result is followed by a simplifying explanation that discusses its impact and how it will be used later.

- Results are linked to the claims proposed in the introduction.

- Simple and intuitive modification of the PPO algorithm in Section 6 that is decently motivated from the preceding theory. 

- Humorous Spiderman references, although reducing the section-title fontsize in section 3 to fit on one line, is probably not allowed.

### Weaknesses
- I find the use of the sensitive word "safe" quite imprudent and unfitting. While we can endlessly debate the concept of safety, and the authors are probably not to blame (but prior work is), as far as I understand the paper is not dealing with cost constraints or risk aversion. Instead, "safe" refers to not "destroying" the world model due to policy updates during learning. Considering that there exist actual risk-sensitive RL applications (e.g., in finance or healthcare), I believe the authors should advertise their method in a different way. How about a word play on "conservative-world-model policy improvement" or "unconfounding regularization"? Which is more accurate in my opinion to the fix presented in Eq.4 \& Eq.5.

- Although section 3.2 presents a clear example, as far I understand this idea lies extremely close to well known prior work that should be cited. Specifically, the $Q^\pi$-irrelevance abstractions. See, Li L. et al. (2006). *Towards a Unified Theory of State Abstraction for MDPs.*

- The after-discussion of Theorem 2 should more strongly state the limitation that the result cannot give guarantees for large discount factors $\gamma \rightarrow 1$. This is ok, and aligns with intuition. The last 2 sentences of lines 301-308 already sort-of discusses this, but I think it should be even more explicit.

- The experimental section is weak and I'm not able to gather much from it, other than confirming that "the authors' method does not break PPO". Performance is not marginally improved or decreased, and varies from environment to environment. So, although the setup is decent, the problem is that we do not learn much from these experiments.
	- I am missing a more didactic experiment that compares the authors' method to a baseline method to validate that they "fix" the OOT and confounding policy update issues that they claim to solve. Instead the experiments go straight into performance and loss curves. Although useful to know, they are not that meaningful in the papers' context.
	- Furthermore, I miss a stronger connection from the theory to e.g., parameter choices for say $\epsilon$ (or $C$) that provides some approximate guarantee of modulating the permitted errors derived in the previous sections. Could a more TRPO-like algorithm be derived from the authors' theory, that explicitly tries to satisfy constrained updates? Although the authors decently build up to the PPO extension, I find the auxiliary loss regularization to read somewhat like an afterthought that undermines the interesting preceding results.
	- The authors claim in paragraph 448-458 that their transition loss is improved, but looking at figure 5 this is not statistically significant compared to the DeepMDP baseline.

- Conclusion line 479-481 overstates the result that DeepSPI improves upon PPO, the performance benefit is quite marginal and more noisy.

 *Minor comments:*
- line 148, the "baseline policy" is known in RL as the behavior policy. I don't see a reason to ignore convention here.
- Appendix page 30, figure is too large, the label overlaps with the page number.
- Line 286-288, can you put this paragraph into an "assumption" environment and clearly state that you're working with this "from this point on". This detail is now buried within the surrounding text, and it's not clear whether the assumption belongs to the later results, or to all results.


*My overall verdict:* 
This is an extremely clearly written document with theoretical results that relatively nicely align with intuition. 
The authors carefully guide the reader through the theory and explain the impact of their result without obfuscating their result with unnecessary detail.
That said, I am not a pure theoretical person in the RL field, so I cannot accurately judge the impact or novelty of the proposed contributions over prior work. 
My main problem with the paper at the moment is the weak experiment section, I want to see a setting where the authors can validate their theoretical result, even if it is simple. I think the PPO extension does not do the rest of the paper justice.
Overall, I would be leaning towards acceptance if most of my points can be addressed and questions answered.

### Questions
1. I don't understand part of the claim on the support for the policy sequence $(\pi_n)_{n\ge 0}$. Why does $\pi_n, n \rightarrow \infty$ guarantee full support?Theorem 1 claims that the sequence converges to an optimal value, and I can understand this yields a (stochastically) optimal policy. However in lines 234-239 it is stated that since $\pi_0$ has full support, all $\pi_n$ have full support as well, which again guarantees full support over the full state space. 
But if $\pi_n \rightarrow \pi^* $ as $n \rightarrow \infty$, then $ \pi^* $ should not have full support. The optimal policy should put zero density/ probability on suboptimal actions. If it doesn't, then $\pi_n$ should not converge such that $V^{\pi_n} = V^*$.
   
2. Could you shortly reflect on the theory and practical impact of your answer to my Q1? How would you revise your claims/ theorem?

3. Can you add equation numbering to the reward-transition loss of lines 257-259, and then refer to this Eq. again when discussing Theorem 2. It took me a moment to see that $L_R^\xi, L_P^\xi$ referred to these losses over the state-occupancy instead of the buffer $\mathcal{B}$.

4. Could the authors think of (and run) a didactic experiment that validates the newly proposed algorithm. This could perhaps be built on the toy MDPs from figure 1 or figure 2, at the moment, it is not clear experimentally whether the authors proposed method

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DeepSPI, a safe policy improvement algorithm that combines mirror learning updates with a learned world model. The work provides theoretical results that jointly consider the impacts of policy updates and world model updates, which motivates an extension of PPO that incorporates a world model for representation learning. Experiments across the Atari benchmark demonstrate improved performance compared to PPO and DeepMDP.

### Strengths
**[S1]**: The paper is clear, well-written, and easy to follow.

**[S2]**: The paper introduces interesting theoretical analysis that combines mirror learning policy updates with world model representation learning. This analysis extends results from DeepMDP [1] to consider the interplay between policy updates and representation learning, and results in bounds with less restrictive assumptions. 

**References:**

[1] Gelada et al., “DeepMDP: Learning Continuous Latent Space Models for Representation Learning.” In ICML 2019.

### Weaknesses
**[W1]**: Experimental results do not demonstrate convincing practical benefits of the proposed approach. The experiments show modest improvements for DeepSPI compared to PPO and DeepMDP, and the performance of DeepSPI is significantly worse than other results that have been reported in the literature on the Atari benchmark. DreamerV3 [2], for example, is also an actor-critic algorithm that leverages a learned world model, and reports better results after only 100k steps (Atari100k results in [2]) compared to the performance of DeepSPI after 10M steps.

**[W2]**: It seems like the learned world model is under-utilized in DeepSPI, only being used for representation learning but not for model-based rollouts as in the Dreamer series. The attempt to consider a model-based implementation using imaginary rollouts (DreamSPI) does not perform well, which the authors suggest may be due to its on-policy nature. It is not clear if the performance guarantees can be extended to account for model-based training with imaginary rollouts.

**References:**

[2] Hafner et al., “Mastering Diverse Domains through World Models.” arXiv, 2023. arXiv:2301.04104.

### Questions
**[Q1]**: Are the theoretical results compatible with model-based RL using imaginary rollouts as in the Dreamer series? The experiments suggest that the use of an on-policy approach without imaginary rollouts cannot achieve performance close to state-of-the-art algorithms like DreamerV3, which limits the practical impact of the work.

**[Q2]**: Are the theoretical results compatible with off-policy RL methods? 

**[Q3]**: How does the approach relate to conservative model-based methods in offline RL (e.g., [3])?

**[Q4]**: Please provide additional implementation details, such as network architectures and the computation time of DeepSPI relative to PPO.


**References:**

[3] Kidambi et al., “MOReL: Model-Based Offline Reinforcement Learning.” In NeurIPS 2020.

### Soundness
3

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
This paper provides a strong theoretical contribution that generalizes safe policy improvement to deep, on-policy RL with learned world models. The integration of SPI principles, Wasserstein-based transition losses, and Lipschitz-continuous latent dynamics is mathematically interesting. The theoretical development is rigorous, connecting mirror descent formulations (Kuba et al., 2022) to practical PPO-style updates, and the proofs (Appendices B–E) seems to be technically comprehensive (although I did not check every detial). The empirical section, though limited to ALE benchmarks, is well-executed and demonstrates the feasibility of theory-grounded improvements.

However, some aspects could be clarified or strengthened: (i) practical relevance of the γ < 1/ C assumption, (ii) the sensitivity of DeepSPI to neighborhood size and Lipschitz constants, (iii) scalability to continuous control tasks, and (iv) the limited ablation on auxiliary losses.

### Strengths
1. The work extends classical SPI (Thomas et al., 2015; Laroche et al., 2019) to high-dimensional, online RL. The neighborhood operator $\mathcal{N}^{C}(\pi)$ (Eq. 2) defines a trust region via importance ratios, bridging SPI and policy regularization in a provably convergent way (Thm. 1).

2. The paper formalizes how transition/reward prediction losses ($L_{P}$, $L_{R}$) act as local regularizers ensuring Lipschitz-consistent latent dynamics (Thm. 2–4). This explicitly ties representation smoothness to policy safety, a novel link not addressed by DeepMDPs or Dreamer.

3. The identification of PPO as a special case of the proposed mirror-learning-based neighborhood update (Eq. 4 vs Eq. 3) is conceptually valuable. DeepSPI’s modified utility (Eq. 5) yields a principled way to blend safety and empirical efficiency.

4. On the ALE-57 suite, DeepSPI consistently outperforms PPO in 43 of 61 environments (Fig. 3), while reducing transition loss and ensuring smoother learning (Fig. 5). The implementation is reproducible and uses standard open frameworks (CleanRL, EnvPool).

### Weaknesses
1. Evaluations are restricted to Atari; no tests on continuous-control or partially observable tasks (e.g., DMControl, Procgen). It remains unclear whether Lipschitz constraints and local losses scale to these settings.

2. Although DeepMDP losses are included, the study omits other model-based SPI or regularized RL baselines (e.g., DreamerV3, SAC + KL). Statistical significance and variance across seeds are not discussed (e.g., arxiv.org/abs/1904.06979).

3. While claimed “on-policy,” DeepSPI introduces additional transition modeling networks. The paper lacks runtime, sample-efficiency, or wall-clock comparisons versus PPO, given the relatively limited improvement versus the increased algorithmic complexity.

4. This paper does noticeably lack a dedicated “proof roadmap” or overview section, which would guide the reader through the logical structure and dependencies of its multiple theorems (e.g., Theorems 1–5).

### Questions
The choice of neighborhood constant C and coefficients αₚ, αᵣ (Eq. 5) critically affects guarantees and performance, but no ablation or stability analysis is reported.

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
The paper presents a method for safe policy improvement (SPI) for an online learning setting in combination with world models and representation learning. The authors offer proof that restricting the update of the policy gives guarantees of convergence. They also introduce auxiliary loss functions on the latent space ensure its stability (i.e. prevents the collapse of states in the representation). Afterwards experiments on the Atari benchmark show the performance improvement over PPO/DeepMDPs. The authors also present an investigation into the quality of the resulting world model.

### Strengths
- The problem addressed in the paper for online on-policy safe improvements is very relevant.
- The proofs and formulations presented in the paper appear correct/rigorous and the analysis done so far forms also a good basis for future works.
- The evaluations are well done over the full Atari suite, and include meaningful benchmarks, albeit all of them designed by the authors themselves.

### Weaknesses
- A lot of the background in section 2 lacks references for some of the very general statements the authors make about the field. E.g., L125-126 or L132-134. Similarly other mentions of "in literature" like L263 don't have references.
- Numeric results seem to only show minor improvement over PPO (however this is not directly comparable since the authors' approach provides SPI guarantees).

### Questions
- In section 4 when bounding the trust region, the choice of setting 1 < C < 2 seems to be arbitrary, why specifically bound it up to 2?
- What happens in the evaluations when changing the stochasticity of the system, i.e., how do DeepSPI compare to PPO when p_a and n_{NOOP} are closer to zero, and similarly when they are even higher? This would further motivate the presented results as it shows that the system is not susceptible to small parameter changes.

### Soundness
4

### Presentation
3

### Contribution
3
