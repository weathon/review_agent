# TIC-GRPO: Provable and Efficient Optimization for Reinforcement Learning from Human Feedback

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Group Relative Policy Optimization (GRPO), recently introduced by DeepSeek, is a critic-free reinforcement learning algorithm for fine-tuning large language models. GRPO replaces the value function in Proximal Policy Optimization (PPO) with group-normalized rewards while retaining PPO-style token-level importance sampling based on an old policy. We show that the GRPO update rule actually estimates the policy gradient at the old policy rather than the current one; however, because the old policy is refreshed every few steps, the gap remains small and the resulting bias is negligible in practice. To validate this, we perform an ablation study that removes importance sampling entirely and instead applies gradients estimated at a fixed old policy across multiple optimization steps. Remarkably, this simplified approach achieves performance comparable to standard GRPO.

Motivated by these findings, we propose a new algorithm: Trajectory level Importance Corrected GRPO (TIC-GRPO). TIC-GRPO replaces token level importance ratios with a single trajectory level probability ratio, yielding an unbiased estimate of the current policy gradient while preserving the critic free structure. Furthermore, we present the first theoretical convergence analysis for GRPO style methods, covering both the original GRPO and our proposed variant.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes TIC-GRPO， which replaces GRPO’s token-level importance sampling with a trajectory-level ratio and integrates two additional tricks: Length-Corrected Group Normalization and Upper-Only Clipping. The paper analyzes the convergence of GRPO and TIC-GRPO. Experiment using Qwen 3 on math reasoning tasks demonstrates the effectiveness of TIC-GRPO.

### Strengths
1. The GRPO gradient decomposition is clear.

2. The proposed method is simple and easy to implement with the existing GRPO code.

### Weaknesses
1. The abstract and introduction highlight that the TIC-GRPO estimator is unbiased, but the derivation of Appendix B shows that the TIC-GRPO estimator is not strictly unbiased. This is a material mismatch between the headline claim and the actual derivation.

2. Assumption 5.1 requires global Lipschitz continuity of the score function for all states. This is a strong assumption for LLMs since there can be low-probability regions where logP can vary sharply.

3. Theorem 5.2’s improved bound explicitly comes only from the length-corrected normalization and upper-only clipping, not from the trajectory-level importance ratio itself. This creates a disconnect with the central framing around trajectory-level sampling.

4. The evaluation scope is too narrow. The evaluation uses only AIME 2024 as the benchmark. The author should include more benchmarks like MATH500, AIME 2025, and OlympiadBench.

### Questions
Please see Weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a theoretically grounded and empirically enhanced variant of the Group Relative Policy Optimization (GRPO) algorithm for reinforcement learning from human feedback (RLHF) in large language models (LLMs). This work makes a substantial contribution by deepening the theoretical understanding of GRPO. And the authors provide the first convergence analysis for GRPO-style methods and introduce a simple yet powerful variant, TIC-GRPO. The combination of novel theory and a practical algorithm makes this a valuable piece of research for the RLHF community, with potential for influencing future theoretical RLHF work.

### Strengths
This work provides the first rigorous convergence analysis for GRPO-style methods, a popular class of critic-free RLHF algorithms. By establishing formal convergence guarantees under standard assumptions, the paper fills a critical theoretical gap in the literature. The convergence analysis is built on a solid foundation of standard and reasonable assumptions.

The paper delivers a crucial and insightful finding and elegantly explains why GRPO works in practice despite the bias. This theoretical clarification of the core mechanism is a significant step forward in understanding RLHF dynamics.

The paper is written with remarkable clarity. Furthermore, the authors demonstrate academic integrity by honestly attributing the tighter convergence bound of TIC-GRPO solely to the two minor modifications, not to the trajectory-level sampling. This conservative and transparent assessment builds trust and accurately scopes their theoretical contribution.

Beyond the specific algorithm proposed, the paper's greatest impact lies in its theoretical rigor. In a field often dominated by empirical results, providing a principled theoretical framework for GRPO is an invaluable service to the community.

### Weaknesses
**Narrow and Potentially Insufficient Empirical Validation**: 
Conducting experiments on only one benchmark (AIME) is highly unusual and insufficient to establish generalizability. A review of other GRPO-related papers (e.g., DeepSeekMath, GSPO) shows they typically use multiple benchmarks. The failure to include, for example, AIME-25, significantly weakens the persuasiveness of the empirical claims.


**Lack of Experiments Directly Supporting Theoretical Claims**: A major contribution is the convergence analysis, yet there are no experiments in the main text that visually demonstrate or validate the improved convergence rate or stability. Including such plots would significantly strengthen the link between theory and practice.

### Questions
I have manually reproduced the derivation of Eq. (7) in your Section 3. In my result, the terms \Xi_g(\theta, \theta_{\text{old}}) and \Xi_c(\theta, \theta_{\text{old}}) do not have the multiplier \frac{1}{\theta_{\text{old}}}. If space permits, could you please provide an explanation for this part?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents TIC-GRPO (Trajectory-level Importance-Corrected Group Relative Policy Optimization), a theoretical and algorithmic refinement of GRPO (Group Relative Policy Optimization), recently introduced by DeepSeek for critic-free RLHF fine-tuning.

The authors first identify that standard GRPO estimates gradients at the old policy rather than the current one, explaining why this bias remains small in practice due to frequent policy refresh. Then, they propose TIC-GRPO, which corrects this by replacing token-level importance weights with trajectory-level probability ratios, yielding an unbiased estimator of the true policy gradient.

The paper provides the first convergence analysis for GRPO-style methods, proving stationarity bounds under Lipschitz and bounded-reward assumptions. Empirical evaluation on AIME and DAPO-17K datasets with Qwen-1.7B and Qwen-8B models shows that TIC-GRPO improves accuracy and convergence speed over GRPO, GSPO, and DAPO baselines.

### Strengths
- Clear theoretical motivation and correction. The decomposition in Eq. 7 demonstrates that GRPO’s update estimates ∇J at π_old rather than π, and TIC-GRPO’s trajectory-level ratio restores unbiasedness. The analysis bridges empirical intuition with formal theory.
- Provable convergence guarantees. Theorems 5.1–5.2 give the first formal stationary-point convergence bounds for GRPO-style methods, showing improved asymptotic dependence after removing terms M_N and σ²_sT,N.
- Simple yet effective modifications. The two “minor” refinements (length correction and upper-only clipping) are shown to individually improve stability, enhancing both fairness and interpretability.

### Weaknesses
- Limited originality relative to concurrent work. The key modification—trajectory-level ratios—is nearly identical to GSPO (Zheng et al., 2025), which the authors acknowledge. While TIC-GRPO adds theoretical analysis and slightly different normalization, the conceptual leap is incremental.
- Experiments are limited in scope. The evaluation focuses on AIME reasoning benchmarks, which are small-scale and synthetic. It’s unclear whether TIC-GRPO generalizes to more diverse RLHF settings (e.g., preference data, summarization, or open-ended dialogue).
- Incremental empirical improvement. Although TIC-GRPO outperforms GRPO by +2–3 points, the margins are modest given additional computation and algorithmic tuning. There is no runtime or stability comparison (e.g., variance, gradient norms, or wall-clock efficiency).
- Ablation isolation could be clearer. The claim that “theoretical improvement stems from two refinements only” (Sec. 5.2) implies trajectory-level importance may not improve the asymptotic rate—suggesting the empirical gains come mainly from the minor tweaks rather than the main theoretical contribution.

### Questions
1. On gradient correctness: How do you empirically verify that TIC-GRPO’s gradient estimator better aligns with the true ∇J(θ)? Can you show cosine similarity between estimated and true gradients (or Monte Carlo rollouts) across updates?
2. On contribution beyond GSPO: Could you clarify what new insights TIC-GRPO adds beyond GSPO besides clipping and convergence proof? Are these differences substantive enough to claim novelty?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper targets the high resource cost of PPO in RLHF caused by its extra value network. The authors first point out that GRPO’s token-level importance sampling actually estimates the policy gradient at the old policy π_old rather than at the current policy. Building on this observation, they propose Trajectory-level Importance-Corrected GRPO (TIC-GRPO). Theoretically, they provide the first convergence rate analysis for GRPO-style algorithms. Experiments on the AIME mathematical-reasoning benchmark show that TIC-GRPO significantly outperforms the original GRPO on both 1.7 B and 8 B models.

### Strengths
- Originality: The work is the first to reveal that GRPO essentially performs gradient estimation at the old policy, and it uses ablation studies to validate this insight, laying an intuitive foundation for further improvements.
- Clarity: Concepts, formulas, and proofs are well presented, and the appendices are comprehensive; however, the meaning of some symbols is not explained.

### Weaknesses
1. In Eq. (7), the subsequent proofs bound some error terms by problem-dependent constants, whereas other bounds are independent of hyper-parameters. Yet in RL the policy changes little between two consecutive steps. What, then, is the justification for decomposing the expression into so many terms in Eq. (7)?
2. The upper bound in the theorem does not contain the hyper-parameters $\epsilon_{high}$ and $\epsilon_{low}$. Does this mean their values do not affect the bound? If so, can the bound be further improved?
3. Main results are reported only on the single mathematical-reasoning task AIME; there is no verification on diverse tasks such as dialogue, code generation, or creative writing. It is therefore unclear whether the gains are task-specific. Experiments at larger scales are also needed.
4. The variance of the trajectory-level importance ratio usually grows exponentially with length. How do the authors handle this issue?
5. The authors claim their method has better sample efficiency. What is the intuitive explanation?
6. There is no dedicated “Conclusion” section.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
