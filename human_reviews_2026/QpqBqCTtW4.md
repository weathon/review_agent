# Unifying Stable Optimization and Reference Regularization in RLHF

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Reinforcement Learning from Human Feedback (RLHF) has advanced alignment capabilities significantly but remains hindered by two core challenges: reward hacking and stable optimization. Current solutions independently address these issues through separate regularization strategies, specifically a KL-divergence penalty against a supervised fine-tuned model ($\pi_0$) to mitigate reward hacking, and policy ratio clipping towards the current policy ($\pi_t$) to promote stable alignment. However, the implicit trade-off arising from simultaneously regularizing towards both $\pi_0$ and $\pi_t$ remains under-explored. In this paper, we introduce a unified regularization approach that explicitly balances the objectives of preventing reward hacking and maintaining stable policy updates. Our simple yet principled alignment objective yields a weighted supervised fine-tuning loss with a superior trade-off, which demonstrably improves both alignment results and implementation complexity. Extensive experiments across diverse benchmarks validate that our method consistently outperforms RLHF and online preference learning methods, achieving enhanced alignment performance and stability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes a unified regularization approach with weights that explicitly balances the objectives of preventing reward hacking and maintaining stable policy updates in RLHF.

### Strengths
The authors found a reasonable limitation of the existing approach to solve reward hacking and maintain stable policy updates, i.e., the conflict of two regularizers that pull the trained policy towards the reference policy and previous-step policy respectively. This well motivates the proposed dual-KL approaches, with the novel and straightfoward idea of combination of the two regularizers, which are presented clearly. The experiments look comprehensive to me.

### Weaknesses
A few clarity issues as shown in the following questions.

### Questions
(1) Is Eq. (2) PPO for RL not RLHF, since there is no KL penalty to $\pi _ {\rm ref}$? The equation in Section 4.1 seems to transform the KL penalty in Eq. (1) into hard constraint $KL<\epsilon$, yes? 

(2) "Empirical Validation" in Section 4.1 involves two dual-KL variants. Why not move "Empirical Validation" to after introducing two dual-KL variants?

(3) Your dual-KL (Eq. 3) looks like a special case of [1] with 2 references, what are your differences and additional contributions?  
[1] Gholamali Aminian, Amir R Asadi, Idan Shenfeld, and Youssef Mroueh. Theoretical analysis of kl-regularized rlhf with multiple reference models. ArXiv:2502.01203, 2025.

(4) In proposition 4.1, $\log\pi_{\theta}(y|x)=\alpha\log\pi_0(y|x)+(1-\alpha)\log\pi_t(y|x)+C(x)$ with a constant normalization $C(x)$ is suggested to ensure policies summing up to 1. 

(5) What are the evaluation metrics of MT Bench in Table 3? 

(6) In Figure 5c, could you explain more about EOS-missing rate, and why $\alpha=0$ increases EOS-missing rate? Is it convenient to add LC-win rate? 

(7) In Algorithm 1, what are $\mu_A$ and $\sigma_A$? What are the meanings of the two weights $w _ {\rm reg}^i$ and $w _ {\rm adv}^i$ and how are they computed?

### Soundness
3

### Presentation
3

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
The paper proposes a *dual-KL regularization* approach that aims to jointly address two RLHF pain points: (i) preventing reward hacking via reference regularization and (ii) ensuring stable optimization via trust-region style control. The method first studies a constrained “PPO-Align” objective and then derives a *weighted dual-KL* objective that yields an interpretable, weighted SFT (DAR) loss by effectively interpolating between the initialization policy $\pi_{0}$ and the current policy $\pi_{t}$ in log space. Experiments on TL;DR, Anthropic Helpfulness, and Harmlessness report improved win rates (Figure 3/Table 2), with evaluations judged by GPT-4 Turbo. Figure 1 provides the conceptual motivation for expanding the search region beyond the intersection of the trust regions to balance stability and reference adherence.

### Strengths
- **Timely problem framing.** The paper clearly motivates the dual goals of stabilizing policy updates while constraining drift from a reference policy, and argues for a unified objective rather than separate mechanisms.
- **Empirical gains.** Across three benchmarks, DAR shows strong win rates against online RLHF (e.g., PPO, GRPO, RLOO) and online DAP baselines; curves in Figure 3 and the summary in Table 2 support the claim.
- **Implementation clarity intent.** The paper points to code/supplement details, which is important given the number of components interacting (policy, judge, datasets, ablations).

### Weaknesses
- In the discussion of PPO stability, the classical TRPO/PPO literature typically regularizes with a KL of the form $D_{\mathrm{KL}}\left[\pi_{\text{old}}\||\pi_{\theta}\right]$, see Schulman et al. (2015, TRPO) and Schulman et al. (2017, PPO). By contrast, the paper’s *PPO-Align* (Sec. 4.1) constrains with $D_{\mathrm{KL}}\left[\pi_{\theta}\||\pi_{t}\right]$ and penalizes $D_{\mathrm{KL}}\left[\pi_{\theta} \||\pi_{0}\right]$. The rationale for changing directions relative to the classical trust-region view is not made explicit. This makes it hard to judge whether the final dual-KL choice is a principled departure or just a convenient variant.
- Figure 3/Table 2 use GPT-4 Turbo as the judge and Qwen2-72B-Instruct as the LLM annotator. Results would be stronger with a stronger contemporaneous judge (e.g., GPT-5) and annotator (Qwen3) for additional validation.
- While DAR (weighted SFT) is argued to be stable, quantitative stability analyses for the *dual-PPO* path are sparse (e.g., per-iteration KL to $\pi_{0}$ and $\pi_{t}$, gradient-norm/entropy trends, collapse rates across seeds).
- Most results focus on Qwen2 and Llama-3.1 settings; adding a recent backbone (e.g., Qwen3) would better probe portability.
- *(Minor)* Colors/encodings are hard to parse quickly; the caption should explicitly map colors/shapes to policies/regions and call out what “search expansion” specifically denotes.

### Questions
- Given the centrality of Table 2, the community would benefit from end-to-end scripts (prompts, judge configs, seeds, filtering) in the supplement to exactly reproduce those numbers. Can we provide that in supplement?
- Would a *mixed-direction* dual-KL be preferable on theory/empirics—e.g., a mode-covering  term $D_{\mathrm{KL}}\left[\pi_{\theta}\||\pi_{0}\right]$ *and* a mode-seeking term around the behavior/current policy $D_{\mathrm{KL}}\left[\pi_{t}\||\pi_{\theta}\right]$?
- Could you report training stability metrics such as gradient-norm statistics, entropy, and seed-wise variance for dual-PPO?
- How do conclusions change with newer backbones (e.g., Qwen3) or different reward models?

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
2

### Summary
This paper proposes a dual-KL regularization framework for RLHF that unifies two objectives usually treated separately: (1) preventing reward hacking via KL to the initial SFT model π₀, and (2) maintaining stability via KL or clipping to the current policy πₜ.
The authors show that these can be merged into a single interpolated reference in log-space, leading to a new weighted-SFT formulation called DAR (Dual-regularized Advantage Regression). DAR is positioned as a simple, RL-free alternative to PPO/GRPO, with theoretical analysis and experiments on Qwen2-7B showing improved reward–KL trade-offs.

### Strengths
1. Clear identification of a long-standing conflict between stability and reference regularization.
2. Mathematical formulation is elegant and internally consistent.
3. DAR simplifies PPO-style RLHF into a regression-like loss that is easier to implement and more stable.

### Weaknesses
1. Outdated baseline setup. All experiments use Qwen2-7B and compare mainly against PPO, GRPO, and RLOO; no comparison to modern alignment frameworks, stronger models, and new RL methods.
2. The novelty is mostly formal: the “dual-KL” is effectively a convex interpolation between π₀ and πₜ, similar to prior multi-reference ideas.
3. Theoretical results rely on clean advantage estimation; no analysis under noisy or biased rewards.
4. Empirical gains are modest and might vanish under stronger baselines.

### Questions
1.How sensitive is performance to α? Could an adaptive trade-off help?
2.Would DAR remain stable under noisy or AI-feedback reward models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an approach to address the trade-offs arising from simultaneously regularizing towards the reference policy (to mitigate reward hacking) and the current policy (for stable policy updates), in RLHF. They accomplish this by regularizing towards a convex combination of the reference policy and current policy. This is done via a weighted supervised fine-tuning loss, which allows for stable training. Experimentally, their proposed approach improves over baselines (both online RL-based approaches and online/offline direct alignment algorithms)

### Strengths
The paper addresses a critical problem that has not been explored in the literature, the impact of regularizing towards both the reference policy and the current policy, in RLHF. Regularizing towards the reference policy is done by a KL penalty to the reward, and regulazing towards the current policy is achieved via clipping or a KL constraint. Together, these two constraint our objective to operate in the intersection of the trust region that becomes increasingly restrictive as training progresses. The paper proposes a simple weighted supervised fine-tuning objective by regularizing towards a convex combination of both the reference policy and the current policy, leading to a stable training algorithm DAR (Dual-Regularized Advantage Regression). Experimental results showcase that DAR surpasses online RL based methods (PPO, GRPO, RLOO) and online/offline direct alignment methods (DPO/IPO/SLiC) across three training domains. The presentation is clear, crisp with no major issues with the grammar and writing flow.

### Weaknesses
One of my concerns with the paper is their choice to regularize towards a convex combination of the reference policy $\pi_{0}$ and the current policy $\pi_{t}$ i.e $\alpha D_{KL}(\pi \vert\vert \pi_{0}) + (1-\alpha) D_{KL}(\pi \vert\vert \pi_{t})$. This inherently leads to incentivizing regularizing to one of the distibutions than the other (when $\alpha$ != 0.5). It would have been better to have two independent multipliers for each of the divergence, which supports the Lagrangian view of the objective when looking at the divergences as constraints i.e $\alpha_{1} D_{KL}(\pi \vert\vert \pi_{0}) + \alpha_{2} D_{KL}(\pi \vert\vert \pi_{t})$.

Additionally, in proposition 4.1, the objective has a KL penalty wrt a reference mixture distribution $\pi_{ref} = \pi_{0}^{\alpha}\pi_{t}^{1-\alpha}$. $\pi_{ref}$ need not be a valid probability distribution, since it may not sum up to 1. The KL term here hence may not be between two distributions. There needs to be a normalizing factor for $\pi_{ref}$ to make it a valid probability distribution. This would affect their proof of the optimal policy in Theorem 4.2, since there would now be normalizing factors of $\pi_{ref}$ in the expressions.

In the DAR derivation in Appendix C.3, ignoring the earlier issue with $\pi_{ref}$ not being normalized, on line 885, they state "we factor out the partition function $Z(x)$ as it is a positive constant and doesnt shift optimal policy". $Z(x)$ depends on $x$ and leads to weighing each prompt differently. Considering a simple setting with two possible $x$, the objective is $Z(x_{1}) f_{\theta}(x_{1}) + Z(x_{2}) f_{\theta}(x_{2})$.  How can $Z(x)$ be factored out without affecting the objective 

The authors state that as $\pi_{t}$ is trained, the log-likelihood interpolation constructs a reference target that is inherently positioned closer to the optimal policy. No proof for this is provided. If this is solely because $\pi_{ref}$ contains $\pi_{t}^{1-\alpha}$, that becomes more optimal over the course of training. If so, standard PPO also has a policy constraint with respect to $\pi_{t}$. How is yours more optimal?

### Questions
1) Is there a reason for choosing a convex combination of the two KL divergences for DAR, instead of choosing independent multipliers?

2) Why is the $\pi_{ref}$ not normalized in the KL constraint for DAR and does this impact the derivation of the optimal policy for DAR?

3) Why can $Z(x)$ be dropped from the objective, without modifying it, in the DAR derivation in Appendix C.3?

4) Is there a theoretical justification for this statement "as $\pi_{t}$ is trained, the log-likelihood interpolation constructs a reference target that is inherently positioned closer to the optimal policy." in line 240?


Minor Questions/Nits

- Equation 2 operates at trajectory level, whereas PPO is defined at token-level
- What the reward (avg, total, etc) in Table 1. Explain it in detail in caption.
- Why no comaprision against PPO in the the Experiments for Standard RLHF (line 317)?
- In Figure 3, why no evolution plots for online direct alignment methods?
- Improvements on MT-Bench and AlpacaEval 2.0 in Table 3 seem marginal. Also the column is named "AlphacaEval".

### Soundness
2

### Presentation
3

### Contribution
3
