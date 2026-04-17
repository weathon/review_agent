# Risk-Sensitive Reinforcement Learning for Alleviating Exploration Dilemmas in Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enhancing Large Language Models (LLMs) on complex reasoning tasks. Yet current methods face an exploration dilemma: standard RL struggles to escape the local optima of pre-trained LLMs’ sharply peaked initial policies, boosting single-solution accuracy (pass@1) but suppressing solution diversity and multi-solution performance (pass@k). As a result, RLVR often distills existing capabilities rather than discovering new reasoning strategies.  We address this with a Risk-Sensitive Reinforcement Learning framework. By adopting a risk-seeking objective that interpolates between mean and maximum rewards, we derive a novel Risk-Sensitive GRPO (RS-GRPO) algorithm that emphasizes hard prompts to drive exploration. Across six mathematical reasoning benchmarks and five LLMs, RS-GRPO consistently improves pass@k performance while enhancing or maintaing pass@1.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a risk‑sensitive RL objective for LLM fine-tuning with verifiable rewards. Concretely, the authors replace the mean-reward objective with the exponential-utility formulation that interpolates between mean and max reward, yielding a simple "drop‑in" advantage estimator for GRPO. The paper argues this encourages exploration from sharply peaked pretrained policies and improves pass@k while preserving pass@1. Empirically, the method is evaluated on six math benchmarks and several LLMs, reporting consistent pass@k gains and mostly non‑degrading pass@1. A small bandit study and lemmas provide intuition/theory.

### Strengths
1. Clear, readable paper with helpful figures. The exposition around the exploration dilemma and how the exponential utility shifts gradient mass toward hard prompts is crisp. The overview figure and the derivation culminating in Equaqtion 7 are easy to follow.

2. RS‑GRPO only alters the advantage computation, making it nearly "plug‑and‑play" for GRPO pipelines. This is appealing for practitioners.

3. The study covers multiple LLMs and benchmarks; pass@k curves across k=1..1024 are shown in Fig. 4, and $\beta$‑ablations (Fig. 5) help quantify the trade‑off between exploration and convergence.

4. Lemmas in section3.2 show that, in a bandit, risk-neutral PG can reduce probability on the optimal arm (Lemma 2), whereas risk‑sensitive PG increases it for sufficiently large $\beta$ (Lemma 3), with diminishing gains if $\beta$ is too large (Lemma 4). The toy bandit (Fig. 3) visually supports this.

### Weaknesses
1. Limited novelty. The exponential-utility objective and risk‑sensitive policy gradients are classical (e.g., Howard & Matheson; Mihatsch & Neuneier), and the paper essentially instantiates that objective as a reweighted advantage for GRPO (Eq. 7-8). The algorithmic delta is small and conceptually straightforward---primarily a different per‑sample weighting. Theoretical results are restricted to a bandit setting; there is no analysis in sequence‑generation or RLVR settings beyond intuition. This makes the contribution marginal.

   - Table 2 (p.8) shows RS‑GRPO usually adds ~+3–5 points in pass@32 over GRPO (sometimes less), while pass@1 is often flat and occasionally lower. For instance, on Llama‑3.1‑8B‑Instruct AIME24, pass@1 drops from 9.9 (GRPO) to 9.4 (RS‑GRPO); on AIME25 it drops from 1.2 to 0.6. The Qwen2.5‑Math‑1.5B MATH500 pass@1 also decreases slightly (80.2$\rightarrow$78.1). These are small but non‑negligible regressions.

2. The comparison to pass@k optimization baselines uses k=4 for those methods and $\beta$=2 for RS‑GRPO. Although the paper argues this equalizes the advantage magnitude (Fig. 8), the mapping between k and $\beta$ is not unique. Baselines might benefit from a small hyperparameter sweep; it is unclear if they were tuned comparably to RS‑GRPO.

3. There are some design choices that look unnatural to me.

   - No KL regularization is used during RL (Table 3). This can lead to policy drift; evaluating safety, verbosity, or formatting regressions would strengthen the case that risk‑seeking exploration is benign beyond accuracy.

   - Risk‑seeking emphasizes tail rewards and may amplify false positives from the verifier. There is no experiment where the reward is intentionally perturbed or partially noisy to test robustness.

   - Many benchmarks use N=1024 samples per problem at evaluation. It would help to quantify gains at small k (e.g., k in {4,16}) under fixed compute budgets and to report training FLOPs/tokens; the training curves (Fig. 5-6) stop at a few hundred steps but do not translate to compute cost.

4. There is a gap between the theory statements and experiments. The claims are theoretically justified only for bandits and do not consider sequence-level credit assignment, KL penalties, or importance sampling present in practical RLVR. Extending the analysis (even approximately) to token-level updates or to a simple MDP with stochastic rewards would increase significance.

### Questions
1. Did you sweep k for the pass@k baselines and report the best? Why is k=4 the sole setting (Appendix D)? Could stronger tuning reverse the small pass@1 gap you observe in Fig. 6?

2. What are the training FLOPs / tokens and wall‑clock for RS‑GRPO vs GRPO? How do gains scale with fewer rollouts (smaller N) or shorter training?

3. You set KL coefficient to 0 (Table 3). Do results hold with a modest KL (as in many RLHF pipelines)? How sensitive are the conclusions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper directly addresses a critical issue in Reinforcement Learning with Verifiable Rewards (RLVR) for LLMs. The exploration dilemma, where LLMs get stuck in suboptimal solutions due to their pre-trained biases, is clearly articulated.

### Strengths
1. Introduction of a risk-sensitive RL framework for LLM fine-tuning.
2.Theoretical and empirical support for the risk-sensitive approach.
3. Demonstration of significant improvements in pass@k performance with maintained pass@1 accuracy.

### Weaknesses
1.The paper focuses on mathematical reasoning. Investigating the applicability of RS-GRPO to other OOD tasks is necessary.
2. While the paper presents an entropy analysis, it acknowledges that entropy loss may be a biased indicator of exploration. Exploring alternative metrics for quantifying exploration in this context could be beneficial.
3. The paper notes that for some models (e.g., Qwen2.5-7B, Llama3.1-8B-Instruct), RS-GRPO failed to outperform the base model at high values of *k*. Further investigation into the reasons for this behavior could lead to more robust performance across different LLM architectures.
4. A key limitation is the use of a fixed risk-seeking parameter.

### Questions
/

### Soundness
3

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
This paper introduces RS-GRPO, a method to address the "exploration dilemma" in RL fine-tuning of LLMs. It argues that standard RL improves pass@1 at the cost of pass@k by overly sharpening the policy. RS-GRPO uses a risk-seeking objective to encourage exploration of diverse solutions, aiming to improve pass@k while maintaining strong pass@1 performance.

### Strengths
1. Applying the established theory of Risk-Sensitive RL to LLM fine-tuning is a well-motivated.

2. Most claims are supported with both theoretical analysis and comprehensive experiments across multiple models and benchmarks.

### Weaknesses
Despite its strengths, the paper has several weaknesses that temper its overall contribution:
1. Unclear Improvement and Potentially Unfair Comparisons: The central claim of improvement is not consistently strong when viewed in context.
- If the main goal is to improve pass@k, the experimental results (Table 2, Figure 6) show that RS-GRPO performs comparably, and sometimes worse than, other baseline methods designed specifically for pass@k optimization.
- More importantly, the comparison to other pass@k methods may be unfair. Other methods like [1, 2] often employ a two-stage training strategy: first, they run pass@k training to enhance exploration and diversity, and then they perform a second stage of fine-tuning to distill this diversity into a strong pass@1 policy. This paper compares its single-stage method against only the first stage of these baselines, which may not represent their full potential, especially on the pass@1 metric.
2. Unverified Advantage on Continuous Rewards: The paper claims the ability to handle continuous rewards as an advantage, but all experiments use binary rewards, leaving this claim empirically unsubstantiated.
3. New Hyperparameter: The method introduces a hyperparameter, β, which requires tuning and adds complexity for practical application.

### Questions
1. Regarding the comparison with other pass@k methods: Could you comment on the fairness of comparing your single-stage method against baselines [1, 2] that often benefit from a second, pass@1-focused fine-tuning stage? Would applying a similar two-stage approach with RS-GRPO provide a more direct and fair comparison of the methods' full capabilities?
2. Regarding the continuous rewards: The ability to handle continuous rewards is presented as a significant advantage. Could you provide any experimental results, even on a smaller scale or a synthetic task, to demonstrate the practical benefits of RS-GRPO in such a setting?
3. The paper's motivation relies on the premise that pre-trained LLMs have a "sharply peaked" and "suboptimal" initial policy. Could you provide more direct evidence to support this? For instance, an analysis of the policy entropy or showing that the most probable reasoning paths are indeed suboptimal would greatly strengthen the paper's foundation.

[1] Chen, Z., Qin, X., Wu, Y., Ling, Y., Ye, Q., Zhao, W. X., & Shi, G. (2025). Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751.

[2] Li, Q., Xue, R., Wang, J., Zhou, M., Li, Z., Ji, X., ... & Yang, J. (2025). Cure: Critical-token-guided re-concatenation for entropy-collapse prevention. arXiv preprint arXiv:2508.11016.

### Soundness
3

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
This paper addresses the mismatch between the risk-neutral training objectives commonly used in RLVR (e.g., GRPO) and the risk-sensitive evaluation metrics such as pass@k. The authors propose a risk-seeking RL formulation based on an exponential utility function, leading to a risk-sensitive variant of GRPO, termed RS-GRPO. The paper derives the policy gradient for this formulation and demonstrates its benefits over standard GRPO in both bandit experiments and math reasoning tasks across seven base LLMs, with evaluation over $k \in [1, 1024]$. An ablation study finds that $k=2$ yields a good trade-off across different $k$ values in pass@k.

### Strengths
- The paper clearly articulates the problem of mismatch between training and evaluation in RL for LLMs and motivates the use of risk-sensitive objectives.  
- The derivation from exponential utility to the risk-sensitive policy gradient is mathematically sound and well presented.  
- Section 3 provides an excellent illustrative toy example that clarifies the intuition behind the approach and shows how risk-seeking objectives can encourage better exploration.  
- The method addresses the vanishing advantage issue for prior pass@k optimization methods.
- Extensive experiments across multiple base models and math reasoning benchmarks convincingly demonstrate consistent improvements over GRPO baseline.  
- The paper is well written, logically structured, and easy to follow, which is commendable given the technical depth of the topic.

### Weaknesses
**Lack of formal connection between risk-seeking objective and pass@k:**  
   The paper repeatedly motivates its approach by the mismatch between risk-neutral objectives and pass@k evaluation, yet it provides no theoretical derivation linking the risk-seeking objective to pass@k performance. While the intuition that risk-seeking encourages exploration is sound, a formal connection would substantially strengthen the argument.

**Lack of formal understanding for the vanishing-advantage analysis:**  
   Figure 8 shows the “vanishing advantage”. A formal proof or at least a more detailed analytical explanation of why prior works have this issue. 

**Fairness of baseline comparison:**  
   The paper compares to baselines trained with $k=4$ but evaluates at $k=1$ and $k=32$. Since prior works *directly* optimize pass@k, such comparison can be misleading. The authors should either match $k$ values during training or explicitly state that the comparison involves different optimization targets. Additionally, analysis of sensitivity to $N$ (number of samples) is needed, as the claimed robustness of RS-GRPO to vanishing advantage depends on this factor.

**Scope:**  
   While the method is domain-agnostic as claimed, experiments focus solely on math reasoning. It would be valuable to discuss whether RS-GRPO naturally extends to multi-turn RL in theory and other reasoning domains like coding in practice.

### Questions
- Figure 1 lacks clear axis labels and is difficult to interpret.  
- Please clarify how “dense signal” in Table 1 is defined.  
- What is the “cumulative solve rate” in the training metric?
- In Figure 3, the y-axis should be labeled “reward” rather than “reward distribution.”?

### Soundness
2

### Presentation
3

### Contribution
3
