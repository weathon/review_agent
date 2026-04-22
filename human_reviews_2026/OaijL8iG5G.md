# CMDPO: Centered Mirror Descent Policy Optimization for Stable and Efficient Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) have shown strong performance in diverse tasks but require post-training alignment, where reinforcement learning plays a key role. Existing methods such as proximal policy optimization (PPO) and direct preference optimization (DPO) suffer from limitations like high computational overhead and overfitting. Although group relative policy optimization (GRPO) addresses some of these issues, its reliance on weighted negative log-likelihood lacks theoretical convergence guarantees. Furthermore, mirror descent policy optimization (MDPO), while more stable, requires computationally expensive partition function estimation. To overcome these challenges, this study introduces centered mirror descent policy optimization (CMDPO), a policy optimization framework that eliminates the need for explicit partition function estimation through group centering. CMDPO ensures unbiased and consistent estimates with strong theoretical guarantees. Optionally, we add two lightweight utilities for improved stability: dynamic reward weighting to balance heterogeneous rewards and token-level discriminative learning to reduce shared-segment dominance. Comprehensive experiments across multiple benchmark datasets demonstrate the effectiveness and robustness of CMDPO, which is further proven theoretically as a promising approach for LLMs' post-training.  The code is accessible at https://anonymous.4open.science/r/CMDPO-0C26.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Centered Mirror Descent Policy Optimization (CMDPO), a novel reinforcement learning algorithm for aligning large language models. It aims to overcome limitations of existing methods like PPO, DPO, and MDPO. CMDPO's core innovation is a group centering technique that mathematically eliminates the need to estimate the problematic partition function (log Z) present in MDPO, thereby achieving unbiased and theoretically sound policy updates. The framework is further enhanced with dynamic reward weighting to balance task and format rewards and token-level discriminative learning to focus on quality-indicative tokens.

### Strengths
1. CMDPO directly addresses a theoretical weakness in MDPO (partition function estimation bias) with a mathematically solution that provides unbiased estimates and guarantees a unique optimal solution.
2. The dynamic reward weighting and token-level discriminative learning mechanisms address practical challenges in RL training, such as balancing different reward signals and improving fine-grained learning focus.

### Weaknesses
1. The overall contribution might be seen as an incremental improvement over MDPO, combined with practical heuristics (dynamic weighting, token-level learning) that are potentially applicable to other RL algorithms.
2. The framework introduces several new components and associated hyperparameters (e.g., $s$, $\tau$ for dynamic weighting; $\delta$ for token learning), potentially making tuning more complex than simpler methods like GRPO or DPO.

### Questions
See Weaknesses

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
4

### Summary
The paper develops an interesting approach for post-training LLMs, improving upon MDPO by introducing a group-centering trick that removes the need to estimate the difficult partition function. It also adds dynamic reward weighting to automatically balance task and format rewards, and token-level discriminative learning to focus on meaningful tokens rather than repetitive text. Experimental results show promising improvements over standard baselines.

### Strengths
- One of the key and primary strength of the paper lies in the group centering idea which cleanly removes estimation of the log-partition and heavy computation thereby keeping MDPO’s convergence guarantees while improving training stability.
- The variance-covariance analysis part is ineteresting and tells why it stabilizes learning
- Dynamic Re-weighting and Token-level Discriminative Learning are simple but effective ideas
- The paper shows promising results by outperforming baselines in several math olympiad and other benchmarks.

### Weaknesses
- One of the most confusing part is that 2-3 components have been added like Dynamic Re-weighting and Token-level Discriminative Learning which I appreciate that improves the performance but is not clearly analyzed
-  Can a detailed plot be shown to understand the benefit of each and how and why each is helping over others?
-  Also these additional two methods come as adhoc to the story which makes it slightly confusing from the central theme that is improving the tractability of MDPO
- How does the batch-size affect the convergence is crucial? A clear study will be helpful to understand the batch-size and how it impacts in the CMDPO performance? How is the sensitivity compared to GRPO?
- In GRPO standard also, the same reward normalization is done but the learning objective is difference, so a clean analysis comparing the two with one factor at a time will be helpful.

I am interested to understand it better and will increase score accordingly.

### Questions
See weakness

### Soundness
3

### Presentation
2

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
The paper presents CMDPO, a centered variant of Mirror Descent Policy Optimization (MDPO), aiming to improve the stability and efficiency of reinforcement learning–based post-training for large models. CMDPO eliminates the need to estimate the intractable normalization term (log Z) in MDPO by introducing a group-centering operation, and further enhances training with two engineering additions: dynamic reward weighting (DRW) to balance result and format rewards, and token-level discriminative learning to down-weight shared segments. The method is theoretically shown to have a unique optimum and an unbiased consistent estimator. Experiments on reasoning, multimodal classification, and visual grounding tasks demonstrate modest but consistent improvements in convergence and reward–KL efficiency compared to MDPO, GRPO, and GPG.

### Strengths
1.The motivation is clear: CMDPO directly addresses the instability in MDPO caused by estimating the partition function (log Z).

2.The centering operation is simple yet effective, achieving stable optimization without changing the underlying objective structure.

3.Theoretical analysis is rigorous, including proofs of optimality and consistency.

4.The method is lightweight and easily integrable into existing GRPO/MDPO training pipelines.

5.The paper is clearly written and well organized, making it easy to follow and reproduce.

### Weaknesses
1.The contribution is basically incremental. The centering trick is intuitive and conceptually similar to normalization or variance-reduction ideas used in GRPO and related works.

2.Experimental improvements are modest (around 1–2 points on reasoning benchmarks) and mainly compared within the MDPO/GRPO family; stronger baselines or modern preference-optimization approaches are missing.

3.Experiments are conducted only on relatively small-scale models (around 1B–3B parameters) and do not include larger-scale settings such as 7B or 32B models commonly used in RLHF pipelines. The absence of large-model verification limits the generality and practical relevance of the proposed method for real-world LLM post-training.

### Questions
None

### Soundness
2

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
This paper identifies a critical, unsolved problem in Mirror Descent Policy Optimization (MDPO): the need to estimate the computationally intractable partition function log Z . While MDPO offers a stable, theoretically-grounded alignment objective, existing approximations of log Z using limited samples introduce significant bias, degrading performance .

To solve this, the authors propose Centered Mirror Descent Policy Optimization (CMDPO). The core contribution is a simple and elegant "group centering" mechanism . By subtracting the batch-mean from both the true reward r(x, y) and the implicit reward R_θ(x, y), the log Z terms on both sides of the equation cancel out, completely eliminating the need for its estimation . The paper proves that this new centered objective, L_CMDPO, retains MDPO's theoretical benefits, including a unique optimal solution (Theorem 1) and an unbiased, consistent estimator (Theorem 2).

The paper also introduces two additional heuristic mechanisms to boost performance:

Dynamic Reward Weighting (DRW): Adaptively balances "result" and "format" rewards .

Token-level Discriminative Learning (TDL): Down-weights the loss contribution of shared, common token segments in a batch to focus learning on discriminative tokens.

Empirically, the full CMDPO framework is shown to outperform other critic-free methods like GRPO, GPG, and MDPO on reasoning and classification benchmarks.

### Strengths
Elegant, Principled Solution to log Z: The primary strength is the CMDPO loss function (Eq. 5) . Eliminating the intractable partition function log Z from the MDPO objective via simple mean-subtraction is a clever, sound, and important theoretical contribution.

Strong Theoretical Guarantees: The paper provides robust theoretical backing for its core loss, including proofs of a unique optimal solution (Theorem 1) and an unbiased, consistent estimator (Theorem 2). This places it on firmer theoretical ground than heuristic-driven methods like GRPO.

Strong Relative Performance: The empirical results clearly show that CMDPO is superior to its closest relatives (MDPO, GRPO, and GPG). The ablation on group size k (Table 7) is particularly effective, showing that CMDPO's advantage over MDPO is largest for small k, precisely where MDPO's log Z approximation bias is worst.

Insightful Analysis: The ablations for the heuristic add-ons (DRW in Table 6, TDL in Table 4) are good and show that these components provide additional, independent gains.

### Weaknesses
Missing Critical Baselines: The most significant weakness is the omission of PPO and DPO as baselines in the main experiments. The paper is motivated by the flaws in PPO and DPO , so failing to compare against them is a major gap. Without this, it's impossible to judge if CMDPO is a true SOTA contender or just the best of the less-common (MDPO/GRPO) methods.

Confounding Three Methods in One: The paper presents a "bag of tricks" (CMDPO + DRW + TDL) as a single algorithm. This is a weak framing. The paper would be much stronger if it presented CMDPO as the core principled contribution, and then showed that its performance can be further enhanced with optional, general-purpose heuristics like DRW and TDL.


Overly Complex Heuristics: The two add-on heuristics, DRW (Eq. 8)  and TDL (Eq. 10), are complex, unprincipled, and add new hyperparameters (s, τ, δ). They detract from the simplicity and elegance of the core CMDPO contribution. The TDL mechanism, in particular, seems computationally expensive as it requires finding the longest common prefix across all k samples at the token level for every step.

### Questions
Why were PPO and DPO, the most widely used alignment methods, not included as baselines in your main experiments (Table 1-3)? This comparison is essential to understand the practical significance and performance of CMDPO relative to the field's standards.

The paper presents three novelties: CMDPO (loss), DRW (reward weighting), and TDL (token weighting). Your ablations show they are additive. Have you tested if the DRW and TDL heuristics could also be applied to other algorithms like GRPO or MDPO? This would help clarify if they are general-purpose improvements or specific to CMDPO.

Could you provide an analysis of the computational overhead of the Token-level Discriminative Learning (TDL) mechanism? Finding the longest common prefix across k sequences at every training step (Eq. 10) seems computationally intensive.

### Soundness
3

### Presentation
3

### Contribution
3
