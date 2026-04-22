# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Recent large reasoning models (LRMs) driven by reinforcement learning algorithms (e.g., GRPO) have achieved remarkable performance on challenging reasoning tasks. However, these models suffer from overthinking, generating unnecessarily long and redundant reasoning even for simple questions, which substantially increases computational cost and response latency.  While existing methods incorporate length rewards to GRPO to promote concise reasoning, they incur significant performance degradation. We identify the root cause: when rewards for correct but long rollouts are penalized, GRPO's group-relative advantage function can assign them negative advantages, actively discouraging valid reasoning. To overcome this, we propose Decoupled Reward Policy Optimization (DRPO), a novel framework that decouples the length-based learning signal of correct rollouts from incorrect ones. DRPO ensures that reward signals for correct rollouts are normalized solely within the positive group, shielding them from interference by negative samples. The DRPO's objective is grounded in integrating an optimized positive data distribution, which maximizes length-based rewards under a KL regularization, into a discriminative objective. We derive a closed-form solution for this distribution, enabling efficient computation of the objective and its gradients using only on-policy data and importance weighting. Of independent interest, this formulation is general and can incorporate other preference rewards of positive data beyond length. Experiments on mathematical reasoning tasks demonstrate DRPO's significant superiority over six efficient reasoning baselines. Notably, with a 1.5B model, our method achieves 77\% length reduction with only 1.1\% performance loss on simple questions like GSM8k dataset, while the follow-up baseline sacrifices 4.3\% for 68\% length reduction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DRPO. 
DRPO tackles overthinking in large reasoning models by decoupling length-based rewards for correct answers from incorrect ones, preventing GRPO’s group-relative advantage from misclassifying verbose correct solutions as negative.
It introduces a KL-regularized, closed-form reweighting of on-policy correct samples that favors concise reasoning without ever flipping their learning signal to negative.
On four mathematical benchmarks, DRPO trims output length up to 77 % while losing only ~1 % accuracy, outperforming six strong baselines that sacrifice 4–7 % for smaller reductions.

### Strengths
* The paper identifies a subtle but critical flaw in GRPO’s group-relative advantage when length penalties are added, and proposes the first principled solution that decouples the normalization of correct and incorrect samples, a conceptual advance that is both novel and generalizable to other composite rewards.
* Extensive comparisons on four standard math-reasoning datasets show DRPO consistently dominates six recent baselines across 1.5 B and 7 B scales; ablations, AES metrics, and per-difficulty breakdowns confirm that the gains hold under rigorous evaluation, while a closed-form derivation guarantees computational efficiency without extra data.
* Good writing and easy to follow

### Weaknesses
* All experiments stop at 7 B parameters and four math domains; without data on >=30 B models or non-mathematical reasoning (code, science, commonsense).
* The length reward $r_l(o)=1−\frac{|o|}{C}$ is essentially linear; ablations with concave, step-wise or difficulty-aware penalties (e.g. LASER-D) are missing.

### Questions
* How does the performance of DRPO compare to the original DisCo algorithm? Adding relevant comparisons in Figure 3 would facilitate better comparison.
* Is DRPO still effective for step-wise rewards (i.e., process-level reward)?
* Could DRPO generalize to different families of LLMs?
* As mentioned in weakness, how DRPO performs on non-mathematical reasoning tasks?

### Soundness
2

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
5

### Summary
This paper proposes a novel RL objective to enhance reasoning efficiency in LLMs by encouraging shorter reasoning paths. The authors observe that applying a length penalty reward may cause long but correct responses to receive negative advantages, inadvertently promoting unnecessarily long reasoning trajectories. To address this issue, they propose a decoupled advantage computation that separately handles positive and negative responses, ensuring positive advantages for high-quality responses. Empirical studies on DeepSeek-R1-Distill-Qwen-1.5B and 7B models across multiple mathematical reasoning benchmarks verify the effectiveness of the proposed approach in reducing reasoning length while maintaining accuracy.

### Strengths
1. The paper is well written and clearly organized.

2. The analysis of positive responses receiving negative advantages is insightful and leads naturally to the proposed DRPO formulation.

3. The proposed DRPO method outperforms other efficient reasoning approaches by a substantial margin.

### Weaknesses
1. The authors claim that redundant reasoning arises from positive responses with negative advantages. However, even the vanilla GRPO without a length penalty still exhibits redundant reasoning, despite all positive responses receiving positive advantages. More in-depth analysis is needed to substantiate the claim that “positive responses with negative advantages greatly harm reasoning efficiency.”


2. The experiments lack a comparison against the vanilla GRPO baseline. In the abstract, the authors mention a 1.1% performance drop of DRPO compared to the baseline. If DRPO exhibits a larger accuracy decline than GRPO, it would weaken the claimed significance of the proposed efficiency improvement.


3. In my opinion, measuring reasoning efficiency solely by token count is not comprehensive. An efficient response should be logical, step-by-step, and free from redundant reasoning, rather than simply shorter in length.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes DRPO, a reinforcement learning framework to improve reasoning efficiency. It identifies a flaw in GRPO where length penalties distort advantage estimation and introduces a decoupled reward formulation that separates positive and negative samples. Experiments on multiple reasoning benchmarks show that DRPO significantly shortens reasoning chains with minimal accuracy loss.

### Strengths
1. Identifies and formalizes a core issue in GRPO under multi-objective rewards.
2. Proposes a principled decoupling mechanism with a closed-form optimization solution.
3. Demonstrates substantial efficiency gains with competitive accuracy across multiple benchmarks.

### Weaknesses
1. Evaluation focuses only on math reasoning; unclear applicability to other domains (e.g., coding, logic).
2. The evaluation relies mainly on **Pass@1**, which can be unstable for reasoning tasks; additional metrics (e.g., Pass@k) would strengthen the empirical validation.

### Questions
1. Could the decoupling idea generalize to other auxiliary rewards?
2. Could λ be learned dynamically during training (e.g., based on question difficulty or model confidence) rather than fixed manually?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the "overthinking" problem in Large Reasoning Models. Existing reinforcement learning-based solutions (e.g., GRPO) generally lead to performance degradation, which the paper attributes to the fact that current methods incorporating length rewards set the advantage value of correct but verbose reasoning to negative due to GRPO's group-relative advantage function, resulting in inverse optimization. To address this issue, the paper proposes the Decoupled Reward Policy Optimization (DRPO) framework, which is core-based on the DisCO discriminative optimization framework. It decouples the learning signals of correct and incorrect reasoning, normalizes length rewards only within the positive sample group, and integrates the optimized positive data distribution that maximizes length rewards under KL regularization, enabling efficient computation using only on-policy data. The paper validates the framework on multiple mathematical reasoning tasks: the 1.5B model achieves a 77% reduction in length with only a 1.1% performance loss on GSM8k, while the 7B model achieves a 51% reduction in length with only a 2.6% performance loss. These results are significantly superior to baseline methods, with a positive Efficiency-Accuracy Score.

### Strengths
- The method design is elegant. By decoupling the learning signals of correct and incorrect reasoning and normalizing length rewards only within the group of correct reasoning samples, it ensures that the rewards for correct but verbose reasoning are only proportionally reduced without becoming negative. The derivation of the optimal positive data distribution is particularly elegant.
- The performance is significant. The experiments demonstrate excellent performance across datasets of varying difficulty levels, achieving significant reductions in output length while maintaining nearly unchanged performance, outperforming baselines.

### Weaknesses
- The premise for DRPO to work is that positive and negative samples can be optimized separately, which strongly relies on the DISCO algorithm. This somewhat weakens the contributions of this paper.
- Missing references. It is recommended that the authors add discussions on the following works [1-4] in the 'large reasoning models' part of related work.
- The presentation of this paper is not very user-friendly, and some parts are hard to follow.

[1] Lyu C, Gao S, Gu Y, et al. Exploring the limit of outcome reward for learning mathematical reasoning[J]. arXiv preprint arXiv:2502.06781, 2025.

[2] Zheng C, Liu S, Li M, et al. Group sequence policy optimization[J]. arXiv preprint arXiv:2507.18071, 2025.

[3] Chen A, Li A, Gong B, et al. MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention[J]. arXiv preprint arXiv:2506.13585, 2025.

[4] Bai L, Cai Z, Cao Y, et al. Intern-s1: A scientific multimodal foundation model[J]. arXiv preprint arXiv:2508.15763, 2025.

### Questions
- Does the author consider some operations on the length optimization for the negative sample? At present, there is only the part for the positive sample in the design of the paper.

### Soundness
3

### Presentation
2

### Contribution
3
