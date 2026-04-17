# Concise Reasoning in the Lens of Lagrangian Optimization

- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Concise reasoning in large language models seeks to generate only essential intermediate steps needed to arrive at a final answer, thereby alleviating issues of overthinking. Most proposed approaches hinge on carefully hand-crafted heuristics, struggling to balance concision with performance, often failing to adapt across domains and model scales. In this work, we address these challenges by introducing a principled and pragmatic strategy, performance-aware length updating (PALU). 
As a principled algorithm, PALU formulates concise reasoning as a constrained optimization problem, minimizing response length subject to a performance constraint, and then applies *Lagrangian* optimization to convert it into a tractable unconstrained problem.
As a pragmatic solution, PALU streamlines complicated update rules through three approximations: *(i)* estimating performance with off-policy rollouts, *(ii)* truncating the *Lagrange* multiplier to two extremes, and *(iii)* replacing gradient-based updates with quantile-driven length adjustments. PALU reduces output length by 65\% while improving accuracy by 15\% when applied to *DeepSeek-Distill-Qwen-1.5B*, averaged over five benchmarks, outperforming a range of alternative methods. Furthermore, PALU is demonstrated to adapt across both domain (logic, STEM and math) and model scale (1.5B, 7B, 14B) entrenching the algorithm as a practical and effective concise reasoning approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PALU (Performance-Aware Length Update), a framework to achieve concise reasoning in large language models (LLMs). The key idea is to cast conciseness as a constrained optimization problem that minimizes output length under a performance constraint, then solve it using a Lagrangian relaxation. To make this computationally feasible, PALU introduces three approximations: (i) Off-policy performance estimation: reuses old rollouts to estimate pass rate. (ii) Regime-based control: replaces continuous λ updates with a two-regime “bang–bang” rule. (iii) Quantile-driven budget update: uses the dispersion of correct response lengths to approximate ∇ₗR, the sensitivity of reward to length. Experiments show 65% reduction in reasoning length with +15% accuracy improvement across MATH, AIME, AMC, MINERVA, and OlympiadBench tasks on DeepSeek-Distill-Qwen models (1.5B–14B).

### Strengths
1. Framing concise reasoning as a Lagrangian optimization problem is novel. While previous studies primarily focus on optimizing accuracy with a length penalty (i.e., $r = a - l$), this paper instead optimizes length with an accuracy penalty applied when performance falls below a predefined threshold c, i.e., $r = l + λ(c - a)$.

2. The three proposed approximations (off-policy, regime-based, and quantile-driven) make the method practically feasible. For instance, the “bang–bang” controller cleverly mitigates instability issues that typically arise from continuous updates of the Lagrange multiplier $\lambda$.

3. The authors conduct thorough experiments across multiple benchmarks, model sizes, and baselines. The consistent performance improvements clearly demonstrate the effectiveness of their proposed method.

### Weaknesses
1. The hyperparameter C plays a central role in PALU, yet its selection can critically affect model performance across datasets of varying difficulty. For example, the reasoning models evaluated in this paper typically achieve around 90% accuracy on GSM8K; setting C = 0.8 in such cases could allow PALU to overly relax the performance constraint, leading to a noticeable drop in accuracy. Moreover, the paper does not clearly analyze how different values of C influence the trade-off between conciseness and performance. In practical applications, the difficulty level of the evaluation dataset is often unknown, making it challenging to set C appropriately. The authors should conduct a systematic sensitivity study on C to validate its robustness and practical applicability.

2. Although the authors acknowledge that PALU may perform suboptimally when output lengths concentrate tightly, and they argue that such concentration is rarely observed in practice, there appears to be a clear trend toward tighter output distributions as model size increases. For instance, the 1.5B model produces outputs ranging from approximately 3,000 to 11,000 tokens, whereas the 32B model’s outputs span only 3,000 to 5,000 tokens. This suggests that larger models inherently generate more uniform reasoning lengths, potentially weakening PALU’s effectiveness in larger models.

### Questions
1. Can you provide the analysis on $C$?

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
4

### Summary
The paper proposes Performance-Aware Length Updating, a method to make LLM reasoning more concise and efficient. It formulates concise reasoning as a constrained optimization problem, minimizing reasoning length while maintaining accuracy above a target threshold, using a Lagrangian optimization framework. To make this practical, the authors introduce three approximations: off-policy performance estimation, a regime-based controller for dynamic budget adjustment, and a quantile-driven update rule that adapts token budgets based on the length distribution of correct responses. Evaluation shows improvements in both efficiency and accuracy across multiple reasoning benchmarks and model sizes.

### Strengths
- The paper tackles a timely and important problem in LLM research.

- The paper provides a clear theoretical formulation of concise reasoning through a Lagrangian optimization framework, giving a principled view of how to balance accuracy and reasoning length.

- The background and positioning are well done. The paper clearly classifies existing approaches (e.g., fixed budget, reward shaping, pruning-based) and situates PALU as a more general, optimization-driven alternative.

- The assumptions and approximations are well-motivated and intuitively appealing: (i) using an off-policy pass-rate estimate to avoid redundant rollouts, (ii) employing a regime-based controller that dynamically shifts between accuracy recovery and compression, and (iii) adopting a quantile-based surrogate to stabilize gradient updates. Together, these make the framework practical and computationally efficient.

- It provides convincing empirical support for the method's design by including ablation studies that isolate the impact of each approximation.

- The method shows  improvements in both efficiency and accuracy, demonstrating that shorter reasoning traces can indeed be achieved without sacrificing correctness.

### Weaknesses
- The evaluation is limited. Experiments are conducted using only one model family (DeepSeek-R1-Distill-Qwen-1.5B), which restricts the generality of the conclusions. It is unclear whether the proposed framework scales similarly to larger or different architectures.


- The computational overhead of the proposed approach is not discussed. Since PALU involves additional monitoring, budget updates, and off-policy estimation, it is important to quantify the real end-to-end runtime or memory cost to validate the claimed efficiency gains.



- The impact of core assumptions and approximations on performance is not analyzed in depth. How sensitive is PALU to these design choices, and could alternative relaxations yield better trade-offs?


- Building on the previous point, there is no systematic ablation to isolate the contribution of each of the three components. Showing how much each assumption contributes to the overall gain (or whether some are redundant) would make the framework more interpretable and robust.


- The paper does not compare alternative theoretical relaxations or optimization formulations. Since the method is inspired by Lagrangian optimization, it would be useful to understand whether simpler or more direct approaches could achieve similar results.



- Although the method is theoretically motivated, the paper lacks deeper intuition or discussion of failure cases. For example, when the quantile update might be unstable or when the off-policy pass-rate estimate becomes inaccurate.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a new RL algorithm named PALU for training LLMs to achieve efficient reasoning while maintaining accuracy. PALU considered the efficient/concise LLM reasoning as a constrained optimization problem, which is to minimize generation length subject to a performance constraint. With the Lagrangian duality, this problem can be converted into an unconstrained min–max problem, and PALU designed 3 pragmatic approximations in this work. Experimental results showed that the proposed PALU can reduce the output token number and maintain or even surpass the baselines.

### Strengths
1. This work considered concise reasoning (efficient reasoning) as a constrained optimization problem with a performance constraint, which is theoretically sound and can align with classical optimization frameworks. Such a perspective provided a clear objective that balances conciseness and correctness, which is better than existing heuristic reward methods.

2. The macro average accuracy of the proposed PLAU was improved, and the token budgets have been decreased to a large margin compared to most baselines.

### Weaknesses
1. The presented PALU estimated the expected reward $R(\theta, L, q)$ for the current policy $\pi_{\text{new}} $ using rollouts from the previous policy $\pi_{\text{old}}$, which is a conservative estimate but without quantifying the direction or magnitude of the bias, nor provide empirical analysis of the bias-variance trade-off. 

2. When training enters the strong coupling between performance and length regime, i.e., the green phase in Figure 2, rollouts from the old policy may overestimate the new policy’s performance, leading to underestimation of $\lambda$ and excessive length compression. Therefore, the monotonically decreasing curve may be an artifact of this biased estimator.

3. Reducing the continuous $\lambda$ to a binary switch converted the optimization into a discrete. And this paper didn’t provide Lyapunov function, fixed-point analysis, or periodic trajectory proofs, and didn’t not address the possibility of limit-cycle oscillations between the token length and performance. 

4. The main experimental results exhibited a better macro average accuracy, but the improvements were mostly from the AMC 2023, while PALU didn’t outperform most baselines on other benchmarks. Also, there is no evaluation on the newest AIME 2025.

5. The Pass@1 is computed over only 10 rollouts (32 for AIME 2024), which may be insufficient for a stable estimation, especially on harder benchmarks like Olympiad or AIME, undermining the confidence for the evaluation.

### Questions
1. This paper estimated the expected reward $R(\theta, L, q)$ for the current policy $\pi_{\text{new}}$ via rollouts from the old policy $\pi_{\text{old}}$. When training enters the strong performance–length coupling regime, would this bias always be positive?

2. Could this discrete-event system exhibit limit cycles or chaotic trajectories?

3. When correct samples are extremely scarce (at the early RL stage), how to prevent the step size from exploding or vanishing?

4. How sensitive is PALU to the C threshold? The paper adopted C = 0.8; what happens if C is larger or smaller? 

5. All experiments were conducted with DeepSeek-R1-Distilled-Qwen models, have the authors tried other reasoning LLMs?

6. Is the macro average accuracy improvement due to denoising or regularization?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Concise reasoning cab seen as constrained optimization! Simply the authors frame it as: minimize response length L subject to performance R(theta,L,q) >= C. Converts to Lagrangian min-max problem L(theta,L,λ) = L + λ(C - R(theta,L,q)) with updates λ <- λ + n(C-R), theta ← theta + nλ∇R, L <- L - n(1-λ∇R). Proposes PALU with three approximations: (i) off-policy performance estimation reuses last epoch's rollouts; (ii) regime-based optimization snaps λ to two extremes - if R>=C reduce L by α_τ^q, else reset L=L_max; (iii) quantile-driven budget update where α_τ^q = Q_1.0^q - Q_{1-τ}^q measures gap between longest correct response and (1-τ) quantile, approximating ∇R via finite difference τ/α_τ^q.

### Strengths
I love this paper. Because it has a principled formulation. Grounding concise reasoning in Lagrangian optimization is elegant and provides theoretical justification for balancing length/accuracy unlike ad-hoc heuristics. Quantile-based step size α_τ^q is clever - using distribution of correct response lengths to determine reduction magnitude is data-driven and intuitive. Strong empirical results - 65% compression with 15% accuracy gain on 1.5B model beats all baselines including multi-stage methods like AutoThink. Figure 2 left shows nice dynamics - negative correlation early (easy redundancies) then positive (hard tradeoffs) with PALU continuing to improve both. Cross-domain and cross-scale experiments (Figure 3) demonstrate genuine adaptivity - stage-based budgeting and overlong punishment fail when initial length differs from assumed 8k target while PALU adapts naturally. AE Score metric properly weights accuracy preservation (theta=5 penalty vs n=3 bonus). Off-policy performance estimation is pragmatic - reusing last epoch avoids expensive recomputation. Generation length assumption validated empirically (Figure 1, Figure 5) showing 2-3x spread supporting premise.

### Weaknesses
Regime-based controller might be too simplistic? - binary snap to L_max or reduce by α_τ discards continuous λ updates that Lagrangian theory prescribes. There is a claim this "preserves sign behavior" but loses magnitude information that could enable finer control. Why not just use gradient ascent with proper learning rates? Off-policy approximation is conservative but introduces lag - performance might improve but policy still thinks constraint violated leading to unnecessary L_max resets? I would love to see an analysis of how often this happens. Quantile approximation  assumes reducing L by α_τ drops success rate by exactly τ but this only holds if L is near Q_1.0. For smaller L values this approximation breaks down? Figure 4 ablation shows α_0.1 vs α_0.5 but doesn't test different τ values in quantile definition. Performance threshold C=0.8 fixed across all experiments - why 0.8? Compute cost comparison missing - paper claims "efficiency" but reports 1100 H200 GPU hours for 1.5B model which seems expensive. How does this compare to baseline GRPO training?

### Questions
How does PALU behave when performance constraint C cannot be satisfied? On AIME24 baseline is 28.5% so initial performance is far below C=0.8. Does L just stay at L_max indefinitely? Table 2 shows final AIME24 accuracy 40.0% still below threshold - what is the training dynamic here? Can you provide analysis of how often regime controller resets to L_max vs reduces by α_τ? This would show whether binary controller actually captures Lagrangian dynamics or just oscillates. The quantile gap α_τ^q requires computing Q_1.0^q which is the longest correct response - but if some questions have very long outlier solutions this could make α_τ^q very large causing aggressive reduction. Do you clip or filter outliers? Why τ=0.5 specifically?

### Soundness
4

### Presentation
3

### Contribution
4
