# Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Policy optimization (PO) has advanced Large Language Models (LLMs), yet training remains constrained by a stability–exploration trade-off. We analyze the coupling between the input environment and the policy in LLM RL, and decouple parameter regularization from the optimization objective by moving regularization to the input side. Concretely, we propose **Environment-Regularized Policy Optimization (ERPO)**, instantiated with **Query-KL (QKL)**, which penalizes the KL divergence between the evolving query distribution and a fixed reference. By regularizing the input (query) distribution rather than the action (response) distribution, QKL indirectly controls policy drift induced by environmental shift while preserving exploration. To avoid premature convergence, we introduce a query-weighted advantage that reweights updates according to estimated query prevalence, reducing estimator variance and improving robustness. Across diverse
mathematical reasoning benchmarks, ERPO achieves KL control comparable to methods with explicit policy regularization, while delivering stronger final performance and smoother training dynamics. Temperature-swept sampling further indicates more stable long-horizon behavior. These results suggest that making the input environment a first-class object—via QKL and query-weighted advantage—
is a principled and practical route to improve the stability–exploration trade-off in PO for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Environment-Regularized Policy Optimization (ERPO), which stabilizes reinforcement learning for LLMs by regularizing the input (query) distribution instead of the output (policy) distribution. It introduces a Query-KL term that limits drift between the current and reference query samplers, treating prompts as part of the training environment. Additionally, ERPO applies a query-weighted advantage, reweighting updates based on query likelihood to reduce variance and improve robustness-while keeping policy-side exploration unrestricted.

### Strengths
1. Tackles an important issue: maintaining training stability vs. exploration in LLM reinforcement learning.

2. Simple and practical: ERPO adds only a Query-KL term and lightweight query-weighting to existing PPO/GRPO code, so it is not bound to any given method or model.

3. The paper is neatly written (except for some small things I'll mention in the next section).

### Weaknesses
1. The math in the Preliminaries section is hard to follow. A lot of symbols are introduced without an explanation. I would suggest a refinement of this section.

2. The method is described as general and compatible with any KL-based optimization method, but it is only compared with GRPO. It would be good to see at least one other method tested.

3. In the paragraph following Figure 3, the paper mentions Pass@k improvements, but Figure 3 itself only reports Avg@32. That line should probably reference Table 1 instead.

4. The claim that ERPO reduces the train–eval performance gap is interesting, but there’s no quantitative evidence shown. A simple table comparing training vs. evaluation accuracy can be helpful to show this.

5. I’m a bit confused about Figure 5-a. The text says GRPO shows larger fluctuations, but all methods seem to degrade similarly as temperature increases. It’s unclear how “fluctuation” is measured here

6. The colors for GRPO and ERPO differ between subplots in Figure 5. Please make the the same across subplots to avoid confusion.

7. In Table 2, ERPO’s advantage in the mean column seems to come almost entirely from the temperature = 1.5 case. Since that’s a pretty extreme temperature, I’m not sure how meaningful that gain is. It would help if the authors clarified why that range is important or provided averages over a more typical range (e.g., ≤1.0).

8. Table 2 also mixes experiments with different numbers of rollouts (8 vs. 16). Comparing across different rollout counts doesn’t seem entirely fair. Do we have GRPO results with 16 rollouts for a cleaner comparison?

### Questions
Please read the above section for questions. Thank you!

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the stability-exploration trade-off in LLM policy optimization by identifying environment non-stationarity (input query drift) as a unmanaged source of instability, even when output policy-KL is constrained.

To tackle this problem, the authors propose the Environment-Regularized Policy Optimization (ERPO), decouples regularization from the action space by: (1) Query-KL (QKL): Penalizing the KL divergence of the query distribution $\rho_\theta$ from a reference $\rho_{\theta_0}$ to control environment drift; (2) Query-Weighted Advantage: Reweighting advantages by query log-likelihood $w_{B}(s)$ to reduce variance.

By regularizing the input (queries) instead of the output (responses), ERPO aims to stabilize training while preserving exploration. Experiments on mathematical reasoning benchmarks show ERPO significantly outperforms the baseline (GRPO) in both final performance and long-term training stability, especially across varied sampling temperatures.

### Strengths
1. The paper identifies a novel and important source of instability: environment non-stationarity induced by the query distribution. This provides a compelling explanation for response distribution shift that goes beyond standard action-space analysis.
2. The proposed ERPO method is a principled solution that directly targets the identified problem. Decoupling the stability constraint (via Query-KL) from the exploration mechanism is an elegant approach to addressing the stability-exploration dilemma.
3. The paper provides convincing empirical support for its claims. Experiments on mathematical reasoning tasks show clear performance gains, and the analysis of long-term training and multi-temperature sampling offers robust evidence that ERPO is significantly more stable and robust to collapse than the baseline.

### Weaknesses
1. The proposed ERPO relies on calculating a log-likelihood for the query, $\log \rho_\theta(s)$. While this is feasible in the paper's experimental setup (mathematical reasoning), it is unclear how this log-likelihood could be computed for a static, offline dataset of human-written prompts (e.g., in standard RLHF).
2. The ablation study (Table 2) suggests that the query-reweighting component, $w_B(s)$, slightly hurts Pass@1 performance at low temperatures compared to using Query-KL alone. This makes its contribution unclear and warrants further explanation.

### Questions
1. In Figure 4(2), ERPO shows a smaller policy KL than GRPO, which directly penalizes policy KL. Could the authors provide some explanation or intuition for this result?
2. For Figure 4(3), could the authors provide the formula used for "Entropy Loss" and explain the mechanism by which ERPO achieves a lower entropy loss than the baseline?

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
The paper proposes a new method ERPO, which learns a weight coefficient for the GRPO objective in a batch. In some sense, this is similar to weighted importance sampling, but with learned weights (self-normalized). It is refreshing to see such idea being investigated in the LLM space. However, the paper's evaluation and reported score shows that this method (ERPO) only improves the performance of models in aggregate (over multiple temperatures). This is very different from how LLMs are typically evaluated -- therefore, the impact of this method seems very limited.

### Strengths
1. Non-stationary MDP/environment is an area worth investigating in LLM setting. However, the tasks covered in this paper appear to be generic/normal/i.i.d. training tasks where distribution shift (covariate drift) does not actually happen.
2. The paper reported the experimental result honestly and with integrity -- emphasizing it's an average over multiple temperatures.
3. The codebase is uploaded and easy to follow

### Weaknesses
1. In Table 2, we can see that ERPO is only narrowly beating GRPO in **1 out of 4** temperature settings. The fact that the mean is higher is only due to GRPO being particularly bad under one of the temperature setting. This evaluation is hardly fair.
2. Reporting average over multiple temperature training is a bit bizarre. Perhaps the author wants to illustrate that under higher temperature, there is more environment shift -- but that's not how distribution shift in MDP is typically defined. When you train on a math task, the initial state distribution (your tasks) did not shift -- regardless of your temperature. This type of slightly strange definition is concerning.

### Questions
Can the authors look at some of the RL work on this topic and rethink whether their setting truly contains any kind of distribution shift? [1] [2] 

It would be great if you can formulate and situate your paper after reading these related work.

[1] Mu, Tong, et al. "Factored DRO: Factored distributionally robust policies for contextual bandits." Advances in Neural Information Processing Systems 35 (2022): 8318-8331.

[2] Tennenholtz, G., Hallak, A., Dalal, G., Mannor, S., Chechik, G., & Shalit, U. (2021). On covariate shift of latent confounders in imitation and reinforcement learning. arXiv preprint arXiv:2110.06539.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Environment-Regularized Policy Optimization (ERPO), a new framework that stabilizes reinforcement learning for LLMs by regularizing the query (input) distribution rather than the policy (output) distribution. The authors introduce Query-KL (QKL) and a query-weighted advantage, which together control environment drift and reduce estimator variance. Empirical results on multiple reasoning benchmarks show improved stability and performance compared to GRPO.

### Strengths
- Treating the evolving input/query distribution as part of the optimization target is a fresh and theoretically grounded idea.
- QKL and query reweighting can be integrated into existing RLHF pipelines with minimal modification.
- The motivation is clear.

### Weaknesses
- The work motivates QKL heuristically but lacks a formal theoretical proof of stability or convergence guarantees.
- All experiments focus on math reasoning; it remains unclear whether the benefits generalize to non-verifiable or open-ended tasks.
- To be honest, the mathematical formulations are somewhat cumbersome.

### Questions
- I couldn’t find any information in the paper about the model size used. Could the authors clarify how many billion parameters their model has?
- If the main contribution lies in introducing the new QKL mechanism, then the experimental evaluation appears too limited. Could the authors conduct more comprehensive experiments on a broader range of datasets beyond RLVR?

### Soundness
3

### Presentation
2

### Contribution
3
