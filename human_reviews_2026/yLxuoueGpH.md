# Focusing on the Riskiest: Gaussian Mixture Models for Safe Reinforcement Learning

- Decision: Reject
- Scores: 4, 2, 4, 8, 6

## Abstract
Reinforcement learning under safety constraints remains a fundamental challenge. While primal–dual formulations provide a principled framework for enforcing such constraints, their effectiveness depends critically on accurate modeling of cost distributions. Existing approaches often impose Gaussian assumptions and approximate risk either by the mean or by CVaR, yet these formulations inherently fail to capture complex, multimodal, or heavy-tailed risks. To overcome these limitations, we propose GMM‑SSAC(Gaussian Mixture Model‑Based Supremum CVaR‑Guided Safe Soft Actor‑Critic), whose core is the Supremum Conditional Value‑at‑Risk (SCVaR) criterion: a coherent and robust safety measure that explicitly targets the worst‑case tail across all components of a Gaussian mixture. To support accurate SCVaR estimation online, we introduce an incremental EM‑based update that refines the GMM parameters by blending instantaneous safety samples with Bellman‑transformed estimates—ensuring unbiased, convergent parameter estimates for reliable SCVaR computation. Empirical evaluations on standard safety benchmarks demonstrate that GMM‑SSAC substantially improves risk sensitivity and safety while maintaining competitive task performance, validating SCVaR as a principled and effective cost estimator for safe reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work mainly considers safe RL with safety constraints. Specially, CVaR may fail to capture complex, multimodal, or heavy-tailed risks, thus this work proposes the Supremum Conditional Value‑at‑Risk (SCVaR) for capturing worst‑case tail across all components of a Gaussian mixture. Consequently, combining with an EM‑based method to update the GMM parameters, the proposed GMM‑SSAC(Gaussian Mixture Model‑Based Supremum CVaR‑Guided Safe Soft Actor‑Critic) can estimate reliable SCVaR. Extensive theoretical and experimental results show that GMM‑SSAC is better than previous CVaR-based RL methods.

### Strengths
- I really like the idea of introducing new metrics in safe RL, thus I think SCVaR is a clear contribution if authors can introduce its advantages in safe RL more clearly (see weakness).

- SCVaR has some obvious insights like it can be easily computed in GMM, which is an expressive distributional framework.

- Lots of theoretical analyses clearly state the properties of SCVaR.

- The writing is great and easy to read.

### Weaknesses
- My major concern is, what is the main advantages of SCVaR compared with CVaR? In my understanding, CVaR considers the tail of the distribution and SCVaR considers the maximum CVaR of each component. What are the benefits of ignoring the tail distribution of other components? Providing some theoretical or experimental insight will make this work more solid. Also, a natural question is, can SCVaR be extended to any distributions that can not be represented by GMM? As CVaR is well defined in all distributions, the application of SCVaR will be limited if it can only be considered on GMM (of course CVaR of complex distribution can not be directly computed but still can be estimated).

- Assume that the ground truth distribution is GMM, there are always estimated gap between our estimated GMM and the ground truth distribution. Under this situation, what about the relationship of the estimated SCVaR and the ground truth SCVaR?

- Assume that the ground truth distribution is **not** GMM, of course we can utilize a GMM to estimate this distribution and calculated our estimated SCVaR, then what is the meaning of this estimated SCVaR?

- In experiments, I think a natural ablation study is that utilizing GMM to estimate the distribution and use CVaR of the estimated GMM to measure the risk, which might be a good comparison of SCVaR and CVaR.

- There are several works on safe RL with different risk measures need to be discussed, like CVaR [1-3] and EVaR [4-5].

Overall, I think this work is currently boardline, I'd like to actively join in the following discussion and adjust my score if the authors can address my concerns.

Ref:

[1] Towards safe reinforcement learning via constraining conditional value-at-risk

[2] Efficient off-policy safe reinforcement learning using trust region conditional value at risk

[3] Risk-sensitive reward-free reinforcement learning with cvar

[4] Risk-sensitive reinforcement learning via Entropic-VaR optimization

[5] Evar optimization in mdps with total reward criterion

### Questions
See weaknesses above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The authors propose a risk-averse safe RL algorithm that maximizes reward while reducing the risk measure of the cost return.
- They parametrically estimate the distribution of the cost return using a Gaussian Mixture Model (GMM).
- They propose a coherent risk measure, called SCVaR, which can compute using GMM.

### Strengths
- While using GMM to estimate the cost return distribution has addressed in prior work (GMAC [1]), proposing SCVaR via this parameterization is novel.
- The authors analyze the convergence of the proposed Bellman operator.
- The presentation is clearly and effectively presented.

[1] Nam, Daniel W., Younghoon Kim, and Chan Y. Park. "Gmac: A distributional perspective on actor-critic framework." *International Conference on Machine Learning*. PMLR, 2021.

### Weaknesses
- The introduction lacks analysis of prior work and appears biased.
    - They mention only methods approximating the cost return distribution with a single Gaussian.
    - However, numerous distributional RL approaches exist for more realistic estimation, such as quantile regression [1], percentile-based methods [2], and moment parameterization [3].
    - Omitting these references reveals a limited understanding of prior work.
    - Additionally, the authors did not cite GMAC, a prior method that estimates return distributions using a GMM, which is closely related to the proposed method.
- While convergence is shown for the critic, it is not guaranteed to achieve an optimal policy.
- Quantile-based parameterization can use various risk measures, but the proposed method is limited to SCVaR.
    - This drawback is neither mitigated nor offset by advantages of the proposed method.
    - While SCVaR is more conservative than CVaR, adjusting $\alpha$ of CVaR could achieve similar effects.
    - Additional analysis of SCVaR's physical properties would help readers intuitively tune $\alpha$ and $K$.
- The experiments include too few risk-constrained RL baselines.
    - CAL focuses on conservative policy updates rather than solving risk-defined constraints.
    - SAC-Lag is risk-neutral.
    - Only WC-SAC is relevant.
    - Others, such as CPPO [4], CVaR-CPO [5], and SDAC [6], should be included.
    - SRCPO [7], which proves convergence to an optimal policy for risk-constrained RL, is essential for comparison.
- In the experimental results, mean + std exceeds the threshold in all tasks except Ant.
    - Despite using risk constraints, this indicates failure to obtain conservative policies.

[1] Bellemare, Marc G., Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning." *International conference on machine learning*. PMLR, 2017.

[2] Dabney, Will, et al. "Distributional reinforcement learning with quantile regression." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 32. No. 1. 2018.

[3] Cho, Taehyun, et al. "Bellman Unbiasedness: Toward Provably Efficient Distributional Reinforcement Learning with General Value Function Approximation." *Forty-second International Conference on Machine Learning*.

[4] Chengyang Ying, Xinning Zhou, Hang Su, Dong Yan, Ning Chen, and Jun Zhu. Towardssafe reinforcement learning via constraining conditional value-at-risk. In Proceedings ofInternational Joint Conference on Artificial Intelligence, 2022.

[5] Qiyuan Zhang, Shu Leng, Xiaoteng Ma, Qihan Liu, Xueqian Wang, Bin Liang, Yu Liu, and JunYang. CVaR-constrained policy optimization for safe reinforcement learning. IEEE Transactionson Neural Networks and Learning Systems, 2024.

[6] Kim, Dohyeong, Kyungjae Lee, and Songhwai Oh. "Trust region-based safe distributional reinforcement learning for multiple constraints." *Advances in neural information processing systems*, 2023.

[7] Kim, Dohyeong, et al. "Spectral-risk safe reinforcement learning with convergence guarantees." *Advances in Neural Information Processing Systems,* 2024.

### Questions
- Can it be shown that SC-MGR converges to the ground truth distribution as the number of GMM components $K$ goes to infinity?
- Figure 3 contains too many equations, making it hard to follow. Can it be simplified?
- According to the primal-dual method, should Equation 22 be written as $-\kappa (\Lambda - d)_{\leq 0}$?
- Experiments on $K$ in SCVaR show that larger $K$ increases conservativeness.
    - However, the relationship between $\alpha$ and $K$ remains unclear, making it difficult to choose an appropriate $K$.
    - Could you provide guidance to help readers select a suitable $K$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GMM-SSAC, which models the cumulative cost distribution with a Gaussian Mixture and introduces SCVaR, the maximum CVaR across mixture components, to emphasize the worst-case tail among multimodal risks. This matters because single-Gaussian critics can underestimate tail risk and miss heavy-tailed/multi-peaked structure in safety-critical settings; in contrast, SCVaR is conservative (upper-bounds mixture CVaR) and coherent. Empirically, GMM-SSAC reduces safety violations both during training and evaluation while maintaining competitive rewards, with $\alpha$ controlling the safety–reward trade-off.

### Strengths
- Addresses the limitation of single-Gaussian critics in modeling multimodal risk.
- Solid formulation of SCVaR and clear integration with SAC.
- Empirical results show fewer safety violations on benchmark tasks.

### Weaknesses
- Novelty is limited relative to the existing WC-SAC (CVaR-SAC with Gaussian costs) and CAL (multiple Gaussian cost estimates with UCB aggregation). The contribution centers on SCVaR and Bellman–EM projection is incremental rather than a fundamentally new safety-risk paradigm.
- The evaluation is conducted on some traditional RL testing benchmarks but the ablations show performance depends on $K$ and $\beta$; very high $\beta$ degrades performance (variance from EM+Bellman), suggesting some instability that merits deeper analysis.
- Runtime/sample-efficiency analysis is missing: while compute setup and wall-times are reported, there’s no per-update overhead or learning-curve comparison against baselines to quantify the cost of EM/GMM (especially as K grows).

### Questions
- The observed multimodality in cost distributions is assumed rather than empirical validated. Any validation for this?
- Does SCVaR fundamentally add value over simply using a lower CVaR confidence level (smaller $\alpha$) with a standard critic?
- How do we know the observed multimodality in cost distributions is real and not an artifact of function approximation noise?
- How stable is the online EM procedure under off-policy distributional shift?

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
4

### Summary
This paper addresses a key limitation in CMDP safety constraints by replacing the standard Gaussian cost distribution assumption with Gaussian Mixture Models (GMMs). The authors introduce SCVaR (Supremum Conditional Value at Risk) to capture worst-case risk across GMM components, providing a more robust safety measure than traditional CVaR. To enable online cost distribution estimation without waiting for episode completion, they propose a Bellman-style incremental update that bootstraps from instantaneous safety measures and historical distribution estimates. The approach is evaluated on Safety Gym benchmarks, demonstrating improved safety guarantees while maintaining performance.

### Strengths
- This is a nice contribution in terms of modeling cost and a right step for studying CMDPs. As an initial approach to CMDP cost modeling, making such estimates more conservative (rather than just expectation over cost) and integrating the Bellman update for cost distribution, this will inspire more methods that will reframe the cost and approach solving CMDPs with more robust methods compared to GMMs (see weakness and/or questions below on this).

- The paper includes an effective ablation study to validate the components - they show comparisons between CVaR and SCVaR, usage of different number of Gaussian components for GMMs and their sensitivity to it. 

- The paper has a nice bit of theory as well, including a proof of Bellman update contraction.

- Sound empirical results including a comparison of SCVaR vs CVaR performance in GMMs.

### Weaknesses
- The presents three images, first for explaining the current usage of Normal distribution for cost, and another image for explaining SCVaR, and the third for explaining the algorithm and integration with RL env - actor cycle. I would have preferred a more linear approach to fig three to sequentially explain the process of RL and integration of SCVaR + GMM - this could have been accomplished by rearranging the figure first showing the RL env-actor cycle, breaking action into three components policy, value, and cost, and breaking cost and explaining the use of SCVaR and GMM.

### Questions
- Why GMM? Perhaps the choice lies in its simplicity and relatively "easy" treatment in terms of theory. But questions arise about accuracy, expressiveness, parameter efficiency and so on. The paper actually motivates this choice in terms of expressiveness and the fact that they have universal approximation capability. We have so many popular, SOTA distributional models that might currently be used for other purposes (e.g. as generative models) that might serve as a better basis or backbone for the SCVaR component. I am curious what the authors think about this.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GMM-SSAC, a safe RL framework that models the safety-cost distribution using Gaussian Mixture Models (GMMs) and introduces Supremum Conditional Value-at-Risk (SCVaR), the maximum CVaR across mixture components, as a conservative risk metric. The safety critic is trained through a Bellman-consistent incremental EM update, while the actor minimizes an SAC-style objective penalized by SCVaR. Theoretical sections show that SCVaR upper-bounds the mixture CVaR and is a coherent risk measure; experiments on Safety-Gymnasium and velocity-constrained MuJoCo tasks show improved constraint satisfaction with comparable reward.

### Strengths
* **Sound extension**: Modeling multimodal or heavy-tailed cost distributions is a reasonable step beyond unimodal Gaussian critics.
* **Intuitive concept**: The SCVaR metric provides an easy-to-understand conservative surrogate for tail-risk control of the GMM.
* **Technical integration**: The paper combines a GMM-based safety critic, an incremental EM fitting step, and a primal-dual-style policy update into a coherent framework.

### Weaknesses
## Conceptual and Empirical Alignment

1. **Multi-modality assumption**: The paper tries to motivate the necessity of multimodal safety cost distribution in Fig. 1, but it is unclear to me whether the true safety cost is multimodal. The constraints tested in the experiment section all seem to be unimodal (e.g., velocity limit, distance to hazard). It'd help to illustrate the scenarios where unimodal cost model fails and multi-modal is required to maintain safety.

2. **Missing connection to distributional robust safe RL**: The proposed critic models a full cost distribution (instead of a point estimate like SAC-Lagrangian) and optimizes a SCVaR. This approach is conceptually closer to distributionally robust CMDP formulations than to standard SAC-Lag baselines. The paper would benefit from comparisons or discussion along that line.

## Theoretical Clarity

3. **Significance of Theorem 1**: The paper could discuss the significance of Theorem 1. It's understood that SCVaR ≥ CVaR and implies conservatism. But since the GMM distribution is estimated and available, why not use the CVaR of the GMM distribution directly? Using an upper-bound (SCVaR) introduces additional conservatism and it's unclear to me why it should be used in the optimization program instead of CVaR.

4. **Intuition on the refinement operator $\mathcal{R}$ and $\beta$**: The historical sample set $\Psi(s, a)$ and Bellman-transformed set $\Psi_{\beta}(s, a)$ are both sampled from target safety critic $\mathcal{G}^{\pi}$, albeit at different time points. The paper could benefit from discussing the conceptual stabilization effect of $\mathcal{R}$ and $\beta$. For example, why not simply use the most up-to-date Bellman-transformed set $\Psi_{\beta}(s, a)$ only?

## Implementation Details

5. **Neural update with MSE loss**: Lines 240–245 describe regressing the network to the EM-updated parameters using an MSE loss, but this step is not accounted for in the convergence analysis. Is there an approximation gap introduced by this?

6. **Constraint formulation**: Eq. 18 converts an episode-level cost limit (problem setting, Eq. 2 and Eq. 7) to a per-step constraint, which can be stricter than the original CMDP constraint. The paper should clarify whether the goal is to enforce stricter per-step limits and justify this choice.

7. **Figure choices**: The "training cost" boxplots (Fig. 5 bottom row) convey coarse trends; line plots with confidence bands might better illustrate progression of the training cost.

8. **$\alpha$ interpretation**: Lines 430-431 claim $\alpha = 0.1$ "achieves precise balance," yet this value yields the lowest rewards. The sweet spot appears to be $\alpha = 0.5$, but I think it could vary significantly from task to task (and safety cost distribution shape). Perhaps the paper could discuss more on this.

9. **Missing detail in ablation**: The $\alpha$ value used in the ablation of Section 4.2.1 is not specified.

### Questions
(related to the Weaknesses above)

1. Could you discuss what example safety tasks demonstrate genuine multimodality?

2. Is the per-step constraint intentional, and how does performance change under episode-level limits?

3. Why is SCVaR preferred over the directly computed mixture-CVaR if both are available from the GMM critic?

### Soundness
3

### Presentation
2

### Contribution
2
