# Achieving $\tilde{O}(1)$ Strong Constraint Violation and Sublinear Strong Regret in Online CMDPs

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
We study safe online reinforcement learning in Constrained Markov Decision Processes (CMDPs) under strong regret and violation metrics. Existing methods that achieve sublinear strong reward regret inevitably incur cumulative strong constraint violation that grows with the number of episodes $T$. To address this limitation, we propose Flexible safety Domain Optimization via Margin-regularized Exploration (FlexDOME), the first algorithm in the literature that provably achieves near-constant $\tilde{O}(1)$ strong constraint violation and ensures a sublinear $\tilde{O}(T^{7/8})$ strong reward regret. FlexDOME, built on the regularized primal-dual framework, introduces a decaying safety margin to the constraint threshold. This margin tightens the feasible region to avoid constraint violation, which relaxes in order $\tilde{O}(t^{-1/8})$ to guarantee feasibility, offering a proper safety-performance trade-off. We then propose a policy-dual divergence potential function that helps establish a non-asymptotic last-iterate convergence guarantee. Experiments demonstrate that FlexDOME significantly enhances safety with negligible reward sacrifice, in full agreement with the theory.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a primal-dual method which employing optimism and regularisation of the dual update attains $\widetilde O(1) $ strong violation e $\widetilde{O}(T^{7/8})$ regret in stochastic CMDPs assuming Slater's condition and knowledge of the Slater parameter. Moreover, the algorithm attains last iterate convergence with respect to the optimal safe policy.

### Strengths
I find the use of the pessimistic $\epsilon_t$ factor in the estimated constraints definition interesting.

### Weaknesses
I have many concerns on the novelty and the significance of the work.

Specifically:

1. It is not clear to me which is the difference between standard episodic CMDP and CMDPs with stochastic thresholds. For instance, assume that standard constraints are of the form $g_t^\top q \leq \alpha_t$, which is exactly the setting studied in this work (I have expanded the expectation using the occupancy measure and I have changed changed $\geq$ to $\leq$ for simplicity, but it is the same). Now, if I want a fixed threshold such as, for example $1/2$, I can rewrite the constraints as $(g_t + \frac{1}{2H}-\frac{\alpha_t}{H})^\top q \leq 1/2$, since the occupancy sums to $H$. Now defining $g^\prime_t=g_t + \frac{1}{2H}-\frac{\alpha_t}{H}$ we have a constraints of there form $g^{\prime\top}_t q \leq \alpha_t$, where $\alpha=1/2$. Thus, we have a reduction from your setting to standard CMDPs.

2. The authors are missing the following reference [1]. In [1], the authors designs a CMDP algorithm (which works with adversarial losses) that achieves constant strong constraint violation and  sublinear $\sqrt{T}$ regret. The regret definition is trivially non-strong since the losses are adversarial. Nonetheless, notice that 1) the strong regret metric makes much less sense that the strong violation one 2) substituting the OMD update with a UCB one the same strong regret can be attained. Thus, I believe that the novelty this work is strongly worsen by [1]. 

3. The regret bound of the work is far from being optimal.

4. The algorithm strongly relies on Muller et al. (2024). Thus, the algorithmic novelty is limited.

Minor: I think there is something wrong in the template. The bottom margin is too large.

[1] Stradi, F. E., Castiglioni, M., Marchesi, A., & Gatti, N. (2025). Learning Adversarial MDPs with Stochastic Hard Constraints. In Forty-Second International Conference on Machine Learning

### Questions
See weaknesses.

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
3

### Summary
This paper proposes FlexDOME, a regularized primal–dual algorithm for online constrained MDPs under strong regret and constraint violation metrics. By introducing a decaying safety margin that tightens and gradually relaxes the constraint thresholds, FlexDOME achieves a near-constant $\tilde{\mathcal O}(1)$ strong constraint violation and sublinear $\tilde{\mathcal O}(T^{7/8})$ strong reward regret, while ensuring last-iterate convergence. Theoretical analyses establish these guarantees, and experiments on tabular CMDPs confirm the method’s superior safety performance compared to baselines.

### Strengths
+ The paper is well written and easy to follow. The technical contribution is clearly presented. To achieve tighter constraint satisfaction, the algorithm introduces a time-decaying safety margin that reduces strong constraint violation from the state-of-the-art $\tilde{\mathcal O}(\sqrt{T})$ to $\tilde{\mathcal O}(1)$, at the expense of higher regret.

### Weaknesses
- The paper is largely an extension of Müller et al. (2024); most algorithmic ideas and analytical techniques are inherited there. 

- The method assumes prior knowledge of the Slater constant ($\gamma$), and the resulting regret bound scales inversely with $\gamma$. In some critical scenarios,  where $\gamma$ is near to zero, the regret may become large. 

- If I understand correctly, the algorithm used TPE for policy evaluation. For sample complexity, it would be more reasonable to take these interactions into account. 

- To achieve constant constraint violation, the algorithm sets the decaying safety margin $\epsilon$ as $\tilde{\mathcal O}(t^{-1/8})$, which leads to higher regret than the state-of-the-art  $\tilde{\mathcal O}(\sqrt{T})$. So could the algorithm be made more adaptive—for example, by adjusting the decay rate dynamically to balance regret and violation?

- The experiments compare only two relatively weak baselines. Given that Table 1 lists stronger alternatives such as (Stradi et al., 2025) and (Zhu et al., 2025), these should be included for a fair and comprehensive evaluation.

### Questions
Please see weakness above.

### Soundness
2

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
4

### Summary
The paper studies online safe reinforcement learning in finite-horizon CMDPs under strong metrics of both regret and constraint violation (i.e., without cross-episode cancellation). It proposes a regularized primal–dual algorithm, FlexDOME, which introduces a decaying safety margin to each constraint threshold and employs entropy and $\ell_2$ regularization to stabilize last-iterate dynamics. With a carefully calibrated schedule for the learning rate, regularization, and margin parameters, the algorithm achieves $\tilde{O}(1)$ cumulative strong constraint violation and sublinear strong regret ($\tilde{O}(T^{7/8})$), along with a non-asymptotic last-iterate convergence guarantee. The analysis follows and extends the recent work of Müller et al., incorporating a vanishing margin mechanism to attain the $\tilde{O}(1)$ strong-violation bound.

### Strengths
- FlexDOME achieves $\tilde O(1)$ cumulative strong violation and $\tilde O(T^{7/8})$ strong regret, with last-iterate convergence---a meaningful milestone for strong safety during learning. 
- The margin-regularized Lagrangian plus entropy/dual regularization induces a well-behaved concave--convex landscape; the potential-function analysis is clean and reusable. 
- The paper is generally well written, and its overall theoretical contribution appears sound. Although I have not verified every technical detail, the employed approaches are standard, and the derivations seem largely correct.

### Weaknesses
- The paper motivates safety in broad terms but does not present a concrete, realistic scenario where strong-metric guarantees, and specifically $\tilde O(1)$ violation during learning, are operationally indispensable, with explicit costs for any violation. 

- The paper overlooks several recent and closely related works that have appeared on adjacent topics.

- The $\tilde O(T^{7/8})$ strong regret is traded for $\tilde O(1)$ strong violation and last-iterate stability. A brief discussion on whether the $7/8$ exponent is tight or proof-artifact (and whether alternative decay could improve it) would help. 

- Treating a stochastic threshold effectively adds an estimated scalar with standard concentration; the contribution feels more like a neat extension than a central novelty. 

- Evaluations are on synthetic tabular CMDPs with one constraint. Including structured environments and sensitivity to schedule constants would strengthen practical relevance.

### Questions
- You regularize with policy entropy $H(\pi)$ and then rewrite over occupancy measures with an entropy $H(q)$. Please justify the exact equivalence.

- Is $t^{-1/8}$ required for $\tilde O(1)$ strong violation with your potential bounds, or could adaptive/data-dependent schedules preserve $\tilde O(1)$ violation while improving the $T^{7/8}$ regret rate? 

- Please expand on realistic application domains where \emph{strong} safety with vanishing cumulative violation is the right operational objective (and why), beyond general safety narratives. 

- Please provide the full MDP specifications used in the appendix (transition kernels, reward/cost functions, threshold distributions, and code). ``Randomly generated'' is insufficient for replication or interpreting oscillations.

### Soundness
3

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
The paper studies online CMDPs under strong metrics with no cancellation for regret and constraint violations and under stochastic thresholds, which constitutes a more general setting than standard CMDPs. The authors propose a regularized primal-dual algorithm with entropy and l2 norm terms, and provide the corresponding strong regret, strong constraint violation, and last-iteration convergence analysis. Experiments on tabular CMDPs show the algorithm keeps lower violation than existing methods.

### Strengths
1. This paper studies a more general and stronger CMDP setting than standard CMDPs, with stochastic thresholds and strong regret/violations.
2. This paper proposes a regularized primal-dual algorithm that can simultaneously achieve constant strong violations, sub-linear regret, and last-iterate convergence.
3. The experiments demonstrate to some extent that the algorithm, with regularizations, is able to tame oscillation while keeping the regret and violations in control.

### Weaknesses
1. The regret $\tilde{O}(T^{7/8})$, though sub-linear, seems still too high, despite the other strong guarantees. If I am understanding this correctly, this strong regret is the consequence of adopting a decay rate of $t^{-1/8}$, which is essential to achieve the theoretical guarantees. If so, what would the lower bound of strong regret be in this case? I would really like to see some discussions and analysis on where the potential sub-optimality (if any) in regret comes from, the regret analysis being too relaxed, or the fundamental problem of this algorithm design?
2. From the last-iterate convergence results, it can be translated that the sample complexity to achieve a $\varepsilon$-optimal policy is $\tilde{O}(\varepsilon^{-7})$, which is too high. Similar arguments and analysis as in Weakness 1 are expected here.

### Questions
Please see weaknesses for my questions. Additionally, I am curious if the authors think these results are improvable, and if so, how would the authors further improve the regret and/or convergence rate?

### Soundness
3

### Presentation
2

### Contribution
3
