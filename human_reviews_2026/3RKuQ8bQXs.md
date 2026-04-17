# Online Learning of Whittle Indices for Restless Bandits with Non-Stationary Transition Kernels

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
We study optimal resource allocation in restless multi-armed bandits (RMABs) under unknown and non-stationary dynamics. Solving RMABs optimally is PSPACE-hard even with full knowledge of model parameters, and while the Whittle index policy offers asymptotic optimality with low computational cost, it requires access to stationary transition kernels - an unrealistic assumption in many applications. To address this challenge, we propose a Sliding-Window Online Whittle (SW-Whittle) policy that remains computationally efficient while adapting to time-varying kernels. Our algorithm achieves a dynamic regret of $\tilde O(T^{2/3}\tilde V^{1/3}+T^{4/5})$ for large RMABs, where $T$ is the number of episodes and $\tilde V$ is the total variation distance between consecutive transition kernels. Importantly, we handle the challenging case where the variation budget is unknown in advance by combining a Bandit-over-Bandit framework with our sliding-window design. Window lengths are tuned online as a function of the estimated variation, while Whittle indices are computed via an upper-confidence-bound of the estimated transition kernels and a bilinear optimization routine. Numerical experiments demonstrate that our algorithm consistently outperforms baselines, achieving the lowest cumulative regret across a range of non-stationary environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work studies a non-stationary variant of the restless multi-armed bandit (RMAB) problem where (unknown) transition dynamics change over time. They adapt standard non-stationary sliding-window and bandit-over-bandit meta-algorithm approaches on the Whittle index-based policy for RMAB to get dynamic regret bounds scaling with the total variation in changes of the transition matrices.

### Strengths
* The paper provides the first dynamic regret bounds for the non-stationary RMAB setting.
* The presentation is clear and easy to follow.
* There are simulation results validating the theoretical results and showing superior regret of their algorithm over other arts.

### Weaknesses
* The work [1] also seems to establish dynamic regret bounds for non-stationary RMAB, albeit with larger scaling in terms of $T$ and variation budget. It would be good to discuss the differences in approaches and results.
* It is unclear to me if the dynamic regret bound shown in this work $O(T^{2/3} \tilde{V}^{1/3} + T^{4/5})$ is optimal. In particular, is the $T^{4/5}$ term an artifact of the analysis or in fact fundamental? 
* Furthermore, it is unclear if the right dependence on other problem-dependent parameters such as $H,P_{\min}$ and the number of states has been captured optimally in the regret bounds as compared to the known stationary regret bounds for RMAB.
* The techniques used in this work (sliding-window and bandit-over-bandit) are fairly standard in works on non-stationary bandits and so the technical novelty of the submission is not very high I feel. Could the authors elaborate on what particular proof challenges or hurdles were overcome for this setting?

[1] Non-Stationary Restless Multi-Armed Bandits with Provable Guarantee. Hung et al., 2025. https://arxiv.org/pdf/2508.10804

### Questions
Please see weaknesses above.

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
3

### Summary
The paper studies restless multi-armed bandits (RMABs) with unknown, time-varying transition kernels and proposes a sliding-window online Whittle policy. Each episode: (i) predict a per-arm variation budget $V_n$
 via a Bandit-over-Bandit (EXP3) scheme; (ii) set window size $w_n$; (iii) build a confidence set for transitions over the last $w_n$ episodes; (iv) pick an optimistic transition kernel $\tilde P_{n,t}$ within the confidence set that maximizes the arm’s value (thus upper-bounding the true value); (v) compute Whittle indices under $\tilde P_{n,t}$ and activate the top-$M$ arms; (vi) update the Lagrange multiplier $\lambda$ to the $M$-th largest index.

### Strengths
1. Timely problem \& clear algorithmic design:
The paper tackles online RMABs with non-stationary kernels—highly relevant in networking, recommendation, and health monitoring—and adapts Whittle policies via sliding windows + optimism + BoB for $V_n$.
The step-by-step procedure is explicit and implementable.

2. Non-stationarity modeling via variation budgets:
Using a per-arm total variation metric aligns with dynamic regret literature and naturally drives the windowing trade-off.

3. Confidence-set optimism specialized to RMABs:
The optimistic kernel selection within a high-dimensional ball, then computing Whittle indices on $\tilde P_{n,t}$, is principled and lets the value function upper-bound enable regret control.

4. S.4. Regret decomposition that exposes the right knobs:
The bound shows how (i) estimation error decreases with $w_n$, (ii) prediction/drift error grows with $\tilde V_n$ and the BoB granularity $J_n$, and (iii) the Lagrangian-relaxation gap contributes $h(N)T$—clarifying where hardness comes from and when sublinearity holds.

5. Sparsity awareness:
If the decision maker knows sets $S_0(s,a)$ of impossible transitions, the confidence region shrinks and the bound improves; yet the method still guarantees sublinear regret even without sparsity.

6. Empirical evidence across two RMABs.
Against strong baselines, the method consistently achieves lower regret, with a transparent evaluation protocol (oracle–Whittle comparator).

### Weaknesses
1. Additive linear term $h(N)T$ can dominate: The dynamic-regret bound contains an additive term $h(N)T$, where $h(N)$ measures the inherent performance gap between the Lagrangian (decoupled) problem and the original coupled RMAB under the same kernels, hence independent of learning error. If $h(N)\neq 0$ (e.g., finite $N$, no global-attractor structure), the linear term dominates for large $T$, nullifying the sublinear part. The algorithm may learn quickly under drift, but if Whittle is structurally sub-optimal in the domain, guarantees are ultimately bottlenecked by $h(N)T$. Hence, a discussion is needed when $h(N)\approx 0$.

2. Computational burden of inner ``optimistic kernel'' maximization (Eq. 12):
Per arm/episode, solves Eq. 12-13 over a high-dimensional confidence polytope $\mathcal B_t^{(n)}$. Outside special 1-D structure, there is no closed form, and the paper itself flags computational intensity. Therefore, without structure, per-episode overhead can limit scalability for large $|S|$ or $N$, creating a gap between theory and deployment.



3. Evaluation scope is narrow:} Experiments run for $T=50$ episodes and rely on domain simplifications to solve Eq.~(12). This under-stresses the asymptotic trade-offs exposed by the theory and does not report wall-clock scaling. Stronger ablations (vary $J_n$, $w_n$, misspecify $S_0$) and runtime plots would better substantiate the method's practicality.

### Questions
Q1: Can you empirically probe regimes where $h(N)$ is non-negligible (finite $N$, no global-attractor), showing the linear term's impact?

Q2: How sensitive is performance to misspecified $S_0(s,a)$? Could one learn $S_0$ online with penalties?

Q3: Any simple exploration schedule to ensure $P_{\min}$ does not collapse when $M\ll N\$?


Q4: Please include runtime scaling (vs. $|S|,N$) for Eq.~(12) in both domains.

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
This paper studies online learning in restless multi-armed bandits (RMABs) where transition kernels are unknown and non-stationary. The authors propose a Sliding-Window Online Whittle (SW-Whittle) algorithm. This method integrates Whittle index computation with a sliding-window and bandit-over-bandit framework to adapt to changing dynamics. They prove a dynamic regret bound and validate their method on two simulated RMAB problems (wireless scheduling and a one-dimensional bandit), showing improved performance over UCWhittle and other baselines.

### Strengths
+ Clear motivation for addressing non-stationary RMABs.
+ Novel algorithm design combining Whittle index, UCB, and sliding-window learning.
+ Strong theoretical guarantees with explicit regret scaling.
+ Experiments show consistent improvement over prior algorithms.
+ Discussion of sparsity in transition kernels adds realistic motivation

### Weaknesses
- The empirical evaluation is limited to small synthetic environments. Real-world data or larger-scale experiments would improve credibility.
- The algorithm’s computational complexity (especially in solving the bilinear optimization in Eq. 12) is acknowledged but not analyzed quantitatively.
- Some assumptions, such as known sparsity structure (assuming indexability is given), may not always hold in practice.

### Questions
1. How sensitive is performance to the quantization level in the bandit-over-bandit step?
2. Could the method extend to continuous or infinite state spaces via function approximation?
3. Can the approach handle partial observability or contextual features?
4. How does computation time scale with the number of arms and states?

### Soundness
3

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
This paper addresses the challenge of resource allocation in restless multi-armed bandits (RMABs) where transition kernels are unknown and non-stationary. The authors propose the Sliding-Window Online Whittle (SW-Whittle) policy, which adapts to time-varying dynamics by using sliding windows for kernel estimation, upper-confidence bounds, and a Bandit-over-Bandit framework to handle unknown variation budgets. Key contributions include the algorithm design, a dynamic regret bound of $\tilde{O}(T^{2/3} \tilde{V}^{1/3} + T^{4/5})$, and numerical simulations demonstrating superior performance over baselines like UCWhittle and WIQL.

### Strengths
- **First work to tackle Non-stationarity in RMABs**: This is the first work to provide dynamic regret bounds for online learning of Whittle indices in non-stationary RMABs, extending stationary methods like UCWhittle (Wang et al., 2023) and addressing a gap in the literature where prior non-stationary RMAB works.

- **Rigorous theoretical analysis**: The dynamic regret bound $\tilde{O}(T^{2/3} \tilde{V}^{1/3} + T^{4/5})$ is well-derived, improving on stationary regrets and directly analyzing the main problem rather than Lagrangian relaxations, as in Wang et al. (2023).

- **Novelty on using Bandit-over-bandit approach**: The algorithm handles unknown variation budgets via Bandit-over-Bandit and exploits kernel sparsity, making it adaptable to real-world sparsity in applications like Age of Information in wireless scheduling.

- **Clear presentation and detailed literature review**: The paper is well-structured, with a detailed literature review and algorithm descriptions, and proofs in appendices.

### Weaknesses
- **Limited experimental validation with synthetic data on a toy setup**: While simulations show lower regret, they rely on synthetic environments; real-world datasets for RMABs exist (e.g., ARMMAN dataset), which could provide more convincing evidence of applicability. This ties into the lack of mention of specific real-world datasets or problems with this non-stationarity modeling, despite bandits' alignment with practical issues such as user preferences drifting over time.

- **Suboptimality gap of Lagrangian relaxation with non-stationarity**: The suboptimality from Lagrangian relaxation (with gap $h(N) \to 0$ as $N \to \infty$) and its interaction with non-stationarity is not thoroughly analyzed/explained. Changing kernels could shift the "goalpost" for large $N$ approximations, yet this is not explicitly addressed.

- **Implications of the non-stationarity model**: The model bounds per-episode changes by $V_n/T$, implying total variation $V_n$, where for sublinear regret $V_n = o(T)$ and average non-stationarity diminishes as $T \to \infty$; this assumes environments that stabilize over time. Was this intentional? The lack of real-world examples tied with the modeling assumptions is making it hard to justify what can be allowed and what is a stretch.

### Questions
- **Choice of $w_n$**: The window size $w_n = \lceil (T/V_n)^{2/3} \rceil$ appears constructed specifically to achieve the regret bound. Any more discussion on where and why this would be the correct choice?

- **Bandit-over-bandit approach**: While this increases the novelty of the overall work, this hasn't been discussed much and not touched much in experiments as well. Can you expand on the results and insights surrounding it more?

- **Impact of Sparsity**: Apologies if I missed this, but after being touched upon, the role of sparsity and its impact on the final results and experiments seem to be undermined. Can you expand on this as well?

- Is the $T^{4/5}$ term in the regret bound necessarily optimal? What is the scope of improvement? Additionally, what real-world problems fit the $V_n/T$ diminishing non-stationarity model?

### Soundness
3

### Presentation
3

### Contribution
2
