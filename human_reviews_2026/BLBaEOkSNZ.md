# Bregman Geometry for Stochastic Online Bilevel Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We study *online bilevel optimization (OBO)* in the *stochastic* setting and ask whether geometry can eliminate the severe dependence on the condition number of the inner problem, $\kappa_g = \ell_{g,1}/\mu_g$. We introduce a family of *Bregman-based algorithms* and analyze both oracle and practical regimes. In the oracle setting, where exact hypergradients are available, generalized Bregman steps achieve sublinear bilevel local regret (i.e., $o(T)$) while *removing the cubic dependence on $\kappa_g$* incurred by Euclidean updates. In the practical stochastic setting, where hypergradients must be estimated, we design single-loop, sample-efficient algorithms that combine Bregman steps with time-smoothed hypergradient estimates. Our analysis shows that Bregman geometry again eliminates the $\kappa_g$-dependence and yields guarantees of sublinear bilevel local regret in this setting. It further reveals a broader insight: time smoothing, previously treated as a heuristic in deterministic OBO, naturally functions as a *variance-reduction mechanism* while keeping bias controlled, clarifying its role across both regimes.  Finally, experiments on preconditioner learning and reinforcement learning support our theoretical findings across a variety of nonstationary loss sequences and large-scale, ill-conditioned datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Bregman geometry into stochastic online bilevel optimization (OBO) and proposes single‑loop algorithms with time‑smoothed hypergradient estimators (SOBBO / SOBOW). It proves local dynamic regret bounds that remove inner‑level condition‑number dependence, and interprets time smoothing as an intrinsic variance–bias trade‑off. Experiments on online hyperparameter selection and non‑stationary Actor–Critic RL show lower regret and better validation loss than OAGD/OBBO/SOBOW baselines.

### Strengths
1. Tightly aligned theory and mechanism. The chain “time smoothing → variance reduction → regret improvement” is rigorously tied together by inequalities and sums.
2. Single‑loop & sample efficiency. Avoids nested loops, suitable for online/streaming.
3. Stable empirical signals. Multiple figures demonstrate how increasing $w$ lowers variance and regret; the RL setting remains favorable.

### Weaknesses
1. Constants and tuning visibility. Many constants appear in the bounds; provide practical estimation heuristics or rules of thumb.
2. Broader baselines. Add meta‑learning/implicit differentiation/accelerated hyper‑optim baselines, especially in non‑stationary setups.
3. Compute–window trade‑offs. Larger $w$ incurs extra history and compute; please report wall‑clock/throughput alongside regret.
4. Scale & diversity. Current tasks are modest; larger models/longer online horizons/more complex non‑stationary distributions would improve external validity.

### Questions
1. Adaptive windows. Can $w$ be adapted to noise level/time without breaking the bounds?
2. Coupling of inner $K$ and $w$. Joint tuning guidance for convergence/regret?
3. Scope of condition‑number independence. Which non‑smooth or non‑strongly‑convex inner problems preserve the guarantees?

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
This paper addresses stochastic online bilevel optimization (OBO) by introducing Bregman-based algorithms that eliminate severe dependence on the inner problem's condition number ($κ_g$).
Main Contributions:

Bregman Geometry for Improved Conditioning: The authors demonstrate that Bregman gradient steps achieve sublinear bilevel local regret ($o(T)$) while removing the $κ_g³$ dependence in the oracle setting and $κ_g⁵$ dependence in the stochastic setting, compared to standard Euclidean updates.
Single-Loop Stochastic Algorithms: Two practical algorithms are proposed:

SOBO (Euclidean): achieves sublinear regret but retains $κ_g⁵$ dependence
SOBBO (Bregman): combines time-smoothed hypergradients with Bregman steps, eliminating $κ_g⁵$ dependence while maintaining sublinear regret

Time Smoothing as Variance Reduction: The analysis reveals that time smoothing, previously used heuristically in deterministic OBO, naturally functions as a variance-reduction mechanism with controlled bias in stochastic settings.
Experimental Validation: Results on preconditioner learning and reinforcement learning tasks demonstrate superior performance over existing online bilevel baselines on nonstationary, ill-conditioned datasets.

The framework provides both oracle and practical stochastic guarantees, offering theoretical insights and practical improvements for large-scale bilevel optimization problems.

### Strengths
This paper demonstrates exceptional academic writing ability. The authors excel particularly in narrative structure, with a clear and coherent logical thread throughout the entire article. The introduction explicitly identifies two critical gaps in existing online bilevel optimization methods, then explains why investigating this problem is necessary. Through comparative tables (Table 1 and Table 2), the authors intuitively illustrate the limitations of existing methods, especially the dependence on the condition number $\kappa_g$.

### Weaknesses
1.The paper focuses on the issues of Bregman geometry and condition number dependence in "Online Bilevel Optimization (OBO)," which are theoretically challenging. However, the necessity of these issues has not been sufficiently validated in practical applications. The authors do not elaborate on the fact that the demand for online settings in bilevel optimization is not strong in most current machine learning tasks, especially in scenarios that require frequent updates to the outer layer parameters and where the inner problem is highly ill-conditioned, which are still relatively rare.

2. In the related work section of the Introduction, the authors could provide a more detailed description. 

3. Section 6 only includes two experimental scenarios; it is recommended to add one more experimental scenario.

### Questions
1. The theoretical step size requirements are rather intricate, depending on numerous unknown constants including $l_F,1, l_g,1,$ among others. How were these step sizes determined in the experimental settings? 
2. While the paper emphasizes the elimination of $\kappa_g$ dependence, the Regret bound in Theorem 6 still retains $\kappa_g$​ terms. Corollary 6.1 indicates that the Bregman parameter must be selected as $\rho = O(\kappa_g^3)$ to remove the $\kappa_g$ dependence. What specific Bregman reference function was employed in the numerical experiments? It would be helpful if the paper provided experimental evidence to justify the necessity of removing $\kappa_g$ dependence. Specifically, what are the typical values of $\kappa_g$ in the tested problems, and how much does the improved dependence translate to practical performance gains? 
3. Before introducing the algorithm formally, could the authors clarify the design logic more explicitly? Specifically: Which are adapted from prior work? Which components are novel contributions? What motivated each of these design choices?
4. Section 6 presents only performance metrics such as regret, but includes no comparison of running times.

### Soundness
3

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
5

### Summary
This paper studies online bilevel optimization in the stochastic setting and proposes using Bregman geometry to eliminate the severe dependence on the condition number of the inner problem.

The authors introduce a single-loop, sample-efficient algorithm that combines Bregman steps with time-smoothed hypergradient estimates. The main theoretical contribution is achieving sublinear bilevel local regret $o(T)$ while exhibiting improved dependence on the condition number. 

Experiments on preconditioner learning and reinforcement learning validate the theoretical results.

### Strengths
The paper studies online bilevel optimization to the stochastic setting with Bregman geometry. The motivation is well-articulated through two concrete gaps: (i) lack of geometry-aware updates for ill-conditioned problems where the inner condition number κ_g is large, and (ii) inability of existing deterministic methods (OAGD, SOBOW) to handle stochastic gradients and nonstationary data streams common in large-scale learning. 


The paper provides comprehensive theoretical analysis for both oracle and stochastic settings. Theorem 3 establishes that Bregman steps achieve sublinear regret $o(T)$ without the $\kappa_g^3$ factor present in Euclidean methods. The main result (Theorem 17) shows SOBBO achieves sublinear regret with single-loop efficiency. 


The experiments demonstrate practical benefits: on GDSC preconditioner learning, SOBBO achieves  lower regret than baselines. The actor-critic RL experiments (Pendulum with scheduled environment changes) validate the approach under nonstationarity. Figure 1 (middle/right) empirically confirms theoretical predictions that increasing window size w reduces both regret and hypergradient variance.

### Weaknesses
The paper has mathematical issues that undermine its soundness:

Theorem 12, Page 13): The proof states “Choosing $\alpha = 1/\ell_{F,1}$” but then claims selecting $\rho = O(\ell_{F,1})$ eliminates $\kappa_g$ dependence. This is circular reasoning: since $\ell_{F,1} = O(\kappa_g^3)$ (Lemma 2), choosing $\rho = O(\kappa_g^3)$ doesn't eliminate the dependence—it just hides it in the Bregman divergence. The computational cost of the Bregman step scales with $\rho$.


Lemma 14): The constant $C_{\mu_g} := 1 + \frac{\ell_{g,1} + \mu_g}{\eta \ell_{g,1}\mu_g}$ becomes unbounded as $\eta \to 0$. The lemma statement requires $0 < \eta \leq \frac{2}{\ell_{g,1} + \mu_g}$ but provides no lower bound. For $C_{\mu_g} = O(1)$, you need $\eta = \Omega(1/\mu_g)$, which should be stated.

Lemma 14 proof): After applying Young’s inequality with arbitrary $\delta > 0$, the proof sets $\delta = \frac{\eta\mu_g}{2}$. But this requires:
$
(1 + \delta)\left(1 - \frac{2\eta\mu_g}{\ell_{g,1} + \mu_g}\right)^K < 1
$
which imposes constraints on $\eta$ and $K$ not established in the lemma statement.

Presentation issues:

1. Equation (2): Fix spacing in the inner problem formulation to improve readability.
2.  There are issues with inconsistent use of `\citep` and `\citet` throughout the text. 
3. Table 1: The notation `O(...)` should be replaced with $\mathcal{O}(...)$ to follow standard mathematical convention.

4. Lines 108–110: Inconsistent use of $\nabla g_t(\lambda, \beta, \zeta)$ vs. $\nabla g_t(\lambda, \beta)$. It's unclear when stochastic vs. deterministic gradients are being used.
5. Algorithm 3:  Inconsistent notation between $\bar{\zeta}_{t,k}$ and $\zeta_{t,k}$.
6. Table 2: Ambiguous formatting — it's unclear whether $\sigma_f^2 + \sigma_g^2$ is inside or outside the $\mathcal{O}(\cdot)$ notation.
7. Equation (7): The random product term involving $\tilde{m} \sim U(0,1,\ldots,m-1)$ is unclear when $\tilde{m} = 0$. This should be clarified (e.g., product equals identity in that case).

### Questions
In addition to the soundness issues, 

Q1) Is the improvement in $\kappa_g$ dependence primarily due to the use of the Bregman divergence, or does it stem from your specific algorithmic design? More generally, does the use of Bregman divergence improve condition number dependence in bilevel optimization? If so, could you cite relevant prior work?

Q2) Can the improvement in condition number dependence be empirically validated in your experiments as well?

Q3) In the experiments, could you provide the values of $\kappa_g$, the specific Bregman function $\varphi(\lambda)$, and all relevant hyperparameters ($K$, $\alpha$, $\eta$, $s$, $m$, $w$)? Also, please include error bars and baseline comparisons.

### Soundness
3

### Presentation
2

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
This paper investigates the problem of online bilevel optimization in both deterministic and stochastic settings. The main contribution is the proposal of a mirror descent type update scheme for solving the problem. The authors show that the regret bound of the proposed method, defined with respect to the mirror descent type update, is independent of the inner condition number $\kappa$. Extensive experiments are conducted to validate the effectiveness of the proposed approach.

### Strengths
+ The proposed method is independent of the condition number $\kappa_g$.
+ The paper is well structured and clearly written in most parts.
+ Experiments are conducted to validate the effectiveness of the proposed method.

### Weaknesses
- One of my main concerns is whether the improvement from removing $\kappa_g$ comes from algorithmic innovation or primarily from a change in the performance measure. In Theorem 6 and Corollary 6, the method does not impose specific requirements on $\phi$ beyond requiring it to be a $\rho$-strongly convex function with sufficiently large $\rho = O(\ell_{F,1})$. In this case, choosing $\phi = \frac{\ell_{F,1}}{2}\Vert \cdot \Vert_2^2$ also appears valid for the proposed method. However, under this choice, the update rule becomes almost identical to online gradient descent, differing mainly in the step size. From this perspective, the observed improvement from removing $\kappa_g$ seems to arise from a change in the performance measure rather than from substantive algorithmic advancement. As also discussed in Corollary 3.1, removing $\kappa_g$ appears difficult when performance is evaluated using the Euclidean norm.

- The significance of obtaining a bound that is independent of $\kappa_g$ is unclear. The paper would be more compelling if it provided concrete examples where $\kappa_g$ is prohibitively large or practically problematic.

- The experiments do not specify which Bregman divergence $\phi$ is used. It would help to state the exact choice of $\phi$ for each experiment.

- The proposed methods appear to require detailed knowledge of Lipschitz constants, and it is unclear how to reduce or remove this dependence in practice.


**Minor Issues**

- Line 131: Please specify with respect to which variable the function is Lipschitz continuous.  
- Line 173: I believe an additional condition should be $\phi(x) = \frac{1}{2}\|x\|_2^2$.  
- Line 184: There is a repeated “=”.  
- Algorithm 3, line 4:* The word “to” is missing.

### Questions
Please refer to the questions listed in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
