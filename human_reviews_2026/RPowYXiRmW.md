# End-to-End Probabilistic Framework for Learning with Hard Constraints

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We present ProbHardE2E, a probabilistic forecasting framework that incorporates hard operational/physical constraints and provides uncertainty quantification. Our methodology uses a novel differentiable probabilistic projection layer (DPPL) that can be combined with a wide range of neural network architectures. DPPL allows the model to learn the system in an end-to-end manner, compared to other approaches where constraints are satisfied either through a post-processing step or at inference. ProbHardE2E optimizes a strictly proper scoring rule, without making any distributional assumptions on the target, which enables it to obtain robust distributional estimates (in contrast to existing approaches that generally optimize likelihood-based objectives, which can be biased by their distributional assumptions and model choices); and it can incorporate a range of non-linear constraints (increasing the power of modeling and flexibility). We apply ProbHardE2E in learning partial differential equations with uncertainty estimates and to probabilistic time-series forecasting, showcasing it as a broadly applicable general framework that connects these seemingly disparate domains.  Our code is available at https://github.com/amazon-science/probharde2e.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents *ProbHardE2E*, an end-to-end probabilistic prediction framework that incorporates hard constraints. The approach first trains an unconstrained probabilistic backbone using the CRPS, chosen for its robustness and existence properties compared to the NLL. The main contribution is the *differentiable probabilistic projection layer (DDPL)*, which projects unconstrained predictions onto constrained probabilistic outputs (mean and covariance) during training and onto corresponding random samples during inference.  

For a location–scale predictive family, the authors derive closed-form solutions of the push-forward through the optimal constrained objective under linear, nonlinear, and convex constraints, allowing sample-free training. The DDPL thus produces uncertainty-aware predictions that satisfy hard constraints. Numerical experiments across time-series and PDE tasks demonstrate improved accuracy and uncertainty metrics over several baselines.

Overall, this is a well-written and valuable paper with a clear contribution, sound theoretical derivations, and strong empirical validation. I have a few main comments that, if clarified, could further strengthen the paper. Given satisfactory responses, I would be willing to raise my score.

### Strengths
The paper provides an interesting combination of probabilistic learning and optimization under hard constraints. To my best judgement, the author have a good grasp of the theory and provide detailed proofs for their theoretical claims. I particularly enjoyed the idea of analyzing the push-forward distribution of the constrained optimization problem and showing ways to derive the anyltical solution for a location-scale family.
The numerical experiments support the author's claims, showing improved performance while fulfilling the hard constraints.

### Weaknesses
I believe, the presentation and readability could be improved, and some details of the method might need further clarification, see questions.

### Questions
## CRPS notation
Equation (2) uses the CRPS from Gneiting & Raftery (2007):  
$$
\mathrm{CRPS}(Y,y)=\tfrac12\mathbb{E}|Y-Y'|-\mathbb{E}|Y-y|.
$$
However, this corresponds to a proper scoring rules that is defined such that $S(Q,Q)\ge S(P,Q)$, i.e., the true distribution maximizes the score. This conflicts with the minimization objectives in (3) and (4).  
Since the closed-form in §3.5 is derived for minimization, as opposed to (2), the sign in (2) should be flipped for consistency. This is likely a notational issue and easily corrected.

---

## Notation in §3.1
The notation in §3.1 is somewhat inconsistent.
You define $\mathcal{Y} \subset \mathbb{R}^{k}$ as the space of predicted distribution parameters,(l. 104), but later (l. 111) use $\mathcal{Y}$ for the space of constrained distribution samples. 
Moreover, ${\mathbf{Y}\_{\theta}} \in \mathbb{R}^{n}$ is introduced as a realization in $\mathbb{R}^{n}$, which is a notation conflict with the parameter space $\mathcal{Y}$. The way I understood it, $\mathbf{Y}\_{\theta}$ is supposed to be a random variable. 

However, with the formulation $\mathbf{Y}_{\theta}(\phi^{(i)})$ it appears like a functional, although it is probably meant as the conditional predictive distribution given $\phi^{(i)}$. Clarity would improve by reserving distinct symbols: e.g., use $\Theta$ for the parameter space, $\Phi$ for network parameters, and define all variables upon first use. This would make it explicit that the network outputs distribution parameters, which then define a predictive measure from which realizations can be sampled.


---

## Framing of log-likelihood and CRPS
Line 19 states that likelihood-based objectives are “heavily biased by their distributional assumptions.” This phrasing feels too strong since the proposed approach also assumes a distributional form (e.g., location-scale or Gaussian), for which the log-likelihood would be an equally valid objective. The CRPS is indeed advantageous, especially under misspecification, but I suggest softening the claim in the abstract and §3.1 or framing it in terms of the observed empirical performance benefits.  
Additionally, it could be interesting to evaluate *ProbHardE2E* with alternative proper scoring rules—such as the energy or kernel scores (Gneiting & Raftery 2007)—as you did for NLL in Q3. Showing that the approach generalizes to various scores would reinforce its flexibility.

---

## Discussion of unique solutions
A brief discussion on the *existence and uniqueness* of the constrained optimization problem (eqs 1 and 7) would enhance completeness.  
If the feasible set is empty, the algorithm naturally terminates; but if the solution is unique, uncertainty vanishes and the probabilistic framework offers no benefit.  
For instance, with linear equalities where $q=n$ and $A$ is full rank, the projection $P_{Q^{-1}}=0$ yields the deterministic solution  
$$
\hat{\mu}=A^{\dagger}b,\quad \Sigma=0,
$$
which is a degenerate distribution and technically not part of the underlying distributional family anymore (maybe depending a bit on the chosen definition). Certainly, in this case uncertainty quantification is not possible and the approach offers no advantage against non-probabilistic solvers. While this is not a limitation of the method, clarifying that *ProbHardE2E* mainly targets under-determined $q<n$ or non-unique settings would improve conceptual precision.
More generally, for any constraint, the transformation $\mathcal{T}$ projects the distribution onto the subspace induced by the constraints and $\Sigma$ does in general not have full rank, i.e. the underlying distribution is degenerate (by design of course). Maybe it would be worth to mention and/or analyze some implication of this, for example that uncertainty cannot be (meaningfully) analyzed across the whole space of the initially learned distribution, but only across the subspace of the transformed distribution.


---

## Minor comments
- $\theta$ appears in l. 106 before being defined.  
- The domains of $\phi^{(i)}$ and $u^{(i)}$ (l. 106) are missing.  
- The spaces of $\xi$ and $z_\theta$ (l. 168–169) should be specified.

---

## Reference
Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* **JASA**, 102 (477), 359–378.  
<https://doi.org/10.1198/016214506000001437>

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors present ProbHardE2E, a probabilistic forecasting framework that incorporates hard operational and physical constraints while providing uncertainty quantification. Their methodology introduces a novel differentiable probabilistic projection layer (DPPL) that can be integrated with a wide range of neural network architectures. The DPPL enables the model to learn the system in an end-to-end manner, in contrast to other approaches where constraints are enforced only through post-processing or during inference.

ProbHardE2E optimizes a strictly proper scoring rule without making any distributional assumptions about the target, allowing it to produce robust distributional estimates. This stands in contrast to existing likelihood-based methods, which are often biased by their distributional assumptions and model choices. Moreover, the framework can incorporate a variety of non-linear constraints, thereby enhancing modeling power and flexibility.

The authors demonstrate the effectiveness of ProbHardE2E in learning partial differential equations with uncertainty estimates and in probabilistic time-series forecasting, establishing it as a broadly applicable general framework that bridges these seemingly disparate domains.

### Strengths
1. The proposed method uses strictly proper scoring rules (e.g., CRPS) instead of log-likelihood objectives, reducing the learning bias caused by incorrect distributional assumptions.
2. The training process can be sample-free, offering potential efficiency advantages.
3. In both time-series forecasting and PDE-solving tasks, the proposed method demonstrates strong empirical performance.

### Weaknesses
1. During inference, the authors indicate that a projection must be computed at every step. Does this imply that the computational cost during inference could be substantial? Could the authors provide an analysis of the time complexity for both training and inference? Similarly, although the training process is sampling-free, repeatedly computing the Jacobian matrix can also increase computational time, especially when dealing with high-dimensional data or scenarios involving multiple constraints.

### Questions
please refer to the weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ProbHardE2E, a unified probabilistic framework for learning under hard constraints with uncertainty quantification. The core module DPPL turns an unconstrained probabilistic predictor (e.g., DeepVAR for time series or VarianceNO for PDEs) into a constraint-satisfying probabilistic model in an end-to-end pipeline. Training is sample-free: the model predicts location–scale parameters $(\mu, \Sigma)$ and DPPL projects these to constrained parameters $(\hat{\mu}, \hat{\Sigma})$ via a delta-method linearization of a projection map $T$, and the paper optimizes CRPS on the constrained distribution. At inference, they sample from the prior and exactly project each sample by solving a constrained least-squares problem to enforce feasibility. The approach aims to handle linear equalities, nonlinear equalities, and convex inequalities within one recipe.

### Strengths
- The method is clearly formulated in a principled “predictor–corrector” view.
- The authors provide an exact handling for linear constraints.
- The training objective is close-formed and sampling free.

### Weaknesses
- First-order DPPL approximation with no error bounds (risk under strong nonlinearity/variance). 
- Loss/evaluation assume independence while the projected posterior is generally correlated. 
- Practical experiments lock $Q$ to diagonal/identity, limiting the benefits of oblique projections and potentially biasing outcomes. 
- Nonlinear equality projection claims “strict feasibility,” but non-convexity and active-set issues are largely handled numerically without theoretical or robustness analysis. 
- Inequality-constraint metrics are thin (CE defined for equalities), so support for those claims is more qualitative than quantitative.

### Questions
1. Can you provide error bounds (or at least empirical error studies) for the delta-method approximation vs exact sampling across varying nonlinearity/variance.
2. Can you report multivariate calibration checks (e.g., energy score/logS), not just diagonal-CRPS, given the fact that you are doing multivariate forecasting, where CRPS is not strictly proper. 
3. Include ablations on $Q$ (identity vs diagonal vs low-rank) with runtime/accuracy trade-offs, since $Q$ defines the projection geometry. 
4. The claim "We demonstrate the importance of using a strictly proper scoring rule for evaluating probabilistic predictions, e.g., the CRPS, rather than negative log-likelihood (NLL)." is completely wrong, NLL is strictly proper.

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
4

### Summary
This paper proposes an end-to-end probabilistic framework that strictly enforces hard constraints during training while producing full predictive distributions (mean + covariance) for uncertainty quantification (UQ). The technical core is the Differentiable Probabilistic Projection Layer (DPPL), which projects an unconstrained probabilistic prediction onto the constraint manifold. For equality/linear constraints, the projection/transformation has closed forms. For nonlinear/convex constraints, the DPPL layer uses Newton/KKT iterations to compute the projection and implicit differentiation (linearized KKT systems) to get the Jacobian for covariance propogation. Then, using CRPS and its closed forms for common local-scale families, it further enables sample-free and end-to-end training, which is faster compared to the sampling-based UQ baselines. The method is shown to imporve forecasting/UQ metrics on several benchmarks in hierarchical time series forecasting and PDE solving problems.

### Strengths
1. The main nolvety of this paper is to propose an end-to-end enforcement of hard constraints, where constraints are optimized as part of learning, not only enforced at post-processing or inference as in convention.

2. It also enables joint UQ and constraint satisfaction, thus suitable for safety/physics/constrained engineering tasks.

3. The sample-free & closed-form CRPS, combined with analytic projection propagations gives substantial training speedups.

4. Generality: the framework covers linear equalities, nonlinear equalities and convex inequalities (with iterative solvers for the nonlinear/convex cases). In addition, it clearly demonstrates how the method works in each scenario with examples and application areas provided.

### Weaknesses
1. The major concern is that the method relies on first-order approximations to linearize the nonlinear function transformation, the KKT system, and the constraints, which together may bring too much estimation errors. Specifically, the covariance propagation relies on a first-order approximation of the Jacobian, thus the UQ under strong nonlinearity constraint can be misestimated. Also the linearized KKT system is only locally valid and can fail to converge or even diverge if the constraint is highly nonlinear or the initial point is far from feasible. 

2. The closed-form sample-free CRPS loss depends on using tractable distribution families (e.g., Gaussian). For multimodal/complex posteriors, it still requires sampling or approximations, thus the advantage of computational efficiency is not guaranteed.

3. In some places, the experimental results are not consistent with the claims made in the paper.

### Questions
1. Is it possible to provide a synthetic analysis of the effect of the first-order approximations on the results, or compute the approximation errors? Have you considered other methods than the linearized KKT, such as Gauss-Newton augmented Lagrangian for nonlinear constraints? The experimental results are not strong enough to support the superior performance of the proposed method (details in the 2nd question below), so maybe relaxing some of the approximations would improve the peformance.
 
2. The paper overclaims its performance based on the experimental results in some places. When analyzing these results,  sometimes it is hard to relate the numbers in the tables to the conclusions stated in the paper. For example, 
   - When addressing Q1, the paper stated "Specifically, when measured against two accuracy metrics across four PDE datasets in Table 2, our method with either oblique (ProbHardE2E-Ob) or orthogonal (ProbHardE2E-Or) projection consistently outperforms both ProbConserv…". However, in Table 2, when comparing the results on Heat, the MSEs for ProbHardE2E-Ob/ProbHardE2E-Or trained with CRPS are higher than ProbConserv trained with CRPS (i.e., 0.036 and 0.031 compared to 0.027); when comparing the results on PME, the MSEs for ProbHardE2E-Ob/ProbHardE2E-Or trained with CRPS are higher than the other methods trained with CRPS (i.e., 9.59 and 9.01 compared to 8.801, 8.187 and 7.945). These results conflict with your statement that your method "consistently outperforms" the other methods. 
   - In Q3, the target is to compare performance using CRPS as training objective to that using NLL, but Table 3 does not provide such experiments or information. I do not see the improvement upon HierE2E addressed Q3, either.
   - In Section 4.2.1, the paper stated "We see an even larger MSE performance improvement of ≈ 15 − 17× when trained on CRPS, and CRPS performance improvement of ≈ 2.5× over the various baselines". This statement only holds true for the case when $m\in[2,3]$, but not for the other two cases. 
   - In Figure 1(b), it seems to me that the uncertainty estimates from the three methods are roughly the same (maybe the blue region is a bit wider than the others; it is hard to tell the difference between the red and purple regions). In Figure 2\(c\), it seems that all the three curves perform roughly the same, none of them captured the ground truth curve.
 
3. Typos/suggested corrections in the manuscript:
   - In Algorithm 1, it would be more clear if you indicate what $\mathbf{Y}_\theta(\phi)$ and $\mathbf{Z}_\theta(\phi)$ represent (constrained/unconstrained random variable), even though you have explained them in the main text. In addition, I think at the end of step 7, "where $z_\theta(\phi)\sim\mathbf{Z}_\theta(\phi)$" should be moved to the end of step 8.
   - The paper frequently mentioned "Problem 7" while I think it would be better to use "Problem (7)" as it appears. 
   - In line 322-323 on page 6, the classical statistical approaches should be listed separately instead of being listed under "(ii) HierE2E".
   - Be consistent with using "HierE2E" and "Hier-E2E" (the paper is using both).
   - In line 424, "approximating" should be "approximately".
   - In lines 1327-1332, $\nabla L_{\hat{u}}$ should be $\nabla_{\hat{u}}L$.
   - In line 1448, in the last vector, $-J^{(*)}$ should be $J^{*}$.
   - In Eq. (44), add definition of $P_I=I-A^{\top}(AA^{\top})A$ (as you mentioned $P_I$ later but it was undefined).

### Soundness
3

### Presentation
4

### Contribution
3
