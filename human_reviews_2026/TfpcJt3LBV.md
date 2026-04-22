# Quantifying Cross-Domain Knowledge Distillation in the Presence of Domain shift

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Cross-domain knowledge distillation often suffers from domain shift. Although domain adaptation methods have shown strong empirical success in addressing this issue, their theoretical foundations remain underdeveloped. In this paper, we study knowledge distillation in a teacher–student framework for regularized linear regression and derive high-dimensional asymptotic excess risk for the student estimator, accounting for both covariate shift and model shift. This asymptotic analysis enables a precise characterization of the performance gain in cross-domain knowledge distillation. Our results demonstrate that, even under substantial shifts between the source and target domains, it remains feasible to identify an imitation parameter for which the student model outperforms the student-only baseline. Moreover, we show that the student's generalization performance exhibits the double descent phenomenon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper builds upon high-dimensional random matrix theory to analytically study cross-domain knowledge distillation (KD) in teacher–student linear regression settings under both covariate and model shifts. It derives asymptotic expressions for the excess risk through bias–variance decomposition, covering both regularized (ridge) and unregularized (ridgeless) regimes with deterministic and random parameters. The analysis includes an exploration of the imitation parameter $\xi$, showing that the optimal $\xi$ may lie outside the interval [0,1] (even negative), and discusses the conditions under which double descent arises.

### Strengths
1. Clear, testable theoretical claims: The paper presents three main results that express the excess risk of cross-domain KD as explicit matrix functions involving Stieltjes transforms. The derivations are mathematically sound and well-grounded in the theory of random matrices.
2. Beyond single-domain baselines: The work demonstrates the existence of an imitation coefficient $\xi$ that allows the student model to outperform pure supervised learning (Proposition 1).
3. Unified explanation of observed phenomena: The analysis connects double descent behavior in KD with the interplay among $\xi$, teacher/student regularization, and the dimensional ratio, offering a coherent theoretical interpretation.

### Weaknesses
1. Limited practical applicability beyond linear models: All results are derived for linear regression; implications for nonlinear or deep KD remain speculative. A discussion or small-scale experiment validating the theoretical predictions in nonlinear settings would strengthen the paper.
2. Empirical validation is light: The experiments are limited to synthetic data, verifying asymptotic formulas.
3. Strong assumptions and limited robustness analysis: Theoretical results rely on independence, bounded moments, and concentration assumptions.

### Questions
See Weakness.

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
4

### Summary
The paper analyzes cross-domain KD for linear regression with ridge (and ridgeless) estimators. A source-domain teacher (\beta_t) is trained on ((X_1,y_1)) and used to supervise a target-domain student on ((X_2,y_2)) via the imitation parameter (\xi) in the objective (L(\xi)=\xi \ell(y_2^t,y_2^s)+(1-\xi)\ell(y_2,y_2^s)). Closed-form resolvent-based expressions yield high-dimensional (HD) bias–variance formulas for the excess target risk (ER(\beta_s)) under covariate shift ((\Sigma_1\neq\Sigma_2)) and model shift ((\beta_1\neq \beta_2)). The paper also treats random-(\beta) and an under-parameterized ridgeless regime. Key theorems (1–3) give deterministic equivalents that expose dependence on (\Sigma_1,\Sigma_2,\beta_1,\beta_2), and on (\xi), and show conditions where the student beats the student-only baseline; (\xi) can even be negative (“anti-learning”).

### Strengths
1. In the under-parametrized, ridgeless setting you show the student is a convex combination of the two OLS estimators and give a closed-form (ER) with optimal (\xi\in(0,1)). This is intuitive and useful. 
 **Allowing (\xi\in\mathbb{R}).** You justify that negative (\xi) can be optimal (isotropic corollary), and you prove existence of a strictly better (\xi) than the student-only baseline under natural conditions.  

2. The paper formalizes (ER=\text{Bias}+\text{Var}) and then supplies deterministic equivalents for both parts under deterministic ((\beta_1,\beta_2)) and random (\beta) models (Theorems 1–2).

### Weaknesses
1. Because both squared losses share the same quadratic term in (\beta), the Hessian of (L(\xi)) is (N_2^{-1}X_2X_2^\top+\lambda_s I), **independent of (\xi)**; hence the problem is convex and well-posed for any real (\xi). It would help readers to add a short lemma right after (1) and your closed form for (\beta_s), making the “negative (\xi) is still safe” point explicit. 

2. Expressions like Theorem 1’s Bias/Var have (o_{\text{a.s.}}(1)) terms while the main terms scale with traces ((\Theta(M)) in isotropic cases). Please specify whether your (o_{\text{a.s.}}(1)) is *absolute* or *per-dimension*, and consider normalizing (ER/M) in statements to make asymptotic orders unambiguous. 

3.  Assumption 1(a) currently asks for all moments; you note it can be relaxed. Give a concrete bound ((4+\epsilon) or (8) moments) sufficient for Lemma 6 / local laws used later, so readers know what’s truly required. 

4. Theorem 1 already shows dependence on (\Sigma_1,\Sigma_2) via (\Pi_1,\Pi_2,S_i(\cdot)). It would help to rewrite one or two key trace terms (e.g., ( \mathrm{Tr},[\Pi_1 \Pi_2 \Sigma_2])) in the eigen-bases of (\Sigma_1,\Sigma_2), or to use an overlap matrix to highlight principal-angle effects. You partly do this in App. B.6 (eq. (28)); elevating a compact “eigenvector-overlap” corollary to the main text would greatly aid intuition.  

5. You note (and Proposition 1 leverages) that (ER(\beta_s)) is a convex quadratic in (\xi). Please collect coefficients (A,B,C) in closed form (from Theorems 1–2) and give a short table for common regimes (isotropic; shared (\beta); pure model shift). This immediately yields (\xi^\star!=!-B/2A) and clarifies when (\xi^\star<0) or (\xi^\star>1) without case-by-case reasoning.   

6. Your analysis assumes/advocates Bayes-consistent teacher probabilities and studies how teacher quality controls the SGD variance term. Ye et al.[1] propose BCDE, a teacher-training objective based on conditional mutual information explicitly aimed at estimating the Bayes conditional distribution for KD. It’s a natural methodological precursor/neighbor to your “Bayesian teacher” prescription and directly relevant to your noise model and guidelines.

[1] Ye, Linfeng, et al. “Bayes Conditional Distribution Estimation for Knowledge Distillation Based on Conditional Mutual Information.” ICLR 2024 (Twelfth).

### Questions
see the weakness above

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
2

### Summary
This paper studies cross-domain knowledge distillation with domain shift through a teacher-student framework. Authors leverage Random Matrix Theory and present a theoretical analysis in the context of linear regression, comprising a deterministic-parameter setting where the teacher and student parameters are non-random and a random-parameter setting where a shared parameter vector is drawn from prior distributions. This paper discovers that the knowledge distillation still works even under substantial domain discrepancies. Authors also observe a double-descent phenomenon in the knowledge distillation process.

### Strengths
- The overall theoretical analysis is clear and well-organized. Authors use tools from Random Matrix Theory and derive precise, high-dimensional asymptotic expressions for the excess risk. The analysis is mathematically sound.
- Interesting phenomenon that knowledge distillation is still possible when the source domain and target domain share substantial domain discrepancies. The Anti-Learning discovery is also insightful, which points out that the best imitation parameter $\xi$ is not limited to the [0, 1] range.

### Weaknesses
- All the derivations in this paper only work for linear regression. In the real world, complex, non-linear models like deep neural networks are much more popular. Therefore, it's unknown whether these insights remain applicable in practice and how much they can provide guidance for Knowledge Distillation.
- Lack of Quantification for the extent of the Substantial Shift. The paper claims efficacy even under substantial domain discrepancies, but the degree of "substantial" is not well quantified. The analysis shows that an optimal $\xi$ exists such that $\text{ER}(\beta_s) < \text{ER}_0$, but how does this performance gain degrade as a function of the domain shift? A potential weakness is that the paper does not introduce an explicit metric to quantify domain shift. The discrepancy between domains is only implicitly reflected through covariance geometry.
- The finding that $\xi < 0$ (anti-learning) can be optimal (Corollary 2) is intriguing but seems rather uncommon in practical scenarios. Does this imply that the teacher performs so poorly that the student benefits from learning the opposite of the guidance? A more in-depth discussion on the underlying reasons for this phenomenon and its practical implications would further strengthen the paper.

### Questions
Please see the weakness.

### Soundness
2

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
2

### Summary
This paper presents a theoretical framework for \textbf{cross-domain knowledge distillation (KD)} under both \emph{covariate} and \emph{model shifts} in a teacher--student ridge regression setting. 
Using tools from \textbf{random matrix theory}, it derives high-dimensional asymptotic expressions for the student's excess risk via bias--variance decomposition. 
The analysis shows that even under substantial domain shift, there exists an optimal imitation parameter $\xi$ such that the student model outperforms the student-only baseline, and the generalization risk exhibits a clear \textbf{double descent} behavior.

### Strengths
The paper provides a rigorous theoretical analysis of cross-domain knowledge distillation (KD) using random matrix theory, deriving precise high-dimensional asymptotic characterizations that extend previous student-only or fixed-ξ formulations. It analytically demonstrates that an appropriately chosen imitation parameter ξenables the student model to outperform the baseline even under significant domain shifts. Moreover, simulation results closely align with the theoretical predictions, validating the framework and revealing a clear double-descent behavior in the student’s excess risk.

### Weaknesses
1. The theoretical analysis is limited to linear ridge regression.   
2. Relies on bounded spectral norms, independence between domains, and high-moment conditions, potentially unrealistic for real-world KD settings.  
3. The ridgeless regression analysis only covers the under-parameterized case (M<N_1,N_2).  
4. The dependence of the optimal imitation parameter ξ and the interaction between λ_tand λ_s lacks intuitive or empirical guidance.   
5. Experiments are entirely synthetic, with no demonstrations on real KD applications (e.g., vision or language models).  

Minors:
1. Theorem 1 and related derivations are notation-heavy; adding a concise notation summary table would improve readability.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
