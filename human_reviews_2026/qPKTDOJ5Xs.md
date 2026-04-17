# Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
For overparameterized linear regression with isotropic Gaussian design and minimum-$\ell_p$ interpolator $p\in(1,2]$, we give a unified, high-probability characterization for the scaling of the family of parameter norms $ \\{ \lVert \widehat{w_p}  \rVert_r \\}_{r \in [1,p]} $ with sample size.

We solve this basic,  but unresolved question through a simple dual-ray analysis, which reveals a competition between a signal 
*spike* and a *bulk* of null coordinates in $X^\top Y$,  yielding closed-form predictions for (i) a data-dependent transition 
$n_\star$ (the "elbow"), and (ii) a universal threshold $r_\star=2(p-1)$ that separates $\lVert \widehat{w_p}  \rVert_r$'s which plateau from those that continue to grow with an explicit exponent. 

This unified solution resolves the scaling of *all* $\ell_r$ norms within the family $r\in [1,p]$ under $\ell_p$-biased interpolation, 
and explains in one picture which norms saturate and which increase as $n$ grows. 

We then study diagonal linear networks (DLNs) trained by gradient descent. 
By calibrating the initialization scale $\alpha$ to an effective $p_{\mathrm{eff}}(\alpha)$ via the DLN separable potential, 
we show empirically that DLNs inherit the same elbow/threshold laws, 
providing a predictive bridge between explicit and implicit bias. 

Given that many generalization proxies depend on $\lVert \widehat {w_p} \rVert_r$, 
our results suggest that their predictive power will depend sensitively on which $l_r$ norm is used.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to characterize one interesting term in overparameterized linear (models) regression: the behavior of $p$-norm minimum interpolator $||\hat{w}_{p}||_r$ against the number of data for different $r$ given $p\in(1, 2]$. To achieve this, the authors propose a novel "dual-ray" analysis to derive the corresponding scaling laws. Then they identify a data-dependent transition point and a universal threshold. They conduct some numerical experiments to verify their theoretical claims.

### Strengths
I think the most significant strength of this paper is the new "dual-ray" proof technique. I find it direct and intuitive, providing a better approach for the problem studied in this paper than approaches like GMT that might be too complex to be applied.

Furthermore, the derived $r$-norm of $p$-norm solution $ || \hat{w}_p ||_r $ as a function of several meaningful constants is also interesting. This form can somewhat support the authors' motivation "using norm-based bounds or proxies should be cautious".

### Weaknesses
However, I still have several concerns, and I will list them below.

1. I find the primary objective of this paper a bit unconvincing: studying the $r$-norm of the $p$-norm interpolator solution. There lacks a compelling motivation for this focus. Because it seems that it is the $p$-norm of the $p$-norm interpolator solution that matters, beyond the cautioning that generalization bounds may depend on the choice of $r$. This makes the scope of this paper a bit narrow. 

2. The theoretical analysis heavily depends on the isotropic Gaussian features and $p \in (1, 2]$. This does not explore the validity of the findings that violates this assumption, e.g., perhaps the most direct case $p = 1$.

3. Theorem 3.1 is based on a general $w_{\star}$ (neither single spike nor flat), but the corresponding scaling laws are only studied in two very extreme case (single spike and flat) to discuss the corresponding laws. I understand that this might be due to the clarity reason, but a thorough discussion for how the scaling laws behave for arbitrary $w_{\star}$ should also be crucial. This also leads to the next weakness.

4. The actionable guidance (line 463 - 467), together with the implications for practice (line 81-85), is a significant overclaim. The theoretical results are built upon linear regression, and there even lacks discussion (both theoretical and empirical) for general $w_{\star}$ in this task, then I think claiming the corresponding actionable guidance for general settings is very misleading. 

Minor: I think this paper should be reviewed in a venue for statistical learning theory or statistics, rather than ICLR (the connection with diagonal linear networks seems unnecessary because the authors directly convert the diagonal linear network to a minimizer of certain norm, which almost has nothing to do with the model itself).

### Questions
1. Why the authors do not include $p = 1$ (and other values of $p$) in the analysis?

2. How can the authors make their actionable guidance more reliable?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proves a closed form for the $\ell_r$ norm of the min $\ell_p$-norm interpolator under certain standard assumptions. Through experiments, the paper demonstrates that the trends predicted by the theory for linear regression, carry over to a slightly broader class of predictors.

### Strengths
- Very well-motivated, and clearly written.
- The mathematical derivations are sound and the application of the proof technique is, to my knowledge, novel in the context of analyzing min-norm interpolators.

### Weaknesses
- Theorem 3.1 is quite dense and could perhaps be split into the closed form result in eq 2, and the result characterizing the different sample size regimes
- It is not particularly easy to grasp what the main terms that determine $||\hat{w}_p||_r$ are. Would it be possible to give an interpretation of the quantities in equation 2?
- To further demonstrate the significance of the result, it would be nice to have a concrete example (perhaps supported by experiments) where one can take advantage of the relationship between $r$, $p$, $n$ and $d$ proved in Theorem 3.1. 

**Minor remarks**
- It would help carry the message across more easily if one could visualize the motivation in lines 55-66 with the help of a teaser plot (perhaps a condensed version of figures 1-2?).
- Some of the markers are not visible in the figures e.g. dotted line in fig 1a.

### Questions
- Is there an explanation for the non-monotonical trend of the $\ell_{1.1}$ norm in figure 1b?
- What is $d$ for figures 1-2? does varying $\kappa=d/n$ change the trends in the figures? also would it be possible to indicate for each figure the predicted transition scale $\n^\star$?
- In [1] the authors try to extend the observations from Donhauser et al, 2022 to inductive biases present in non-linear models. Would a similar extension be applicable to the results of this paper?

[1] - https://arxiv.org/pdf/2301.07605

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
This work studies the ell_r norm of the minimum ell_p norm interpolator for regression over gaussian data in the overparametrized setting where number of data n >> number of dimensions d. the setting assumes a sparse ground truth w^star.

In particular, they focus on how the ell_r norm grow when n grows when d/n has a limit kappa > 1.


The main finding (theorem 3.1) is that for r <= 2 ( p - 1 ), the ell_r norm of the min ell_p norm interpolator grows, while r > 2 (p -1), the same quantity plateaus.

Another finding is a "phase transition" from spiky-to-bulk dependency in terms of the number of samples (denoted n_star).

### Strengths
The theory is well-supported as seen in Figure 1. The provable behavior from the linear model seems to also carry over empirically to diagonal linear networks.

### Weaknesses
Can the authors give a more explicit reference to previous work that uses ell_r for bounding generalization error where r is not in {1,2}? I'm not convinced whether ell_r for r not in {1,2}. A more specific reference to a paper + lemma number would be helpful.

The legend of figure 1 looks to suggest that there should be error bars/confidence bands, but it's not visible in the plot. Is the result replicated?

### Questions
While i understand the spiky-vs-bulk part of X^TY, it's not clear to me how why the RHS of eqn (4) is said to be "spike-dominated". By contrast, in eqn (5), kappa_bulk clearly shows up on the RHS, so that makes sense.

Line 303 says middle panel exhibits a clear elbow near the predicted n_star. but it's not clear to me where is n_star on the middle panel plot.

The definition of t_star in eqn (1) is confusing. since it is a "definition", why is there a "w.h.p" at the end?

Minor:

Line 229 equation (8) links to the appendix when it could link to eqn (3) instead. Same issue in the caption of figure 1.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a unified theoretical and empirical study of how the parameter norms of overparameterized linear regression models scale with data under an ℓp-biased interpolation. Specifically, for minimum-ℓp interpolators with $p \in (1,2]$, the authors derive closed-form, high-probability laws describing how the family of norms $\{\lVert \widehat{w}_p \rVert_r\}$ for $r\in[1,p]$ changes with sample size $n$. A central insight is the identification of a data-dependent transition point n^\star (the “elbow”) separating bulk- and spike-dominated regimes, and a universal threshold $r^* = 2(p-1)$ that determines which norms plateau and which continue to grow. Using “dual-ray” analysis, the paper provides explicit expressions for these quantities and verifies the predictions empirically across both explicit $\ell_p$ regression and diagonal linear networks (DLNs) trained by gradient descent. By calibrating the DLN initialization scale $\alpha$ to an effective $p{\text{eff}}(\alpha)$, the authors demonstrate that DLNs inherit the same scaling laws, revealing a consistent connection between explicit and implicit bias. Overall, the work offers a clear, closed-form picture of how different norm measures behave with data growth and highlights that the predictive power of norm-based generalization proxies crucially depends on which ℓr norm is chosen.

### Strengths
See summary. The paper has a very clear and concrete contribution. The paper’s main strength lies in its clear and unified theoretical characterization of how all $\ell_r$ norms scale under $\ell_p$-biased interpolation. It successfully bridges explicit and implicit bias through an effective p_{\text{eff}}(\alpha) mapping in diagonal linear networks, and its empirical results convincingly validate the theoretical predictions across multiple regimes, offering practically valuable insights into the sensitivity of norm-based generalization measures.

### Weaknesses
I do not seen any major weaknesses of the study. The paper addresses a clear question. One minor weakness is that the connection to generalization is still less clear to me. It would be good to expand on this, and point out other work that also share the same weaknesses.

### Questions
Can authors comment on the Gaussian design choice? Do they expect Gaussian universality, or can the results significantly change under other distributions, and how under other natural distributions?

### Soundness
3

### Presentation
4

### Contribution
3
