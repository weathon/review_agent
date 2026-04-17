# Identification and Estimation of Treatment Effects under Coupled Confounding and Collider Biases

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In causal inference, confounding bias and collider bias pose two major challenges for treatment effect estimation in observational studies. Confounding bias arises from unobserved common factors that simultaneously affect the treatment and outcome, while collider bias results from non-random sample selection caused by both variables. Existing methods focus on bias correction for a specific bias, such as using Instrumental Variables (IVs) to address confounding bias and Selection IVs (SIVs) to mitigate collider bias. However, real-world data frequently exhibit coupled confounding and collider biases, where unmeasured confounders directly affect the selection mechanism. Currently, the coupled biases problem remains an unaddressed challenge. In this paper, we propose a new identification theory for treatment effects under coupled biases with an IV set, which contains subsets serving as IV and SIV, respectively. Based on this theory, we propose a novel treatment effect estimation method, DualDebiasIV (DDIV), which decomposes the IV set to separately obtain the SIV and IV, using them for biases decoupling and correction. To the best of our knowledge, this is the first work to provide a solution for the identification and estimation of treatment effects under coupled biases. DDIV is theoretically guaranteed, with proofs provided for the correctness of the decomposition and the consistency of the estimates. Extensive experimental results on semi-synthetic and real-world datasets show that DDIV achieves significant performance improvements, further demonstrating its practical effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies treatment-effect estimation when two biases act together: unmeasured confounding and collider (selection) bias. The authors assume access to an "IV set" $D$ that contains variables influencing treatment and selection separately, with exclusion and independence properties. They prove an identification result for $E[Y \mid d o(T=t), X]$ under a separable selection model that allows unmeasured factors to affect selection. They then propose DualDebiasIV (DDIV), a two-stage estimator that (i) decomposes the IV set into an IV-like representation $\Phi_t$ and a selection-IV representation $\Phi_s$ via mutual-information constraints, fits $T$ on ( $X, \Phi_t$ ), and learns the selection score $\pi\left(X, T, Y, \Phi_s\right)$; and (ii) reweights by $\pi^{-1}$ and regresses $Y$ on $\left(X, \hat{T}, \Phi_s\right)$. The paper presents consistency arguments and experiments on a semi-synthetic demand benchmark and the Wage2 dataset, showing lower MSE and ATE estimates closer to a reference value.

### Strengths
Originality.
- The paper targets the coupled-bias setting explicitly and gives an identification theorem that combines an IV subset for treatment and a selection-IV subset for sampling. This is a new blend of ideas that is not handled by standard IV or selection models alone. The decomposition of an IV set into $\Phi_t$ and $\Phi_s$ with MI penalties is also novel as a practical mechanism to operationalize the theory.

Quality.
- The identification section states clear structural assumptions (IV-set exclusion and independence; separable selection with unobservables), and the estimation section aligns with those assumptions.
- The algorithm is consistent with the theory: first remove selection bias with $\Phi_s$ and $\pi$, then address confounding with $\Phi_t$ (via fitted $\hat{T}$ ).

Clarity.
- The problem is motivated with graphs and a real-world story. The assumptions are labeled, and the two-stage pipeline is easy to follow.
- Limitations are acknowledged (need for an IV set; tuning and architecture choices).

Significance.
- Many ML+causal settings face both residual confounding and non-random selection. A framework that unifies identification and estimation for this case can influence practice across economics, health, and policy analytics.
- The approach is modular: it can plug into representation learners and outcome models, making it relevant to ICLR.

### Weaknesses
1) Assumptions and their testability.

The main identification rests on three strong pieces:
- (a) a supplied IV set $D$ that is independent of $(X, U)$ and excluded from $Y$;
- (b) a known split of $D$ into $D_t$ (for confounding) and $D_s$ (for selection);
- (c) an additively separable selection model.

Assumption (b) is especially demanding in practice. It asks the user to know, in advance, which elements of $D$ shift treatment only ($D_t\not\perp T\mid X$) and which shift selection only ($D_s\not\perp S\mid (X,T); D_s\perp T$). This is not clear in real data how to determine $D_s$ from $D_t$. Also, the paper should provide tools and evidence that make (a) and (b) more credible and more robust, e.g., provide a sensitivity analysis that perturbs (i) exclusion in $D$ and (ii) separability in selection, and report how bias/variance change.

2. Role and discovery of the IV set.
- The method presumes a supplied $D$ with known $D_t$ and $D_s$. In many ML applications, $D$ is high-dimensional and noisy, with some components weak or invalid. Please add experiments that (a) inject invalid instruments into $D$ (direct paths to $Y$ or correlation with $U$ ) and (b) vary instrument strength and overlap. The authors can clarify when $\Phi_t, \Phi_s$ remain valid under various contaminations.

3. Mutual-information penalties and identifiability in practice.
- Proposition 4.1 hinges on "sufficiently rich models," consistent MI estimation, and large penalty weights. In finite samples, CLUB can be biased and high-variance. Can the authors add ablations over MI estimators (CLUB vs. NWJ vs. MINE), penalty magnitudes, and representation capacity?

4. Target estimand and heterogeneity.
- The identification section mixes conditional effects $E[Y \mid do(T=t), X]$ and ATE in experiments. Please be explicit about the target estimand for each experiment and about if the homogeneity assumption (Assumption 2.4) is needed to link IV identification to that estimand. Another question is that how does general effect heterogeneity (general CATE) affect the validity of Assumption 2.4? In other words, can you give an example of data generating models with general CATEs which simultaneously satisfies Assumptions 2.1, 2.2, 2.4, 3.1, and 3.2? If the target causal estimand is ATE, can the authors provide valid confidence intervals?

### Questions
1. Diagnosing the IV set.

How should a user check that the supplied $D$ is plausibly independent of ($X, U$) and satisfies exclusion with respect to $Y$ ? Can you propose empirical proxies or falsification tests that your pipeline can output by default?

2. When $D$ is partially invalid.

If a fraction of variables in $D$ violate exclusion or independence, do the MI constraints still recover usable $\Phi_t$ and $\Phi_s$ ? Please provide theory or a robustness experiment that quantifies the breakdown point.

3. Separable selection model.

Your identification needs logit$P\left(S=1 \mid U, X, T, D_s, Y\right)=q\left(X, T, D_s\right)+h(U, X, T, Y)$. How sensitive are results to mild non-separability (e.g., small $D_s \times Y$ interactions)? Can you add a sensitivity curve that sweeps an interaction term from 0 to moderate values?

4. Instrument strength and overlap.

What happens as $D_t$ becomes weak or as overlap in $P(T \mid X)$ or $P(S \mid X)$ worsens? Please report performance vs. first-stage $R^2$, F-statistics, and minimum/maximum selection propensities, including failure thresholds.

5. Computational cost.

What is the training time relative to DeepIV/DeepGMM on the same hardware? Include wall-clock and parameter counts; discuss scalability to large $D$ or image/text covariates.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the problem of identifying and estimating treatment effects when both confounding bias and collider bias exist and are coupled, meaning that unobserved confounders influence both treatment assignment and sample selection. The authors propose a new identification theory using an instrumental variable (IV) set that includes subsets serving as both IVs and selection IVs (SIVs). Based on this theory, they develop DualDebiasIV (DDIV), a two-stage estimation approach that decomposes the IV set into IV and SIV representations using mutual information constraints, and then corrects for coupled biases through reweighting and regression. The paper provides proofs of identification and consistency, and experimental results on semi-synthetic and real datasets show that DDIV outperforms existing methods.

### Strengths
1. The paper is well written and easy to follow. The motivation and problem setup are clearly presented, with intuitive examples and well-structured explanations.

1. The problem studied is both interesting and important, addressing a setting rarely discussed in causal inference.

1. This paper introduces a method that decomposes IVs into IV and SIV representations via mutual information minimization.

### Weaknesses
1. The claim that “the coupled bias problem remains unresolved” is somewhat misleading, as it has been studied in [1]. The paper should better position its contribution by explicitly discussing how it differs from or extends prior work.
1. It is unclear whether inference stages can be conducted to establish the asymptotically normal property. A discussion on this aspect would strengthen the theoretical contribution.
1. It is clear that $D_t \not \perp T$ and $D_s \perp T$. Thus, they are easily separated by independence test. Why do the authors rely on representation learning, which may introduce inaccuracy due to model complexity and the difficulty of mutual information estimation?
1. The reason for splitting the data (Batch 1 and Batch 2(A)) in Algorithm 1 is not explained. Clarifying the motivation and necessity of this step would help readers understand the algorithmic design.
1. Real-world validation is limited to one dataset where coupled biases are artificially introduced, reducing the practical credibility of the results.



[1] Li, Baohong, et al. "Two-stage shadow inclusion estimation: an IV approach for causal inference under latent confounding and collider bias." *Forty-first International Conference on Machine Learning*. 2024.

### Questions
see above

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
3

### Summary
This paper claims to address a long-standing challenge in causal inference for observational studies: coupled confounding and collider biases. To solve this, this paper proposes an identification theory for treatment effects under coupled biases, which relies on a strong assumption that a known IV set containing subsets that act as IV and SIV respectively. This paper validates the method on semi-synthetic  and real-world datasets, showing it outperforms baselines.

### Strengths
1. The paper’s identification theory is novel.
2. The paper meets good standards of quality across theoretical, methodological, and experimental dimensions.
3. The paper is well-organized and accessible to both causal inference experts and researchers familiar with machine learning.

### Weaknesses
1. The paper’s foundational Assumption 3.1 (“Known Instrumental Variable Set”) requires a pre-identified set containing disjoint IV and SIV subsets. This assumption is unrealistic for most real-world scenarios.
2. The paper claims to be the “first work to address coupled confounding and collider biases”, but this is misleading. Prior studies such as [1] already tackle joint bias correction.
3. The paper’s experiments are well-controlled but fail to validate DDIV’s performance in scenarios that reflect real-world challenges.

[1] Li, B., Wu, A., Xiong, R. &amp; Kuang, K.. (2024). Two-Stage Shadow Inclusion Estimation: An IV Approach for Causal Inference under Latent Confounding and Collider Bias. <i>Proceedings of the 41st International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 235:28949-28964 Available from https://proceedings.mlr.press/v235/li24bu.html.

### Questions
See above.

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
This paper studied the problem of causal effect estimation under coupled confounding and collider biases. Under this framework, the paper proposed a new identification method for treatment effects under coupled biases with an IV set. The proposed estimation method DualDebiasIV (DDIV) aims to decompose the IV set to separately obtain the SIV and IV, then use them for debiasing and decoupling. Theoretically, the paper showed the correctness of the decomposition and the consistency of the estimate of DDIV. Empirical results further supported the effectiveness of the proposed method.

### Strengths
- Overall I found this paper presented an interesting framework for causal effect identification with coupled confounding and collider biases. Such a scenario can be common in practice and the proposed method brings useful insights for practitioners. 
- the proposed DDIV algorithm uses a two-stage solution to first identify IV and SIV variables, then adjust for the two biases to estimate the causal effect. This process is very intuitive and natural to apply. 
- Theoretically, it is shown that DDIV achieves unbiased estimation. Emprically, the experments with synthetica and real-world data further supported that result.

### Weaknesses
My main concerns are about the technical novelty and contribution:
- as the authors claimed, there has been few prior work studying the scenario where the two types of biases are coupled. However, the main technical difficulty seems to stem from the usage of mutual information to identify the IV and SIV sets. Once such sets are obtained, adjusting two biases via regression has been a well-studied solution. Therefore I remain some concern about the technical contribution from this work.

### Questions
what is the time complexity for DDLV? is it optimal?

### Soundness
3

### Presentation
3

### Contribution
2
