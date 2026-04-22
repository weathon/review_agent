# Debiased Front-Door Learners for Heterogeneous Effects

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
In observational settings where treatment and outcome are confounded by unobserved factors but an observed mediator satisfies front-door conditions, estimating heterogeneous treatment effects remains underdeveloped. We introduce two debiased learners for heterogeneous front-door effects: FD-DR-Learner and FD-R-Learner. Both methods are constructed to be robust to nuisance estimation error, and we show they achieve fast quasi-oracle rates even when nuisance functions converge as slowly as $n^{-1/4}$. We provide error analyses that clarify their behavior under overlap and nuisance misspecification. In synthetic experiments varying sample size, nuisance noise, and overlap severity, both learners consistently outperform a plug-in baseline, with FD-R showing stronger stability under weak overlap. In a real-world case study using FARS data on primary seat-belt laws, the methods deliver reliable personalized effect estimates and interpretable heterogeneity patterns. Overall, the proposed learners offer practical and sample-efficient tools for heterogeneous causal estimation under front-door identification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose an adaptation of the R-learner for CATE estimation using a front-door structure with a binary mediator (instead of a back-door adjustment which the R-learner was developped for). They fill the gap with prior work that had been focusing on targeted averages, while their method targets conditional effects $\tau(.)$. To do this, they adapt two learners (DR and R-learners) to the FD setting. They reach quasi-oracle

### Strengths
The theoretical properties are clear and the setting too. The subject is interesting and well positioned. The empirical evaluation design seems sound.

I especially like the apparent soundness of the results

### Weaknesses
I found the paper's notation quite extensive and hard to read. Especially equations 1, 2 and 3 need to be described, and more importantly there is a lack of explanation in the. main text of the nuisance functions defined in Eq. 15, 16 and 17. The authors need to explicitly state what $\bar x$ refers to too.

Why did you not consider comparing your performances with Chen et al. 2025? 

I am not sure whether the environments for definitions, propositions and theorems are in accordance to the conference's template

### Questions
Could you consider a simulation with controlled FD violation to show the behavior of the estiamtor under misspecification?

Can you give more details on density ratios?

What happens in small sample sizes? The sizes considered in the synthetic study are very big and FD-PI seems to have better RMSE in lower sample sizes, can you comment on this?

Can your method provide confidence intervals for $\tau(.)$?

Other baselines could be considered in the experiments such as the oracle estimator and the mentionned Chen et al. 2025 method.

### Soundness
3

### Presentation
2

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
This paper considers the front-door criterion to identify heterogeneous treatment effect where there exists unobserved confounding. The authors proposed two estimator for this regime: FD-DR-learner and FD-R-learner. The FD-DR-learner follows the DR-learner framework that constructs a doubly robust pseudo-outcome leveraging FD identification of the treatment effect first, then regresses the covariates on the pseudo-outcome. The FD-R-learner first uses the (backdoor) R-learner to fit some of the nuisances, then constructs a "pseudo-outcome/function" from the learned nuisances to obtain the final estimator. Moreover, the author also provides an error analysis for the proposed estimator.

### Strengths
- Proposed two novel estimators with discussion on their performance, and when one is preferred over the other. 

- Provided error analysis that demonstrated the robustness of the proposed estimators.

### Weaknesses
- Proposed estimators are mainly built on top of ideas from exiting estimators.

- The FD-R-learner requires the estimation of many nuisance functions (has to split data into 3 folds). This could suffer when the sample size is small. 

- The paper discussed cases where one estimator is preferred over the other, but did not include experiments to support the claims.

### Questions
- This paper only considers binary mediators, can the estimators be extended to higher dimensional or continuous mediators?

### Soundness
3

### Presentation
2

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
This paper tackles the challenge of estimating heterogeneous treatment effects (HTEs) under unmeasured confounding using the front-door (FD) criterion. It introduces two novel debiased estimators—FD-DR-Learner and FD-R-Learner—that achieve quasi-oracle convergence rates even when nuisance functions converge slowly at $n^{-1/4}$. FD-DR leverages doubly robust pseudo-outcomes, while FD-R decomposes effects into interpretable causal pathways. Both methods demonstrate superior performance and robustness in synthetic and real-world experiments, providing accurate and stable HTE estimates where traditional plug-in estimators fail. Overall, the study advances causal inference by enabling reliable, personalized effect estimation under front-door settings with theoretical and empirical validation.

### Strengths
1. This paper is among the first to tackle HTEs under FD.
2. The proposed method is supported by rigorous theoretical analysis.
3. Experimental results demonstrate that the proposed approach.

### Weaknesses
1. The proposed methods and theory are restricted to binary mediators.
2. The paper does not include empirical comparisons with existing HTE-FD methods (e.g., Chen et al., 2025, LobsterNet).
3. Beyond double robustness, a key advantage of debiased estimators lies in their asymptotic normality, which enables valid statistical inference. However, the paper does not provide any inference results or variance estimation, leaving the uncertainty quantification of the proposed estimators unexplored.

### Questions
1. While theoretically appealing, the front-door identification relies on strong and often unverifiable assumptions—such as the absence of unmeasured confounders between mediators and outcomes. As noted by Imbens and others [1], these conditions are rarely satisfied in practice, raising concerns about the empirical credibility and real-world applicability of the proposed methods.
2.  Figure 3(a)’s histogram suggests that the sampling distribution of the estimator deviates noticeably from normality.



[1] Potential Outcome and Directed Acyclic Graph Approaches to Causality: Relevance for Empirical Practice in Economics

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
This work presents estimators for the CATE under a front-door adjustment formula. This contrasts with existing works, which focus on a back-door formula. The authors derive variants of the DR- and R-leaners for this setting and provide an error analysis that reveals double robustness.

### Strengths
To my knowledge, this the first estimator of the CATE under front-door adjustment. This is important since it makes it possible to estimate the CATE in a wider variety of settings than was previously possible.

The theoretical arguments appear to be sound.

### Weaknesses
The theoretical analysis ends up being somewhat similar to that of existing backdoor methods, but that's no fault of the authors - that analysis strategy works well.

### Questions
Minor comment on references on line 41-42:

* R-learner was first proposed in Corollary~9.1 of
Robins, James M. "Optimal structural nested models for optimal sequential decisions." Proceedings of the Second Seattle Symposium in Biostatistics: analysis of correlated data. New York, NY: Springer New York, 2004.
* DR-learner was proposed in Section 3.1 of 
van der Laan, Mark J. "Targeted Learning of an Optimal Dynamic Treatment, and Statistical Inference for its Mean Outcome." (2013).

### Soundness
4

### Presentation
4

### Contribution
3
