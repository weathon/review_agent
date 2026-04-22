# Achieving Fairness-Utility Trade-offs through Decoupling Direct and Indirect Bias

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Fairness in regression tasks is critical in high-stakes domains such as healthcare, finance, and criminal justice, where biased predictions can lead to unequal treatment. Bias can arise both directly, when sensitive attributes explicitly influence predictions and indirectly, when predictors correlated with sensitive attributes act as proxies. Existing fairness-aware regression methods often fail to address both forms of bias simultaneously, or sacrifice predictive performance. We propose Fair Envelope Regression Models (FERM), a novel framework that brings structure-aware subspace decomposition techniques from envelope regression into fairness-aware learning. FERM decomposes the predictor space into four orthogonal components: variation uniquely informative about the response, variation associated with sensitive attributes, shared variation, and residual noise. By penalizing only the sensitive component, FERM provides explicit and interpretable control over the fairness-utility trade-off. Unlike black-box approaches, FERM offers interpretable estimators with statistical efficiency guarantees under a fully parametric linear model. We validate FERM through extensive simulations and real-world experiments, showing improved fairness and predictive accuracy compared to prior work. Our results highlight envelope-based decomposition as a principled and powerful tool for building fair, efficient, and interpretable regression models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to improve fairness in regression through decomposing the input space into components that vary with the label Y and/or a sensitive attribute S. Theoretically they show how this can be estimated consistently and show it is a variance-reducing projection. Empirically, they demonstrate some wins on simulated and real data in some combination of predictive MSE and fairness.

### Strengths
- believe there is novelty in the algorithm, in the application of envelope regression to fair regression, and seems like a good fit of techniques
- synthetic experiment is compelling, showing a nice win in predictive MSE that one doesn't usually see in fairness papers
- theoretical results seems sound, and the interpolation/decomposition result in 5.4 seems intuitive and useful

### Weaknesses
- framing: authors introduce the idea of this paper as decomposing direct/indirect bias and I'm not quite sure that the method actually touches on that. For instance, indirect bias could be present in either XS or XSY as either can contain proxy variables.
- related work: it would be nice to see more discussion of + comparison to a) work from the causal literature that claims to do direct/indirect bias decomposition (are these the same/different ideas?) b) work from the fair representation learning literature that aims to learn de-correlated predictors through pre-processing
- some lack of clarity around (1) and (2) in the Limitations section in L185 - I don't think the rest of the paper really demonstrates how we get efficiency or interpretability gains with this method; certainly there isn't an empirical demonstration
- background: it would be great to get more background on envelope regression in the main body, given that it's the central technical tool in this paper. What does it do, and how should I think about it?
- Alg 2: not clear to me why if we have r=1, we wouldn't be fitting just normal OLS on all of X (given that the XS-estimation process is probably noisy, doing the unconstrained thing may be better)
- in Sec 6, should give more clarification on the difference in the two FERM methods: is one on XY + XSY and the other on XY only? something else?
- In general in the experiments, would be good to have more comments on what I should be looking for here - better MSE? better fairness? both? for instance in the Fig 3, we don't see an MSE improvement, or much of an unfairness improvement - would be helpful to communicate better what in the graph I should be taking away


smaller points:
- Fig 1 is more confusing than helpful I think - I'd recommend visualizing X as a vector rather than a space 
- “re Γ spans directions of X associated with S and Γ0 its invariant complement; Φ spans predictive directions for Y and Φ0 the immaterial ones.” - not sure why the authors use different terminology for Y and S? Are the concepts different? (eg associated/predictive, invariant/immaterial)
- L313: would be good to get more clarity on what it means to "exploit the envelope structure",  as well as the difference between the approaches outlines in equations on L313 and L315
-

### Questions
what is the exact relationship of the method  to direct/indirect bias decomposition?
how does the method improve on the 2 limitations from L185?
what does envelope regression do, and how?
what should the reader be looking for in the experiments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a framework for using subspace decomposition using envelope regression to improve the fairness in regression algorithms. By decomposing a predictor space into four components and isolating the part that corresponds to sensitive attributes, this provides a way to distinguish direct and indirect bias. The model introduces a ridge penalty to the sensitive components, which theoretical and empirical results show a way to trade-off between accuracy and fairness that improves on prior methods.

### Strengths
S1) The paper presents a novel and principled decomposition to address an important problem.

S2) Theoretical results are strong and show smaller asymptotic variance. 

S3) Empirical results are clear and compelling

### Weaknesses
W1) The framework relies on linear subscape decompositions. Further discussion about this assumption and the prevalence in real-world settings would be helpful.

### Questions
Other clarification questions:
Q1) I can't find additional implementation details about the baselines including hyperparameter tuning, and that would be helpful for understanding the experiments better.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this paper, the authors consider fairness in regression and state the importance of fairness. They introduce a fairness framework that adapts subspace decomposition techniques from envelope regression. The predictor space is decomposed into four orthogonal components: response-specific variation, sensitive variation, shared variation, and residual noise. This angle is good. Moreover, this decomposition makes it more interpretable in fairness regression.

### Strengths
- The decomposition is good, which makes it more interpretable in fairness learning.
- This paper provides numerical experiments on simulated and real datasets.

### Weaknesses
- Only consider the linear relationship between response and feature $X$.
- The authors should give the full names when they use at the first time.

### Questions
- Only consider the linear relationship between response and feature $X$. My main concern is about non-linearity. When we evaluate on tubular datasets (real datasets), 3-layer or 4-layer fully connected neural networks are used. In this paper, the author decompose the predictor space into 4 parts and consider linear regression on 2 of them.
- ``Envelope'', what is the meaning of this word?
- Lines 318-319, is it a definition of asymptotic variance matrix? Also, $T$ is a random variance? $\theta$ is a mean value? I am not sure my guess is right or wrong?
- Line 161, there is a mistake about $Cov(S,\hat U)=0$, it should be $S\perp\hat U$, since you need to subtract the mean values when calculating covariance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FERM, a fairness-aware regression method which decomposes the predictor space into response-specific, sensitive, shared and residual variations using envelope regression, in order to disentangle direct and indirect biases. FERM applies Ridge penalty only to the sensitive subspace to yield fine-grained, interpretable control over how sensitive attributes influence prediction. Authors provide theoretical results about consistency and efficiency of estimation with provable reductions in asymptotic variance relative to OLS and provide a closed-form characterization of fairness and utility. While the theory is sound, empirical validations are limited as they provide only one baseline, which is not consistently inferior to proposed method (example for smaller sample size or larger threshold $r$). I believe the paper could benefit from more extensive experiments to (i) validate theoretical claims about consistency and nonlinear robustness, (2) consider additional baselines and real datasets, and (iii) identify conditions under which the proposed method is superior to existing ones. The presentation of the paper could improve.

### Strengths
- Envelope estimation enables more statistically efficient (lower-variance) estimators and grounded understanding of model components.

- Applying a ridge penalty only to the shared (sensitive + response) subspace permits continuous interpolation between full fairness (no sensitive influence) and unconstrained prediction (maximum utility); assign Eq (6). 

- The proposed decomposition is structured for a multivariate setting, which permits handling multiple sensitive attributes, which could generalize better than pairwise or moment constraints. 

- Theoretical guarantees of efficiency and consistency, as well as closed-form for fairness-accuracy tradeoff, are provided in Section 5.

### Weaknesses
- Empirical validation is very limited--testing on only one real dataset and comparing against a single baseline--leaving method's generalization questionable and limiting strength of conclusions. Including a broader range of conceptually aligned fair regression methods and more real datasets will improve validation. 

- In the real-data experiment, FERM and the baseline FRRM exhibit "somewhat" comparable performance, with FRRM occasionally outperforming FERM at some unfairness thresholds (levels). This suggests that FERM’s practical benefits may diminish outside controlled settings. Further validation is needed. 

- While the experiments validate improved efficiency and some fairness–accuracy trade-off, they fall short of evaluating other theoretical claims such as estimator convergence and asymptotic fairness as established in Proposition 5.2 and Lemma 5.3.

- The synthetic experiment design at beginning of Sec 6 is confusing (not aligned with regression models in Eqns (4)-(6)). Both FERM-decorrelated and FERM-predictive sound different than the interpolated model, yet both apply a Ridge penalty which according to theory, should be on shared subspace. Also, both perform somewhat similarly. 

- Wouldn't the controlled linear dependence in data generation imply separable predictive & sensitive subspaces, hence the penalty could be unnecessary. Evaluating results without the penalty may clarify whether fairness arises from the decomposition itself or from regularization.

### Questions
Please see weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
