# Causal Discovery via Quantile Partial Effect

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Quantile Partial Effect (QPE) is a statistic associated with conditional quantile regression, measuring the effect of covariates at different levels. Our theory demonstrates that when the QPE of cause on effect is assumed to lie in a finite linear span, cause and effect are identifiable from their observational distribution. This generalizes previous identifiability results based on Functional Causal Models (FCMs) with additive, heteroscedastic noise, etc. Meanwhile, since QPE resides entirely at the observational level, this parametric assumption does not require considering mechanisms, noise, or even the Markov assumption, but rather directly utilizes the asymmetry of shape characteristics in the observational distribution. By performing basis function tests on the estimated QPE, causal directions can be distinguished, which is empirically shown to be effective in experiments on a large number of bivariate causal discovery datasets. For multivariate causal discovery, leveraging the close connection between QPE and score functions, we find that Fisher Information is sufficient as a statistical measure to determine causal order when assumptions are made about the second moment of QPE. We validate the feasibility of using Fisher Information to identify causal order on multiple synthetic and real-world multivariate causal discovery datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Quantile Partial Effect (QPE) as an observational-level object for causal discovery. QPE is defined via conditional quantiles and admits multiple equivalent forms, enabling estimators that do not assume a specific functional causal model (FCM). The core theory shows identifiability of causal direction when each component of the QPE lies in a finite linear span of basis functions, formalized by Theorem 3.6. Building on this, the authors give two bivariate procedures: (i) a kernel QPE with an OLS basis test (QPE-k) and (ii) a flow-based QPE with a neural basis test (QPE-f)—and report strong empirical results. For multivariate discovery, they leverage a PDE link between QPE and the score function to derive Fisher-Information Causal Ordering (FICO), which greedily removes leaf nodes using marginal score variances under an additional assumption (Assumption 4.5). Experiments show QPE-f competitive or superior to many bivariate baselines, and FICO performs competitively on synthetic and real benchmarks.

### Strengths
a. The equivalence of QPE and the Wronskian-based identifiability criteria are clear and self-sufficient at the observational level.

b. Unifies causal-velocity style reasoning with quantile/CDF views and score-based analysis.

c. Two practical bivariate pipelines: kernel, OLS basis test (fast), and flow, neural basis test (accurate). And, for multivariate cases, FICO is simple to implement given score estimators.

### Weaknesses
a. The central finite-span assumption (Assumption 3.5) is elegant but potentially restrictive; practical guidance on choosing/validating basis sets is limited. How sensitive are conclusions to basis mis-specification?

b. Assumption 4.5 for FICO appears to lack easy a-priori diagnostics.  The performance on Sachs suggests it fails in practice, which narrows applicability.

c. Multivariate ranking depends on accurate scoring function estimation; the paper uses first-order scores for speed but does not deeply analyze robustness to estimator bias or noise.

### Questions
a. How is the assumption on lines 96–97 used downstream? e.g., in which lemmas/theorems or in the estimator design. and what would break without it?

b. The paper claims that QPE does not require the (causal) Markov assumption. However, some steps appear to rely on independence of noise terms (exogeneity)?

c. How should practitioners select basis families for Assumption 3.5 in new domains? Could the authors provide data-driven diagnostics or model selection among candidate bases?

d. How sensitive are ODRs to the estimator class and its hyperparameters in FICO?

e. Why Sachs is hard? Can the authors analyze where Assumption 4.5 breaks on Sachs?

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
The authors propose estimating the quantile partial effect (QPE) as a means of functional causal discovery. The QPE generalizes causal velocity to the distributional level, which avoids making SCM assumptions. They derive novel conditions for both identifying causal direction in the bivariate case as well as sink nodes in a multivariate setting. Based on these conditions, they propose two methods for bivariate causal discovery (QPE-f, QPE-k) and an order-based approach for the multivariate case (FICO).

### Strengths
- The authors propose a Wronskian criterion for bivariate causal discovery that is not only theoretically interesting but also directly informs novel methodology based on a practical test, which shows strong performance in the common benchmarks. 
- Mathematical erivation and discussions (especially in the Appendix) are complete and rigourous
- Experiments are exceptionally comprehensive and easy to follow 
- Overall writing is clear, the paper is well-structured and logical

### Weaknesses
- The theoretical generalization beyond causal velocity is limited in practice, since the discussion in Appendix A.2 on identifiability still requires the SCM assumption which equates QPE and causal velocity. 
- Assumption 4.5 for FICO is specifically designed for Corollary 4.3 and is difficult to interpret in practice (although I appreciate the authors acknowledging this).
- The resulting algorithm, FICO, seems very similar to the SCORE algorithm (Rolland et al., 2022).

### Questions
- Question on your Appendix A.2: You show that, given a joint distribution (up to $p(x)$) the class of forward models $p(y|x)$ are non-parametric, while the class of backward models $p(x|y)$ are parametric, based on an extension of the ANM argument in Hoyer et al., 2008. My question is this: does the (k+2)-th order ODE for $\log p(x)$ somehow depend on the anti-causal model, either via $\eta_{X \mid Y}$ or via the basis functions $\phi(x)$ in Assumption 3.5? And if it does, will changing these unknown properties also change the ODE? 

- Question on FICO: can you clarify the relationship between FICO and SCORE? You still have to estimate the score function for FICO, correct? SCORE looks for the smallest variance in score across nodes, while FICO looks for the smallest FI, or squared score. But asymptotically under mild conditions the FI and variance are the same (the experimental results also look very similar). Is Corollary 4.3 a condition on when we expect SCORE to work for non-linear, non-additive noise models?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an approach to causal discovery from observational data, based on the notion of quantile partial effect (QPE). In particular, three distinct approaches utilizing QPE are proposed:

(i) A kernel-based, non-parametric approach for bivariate discovery which assumes that the QPE can be represented as a linear combination of known basis functions,

(ii) A parametric approach for bivariate discovery using the idea of normalizing flow, where each flow is parameterized by a neural network,

(iii) A parametric approach, which recovers a causal ordering among a set of variables, based on the Fisher information matrix.

### Strengths
(S1) Causal discovery is an important topic in causal inference, and further advances in this context are needed. The paper deals with an important topic.

(S2) The framing of previous methods (LiNGAM, ANM, HNM) as specific cases of the general setting in which QPE lies in the linear span of fixed basis functions, seems interesting and novel. Getting a generalization of current approaches assuming different functional constraints is a good direction to explore.

(S3) The performance of the flow-based (neural) approach for QPE testing seems to perform quite well empirically.

### Weaknesses
(W1) The exposition of the paper could be substantially improved. A range of assumptions are discussed, and in the current version, it is non-trivial to follow for which of the proposed approaches exactly which assumptions are used. This could be streamlined through an overview table.

(W2) Causal Velocity comparison is misleading: Some of the language around causal velocity / bijective causal models seems concerning (abstract, after Proposition 3.3). 

When a bijective causal model assumption is invoked, some collapse of the layers does occur. This seems to be an important point for causal velocity.

QPE is defined purely at the observational level, and does not pertain to any structural assumptions — it is simply a functional of the underlying observational distribution. Therefore, comparing these two in such a way (saying that “QPE does not need a monotonicity assumption”) seems quite confusing. It seems more likely that (i) the definition of QPE does not require monotonicity; but (ii) for the proposed discovery methods to work, monotonicity is assumed?

(W3) Removal of the Markovian assumption: the claims on the removal of the Markov assumption are very curious. Most work in causal discovery using functional assumptions considers the Markovian case. If the authors are truly relaxing this assumptions, this seems like a big deal, and should be emphasized strongly. If not, explaining exactly how the Markov assumption is “not needed” is very important — is the case again that QPE can be defined without invoking the Markov assumption, but to apply QPE-based discovery methods, the Markov assumption is needed? If the latter is true, these claims need to be reframed properly.

(W4) It is not uncommon for causal discovery methods to provide some form of formal guarantees that the method will work with high probability etc. Are the formal guarantees needed / missing from the paper?

(W5) Line 289 states that “OLS assumes a Gaussian distribution”. OLS is a method, which does not assume any distribution implicitly. Of course, with Gaussian noise, OLS may exhibit specific properties (become the MLE etc.). Clarifying this would be important.
Furthermore, in this setting arbitrary distribution of $x_t, y_t$ can now be handled — is this previously not the case? This is definitely worth clarifying.
This lack of clarity on the exact assumptions connects back to point (W1).

(W6) In Lines 338, it is said that $\partial_y \psi_{Y | X,i} = 0$ is “e.g. ANM”. To be clear, in this context, additive noise is assumed in addition to the fully parametric model used in this setting? 

ANM usually allows arbitrary functions $a(X)$; which is not the case here? It would be more precise to say “additive noise” than ANM in this case?

Same comment for the “e.g. HNM” model.

(W7) Assumption 4.5 seems to be important for the FisherInfo-based approach. I currently have no idea how one might judge the validity of such an assumption, or even start thinking about it? It seems that this is insufficiently discussed, and raises the concern whether the exact condition needed for the method is assumed, regardless of its plausibility.

I note that I would be willing to reconsider my grade in light of the authors' responses to weaknesses / questions.

### Questions
(Q1) Is the representation in Table 1 actually true?: specifically, for PNL-ANM and PNL-AHM models, the basis functions $\bar g, g^{-1}\bar g$, etc. are needed. Isn’t it generally the case that the function $g(\cdot)$ is not known in these settings? How does this tie with the fact that the basis functions in $\Phi$ need to be known?

(Q2) Relating to the previous point, is there any nuance w.r.t. choosing the basis? The choice of the kernel bandwidth parameter is mentioned as a limitation, but the choice of basis (which is even more fundamental?) is not really mentioned.

(Q3) The performance of QPE-f seems to be very strong empirically. I wonder, for settings where the data-generating model is actually an ANM, does the ANM method perform better? (similar question for LiNGAM & others). If I understand, current experiments would seem to imply the answer is no? If so, it is curious.

(Q4) Relating to previous point — given the strong performance of QPE-f, is there any way to move beyond bivariate discovery with such an approach?

### Soundness
3

### Presentation
2

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
The paper studies causal discovery from observational data using distribution shape constraints. Authors motivate non-graphical assumptions about the gradient of the joint distribution that emphasize the asymmetry between the cause and effect in a bivariate case, and extend the results to multivariate causal discovery by iteratively finding the last variable in the causal order. Through examples, they show how the shape constraints introduced in their work is distinct from the graphical constraints commonly employed in this problem, and experiments validate usefulness of their approach.

### Strengths
1. Clear statement of the assumptions underlying the theoretical results.
2. Motivating examples throughout the paper allow the more curious reader to understand the motivation.
3. Algorithm 1 is impressively simple, and the connection between fisher information and the shape constraints are quite insightful.

### Weaknesses
1. Appendix A3 contains very important remarks that deserve to be presented in the main text. The paper argues that the shape assumptions are more *relaxed* than the existing assumptions employed in causal discovery, and this subtle points needs to be established carefully, or else the reader may find the contributions not well-motivated.
2. Sections 3.3 and 3.4 are not contributing to what comes later in the manuscript; they are useful extensions of 3.2, though, I recommend a restructuring as they hurt the coherency.
3. Missing discussion about related work [1,2,3,4] that would enrich the work.

[1] Hyvarinen, A., Sasaki, H., & Turner, R. (2019, April). Nonlinear ICA using auxiliary variables and generalized contrastive learning. In The 22nd international conference on artificial intelligence and statistics (pp. 859-868). PMLR.

[2] Jalaldoust, K., Salehkaleybar, S., & Kiyavash, N. (2025). Multi-domain causal discovery in bijective causal models. In Proceedings of the Fourth Conference on Causal Learning and Reasoning.


[3] Guo, S., Tóth, V., Schölkopf, B., & Huszár, F. (2023, December). Causal de finetti: On the identification of invariant causal structure in exchangeable data. In Thirty-seventh Conference on Neural Information Processing Systems.

[4] Reizinger, P., Sharma, Y., Bethge, M., Schölkopf, B., Huszár, F., & Brendel, W. (2023). Jacobian-based causal discovery with nonlinear ICA. Transactions on Machine Learning Research.

### Questions
Please address my concerns above.

### Soundness
4

### Presentation
3

### Contribution
4
