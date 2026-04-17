# Reliable Probabilistic Forecasting of Irregular Time Series through Marginalization-Consistent Flows

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Probabilistic forecasting of joint distributions for irregular time series with missing values is an underexplored area in machine learning. Existing models, such as Gaussian Process Regression and ProFITi, are limited: while ProFITi is highly expressive due to its use of normalizing flows, it often produces unrealistic predictions because it lacks marginalization consistency—marginal distributions of subsets of variables may not match those predicted directly, leading to inaccurate marginal forecasts when trained on joints.
We propose MOSES (Mixtures of Separable Flows), a novel model that parametrizes a stochastic process via a mixture of normalizing flows, where each component combines a latent multivariate Gaussian with separable univariate transformations. This design allows MOSES to be analytically marginalized, enabling accurate and reliable predictions for various probabilistic queries.
Thanks to its inherent marginalization consistency, MOSES significantly outperforms all baselines—including ProFITi—on marginal predictions.
For joint predictions, it beats all other consistent models and performs close to or slightly worse than ProFITi. Implementation details:~\url{https://github.com/yalavarthivk/separable_flows}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a model named MOSEs, designed to achieve marginalization consistency in probabilistic irregular time series forecasting. Formally, given an input $X$ and a full query set $Q$, marginalization consistency requires that for any subset $S \subseteq Q$, $p(y_{S} \mid S, X) = \int p(y \mid Q, X), dy_{Q \setminus S}$ i.e., the marginal distribution obtained by direct querying should be consistent with that obtained through marginalization. MOSEs enforces this property using a mixture of separable normalizing flows, where each flow is constructed from a parameterized source Gaussian distribution followed by separable univariate transformations. Multiple flows are then combined with softmax-weighted mixing coefficients. Experimental results demonstrate that MOSEs achieves superior marginalization consistency compared with baseline models.

### Strengths
1. The issue studied in this paper—marginalization consistency—is crucial for real-world probabilistic forecasting applications.

2. The paper is well-structured and easy to follow.

3. The proposed model theoretically guarantees marginalization consistency in closed form.

### Weaknesses
1. The separability constraint (Equation 8) is overly restrictive, substantially limiting the flexibility of the normalizing flow. A flow defined as $f(x_1, x_2) = (f(x_1), f(x_2))$ effectively models only the marginal distributions and implicitly assumes $p(x_1, x_2) \ne p(x_1)p(x_2)$. Although the use of parameterized source Gaussians and mixture modeling can partially mitigate this issue, the restriction still reduces the expressive power that typically makes flow-based models advantageous.

2. Consequently, the prediction accuracy (as measured by njNLL) is inferior to that of non-separable flow models. This limitation is further supported by the results shown in the Figure 2, where the GMM achieves better approximations, despite MOSEs employing more complex neural architectures.

### Questions
1. To further evaluate the expressiveness of separable flows, please compare **MOSES(1)** with ProFITi on the following toy example: a 2D Gaussian mixture distribution with two components—one centered at $(1, 1)$ and the other at $(-1, -1)$. In this case, $p(x_1, x_2) \ne p(x_1)p(x_2)$, making it suitable to test whether the separability assumption limits modeling capacity.

2. Since ProFITi performs better in terms of njNLL, is it possible to marginalize its joint distribution to obtain the corresponding marginal distributions? If so, how does this method perform on njNLL compared with MOSES?

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
4

### Summary
This paper treats the problem of probabilistic forecasting of multivariate irregular time series with missing values. The emphasis is put specifically on obtaining the joint distribution over all covariates at new unobserved query timestamps. The authors identify two main issues with the existing methods: Inconsistent marginals in the case of baselines that directly learn the joint distribution with no consistency guarantess (e.g. ProFitI), and Lack of expressivity of baselines that provide consistent marginals by design (e.g. GPs). In order to fill this gap, the authors propose MOSES, a method that learns a mixture of separable conditional normalizing flows, enabling marginals consistency through separability, and expressivity through the flows' invertible transformations. Experiments in toy datasets and 4 real world datasets show that the proposed method indeed achieves better marginals consistency, trading-off errors in the joint distribution estimation compared to baselines (ProFITi).

### Strengths
- The paper is clearly written, highlighting all the important aspects of the method, its comparison with existing baselines, and the experimental results.
- The approach is theoretically well-motivated, and innovative in imposing a dependency structure in the source distribution of the NF, conserving separability afterwards through component-wise transformations $\phi$.
- The experimental results are consistent with the paper's narrative in the sense that they show the proposed approach indeed improves the marginal distributions negative log-likelihood.

### Weaknesses
- My main concern is that unlike the claim in the abstract "it matches ProFITi in joint prediction performance", the proposed method achieves statistically-significant worse joint negative log-likelihood than ProFITi in 3 out of 4 real-world tasks considered in the paper (referring to Table 1). Furthermore, in the only dataset where it outperforms ProFITi, it's shown in the ablation study in the appendix (Table 11) that the mixture components ($D>1$) are important for this dataset, suggesting the improvement could be related to the mixture rather than the marginals consistent design of MOSES. The last matter make me wonder whether ProFITi can be augmented to fit a mixture joint density, in which case it could maybe win also in the 4th dataset (and perhaps in univariate marginals as well).

### Questions
- Regarding the illustrative examples in Figure 1, although they do a good job in showing the marginal consistency feature of MOSES, I believe that they fail in providing a comprehensive comparison with ProFITi. Specifically, I'm referring to the fact that in these examples, MOSES appear to have a better fit even for the joint density (first line of plots showing the samples). Dont the authors think that it would be better to illustrate a case where ProFITi achieves a better joint density, all while having inconsistent marginals (similar to the results on Physionet'12, MIMIC-III, and MIMIVC-IV)? In this case, the tradeoffs between fitting the joint density and having consistent marginals would be clearer in my opinion. 
- I have a question regarding how the transformation parameters $\theta_{\text{flow}}$ are learned, it's mentioned in Appendix A.4 that these parameters are set based on the linear parameters of the LRS model, the knots, their derivatives, etc, and that they are common between variables and mixture components. Doesn't this violates the separability characteristic of the NF in the sense that the transformation $\phi$ now depends on other query embeddings through the knots?
- Correct me if I'm wrong but you learn $D$ query embeddings (so $D$ different sets of parameters $\theta_{\text{d}}^{\text{query}}$) then you train everything end-to-end, so I guess in the final likelihood term on $y$, the gradients flowing to a given set of parameters $\theta_{\text{d}}^{\text{query}}$ are coming only from the corresponding mixture terms $p_{\mathrm{Y}_d}(y|\mathbf{h}_d)$. If this is correct, how do you prevent these query representations from collapsing into the same representation?

### Soundness
3

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
3

### Summary
The paper proposes MOSES, a mixture of separable flows for probabilistic forecasting on irregular multivariate time series. The core idea is to guarantee marginalization consistency by (i) using a consistent Gaussian base with low-rank-plus-identity covariance, (ii) applying per-target invertible spline transforms whose parameters depend only on that target’s query and the shared context, and (iii) mixing such components with query-independent weights. Experiments on several datasets show competitive joint NLL and markedly better marginal NLL and near-zero marginal-inconsistency.

### Strengths
- Clear formalization of requirements (e.g., marginalization consistency) and an architecture that enforces them by construction.
- Clean, scalable base: low-rank plus diagonal covariance and diagonal Jacobian flows give tractability while retaining flexible marginal shapes.

### Weaknesses
- Architectural assumption is critical. Consistency hinges on each target’s transform depending only on its own query + context (not other queries). This should be stated as a formal assumption in the lemmas.
- The low-rank plus covariance shared via a global parameter limits cross-dimensional dependence. The “flow-level expressiveness” claim should acknowledge this trade-off.
- Complexity claims are theoretical; wall-clock/memory vs. strong baselines at larger $K$ are missing.
- “Fully identifiable” is too strong.

### Questions
- Can you show that your R3 implies full projective consistency for any two finite index sets, and formalize the “conditional on $X$” version of Kolmogorov?
- Can you provide scaling plots (time/memory) vs ProFITi and others as $K$ and component count $D$ grow.
- Add other metrics (e.g., CRPS, Energy Score) in the main text to complement NLL.
- Include ablations: sensitivity to mixture size $D$, latent sizes, and spline bins; discuss the marginal-vs-joint performance trade-off and when query-independent mixing may hurt.

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
This paper provide a time series prediction method applied in irregular (missing) dataset. Requirements 1~3 satisfy the problems that irregular dataset facing. The key contribution of R3 is crucial for irregular forecasting because the consistency of probability of one variable if some observations are missed. For implementing abovementioned proposition, MOSES use Separable Normalizing Flows through dealing each variable individually. For forecasting, MOSES train a weight matrix to combine variables. The experiments results suggest MOSES take a large margin than baselines.

### Strengths
1.	The probability consistency is crucial for irregular forecasting.
2.	The pave to accomplish probability consistency is efficient.
3.	Verify the importance of probability consistency fully in toy dataset, and MOSESE make a big gap to baseline in four real world dataset.

### Weaknesses
1.	Though Kolmogorov’s extension theorem make sure the exchangeable of R3, it still need handle different size of subset for ensuring probability consistency.
2.	MOSES only ensure the probability consistency of hidden variable z, then bridge this property to y through function f. So the representability of enc might be vital.
3.	In line 257, author claim w depend only on hobs, not on queries. Whereas hobs come from X, and Q is a subset of X. So queries might effect w.

### Questions
1.	Ablation experiments may be conducted in ENC to demonstrate its effects.
2.     As Weaknesses.2 says, is there some method to make probability consistency not only in hidden variable z but also in observation y, or to bridge them to some degree?
3.	Though the frequency-time embedding method is employed in MOSES, I'm still concerned about the accuracy degeneration when query-t has a big gap between observe-t, because the query-t must be greater than observe-t which is used in the training stage.
4.	The time series generation model showed impressive performance in irregular datasets, the comparison with them may be interesting. 
Li Y, Lu X, Wang Y, et al. Generative time series forecasting with diffusion, denoise, and disentanglement[J]. Advances in Neural Information Processing Systems, 2022, 35: 23009-23022.
Narasimhan S S, Agarwal S, Akcin O, et al. Time weaver: A conditional time series generation model[J]. arXiv preprint arXiv:2403.02682, 2024.

### Soundness
3

### Presentation
3

### Contribution
3
