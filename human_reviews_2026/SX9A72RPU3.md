# Structured covariance estimation via tensor-train decomposition

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
We consider a problem of covariance estimation from a sample of i.i.d. high-dimensional random vectors. To avoid the curse of dimensionality, we impose an additional assumption on the structure of the covariance matrix $\Sigma$. To be more precise, we study the case when $\Sigma$ can be approximated by a sum of double Kronecker products of smaller matrices in a tensor train (TT) format. Our setup naturally extends widely known Kronecker sum and CANDECOMP/PARAFAC models but admits richer interaction across modes. We suggest an iterative polynomial time algorithm based on TT-SVD and higher-order orthogonal iteration (HOOI) adapted to Tucker‑2 hybrid structure. We derive non-asymptotic dimension-free bounds on the accuracy of covariance estimation taking into account hidden Kronecker product and tensor train structures. The efficiency of our approach is illustrated with numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper extends previous approaches based on low-rank tensor-structured models for covariance matrix estimation. Specifically, it introduces a Tucker-2 structured covariance matrix model, as per (5), which has a higher degree of flexibility than the existing sum-of-Kronecker-products model (2).

It then resorts to a quite standard high-order orthogonal iteration (HOOI) algorithm (Algorithm 1) initialized by a truncated high-order SVD (HOSVD), with respect to two modes (in line with the Tucker-2 structure).

Its main result (Theorem 2.2) is then a bound on the quadratic estimation error (with respect to the true covariance) made by Algorithm 1. This bound and the conditions of Theorem 2.2 are stated without reference to the ambient space dimension, but rather involve three introduced quantities ($r_1, r_2, r_3$) which are said to play the role of "effective dimensions".

A set of numerical experiments is described in the paper, but only involving synthetic, randomly generated covariance matrices having the postulated structure by construction.

### Strengths
The proposal of new, richer tensor model structures for learning high-dimensional covariance matrices is certainly of some interest, as well as deriving dimension-free guarantees regarding the approximation error delivered by a simple approximation scheme.

### Weaknesses
1) My main concern is that, in spite of the interest of its proposal, this paper suffers from serious readability and clarity issues:
- When referring to the model, there is a constant ambiguity between Tensor Train (TT) and Tucker-2 structure, with the terminology going back-and-forth along the paper, which does not help. I don't see why one should use the former terminology, since the model is quite clear a Tucker-2 model with two matrix factors and a core tensor. This is how it is written in (9) and how Algorithm 1 is devised. If the authors wish to point to a connection with the TT model, I suggest that they use some remark to do it, but keep the discussion focused on the Tucker-2 structure for clarity.
- The claim that the CP is a special case of (5) is far-fetched, since in (5) if one eliminates one of the summations (say, setting $J=1$), then one gets a special CP in which the first vector is repeated in all terms. Moreover, this claim is somewhat misleading in the sense that it might lead the reader to think that an algorithm capable of estimating a model with the structure (5) would also be able to handle the CP as a special case. The TT model cannot be said to be a more general model than CP, it is simply a different one. In fact, it turns out that several different tensor models can be thought each one as a special case of the other (CPD, Tucker, BTD, ...), but in terms of actually estimating these models, the algorithms are specialized, and the properties and issues faced in practice are not the same.
- The multilinear transformation (such as in (9)) is not denoted as usual (with the core tensor coming first, and then the transforming matrices).
- Line 240 refers to a randomized truncated SVD, while the operator SVD$_J$ is defined before as a standard truncated SVD. This causes confusion, since the paper never discusses the use of a randomized SVD, and possible errors due to randomization do not seem to be taken into account in the analysis.
- The statement of Theorem 2.2 is very cumbersome and hard to parse and interpret. I believe that further work is needed to simplify/distillate as much as possible that statement and thereby render it more accessible/useful.
- What is "the variance part" of $\tilde{r}_T$ mentioned in line 305?
- Why is homoscedastic used to mean i.i.d. in line 336, when it actually means "of equal variance"?
- It is hard to conclude anything from the discussion on the choice of $J$ and $K$. Are those given estimators practical? How can $C'$ be inferred?
- Why are singular values referred to as "singular numbers"? This terminology sounds very unusual to me. 

2) Algorithm 1 is not novel at all, and cannot be claimed to be proposed by the authors, as done in the Conclusion. This algorithm is in fact just a standard high-order orthogonal iteration (HOOI) applied to a Tucker-2 model for estimating its two factors and core, and initialized in a very usual way by means of a truncated high-order SVD. 

3) The contents of the numerical experiments are insufficient in my view, mainly because only random synthetic examples matching the postulated model are considered. This is of course useful for measuring the error with respect to the ground truth, but the whole motivation discussed in the Introduction is to introduce model (5) for greater flexibility with respect to previous approaches. Hence, a comparison with other models using real-world sample covariance matrices should be given.

4) Finally, a careful revision is needed regarding English usage. Some examples: "yielding rough variance proxy..." (line 324); "impact of ... to the perturbation" (lines 337-341); "for $n$ one has bounds" (line 360); "which diagonal" (line 395).

### Questions
1) How should the condition stated in (11) be compared to that in your result?

2) I do not understand the claim "This highlights the difference between low-rank tensor estimation problem and low-rank matrix estimation problem, since for the latter there is no significant difference between soft-thresholding and hard-thresholding estimation." Estimators relying on soft- and hard-thresholding singular values can behave quite differently depending on the example at hand. What do you mean exactly here?

### Soundness
2

### Presentation
1

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
The paper studies high-dimensional covariance estimation under a flexible structural assumption that the covariance matrix can be well approximated by a sum of double Kronecker products arranged in a tensor-train (TT) format. Concretely, the model writes $\Sigma$ as a sum over factors $U_j\otimes V_{jk}\otimes W_k$ with TT ranks $(J, K)$, which strictly generalizes classical Kronecker-product and CP covariance models and allows richer cross-mode interactions. The authors show that $\Sigma$ can be rearranged into a third-order tensor $\mathcal{R}(\Sigma)$ that admits a Tucker-2 structure, and they propose a polynomial-time estimator based on iterative truncated SVDs on the mode-1 and mode-3 unfoldings. The algorithm, named HardTTh (Hard Tensor-Train Thresholding), alternates subspace updates via rank-$J$ and rank-$K$ truncated SVDs and then projects the observed tensor onto the estimated subspaces.

On the theoretical side, under a mild sub-exponential-type assumption on the standardized data (a Hanson–Wright-style moment generating function bound), the paper derives non-asymptotic, high-probability, dimension-free Frobenius error bounds for the estimator. The bound separates into a bias term (capturing model misspecification), a leading variance term that scales as $\omega||\Sigma||$ times a sum of effective dimensions divided by $n$, and smaller remainder terms that decay with the number of iterations. The effective dimensions are defined via norms of partial traces of $\Sigma$ and capture intrinsic complexity of the covariance structure, analogously to effective rank in unstructured covariance estimation. The analysis also improves on prior Kronecker-sum results by expressing the variance in terms of $||\Sigma||$, and it explicitly treats misspecification. Experiments on synthetic data show that HardTTh substantially improves over one-shot TT-HOSVD when spectral gaps are reasonable and the sample size is moderate-to-large, and it attains Tucker+HOOI accuracy with noticeably lower runtime. The results also illustrate that when spectral conditions fail (e.g., small $n$ or weak signal), iterative refinement does not help, which matches the theory’s requirements.

### Strengths
Originality. The paper introduces a well-motivated TT-structured covariance model that subsumes widely used Kronecker and CP models while enabling richer multiway dependencies. It proposes an efficient HOOI-like estimator specialized to Tucker-2/TT covariance structure, and it develops a nontrivial analysis that yields dimension-free guarantees based on partial-trace effective dimensions. This combination of modeling, algorithmic design, and analysis is novel and meaningful.

Quality. The theoretical development is careful and modular. The authors first establish a deterministic perturbation bound for their iterative SVD procedure and then derive high-probability error bounds using a PAC–Bayes approach tailored to the structured noise arising from sample covariance tensors. The bias–variance decomposition is explicit, misspecification is handled cleanly, and the variance term scales with interpretable effective dimensions. The algorithmic complexity is analyzed, showing practical scaling with truncated randomized SVDs. The experiments are designed to probe when additional iterations help, how performance varies with sample size, and why soft-thresholding (PRLS) can be suboptimal in multi-mode settings.

Clarity. The paper uses heavy notation, but it is consistent and well-documented. The rearrangement operator and mode-wise unfoldings are clearly defined, and the key algebraic identities are stated. The main theorem is presented with a helpful collection of ancillary terms. The proof sketch guides the reader through the argument, and the appendices provide the necessary technical details.

Significance. The work bridges an important gap by providing a computationally efficient covariance estimator for multiway structured models with non-asymptotic, dimension-free guarantees beyond simple rank-one settings. It contributes a practical alternative to CP/nuclear-norm approaches that face computational hardness, and it extends effective-dimension ideas from unstructured covariance estimation to a richer tensor setting. Given the prevalence of multiway data (e.g., spatio-temporal, multi-sensor, MIMO), the proposed framework has the potential for notable impact.

### Weaknesses
1. The main theoretical guarantees rely on lower bounds on the $J$-th and $K$-th singular values of the mode-1 and mode-3 matricizations of $\mathcal{R}(\Sigma)$. While these conditions are standard in tensor estimation, they may be restrictive in some realistic regimes. The paper acknowledges this statistical–computational gap, but it would benefit from more actionable guidance on how practitioners can assess these conditions from data.

2. The analysis indicates that the noise structure induced by sample covariance is heteroscedastic, and that an ideal version of the algorithm would include a debiasing step before applying SVD. The present work leaves such debiasing to future research. Even a pragmatic approximate debiasing scheme could strengthen the practical contribution.

3. The proposed data-driven selection of $J$ and $K$ depends on $\omega$ and on concentration of partial traces, and the paper defers establishing the needed concentration bounds. An empirical study examining the stability and accuracy of the rank selection rule would make the method more turnkey.

4. All experiments are synthetic. Although they are informative, adding at least one real-data example would improve the credibility and applicability of the method and help illustrate interpretability of the learned structure.

5. The effective dimensions $r_i(\Sigma)$ are insightful but somewhat abstract. The paper could provide more intuition through examples where $r_i(\Sigma)$ is demonstrably much smaller than ambient dimensions, helping readers anticipate when the dimension-free rates will be most favorable.

### Questions
1. How robust is HardTTh to mild violations of the spectral gap conditions? It would be helpful to include experiments where the signal-to-noise ratio is near the threshold and to quantify the degradation of the subspace estimates and final error.

2. For the rank selection scheme, $\omga$ may be unknown. Do you suggest estimating $\omega$ from the data (e.g., via Gaussian calibration of quadratic forms or via residual-based methods), or using data splitting or bootstrap techniques to tune the thresholds? An empirical comparison between fixed ranks and the proposed selection (reporting rank recovery rates and sensitivity to n) would be very helpful.

3. The bias term involves suprema over orthonormal $U$ and $V$ of the projected misspecification energy. Can you provide more interpretable sufficient conditions that translate into explicit bias rates, such as tail bounds on singular values of the mode unfoldings or decay conditions on TT cores?

4. In practice, when should one prefer the proposed TT covariance model over CP or pure Kronecker models? Brief guidelines contrasting modeling flexibility, sample complexity, and computational demands across these alternatives would aid practitioners.

5. Do your methods and theory extend cleanly to orders higher than three (e.g., four-way TT covariance)? Which parts of the analysis (partial trace definitions, PAC–Bayes steps, recursion) generalize directly, and where would new technical work be needed?

6. In the Gaussian setting, can sharper constants or reduced logarithmic factors be achieved using exact Hanson–Wright or Gaussian isoperimetric inequalities? If so, would this change the iteration complexity needed for the remainder terms to be dominated?

7. On the empirical side, it would strengthen the paper to add a real-data case study (even small-scale), report measured proxies of the effective dimensions, and provide ablations on the number of iterations, randomized SVD settings, and misspecified ranks to guide practical deployment.

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
3

### Summary
The paper develops a new structured covariance estimation framework based on tensor-train (TT) decomposition, extending Kronecker and CP-type models. It proposes a polynomial-time algorithm, HardTTh, and provides dimension-free non-asymptotic error bounds under assumption 2.1, in particular under sub-Gaussian assumptions. Experiments on synthetic data show performance improvements over Tucker/HOOI and PRLS baselines.

### Strengths
**Originality:**

Combining TT decomposition with covariance modeling is mathematically elegant.

Rigorous derivation of dimension-free Frobenius error bounds, uncommon in tensor-structured covariance literature.

**Quality:**

Theoretical analysis is rigorous, clearly distinguishing bias and variance contributions to the estimation error.

**Clarity:**

Notation is heavy but consistent; theoretical results are clearly labeled and contextualized.

Numerical results are clearly tabulated, showing systematic improvement with increasing sample size.

**Significance:**

Provides the first dimension-free rates for a tensor-train covariance estimator.

The balance between statistical guarantees and computational feasibility could influence future research on structured covariance learning.

### Weaknesses
All experiments are toy synthetic, with modest dimensions (≈ 10³ parameters). There is no demonstration on real, large-scale, or practical ML datasets. 

The contribution is primarily theoretical. While interesting for statistical estimation, it lacks connection to modern ML problems (representation learning, deep generative models, etc.). 

Unproven scalability:
Complexity analysis is asymptotic, there is no empirical scaling or memory footprint evaluation for large d. This makes it hard to evaluate its practicability for $d>10^4$ 

In the experiments, the algorithm assumes known TT-ranks, the proposed selection rule (Eq. 13) is untested experimentally. 

The theoretical guarantees rely on sub-Gaussian assumptions and large spectral gaps. No robustness analysis for real conditions is provided.

### Questions
Can the approach handle non-Gaussian or heavy-tailed data in practice?

Have you attempted any real data (e.g., spatiotemporal or vision tensors)?

How sensitive is performance to the choice of ranks J,K?

How does HardTTh scale when the tensor shapes exceed a few hundred?

Could this framework integrate into neural-network covariance or feature-space modeling (feature covariance regularization in deep embeddings, low-rank approximations of neural Fisher information matrices, structured priors in Gaussian processes or diffusion models or covariance-aware adaptation in federated or meta-learning)?

### Soundness
3

### Presentation
3

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
The paper studies covariance estimation from i.i.d. samples under a structured model for the covariance. Namely, the authors assume that the covariance is the sum of a small number of terms, with each of them given by the tensor product of three matrices. 
Under suitable rearrangement of the entries, such a matrix can be written as a low rank tensor of order three. Hence the matrix estimation problem can be recast as a low-rank tensor estimation problem.
The author study an algorithm based on tensor unfolding and tensor power iteration, and derive a dimension-free bound on the estimation error of this algorithm under a condition on the singular values of the components of the covariance.

### Strengths
The paper is clear.
The results seem to be solid although the assumption on the effective dimensions is possibly suoptimal.

### Weaknesses
1. The model is a generalization of earlier models for matrices with tensor product structure. The new model is obviously more expressive than earlier ones,  but expressivity could have been achieved in other ways. No substantive motivation is provided for studying this specific model.
2. The mapping onto low-rank tensor denoising is obvious and once this is done, the application of tensor estimation method is straightforward (of course bounding the estimation error is non-trivial).

### Questions
Related to the above.
1) Is there a substantive motivation for studying this model?
2) Is there a difference in the estimation algorithm with respect to earlier work in tensor estimation?
3) It would be useful to describe the proof strategy in the main text.

### Soundness
4

### Presentation
3

### Contribution
2
