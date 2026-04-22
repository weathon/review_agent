# Bias-variance Tradeoff in Tensor Estimation

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
We study denoising of a third-order tensor when the ground-truth tensor is **not** necessarily Tucker low-rank. Specifically, we observe
$$
Y=X^\\ast+Z\in \\mathbb{R}^{p_{1} \\times p_{2} \\times p_{3}},
$$
where $X^\\ast$ is the ground-truth tensor, and $Z$ is the noise tensor. We propose a simple variant of the higher-order tensor SVD estimator $\\widetilde{X}$. We show that uniformly over all user-specified Tucker ranks $(r_{1},r_{2},r_{3})$,
$$
\\| \\widetilde{X} - X^\ast \\|^2_{\\mathrm{F}} = O \\Big( \\kappa^2 \\Big\\{ r_{1}r_{2}r_{3} + \\sum_{k=1}^{3} p_{k} r_{k} \\Big\\} \\; + \\; \\xi_{(r_{1},r_{2},r_{3})}^2 \\Big) \\quad \\text{  with high probability.}
$$
Here, the bias term $\xi_{(r_1,r_2,r_3)}$ corresponds to  the best achievable approximation error of $X^\ast$ over the class of tensors with Tucker ranks $(r_1,r_2,r_3)$;    $\kappa^2$   quantifies the noise level; and the variance term  $\kappa^2 \\{r_{1}r_{2}r_{3}+\sum_{k=1}^{3} p_{k} r_{k}\\}$ scales with the effective number of free parameters in  the estimator $\widetilde{X}$. Our analysis achieves a clean rank-adaptive bias-variance tradeoff: as we increase  the ranks of estimator $\widetilde{X}$, the bias   $\xi(r_{1},r_{2},r_{3})$ decreases and the variance increases. As a byproduct we also obtain a  convenient bias-variance decomposition for the vanilla low-rank SVD matrix estimators.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is focused on denoising a third-order tensor signal corrupted by additive noise, by means of a low-multilinear-rank approximation.

Its main message is that, by employing a simple spectral algorithm that is essentially a one-step HOOI initialized by truncated HOSVD (THOSVD), the resulting approximation error (with respect to the sought signal) can be decomposed into the sum of a bias term which decays in (each component of) the multilinear rank of the model, plus a variance term which grows in (the components of) the rank.

### Strengths
1) This bias-variance analysis is certainly novel and of interest to the community.

2) The raised connections with existing results, in particular those of Zhang and Xia (2018), are also of interest.

### Weaknesses
The paper has several presentation issues, thus needing substantial polishing. My main concerns in this regard are:

1) The designed algorithm per se is not novel, simply being a variant of a one-step HOOI initialized by truncated HOSVD (THOSVD).

2) The paper fails to cite and acknowledge important results and previous work on this topic, notably the fact that the THOSVD and similar algorithms such as the sequentially truncated HOSVD are quasi-optimal, as shown in the relatively celebrated paper:
[Vannieuwenhoven, N., Vandebril, R., & Meerbergen, K. (2012). A new truncation strategy for the higher-order singular value decomposition. SIAM Journal on Scientific Computing, 34(2), A1027-A1052.]
In particular, some of the proof techniques used in this paper, such as the telescoping sums introduced to control the approximation error by multilinear projection (see Appendix A), seem to be inspired by the classical proof of this quasi-optimality result, as given in the above paper. These connections should be pointed out. 

3) The numerical experiments are in my view ill-designed, and the corresponding section is not sufficiently clear:
- In Section 4.1, it is quite difficult to see differences among the given images. The paper should include zoom boxes or arrows for highlighting specific details which support the conclusions, or even change the parameters in order to illustrate its findings in a more clear way. Also, why does the discussion here stops short of considering high values or rank $r$ for seeing its effect on the variance? 
- While the paper focuses on the Tucker model, the synthetic tensors described in 4.2 follow instead an orthogonal canonical polyadic decomposition (due to the diagonal core). Why that choice? Moreover, this model is exactly low-rank (as seen in Table 1, which by the way is redundant since this information is already shown in Table 2), contrarily to the whole point of the Introduction of handling the case of approximately low-rank signals.
- The following conclusion given in Section 4.2 is quite vague, and makes us think that the goal here is to evaluate the given algorithm instead of illustrating the bias-variance tradeoff: "We observe that the error consistently decreases as the SNR parameter $\lambda$ increases. Overall, one-step HOSVD is robust across the tested sizes and ranks, yielding accurate estimates on the synthetic tensors." Yet, if that is the case, then a comparison with other similar algorithms (at least THOSVD as a baseline) is required. But in my view, given that the algorithm is not novel, the paper should rather focus on showing how the utility of the derived result on the bias-variance tradeoff, for instance by plotting separate curves for the bias and the variance terms as a function of the ranks and showing that they follow the predicted trends (bias decay and variance growth) with the correct scaling.
- By the same token, if an exactly low-rank model is to be used, then experiments such as those reported on Table 2 should not stop at a model rank $r$ below the true rank $s$: this would allow seeing the effect of a rank overestimation on the variance.

4) The paper states that the results given for the matrix are a "byproduct" of the tensor analysis, but this claim seems misleading to me. The statement of Theorem 2 is not a particular case of Theorem 1; these results are in fact of a quite different nature: Theorem 1 has a probabilistic flavor and requires a spectral gap condition in order to hold, while Theorem 2 is completely deterministic and valid for any signal. Even the proof arguments are quite dissimilar.

5) Last but not least, a careful revision is in order. In particular:
- Several results referred to as Theorems (for instance, in lines 280 and 281) are actually other types of results (Lemma, Corollary).
- It seems that the constant in (6) should be denoted by C, according to the discussion in lines 242-243.

### Questions
1) Would it be possible have a deterministic result similar to that of Theorem 2 for tensors, with a bound depending on the model degrees of freedom and, say, some norm (spectral?) of the noise tensor?

2) The paper argues that its results extend in a way those of Zhang & Xia, and in particular match those previous results in the case of an exactly low-rank model. Have the authors also checked that their bounds match the known results for rank-one tensor PCA? Namely, for $p_1=p_2=p_3=p$, an appropriately normalized noise (say, $\kappa = 1/\sqrt{p}$) and $r_1=r_2=r_3=1$, the condition on the signal magnitude for having an $O(1)$ error should match the known conjectured computational threshold $p^{1/4}$. It seems to me that this is the case, which would be interesting to point out.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studies the problem of noisy tensor estimation when the the ground-truth tensor is not necessarily Tucker low-rank. By utilizing some  classical linear algebra results such as Mirsky’s and Ky Fan’s theorems, the authors derive a clean rank-adaptive bias–variance tradeoff. A series of simulations is carried out to confirm the existence of tradeoffs across different regimes of spectral decay and noise.

### Strengths
1. A new tensor estimation setting has been not considered in the literature. 
2. A rigorous theoretical analysis is carried out to show clean the rank-adaptive bias–variance tradeoff in such a tensor estimation problem. 
3. As a byproduct, a convenient bias-variance tradeoff in matrix estimation has been obtained.

### Weaknesses
1. The minx lower bounds for both tensor and matrix cases are not provided. 
2. The main theoretical derivations seem to be direct extensions of the work Zhang & Xia (2018).
3. How to tune the target Tucker rank for the proposed Algorithm 1 has not been provided.

### Questions
1. Why the authors consider the simple one-step HOSVD algorithm rather than the popular HOOI algorithm?
2. Can the proposed proof strategy be readily extended to accommodate general $d$th-order tensors, and if so, what specific steps or modifications would be required to achieve this?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proves a new bias-variance tradeoff for Tucker decompositions of general third-order tensors.  No assumption is made on the input tensor being close to low-rank, which makes the results highly applicable to real data.  The authors provide demonstrations with real datasets to reinforce this.  Overall, this is a very nice submission.

### Strengths
- No assumption is made on the input tensor being exactly low-rank or close to low-rank.  This makes the result applicable to any input tensor, and thus very useful.

- As the authors explain, the proven bound is optimal up to constants.

- The authors show that the HOSVD algorithm achieves the bound.

- There are nice real data illustrations.

- The paper is very clearly written.

### Weaknesses
- It would have been nice if the authors extended their analysis to tensors beyond third order tensors.

- A few useful references are missing.

### Questions
(1) Line 35: missing both older and more recent papers on latent variable learning.  Obviously the authors should cite 
	Animashree Anandkumar, Rong Ge, Daniel J Hsu, Sham M Kakade, Matus Telgarsky, et al. "Tensor decompositions for learning latent variable models." J. Mach. Learn. Res., 15(1):2773–2832, 2014.  
here instead of the community detection part.
I also suggest the authors cite 
	Yifan Zhang, Joe Kileel, "Moment estimation for nonparametric mixture models through implicit tensor decomposition", SIAM Math. Data Sci. 5.4 (2023), 1130-1159.
as well as 
	Samantha Sherman, Tamara G. Kolda. "Estimating higher-order moments using symmetric tensor decomposition." SIAM Matrix Anal. Appl. 41.3 (2020), 1369-1387.



(2) Lines 43-50: regarding the low-rank Tucker tensor decomposition, the authors should cite 
	Ruhui Jin, Joe Kileel, Tamara G. Kolda, Rachel Ward. "Scalable symmetric Tucker tensor decomposition." SIAM Matrix Anal. Appl. 45.4 (2024), 1746-1781



(3) Can the authors add some remarks on how the analysis should go for higher than third order tensors?

### Soundness
4

### Presentation
4

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
This paper investigates the problem of denoising a third-order tensor $Y = X^* + Z$, where $Y$ is the observed noisy tensor, $ X^* $ is the ground-truth signal, and $ Z $ is a noise tensor. Existing work mainly assumes that $ X^* $ is of low Tucker-rank as compared to the dimension. In contrast, this paper does not impose this assumption, and considers a simple variant of the HOSVD algorithm. They show that the recovery error of this algorithm can be upper bounded by the sum of two terms: a bias term that depends on the best achievable estimation error of $X^*$ over the space of tensors of low Tucker-rank $(r_1, r_2, r_3)$, a variance term that characterizes the estimation error if the true signal itself is indeed of Tucker-rank $(r_1, r_2, r_3)$. 

As a byproduct, the paper provides a similar (and even simpler) non-asymptotic bias-variance decomposition for the standard truncated SVD estimator for matrices.

### Strengths
This paper's main result (Theorem 1, the bias-variance tradeoff error bound) is novel and interesting, as it extends the exact low-rank setting, which is common in existing literature, to the approximate low-rank setting. This is a natural generalization and one would expect both the variance and bias terms in the error bound to be optimal up to a constant. Further, this error bound holds uniformly for all low Tucker-rank tensors, making it adaptive to all target Tucker ranks. The paper is also clearly written.

### Weaknesses
(i) Given the upper bound on the variance term in existing literature, it seems to me that the main contributions of this paper is a bit incremental, especially that the proof techniques are also similar.
(ii) Theorem 1 has an assumption on the singular value gap of the true signal $X^*$. Is this generally satisfied in practice? It would be better to give some examples for which this assumption holds.

### Questions
Is it possible to compute the bias term $\xi (r_1, r_2, r_3)$ for some examples? Again, this would be helpful in understanding how the approximate low-rank case compares to the exact low-rank case.

### Soundness
4

### Presentation
3

### Contribution
3
