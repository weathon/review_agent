# Exact Rates and Saturation Effect of Kernel Ridge Regression over Unbounded Input Space in Large Dimensions

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
This work presents a theoretical analysis of Kernel Ridge Regression (KRR) in a large dimensional regime where both the input dimension $d$ and sample size $n$ grow, satisfying $n \asymp d^\gamma$ for some $\gamma > 0$.  We extend prior studies,  which focus on inner product kernels on the sphere $\mathbb{S}^{d-1}$, to broader classes of kernels defined on unbounded domains $\mathcal{X} \subseteq \mathbb{R}^d$. Suppose that the true function $f_{\rho}^* \in [\mathcal{H}]^s$, where $[\mathcal{H}]^s$ denotes the interpolation space of RKHS $\mathcal{H}$ with source condition $s>0$. Our primary contribution is a precise characterization of the generalization error of KRR, treating the cases $s \ge 1$ and $s < 1$ separately.

Surprisingly, after adopting the result to the Gaussian kernel in large dimensions and deriving precise asymptotics for the corresponding eigenvalues, our analysis of Gaussian kernel ridge regression reveals that it:
$i)$ achieves minimax optimality when $0 < s \leq 1$;
$ii)$ fails to attain the minimax lower bound for $s > 1$, demonstrating a $\textit{saturation effect}$ where additional smoothness beyond this point does not improve the convergence rate.
Furthermore, we identify two phenomena unique to the large dimensional regime: a $\textit{periodic plateau phenomenon}$ in the convergence rate and a $\textit {multiple-descent behavior}$ with respect to the sample size $n$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors investigate the error rates of kernel ridge regression in the large dimensional, polynomial regime $n\sim d^{-\gamma}$, under a source condition on the target function. Similarly to previous works, they derive the exact convergence rates, and the minimax rates, and evidence a multiple descent behavior. The central technical contribution lies in the fact that the present work succeeds in prescinding the need of assuming bounded (typically spherical) data distribution.

### Strengths
The paper is very clearly written, and addresses a problem of interest. The technical advance allows in principle to lift a rather restrictive data assumption common to many prior studies, e.g. Xiao et al, 2022, Zhang et al, 2024, and could prove of interest to the community. I have however not carefully read the proofs, and have limited familiarity with the technical tools the paper uses.

### Weaknesses
My main concern is the close proximity of the paper with the reference Zhang et al, 2024. While the authors carefully explain the difference between the current theorem 3.4 and theorem 1 in the latter, such detailed comparison is lacking in the discussion of subsequent results. For instance, the exact rates (Theorems 4.8, 4.10) and minimax lower bounds (Theorem 4.13) all appear, to the best of my understanding, in  Zhang et al, 2024, and the expressions are identical. A discussion of why this is the case, and high-level intuition, would be helpful. By the same token, the phenomenology (plateaus and multiple descent) discussed from line 414 seem to be unaltered from the spherical data case of Zhang et al, 2024. Thus, while the technical contribution may be interesting, there are limited new results (both in terms of rate and phenomenology) compared to the spherical case. I believe more comparison should at least be included, and would be willing to increase my score if the authors include further detailed discussion on the differences to the spherical case.

### Questions
- The similarity of the phenomenology to the spherical case could stem from the assumptions of isotropic Gaussian data, which while unbounded, is intuitively very close to a spherical distribution in high dimensions. Would it be possible to illustrate the results on another distribution, or would this be technically out of reach ?
- To the best of my understanding, (Pandit et al, Universality of Kernel Random Matrices and Kernel Regression in the Quadratic Regime, 2024) already consider potentially unbounded data (e.g. Gaussian), for the quadratic regime $n\sim d^2$.
- "their applicability to the most commonly used kernel, the Gaussian kernel, remains unverified" (l.70). I think the phrasing is slightly confusing, as one could very well apply a Gaussian kernel on spherical data.
- Is the setting of section 4 not satisfying the conditions of applicability of Theorem 1 in Zhang et al, 2024? Despite the discussion around l. 224, I do not believe this question is addressed. I am curious to know whether something fundamentally breaks in the unbounded case, warranting Theorem 3.4, or if the latter is just easier to verify.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the generalization error of kernel ridge regression in high dimensions for general data distributions on unbounded domains and without assumptions of the eigenfunctions of the kernel. This is an improvement over prior works that consider restricted data distributions, such as the sphere or hypercube, and/or inner product and hypercontractivity conditions on the kernel function. They apply their general result to the specific case of the Gaussian kernel and Gaussian data. As a result, the authors show the Gaussian kernel achieves minimax optimality when the target is sharper than the RKHS. The authors also show a saturation effect where KRR fails to achieve the minimax rate and does not see additional improvements in generalization error with smoothness above that of the RKHS.

### Strengths
-	The paper studies fundamental theoretical questions in machine learning: (1) How do non-parametric estimators generalize in high dimensions? What is the interaction of the smoothness of the target function with that of the function class used to estimate it?
-	The paper establishes satisfying results in these questions that seem to contribute substantially over prior works.

### Weaknesses
-	The main weakness is I would like to see some experimental evidence of the saturation effect for the setting considered in this work – Gaussian kernel and Gaussian data. In particular, it could be quite nice to have some plots illustrating why additional smoothness in the target function does not benefit the target estimate.

### Questions
-	Do we see saturation effects for non-Gaussian kernels? I suspect this may be a general phenomenon for kernel ridge regression.
-	Is there some intuition for why the saturation effect occurs? My feeling is if you know the target function is smoother than the minimum allowed by your function class, your function space is ‘too wide’ and you can truncate it without losing generalization error. It could be nice to see experimentally that trimming your function space improves error.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper first derives excess risk upper bounds for KRR on general domains with bounded kernels, expressed cleanly in terms of the kernel spectrum and source conditions. It then specializes to the high-dimensional Gaussian kernel, where it characterizes the saturation effect and documents the periodic plateau behavior of the convergence rates.

### Strengths
This paper presents a clear exposition with rigorous, carefully organized proofs. It extends prior high-dimensional KRR results from spherical inner-product kernels to kernels on general domains—a theoretically significant contribution that broadens applicability.

### Weaknesses
The only weakness of this paper is that it provides concrete high-dimensional rate calculations only for the Gaussian kernel.

### Questions
If the kernel is not Gaussian, will the periodic plateau rate curve and the source condition threshold for the saturation effect change?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper considers the generalization properties of kernel ridge regression in increasing dimension. The main result is a generalization error of KRR that can be applied to a broad class of kernels (and not only inner product kernels on the sphere as some of the previous). Under source conditions on the target depending on $s$, they characterize the error of kernel ridge regression. This bound applied to the Gaussian kernel with Gaussian data matches the minmax lower bound for large-dimensional Gaussian kernel regression for $\le s<1$ but not $s>1$ and shows that this kernel exhibits the periodic plateau and multiple descent phenomena.

### Strengths
1. The topic of the main result in the paper is central to the understanding of kernel ridge regression, and it is especially interesting as it explores non-typical settings beyond the inner-product kernels on the sphere.
2. The bounds on the eigenvalues of Gaussian kernel with Gaussian data are interesting in their own right and can further the understanding of the generalization properties of KRR.

### Weaknesses
1. A related and important line of work is missing from the discussion on related works and the context of the paper’s result -  the work on eigenframework (e.g. Simon et al “The Eigenlearning Framework: A Conservation Law Perspective on Kernel Regression and Wide Neural Networks” and many others). Further, the work on the deterministic equivalent of test risk (e.g. Misiakiewicz and Saeed “A non-asymptotic theory of Kernel Ridge Regression: deterministic equivalents, test error, and GCV estimator”) is only briefly discussed because it puts explicit assumptions on kernel eigenfunctions. Both of these lines of work give estimates for the generalization error of KRR with a general input space $\mathcal X$ and kernels beyond the inner product kernels. In particular, both lines of work give estimates on the test risk of KRR in the same form as Theorem 3.4. For example, Thereom 1 in Misiakiewicz and Saeed and the main equation of eigenframework form Simon et al look very similar to Theorem 3.4. It is essential to properly discuss the similarities and differences between this work and existing work, especially when they are so closely related. Furthermore, both of these lines of work consider quantities that are more general to the ones in Definition 3.1 and Eq. (4). Namely, the regularization parameter $\lambda$ is replaced with “effective regularization”. This makes me wonder if the result in this paper misses a key detail of the behavior of KRR. Not discussing the eigenframework is especially alarming.
2. The motivation for considering a Gaussian kernel with $x\sim N(0,\sigma^2 I_d)$ is unclear, even with the remark on line 251. The reasoning there (and in Remark 4.1) is mostly focused on explaining why the proof techniques won’t work in this case. The more interesting question (to me at least) is whether the behavior of Gaussian Kernel with Gaussian data in large dimensions is in any way different from an inner product kernel on the sphere. The discussion in lines 414-423 does not discuss the differences in the behavior but (if I understand correctly) only the similarities (that the periodic plateau phenomenon and multiple descent behavior were shown in these other papers that “consider large dimensional spectral algorithms on the sphere”). 
3. The limitations of the result could be expanded by a discussion of the phenomena the result does not capture. It is well known that KRR in increasing dimension n=d^{\gamma} can overfit benignly (e.g. Misiakiewicz 2022, Barzilai and Shamir 2023). The result in this paper does not discuss this regime, namely, the conditions on Eq (5) do not allow for setting $\lambda=0$. This is not the case for prior work (see weakness #1).

### Questions
1. In light of weakness #1, how is the main result of this paper (Theorem 3.4.) different/new compared to the already existing results on the deterministic equivalent of test error of KRR (Misiakiewicz and Saeed 2024, Theorem 1)? Are assumptions in Eq (5) not covered by the already existing results? If not, in what ways is it an improvement? Is it the avoidance of reliance on eigenfunctions? Because the result in Misiakiewicz and Saeed also does not rely on eigenfunctions.
2. Why are the $d^{\gamma}$-th moments of $X$ (scaled by $\sigma \sqrt{d}$) important to the generalization behavior of KRR (except in the particular proof approaches)? For large $d$, the distribution $x\sim N(0,\sigma^2 I_d)$ is close to $x\sim d*S^{d-1}$ because of CLT, and on $S^{d-1}$ the Gaussian kernel is an inner product kernel, so the Gaussian kernel considered here is also close to an inner product kernel up to scaling.
3. What other kernels except the Gaussian kernel with Gaussian data is the main result applicable to? 
4. Is there a kernel to which the main result applies that is qualitatively different from inner product kernels on the sphere? If not, then what is the contribution of the result?

### Soundness
2

### Presentation
1

### Contribution
1
