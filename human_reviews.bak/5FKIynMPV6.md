# Bounds on the Reconstruction Error of Kernel PCA with Interpolation Spaces Norms

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 3, 6, 8

## Abstract
In this paper, we utilize the interpolation space norm to understand and fill the gaps in  some recent works on  the reconstruction error of the  kernel PCA. After rigorously proving a simple but fundamental claim appeared in the kernel PCA literature, we  provide  upper bound and lower bound of the reconstruction error of the empirical kernel PCA with interpolation space norms under the assumption $(C)$, a condition which is taken for granted in the existing works. Furthermore, we show that the assumption $(C)$ holds in two most interesting settings ( the polynomial-eigenvalue decayed kernels in fixed dimension domain and the inner product kernel on large dimensional sphere $\mathbb S^{d-1}$ where $n\asymp d^{\gamma}$) and compare our bound with the existing results. This work not only fills the gaps appeared in literature,  but also derives an explicit lower bound on the sample size to guarantee that the (optimal) reconstruction error is well approximated by the empirical reconstruction error. Finally, our results reveal that the RKHS norm is not a relevant error metric in the
large dimensional settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on understanding and bounding the reconstruction error of kernel Principal Component Analysis (PCA) using interpolation space norms. This work fills gaps in previous studies on kernel PCA by providing rigorous proofs and new bounds on the reconstruction error under specific conditions. Key contributions include: 1.Upper and Lower Bounds on Reconstruction Error; 2.Applications to some interesting settings including Fixed Dimension Domain, for polynomially eigenvalue-decayed kernels and High-Dimensional Sphere, for inner-product kernels where the dimension grows along with sample size. Moreover, he paper reveals that using [\mathcal{H}]^{1}-norm in high-dimensional settings may be unsuitable due to inconsistent error behavior. In addition, a "periodic plateau" phenomenon in convergence rates is observed, where the reconstruction error rate stabilizes over certain intervals as the number of components (\ell) changes.

### Strengths
The most significant strength is I think overall the paper addresses a gap that exists among the most recent works in a rigorous and inspiring way. It involves both the rigorous theoretical contribution mentioned above in the summary but also provides a novel use of Interpolation Norms that I personally find helpful and interesting for technical proofs. For applications, the high-dim behavior insights mentioned above are also of great practical importance given the topic kernel PCA is a practically popular method. Besides, the paper is well distinguished from the existing works. The paper offers a thorough comparison with existing bounds, demonstrating improvements over previous results and discussing where previous work lacked rigor. This transparency about advancements and limitations strengthens the credibility of the results.

### Weaknesses
Overall, the paper has few weaknesses. One point can be that its results are largely theoretical, with only limited empirical validation. More comprehensive experiments across various datasets and settings would strengthen the paper by providing practical evidence to support the theoretical claims. Since after kernel PCA is a widely applied method, adding various types of empirical behavior would definitively make the paper more appealing. The other point, which would rather be some improvements, is more from a practical point of view that for example, there are parameters say $s$ in 3.4 Corollary is in practice unknown. How to do adaptation to find a data driven $n$ is also needed here as there are some works focusing on adaptation on smoothness parameters in terms of estimation and regression etc. In general, the weaknesses mainly lie in the practical side (not the main focus of the paper), which however the strengths far exceed.

### Questions
1. Can you please elaborator more on your paper comparing with "Rosasco L, Belkin M, De Vito E. On Learning with Integral Operators[J]. Journal of Machine Learning Research, 2010, 11(2)", where the kernel $\Sigma$ in your paper is actually an integral operator?

2. How about extending the current result to the manifold setting? As there are existing results of the convergence of some particular kernel based Laplacians on the manifolds.

3. Given similarity between MDS and kernel PCA, would the error bounds work also for MDS?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper examines the reconstruction error of kernel Principal Component Analysis (PCA) using interpolation space norms. The authors derive upper and lower bounds for the reconstruction error of empirical kernel PCA under specific conditions. They apply these bounds to two scenarios: polynomial-eigenvalue decayed kernels in a fixed-dimension domain, and the inner product kernel on a high-dimensional sphere, comparing their bounds to existing results. Notably, this work establishes a lower bound on the sample size necessary to ensure that the empirical reconstruction error approximates the optimal reconstruction error accurately. Additionally, the authors conclude that the $H^1$-norm is unsuitable for large-dimensional settings.

### Strengths
The paper provides a solid theoretical analysis of kernel PCA within the framework of a generalized norm, referred to here as the interpolation space norm.

### Weaknesses
The presentation is suboptimal, making the paper challenging to read in its current form. There are multiple instances where notations or concepts are referenced before they are formally defined, impeding the reader’s ability to verify the correctness of the claims.

### Questions
- In page 1, notation $\otimes_H$ appears without definition. Also it is unclear whether $f(X)$ represents a vector or a matrix. Please clarify this notation and provide definitions for these terms.

- In page 3 there are discussions about the interpolation space norm, but this norm has not yet been defined. It’s difficult to follow the paper with references to undefined terms. Furthermore, a motivating explanation in the introduction about the significance of the interpolation space norm would be helpful. Why is this norm important, and how does it enhance the understanding of kernel PCA?

- What specific norm is used in equation (2)? Is it Frobenius norm?

- The presentation of this proposition could be improved. It references "condition (2.11) in Reiß & Wahl (2020)," yet does not restate the condition. Including the condition here would make the proposition more self-contained.

- In remark 2.3, what is meant by H norm?

- On page 5, inclusion map is not defined.

- In line 301 it is written $\langle \lambda_i^{(s-1)/2} \phi_i, \lambda_j^{(s-1)/2} \phi_i \rangle_{[H]^s} = \delta_{ij}$. Is this a definition of this inner product or is it deduced from some other fact or property? For example shouldn't the right hand side be $\lambda_i^{s-1} \delta_{ij}$ instead?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work is concerned with kernel PCA, and specifically with the statistical performance of the empirical estimator, defined as the $\ell$ principal components of the empirical kernel matrix $\widehat{\Sigma}_{ij} = k(X_i, X_j)$.
The main contributions are upper bounds on the empirical estimator's reconstruction error as a function of $\ell$, for several error metrics, and under several statistical settings.

The error metrics considered are the interpolation space norms $[H]^s$ for $0 \leq s \leq 1$, defined in Section 2.4. For $s=0$ this amounts to considering the estimator's $L^2$ reconstruction error, and for $s=1$ it amounts to the RKHS-norm reconstruction error.

There are three statistical settings considered:
- An abstract setting (section 3.1) where the only assumption is a condition referred to as "assumption $(C)$" in the paper. The following subsections make use of this abstract result.
- The classical setting (section 3.2) where dimension $d$ is constant and where the kernel $k$ has polynomially decaying eigenvalues.
- A high-dimensional setting (section 3.3) where sample size $n \asymp d^\gamma$ for some fixed $\gamma>1$.

Another contribution is a rigorous proof that the minimum $[H]^s$-norm reconstruction error admits a simplified expression for all $0 \leq s \leq 1$ (Theorem 2.5).

A secondary contribution is the remark that, in the high-dimensional setting of section 3.3, the RKHS-norm reconstruction error _of any estimator_ does not vanish as $n \to \infty$, implying that this error metric is unsuitable in this setting. Moreover, high-dimensional kernel PCA is shown to exhibit a similar phenomenon as in high-dimensional kernel regression: the periodic plateau behavior.

### Strengths
I am unable to assess the originality of this work w.r.t related literature, as I am not sufficiently familiar with this literature.

Identifying the important role played by "assumption (C)" in statistical analyses of kernel PCA is an interesting point.

The paper is very well structured and easy to follow.

### Weaknesses
In the proof of Theorem 2.5, I don't understand why the orthonormality constraint $(\psi_1, ..., \psi_\ell) \in B_\ell$ does not appear in the stationarity condition of $\mathcal{R}_s$, equation (6). In fact I did not manage to recover equation (6) at all. This step of the proof deserves more details.

The last sentence of the abstract is not very clear, and "$[H]^1$ norm" is not defined at that stage. Perhaps a more precise statement would be that "the RKHS norm is not a relevant error metric in high dimensions".

The figures are difficult to read, the paper would greatly benefit from making them bigger. (E.g with matplotlib, reduce figsize and increase dpi.)

The paper contains many typos and unusual wordings:
- line 17, in the abstract, remove space after opening parenthesis
- line 118, add "In", or use "contain"
- line 343, the statement of assumption (C), remove "If"
- line 376, replace "interested" by "interesting"
- throughout, consider using the word "setting" in place of "circumstance", which is less commonly used
- line 424, remove space after opening parenthesis
- line 470, add "A" at the beginning of the sentence
- line 531, replace "decayed" by "decaying", and "provide" by "provided"
- line 537, replace "on" by "of"

### Questions
- Please address the missing step in the proof of Theorem 2.5 (see Weaknesses).
- In Corollary 3.4, what is the dependency of the constant $C_3$ on problem parameters? is it polynomial?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper gives bounds on the recontruction error of kernel PCA, measured in the full scale of interpolation spaces $[H]^s$ between $L_2$ ($s=0$) and the RKHS ($s=1$). 
Analogous results exist in the literature for $s=0$ and $s=1$. However, the paper identifies a gap in the existing proofs for $s=0$ and gives an alternative, correct proof. Moreover, the results for $0<s<1$ are completely novel up to my knowledge, and they correspond to the existing ones in the limiting cases.

### Strengths
- The topic is of current interest, and the results are clearly presented, discussed, and placed in the context of the existing literature. 
- The new bounds extend the existing results to any $0\leq s\leq 1$. The extension is significant and relevant for applications.
- The identification and correction of a bug in the result $s=0$ is interesting and relevant to the community.

### Weaknesses
- The results in large dimensions hold on the sphere, as is clearly explained in Section 3.3. and especially in Theorem 3.8. This is a reasonable setting, but it should be made clear upfront in the introduction (e.g., in Table 1 and in Section 1.2 under "Convergence rate of empirical kernel PCA in large dimensions"). It would also be interesting to know more precisely what are technical limitations beyond the sphere?
- The results are in part motivated by filling gaps in the existing literature, namely those discussed in Appendix C1 and Appendix C2. To the best of my understanding, these gaps are identified correctly. But the claim is quite significant, and it should be better discussed in the main text. Also, according to the two appendices, the existing arguments fail for very specific corner cases: An effort should be made to clarify if these are cases of general interest.
- There are results in the approximation theory literature that seem to be closely related and should be discussed, e.g. Theorem 3 in [1] seems to prove a version Theorem 2.5 for s=0. See also [2].


[1] G. Santin and R. Schaback, Approximation of eigenfunctions in kernel-based spaces, Adv. Comput. Math. (2016)

[2] I. Steinwart, A short note on the comparison of interpolation widths, entropy numbers, and Kolmogorov widths, J. Approx Theory (2016)

### Questions
Besides the points discussed above, there are the following minor points: 
- Several constants are used in the introduction (Section 1) without being introduced. This makes the discussion sometimes difficult to follow.
- What does condition (2.11) in Reiss and Wahl (2020), quoted in Proposition 2.2, mean?
- Figure 1: The experimental setup is missing here, and it's unclear whether the plots correspond to an actual experiment. This should be specified.

### Soundness
3

### Presentation
3

### Contribution
4
