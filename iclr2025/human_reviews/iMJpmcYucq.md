## Human Reviewer 1

### Summary
This paper considers the variational inference (VI) problem by minimizing the Kullback-Leibler (KL) divergence over the Bures-Wasserstei space of Gaussian distributions. Building on the work by Diao et al., this paper employs the forward-backward Euler method to tackle the composite nature of the KL objective and non-smoothness of the entropy. The forward step involves computing the Bures-Wasserstein (BW) gradient, which needs to be estimated using Monte Carlo (MC) method with a single sample. However, this approach results in noisy BW gradients, leading to inefficient convergence. The main contribution of this paper is to overcome this limitation, proposing a variance-reduced estimator for the BW gradient, leveraging a control variate approach. In addition, this paper also provides convergence analysis and present experimental results that demonstrate the effectiveness of the proposed method.

### Strengths
The paper is well structured, clearly presenting the problem and the proposed solution. The main idea of using a control variate to reduce the variance of BW gradient is both well-motivated and reasonable. Additionally, it provides strong theoretical foundations for the derived variance-reduced Gaussian VI. The experimental results further validate the effectiveness of the proposed method.

### Weaknesses
I have two major concerns regarding the proposed method in this paper:

1) A straightforward approach to addressing the noisy BW gradient is to sample multiple $X_{k}$ from $\mu_{k}$. This is both effective and efficient, requiring low extra computational cost if estimating $\nabla V$ and $\nabla^{2}V$ is not expensive. I am curious how the performance of the proposed method compares to this trivial approach.

2) The paper only considers the case where $\mu_{k}$ is Gaussian, and the proposed method appears to work well for simple target distributions. Note that both compared methods (BWGD and SGVI) can be extended for Gauss mixtures, while extending the proposed method in this paper to cases of Gaussian mixtures may not trivial. In such cases, the proposed control variate approach might lose its effectiveness and efficiency, as $E[Z_{k}]$ is not zero (the mean) and estimating it is non-trivial.

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper propose improvements on a gaussian variational inference algorithm in the Bures-Wasserstein manifold.  The idea is to find the best Gaussian approximation (in the KL sense) to a probability distribution of the form $Ce^{-V}$ (where $C$ is unknonw) and to use the Wasserstein geometry on the space of Gaussian distribution (Bures-Wasserstein manifold) to guide us toward the optimal solution.  Several authors have studied this question in the last few year leading to an implicit-explicit algorithm.  In this framework the implicit backward (JKO-style) step is computable explicitly and the explicit forward step involves the computation of the the expectations of the gradient and  the Laplacian of $V$ under gaussian measures.  The central contribution of the paper is to perform a variance reduction (using control variates)  at each step and to study the effect of this on the rate of convergence of the algorithm.  Some synthetic examples are studied.

### Strengths
I found the paper to be very overall very well-written with a very clear presentation of the background, literature, and of the technical ideas involved.  The results and the proofs are clearly presented.  Since the measure involved are Gaussian, the analysis of the control variate is quite straightforward and this lead to an improved algorithm at essentially no additional cost.  I think that it is the main contribution of the paper to point out this very cheap improvement to the algorithm. The empirical results also supports this improvement of the stability of the algorithm leading to a substantially better approximation.  At the technical level, control variates for Gaussian measures are quite straightforward and the analysis of the variance reduction relies on standard techniques as well (Stein's lemma).  The convergence of the algorithms is then analyzed by leveraging the results recently obtained Diao et a. (ICML 23) and propagating the improved variance through the convergence analysis of Diao et al.  This leads to improved convergence results.

### Weaknesses
I think the main weakness is the lack of interesting examples. The algorithms seems to do great on the synthetic examples but it remains to be seen how it will perform in a realistic  Bayesian inference example based on real data.

### Questions
+  Are there any obstacles in applying this to more realistic statistical inference problems?  I really do appreciate the mathematical elegance  of the proposed method but t would be nice to have a better test problem  to make the case for the method.

+ Any head-to-head comparisons with different inferences methods such that MCM, Hamiltonian MC, or likelihood free inferences (e.g. using diffusions models) on a test problem would be helpful to convince the reader of the efficiency of the method., 

+  The improvement due to control variates looks quite impressive (as it  sometimes does with control variates!).  The convergence analysis  is nice but it is not clear it shed light on this improvement. Any other explanation, even the hand-waving kind?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
- The authors propose a control-variate-based variance reduction technique for gradient estimation in stochastic Gaussian variational inference (SGVI).
- They theoretically demonstrate that, in specific cases such as high-curvature distributions, the gradient estimator with their method achieves lower variance than the standard SGVI.
- Additionally, they show that their variance reduction scheme enhances the convergence of SGVI when $V(x)$ is (strongly) convex.
- Empirically, they confirmed that their method improves the convergence properties of SGVI.

### Strengths
- The authors extend the control variate method—a well-established variance reduction technique in stochastic variational inference (VI)—to variational inference in Bures-Wasserstein space.
- The variance reduction performance of the proposed method is theoretically guaranteed in (strongly) convex settings.
- The authors demonstrate improved convergence over existing SGVI and Bures-Wasserstein gradient descent (BWGD) methods through several numerical experiments.

### Weaknesses
I would like to express my sincere respect for the efforts the authors have invested in this paper. However, I am unable to strongly recommend it for acceptance at ICLR 2025 for the following reasons:

### Concerns Regarding Theoretical Analyses:
- The evaluation of the derived bound is primarily qualitative and limited to a comparison with the existing upper bounds in [Diao et al., 2023]. However, detailed analysis of each term and the behavior of the bound as the iteration count $N$ approaches infinity—specifically in terms of convergence and the conditions required for convergence—is somewhat lacking. This limits a comprehensive understanding of the proposed method’s advantages and challenges the assessment of this paper’s contribution.
- In the convex case, the variance reduction effect is guaranteed only in the neighborhood of the optimal solution (Thm. 1). However, the convergence analysis in Thm. 3 assumes that variance reduction holds “throughout all iterations of the algorithm,” which seems to suggest that parameters are already near the optimal solution from the outset—a somewhat unrealistic assumption (or am I misinterpreting this?).
- Additionally, from the proof, it appears that the improvement in the bound is only valid under the aforementioned assumption: “when variance reduction is assured for all iterations.” This assumption seems to diverge from the result in Thm. 1. Is there a clear link between them? Is the radius $r$ of this region always sufficiently small?

### Concerns Regarding Numerical Experiments:
- One key motivation for variance reduction in stochastic variational inference is to enhance convergence and generalization in real-world data applications (e.g., [Ruiz et al. (2016); Roeder et al. (2017); Miller et al. (2017); Buckholz et al. (2018)]). However, the numerical experiments in this paper are primarily conducted on synthetic data, and the discussion of the practical applicability of the proposed method is somewhat limited.
- Additionally, there are no direct measurements of gradient variance, leaving insufficient empirical evidence regarding the extent of gradient variance reduction compared to existing methods. If the empirically measured gradient variance does not differ substantially from that of other methods, it is possible that other factors are driving the observed improvement in convergence. Furthermore, based on the results presented, the effectiveness of gradient variance reduction appears to depend on the setting of $c$. Practical guidance on choosing $c$, based on an analysis of the sensitivity of variance-reduction performance to $c$, would be beneficial. Currently, such guidance is absent, leaving limited information on how to apply the proposed method in practice.
- When comparing with variance-reduction methods in SVI in Euclidean space (EVI), this study adopts the method of Roeder et al. (2017); however, the rationale for this choice is unclear. While both methods share a similarity in achieving performance improvements through modest algorithmic modifications, concluding that the proposed method outperforms EVI without comparison to more recent approaches seems premature (e.g., [Miller et al. (2017); Buckholz et al. (2018)]).

### Lack of Discussion on Other Variance Reduction Studies in EVI:
- Recent years have seen substantial progress in variance reduction research within EVI, with many theoretical analyses and methods proposed (e.g., [Kim et al. (2023); Domke et al. (2023)]). These approaches extend beyond control variate methods to include enhancements to Monte Carlo methods and techniques such as reparameterization tricks in Gaussian settings, as in Roeder et al. (2017). While the use of control variates as an initial approach to variance reduction in SGVI is reasonable, the motivation for extending only the control variate approach may seem somewhat limited given the variety of variance reduction techniques available. It might be helpful to mention, for example, the specific challenges or limitations that prevent the application of alternative variance reduction techniques in SGVI, if applicable.

### Citation:
- Ruiz et al. (2016): Francisco J. R. Ruiz, Michalis K. Titsias, and David M. Blei. The Generalized Reparameterization Gradient. NeurIPS2016. https://arxiv.org/abs/1610.02287.
- Miller et al. (2017): Andrew C. Miller, Nicholas J. Foti, Alexander D'Amour, and Ryan P. Adams. Reducing Reparameterization Gradient Variance. NeurIPS2017. https://arxiv.org/abs/1705.07880
- Buckholz et al. (2018): A. Buchholz, F. Wenzel, and S. Mandt. Quasi-Monte Carlo Variational Inference. ICML2018. https://arxiv.org/abs/1807.01604.
- Kim et al. (2023): K. Kim, K. Wu, J. Oh, and J. R. Gardner. Practical and Matching Gradient Variance Bounds for Black-Box Variational Bayesian Inference. ICML2023. https://proceedings.mlr.press/v202/kim23w.html.
- Domke et al. (2023): J. Domke, R. Gower, and G. Garrigos. Provable convergence guarantees for black-box variational inference. NeurIPS2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/d0bcff6425bbf850ec87d5327a965db9-Abstract-Conference.html.

### Questions
Based on the concerns summarized in the weaknesses section, I would like to pose the following questions, categorized into “theoretical” and “experimental” aspects:

### Questions on Theoretical Analyses:
1.	Could the authors provide a more detailed discussion on the behavior of each term in the derived bound and how the bound behaves as the iteration $N$ approaches infinity, particularly in terms of convergence and conditions for convergence?
2.	In the convex case, variance reduction is guaranteed only near the optimal solution (Thm. 1), while the convergence analysis in Thm. 3 assumes variance reduction is achieved “in all iterations.” Does this assumption imply that parameters are initially close to the optimal solution, and if so, how realistic is this assumption?
3.	The improvement in the bound relies on the assumption that variance reduction is guaranteed for all iterations. How does this assumption align with the results of Thm. 1? Is the radius $r$ of the region always small enough to ensure consistency between the two theorems? If not, there may be a significant gap between these results.

### Questions on Experimental Analyses:
1.	The numerical experiments are primarily conducted on synthetic data. Could the authors consider conducting experiments on real-world datasets to provide more practical insights into the usefulness of the proposed method?
2.	There are no experiments measuring the reduction in gradient variance. Could the authors include empirical measurements of gradient variance to illustrate how much variance is reduced compared to existing methods? The following references may be relevant for this consideration: [Miller et al. (2017); Buckholz et al. (2018)].
3.	In the proposed method, the degree of variance reduction appears to depend on the parameter $c$. Could the authors provide additional practical discussions and sensitivity analyses on the selection of $c$ and its impact on variance reduction performance, particularly in terms of gradient variance? Although Figure 2 presents the relationship between $c$ and the degree of variance reduction, it would be beneficial to confirm whether this behavior is consistent across various experimental settings.
4.	The comparison with the method of Roeder et al. (2017) in Euclidean SVI is understandable; however, could the authors clarify the rationale for not including comparisons with more recent variance-reduction methods?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes an improved optimization technique for variational inference with a multivariate Gaussian variational family.

Building on the prior work that proposed optimization on the Bures–Wasserstein manifold and simple one-sample stochastic gradient estimators, this work offers improved stochastic gradient estimators. These are based on a simple idea: as iterates get closer to the optimum, they provide additional information about the next gradient that can be used to reduce variance of its stochastic estimator.

Authors support the superiority of the new optimization technique with both theoretical analysis and empirical evaluation, the latter on a set of synthetic benchmarks.

### Strengths
- The paper is written very well. It is a pleasure to read.
- The theoretical analysis is convincing (I didn’t rigorously check the proofs but I don’t doubt the claims and I am pretty sure the methods authors use are quite appropriate).
- Empirical evaluation shows very strong improvement over the baselines, more so than the theoretical analysis (which is not claimed to be tight) suggests.

### Weaknesses
-I’m not an expert in optimization, so it’s somewhat hard for me to judge the impact. I can speculate that the problem might seem somewhat narrow: variational inference with _Gaussian_ variational family. However, I don’t really think so myself. Furthermore, in my opinion, a good solution to even a somewhat narrow problem definitely warrants publication.

I would actually give this paper a score of 9, but only 8 and 10 are available.

### Questions
As mentioned above, in my opinion the paper is very well-written. I therefore don’t have any content-related questions, only a small number of typographical suggestions for the authors:
- Perhaps use the en-dash (--) instead of the simple dash (-) in "Bures–Wasserstein". I believe this is the standard when it comes to joining two names.
- I believe that "FB" abbreviation (forward-backward) is never introduced.
- The term “L-smooth” seemed weird to me, why not “L-Lipschitz”?
- Page 4, line 174, “affine map” -> “affine maps”.
- Page 4, line 198, “shows that” -> “showed that”.
- Page 5, line 233, I would remove parentheses around the “also see ...”, they look ugly immediately after a citep.
- Page 5, line 237, perhaps add commas around “where $c \in \mathbb{R}$”, otherwise the sentence is hard to read.
- Perhaps, when introducing Algorithm 1, you could briefly mention there are recipes for defining the constants $c_k$ that you investigate further in the text. An algorithm without parameters you don’t know how to set would come across as more useful.
- Page 8, line 387, “We also note that conditioning” -> “We also note that by conditioning”.
- Page 8, footnote 1, please expand what is the minor correction. Also, perhaps start with a capital letter.
- Page 9, line 457, missing “e” in “covariancs”.
- Page 9, line 472, “In The” -> “In the”.

Note: no need to respond to the small items above in the rebuttal, save yourself some time.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4