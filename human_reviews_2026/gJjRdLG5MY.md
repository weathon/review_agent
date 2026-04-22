# A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 6, 8, 2, 6

## Abstract
The representer theorem is a cornerstone of kernel methods, which aim to estimate latent functions in reproducing kernel Hilbert spaces (RKHSs) in a nonparametric manner. Its significance lies in converting inherently infinite-dimensional optimization problems into finite-dimensional ones over dual coefficients, thereby enabling practical and computationally tractable algorithms. In this paper, we address the problem of estimating the latent triggering kernels--functions that encode the interaction structure between events--for linear multivariate Hawkes processes based on observed event sequences within an RKHS framework. We show that, under the principle of penalized least squares minimization, a novel form of representer theorem emerges: a family of transformed kernels can be defined via a system of simultaneous integral equations, and the optimal estimator of each triggering kernel is expressed as a linear combination of these transformed kernels evaluated at the data points. Remarkably, the dual coefficients are all analytically fixed to unity, obviating the need to solve a costly optimization problem to obtain the dual coefficients. This leads to a highly efficient estimator capable of handling large-scale data more effectively than conventional nonparametric approaches. Empirical evaluations on synthetic and real-world datasets reveal that the proposed method achieves competitive predictive accuracy while substantially improving computational efficiency compared to state-of-the-art kernel method-based estimators.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors consider nonparametric estimation of the triggering kernels in a multivariate linear Hawkes process using an RKHS (reproducing kernel Hilbert space) framework. They formulate a penalised least-squares loss for the latent triggering functions, and derive a novel representer theorem showing that the optimal kernel estimators admits an expansion over a family of transformed kernels evaluated at data points, and remarkably, the dual coefficients are analytically fixed to unity, so one does not need to solve an optimization problem. These transformed kernels are defined by a system of Fredholm integral equations, which allows an efficient random-feature approximation algorithm, further enabling scalability since number of random features can be typically set much smaller than number of samples. 

The method applies to linear multivariate Hawkes (identity link function, no inhibitory effects or non-linear link). So the generality is narrower than the Bonnet & Sangnier (2025) paper which handles inhibition and non-linear link functions. The trade-off is computational simplicity and analytic coefficient expressions.

Empirical results on synthetic toy-models show competitive accuracy and much better efficiency compared to existing kernel-based methods.

### Strengths
(1) Novel representer theorem: While prior works (Bonnet & Sangnier 2025) derived a general representer theorem for approximate loss formulations (and requiring optimization of dual coefficients), this paper focuses on linear Hawkes processes and provides one for the exact penalised least squares loss obtaining that dual coefficients equal unity w.r.t. transformed kernels characterised by Fredholm integral equations  That is a non-trivial theoretical advance.

(2) Practical algorithm: To obtain practical algorithm, authors use random features to avoid discretization approximation of the integral operators,  and, in process, reduce the inversion to a matrix independent of data size. This results in practical algorithm that scales much better than previous kernel-based methods for Hawkes. Further, authors empirically show that you can maintain competitive predictive accuracy while significantly improving computational efficiency. This allows algorithm's deployment in large event-data applications.

(3) Paper is generally well written and related works are, up to my knowledge, well covered.

### Weaknesses
(1) Limited depth: The paper is fundamentally methodological. New kernel method is backed by the representer theorem and coupled with random features to result in practical algorithm. However, any form of learning guarantees is lacking. 

(2) Scalability via random features: Using random features and reducing the inversion to a matrix independent of data size is golden standard in scaling kernel methods. Hence, there is hardly some big novelty in this. Even more so, since there is lack on any theory such as e.g. convergence w.r.t. number of random features.

(3) Empirical evaluation: Since the paper is methodological,  testing it only on synthetic toy-models is a bit underwhelming. 

To conclude, up to my knowledge, the paper delivers a clear meaningful advance: the theoretical representer result is strong (analytic coefficients) and the algorithmic scalability is obtained via random features. As such, it can make a solid contribution to kernel-based nonparametric estimation for Hawkes processes if the empirical study is improved, or some theoretical  learning guarantees included.

### Questions
Minor issues:
- line 278 it should read "number"
- line 132, definition of ${\cal N}_i$ is missing, though intuivily clear what it should represent
- line 147, would be good to explicitly write what dual coefficients equal to
- line 155 personally I would prefer using $\dagger$ for pseudo inverse sainted of $*$ (typically adjoint)

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
4

### Summary
The paper aims to provide a theoretically principled methodology for reducing the computational complexity of nonparametric estimation of the intensity functions in multivariate linear Hawkes process, where the triggering "kernels" $g_{ij}$ are assumed to belong to an RKHS $\mathcal{H}_{k}$ with reproducing kernel $k$ and a Tikhonov regularization is employed. 

The novel technical contribution of this paper and the workhorse behind this paper is a representer theorem for the sample based estimates $\hat{g}_{ij}$ in terms of  linear combinations of Fredholm integral transformed versions of the base kernel function $k$, denoted by $h_j$'s. The representer theorem ensures that there is no need for dual variable (coefficients in terms of $h_ij$'s) optimization, leaving only the system of Fredholm integral equations for $h_j$'s to be solved. Using the machinery of Random Fourier Features (RFF), a practical algorithm is proposed with corresponding experiments based on synthetic data that achieves $O(N^{2}M^{2}U^{2} +M^{3}U^{3})$ computational complexity where $N$ is the maximum sampling frequency in each dimension, M is the number of Random features and U is the dimensionality of the multivariate linear Hawkes process, improving upon $O(N^{4}U^{2})$ computational complexity of the SOTA method (Bonnet and Sangnier, 2025), along with the usual memory complexity improvement expected from RFF. While computation time on synthetic experiments drops by around 2 orders of magnitude, inferential accuracy loss is reasonably low.

### Strengths
The paper presents a very clear and precise solution to a significant computational problem in the context of linear Hawkes processes,and explains the utility behind analyzing the continuous variational formulation of the data fitting problem, along with a clean and complete derivation of the key technical result being included. The representer theorem is novel in the context of this particular problem, and shows the theoretical advantage of using the Tikhonov regualrization of RKHS norm in contrast to the Hakwes process log-likelihood based estimation approach. Synthetic experiments also show around 100 times relative improvement in computational complexity compared to SOTA with small loss of inferential power for the trigerring "kernels" $g$, in terms of integrated squared error.

### Weaknesses
While the RFF approach indeed allows a practical solution to the problem, from a theoretical perspective it would be helpful if the authors can report the specific conditions (properties of kernels etc.) under which the Fredholm integral equations for $h_j$'s (Equation 6 under Theorem 1) admit a solution. The authors themselves acknowledge the limitation of the paper being restricted to linear Hawkes processes and requires adhoc post-processing in empirical experiments, while approach of the SOTA method is more general and is applicable to nonlinear processes as well. Also, from a empirical perspective, the implication of the small loss of inferential power for the latent trigerring "kernels" $g_{ij}$'s for actual predictive performance for the point process is not immediately clear.

Minor typos:
1. Replace $x$'s by $s$ in Equation 22 (Lines 452 and 453)
2. Replace "quantifing" by quantifying on Line 096
3. Replace "evet" by event on Ine 134

### Questions
Based on the discussion of the weaknesses, the paper will improve if
1.  the authors can report the specific conditions (properties of kernels etc.) under which the Fredholm integral equations for $h_j$'s (Equation 6 under Theorem 1) admit a solution and 
2. demonstrate the implication of the small loss of inferential power for the latent trigerring "kernels" $g_{ij}$'s for actual predictive performance for the point process. 

Although this did not play a role in my assessment, but, as always, a real-life experimment will always help, especially regarding the evaluation of predictive performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new representer-theorem–based method for estimating triggering kernels of linear multivariate Hawkes processes in an RKHS framework. By formulating a system of simultaneous integral equations, the optimal estimator is expressed as a linear combination of transformed kernels evaluated at data points, with all dual coefficients analytically fixed to one. As a result, the method avoids solving high-dimensional optimization problems and achieves substantial computational efficiency gains. Experiments on synthetic data show that the approach matches state-of-the-art predictive accuracy while being significantly faster than existing nonparametric kernel estimators.

### Strengths
This paper establishes the first representer theorem for the non-approximated penalized least squares formulation of linear multivariate Hawkes processes, filling a clear theoretical gap in kernel-based point process modeling. The resulting estimator has all dual coefficients analytically fixed to one, eliminating the need for high-dimensional nonlinear optimization and dramatically improving scalability over prior methods. The proposed approach yields a closed-form solution of the integral equations via random feature approximations, avoiding Riemann-sum discretization used in earlier work and enabling efficient computation. The final estimator consists only of additive matrix operations and a matrix inversion whose size is independent of the data size, making the method lightweight, scalable, and well-suited for large-scale Hawkes process datasets. Overall, the paper offers both theoretical novelty and practical computational advantages, advancing kernel-based learning for multivariate Hawkes processes.

### Weaknesses
Although the proposed method offers strong computational advantages, it relies on the identity link function and therefore applies only to linear Hawkes processes; models requiring non-linear link functions to enforce non-negativity or capture inhibitory effects fall outside its scope. In addition, the representer theorem is developed under a penalized least-squares formulation rather than maximum likelihood, which may limit statistical efficiency in some settings. The approach also requires solving Fredholm integral equations whose accuracy depends on the random feature approximation, potentially introducing approximation error relative to exact kernel evaluations. Finally, the empirical evaluation is conducted on synthetic datasets, leaving open questions about robustness and performance on real-world event data.

### Questions
1. The analytical framework appears to rely on classical techniques, and at first glance the contribution seems incremental. Could you clarify precisely where the conceptual or technical novelty lies in your representer-theorem formulation for Hawkes processes?
2. In the era of deep neural networks, kernel methods are sometimes viewed as outdated or less competitive. Why is a kernel-based approach appropriate and timely for ICLR, and what advantages does it offer over contemporary deep-learning-based models for point processes?
3. The representer theorem is a well-established and classical concept in kernel methods. How does the representer theorem presented in this paper differ from existing versions, including variants used in prior work on point processes? Is the contribution more than a minor extension, and in what sense is it a meaningful and substantial generalization?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new representer theorem for linear multivariate Hawkes processes trained with penalized least squares in an RKHS. The theorem says that each triggering kernel can be written as a sum of equivalent kernels** that solve a system of Fredholm integral equations. A key outcome is that the dual coefficients are all 1, so there is no separate optimization over them. The authors show an efficient construction using random Fourier features, which reduces learning to one matrix inversion of size $MU \times MU$ (independent of the number of events). On synthetic datasets, the method keeps similar accuracy to strong baselines while being much faster. The paper also discusses limits: linear link (no guaranteed non-negativity) and cubic cost in the process dimension $U$.

### Strengths
- **Originality:** First representer theorem for the non-discretized penalized LS problem in linear Hawkes; fixing all dual coefficients to 1 removes a costly optimization step.  
- **Quality:** Clear path from theorem to algorithm using degenerate kernels / random Fourier features, with closed-form estimators.  
- **Clarity:** Assumptions and complexity are explicit; the narrative from theory to implementation is easy to follow.  
- **Significance:** Very large speed-ups compared to strong baselines while keeping similar accuracy for larger datasets.

### Weaknesses
- **Evidence scope:** Only synthetic data are used; real-world robustness and data issues are not tested.  
- **Proof placement:** The main-text sketch uses inverse operators and delta distributions and may feel less rigorous; the appendix version is better suited for details.  
- **Modeling limit:** Linear link does not guarantee non-negativity of intensity; post-hoc clipping or other fixes may be needed.  
- **Scalability in $U$:** Cost scales as $O(N^2 M^2 U^2 + M^3 U^3)$; very high-dimensional processes or very long sequences can still be heavy.

### Questions
1. For large \(U\), do **conjugate-gradient** solvers or block preconditioners help? Please provide scaling curves vs. \(U\).  
2. How sensitive are results to the **support window $A$**? Can $A$ be learned ?  
3. Consider moving the long proof to the **appendix**, and keeping only a short, readable proof sketch in the main text.

### Soundness
3

### Presentation
2

### Contribution
3
