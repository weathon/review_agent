# Fast kernel methods: Sobolev, physics-informed, and additive models

- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Kernel methods are powerful tools in statistical learning, but their cubic complexity in the sample size $n$ limits their use on large-scale datasets. In this work, we introduce a scalable framework for kernel regression with $\mathcal{O}(n \log n)$ complexity, fully leveraging GPU acceleration. The approach is based on a Fourier representation of kernels combined with non-uniform fast Fourier transforms (NUFFT), enabling exact, fast, and memory-efficient computations. 
We instantiate our framework in three settings: Sobolev kernel regression, physics-informed regression, and additive models. When known, the proposed estimators are shown to achieve  minimax convergence rates, consistent with classical kernel theory. Empirical results demonstrate that our methods can process up to tens of billions of samples within minutes, providing both statistical accuracy and computational scalability. 
These contributions establish a flexible approach, paving the way for the routine application of kernel methods in large-scale learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a NUFFT-based framework for kernel regression that is exact within a truncated Fourier space and runs in 
$O(nlog n)$ with GPU acceleration. It studies three settings: Sobolev kernels, physics-informed regression with PDE penalties, and additive models, and claims minimax-rate guarantees together with significant scalability improvements. Experiments demonstrate the great improvement in computation gain.

### Strengths
1. This paper gives a computational story such that covariance vectors/matrices reduce to NUFFT computations and a block-Toeplitz structure, giving significant computation cost reduction.
2. The theory recovers the standard minimax rate of kernel methods, while speeding up the computation.
3. The authors consider three different applications covering Sobolev KRR, physics-informed KRR, and addictive models.
4. The experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The literature review on scalable kernel methods and the comparison are limited.
2. The assumed target function space is limited to a Sobolev RKHS, not a general RKHS, as well as the bounded Lipschitz domain.
3. The main contribution is the fast optimization in Section 2.3, however, the implementation is not so clear, and more details should be clarified to the non-experts.
4. The experiments are somewhat limited to very low-dimensional datasets.

### Questions
1. The random feature method is also a finite-dimensional approximation. What is the precise difference between your truncated-Fourier approach (e.g., lines 101–119) and RF? Can a GPU-optimized RF pipeline achieve comparable computational gains?
2. As mentioned in line 44, the NUFFT method is proposed by Shih et al. 2021. What methodological advances does this work introduce relative to Shih et al. 2021?
3. While I acknowledge that one contribution of this paper is to establish the optimal learning rates of the proposed methods in three settings, however, the theory is well-established without NUFFT in Sobolev KRR, physics-informed KRR, and addictive models. What is the main difference or biggest challenge between your paper and the literature?
4. This paper assumes bounded domains and densities. In practice, data live on $R^d$  and are non-uniform. Can the author discuss the boundary handling and NUFFT sensitivity under non-uniform settings?
5. The implementation of NUFFT is typically limited to $d=1,2,3$, can the method have good performance in higher dimensions?
6. In Figure 6, why CPU kernel faster than the GPU kernel when $n<10^4$?
7. Theory tunes $m$ via $s$, which is unknown in practice. Can you provide an adaptive rule to determine $m$ while preserving the theoretical guarantees?
8. Can the author compare against other large-scale kernel baselines (e.g., Nyström/FALKON-style solvers, RFF model).

*Minor issue*:

1. In line 91-92, why $f^*$ has a bounded RKHS norm?

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
3

### Summary
This paper proposes a scalable framework for kernel regression that achieves O(n log(n)) computational complexity by leveraging a Fourier feature representation of Sobolev-type kernels combined with GPU-accelerated non-uniform fast Fourier transforms (NUFFT). The authors show how this formulation enables fast kernel learning for three settings: standard Sobolev regression, physics-informed regression with linear PDE constraints, and additive models for high-dimensional inputs. They establish minimax-optimal convergence rates in these scenarios and show empirical scalability to extremely large synthetic datasets, reportedly up to billions of samples processed in minutes on a single GPU.

### Strengths
The paper tackles a central limitation of kernel methods — their poor scalability — and presents a unified framework that achieves nearly linear complexity by combining Fourier kernel representations with NUFFT-based GPU acceleration. This allows the authors to train kernel models on extremely large datasets, demonstrating impressive empirical scalability. The work is well-grounded theoretically, with minimax-optimal learning guarantees established in the studied Sobolev, physics-informed, and additive settings. Overall, the paper offers a compelling and technically solid advancement toward making kernel methods practical at modern data scales.

### Weaknesses
-The introduction is quite limited and does not sufficiently position the contribution within the broader landscape of scalable kernel methods. The related work focuses almost exclusively on Meanti et al. (2020), which is highly reductive given substantial literature on efficient kernel learning (e.g., random features, sketching techniques, and Nyström methods combined with fast solvers). Moreover, this prior work is characterized as complicating analysis and potentially degrading performance, whereas results such as “Less Is More: Nyström Computational Regularization” (Rudi et al., 2015) show that Nyström approximations can preserve optimal statistical guarantees while scaling to large datasets.
-A clearer articulation of the applicability and limitations of the proposed framework would greatly strengthen Section 2. The method fundamentally relies on Sobolev-type kernels, which restricts the generality relative to widely used kernels like Gaussian RBFs. The paper should provide explicit guidance on when the approach can or cannot be applied in practice, with a fair comparison with broadly applicable methods like FALKON
-The empirical evaluation is based entirely on synthetic data and lacks comparisons against established scalable kernel baselines. Given the strong practical motivation of the work, including such comparisons would be essential to assess empirical competitiveness.
-In general, some more explanations can be useful, like a concise presentation of the introduced NUFFT method ( type-I is mentioned for example).

### Questions
-The method is developed specifically for Sobolev-type RKHSs. Could you clarify why this functional setting is particularly important in practice, and discuss to what extent widely used kernels such as Gaussian RBFs or Matérn kernels fall within—or outside—the scope of the proposed framework? More broadly, can you articulate the main limitations of the approach and provide guidance on when it may not be applicable?

-The empirical evaluation focuses entirely on synthetic data and does not include comparisons with established scalable kernel methods (e.g., Nyström + FALKON, random features). Given the practical motivation and scalability claims of the work, could you explain the absence of such baselines or provide experiments on real datasets to better assess competitiveness?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript leverages a Fourier representation of kernels to accelerate kernel regression. Key to the efficacy of the method is the use of a NUFFT. Several experiments are provided that illustrate the effectiveness of the scheme.

### Strengths
The main strengths of this manuscript are in sections 3-5 where concrete use cases for the proposed method (and some accompanying analysis) are outlined and explored. The experiments do effectively show that the implemented method is effective for a variety of problems; moreover, the theory provides reasonable, albeit often illustrative rather than tight, control over the behavior of the methods.

### Weaknesses
Unfortunately, as written the main contribution of the manuscript seems to be articulated as the method for working with kernel matrices and I do not believe that is new. The key references are:

Efficient Fourier representations of families of Gaussian processes by Philip Greenghard [see https://arxiv.org/abs/2109.14081 and https://link.springer.com/article/10.1007/s11222-022-10124-z] 

followed by more recent work such as 

Greengard, Philip, Manas Rachh, and Alex H. Barnett. "Equispaced Fourier representations for efficient Gaussian process regression from a billion data points." SIAM/ASA Journal on Uncertainty Quantification 13, no. 1 (2025): 63-89. (arxiv: https://arxiv.org/pdf/2210.10210)

and

Barnett, Alex, Philip Greengard, and Manas Rachh. "Uniform approximation of common Gaussian process kernels using equispaced Fourier grids." Applied and Computational Harmonic Analysis 71 (2024): 101640.

While there are seemingly some variations on the style of theoretical results provided (it is not clear the provide anything substantially new or more informative), if the main contribution of this manuscript is the "algorithm" for working with kernel matrices then it needs to either: (1) articulate that there is a distinction/difference/improvement relative to this prior work (e.g., did I miss something) or (2) significantly refocus its contributions in relation to this prior work.

I will add that I think the presentation of the manuscript could be improved. The above question is illustrate of a lack of details provided in certain places that make it a bit tricky to assess the numerical results. In addition the method outlined in Section 2 could use some sort of "algorithmic" presentation to more clearly specify the computations.

### Questions
Some of the discussion following Prop. 5.1 is a bit odd to me; specifically, if I am interpreting the problem being solved correctly for the grid search in $\lambda,$ why are multiple inversions needed (as suggested by the text). Generally, the use of, e.g., Krylov methods absent preconditioner (or even with in certain situations) enable efficient computation of the solutions corresponding to multiple $\lambda$ more efficiently than independently solving a bunch of linear systems. (Perhaps I missed something about the setup?) I guess the answer to this question does sort of influence how to think about the results in Fig. 4. Clarification about what exactly is being timed would be helpful.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a highly scalable framework for kernel regression that reduces computational complexity from the typical O(n³) to O(n log n), enabling efficient processing of massive datasets (up to tens of billions of samples) on GPUs. The core innovation is the use of a Fourier representation of kernels combined with the Non-uniform Fast Fourier Transform (NUFFT).

### Strengths
The author combining several existing facts to give a fast implementation of kernel regression.

### Weaknesses
I am curious about the novel theoretical contributions of this work, as the core components appear to be established.

1.The convergence rates in Propositions 3.1, 3.2, and 5.1 are known results in the literature. More specific, it is well known that suppose that f\in [\mathcal H]^{s}, and the eigenvalue decay rate of \mathcal H is \beta, the the optimal rate is n^{-\frac{s\beta}{s\beta+1}}. ( see e.g.,  Optimal rates for regu-
larized least squares algorithm, 2007. A. Caponnetto and E. De Vito.,
,  Sobolev Norm Learning Rates for Regularized Least-Squares Algorithms, 2020.  Simon Fischer, Ingo Steinwart;  and On the Optimality of Misspecified Kernel Ridge Regression, 2023.  Haobo Zhang, Yicheng Li, Weihao Lu, Qian Lin)

2.The computational acceleration relies on the NUFFT (Shih et al., 2021), which is the established solution for non-uniform sampling.

3.The foundation of using Fourier transforms for Sobolev spaces is also well-known ( see e.g., wiki https://en.wikipedia.org/wiki/Sobolev_space ).

While the combination of these elements is competently executed, the paper currently reads as a straightforward application of existing techniques. A significant theoretical contribution would be an extension to more general Reproducing Kernel Hilbert Spaces (RKHS), which would make this a much stronger paper

### Questions
Similar to the weakness mentioned above, I would be more than happy to change my current assessment if the authors could propose a reasonable method for a general RKHS or provide new theoretical results.

### Soundness
2

### Presentation
3

### Contribution
2
