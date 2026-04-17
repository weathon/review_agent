# A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
We propose HALO, a memory-efficient hierarchical algorithm for solving large-scale optimal transport (OT) problems with squared Euclidean cost, particularly effective in moderate-dimensional settings. 
The core of \ours lies in combining a hierarchical representation of the OT problem with parallel-friendly linear programming solvers, within which an active pruning technique is integrated to further reduce memory usage and computational cost.
Theoretically, we establish a scale-independent iteration-complexity upper bound for the refinement phase, which is consistent with our numerical observations. 
Numerically, experiments on the image dataset \dataset and the 3D point cloud dataset \datasetnongrid demonstrate that \ours effectively alleviates the memory and scalability bottlenecks of existing solvers.
Our method demonstrates significant advantages compared to state-of-the-art baselines: for images with $n=1024^2$ pixels, it achieves an $8.9\times$ speedup and $70.5$% reduction in memory usage under comparable accuracy; for 3D point clouds at scale $n=2^{18}$, it achieves a $1.84\times$ speedup and an $83.2$% reduction in memory usage with $24.9$% lower transport cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces HALO, a hierarchical and memory-efficient algorithm for large-scale optimal
transport (OT) problems on 2D supports with squared Euclidean cost. The method combines a coarseto-
fine multi-scale framework with a GPU-optimized PDHG (Primal-Dual Hybrid Gradient) solver
and an active-pruning mechanism for sparsity. The authors prove a scale-independent iteration
complexity bound for the refinement phase and show strong empirical results on the DOTmark
benchmark, achieving significant speedup and memory usage saving compared to methods such as
HOT, ShortCut, and M3S.

### Strengths
1. Theorem 1 (scale-independent iteration complexity) provides a nontrivial and interpretable
convergence guarantee, addressing the gap in multi-scale OT methods (which often lack formal
complexity bounds).
2. HALO merges the hierarchical multi-scale structure with a GPU-based LP solver. This is a
thoughtful contribution, given the growing interest in efficient OT solvers.
3. The paper is clearly written and well-organized. Figures 1–3 effectively illustrate the hierarchical
process and empirical trends. Proofs are included in the appendix and seem technically sound.

### Weaknesses
1. HALO is currently limited to 2D support with squared Euclidean cost. The hierarchical design
heavily relies on the specific problem (OT between two 2D images), while applications of OT
(like in generative models) are on more general settings (Wasserstein GAN [1] and optimal flow
matching [2]) where the supports are high-dimensional. In this case, the HALO may not work well
as in the image problems in DOTmark because the data locality property doesn't hold in these
problem. The limited problem setting is mentioned but not stated clearly in the introduction.

2. Although the paper fairly acknowledges this, HALO’s performance heavily relies on a third-party
GPU solver. A deeper analysis of how HALO interacts with other first-order solvers would make
the contribution more self-contained.

3. The parameter beta in the dual-violation step appears to influence sparsity and runtime, but the
sensitivity analyses are missing. Reporting how beta impacts convergence and memory would be
valuable.

4. In DOTMark [3], there are various types of images. It would be interesting to see how HALO
performs on these classes of images respectively to have a clearer idea of the practicality of the
assumptions and the influence of the data locality on the convergence of HALO.

References:
[1] Adler, Jonas, and Sebastian Lunz. "Banach wasserstein gan." Advances in neural information
processing systems 31 (2018).
[2] Kornilov, Nikita, et al. "Optimal flow matching: Learning straight trajectories in just one
step." Advances in Neural Information Processing Systems 37 (2024): 104180-104204.
[3] Schrieber, Jörn, Dominic Schuhmacher, and Carsten Gottschlich. "Dotmark–a benchmark for
discrete optimal transport." IEEE Access 5 (2016): 271-282.

### Questions
1. HALO is currently limited to 2D support with squared Euclidean cost. The hierarchical design
heavily relies on the specific problem (OT between two 2D images), while applications of OT
(like in generative models) are on more general settings (Wasserstein GAN [1] and optimal flow
matching [2]) where the supports are high-dimensional. In this case, the HALO may not work well
as in the image problems in DOTmark because the data locality property doesn't hold in these
problem. The limited problem setting is mentioned but not stated clearly in the introduction.

2. Although the paper fairly acknowledges this, HALO’s performance heavily relies on a third-party
GPU solver. A deeper analysis of how HALO interacts with other first-order solvers would make
the contribution more self-contained.

3. The parameter beta in the dual-violation step appears to influence sparsity and runtime, but the
sensitivity analyses are missing. Reporting how beta impacts convergence and memory would be
valuable.

4. In DOTMark [3], there are various types of images. It would be interesting to see how HALO
performs on these classes of images respectively to have a clearer idea of the practicality of the
assumptions and the influence of the data locality on the convergence of HALO.

References:
[1] Adler, Jonas, and Sebastian Lunz. "Banach wasserstein gan." Advances in neural information
processing systems 31 (2018).
[2] Kornilov, Nikita, et al. "Optimal flow matching: Learning straight trajectories in just one
step." Advances in Neural Information Processing Systems 37 (2024): 104180-104204.
[3] Schrieber, Jörn, Dominic Schuhmacher, and Carsten Gottschlich. "Dotmark–a benchmark for
discrete optimal transport." IEEE Access 5 (2016): 271-282.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposed a hierarchical algorithm to solve large-scare OT problems. By several techniques including hierarchical expansion, active support update, and GPU based PDHG. The algorithm justifies its efficiency by improving the state-of-the-art baselines and theoretical scale independent iteration complexity bound.

### Strengths
By carefully tuning the three approaches in discrete OT techniques, the algorithm justify itself by testing on common data sets.

### Weaknesses
1. A well known issue of the hierarchical approach, is the scalability to higher dimensions rather than 2D mesh like topology. The author shall consider a least a remark on extension to higher dimensions.
2. Another issue of the manuscript is the theoretical justification relies on rather strong assumption (4,5), also as the authors mentioned. Since the complexity of HALO is only an upper bound under strong assumptions, the practical superiority of the HALO is not well-explained. The author may need further discuss the structural advantages comparing with other algorithm, for instance Multiscale-OT.

### Questions
Beyond the weakness that requires the author to address.
1. The active support in Def. 1 also appears in [1] and even earlier (see references in [1]), will such randomized method improve the HALO?
2. In experiment, the ShortCut is only implemented on CPU, given its low memory cost and rather low runtime in lower resolution, is there theoretical barrier for ShortCut to implement on GPU and yield an even better result?

[1] Xie, Yue, Zhongjian Wang, and Zhiwen Zhang. "Randomized methods for computing optimal transport without regularization and their convergence analysis." Journal of Scientific Computing 100.2 (2024): 37.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces HALO (Hierarchical Algorithm for Large-scale Optimal Transport), a GPU-based, memory-efficient method for solving large-scale optimal transport (OT) problems in the plane with squared Euclidean cost. HALO combines a multiscale hierarchical framework with a sparse, active-support refinement scheme and a primal–dual hybrid gradient (PDHG) solver to overcome the severe memory and scalability limitations of existing OT solvers. By solving coarser OT problems first and using their solutions to warm-start finer levels, HALO efficiently refines the transport plan while keeping memory usage to  𝑂(r^2), where r is the number of pixels per dimension.

The method also includes a dual-violation augmentation step that improves robustness, and the authors establish a scale-independent iteration bound, proving that each refinement level requires only 𝑂(1) iterations. On the DOTmark benchmark, HALO outperforms state-of-the-art methods such as HOT, ShortCut, and M3S—achieving up to 8.9× speedup and 70% lower GPU memory use for 1024×1024 images, while maintaining comparable or superior accuracy. Overall, HALO demonstrates near-linear runtime scaling and high parallel efficiency, offering a theoretically grounded and practically scalable solution for large-scale OT computation.

### Strengths
The paper’s main strengths lie in its combination of theoretical rigor and practical scalability. HALO introduces a hierarchical, GPU-friendly framework that reduces the memory requirement of large-scale optimal transport (OT) from 𝑂(𝑟^4) to 𝑂(𝑟^2), allowing it to handle very high-resolution problems that were previously infeasible. Its coarse-to-fine multiscale design and sparsity-based active support updates enable near-linear runtime scaling, while the factorization-free PDHG solver fully exploits GPU parallelism for efficient computation.

Equally important, the paper provides strong theoretical guarantees and robust empirical validation. The authors prove a scale-independent iteration bound, ensuring that refinement at each level converges in a constant number of steps, and introduce dual-violation augmentation to improve robustness without sacrificing sparsity. Extensive experiments on the DOTmark benchmark confirm that HALO achieves up to 8.9× speedup and 70% memory savings compared to leading solvers, while maintaining high accuracy. Overall, the method is both theoretically elegant and practically impactful, setting a new benchmark for scalable OT computation.

### Weaknesses
The main weaknesses of the paper lie in its limited generality and empirical scope. HALO is tailored for 2D optimal transport problems with squared Euclidean cost on regular grids, and both its theoretical guarantees and hierarchical design rely on this structure. As a result, the method’s applicability to higher-dimensional settings, irregular domains, or non-Euclidean costs remains unclear. Moreover, while the algorithm performs impressively on the DOTmark benchmark, its evaluation is confined to this dataset, leaving open questions about robustness and generalization to more diverse or real-world applications.

Another limitation is the dependence on heuristic components such as dual-violation augmentation and Top-K active-set selection, whose performance may vary with parameter choices not deeply analyzed in the paper. The implementation is also technically complex—combining multiscale hierarchy, active-support refinement, and GPU-based PDHG—which could make reproduction or extension challenging for practitioners. Overall, while HALO is methodologically strong and achieves excellent performance, its restricted scope, heuristic tuning, and limited experimental diversity temper its broader applicability.

### Questions
The following questions need to be addressed to further improve the quality:

1. transportation cost: if the transportation cost is not the squared Euclidean distance, like L1 distance, will the method be applicable ?
2. dimension : if the problem is not restricted on 2d, but general n dimensional Euclidean space, can the method work ?
3. range: if the support of the target measure is not convex, but a concave Jordan domain, can the active-support refinement  algorithm still work ? In this situation, there will be complicated singularities in the domain, determining the singularity will be challenging.
4. HALO is built on a first -order optimization framework, is it possible to use Newton's method ?

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
- The authors proposes HALO (Hierarchical Algorithm for Large-scale Optimal Transport), a novel GPU-friendly method for solving large discrete optimal transport (OT) problems efficiently. 
- The key idea is to build a coarse-to-fine multi-scale hierarchy and iteratively refine the transport plan while maintaining an active support set that captures potentially non-zero couplings. 
- Each refinement step solves a restricted OT problem via a primal-dual hybrid gradient (PDHG) method optimized for GPU computation.
- Experiments on large-scale image OT tasks (DOTmark dataset) demonstrate significant speed and memory savings ove baselines like HOT, ShortCut, and M3S.

### Strengths
- The paper successfully solves the scalability bottleneck of OT on high-resolution data with well-designed hierarchical + active-support framework and solid theoretical justification.
- The experimental results are promising with great speed and memory advantage.

### Weaknesses
- Currently limited to 2D grid supports with squared Euclidean cost.
- The active-support update relies on heuristic parameters (e.g., β) without sensitivity analysis.
-Experiments only cover image data; no test on non-grid or higher-dimensional problems. Does it work on other distributions?

### Questions
Please mainly see the above weakness part

### Soundness
3

### Presentation
2

### Contribution
3
