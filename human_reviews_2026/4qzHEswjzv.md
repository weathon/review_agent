# Trajectory Seriation via Spectral Tangent Alignment and Global Embedding

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
This paper addresses the problem of linear seriation: recovering the intrinsic order of noisy samples drawn from an unknown one-dimensional manifold embedded in a higher-dimensional space. We propose a multi-stage approach that first robustly estimates local tangent directions using Principal Component Analysis (PCA) on neighborhoods, establishing theoretical consistency for these local estimates. Global orientation consistency of these tangents is then achieved through a spectral relaxation of a pairwise alignment objective. Finally, a globally consistent 1D embedding is computed by solving a carefully formulated linear system (or equivalently, a spectral problem on a derived Laplacian) that aligns the embedding with the oriented local projections. This method effectively leverages local geometric information while ensuring global coherence, producing an ordering robust to noise, curvature, and initial data rotation. We demonstrate its performance on simulated manifold data and discuss the theoretical underpinnings of its core components.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new algorithm for linear seriation of data in R^d. i.e. assigning an order to a point cloud when we assume the points roughly follow a one-dimensional curve.

The algorithm is divided into three steps:
1. Estimate the local tangent (up to sign) at each data point using local PCA.
2. Pick signs for all the tangent vectors so that they are globally compatible.
3. Find the order of the points by solving a least-squares problems that assigns values to the points so that local differences approximate the oriented tangent vectors.

The authors test their method and compare it to the Fiedler vector method of Atkins et al. [1], t-SNE [2], UMAP [3] and Recanati et al. (2017), demonstrating superior performance. The paper contains a proof that shows that under certain assumptions, the resulting ordering is highly correlated with the ground-truth order.

### Strengths
* The paper is very clear and a pleasure to read.
* The described method is short and elegant.
* Empirically, the proposed method (slightly) outperforms existing method in terms of error and runtime.
* The paper contains a theorem that shows that, with some assumptions on the data distribution, with probability > 0.99, the order that is found is close to the truth (in a Kendall tau sense). However, I believe there is an issue with the current theorem statement (see below).

### Weaknesses
* As written, there appears to be an error in Theorem 1. The curve is required to be parameterized by an injective function of bounded curvature, which prevents self-intersections. However, it can be *close* to self-intersection. This could cause the local tangent estimation to fail because it would mix points from two branches of the curve.  While a major error in the theorem statement, I don't think it is difficult to fix by adding an assumption on the reach of the manifold (a formal notion of the distance to self-intersection). I think it may enough to require that the radius r used in the algorithm be smaller than reach(\gamma)-2\rho, but please check this carefully.

* The paper claims that methods based on a similarity matrix are sensitive to outliers and thus "can lead to a loss of finer geometric information inherent in the point cloud coordinates" but I don't think this claim is sufficiently supported in the manuscript. In particular, there are no theoretical results that support it. The empirical evaluations show that Recanati (2017) actually performs very well in the empirical comparisons (Table 1, Table 2, Table 3, Table 4), even in high dimensions such as 200.

### Questions
* Why \alpha=2.3 in the simulation in Section 5.1? This value raises suspicion of cherry-picking.

* More of a comment than a question: in addition to the issue raised w.r.t. Theorem 1, I would suggest that you modify the theorem statement such that it holds w.p. > 1-\epsilon for any epsilon. I believe you could even guarantee convergence to the true order w.h.p. under appropriate conditions if you are willing to consider a curve-orthogonal noise model. i.e. where the noise at each point is a Gaussian in the d-1 subspace orthogonal to the tangent of the curve at the point.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces STAGE (Spectral Tangent Alignment and Geometric Embedding), a multi-stage algorithm designed to solve the linear seriation problem, which involves correctly ordering noisy data points sampled from a one-dimensional curve embedded in a high-dimensional space. The method first estimates the local tangent direction at each data point using Principal Component Analysis on its neighbors, then employs a spectral technique to ensure these tangent vectors are globally aligned in a consistent orientation. Following this alignment, the algorithm computes a final one-dimensional embedding by solving a least-squares problem that arranges the points according to this consistent directional flow, with the final order determined by sorting these embedded values. The authors demonstrate that STAGE is computationally efficient and often outperforms existing methods, particularly in high-dimensional and noisy settings, showing its effectiveness on some toy synthetic datasets and for pseudotime ordering in real-world single-cell RNA-seq data.

### Strengths
1. The paper's primary strength is its shift from traditional scalar-based methods to a novel vector-based approach. While algorithms like ISOMAP preserve distances and LLE preserves reconstruction weights, STAGE introduces the more dynamic concept of preserving the manifold's directional flow. Its core innovation lies in estimating local tangent vectors and, crucially, solving the non-trivial problem of aligning them into a globally consistent vector field, a challenge not addressed by the other methods.
2. The contribution is not simply a new heuristic in manifold learning but the authors support their algorithm with a solid theoretical analysis that provides guarantees on its ability to recover the correct ordering. This mathematical foundation, combined with its demonstrated computational efficiency and successful application to complex real-world problems like single-cell RNA-seq pseudotime ordering, makes the method both powerful and practical for researchers.

### Weaknesses
The paper has some weaknesses and I will try to write them down in a somewhat decreasing order of significance that would hopefully help the authors to fix these issues and improve the quality of their paper.

1. A notable weakness of the paper is its insufficient engagement with foundational methods in manifold learning, specifically ISOMAP and Locally Linear Embedding (LLE). While the algorithm is clearly inspired by this field, as acknowledged through a subtle citation to the original LLE paper, it stops short of providing a direct, quantitative comparison (e.g. page 2, lines 102-105). For a new method to establish its place, it is crucial to benchmark it against the very algorithms that represent the benchmarks like in distance preservation (ISOMAP) and in local reconstruction (LLE). By omitting such a direct comparison, the authors leave a critical gap in their analysis, making it difficult for readers to fully appreciate the practical performance gains and trade-offs of their vector-based approach relative to these well-established techniques.
2. Furthermore, the empirical validation of the STAGE algorithm is conducted on datasets of a relatively modest scale, with the largest example containing fewer than 17,000 genes. This scope raises significant questions about the method's scalability and its applicability to the large-scale "big data" problems prevalent today. An algorithm's performance on thousands of samples does not guarantee its feasibility on millions, where computational and memory constraints are really important. The paper would be substantially stronger if it explored how STAGE performs against modern, highly scalable embedding techniques, such as a simple autoencoder with some neural network architecture, which are designed to handle massive datasets efficiently and could potentially offer a more practical solution for a large number of samples.
3. The evaluation of the algorithm's success is also somewhat limited by its focus on the intrinsic quality of the ordering, rather than the extrinsic utility of the resulting embeddings. A truly powerful manifold learning algorithm should produce a low-dimensional representation that is useful for downstream machine learning tasks. A more robust validation strategy, as demonstrated in prior work like Paraskevopoulos et al. (2018) [A], would involve a task-based evaluation. For example, by feeding the embeddings generated by STAGE and other competing algorithms (such as SMACOF, LLE, and ISOMAP) into a standard k-NN classifier and comparing the resulting accuracies, one could obtain a clear, objective measure of how well each method preserves the essential structure of the data for a practical application.
4. Finally, I think the authors should include some computational complexity numbers of their algorithm vs the above referenced SOTA manifold learning algorithms to show maybe that their method is of similar or lower complexity (e.g. execution time or actual memory footprint)

[A] Paraskevopoulos, G., Tzinis, E., Vlatakis-Gkaragkounis, E.V. and Potamianos, A., 2018. Pattern search multidimensional scaling. arXiv preprint arXiv:1806.00416.

I would gladly increase my score if the authors work properly to address the above issues to a proper degree.

### Questions
Considering the novelty of STAGE for vector-based alignment, could you elaborate on its practical trade-offs against foundational methods like ISOMAP and LLE, specifically regarding how its computational complexity scales versus the utility of its embeddings for a downstream task (such as k-NN classification) when applied to large-scale datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a three-step procedure for trajectory seriation consisting of: Local estimation, Alignment, 1D embedding, and provided a theoretical guarantee and simulations to show its superiorty over competing alogithms.

### Strengths
The paper presents a complete algorithmic framework, supported by simulations that demonstrate its advantages over competing methods.

### Weaknesses
While the structure of the approach is clear, the novelty of the contribution appears limited, as each step builds upon existing methods without a substantial new theoretical or algorithmic insight. The only novelty might be the step 3 of the "Constructing a Geometrically-Informed Global Embedding", but I am not sure how much improvement this step gives.

The statement of Theorem 1 is not clearly written and without discussion, hard to understand. The meanings of key quantities such as 𝑓_max and 𝐾_𝜅 are not explained, making it difficult to interpret the result. Moreover, the result does not seem to establish consistency, since the first term on the right-hand side is independent of n. The authors should clarify what the theorem is intended to demonstrate and, if possible, provide a discussion of the asymptotic behavior of the estimator.

To make the paper more compelling and easier to understand, it would be helpful to include a simple illustrative example. For instance, the authors could demonstrate their method on a one-dimensional manifold such as a straight line, a circle, or a half-circle. Such an example would concretely show how the proposed procedure works in practice and highlight its advantages or limitations.

### Questions
See weakness--
1. The meanings of key quantities such as 𝑓_max and 𝐾_𝜅 are not explained, making it difficult to interpret the result.
2. The result does not seem to establish consistency, since the first term on the right-hand side is independent of n. When does the theorem shows that the algorithm performs well with a small error bound on the RHS?
3. The authors should clarify what the theorem is intended to demonstrate and, if possible, provide a discussion of the asymptotic behavior of the estimator. It would be helpful to include a simple illustrative example. For instance, the authors could demonstrate their method on a one-dimensional manifold such as a straight line, a circle, or a half-circle.

### Soundness
3

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
4

### Summary
This paper proposes STAGE, a geometric algorithm for linear seriation — the problem of recovering the intrinsic one-dimensional order of noisy samples drawn from an unknown curve embedded in high-dimensional space. The method proceeds in three main stages: (1) estimating local tangent directions via neighborhood PCA, (2) enforcing consistent global orientation of these tangents through a spectral relaxation of a sign-alignment objective, and (3) constructing a globally coherent 1-D embedding by solving a least-squares system based on the oriented tangents. Theoretical analysis establishes finite-sample recovery guarantees under bounded curvature and noise, providing probabilistic bounds on Kendall’s τ correlation with the ground-truth order. Empirical results on synthetic Fourier-curve data and single-cell RNA-seq pseudotime datasets show that STAGE achieves state-of-the-art rank recovery accuracy and runtime efficiency compared to spectral and manifold-learning baselines such as t-SNE, UMAP, and Recanati et al. (2018).

### Strengths
The paper is exceptionally well written and mathematically coherent, combining geometric intuition with rigorous analysis. Its key contribution is to operate directly on the point-cloud geometry rather than collapsing data into a global similarity matrix, thereby preserving fine-grained local structure. The use of local PCA to estimate tangents, followed by spectral sign alignment, is elegant and computationally efficient. The theoretical results are detailed and nontrivial, bridging tools from differential geometry, random matrix concentration, and spectral graph theory to derive rank-consistency bounds. The empirical section is solid and demonstrates that STAGE performs competitively across a wide range of noise levels and dimensions, often outperforming t-SNE and UMAP while being faster. The combination of geometric consistency, spectral reasoning, and provable robustness makes this work a valuable contribution to the intersection of manifold learning and seriation.

### Weaknesses
While theoretically sophisticated, the method’s scope is limited to one-dimensional manifolds. The authors acknowledge that the orientability assumption and the global sign-alignment procedure fail for higher-dimensional or non-orientable manifolds. Thus, the generalization of STAGE to k>1 manifolds remains open and would require synchronization over SO(k), which is nontrivial. Additionally, several technical assumptions — such as bounded curvature and strong connectivity of the neighborhood graph — may be difficult to verify in practical data. From an empirical standpoint, although the synthetic and biological experiments are convincing, it would be informative to test the algorithm on more diverse real-world domains (e.g., motion trajectories, time-series embeddings). Finally, while the theoretical analysis provides finite-sample guarantees, it focuses on asymptotic concentration without a detailed treatment of bias introduced by approximate tangent estimation, which might affect robustness at small sample sizes.

### Questions
1.	Extension to higher-dimensional manifolds.
Could the tangent-alignment stage be generalized to k>1 dimensional manifolds via frame synchronization over \mathrm{SO}(k)? For instance, do you see a viable path using connection Laplacians or orthogonal Procrustes–type relaxations, or are there conceptual obstacles beyond technical difficulty?
	2.	Sensitivity to noise and curvature / phase transition.
How sensitive is your theoretical guarantee to violations of the “small curvature / sub-Gaussian noise” assumptions? Do you expect (or empirically observe) a phase-transition–type behavior in Kendall’s \tau as curvature, noise level, or tube radius increase—i.e., a regime where recovery abruptly ceases to improve toward \tau \approx 1?
	3.	Asymptotic behavior of the main bound.
In the main theorem, could you comment on the asymptotic rate of each term in the upper bound as a function of n, curvature \kappa, noise level \sigma, tube radius \rho, and graph parameters (e.g., r, \lambda_2(L))? In particular, is it correct that, under your current assumptions, the first summand behaves like a (possibly large) constant determined by geometry and noise, so that only the n^{-1/2} term vanishes with growing sample size? Or is there a natural scaling of r, \rho, etc., under which the first term can also be driven to zero?

### Soundness
3

### Presentation
2

### Contribution
2
