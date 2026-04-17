# A Spectral-Grassmann Wasserstein metric for operator representations of dynamical systems

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
The geometry of dynamical systems estimated from trajectory data is a major challenge for machine learning applications. Koopman and transfer operators provide a linear representation of nonlinear dynamics through their spectral decomposition, offering a natural framework for comparison. We propose a novel approach that represents each system as a distribution over its joint operator eigenvalues and spectral projectors and defines a metric between systems leveraging optimal transport. The proposed metric is invariant to the sampling frequency of trajectories. It is also computationally efficient, supported by finite-sample convergence guarantees, and enables the computation of Fréchet means, providing interpolation between dynamical systems. Experiments on simulated and real-world datasets show that our approach consistently outperforms standard operator-based distances in machine learning applications, including dimensionality reduction and classification, and provides meaningful interpolation between dynamical systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed a representation of each dynamical system as a probability distribution over its joint operator eigenvalues and spectral projectors on the Grassmann manifold. They defined a metric between systems using optimal transport with a cost function that balances spectral information and geometric information, i.e., the eigenspace distances on the Grassmann manifold.

### Strengths
The paper introduces a new metric called Spectral-Grassmann Optimal Transport (SGOT) distance for comparing dynamical systems through their Koopman or transfer operator representations. Its main strength lies in integrating spectral and subspace information within an optimal transport framework, which is an original formulation that helps address several limitations of prior operator-based metrics. The theoretical analysis includes finite-sample guarantees, and the presentation is generally clear. The experiments provide evidence supporting the method’s potential usefulness in practical applications.

### Weaknesses
(1) All experiments use linear kernels. There is no exploration of RBF, polynomial, or learned kernels;
(2) Assumption (A3) requires a shared RKHS $\mathcal{H}$ that simultaneously contains the low-rank images of all Koopman operators. It is unclear how one could construct or validate such a space in real applications. Different dynamical systems may have very different scales and smoothness properties;

### Questions
(1) In Theorem 2, how would you verify the conditions $[(C_x^k)^{\dagger}]^{\frac{\alpha - 1}{2}}T_k$ and $\lambda_i(C_x^k)^{\dagger}$ by data?
(2) Theo theory replies on assumptions A1-A3, which is convenient to compare between the operator from different systems, but are they assured in practical cases?
(3) When $\lambda_j$ has multiplicity $m_j>1$, how is the space $V_j$ uniquely defined?
(4) Do we also need to show that $V_j$ is independent of basis choice?
(5) Is $d_{\mathcal{S}}(T, T')$ independent of the choice of bi-orthogonal basis used to represent spectral decompositions? It is unclear whether $d_{\mathcal S}(T, T')$ is intrinsic, i.e., independent of the choice of bi-orthogonal basis used to represent the spectral decomposition when eigenvalues have multiplicity > 1.

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
This paper addresses the problem of quantifying and comparing the geometry of dynamical systems from trajectory data, a central challenge in machine learning for dynamical systems. The authors build on the Koopman and transfer operator framework, which provides a linear representation of nonlinear dynamics via spectral decomposition.
They propose a new operator-based metric that represents each system as a joint distribution over operator eigenvalues and spectral projectors, and defines distances between systems using optimal transport theory. The resulting metric is invariant to the sampling frequency of trajectories, computationally efficient, and theoretically supported by finite-sample convergence guarantees. Furthermore, it allows the computation of Fréchet means, enabling smooth interpolation between dynamical systems.
Comprehensive experiments on both synthetic and real-world datasets demonstrate that the proposed method consistently outperforms standard operator-based distances in machine learning tasks such as dimensionality reduction and classification, while also yielding meaningful interpolations between dynamical systems.

### Strengths
This paper tackles a fundamental and timely problem in machine learning for dynamical systems — defining a theoretically sound, interpretable, and computationally tractable metric between data-driven operator representations of nonlinear and stochastic dynamics. The proposed Spectral-Grassmann Optimal Transport (SGOT) framework elegantly combines spectral theory, Grassmannian geometry, and optimal transport, yielding a true metric that is invariant to sampling frequency and robust to operator estimation errors. The method is mathematically principled, supported by finite-sample convergence guarantees, and remains computationally efficient, overcoming the typical trade-off between theoretical rigor and practicality. Extensive experiments demonstrate that SGOT outperforms existing operator-based similarities in both unsupervised and supervised learning tasks, and its ability to compute Fréchet means enables meaningful interpolation between dynamical systems — a capability with broad implications for model comparison, system identification, and generative modeling of dynamics. Overall, the paper makes a significant and original contribution to bridging operator-theoretic dynamical systems and modern machine learning.

### Weaknesses
While the proposed SGOT framework is theoretically elegant and empirically strong, the paper could be further strengthened by a clearer characterization of its practical limits. In particular, it remains unclear how strongly nonlinear systems the approach can accurately handle, and how robust the method is under varying levels of stochastic noise in trajectory data. Although the finite-sample convergence guarantees are valuable, the mathematical conditions quantifying robustness to nonlinearity and noise are not explicitly derived. Providing such analyses — for example, bounds on operator estimation errors or perturbation stability with respect to noise amplitude — would help clarify the method’s applicability range and further enhance its theoretical completeness.

### Questions
1.	To what extent do the theoretical properties of the Koopman operator, such as boundedness or continuity, play a critical role in the proposed SGOT framework?
2.	Regarding the domain and codomain of the Koopman operator, what kind of function spaces are suitable for ensuring the validity of the spectral–Grassmann representation? For example, does the analysis require specific regularity, integrability, or compactness assumptions on observables?
3.	Can the proposed approach be extended to highly complex dynamical systems, such as stochastic nonlinear partial differential equations (SPDEs), where both randomness and infinite-dimensional state spaces come into play? If so, what are the main theoretical or computational challenges expected in such settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Spectral-Grassmann Optimal Transport (SGOT), a metric that represents each linear operator by its eigenvalue subspace atoms and compares operators by a Wasserstein distance. The authors show how to compute this metric via kernel representations, provide statistical guarantees for finite-sample estimation, and propose a parametric formulation to compute barycenters in practice. Empirical results on synthetic systems and fluid flow data illustrate that SGOT responds to changes in frequency, decay and subspace structure compared with the baselines, better performances in ML tasks, and can produce visually meaningful interpolated modes.

### Strengths
The paper proposes a coherent, high-level framework that combines spectral information (eigenvalues) with subspace geometry (Grassmann) under an optimal-transport view, which is an original and conceptually clean way to compare linear dynamical operators.
Methodologically, the authors introduce a concrete, computable metric (SGOT), give a parametric barycenter formulation and an optimization scheme that bridges the abstract definition and practical computation.
Experiments demonstrate the metric’s sensitivity to controlled changes, its usefulness for ML tasks (classification/embedding), and the interpolation behavior on physical flow data.

### Weaknesses
The paper’s choice of distances is only one possible design: Eqs. (4)  and (6) are reasonable, but the authors do not justify them or compare to clear alternatives. The empirical claims may need clear explanations. Finally, there are some poor presentation issues. Please see the questions below in detail.

### Questions
1. I agree that summing an eigenvalue distance and a subspace distance in Eq. (4) is reasonable, but it’s just one possible choice. For example, distances based on Koopman trace/determinant kernels (Fujii et al., 2017) could also be defined. Please explain why you chose the current form and mention other viable alternatives.

2. Similarly, the second term in Eq. (6) is only one concrete realization of a subspace distance. Eq. (6) evaluates subspace differences as a quadratic form with kernel matrices and effectively uses a trace kernel-like representation. Why did you select this particular form? Please discuss alternatives (e.g. principal angles) and comment on related prior work (e.g. Kawahara et al., 2016 [a]), which is not cited but is relevant to distances in Koopman spectral decomposition.

3. The results in Fig. 1 can be interpreted, but the current explanation lacks enough information for a reader to judge whether SGOT is truly better. In particular, please justify the claim that “grows linearly with the shifts almost everywhere” is a desirable property and explain why this behavior indicates superiority.

4. Figure 5 may visualize the intermediate modes computed by SGOT, but a barycenter does not necessarily correspond to a physically realizable intermediate system. To make Fig. 5 more convincing, please evaluate the barycenter against concrete metrics that relate to physical realism. For example, mode frequencies, decay rates, and reconstruction/forecast RMSE.

5. The experiments report that Koopman operators are estimated with a linear kernel (including Appendix H). Please state why you chose a linear kernel (interpretability, computational reasons, or data suitability) and discuss the relationship with non-linear kernels such as RBF.

Minor points
6. From Section 2 onward you use \cite where \citep would be appropriate.

7. Line 308 contains two periods in a row.

8. Figure 2 appears after Fig. 5 and a label seems to be “ilbert”.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel metric, named the Spectral-Grassmann Optimal Transport (SGOT) metric, for comparing dynamical systems represented by Koopman or transfer operators. The core idea is to represent each dynamical system as a probability distribution over its joint spectral data, specifically its eigenvalues and corresponding spectral projectors (viewed as points on a Grassmann manifold). The distance between two systems is then defined as the Wasserstein distance between these two distributions. The ground metric for the optimal transport problem is a weighted combination of the distance between eigenvalues and a Grassmannian metric between the projector subspaces. The authors provide theoretical guarantees for the metric, including finite-sample convergence rates for its estimation. They also propose an algorithm to compute the Fréchet mean (barycenter) of a set of dynamical systems under this metric, enabling interpolation. The method's effectiveness is demonstrated through experiments on simulated and real-world datasets, where it is shown to outperform other operator-based metrics in tasks like dimensionality reduction and classification, and provides more meaningful interpolations.

### Strengths
1.  **Principled Metric Definition:** The SGOT metric is well-founded in theory, combining optimal transport with the geometry of Grassmann manifolds. It offers a holistic comparison of operators by considering both their spectral values (eigenvalues) and their geometric structure (eigenspaces).
2.  **Theoretical Guarantees:** The paper provides finite-sample convergence guarantees (Theorem 2), which is a significant theoretical result that adds rigor to the proposed data-driven metric.
3.  **Key Invariances:** The metric is designed to be invariant to the sampling frequency of trajectories and permutations of the spectral decomposition, which are desirable properties for a robust similarity measure.
4.  **Enables Geometric Operations:** The framework naturally allows for the computation of barycenters, providing a principled way to average and interpolate between dynamical systems. The experimental results show this leads to more meaningful interpolations than simpler methods.

### Weaknesses
1.  **Strong and Potentially Impractical Assumptions:** The requirement of a "Common functional space" (A3) is a major limitation. It is not clear how one would verify this assumption in practice or how to choose a suitable RKHS when comparing a diverse set of dynamical systems. The paper does not address the sensitivity of the method to violations of this assumption.
2.  **Limited Generality of Guarantees:** The statistical guarantees are tied to a specific operator estimation method (RRR) and kernel type. It is unclear if these guarantees would extend to other popular and powerful estimation methods, such as those based on neural networks.
3.  **Scalability Concerns:** The metric computation involves calculating a cost matrix and solving an OT problem. While the paper claims it is computationally efficient for small ranks (`r`), the complexity `O(n^2 * r^2)` can be prohibitive for high-resolution data (large `n`) or when a large number of modes are needed to accurately represent the system (large `r`). The barycenter computation adds another layer of iterative optimization.
4.  **Limited Experimental Scope for Barycenters:** The interpolation experiments are primarily demonstrated on simple 1D linear systems. The single fluid dynamics example is more compelling, but a more extensive evaluation on a wider range of complex, nonlinear systems is needed to fully validate the claimed superiority of SGOT for interpolation and averaging tasks.

### Questions
1.  How can a practitioner verify or enforce the "Common functional space" assumption (A3) when given a collection of time-series datasets from potentially very different underlying systems? What are the consequences for the metric if this assumption is violated (i.e., the relevant eigenspaces do not lie in the chosen RKHS)?
2.  Your statistical guarantees are derived for the RRR estimator. Could you comment on the challenges of extending these guarantees to Koopman operator estimators based on deep neural networks, which are widely used for complex systems? Does the metric `ds` remain well-behaved if the estimated operators `T_k` come from different model classes?
3.  Regarding scalability: What is the practical limit on the number of samples (`n`) and the rank (`r`) for which the SGOT metric and its barycenter can be computed in a reasonable amount of time? How does this compare to the scale of problems typically found in fields like fluid dynamics or climate science?
4.  In Figure 1, the SGOT metric shows a desirable linear-like response. However, this is dependent on the hyperparameter `η`. The paper mentions a sensitivity analysis in the appendix but could you elaborate on how `η` should be chosen in practice? Does it require extensive cross-validation, and does the optimal `η` vary significantly across different tasks and datasets?

### Soundness
3

### Presentation
3

### Contribution
3
