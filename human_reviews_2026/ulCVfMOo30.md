# Exponential-Wrapped Mechanisms: Differential Privacy on Hadamard Manifolds Made Practical

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4

## Abstract
We propose a general and computationally efficient framework for achieving differential privacy (DP) on Hadamard manifolds, which are complete and simply connected Riemannian manifolds with non-positive curvature. Leveraging the Cartan-Hadamard theorem, we introduce Exponential-Wrapped Laplace and Gaussian mechanisms that achieve $\epsilon$-DP, $(\epsilon, \delta)$-DP, Gaussian DP (GDP), and Rényi DP (RDP) without relying on computationally intensive MCMC sampling. Our methods operate entirely within the intrinsic geometry of the manifold, ensuring both theoretical soundness and practical scalability. We derive utility bounds for privatized Fréchet means and demonstrate superior utility and runtime performances on both synthetic data and real-world data in the space of symmetric positive definite matrices (SPDM) equipped with three different metrics. To our knowledge, this work constitutes the first unified extension of multiple DP notions to general Hadamard manifolds with practical and scalable implementations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces "Exponential-Wrapped" mechanisms for achieving differential privacy on curved geometric spaces (Hadamard manifolds) like hyperbolic space and symmetric positive definite matrices used in medical imaging. Instead of using slow MCMC sampling like existing methods, they simply sample from distributions in flat tangent space and map them to the manifold using the exponential map. This approach is computationally efficient, works for multiple types of differential privacy (ε-DP, GDP, RDP), and achieves better utility-privacy tradeoffs than previous Riemannian privacy methods. They demonstrate strong performance on both synthetic and real-world data, particularly for high-dimensional medical imaging applications.

### Strengths
1) The paper tackles differential privacy for manifold-valued medical data (diffusion tensor imaging, OCT scans) where traditional Euclidean methods fail due to geometric incompatibility. This is increasingly important as healthcare AI systems require both geometric fidelity for accuracy and rigorous privacy guarantees for patient data protection.

2) The EWG mechanism achieves runtime improvements of several orders of magnitude over existing MCMC-based methods, with the speedup increasing in higher dimensions. This makes differential privacy practically feasible for real-world medical imaging applications where previous methods were computationally prohibitive, especially in high-dimensional SPDM spaces.

3) Comprehensive experiments span synthetic data on three different SPDM metrics and hyperbolic space, plus real-world OCTMNIST medical imaging data. The EWG mechanism consistently demonstrates superior utility across 100 Monte Carlo replications, multiple dimensions (d ∈ {3, 10, 15}), and wide privacy budget ranges, with particularly strong performance in high-dimensional regimes where it matters most.

### Weaknesses
1) Performance degrades on affine-invariant metric at low dimensions (d=3) with high privacy budgets due to footpoint misalignment; paper acknowledges this but provides no principled solution for choosing footpoint $p_{0}$

2) Only tests mean estimation; no evaluation on other statistical tasks like principal geodesic analysis or regression despite these being mentioned as important applications

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present a framework for achieving differential privacy with a summary taking values over Hadamard Manifolds (complete, connected Riemannian manifold with non-positive sectional curvature).  The authors achieve this by introducing the "exponential wrapped mechanism", which sanitizes the statistic on a tangent plane before mapping back to the manifold.  They demonstrate that this mechanism can be used to achieve basically any popular notion of DP (pure, approximate, gaussian, and renyi).  The authors provide an interesting theoretical analysis, bounding the noise injected for privacy.  In the case that the sectional curvature is bounded, they show that the noise scales like (1/n) which matches the results for Euclidean geometry.

### Strengths
The results are timely as machine learning on manifolds and with privacy are both active fields.  The framework presented by the authors is fairly complete and the mathematical results are interesting.  Extensive numerical work is presented to highlight the strengths of the approach (both in terms of accuracy and computation).

### Weaknesses
The major weakness of the method is that a footpoint is required to implement the methodology and this footpoint is currently not data driven (for privacy reasons).  This is reasonable for a new method and acknowledged by the authors, but the paper suffers from not having any simulations showing how sensitive their mechanism is to the choice of footpoint (even in the appendix would suffice).  

The other weakness is the limit to non-positive curvature.  Though this is fairly common given how different manifolds are with non-positive and non-negative curvature.

I don't view either of these weaknesses as fatal, though the first one would be fairly easy to address.

### Questions
See weaknesses.

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
In this paper the authors focus on wrapped distributions (distributions defined on tangent spaces and pushed onto the manifold via the exponential map) for Hadamard manifolds to achieve differential privacy.  The authors propose two mechanisms and show how they achieve a variety of DP definitions over many example manifolds. The insight of this paper in Theorem 3.1 shows how designation of the footprint of the mechanism directly effects the statistical utility.

### Strengths
The authors are thorough on manifolds and SOTA definitions of privacy.
The authors focus on: 
-two mechanisms (Laplace and Gaussian) 
-SPDM manifold under three metrics (affine-invariant, log-Euclidean, Log-Cholesky) 
-Hyperbolic space manifold
-three definitions of privacy ( $\epsilon,\delta$-DP, Gaussian DP, and Renyi DP).

Further the authors implement their methodology under varying dimension sizes.

### Weaknesses
A weakness is the lack of focus on varying sample sizes. While I understand that for a fixed sample size one can see how the dimension effects utility (Fig 1.) and hence, in a sense, verifies Theorem 3.1, some experiments on what happens as n increases would be useful.

Theorem 3.1 also needs to be reworked. See questions below.

There are some typos here and there which should be fixed. For instance, many of the citation styles are incorrect in the opening paragraph.

Sometimes $p_0$ is called the footPOINT other times footPRINT. This is one example, but other notations are inconsistent

### Questions
I am having a difficult time understanding the second half of Theorem 3.1. First, does $|Sec_{\mathcal{M}}|$ refer to the determinant or norm of the sectional curvature? The added confusion here is that $K\geq 0$ but this paper is about Hadamard manifolds which have non positive curvature.
(The footnote is poorly placed, at first read the 2 looked like a power and hence $K^2$. Further the footnote itself uses $m$ rather than $\mathcal{M}$)
This is further confusing in line 284 where $K$ is referred to as an upper bound. Perhaps we should have $K\leq0$?

I am not convinced Line 138 is correct "our method only requires a rate of... across all manifolds." It seems the theory is limited to Hadamard manifolds, which clearly is not ALL manifolds.

The sampling section in the appendix should have exp as the second step not log, correct?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Exponential wrapped mechanism for DP on Hadamard manifolds. As these manifolds admit a global exponential map and Lipschitz log map, the Exponential wrapped mechanism draws noise in a Euclidean tangent space and push it forward to the manifolds via Exp. This allows efficient sampling Laplace, and Gaussian noise without MCMC.  They provide utility bounds and some simulation results.

### Strengths
- The observation of using exponential map on Hadamard manifolds give us a clean geometry-aware DP mechanisms, and avoid MCMC burden.
- The paper also provide results beyond pure DP (approximate DP, GDP, and RDP)

### Weaknesses
- The utility guarantees are weak on curved manifolds.  The bound for Frechet mean contains a error term that depends on the footprint $d(p_0, \bar{x})$ which does not go to zero with the number of data $n$ unless $p_0 = \bar{x}$.  Though the paper suggests privately estimating $p_0$, it does not provide a concrete DP procedure for selecting $p_0$.  


Minor comment: 
- the definition of exponential wrapped mechanism can be clearer.  Is $\eta$ in definition 1 the original output $f(D)$?
- Should be footpoint or footprint?

### Questions
Can you provide a concrete DP mechanism for p_0 selection?

### Soundness
3

### Presentation
3

### Contribution
3
