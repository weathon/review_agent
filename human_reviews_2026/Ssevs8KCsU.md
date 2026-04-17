# Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
The high-dimensional parameter space of deep neural networks --- the neuromanifold --- is endowed with a unique metric tensor defined by the Fisher information. Reliable and scalable computation of this metric tensor is valuable for theorists and practitioners. Focusing on neural classifiers, we return to a low-dimensional space of probability distributions, which we call the core space, and examine the spectrum and envelopes of its Fisher information matrix. We extend our discoveries there to deterministic bounds for the metric tensor on the neuromanifold. We introduce an unbiased random estimator based on Hutchinson's trace method and derive related bounds. It can be evaluated efficiently with a single backward pass per batch, with a standard deviation bounded by the true value up to scaling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work presents a new approach along with theoretical results for estimating the Fisher Information Matrix (FIM) in deep neural networks. The deterministic upper and lower bounds were examined via the geometric properties in low-dimensional probability simplex spaces, followed by extending to high-dimensional parameter spaces using pullback metrics. This estimator can be computed with just a single backward pass, providing significant computational benefits over existing Monte Carlo and empirical FIM methods for large networks like DistilBERT.

### Strengths
1. Theoretical results are sufficient and rigorous.

2. The ease of implementation in standard deep learning frameworks, which is evident by the empirical validation with better MAE while keeping computational costs similar. The framework's independence from architecture and its ability to apply to both diagonal and low-rank approximations offer flexible trade-offs between accuracy and computational cost.

### Weaknesses
1. Empirical validation relies solely on DistilBERT and the results from two text classification datasets. How computational overhead scales with network depth and width, memory needs for large models?

### Questions
None

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
This paper proposes deterministic bounds and Hutchinson-based stochastic estimators for the Fisher Information Matrix (FIM) on neural manifolds. The method provides unbiased, bounded-variance FIM estimates requiring only one backward pass, making it scalable to large networks. Experiments on DistilBERT show feasible and stable approximations compared with empirical and Monte Carlo estimators.

### Strengths
- Provides a unified and theoretically grounded framework for deterministic and stochastic FIM estimation with clear variance guarantees.  
- The Hutchinson estimator is efficient and easy to integrate into deep learning pipelines, enabling scalable information-geometric analysis.

### Weaknesses
- The paper does not validate the estimator on analytically known probability distributions, especially in high-dimensional settings. This makes it difficult to verify the claimed accuracy of the proposed bounds and stochastic estimates.
- The practical impact of improved FIM accuracy on downstream tasks (e.g., optimization, generalization) is not demonstrated. In some real applications [a], approximation may even be preferable to precision, since neural networks themselves are inherently approximate. The authors should strengthen the discussion and empirical evidence on how their estimator benefits real learning tasks.

[a] Deep CNNs Meet Global Covariance Pooling: Better Representation and Generalization

### Questions
see wk

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an unbiased estimator for the Fisher metric in the parameter space of a neural network, in the setting of multiclass classification, where the underlying geometry is determined by the simplex. The main estimator, along with two additional approximations, is straightforward to compute and leverages automatic differentiation. The authors provide a theoretical analysis to analyze the accuracy and some properties of the proposed estimators. The theoretical claims are further supported by empirical demonstrations.

### Strengths
- The problem of approximating the Fisher metric in neural network settings is timely and very important. The proposed estimators are simple and can be easily computed in practice.
- The technical part of the paper appears to be sound and the theoretical claims correct, but I have not followed every proof in detail. The experiments support the theoretical analysis.
- The paper is generally well written and accessible.

### Weaknesses
- Since the paper proposes an estimator for the Fisher metric, I would expect additional empirical demonstrations regarding the estimator’s accuracy and efficiency. I acknowledge that the theoretical results constitute the primary contribution, but I believe that further benchmark experiments would help demonstrate the properties of the estimator and also consider comparisons to related methods. The current experimental section provides some insight, but additional comparisons would strengthen the paper.
- As the authors mention in Section 5, Hutchinson-type estimators have been used in related contexts. While the extension to neural networks is appreciated and well motivated, as a non-expert in Fisher information analysis, I cannot argue which of the theoretical contributions are novel and which may already be known.

### Questions
Q1. It would be helpful to include a comparison with a Monte Carlo approach in terms of both computational complexity and accuracy.

Q2. Is it correct that the LB estimator typically performs better when the predicted distribution approaches one of the corners of the simplex (see Lemma 3)?

Q3. I suggest including the true Fisher information in the experimental comparison, at least in settings where it is feasible.

Q4. A simple synthetic example where the full Fisher information matrix can be computed and visualized would help demonstrate the behavior of the estimator.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper analyzes the Fisher Information Matrix (FIM) for neural classifiers through deterministic bounds based on spectral analysis and proposes a novel Hutchinson-based estimator. The theoretical development is rigorous with complete proofs in the appendix, though experimental validation remains limited.

### Strengths
Solid mathematical foundation: All major theorems have complete proofs in the appendix. The spectral analysis of the simplex FIM (Theorem 1) and envelope characterizations (Lemma 2) are rigorously established.

Novel computational approach: The Hutchinson estimator with “detach-and-mix” construction provides unbiased FIM estimates with provably bounded diagonal CV ≤ √2, addressing variance issues in Monte Carlo methods.

Comprehensive theoretical analysis: The paper provides both deterministic bounds and stochastic estimates with detailed variance analysis for Gaussian and Rademacher distributions.

### Weaknesses
Experiments: Only DistilBERT tested on AG News and SST-2. Missing: (i) comparison with K-FAC, diagonal empirical Fisher (Adam), exact Gauss-Newton; (ii) optimization performance metrics; (iii) other architectures (CNNs, ResNets). Just I guess a general expansion in this area would be nice, though you note each formulation may require a from-scratch derivation for each. Please comment on this

Unclear practical advantage: No demonstration of how the estimator improves optimization or learning dynamics compared to existing methods.

Terminology issue: “Singular semi-Riemannian metric” is incorrect I believe - the FIM is degenerate PSD, not indefinite.

### Questions
Technical Concerns

Computational complexity underspecified: Power iteration convergence for computing λ_C, v_C could be slow near uniform distributions (small spectral gap). The O(MC|𝒟_x|) cost needs quantification.
Implementation gaps: No guidance on choosing probe count for target accuracy, mini-batch handling, or numerical stability considerations.

Minor Issues

Notation overload (multiple uses of F̂ for both empirical and MC Fisher)
Some bounds are loose (e.g., diagonal upper bound always has error ≥ 1/C). Correct me if im wrong/misinterpret

Questions for Authors

What is the wall-clock time comparison with K-FAC for equivalent accuracy?
Can you demonstrate improved optimization performance in a practical setting?
How does mini-batching affect the variance bounds?

Verdict

The paper makes solid theoretical contributions with rigorous mathematical development. However, it lacks the experimental validation needed to demonstrate practical value. The theory is great and extensive, but the work feels a little missing in terms of comprehensive experiments showing real optimization improvements. But I am also open to understanding if this mainly geared to a paper of more theoretical build up value (or step in that direction) than a fully practical ready usage.

### Soundness
3

### Presentation
4

### Contribution
3
