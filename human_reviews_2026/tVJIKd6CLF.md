# Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Learning population dynamics involves recovering the underlying process that governs particle evolution, given evolutionary snapshots of samples at discrete time points. Recent methods frame this as an energy minimization problem in probability space and leverage the celebrated JKO scheme for efficient time discretization. In this work, we introduce ``iJKOnet``, an approach that combines the JKO framework with inverse optimization techniques to learn population dynamics. Our method relies on a conventional *end-to-end* adversarial training procedure and does not require restrictive architectural choices, e.g., input-convex neural networks. We establish theoretical guarantees for our methodology and demonstrate improved performance over prior JKO-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is an attempt to use the (famous) JKO optimization scheme to learn a dynamical system from a time- series $\rho_1, \cdots, \rho_M$  where $\rho_i$ represent the data of the system at time $T_1< T_2 < \cdots < T_M$.  The JKO scheme are expressed as proximal operator of the $W_2^2$ optimal cost with a functional (typically of the form energy plus entropy plus possibly interaction terms described by a kernel).   The approach taken here is to parametrize a transport map (for the $W^2$ term) and to parametrize  the energy terms using neural architectures and to estimate the entropy terms via an entropy estimator.

### Strengths
The JKO formulation of gradient flow as a proximal optimzation problem is a fundamental tool in probbaility theory.  It has been found difficult to use it effectively as a generating tool and and the reformulation of the JKO functional in this paper  is interesting and could be promising.

### Weaknesses
1) The main weakness is that the paper does not fully implement the proposed loss functional. Indeed the the interaction terms and the entropy terms are ignored and the same architecture as in JKONet is used.  This reduces the class of PDEs considered to a very small class of linear PDE. The mismatch between the theory which considers general PDEs and the implementation is too great to make the paper convincing.   Ignoring the entropy term,  which I suspect is very difficult to estimate,  is a serious defect of the approach.  

2) The proposed architecture requires the construction of a transport map which is both expensive and only works theoretically when the measures have density.  The authors propose to replace input convex neural networks (since the transport maps are gradient of a convex functions) by a more general architecture but I do not see why this will make the problem more scalable or stable.  Maybe a approach using triangular map as in the Yousef Mazrouk group at MIT may be more scalable?

### Questions
1) Can you clarify what are the obstacles to implement an interaction kernel and the entropy term in the algorithm?

2) Can you comment on the difficulty of computing the transport maps?

### Soundness
3

### Presentation
3

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
In this paper, the author propose  a method to learn the population dynamics, which combine JKO framework with inverse optimization techniques. In addition, this method relays on adversarial training procedure. Theoretical guarantees are also provided in this paper.

### Strengths
1. This paper is well written.

2. The figures are well-prepared and greatly facilitate the understanding of the content.

3. The theoretical proofs is provided, which give solid theoretical guarantees of the proposed method.

4. The background section is clearly presented and offers helpful context.

### Weaknesses
1. The interchange of the min and Σ operators between Equations (10) and (11) lacks justification or analysis. It is unclear under what conditions this exchange is mathematically valid.

2. The method relies on an adversarial training procedure, which is known to introduce instability. It would be important to discuss whether any measures were taken to mitigate this issue.

3. The work builds heavily on existing studies (e.g., Terpin et al., 2024; JKOnet∗). The specific contributions and novelty of the present approach should be more clearly emphasized.

4. Several relevant references appear to be missing from the related work section, such as:

[1] Zhang Z, et al. Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schr\" odinger Bridge[J]. arXiv preprint arXiv:2505.11197, 2025.

[2] Li R, et al. WeightFlow: Learning Stochastic Dynamics via Evolving Weight of Neural Network[J]. arXiv preprint arXiv:2508.00451, 2025.

[3] Zhang Y, Levin M. Equilibrium flow: From Snapshots to Dynamics[J]. arXiv preprint arXiv:2509.17990, 2025.

### Questions
- Could you explain about the potential stability of the proposed method resulted by the adversarial training procedure, and what further improvements could be made?

- Could you more explicitly summarize the key algorithmic or theoretical innovations that distinguish your approach from its predecessors?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces iJKOnet, an approach to learning population dynamics by bridging inverse optimization techniques with the Jordan-Kinderlehrer-Otto (JKO) scheme. The method frames the task as a min-max optimization problem for identifying the underlying energy functional that governs observed distributions, enabling end-to-end adversarial training without restrictive neural architectures. Theoretical guarantees are provided for recovery of potential-driven dynamics, and empirical results on both synthetic and real datasets show improved performance over prior JKO-based baselines.

### Strengths
- iJKOnet is different from previous methods that either require potential-only energies or precomputed OT couplings. The formulation of energy functional recovery using inverse optimization within the JKO scheme provides a conceptually clear route to modeling population level dynamics from discrete snapshots, directly leveraging optimal transport geometry.
- Theorem 3.1 gives a non-trivial quality guarantee, explicitly bounding the distance between the gradients of learned and ground-truth potential functions in terms of the inverse optimization gap. This is a concrete advance compared to many contemporary works that lack precise recovery guarantees.
- iJKOnet dispenses with input-convex neural networks for parameterizing transport maps, allowing the use of more scalable and widely applicable neural network architectures such as MLPs and ResNets

### Weaknesses
- The main theoretical guarantee (Theorem 3.1) exclusively addresses potential energy functionals with strongly convex smooth potentials, while the practical method is claimed to work for broader energy functionals (e.g., including interaction and internal terms). This leaves a theoretical gap between what the analysis covers and what is demonstrated empirically. Specifically, the inability to provide quality bounds for non-potential and more general functionals limits the rigor of the claims. (This could explain why the performance of iJKOnet is much worse than baseline in Table 2)
- While iJKOnet is, in principle, applicable to full free-energy functionals, all main experiments revert to only learning potential energy, explicitly setting interaction and entropy terms to zero. As a result, the empirical support for learning richer dynamics is missing, and the scalability to very high-dimensional or truly multimodal interaction-dependent data is unproven.
- Some recent and directly related works [1,2,3,4] on similar task with unknown parameters, alternative population-level diffusion modeling, and advanced regularization techniques are not discussed.

ref:

[1] Computational and Statistical Asymptotic Analysis of the JKO Scheme for Iterative Algorithms to update distributions

[2] An Eulerian approach to regularized JKO scheme with low-rank tensor decompositions for Bayesian inversion

[3] WeightFlow: Learning Stochastic Dynamics via Evolving Weight of Neural Network

[4] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrodinger Bridge

### Questions
- In which regimes or datasets is JKO-based modeling justified or superior to flow-matching, neural ODEs, or Schrödinger bridge-based methods?
- How sensitive is iJKOnet to missing, irregularly spaced, or noisy population snapshots? Do learned energy functionals overfit to sampling artifacts in sparse settings?

### Soundness
3

### Presentation
3

### Contribution
3
