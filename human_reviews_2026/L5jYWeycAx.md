# Learning on a Razor’s Edge: Identifiability and Singularity of Polynomial Neural Networks

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8, 8

## Abstract
We study function spaces parametrized by neural networks, referred to as neuromanifolds. Specifically, we focus on deep Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) with an activation function that is a sufficiently generic polynomial. First, we address the identifiability problem, showing that, for almost all functions in the neuromanifold of an MLP, there exist only finitely many parameter choices yielding that function. For CNNs, the parametrization is generically one-to-one. As a consequence, we compute the dimension of the neuromanifold. Second, we describe singular points of neuromanifolds. We characterize singularities completely for CNNs, and partially for MLPs. In both cases, they arise from sparse subnetworks. For MLPs, we prove that these singularities often correspond to critical points of the mean-squared error loss, which does not hold for CNNs. This provides a geometric explanation of the sparsity bias of MLPs. All of our results leverage tools from algebraic geometry.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper studies the function spaces corresponding to CNNs and MLPs with polynomial activations using tools from algebraic geometry. Due to polynomial activations, these function spaces are manifolds. The authors show that almost everywhere, each function on this manifold corresponds to at-most finitely many parameter values of the network (the identifiability problem) and stronger result for CNNs (almost everywhere uniformly identifiable). The authors also characterize the singularities of the neuromanifolds. 


I want to note that I have no training in algebraic geometry and most of the results and ideas in this paper appeared opaque to me. I have set my confidence level to be low to reflect this fact.

### Strengths
The paper tackles the important question about the class of functions represented by neural networks. While standard results with non-linear non-polynomial activation functions consider approximation properties, this considers a different of view: that of characterizing the set of functions exactly represented by neural network families. The paper answers some deep questions about this function space such as identifiability and singularity.

### Weaknesses
- The authors claim that the identifiability results were previously only known for sigmoid and tanh activation functions. Given that they are more widespread in ML than polynomial activations, I would say that the current results are not as relevant to the community. 

- It does not directly give us relevant insights into practically relevant activation functions such as ReLU or sigmoid.

### Questions
- Theorem 4.1 requires $r$ to be large enough. Is there a quantitative estimate about how large it should be? 

- In Theorem 4.2 it is not clear what "enough coefficients of \sigma are non-vanishing" means. 

- The term "generic polynomial" is used throughout the paper, but I could not gather the meaning of this term. This could be because I am not trained in algebraic geometry.

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
3

### Summary
The paper studies the (dis)connectedness and the existence of singularities in the parameter space of ReLU neural networks whose architectures admit a DAG computational graph representation. The authors then leverage these results to study whether these elements can exert impact on the dynamics of gradient flow (GF) of standard training algorithms.

### Strengths
I generally like the paper. Here are several remarkable points:
1. The conservative law for ReLU networks in the DAG case is very elegant. 
2. Theorem 1 makes a very nice connection to flow problem in graph theory.
3. Theorem 2 - Proposition 5 to 7 provide a very clean picture on the dynamics of gradient flow in the existence of singularities

### Weaknesses
1. I have a hard time distinguishing what are the main contributions and what are already proved in the literature. Authors might want to re-organize the section 2, and credit properly all the results (theorems, propositions, definitions) if they are ever taken/inspired by previous works.

2. Do the author forget to define the notion of stable by forward/backward edges in the announcement of Theorem 1? Otherwise, I believe that Theorem 1 needs rephrasing to be easier to understand.

### Questions
1. Do Proposition 6 - 7 imply that singularities are truly rare? It seems to me that the limit of GF can still be a singularity (or a sparse subnetwork). If the GF dynamics does not bias towards sparse subnetworks, do you have any idea which points are preferable for the convergence of GF?

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
The paper investigates the geometry of neural networks with polynomial activation functions through algebraic geometric tools. The authors claim three main contributions:
Identifiability results: For MLPs with generic polynomial activations, almost all functions have finitely many parameter representations; for CNNs, the parametrization is generically one-to-one.
Singularity characterization: Sparse subnetworks constitute singular points of neuromanifolds for both MLPs and CNNs.
Critical exposedness: Subnetworks of MLPs are "critically exposed" (contain critical points of the loss with positive probability), providing a geometric explanation for sparsity bias. CNNs do not exhibit this property.

### Strengths
Important theoretical questions: The paper addresses fundamental issues in deep learning theory - identifiability, singularities, and sparsity bias - that have significant implications for understanding optimization and generalization.
Novel geometric perspective: Connecting sparsity bias to singularities of neuromanifolds is creative and could provide new insights into the lottery ticket hypothesis.
Architectural comparison: The distinction between MLP and CNN geometry regarding critical exposedness is novel and aligns with empirical observations about their different behaviors.

### Weaknesses
The paper relies heavily on citations in a way that makes the intuition of the proofs difficult to follow . I would appreciate a more self contained mathematical exposition. I would be happy to improve my rating if more exposition was provided, as well as the following are addressed. 

Issues with specific proofs: 
Theorem 4.1 (MLP Identifiability)
- The constraint β₁ > 6m² - 6m appears without much justification. Why this specific bound?
- The dependence on this proof and a good bit of discussion in the paper on the Zariski topology necessitates a more thorough exposition/ explanation of this topological space, as it is a rather different topology than commonly used in Learning. 

Theorem 4.2 (MLP Singularities)
- There should be more said about the dominance of σ ◦ fW′  . In particular it isnt clear that the image of σ ◦fW′ having a non-empty interior is sufficient to show Zariski density .

Theorem 4.6 (CNN Singularities) 
More said at all points.

### Questions
Add concrete examples: Provide explicit small examples (e.g., 2-layer networks) where singularities and exposedness can be computed directly, if possible.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies the function spaces (“neuromanifolds”) of deep MLPs and CNNs with polynomial activations, using tools from algebraic geometry. The core results:

Identifiability. For MLPs with a generic sufficiently high-degree polynomial activation, the parametrization is generically finite-to-one; hence the dimension of the neuromanifold equals the number of parameters (Theorem 4.1). For CNNs, it is generically one-to-one and regular off the zero fiber (Theorem 4.4).

Singularities. Subnetworks (deactivating neurons/filters) yield singular points: fully characterized for CNNs (Theorem 4.6) and partially for MLPs (Theorem 4.2).

Optimization bias. The paper introduces “critically exposed” parameter sets. Strict subnetworks of MLPs are critically exposed for quadratic losses (Theorem 4.3), but not for CNNs (Proposition 4.5). This gives a geometric account of sparsity bias in MLPs.

### Strengths
Conceptual advancement. Provides a clean algebraic–geometric framework for identifiability and singularity in deep networks with polynomial activations, extending prior results beyond monomials.

Generality of MLP result. Finite identifiability for generic polynomials closes a dimension conjecture (dimension = #params) and generalizes known tanh/sigmoid cases to large-degree polynomials (Theorem 4.1).

CNN result. The regularity and generic injectivity of CNN parametrizations off the zero fiber (Theorem 4.4) is technically strong and explains why CNN singularities are mild and do not create optimization equilibria.

Sparsity bias lens. The critical exposedness notion and proofs (Theorem 4.3 vs Prop. 4.5) give a principled account of why MLPs tend to collapse to sparse subnetworks whereas CNNs typically recover from near-zero initializations—matching known empirical phenomena.

### Weaknesses
Activation assumptions: Many results require “generic” high-degree polynomials (often with σ(0)=0 and nonzero top coefficients). Practical popular activations (ReLU, GELU, tanh) are non-polynomial; while the authors argue approximation plausibility (Remark 4.1, §5), formal transfer to non-polynomial nets is not proven.

Optimization link: While “critically exposed” is compelling and geometric, the paper doesn’t classify whether exposed subnetworks are local minima vs saddles, which matters for SGD dynamics and generalization.

### Questions
1) Can you sketch the proof of  the limit argument suggested in Remark 4.1: under what conditions do MLP/CNN singularities persist under uniform polynomial approximation of non-polynomial activations?

2) Are there settings where subnetwork critical points in MLPs are provably local minima with non-negligible measure?


3) Can dimension counts for U_S (Eq. 8) yield measure bounds for how often subnetworks are equilibria?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies the function spaces (neuromanifolds) parameterized by deep MLPs and CNNs with generic high-degree polynomial activations. It proves finite identifiability for MLPs (generic outputs come from finitely many parameters) and generic one-to-one identifiability for CNNs, hence the dimension equals the number of parameters in both settings. It then characterizes singular points of these neuromanifolds: (i) for MLPs, many subnetworks yield singularities and are shown to be critically exposed (they occur as critical points of squared-loss for a set of targets with nonempty interior); (ii) for CNNs, all singularities are precisely the subnetworks with edge zero-padding that satisfy an integrality constraint, and such sets are not critically exposed (away from the zero fiber). This geometric picture explains sparsity bias in MLPs but not in single-channel CNNs and resolves a long-standing dimension conjecture from prior work on polynomial networks. Figures 1--2 visualize how subnetworks create singular points and how MLP cuspidal-type vs. CNN nodal singularities differ.

### Strengths
* Originality: Moves from monomial/linear models to generic polynomial activations; formalizes critical exposure; delivers complete CNN singularity characterization. 
* Quality: Careful use of fiber-dimension, Vandermonde invertibility, and toric lattice ideals; clean separation between parametrization criticality and image singularity (Appendix A). Proof architecture is transparent via modular lemmas. 
* Clarity: Clear definitions of neuromanifolds/subnetworks; optimization setup with quadratic loss; intuitive figures (Fig. 1--2) and didactic examples (nodal vs. cuspidal curves in Fig. 3) to illustrate singular behaviors. 
* Significance: Fixes dimension = parameter count generically for both MLPs and CNNs; connects geometry to sparsity bias and to the presence/absence of spurious critical points across architectures.

### Weaknesses
* Generality vs. practicality: Many results require very high polynomial degree and generic coefficients. Concrete degree thresholds are not explicit beyond \(r \gg 0\) (depends on architecture). Giving quantitative bounds (even conservative ones) would improve applicability. 
* Model scope: CNN analysis is single-channel, 1D; multi-channel and higher-D details are asserted “similarly” but not proved. Because modern CNNs are multi-channel, a pathway or obstacles to generalization would be valuable. 
* Singularity coverage in MLPs: The paper shows many singularities arise from subnetworks but leaves open whether all singularities do (contrast with linear MLPs and with CNNs, where a full characterization is given). Clarifying non-subnetwork singularities would strengthen the picture. 
* Type of critical points: Criticality vs. local minima vs. saddles is not analyzed; given the optimization motivation (sparsity bias), even partial results or conjectures on stability types would be informative. 
* Beyond polynomial activations: While approximation arguments are discussed (Remark 4.1), formal extension to ReLU/Tanh/Softmax is left for future work; clarifying which parts port over under approximation limits would broaden impact.

### Questions
Non-polynomial activations: Can the polynomial approximation idea in Remark 4.1 be made quantitative (e.g., stability of singularity types under uniform approximation on compact sets)? Which parts of identifiability/exposedness survive in the ReLU or tanh settings?

### Soundness
3

### Presentation
3

### Contribution
3
