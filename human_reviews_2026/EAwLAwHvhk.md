# Detecting Invariant Manifolds in ReLU-Based RNNs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Recurrent Neural Networks (RNNs) have found widespread applications in machine learning for time series prediction and dynamical systems reconstruction, and experienced a recent renaissance with improved training algorithms and architectural designs. Understanding why and how trained RNNs produce their behavior is important for scientific and medical applications, and explainable AI more generally. An RNN's dynamical repertoire depends on the topological and geometrical properties of its state space. Stable and unstable manifolds of periodic points play a particularly important role: They dissect a dynamical system's state space into different basins of attraction, and their intersections lead to chaotic dynamics with fractal geometry. Here we introduce a novel algorithm for detecting these manifolds, with a focus on piecewise-linear RNNs (PLRNNs) employing rectified linear units (ReLUs) as their activation function. We demonstrate how the algorithm can be used to trace the boundaries between different basins of attraction, and hence to characterize multistability, a computationally important property. We further show its utility in finding so-called homoclinic points, the intersections between stable and unstable manifolds, and thus establish the existence of chaos in PLRNNs. Finally we show for an empirical example, electrophysiological recordings from a cortical neuron, how insights into the underlying dynamics could be gained through our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a new algorithm for identifying stable and unstable manifolds in piecewise-linear recurrent neural networks (PLRNNs). The authors leverage the linearity within each ReLU activation region to derive an efficient, semi-analytical method that constructs manifolds by iteratively propagating sampled points across subregions. This approach enables explicit detection of manifolds that delineate basins of attraction and reveal multistability or chaotic structures in RNN dynamics. The authors demonstrate the method on several systems including the Duffing oscillator, Lorenz-63 attractor, decision-making RNNs, and real neuronal recordings. The results show close agreement between the algorithmic manifolds and those computed analytically or numerically from the underlying dynamical systems, suggesting both accuracy and scalability beyond traditional continuation techniques.

### Strengths
The paper is clearly written and well-organized. The proposed algorithm effectively leverages the piecewise-linear structure of PLRNNs, leading to an elegant and computationally efficient solution. The authors successfully ground their work in established theoretical principles from dynamical systems theory and neuroscience, making a strong case for its relevance to both machine learning and scientific modeling. The experimental section is broad and persuasive, spanning toy systems, classical chaotic dynamics, and real neural recordings, which together illustrate the versatility of the method and its capacity to illuminate the internal mechanisms of learned recurrent dynamics. The addition of invertibility regularization and supporting theoretical analysis further reflects the authors’ careful consideration of the method’s stability and practical applicability.

### Weaknesses
As far as I can tell, the paper does not engage with the existing literature on Threshold-Linear Networks (TLNs), particularly the extensive work by Carina Curto and collaborators. This omission is notable, as TLNs share strong conceptual and mathematical parallels with the proposed framework. The TLN literature provides rigorous theoretical results on how ReLU-based recurrent networks partition state space into distinct regions, each exhibiting its own dynamical behavior—precisely the type of structure this paper seeks to characterize.  

For instance, see [this paper](https://arxiv.org/pdf/2109.03198) and [this paper](https://arxiv.org/pdf/1605.04463).

### Questions
Can the authors please clarify the relationship of PLRNNs to TLNs, and situate their findings within this literature?

### Soundness
3

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
This paper discusses the problem of finding stable and unstable manifolds in a class of recurrent neural networks. The authors treat RNN's involving the ReLU activation function which allows them to treat the system as a piecewise linear system. From what I understand of the paper, the authors exploit the piecewise linear property of the system and use the eigendecomposition of the Jacobian as the basis for computing the manifolds. They have performed some tests on various systems, some simple, some more complex and their algorithm appears to work well.

### Strengths
I found the paper interesting and relevant. I have read numerous papers on computing stable/unstable manifolds in nonlinear systems, but these have typically been in continuous time and I think this paper is one of the few (or perhaps only) paper which treats this specific type of system. It should be observed that generally the computation of stable/unstable manifolds for nonlinear systems is a very hard problem and one which gets harder as the order of the system increases and the dynamics become more complex. 

The writing in the paper was good in terms of well constructed prose and a well-presented manuscript. 

The authors' algorithm appeared to work well on the examples tested.

### Weaknesses
My chief complaint about the paper is it meandered awkwardly between giving the reader quite basic details (and perhaps too much of these) in some places and then depriving the reader of significant details of the authors' contribution. In some places it was not clear what was obvious, what was the authors' work and what was new. The inclusion of a few more references at particular places would be useful. The paper is quite easy to understand up until Section 3.3 when I felt the authors needed to give significantly more detail. In particular I thought:-

1. The authors should give more detail about Algorithm 1, which does not appear to have been described comprehensively. Indeed, it does not seem to referenced in the main text until Section 4. I would like to see a better, more comprehensive explanation of the ideas behind it, its construction, and a more detailed description of its steps and the various mathematical objects it contains. For example in Algorithm 1 $\sigma$ is an index, but what values does it take (one can guess? One expects $Q$ to be the manifold, but in the algorithm it seems like $Q^{\sigma}_k$ is a selection of points in that manifold? I am not sure. Also in Step 4 it seems like $n$ denotes iteration number, but is this actually $k$? These issues are probably resolvable, but without some sort of description, make the paper only possible to appreciate at a superficial level. 

2. I was not entirely sure how the SCYFI package was used. Does this provide the points $P$ for Algorithm 1? Please explain and perhaps also indicate how $N_{max}$ is chosen. 

3. In Section 3.4 the authors discuss regularization which certainly makes sense. However, the details of how this regularizing term is included is not clear: to which "loss" is the term (6) included? In the discussion below equation (6), the authors refer to Figure 3 (top), (bottom) and (right), but this does not seem to correspond with the figures included in the paper - there is only one row of figures in Figure 3. I was confused.

4. I found the appendices quite long and strangely structured. It was not quite clear to me how central some of these were to the paper. For example the first few paragraphs of Appendix D seemed to be only weakly relevant to the paper. Section D.1 seemed relevant, but then Section D.2 seemed rather verbose and un-focused. The relevance of the appendices to the main text needs more focus. Finally, it seems that some sort of "training" is needed for the algorithm reported, according to the appendix, but this does not seem to be discussed in the main text. Some clarity here would be useful for the reader. 

5. Finally a comment about the literature. The system which the authors consider belong to a type of so-called "Lur'e" or "Lurie" or "Lurye" systems considered in the control theory literature. Furthermore the PLL RNN which they consider seems quite close to the piecewise linear systems again considered in the control theory literture. I am not asking the authors to cite any specific papers, but they may find it useful to consult some of this literature.

### Questions
The examples look good, but to me it was not clear exactly what the algorithm returned when computing manifolds - was it just samples from the manifold and then one would deduce from these samples the likely shape of the manifold? This would of course be dependent on the density of samples used, particularly in critical regions of the state-space. Some more explanation and guidance would be useful.

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
4

### Summary
The authors propose an algorithm to detect stable and unstable manifolds in piecewise linear RNNs. The main challenge is moving across regions, and the algorithm handles this by testing for self-consistency across the boundary, and rejecting inconsistent continuations. The authors use the method to locate boundaries between basins of attraction, and also use it to analyze a chaotic system.

### Strengths
Identifying stable and unstable manifolds is an important problem. Most RNN analysis methods deal with fixed points or trajectories.

Limiting the analysis to PLRNN provides a feasible algorithm.

### Weaknesses
Discrete vs. continuous time: The definitions in equation 4 (for instance) are of discrete time, where stability is determined by the norm of eigenvalues. But section 3.3 (and later) describe continuous time conditions – negative/positive real part.

RNNs are usually high-dimensional, as mentioned by the authors. Most examples in the paper are low dimensional. And the high-dimensional examples are only visualized in low-D as a check of their validity.

Many results are qualitative, and only visual inspection is provided as a measure of correctness.

The presentation of the paper is not very clear.  For instance, the argument about curved surfaces. If there is a complex eigenvalue pair, it is associated with two eigenvectors spanning a hyperplane. The description in Section 3.3 is of sampling points on this hyperplane. It does not describe following one specific trajectory (which is usually curved also for real eigenvalues).

### Questions
Line 133: question mark

The stable manifold is a high-D object in general. Figure 1 is a 1D case.

Line 338: should this be figure 2?

Line 365 – this is a relatively high-dimensional example. But the validation of the reconstruction is only done visually in a 3D projection. It would be useful to quantify the quality of the stable manifold. For instance, running the dynamics forward from nearby points.

Scalability. How does the algorithm scale with system dimensionality? In general, validation on high dimensional examples is not clear.

Figure 5: How does the figure show that the intersections specifically are the chaotic attractor? Figure 5B shows points on all the unstable manifold – not just the intersections.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes an algorithm for detecting stable and unstable manifolds of fixed and cyclic points in piecewise-linear RNNs with ReLU activation functions. The authors claim this is the first algorithm for detecting stable/unstable manifolds in ReLU-based RNNs. The paper then utilizes some dynamical system examples such as Lorenz-63 attractors and single-cell recordings to illustrate the proposed algorithm.

### Strengths
- Clarity: detailed math background and derivations, and pseudocodes facilitate the understanding and reproducibility of the proposed approach.

- Mathematically rigorous. 

- Validation on across synthetic and real-world problems. And the algorithm handles high-dimensional models with sublinear scaling in subregion sampling. Its efficiency enables previously infeasible analyses, such as identifying attractor basins in decision-making tasks or chaos signatures in electrophysiological data, bridging theoretical insights with empirical applications.

### Weaknesses
To begin with, I'd like to disclose that I also reviewed this paper in NeurIPS 2025 and I gave quite positive score. Below I just attach my weaknesses when I reviewed this paper in NeurIPS 2025, and the authors' rebuttal in NeurIPS 2025 have addressed nearly all my concerns.

1. I would have liked a bit more concrete discussion on which parts of the algorithm are heuristic and which are exact. And when they are heuristic, how do potential hyper parameter choices affect the outcome, and how are they picked? I will give some concrete examples here:

- Use of approximate algorithm for finding the fixed points: Is there any concern in case one doesn't find all fixed points?
- Sampling seed points: how many are sampled? How are they sampled? It doesn’t seem trivial (or even always possible) to me to e.g., uniformly sample within one region of linear dynamics.
- Propagating seed points. I think one is not always sure we reach all reachable manifolds from a given sub-region? (E.g., if we undersample seed points, or all seed points / most of the space ends up converging in one particular direction).
- The fallback algorithm 3 still doesn’t have any guarantees of finding the right subspace, does it?

2. From results shown it is unclear to me that the regularisation (Eq. 6) actually reaches the desired effect. Besides the effect on training, does it actually lead to networks that are more likely to be invertible? Is there some way to verify this, maybe by comparing the average condition numbers of the to be inverted matrices when running your manifold construction algorithm on a model trained with and without the regulariser?

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3
