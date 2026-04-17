# Deep Neural Networks Divide and Conquer Dihedral Multiplication

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
We find multilayer perceptrons and transformers both learn an instantiation of the same divide-and-conquer algorithm and solve dihedral multiplication with logarithmic feature efficiency. Applying principal component analyses to clusters of neurons where neurons have similar activation behavior reveals remarkably clear structure: \textit{the neural representations correspond to Cayley graphs.} It's noteworthy that these Cayley graphs are the low-dimensional manifolds networks are predicted to learn under the manifold hypothesis, and thus each neural representation learned by networks corresponds to a manifold. To our knowledge, our novel methodology allows this to be the first work that fully characterizes and describes the neural representations distributed over many neurons on a task.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper provides evidence for the universality hypothesis by demonstrating that both multi-layer perceptrons (MLPs) and transformers learn the same abstract divide-and-conquer algorithm to solve dihedral group multiplication. The authors employ a methodology using the Group Fourier Transform (GFT) to cluster neurons with similar activation patterns. Subsequent Principal Component Analysis (PCA) on these clusters reveals that the network's distributed neural representations form distinct, geometric structures identified as Cayley graphs. The global algorithm then works by combining the outputs of these different Cayley graph representations—each solving a simpler subproblem—to maximise the logit for the correct answer.

### Strengths
1. Novel Perspective on Neural Representations: The shift from analysing single neurons to the emergent structures formed by a neuron cluster is more appropriate under the superposition hypothesis. This goes beyond prior work that modelled individual neuron activations with sinusoidal functions to provide evidence for discrete distributed representation, showing how sinusoidal components collectively build a higher-level geometric object (a Cayley graph).
2. Strong Methodology: The task of dihedral multiplication is rigorously analysed from a theoretical perspective to arrive at an appropriate hypothesis for the behaviour of these deep neural networks. The subsequent implementation of GFT and PCA provides compelling evidence that the proposed algorithm is being implemented. Furthermore, the replication of results across different architectures, random initialisations, and numerous (i.e., different values of n) problem instances strongly supports the claim.

### Weaknesses
1. Claim of Universality: Although the authors consider multi-layer perceptrons and transformers, the consideration of more fine-grained architectural choices (i.e., depth, width, activation function) is not considered. To provide comprehensive evidence for the universality hypothesis, the effects of these architectural choices, as well as optimiser, training hyperparameters, etc, would need to be considered.
2. Theoretical Justification: There is little discussion as to why deep neural networks learn this particular algorithm compared to other algorithms for performing dihedral multiplication. 
3. Presentation: A significant portion of the paper is devoted to introducing the problem, reviewing related work, and presenting figures. It would perhaps be more appropriate to provide the results of various ablation studies to support the claim of universality.

### Questions
1. Does the emergence of this algorithm appear at a particular point in training, say as a grokking effect?
2. How do other architectural factors – such as activation function, depth and width – influence the emergence of the divide and conquer algorithm?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates how deep neural networks (MLPs and transformers) learn to perform multiplication in the dihedral group, which is 
a non-commutative finite group representing the symmetries of an n-gon.

The authors claim that neural networks learn a divide-and-conquer algorithm for this operation. 
Their internal representations correspond to Cayley graphs and coset structures of the group. This supports a “universality hypothesis”: networks trained on algebraic tasks consistently discover similar algorithmic structures, 
regardless of architecture or random seed.

### Strengths
- Very interesting approach to gain insight into learning
- The “divide-and-conquer” hypothesis is quite reasonable and insightful
- Comprehensive experimentation: many thousands of training runs were studied
- High reproducibility: many experiments over random seeds
- The manuscript is well written

### Weaknesses
The paper is essentially an exploratory analysis rather than a hypothesis-driven study. 
No explicit, testable hypotheses are formulated or quantitatively evaluated.
Most conclusions rest on visual inspection of PCA- and Group Fourier Transform -derived figures, whose interpretations are ambiguous.
It remains unclear whether these structures reflect genuine inductive biases or are artifacts of the analytical lens itself - especially given that the Group Fourier Transform already encodes the group structure being “discovered.”
This raises a risk of circular reasoning.
To be more convincing, the authors should (i) define explicit hypotheses about what structures should appear under what conditions, (ii) quantify the strength or prevalence of these patterns statistically, and (iii) include negative or counterexample experiments where no such structure is expected.
Such controls would clarify whether the observed patterns genuinely reflect learned algorithmic structure rather than the setup or analysis.

In this sense, the approach is a visually appealing first step, but scientifically, it is wholly unclear what the results even mean.

### Questions
1. Could the authors formulate explicit, testable hypotheses about what structures (e.g., Cayley graphs, cosets, scaling laws) should appear under specific conditions?
2. How would those hypotheses be falsified -- what results would not support the proposed divide-and-conquer or universality interpretation?
3. Have the authors considered applying the same training and analysis pipeline to a problem where such structured behavior is not expected (e.g., random multiplication tables, corrupted group laws, or non-algebraic mappings)?
4. Would the same Cayley-graph patterns and O(log n) scaling appear in those cases?
5. If so, how would that affect the interpretation of the current findings?
6. Given that the Group Fourier Transform is defined directly in terms of the dihedral group’s representation theory, how can the authors ensure that the observed structure is not a byproduct of analyzing activations in a basis already aligned with that structure?
7. Can the authors provide quantitative metrics (e.g., distances between embeddings and Cayley-graph adjacency matrices, reproducibility statistics across seeds) rather than relying primarily on visual inspection?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies how deep neural networks (DNNs) perform the *dihedral* group operation (the group of symmetries of a regular polygon, which includes rotations and reflections). By visualizing the neural activation using a frequency-based remapping, the authors argue that DNNs learn neural representations that correspond precisely to Caley graphs.

### Strengths
- First comprehensive mechanistic analysis of neural networks trained on non-commutative group tasks.
- The experiments include hundreds to thousands of random seeds, showing robustness across architectures (MLP vs transformer).
- Results scale across orders of magnitude (2-512), demonstrating the claimed logarithmic feature efficiency.

### Weaknesses
- The experimental setup is incompletely described. For example, it is not stated how many layers are used for the MLPs and Transformers, how the sequences in the language are sampled for transformers, etc.
- The novelty of this work is not properly discussed in the context of existing work. Specifically, the algorithms learned by DNNs to perform group operations have been already studied by [1]. The current work focuses specifically on the non-commutative dihedral group, but this appears to be just a subcase of [1] since every group is isomorphic to a subgroup of a symmetric group (Cayley’s Theorem). It is not clear if the current work has any additional insights compared to [1].
- The algorithm learned by DNNs to perform the dihedral group operation is not explained with sufficient detail and clarity. The discussion in section 5 relies heavily on cosets and the Chinese Remainder Theorem (CRT), but these concepts are not adequately introduced.
- The "algorithm" recovered by the authors is quite abstract. The authors do not say how the networks weights are able to compute the Caley graph representations. 

1. Chughtai, Bilal, Lawrence Chan, and Neel Nanda. "A toy model of universality: Reverse engineering how networks learn group operations." International Conference on Machine Learning. PMLR, 2023. https://arxiv.org/pdf/2302.03025

### Questions
- Is the Group Fourier Transform applied layer-wise or neuron-wise?
- How are clusters defined quantitatively (e.g., thresholding on frequency similarity, correlation distance)?
- Could the same procedure be applied to arbitrary networks without access to group labels?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The work is centered around *how* feed forward and transformer neural networks learn dihedral multiplication. The authors present an argument for how the cyclic nature of dihedral multiplication presents an interesting testbed. They follow this up studying the activations and their principal components. This study reveals that the manifold of learned neural representations correspond to Cayley graphs.

### Strengths
The PCA approach to neural representation for their test bed of learning group multiplication in the dihedral group $D_n$ is novel and very interesting. I found the exposition instructive, given some background. Their results align perfectly with the hypothesis in toy setup. They position their work in light of previous work very well. They compare to literature on grokking and interpretability using analytically generated datasets. I found the result in section 5: neural networks learn $O(\log n)$ algorithm for group operation to be very interesting.

### Weaknesses
One challenge I faced while reading the work was the background. I believe the authors can present the following topics in the main body of the paper:
1. Group Fourier Transform: the authors use and re-use this idea throughout but the background is relegated to the Appendix. I believe a brief intro would benefit all readers with varying levels of familiarity with Fourier analysis.

2. An example of a coset (e.g. all with + sign or - sign) would help a reader like me.

3. I am not sure how the learning dynamics fit into the picture? Since the problem is motivated from a Grokking perspective, which has to do with learning dynamics, I would be interested in at what stage in their training do NNs Grok these divide and conquer algorithms.

------------------------------------------

Minor issues:

Line 101-103: I would use citep here to put all the references in ()

Line 432-433: broken reference "??"

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
