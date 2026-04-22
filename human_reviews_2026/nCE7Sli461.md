# Why High-rank Neural Networks Generalize?: An Algebraic Framework with RKHSs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 4

## Abstract
We derive a new Rademacher complexity bound for deep neural networks using Koopman operators, group representations, and reproducing kernel Hilbert spaces (RKHSs).
The proposed bound describes why the models with high-rank weight matrices generalize well.
Although there are existing bounds that attempt to describe this phenomenon, these existing bounds can be applied to limited types of models.
We introduce an algebraic representation of neural networks and a kernel function to construct an RKHS to derive a bound
for a wider range of realistic models.
This work paves the way for the Koopman-based theory for Rademacher complexity bounds to be valid for more practical situations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a Rademacher complexity bound for deep NNs derived using Koopman operators, group representations and RKHSs. Distinct from other bounds, the bound includes the determinant of the weight matrices in its denominator, and is therefore tighter than comparable bounds in the case of high-rank weight-matrices.

### Strengths
1. The paper builds on previous work on Rademacher complexity bounds via Koopman operators, removing restrictions which limited the applicability of the previously derived bounds.
2. The resulting bounds scale much better than comparable bounds when the weight matrices are high-rank.
3. The experimental results appear to back up the claims make in the paper.

### Weaknesses
My main problem with this paper is readability. While I understand that the material is intrinsically difficult, it would nevertheless be helpful if more time was spent systematically introducing the various notations/definitions etc that are used. In particular, I found section 5 to be quite heavy going, and section 5.3 in particular was extremely difficult to parse. Perhaps the authors could include a table (even if it is located in the appendices?) presenting the various notations/definitions (and their inter-connections) in a more systematic way so that the reader isn't left constantly scanning back and forward through paragraphs of dense definitions/notations interleaved with prose to remind themselves what something means when it is eventually used?

More generally it seems to me that this is a somewhat incremental improvement on the previous work in Hashimoto et al, but I am open to argument in this regard as this is open to interpretation.

Specific points and questions:

- Lemma 2.5: it seems to me that this result effectively rules out applying the method to ReLU networks as it blows up in the relevant limit. Can you see any way around this?
- Example 3.3: is the limit in line 205(ish) due to $p_{c,x}(y)$ approaching the Dirac-delta as $c \to \infty$?
- Theorem 4.3: is the supremum here actually bounded? The determinant cannot be zero for the invertible case, but it can be arbitrarily small, so the supremum can presumably be arbitrarily large. Also I assume $A_l$ here is the Koopman operator?
- Theorem 5.1: it is confusing to use $\alpha (f_l)$ here before introducing it (in remark 5.3). Also what role does this factor play? Could it just be replaced by $1$ for clarity in the theorem?
- Section 5.3: the first paragraph here is far to mathematically dense and needs to be expanded for clarity. Also it strikes me that many of these are essentially geometric constructs, so perhaps a diagram could help?

### Questions
See weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new Koopman-operator-based Rademacher complexity bound to explain why neural networks with high-rank weight matrices can generalize well—an observation not captured by existing norm-based or compression-based generalization theories.

The key idea is to construct a reproducing kernel Hilbert space (RKHS) on the parameter space of neural networks using group representations and Koopman operators, enabling a generalization bound applicable to realistic models with bounded input domains and nonsmooth activations (e.g., tanh, sigmoid, Leaky ReLU).

### Strengths
Addresses a known theoretical gap:
Most generalization bounds (norm-, compression-, or PAC-Bayes–based) explain why low-rank networks generalize, but fail for empirically observed high-rank networks. This work provides a principled explanation via Koopman-based operator theory.

Mathematical generality:
The approach unifies algebraic, operator-theoretic, and kernel methods, supporting both bounded and nonsmooth activations—something prior Koopman-based bounds (Hashimoto et al., 2024) could not handle.

Elegant use of RKHS:
By defining an RKHS on the parameter space, the authors convert the non-linear deep network mapping into a linear setting where standard Rademacher complexity tools apply — a conceptually clean and general construction.

Practical implications validated by experiments:
The three plots in Figure 1 show consistent improvement in generalization across synthetic regression, MNIST dense nets, and LeNet CNNs when regularization terms derived from the bound are added.

Connections to high-level mathematical structures:
The use of group representations (affine, Heisenberg) and operator algebras (Schur’s Lemma, von Neumann double commutant theorem) to describe neural networks is novel and could inspire deeper theoretical analyses of neural architectures.

### Weaknesses
Dependence on bounded activation assumptions:
The proofs assume Koopman operator boundedness (Assumption 2.2), which fails for ReLU (derivative = 0 on half-line).
The authors acknowledge this limitation (Sec. 7) and suggest exploring weighted Koopman operators, but current results exclude the most popular activations.


Interpretability of constants:
The bounds depend on abstract constants that are not easy to estimate in practice. Consequently, the bound is more qualitative than quantitative.

Comparative evaluation:
While it compares against the 2024 Koopman bound, other modern generalization frameworks (e.g., PAC-Bayes, spectral norms, neural tangent kernel analyses) are not benchmarked.

No discussion of sample complexity scaling:
The 1/sqrt(S) dependence mirrors classical Rademacher bounds, but no new scaling exponents are derived.

### Questions
1) Can your framework extend to ReLU networks via non-smooth or distributional Koopman operators?

2) How sensitive is the determinant-based bound to weight scaling (e.g., if |det (W)| >> 1 due to scaling, not diversity)?

3) In practical networks, rank deficiency often correlates with overfitting. Does your theory predict the same phenomenon in the inverse direction?

4) How does the bound behave for randomly initialized networks versus trained ones?

5) Could the RKHS kernel you define (Eq. 3.2) be computed empirically for small models to verify the bound numerically?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper develops a new Rademacher complexity bound for deep neural networks by combining Koopman operator theory, group representations, and RKHS methods. Unlike traditional norm- or rank-based bounds, the proposed framework explains why high-rank networks can still generalize well. The authors construct a kernel on the parameter space and prove an isometric correspondence between this Koopman-RKHS and the function space of the network. Overall, the work provides a mathematically elegant perspective on neural network generalization, though some aspects of proofs raise some concerns about their validity.

### Strengths
The paper offers a novel and mathematically rigorous framework that unifies Koopman operator theory, RKHS analysis, and group representations for studying neural network generalization. It extends existing Koopman-based approaches to handle nonsmooth activations and bounded domains, broadening their practical relevance.

### Weaknesses
The main weakness is a couple of issues that I found in proofs.

### Questions
Example 3.3 introduces $p_{c,x}$ which is  Gaussian centered at $x$ whose variance goes to zero, as $c$ goes to infinity. 

This function satisfies 
$||p_{c,x}||_{1}=1$ 

but not 

$$||p_{c,x}||_{2}=1$$ 

as claimed in paper. 
The latter claim, i.e. 
$$||p_{c,x}||_{2}=1$$ 
is then used in the proof of Theorem 4.3 where the dependence on $c$ vanishes (which I doubt is correct).

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new bound on the Rademacher complexity of deep neural networks, with the goal of better accounting for the observed good generalization of neural networks with high-rank weight matrices.  The paper improves upon earlier work by Hashimoto et al. 2024 that introduced an approach based on Koopman operators and RKHSs to represent and reason about neural networks.  The main improvement is the analysis of models in a different RKHS than in prior work, thus accounting for a larger and more practical model class, for example including non-smooth activation functions.  A small-scale experiment on MNIST digit classification shows that a regularizer derived from the bound improves on one derived from the earlier bound.

### Strengths
- Expanding the reach of existing complexity bounds to a more realistic model class is a very important contribution.

- The style of Koopman-based bounds appears to differ from much of the other existing work studying generalization in neural networks (besides the cited Hashimoto et al. 2024), and so may be complementary to other results.

- The inclusion of empirical evidence, while limited, is helpful.

### Weaknesses
- The main weakness is with the presentation.  I fear that in its current form, the paper will be accessible only to an very narrow segment of the ICLR audience.  While I am admittedly not an expert in the recent theoretical ML literature, I believe the paper should be able to make its results and implications much clearer to most of the ICLR community.  By the end of the paper, it is not clear to me in what way exactly the paper improves over Hashimoto et al. 2024, and how this improvement was achieved.  It is possible that parts of my summary above are incorrect as a result (please let me know if so!).  I would have expected a clear statement of the Hashimoto et al. 2024 bound and the new bound, both mathematically and intuitively, and ideally also other existing bounds for comparison.  I also would have expected a more intuitive "walk-through" of the key steps.  There are two paragraphs devoted to introducing RKHSs -- a broadly familiar concept, I believe -- while there are many other more obscure concepts and notation that are relied on but not explained.  While they cannot all be defined for the novice reader, some intuitive explanations of the ideas would go a long way.

- A second possible and more minor weakness (which may be not a weakness at all, and rather due to my own misunderstanding -- again, please let me know if so!) is with the empirical results.  I am not able to understand why the bound was not compared directly to the prior bound (e.g. in Fig. 1(a)) but only as a form of regularization (in Fig. 1(b, c)).  I also wonder why there are no empirical results with more practical, larger datasets and models (say, ImageNet).  It is reasonable for a theoretical paper to limit itself to small experimental settings, if needed.  But it should be stated what are the limitations that prevent scaling up the experiments.

### Questions
My remaining questions are mainly about low-level details:

- Regarding these sentences:  "... phenomena in which models with high-rank weight matrices generalize well have been empirically observed (Goldblum et al., 2020).  Since the norm-based and compression-based bounds focus only on the low-rank and nearly low-rank cases, they cannot describe these phenomena."  The second sentence need not be true if the high-rank matrices in Goldblum et al. 2020 are in fact nearly low-rank.  I guess that is not the case, but could you please confirm?

- There is some imprecise language that would be good to clean up.  Examples:
  - "It describes how the model can fit unseen data" -- not clear what "It" refers to.
  - "This bound is described by the ratio of the norm to the determinant of the weight matrix" -- since there are multiple layers and multiple weight matrices, this can't be precise.
  - "the generalization bound is described by the Rademacher complexity" -- unclear what it means for the Rademacher complexity to "describe" a bound.
  - "Our framework fills the gap between the Koopman-based analysis of generalization bounds and practical situations" -- I think this is a bit stronger than intended (i.e. the paper does not fully fill the gap).

- Typo:  "The Hilbert space H to which the modes belong..." --> "... the models belong"

- In Fig. 1(a), I have a hard time seeing the color gradient.  Perhaps a different gradient would be visually clearer.

### Soundness
3

### Presentation
2

### Contribution
3
