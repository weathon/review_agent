# Permutation Equivariant Neural Networks for Antisymmetric Tensors

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 6, 2

## Abstract
Antisymmetric tensors, which change sign under index swaps, appear naturally in physics, yet learning from them remains largely unexplored. We provide a complete characterisation of all linear permutation equivariant functions between antisymmetric power spaces of $\mathbb{R}^{n}$. To make this characterisation practical, we introduce a memory-efficient implementation that eliminates the need to create and store large weight matrices. We demonstrate that our approach is efficient in learning functions that depend on the antisymmetric structure of the input and outperforms models that do not incorporate this structure as an inductive bias.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces permutation-equivariant models for antisymmetric tensors, i.e., for the antisymmetric powers of $\mathbb{R}^n$.

### Strengths
- I find the theoretical contribution to be **the strongest aspect of the paper**. Sections 3 and 4 are particularly enjoyable for readers interested in theory, as they provide a complete characterization of linear permutation-equivariant scalar-valued (and vector-valued, respectively) functions on antisymmetric tensors in $(\mathbb{R}^n)^{\otimes k}$ as elements of $\mathrm{Hom}{S_n}(\Lambda^k(\mathbb{R}^n), \Lambda^0(\mathbb{R}^n))$ (and $\mathrm{Hom}{S_n}(\Lambda^k(\mathbb{R}^n), \Lambda^1(\mathbb{R}^n))$, respectively). The authors also supply all necessary proofs in the Appendix. A solid background in representation theory would likely help readers fully appreciate the results.

- This theoretical foundation naturally leads to the efficient implementation of the proposed layers in Section 5, which is then applied in the experimental results presented in Section 6. While I am able to follow the high-level ideas in Section 5, I do not feel fully confident evaluating the correctness of the implementation details, as I lack a strong background in coding. However, I did notice that imposing equivariance appears to constrain many parameters in the linear layers--forcing some to be zero or equal to one another--which may substantially reduce the expressiveness of the resulting model.

### Weaknesses
I believe the empirical aspect of the paper is its weakest part. 

- Both experiments are conducted on very small synthetic datasets, which makes it difficult to draw meaningful conclusions about the practical effectiveness or generalizability of the proposed approach. 

- As mentioned earlier, the equivariance constraints drastically reduce the number of learnable parameters--down to only 5 in the first experiment and 4 in the second. Even the next-best baseline models are extremely small (9 and 8 parameters, respectively). In contrast, the unconstrained MLPs, while still modest in size (664 and 93 parameters), have significantly more freedom to learn. 

Although I understand that the symmetry constraints inherently shrink the parameter space, experiments on such tiny models and toy datasets provide limited insight. They do not convincingly demonstrate whether the proposed architectures can scale, whether they can handle realistic noise and variability.

### Questions
To strengthen the empirical section, I would encourage the authors to include at least one experiment on a real dataset, and a significantly larger synthetic dataset, together with a scaling analysis illustrating how the method behaves as model size and data complexity increase. Without such evidence, the experiments read more as a proof-of-concept demonstration rather than a convincing empirical validation.

---

My current score reflects the fact that, while I acknowledge and appreciate the impressive theoretical contribution of the paper, I believe that for a conference focused on Machine Learning, the empirical aspect also plays an important role. In its present form, the work feels more suitable for a theory-oriented venue, such as a journal, where empirical validation is typically less central to the evaluation criteria.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a classification of all linear maps from $\wedge^k(\mathbb{R}^n)$ to $\wedge^l(\mathbb{R}^n)$ which are permutation equivariant. This is relevant to deep learning since it is equivalent to providing a full classification of all linear layers that are permutation equivariant and preserve the antisymmetric structure of these spaces (something that does not naturally happen when $\wedge^k(\mathbb{R}^n)$ is interpreted as a product of standard vector spaces during implementation). The paper begins by reviewing basic properties of $\wedge^k(\mathbb{R}^n)$. Then the paper gives its classification of all linear maps in $\hom_{S_n}(\wedge^k(\mathbb{R}^n), \wedge^l(\mathbb{R}^n))$ using some simple diagrammatics and a combinatorial model which the paper calls antispherical bipartitions. Since elements of $\hom_{S_n}(\wedge^k(\mathbb{R}^n), \wedge^l(\mathbb{R}^n))$ can become very large when explicitly manifested, the paper describes an efficient way for encoding these via what the paper calls *$IO$ patterns*. Finally, as a proof-of-concept, the paper presents some small-scale numerical experiments that show that models using the layers described in the paper perform better than naïve methods.

### Strengths
**Clarity:** This is a well-written paper, that reads like a math paper (in a good way). Notation is clearly laid out so that no guessing is required on the part of the reviewer. The paper reviews the exterior product and establishes its basic properties before launching into the analysis of $\hom_{S_n}(\wedge^k(\mathbb{R}^n), \wedge^l(\mathbb{R}^n))$. Despite this being in some ways a fairly abstract paper, the writing style makes it easier to read than many more concrete papers that deal with simpler constructions.

**This paper contains elegant mathematics that tells a nice story:** The reviewer appreciated the use of combinatorial and diagrammatic constructions in the paper’s classification of elements of $\hom_{S_n}(\wedge^k(\mathbb{R}^n), \wedge^l(\mathbb{R}^n))$. This style has a long history within the pure math community and to some degree physics, but one rarely sees it in machine learning. In this paper, the approach really brought the results to life and made them much more tangible than would be the case if the analysis was performed in a more brute force style (e.g., complicated descriptions of matrices). As a result of the way the mathematics is developed, a lot of the theorems appear evident, even though they are non-trivial.

### Weaknesses
**What is already known and what is new?:** Exterior powers are a classical construction and as such, much is known about their structure, including their Hom spaces. It was unclear to the reviewer what exactly was new here and what is already known and simply being applied to the problem of layers in neural networks. It was at least stated that the antispherical bipartitions are a new construction. Clarifying these points would help the reader better understand how the work fits into the research landscape.  

**Further experimental analysis:** The reviewer appreciates that the main point of this work is not experimental and that realistic data with antisymmetric structure (e.g., data coming from physics or differential geometry) is likely to be unhelpful to the general reader. However, there are a range of interesting experiments that could be run with the synthetic data that is already used in the paper. For example, the reviewer would have been interested to see what happened when $n$, $k$, or $l$ are varied. When does learning break down?

**More information on the types of applications where this is relevant:** One limitation of this type of work is that most of the ICLR community probably doesn’t work with data that has an antisymmetric structure. One idea for engaging such a reader would be to describe how this structure arises in an application (preferably the most accessible one available). This would help motivate and ground the rest of the work. The mix of combinatorics and algebra that this work taps into is very beautiful so making it accessible to the broader community would be a benefit to everyone.

**The topic is somewhat niche:** While exterior products and morphisms between them are certainly a fundamental part of mathematics, it is unclear how much the machine learning community needs networks that can accommodate them. The work seems to suggest that so far, few papers actually tackle this problem. It was unclear to the reviewer if this was because there simply isn’t a demand in applications or if there is a demand and practitioners are simply using unsuitable architectures.

**Nitpicks:**
- Line 025: I believe that the history of equivariant networks goes back quite a bit further than this. It is this reviewer’s understanding that this was a central feature in the development of convolutional neural networks for instance.
- Line 157: The reviewer liked the use of this colored box. But it would be useful to put a title on it for later reference.

### Questions
- Networks typically contain a few different types of layers, how do layers like non-linearity, norms, etc. interact with this type of equivariance?
- What sizes of $k$, $l$ can we realistically expect to compute with?

### Soundness
4

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
2

### Summary
This paper characterizes all linear permutation-equivariant maps between antisymmetric power spaces $\Lambda^{k}(\mathbb{R}^{n}) \to \Lambda^{\ell}(\mathbb{R}^{n})$ by giving an explicit basis for $\mathrm{Hom}{S_n} (\Lambda^{k} (\mathbb{R}^{n}), \Lambda^{\ell}(\mathbb{R}^{n}))$.
It provides a memory-efficient implementation that applies these maps directly to antisymmetric tensors without forming large unrolled matrices.
Experiments on $S_{8}$-equivariant and $S_{4}$-equivariant tasks show consistent gains over MLPs and standard permutation-equivariant baselines.

### Strengths
- The construction of linear permutation-equivariant maps for antisymmetric tensors is presented clearly and rigorously.
- The empirical results demonstrate consistent gains over MLP and permutation-equivariant baselines, achieving lower test MSE across data sizes while using fewer parameters.

### Weaknesses
The main weakness concerns experimental scope. Both the $S_4$-invariant and $S_8$-equivariant tasks are synthetic, so evidence for real-world applicability remains limited. It would be more convincing to evaluate the proposed antisymmetric permutation-equivariant maps on real datasets.

### Questions
Overall I think it is a good contribution to theoretically study the equivariant map for antisymmetric tensor. My main question is how the proposed maps translate to practice. Specifically, how could these maps be applied to relevant tasks? For example, in molecular/fermionic modeling or physics simulations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a framework to construct permutation equivariant networks for antisymmetric tensors. By considering a canonical form where indices appear in lexicographical order, the authors are able to enumerate the basis of the transformation space, hence characterizing the equivariant transformations. Proof-of-concept experiments are presented in synthetic antisymmetric datasets to validate the constructions presented.

### Strengths
The authors go to great lengths to provide illustrations and explain their algorithms and the generating procedures. The constructions are intuitive and the overall structure of the paper is clean and logical.

### Weaknesses
The paper is definitely a difficult read and requires very careful and thorough pace. I found a lot of definitions taken for granted, which I think hurts exposition, especially for a general ML audience.

More on the general audience, I think it would have been extremely helpful to visualize (or give specific examples) of the vector spaces that are being considered. Concretely, (14) is a 3x3 matrix. How this matrix acts on elements of $\Lambda^2(\mathcal{R}^3)$ (to yield an element of $\Lambda^1(\mathcal{R}^3)$) is not trivial and concrete explanations about the practical implementation of the method would help presentation.

Finally, a lot of the motivation is coming from examples in the sciences, however the experiments don't have any experiments on real data, and the synthetic experiments are on very small models.

### Questions
1. How are zero length tuples defined? They are used in Proposition 3.4.
2. The proof of Proposition 4.1 is very difficult to read. As an example, $\\{E_{I, J}\\}_{I\in \Lambda[n]^l, J\in\Lambda[n]^k}$ is defined as having 1 in the $(I, J)$ position. However, these are tuples, so how is a position for a tuple of tuples defined? Standard ideas from linear vector spaces don't seem to suffice, and the background is missing.
3. More of a comment than a question, but the notation of $(k, l)$-antispherical bipartition obfuscates $n$, which seems problematic. It is not possible to deduce the vector space.
4. The algorithm above Proposition 4.9 is hard to parse, it's unclear why it is needed or what the intuition behind it is, and is combinatorial in order. Specifically for step 3, it is unclear what the order of a spider is.
5. There is no proof for Proposition 4.9, which seems like a central result of the paper.
6. In Appendix C an algorithm is presented, followed by the statement "Clearly, this generates all possible partitions". That is not obvious, and that statement is used in the sequel to count the number of bipartitions. A proof using induction (or any other proof tool) is necessary for statements like this.
7. The main text claims that the nonlinearity needs to be odd in order to preserve the antisymmetric nature, however ReLU is used for the experiments, which is clearly not odd. Moreover on the experiments, how is the generalization evaluated on differently sized tensors? Is $n$ the only thing that is changing, while the length of the tuples are kept the same? If so, how are the different indices casted to matrices?
8. Why is there no test accuracy reported for the first experiment, and why are the models trained for only 20 epochs, with such small model sizes?
9. Why are the running times not reported for the second experiment? In general, a complexity analysis for the method seems to be missing.

### Soundness
3

### Presentation
2

### Contribution
2
