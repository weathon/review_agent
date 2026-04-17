# It's All Just Vectorization: einx, a Universal Notation for Tensor Operations

- Decision: Accept (Oral)
- Scores: 4, 8, 6, 6

## Abstract
Tensor operations represent a cornerstone of modern scientific computing. However, the Numpy-like notation adopted by predominant tensor frameworks is often difficult to read and write and prone to so-called shape errors, i.a., due to following inconsistent rules across a large, complex collection of operations. Alternatives like einsum and einops have gained popularity, but are inherently restricted to few operations and lack the generality required for a universal model of tensor programming.

To derive a better paradigm, we revisit vectorization as a function for transforming tensor operations, and use it to both lift lower-order operations to higher-order operations, and conceptually decompose higher-order operations to lower-order operations and their vectorization.

Building on the universal nature of vectorization, we introduce einx, a universal notation for tensor operations. It uses declarative, pointful expressions that are defined by analogy with loop notation and represent the vectorization of tensor operations. The notation reduces the large APIs of existing frameworks to a small set of elementary operations, applies consistent rules across all operations, and enables a clean, readable and writable representation in code. We provide an implementation of einx that is embedded in Python and integrates seamlessly with existing tensor frameworks: https://github.com/fferflo/einx

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces einx, a notation and library for expressing tensor operations as vectorizations of elementary functions. The authors define a consistent syntax based on loop notation and implement it in Python for NumPy, PyTorch, and JAX backends. The goal is to replace the heterogeneous and sometimes confusing APIs of current tensor frameworks with a small, uniform set of composable operations. The paper is clearly written and technically sound.

### Strengths
- The paper is well organized and easy to follow.
- The notation is clearly defined and internally consistent.
- The implementation is described in sufficient technical detail and appears complete.
- The examples and tables illustrate coverage across many tensor operations.
- The case study on multi-head attention demonstrates syntactic conciseness.

### Weaknesses
- The contribution is about notation and implementation, not about new ML algorithms or empirical insights.
- The main problem addressed (API inconsistency) is largely syntactic and not shown to have a measurable impact on ML research or practice.
- No experiments, user studies, or adoption data support the claim that einx improves model development or reduces errors.
- The approach generalizes existing ideas from einsum and einops rather than introducing a new paradigm.
- The motivation section frames the issue as readability and conceptual elegance, but provides no evidence that current tools are a significant bottleneck.
- The comparison to other libraries is descriptive only, without performance or usability analysis.

### Questions
- Are there examples where einx allows expressing a model or computation that cannot be easily written with existing tools?
- Can you provide any data, even informal, on user adoption or code simplification in real ML workflows?
- How does einx handle contraction path optimization? Can it automatically determine and apply efficient evaluation orders for chained tensor products (e.g., einx.dot("m [a], [a b], [b] -> m", M, A, v)), or does it always execute operations in the explicit order implied by the expression?
- einx accepts sparse tensors for simple elementwise operations (e.g., scalar multiplication) but fails for contractions, additions, or reshapes because the PyTorch backend invokes dense operations (einsum, reshape, add(sparse, dense)). Could the authors clarify whether sparse interoperability is within the design scope of einx? Are there plans to support dispatching contractions to sparse-aware kernels (e.g., torch.sparse.mm), or is einx currently intended primarily for dense tensors?
- How much runtime overhead does the einx compilation and caching mechanism add compared to equivalent direct calls in PyTorch or NumPy, and are there cases where it limits backend optimizations (e.g., for dynamic shapes or JIT execution)?

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
4

### Summary
This paper proposed a new extended grammar for describing tensor operations that generalizes those common in `einsum` and `einops`. The authors cast vectorization as a general transformation that lifts lower-order operations to higher-order operations with a pointful (instead of pointless) style that is more declarative. It can naturally subsume all kinds of indexing (gather, scatter, index_select); tensor contraction (multiply, inner, outer, kronecker product, etc.); and all kinds of reshaping operations (broadcast, repeat, squeeze, etc.). Additionally, the authors also included a `(a+b)` notation that covers stack and concat. 

The case study with multi-head attention is pretty illuminating. This library would be beneficial to all machine learning and scientific computation practitioners.

### Strengths
- A good generalization of `einsum` and `einops` that would be beneficial to all ML practitioners.
- Paper is clearly written, with lots of examples to showcase the semantics the proposed grammar.

### Weaknesses
- The semantics of `[x]` is not clear enough: at sometimes it is for axes to be contracted; sometimes it can be used in a `get_at` expression whose semantics is a bit vague.

### Questions
- In the abstract, please cite `einsum` and `einops`.
- L165 Named tensors: Missing citations to include:
  - T Chen (2017): Typesafe abstractions for tensor operations. https://dl.acm.org/doi/10.1145/3136000.3136001
  - D Chiang, A M Rush, B Borak (2021): Named Tensor Notation. https://arxiv.org/abs/2102.13196
- L309: The semantics of `[ ]` is vague: it seems to indicate matched axes for tensor contraction, but sometimes it has other uses: please clarify

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes einx, a “universal” notation for tensor operations that extends einsum. It also claims other features, including declarative and Interpretable. The paper compares einx with einops using multi-head attention as a case study.

### Strengths
+ Brings many NumPy-like operations under a unified function signature.
+ The declarative nature of the notation is advantageous; for instance, Line 374 demonstrates how the necessary permutation is inferred and performed implicitly.

### Weaknesses
- Section 3 discusses tensor ops and vectorization, but lacks formal definitions of lower-order vs higher-order vectorization; this makes the argument hard to follow
- The core conclusion in Section 3 —“It’s all just vectorization”—reflects a well-known aspect of tensor computation (same computation applied across slices). The paper does not clearly present how this finding leads to the specific design choices in the einx signature.
- It’s unclear, in notation and capability, how the proposed {vectorization}/bracket syntax differs in practice from PyTorch einsum (or its common extensions).
- Table 1 categorizes Numpy-like notations into 4 groups. It might be interesting to discuss the intuition of such a classification.

### Questions
1. How does the finding about vectorization lead to the design of einx?
2. What problems do brackets solve that einsum cannot, and how often do those cases arise in practice?
For example, this notation seems unnecessary in many cases in the paper.
Line 358, einx.sum("a [b]", x) seems equivalent to torch.einsum(“a b -> a”, x).
Line 308, einx.dot("a [b], [b] c -> a c", x, y)  seems equivalent to torch.einsum("a b, b c", x, y)
Line 472, einx.softmax("[k]", A) seems equivalent to torch.softmax(a, axis=1). 
3. What are the key differences between einx and PyTorch einsum (or other einsum extensions) in terms of notation design? Which operations can einx express that others cannot, beyond the softmax example above?
4. What is the scope? Is it really universal?
a. Convolution operators are important; why are they not covered or discussed in this paper? There are works extending einsum to convolution operations [1][2].
b. How does einx represent pixel shuffle?
c. How does einx represent jacobi-2D? 
5. Table 1 classifies Numpy-like operations into four categories. What is the intuition for such classification? 
6. How can the proposed classification (Table 1) be extended to cover custom operations? Can it serve as a framework or guideline for defining new operations within einx?
7. Section 5.2 presents multi-head attention as an example; however, similar high-level fused attention layers already exist. Please include additional examples where einx clearly shortens code, improves readability, or demonstrates other unique advantages.

[1] Dangel, Felix. "Convolutions and More as Einsum: A Tensor Network Perspective with Advances for Second-Order Methods." Advances in Neural Information Processing Systems 37 (2024): 96671-96727.

[2] Rabbani, Tahseen, et al. "conv_einsum: A Framework for Representation and Fast Evaluation of Multilinear Operations in Convolutional Tensorial Neural Networks." arXiv preprint arXiv:2401.03384 (2024).

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new notation for tensor operations called einx. Previous attempts, such as einsum and its variants, have been proposed and widely used in the machine learning community. However, these notations have limitations in expressive power by definition, and their semantics are not always straightforward to understand. In this paper, the authors propose einx, a new notation focusing on vectorization. In the proposed notation, an einx expression can be derived by rewriting an implementation originally written using for-loops. The proposed notation enables concise descriptions of a wide range of tensor operations.

### Strengths
The strengths of this paper are as follows:

## Convenience of a unified notation
As the authors claim, the proposed notation enables a unified expression of various tensor operations. In particular, the ability to represent concatenated axes, such as in `einx.id`, is interesting. Specifically, in the expression of attention in Sec 5.2. 5.2, the use of `einx` indeed allows the attention mechanism to be written in a small number of lines.

## Beginner-friendly for elementary examples
In the elementary procedure of this method, one first considers a standard loop-based formulation and then converts it into `einx` notation by arranging the corresponding ids. This step-by-step approach seems intuitive and accessible even to beginners.

## Introduction and related work
I appreciate Sections 1 and 2 of this paper. The authors provide a summary of the history of past `ein*` notations, which should serve as valuable material for future researchers considering similar notational systems.

### Weaknesses
On the other hand, the weaknesses of this paper are as follows:

##  Mismatch with the venue due to a lack of quantitative evaluation
While the proposed notation is interesting, there is no quantitative evaluation demonstrating its actual effectiveness. Therefore, it isn't easy to judge whether the proposed method is scientifically sound as a research contribution. Since quantitative evaluation is not necessarily required for notation proposals, this work would be more suitable for venues such as MLOSS or the ACMMM Open Source Competition rather than the ICLR main conference.

## Still complex for complex examples
Although the proposed notation is easy to understand in elementary examples, it remains complex for complicated cases. For instance, an expression like `z = einx.multiply("a b c, b -> c b a", x, y)` is understandable by reasoning from the corresponding loops. Still, the attention example requires considerable familiarity to interpret. Therefore, the effectiveness of this notation (i.e., whether it truly makes complex operations easier to understand) depends heavily on the user's level of expertise with the notation.

### Questions
In Sec. 3.22, the phrase "The sum-reduction operation `np.sum(x, axis=1)` ... is decomposable" appears, but this wording may be somewhat misleading and might benefit from rephrasing. When I first read it, I interpreted it to mean that the `sum` computation can be "decomposed" into two operations, namely `y[i] = sum(x[i, :])` and `y[i] += x[i, j]`. However, that is not the intended meaning here; instead, it means that the sum operation can be expressed in two distinct ways, correct?

This is more of a comment than a question, but for complex notations (such as `einx.dot("b q (h [c]), b k (h [c]) -> ...` in Sec. 5.2), I believe understanding would be deepened by a visualization that explains them visually. For example, a simple HTML page that can be opened in a browser, where entering an `einx` expression produces a real-time visualization.

### Soundness
3

### Presentation
3

### Contribution
3
