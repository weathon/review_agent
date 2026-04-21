# Black-Box Approximation and Optimization with Hierarchical Tucker Decomposition

- Avg Score: 4.25
- Decision: Reject
- Scores: 5, 6, 3, 3

## Abstract
We develop a new method HTBB for the multidimensional black-box approximation and gradient-free optimization, which is based on the low-rank hierarchical Tucker decomposition with the use of the MaxVol indices selection procedure. Numerical experiments for 14 complex model problems demonstrate the robustness of the proposed method for dimensions up to 1000, while it shows significantly more accurate results than classical gradient-free optimization methods, as well as approximation and optimization methods based on the popular tensor train decomposition, which represents a simpler case of a tensor network.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper is concerned with the approximation and maximization (or minimization) of a real-valued function defined on a subset of $\mathbb{R}^d$. The function is considered as being a "black box", meaning its properties are unknown, but one can evaluate it over a predefined discretization grid, implicitly defining a tensor (of point function evaluations). Specifically, the authors propose extensions of the TT-Cross and TTOpt algorithms, which rely on the so-called tensor train (TT) tensor format, to a more general hierarchical Tucker (HT) format that is specified by a tree topology which indicates how the cores (factors) involved in the low-rank approximation are connected. The main technical difficulty seems to be how to navigate that tree for selecting a subset of indices used for building the approximation.

### Strengths
S1) The paper generalizes tensor-tain-based function approximation algorithms to more general hierarchical Tucker structures.

S2) Its simulation results convincingly illustrate the benefits of the proposed approach in terms of function approximation and maximization performance, presumably thanks to the higher degree of flexibility of HT in comparison with TT.

### Weaknesses
W1) Despite the proposed algorithms being potentially useful for practitioners, in my view the contribution of this paper is fairly incremental, as it extends some well-known algorithms to a different tensor format, but without introducing new ideas, insights or relevant theoretical results.

W2) The paper is fairly hard to read for non-experts, which I believe will be the case for the majority of the ICLR audience. In that regard:
- A brief description of the rationale behind the TT-Cross and TTOpt should be given, so that the readers understand what the authors are trying to implement. This will make the paper minimally self-contained, since the extension of these algorithms is the main goal here. 
- Related do that point, is not quite clear what is the output of Algorithm 1. It seems to be the cores that will define the approximation, but at the same time this algorithm is expected to yield the max (or min) of the function, as announced in the paper.

W3) The paper should also clearly state exactly where the novelty is in terms of the introduced algorithms. 

W4) A careful revision is needed for improving English usage. Examples:
- "possible maximum rank increment" -> maximum possible rank increment
- "then we go to the random side" -> then we randomly choose one side to go to.
- "see Figure 3 for example of path" -> for an example of a path
- "For the core of the root node it is hold"  (unclear what is meant)

### Questions
Q1) I believe the paper would be considerably improved by taking into accound the point W1 stated above. In particular, I suggest the following:
- Briefly recall the rationale and the mechanics of TT-Cross and TTOpt.
- Write Algorithm 1 in a more explicit fashion, clearly stating its inputs and outputs, using a more formal notation.

Q2) In Section 3, the authors state that by HT they "mean a tensor tree that is not necessarily balanced". Thus, the topology of the HT structure used in the numerical experiments should be specified. Is it always taken to be simply a balanced tree? 

Q3) Related to the previous question, how is this topology chosen in practice? Is it simply by trial-and-error or are there some heuristics for doing so? 

Q4) Intuitively, the higher flexibility of the proposed scheme in comparison with the TT-based methods should lead to smaller approximation errors, provided that the HT structure is correctly chosen. Hence:
- Is it possible to derive any quantitative comparison between the TT and HT approaches?
- More generally, can you give any approximation guarantees of your method for some reasonable classes of functions, say as function of the number of point evaluations? 

Q5) Regarding the point W3 raised above, is the novelty here all in the way where HTBB selects the next node that should be visited, the rest of it being similar to TT-Cross and TTOpt?

Q6) Some details and relevant discussion are lacking in the numerical experiments. Namely, what are the TT ranks used in the experiments of sections 6.1 and 6.2? What are the total number of parameters in both models? If the TT ranks are increased (but the HT ones are held fixed), is the TT model able to gradually bridge the performance gap, and if so, at which additional cost?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose a low-rank hierarchical Tucker cross-approximation method, utilizing the MaxVol index selection procedure. The paper is well-structured, with a comprehensive introduction. In my assessment, this work constitutes a notable contribution to the field.

### Strengths
The authors propose a low-rank hierarchical Tucker cross-approximation method, utilizing the MaxVol index selection procedure. The paper is well-structured, with a comprehensive introduction. In my assessment, this work constitutes a notable contribution to the field.

### Weaknesses
Please refer to the questions below.

### Questions
1. The introduction of the equation in Section 3 appears abrupt. It is recommended that the authors include a figure to provide further illustration and clarity.

2. What would be the performance of Algorithm 2 if QR decomposition were replaced with singular value decomposition (SVD)?

3. Given the existence of a canonical form for the hierarchical Tucker decomposition, could the authors provide a comparison between the cross-approximation method and an SVD-based method in the context of hierarchical Tucker?

4. In the first experiment, in addition to the TT cross-approximation, the Tucker cross-approximation should be considered, as hierarchical Tucker is an extension of the Tucker decomposition.

5. The experimental results should include performance comparison with regard to different tensor orders, as this would provide insight into the scalability of the proposed method.

6. All experimental results should be normalized for comparability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes a method called Hierarchiacl Tucker Black Box (HTBB) based
on low-rank hierarchical Tucker decomposition and the MaxVol subroutine for
finding good submatrices [Goreinov et al., Matrix Methods: Theory, Algorithms
and Applications 2010]. The approach is inspired by the algorithms:
* TT-cross [Oseledets-Eugene, Linear Algebra and Its Applications, 2010]
* TTOpt [Sozykin et al., NeurIPS 2022]

with the goal of extending these works
to hierarchical Tucker tree structures. The main techincal contribution of
this work is Algorithm 2, which decomposes dimensions of the tensor to
construct a good low-rank hierarchical Tucker tree structure. Finally, this
work compares HTBB with several other algorithms across a benchmark of
high-dimensional tensors representing discretized continuous functions.

### Strengths
- The main technical details in Section 4 are sophisticated and well organized.
  That said, they could benefit from more context and a better setup in the
  preceeding sections.
- Experiments are run against well defined benchmarks tensors from previous
  work, e.g., [Jamil-Yang, J. of Math. Modeling and Num. Opt. 2013]. These
  tensors are discretized continuous functions with known ground truth.
- Table 3 and Figure 4 in the experiments suggest that HTBB has a lot of
  potential, e.g., there are clear improvements in the $10^4$ number-of-requests
  regime.

### Weaknesses
- First and foremost, the presentation quality of this work makes it
  challenging to (1) have a precise understanding of what problem is being
  solved, and (2) the difference between your approach and other existing works.
  The good ideas in this work are getting lost, largely because there is not a
  crisp and standalone mathematical problem definition (relies too much on
  pointing to related work).
- Related works cover many different method types (e.g., particle swarm
  optimization).  How many of these solve the same problem and could be
  compared against in the experiments?

### Questions
- For the 14 benchmark instances, [Jamil-Yang, J. of Math. Modeling and Num. Opt. 2013] has 100+
  tensors in their dataset. How did you choose this subset of 14 across multiple works?
- "gradient-free" methods are mentioned regularly throughout the paper. Can you give an example
  of an effective method that you compare against that is gradient-based?
- Are the sizes (i.e., the number of optimizable parameters) in HTBB and TT-cross equivalent in Table 1?
- In Figure 4, why are the SPSA and OPO methods better up until $10^3$ requests?
- Note: The [Goreinov et al., 2010] citation should include the journal.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper develops new black-box approximation and gradient-free optimization methods for multidimensional functions based on Hierarchical Tucker (HT) decomposition. Numerical experiments were conducted to illustrate the improvement over the prior method that was developed based on Tensor Train (TT) decomposition.

### Strengths
- The proposed algorithm achieved better numerical performances than the prior works.
- The authors did a good job explaining the details of the proposed algorithm despite the complicated notations, which are common in tensor-related work.

### Weaknesses
- A key concern is the lack of theoretical guarantees. Compared to the existing algorithms like TT-Cross and TTOpt, there is no clear improvement in terms of time complexity, approximation guarantees, or optimization guarantees with respect to the dimension $d$ and rank $r$. The algorithm seems too heuristic to me and the contribution is not enough for an ICLR publication.
- In addition, I have reservations about the practicality of the proposed "black-box" approach. In the numerical experiments, all functions have well-behaved analytical expressions, which may not reflect real-world scenarios. It remains unclear whether the approach would perform well in applications where functions may lack favorable properties, such as continuity or convexity.
- The presentation of the paper, particularly in terms of English language usage, could be improved.

### Questions
- typo on page 5, line 47: $i^{down}_{l,j}$
- typo on page 6, line 323: "As each visited node"

### Soundness
3

### Presentation
2

### Contribution
2
