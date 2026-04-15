# How to Capture Higher-order Correlations? Generalizing Matrix Softmax Attention to Kronecker Computation

- Decision: Accept (spotlight)
- Scores: 8, 8, 8

## Abstract
In the classical transformer attention scheme, we are given three $n \times d$ size matrices $Q, K, V$ (the query, key, and value tokens), and the goal is to compute a new $n \times d$ size matrix $D^{-1} \exp(QK^\top) V$ where $D = \mathrm{diag}( \exp(QK^\top) {\bf 1}_n )$. Here, $\exp()$ is applied entry-wise and ${\bf 1}_n$ denotes a length-$n$ vector whose entries are all ones.

Intuitively, attention computation captures pairwise information between words in a sentence, but not higher-order information. Indeed, recent work \cite{sht23} has shown that attention units cannot solve simple problems about detecting triples of connected words.

In this work, we study a generalization of attention which captures triple-wise  correlations. The generalization is based on computations involving tensors defined by tuples of words. More formally, given five $n \times d$ size matrices $Q, K_1, K_2, V_1$ and $V_2$ (generalized query, key, and value tokens), our new goal is to compute an $n \times d$ size matrix $D^{-1} \exp( Q ( K_1 \oslash K_2)^\top ) (V_1 \oslash V_2) $ where $D = \mathrm{diag}( \exp( Q ( K_1 \oslash K_2)^\top ) {\bf 1}_{n^2} )$ and $K_1 \oslash K_2 \in \mathbb{R}^{n^2 \times d}$ denotes the column-wise Kronecker product of $K_1$ and $K_2$. This generalization is indeed able to solve problems about detecting triple-wise connections that were shown to be impossible for transformers.

The potential downside of this generalization is that it appears as though computations are even more difficult, since the straightforward algorithm requires cubic time in $n$. However, we show that in the bounded-entry setting (which arises in practice, and which is well-studied in both theory and practice), there is actually a near-linear time algorithm. More precisely, we show that bounded entries are both necessary and sufficient for quickly performing generalized computations:

$\bullet$ On the positive side, if all entries of the input matrices are bounded above by $o(\sqrt[3]{\log n})$ then we show how to approximate the ``tensor-type'' attention matrix in $n^{1+o(1)}$ time.

$\bullet$ On the negative side, we show that if the entries of the input matrices may be as large as $\Omega(\sqrt[3]{\log n})$, then there is no algorithm that runs faster than $n^{3-o(1)}$ (assuming the Strong Exponential 
Time Hypothesis from fine-grained complexity theory).


We also show that our construction, algorithms, and lower bounds naturally generalize to higher-order tensors and correlations. Interestingly, the higher the order of the tensors, the lower the bound on the entries needs to be for an efficient algorithm. Our results thus yield a natural tradeoff between the boundedness of the entries, and order of the tensor one may use for more expressive, efficient attention computation.

Our constructions make use of a novel connection with a higher-order variant on the kernel density estimation problem. They combine a number of technical tools, including the polynomial method, algebraic geometry codes, and multiparty Merlin-Arthur communication protocols.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper solves the transformer attention scheme over triplets under some mild complexity theoretical assumptions.

### Strengths
Originality:
This work is original to the best of my knowledge.

Quality:
Quality is high.

Clarity:
Writing is clear.

Significance:
The results of this paper are important due to their connections to LLMs and other AI applications.

### Weaknesses
None.

### Questions
Page 2:
Could you please explain the column-wise Kronecker product of V_1 and V_2 here as well?

Page 3:
Please elaborate on SETH.
I know it is standard, but perhaps too strong? :)

Page 4:
Please provide some intuition for Definition 1.5.

Page 5:
I do not understand "Approximating A."

Page 6:
Could you please sketch an short example for the reduction from GapMaxIP to ATAttc?

Page 8:
Could you please add some discussion about Theorem 4.7?

Page 9:
Line 12 from bottom:
Why is there such a \mu?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the computational aspects of a so-called "third-order tensor attention variant" recently defined by Sanford et al (2023). While standard attention captures pairwise interactions between tokens, the third-order tensor variant was suggested in the context of capturing triplet interactions. 

The current paper shows that if the input entries are bounded, and if it suffices to compute each entry of the attention output approximately rather than exactly, then there is an almost-linear time algorithm for computing this operation. The bound on the entry magnitude is asymptotically (conditionally) tight, as the paper also shows that without this bound, computing this attention output in time significantly better than the trivial n^3 time computation is SETH-hard, in the complexity theoretic sense. Both the algorithm and the hardness result are generalizations (in terms of both the results themselves, and the techniques used to prove them) of a recent work of Alman and Song (2023) that proved them for standard attention. The results also extend to tensors of higher orders (than 3).

### Strengths
The paper is generally well-written and the mathematical content seems interesting.

### Weaknesses
The paper is purely theoretical and seems quite removed from application. It is entirely about a form of tensor attention that has been suggested as a bit of an afterthought in a recent work (Sanford 2023), that deemed it likely impractical and anyway did not implement it. Thus it is not about an architecture that is actually used or presently considered usable. This raises the question of what is the import of showing the existence of an almost-linear time approximation algorithm for it, and whether this algorithm makes sense as part of a neural network (I believe the paper does not touch on this point). A negative outlook on this paper would be that the interesting results and the conceptual message in this line of research were already given in Alman and Song (2023), and the extension to higher-order tensors in this manuscript might be an elegant intellectual and mathematical exercise, albeit without much consequence to the ML community. Nonetheless, since the content does seem elegant, and there is always the issue of benefit-of-doubt about whether a piece of theoretical research would have implications down the road, I prefer to vote for acceptance.

### Questions
I'd be interested to hear from the authors what do they consider to be the importance and implication of their algorithm, and whether they deem it implementable within a neural network architecture?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work explores the computational complexity of generalized matrix attention scheme, which is used to capture high-order correlation among the input sequence. 
To capture the correlation among triplets of tokens in an input sequence of length $n$, the generalized attention scheme outputs an attention matrix by computing the column-wise Kronecker products based on one query matrix, two key matrices and two value matrices. 
In such a case, this work shows if the max entries of the query, key and value matrices are bounded by $o(\sqrt[3]{\log n})$, one can compute an approximation to the generalized attention matrix in almost linear time. On the other hand, if the max entries of the input matrices are at least $\Omega(\sqrt[3]{\log n}$), one cannot hope to efficiently approximate  the attention matrix in less than cubic time, assuming SETH. The latter hardness result is shown by reducing from the Gap Max IP problem, whose hardness is then shown through a combination and generalization of previous techniques.  Furthermore, the work shows the techniques developed above can be extended to characterize the gap in computational complexity in $k$-th order generalization of the attention scheme.

### Strengths
- This work considers an interesting problem of computing generalized matrix attention scheme. Since the generalized schemes involves computing the Kronecker products between a set of matrices, this is apparently a computationally expensive operation. It is hence natural to explore the computational complexity of this problem.

- Both the upper bound and the lower bound results (especially the latter one) presented in the work are interesting. 

- A high-level summarization and intuition behind the techniques used to derive the upper and the lower bound helps the reader.

### Weaknesses
The presentation needs to be improved. 

- In Section 3.2 hardness, “3-Max IP” and “Gap-MaxIP” seem to refer to the same problem. It is confusing to give two names to the same problem.

- It might be clearer to present the upper bound (UB) and the lower bound (LB) in two separate sections, instead of giving an overview of the UB + LB, and then elaborate on the LB.

- Is it possible to give a few sentences of description of the mysterious $U_1, U_2, U_3$ matrices and how to compute them in Section 3.1?

- In Section 4 “hardness” which elaborates on key steps in showing the LB, presenting Hypothesis 4.1, Definition 4.2, Conjecture 4.3 and Theorem 4.4, all of which are from prior works, in the main paper do not help much on understanding and appreciating the novelty / challenges addressed in extending and generalizing current proof techniques to show the LB on computational complexity for approximating the generalized attention matrix. Some of them can indeed be moved to the Appendix. It would be better to give more intuition on the (technical) difference between the three-party and four-party communication protocol that computes set disjointness, how algebraic geometry code is applied in extending the proof from three-party to four-party communication and how the new protocol is used in showing the LB on computation time of Gap Max-IP.

- Minor issue: the last paragraph in page 5 states “showing that the same line of attack”? “attack” here means “techniques”, I assume?

### Questions
I am not a complexity expert. I have no comments on the proof techniques presented in this work. However, I do have a few questions for the authors.

- In Definition 1.2, why is approximating the higher order attention matrix in the $\ell_{\infty}$ norm considered a good metric to evaluate the approximation of a matrix that captures higher order correlation? Why not the other norms?

- In Section 3.1, does “$\widetilde{D} \approx D$” mean $\tilde{D}$ and $D$ close in the $\ell_{\infty}$ norm? (and so does “$\widetilde{A} \approx A$”?)

- In Section 3.1, why can $\widetilde{A}(V_1 / V_2)$ be computed in $O(n^{1+o(1)})$ time, while $\widetilde{D}$ needs to be computed in $O(nd)$ time?

- Why does the construction of the algorithm in Section 3.1 fail when there are large entries $\Omega(\sqrt[3]{\log n})$ in the input matrices?

- In Section 4, what is the major challenge of extending the three-party communication protocol to a four-party communication protocol in Section 4.2? Why does one need to use the algebraic geometry code?

- In Section 4, where does $B = O(\sqrt[3]{\log n})$ pop up in the LB proof?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
