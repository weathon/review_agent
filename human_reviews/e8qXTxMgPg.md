# Beyond Worst-Case Dimensionality Reduction for Sparse Vectors

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
We study beyond worst-case dimensionality reduction for $s$-sparse vectors (vectors with at most $s$ non-zero coordinates). Our work is divided into two parts, each focusing on a different facet of beyond worst-case analysis:

\noindent (a)  We first consider average-case guarantees for embedding $s$-sparse vectors. Here, a well-known folklore upper bound based on the birthday-paradox states: For any collection $X$ of $s$-sparse vectors in $\mathbb{R}^d$, there exists a linear map $A: \mathbb{R}^d \rightarrow \mathbb{R}^{O(s^2)}$ which \emph{exactly} preserves the norm of $99\%$ of the vectors in $X$ in any $\ell_p$ norm (as opposed to the usual setting where guarantees hold for all vectors). We provide novel lower bounds showing that this is indeed optimal in many settings. Specifically, any oblivious linear map satisfying similar average-case guarantees must map to $\Omega(s^2)$ dimensions. The same lower bound also holds for a wider class of sufficiently smooth maps, including `encoder-decoder schemes', where we compare the norm of the original vector to that of a smooth function of the embedding. These lower bounds reveal a surprising separation result for smooth embeddings of sparse vectors, as an upper bound of $O(s \log(d))$ is possible if we instead use arbitrary functions, e.g., via compressed sensing algorithms.


 (b) Given these lower bounds, we specialize to sparse \emph{non-negative} vectors to hopes of improved upper bounds. For a dataset $X$ of non-negative $s$-sparse vectors and any $p \ge 1$, we can non-linearly embed $X$ to $O(s\log(|X|s)/\varepsilon^2)$ dimensions while preserving all pairwise distances in $\ell_p$ norm up to $1\pm \varepsilon$, with no dependence on $p$. Surprisingly, the non-negativity assumption enables much smaller embeddings than arbitrary sparse vectors, where the best known bound suffers an exponential $(\log |X|)^{O(p)}$ dependence. Our map also guarantees \emph{exact} dimensionality reduction for the $\ell_{\infty}$ norm by embedding $X$ into $O(s\log |X|)$ dimensions, which is tight. We further give separation results showing that both the non-linearity of $f$ and the non-negativity of $X$ are necessary, and provide downstream algorithmic improvements using our embedding.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper investigates dimensionality reduction techniques for sparse vectors by addressing two central research questions. It explores average-case guarantees, presenting lower bounds that show that essentially the "birthday paradox map" (simply randomly map into the lower dim space with very low chance of coordinate collision) which uses O(s^2) dimension is optimal in many relevant scenarios. In particular, any linear map with similar average-case properties must use Omega(s^2) dimensions; this is then also extended to the case where the map is potentially non-linear but smooth. The authors also propose improved dimensionality reduction for *non-negative* sparse vectors, showing that they can be embedded into fewer dimensions, i.e., O(s log (#vectors) / eps^2) than arbitrary sparse vectors. This shows that non-negativity enables more efficient embeddings, especially in high-dimensional spaces; this is also necessary for smaller embeddings as also shown by the authors. The papers also includes several interesting down-stream applications from optimization.

### Strengths
I really like the paper, although I have to admit not being an expert in the field, so I might over-/underestimate the relevance and the first 9 pages only include high-level exposition with no proofs and I did not have the time to check the proofs in details in the appendix. The results seems to be quite "complete" and "complementary" in the sense that basically for each relevant case and bound the corresponding lower bound or construction is presented. The results regarding the additional dimension-reduction in the case of non-negative vectors were somewhat surprising. 

It would be great if the paper could give a little bit more intuition why the non-negativity makes such as significant difference. In particular, the authors write that simple translation is not good enough as it might affect sparsity. Intuitively I tend to agree (as this happens to be the case in many similar scenarios). 

I think the paper, assuming everything is correct etc, can have quite a few downstream application as dimensionality reduction is at the core of many ML and optimization algorithms.

### Weaknesses
Not much to complain about - my only concern or comment is that the first 9 pages only contain high-level summaries. A little bit more detail and intuition, e.g., for the non-negativity induced improvement would have been nice.

### Questions
see above

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors focus on the problem of sparse dimensionality reduction in the average case $\ell_p$ norms, where, instead of preserving distances across a high percentage of problem instances, the goal is to preserve distances for a high percentage of pairs of vectors within each instance.

They present a bunch of results for this problem, mainly some are:

1. They show that any linear map, resembling the so called Birthday Paradox map if optimal for the case where $p$ is even. They provide lower bounds to match existing $O(s^2)$ upper bounds on the embedding dimension for $s$-sparse vectors. Their hard cases involves distributions where one first samples co ordinates uniformly at random, and then assigns mean 0 i.i.d. Gaussian value to the non zero co ordinates. 
2. They extend this lower bound for weaker guarantees for the case of $\ell_2$ norms, providing stronger results. These extend to continuous, twice differentiable maps, and compositions of such maps that can be interpreted as an encoder-decoder scheme. 
3. For the case where the entries could be non negative, they show improved upper bounds by showing the existence of a nonlinear map that achieves a reduced embedding dimension. These results are also paired with lower bounds.

### Strengths
Apart from the interesting hard cases, their results on nonlinear mappings for non negative vectors, which provide guarantees for preserving $\ell_\infty$ distances, highlight the importance of moving beyond linear i.i.d. mappings for certain problems. 

Their overall theory results are good and I recommend an accept for this venue.

### Weaknesses
I don't recommend a strong accept mainly because of lack of a cohesive insight across the two main research questions they address -- average case lower bounds for general sparse vectors and improved upper bounds for non negative sparse vectors. 

The paper's array of results could also be supplemented with more insight into the hard distributions (of the type $Unif_{t,r}$) that they constructed. They also mention that Birthday Paradox like maps are folklore, but it would provide more completeness if they referenced mentions or proof of the upper bound. Moreover, their proof techniques might not be highly novel -- for instance the authors ultimately leverage a linear structure of the nonlinear mapping through a Taylor expansion, which is fairly standard. 

As a result, while the overall contributions are good, if there were an option for giving a 7, I would give that. However, I do not consider this work sufficiently non trivial to warrant a strong accept.

### Questions
Have the authors explored separation results between non-i.i.d. maps and linear i.i.d. maps for this problem?

### Soundness
3

### Presentation
2

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
This paper studies the problem of dimensionality reduction for sparse vectors.

Suppose $x\in \mathbb{R}^d$ is an $s$-sparse vector.
We would like to map $x$ to an $m$-dimensional vector $f(x)$ such that $\lVert f(x) - x \rVert$ is preserved.
Note that this map can be a nonlinear map.
Our goal is to minimize $m$.

This paper provides two main results.
The first one is for the general setting.
The authors show that $m$ needs to be at least $s^2$ for a class of smooth maps $f$.

The second result is for vectors with non-negative entries.
Namely, we further assume that the entries of $x$ are all non-negative.
The authors construct a novel map and show that $s$ is at most $s\log n$ to preserve the pairwise distance among $n$ points.
Also, the authors prove the lower bounds in this setting.

### Strengths
The presentation of this paper is generally clear. 
Readers from different backgrounds should be able to follow the main idea of this paper.


For the second result, the map constructed by the authors seems to be novel and interesting.
Researchers who work on this area may be able to learn the insight from this construction.

### Weaknesses
Regarding the first result, it seems to be natural that one should expect a lower bound of $s^2$ because of the birthday paradox.
Indeed, the main analysis of the lower bound is mostly a probability calculation and the construction is based on uniform sampling which is not particularly surprising.
I am not quite able to see new techniques introduced in this part.

### Questions
Line 431: "... a map (randomized) ..." $\to$ "... a (randomized) map ..."?

Line 456: "... (they maybe $0$)." $\to$ "... (they may be $0$)."


Line 842: "... let $i$ be it's ..." $\to$ "... let $i$ be its ..."

Line 908: Should it be the second sum (in the expression for $Z^2$ in line 891)?

Theorem E.6: "datastructure" $\to$ "data structure"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies dimensionality reduction for sparse vectors. Suppose we have a set of s-sparse vectors X in \R^d (i.e., each vector has s non-zero entries), the goal of dimension reduction is to find a transformation f : \R^d -> \R^k, such that k is as small as possible, and || f(x) || \approx ||x|| for "most" x \in X.

Different variants of this basic question are considered:
- if the norm is \ell_p and the goal is to exactly preserve the norms of 99% of the vectors, the authors show a lower bound k = \Omega(s^2). The nice thing here is that it matches the "trivial" birthday paradox based embedding. 
- when the norm is \ell_2 and the goal is approximate preservation but to a "small enough" error factor, the authors show that the same lower bound holds.
- the above two lower bounds are for the case when the embedding function f() is assumed to be linear. The authors extend this result to more general non-linear embeddings (assuming smoothness of derivatives), and then argue that a similar lower bound holds on the embedding dimension.

On the upper bound side, the authors consider a set of n vectors, each s-sparse, with _non-negative_ entries. Here, they argue that in order to maintain pairwise distances between these vectors, k = O(s \log n) suffices (with an extra dependence on \epsilon, which is the parameter measuring the approximation error). The statement is interesting, because (a) it overcomes the upper bound shown in the earlier results, and (b) such embeddings are typically difficult for general \ell_p norms (e.g., p=1).

The lower bounds are shown via a nice use of anti-concentration properties of the Gaussian distribution. While not too surprising, it's nice that a simple construction works out. It is nice to see that the lower bound also holds for general smooth embeddings (not just linear). The upper bound is via a non-linear embedding that builds on the birthday construction but with more tricks.

### Strengths
- The paper is very well written and the authors do a thorough study of dimension reduction for sparse and non-negative sparse vectors.

- While the proof is not hard, the fact that the lower bound matches the birthday embedding is interesting. It is also interesting that the lower bounds extend to non-linear but smooth embeddings.

- The upper bound also felt interesting, but perhaps the authors can clarify things a bit more (see questions below).

### Weaknesses
- The results are nice, but it does feel like a pure theory paper, raising the question of whether ICLR is the right venue. That said, dimension reduction is ubiquitous in ML, and so I feel it is justified.

- The setting considered in the lower bounds and upper bounds are quite different -- lower bounds work only for linear and smooth embeddings while the upper bound uses a non-linear embedding. So stating the results as a "separation" between general and non-negative results does not seem convincing to me. (Of course, there is also the bigger difference that one case considers distances between the vectors, while the other case considers the vectors themselves.)

### Questions
- Can the authors say more about whether the known non-linear embeddings (eg., compressed sensing) work for the non-negative vector setting considered in the paper? (will it be that most pairs of distances will be preserved but not all?)

- Please comment on the second point in the weakness above, in case I missed something in the paper.

### Soundness
3

### Presentation
4

### Contribution
3
