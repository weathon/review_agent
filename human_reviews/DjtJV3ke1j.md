# Dynamic Kernel Sparsifiers

- Decision: Reject
- Scores: 6, 3, 6, 3

## Abstract
A geometric graph  associated with a set of points $P= \{x_1, x_2, \cdots, x_n \} \subset \mathbb{R}^d$ and a fixed kernel function $\mathsf{K}:\mathbb{R}^d\times \mathbb{R}^d\to\mathbb{R}_{\geq 0}$ is a complete graph on $P$ such that the weight of edge $(x_i, x_j)$ is $\mathsf{K}(x_i, x_j)$. We present a fully-dynamic data structure that maintains a spectral sparsifier of a geometric graph under updates that change the locations of points in $P$ one at a time. The update time of our data structure is $n^{o(1)}$ with high probability, and the initialization time is $n^{1+o(1)}$. Under certain assumption, our data structure can be made robust against adaptive adversaries, which makes our sparsifier applicable in iterative optimization algorithms. 

We further show that the Laplacian matrices corresponding to geometric graphs admit a randomized sketch for maintaining  matrix-vector multiplication and projection in $n^{o(1)}$ time, under \emph{sparse} updates to the query vectors, or under modification of points in $P$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a dynamic data structure that maintains a spectral sparsifier for a geometric graph, allowing for efficient updates when point locations change. This data structure is initialized in $n^{1+o(1)}$ time and can handle point location changes in $n^{o(1)}$ time with high probability. The key component of the data structure is a dynamic well-separated pair decomposition (WSPD), which efficiently partitions the graph into subgraphs with similar edge weights. Leveraging JL projections and a smooth resampling technique, the data structure maintains low-dimensional sketches for efficient updates and queries. Additionally, the data structure can be made robust against adaptive adversaries. The paper also present a randomized sketch for maintaining matrix-vector multiplication and projection in $n^{o(1)}$ time under sparse updates to query vectors or point modifications.

### Strengths
**S1)** The studied problem is interesting and relevant. Constructing spectral sparsifiers in the dynamic setting is an active area of research. For geometric graphs, this is extra challenging, because any change in data point directly induces $\theta(n)$ edge updates. A lot of previous work on dynamic sparsifiers focuses on single edge updates (Abraham et al., 2016) - this work circumvents that.

**S2)** The overall results are quite strong. The main technical contribution of this work is to carefuly work out how to (i) dynamically maintain the WSPD; and (ii)  maintain uniform random samples of the bicliques induced by the WS pairs. The work notes that (i) can be achieved with the algorithm of Har-Peled (2011). To achieve (ii), a careful resampling is designed for the bicliques using biased coins. Designing the full procedures is clearly tedious work, and therefore I consider it a significant contribution that future work could benefit from. The results on the adversarial guarantees and randomized sketch for matrix-vec multiplication seem to follow from the main result, and they are a nice addition to the paper.

### Weaknesses
**W1)** The current manuscipt write-up is subpar. The first 9 pages are difficult to read, and I believe the writeup could be significantly improved.

**a.)** First, the references are poorly formatted. Most references are missing their brackets. This makes the paper difficult to read at points. Probably this is an easy fix using \citet instead of \cite.

**b.)** Section 4 contains a lot of dense technical content, that is difficult to follow. Given that the contribution of this work is very technical, I understand that it's not possible to fully describe the result in the main body, and that only high-level ideas and sketches can be given. However the current writeup makes it difficult to grasp the ideas in the paper. For example, sections 4.1.1 and 4.1.2 are dense with a lot of technical discussion. There are a lot of references to the appendix (Lemma A.22, Definition A.15, Figure 2) that require the reader to go back and forth a lot. Replacing some of the long descriptions with figures (like fig 1 and fig 2) would add a lot of clarity. It is not expected that all the results can be accurately verified within the first 9 pages, however, these pages should convice the reader/reviewer that the result is solid. At the moment, for me, that is difficult to do.

**c.)** It might be helpful to convert some of the text into algorithm descriptions (e.g., lines 285-292, lines 317-320, lines 329-333), as these lines describe pseudocode.

**d.)** A lot of typos and mistakes in formulation. See the Minor points/typos section. 

**W2)** Some of the assumptions in this work are quite strong, such as fixed dimensionality and being limited to $(C, L)$-Lipschitz functions (although constructing static sparsifiers for arbitrary kernel functions is hard, see Alman et al. (2020)). In particular the assumption on the aspect ratio of the data might be too strong; under adversarial updates, this assumption is broken easily.

**W3** This is a fairly minor weakness, but no experimental results are reported. Adding experiments would improve the significance and impact of this work.

**Minor points/typos**
- On line 029-031 the references look wrong - I think there should be brackets around "Alaoui (...) Lee et al. (2020)". There are multiple other places where this happens, making the text hard to read at points. Probably a \citep vs \citet issue. 
- Line 056: "Extendin" --> "Extending"
- Line 207: "Before presenting our dynamic data structure, we first have a high level (..)" --> "(..) we first give a high level (..)"?
- Line 218 and 220: "A s-WSPD" --> "An s-WSPD"
- Line 283: The sentence "When a point location update (...)" ends abruptly. Should there be a comma instead of a period?
- Line 392: "To make sure this" --> "To ensure this"?
- Lines 391: "From another direction, we need to make that" --> "From another direction, we need to make sure that"?
- Line 950: "Algortihm" --> "Algorithm"
- Line 432: What is DynamicGeoSpar? I believe this is only introduced in the appendix.

### Questions
**Q1)** Is any adjustment to the algorithm of Har-Peled required to obtain your results? Or can it be applied as a black-box. 

**Q2)** How tight are the results? Could any of the update times be improved?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies the following problem: given points $x_1, \ldots, x_n \in \mathbb{R}^d$ and a kernel function $K : \mathbb{R}^d \times \mathbb{R}$, define a complete graph with $x_1, ..., x_n$ as nodes and the weight of the edge between $x_i, x_j$ is defined to $K(x_i, x_j)$. The goal of this paper is to maintain a dynamic spectral sparsifier of the complete graph where dynamic refers to the abilitiy of handling vertex movements quickly. In particular, if a given vertex $x_i$ is moved to a new location $z \in \mathbb{R}^d$, we want the spectral sparsifier with respect to the new graph to be computed quickly. Note that moving a single vertex affects $n-1$ edge weights (assuming $K(x, x) = c$ for all $x$) and therefore $O(n)$ entries of the Laplacian matrix are changed by a single vertex movement. The goal of the paper is to compute a spectral sparsifier that can be updated in $n^{o(1)}$ time for each vertex update.

The high-level algorithm is as follows:
1. Project the points in $\mathbb{R}^d$ into $\mathbb{R}^k$ using ultra-low dimensional JL embedding which preserves relative distances up to a reasonable multiplicative factor.
2. Construct a well separated pair decomposition using quadtrees.
3. Use the approxiamtion guarantees of the JL transform to argue that the WSPD computed on the projected points is also a WSPD.
4. For each tuple, in the WSPD, approximate the laplacian by uniform sampling, which is equivalent to leverage score sampling up to multiplicative factors and hence gives a spectral sparsifier when sampling appropriate number of edges uniformly at random.
5. To make the algorithm support vertex movements dynamically, the authors show that there exists a WSPD where each vertex is part of only a small number of tuples and that when a vertex is moved only the edges in the affected tuples need to be resampled. To show that this can be done quickly, they argue that the resampling can be done keeping a large number of sampled edges from the original graph and modifying only a few random edges.

The above leads to a fast dynamic algorithm to update the spectral sparsifier. Then the authors give algorithms to robustify the randomness of JL transform to sequential updates by computing a net and using a union bound over the net vectors. Rounding off the vectors to their closest net vector would then mean that the randomness of the JL transform is robust to sequential updates.

### Strengths
The problem is interesting to study and a reasonably fast algorithm with support for a broader range of kernel functions may have practical utility.

### Weaknesses
I have reviewed an earlier version of the paper. I am slightly paraphrasing my earlier review below.

1. The randomness of uniform sampling in a pair of WSPD is correlated across multiple rounds of vertex movements. Then how is the algorithm adversarially robust? This wasn't considered at all in the paper.
2. The motivation of not being able to directly update $L_H u$ dynamically is stated many times in the paper and I don't think it is adequate. What is the model here? Do we know the vector $u$ at the beginning or is it only revealed later? If it is revealed later, do we need to support answering queries for multiple adaptively chose vectors $u$? In that case how are the sketches robust?
3. Not much significant contributions necessary to convert the static version of Alman et al., into the dynamic version.

### Questions
See the weakness section.

### Soundness
1

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
This paper introduces a dynamic data structure designed to efficiently maintain spectral sparsifiers for geometric graphs constructed on a set of points $P = {x_1, \ldots, x_n} \subset \mathbb{R}^d$ with edge weights determined by a kernel $K: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}_{\geq 0}$. The authors propose a randomized dynamic algorithm that, when $K$ satisfies a multiplicative Lipschitz condition, updates an almost linear spectral sparsifier with high probability in time $O(n^{o(1)})$ (with an initialization time of $O(n^{1+o(1)})$).

Furthermore, the authors show that with additional constraints on $P$, specifically involving its "aspect ratio," the algorithm can be made resilient to adversarial modifications. The paper also includes algorithms for efficiently maintaining approximate matrix-vector queries for the graph’s Laplacian and its generalized inverse, achieving a time complexity of $O(n^{o(1)})$ for these operations as well.

### Strengths
1- **Originality and mathematical rigor**: The authors provide technically rigorous arguments, integrating ideas from data structures, spectral sparsification, resampling, and sketching. For example, they employ the well-separated pair decomposition (WSPD) method from Har-Peled (2011) to construct and maintain sparse graph representations, leveraging this decomposition to create a spectral sparsifier through efficient sampling. Additionally, they utilize Johnson-Lindenstrauss projections to maintain a lower-dimensional representation, which is essential for preserving the algorithm’s efficiency.

2- **Clarity and presentation**: The paper is generally well-written and effectively organized. The main body focuses on clearly defining the problem, presenting the key results (in informal versions for accessibility), and outlining the principal proof strategies and techniques. The detailed technical proofs and methodological developments are provided in the appendix, which is largely self-contained and complements the main text well.

### Weaknesses
**Limitations and their assessment**:    One limitation is that the method’s performance may degrade in high-dimensional settings. Although the authors acknowledge this in the limitations section, noting that their arguments are optimized for fixed dimensions, it would be helpful to have explicit statements on the relationship between $d$ and $n$ for each main result. For the adversarial setting, this is more clear due to the condition $\alpha^d=O(\text{poly}(n))$, suggesting that these results might deteriorate more quickly as the dimension increases. Could the authors confirm this interpretation?

The $(C,L)$-Lipschitz assumption is also presented as a limitation. To better understand applicability, it would be useful to clarify which types of kernels satisfy this assumption, with examples. For instance, the classic kernel $K(x,y)=\mathbb{1}_{|x-y|_2 \leq \delta}$ for some $\delta \geq 0$ does not appear to satisfy this condition, in general. Similarly, a kernel like $K(x,y)=e^{-\frac{1}{\sigma} |x-y|_2^2}$ with large $\sigma$ might encounter similar issues. Providing insights into the impact of this assumption for such common kernels would be valuable, as the role of the Lipschitz condition remains somewhat unclear (see additional comments below).

### Questions
1- In the main results, the role of the constant $C$ in the $(C,L)$-Lipschitz condition is unclear. Specifically, it appears that any function could satisfy this condition by setting $C=1$, with a corresponding choice of $L=1$, for example. Am I overlooking a situation where $C$ significantly affects the analysis?

2- In line 055, could you define geometric kernel? 

3- typo: in line 056 in the word extending.

4- Line 076: an entire row update of $K$...is an entire row and the corresponding column, right?

5- Similar to point 1: in line 183 of the related work you mention several examples of kernels, it would be interesting to mention the applicability of your results to those situations.

6- In line 963: shouldn't be $w_ {f(G)}$ instead of $w_ {G'}$?

7- Line 1053: why is $\log(1/\delta)>2$.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Given a set of $n$ points and a kernel function $k$, we can consider an all-pairs weighted graph where the weight of every pair is given by the corresponding kernel value. The paper studies algorithms for the laplacian matrix for this graph. The main result that I mostly focused on is assuming the kernel is (C,L)-multiplicative Lipschitz, the paper presents an algorithm for constructing a spectral sparsifier which can be dynamically updated. Algorithms for constructing this graph in near linear time was studied already in prior works (e.g. Alman et al 20), and the new contribution is have fast updates when the points are dynamically updated. The authors show that for a small number of updates (see more below), we can have update time $n^{o(1)}$.

The main idea seems to be the following: one can modify the classic JL lemma to project onto slightly smaller than $O(\log n)$ dimensions (the paper takes it to be $o(\log n)$ dimensions). This can cause the distances to be distorted by $n^{o(1)}$ factors. However, in such small dimensions, we can afford to build dynamic data structures that require exponential in dimension space/query time (where exponential in dimension would be $n^{o(1)}$ due to the projection dimension). These underlying algorithms use sampling for spectral sparsification, and so we can get rid of the distortion issue by simply oversampling by the distortion factor. 

Extensions to other settings such as adversarial queries are also presented, if the dimension of the point set is not too large.

### Strengths
The technique of using an `ultra low' dimensional embedding seems to be novel to me. It is nice that in this setting, one can be resilient against the large distortion factors cause by very small embeddings by simply over sampling.

### Weaknesses
Unfortunately, the writing leaves a lot to be desired. In particular, I found many parts of the paper to be very vague, including the statement of the 'rigorous' version of the main theorem. Let's start there at Theorem D.3 which is the 'main theorem'. 

- it is stated there that $k = o(log n)$. Can I take $k$ to be just $1$ or a constant? In other parts of the paper, $k$ is set to be $O(\sqrt{n})$. What should I set $k$ to? Why not just set it to be a fixed value in the theorem statement or can I take any value?
- There is no dependence of $C$ in the parameters of the theorem statement, even though it is a parameter of the kernel. Is $C$ thought to be a constant?
- A major issue is that everywhere the statement 'with high probability' is used, including Theorem D.3. Typically one thinks of this to be failure probability $1/\text{poly}(n)$. Indeed, this is what is stated in the 'informal' version in the main body. However, the dependence on the failure probability seems to be much worse in the theorem. The theorem relies on prior lemmas (such as Lemmas D.10, D.11), which for failure probability $\delta$, have dependence $1/\delta$. This means that if we naively set $\delta = 1/\text{poly}(n)$, the running time for every update would be polynomial. This would mean one can just use Alman et al to rebuild the data structure every time. Thus, the data structure seems to only handle very few updates, which makes it less theoretically interesting. Besides this, it maybe a bit misleading that failure probability $1/\text{poly}(n)$ is stated in the main body, but it does not seem to be the case in reality when examining Theorem D.3. 
- The paper never gives an explicit example of a kernel that is allowed by their main theorem. The best description I can find is on line 97: the kernel must be (C,L)-multiplicative Lipschitz. What are some examples of such kernels? Must the condition 1/c^L <= f(cx)/f(x) <= C^L hold for all x? It seems like such a definition is only relevant for polynomially decaying kernels and not more popular kernels such as Guassians etc unless the diameter of the point set is bounded. This is fine, but it is never explicitly explained in the text.
- I don't think empirical evaluations are strictly necessary, but they could benefit the paper since the theoretical results are not so strong (due to the poor dependence on the failure probability). 

Minor:
- The paper seems to be missing discussions on related works such as https://openreview.net/pdf?id=74A-FDAyiL. There another spectral sparsification algorithm and lower bounds are discussed.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
