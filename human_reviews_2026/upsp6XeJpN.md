# Testing Fourier Sparsity via Implicit Sensing

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8, 8

## Abstract
Boolean functions constitute a fundamental object of study in machine learning and theoretical computer science. Among their various complexity measures, Fourier sparsity, the number of nonzero coefficients in a function’s Fourier expansion, serves as a natural indicator of structural simplicity. For more than three decades, the problem of learning Boolean functions with sparse Fourier representations has occupied a central place in computational learning theory. A major line of progress has produced algorithms whose complexities depend primarily on the sparsity parameter itself. However, these methods typically assume that this parameter is known in advance.
In this work, we explore the problem of Fourier sparsity testing, which naturally relates to this question. Given query access to a Boolean function $f : \mathbb{F}_2^n \to \{ -1, +1 \}$, we seek to determine whether it is $s$-Fourier sparse or far (under Hamming distance) from every such function.

Our contributions are twofold. On the algorithmic side, we design a new tester with query complexity $\widetilde{O}(s^4)$, independent of the ambient dimension. On the lower bound side, we prove that any tester requires at least $\Omega(s)$ queries. Both bounds improve upon the best known results of Gopalan et al. (SICOMP 2011), who obtained a tester with query complexity $\widetilde{O}(s^{14})$ and a lower bound of $\Omega(\sqrt{s})$. For the upper bound, we introduce a refined notion of a sampler inspired by the junta testing framework and combine it with $\ell_1$-minimization-based compressed sensing techniques. In doing so, we develop a novel method for sampling leaves of parity decision trees associated with Fourier-sparse Boolean functions. The lower bound is obtained via a reduction from communication complexity, leveraging structural properties of Fourier coefficients of a specific class of cryptographically hard functions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the property testing problem of Fourier sparsity under Hamming distance. The goal is to decide, given query access to $f :\mathbb{F}^n_2 \rightarrow \{ \pm1 \}$, whether $f$ is $s$-Fourier-sparse or $\epsilon$-far from every such function, with query complexity independent of $n$. The authors give a non-adaptive tester with complexity $\tilde{O}(s^2 \cdot \max(s^2, 1/\epsilon))$, improving the prior bound of $\tilde{O}(s^{14})$. In addition, this paper shows an $\Omega(s)$ lower bound , improving the prior $\Omega(\sqrt{s})$. 

For the upper bound, the tester combines a "implicit sensing" sampler (built in the spirit of junta testing) with an $\ell_1$-minimization step over a low-dimensional projection of $f$. The sampler produces uniform coset samples corresponding to leaves of an appropriate parity decision tree; from these, the analysis identifies the span of heavy Fourier characters (up to a linear change of variables), and reduces the task to recovering a sparse spectrum in dimension $O(\sqrt{s})$. This is done via compressed sensing on random Walsh–Hadamard rows. For the lower bound, the authors reduce testing to a linear-algebraic communication problem instantiated with Maiorana–McFarland functions.

### Strengths
- The quantitative improvement is significant: although the new upper bound $\tilde{O}(s^4)$ is still a large polynomial, comparing with the previous bound of $O(s^{14})$, cutting the exponent from 14 to 4 indeed materially advances the state of the art.

- The recovery step leverages robust, well-understood RIP machinery for subsampled Hadamard matrices. I appreciate that the paper plugs into a mature toolbox, which increases credibility and portability. The pipeline of (1) isolating a low-dimensional structure via coset sampling, then (2) running a standard compressed-sensing recovery in that reduced space feels neat and intuitive. Conceptually, the RIP lens clarifies why the query complexity loses all dependence on $n$: after the dimension to $O(\sqrt{n})$, the complexity of compressed sensing scales with the sparsity and the log of the reduced dimension, hence becomes independent of the ambient $n$. This aligns with the property-testing goal and makes the result feel principled rather than ad hoc.

### Weaknesses
I think the presentation of the paper could be significantly improved. The paper is understandably notation heavy, but this is exacerbated by the fact that a major part of both the algorithm and the upper-bound proof is deferred to the appendix, which makes cross-referencing difficult. It seems the authors intend to use Algorithm 1 as an "oracle" interface for Algorithm 3 to aid exposition; however, Algorithms 1 and 3 have quite different specifications (inputs/outputs, etc.), which limits the usefulness of this design. I suspect this formatting is driven by ICLR’s tight page limits. Rather than splitting essential descriptions and proofs between the main body and the appendix, it might be better to submit to a TCS venue with a more permissive page limit so the proof can be presented in a self-contained manner. I have not checked all proofs in detail, but even at a semantic level there appear to be some inconsistencies:

1. Lemma 4.2 gives only an upper bound on the support size, whereas the proof of Lemma 4.4 refers to it giving an exact equivalence (i.e., both upper and lower bounds).

2. In Algorithm 3, Step 6 constructs b from the signs $Q[x][B_j]$, thus $b$ seems to correspond to $Ux$, where $U$’s rows are the (unknown) $\alpha$'s. However, on Lines 918-933 it is asserted that $Mx = b$ with $M$’s rows being the standard unit vectors. I am not sure how these two statements are reconciled?

3. In the proof of Lemma A.5, when bounding the expectation $E[|\mathcal{P}_C(z)  - \mathcal{P}_C^*(z) |]$, why is it legitimate to drop the absolute value inside the expectation?

### Questions
See the above section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies property testing of the Fourier sparsity of a Boolean function. Specifically, for a parameter s, we want to perform few queries and distinguish between the cases where the Fourier sparsity is at most s, or the function is eps-far in Hamming distance fomr any function of Fourier sparsity at most s. This is a standard setup in property testing, and the problem here is particularly important: many learning/testing/etc algorithms in the literature assume one knows the Fourier sparsity of the function, so to apply these to functions where the Fourier sparsity is unknown and only query access is available, this type of algorithm is needed.

### Strengths
- This is a very natural problem in computational learning theory. The setup and sdetting are very clean and general.
- The paper improves the state-of-the-art for both upper and lower bounds. It's nice that the new insights here are able to make progress in both directions.
- The paper is written well and easy to follow for the most part.

### Weaknesses
The main weakness is just that the results seem somewhat incremental. I also think the intro/abstract overstate how important the results are; see questions below.

### Questions
- It seems misleading to say there's a "gap in the literature" (in the abstract) or a "critical gap" (page 2) about assuming you know the Fourier sparsity when Gopalan et al. already studied this question, and here the bounds are being improved but not completely new. Could you comment on this; is there some aspect that you address that was missing in the prior work? 
- I was expecting the problem to test whether the Fourier sparsity is at most s or at least s + g for some parameter g, rather than using Hamming distance of f to an s-Fourier-sparse function. Wouldn't this more accurately test for whether the function can be used in learning algorithms that need small Fourier sparsity?
- Could you highlight which of the algorithmic techniques are most novel/intereting in this context?
- In applications to learning (i.e., other learning algorithms where it is assumed you know the Fourier sparsity), I would imagine that a powering approach might work where you guess the Fourier sparsity is 1, then 2, then 4, then 8, and so on, until the algorithm succeeds, and only lose a poly factor. Are there any important applications where this type of approach doesn't work, and testing the Fourier sparsity like this is important?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a tester for determining whether a given Boolean function is $s$-Fourier sparse or $\epsilon$-far from every $s$-Fourier sparse function, using only $O(s^4)$ queries (ignoring polylogarithmic factors) under the Hamming distance. A Boolean function is $s$-Fourier sparse if it has at most $s$ nonzero Fourier coefficients. Furthermore, the paper complements this result by providing an $\Omega(s)$ lower bound on the query complexity of the problem. The authors claim that both these results improve upon the previously best-known bounds of $O(s^{14})$ and $\Omega(\sqrt{s})$, respectively.

Unfortunately, due to time constraints and limited expertise in this specific subarea, I am unable to verify the technical details.

### Strengths
The results appear to significantly advance the state-of-the-art bounds for the problem (although I have some strong reservations about this claim; see Weakness). The techniques also seem novel and potentially interesting.

### Weaknesses
Theoretically, the upper bound result is very interesting given the claimed improvement over existing bounds. However, I am not convinced that the improvement is correctly compared to the existing results. This is partly because the paper does a poor job at comparing their work with prior works.
For instance, the paper refers to the work of Yaroslavtsev–Zhou (SOSA’20), who study the same problem under the $\ell_2^2$ distance instead of the Hamming ($\ell_1$) distance considered here, and propose a tester using only $O(s)$ queries. However, it seems to me that their result can be extended to the $\ell_1$ setting with only constant-factor loss in accuracy. Indeed, for any two Boolean functions $f, g$, we have
$||f-g||_2^2 = E_x[(f(x)-g(x))^2] = 4 \cdot Pr_x[f(x)\neq g(x)] = 4 \cdot ||f-g||_1$

where $||f - g||_1$ denotes the Hamming distance between $f$ and $g$. Let $F_s$ be the set of s-Fourier sparse functions.

Therefore, the learner of Yaroslavtsev–Zhou, which estimates $||f - F_s||_2^2$ with additive error $\delta$ using $O(s)$ queries, can be used to distinguish whether the Hamming distance of $f$ to the set $F_s$, of $s$-Fourier sparse functions is at most $\delta$ or at least $4\epsilon - \delta$,  ($\delta$ set to $\epsilon / 1000$ to create a suitable gap). This formulation exactly matches the testing problem studied in the present paper. Moreover, the prior tester achieves this with only $O(s)$ queries, whereas the current paper’s tester requires $O(s^4)$ queries (ignoring polylogarithmic factors). Thus, unless I am missing a critical technical distinction, I do not see how the upper bound in this paper improves upon existing results.

### Questions
see above (weakness).

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies testing the sparsity of Fourier coefficients of boolean functions under Hamming distance. The authors proves an upper bound of $\tilde{O}(s^{4})$ and a lower bound of $\Omega{s}$, significantly improving previous bounds.

### Strengths
1. Testing Fourier coefficients of boolean functions is a fundamental problem in theretical compter science. The authors significantly improve existing upper and lower bounds on testing sparsity under Hamming distance.
2. The techniques are interesting. It is inspired by prior works on testing junta functions, but requires non-trivial and novel design in the algorithms to work for testing sparsity in Fourier coefficients.

### Weaknesses
1. A recent work [1] obtain nearly tight bound for testing sparsity of Fourier coeficients under L2/Euclidean distance. While the distance metric is different, it is necessary to discuss differences in the technical ideas. The authors mention that closesness in Hamming distance implies closeness in Euclidean distance, and therefore it is more difficult to design algorithm for the former case, but this also means that lower bound for Euclidean distance could imply a lower bound for Hamming distance.

[1] Arijit Ghosh, Manmatha Roy, Price of Parsimony: Complexity of Fourier Sparsity Testing, to appear in NeurIPS 2025, https://openreview.net/forum?id=7bCPXHq8xV

### Questions
My main questions are related to the recent work [1]
1. What are the key distinctions with the algorithm used in [1]? 
2. Does the lower bound in [1] imply a meaningfule lower bound for the Hamming distance?

[1] Arijit Ghosh, Manmatha Roy, Price of Parsimony: Complexity of Fourier Sparsity Testing, to appear in NeurIPS 2025, https://openreview.net/forum?id=7bCPXHq8xV

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents new algorithms for testing Fourier sparsity of a boolean function given query access to that function. A boolean function is $s$-Fourier sparse if it's Fourier representation has at most $s$ nonzero components. This assumption is satisfied by classes of simple functions such as low-depth decision trees. Fourier-sparse family of functions that has received extensive theoretical investigation because of connections to a wide range of combinatorial and algorithmic questions. 

The main results improve the known upper and lower bounds for the query complexity of testing Fourier-sparseness: the upper bound drops from about $s^{10}$ to about $s^4$. The lower bound jumps from about $\sqrt{s}$ to about $s$.

The paper achieves these results by refining existing approaches and developing new ones based on the structure of Fourier-sparse functions.

### Strengths
* This paper provides significant improvements on a clean algorithmic problem. 

* The techniques seem like a nice contribution to the literature. I did not check the proofs, but the overall writing and presentation seem like a good fit for a specialist audience. It was nice to see many recent insights from the complexity/learning literature applied to this problem (which hadn't been directly attacked since 2011 or so).

* Applications of this result are not discussed in detail, but the authors point out that testers for Fourier-sparseness could, in principle, be used to help select the parameters of a learning algorithm that is to be run on a dataset. 

* Section 3.1 does a nice issue of explaining the basic outline of the algorithm.

### Weaknesses
* My main hesitations about the paper have to do with significance and fit for ICLR. Specifically: 

   * Exact vs tolerant testing: the authors suggest in the abstract that their algorithm could be used to select the parameter $s$ for a downstream learning algorithm that learns s-Fourier sparse functions. However, such an application would presumably require a tolerant tester, and would only be useful when the additional powers of $s$ in the testing results are dominated by the dimension-dependent factors required for learning. It would be good, for an ICLR audience, see this path spelled out—or at least a clear-eyed case be made for why this line of theoretical work should interest the ICLR audience. 

   * Hamming versus Euclidean distance (similar to the tolerant testing issue above): The paper explains why testing for Hamming distance may be harder than testing for Euclidean distance, but not why that difference really matters for downstream use. 

* The presentation is really tailored towards experts. Section 3.1 was pretty readable (modulo some notational issues), but Section 3.2 delved into specific jargon (e.g. "nonadaptive parity decision trees") pretty quickly. 

* Minor issue: the dimensions and notation for the discussion of $f$, $f^*$, $L$, $U$, and $R$ (lines 200–206) was confusing. I guess you meant that $f = f^* \circ L$? and $R$ goes from $r$ to $n$ dimensions, not the other way around? In any case, a bit more explanation ehre would help, as would explicitly matching the notation with that of Section 3.2.

### Questions
I would like to see the authors make a clear and hones) case for the significance of their work to the ICLR community. Why is this the right paper for the venue and the right venue for the paper? Of the many possible lines of basic theory work that are adjacent to ML, which ones fit ICLR? (I guess "all of them" is a fine answer, but maybe not one that would see wide support.)

On a more technical level: what are the implications of this work for tolerant testing, if any? For what settings of parameters do these algorithms improve on those that actually learn a nearby s-Fourier sparse function? (What are specific settings where these algorithms would be useful for parameter selection?) Note that even query-based learning of s-Fourier sparse functions is already a fairly stylized theory question, albeit a fundamental and beautiful one.

### Soundness
4

### Presentation
3

### Contribution
3
