# Tight Bounds and Achievable Upper Bounds of Minimal Dimensions for Embedding-based Retrieval

- Decision: Reject
- Scores: 4, 2, 6, 8, 6

## Abstract
This paper studies the minimal dimension required to embed subset memberships ($m$ elements and ${m\choose k}$ subsets of at most $k$ elements) into vector spaces, denoted as Minimal Embeddable Dimension (MED).
The tight bounds of MED are derived theoretically and supported empirically for various notions of "distances" or "similarities", including $\ell_2$ metric, inner product, and cosine similarity.
In addition, we conduct numerical simulation in a more achievable setting, where the ${m\choose k}$ subset embeddings are chosen as the centroid of embeddings of the contained elements. Our simulation easily realizes a logarithmic dependency between the MED and the number of elements to embed.
These findings imply that embedding-based retrieval limitations stem primarily from learnability challenges, not geometric constraints, guiding future algorithm design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provides theoretical bounds for the so-called minimal embeddable dimension, which is the smallest dimension for which some configuration of m points with a given functional family can be k-shattered. It shows that both lower and upper bounds are independent of the number of points, and only depend on k, in the special case where the functional family is given by the 3 standard scoring functions: inner product, cosine, and L2 distance. 

The paper also defines so-called minimal achievable embeddable dimension, where k-shattering is replaced by k achievable-shattering, defined by evaluating a scoring function evaluated at the centroid of k nearest points. Here nearest just means highest scores. Then the paper uses a union bound to show that the O(k^2 log m) dimension is sufficient to find a configuration of m points with the k achievable-shattering property. Finally the paper runs a simulation to verify that the true relation between achievable configuration and dimension is indeed logarithmic as opposed to cubic in a referenced paper.

### Strengths
The context of the problem being addressed is of significant interest in the ML community. For instance, in K nearest neighbor retrieval, we need to find the right kind of dimension to ensure most if not all query embeddings can find its nearest k item embeddings simply using dot product, L2 distance, or cosine distance (dot product with item embeddings L2-normalized).

The construction using the moment curve is interesting mathematically. 
The proof of the achievable upper bound is also standard and reasonable. The simulation result also supports its general order of magnitude.

### Weaknesses
The exposition of the paper is quite cryptic sometimes. I will list some examples
l070-075: looks like 3 things are being compared here: MED, MAED, and real life practical situation. The first sentence says MAED is weaker than real life, but the second sentence then concludes that MAED upper bounds MED. The logic simply doesn’t follow.
It would be helpful to tabulate the results for the 3 kinds of scoring functions to make their relationship more transparent. 

There are many typos in the paper, including
l293: “We use optimize m embeddings randomly initialized”
l028-029: "retrieving the top-k answers of top-k largest scores" should be rephrased as "retrieving the answers with the k largest scores".

l019 (abstract): "Our results also align well with existing practices in large language models, vector databases, and other related fields." seem irrelevant.

l097: C_k definition doesn't need min

The paper can also make the result statements as well as the proofs more accessible to general ML practitioners not familiar with proof heavy literature, involving VC dimensions. 

The upper bound result of the MED is pretty contrived. It doesn’t really show that the dimension n doesn’t depend on the number of points, but only that for any m, one can find a configuration of m points in R^n with the property that the k nearest points of any point can be separated by one of the scoring functions. This makes the result of little practical value. 

Even in the MAED case, the so-called achievable query, where it’s required to be the centroids of its k nearest neighbors, seems rather special. In addition, it’s again not saying the dimension upper bound works for all configurations, but rather one can find some configuration with the achievability constraint under the upper bound, even though it should work for almost all cases. I think it’s very much worth highlighting the limitations of these results, and try to make some attempt connecting the results to practical cases, such as in a unsupervised KNN learning task.

The setup for the simulation requires more detailed explanation as well as motivation. The use of gradient descent to search for an achievable k-shattering configuration does not guarantee optimality, but is sufficient for the upper bound. Some references would be useful to compare against other such simulation work for such a theoretical result.

### Questions
Overall I think the paper has some interesting probabilistic results. But it needs to explain the results in a more accessible manner. There should be plenty of space left to add more details in the main text.
I would like to see the paper much better polished in terms of writing style and motivation. Focus on the main claims, namely the 2 upper bounds and 1 lower bound, as well as the special treatment for each scoring function, and making sure the setup, definitions, and proof strategy are completely transparent. Leave some of the propositions to the appendix.

### Soundness
3

### Presentation
1

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
This work studies the *minimal embeddable dimension* (MED) problem, where given a set of $m$ objects
and a pairwise scoring function $f$, we want to know the minimum embedding dimension $n$ such that
we can perfectly recover the top-k (object, query) results according to $f$. The authors consider
the following score functions: Euclidean distance, cosine similarity, and a related inner product.
They also study a so-called "achievable" setting (MAED) where query vectors are of the form
$1/|S| \sum_{i \in S} \mathbf{x}_i$ for some set $S \subseteq X$ of size $k$. They give a clean
proof of the lower and upper bounds for MED (tight up to a factor of 2). They also give
$O(k^2 \log m)$ upper bounds for MAED and some synthetic experiments to demonstrate this result.

### Strengths
- The authors introduce the study of the achievable setting.
- The MED and MAED problem statements and analysis are built on a clean definition of *$k$-shattering*.
- The cyclic polytope example in Section 3.1 is instructive and succinct.

### Weaknesses
- Manuscript is quite unpolished.
- Experiments (Section 4.2) are very interesting but incomplete. It would be
  good in a future version of this paper to strengthen these results (e.g.,
  revisiting the cyclic polytope as a warm up).

### Questions
**Questions**

- What are the lower bounds for MAED (Table 1)?

**Misc**

- [049] Typo: "can be found in Section ??"
- [081] Typo: two different capitalization styles for list items, i.e., "Standard setting" and "achievable setting"
- [081] Typo: "simulation results on the ."
- [095] Nit: How do we handle having the same vector for two different items given the notation $\{\mathbf{x}_{i}\}_{i=1}^m$?
- [102] Nit: Inconsistent use of normal and boldface letters for scalars and vectors.
- [246] Typo: "top-k" --> "top-$k$"
- [281] Sugestion: Add some horizontal space between the captions of Figure 1 and Figure 2 so it's more clear they're separate.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the minimum dimension of the vector space required for embedded retrieval systems, aiming to challenge the view on the "vector bottleneck" in the field. The authors introduce two core settings to analyze this problem:

1. Standard Setting. Under this theoretically idealized setting, the paper proves that the Minimum Embedding Dimension (MED)—required to perfectly retrieve all queries with no more than k answers—has a linear relationship only with $k$ (i.e., $\Theta(k)$) and is independent of the total number of objects $m$ in the corpus.
2. Achievable Setting. Under this more practically relevant setting, query vectors are constrained to be the centroid of the answer set vectors. The paper theoretically derives and experimentally verifies that the Minimum Achievable Embedding Dimension (MAED) required for this constructive method has a logarithmic relationship with the total number of objects $m$ (i.e., $d=\text{O}(k^2\text{log}m)$).

### Strengths
The theoretical proof on the $\Theta(k)$ bound for the Minimum Embedding Dimension provides an entirely new, more optimistic perspective for understanding the theoretical limits of embedded retrieval. On top of that, the paper successfully reframes the "vector bottleneck" problem, shifting it from a seemingly immutable hardware constraint (space dimension) to an optimizable software issue (embedding construction method).

### Weaknesses
1. The experimental validation for the "achievable" $O(\log m)$ bound is not truly achievable, as its training method requires checking all $\binom{m}{k}$ combinations, which is computationally unfeasible for large $m$.

2. The experimental comparison to prior work [Weller et al., 2025a] is misleading because it compares results from two different paradigms (MAED vs. MED). A fair comparison would require the authors to re-run experiments under the same (e.g., MED) setting to prove their optimization method is truly better.

3. The centroid query method proposed in the paper cannot handle complex compositional queries, which limits the applicability of the achievable method. It is important to more honestly define the scope of applicability of their method.

4. There are several minor but noticeable typographical and notational issues in the manuscript.

### Questions
1. Most importantly, to truly support the "achievable" claim, you should demonstrate that this O(logm) bound can also be reached using a scalable, practical training algorithm (e.g., one based on negative sampling or contrastive loss) rather than the full-combination check.

2. To make a more robust claim, you should run your optimization method under the same MED (Standard Setting) as Weller et al. This would provide a true "apples-to-apples" comparison and prove that the difference is due to your superior optimization, not the change in settings. Alternatively, provide a clear theoretical proof or new experimental evidence within your paper demonstrating that the MAED (centroid query) is indeed a harder problem than MED (free query).

3. You should more clearly define the limited scope of the MAED model. It would be valuable to discuss the gap between this "centroid query" model and a more realistic "independent query" model, and what new challenges the latter might introduce.

4. There are several minor but noticeable typographical and notational issues in the manuscript. It is recommended that the authors carefully proofread the paper to improve readability and consistency. In particular:
    - There are some cross-reference errors, including "Section ??"  and "on the \space(a space)" in Section 1.
    - Some misspellings are present, such as "MEAD" in Section 2.3 and "k-shuttering" in Section 3.2.
    - Equation 10 in A.2 should be $\langle v_1, \sum_{u\in S} u\rangle - \langle v_2, \sum_{u\in S} u\rangle$.

### Soundness
2

### Presentation
4

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
The paper studies the problem of finding the appropriate dimensionality to embed data in vector spaces. In contrast with recently published work, the formal findings in this paper show an encouraging picture for embedding based retrieval. First, for common similarity measures, the minimal number of dimensions does not depend on the cardinality of the set to embed. Second the minimal dimensionality is a low degree polynomial of the number k of retrieved vectors: between k and 2k in the general case and quadratic in the "achievable" setting. Considering that the number of retrieved vectors is a small number in practice, these theoretical bounds paint a positive picture for vector search.

### Strengths
- The topic is of great relevance in practice, even though the results are of a very theoretical nature.
- The dimensionality bounds are novel and they bring a much needed formal understanding to the important area of retrieving unstructured data.
- These small bounds also highlight that more work is needed in the embedding models and that such practical work is not a lost cause.
- To the best of my knowledge, the proofs are correct and the level of rigor exhibited is appropriate for ICLR.

### Weaknesses
- The discussion of the achievable setting in the introduction (line 70) feels a bit lacking and its description in the contributions (lines 81 to 84) too vague. The authors should position the achievable setting more clearly in a "hardness of embbedability" scale.
- How tight are the MAED bounds? Although the authors state that this bound may not be tight, I would have appreciated a more detailed discussion of what this bound means in practice.
- How important is the fact that random vectors are used to get the MAED bound? What would happen in a different setting? Would the bound get worse? It is important to clearly state whether the authors are covering a best or worst case scenario here (or neither and it is just a particular one). This should be more clearly stated in the introduction and abstract, because those section seem to implicitly indicate the MAED bound is general.

### Questions
- In the abstract, it is unclear what the authors mean by achievable. It would be useful to have a succint definition.
- In the third paragraph  of the introduction, too little context is given when taking about the work by Weller et al. It is not clear what the query-relevance matrix and rank_{+/-} mean. Adding a couple of sentences might help. Alternatively, the authors should up-level that discussion to a more intuitive explanation.
- There is an undefined symbol at the end of the third paragraph (line 49).
- In the contributions, there is an unfinished sentence in the item about the "achievable setting" (line 81).
- After Definition 2.11, "that if MAED" -> "that MAED" (line 181).
- After Proposition 2.12, "MEAD" -> "MAED" (lines 186 and 187)
- Please add a horizontal space between Figures 1 and 2 as the captions are hard to read.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This submission looks at the problem of determining the minimum dimension needed to encode an arbitrary set $X$ of $m = |X|$ objects into $\mathbb{R}^n$ such that a retrieval query on $X$ with $k$ answers in the set is perfectly retrievable. This is done by way of a combinatorial argument:
1. $k$-shattering is used to define the minimum embeddable dimension (MED)
2. A simple VC-dimension bound falls out of the definitions, parametrized by $m$ and $k$, and works for an arbitrary scoring function "family" $\mathcal{F}$
3. For specific scoring functions (dot products, $\ell_2$, cosine sim.), one can get tight bounds on the MED of $\Theta(k)$
4. A minimum achievable dimension (MAED) is designed to model practical scenarios and provides an upper bound on MED
There is also an upshot given by empirical results, which suggest that how we construct the embeddings (or generate them with a neural net) matters much more than the available dimensions.

### Strengths
1. The bounds on MED are quite surprising and make for a great result.
2. The contrasting optimism for low-dimensional dense retrieval to the prior work of Weller, et al. will make for interesting and important discussion on the limits of vector search in the AI landscape.
3. Careful effort is made to reconcile empirical results with the theoretical results in an intuitive manner. There is also a clean comparison with the prior work, which makes it easier to reconcile the position of this work with existing results.

### Weaknesses
1. Typos (e.g. lines 49, 81, 283): some are quite substantial, definitely get these fixed
2. Considering that this submission aims to contradict earlier work, some more discussion about the earlier work, what is acking in it, and motivation to pursue this approach in place of the prior work should appear earlier on in the manuscript.
3. The MAED discussion is perhaps oversimplifying the practical scenarios that it tries to represent. While it does model the in-distribution setting of vector search, it fails (at the admission of the authors) to capture the nuance that comes with embeddings generating with a neural network. This leads to an empirical section that is, I feel, lacking. To complement the existing results, there should be experiments based on real data with neural network based embeddings that support the paper's results and an effort to quantify the issues that come with such a setting.

### Questions
1. It's common in practice to retrieve a larger number (than $k$) of candidates, then rerank them down using a stronger similarity function into $k$ final results. Is there a way to model that setting with this framework? If we were to naively apply the theoretical results to this method, it's possible we would see poor results (as we rely on $k << m$), but this seems to work remarkably well in practice.

### Soundness
3

### Presentation
2

### Contribution
4
