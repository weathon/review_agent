# From Fields to Random Trees

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
This study introduces a novel method for performing Maximum A Posteriori (MAP) estimation on Markov Random Fields (MRFs) that are defined on locally and sparsely connected graphs, broadly existing in real-world applications. We address this long-standing challenge by sampling uniform random spanning trees(SPT) from the associated graph. Such a sampling procedure effectively breaks the cycles and decomposes the original MAP inference problem into overlapping sub-problems on trees, which can be solved exactly and efficiently. We demonstrate the effectiveness of our approach on various types of graphical models, including grids, cellular/cell networks, and Erdős–Rényi graphs. Our algorithm outperforms various baselines on synthetic, UAI inference competition, and real-world PCI problems, specifically in cases involving locally and sparsely connected graphs. Furthermore, our method achieves comparable results to these methods in other scenarios. The code of our model can be accessed at \url{https://github.com/LOGO-CUHKSZ/From-fields-to-random-trees.git}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a spanning-tree–based MAP inference algorithm (called SPT) for Markov Random Fields (MRFs) on locally and sparsely connected graphs, which is a regime common in power grids, communication networks, and transportation networks. The key idea is to sample uniform random spanning trees (RSTs) from the original graph; then run exact BP on each sampled tree (cycle-free so inference is tractable). Experiments on both synthetic an real datasets show the better performance of the proposal.

### Strengths
- The paper has clear motivation and introdution for the problem setup and the proposal.
- The complexity analysis and convergence guarantee is given and offer insights into the dependence on the problem parameters.
- The proposed algorithm is scalable thus supports inference for large graphs.
- The empirical validation on synthetic + UAI + real PCI datasets shows promising gains of the proposed approach.

### Weaknesses
See questions.

### Questions
- In Theorem 1, the error bound decreases as the number of spanning trees $\mathcal{K}$ increases. The proposed algorithm is taliored for sparse graphs. However, a sparse graph should have less spanning trees compared to a dense graph, which leads to a worse error bound. How do we understand this?
- In the experiment of Figure 1, where we compare energy against number of iteration, I wonder if this is a fair comparison. As these methods can be doing very different things and computation in each iteration step.
- Maybe a better comparison is to look at the wall clock time needed for each baseline to reach a certain energy accuracy.
- Why is the proposed algorithm inconsistent on ER graph?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper explores sparsity and local connectedness in MAP estimation of Markov Random Fields (MRFs) proposing to combine multiple (spanning) subtrees learned exactly by sampling multiple spanning trees from the original graph and applying conventional belief propagation (BP). These exactly solved spanning subtrees are combined through an inverse weighting function accounting for the probability of sampling given edges to provide aggregated approximate solutions to the entire MRF. The approach is simple and straightforward to implement and appears to work well in practice as highlighted in the papers experimental section.

### Strengths
•	The approach is well related to the existing literature in the literature review.

•	The approach appears to work well in practice and is simple to implement.

•	The weighting scheme to correct for how edges in the spanning tree are sampled compared to uniform sampling appears as a simple, elegant, and valid practical approach to correct for biases by the sampling procedure.

Originality:
The approach combining multiple spanning subtrees appears new and original.

Quality:
The paper is generally clear and easy to follow. The experimentation considers comparison to a limited number of baselines which can be improved.

Clarity:
The paper can be improved in its writing - please also see the minor comments under Weaknesses, however, the developed methodology is clear. 

Significance:
Minimizing pairwise MRFs are an extensively studied field with many contributions and approaches developed over the years. The paper here develops an interesting new approach which could warrant publication, but the significance of the results are unclear as no error bars are reported and the paper only compared to a limited number of alternatives.

### Weaknesses
The present procedure relies on estimating the spanning three exactly requiring O(N^3) which limits the approach to small graphs and subtrees. While scalable approaches are discussed it is unclear how well they perform in practice.

Whereas applications where the methodology is important is discussed in the motivation it is unclear how the solutions benefit practical applications. It would strengthen the paper to consider the solutions for at least one of the given problem domains highlighted in the motivation (introduction) and the practical implications.

The method is compared to very few competing methods, whereas the literature and approaches minimizing the energy of MRF including for the considered pairwise MRFs is vast as also highlighted in the rather old survey:

Wang, Chaohui, Nikos Komodakis, and Nikos Paragios. "Markov random field modeling, inference & learning in computer vision & image understanding: A survey." Computer Vision and Image Understanding 117.11 (2013): 1610-1627.

Where submodular random fields have also been considered for pairwise random fields such as the papers cited in the related works section:

H. Ishikawa, Exact Optimization for Markov Random Fields with Convex Priors, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 25 (10) (2003) 1333-1336.

D. Schlesinger, B. Flach, Transforming an Arbitrary Minsum Problem into a Binary One, Tech. Rep. TUD-FI06-01, Dresden University of Technology (2006)

Furthermore MRF estimation has been scaled using massively parallelized implementations as in:
https://download.mmag.hrz.tu-darmstadt.de/media/FB20/GCC/paper/Thuerck-2016-HPG.pdf

In this context, I find the experimentation to only include a very limited  number of comparisons to existing methods, i.e. Mean field, LBP, and TRBP whereas the literature is vast with many proposed methods. 

Minor issues of the paper needing some proof-reading:
Problem equation 1 is known NP-hard in general -> The problem in equation 1 is known to be NP-hard in general

solve equation 1 can date back 80’s of the previous century -> solve equation 1 dates back  to the 80’s of the previous century
approach to infer on MRFs  -> approach to infer MRFs 

Strange use of past tense of “could” in the Methods section which should be “can” throughout please check. 

Since Acquire the true tree -> Since Acquiring the true tree

### Questions
Consider include scalable approaches based on approximate spanning tree estimation procedures and compare how the exact to such scalable approaches compare.

Please provide error bars in Figure 1 and the Tables across multiple runs. As the procedure is non-deterministic it will be good to see how much variability this induces on the results.

MRF estimation has been demonstrated to benefit from massively parallelization – how does the proposed approach compare to such parallelized implementations as in:

https://download.mmag.hrz.tu-darmstadt.de/media/FB20/GCC/paper/Thuerck-2016-HPG.pdf

Furthermore, how does the procedure compare to other methods covered in the related works section such as 

H. Ishikawa, Exact Optimization for Markov Random Fields with Convex Priors, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 25 (10) (2003) 1333-1336.

D. Schlesinger, B. Flach, Transforming an Arbitrary Minsum Problem into a Binary One, Tech. Rep. TUD-FI06-01, Dresden University of Technology (2006)

In equation 12 the product does not produce as I understand it a valid conditional distribution \tilde{p}(x_i|X\{x_i}). Please clarify how samples are drawn from the distribution, is it simply renormalized by \tilde{p}(x_i|X\{x_i})/(\sum_{x_i} \tilde{p}(x_i|X\{x_i})) and shouldn’t "=" then be "\propto" (if it is not a normalized distribution)?

In summary, I consider this a borderline leaning accept paper but with room for improvements in particular in terms of establishing their approach to more alternatives by providing a wider comparison and include error bars for the assessments of results.

### Soundness
3

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
3

### Summary
The paper studied the problem of finding the minimum energy state in a Markov random field (MRF). The problem is NP-hard in general because it can in-code the max-cut problem. The Belief Propagation (BP) is a famous algorithm for computing marginal distributions (inference) of MRFs. The algorithm can solve the inference problem exactly if the graph is a tree. The main contribution of this paper is a heuristic algorithm for finding the minimum energy state of a MRF. 
- The algorithm generate some i.i.d. samples of uniform random spanning trees.
- For each tree (re-weight the edge according to the marginal of random spanning trees), run BP to compute marginals of every nodes. 
- Merge the results of all trees by taking the product in (12) line 237.
- Given the marginals, using a greedy algorithm and a simple sampling (sample each marginal independently) algorithm to find a state. Update the current best answer.

The paper gives theoretical analysis on some step of the algorithm. The paper then did a lot of experiment on both synthetic data and real-world data.

To summarize, I think the theory part is of this paper is simple. The idea share some similarity with graph sparsification algorithm, which also use the random tree to sparsify the graph and change the edge weight (you may add some discussion). Some main theoretical results is the correctness of expectation and a concentration bound. However, it is interesting to see a simple algorithm works in many real application.

### Strengths
- The paper proposed to use random trees to approximate the MRF, which connects the concept of graph sparsification with the MAP (Maximum A Posteriori) inference problem in probabilistic graphical models. Intuitively, by constructing a collection of randomly sampled tree structures, one can run existing algorithms on trees while random trees preserves some information of the original MRF.

- The experimental evaluation on synthetic data and real-world data is comprehensive and well-structured. Detailed tables and visualizations are provided to illustrate the results, including comparisons with baseline methods. The findings shows that their algorithm can achieve a good performance in many situations.

### Weaknesses
- The MAP problem is trivial in trees because one can use a dynamic programming (the algorithm called max-product or min-sum algorithm). In this paper, the authors use BP to compute marginal distributions on trees and use marginal on trees to approximate marginal on MRFs.  The connection between computing marginal distributions and identifying the minimum energy configuration is not stated in the paper.  Even if this is a heuristic, it would be good to explain some intuition here. (See more detailed questions in the next section)

- Some statement in the theorems and lemmas and some proofs look confusing.  (See more detailed questions in the next section)

### Questions
Consider a Markov Random Field (MRF) defined on a general graph, inducing a Gibbs distribution $\mu$ over $\mathcal{X}^V$ (assume $\mathcal{X} =$ {0,1}). Although computing marginals is generally intractable, suppose we have access to an oracle that provides the marginal distribution for each node. For any $v \in V$, the oracle returns $p_v = \text{Pr}_{X \sim \mu}[X_v = 0]$. With this information, we can directly run the GibbsSampler and GreedySelector as described in Algorithm 1. The question is: does the algorithm find a good state with low energy?

Here is a simple example. Consider the energy function defined in (1), with $\mathcal{X}$ = {0,1} and $\theta_i(x_i) = 0$ for all $i$ and $x_i$. For the pairwise interaction term $\theta_{ij}$, define:
- $\theta_{ij}(x_i, x_j) = 1$ if $x_i \neq x_j$
- $\theta_{ij}(x_i, x_j) = 0$ if $x_i = x_j$

Clearly, the minimum energy configuration corresponds to the MAX-CUT of the graph, which is NP-hard. In this case, due to the symmetry between values $0$ and $1$, the marginal distribution for each node is uniform over {0,1}. This suggests that marginals alone may not help in finding the minimum energy state.

**Question:** In general, what is the relationship between computing marginals and finding the minimum energy state? Since the problem itself is NP-hard, we cannot expect the algorithm work for all cases. It is better to give some intuition and explanation on which situation this idea could provide a good solution. 

---

**Lemma 1** relies on an unrealistic assumption. The set $\mathcal{K}$ is a randomly sampled set in the algorithm and may contain duplicate elements. The lemma assumes $\mathcal{K} = \mathcal{T}$, but this assumption is problematic for two reasons:
1. Let $N = |\mathcal{T}|$ be the number of spanning trees. Since $N$ can be exponential in $n$, i.e., $N = \exp(O(n))$, it is infeasible to generate that many samples.
2. Even if it were feasible, the condition $\mathcal{K} = \mathcal{T}$ implies no duplicates in $\mathcal{K}$ (otherwise, equation (17) in the proof of Lemma 1 fails). However, the algorithm samples with replacement, making this event extremely unlikely.

---

**Lemma 2** is more reasonable. It states that the expected energy is correct. Some notational improvements can be made in equation (13):
- You can remove the factor $\frac{1}{|\mathcal{K}|}$ and the summation over $T_k \in \mathcal{K}$, and simply replace $T_k$ with $T$. It suffices to show that the expectation over a single random spanning tree $T\sim \Omega(\mathcal{T})$ is correct. Since $\mathcal{K}$ consists of i.i.d. samples, the expectation of their average is also correct.
- If you prefer to keep the set $\mathcal{K}$, then the expectation should be taken over the randomness of all i.i.d. samples in $\mathcal{K}$, where each $T_k \sim \Omega(\mathcal{T})$.

### Soundness
3

### Presentation
3

### Contribution
2
