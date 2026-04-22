# Branch and Bound Search for Exact MAP Inference in Credal Networks

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Credal networks extend Bayesian networks by incorporating imprecise probabilities through convex sets of probability distributions known as credal sets. MAP inference in credal networks, which seeks the most probable variable assignment given evidence, becomes inherently more difficult than in Bayesian networks because it involves computations over a complex joint credal set. In this paper, we introduce two tasks called \emph{maximax} and \emph{maximin} MAP, and develop depth-first branch-and-bound search algorithms for solving them \emph{exactly}. The algorithms exploit problem decomposition by exploring an AND/OR search space and use a partitioning-based heuristic function enhanced with a cost-shifting scheme to effectively guide the search. Our experimental results obtained on both random and realistic credal networks clearly demonstrate the effectiveness of the proposed algorithms as they scale to large and complex problem instances.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a new way of exactly deriving versions of the MAP estimate for credal networks, a generalization of Bayesian networks to sets of probabilities. They provide experimental evidence testifying the effectiveness of their method.

### Strengths
The paper is well written, well motivated, it solves an open problem, and it does so exactly. The experiments show that the proposed method is indeed effective.

### Weaknesses
This is a strong paper, with only two minor -- almost cosmetic -- weaknesses, and it deserves to be accepted.

(i) There are a few typos, e.g. in Line 2, it should be "Instead of precise [...]"; there are two parentheses in the second line of section 2, so it is displayed $P_i = P(X_i | \Pi_i))$

(ii) The importance of causal inference in general, and credal networks in particular, for the field of Imprecise Probabilistic Machine Learning could be briefly mentioned in the related work section.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents new depth-first branch-and-bound algorithms for performing exact Maximum a Posteriori (MAP) inference in credal networks—a generalization of Bayesian networks that allow imprecise probabilities.
The authors introduce two related inference tasks, maximax and maximin MAP, and develop algorithms that leverage an AND/OR search space to exploit problem decomposition.
They also propose a partitioning-based heuristic enhanced with a cost-shifting (moment-matching) strategy to guide the search.
Extensive experiments on both synthetic and real-world credal networks demonstrate significant efficiency improvements and scalability up to models with over 3000 variables.

### Strengths
The paper makes a meaningful step forward by providing the first exact branch-and-bound framework for MAP inference in credal networks, addressing a clear gap in the literature.

The proposed AND/OR Branch-and-Bound (AOBB) approach is well-motivated, theoretically grounded, and effectively extends existing frameworks from Bayesian networks to the more general credal setting.

The introduction of partitioning-based mini-bucket bounds with moment-matching (MBMM) is both technically interesting and empirically effective, improving heuristic accuracy without excessive computational cost.

### Weaknesses
While the paper acknowledges weaker heuristics and higher computational difficulty for the maximin case, the discussion could better analyze why this occurs and suggest concrete mitigation strategies.

The theoretical complexity results are concise but could benefit from a more intuitive discussion of practical bottlenecks—especially regarding heuristic pre-compilation and space trade-offs.

### Questions
How sensitive is the algorithm’s performance to the choice of pseudotree structure? Could dynamic variable ordering further enhance pruning efficiency?

For the maximin MAP task, have you explored any alternative bounding schemes beyond mini-buckets that could yield tighter lower bounds?

Could the proposed AOBB framework be adapted for anytime inference (as hinted in Section 5), and if so, how would heuristic accuracy affect the anytime performance curve?

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
This paper focuses on Maximum a Posteriori (MAP) inference in credal networks, which generalize Bayesian networks by allowing imprecise probabilities represented through convex sets of distributions (credal sets).

The authors define two MAP tasks, maximax MAP (finding assignments maximizing the upper probability) and maximin MAP (maximizing the lower probability). To solve these tasks, the paper introduces:

1- Depth-first Branch-and-Bound algorithms for exact MAP inference in credal networks, extending prior AND/OR search formulations used for Bayesian MAP inference.

2- A partitioning-based mini-bucket heuristic with potential approximation and cost-shifting (moment matching) to guide the search and reduce runtime.

The algorithms are evaluated on random and real-world credal networks (e.g., ALARM, Link, Mastermind) and demonstrate scalability to networks with over 3,000 variables.

### Strengths
1- The paper is mathematically sound, with clear definitions of credal networks, the maximax/maximin MAP formulations, and the bounding procedures.

2- Well-written and organized, following the style of classical graphical model research.

3- Empirical results demonstrate strong performance improvements over simpler search strategies and show good scalability.

4- Provides one of the few systematic attempts to perform exact inference in credal networks, which are otherwise rarely explored.

### Weaknesses
1- The overall contribution is incremental, mainly extending existing AND/OR branch-and-bound frameworks and mini-bucket heuristics from Bayesian to credal networks. The adaptation is conceptually straightforward, and the heuristic improvements are mostly engineering refinements rather than theoretical advances.

2- The runtime reduction relies on heuristic approximations (mini-buckets, Pareto least upper bounds, and moment matching). These are not guaranteed to always yield efficient pruning or optimal bounds, and no theoretical runtime guarantees are provided.

3- While the reported results are promising, they are mostly demonstrated on synthetic credal networks. Extending the experiments to more practical domains could further validate the effectiveness and real-world applicability of the proposed approach.

4- The claim that the proposed method “solves these tasks exactly in practice” is somewhat ambiguous. In principle, the algorithm can find the true optimal MAP solution if given enough time, since branch-and-bound guarantees exactness through exhaustive search with pruning. However, in practice, the approach relies heavily on heuristic bounds to reduce runtime, meaning that its “exactness” depends on the tightness of those bounds. This makes the statement somewhat overstated and potentially misleading without a clearer discussion of the trade-off between theoretical exactness and practical efficiency.

### Questions
See weaknesses.

### Soundness
4

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
3

### Summary
This paper uses various techniques developed that have been previously applied to MAP inference in Bayesian networks, and extends them to credal networks, specifically to the maximax and maximin MAP tasks. The resulting algorithm is provably exact and empirically more efficient than existing algorithms (both exact and not).

### Strengths
The algorithm is a substantial improvement over existing work, in a challenging and significant problem. The theoretical quality seems to be very high. The work is mostly clearly presented, though improvement is possible there.

### Weaknesses
The appendices include a lot of additional material, but the individual appendices are not referenced from the main text, making it an "exercise for the reader" to discover when they should skip from the main text to an appendix. The storyline of the main paper came across as incoherent in section 4 due to this.

### Questions
1. Section 1 mentioned Marginal MAP, but doesn't define it. How does it relate to the MAP tasks studied in this paper: Is one harder than the other? For which real-world problems are they best-suited?
2. Figure 1(b): according to the definition, it seems there should be an arc between C and D, correct? Does this affect the solution tree or the way the algorithms operate on this example?
3. Some questions about Algorithm 1:
  - there seems to be a hat missing on $\mathbf{x}_k$ in line 12 of the algorithm (similarly, line 216 of the text has a bar instead of a hat);
  - $S$ is not mentioned anywhere else in the algorithm. Is it an alias for $v(s)$? If so, I suggest to write that instead.
4. What is the distribution of the \texttt{random} graphs?
5. According to the main text, the algorithm for maximin MAP is much less efficient than the maximax one. Yet when comparing mastermind3 in tables 3 and 4, the maximin case is much faster. What is going on here?

### Minor comments

- line 27: "Instead *of*"
- definition of $\max$ on line 251: this should rule out the possibility of taking $q(Y)$ the same as $p(Y)$; as written, $\max$ is always the empty set. Same for $\min$ (line 310)
- the two-column form of the algorithms makes it impossible to see at which nesting level the first line of the second column is. Please resolve this, for instance by adding vertical helper lines.
- section "A Appendix" is empty; remove/replace this header
- Definition 6 and the text below it in Appendix C replicates part of the main text
- line 748: stray "and"

### Soundness
3

### Presentation
2

### Contribution
3
