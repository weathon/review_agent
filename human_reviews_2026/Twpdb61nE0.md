# Efficient Differentiable Discovery of Causal Order

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
We introduce a differentiable and scalable regularizer for causal order, which can be integrated into gradient-based learning systems to inject causal inductive bias. Our approach builds on Intersort (Chevalley et al., 2025), a recently proposed score-based method that infers causal orderings from interventional data. While effective, Intersort requires discrete combinatorial optimization, making it computationally expensive and non-differentiable. We address this limitation by relaxing the score with continuous optimization techniques, including differentiable sorting and ranking, yielding a differentiable surrogate objective. The resulting formulation can be used as a regularizer to encode prior knowledge about causal order—whether derived from interventions, domain heuristics, or learned proxies. As a proof of concept, we instantiate this regularizer using single-variable interventions and demonstrate significant improvements in causal discovery tasks across diverse synthetic datasets. Our work enables scalable, differentiable, and flexible integration of causal order into modern learning systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a differentiable and scalable causal order regularizer based on Intersort (Chevalley et al. 2025c). Intersort is a score-based algorithm that discovers causal order using intervention data, but it relies on discrete combinatorial optimization, is computationally expensive, and is non-differentiable. By continuously relaxing the Intersort score using differentiable sorting techniques (specifically the Sinkhorn operator), this paper derives a differentiable surrogate objective (DiffIntersort). The authors repurpose this differentiable score as a regularizer to inject an inductive bias toward causal order into gradient-based learning systems. As a proof-of-concept, the authors apply this regularizer to a simple differentiable causal discovery model (called CausalDisco in the paper) and demonstrate on a variety of synthetic datasets (linear, RFF, GRN, and NN) that the inclusion of this regularizer significantly improves the model's structural recovery capabilities (SHD, SID) compared to the model without it, and surpasses baselines such as GIES and DCDI on certain nonlinear datasets.

### Strengths
1. The tech of a differentiable operator is novel to me and may provide insights into other fields.

2. The experiments are comprehensive.

### Weaknesses
1. This approach is very similar to DP-DAG (Charpentier et al. 2022) and BayesDAG (Annadani et al. 2023). DP-DAG is also permutation-based, differentiable (Sinkhorn), and supports intervening data. It is the approach's most direct and significant competitor. The lack of experimental comparisons with DP-DAG significantly reduces the persuasiveness of our empirical results, particularly in terms of scalability and accuracy. 

2. Although the authors conduct experiments on RFF, GRN, and NN (non-linear) datasets, the paper does not explain how the model $f(\cdot)$ and the fitting loss $\mathcal{L}_{fit}$ are parameterized in these cases. This is a key missing implementation detail.

### Questions
above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to recover the correct causal order of a causal model with observational and single-variable interventional distributions. It extends the Intersort score (Chevalley et al. 2024), a metric for identifying the causal order, to a differentiable form using the Sinkhorn operator, thus enabling efficient computation. The method of using potential to represent causal order is similar to Annadani et al. (2023). The paper also proposes to use the Intersort score as a regularization to impose learning the causal structure.

### Strengths
- The development of scalable approaches is of interest in the field of causal discovery.

- Experiment shows the proposed method achieves efficient computation when the number of variables is large.

### Weaknesses
- The novelty is unclear to me, as the method resembles Annadani et al. (2023).
- The diffIntersort score is not convex, which might result in converging to a local minimum.
- The paper aims to improve computational efficiency with respect to the number of variables d. However, for enhancing the scalability of score-based causal discovery methods, it may be more important, in my view, to evaluate both accuracy and computational efficiency with respect to the dataset size instead. Especially, Intersort (Chevalley et al. 2024) has already achieved $O(d^3)$ computation complexity, which is not large.
- The section of diffIntersort as a regularizer is unclear, as illustrated in Questions.

### Questions
- When DiffIntersort score is utilized as a regularizer, does that make the loss function $\mathcal{L}_{fit}(\theta, p)$ non-differentiable?
- Conceptually, the correct model should maximize the DiffIntersort score. Why $\lambda > 0$ in equation (6)? Minimizing $\lambda S(p)$ would encourage worse causal model fitting.
- The linear SEM in equation (7) is confusing unless I miss some important points. If the observational distribution (and interventional distribution) is available, why can't $\boldsymbol{W}$ be inferred directly? The causal structure will be naturally given after we find that some $W_{j,i}$ is close to $0$.
- The analytical computational complexity of diffIntersort is $O(d^2 T)$, where $T$ is the iteration number that is typically $500$. It is unclear how this number is selected. Should there be a stopping criteria according to the convergence of $S(\boldsymbol{p})$?
- Is there any Bayesian interpretation for using the DiffIntersort score as a regularizer?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a method for causal order discovery based on interventional data that can be optimized through gradient descent. The work heavily builds on the INTERSORT method and provides a gradient-based optimization mechanism for the same score. A significant part of the contribution is the theory stating that the goal can be efficiently optimized. It is shown empirically that integrating DiffIntersort into the causal discovery method improves its performance.

### Strengths
1. The experimental evaluation is extensive.
2. The paper proposes a novel solution that significantly improves the scalability of previous methods.

### Weaknesses
1. The paper is hard to parse and is not self-contained:
* The notation is unclear (for example, in line 128, C in the upper index is not introduced). 
* The assumptions, definitions, and theorems are informal and difficult to understand (for example, in assumption 2.3, what is a “detectable change”?; definitions 2.1 & 2.2 mix comments with the actual definitions; definition 2.7 is completely unclear to me)
* Some algorithms and theorems, which constitute a part of the proposed solutions, are not introduced properly: SORTRANKING, CausalDisco,  Theorems 2 & 4 from Chavalley at al. 2025c (line 348).
* Assumptions are not stated clearly. For example, in Assumption 2.3,  do the Authors assume single-node interventions?
* Section 2.2 could use additional structure to improve clarity.
2. The reproducibility of the results is low due to a lack of clear and detailed descriptions about the experimental setup and specifics about the proposed causal discovery method. This significantly reduces the potential impact and soundness of the work.

### Questions
1. Are the upper bounds from Thm2 and Thm4 upper-bounding the error or the accuracy of the method? Which method: Intersort, SORTRANKING, or both?
2. The CausalDisco method is not described. How does the proposed method for causal discovery work? Especially, how does it handle non-linear data and interventional data?
3. Figure 1 - What are the black lines (variance, standard error, or confidence intervals)? How were they computed? According to the caption, only one run was conducted in each configuration. Also, what are E and #edges? Which is the parameter of the setup, p_e or E, or #edges? Why not randomly select a certain number of variables for intervention for more reproducible results instead of sampling independently with probability p_int? Why does the performance of SORTRANKING often violate theoretical upper bounds?
4. Can DiffIntersort be used to regularize other causal discovery methods? Which ones?
5. What statements from section 2.2 apply to DiffIntersort? What are the theoretical guarantees for its convergence?
6. line 199 typo “finite t and T” -> “positive t and finite T”

### Soundness
2

### Presentation
1

### Contribution
3
