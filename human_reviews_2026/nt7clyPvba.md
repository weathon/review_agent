# Moment Matters: Mean and Variance Causal Graph Discovery from Heteroscedastic Data

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
This paper proposes a Bayesian causal discovery approach 
  to uncover the causal mechanisms 
  underlying heteroscedasticity, 
  where the variance of one variable is influenced 
  by the values of the others.
  To distinguish between the causes that affect the mean
  and those that influence the variance,
  we infer the posterior distribution over 
*mean* and *variance causal graphs*,
  whose structures can be different, 
  depending on the moment information.
  We establish identifiability conditions for these causal graphs 
  by extending the results on heteroscedastic noise models (HNMs).
  Building on these conditions,
  we develop a variational inference framework that can 
  incorporate prior knowledge about 
  the node orderings of the underlying graphs.
  We experimentally show that our method can successfully infer both mean and variance causal graphs,
  outperforming the state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new causal discovery framework that distinguishes between causes affecting the (conditional) mean and those affecting the (conditional) variance of a variable. It introduces the concept of mean and variance causal graphs, extending existing heteroscedastic noise models (HNMs). The authors prove identifiability conditions for these graphs and develop a Bayesian variational inference method that infers their posterior distributions from observational data. Empirical results on synthetic, semi-synthetic, and real datasets show competitive performance compared to prior methods.

### Strengths
1. The proposed Bayesian inference with uncertainty quantification is very important, not only for this specific case of discovery in heteroscedastic noise models, but generally to all causal discovery methods. Unlike point-estimation methods, such full posterior estimation gives more robust results especially in small data regimes.

2. Theoretical results and the whole paper progression is generally well structured. The assumptions are stated clearly and used transparently in the derivation of identifiability conditions.

### Weaknesses
1. The motivation for separating the mean graph and the variance graph is still unclear to me. While the authors argue for the importance of distinguishing between causes of mean and variance (e.g., in drug design or economic variability), the practical necessity of _explicitly modeling two separate causal graphs_ remains somewhat unclear. Intuitively, one could always first estimate a moment-agnostic causal graph and then use flexible (e.g., nonparametric) regression to analyze how each parent affects the target variable (referred to as "double dipping" in this paper). The necessity could be made more precise and convincing.

2. More literature review and comparison to other HNM models and methods are needed, especially since this is a relatively new area. The authors show two main advantages of their model comparing to others (separating, and Bayesian estimation). But for a balanced comparison, what are the other aspects that other methods may be able to address but not this one? In particular,

   - Are there existing models (e.g., beyond HNM) that allow more general functional forms, such as multiplicative or non-additive noise structures (instead of multiplier only posed on the exogenous noise)?

   - Do any existing methods relax the Gaussian noise assumption?


3. More elaboration about the identifiability conditions are needed. Specifically,

   - Are the stated conditions only sufficient, or are they also necessary for identifiability?

   - What happens when the conditions are violated? For instance, could the authors give simple concrete examples where identifiability fails (e.g., two models with different graphs but same distribution), to illustrate the role of nonlinearity or piecewise variance functions?

4. Minors:

   - The title “Moment Matters” may be slightly misleading to readers expecting techniques involving higher-order moments (e.g., skewness, kurtosis), when in fact the method focuses on first and second moments only (mean and variance). May consider rephrasing or clarifying this early in the introduction.

   - Assumption 3.2 (causal minimality) can be explained more in details in the main body, especially to emphasize that it is weaker than the standard faithfulness assumption.

### Questions
see "weaknesses".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Bayesian causal discovery approach that explicitly disentangles and recovers mean and variance causal graphs. Building on heteroscedastic noise models (HNMs), the authors establish identifiability conditions for mean and variance causal graphs and propose a variational inference approach that enables uncertainty quantification. The method is evaluated on synthetic, semi-synthetic, and real-world biological datasets.

### Strengths
1. The proposed approach can qualify uncertainty, which is an improvement over most prior point estimation methods.

2. The method allows domain knowledge to be flexibly incorporated into the learning process via a differentiable relaxation of the permutation matrix. This is important for practical small-sample applications.

3. The authors provide comprehensive experimental results to validate the effectiveness of the proposed appoach.

### Weaknesses
My main concern is that the novelty of this paper is somehow limited.

- Theorem 1 in (Yin et al., 2024) has already established identifiability of HNM under three conditions. Theorem 3.5 in this paper is very similar to Theorem 1 in (Yin et al., 2024) and the proof of the former relies heavily on the latter.
- While the exploration on mean/variance graph separation is well-motivated, the variational Bayesian treatment, use of Gumbel-Softmax relaxations, and exploitation of domain knowledge via permutation matrix regularization are closely related to existing Bayesian DAG learning frameworks such as DDS (Charpentier et al., 2022), MC3 (Giudici and Castelo, 2003), BayesIMP [1], and BayesDAG [2]. The transition to mean/variance-specific graphs is meaningful, but primarily an extension rather than a conceptual breakthrough.

[1] Bayesimp: Uncertainty quantification for causal data fusion. NeurIPS 2021

[2] Bayesdag: Gradient-based posterior inference for causal discovery. NeurIPS 2023.

### Questions
There is a typo in line 179: "if $\pi (i) < \pi (j)$, then $X_{\pi (j)}$ cannot have a directed path to $X_{\pi (i)}$", it seems that $X_{\pi (j)}$ should be "$X_ j$ cannot have a directed path to $X_ i$".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a mean-variance heteroscedastic noise model (HNM) that separates causal influences on a variable’s mean and variance via two graphs, $G^M$ and $G^V$, assumed to share a topological order. This model is a reparameterization of the original HNM that allows for more interpretable causal discovery. The authors prove identifiability of both graphs under standard HNM assumptions plus the shared ordering constraint, and develop a variational inference method using Gumbel-Softmax and SoftSort to learn both graphs jointly. Experiments on synthetic and real datasets show the method can uncover causal links missed by standard approaches, especially variance-only effects.

### Strengths
a. The paper establishes identifiability of dual causal graphs ($G_M$ and $G_V$) under clear conditions. The proof extends existing heteroscedastic causal discovery theory (e.g. Yin et al., 2024) to show that one can recover not just the overall DAG structure but also which edges belong to mean vs. variance relationships, given a shared ordering.

b. It proposes a Bayesian variational approach to infer two linked DAGs simultaneously. The formulation cleverly uses a shared permutation (ordering) for both graphs and generalizes differentiable DAG sampling (DDS) to handle two adjacency matrices. Techniques like Gumbel-Softmax and SoftSort are employed to maintain differentiability, which is an innovative extension of prior continuous DAG optimization methods.

c. The framework naturally incorporates prior knowledge (e.g. known partial ordering of nodes) into the inference procedure. This is valuable for real applications where domain knowledge about causal ordering exists and can guide the search.

### Weaknesses
a. The identifiability and method rely on the assumption that the mean and variance causal graphs share a single topological ordering. In practice, this means no cause-effect relationship flips between mean and variance graphs (an edge present in one cannot appear reversed in the other). It might be too restrictive in some real systems. If the true causal mechanism violates this shared order (e.g. a variable influences another’s variance but is downstream in mean effects), the current approach may struggle or require the user to know and enforce the correct ordering upfront.

b. Like standard HNM, the theory requires Gaussian noise and specific nonlinear forms (mean functions must be nonlinear; variance functions non-constant piecewise). These assumptions are crucial for identifiability but limit generality.

c. It seems that most distributions under this two-graph model could also be captured by a standard single-graph HNM, with appropriate functional form (mask). The mean-variance HNM, while more interpretable, does not expand the class of distributions that can be represented compared to the original HNM.

### Questions
a. How critical is the Gaussian noise assumption in practice? Could the method be adapted for non-Gaussian heteroscedastic noise in a way that still yields at least partial identifiability?

b. The approach assumes a shared causal order. If this assumption is mildly violated in reality (for instance, one edge slightly conflicts between mean and variance ordering), how robust is the inference?

c. Given the two-graph sampling scheme, what are the practical limits on the number of nodes $d$ the method can handle? Have the authors considered any heuristics or structure in the permutation search (besides simple priors) to improve scalability?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a Bayesian causal discovery method for mean-variance hierarchical noise model causal graphs: where the mean and variance of a node depends on its parents. Crucially, the mean and variance graphs can be different.

The authors provide an identifiability result and provide a variational method for recovering an approximate posterior using neural networks to approximate the mean and variance functions. Some improvements in the implementation are introduced to make the optimization more tractable, and the method is compared against several competing Bayesian methods on a variety of data sets.

### Strengths
+ The identifiability result is, as far as I can tell, correct and nicely motivated.
+ The parameterization is well-structured and clearly chosen in a way that makes the variational inference tractable.
+ The implementation tricks such as the two-phase optimization and prior knowledge incorporation are novel applications to improve the performance of the approach.
+ The empirical results are quite strong on these small datasets.

### Weaknesses
+ I am not sure how effectively the proposed probabilistic model can actually capture the posterior of the mean-variance HNM. As I understand it, the mean and variance functions are represented by an MLP which is independent of $A^M$ and $A^V$ (except insofar as the edges are masked out with the adjacency matrices when applying the MLP). Therefore, the MLP would have to learn mean and variance functions that would be applicable when the underlying inputs correspond to different nodes. As a somewhat trivial example, if we had some posterior density on $X_1 \rightarrow X_2$ and also on $X_1 \leftarrow X_2$, then the MLP would have to be able to represent the mean function from $X_1$ to $X_2$, but also from $X_2$ to $X_1$. It's possible I have misunderstood this point, but it seems like this independence assumption could be quite limiting.

+ The contribution of the prior knowledge incorporation is a bit unclear, since there's no baseline for the other methods. It's hard to tell if the improvement is just because of the fact that some of the edges are being specified to correctly exist/not exist.

### Questions
+ Is it possible to adjust your method to condition the MLP on the permutation or adjacency matrix?
+ Could you provide some small-dimensional (2,3,4)-node cases where the posterior can be calculated exactly or via direct sampling, and compare this approach to the exact posterior?
+ Could you provide a comparison to the other methods when they are given the prior knowledge incorporation? You could for instance take the methods' solution and specify that an edge must exist in the solution when given the prior information.
+ Did you experiment with different values of the temperature parameters $\tau$? How does the solution accuracy/optimization smoothness trade-off?

typos: 'VARINACE' in table 4.

### Soundness
3

### Presentation
3

### Contribution
3
