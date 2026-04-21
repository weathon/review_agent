# ADDITIVE SEPARABLE GRAPHON MODELS

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6

## Abstract
The graphon function is fundamental to modeling exchangeable graphs, which form the basis for a wide variety of networks. In this paper, we introduce the additive separable model as a parsimonious representation of the graphon, capable of generating a low-rank connection probability matrix for network data. This model effectively addresses the well-known identification challenges associated with graphon functions. We develop an efficient estimation approach that leverages subgraph counts to estimate the low-rank connection matrix and uses interpolation to recover the graphon functions, achieving the minimax optimal estimation rate. We provide the convergence rate of our method, and validate its computational efficiency and estimation accuracy through comprehensive simulation studies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an additive separable model for graphons, producing a low-rank connection matrix and addressing certain identification issues. An estimation method using subgraph counts has been proposed to estimate the graph parameters. Several numerical experiments are showed to highlight the performance.

### Strengths
1. The proposed generalization approach is simple and easy to follow. 
2. The bounds and minmax rates obtained are reasonable and the authors did a good job in explaining the results and the implications. 
2. The numerical experiments encompasses a wide class of gryphon functions and several competing methods. Unlike cherry picking the results, the authors also showed scenarios where their method did not perform well.

### Weaknesses
The utility of the proposed generalization is questionable from both theoretical and empirical perspectives. Empirically, as shown in the comparison tables, the proposed method yields only marginal improvement over existing methods. Theoretically, it raises the question: why is an additive separable scheme considered reasonable? Why not a multiplicative or another type of schema?

### Questions
n/a

### Soundness
2

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
This paper explores additive separable graphon models (ASG) as a a flexible framework to capture low-rank structures in network data. The authors propose simple and efficient algorithm based on subgraph counts for probability connection matrix rank $r$ either being $1$ or $2$, and further use interpolation to recover the graphon functions. A wide range of simulations are included to back up the performance of their method.

### Strengths
- The general framework of additive separable graphon models is appealing, which naturally introduces the low-rankness for network data.
- I do enjoy reading the paper, as it presents ideas in a clear and logical way. This makes the methodology and its contributions straightforward to follow. 
- The simulations showcase both the efficiency and effectiveness of their method. This makes it easy to assess how the method performs across a range of synthetic network settings.

### Weaknesses
-  Overall, I would regard the proposed method as a  method for  estimating the network moment or motifs. That said, I doubt that it would scale easily to even moderate values of $r$, especially due to computational demands. This constraint could make it less practical for most commonly studied stochastic block models, where the connection probability matrix rank is often higher than $2$. 

- Despite what’s suggested in the discussion, I think it would help if the authors could explicitly address the estimation process for general ASG(r) as this is particularly important for practitioners looking to apply it more broadly.

- As a follow-up question, deciding the appropriate rank $r$ could be a challenge in practice. A brief discussion on rank selection—whether there’s a heuristic, or a data-driven way to guide practitioners would make the approach more usable.

- Given the wide range of applications inspired by network literature, I suggest the authors analyze at least one real-world dataset to demonstrate the practical utility of their method.

### Questions
See weakness part.

### Soundness
3

### Presentation
3

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
This paper proposes a method for estimating both the connection probability matrix and the graphon function for low-rank graphons of rank 1 and 2. The authors provide finite-sample error bounds for the connection probability matrix estimation, specifically in the max-norm (i.e., $\|A\|_ {max} = \max_ {ij} |A_{ij}|$). For the graphon function estimation, they establish both asymptotic and finite-sample bounds under the sup-norm on $[0,1]^2$. The experimental results on synthetic data highlight the method's performance across various rank-1 and rank-2 graphons.

### Strengths
1- **Clarity and Mathematical Rigor**: The paper is generally well-written and accessible, with clear explanations that guide the reader through the methodology.  The methods are simple, and the proofs appear mathematically sound and contribute to a solid theoretical foundation.

2- **Competitive Experimental Results**: The proposed methods demonstrate competitive performance on synthetic datasets.

### Weaknesses
1- **Novelty**: Regarding the proposed model Additive Separable Graphons ASG($r$), the authors state in line 076 that this model is new. However, it appears to me as a generic low-rank graphon, which is a well-recognized particular case of a general graphon already discussed in the literature (e.g., Chan and Airoldi, 2014; Xu, 2018). Indeed, for any rank-$r$ graphon, the decomposition given in Eq. $(1)$ follows directly from the spectral theorem. I struggle to see this as a new model, and in my opinion, it would be beneficial for the authors to justify why ASG($r$) is indeed novel. 

In terms of the methods, **Algorithm 1** resembles a step of the power iteration applied to the adjacency matrix $E$ with the starting vector $\mathbf{1}$ (the vector of all ones).  Taking $v= \frac{E\mathbf{1}}{\|E\mathbf{1}\|_ 2}$, we have $v_i=\frac{\sum_ {j}E_{ij}}{\|E\mathbf{1}\|_ 2}=\frac{d_i}{\|d_i\|_ 2}$, where $d_i$ is the degree of the node $i$, as in the paper. On the other hand, the estimation of the eigenvalue is given by the Rayleigh quotient $\lambda'_ 1=\frac{\mathbf{1}^\top E \mathbf{1}}{n}=\frac{\sum_ {i,j}E_{ij}}{n}=\frac{\sum_ {i,j:i\neq j}E_{ij}}{n}$. With this normalization $v_i$ is an estimation of $\frac{G_1(U_i)}{\sqrt{n}}$, then putting this together yields an estimation of probability matrix: $p'_ {ij}=n\lambda'_ 1(vv^\top)_ {ij}=\frac{\sum_ {i,j:i\neq j}E_{ij}}{\sum_ {i,j}d_id_j}d_id_j$. The proposed estimator is $\hat{p}_ {ij}=1\wedge \frac{\sum_ {i,j:i\neq j}E_{ij}}{\sum_ {i,j:i\neq j}d_id_j}d_id_j$. Discounting the clipping on $1$, which one can always do knowing that the matrix to be estimated has entries below $1$, the only difference is the factor $\sum_ {i,j:i\neq j}d_id_j$ is the denominator of $\hat{p}_ {ij}$, compared to $\sum_ {i,j}d_id_j$ in $p'_ {ij}$. In the dense regime treated here, I believe that the difference is negligible.I suspect **Algorithm 2** could be similarly treated. The power iteration is known to converge in one iteration for matrices of rank 1 and fast for low rank matrices. Comparing with this approach may be beneficial, as the direct competitors here are spectral methods, and it’s possible the methods are analogous. 

2- **Difficulty in Extending Results**: Although the authors discuss potential extensions, constructing estimators for lower-rank graphons appears challenging as the equations grow complex quickly. Furthermore, the order (in powers of $E$) of the quantities needed increases with the rank, complicating the computation. Although the subsampling technique seems feasible, it lacks a thorough theoretical foundation. For example, in **Remark 2**, the justification only considers asymptotics, and I suspect finite sample considerations could introduce additional variance. Providing more detail on possible extensions would add value.

3- **Somewhat Unfair Experimental Comparisons**: In their experiments, the authors compare their method to more generic methods that work under broader assumptions (not strictly rank 1 or 2). For example, **USVT** can adapt to various low-rank situations. I consider this comparison somewhat unfair, as the proposed method is tailored to specific ranks in the application, effectively using additional information. Additionally, the authors note that methods are used with default values. Spectral methods with low-rank information might perform as efficiently as the proposed method. 

4- **Limited Applicability**: Limiting the methodology to rank-1 and rank-2 graphons restricts its applicability. Additionally, for rank-2 graphons, **Assumption 1** feels restrictive. For instance, the condition $\int^1_0G_k(u)du\neq 0$, for all $k$, is unmet for $ k = 2 $ when \$ G_1 $ is constant (given $L_2 $ orthogonality). It seems natural to consider polynomial bases as the elements $ G_k $ in a low-degree graphon, which will typically include a constant function.

### Questions
1- Could you address my comments in the weaknesses section above, point by point?

2- The notation $O_p$ used throughout the paper is not defined. 

3- In the finite sample results, such as in Theorem 1, the results should hold with high probability—this is currently unstated.

4- In line 434, the authors mention that the absence of a tuning parameter "enhances robustness." Could they provide a clearer justification for this claim?

5- f I am following their proof correctly, it seems that alternative equations involving $\hat{\lambda}_ 1$ and $\hat{\lambda}_ 2$ could be derived for the rank-2 case.  Specifically, it seems that $\sum_{i,j}A_ {ij}A_{ji}$ should approximate up to rescaling $\hat{\lambda}^2_1+\hat{\lambda}^2_2$. Is this correct, and would it be useful? How does this approach compare with the method they propose?

### Soundness
3

### Presentation
3

### Contribution
2
