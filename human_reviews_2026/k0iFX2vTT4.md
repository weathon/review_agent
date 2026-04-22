# Sharp Statistical Limits and Algorithm for Attributed Graph Alignment

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 0, 8

## Abstract
This paper investigates the problem of recovering hidden vertex mappings between two correlated weighted graphs with both edge structure and node features. While most existing studies on graph alignment focus solely on edge information, many practical scenarios also provide node features in addition to graph topology. To address this setting, we introduce the featured correlated Gaussian Wigner model, in which the graphs are correlated through a latent vertex permutation, and the associated features are also correlated under the same permutation. We establish the optimal information-theoretic thresholds for recovering the latent vertex mappings. Furthermore, we propose QPAlign, a fast algorithm leveraging quadratic programming relaxation to the Birkhoff polytope, and validate its effectiveness on both synthetic and real datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the information-theoretic limits of graph alignment when both edge structures and node covariates are observed, modeled through a feature-correlated Gaussian Wigner model. The authors show that incorporating node features improves the recovery thresholds for both partial and exact recovery compared to the purely structural (correlated Wigner) case. On the algorithmic side, they propose a practical approach based on a Birkhoff relaxation of the quadratic assignment problem (QAP) corresponding to the maximum likelihood estimator, and evaluate its performance on synthetic and real datasets.

### Strengths
The main strength of the paper lies in establishing, in a rigorous manner, the information-theoretic thresholds for exact and partial recovery in the graph matching problem under the feature-correlated Gaussian Wigner model. The analysis seems technically sound and provides a consistent extension of existing results to a setting that combines structural and feature information.

### Weaknesses
1) **Motivations.** The new model introduced in this work is motivated by the existence of networks with negatively correlated features or negative edge weights (lines 44–48). However, the formal definition of the model only considers positive correlations $r\in (0,1)$, and the experimental datasets involve graphs with positive edge weights. This discrepancy weakens the motivation. Moreover, the influence of node covariates on graph matching has already been studied under models such as the correlated Gaussian-attributed Erdős–Rényi model. The present model thus appears to be a slight variation, primarily chosen for analytical tractability rather than capturing new phenomena.

2) **Novelty of the contribution.** From a technical standpoint, it is unclear what new challenges the proposed model introduces. The proof seems to mimic the approach used by Wu et al. (2022). Because edges and node covariates are assumed independent, the likelihood conveniently decomposes into two independent terms, making the analysis of the key statistic  $Z$ (used to bound the misalignment probability) relatively straightforward.

3) **Presentation and clarity of the writing.** The presentation could be significantly improved by clarifying the motivation and the formal definition of the model (see remarks below) and by providing more explicit comparisons with prior work, especially regarding the proof techniques used to establish the main results. The limitations of the lower bound should also be discussed, as it does not always match the upper bound; this gap weakens the claim that the derived threshold is optimal.

### Questions
1) The definition of the model needs to be clarified. The vertex sets $V(G_1)$ and $V(G_2)$ are not defined. Moreover, the definitions of $G_1$ and $G_2$ themselves rely on these vertex sets, creating a circular dependence.

2) There is no discussion after Theorem 2 about the condition $r\geq 40/d$, and the additional $4\ log\ d$ additive factor appearing in the lower bound. To my understanding, the lower bound matches the upper bound only if one assumes a specific scaling for $d=d(n)$. This condition should be made explicit. 

3) The proof sketch should emphasize what specific challenges arise in the new setting studied. As currently written, it follows the standard approach for deriving information-theoretic thresholds and does not make clear what aspects of the new setting require different arguments. 

4) I don't understand the computational complexity estimate: for each gradient step, one needs to compute $A_1 \Pi^{(t)}$, but since $ \Pi^{(t)}$ is not a permutation matrix, I think the sample complexity should correspond to the general sample complexity for dense matrix multiplication and is not of order $n^2$.

5) If I'm not mistaken, the use of the Sinkhorn (1964) projection algorithm requires the input matrix to have non-negative entries, but due to the gradient step, it is not clear that $\Pi^{(t)}$ still has non-negative entries. 

6) Typos: 
- In Remark 2, isn't the constraint added to program 8 instead of 7?
- line 294: "we derive at the quadratic programming"
- Ponctuation is missing after equation (5).
- Legend in Figure 3: it should be QPAlign and not QPALign

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the graph alignment (or graph matching) problem under a featured correlated Gaussian Wigner model, where the goal is to recover a latent bijective mapping between two correlated graphs.
Each graph consists of correlated weighted edges—modeled as correlated standard normals with correlation coefficient
$\rho \in (0,1)$—and correlated node features, modeled as multivariate Gaussian vectors with correlation $r \in (0,1)$.

The authors first derive information-theoretic limits for partial and exact recovery using a maximum-likelihood estimation (MLE) framework. Their results show that the combined signal-to-noise ratio (SNR) from both edges and features, given by
$n \log\frac{1}{1-\rho^2} + d \log\frac{1}{1-r^2},$
must exceed the threshold
$(4\pm \epsilon)\log n$ for successful recovery.

Since MLE is computationally infeasible due to its exponential complexity in $n$, the authors propose a polynomial-time algorithm (QPAlign) based on a quadratic programming relaxation of the MLE objective. The relaxation replaces the permutation constraint with the Birkhoff polytope (the set of doubly stochastic matrices) and adds a regularizer to guide the solution.
Numerical experiments on both synthetic correlated Gaussian-attributed Erdős–Rényi graphs and real-world datasets (ACM-DBLP and Douban) demonstrate the effectiveness of QPAlign for partial recovery.

### Strengths
* The paper extends classical correlated Gaussian Wigner models to include correlated node features, establishing a unified featured correlated Gaussian Wigner model and showing that the joint use of edges and features enhances the effective SNR for alignment.

* It proposes a computationally efficient relaxation of the quadratic assignment formulation using the Birkhoff polytope, achieving $O(n^3)$ complexity and demonstrating promising empirical results.

### Weaknesses
* Gap in information-theoretic bounds:
The achievability and converse results (Theorems 1 & 2) are not tight. In particular, achievability requires $d = \omega(\log n)$, while the converse does not. The paper lacks discussion or intuition for this additional condition, leaving an unexplained theoretical gap.

* Limited theoretical novelty:
The MLE-based analysis largely follows established techniques from prior works on correlated Gaussian Wigner models (Wu et al., 2022), Gaussian databases (Dai et al., 2019a), and correlated Gaussian-attributed ER models (Yang & Chung, 2024). Since the edge and feature correlations are independent, the overall SNR simply adds, leading to minimal new analytical challenges. The authors should clarify any technical novelty introduced in their proofs.
Additionally, a relevant reference—“Exact Matching in Correlated Networks with Node Attributes for Improved Community Recovery” (Yang & Chung, IEEE T-IT 2025)—also derives similar IT limits under correlated SBMs. The relation and any technical differences from these prior works need to be discussed.

* No theoretical guarantees for the algorithm:
While the QAP relaxation via the Birkhoff polytope has been adopted in the literature (e.g., Vogelstein et al., 2015; Bonmakanti et al., 2024), the paper provides no theoretical connection between the proposed algorithm and the derived IT limits. The lack of recovery or approximation guarantees leaves a gap between theory and computation.

* Experimental limitations:
The experiments evaluate QPAlign but do not compare its empirical performance against the information-theoretic thresholds derived earlier. Moreover, for real datasets such as ACM-DBLP and Douban, the node and edge distributions are highly non-uniform due to community structure, casting doubt on the practical relevance of the uniform correlated Gaussian Wigner assumption mainly considered in the theoretical derivations of this paper.

### Questions
1. What is the intuition behind requiring $d = \omega(\log n)$ for achievability, while the converse result has no such condition? Can this gap be closed, or is it a fundamental limitation of the current proof technique?
2. Beyond combining independent edge and feature correlations, what new analytical challenges or proof techniques distinguish this work from prior studies (Wu et al., 2022; Dai et al., 2019a; Yang & Chung, 2024)?
3. Does QPAlign possess any recovery or approximation guarantees—e.g., conditions under which it converges to the true permutation or achieves a bounded alignment error? 
4. Can you establish a connection between the algorithm’s empirical success and the derived information-theoretic thresholds?
5. For real datasets with community structure, how realistic is the assumption of uniform edge/feature distributions in the correlated Gaussian Wigner model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper studies graph alignment with both edge structure and node features under the featured correlated Gaussian Wigner model. The authors establish information-theoretic thresholds for vertex mapping recovery and propose QPAlign, a quadratic programming-based algorithm.

### Strengths
- Addresses a relevant problem by incorporating node features alongside graph topology
- Aims to provide theoretical guarantees with information-theoretic analysis
- Proposes a practical algorithm with experimental validation

### Weaknesses
Critical issue with problem formulation: 
- The paper suffers from a fundamental lack of clarity in its problem definition. Definition 1 is ambiguous and appears inconsistent: It is unclear whether graphs $G₁$ and $G₂$ are given as input or generated by the model. If they are given, the condition that both ${u,v}$ and ${π*(u),π*(v)}$ must be edges in $G₁$ and $G₂$ respectively only makes sense if $G₁$ and $G₂$ are isomorphic.
- Equation (1) does not clarify this confusion, as the generation process for $G₁$ and $G₂$ is not properly specified.
- If the graphs are indeed assumed to be isomorphic, the claimed results that do not depend on edge density become highly questionable.

The paper requires major revision to properly define the problem before its contributions can be evaluated.

### Questions
1- Are $G₁$ and $G₂$ assumed to be isomorphic? Please clarify this explicitly.
2- How exactly are $G₁$ and $G₂$ generated in your model?
3- How does edge density affect your theoretical results?

### Soundness
2

### Presentation
1

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
This paper addresses the problem of attributed graph alignment, where the goal is to recover a hidden vertex correspondence between two correlated graphs. The available observations consist of the pair of graphs to be matched, together with vertex-associated side information in the form of feature vectors. The main theoretical results are established under the featured correlated Gaussian Wigner model.

The authors derive the maximum likelihood estimator (MLE) and provide high-probability upper bounds on its error under suitable assumptions on the model parameters. They also establish impossibility results in the form of lower bounds, identifying conditions under which recovery is information-theoretically infeasible. The resulting threshold for successful recovery depends on the sum of two components: one reflecting the information contained in the graph structure, and the other arising from the features. This characterization demonstrates that combining both sources of information yields strictly better recovery guarantees than relying on either one alone.

Finally, the authors propose a relaxation of the MLE over the Birkhoff polytope and evaluate its empirical performance on both synthetic and real datasets.

### Strengths
The graph alignment problem is both challenging and highly relevant, and as the authors note, its attributed variant is the most appropriate formulation for many real-world applications. The proposed model extends the correlated Gaussian Wigner model, one of the main statistical frameworks for studying graph matching. In this sense, the work makes a meaningful contribution by addressing important gaps in the existing literature.

One of the paper’s main strengths lies in its rigor and mathematical soundness. The proofs I examined appear correct and well executed.

Another strength is the clarity of exposition, for the most part, which makes the paper accessible and easy to follow despite its technical and theoretical nature.

### Weaknesses
I believe the paper would benefit from the inclusion of a **conclusion section**, where the authors could elaborate on natural extensions of their work. For example, it would be interesting to discuss how the results might generalize to other attributed correlated random graph models, or to settings where dependencies exist between the edges and the features.

In addition, a brief discussion of the optimization aspects of the proposed relaxation would be valuable. For instance, since the relaxed problem is convex, one could consider guarantees for projected gradient descent methods. Although the rounding step remains more challenging to analyze, it would still be informative to comment on possible strategies to improve optimization efficiency—for example, through adaptive step sizes or other acceleration techniques. Moreover, given that the gradient is linear, the Frank–Wolfe algorithm presents itself as a natural and computationally appealing alternative.

### Questions
1. **Clarity of assumptions in proofs**  
   As a suggestion, it would be helpful to remind the reader of certain parameter assumptions within the proofs to improve readability.  
   For instance, it would be good to restate the conditions on $\rho$ and $r$ when discussing the inequality in line 1021.

2. **On the Sinkhorn method**  
   The Sinkhorn method does not yield a Euclidean projection.  
   Is this choice primarily motivated by computational efficiency?  
   How does its performance and accuracy compare to the Dykstra-based approach?

3. **Regularization term (Remark 2)**  
   I found the regularization term discussed in Remark 2 particularly interesting.  
   Could the authors provide some insight into how the algorithm’s performance changes **with and without** this term?

4. **Comparison with Fan et al. (ICML 2020)**  
   Related to the previous point, if I am not mistaken, in *Spectral Graph Matching and Regularized Quadratic Relaxations: Algorithm and Theory*  
   (Zhou Fan, Cheng Mao, Yihong Wu, and Jiaming Xu, ICML 2020), the authors introduce a regularization term with the opposite effect—encouraging solutions closer to the center of the Birkhoff polytope.  
   Could the authors clarify why the regularization in the present work acts in the opposite direction?

5. **On the assumption about $\pi^*$**  
   In Section 3, the authors state that the ground truth permutation is assumed to be uniform over $ \mathcal{S}_n$ under the proposed model.  
   However, the model itself appears to assume a fixed permutation.  
   Could the authors elaborate on this point?  
   It seems that the uniform assumption might be introduced mainly for the lower-bound argument, whereas in the model, $\pi^*$ could be viewed as a fixed but unknown parameter.

### Soundness
4

### Presentation
3

### Contribution
3
