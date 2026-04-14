# Rethinking Degree-Corrected Spectral Clustering: a Pure Spectral Analysis & Extension

- Decision: Reject
- Scores: 5, 8, 3, 5, 3

## Abstract
Spectral clustering is a representative graph clustering technique with strong interpretability and theoretical guarantees. Recently, degree-corrected spectral clustering (DCSC) has emerged as the state-of-the-art for this technique. While prior studies have provided several theoretical results for DCSC, their analysis relies on some random graph models (e.g., stochastic block models). In this study, we explore an alternative analysis of DCSC from a pure spectral view. It gives rigorous bounds for the mis-clustered volume and conductance w.r.t. the optimal solution while involving quantities that indicate impacts of (i) high degree heterogeneity and (ii) weak clustering structures to DCSC. Inspired by recent advances in graph neural networks (GNNs) and the associated over-smoothing issue, we propose ASCENT (Adaptive Spectral ClustEring with Node-wise correcTion), a simple yet effective extension of DCSC. Different from most DCSC methods with a constant degree correction for all nodes, ASCENT follows a node-wise correction scheme. It can assign different corrections for nodes via the mean aggregation of GNNs. We further demonstrate that (i) ASCENT reduces to conventional DCSC methods when encountering over-smoothing and (ii) some early stages before over-smoothing can potentially obtain better clustering quality.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents an alternative analysis of degree-corrected spectral clustering (DCSC) from a spectral perspective. The authors derive theoretical bounds on the mis-clustered volume relative to an optimal solution for conductance minimization. In addition, they propose ASCENT, an extension of DCSC that uses a node-wise correction scheme, which aims to improve clustering quality by adaptively correcting node-specific degrees. The authors validate ASCENT against established spectral clustering and DCSC baselines, reporting improved results on synthetic and real-world datasets.

### Strengths
1. The pure spectral approach is interesting and sidesteps the usual model assumptions, which makes the results more broadly applicable.
2. I do appreciate the ASCENT's node-wise correction scheme, which shows the potential for handling challenging regimes, especially with severe degree heterogeneity.
3. The authors test the proposed algorithm on a range of datasets, demonstrating that it can outperform many traditional methods, which strengthens the practical utility.

### Weaknesses
1. The paper's notation is dense and at times difficult to follow. For example, definitions for $E(S, V \setminus S)$ and $\mu(S)$ should be provided as separate statements above Definition 1 to improve clarity;  $m_K$ is used in (5) before it is defined in (6), which breaks the flow; In Proposition 9, $\Phi$ is redefined with a minor variation ($\tau_{max}$), which could be better handled by using a different symbol or subscript to make the distinction clearer. 
2. Some parts of the derivations, especially in Section 3, appear redundant. For example, the sentence following Lemma 6 that explains the proof idea seems unnecessary since the lemma is cited from prior work. Overall, the buildup before Theorem 8 (the main result) feels too lengthy; some of these derivations could be relocated to the appendix. The technical depth in these derivations does not appear to add significant new insights unless I’ve overlooked a non-trivial component.
3. As previously mentioned, the ASCENT algorithm and its node-wise correction scheme should be the highlight of this paper. However, the theoretical results in Section 4 don’t convincingly showcase the advantages of ASCENT. Proposition 9, for example, does not demonstrate the benefit of the iterative correction step, and Theorem 10 feels somewhat disconnected, reading more like a heuristic explanation of Figure 1. I recommend replacing Theorem 10 with a result that ties more directly to the ASCENT algorithm, or alternatively, presenting a simpler explanation in text form.
4. Selecting suitable values for parameters $(\theta, L, K)$ could be challenging in practice. While $L=3$ seems effective in these experiments, a more adaptive approach or further guidance on parameter selection for real-world datasets would be helpful. 
5. The paper contains several typographical errors, and a thorough proofreading is recommended. Examples include: (1) Line 100: (iii) (iv) should be (i) and (ii); (2) Line 154: $1 > \lambda_1 \le \lambda_2 \cdots $ (3) Line 1163: The sentence ending in "..in Fig." is incomplete (4) "Eigenvalue decomposition" could be replaced with "eigen-decomposition," which is the more common term.

### Questions
See weakness part. Also
1. What is the rationale for selecting $K \in \{2, 8, 32\}$ in the experiments?   
2. The range of $\theta$ values tested in the experiments seems broad. Could you discuss whether the algorithm is sensitive to the choice of $\theta$?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper explores advancements in graph clustering, specifically focusing on the Degree-Corrected Spectral Clustering (DCSC) method. The authors critique traditional analyses of DCSC, which often rely on random graph models like SBM. They propose an alternative analysis framework based on spectral theory alone, presenting new metrics for assessing mis-clustered volumes and analyzing the algorithm's ability to handle diverse node degrees and weak clustering structures.

A key contribution is also the introduction of a novel clustering algorithm, ASCENT (Adaptive Spectral Clustering with Node-wise Correction), which differs from DCSC by applying individualized correction values to each node rather than a single global correction factor. This node-wise adjustment, inspired by techniques in Graph Neural Networks (GNNs), helps ASCENT overcome issues related to graph over-smoothing. In tests, ASCENT demonstrated superior performance over both conventional spectral clustering and DCSC approaches in various scenarios involving high degree heterogeneity and weak clustering structures.

### Strengths
Traditional analyses of DCSC often depend on assumptions from random graph models, like the Stochastic Block Model (SBM), to derive theoretical guarantees. The authors propose a new approach by analyzing DCSC solely through spectral graph theory. This analysis provides an upper bound on the mis-clustered volume relative to an optimal solution, without relying on random graph models.

The ASCENT algorithm seems promising as a simple improvement over DCSC.

### Weaknesses
1) Over-Smoothing Control: Although ASCENT attempts to mitigate over-smoothing, the choice of the parameter L (number of correction iterations) is critical. Incorrectly setting L could lead to suboptimal clustering, and a more dynamic method for determining this parameter might enhance robustness.

2) Complexity and Scalability: Although ASCENT aims to address high degree heterogeneity and weak clustering structures, node-wise correction might increase computational complexity, especially on large-scale graphs with millions of nodes. The over-smoothing issue addressed by ASCENT could still emerge in massive networks where iterative corrections might become infeasible.

3) Initialisation in ASCNET: Since initial corrections are based on node degrees, ASCENT’s performance could vary depending on the graph’s degree distribution. In graphs with degree heterogeneity, initial corrections may differ widely, and ASCENT’s iterative process may need more iterations to bring node corrections to a stable, locally consistent state, or possibly oscillate ?

4) The presentation is a bit dense and it could useful to try to re-organize a bit the paper to make it more more reader-firendly.

### Questions
The questions are related to the identified weaknesses:
1) Can you effectively  do cross validation to find L?
2) How would you consider scaling the algorithm to big graphs?
3) It would be useful to define over smoothing more rigorously, especially in the context of ASCENT which is pretty different from GNN?
4) The paper also notes that, after enough iterations, ASCENT effectively reduces to a conventional DCSC method with a global, constant correction term, indicating that the algorithm reaches a state where further updates do not change the corrections meaningfully. Is there  a convergence guarantee in the strict mathematical sense? Are there possible guarantees  against oscillations?
5) Maybe give more intuition on crucial parameters like psi

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes an extension of the degree-corrected spectral clustering algorithm, ASCENT, from a spectral perspective to address networks with high degree heterogeneity or weak clustering structures.  The method assigns different corrections for nodes via
the mean aggregation of GNNs, instead of constant degree correction.  Theoretically, the paper gives a rigorous bound for the mis-clustered volume.

### Strengths
1) Without relying on specific random graph models,  the paper provides a rigorous bound on the mis-clustered volume.
2) Being able to assign different corrections for different nodes.
3) The experiment identifies some interesting phenomena relates to 'early stages'.

### Weaknesses
The theoretical results on the mis-clustered volume (i.e. Theorem 3 and Theorem 8) are nearly identical to Theorem 1 and Theorem 5 in the paper 'Fangmeng Liu, Wei Li, and Yiwen Zhong. A further study on the degree-corrected
spectral clustering under spectral graph theory. Symmetry, 14(11):2428, 2022.' . Additionally, the structure of the current paragraph closely mirrors that of the previous paper. However, the prior work is not cited, which raises concerns regarding proper attribution. As a result, the novelty of this paper is limited, and it would benefit from further clarification on how it advances the existing literature.

The paper provides insufficient detail regarding the tuning of the parameter $\theta$, lacking a clear explanation of how it was chosen during the data analysis process. Furthermore, the experimental results in Section 5.1 show only a slight improvement over the baseline methods. This raises concerns that if an inappropriate value for $\theta$ is selected, the proposed approach may offer no noticeable advantage over existing spectral clustering methods.

### Questions
In Theorem 8, the assumption of an upper bound on $\Psi$ is made without providing any justification. It would be beneficial to offer an intuitive explanation for this assumption and validate the upper bound using both simple network structures (e.g. the Erdos-Renyi model) and real-world networks to demonstrate its practical relevance. Additionally, theoretical analysis or sufficient conditions that highlight the universality of this assumption would further strengthen the argument.

 The theoretical bound for the mis-clustered volume is derived based on the solution of minimal conductance, which is an unconventional approach in network clustering analysis. It would be helpful to explain whether this mis-cluster rate is valid in practical applications. Ideally, the mis-cluster rate should be evaluated with respect to the ground truth, as this is a more commonly accepted and meaningful metric.

The limitation on the number of iteration steps in the ASCENT model is introduced to address over-smoothing issues. However, the impact of neglecting the over-smoothing problem on clustering results remains unclear. The original degree-corrected spectral clustering (DCSC) method already performs well, and the proposed approach shows only slight improvements by adding two extra parameters, $L$ and $\theta$. If the tuning parameters does not selected optimally, the performance may be inferior than that of the original DCSC method. Also it is unclear whether the performance of the method is sensitive to the tuning parameters.

 The authors should provide an analysis of the computational complexity of the proposed method ASCENT. Additionally, a comparison of ASCENT with other existing approaches in terms of runtime would be beneficial. Given that the proposed method involves iterative steps for the correction of node-wise parameters $\{\tau_i\}_{i=1}^n$, this analysis is necessary for assessing the efficiency and practicality of the approach.

The authors should justify the use of the mean aggregation operation in the computation of the node-wise parameters $\{\tau_i\}_{i=1}^n$. Additionally, please explain the rationale behind employing a GNN-based graph clustering method within the context of spectral clustering analysis.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper examines Degree Corrected Spectral Clustering (DCSC) from a pure spectral view,  not requiring any model assumptions. In theory, the author establish a rigorous bound on the mis-clustered volume for the ISC approach (Qing and Wang) with respect to the optimal solution under conductance minimization. This bound depends on both high degree heterogeneity and weak clustering structure.  In algorithm, inspired by recent advances in GNNs, they propose ASCENT which iteratively updates the degree correction term $\tau$, which is used to construct the Laplacian. Experiment results demonstrate  that ASCENT is reduced to the DCSC algorithm, ISC, when the number of iteration is large, due to the over-smoothing issue. However, at early iterations, ASCENT can achieve  better accuracy compared to other DCSC algorithms. Results on real datasets further compare the performance of ASCENT with other DCSC algorithms.

### Strengths
The research question addressed is both intriguing and significant. Existing literature generally requires model assumptions, meaning that users must first confirm that the network data indeed fits the model assumptions before applying the corresponding results and error bounds. Understanding the robustness of spectral clustering under minimal assumptions is therefore a valuable contribution to the field, complementing existing work. This paper not only offers rigorous theoretical insights but also introduces an extension of the DCSC algorithm. The comprehensive analysis on various real datasets further strengthens the study.

### Weaknesses
The theoretical contributions of the paper could be presented more clearly. While several lemmas and theorems are introduced in Section 3, their purposes and connections are not immediately apparent, making it difficult to follow the logical flow. I suggest adding a high-level overview of the theorems and lemmas to clarify their roles in the analysis before diving into the details. Additionally, I am uncertain how useful the main result (Theorem 8) since the recovery accuracy by spectral clustering without a model assumption is still unknown. Theorem 8 appears to provide a bound comparing the spectral clustering to the optimal solution under conductance minimization, yet it does not provide insight into the accuracy of this optimal solution itself. Furthermore, I am concerned that the condition required for $\Psi$ in Theorem 8 might be too stringent and challenging to satisfy for high degree heterogeneity case. Another limitation is that conductance minimization may not be appropriate for disassortative networks.
  
Regarding  the ASCENT algorithm, the motivation behind iteratively updating the degree correction term $\tau$ for each node is unclear. While ASCENT demonstrates some advantages in the early stages, it remains unclear  what the number of iteration should be in practice, and the accuracy is only slightly improved as shown in Figure 2. I believe further justification for the optimal iteration number, such as if it is related to certain graph structures, would strengthen the paper. The authors might also consider some analysis to determine optimal iteration numbers for different types of graphs.

### Questions
1. Is it possible to consider alternative optimization objectives other than conductance minimization, which might yield an optimal solution that is closer to the output of ISC? Additionally, if we were to use SCORE+ instead of ISC, would this result in a very different upper bound?

2. Is it correct to select the first $K$ eigenpairs of the Laplacian matrix when the eigenvalues are ordered in a decreasing order? If we write $A =\mathbb E A + W$, where $\mathbb E A$ characterizes the low-rank signal of the network model, it is possible that $\mathbb E A$  may have a negative eigenvalue, $\lambda_k<0$, with relatively large magnitude. In this case, selecting the top eigenvalues by their value rather than their magnitude  could lead to the exclusion of the eigenvector associated with $\lambda_k$. This could potentially affect the performance of K-means clustering.  I would like to see the authors address this potential issue in their methodology section and discuss its implications for their results.


3. I don't understand why the clustering cost function include $d_i$ in front of $\| \tilde F_{i,:} - w_r\|^2_2$. It seems more intuitive to consider simply sum of $\| \tilde F_{i,:} - w_r\|^2_2$, which aligns with  the K-means objective. Additional explanation of this choice would be helpful.

4. The paper states that the upper bound $\Psi$ depends on a quantity that indicates the weak clustering structure. However, this is not convincing to me,  especially using a relatively loose upper bound for $\Psi$ to explain this. While this upper bound includes the term $1- \lambda_{K+1}/\lambda_K$ that reflects the weak signal level, I am not sure it is reasonable to say $\Psi$ shows the weak clustering structure. From the definition of $\Psi$, it only depends on $1 - \lambda_{K+2}$, and I don't think we can infer that $1 - \lambda_{K+2}$ is comparable to $1- \lambda_{K+1}/\lambda_K$. 


5. In Lemma 4, why are  the eigenvectors compared to the columns of the orthogonal matrix $O$? shouldn't they be compared to the columns in $G$, as they are in Theorem 3?

6. In Theorem 8, the upper bound for $\mu(C_r\Delta \hat S_r)$ holds when 
$
\Psi \leq \mu_{\min}/[132(1+ \lambda_1)^2 \alpha K \mu_{\max} ].
$
The RHS is a very small constant when $\alpha >0$ is fixed.  The definition of $\Psi $ says that 
$
\Psi = \frac{1}{1 - \lambda_{K+2}} [ 1- \frac{d_{\min} }{d_{\max} + \tau} (1- \bar \phi_K(G)].
$
When there is very high degree heterogeneity such that $\frac{d_{\min} }{d_{\max} + \tau} (1- \bar \phi_K(G)= o(1)$, $\Psi \approx1/(1- \lambda_{K+2})$. Furthermore, $\lambda_{K+2}$ could  be quite small. For instance, for the laplacian of a DCBM, under mild conditions, $|\lambda_{K+2}|\leq C/ \sqrt{n\bar \theta^2} \approx 1/\sqrt{\bar d}$ where $\bar \theta$ is the average of the degree parameters. In this case, if $n\bar \theta^2 \to \infty$ or $\bar d\to \infty$, then $\lambda_{K+2} = o(1)$, and $\Psi \approx 1$.  Given these, I am concerned about whether the condition can be satisfied under high degree heterogeneity. I suggest providing a more thorough discussion of the practical implications of this condition, particularly in the context of high degree heterogeneity, along with specific examples where the condition can be met.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper provides novel analyses of the performance of degree-corrected spectral clustering.
Compared to the existing analyses, the advantage of this analysis is that it does not assume a specific random graph model.
To build on the analysis, this paper also proposes ASCENT, inspired by the recent over smoothing discussion of GNNs. 
This paper also empirically demonstrates the effectiveness of the proposed ASCENT.

### Strengths
-- The theoretical bound on the mis-clustering rate without assuming the random model graph.

-- The node-wise correction model of the DCSC and also theoretical results for the model

### Weaknesses
-- It is not clear how the bound is useful/non-trivial. How large the coefficient is? For instance, from the bound of Qin & Rohe, the spectral clustering on DCSBM is shown to be consistent, i.e., the bound is effectively smaller with respect to the size of the graph. Due to this, the bound of DCSBM is non-trivial. Can you say your bound is not trivial?

-- Since the $\Phi$ needs to be lower than \mu_{\min}/(132 (1+\lmabda_{1})^2 \alpha K \mu_{\max}), when this is satisfied? For a heterophilous graph, $\lambda_{1}$ tends to be smaller since the topological information for a heterophilous graph spans from smaller to larger frequencies. Thus, the analysis of heterogeneity carries this assumption. Yes, in this sense, while this paper does not assume the random graph, you still have some assumptions for the graph. Thus, you need to clarify what this assumption is.

In any case, since the analysis is still weak for the bound, I vote for rejection at this point.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
