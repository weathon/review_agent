# Generalized Sobolev IPM for Graph-Based Measures

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
We study the Sobolev IPM problem for measures supported on a graph metric space, where critic function is constrained to lie within the unit ball defined by Sobolev norm. While Le et al. (2025) achieved scalable computation by relating Sobolev norm to weighted $L^p$-norm, the resulting framework remains intrinsically bound to $L^p$ geometric structure, limiting its ability to incorporate alternative structural priors beyond the $L^p$ geometry paradigm. To overcome this limitation, we propose to generalize Sobolev IPM through the lens of *Orlicz geometric structure*, which employs convex functions to capture nuanced geometric relationships, building upon recent advances in optimal transport theory---particularly Orlicz-Wasserstein (OW) and generalized Sobolev transport---that have proven instrumental in advancing machine learning methodologies. This generalization encompasses classical Sobolev IPM as a special case while accommodating diverse geometric priors beyond traditional $L^p$ structure. It however brings up significant computational hurdles that compound those already inherent in Sobolev IPM. To address these challenges, we establish a novel theoretical connection between Orlicz-Sobolev norm and Musielak norm which facilitates a novel regularization for the generalized Sobolev IPM (GSI). By further exploiting the underlying graph structure, we show that GSI with Musielak regularization (GSI-M) reduces to a simple *univariate optimization* problem, achieving remarkably computational efficiency. Empirically, GSI-M is several-order faster than the popular OW in computation, and demonstrates its practical advantages in comparing probability measures on a given graph for document classification and several tasks in topological data analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper aims to generalize the Sobolev IPM [1] using the idea of an Orlicz geometric structure, which has previously been used to extend the Wasserstein distance on L_p spaces to the Orlicz–Wasserstein (OW) distance. By assuming an underlying graph structure, the computation of the proposed generalized Sobolev IPM reduces to a simple univariate optimization problem, which is significantly faster than OW.

### Strengths
The main computational advantage of the proposed method lies in the assumption that the compared probability measures are supported on the same graph. Under this assumption, the computation of the optimal transport plan, required in Wasserstein or OW distances, can be avoided, similar to the case where a closed-form solution exists for the Wasserstein distance in the one-dimensional setting. This advantage has also been extensively exploited in several prior works [1, 2, 3].

### Weaknesses
- At first glance, I thought this paper extends [1], in the sense that the Sobolev IPM in [1] is defined on an L_p geometric structure, while the proposed method generalizes it by replacing the L_p structure with an Orlicz geometric one. However, I found that the method called generalized Sobolev transport (GST, [2]) also employs an Orlicz geometric structure to generalize L_p. Moreover, my understanding is that the proposed method is essentially a weighted version of [2], that is, it introduces weights into the definition of the Orlicz–Sobolev space, resulting in what the authors call the generalized Sobolev IPM with Musielak regularization (GSI-M) in Eq. (11). I cannot see why [1] is cited in the abstract but not [2], since [2] appears to be more closely related. 
- regarding the weighting functions, [2] fixes the weights as w(x) = 1+\lambda(\Lmabda(x)), while in this paper, the weights are defined as w(x) = 1+\lambda(\Lmabda(x))/\lambda(G). A question arises: is it possible to use other weight functions, such as user-defined functions. Furthermore, as the proposed method is a weighted variant of GST [2], the paper should include a more careful discussion on the choice and impact of the weighting functions.
- In the experiments, the authors compare OW (without graph structure), GST [2], and the proposed GSI. It appears that GST and GSI achieve comparable performance in terms of both accuracy and running time. In other words, as mentioned above, although the proposed GSI is the weighted version of GST [2], the paper fails to clearly explain what benefits the weighting scheme of GSI .
- Overall, the paper is not written in a way that is accessible to those who are not familiar with this research line. The proposed method seems to be a weighted variant of [2], but the motivation and practical advantages of introducing this specific weighting scheme remain unclear.

[1] Tam et al., Scalable Sobolev IPM for Probability Measures on a Graph, ICML2025. 

[2] Tam et al., Generalized Sobolev Transport for Probability Measures on a Graph, ICML2024 

[3] Tam et al., Sobolev transport: A scalable metric for probability measures with graph metric, AISTATS2022

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Generalized Sobolev IPM (GSI) with Musielak regularization for measuring distances between probability measures on graph metric spaces. Building on Sobolev Transport (ST) and Generalized Sobolev Transport (GST), the authors develop theoretical connections between GSI and other transport distances including Orlicz-Wasserstein. They prove metric properties (Theorem 3.4) and establish computational efficiency improvements. The paper demonstrates applications in document classification and TDA, showing competitive performance with faster computation times compared to existing methods.

### Strengths
**Theoretical innovation and significance:** The paper makes a significant contribution by generalizing Sobolev IPM through Orlicz geometry, creating meaningful connections between integral probability metrics and transport distances on graphs. The rigorous proofs establishing relationships between GSI-M and GST (Proposition 4.6: $\frac{1}{2} \text{GST}\_{\Phi}(\mu,\nu) \leq \hat{GS}\_{\Phi}(\mu,\nu) \leq \text{GST}_{\Phi}(\mu,\nu)$) provide valuable theoretical insights that advance beyond prior work (Le et al., 2022, 2024).

**Clarity and practical relevance:** The presentation is exceptionally clear, with well-structured theoretical development that makes complex concepts accessible. The discrete case formulation (Theorem 3.5) provides practical computational methods, and the paper effectively demonstrates applications in document classification and TDA where computational efficiency matters, addressing real-world limitations of existing approaches.

### Weaknesses
**Limited experimental validation with key baselines:** While the paper cites Fused Gromov-Wasserstein (FGW) and Fused Partial Gromov-Wasserstein (FPGW) (Bai et al., 2025; Brogat-Motte et al., 2022), it lacks direct comparisons with these methods. Given that FGW has become a standard for structured object matching, including these comparisons would significantly strengthen the empirical validation and better position GSI-M within the broader landscape of graph-based distance metrics.

**Narrow experimental scope:** The evaluation focuses primarily on document classification (Orbit) and TDA (MPEG7) datasets but misses opportunities to test on more diverse graph-structured problems. Additional experiments on graph matching tasks or node classification would better demonstrate the versatility of GSI-M across different application domains where graph structure plays a critical role.

**Insufficient parameter sensitivity analysis:** While the paper demonstrates computational advantages, a more thorough ablation study on how the choice of N-function $\Phi$ affects practical performance would help practitioners understand when and why to choose specific configurations. The current experiments don't fully explore how different Φ functions impact results across varying data characteristics.

### Questions
1. In Proposition 4.8, you establish $1/2 \text{OW}(\mu,\nu) \leq \hat{GS}_\Phi(\mu,\nu) \leq \text{OW}(\mu,\nu)$ for tree graphs. How does this bound degrade for graphs with cycles, and is there a tight bound based on graph properties like treewidth?

2. Your Equation (14) shows the discrete case formulation. How sensitive is the computational performance to the sparsity pattern of $\gamma_e$, and have you observed cases where non-standard Φ functions provide meaningful advantages over standard Sobolev IPM?

### Soundness
3

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
This paper proposes a generalization of the Sobolev Integral Probability Metric (IPM) for measures on graphs. They extend the standard $L^p$ geometric structure based solution to Orlicz geometric structure using N-functions, termed as Generalized Sobolev IPM (GSI)
Based on an equivalence between Orlicz-Sobolev norm and a weighted Musielak norm, the paper then introduces an efficiently-computable GSI with Musielak regularization.

### Strengths
- The extension of Sobolev IPMs from $L^p$ spaces to Orlicz spaces is reasonable.
- The authoors address the computational tractability of this metric. 
- The paper shows that the proposed GSI-M is a metric and is equivalent to GSI.
- The empirical results show that GSI-M is computationally more efficient than the related Orlicz-Wasserstein (OW) distance.

### Weaknesses
- The novelty of the core technical insight appears largely incremental and derivative. The paper leverages the exact same weight function $\hat{w}(x)$ that was a key finding in Le et al. (2025) to relate the norms. This makes the paper feel like a direct substitution of $L^p$ norms with Orlicz/Musielak norms onto the framework of Le et al. (2025). 
- The practical motivation for the generalization is weak. The experiments do not demonstrate a compelling advantage for using the more complex Orlicz functions ($\Phi_1, \Phi_2$) over the limit case ($\Phi_0$). The limit case $\Phi_0$ reduces to the 1-order ST (which has a closed-form) and performs comparably, suggesting the added complexity of the Orlicz structure offers marginal practical benefit for these tasks.
- Section 4, which lists connections to other metrics (ST, GST, OW, OT), reads like an appendix. It provides a long list of propositions without sufficient intuition or discussion of their implications or motivation.
- The paper provides very little background on Sobolev IPMs or their limitations (around Eq. 5), assuming significant prior knowledge. This makes it difficult for the broader community to appreciate the starting point and the motivation for the proposition.
- The paper is not written clearly, and motivation and intuition seems lacking; the thought processes are hard to follow.

### Questions
- Could you explicitly clarify the technical novelty compared to Le et al. (2025)? Specifically, is the core contribution the proof that the norm equivalence (Thm 3.2) also holds for Orlicz/Musielak spaces using the same weight function, or is there a more fundamental difference? Would it be easier to reuse prior art in this line of research rather than potentially restating similar results?

- The empirical benefit of the Orlicz generalization (using $\Phi_1, \Phi_2$) over the $L^1$-like $\Phi_0$ case seems minimal. Can you provide evidence of (or hypothesize about) specific applications, graph types, or N-functions where this generalization would provide a clear and significant practical advantage?

- To improve readability, you might consider moving Sec. 4 to the appendix, keeping only the major Theorems. The saved space could then be used to to motivate the problem for a broader audience.

### Soundness
3

### Presentation
2

### Contribution
2
