# A Spectral Characterization of Generalization in GCN: Escaping the Curse of Dimensionality

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Empirically it is observed that Graph Convolution Networks (GCNs) often generalize better than 
fully connected neural networks (FCNNs) on graph-structured data. While this observation is often attributed to the ability of GCNs to exploit knowledge about the underlying graph structure, a rigorous theoretical explanation remains limited. In this work, we theoretically prove that one factor for the improved generalization of GCNs arises from the spectral representation of the filters or graph convolutional layers. Specifically, we derive generalization bounds that are independent of the number of parameters and instead scale nearly linearly with the number of graph nodes, offering a compelling explanation for their superior performance in over-parameterized regimes. Furthermore, in the limit of infinite number of nodes, we prove that under certain regularity conditions on the spectrum, GCNs escape the curse of dimensionality and continue to generalize well. We demonstrate our conclusions through numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a new theoretical framework for analyzing the generalization properties of GCNs by leveraging classical tools from signal processing and modern statistical learning techniques. The approach differs from prior work in that it leverages the graph spectral structure rather than relying purely on the parameter space. The authors argue that GCN filters, viewed in the graph Fourier domain, have lower intrinsic dimensionality, allowing the derivation of tighter generalization error bounds that are independent of the number of parameters. The paper further extends the analysis to graphons. Numerical experiments validate the scaling of the theoretical bounds.

### Strengths
1. Sharp generalization bounds.
2. The paper is generally well-written and clearly motivated.
3. The paper provides some insights in Sec. 4.
4. This seems to be the first theoretical results that analyze generalization properties in the spectral domain. The spectral viewpoint provides an elegant and under-explored angle for bounding generalization in GCNs.
5. Tighter bounds: The derived bounds scale as $\sqrt{n_x/N},$ independent of the total number of parameters, and improved over VC-dimension/PAC-Bayes/Rademacher complexity bounds.

### Weaknesses
1. The paper claims sharp bounds but has not shown the actual gap between empirical and theoretical error bounds.
2. The sub-Gaussian assumption may not hold for real-world graph data.
3. The bounds involve constants such as $K, K'', L_\mathcal{X},$ and $L_\mathcal{H},$ which can benefit from better explanation and intuition of why they are defined in this way.
4. Some typos/grammar issues, e.g., in lines 135-136 it should be "an empirical"; in lines 293-294 "While" should be removed.

### Questions
1. How tight are your theoretical bounds to empirical values?
2. How sensitive are your results to the Lipschitz or sub-Gaussian assumptions? Would small violations lead to large deviations in generalization behavior?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a theoretical study of generalization in Graph Convolutional Networks (GCNs) from a spectral perspective. It argues that the spectral representation of graph filters provides a natural space to analyze generalization, yielding bounds independent of the number of parameters. The authors derive sharp non-asymptotic generalization bounds that scale nearly linearly with the number of graph nodes and remain finite in the infinite-node (graphon) limit under mild regularity assumptions. Theoretical results are complemented by numerical simulations on both homophilous and heterophilous datasets, verifying the dependence of generalization on Lipschitz constants and spectral properties.

### Strengths
1. The spectral characterization of GCN generalization is interesting and deserves in-depth theoretical analysis;
2. The paper includes comparisons with classical frameworks (VC dimension, PAC-Bayes, Rademacher complexity) and state-of-the-art works, highlighting the improvement in sample complexity.

### Weaknesses
1. The readability is not high, as the notation is complex and difficult to follow.

2. The link between the simulation studies and the theoretical insights is not clearly established.

3. If the authors focus on the graph classification setup, what about the case of node classification?

4. The current theoretical result does not consider the over-smoothing issue, why?

### Questions
1. How sensitive are the derived bounds to violations of the sub-Gaussian assumption?

2. Can the spectral regularity (Lipschitz or low-pass) conditions be empirically estimated from real GCNs during training?

3. Would incorporating stochasticity in the graph structure (e.g., random graph models) affect the generalization scaling behavior?

4. Could the theoretical insights guide the design of regularization terms or spectral constraints to improve practical generalization?

### Soundness
2

### Presentation
2

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
This paper provides new generalization bounds for GCNs. The bounds are derived by evaluating the covering number of the hypothesis space in the Fourier domain of convolution operators. Specifically, when the spectrum is bounded, the bound does not depend on the number of trainable parameters. When the spectrum decays rapidly, the bound is independent of the number of nodes on a graphon, which can be interpreted as a graph with an infinite number of nodes.

### Strengths
1. (Originality) To the best of my knowledge, this paper is the first to derive a generalization performance bound for GCNs utilizing spectral decay.
2. (Quality) It provides a thorough review of previous statistical learning theory research on GCNs, clearly positioning this work within this line of research.
3. (Clarity) The writing is clear. The paper's structure is appropriate, and the mathematical descriptions are accurate. I had no difficulties in understanding the paper's main claims.
4. (Significance) For single-layer GCNs, the proposed method's performance bound achieves a better order with respect to node size than existing methods.

### Weaknesses
1. The derivation of bounds employs evaluation using the covering number, which is a relatively classical statistical learning theory method. Therefore, its novelty from this perspective is limited.
2. There is a discrepancy in the problem setting between the theoretical analysis and the numerical experiments. The theoretical analysis considers a problem setting where the graph signal and teacher signal are given in the I.I.D. setting. On the other hand, the numerical experiments consider a transductive node classification problem.

### Questions
I would like the authors to address the concerns raised in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the generalization properties of Graph Convolutional Networks (GCNs) and attributes their empirical superiority over fully connected neural networks (FCNNs) to the spectral representation of graph convolution filters. The authors derive generalization bounds $O(\sqrt{\frac{n_x}{N}})$ that are independent of the number of parameters. They show that under mild spectral regularity conditions, GCNs can escape the curse of dimensionality. However, its immediate practical and theoretical claims are significantly weakened by several critical methodological flaws.

### Strengths
(1) The core idea that GCN generalization is determined by spectral complexity (covering numbers) rather than the parameter count is novel and addresses an open problem in GNN theory.

(2) If the results were fully valid, they would provide a rigorous theoretical explanation for a widely observed phenomenon: GCNs generalizing better than FCNNs on graph-structured data.

### Weaknesses
(1) Unrealistic Assumptions: The author assumes sub-Gaussian for graph signals and convex, smooth loss. In practice, graph node features are often sparse, categorical, or heavy-tailed. They are **not sub-Gaussian**. Classification losses (e.g., cross-entropy in **multi-class classification**) are **non-convex**. This limits the applicability of the theoretical bounds. 

(2) Single-layer GCN analysis: All theoretical results in section 3.3 are derived for one layer $L=1$, whereas practical GCNs usually have 2 layers. **Multi-layer extensions are nontrivial**, and the bounds may degrade significantly due to Lipschitz composition and over-smoothing effects.

(3) Mismatch between theory and experiments: Experiments are conducted on ChebNet rather than standard GCNs, introducing a gap between the theoretical model and empirical validation. The **polynomial order $K$** in ChebNet affects receptive fields and generalization, which is not addressed.

(4) Negative generalization error in plots: Left one in Figure 4(a) shows **negative generalization errors** despite defining GE as absolute value in Eq. 4, suggesting either an inconsistency in the plot or a mismatch between theory and implementation.

### Questions
(1) How sensitive are your bounds to the assumption of sub-Gaussian node features, and can they be relaxed for sparse or categorical inputs?

(2) Can your $L=1$ layer analysis be generalized to multi-layer GCNs, and if so, how does the bound scale with $L$?

(3) Why were ChebNet models used in experiments instead of the theoretical GCNs, and how does the polynomial order $K$ affect the generalization?

(4) Explain the negative generalization error in left figure in Figure 4(a) when generalization error is defined as the absolute value in Eq. 4?

### Soundness
2

### Presentation
2

### Contribution
2
