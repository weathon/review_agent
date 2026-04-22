# On the Universality and Complexity of GNN for Solving Second-order Cone Programs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Graph Neural Networks (GNNs) have demonstrated both empirical efficiency and universal expressivity for solving constrained optimization problems such as linear and quadratic programming. However, extending this paradigm to more general convex problems with universality guarantees, particularly Second-Order Cone Programs (SOCPs), remains largely unexplored.
We address this challenge by proposing a novel graph representation that captures the inherent structure of conic constraints. We then establish a key universality theorem: *there exist GNNs that can provably approximate essential SOCP properties, including instance feasibility and optimal solutions*. We further derive the sample complexity for GNN generalization based on Rademacher complexity, filling an important gap for Weisfeiler-Lehman-based GNNs in learning-to-optimize paradigms.
Our results provide a rigorous foundation linking GNN expressivity and generalization power to conic optimization structure, opening new avenues for scalable, data-driven SOCP solvers. The approach extends naturally to $p$-order cone programming for any $p \geq 1$ while preserving universal expressivity and requiring no structural modifications to the GNN architecture. Numerical experiments on randomly generated SOCPs and real-world power grid problems demonstrate the effectiveness of our approach, achieving superior prediction accuracy with significantly fewer parameters than fully connected neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed a way to encode SOCP problems with graphs, which is an important extension of LP, QP, QCQP problems. The authors designed a message passing scheme on top. Besides, the separation power of WL on such graphs as well as generalization bounds are established. The empirical results also exhibited the sigfinicance of the work.

### Strengths
- The problem is well motivated. SOCP is an important extension of LP, QCQP, and this work is an important milestone towards more general convex cones. 
- The paper is well written and easy to read. 
- The design of the graph encoding for SOCP problems make a lot of sense, and faithfully encode all the information of an SOCP problem.
- The theoretical results are well established.

### Weaknesses
- I think the separation power of WL on SOCP is also an important point and should be mentioned in the main paper as well. 
- The notations of graph nodes are a bit confusing, they are inconsistent in section 4.1 and figure 2. 
- I understand there might be lack of baselines, but FCNN baseline is a bit too trivial.

### Questions
- What are the hardwares for comparing solving time of SOCP-GNN and MOSEK solver? It seems that the GNN has constant solving time with the growth of the instance size, so I guess it is on GPU. Did you also run MOSEK on GPU? If not that is not fare comparison.
- It is mentioned in the appendix that extending the work to more general $p$ cone only requires encoding $p$ parameter in the graph. It is plausible in the perspective of WL test, but does it hurt the generalization performance? For example, will there be such a family of problems, where different $p$ leads to the same solution?

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
The author proposed a novel graph representation for SOCPs, that is a major class of convex optimization problems. Based on this representation, the author proposed a GNN structure called SOCP-GNN, that is simple enough to be effective. Additionally, the author analyzed the representation power and generalization capability of SOCP-GNN. Finally, the author provided the results a series of experiments to validate the proposed framework.

### Strengths
- The new graph representation extends to SOCP problems, which is more general than previously considered LP, QP, and QCQPs. And on the representation, the proposed structure is simple enough, while having enough representation power to express the inherent structure of SOCP instances.
- The author provides a framework for analyzing the generalization capability of SOCP-GNN or other structures with representation power guarantee established by WL-test and relevant tool(i.e. Lusin's theorem and Generalized Stone-Weierstrass Theorem).

### Weaknesses
- The previous representation on QCQPs **benefits from sparsity** to get better MP complexity and the SOCP-GNN(when applied on equivalent reformulated QCQPs) benefits from **the constraints being low-rank**. The author claimed that the proposed structure is less complex when used on equivalent reformulated convex QCQPs, but there is no argument about relation between sparsity of the original QCQP instance and the complexity of the reformulated instance.
- SOCPs do not cover **general** QCQPs, since only convex QCQPs can be directly reformulated to equivalent SOCPs, and SOCPs reformulated to special non-convex QCQPs (since the special combination of non-convex quadratic constraints and linear constraint makes it **implicitly convex** to be solved in polynomial time). So there is not too much improvement with respect to general non-convex QCQPs.

### Questions
- In the comparison of node complexity and MP complexity, the ranks of the cone constraints are denoted by $k_{i} $instead of $r_{i}$. $r_{i}$ is used to denote the rank of quadratic constraints, but here we are not discussing quadratic constraints.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The present manuscript addresses the solution of Second-Order Cone Programs (SOCPs) by means of Graph Neural Networks, specifically designed to process the variables-constraints graph arising from such problems, with a lightweight implementation that is competitive in terms of computational complexity.
The authors provide universal approximation results and generalization bounds for the proposed model, and evaluate their model against  traditional SOCP solvers.

### Strengths
The literature review is accurate and broad, and the related work is addressed (almost) properly. 
The theoretical results are meaningful and derived with deep rigorousness (the Appendix has been checked). 
All the needed mathematical concepts are clearly introduced and explained, and intuitive readings of the theoretical results are provided. 
The experimental evaluation on synthetic data is well set and well conducted.

### Weaknesses
Major concerns:
- The literature review on VC dimension for GNNs is not complete, please check [1];
- at line 306, a Lipschitz assumption on GNN is stated; although, as noted by the authors, this is frequent in theoretical analysis of generalization capabilities of GNNs, I don't think that such property has been properly addressed when switching to the experimental framework;
- a similar concern arises about connecting the generalization bound with the experiments, i.e.  there are not connections between Theorem 2 and the experiments in Section 7;
- lastly, even if this paper is a proof of concept, the authors propose a benchmarking over AC-OPF solvers; in this case, I think that a comparison with actual deep learning OPF solvers is needed.

Minor concerns:
- line 104, "specially" -> "specifically"
- line 128, "representations" -> "representation"
- line 1845-1847: "Lipshitz" -> "Lipschitz"

[1] D’Inverno, G. A., Bianchini, M., & Scarselli, F. (2025). VC dimension of Graph Neural Networks with Pfaffian activation functions. Neural Networks, 182, 106924.

### Questions
- I would suggest the authors to revise the literature over generalization results for GNNs, as suggested above;
- I would suggest the authors make explicit connections between the experimental validation and the theoretical results;
- Lastly, I would strongly encourage the authors to insert a (possibly) fair comparison with deep learning-based OPF solvers.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This ICLR submission proposes a graph representation and Graph Neural Network (GNN) architecture, termed SOCP-GNN, for solving Second-Order Cone Programs (SOCPs). The graph representation has variable, polyhedral, minor conic, and major conic nodes, enabling message-passing GNNs to predict key properties like feasibility and optimal solutions. They prove universal approximation capabilities using Weisfeiler-Lehman-based expressivity and derive sample complexity bounds via Rademacher complexity for generalization. The approach generalizes to p-order cone programs and demonstrates empirical superiority on synthetic SOCPs and real-world power grid problems, using fewer parameters than fully connected networks.

### Strengths
1. The paper is in general well written, and the main results are easy to follow.

2. The paper provides the generalization analysis beyond WL-test-based GNN analysis, deriving sample complexity bounds via Rademacher complexity.

### Weaknesses
The WL-test-based universal approximation analysis is not completely new; it uses ideas from (Chen et al., 2022b), as the authors explicitly noted.

### Questions
Is it possible to add quadratic terms in the objective function, as in (Wu et al., 2024; Chen et al., 2024b)?

### Soundness
3

### Presentation
3

### Contribution
3
