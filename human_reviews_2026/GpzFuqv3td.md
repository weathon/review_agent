# Adverseness vs. Equilibrium: Exploring Graph Adversarial Resilience through Dynamic Equilibrium

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Adversarial attacks to graph analytics are gaining increased attention. To date, two lines of countermeasures have been proposed to resist various graph adversarial attacks from the perspectives of either graph per se or graph neural networks. Nevertheless, a fundamental question lies in whether there exists an intrinsic adversarial resilience state within a graph regime and how to find out such a critical state if exists. This paper contributes to tackle the above research questions from three unique perspectives: i) we regard the process of adversarial learning on graph as a complex multi-object dynamic system, and model the behavior of adversarial attack; ii) we propose a generalized theoretical framework to show the existence of critical adversarial resilience state; and iii) we develop a condensed one-dimensional function to capture the dynamic variation of graph regime under perturbations, and pinpoint the critical state through solving the equilibrium point of dynamic system. Multi-facet experiments are conducted to show that the proposed approach can significantly outperform the state-of-the-art defense methods under five commonly-used real-world datasets and several representative attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an approach to graph adversarial defense by drawing inspiration from the theory of complex dynamical systems. The authors propose the EquiliRes framework, which uses concepts like Lyapunov stability and Laplace transforms to prove the existence of this equilibrium. The framework models the graph’s dynamics with a condensed one-dimensional function based on degree centrality and iteratively modifies the adjacency matrix to move the graph toward the equilibrium point, thereby “purifying” it from adversarial perturbations.

### Strengths
1.The paper’s approach is highly original, combining concepts from dynamical systems and graph adversarial learning in a novel way. This cross-disciplinary perspective could potentially open up new research directions.

2.The paper tackles a bold, high-level idea with the potential to transform the way we think about graph adversarial defense. The theoretical framework, if fully validated, could offer a profound contribution.

### Weaknesses
1.The connection between the theoretical framework (Lyapunov stability and Laplace transforms) and the concrete implementation based on degree centrality is not adequately justified. This “leap of faith” leaves a critical gap in understanding why such a simplification is both valid and effective for adversarial defense in graphs.

2.The empirical comparison is insufficient, primarily involving older and simpler baselines like GCN and GCN-SVD. To demonstrate the true effectiveness of the method, comparisons to more recent and robust defense methods would be necessary.

### Questions
1.Could the authors provide a clearer, step-by-step explanation of how this simplification is both valid and sufficient to capture the dynamics of graph adversarial perturbations?

2.Why were more recent, state-of-the-art defense methods such as Pro-GNN not included in the comparisons? How does EquiliRes perform in direct comparison to these methods?

3.The method focuses on purifying graph topology. Does the theoretical framework provide any insight into defending against perturbations to node features as well?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel theoretical and algorithmic framework, EquiliRes, for understanding and improving graph adversarial resilience through the concept of dynamic equilibrium. The authors model adversarial perturbations on graphs as a complex dynamic system and establish the existence of an asymptotically stable equilibrium point that represents the graph’s critical adversarial resilience state. Building upon this theoretical foundation, the paper introduces a one-dimensional perturbation mapping and an iterative procedure to construct an attack-resilient topology. Experimental results on five common benchmark datasets demonstrate promising robustness improvements compared to several baseline defenses.

### Strengths
* The paper presents an interesting theoretical perspective that connects adversarial graph learning with complex dynamic systems.

* The equilibrium-based framework offers a potentially generalizable and interpretable way to reason about adversarial robustness.

* Experimental results indicate consistent performance gains across different attacks and datasets, suggesting the practical potential of the proposed approach.

### Weaknesses
* Insufficient baselines:
The comparison is incomplete. Key recent robust GNN methods, such as RUNG [1], are missing from the evaluation. Including these would make the empirical validation more convincing.

[1] Hou, Z., Feng, R., Derr, T., & Liu, X. (2023). Robust Graph Neural Networks via Unbiased Aggregation

* Weak experimental completeness:
The experiments are not comprehensive. For instance, in Table 2 (GOttack), only one baseline (GCN-SVD) is reported, which is far from sufficient for a fair comparison. Similarly, ablation and parameter sensitivity analyses could be expanded to strengthen the empirical section.

* Poor writing quality:
The overall writing, particularly in the experimental sections, is difficult to follow. Descriptions of settings, metrics, and observations are often unclear and overly verbose. The readability and logical flow of the experimental analysis need major improvement.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper asks if a graph itself possess an intrinsic, critical adversarial-resilience state, and if so, how do we find it and use it to build a more attack-resilient topology? It treats adversarial learning on graphs as a multi-object dynamic system. Under bounded, continuous perturbations, the graph’s state can converge to an asymptotically stable equilibrium point (ASEP), or the graph’s critical resilience state.

### Strengths
1. The paper introduces a complex multi-object dynamic system perspective to study adversarial robustness of graph neural networks.

2. The paper obtains an asymptotically stable equilibrium point.

### Weaknesses
1. The intrinsic adversarial-resilience state exists only under bounded, smooth perturbations and the paper’s modelling assumptions, which can be a significant limitation for general graph structures. 

2. The use of the analogy in Fig. 1 is helpful to understand the motivation of the method. However, using the analogy to guide the method design somehow oversimplifies the problem space, e.g., the node/edge states in a graph may have the correlations or dependencies that cannot be captured by the three states in the paper.

3. Decomposing the adversarial perturbation into linear and non-linear part shares similarity to gradient-based perturbation (Wu et al., 2019). How does the decomposition differ to the graph per se methods built on gradient-based perturbation patterns? 

4. The method deals with topology only and node features are not handled, therefore strong defence like pro-gnn is not compared.

### Questions
see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
