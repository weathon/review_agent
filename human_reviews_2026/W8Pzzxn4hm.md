# Differentiable Cluster Discovery in Temporal Graphs

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
Existing temporal graph clustering methods suffer from poor optimization dynamics due to reliance on heuristically initialized cluster assignment distribution without considering the dynamic nature of the evolving graph. The target cluster assignment distribution often conflicts with evolving temporal representations, leading to oscillatory gradients and unstable convergence. Motivated by the need for differentiable and adaptive clustering in dynamic settings, we propose $\textbf{TGRAIL}$ ($\textbf{T}$emporal $\textbf{Gr}$aph $\textbf{A}$lignment and $\textbf{I}$ndex $\textbf{L}$earning), a novel framework for temporal graph clustering based on Gumbel–Softmax sampling. TGRAIL enables discrete cluster assignments while maintaining gradient flow. To ensure stable training, we formulate the clustering objective as an expectation over Monte Carlo samples and show that this estimator is both unbiased and variance-reduced. Furthermore, we incorporate a temporal consistency loss to preserve the order of interactions across time. Extensive experiments on six real-world temporal graph datasets demonstrate that our approach consistently outperforms state-of-the-art baselines, achieving higher clustering accuracy and robustness. Our results validate the effectiveness of jointly optimizing temporal dynamics and discrete cluster assignments in evolving graphs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes TGRAIL, a temporal graph clustering framework based on Gumbel-Softmax sampling that enables discrete cluster assignment while preserving differentiability. The method jointly learns node embeddings and cluster centroids through a Monte Carlo expectation of the clustering loss, and introduces a temporal consistency loss to encourage alignment across timestamps. The authors provide theoretical guarantees on gradient variance and convergence, and evaluate the approach on six temporal graph datasets, reporting improvements over several static and temporal baselines.

### Strengths
The paper tackles a relevant and challenging problem by enabling discrete cluster assignment in dynamic graphs through Gumbel-Softmax reparameterization. This allows joint learning of node embeddings and cluster centroids in a fully differentiable manner, addressing optimization instability in prior approaches. Theoretical analysis supports the proposed estimator with variance bounds and a convergence guarantee, and the method maintains linear scalability with respect to the number of temporal interactions. Empirical evaluation on several real-world datasets demonstrates consistent performance gains competitors.

### Weaknesses
While the paper provides a technically sound method for differentiable temporal clustering, several important aspects limit the strength of the empirical and methodological claims.  
1) The evaluation setup relies on static semantic node labels as ground-truth clusters, which does not necessarily reflect temporal community structure and risks turning the task into temporal node classification rather than clustering. 
2) The datasets used contain primarily static features, despite the problem formulation assuming time-dependent node attributes; this mismatch weakens the applicability of the method to settings where features genuinely evolve.  
3) Although the paper emphasizes “clustering coherence” and “temporal alignment,” these concepts remain informal and lack associated metrics or analyses to demonstrate that the model preserves meaningful dynamic community patterns.  
4) Baseline selection excludes classical dynamic community detection approaches from network science, where temporal clustering has been extensively studied, resulting in an incomplete view of comparative performance.   
5) The empirical analysis does not explore the evolution of clusters over time nor scalability beyond the provided datasets, leaving open whether the method meaningfully captures dynamic community behavior or maintains its claimed efficiency at larger scales.

### Questions
1) Since ground-truth labels are static and do not reflect temporal community evolution, how do the reported metrics validate the model’s ability to discover dynamically changing clusters rather than performing temporal node classification?


2) Most datasets used provide static or minimal node attributes, yet Problem 2.1 assumes time-dependent features. Could you clarify this inconsistency and specify the actual features used in each dataset (e.g., SCHOOL)? If features are missing, what initialization is applied?


3) Could you formally define “clustering coherence” and “temporal alignment,” and provide quantitative evaluations showing that TGRAIL achieves these properties in practice?


4) Can you justify the absence of dynamic community detection baselines from network science (e.g., evolutionary modularity, dynamic SBM, Louvain)? Which characteristics of TGRAIL differentiate it from such models?



5) The proposed method’s stability relies on temperature annealing and Monte-Carlo sampling. Could you include ablations or sensitivity analyses for these hyperparameters to demonstrate robustness?


6) Given the strong claims of linear scalability, would you consider adding synthetic experiments to empirically validate runtime and memory efficiency as the graph size and timestamps increase?


7) The dataset presentation lacks detail, making reproduction difficult. Could you enhance the appendix 11 with dataset feature descriptions, preprocessing steps, and cluster count selection rationale?


8) It would strengthen the claim of performing temporal clustering rather than temporal node classification to analyze how communities evolve under TGRAIL. For instance, visualizing cluster dynamics over time (e.g., community size changes, node transitions between clusters, formation or dissolution of communities) would help demonstrate that the method captures meaningful structural evolution rather than merely predicting static labels. Could you include such an analysis or provide evidence that TGRAIL discovers temporally coherent community trajectories?


9) There are instances where claims of interpretability are associated with graph clustering methods. Could you clarify what interpretability TGRAIL provides and how users would benefit from cluster assignments?


10) Could you increase the size of the labels/titles, etc, of figures?


11) Are there specific reasons for not releasing the official code? Making it publicly available would greatly improve the usability of the proposed method

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
The paper proposes TGRAIL, a differentiable framework for temporal graph clustering based on Gumbel–Softmax reparameterization, enabling end-to-end learning of discrete cluster assignments with temporal consistency. It replaces traditional t-distribution–based soft assignments with a Monte Carlo Gumbel estimator to stabilize gradients and align cluster evolution with dynamic node embeddings.

### Strengths
- The paper unifies discrete clustering and temporal embedding learning.
﻿
- Theoretical analysis is provided.
﻿
- Comprehensive experiments on multiple real-world temporal graph datasets demonstrate performance improvements.

### Weaknesses
- The proposed temporal-consistency loss builds on existing Hawkes-process–based similarity modeling but lacks detailed ablation to show its standalone contribution.
﻿
- The complexity analysis is largely theoretical; practical runtime and GPU utilization comparisons with baselines are not quantitatively reported.
﻿
- The framework depends on multiple hyperparameters , yet sensitivity studies are minimal.
﻿
- Some sections of the manuscript exhibit a very uniform writing style and phrasing patterns that may suggest the use of AI-assisted writing tools. 
﻿
- The baseline methods used for comparison are outdated. The authors must update the experimental section with stronger and more recent baselines to validate the claimed advantages.

### Questions
Please see the weaknesses.

### Soundness
2

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
2

### Summary
The paper introduces TGRAIL, a novel framework for clustering nodes in temporal graphs. Unlike traditional methods, which struggle to account for the evolving nature of graphs, TGRAIL employs Gumbel-Softmax sampling to allow for differentiable clustering that adapts to dynamic temporal representations. The paper demonstrates that this approach results in more stable training and improved clustering accuracy, with experiments showing superior performance across several real-world temporal graph datasets compared to existing methods.

### Strengths
1. TGRAIL introduces a differentiable clustering mechanism that can adapt to the temporal evolution of graphs, offering a clear improvement over static and post-hoc clustering methods
2. The authors conduct extensive experiments across six diverse real-world temporal graph datasets, demonstrating that TGRAIL consistently outperforms state-of-the-art methods, showing strong empirical evidence of its effectiveness.
3. The paper provides a rigorous theoretical analysis, including unbiased gradient estimation and convergence theorems, which ensures the soundness of the proposed approach and guarantees stable training dynamics.

### Weaknesses
1. The motivation section needs a clearer explanation; the significance of clustering techniques in temporal graphs does not seem obvious. What are the real-world application scenarios?
2. How was the number of clusters K chosen in this paper? If K is fixed, will it affect its adaptability in dynamic environments, thus contradicting the dynamic adaptability of the temporal graphs itself?
3. The font in the image is too small to read, as shown in Figure 4.

### Questions
How was the number of clusters K chosen in this paper? If K is fixed, will it affect its adaptability in dynamic environments, thus contradicting the dynamic adaptability of the temporal graphs itself?

### Soundness
3

### Presentation
4

### Contribution
3
