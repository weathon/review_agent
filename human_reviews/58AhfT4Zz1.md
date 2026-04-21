# Causal-aware Graph Neural Architecture Search under Distribution Shifts

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
Graph neural architecture search (Graph NAS) has emerged as a promising approach for autonomously designing graph neural network architectures by leveraging the correlations between graphs and architectures. However, the existing methods fail to generalize under distribution shifts that are ubiquitous in real-world graph scenarios, mainly because the graph-architecture correlations they exploit might be spurious and varying across distributions. In this paper, we propose to handle the distribution shifts in the graph architecture search process by discovering and exploiting the causal relationship between graphs and architectures to search for the optimal architectures that can generalize under distribution shifts. The problem remains unexplored with the following critical challenges: 1) how to discover the causal graph-architecture relationship that has stable predictive abilities across distributions, 2) how to handle distribution shifts with the discovered causal graph-architecture relationship to search the generalized graph architectures. To address these challenges, we propose a novel approach, Causal-aware Graph Neural Architecture Search (CARNAS), which is able to capture the causal graph-architecture relationship during the architecture search process and discover the generalized graph architecture under distribution shifts. Specifically, we propose Disentangled Causal Subgraph Identification to capture the causal subgraphs that have stable prediction abilities across distributions. Then, we propose Graph Embedding Intervention to intervene on causal subgraphs within the latent space, ensuring that these subgraphs encapsulate essential features for prediction while excluding non-causal elements. Additionally, we propose Invariant Architecture Customization to reinforce the causal invariant nature of the causal subgraphs, which are utilized to tailor generalized graph architectures. Extensive experiments on synthetic and real-world datasets demonstrate that our proposed CARNAS achieves advanced out-of-distribution generalization ability by discovering the causal relationship between graphs and architectures during the search process.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents CARNAS (Causal-aware Graph Neural Architecture Search), a novel approach that addresses the challenge of distribution shifts in Graph Neural Architecture Search. 
Unlike existing methods that rely on potentially spurious correlations between graphs and architectures, CARNAS focuses on discovering and leveraging causal relationships to achieve better generalization.

The solution consists of three main components:

1. Disentangled Causal Subgraph Identification: Discovers subgraphs with stable predictive capabilities across different distributions
2. Graph Embedding Intervention: Works in latent space to preserve essential predictive features while removing non-causal elements
3. Invariant Architecture Customization: Reinforces causal invariance and uses it to design generalized architectures

The approach's effectiveness is validated through experiments on both synthetic and real-world datasets, demonstrating superior out-of-distribution generalization compared to existing methods.

### Strengths
1. The application of neural architecture search to address graph out-of-distribution (OOD) problems represents a significant innovation compared to traditional fixed-backbone approaches, as demonstrated in Appendix G.3. This shift from static to adaptive architectures opens new possibilities for graph OOD generalization.
2. The method's effectiveness is convincingly demonstrated through comprehensive experiments on SPMotif and OGBG-Mol* datasets, where it consistently outperforms existing approaches in handling distribution shifts.
3. The paper stands out for its clear and organized presentation, featuring well-designed figures that effectively illustrate complex concepts, complemented by rigorous mathematical formulations that provide a solid theoretical foundation.

### Weaknesses
1. Computational Efficiency Concerns:  Section 3.3 reveals a potential limitation. The method searches through an extensive space of graph neural networks across all layers, which could be computationally more intensive than existing baselines. This increased computational and memory overhead might present challenges when applied to large-scale graphs.
2. Limited Experimental Validation: While DIR is included as a baseline, other important causal subgraph-based methods from recent works (such as [1],[2]) are not considered. 
Additionally, the evaluation datasets - SPMotif (synthetic) and OGBG-Mol* (relatively small graph sizes) - leave questions about scalability. 
It would be valuable to see performance on larger-scale datasets like DrugOOD or GOOD to address concerns about memory consumption and computational time.
3. Novelty Discussion:
While the neural architecture search component is innovative, the underlying methodology shares significant similarities with existing graph OOD approaches like DIR and related works [1-3]. The core mechanisms - using weighted top-k for causal subgraph identification and random combination for spurious subgraph intervention - closely parallel previous methods. This raises questions about the method's novelty beyond the architecture search component.
[1] Learning causally invariant representations for out-of-distribution generalization on graphs. 
[2]Learning invariant graph representations for out-of-distribution generalization.
[3] Improving subgraph recognition with variational graph information bottleneck

### Questions
1. Could you conduct an ablation study focusing on the neural network search details provided in the added Appendix C.1? 
Specifically, I'm interested in understanding:
- The effectiveness of different backbones to OOD distribution shifts, possibly illustrated through weight distributions.
- How about time and memory requirements during the search process?
- Are all of the backbones crucial for effective OOD distribution handling? 

2. How does your method behave in the large graph compared to the previous fixed-network methods since you may search in a large network space?  Will it be out of memory or the time computation will exponentially grow？

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The study proposes a novel method called Causal-aware Graph Neural Architecture Search (CARNAS) to address the challenges posed by distribution shifts in the process of Graph Neural Architecture Search (Graph NAS). Existing methods face limitations when handling distribution shifts in real-world scenarios, as the correlations they exploit between graphs and architectures are often spurious and subject to variation across different distributions. CARNAS aims to discover and leverage the causal relationship between graphs and architectures to search for optimal architectures capable of maintaining generalization under distribution shifts.

### Strengths
1. This paper innovatively proposes using NAS to address the problem of causal information identification in graph data.  
2. The paper conducts extensive experiments to validate the proposed method.

### Weaknesses
1. I believe the paper does not clearly explain why NAS can help adjust GNNs to identify causal information, which I consider the main issue of the paper. In my view, NAS optimizes the structure of GNNs, enhancing their efficiency or expressiveness, but it does not inherently enable GNNs to determine what type of data to model. At the very least, the authors did not provide a clear explanation of this point in the paper.  
2. The paper lacks theoretical justification for the regulatory capability of NAS.  
3. The survey and introduction of related work are insufficient.

### Questions
Could you explain how NAS can guide GNNs to model specific data causal relationships?

### Soundness
1

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
3

### Summary
This paper addresses the challenge of graph neural architecture search (Graph NAS) under distribution shifts. The authors observe that existing Graph NAS methods fail to generalize well when there are distribution shifts between training and testing data, since they may exploit spurious correlations that don't hold across distributions. To tackle this, they propose CARNAS (Causal-aware Graph Neural Architecture Search), a novel approach that discovers and leverages causal relationships between graphs and architectures. 

The main contributions of this work is 1) it is the first work to study Graph NAS under distribution shifts from a causal perspective; 2) they propose a novel framework with a disentangled Causal Subgraph Identification to find stable predictive subgraphs, a Graph Embedding Intervention component to validate causality in latent space and Invariant Architecture Customization to handle distribution shifts. Comprehensive experiments on synthetic and real-world datasets showing superior out-of-distribution generalization compared to existing methods.

### Strengths
1. The paper is the first to study Graph NAS under distribution shifts using causality. The problem is well-motivated with clear real-world relevance and applications. 
2. The authors present comprehensive experiments on both synthetic and real-world datasets that demonstrate clear performance improvements over existing baselines. The thorough ablation studies effectively validate each component of the proposed method, and the analysis provides valuable insights into the model's behavior.
3. The paper is well-structured. The experimental analysis is clearly presented, making the work reproducible.

### Weaknesses
1. While the paper shows good performance on the tested datasets, it lacks a detailed analysis of computational complexity and memory requirements. Specifically, the time complexity of $O(|E|(d_0 + d_1 + |O|d_s) + |V|(d_0^2 + d_1^2 + |O|d_s^2) + |O|^2d_1)$ could become prohibitive for very large graphs. The authors should discuss how their method performs on graphs with millions of nodes and edges, which are common in real-world applications like social networks.
2. The method requires careful tuning of four critical hyperparameters ($t$, $\mu$, $\theta_1$, $\theta_2$), which may significantly impact performance. In particular, the edge importance threshold t in Eq.(6) and the intervention intensity $\mu$ in Eq.(10) show high sensitivity in experiments. While the authors provide some sensitivity analysis on BACE dataset, they don't fully explain how to effectively tune these parameters for new datasets or application domains. 
3. The paper lacks formal theoretical guarantees for the causal discovery process. While the empirical results are strong, the authors should clarify under what conditions their method is guaranteed to identify true causal relationships and provide bounds on the probability of discovering spurious correlations. Additionally, the relationship between the intervention loss and causal invariance could be more rigorously established.

### Questions
1. In the graph embedding intervention module, you use $\mu$ to control intervention intensity in Eq.(10): $H_{v_j} = (1-μ)·H_c + μ·H_{s_j}$. Have you considered using adaptive intervention strategies where $\mu$ varies based on the structural properties of $G_c$ and $G_s$? This could potentially better handle graphs with varying degrees of spurious correlations.
2. The overall objective function (Eq.(17)) uses a linearly growing $\sigma_p$ corresponding to epoch number. Could you elaborate on why linear growth was chosen over other schedules (e.g., exponential, step-wise)? How does the schedule of $\sigma_p$ affect the trade-off between causal structure learning and architecture optimization?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel method, Causal-aware Graph Neural Architecture Search (CARNAS), to enhance the generalizability of Graph Neural Network (GNN) architectures under distribution shifts. By discovering stable causal relationships between graph structures and GNN architectures, CARNAS aims to mitigate issues with spurious correlations that often degrade performance across varying distributions. CARNAS introduces three core modules: Disentangled Causal Subgraph Identification, Graph Embedding Intervention, and Invariant Architecture Customization. Experimental results on both synthetic and real-world datasets show significant performance gains, especially in out-of-distribution generalization.

### Strengths
1. Well-structured modular approach: The CARNAS framework is thoughtfully organized, with each component clearly contributing to improved generalization under distribution shifts.

2. Robust experimentation: The paper includes extensive experiments across synthetic and real-world datasets, highlighting the robustness of the proposed method.

3. Component-level contribution clarity: Each module’s individual contribution is demonstrated, providing transparency and supporting the effectiveness of the approach.

### Weaknesses
1. Clarity in Section 3.3: Given I’m having limited familiarity with Graph NAS, the dynamic graph neural network architecture production and optimization process described in Section 3.3 remains somewhat unclear for me. A visual representation and a more detailed explanation would significantly improve the paper's readability.

2. Causal-Aware Solution's Justification: While the paper presents a causal-aware solution for handling distribution shifts, some aspects require stronger theoretical support to underscore the novelty and significance of the approach:

2.1. Limited Theoretical Support: The causal-aware Graph NAS solution leans heavily on implementation specifics, which limits the theoretical grounding of the method and may impact the perceived novelty.

2.2. Reliability of Causal Subgraph in Latent Space: The representation of causal subgraphs in latent space is an interesting approach; however, it is not entirely clear if the model reliably learns the true causal components or overfits to the training set to optimize the objective.

2.3. Overlap with Prior Work: Section 3.1 closely mirrors aspects of PGExplainer [26], which limits the novelty of this part of the approach.

### Questions
1. In Line 113, $N_{tr}$ and $N_{te}$ are not explicitly defined, though their meanings seem clear from the context. Please clarify these terms in the final manuscript.

2. How are the subgraphs in Eq. 6 represented (e.g., soft edge mask or hard crop)? Where and how are they used in subsequent steps?

3. To enhance the clarity of your approach, it would be helpful to visualize the causal and non-causal subgraphs for each dataset used in the case studies.

### Soundness
2

### Presentation
3

### Contribution
2
