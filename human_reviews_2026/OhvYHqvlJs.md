# Learning Fair Graph Representations with Multi-view Information Bottleneck

- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Graph neural networks (GNNs) excel on relational data by passing messages over node features and structure, but they can amplify training data biases, propagating discriminatory attributes and structural imbalances into unfair outcomes. Many fairness methods treat bias as a single source, ignoring distinct attribute and structure effects and leading to suboptimal fairness and utility trade off.
To overcome this challenge, we propose FairMIB, a multi-view information bottleneck framework designed to decompose graphs into feature, structural, and diffusion views for mitigating complexity biases in GNNs. Especially, the proposed FairMIB employs contrastive learning to maximize cross-view mutual information for bias-free representation learning. It further integrates multi-perspective conditional information bottleneck objectives to balance task utility and fairness by minimizing mutual information with sensitive attributes. Additionally, FairMIB introduces an inverse probability-weighted (IPW) adjacency correction in the diffusion view, which reduces the spread of bias propagation during message passing. Experiments on five real-world benchmark datasets demonstrate that FairMIB achieves state-of-the-art performance across both utility and fairness metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces FairMIB, a novel multi-view information bottleneck (MIB) framework for learning fair graph representations in Graph Neural Networks (GNNs). It addresses biases amplified by GNNs in relational data, which stem from node attributes, graph structure, and message-passing propagation. FairMIB decomposes the graph into three views: a diffusion view, a feature view, and a structural view. It employs contrastive learning to maximize cross-view mutual information for robust, bias-free representations, while integrating a multi-view conditional information bottleneck (MCIB) to balance utility and fairness. The framework optimizes a loss that combines compression, fair prediction, consistency constraints, and cross-entropy for node classification. Experiments on five real-world datasets show superior performance on fairness and utility metrics.

### Strengths
1. The paper decomposes the sources of bias in graphs into three complementary views feature, structure, and diffusion. It introduces a multi-view conditional information bottleneck (MCIB) framework that constrains the learned representations to preserve task-relevant information while being disentangled from sensitive attributes. In addition, a cross-view contrastive consistency mechanism is incorporated to align information across views. Together, these components form an end-to-end fair representation learning framework that brings conceptual novelty and achieves strong performance in fair graph neural networks.

2. In the diffusion view, causal intervention is applied to mechanistically suppress the amplification of sensitive attributes during message passing.

3. The proposed framework’s multi-view disentanglement and information-theoretic design are both innovative and well-motivated.

4. Experiments on five real-world datasets provide strong evidence for the effectiveness of the proposed method. The approach is compared with SOTA baselines from different categories, and the results are highly convincing. Comprehensive ablation studies are conducted, covering different views and modules.

### Weaknesses
1. The details of IPW estimation and stabilization are insufficient. Although the paper clearly states the use of IPW and provides the weighting formula, it does not specify the estimation model for the propensity score $e(i)$, nor clarify key implementation aspects such as whether it is estimated using only non-sensitive features.

2. The framework introduces additional modules, which inevitably increase time complexity compared with standard GNNs or simpler fairness methods. It would be beneficial if the paper could briefly discuss this aspect.

### Questions
1. What type of model is used to estimate the propensity score $e(i)$? Is it estimated using only non-sensitive features?

2. How does the method perform when $S$ is partially missing or noisy?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FairMIB, a multi-view information bottleneck framework for learning fair graph neural network representations. The approach decomposes graphs into three distinct views to address biases from different sources. The framework employs contrastive learning to maximize cross-view mutual information while using a conditional information bottleneck to balance task utility and fairness. It further introduces inverse probability weighting in the diffusion view to reduce bias propagation during message passing. Experiments on five real-world datasets demonstrate state-of-the-art performance in both utility and fairness metrics.

### Strengths
1 - The multi-view decomposition approach is well-motivated and addresses a genuine limitation of existing methods that treat bias as a single source. The separation into feature, structural, and diffusion views provides a principled way to disentangle different sources of bias in graph data.

2 - The theoretical framework is solid, building on established information bottleneck principles and extending them appropriately to the multi-view conditional setting. The mathematical formulation clearly connects the compression and fairness objectives through mutual information terms.

3 - The experimental evaluation is comprehensive, including comparisons with seven state-of-the-art baselines across five datasets, thorough ablation studies validating each component, and sensitivity analyses demonstrating robustness to hyperparameter choices.

### Weaknesses
1 - The computational complexity and scalability concerns are not fully addressed. The method requires three separate encoders and additional contrastive learning computations, but there is no analysis of training time, memory requirements, or comparison of computational costs with baseline methods.

2 - The diffusion view construction using APPNP with IPW correction appears somewhat arbitrary. The paper doesn't justify why APPNP specifically was chosen over other propagation methods, nor does it provide ablation studies comparing different diffusion mechanisms or validating the effectiveness of IPW correction in isolation.

3 - The writing quality needs improvement, with several grammatical errors and awkward phrasings throughout (e.g., "d remain predictive" on page 2, inconsistent capitalization). The paper also has organizational issues, with some experimental details relegated to appendices that would be better integrated into the main text.

4 - The method's limitation to binary sensitive attributes is a significant practical constraint not adequately discussed. Real-world fairness scenarios often involve multiple or continuous sensitive attributes, and the paper doesn't address how the framework would extend to these cases.

### Questions
1 - How does the computational cost of FairMIB compare to baseline methods in terms of training time and memory usage?

2 - How would the framework extend to handle multiple sensitive attributes or continuous sensitive variables?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes FairMIB, a fairness-aware framework for GNNs that decomposes graph information into feature, structural, and diffusion views. It applies a multi-view conditional information bottleneck (MCIB) objective to balance utility and fairness by minimizing mutual information with sensitive attributes while maximizing task-relevant information. Experiments on five benchmark datasets (German, Bail, Credit, Pokec-z, Pokec-n) show improved fairness metrics compared with existing baselines such as FairVGNN, GRAFair, and DAB-GNN.

### Strengths
1. The paper is clear and easy to follow.

2. They evaluate on five datasets and include several baseline comparisons and ablation studies, which gives some empirical support for the proposed approach.

### Weaknesses
1. The paper has limited novelty. Nearly every component of FairMIB, i.e., information bottleneck, multi-view consistency, contrastive learning, IPW correction, has been previously published. The contribution is primarily a combination of existing tools rather than a conceptual breakthrough. The paper’s framing as a “multi-view information bottleneck” for fairness appears incremental relative to GRAFair (Zhang et al. 2025) and FDGIB (Zheng et al. 2024). There is insufficient discussion of how FairMIB fundamentally advances beyond these prior works.

2. The paper claims to disentangle multiple sources of bias by constructing three graph views. However, there is no formal proof or empirical verification that the proposed decomposition actually isolates independent bias factors. In practice, node attributes, topology, and diffusion processes are highly entangled, treating them as separable “views” could be theoretically fragile and arguably artificial. 

3. The advance over prior fairness models (e.g, GRAFair, DAB-GNN) is incremental at best, the paper provides little justification for its distinct contribution.

### Questions
1. How is the independence between views ensured or even approximately satisfied? If not independent, the theoretical motivation for a multi-view bottleneck collapses. 

2. What is the true novelty over FDGIB and GRAFair? Beyond adding a diffusion view, FairMIB appears to remix existing elements.

3. In the ablation studies Figure 2 (b), for German and Bail datasets, why does removing the structure view lead to decreasing DP scores?

4. What is the runtime of FairMIB compared to baselines like FairVGNN and DAB-GNN?

5. Did you retune $\lambda$ and $\gamma$ in ablations? If not, results might unfairly penalize variants.

### Soundness
2

### Presentation
2

### Contribution
2
