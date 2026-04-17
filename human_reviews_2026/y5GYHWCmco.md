# TAME: A Task-Agnostic Framework for Robust Graph Neural Network Explanations via Structural Mixup

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Graph Neural Networks (GNNs) have demonstrated remarkable performance across a range of applications involving graph-structured data, particularly in high-stakes domains. However, the opaque nature of their decision-making processes limits their trustworthiness and broader adoption. 
Existing post-hoc explanatory methods aim to improve interpretability by identifying subgraphs that influence GNN predictions. Yet, these approaches are typically restricted to a specific type of task, such as classification with discrete decision boundaries or regression with continuous ones, which limits their general applicability. 
In this work, we propose TAME, a unified, task-agnostic framework for GNN explanation that addresses both the limitations of task-specific methods and the distribution shift caused by subgraph extraction. Our approach integrates contrastive learning into the Graph Information Bottleneck (GIB) framework, enabling consistent explanation across both classification and regression tasks. 
Furthermore, we introduce a novel mixup strategy built upon graph pooling, which generates in-distribution explanations through hard structural perturbations. Extensive experiments on diverse tasks demonstrate that TAME achieves state-of-the-art performance in generating robust and interpretable explanations across both synthetic and real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TAME, a unified, task-agnostic framework for GNN explanation that addresses both the limitations of task-specific methods and the distribution shift caused by subgraph extraction. The proposed approach integrates existing contrastive learning-based GIB objective for GNN explanation, enabling consistent explanation across both classification and regression tasks. To better generate in-distribution explanations, a novel mixup strategy built upon graph pooling is introduced.
Experiments show TAME achieves state-of-the-art performance in generating robust and interpretable explanations across both synthetic and real-world datasets.

### Strengths
- This paper extends the experimental boundaries of the GIB objective to classification tasks.

- This paper introduces a novel mixup strategy built upon graph pooling, which generates in-distribution explanations through hard structural perturbations.

### Weaknesses
1. The contrastive learning-based GIB objective for GNN explanation was proposed by RegExplainer [1], and its motivation is to address the issue that estimating I(G∗; Y) in regression tasks is not as straightforward as in classification tasks. Applying RegExplainer objective to classification tasks is also straightforward, and TAME merely further validates this objective in classification tasks—its contribution in this regard is somewhat limited.

2. The idea of Mix-up is also very similar to that of RegExplainer [1]. The innovation of TAME lies in replacing the soft-mask-based mixup strategy with a structural mixup strategy, which explicitly disentangles explanatory subgraphs from label-irrelevant ones. However, the structural mixup strategy involves the matching problem of two graphs and has many limitations:  
   - Structural Mixup requires (A')* and A* to have the same size, but in practice, the sizes of effective explanatory subgraphs vary across different graphs.  
   - A* replaces (A')* according to fixed-order node indices, which may introduce spurious features. Considering the rotation and translation invariance of graphs, it would be better to shuffle the node indices of A* first before replacing (A')*.

3. There is a lack of verification on whether the structural mixup strategy is more effective than other strategies (e.g., the soft-mask-based mixup strategy) in addressing the out-of-distribution (OOD) problem. The evaluation method for GNN explanation in benchmark [2] takes OOD into account, and the authors are advised to refer to [2] for further analysis.

4. Can TAME train a unified state-of-the-art explainer with shared weights for all tasks? If not, the term "TASK-AGNOSTIC FRAMEWORK" is somewhat confusing to me, and it is recommended to use "TASK-AGNOSTIC Objective" instead.

In summary, TAME's claim of being the first to propose a contrastive learning-based, task-agnostic GIB objective for GNN explanation is somewhat overstatement for me. The main contributions of TAME are twofold: first, it extends the experimental boundaries of the GIB objective (rather than proposing or optimizing the GIB objective itself); second, it replaces the soft-mask-based mixup strategy with a structural mixup strategy based on graph pooling, though the verification of this design is insufficient.

[1] RegExplainer: Generating Explanations for Graph Neural Networks in Regression Tasks  
[2] Evaluating Post-hoc Explanations for Graph Neural Networks via Robustness Analysis

### Questions
Please refer to the above weakness section for suggestions and questions.

### Soundness
3

### Presentation
4

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
This paper proposes a Task-Agnostic structural Mixup Explanation (TAME) framework for GNN explanation, which addresses both the limitations of task-specific methods and the distribution shift caused by subgraph extraction.

### Strengths
1. This is the first work that integrates contrastive learning into the Graph Information Bottleneck (GIB) framework.

2. This paper designs a novel mixup strategy that structurally replaces explanatory subgraphs, which generates in-distribution explanations
through hard structural perturbations. 

3. Experiments on both synthetic and real-world benchmarks demonstrate that TAME outperforms existing methods across diverse tasks, achieving up to a 30.1% improvement in AUC.

### Weaknesses
1. This paper focuses on post-hoc explanations, but ignores the scalability of dynamic graphs or large-scale GNNs, and does not mention the computational overhead.

2. The definition of Problem 1 (Post-hoc Instance-level GNN Explanatory Method) lacks a reference; is it widely acknowledged by other researchers, or is there any new novelty in this definition?

3. TAME is primarily designed for graph- and node-level explanations with well-defined subgraph dependencies, but not for edge-level tasks.

4. "Regarding hyperparameters, we apply grid search to determine the loss weight βand set the top-r ratios according to the ground-truth explanations." However, the detailed processes are not provided.  Are there any theoretical promises for selecting such hyperparameters?

5. "TAME introduces a theoretically grounded objective based on GIB." However, the theoretical contributions are weak, it lacks proof (such as the derivation of the lower bound of mutual information).

### Questions
1. The definition of Problem 1 (Post-hoc Instance-level GNN Explanatory Method) lacks a reference; is it widely acknowledged by other researchers, or is there any new novelty in this definition?

2. How to optimally choose hyperparameters is not well illustrated. Are there any theoretical promises for selecting such hyperparameters?

### Soundness
2

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
4

### Summary
This paper proposes TAME, a unified task-agnostic framework for GNN explainability for both graph classification and regression tasks, and also addresses distribution shift problems of explanatory graphs.

### Strengths
S1. A task-agnostic explainer on graphs presents an interesting and efficient approach for generating graph explanations across diverse problem settings, including regression tasks.

S2. Additionally, authors attempt to provide robust explanations by addressing OOD problems of explanatory graphs.

### Weaknesses
W1. The proposed TAME is not the first to propose a task-agnostic explainer model. TAGE (NeurIPS 2022) [1] also introduced task-agnostic explainers based on conditioned contrastive learning, demonstrating performance on both graph classification and node classification tasks, with potential extensibility to regression tasks as well.

W2. The effectiveness of structural mixup in Section 3.2 in addressing distribution shift may be limited in certain cases. Due to the inductive biases learned by GNNs, nodes with similar attributes or global structural roles can have similar embeddings even when they are not neighbors. In such scenarios, the proposed structural mixup strategy may have restricted effectiveness.

W3. In Table 3, the explanation embeddings are shown to be much closer to the original graph distribution than to the ground truth, suggesting that the extracted subgraphs should differ from the ground truth. However, Figures 4 and 5 show that the explanations are nearly identical to the ground truth, which seems contradictory.

[1] Task-Agnostic Graph Explanations, Neurips 22 (https://arxiv.org/pdf/2202.08335)

### Questions
Q1. Please clarify the novelty of TAME compared to TAGE (NeurIPS 2022). What are the key technical differences and contributions that distinguish your approach? Additionally, include TAGE as a baseline in your experiments and provide a comparative performance analysis.

Q2. The out-of-distribution (OOD) experiments in Table 3 are limited to comparisons with ground truth only. Please demonstrate the effectiveness of your proposed method by comparing it not only with ground truth but also with other baseline methods, such as MixupExplainer and ProxyExplainer, that address distribution shift.

Please consider the weaknesses in conjunction with the questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a task-agnostic framework for explaining Graph Neural Networks (GNNs) named TAME,  which unifies explanation across both classification and regression tasks. TAME reformulates the Graph Information Bottleneck (GIB) framework through InfoNCE. And it introduces a structural mixup method based on graph pooling with a joint sampling strategy to generate in-distribution mixup graphs. Experiments show that TAME significantly outperforms prior explainers in accuracy and robustness.

### Strengths
1. This paper provides a rigorous theoretical foundation for deriving the proposed task-agnostic loss function.

2. The paper is easy to follow, and the figures are well-designed and visually appealing.

3. The experimental results in this paper support the method proposed in this paper.

### Weaknesses
1. The contribution of the paper appears limited. TAME mainly improves upon the structural sampling step of the previous structure mixup method like MixupExplainer. Can the author clarify in more detail how TAME is substantially different from MixupExplainer. In addition, please explain how the InfoNCE-based loss in TAME differs from the loss function used in RegExplainer.
2. As a post-hoc explanation method, the paper should consistently use the term “explainability” rather than “interpretability”, which is more appropriate in this context [1].
3. The Sampling Neighbors procedure is not clearly described. In particular, it is unclear how the negative neighbors $\left\{{y_j^-} \right\}_{j=1}^{N-1} are selected during the sampling process.
4. In the Introduction, the paper states that “soft-mask-based explanatory methods struggle to effectively drive edge weights toward a binary form.” Please clarify how TAME overcomes this limitation, given that it still employs a soft mask generated by an MLP.
5. The paper claims that TAME is task-agnostic. Could the authors discuss whether the proposed framework can be extended to other tasks such as node-level classification or edge-level link prediction?
6. The paper does not specify which explanation network architectures were used for MixupExplainer and RegExplainer when serving as baselines. Were their original explainer modules reimplemented according to their respective papers, or were they unified under the same explainer structure for fairness? Clarifying this is crucial for assessing the validity of the comparisons.

Reference: [1] Yuan, H., Yu, H., Gui, S., & Ji, S. (2022). Explainability in graph neural networks: A taxonomic survey. IEEE transactions on pattern analysis and machine intelligence, 45(5), 5782-5799.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
