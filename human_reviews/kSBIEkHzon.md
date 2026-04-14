# Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-trees

- Decision: Reject
- Scores: 3, 6, 6, 6

## Abstract
Foundation models aim to create general, cross-task, and cross-domain machine learning models by pretraining on large-scale datasets to capture shared patterns or concepts (generalities), such as contours, colors, textures, and edges in images, or tokens, words, and sentences in text. However, discovering generalities across graphs remains challenging, which has hindered the development of graph foundation models. To tackle this challenge, in this paper, we propose a novel approach to learn generalities across graphs via task-trees. Specifically, we first define the basic learning instances in graphs as task-trees and assume that the generalities shared across graphs are, at least partially, preserved in the task-trees of the given graphs. To validate the assumption, we first perform a theoretical analysis of task-trees in terms of stability, transferability, and generalization. We find that if a graph neural network (GNN) model is pretrained on diverse task-trees through a reconstruction task, it can learn sufficient transferable knowledge for downstream tasks using an appropriate set of fine-tuning samples. To empirically validate the assumption, we further instantiate the theorems by developing a cross-task, cross-domain graph foundation model named Graph generality Identifier on task-Trees (GIT). The extensive experiments over 30 graphs from five domains demonstrate the effectiveness of GIT in fine-tuning, in-context learning, and zero-shot learning scenarios. Particularly, the general GIT model pretrained on large-scale datasets can be quickly adapted to specific domains, matching or even surpassing expert models designed for those domains.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper centers on learning shared structures, or “generalities,” across various graphs using a unique construct called "task-trees." The proposed model, Graph Generality Identifier on task-Trees (GIT), is designed for broad applicability across graph-related tasks, such as node, edge, and graph classification.

### Strengths
1. The presentation is generally clear.
2. Both theoretical and experimental sections are well-executed, with experiments spanning a range of graph-based tasks across multiple domains.

### Weaknesses
1. My primary question is: what exactly differentiates the proposed Task-Tree Space from existing concepts like prompt graphs, contextual graphs, anchor nodes, or nodes of interest[1][2][3][4]? We already know that subgraphs can be used to unify node-, edge-, and graph-level tasks, and this paper seems to be doing the same thing, just under a different name.
2. While task-trees provide a generalized approach, there is a potential trade-off in terms of performance on domain-specific tasks, as noted in the specialization experiments. 
3. Could the authors clarify the computational impact of task-tree transformations, particularly for large-scale graphs?

[1]Sun, X., Cheng, H., Li, J., Liu, B., & Guan, J. (2023, August). All in one: Multi-task prompting for graph neural networks. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2120-2131).

[2]Liu, H., Feng, J., Kong, L., Liang, N., Tao, D., Chen, Y., & Zhang, M. (2023). One for all: Towards training one graph model for all classification tasks. arXiv preprint arXiv:2310.00149.

[3]He Y, Hooi B. UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language[J]. arXiv preprint arXiv:2402.13630, 2024.

[4]Huang, Q., Ren, H., Chen, P., Kržmanc, G., Zeng, D., Liang, P. S., & Leskovec, J. (2024). Prodigy: Enabling in-context learning over graphs. Advances in Neural Information Processing Systems, 36.

### Questions
See weaknesses.

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
4

### Summary
The paper presents GIT, learning general patterns using task-trees. By pretraining on task-trees, GIT captures transferable knowledge, validated through experiments on 30 graphs across five domains. GIT demonstrates strong performance in fine-tuning, in-context, and zero-shot learning, often outperforming domain-specific models.

### Strengths
1. The paper is well-written and easy to follow.
2. The paper provides a theoretical analysis of the effectiveness of task-trees.

### Weaknesses
1. **Inaccurate Statements**: The authors make some inaccurate statements. For example, in Appendix A, they claim that "most of these approaches rely on subgraphs as the primary learning instances, which can result in inefficient training and reduced expressiveness, as discussed in the main paper." However, many foundation models [1,2] do not rely on subgraphs.
2. **Limited Novelty in Task Trees as Unified Instances**: The novelty of leveraging task trees as unified instances is limited. Building task trees based on task-relevant nodes as unified instances is quite similar to using subgraphs as instances [3,4]. Subgraphs are a versatile structure that can represent node-, edge-, and graph-level instances [3].
3. **Limitations in Building a Graph Foundation Model**: I believe the paper is far from building a graph foundation model. Although a unified instance is proposed to unify various tasks, the semantic and structural gaps between various domains [1,2] remain unaddressed. These are important challenges in building a foundation model.

[1] Zhao et al. "All in one and one for all: A simple yet effective method towards cross-domain graph pretraining." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.\
[2] Yu et al. "Text-Free Multi-domain Graph Pre-training: Toward Graph Foundation Models." arXiv preprint arXiv:2405.13934 (2024).\
[3] Liu et al. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks[C]//Proceedings of the ACM Web Conference 2023. 2023: 417-428.\
[4] Hao et al. Motif-based prompt learning for universal cross-domain recommendation[C]//Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 2024: 257-265.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes GIT, which learns generalities across graphs via a novel concept of task trees. The task trees are defined as the basic learning instances in graphs and they can preserve the generalities shared across graphs. The extensive experiments demosntrate the effectiveness of GIT in fine-tuning, in-context learning, and zero-shot learning.

### Strengths
1. This paper proposes a novel concept of task trees.
2. The authors show that task trees can preserve the generalities shared across graphs by several theorems and extensive experiments.

### Weaknesses
1. As pointed out by Line 90, message-passing GNNs---which have been used in many existing graph prompt methods---can fully capture task trees. Thus, the existing graph prompt methods may also implicitly use task trees (the definition may be different). Please compare GIT with the existing graph prompt methods in detail and explain why GIT can capture the task generalities better than the existing graph prompt methods.
2. I suggest moving the related work into the main body.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a solution towards graph foundation model which works in cross-task and cross-domain scenarios. With a task-tree-based appraoch to unify different graph tasks, along with a reconstruction-based pretraining objective, the authors provide theoretical analysis on the stability and transferability of their proposed method. By pretraining on a diverse set of graph datasets, the pretrained model is able to generalize to unseen graphs from various domains, and the specification process can further improve the performance.

### Strengths
1. The paper is well written and contains extensive information from theoretical and empirical perspectives. 

2. Based on the experimental results, the proposed method achieves the goal of 'cross-task, cross-domain' transfer learning, which is quite impressive compared with existing GFM research.

### Weaknesses
I would urge the authors to provide a more detailed discussion of the difference between their proposed 'task-tree' method and existing subgraph-based methods.

### Questions
1. What are the differences between the proposed 'task-tree' method and existing subgraph-based methods? Could you explain this further, in terms of efficiency (i.e., why the task-tree method is more efficient), and effectiveness (i.e., how can the task-tree representation improve the performance of graph tasks). 

2. The authors adopted a reconstruction-based objective for pretraining. Could you provide some intuitions to justify this choice? Is it possible to use a contrastive learning method instead? 

3. In the fine-tuning experiments, did the model go through the specification process, or were they directly fine-tuned on specific downstream datasets after general pretraining?

### Soundness
3

### Presentation
3

### Contribution
3
