# HSelKD: Selective Knowledge Distillation for Hypergraphs using Optimal Transport

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Hypergraph Neural Networks (HGNNs) excel at modeling high-order dependencies through hyperedges, but their heavy inference cost limits deployment in latency-sensitive industrial scenarios. Knowledge Distillation (KD) offers a promising way to combine the expressiveness of graph-based models with the efficiency of lightweight Multi-Layer Perceptrons (MLPs). However, existing KD methods typically transfer the full output distribution of the teacher, overlooking the practical setting where only a subset of knowledge is necessary or beneficial. To address this, we propose HSelKD, a selective KD framework that transfers task-relevant knowledge from an HGNN teacher to a lightweight MLP student. HSelKD leverages Inverse Optimal Transport to distill the most informative parts of the teacher’s knowledge in a capacity-aware manner. We further introduce two principled variants: (1) Task-Aware Distillation, which specializes the student on task-relevant labels, and (2) Reject-Aware Distillation, which equips the student with the ability to abstain from uncertain or out-of-scope predictions. Extensive experiments on hypergraph and graph benchmarks show that HSelKD consistently outperforms lightweight baselines, matches the accuracy of structure-aware teachers, and delivers faster inference by up to 53× with lower training cost and computational overhead. These results establish HSelKD as a practical and scalable solution for real-world, latency-constrained deployments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies knowledge distillation for hypergraph neural networks. It proposes HSelKD, a selective knowledge distillation framework that transfers task-relevant knowledge from an HGNN teacher to an MLP student. HSelKD leverages inverse optimal transport to distill the most informative parts of the teacher’s knowledge in a capacity-aware manner and has two variants: task-aware and reject-aware distillation. The authors conducted experiments on real datasets for evaluation and comparison.

### Strengths
1.	Knowledge distillation for hypergraph neural networks is an important research topic for speeding up inference by distilling knowledge to a lightweight MLP.
2.	The paper is clearly written and easy to follow.
3.	The experiments include various settings for evaluation and comparison.

### Weaknesses
1.	From a technical perspective, the idea of using optimal transport for knowledge distillation is not new. For example, SelKD: Selective Knowledge Distillation via Optimal Transport Perspective (ICLR 2025) employs a similar methodology, albeit on different types of data.
2.	Figures 1b, 1c, and 1d are presented in Section 1 but are only discussed in Section 4, making them difficult to understand when reading the paper. The explanation of these figures on page 7 is unclear. Additionally, the speedup improvement in Figure 1b is not significant, e.g., between HGSelKD-W32 and its counterpart HGNN-W32.
3.	The design in Section 3.2 for different scenarios appears ad hoc. Why are these two modes important to consider? Why not have a method that supports both without overly specific technical designs? I do not see the necessity or contribution of these two extensions with modifications to the objective function. Without Section 3.2, the technical content in Section 3.1 is short and limited.
4.	In Section 3.1, the approach is a relatively straightforward adoption of the OT technique for hypergraph distillation. What are the challenges and new contributions here?
5.	The latest GNN-to-MLP methods are not compared, such as:
-	Teach Harder, Learn Poorer: Rethinking Hard Sample Distillation for GNN-to-MLP Knowledge Distillation (CIKM 2024)
-	AdaGMLP: AdaBoosting GNN-to-MLP Knowledge Distillation (KDD 2024)
-	TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation (ICML 2025)
6.	HSelKD is inferior to HGNN in most settings and datasets in Table 2, where the authors did not highlight the best results in bold.
7.	In Figure 3, when the noise ratio varies, the performance is suboptimal compared to HGNN.

### Questions
1.	In Table 1, why are the results for all baselines not included in Tables 1 and 2, but only those for lightHGNN+ are presented?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses GNN-to-MLP knowledge distillation for hypergraphs. It specifically tackles a sub-problem where the MLP student is only required to predict on samples from a subset of classes. The authors employ Optimal Transport (OT) to formalize this selective distillation and propose a method to transfer task-relevant knowledge from the teacher to the student. Experimental results show the method outperforms lightweight baselines, achieves performance comparable to the structure-aware teacher, and obtains a 53x inference speedup.

### Strengths
1. The experimental results are strong, convincing, and extremely thorough.

2. Modeling the knowledge distillation process with Optimal Transport is highly interesting. It reframes the alignment as a global constraint rather than an instance-level one, revealing a valuable internal structure within the distillation task.

3. The work is supported by solid theoretical proofs.

### Weaknesses
1. The clarity of the core methodology in Section 3.1 is a concern, as it assumes deep expert knowledge of Optimal Transport. The paper fails to provide an intuitive explanation for what "transport" signifies in this context (i.e., as a global node-to-class alignment or matching). Critically, the marginal distributions $\mu$ (over nodes) and $\nu$ (over classes), which are fundamental to the OT problem $\Pi(\mu, \nu)$, are never explicitly defined, forcing the reader to assume their implementation (e.g., uniform distributions) and hindering the work's self-containedness.

2. The novelty of the proposed bi-level OT formulation is questionable. Ultimately, the framework still relies on aligning soft probability distributions—in this case, the OT-derived coupling matrices ($P_T$ and $P_S^{\theta}$)—which essentially serve as a more elaborately computed set of soft labels. The student parameters are then updated by minimizing the divergence between these two matrices, a process conceptually similar to standard soft-label distillation, just with a more complex, non-local alignment mechanism. It is unclear if this added complexity provides a fundamental advantage over more direct methods of matching teacher and student output distributions. Furthermore, I didn't see any relation between this OT formulation and hypergraph structure.

3. The use of a "rejection option" is a well-known and general technique in machine learning [1]. Its application in this context does not appear to offer significant novelty.

[1] Hendrickx, K., Perini, L., Van der Plas, D., Meert, W., and Davis, J., Machine Learning with a Reject Option: A survey, arXiv:2107.11277, 2021.

### Questions
1. What is the "contrastive loss" mentioned in lines 165-166?

2. What is the notation "L" in line 151?

3. What is the exact distribution of $\mu$ and $\nu$?

### Soundness
3

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
This paper addresses the significant challenge of high inference costs associated with Hypergraph Neural Networks (HGNNs), which limits their practical deployment. The authors propose HSelKD, a novel selective knowledge distillation (KD) framework to transfer knowledge from a large, powerful HGNN (teacher) to a lightweight, graph-free MLP (student).

The core novelty lies in its "selective" approach, which contrasts with standard KD methods that transfer the entire teacher's output distribution. HSelKD is built on a principled Inverse Optimal Transport (IOT) formulation, which aligns the student's and teacher's node-to-class prototype mappings (transport plans). This framework is extended with a hyperedge-aware loss to ensure high-order structural information is preserved.

### Strengths
1. Principled OT-Based Method: The use of Inverse Optimal Transport provides a strong theoretical foundation for the knowledge transfer, moving beyond simple logit-matching.

2. Thorough Evaluation: The comprehensive experiments, including ablations, parameter sensitivity, and large-scale datasets, build a very strong case for the method's effectiveness.

### Weaknesses
1. Training Complexity: While the inference is extremely fast (the main goal), the training of HSelKD is more complex than standard KD. The objective has three loss terms with three balancing hyperparameters ($\alpha$, $\beta$, $\lambda$), which likely requires careful tuning. Furthermore, as noted in Appendix D, the training complexity includes terms for Sinkhorn iterations ($O(IN_s C^2)$) and contrastive alignment ($O(N_s^2 C)$), which could be computationally intensive, although the paper states this is mitigated by mini-batching.

2. Heuristic Hyperedge Loss: The hyperedge-aware JSD loss is based on a simple averaging of node couplings within a hyperedge. While effective, this is a heuristic approach to capturing high-order information and may not fully distill the complex message-passing dynamics of the teacher.

### Questions
1. Regarding training cost: The paper's focus is on inference speed, which is excellent. However, given the complexity of the training objective (three losses, IOT, contrastive alignment), could the authors comment on the wall-clock training time of HSelKD compared to the baselines like LightHGNN+?

2. Regarding the "Task-Aware" mode: The experiments show strong performance on subtasks with 2-5 labels. How does the method perform in a more extreme "few-class" setting (e.g., distilling knowledge for only 1 or 2 classes from a teacher trained on 40+)? Does the MLP student still capture the teacher's knowledge effectively?

3. Regarding the hyperedge-aware loss: The simple average for aggregating node couplings is effective. Did the authors experiment with other aggregation functions (e.g., max-pooling, or a learnable attention mechanism) to create the hyperedge-level distributions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposed an Inverse OT-based hyper-graph distillation method that transfers topology information throguh hyper-edge level alignment. The paper further incorporate task-aware and reject-aware mode to handle cases of classifying only subset of classes and facing OOD classes. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. KD of hypergraph is an important task and less studied.
2. The paper gives comprehensive theoretical analysis that demonstrate its soundness.
3. Comprehensive experiments under different settings demonstrate the proposed method achieves good performance in multiple classification senarios.

### Weaknesses
1. The proposed method has limited relevance to hyper-graph KD. For graph/hyper-graph KD, the key question is how to enable MLP to encode structural information of graphs, but the paper doesn't target at the problem while handles some general problems like distilling most informative parts, and some special classification cases.
2. IOT-based KD has been studied; here it is used mainly to enable task-aware and reject-aware distillation. These are useful but do not constitute a core advance in hypergraph KD.
3. The paper didn't compare training and inference efficiency to othe graph KD methods.

### Questions
1. In Figure 3(a), the performance w.o. Hypergraph-Level Info generally achieves the same performance as full version, why hyper-edge information seems not important?
2. Why structural information is not sent to MLP at inference? Without structure as input, how can MLP encode graph structure for downstream tasks.

### Soundness
2

### Presentation
3

### Contribution
2
