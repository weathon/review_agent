# Bidirectional Reverse Contrastive Distillation for Progressive Multi-Level Graph Anomaly Detection

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Graph Anomaly Detection (GAD) faces the challenge of identifying irregular patterns across multiple structural scales while maintaining computational efficiency for real-world deployment. Existing knowledge distillation approaches rely on unidirectional teacher-student alignment, producing brittle embeddings that fail to establish robust decision boundaries between normal and anomalous patterns. We introduce \textsc{ReCoDistill}, a unified framework that combines bidirectional contrastive learning with progressive checkpoint-based distillation using a single teacher network. Our approach simultaneously optimizes two complementary objectives: (1) attracting student embeddings toward clean teacher representations while (2) repelling them from structured multi-scale noisy teacher outputs. We develop a dynamic curriculum mechanism that selects optimal teacher checkpoints based on complexity-compatibility trade-offs, progressing from local to global semantics. Unlike existing methods requiring multiple teacher networks, \textsc{ReCoDistill} achieves superior efficiency through single-teacher architecture while maintaining state-of-the-art performance. Evaluation on 14 benchmark datasets demonstrates that \textsc{ReCoDistill} achieves the best detection accuracy (88.93\% AUROC on Amazon, 89.80\% on BM-MN), superior zero-shot transfer performance across 9 out of 12 cross-task scenarios, and substantial computational efficiency improvements. Our theoretical analysis provides convergence guarantees and generalization properties, establishing \textsc{ReCoDistill} as the first computationally efficient GAD framework to unify bidirectional contrastive learning with progressive distillation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ReCoDistill, a teacher–student framework for graph anomaly detection (GAD). A single 3-layer GCN teacher produces “clean” and perturbed “noisy” views; a 1-layer GCN student is trained with a bidirectional contrastive objective and a progressive checkpoint curriculum. At inference time, only the student is used with a dual anomaly score (reconstruction + Mahalanobis).

### Strengths
1. GAD is important and practically relevant.
2. Results are reported on 14 datasets with AUROC/Macro-F1/AUPRC and multiple baselines.

### Weaknesses
1. Questionable Motivation for Computational Efficiency: The paper's central motivation of computational efficiency is undermined by the choice of a 3-layer GCN teacher and 1-layer GCN student. GCNs are already memory-efficient, and the compression from 3 layers to 1 layer provides minimal practical benefit. The paper lacks crucial metrics to justify their motivations: (i) peak training memory, (ii) inference memory, (iii) per-epoch train time, and (iv) per-graph/node inference time, for both teacher and student.


2. Missing Related Work Section: The absence of a dedicated related work section is a critical omission for a conference paper. This makes it difficult to position the work within existing literature and understand what specific gaps are being addressed. The paper should clearly delineate contributions relative to existing GAD, contrastive learning, and knowledge distillation methods. 


3. Multi-Scale Perturbations (Section 2): The perturbations (Gaussian noise, edge dropping and adding) are standard augmentation techniques where previous works use to do consistent training[1]. In these previous works, the labels of the perturbed nodes remain unchanged. However, the paper doesn't justify why these would create valid anomalies rather than just noisy normal samples in their setting, especially given that the perturbation strengths is weak in the experiment section. Also, graph-level "rewiring" is mentioned but never fully described. 


4. Bidirectional Learning Concerns: Equation (5) shows the teacher being pulled toward student embeddings. This could degrade teacher performance or lead to representation collapse. But the paper doesn't address such stability concerns or provide convergence analysis.

5. The compatibility and complexity metrics in Eq. (7) lack formal definitions. Without knowing what these measure, the checkpoint selection is hard to understand.

6. Incomplete Training Details: "Clean data" is ambiguous - does this mean graphs without anomaly labels or without perturbations? The self-supervised objectives for teacher pre-training are mentioned but not specified. In Equation (15), the attention vector u appears without explanation of its training procedure.

7. Mathematical Inconsistencies: In Equation (12), the reconstruction error is computed between h-dimensional and h'-dimensional vectors, where dimensions are incompatible.

8. Figure 3's t-SNE visualization shows no clear separation between normal/anomaly nodes where anomaly nodes are distributed evenly among normal nodes. This might suggest that the teacher may not learn meaningful representations.

9. Presentation Issues: Figures 3 and 4 consume excessive space without conveying proportional information. The writing could be more concise to accommodate proper related work discussion. Table 3's ablation study has unclear row/column semantics.

[1] Feng, Wenzheng, et al. "Graph random neural networks for semi-supervised learning on graphs." Advances in neural information processing systems 33 (2020): 22092-22103.

### Questions
See Weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes RECODISTILL, a knowledge distillation framework for graph anomaly detection. The key idea is to use a single teacher model to produce both clean and perturbed (anomaly-like) graph representations. A bidirectional contrastive distillation objective encourages the student to align with clean teacher embeddings and be repelled from noisy ones, improving anomaly-aware decision boundaries. Extensive experiments on 14 benchmarks show consistently strong performance and theoretical analysis provides convergence and generalization insights.

### Strengths
1.Bidirectional contrastive distillation with clean and noisy representation alignment is novel and makes strong intuitive sense in anomaly detection.
2.Convergence guarantees and error bounds provide additional insight in the framework.
3.Comprehensive experimental validation.

### Weaknesses
1. Synthetic perturbations may not match real anomalies. It remains unclear how the method behaves if the anomaly distribution diverges from the designed perturbations.

2. The paper positions efficiency as a main practical advantage — using a single teacher and discarding it during inference. However, no quantitative evidence (training time, inference latency, memory usage, FLOPs, etc.) is provided to support this claim.

3. Although embedding separability is visualized, there is little discussion on what types of anomalies are detected.

4. Section 2.0.0.1 appears to introduce an unnecessary extra numbering level that disrupts the logical structure of the methodology section.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

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
This paper introduces ReCoDistill, a novel framework for Graph Anomaly Detection (GAD) that innovatively combines the teacher-student knowledge distillation paradigm with bidirectional contrastive learning. The method addresses key limitations in existing GAD approaches, such as their lack of anomaly awareness and multi-scale adaptability. Its core contributions include a bidirectional contrastive objective that enables mutual refinement between teacher and student models, a dynamic checkpoint curriculum for progressive learning from local to global patterns, and a multi-scale perturbation strategy to generate realistic anomalous views. The framework is designed for efficiency, requiring only the lightweight student model during inference, which utilizes a dual scoring mechanism based on reconstruction error and distributional deviation. The paper is supported by extensive experiments on 14 benchmarks, demonstrating state-of-the-art performance, and is further bolstered by a comprehensive theoretical analysis providing convergence and detection guarantees.

### Strengths
1. The bidirectional contrastive distillation paradigm is a novel and creative fusion of knowledge distillation and contrastive learning.

2. Rigorous experiments on 14 datasets with strong baselines, ablation studies, and zero-shot transfer tests solidly validate the method.

3. The student-only inference provides significant efficiency gains without sacrificing performance, which is crucial for real-world deployment.

### Weaknesses
1. The multi-scale perturbations, while systematic, are based on relatively simple mechanisms like Gaussian noise and random edge flipping. Real-world anomalies can be more complex and adversarial. The paper would be strengthened by a discussion on whether these simple perturbations are sufficient to simulate the true anomaly distribution or if more sophisticated, data-driven perturbation methods could be explored in future work.

2. The theoretical analysis relies on assumptions such as bounded embeddings and Lipschitz-continuous networks. A discussion on the practical validity of these assumptions for real-world graph data and their potential impact on the realized performance would make the theory more accessible and grounded.

3. Although a theoretical complexity analysis is provided, the paper lacks concrete, empirical measurements of inference speed-up and memory reduction compared to the teacher model and other large baselines. Reporting actual latency, throughput, or memory usage would make the efficiency claims more tangible and compelling for practitioners.

### Questions
This review would be strengthened by the inclusion of direct empirical measurements of inference speed and memory usage, alongside a discussion on the practical validity of the theoretical assumptions for real-world graph data.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
RECODISTILL introduces a unified framework that combines bidirectional contrastive learning with progressive checkpoint-based distillation using a single teacher network, where student embeddings are attracted toward clean teacher representations while repelled from structured multi-scale noisy teacher outputs. The approach simultaneously optimizes two complementary objectives through a dynamic curriculum mechanism that selects optimal teacher checkpoints based on complexity-compatibility trade-offs, progressing from local to global semantics. Evaluation on 14 benchmark datasets demonstrates that RECODISTILL achieves the best detection accuracy (88.93% AUROC on Amazon, 89.80% on BM-MN) and superior zero-shot transfer performance across 9 out of 12 cross-task scenarios.

### Strengths
1. The paper proposes a novel bidirectional contrastive distillation framework where the student learns to attract clean teacher embeddings while repelling noisy ones, and the teacher is regularized to maintain separation between clean and corrupted representations. 
2. The method incorporates a progressive checkpoint selection mechanism that dynamically matches teacher complexity to student capacity, enabling curriculum learning from local patterns to global graph-level anomalies without requiring multiple teacher networks. 
3. Comprehensive experiments on 14 diverse datasets demonstrate state-of-the-art performance with 88.93% AUROC on Amazon and superior zero-shot transfer capabilities across 9 out of 12 cross-task scenarios.

### Weaknesses
1. The motivation for using knowledge distillation to achieve efficiency is unclear, as inference on graphs with millions of nodes using shallow GNNs is already efficient, and dynamic scenarios can be handled efficiently by performing inference only on new nodes with neighbor sampling.
2. The proposed method combines existing techniques (contrastive learning, knowledge distillation, multi-scale perturbations, progressive checkpoints) without clear individual novelty, and the ablation study sometimes shows modest performance drops when removing key components, making it difficult to identify the core contribution.
3. Despite emphasizing "EFFICIENT" in the title, the paper lacks comprehensive efficiency experiments, and there is a title inconsistency between the submission system and PDF version.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
