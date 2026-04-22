# ProMoS: Prototype-Guided Distillation for Generalist Graph Anomaly Detection

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Graph anomaly detection (GAD) is crucial in high-stakes domains. Recently, generalist GAD is a type of GAD that trains a single detector and can be transferred to new graphs, and has attracted attention. However, existing methods often rely on scarce and costly annotations for training and sometimes even require few-shot support at inference, which limits their robustness to diverse and unseen anomaly patterns. To address this limitation, we introduce ProMoS, the first unsupervised generalist GAD framework, which detects anomalies by modeling the abundant normality in unlabeled data. Specifically, we introduce a knowledge-distillation (KD) architecture that distills normality representations from a frozen self-supervised graph neural network (GNN) teacher to a mixture-of-students (MoS) model. The MoS employs a shared branch to capture global patterns and a lightweight personalized branch to extract local normality from the teacher, avoiding learning normality from scratch while improving both expressiveness and efficiency. Second, we propose prototype-guided soft-label distillation to align the student with the teacher in a shared prototype space, thereby improving cross-graph transferability and generalizability. During inference, ProMoS performs zero-shot anomaly detection on unseen graphs based on teacher-student distillation bias and prototype geometric deviation. Extensive experiments on eleven zero-shot GAD tasks show that ProMoS consistently outperforms state-of-the-art supervised, unsupervised, and generalist baselines while reducing computational overhead, charting a practical path toward label-free, zero-shot generalist GAD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on generalist graph anomaly detection and introduces a method named ProMoS. It is an unsupervised generalist GAD framework that consists of a self-supervised GNN teacher and a mixture-of-students. The model optimization is performed via prototype-guided soft-label distillation.

### Strengths
1. The paper is well-structured and the code is released.

2. The studied problem is practical and important.

3. Experiments are comprehensive, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The introduction of discrepancy-aware commitment and refinement is not very clear. More detailed descriptions are needed.

2. What does r_i mean in Eq.2?

3. Since UNPrompt and AnomalyGFM are pre-trained on one dataset originally, how are they pre-trained in the proposed setting?

4. The authors are encouraged to provide more analysis or visualizations to demonstrate the effectiveness of the two-branch design.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ProMoS, an unsupervised generalist graph anomaly detection framework capable of zero-shot detection on unseen graphs. Specifically, it first builds a self-supervised GNN teacher and transfers normality representations to a mixture-of-students (MoS) model with local and global branches. Moreover, through prototype-guided soft-label distillation, ProMoS enhances cross-graph generalization.

### Strengths
1.	This paper focuses on unsupervised generalist graph anomaly detection, which is a challenging and practical problem. Moreover, the code is released.
2.	The utilization of knowledge distillation and prototypes alignment enhances cross-graph transferability and generalizability.
3.	The proposed method achieves better performance than the used baselines, demonstrating its effectiveness.

### Weaknesses
1.	For the pre-trained teacher, can it be replaced with other non-SSL methods?
2.	Does the shared branch and personalized branch share the prototypes as it says “share” in Figure 1? Moreover, how are they initialized, from teacher model or the student model?
3.	The authors should provide more analysis as to why Eq.9 could measure the reliability of nodes. For Eq.11, there is no routing regularization term.
4.	The authors use PubMed, Flickr, Questions and YelpChi as the pretraining datasets. Is there a specific reason? If not, what is the performance when using different pretraining datasets.
5.	In Table 5, the authors argue that it reports empirical runtimes. But only the complexity of methods is provided.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an unsupervised generalist GAD framework, ProMos, which transfers prior knowledge from a pre-trained graph self-supervised learning teacher and introduces MOS to balance expressiveness and efficiency. The framework is jointly optimized using a tailored set of loss functions, including prototype distillation, as well as discrepancy-aware commitment and refinement losses.

### Strengths
(1) This paper is well-motivated and well-written. Distilling knowledge from a pre-trained SSL model is an effective approach that aligns well with intuitive understanding.

(2) The authors propose a fine-grained, prototype-guided method that goes beyond the conventional binary classification of normal and abnormal classes.

### Weaknesses
(1) The details regarding the pre-training of the teacher model are not very detailed, particularly how the training across multiple graphs is integrated with the clustering process. There are four graph inputs—do the authors directly merge all nodes from these graphs and then perform clustering on the combined set? 

(2) It is also unclear whether the inputs to the student model are identical to those of the teacher model. The distillation process is intended to learn invariant features across different graphs, where the guidance from prototype learning helps the model capture the underlying normal patterns.

(3) Could the authors include a t-SNE visualization for one of the datasets to illustrate the effectiveness of the commitment loss and refinement loss? It would also be helpful to show the difference between the teacher’s and the student’s feature distributions.

### Questions
See above **Weaknesses**

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ProMoS, which aims to perform graph anomaly detection in a general and unsupervised manner. The approach uses a frozen self-supervised GNN as the teacher to provide representations, while a mixture-of-students model learns to capture multiple normality patterns through prototype-based soft supervision and discrepancy-aware refinement. Experiments on several benchmark graphs show that the method achieves strong zero-shot detection performance and demonstrates good generalization across different graph domains.

### Strengths
1. The motivation is novel and meaningful, as it considers the diverse anomaly patterns that exist across different graph datasets.
2. The overall framework and training objectives are clear and well explained.
3. The figures and layout are well organized and easy to follow, and the writing is clear with comprehensive experiments.

### Weaknesses
1. The method does not effectively ensure diversity among student models. Although multiple students are used to capture different modes, no concrete strategy enforces their differentiation.
2. Some formula notations are incorrect, such as in Eq. (2), where ‘ei’ and ‘ri’ appear inconsistent and likely refer to the same variable.
3. The ablation study shows limited improvement from modules PB, SB, and DIS, indicating their contributions are not significant.
4. The Discrepancy-aware Commitment and Refinement stage is somewhat confusing. It is unclear whether simultaneously adjusting the teacher embeddings and prototype vectors might weaken the distinctiveness and effectiveness of the prototypes.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
