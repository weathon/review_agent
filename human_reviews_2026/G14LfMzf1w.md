# Normality Calibration in Semi-supervised Graph Anomaly Detection

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Graph anomaly detection (GAD) has attracted growing interest for its crucial ability to uncover irregular patterns in broad applications. Semi-supervised GAD, which assumes a subset of annotated normal nodes available during training, is among the most widely explored application settings. However, the normality learned by existing semi-supervised GAD methods is limited to the labeled normal nodes, often inclining to overfitting the given patterns. These can lead to high detection errors, such as high false positives. To overcome this limitation, we propose $GraphNC$, a graph normality calibration framework that leverages both labeled and unlabeled data to calibrate the normality from a teacher model (a pre-trained semi-supervised GAD model) jointly in anomaly score and node representation spaces. GraphNC includes two main components, anomaly score distribution alignment ($ScoreDA$) and perturbation-based normality regularization ($NormReg$). ScoreDA optimizes the anomaly scores of our model by aligning them with the score distribution yielded by the teacher model. Due to accurate scores in most of the normal nodes and part of the anomaly nodes in the teacher model, the score alignment effectively pulls the anomaly scores of the normal and abnormal classes toward the two ends, resulting in more separable anomaly scores. Nevertheless, there are inaccurate scores from the teacher model. To mitigate the misleading by these scores, NormReg is designed to regularize the graph normality in the representation space, making the representations of normal nodes more compact by minimizing a perturbation-guided consistency loss solely on the labeled nodes. Through comprehensive experiments on six benchmark datasets, we show that, by jointly optimizing these two components, GraphNC can 1) consistently and substantially enhance the GAD performance of teacher models from different types of GAD methods and 2) achieve new state-of-the-art semi-supervised GAD performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes GraphNC, a teacher–student framework for semi-supervised graph anomaly detection (GAD). It introduces two main components: Score Distribution Alignment (ScoreDA) aligns the student’s anomaly scores with the teacher’s distribution, aiming to calibrate the notion of “normality.”  Perturbation-guided Normality Regularization (NormReg) applies consistency regularization to labeled normal nodes, enhancing intra-class compactness.

### Strengths
* The method can be applied on top of various existing GAD models, showing flexibility.

### Weaknesses
1. The proposed method is essentially a straightforward combination of existing ideas. The teacher model is drawn directly from the baselines, while the student model is a simple contrastive learning framework augmented by an extra loss term (L_{ScoreDA}). There is little genuine innovation in either architecture or learning strategy. Yet, the method achieves surprisingly high performance, which raises questions about where the actual gain comes from — this will be discussed further below.

2. The authors train the teacher model on each dataset exactly as in the original papers, under the same semi-supervised setting. Then, on the same dataset, they train the student model by aligning its predictions with the teacher’s anomaly scores. From the loss formulation, it is clear that the teacher’s predicted scores are treated as ground truth, without considering possible teacher errors or noise. Consequently, the overall performance depends entirely on how good the teacher is.
Table 5 confirms this dependency: when the teacher performs poorly on a dataset, the proposed method can even hurt performance instead of improving it.

3. Conceptual inconsistency with the teacher–student paradigm and semi-supervised learning:
This experimental design violates the fundamental purpose of knowledge distillation. In typical distillation, the goal is for a smaller or cheaper student model to approximate or even surpass a strong teacher using the same data or fewer resources. However, here the student is trained after a full teacher pretraining phase on the same dataset, effectively making the process redundant and inefficient.
Moreover, when the teacher model already achieves very high accuracy (e.g., We regard teacher scores as near-supervised labels when (i) AUPRC ≥ 0.8 on datasets with anomaly rate ≤ 1%, (ii) Precision@Top-1% ≥ 90%, and (iii) OOF-estimated label noise within the top-scored set ≤ 5%), this pipeline can no longer be considered semi-supervised. The student is simply learning from near-perfect labels, making the task closer to a fully supervised binary classification problem rather than genuine graph anomaly detection. This undermines the originality and validity of the proposed framework.

### Questions
* Did you use an Out-of-Fold (OOF) protocol to avoid leakage from teacher to student?

* Equation (3) sign/direction appears inconsistent with the text (minimize vs. maximize), and Algorithm 1 repeats the issue.  Equation (3) defines L_NormReg =  -XXX. , while the text says it minimizes the discrepancy between representations. This negative sign contradicts both the description and Algorithm 1 (which minimizes the total loss). Could the authors clarify whether L_NormReg is minimized or maximized in implementation, and correct the sign accordingly?

### Soundness
3

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
4

### Summary
This paper presents GraphNC, a framework for semi-supervised graph anomaly detection that improves a student model by leveraging a pre-trained teacher model. Specifically, ScoreDA aligns the anomaly score distribution between teacher and student to enhance the capability of the student model, while NormReg, a perturbation-based masking regularization, mitigates the negative effect of inaccurate teacher outputs by enforcing compact representations of normal nodes.

### Strengths
1. The paper is clearly written, with a clear problem statement and a logically organized presentation.
  2. The experimental evaluation is fairly comprehensive, covering multiple datasets and various baseline methods.

### Weaknesses
1. The novelty of the paper is limited. the overall framework essentially follows a distillation paradigm, with the main contribution being the incorporation of a graph augmentation–based consistency regularization technique to improve the anomaly detection capability of the student model. However, the methodological innovation of the framework is relatively low, as it largely combines and applies existing ideas from distillation and consistency regularization.
2. The method assumes the teacher produces reasonably accurate scores for many nodes. If the teacher is poor, ScoreDA could harm performance.

### Questions
1. In Eq. (3), the loss term is defined with a negative sign, while the hyperparameter α is positive. This setup would effectively increase the discrepancy between original and augmented representations, which contradicts the explanation in lines 260–261 stating that NormReg minimizes this discrepancy. Could you clarify whether this is a typographical or conceptual mistake in the paper, or if I have misunderstood the formulation?
2. The primary objective of this work is to enable the student model to surpass the teacher model through a distillation-like approach. However, if the teacher model performs poorly on certain datasets, can the proposed framework still function effectively? It is recommended that the authors provide robustness experiments under such conditions or explicitly clarify this limitation in the paper. 
3. In the NormReg module, random masking is applied to labeled normal nodes to generate augmented samples, and the model minimizes their representation discrepancy. However, masking may alter node features and deviate them from the normal distribution. How do the authors ensure that these augmented nodes remain normal? If some augmented samples lose their normality, wouldn’t minimizing their discrepancy reduce the model’s ability to distinguish anomalies?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies semi-supervised graph node anomaly detection (GAD) problem, where the model is trained with a subset of labeled normal nodes and then infers on the full graph. They propose a framework to enhance existing semi-supervised GAD methods with two components: (1) Anomaly Score Alignment, which aligns the anomaly score distirbution of the teacher model (namely one off-the-shelf semi-supervised GAD method) with those of the student model (namely the enhanced model); By using aligning with raw anomaly scores from the teacher model, the student model can better separate normal and abnormal instances in their anomaly score distribution. (2) However, the teacher model can make mistakes. In other words, the anomaly scores produced by the teacher model may be misleading. To mitigate this, they employ a regularizer to make the represenations of normal instances more compact in the latent space. One can see the idea is very intuitive but still interesting. They perform many experiments to evalute the effectiveness of their framework. Though it is a good paper already, I have two major concerns (see weak points), which should be addressed to fulfil the (high) publication standard  of ICLR.

### Strengths
1. This paper is very well presented, including the organisation and writing.
2. The idea is intuitive but interesting. (I will not call an intuitive idea ``lack of novelty").
3. Most claims are supported either by theoretical analysis or empirical studies.

### Weaknesses
##Major Concerns

1. [Implicit Assumption]: During the construction of GraphNC, the authors (implicitly) assume the teach model performs (reasonably) well. However, this is not always the case. I would like to see a study where the teacher model performs poorly (say with ROC < 0.5). In this case, most predited labels given by the teacher model are wrong (namely the anomaly scores given by the teacher model are misleading), will your framework amplify such wrong signals (and thus get worse results, i.e., lower accuracy)?

2. [Unjustified results]: In figure 4(e) and (f), one can clearly see that the performance of GraphNC first increases but then decreases with the increase of training size. This is counter-intuitive but the authors did not investigate the reasons. Instead, they wrote that ``Similar with other competing methods, with the increase of training size $R$, our model GraphNC generally performs better...", which is not true and not responsible. Importantly, one can see that GraphNC obtains the best results when the ratio is 0.15 (which corresponds to the setting of results reported in Table 1 and other main tables). However, if the ratio changes (either increases or decreases), the AUROC may substantially decreases (e.g., from 0.76 to 0.68 on Photo; Similar trends may also hold for other datasets, and I would like the authors to provide full results on all datasets regarding this.) Overall, this is a critical issue that needs to be investigated and solved. Otherwise, the conclusions given by this paper should be treated with caution.

##Minor Concerns

1. [False y-axis labels]: In Figure 1, the range of  false positive rate (and false negative rate) should be [0,1]. That is why we call it a "rate".

2. [Writing]: Page 6, ``...Note that some unsupervised methods, such as CoLA (Liu et al., 2021b), GRADATE (Duan et al., 2023), and HUGE (Pan et al., 2025), cannot be adapted to semi-supervised setting due to their design more relying on fully unlabeled data, which are excluded from our comparison." this sentence should be put behind the Competing Methods rather than Datasets.

3. [Writing]: Page 7, ``...The AUROC and AUPRC results are shown in Table 3....", it should be Table 2.

I will raise my ratings if my concerns are (partially) addresses.

### Questions
Please check the weak points.

### Soundness
2

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
4

### Summary
The authors propose a framework, GraphNC (Graph Normality Calibration), designed to improve semi-supervised graph anomaly detection (GAD). The paper addresses the problem of existing models overfitting to the limited patterns of labeled normal nodes, which leads to high false positive rates. The main idea is to use a teacher-student framework to "calibrate" the normality learned by a pre-trained GAD model (the teacher). The authors propose two main components: (i) Anomaly Score Distribution Alignment (ScoreDA), which trains the student model by aligning its output scores with the teacher's score distribution across all nodes , and (ii) Perturbation-based Normality Regularization (NormReg), which mitigates the impact of the teacher's inaccurate scores by enforcing a consistency loss on augmented, labeled normal nodes to learn a more compact normal representation. The authors then demonstrate that GraphNC can be flexibly applied to different types of teacher models (e.g., reconstruction, one-class) and improves their performance across six benchmark datasets.

### Strengths
There are a few things I like about the paper:
1. The paper addresses a practical challenge in semi-supervised GAD: overfitting to the seen normal patterns and the resulting high false positive rate, which indicates the need for further calibration on the prediction.
2. The proposed GraphNC framework can serve as a plug-and-play module to enhance existing semi-supervised GAD models.
3. The proposed components are well-motivated. ScoreDA leverages the teacher's general knowledge, while NormReg specifically targets the teacher's weaknesses (inaccurate scores) by introducing a regularization term on labeled data.
4. The authors provide a theoretical analysis of the proposed approach.
5. The demonstrated consistent improvement over semi-supervised models without normality calibration over multiple datasets, as well as provided ablation study of the model.

### Weaknesses
1. [Baselines]. Some of the baselines models used as the teacher models in the paper are relatively old, like DOMINANT. I would suggest the authors add more recent baselines.
2. [Ablation]. The ablation study is insightful but reveals a potentially negative interaction. The variants with just NormReg perform poorly, sometimes even worse than the original teacher (OT). This suggests NormReg is not a standalone regularizer and is highly dependent on the ScoreDA. A deeper analysis of why it fails on its own would strengthen the paper.
3. [Motivation]. The paper started with the motivation for high FPR in the introduction. However, there is not much discussion on FPR rate in the experiments.
4. [Overhead] The framework requires pre-training a teacher model and then training a separate student model. However, the paper does not discuss the total computational overhead (training and inference time) compared to just using the (already strong) teacher model.

### Questions
Please address the weaknesses mentioned above.

### Soundness
2

### Presentation
3

### Contribution
3
