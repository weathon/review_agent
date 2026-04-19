# EPIC: Compressing Deep GNNs via Expressive Power Gap-Induced Knowledge Distillation

- Decision: Reject
- Scores: 5, 5, 3

## Abstract
The teacher-student paradigm-based knowledge distillation (KD) has recently emerged as a promising technique for compressing graph neural networks (GNNs). Despite the great success in compressing moderate-sized GNNs, distilling deep GNNs (e.g., with over 100 layers) remains a tough challenge. A widely recognized reason is the *teacher-student expressive power gap*, i.e., the embeddings of a deep teacher may be extremely hard for a shallow student to approximate. Besides, the theoretical analysis and measurement of this gap are currently missing, resulting in a difficult trade-off between the needs of being "lightweight'' and being "expressive'' when selecting a student for the deep teacher. To bridge the theoretical gap and address the challenge of distilling deep GNNs, we propose the *first* GNN KD framework that quantitatively analyzes the teacher-student expressive power gap, namely **E**xpressive **P**ower gap-**I**ndu**C**ed knowledge distillation (**EPIC**). Our key idea is to formulate the estimation of the expressive power gap as an embedding regression problem based on the theory of polynomial approximation. Then, we show that the minimum approximation error has an upper bound, which decreases rapidly with respect to the number of student layers. Furthermore, we empirically demonstrate that the upper bound exponentially converges to zero as the number of student layers increases. Moreover, we propose to select an appropriate value for the number of student layers based on the upper bound, and propose an expressive power gap-induced loss term to further encourage the student to generate embeddings similar to those of the teacher. Experiments on large-scale benchmarks demonstrate that EPIC can effectively reduce the numbers of layers of deep GNNs, while achieving comparable or superior performance. Specifically, for the 1,001-layer RevGNN-Deep, we reduce the number of layers by 94\% and accelerate inference by roughly eight times, while achieving comparable performance in terms of ROC-AUC on the large-scale benchmark ogbn-proteins.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper theoretically and experimentally analyzes the expressive power gap between deep GNN teachers and lightweight students. It formulates the estimation of the expressive power gap to an embedding regression problem. The results on three large-scale datasets demonstrate the effectiveness of the proposed method. Overall, the paper is well written with an interesting topic.

### Strengths
- The analysis of graph knowledge distillation from the perspective of expressive power is novel.
- The paper is well written and presented, especially about the notation and background.
- EPIC outperforms 1001-layer GNNs with a 4-layer GNN.

### Weaknesses
- My main concern is the motivation for the paper. Although deep GNNs have been a popular research topic, deep GNNs are prone to suffer from overfitting and over-smoothing and so far have not been widely used as shallow GNNs. I am confused about the significance of studying distillation for deep GNN.
- The authors did not conduct their experiments in a transductive setting as GLNN and NOSMOG did.
- Why does the performance of NOSMOG drop so much in Table 1? According to their original paper, NOSMOG can achieve performance even better than the teacher GNN.
- Can the authors provide more results on the different layers of teacher GNN in Table 1? In particular, I would like to know how generalizable EPIC is in a shallow teacher GNN setting?
- As a GNN-to-GNN distillation method, the authors mainly compare EPIC with two GNN-to-MLP methods (GLNN and NOSMOG) in their experiments. Comparisons with some SOTA GNN-to-GNN distillation methods are missing, such as those introduced by the authors in related work.

### Questions
Please explain (3-5) in the weakness part.
How about the time complexity (especially the bound estimation) and running time?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the knowledge distillation of deep GNNs, such as that with over 100 layers. To address the issue of the representation ability gap between teacher and student networks, this paper proposes a framework, EPIC, leveraging the embedding regression based on the theory of polynomial approximation. Specifically, EPIC first shows the EPIC bound exponentially converges as the number of student layers increases with experiments. Then, with this observation, it selects an appropriate layer number of the student network by analyzing the EPIC bound.  Furthermore, an expressive power gap-induced loss term is proposed to reduce the gap. In the experiments, it reduces 94% of layers, improves 8x speed, and obtains comparable performance for 1,001-layer RevGNN-Deep.

### Strengths
1. The field of distilling a very deep GNN model is less explored. By distilling, the inference speed can be improved by a large margin, which benefits the potential applications.
2. A theoretical framework is proposed to analyze the gap between the student and teacher network from a spectral perspective. 
3. The theoretical results with an EPIC bound are used to select the appropriate layer number. It is interesting and meaningful to reduce the cost of the hyper-parameter tuning.

### Weaknesses
The main concerns lie in the method details and experimental settings. The details are shown below:
1. The hyper-parameter $\gamma$ in Eq. (8) weakens (somehow) the theoretical results. The optimal # layers $M^*$ are dependent on this hyper-parameter, regardless of the EPIC bound. How to choose $\gamma$, is there any guidance?   Does it involve the extra hyperparameter tuning cost?  When comparing tuning M (layer number) and $\gamma$, what is the advantage of the latter?
2. The scalability is also a concern, since it requires eigenvalue decomposition to find out all the eigenvalues, incurring high memory and time complexity, especially for large-scale graphs, which is the setting in this work. It is suggested to add some experimental cost to the process of calculating EPIC bounds. 
3. The singular value decomposition (SVD) and Laplacian requires the assumptions of linearity.  I understand that this may be easier for derivation. However, it still would be helpful if you could explain why this linear-based theory could be used for non-linear graph models, such as GCN and GAT.  It seems that the used teacher model is also non-linear. 
4. Novelty concerns on the EPIC loss.  The paper claims the new EPIC loss (expressive power gap-induced loss). However, If I understand correctly, H_S and H_T are representations of student and teacher models respectively. It seems the same as standard knowledge-distilling loss where the embeddings in the student network mimic the ones in the teacher network. It is suggested to mention this and avoid the new terms if they are similar. Otherwise, it is better to clarify the differences and provide corresponding evidence (insights and experimental results.)
5. There are some concerns in the experiment design. First, the comparison needs to be more fair. As for GLNN and NOSMOG, the student model is MLP, while for EPIC, the student model is a GNN, the performance gain may be caused by the enhanced expressivity of student models. Second, this experiment can't be used to verify the correctness of the proposed theorem (i.e., is this bound tight when deciding the # of layers in student GNNs). Do the results of the real experiments align with the curves in Figure 1?  I suggest redesigning this experiment to stress the correctness of the proposed theorem. Third, It is suggested to add baselines that train the GNN with fewer layers from scratch.  Again, the cost of hyper-parameter $\gamma$ is an issue. 
6. (Minor), It is suggested to refine Figures 1 and 2.  For example, it is better to provide more details in Figure 2. Currently, the details of EPIC and Bound and EPIC loss are lost.

### Questions
1) Regard the $\gamma$. How to choose $\gamma$ in the experiments? Does it involve the extra hyperparameter tuning cost (How much)?  When comparing tuning M (layer number) and $\gamma$, what is the advantage of the latter?

2) What is the experimental cost of the calculating process of EPIC bounds?

3) Could you please explain why the proposed linear-based theory could be used for non-linear graph models?

4) What is the unique contribution of the proposed EPIC loss?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The author(s) theoretically demonstrate that the shallow student model lacks sufficient expressive power to mimic the teacher;
hence, distilling deep GNNs remains a tough challenge.
The derived upper bound allows for quantitative analysis of the gap, making it easy to determine an appropriate number of layers for student models.

### Strengths
1. Distilling very deep GNNs is challenging and important. 
2. This work provides some valuable suggestions on layer number selection.

### Weaknesses
1. The conclusion obtained from Theorem 1 is obvious, and the upper bound derived therein appears difficult to apply in practice.
While the hyperparameter $\gamma$ can achieve this to some extent, the value of $\gamma$ is not directly related to classification performance gap. Therefore, a search is also required to determine the appropriate value.
2. The complexity of computing EPIC bounds should be analyzed (since SVD for large-scale graphs is expensive).
3. Feature distillation (namely the EPIC loss presented in this paper) is a well-known technique.

### Questions
Please refer to Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
