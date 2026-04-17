# Geometric properties of neural multivariate regression

- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
Neural multivariate regression underpins a wide range of domains such as control, robotics, finance, and meteorology, yet the geometry of its learned representations remains poorly characterized. While neural collapse has been shown to benefit generalization in classification, we find that analogous collapse in regression consistently degrades performance. To explain this contrast, we analyze models through the lens of intrinsic dimension. Across synthetic benchmarks and control tasks, we estimate the intrinsic dimension of last-layer features ($ID_H$) and compare it with that of the regression targets ($ID_Y$). Collapsed models exhibit $ID_H < ID_Y$, leading to over-compression and poor generalization, whereas non-collapsed models typically maintain $ID_H > ID_Y$. Moreover, the gap between IDH and IDY reliably predicts test error. From these observations, we identify two regimes—over-compressed and under-compressed—that determine when expanding or reducing feature dimensionality improves performance. Our results provide new geometric insights into neural regression and suggest practical strategies for enhancing generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on neural collapse in neural multivariate regression. The neural collapse is defined by NRC1 metric which is measuring the remaining signal of the last layer features from the projection onto the top principal components.  In the experiments the authors observe that intrinsic dimension of the last layer features is smaller than the intrinsic dimension of the labels in collapsed models whereas in the non-collapsed models it is the other way around. They claim that poor perfromance is observed in collapsed models while the performance in non-collapsed models depend on the data quality and the dataset size. Authors also provide emprical suggestions on last layer dimension based on the intrinsic dimension of the last layer and data quality. This paper also suggests that this observation of collapsed models having poor generalization is completely the opposite observation of previous work on neural collapse in classification tasks.

### Strengths
First, this paper shows that there is an important correlation between the NRC1 metric and the intrinsic dimension as stated in the Fig.4. Even though this is very expected behavior, this phenomenon is analyzed in the regression problems for the first time as authors suggested. Moreover, the behaviour of the intrinsic dimension in different layers in the training process (Fig.5) is analyzed well. Especially, providing the relation between IDP vs IDY for different settings (collapsed and non-collapsed) is good. In addition, the results in Fig.6, are very clean and shows interesting results like U-shape in (b)(e)(h) which actually unlocks new questions regarding the intrinsic dimension of the last layer and the intrinsic dimension of the labels.

### Weaknesses
I think the main weakness is that the paper states that NRC1 metric and the intrinsic dimension of the last layer (IDH) is highly correlated. However, as far as I understand, this is very expected. As the intrinsic dimension is something similar to effective-rank, even if it is not a linear metric. NRC1 directly calculates how much noise is remain if you remove the first principal components. So, as far as I understand, by their definitions, they should be measuring very similar concepts. So, their correlation provided in Fig.4 is not suprising at all. 
In addition, some of the conclusions are very expected and do not contribute to the field. For example, when ID_H < ID_Y, it is also expected that the final layer has no enough representative power to learn the labels.

### Questions
1 - By their definitions of ID and NRC1, I feel that their correlation provided in Fig.4 is not suprising. I feel that they are two different approaches to measure the same phenomenon. What makes it interesting in regression tasks ? 

2 - In the paper, it is stated that when ID_H >= ID_Y , the test error depends on noise and dataset richness. And Fig.6, (b,e,h) are given as examples of low-data tasks and noisy-target tasks, while (k,n,q) are given as examples of high-data tasks or low-noise tasks. How do you say these are low-data tasks or noisy target tasks ? Is it just intuition or is there a clear definition ?
 
3 - I might have missed but I did not see the definition of ID_P in the paper even though it can be understood in the analysis of Fig.5, it is good to provide a definition.

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
The current paper studies neural collapse, i.e., the "degeneracy" of last layer features, for multi-variate regression. Previously, a similar phenomenon was studied for classification and it was shown that the last layer features tightly cluster around class centroids. The current paper studies the problem from the angle of intrinsic dimension (ID), and presents empirical results of models trained with regularized least squares loss (by weight decay). By varying the regularization strength, models of different capacity are obtained and they lead to different behavior on two neural collapse metrics, NRC1 and ID. The paper shows that ID is a better metric for regression, ID is reduced during training and decreases with depth. The paper discusses correlations between these metrics and model generation, and suggests the critical regime being $ID_H=ID_Y$, i.e., the ID of last layer features need to be at least the ID of targets.

### Strengths
The ideas are very intuitive and the empirical observations match the intuitions.

### Weaknesses
My main concern is that, while the findings are interesting and intuitive, the paper lacks more rigorous analysis. The only theorem of the paper states that a collapsed model can not fit the data well. But this does not answer why collapse happened, and why ID is reduced during training (and maybe at what speed). Compared with most cited work on neural collapse for classification (accepted at top ML venues), the current paper is somewhat lacking in theoretical depth.

### Questions
1. I hope the authors can address the main weakness mentioned above.

2. The current paper relied on weight decay to vary the capacity of the model. My understanding is that weight decay is not very often used in practice currently. Are there other interesting observations regarding neural collapse when weight decay is not used?

### Soundness
3

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
3

### Summary
This paper studies the geometric properties of neural multivariate regression. Based on the experiments on regression tasks, the submission explores the relations between the Neural Regression Collapse metric and the estimated intrinsic dimension via 2 nearest neighbors.

### Strengths
1. It is interesting to investigate the geometric properties of neural multivariate regression.

2. The conclusion seems intuitive.

### Weaknesses
1. From Figure 1, we observe the relations between the weight decay parameter and the collapse degree. What is the deeper explanation regarding this observation?

2. In the experiments, only two kinds of datasets with modest sizes are used. In order to draw a solid conclusion from experiments, it would be better to test more kinds of tasks with varying sizes.

3. The intrinsic dimensions of $Y$ in the experiments are small. It is interesting to explore if the same conclusion can be drawn from high dimensional situations.

4. It seems that the vision-based regression tasks are constructed on classification datasets. It is encouraging to test the regression vision tasks.

5. In general, the conclusion of the submission is intuitive. However, the deeper theoretical analysis is missing. 

6. The practical guideline is vague. How to increase or reduce the intrinsic dimension of the embeddings? In addition, the ideal $ID_H$ is related to $ID_Y$. However, it is difficult to accurately estimate $ID_Y$ in practice, making it challenging to determine if one should increase $ID_Y$ or not.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper examines the geometry of representations in neural multivariate regression and investigates why neural collapse—beneficial in classification—often degrades regression performance. It begins with the Neural Regression Collapse (NRC1) metric, which quantifies how tightly last-layer features concentrate in an $n$-dimensional linear subspace ($n$ here is the number of targets in multivariate regression). Although low NRC1 correlates with higher test error, the paper shows that NRC1 is limited because it captures only linear collapse and misses non-linear reductions in representational degrees of freedom. To address this, the authors estimate the intrinsic dimension $ID_H$ of the learned features and compare it with that of the targets $ID_Y$. They find that collapsed models satisfy $ID_H < ID_Y\, $ , indicating over-compression and poor generalization, while non-collapsed models maintain $ID_H \ge ID_Y$. The study thus argues that the intrinsic dimension provides a more faithful geometric measure than NRC1 for understanding regression generalization.

### Strengths
Well-executed and engaging study: I enjoyed reading this paper. It presents a careful and comprehensive empirical investigation of representation geometry in neural multioutput regression, combining the NRC1 metric with intrinsic dimension analysis across diverse tasks. The experiments are clearly presented, and the findings—linking over-compression $ID_H < ID_Y$ to poor generalization—are convincing and well-supported.

See the summary.

### Weaknesses
I do not see any major weaknesses in the study itself. I have minor issues with the presentation. Especially, the intrinsic dimension should be part of the main text, with enough intuition. Based on the current version, I was not sure what I was supposed to think when I was trying to understand Figure 4. Perhaps this quantity is standard, but I am not familiar with it, and since it constitutes a significant part of the study, it should have been introduced more thoroughly.

### Questions
The main questions I have are:
(1) Could you precisely walk me through the definition and intuition with intrinsic dimension and walk me through it? I am not familiar with prior work that introduces this quantity. I tried to understand from Appendix C, but I still lack a clear intuition of what this estimate captures.
(2) Any limitations and future work of the study that the authors see?

### Soundness
3

### Presentation
2

### Contribution
3
