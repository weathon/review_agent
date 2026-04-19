# Transfer Learning Under High-Dimensional Graph Convolutional Regression Model for Node Classification

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 6, 6

## Abstract
Node classification is a fundamental task, but obtaining node classification labels can be challenging and expensive in many real-world scenarios.  Transfer learning has emerged as a promising solution to address this challenge by leveraging knowledge from source domains to enhance learning in a target domain.  Existing transfer learning methods for node classification primarily focus on integrating Graph Convolutional Networks (GCNs) with various transfer learning techniques. While these approaches have shown promising results, they often suffer from a lack of theoretical guarantees, restrictive conditions, and high sensitivity to hyperparameter choices. To overcome these limitations, we employ a Graph Convolutional Multinomial Logistic Lasso Regression (GCR) model which simplifies GCN, and develop a transfer learning method called Trans-GCR based on the GCR model. We provide theoretical guarantees of the estimate obtained under the GCR model in high-dimensional settings. Moreover, Trans-GCR demonstrates superior empirical performance, has a low computational cost, and requires fewer hyperparameters than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed a novel transfer learning approaches to tackle transfer learning on graph domain data. Specifically, according to the SGC in 2019, the author remove the non-linearity and reformulate the GCN to be GRC model which first aggregate the graph features using symmetric normalized adjacency and then feed it into multinomial logistic lasso regression model assuming a linear relation between graph features and labels. Loss is a commonly used L1 regularized negative log likelihood loss for sparse coefficient learning. For input, pooling is used to combine source and target domain data for training. The training stage involves estimation of source domain coefficient and estimation of domain shift. Finally, estimation of target domain coefficient is learned through the addition of source domain coefficient and domain shift. Additionally, the paper provides a simple way to evaluate a score for each source domain dataset to select dataset that are close related to the target domain data.
For theoretical analysis, the author provides a theoretical analysis on the gap between the estimation and the true target coefficient with several assumptions.

### Strengths
1. Overall the presentation and logic is clear and sound. I find most part easy to understand and follow.
2. The model removes non-linearity and therefore is generally efficient and computationally inexpensive, it is more like a simple machine learning model than a generally deep learning approach.
3. The theoretical analysis provides a good estimate on how well the model can achieve given the number of nodes, the dimensionality of the feature, and the sparsity of the learned coefficient. Assuming the theorem is proper suitable for common cases, the high dimensionality problem seems to be alleviated with log terms.
4. The paper provides experiments on both simulated data and real world dataset.

### Weaknesses
1. The paper mentions several previous transfer learning approaches in GNN in the introduction, However when compared in the baseline in real datasets, only two adaptive based approach has been used and the rest is naive transfer learning with different GNN models. As the problem is focused on transfer learning scheme, It therefore doesn't seem to be convincing that the approach is fairly compared with existing STOA.
2. In the experiment, the hyperparameter in the proposed method is selected through cross validation, but the other methods are fixed with the same hyperparameter settings. For many GNN methods, performance is sensitive to the selection of the hyperparameters, the compared result is therefore not a rather fair comparison from my understanding. When checking on the sensitive analysis on the lambda and M, it is shown that the results in the proposed model have moderate fluctuations with different values.

### Questions
1. For the theoretical analysis, how do we obtain the assumption 4.3 (sparsity), is it assumed directly from the conclusions and why is it equal to O(1)?
2. If the author could provide a comparison results with baselines and the averaged results of the proposed results for different hyperparameter settings with some more recent transfer learning baselines on GCN, I think the experimental results will be much more convincing.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper studies transfer learning for node classification using a so called Graph Convolutional
Multinomial Logistic Lasso Regression (GCR). Experiments on limited datasets are conducted to show the performance of GCR.

### Strengths
1. The presentation of the proposed method is clear.

2. Assumptions based on which GCR and its theoretical results are derived are presented clear.

### Weaknesses
There are several major concerns for this paper.

1. GCR is based on an indeed very strong assumption, that is, there is a linear relationship between aggregated features and labels. There is an obvious concern that features clearly depends on the GCN architecture and its training process, and it is risky to assume that there is linear relationship between the features obtained by a particular GCN architecture and its training process without clear theoretical or empirical study.

2. The assumptions 4.1-4.3 for the theoretical results in Section 4 are particularly restrictive and some of them can hardly hold in practice. For example, Assumption 4.1 needs to the node attributes to follow sub-gaussian distribution, which are often not the case in real data including the real data used by this paper in the experiments. For another example, when can the sparsity parameter $s$ satisfy the particular condition in line 227-228? Furthermore, how sharp is the bound in Theorem 4.4, and how does it compare to the literature? Without a clear comparison to prior art, the significance of Theorem 4.4 is unclear.

3. Experiments are very limited, and the real graph data used in this paper are all small graphs. Experiments on graphs of much larger scale are expected to justify the effectiveness of GCR.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of node classification, where labeling nodes can be costly. The authors propose Trans-GCR, a transfer learning method using a simplified Graph Convolutional Regression (GCR) model. Unlike existing methods, Trans-GCR offers theoretical guarantees, performs well empirically, and is computationally efficient with fewer hyperparameters, making it suitable for practical use in complex graphs.

### Strengths
1. **Theoretical Convergence Analysis:** The authors provide a convergence analysis, offering theoretical guarantees for the convergence.

2. **Simplicity and Efficiency:** The model’s structure is both simple and fast, enhancing accessibility and scalability without compromising performance.

3. **Innovative Transfer Learning Framework:** The paper introduces a focused transfer learning framework for deep graph learning, addressing a significant gap in handling graph-structured data with limited labels, which can benefit many practical applications.

### Weaknesses
1. **Triviality of Convergence Proof**: The theoretical proof for convergence appears trivial. Although the paper claims that $Z$ is non-i.i.d., Assumption 4.1 indicates that $X$ is i.i.d., leading to a clear covariance matrix for $Z$ as $A^\top A / np$. Following Modification 1 in Appendix C, previous theorems can be trivially adapted with straightforward modifications. Thus, the convergence proof lacks substantial novelty and rigor.

2. **Limited Parameterization**: According to Equation 2.2, it seems that only the final logits layer contains parameters $\beta$, while the preceding $S^M X$ lacks parameters. The absence of parameters in these earlier layers raises concerns about why only the last layer is parameterized, which could lead to over-smoothing due to unparameterized iterations of $S X$ and consequently limit the model’s expressiveness.

3. **Basic Transfer Learning Approach**: The transfer learning method employed, a simple $\delta$ fine-tuning, appears overly basic. There is little exploration of alternative, established methods in transfer learning or meta-learning that could potentially enhance the model’s adaptability and robustness.

4. **Issues in Hyperparameter Sensitivity Testing**: The sensitivity experiments on hyperparameters are limited. For instance, in the $\lambda$ experiment, the model fails to achieve the optimal solution seen at $M=5$. Additionally, the range of $\lambda$ tested is narrow; a broader, exponential scale (e.g., 0.01, 0.001, 0.0001) would provide a more comprehensive understanding of the model’s sensitivity.

5. **Lack of Notational Clarity**: The notation lacks clarity and could benefit from a dedicated section outlining all definitions. Many symbols, such as $X_j$, are undefined in Appendix A. A coherent notation guide would improve readability and help readers follow the technical details more effectively.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of node classification in high-dimensional settings, particularly in scenarios with limited labeled data. It introduces a novel transfer learning method called Trans-GCR, based on a Graph Convolutional Multinomial Logistic Lasso Regression (GCR) model. The GCR model assumes that classification labels depend on graph-aggregated node features followed by a multinomial logistic regression, allowing for effective handling of high-dimensional features. The authors highlight the limitations of existing GCN-based transfer learning methods, which often lack theoretical guarantees, are sensitive to hyperparameters, and impose restrictive conditions. In contrast, Trans-GCR provides theoretical guarantees under mild conditions and demonstrates superior empirical performance with a lower computational cost. The method requires only two hyperparameters, making it more accessible for practical applications.

### Strengths
1. The paper is well-written and presents its ideas clearly.
2. The proposed Trans-GCR method appears to be a sensible solution to the challenges of graph transfer learning, addressing the issue of node classification effectively. Additionally, the authors provide theoretical guarantees for their method under specific conditions, enhancing the robustness of their approach.

### Weaknesses
1. Some parts of the paper are confusing. In line 260, the authors state that $p$ is the expected degree, but according to the definition in Assumption 4.2, the network connectivity parameter $p$ should be a probability. This inconsistency may leads to misunderstanding.
2. I find the condition $p \log d \to 0 $ as $ n \to \infty $ in Assumption 4.2 somewhat perplexing. It would be helpful for the authors to explain why the feature dimension $d$ is relevant when defining network connectivity.
3. Since one of the contributions of the paper is providing theoretical guarantees, and the authors have made several modifications to the proof approach compared to previous work, a brief sketch of these changes would be helpful for readers to understand the arguments better.
4. On synthetic data, since the true $\beta$ values and sparse patterns are known, using MSE as the evaluation metric is insufficient. It would be more comprehensive to consider additional metrics such as True Positive Rate (TPR) and False Discovery Rate (FDR) to assess the model's performance.

### Questions
1. Please address the concerns mentioned in the Weaknesses.

2. Is Assumption 4.3 a technical condition? Specifically, does the term $s \frac{\log d}{n} \times \log d \times \psi(p)$ provide additional insight compared to the condition $s \frac{\log d}{n}$?

3. What are the technical challenges in proving results for the SBM in the context of this paper compared to the ER graph? Specifically, under conditions similar to Assumption 4.2, given that $p, q = \omega(\log n / n)$, what complexities arise in this setting?

### Soundness
2

### Presentation
2

### Contribution
2
