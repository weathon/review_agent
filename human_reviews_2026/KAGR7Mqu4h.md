# Unified and Efficient Multi-view Clustering from Probabilistic Perspective

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Multi-view clustering aims to segment the view-specific data into the corresponding clusters. There have been a large number of works for multi-view clustering in recent years. As representive methods in multi-view clustering, works built on the graph make use of a view-consistent and discriminative graph while utilizing graph partitioning for the final clustering results. Despite the achieved significant success, these methods usually construct full graphs and the efficiency is not well guaranteed for the multi-view datasets with large scales. To handle the large-scale data, multi-view clustering methods based on anchor have been developed by learning the anchor graph with smaller size. However, the existing works neglect the interpretability of multi-view clustering based on anchor from the probabilistic perspective. These methods also ignore analyzing the relationship between the input data and the final clustering results based on the assigned meaningful probability associations in a unified manner. In this work, we propose a novel method termed Unified and Efficient Multi-view Clustering from Probabilistic perspective(UEMCP). It aims to improve the explanation ability of multi-view clustering based on anchor from the probabilistic perspective in an end-to-end manner. It ensures the consistent inherent structures among these views by learning the common transition probability from data points to categories in one step. With the guidance of the common transition probability matrix from data points to categories, the soft label of data points can be achieved based on the common transition probability matrix from anchor points to categories in the learning framework. Experiments on different challenging multi-view datasets confirm the superiority of UEMCP compared with the representative ones.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a novel method termed Unified and Efficient Multi-view Clustering from Probabilistic perspective (UEMCP). It aims to improve the explanation ability of multi-view clustering based on anchor from the probabilistic perspective in an end-to-end manner. It ensures the consistent inherent structures among these views by learning the common transition probability from data points to categories in one step. With the guidance of the common transition probability matrix from data points to categories, the soft label of data points can be achieved based on the common transition probability matrix from anchor points to categories in the learning framework.

### Strengths
The proposed method, which proposes a novel method termed Unified and Efficient Multi-view Clustering from Probabilistic perspective. It aims to improve the explanation ability of multi-view clustering based on anchor from the probabilistic perspective in an end-to-end manner. The soft label of data points can be achieved based on the common transition probability matrix from anchor points to categories in the learning framework.

### Weaknesses
1. The authors summarize the advantages of the proposed method based on the Adaptive weighted, Probabilistic and Consistent property. However, there is no further explanation for Probabilistic property in illustrating the advantages of the proposed UEMCP. The authors should add the related further explanation for Probabilistic property in the paper.
2. The second best results in Tables for the experiment part can be highlighted to make the authors more obviously grasp the clustering performance gains.
3. The authors do not give detailed analysis regarding the convergence study of the proposed UEMCP. More analysis can be given for the convergence study part in the experiment, i.e., the iteration speed for the proposed UEMCP in the paper.
4. The authors study the influence of the anchor number for the final clustering results in the experiment. They vary the anchor number in the range of {2k, 3k, 4k, 5k, 6k} and investigate the sensitivity in terms of different metrics, where k is the number of clusters in the dataset. The authors are expected to give the reason why they choose {2k, 3k, 4k, 5k, 6k} as the range in the experiment.
5. The authors are expected to adjust the size of the presentation for Algorithm 1 of the proposed method in the paper.

### Questions
The authors are expected to give the reason why they choose {2k, 3k, 4k, 5k, 6k} as the range in the experiment for the proposed method.

### Soundness
3

### Presentation
3

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
This paper proposes a method termed Unified and Efficient Multi-view Clustering from Probabilistic perspective (UEMCP), which assigns the probabilistic meaning to the anchor graph and soft label of data points to increase the explanation ability of multi-view clustering model in an end-to-end manner. UEMCP learns the common transition probability from data points to categories shared by multiple views with one step, which is able to ensure the consistency of inherent structures for these views. Experiments are performed on several datasets to validate the effectiveness and efficiency of the proposed UEMCP under different metrics

### Strengths
Most existing multi-view clustering methods for large-scale data usually rely on anchor graphs to reduce the algorithmic complexity. The structure of the data can be represented by choosing $ m $ anchors to reflect the entire distribution for the dataset. The relation between data points and anchors can be built based on the anchor graph and the correlation tends to be stronger when a data point and anchor belong to the same category. Besides, a consensus data distribution shared by different views is assumed in our multi-view clustering setting. The anchor graph can be considered as the probability transition matrix between data points and anchors due to the non-negative properties and summarization being one for the row. The anchor selection and the probability transition matrix construction are separated from each other, which will inevitably affect the final performance. Different from the traditional strategy, the authors automatically learn anchors instead of simple sampling

### Weaknesses
. To study the influence of parameter $ \lambda $ on the final clustering performance, the authors perform experiments for investigation in the range of $ \{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1\} $. According to Fig. 1, the authors find that choosing the proper parameter is crucial in improving the final clustering results. The parameter $ \lambda $ with too large or small values is not helpful in achieving the desired performance. However, the authors do not give detailed reason why choose $ \{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1\} $ as the range in performing experiments for investigation.
2. The authors present the clustering results of the proposed UEMCP and compared methods on seven benchmark datasets. 'N/A' is adopted to show that the method encounters the out-of-memory issue on the dataset. According to Tables 2-5, we can draw conclusions. However, the authors do not bold the best clustering results in Tables 2-5 for the experiment part. Therefore, the authors are expected to bold the best clustering results in Tables 2-5 for the experiment part.
3. To validate the computational efficiency, we list the running time of our UEMCP on different datasets. As shown in Fig. 4, we observe that UEMCP needs relatively shorter running time on different datasets, which demonstrates the computational efficiency of our method. Though less running time is needed by BMVC, the simple procedure ignores to fully explore the information from multiple views, leading to relatively poor performance. Considering that the running time analysis is given in this paper as shown in the above, the authors are expected to list the memory of the adopted device in performing the experiment.
4. In the reference part, some publication name is called for short, i.e., Inf. Sci., and some publication is called for the whole name, i.e., Proceedings of the AAAI Conference on Artificial Intelligence. It is observed that these manners of publication names are not consistent. Then the authors should correct this issue and ensure that the forms of the references in this paper are consistent.

### Questions
1.The authors are expected to give detailed reason why choose $ \{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1\} $ as the range in performing experiments for investigation.

2.The difference from existing jobs, especially those related to probability transitions, is not very prominent, It is recommended to supplement relevant work, highlight the differences of the paper

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a novel anchor-based multi-view clustering method, UEMCP. The primary goal is to improve the interpretability of anchor-based clustering by framing it from a probabilistic perspective within an end-to-end learning model. The method aims to ensure structural consistency across views by learning a common transition probability matrix from data points to cluster categories in a single step.

### Strengths
The paper proposes an end-to-end framework to enhance the explanation ability of anchor-based multi-view clustering, which is a valuable goal. The method is designed to ensure the consistency of inherent structures across views. A key aspect is the use of a common transition probability matrix from anchors to categories, which guides the soft label assignment for data points within the learning framework. The experimental results on several datasets are strong and show a clear improvement over existing methods.

### Weaknesses
1.  The methodology is plagued by confusing and inconsistent notation. For example, in Eq.1, $P_v$ should be $d_v \times d$, but its constraint is $P_v^T P_v = I_m$ (it should be $I_d$). The variable $S$ is defined with different dimensions in Eq.1 ($m \times n$) and Eq.2 ($l \times n$). Similarly, $H$ has conflicting dimensions in Eq.2 ($c \times l$) and Eq.3 (implying $c \times m$). The dimension $l$ is introduced in Eq.2 without definition. These issues severely affect readability.
2. The paper presents a running time analysis, but the memory used for the experiments are not provided. Besides, the discussion of the running time is very brief. A more detailed comparison and analysis against the other methods are needed.
3. The formatting of the references is inconsistent and should be carefully checked throughout the paper.

### Questions
Could the authors provide a more detailed analysis of the running time results shown in Figure 4? Specifically, please elaborate on the comparative efficiency of UEMCP against the baselines.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows a new method termed Unified and Efficient Multi-view Clustering from Probabilistic perspective (UEMCP), which assigns the probabilistic meaning to the anchor graph and soft label of data points to increase the explanation ability of multi-view clustering model in an end-to-end manner. UEMCP is able to learn the common transition probability from data points to categories shared by multiple views with one step, which ensures the consistency of inherent structures for these views.

### Strengths
The authors propose a novel method termed Unified and Efficient Multi-view Clustering from Probabilistic perspective for increasing the explanation ability of multi-view clustering based on anchor from the probabilistic perspective. Besides, the expression is very clear with satisfied writing and novelty. With the guidance of the common transition probability matrix from anchor points to categories in the learning framework, the soft labels of data points are able to be achieved.

### Weaknesses
1. The authors can give the brief operations in Optimization part for Section 3.2. Then the main idea for optimization is more clear for readers.

2. To study the influence of parameter λ on the final clustering performance, the authors perform experiments for investigation in the range of {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1}. The authors should explain why they choose such range in parameter selection.

3. The authors can give more details for running time analysis part in this work, i.e., the reason why the proposed UEMCP needs relatively less time in the experiment.

4. The authors should confirm the typo error and check the whole paper to avoid such issues for the paper.

### Questions
See the Weakness box.

### Soundness
3

### Presentation
3

### Contribution
3
