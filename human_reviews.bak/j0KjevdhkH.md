# SIG: Self-Interpretable Graph Neural Network for Continuous-time Dynamic Graphs

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
While graph neural networks have demonstrated potential across various applications, explaining their predictions on dynamic graphs remains largely under-explored. This paper introduces a new research task: self-interpretable GNNs for continuous-time dynamic graphs (CTDGs). We aim to predict future links within dynamic graphs while simultaneously providing causal explanations for these predictions. There are two key challenges: (1) capturing the underlying structural and temporal information that remains consistent across both independent and identically distributed (IID) and out-of-distribution (OOD) data, and (2) efficiently generating high-quality link prediction results and explanations. To tackle these challenges, we propose a novel causal inference model, namely the Independent and Confounded Causal Model (ICCM).  ICCM is then integrated into a deep learning architecture that considers both effectiveness and efficiency. Extensive experiments demonstrate that our proposed model significantly outperforms existing methods across link prediction accuracy, explanation quality, and robustness to OOD data. Our code and datasets are anonymously released at https://github.com/2024SIG/SIG.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper studies the problem of self-interpretable GNNs for continuous-time dynamic graphs (CTDGs), i.e., predict labels while generating a causal subgraph for interpretation. The authors first provide a theoretical analysis of the causal effect, which motivates the proposed causal inference model, the Independent and Confounded Causal Model (ICCM). Based on ICCM, the paper proposes self-interpretable GNN (SIG) to predict labels while offering an explanation. Finally, the paper conducts extensive experiments to demonstrate the effectiveness of the proposed method.

### Strengths
- The paper studies a valuable problem about self-interpretable GNNs for CTDGs.

- The authors conduct thorough experiments to verify the performance of link prediction and explanation capability for SIG.

### Weaknesses
- The paper writing needs to be improved for clarity, e.g., 

&emsp; (1) The goal of the paper is to conduct link prediction while offering an explanation for prediction, the proposed method should be SIG, but it only introduces ICCM for causal analysis in the Abstract.

&emsp; (2) The relationship between SIG and ICCM confuses readers before they understand Sections 4 and 5.

&emsp; (3) A real-world case for confounders in CTDGs should be offered to better understand causal analysis and the design of SIG.

- Confounder mining is done by clustering the representations of edges, but there is a lack of explanation on how this process works.

- In Section 5.4, the proposed method is optimized by maximizing mutual information. Is the goal of this optimization to enable the model to capture causal subgraphs? If so, would maximizing the mutual information between the predicted relevant subgraphs and labels also make the model capture spurious connections, thus having the opposite effect?

- Similar performance can be observed in DIDA (Table 1) and SIG without ICM (Table 8). A deeper analysis is needed to explain this phenomenon, focusing on the similarities between DIDA and SIG without ICM.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel task for CTDGs aimed at predicting labels and generating explanations through causal subgraphs. It presents multiple insights from a causal analysis of SIG, along with an implementation of the method. SIG effectively captures both structural and temporal aspects of the data. Extensive experimental results compare SIG with various methods in the literature, evaluating link-prediction accuracy and explainability. Overall, SIG outperforms the provided baselines in terms of both accuracy and efficiency.

### Strengths
1) The paper addresses critical fields: link prediction, dynamic graphs, and explainability, which hold long-term relevance for both academia and industry.
2) By sharing the code and providing a detailed description of how to run it, the authors enhance the paper's reliability.
3) The authors first present the insights and theoretical foundations of the method, followed by its practical implementation.

### Weaknesses
Although multiple experimental results are provided, there are certain flaws that need to be addressed.

1) Following the sentence, "Regarding the link prediction task, negative samples were set at a ratio of 1:5 in the training set and adjusted to 1:1 in both the validation and test sets" (Line 905), I understand that the test set has been sampled, which should not be the case. If sampling is applied, that will introduce biases into the results. I would expect the authors to conduct testing on the complete set of test edges and provide precision@k (where k can be the number of positive edges or its small multiplier) results to better understand the method's true quality in real-life setting.
2) To avoid bias, I won’t specify paper names, but I believe the method should be compared to studies from 2023 and 2024 at top ML/AI conferences for explainable AI methods for graphs and dynamic graph architectures.
3) The method uses multiple components from Cong et al., which already performs well on the provided link-prediction datasets. I recommend that the authors compare methods under unbiased testing conditions and, if possible, add a more challenging dataset where Cong et al. does not perform well.
4) The experiments need to be run multiple times to see the deviations between runs to test robustness, not just with a single random seed.

### Questions
My main concerns are around the experiment results. I liked the development of the paper and uses of the idea of causal inference model. I'm open to improve my score if the (some) items are provided. If not provided, I would like to hear the explanations from the authors. So to summarize my questions:

1) If the sampling is applied to the testing, can authors provide full-test results (i.e., without sampling test edges), and provide different metric, precision@k where k is the number of positive edges or its small multiplier?
2) Can authors provide more recent baselines for both tasks, link prediction and explainability?
3) Can authors provide more deeper analysis about the differences between Cong et al., and SIG on the experiment results? Can authors find more challenging dataset which can demonstrate SIG's capabilities beyond just building on Cong et al.'s strong performance? 
4) Can authors run their method multiple times (at least 5 times with different seeds) and provide mean/standard deviation across runs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies the self-interpretable GNNs for continuous-time dynamic graphs. The authors propose a Independent and confounded causal model which incorporates independent causal model and confounded causal model. Experimental results demonstrate the proposed method works well on both IID and OOD data.

### Strengths
1. This papers study the model explanation on dynamic graphs, which consists of both temporal information and structure information.
2. The authors consider both IID and OOD setting.
3. The paper is well-written and easy to follow.

### Weaknesses
1. Although the authors claim to provide theoretical analysis, I found no theoretical insight specifically focused on the proposed method.

2. The datasets chosen appear relatively simple, as several baseline methods achieve close to 100% AUC and AP, limiting the scope of meaningful comparison.

3. The baselines included are limited and somewhat outdated. More recent relevant works, such as [1, 2, 3], should be considered to provide a more comprehensive evaluation.

4. For the temporal causal subgraph extraction, the authors only extract most recent temporal edges, neglecting the graph's evolution over time and potentially overlooking key historical patterns that inform causal relationships.

[1] Wang, Tianchun, et al. "DyExplainer: Explainable Dynamic Graph Neural Networks." arXiv preprint arXiv:2310.16375 (2023).
[2] Wang, Jie, et al. "Trustguard: Gnn-based robust and explainable trust evaluation with dynamicity support." IEEE Transactions on Dependable and Secure Computing (2024).
[3] Liang, Guojun, et al. "Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatiotemporal Forecasting." IEEE Transactions on Neural Networks and Learning Systems (2024).

### Questions
1. What is the difference between discrete-time dynamic graphs  and continuous time dynamic graphs? The  continuous time  can be divided into several discrete steps.

2. How to define the ground truth for an explanation subgraph?

### Soundness
2

### Presentation
3

### Contribution
2
