# Fed-REACT: Federated Representation Learning for Heterogeneous Time Series Data

- Avg Score: 4.25
- Decision: Reject
- Scores: 6, 3, 3, 5

## Abstract
Motivated by high resource costs and privacy concerns that characterize centralized machine learning, federated learning (FL) emerged as an efficient alternative that allows the participating clients to collaboratively train global model while keeping their data local.
In practice, distributions of clients' data vary over time and from one client to another, creating heterogeneous conditions that deteriorate performance of conventional FL algorithms. In this work, we study an FL framework where clients train on heterogeneous time series data and introduce to these settings Fed-REACT, a novel federated learning method leveraging representation learning and evolutionary clustering. The algorithm consists of two stages: (1) in the first stage, the clients learn a model that extracts meaningful features from local time series data; (2) in the second stage, the server adaptively groups clients into clusters and coordinated cluster-wise learning of task (i.e., post-representation) models for local downstream tasks, e.g., classification or regression. We demonstrated high accuracy and robustness of the proposed algorithm in experiments on real-world time series datasets, and provided theoretical analysis of its performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel federated learning framework designed to address the challenges of heterogeneous time series data across multiple clients. The proposed method, Fed-REACT, consists of a two-stage learning process: the first stage focuses on self-supervised learning to extract meaningful features from local time series data, while the second stage employs evolutionary clustering to adaptively group clients and train task models that reflect data heterogeneity. The authors present an adaptive evolutionary clustering approach that enhances stability by incorporating both current and historical model weights. Through theoretical analysis and experimental validation on given datasets, the paper demonstrates that Fed-REACT improves model performance compared to existing methods.

### Strengths
1. The paper is well-structured and well-written, making it easy to understand.
2. The paper provides a novel approach that combines representation learning and evolutionary clustering.
3. The paper also gives a theoretical analysis of the proposed method, including convergence properties and a global regret function.
4. The authors include a thorough experimental analysis on the given datasets demonstrating the effectiveness of Fed-REACT.

### Weaknesses
1. While the paper discusses the necessity of federated learning in non-IID settings, it does not include detailed experimental results. Only results on one dataset do not help generalize the performance of the method over non-iid.
2. A detailed privacy analysis of the proposed method is missing which is very important in a federated setting.
3. A more thorough literature review would be nice, including state-of-the-art methods in time series federated learning

### Questions
1. How does the proposed method perform in non-IID data distributions among various clients (include more datasets)?
2. Can you provide a detailed privacy analysis of the proposed methods, and how do you plan to address this important aspect in the context of federated learning?

### Soundness
2

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
2

### Summary
This paper investigates the federated representation learning problem for time series data. A new method based on evolutionary clustering is proposed to handle the problem. Theoretical analysis is given to discuss the properties of the proposed method. Experiments are conducted to show the effectiveness of the proposed method.

### Strengths
[1] Sufficient theoretical analysis is provided, which provides a solid guarantee of the usefulness of the proposed method. 

[2] The idea of balancing the contribution of structure and feature information makes sense, especially when the missing situation is unknown or uncertain.

### Weaknesses
[1] Experiments are carried out on only two datasets. More datasets should be considered to lead to comprehensive performance results.

[2] There should be more discussion on the unique challenges in time series representation learning and how existing federated learning/federated representation learning methods fail to perform well. 

[3] Usually, representation learning aims to learn high-quality representations that can be utilized in various downstream tasks. However, this paper only discusses the optimization for the SVM classifier, which is not aligned with the goal of representation learning. 

[4] More up-to-date baselines should be compared with the proposed method in the experimental part.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This study presents Fed-REACT, a federated learning method tailored for heterogeneous time series data, integrating representation learning and evolutionary clustering. The method operates in two stages: first, clients employ self-supervised learning to extract features from their local data; second, the server uses evolutionary clustering with an adaptive forgetting factor to cluster clients based on their features. Clients from the same cluster collaborate in order to learn the parameters of a downstream task. 

The paper provides a theoretical analysis for self-supervised feature extraction, when a rank one approximation of  the covariance matrix is employed. 

The paper provides numerical simulations demonstrating the effectiveness of the proposed approach, in particular in settings with a small number of clients.

### Strengths
* The paper provides a reasonable number of numerical simulations in order to evaluate the performance of the proposed approach. 
* The paper does not have any major errors, and most of the claims are correct.

### Weaknesses
* The paper is overall hard to follow due to the confusion notation. For example, (1) introduces the contrastive loss that would be used in the first phase of feature extraction. However, the theoretical analysis does not make any reference to (1); instead it introduces an other objective function, $f_{\text{SSL}}$. An other example of inconsistent notation, is the introduction of $\phi_t$, $W_t$, and $N_t$, which are not explicitly connected to the parameter  of the problem $\theta$ and $\theta_{\text{task}}$. 
* I do not really see how Section 3 fits in the paper and how it serves its goal. This section proves the convergence of federated projected gradient descent under a restrictive set of assumptions to a neighborhood of a stationary point. 
* The numerical simulations show the advantage of the proposed method when the number of clients is small, but the advantage of this method is small when the number of clients is relatively large (>50).
* The paper is principally proposing to learn a feature extractor in a federated self-supervised fashion. Afterwards, evolutionary clustering is used to learn one model by cluster. Both these techniques are already known, therefore the novelty of the paper is limited.
* The proposed approach is claimed to be tailored for time series, but it could be used for any type of data.

### Questions
Can you please explain how is the proposed approach specific to time series?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors proposed a method Fed-REACT to tackle the time-series analysis in federated learning. They adopt self-supervised training in the first stage to learn useful representations. In the second stage, they use evolutionary clustering to cluster clients for local downstream tasks. Extensive experiments are conducted to evaluate the performance of the proposed mechanism.

### Strengths
1. Overall, the paper is well-written with a clear presentation. 
2. Investigating time-series analysis in FL is a significant problem and direction.
3. Extensive experiments have been conducted to evaluate the performance of the proposed framework.

### Weaknesses
1. The explanation of motivation in the introduction is unclear. The reason for applying clustering and self-supervised learning to investigate time-series analysis in FL could be clarified.
2. The novelty of Weighted Averaging with Forgetting is limited. It is quite common to do clustering or aggregation by clients’ current and historical weights.
3. The compared baselines are too old. Some SOTA works about time-series FL or clustered FL could be considered.
4. As a heuristic algorithm, what is the efficiency and cost performance of evolutionary clustering?
5. The author could consider comparing with SOTA time series analysis models, such as Time-LLM, PatchTST, TimesNet, etc.
6. Are the experimental results repeated multiple times? Are there any standard deviation results shown?

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
2

### Contribution
3
