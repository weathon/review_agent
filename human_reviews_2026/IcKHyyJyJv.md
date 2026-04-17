# Data Valuation and Selection in a Federated Model Marketplace

- Decision: Reject
- Scores: 4, 6, 4, 2, 4

## Abstract
In the era of Artificial Intelligence (AI), marketplaces have become essential platforms for facilitating the exchange of data products to foster data sharing. Model transactions provide economic solutions in data marketplaces that enhance data reusability and ensure the traceability of data ownership. To establish trustworthy data marketplaces, Federated Learning (FL) has emerged as a promising paradigm to enable collaborative learning across siloed datasets while safeguarding data privacy. However, effective data valuation and selection from heterogeneous sources in the FL setup remain key challenges. This paper introduces a comprehensive framework centered on a Wasserstein-based estimator tailored for FL. The estimator not only predicts model performance across unseen data combinations but also reveals the compatibility between data heterogeneity and FL aggregation algorithms. To ensure privacy, we propose a distributed method to approximate Wasserstein distance without requiring access to raw data. Furthermore, we demonstrate that model performance can be reliably extrapolated under the neural scaling law, enabling effective data selection without full-scale training. Extensive experiments across diverse scenarios, such as label skew, mislabeled, and unlabeled sources, show that our approach consistently identifies high-performing data combinations, paving the way for more reliable FL-based model marketplaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new data valuation method based on private Wasserstein distance estimation that is suitable for a decentralized data market. This setting is novel and realistic in that data buyers request an FL model trained on a combination of inaccessible and heterogeneous data sources, while data owners bid by selecting a portion of their data to be used for training.

### Strengths
- It is intriguing to expand the usage of Wasserstein distance, as a credible model performance predictor, into a non-IID, data-decentralized setting. 
- The proposed CombineWad method is theoretically and empirically supported.
- In practice, the target problem setting itself is critical and well-aligned with the needs of data-centric FL. It is impossible to train FL models for all combinations of private datasets; thus, preliminary dataset curation should be considered before operating FL.

### Weaknesses
- The existence of trusted central platform other than a model buyer can be a strict assumption in practice. 
- The empirical validation is not sufficient overall to prove the efficacy of the proposed method.
  - The number of clients is only limited to three, which is far from realistic setting. Either two options below could possibly resolve this issue: 
    - 1) Please consider limiting the target FL setting into a cross-silo FL setup where moderate number of credible clients are participating, to hopefully persuade the small number of clients in the experiments.
    - 2) Or, please consider providing additional results for proving the scalability of the proposed method using widely-used FL benchmark having sufficient number of clients, e.g., `FEMNIST` (3K+ clients), `Sent140` (600K+ clients), and `StackOverflow` (300K+ clients).
  - The aggregation algorithms should have been more diverse and up-to-date, including state-of-the-art adaptive aggregation algorithms, e.g., `TERM` and `AAggFF`. This is essential in order to conclusively demonstrate that the 'Technical Challenge 2' (line 89) has been resolved.  

- `FEMNIST` & `Sent140`: https://arxiv.org/abs/1812.01097
- `StackOverflow`: https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data
- `TERM`: https://arxiv.org/abs/2007.01162
- `AAggFF`: https://arxiv.org/abs/2405.20821

### Questions
- The existence of validation data for calculating Wasserstein distance should have been more convincing. For example, the validation dataset itself has too limited a diversity of sample sizes to allow the model buyers to train an independent model.
  - It seems that $k$ in Theorem 4.2 should be in similar magnitude with the number of pilot ($n$) and validation dataset ($m$). In practice, this can incur an unavoidable communication cost, if $n$ and $m$ are large. Any guide to set $k$ in Theorem 4.2, and what values are used throughout experiments?
- Just to confirm, does Figure 1 imply that the utility of pilot data is maintained linearly, proportionally to the training budget?
- Could the authors please clarify the "single-seller limitations" in line 294?
- Setting $t=0$ for Eq. (6) coincides with the empirical distribution of the pilot dataset. Doesn't this result in privacy leakage?
- Does the Eq. (9) guarantee convergence to global optimum, i.e., is Eq. (8) convex in $\mathbf{p}$? Any reference is appreciated.
- Why there exist periodic fluctuations in validation loss and accuracy in Figure 4 and 13?
- Which dataset do the results in Figure 4 correspond to?
- With the existence of va, is there sharing of pilot data using randomized SVD?
- Please consider using principled label-noise simulation methods for the "Data Selection with Mislabeled Data" experiments in line 469, such as the pairwise flipping or symmetric flipping.

* Pairwise Flipping: https://arxiv.org/abs/1804.06872
* Symmetric Flipping: https://arxiv.org/abs/1505.07634

### Minor comments
- Please consider changing citation style in the draft, e.g., (Alphabet et al., 2021), for better readability.
- Missing citation: "The Monge problem" in line 129; "The Kantorovich's relaxed formulation" in line 132; all benchmark datasets except the last one.
- Typo: "Table ??" in line 161; "as reported in ?" in line 292; "combineWad -> CombineWad' in line 280, 284, and 331.
- Please add a legend in Figure 1.
- $\alpha$ is already used in Theorem 4.1 Please consider changing the step size notation in Eq. (9) to others to avoid any confusion.
- Please consider increasing font sizes and DPI in Figure 4.

### Soundness
2

### Presentation
2

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
The paper tackles the problem of data valuation and selection in FL-based data marketplaces, where raw data cannot be shared due to privacy constraints. The authors propose a novel framework centered on a Wasserstein-based estimator, CombineWad, that predicts model performance across unseen data combinations. The approach further introduces a privacy-preserving approximation of Wasserstein distance and leverages neural scaling laws to project performance at larger scales without full training. The framework is demonstrated on multiple datasets (CIFAR-10, MNIST, ImageNet, RSNA)

### Strengths
1. Introducing a federated variant of Wasserstein-based performance estimation (CombineWad) is innovative. The theoretical link between Wasserstein distance and model generalization in non-i.i.d. FL settings is interesting.
2. The experiments cover multiple FL algorithms, datasets, and heterogeneity scenarios.
3. The theoretical analysis is solid with Theorem 4.1 and Theorem 4.2.

### Weaknesses
1. The paper is dense and mathematically heavy. Some sections (e.g., §4.2, §C) are difficult to follow.
2. The experiments seem limited to small FL scenarios (three clients). It would strengthen the claim to test scalability (e.g., 10–20 clients with non-i.i.d. data).
3. The RSNA dataset (real-world medical dataset) is mentioned, but detailed results are missing.

### Questions
1. How sensitive is CombineWad to the choice of the Gaussian reference distribution?
2. Does CombineWad generalize well when the validation set is small or noisy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles a long-standing yet unsolved challenge in federated data and model marketplaces — how to value and select data from multiple heterogeneous sources without sharing raw data.
The authors propose CombineWad, a Wasserstein-distance–based estimator that predicts model performance across unseen data combinations in a federated learning (FL) setup.

They further introduce a privacy-preserving distributed approximation of the Wasserstein distance (via interpolating measures between each party’s data and a shared Gaussian reference) and couple it with a neural scaling law–based extrapolation, enabling reliable performance prediction on larger datasets without full-scale training.

Extensive experiments across CIFAR-10, MNIST, ImageNet, and RSNA bone age datasets demonstrate that CombineWad consistently correlates with model validation performance under non-i.i.d. conditions, supports algorithm selection (e.g., FedProx vs. FedAvg), and facilitates robust data selection in label-skewed, mislabeled, and even unlabeled cases.

### Strengths
1. Conceptual originality: The paper combines optimal transport, privacy-preserving computation, and neural scaling laws in a new way tailored for federated marketplaces.

2. Theoretical justification: The relationship between validation loss and combined Wasserstein distance is formalized and empirically validated.

3. Strong experimental evidence:
CombineWad shows a stable negative correlation with validation performance across FL algorithms (FedAvg, FedProx, Scaffold, FedNova).
Outperforms AggWad and baseline estimators (Linear, Quadratic, Rational).
Works for noisy, label-skewed, and unlabeled data selection.

4. Practical implications: Provides a lightweight, privacy-aware tool for performance estimation and data acquisition without full retraining.

### Weaknesses
1. Dependence on convergence and algorithm stability:
CombineWad’s correlation with model accuracy weakens under unstable or early-stage FL training (especially for FedAvg). The method implicitly assumes good convergence, which may not always hold.

2. Scalability and sensitivity of private Wasserstein approximation:
The interpolation-based approach introduces hyperparameters whose influence on accuracy and privacy is not fully analyzed.
Its computational and communication cost under high-dimensional embeddings or large m (many clients) remains unclear.

3. Limited robustness and extrapolation analysis:
The scaling-law extrapolation achieves r^2 = 0.88 but lacks an analysis of failure cases (e.g., heavy distribution shift or long extrapolation range).

4. Lack of market/pricing integration:
The paper focuses on valuation and selection but does not connect these to pricing or trading mechanisms, which are critical for a complete “marketplace” framework.

5. Baselines and privacy comparisons:
No direct comparison with DP-based or gradient-based privacy-preserving data valuation methods (e.g., DP-OT, CLUES), which would strengthen the empirical positioning.

### Questions
Q1. How can CombineWad be reliably used before full convergence? Could the authors quantify its reliability or propose a convergence-detection heuristic?

Q2. How should the hyperparameters t, k, σ_γ of the private Wasserstein approximation be chosen? How sensitive are results to these choices?

Q3. What is the computational complexity and communication cost for multi-source (large m) scenarios?

Q4. How far can the scaling-law extrapolation be trusted under strong distribution shifts or limited pilot data?

Q5. Could CombineWad’s gradient signal be used to define shadow prices for data in a federated marketplace?

Q6. Have the authors evaluated privacy leakage risks (e.g., membership inference) from the exchanged interpolating measures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study investigates how model buyers can evaluate the values of distributed datasets and select appropriate data subsets from data sellers who act as clients in an FL setting. The authors propose a scaling-law-style method for data valuation, by which data buyers can use small-scale data samples to predict large-scale model performance with varying sizes of data subsets. Based on this approach, data buyers can select the optimal data subset sizes for data selection to maximize the model performance.

### Strengths
S1. This paper is well motivated and studies a significant research problem regarding data valuation.

S2. The proposed data valuation method is grounded in a solid theoretical foundation.

S3. The idea of using scaling laws for data valuation and selection is interesting.

### Weaknesses
W1. The writing of this paper requires improvement. First, the introduction to Wasserstein Distance in Section 2 fails to clarify its relevance to the target research problem. Second, many choices lack justification. For instance, why do the authors consider the AggWad and CombineWad calculation methods? Why is the non-IID setting examined using only three clients? Why does the non-IID setting assume non-overlapping labels (which is unrealistic in my opinion)? Third, the proposed method has not been completely introduced and explained. For example, Formula (8), which is used for performance prediction, considers only the case of fitting at two scales. Where is the predictor fitting at more scales?

W2. The authors assume that model buyers have a budget N restricting the amount of training data to purchase, which is unrealistic. Typically, model buyers have a monetary budget for purchasing training data or models. Assuming a quantity-based budget implies that all data sellers charge the same unit price, which is uncommon in a non-IID setting.

W3. The trust model in this paper is internally inconsistent. In Section 3, the authors assume that the central platform responsible for computing the Wasserstein distance is trusted. However, in Section 4.2, they consider a privacy-preserving setting in which clients' data privacy must be protected from the central platform, implying that the central platform is semi-trusted or untrusted.

W4. Some important relevant works on data valuation in FL have not been discussed, including the following papers.

[a] A principled approach to data valuation for federated learning.

[b] Secure Shapley Value for Cross-Silo Federated Learning.

[c] Fair and efficient contribution valuation for vertical federated learning.

### Questions
Please address the questions shown in the weaknesses part.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces \textbf{CombineWad}, a Wasserstein-distance–based framework for data valuation and selection in federated model marketplaces. It predicts model performance and selects optimal data combinations without accessing raw data, using a privacy-preserving Wasserstein estimator and neural scaling laws for extrapolation. Experiments on multiple datasets show strong correlation between predicted and actual performance.

### Strengths
1.	Tackles a real and important problem of data valuation under privacy constraints in federated learning.
2.	Provides sound theoretical grounding linking Wasserstein distance to validation performance.
3.	Empirical results show promising predictive accuracy and practical relevance

### Weaknesses
1.	Some important references are missed in the article, e.g. “we provide Table ??, which summarizes the important notations used throughout this work.” In the third Section.
2.	Experiments are mostly synthetic or small-scale, which lack the robustness needed to demonstrate real-world scalability.
3.	Computational and communication overheads of the Wasserstein estimator in high-dimensional data or with many clients are not analyzed.

### Questions
1. How does the proposed private Wasserstein estimator guarantee privacy formally (e.g., under $\epsilon$-DP or other quantifiable bounds)?
2. Could the estimator generalize to vertical FL or cross-domain settings?
3. How sensitive is the method to pilot data size? Would a small pilot set (e.g., 1% of data) still yield accurate predictions?

### Soundness
3

### Presentation
2

### Contribution
2
