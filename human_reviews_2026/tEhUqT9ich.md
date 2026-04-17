# TabINR: An Implicit Neural Representation Framework for Tabular Data Imputation

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Tabular data builds the basis for a wide range of applications, yet real-world datasets are frequently incomplete due to collection errors, privacy restrictions, or sensor failures. As missing values degrade the performance or hinder the applicability of downstream models, and while simple imputing strategies tend to introduce bias or distort the underlying data distribution, we require imputers that provide high-quality imputations, are robust across dataset sizes and yield fast inference. We therefore introduce \textsc{TabINR}, an auto-decoder based Implicit Neural Representation (INR) framework that models tables as neural functions. Building on recent advances in generalizable INRs, we introduce learnable row and feature embeddings that effectively deal with the discrete structure of tabular data and can be inferred from partial observations, enabling instance adaptive imputations without modifying the trained model. We evaluate our framework across a diverse range of twelve real-world datasets and multiple missingness mechanisms, demonstrating consistently strong imputation accuracy, mostly matching or outperforming classical (KNN, MICE, MissForest) and deep learning based models (GAIN, ReMasker, DiffPuter, CACTI, GRAPE, UnIMP), with the clearest gains on high-dimensional datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TabINR, a missing value imputation method for tabular data. The key idea is to leverage Implicit Neural Representations (INRs) for imputation. For each cell, TabINR learns a row embedding and a feature embedding, and then uses an MLP to map their combination to a scalar value as the imputed result. To address the absence problem of row embeddings for unseen test samples, a test-time adaptation strategy is introduced to retrieve the most suitable row embedding from the training set. Experiments on 12 datasets across MCAR, MAR, and MNAR settings and different missing ratios are done to validate the effectiveness of the proposed method.

### Strengths
1. The paper is clearly written and easy to follow.

2. The proposed method is lightweight and memory-efficient.

### Weaknesses
1. The technical novelty of the proposed method is limited. The key idea is to learn row and feature embeddings for each row and column in the tabular data, which is common.

2. The proposed method does not seem to be effective, as it has higher NRMSE and lower AUROC compared with ReMasker in many cases in Figure 3. Similar trends are observed in Figure 7 - 16 under different settings.

3. The proposed test-time adaptation mechanism could be computationally expensive for large-scale datasets, since it requires searching over all row embeddings to find the one that minimizes the imputation error of all observed features in a test instance.

4. It is mentioned in the introduction that in tabular datasets, "rows are often assumed to be independents." This may not be true as many methods leverage the correlation between rows for imputation. In addition, the MCAR, MAR, and MNAR are three common settings. It may not be necessary to elaborate here.

### Questions
1. How are the training, validation, and test sets constructed? Do test samples appear during training such that they have pre-learned row embeddings, or are row embeddings only learned for training samples?

2. Could you clarify the masking procedure used during training and testing? In Line 215, it is mentioned that "During training, we applied random masking of 10–70 % of entries to simulate missingness and evaluate reconstruction fidelity." However, in Table 5, the masking ratio is set to 0.3.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem of missing values in tabular data. The authors propose TAB-INR, an auto-decoder–based Implicit Neural Representation (INR) framework that employs learnable row and feature embeddings for adaptive imputation from partial observations. Experiments on twelve real-world datasets show that TAB-INR achieves consistently superior imputation accuracy.

### Strengths
S1: A new Implicit Neural Representation framework for missing data imputation.
S2: Comprehensive experiments are conducted.

### Weaknesses
W1: The paper’s presentation needs improvement. In particular, the Introduction lacks sufficient discussion of related works on both data imputation and Implicit Neural Representations (INRs). As a result, the motivation for applying INR to imputation is unclear—specifically, what unique advantages INR offers for this problem and how the challenges in imputation naturally align with the strengths of INR.

W2: The paper misses comparisons with several state-of-the-art imputation methods, especially recent graph-based approaches such as
[1] Handling Missing Data with Graph Representation Learning and
[2] On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing.
These methods generally outperform older baselines like ReMasker and GAIN, and they also employ learnable representations for rows and columns. It remains unclear why the authors chose INR-based learnable features over graph-based ones or how their design offers any advantages.

W3: The experimental evaluation primarily compares against outdated baselines (e.g., GAIN, MissForest, MICE) while omitting more recent generative or graph-based models. Even within these limited comparisons, the improvements are not significant, which weakens the empirical contribution.

W4: The core contribution of the paper appears to be the straightforward application of the INR framework to the imputation task. Beyond this adaptation, the paper’s novel technical contribution is not clearly articulated, making it difficult to distinguish the originality or necessity of the proposed method.

### Questions
NAN

### Soundness
2

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
4

### Summary
This paper proposes TabINR, which applies Implicit Neural Representations (INRs) to tabular data imputation. The approach models table entries as outputs of a neural function conditioned on learnable row and feature embeddings. At test time, new instances are handled via auto-decoder-style optimization of row embeddings while keeping the network frozen.

### Strengths
1. Applying INRs to tabular imputation is relatively unexplored (as far as I know) and conceptually interesting.

2. TabINR demonstrates faster inference than iterative methods, per dataset once trained, which is quite impressive and is practically useful.

3. The auto-decoder approach for inferring embeddings from partial observations is neat and enables instance-specific imputation.

4. The ablation analysis are well done and comprehensive.

### Weaknesses
1. TabINR shows competitive performance but gains over baselines are modest and inconsistent. ReMasker frequently matches or outperforms it imputation and downstream classification. While exploring a novel approach justifies some performance variability, the more critical concern arises from the fact that the benchmark omits recent state-of-the-art methods [1,2], making it unclear whether this represents meaingful progress in tabular imputation. Additionally, there appears to be a hyperparameter tuning imbalance. TabINR underwent grid search across datasets, while baselines used default settings configurations. This could be potentially inflating TabINR's relative performance. 

2. The results in Sec 3.1.2 are a good starting point to demonstrate permutation robustness. However, they demonstrates permutation invariance only under MCAR, where missingness is completely random and uninformative. Under this setting its almost certainly expected that the all models are permutation invariant. The meaningful test would be under MNAR, where missingness patterns carry information about the data structure. Without MNAR results, the claim of permutation robustness remains unconvincing.

#### References
[1] DiffPuter: Empowering Diffusion Models for Missing Data Imputation (ICLR 2025)

[2] CACTI: Leveraging Copy Masking and Contextual Information to Improve Tabular Data Imputation (ICML 2025)

### Questions
1. How does test-time optimization cost compare to baseline inference in wall-clock time?

2. The SIREN activation function introduces high-frequency components, is this actually beneficial for tabular data, which typically lacks such structure?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose an Implicit Neural Representation (INR) based imputation method for tabular data. It represents the dataset as a function (in fact, a neural network) that maps a cell (represented by row embedding and column embedding) in the table to its corresponding numerical value. This function is trained on the non-missing cells and then used to fill in missing values as needed. The authors provide an evaluation against different baselines (both classical and neural network-based). The reported performance is on par with, if not superior to considered baselines.

### Strengths
* The paper tackles a fundamental issue of handling missing data, which is very common in practical data analysis and machine learning problems, where some data are missing due to reasons like sensor malfunction, communication problems, or privacy concerns.

* The proposed method is targeted to work well not only in a relatively simple MCAR case, but also in the case of MAR and MNAR missing mechanisms.

* The proposed approach takes a different approach for missing value imputation.

### Weaknesses
* The paper lacks justification and a clear motivation for why the proposed method works. Section 2, the authors explained the proposed method step-by-step, but did not discuss why. It is more like a technical report than a research paper.

*  The presentation is not entirely clear to readers, particularly how missing values are imputed at inference (Sec 2.3).

* While the proposed method is competitive with the state-of-the-art methods in 12 datasets used, it is not particularly good in any case. The authors fail to provide any conclusion on what sort of scenario/case the proposed approach may be better than the SOTA methods.

### Questions
We would appreciate the authors' responses to the concerns raised in the weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
2
