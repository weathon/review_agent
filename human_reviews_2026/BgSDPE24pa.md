# Prior-free Tabular Test-time Adaptation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 8, 4, 2

## Abstract
Deep neural networks (DNNs) have been effectively deployed in tabular data modeling for various applications. However, these models suffer severe performance degradation when distribution shifts exist between training and test tabular data. While test-time adaptation (TTA) serves as a promising solution to distribution shifts, existing TTA methods primarily focus on visual modalities and demonstrate poor adaptation when directly applied to tabular modality. Recent efforts have proposed tabular-specific TTA approaches to mitigate distribution shifts on tabular data. Nevertheless, these methods inherently assume the accessibility of source domain or prior and fail to fundamentally address feature shift while overlooking unique characteristics of tabular data, leading to suboptimal adaptation. In this paper, we focus on the problem of \textit{prior-free tabular test-time adaptation} where no access to source data and any prior knowledge is allowed, and we propose a novel method, \underline{P}rior-\underline{F}ree \underline{T}abular \underline{T}est-\underline{T}ime \underline{A}daptation (PFT$_3$A), which has three designs to simultaneously address label shift and feature shift without source domain or prior access. Specially, PFT$_3$A contains the \textit{Class Prior Estimating} module for estimating source-target class priors to calibrate prediction, eliminating dependency on source class prior and mitigating label shift; the \textit{Robust Feature Learning} module for learning robust feature by aligning source-like and target-like features to mitigate feature shift; the \textit{Representative Subspace Exploration} module for eliminating redundant features by projecting feature into subspace to enhance feature alignment. Extensive experiments demonstrate the effectiveness and generalization of PFT$_3$A in tabular TTA tasks. The implementation is at \url{https://github.com/rundohe/PFT3A}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on the tabular TTA problem without the accessibility of the source domain or prior. Based on the analysis, this paper proposes PFT3A method, which consists of a Class Prior Estimating module for estimating source-target class priors, Robust Feature
Learning module for learning robust features by aligning source-like and target-like features to mitigate feature shift, and Representative Subspace Exploration module for selecting informative features. Experiments conducted in this paper show the effectiveness of PFT3A under prior-free tabular test-time adaptation problem.

### Strengths
1. This paper is well-written, which makes it easy to follow. The motivation is clear, and the math symbol is easy to follow.
2. The method introduced in this paper is novel; The idea of optimizing the Kullback-Leibler (KL) divergence of the source distribution and the target distribution with informative columns provides insights for tabular machine learning.
3. The experiment conducted in this paper is well-designed, showing the effectiveness in performance and robustness in parameters of the PFT3A method.

### Weaknesses
1. The motivation behind the method’s design needs to be explained. In Eq. 6, the estimation of the prior leverages both the model’s predicted outputs and the covariance matrix of the features. This design links the predicted probabilities with the feature information of the data; however, the paper does not clarify the rationale for combining these two components.
2. Typos: 

a. In line 298, the cite format would be better using \citep: 
>  Assuming the feature distributions follow a Gaussian distribution Adachi et al.
(2024), --> Assuming the feature distributions follow a Gaussian distribution (Adachi et al. 2024).

b. The symbol $\zeta$ does not appear until section 5.3, and it is not clearly defined. According to the explanation in line 466, does it refer to $\epsilon$, or what is the relationship between $\zeta$ and $\epsilon$?

### Questions
See Weakness above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles test-time adaptation (TTA) for tabular data in a realistic, prior-free setting where no source data or distributional priors are available. The proposed method, PFT₃A, comprises three key modules: (1) a Class Prior Estimation module that addresses label shift via unsupervised entropy-based techniques; (2) a Robust Feature Learning module that aligns features using KL divergence without relying on confidence thresholding; and (3) a Representative Subspace Exploration module that identifies and adapts discriminative subspaces through PCA-inspired methods. Comprehensive experiments on five datasets with various backbones demonstrate consistent superiority over existing TTA baselines, including prior-free and tabular-specific ones. Ablation studies and sensitivity analyses validate the efficacy of each component.

### Strengths
1). The paper identifies and formalizes an under-explored challenge in tabular TTA: the fully prior-free and source-free scenario. This is motivated by empirical evidence in Figure 2, highlighting limitations of prior methods under real-world constraints.

2). PFT₃A integrates unsupervised class prior estimation from prediction entropy (to handle label shift), KL-based feature alignment (for covariate shift), and a principled subspace selection strategy via PCA, which mitigates overfitting to non-discriminative features.

3). The method is evaluated on five TableShift benchmark datasets using three deep tabular backbones, showing improvements over a range of baselines. The inclusion of ablations and analyses enhances the credibility of the results.

### Weaknesses
1). Although the method is compared to various prior-free and vision-inspired TTA approaches, it lacks direct comparisons or discussions with recent tabular-specific test-time augmentation (TTAug) techniques, such as those in Kozodoi (2021) or Brownlee (2020). These augmentation-focused methods could serve as competitive baselines or complementary strategies, and their omission limits the benchmarking scope.

2). The paper under-explores connections to emerging training-free TTA adapters in vision-language or cross-modal domains (e.g., Karmanov et al., 2024). Incorporating such perspectives in the Related Work section or as experimental baselines could reveal transferable insights for tabular adaptation.

3). In Sections 4.2 and 4.3, the KL divergence is computed assuming Gaussian-distributed feature projections for alignment. However, the justification for this Gaussian assumption is insufficient, particularly for tabular data with high-cardinality categorical features. Details on handling low-variance or near-singular subspaces—such as eigenvalue thresholding or covariance regularization—are not explicitly described, despite mentions of numerical instability. Furthermore, the bias correction in class prior estimation (Section 4.1.2, Eq. for $\tilde{\mathbf{p}}_T^j$) references a batch-specific covariate matrix without a full explanation of its computation, regularization, or derivation, which could hinder reproducibility.

4). The entropy-based sample selection for distinguishing source-like and target-like instances may introduce bias in severe label shift scenarios (e.g., as observed in Table 10 for HELOC/Health Ins. datasets), where initial model uncertainty might not reliably indicate domain alignment. This affects the accuracy of class prior estimation and suggests the need for additional calibration or robust alternatives.

5). The method's per-batch covariance computations and eigen-decompositions could incur high computational costs on large-scale or high-dimensional datasets. A quantitative comparison of runtime and memory usage against baselines would strengthen the practical claims.

### Questions
1). Could you clarify if the Gaussian assumption for feature representations was empirically validated across all evaluated tabular datasets, especially those with high-cardinality categorical variables? 

2). For the class prior update equation in Section 4.1.2 ($\tilde{\mathbf{p}}_T^j = \operatorname{Norm}(\tilde{\mathbf{p}}_T^{j-1} - \hat{C}^{j-1} \tilde{\mathbf{p}}_S^j$), please provide a detailed derivation, intuition, and procedure for computing and regularizing the batch covariate matrix, particularly in cases of low-rank or ill-conditioned features.

3). Have you evaluated the runtime and memory footprint of PFT₃A on larger batch sizes or datasets with hundreds of features, and how does it compare to baseline methods?

4). The "Limitations and Future Work" section (Appendix A.5) emphasizes adversarial robustness but overlooks scenarios where the method underperforms, such as extreme distributional shifts or small-batch settings. Could you elaborate on these failure modes and potential mitigations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Prior-Free Tabular Test-Time Adaptation (PFT3A) to solve distribution shifts in Tabular data by test-time adaptation. Specifically, they achieve prior-free test-time adaptation by estimating the source data prior through low-entropy target data at the first test batch, and learn robust and representative representations by aligning pseudo-source and target features during test-time adaptation. Experiments on the TableShift benchmark demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is easy to follow.

2. Experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The novelty of the proposed method is limited. The idea of using high-confidence target samples as source-like samples have already been explored in many source-free adaptation or test-time adaptation papers [1]. The idea of aligning source and target features for test-time adaptation has also been widely used [2, 3]

2. The class-prior estimating module seems highly relies on the threshold of the entropy. But it is not clear how does the threshold set for various datasets.

3. Following the above question, there is no guarantee that the estimated source prior can cover the real source data. If the first batches are very bad (e.g., heavily shifted or extremely class-imbalance, which can usually happen in real applications), the estimation can be biased and misclassified. And the error will also be accumulated to the following adaptation and prediction, leading to worse and worse results. Including experiments on bad first-batch test data is also necessary for more insights of this problem.

### Questions
1. From Figure 4, it seems that the method are sensitive on some hyperparameters (e.g., beta_2 and m). It is not clear whether different dataset need specific hyperparameters. And how does these hyperparameters set for different datasets?

2. In Section 4.2, why do the author assume feature as Gaussian distributions? motivation and advantages? How about other distributions or just deterministic features with alignment by L2?

3. How about the costs of the method on test-time adaptation?

### Soundness
2

### Presentation
3

### Contribution
1
