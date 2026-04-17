# One-Shot Weighted Ensemble Estimation for Federated Quantile Regression: Optimal Statistical Guarantees under Heterogeneous Structured Data

- Decision: Reject
- Scores: 8, 6, 2, 4

## Abstract
Federated Quantile Regression (FQR) has emerged as a powerful modelling paradigm for estimating conditional quantiles, offering a more comprehensive understanding of response distributions than standard conditional mean regression. However, achieving communication efficiency and optimal statistical guarantees for FQR remains challenging, particularly due to the nonsmooth nature of quantile loss functions and the presence of heterogeneously structured data, where each local agent trains its conditional quantile models with distinct sets of features. In this paper, we propose a data-driven, one-shot weighted ensemble estimator for FQR that incorporates scalable weighting schemes to effectively leverage the partially observed features at each local agent, thereby enjoying both communication efficiency and estimation optimality. Theoretically, we present a unified analysis of the proposed learning procedure, establishing that the resulting estimator exhibits asymptotic normality and attains uniformly minimum variance. Furthermore, we investigate the estimator's sensitivity to perturbations introduced by local agents and derive conditions under which the estimator achieves stability and enjoys strong out-of-sample generalization. Extensive simulations under various scenarios validate the asymptotic normality of our estimator and demonstrate its superior estimation accuracy and uniform convergence compared to several baseline methods across a range of quantile levels.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper develops a federated quantile regression framework for heterogeneous data and provide corresponding theoretical guarantees.

### Strengths
While some assumptions are strong (e.g., Assumption 4.1), the authors provide a comprehensive analysis with detailed theoretical justification.

### Weaknesses
I'm wondering whether it is possible to relax Assumption 4.1 to isotropic sub-Gaussian features? If not, what are the restrictions in the proof?

Another question is that, in the proof, how should you handle the o_p terms if M can also asymptotically increase? Based on previous literature, e.g., Volgushev, Stanislav, Shih-Kang Chao, and Guang Cheng. "Distributed inference for quantile regression processes." (2019): 1634-1662. they provide an upper bound on the number of workers. I'm wondering if there is a similar result like that in the federated scenario?

### Questions
Please answer my questions in the weakness section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel one-shot weighted ensemble estimator for FQR that effectively handles heterogeneously structured data. By incorporating an optimal weighting scheme, the method achieves communication efficiency while establishing optimal statistical guarantees, which is validated through extensive experiments on both synthetic and real-world data.

### Strengths
This paper introduces a communication-efficient FQR method, specifically designed for a unique form of heterogeneous structured data. Its primary strength lies in its theoretical results, which demonstrate that the proposed estimator achieves both asymptotic and non-asymptotic theoretical guarantees.

### Weaknesses
There are some critical issues the authors need to address, including: (1) strong modeling assumptions and an unusual definition of heterogeneity; (2) questionable stability arguments with respect to non-i.i.d. agents; and (3) a simple experimental design with limited experimental baselines.

### Questions
1. The assumption of a linear data-generation process in the problem setup is overly strong and does not align well with most real-world scenarios.
2. The definition of heterogeneity employed in the paper is somewhat unusual. In typical federated learning settings, it is more common to assume heterogeneity in the model parameter (e.g., $\beta^*$).
3. The paper is mathematically dense and uses a large number of notations. It would be beneficial to include a dedicated section that introduces and summarizes all of the notations used throughout the manuscript.
4. In Remark 3.1, the authors present the communication cost of the proposed method, but they do so without providing comparison for other FQR methods. For readers less familiar with this field, it would be helpful if the authors compared the communication overhead of their method with that of existing methods, and discussed whether the communication cost becomes especially large in applications with very large problem scales.
5. In Section 4.2, we know that traditional sample-level algorithmic stability typically assumes each sample is i.i.d. However, in the present setting the authors consider agent-level stability where each agent may be non-i.i.d. Therefore, it is unclear whether the standard algorithmic stability framework can be directly applied to this setup.
6. In the numerical experiments, the authors compare the proposed method only against two very simple baselines (“Naive-local” and “Naive-OSFL”). The limited number and simplicity of these baselines weaken the evidence for the superiority of the proposed method. Additionally, the authors are encouraged to report mean values together with standard errors (over multiple runs) in order to demonstrate the robustness of their results.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a quantile regression framework for federated learning with structured missing data and provide a one-shot weighted ensemble estimation algorithm. Theoretical guarantees, including convergence rates and asymptotic normality, are presented and are noted as being solid and comprehensive.

### Strengths
*   Well-established theoretical results with detailed proofs covering both convergence rates and asymptotic normality.
*   The proposed algorithm addresses the challenge of structured missing data in a federated learning context.

### Weaknesses
*   **Lack of Clear Innovation:** The core methodological innovation appears limited, as the algorithm seems to be a direct adaptation of existing work (Chen Cheng, 2023) by merely substituting the gradient and Hessian calculations for the quantile loss, without a significant abstraction or generalization.
*   **Insufficient Motivation for Quantile Regression:** The paper lacks a dedicated discussion on the specific properties and challenges of the quantile loss function (e.g., non-differentiability, computational aspects), failing to justify why this specific loss is used and how its peculiarities are handled.
*   **Inadequate Experimental Evaluation:** The experiments are considered thin. Key baselines and analyses are missing, including:
    *   Comparison with a pooled estimator.
    *   Comparison with an estimator using fully observed data.
    *   Experimental validation of the theoretical asymptotic normality.
    *   Experiments on heavy-tailed distributions (e.g., Cauchy).
    *   Analysis of estimation error variability.
    *   Sensitivity analysis regarding the number of quantiles (M).

### Questions
1.  What is the fundamental methodological advancement beyond the work of Chen Cheng (2023)? Specifically, how does the proposed framework abstractly generalize the problem for a broader class of loss functions, or, if it is specific to quantile regression, what unique technical challenges does it solve?
2.  Why is there no discussion on the inherent properties of the quantile loss function (like non-differentiability), and how does the proposed algorithm effectively manage these challenges?
3.  Can the authors provide more comprehensive experiments, including the missing baselines (full-data ) and a verification of the asymptotic normality results?
4.  How does the method perform under heavy-tailed noise distributions, and how does the number of quantiles (M) impact the estimation stability and accuracy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a one-shot Federated Quantile Regression (FQR) method for heterogeneous structured data where agents observe distinct feature subsets. Each agent fits local linear QR on its features  then the server solves a weighted ERM mixing local estimators. Theory claims provided include asymptotic normality for any positive-definite weights; existence of variance-optimal W⋆,  and generalization bound via agent-dependent stability among few more things.

### Strengths
1) Relevant problem and one-shot design: FQR with heterogeneous feature access is under-explored;  transformation reconciles partial features via cross-correlations rather than truncation; avoids iterative rounds
2)  Good theory contributions ie: asymptotic normality for broad weights, variance-optimal W⋆ characterization, practical plug-in construction, and agent-perturbation stability bound
3) Transparent communication: Per-agent payload d²ᵢ+3dᵢ explicit which is a good sign  and n-independent

### Weaknesses
1) Since this is mainly a theory based paper it seems the major results require gaussian designs and structural coverage requiring that the union of local supports spans the full feature space. Computing the key quantities Tₘ and W⋆ requires access to matrices A and B that depend on unobserved counterfactual data x⁻. Broadly speaking a little gap seems to exist in the theory and empirical part of the paper. More on this in my next point

2) The empirical part of the paper seems a bit thin when compared to the strength of the theory claim: For instance, the paper compares only against weak baselines (Naive-Local and simple averaging), using a single small real-world dataset (California Housing, median quantile τ=0.5). I would appreciate if the experimental section can be expanded to consider more variant setting

### Questions
Weaknesses and Questions merged

### Soundness
3

### Presentation
3

### Contribution
3
