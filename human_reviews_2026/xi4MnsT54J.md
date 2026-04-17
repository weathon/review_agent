# Federated Data and Feature Selection by Generalized CUR Decomposition

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
With the advance of federated learning (FL) in privacy-sensitive domains such as healthcare, finance, and mobile intelligence, the need for efficient and robust training becomes increasingly urgent. Communication bottlenecks, heterogeneous client distributions, and fairness requirements make it essential to select the “right” data and features for model training. Yet existing FL research often addresses feature selection and data selection separately, ignoring their interplay in real-world high-dimensional and noisy datasets, leading to suboptimal performance. In this paper, we propose a unified framework for data and feature selection by formulating the problem as a generalized CUR decomposition problem. We introduce FedGCUR, a practical framework that integrates a federated column-pivoted QR (FedCPQR) decomposition routine with per-silo row selection. Specifically, FedCPQR is designed to securely compute a global pivot order without exposing raw data, while FedGCUR leverages this to jointly select shared features and silo-specific samples. We prove that FedCPQR produces exactly the same decomposition results as centralized CPQR and establish an upper bound of the reconstruction error of FedGCUR. Extensive empirical results show that the proposed framework achieves higher accuracy compared to the baselines of data and feature selection methods, demonstrating its effectiveness and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a data and feature selection algorithm in federated learning (FL) setting. The idea is to use CUR decomposition to reduce the dimensionality and select the top rows and columns which correspond to the top samples and features.

### Strengths
- The idea of using CUR decomposition for data and feature selection is novel and interesting.
- Extensive experiments were performed and detailed analysis on different aspects (complexity, comm cost, privacy, etc.) of the method are provided.

### Weaknesses
- The motivation is not clear. There is little evidence as to why separate data and feature selection may hurt the utility. In addition, there is no complexity comparison to existing data and feature selection methods. The computation cost of the proposed method can be heavy due to the CUR decomposition. A computation-utility tradeoff should be discussed with comparison to other data/feature selection baselines.
- There are many recent data and feature selection methods for FL settings as mentioned in the related work section. However, the author chose older methods that are not designed for FL as baselines.
- Why does FedCPQR use secure aggregation? The inputs to the sum are protected, but the sum is revealed to all parties, so there is no formal privacy protection. 
- The experiments use only 10 clients. This is very small number for a horizontal federated learning setting. Further it assumes that all clients are available at all times, which is not necessarily the case in FL. How does the method scale to larger numbers of clients? What about client unavailability?
- The experimental datasets are also on the small side. Can the method scale to larger datasets?
- Minor: main results in Table 4 are too compact which makes it hard to read.

### Questions
- Can the author demonstrate with examples how separate data and feature selection might lead to suboptimal results?
- What is the complexity of the proposed algorithm compared to other data/feature selection baselines?
- What are the reasons for the chosen baselines?

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
3

### Summary
The paper methodological core (FedCPQR exactness; clean FedGCUR design; reconstruction-error bound) is solid and clearly written, and correctness experiments are convincing. However, the practical value proposition isn’t yet demonstrated convincingly: absolute accuracy drops are large versus "all data", budget-fair comparisons are partial, privacy is not quantified properly and seems rather superficial, and systems measurements (bytes, rounds, WAN) are absent.

### Strengths
The positives can be summarised as follows,

- Clear, unified problem framing. The paper tackles joint feature and data selection in cross-silo FL via a generalized CUR factorization, with a clean split: global feature selection (shared columns) and per-silo data selection (rows).
- FedCPQR is well-specified and theoretically exact. The modified Gram-Schmidt style protocol with secure aggregation of norms/inner products provably reproduces centralized CPQR pivoting and factors; the algorithm is explicit (Alg. 1) and its communication pattern is easy to reason about.
- Empirical breadth and correctness checks. Six OpenML datasets (varied sizes/classes), both IID and Dirichlet non-IID splits including multiple ranks with FedCPQR matching SciPy CPQR to numerical precision, and the paper reports run-time scaling across number of parties/ranks.

### Weaknesses
The weaknesses can be summarised as follows,

- Utility gap vs. "all data" remains large. Even when FedGCUR/FedCPQR beat other selection baselines, absolute accuracy often drops markedly relative to using all data
- Privacy analysis is rather superficial and not quantitative. Revealing P, R (hence A^TA=PR^{T}RP^{T}) and residual norms may leak feature norms/correlations; special-case leakage for two parties is acknowledged but not mitigated. No formal membership/attribute-inference or DP bounds are provided.
- Missing communication-byte accounting & systems costs. The paper counts scalars per iteration but does not report end-to-end message sizes/rounds, cryptographic overhead, or wall-clock under WAN conditions; runtime plots exist, but not network/energy budgets.
- Limited task realism. All experiments are small image/tabular classification with a 3-layer MLP and only 10 FedAvg rounds; no results on modern deep FL tasks, no ablations on per-silo quotas.
- Lack of reproducibility as the code is not suppled and its relase is defered for later upon publication.

### Questions
The questions can be inferred mostly from the weaknesses mentioned prior but further,

- Code & reproducibility: The checklist promises fixed seeds; when will code/configs be made available?
- Robustness to noisy/poisoned silos: If one silo contains substantial noise or adversarial rows, does local CPQR row selection isolate/remedy that, or can it still influence global pivots via FedCPQR? Any robustness results?
- Heterogeneity and fairness: Beyond macro accuracy, how does FedGCUR affect per-silo performance variance and minority-class performance under non-IID splits? Could you please report per-client metrics?
- Privacy risk quantification: Since P, R, and residual norms imply A^{T}A (up to permutation), can you quantify reconstruction or attribute-inference risk (e.g., via Shokri MIAs) and report attacks under small-party (s=2–3) regimes? Any plan for DP (e.g., noise on aggregated scalars)?

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
4

### Summary
The paper introduces a federated method for "joint data and feature selection" via column-pivoted QR decomposition.
They extend federated QR to support column pivoting by securely aggregating scalar quantities (norms, inner products) across silos without sharing raw data. 
Then they use the global column pivots from FedCPQR and performs local row selection per silo to construct CUR-like factors. 
The paper provides an exactness claim for FedCPQR under ideal secure aggregation, a reconstruction error bound, and empirical validation under i.i.d. and non-i.i.d. splits.

### Strengths
- The paper introduces a straightforward extension of modified Gram-Schmidt CPQR to the federated setting. 
- The authors support the method with a theorem that captures equivalence to centralized CPQR under exact aggregation.
- They empirically check that FedCPQR produces pivot orders matching SciPy with negligible numerical error.
- The evaluation covers both i.i.d. and non-i.i.d. splits.
- The paper seemingly provides a reconstruction bound relating column and row projection errors to singular values.

### Weaknesses
- The claim of a "unified framework for data and feature selection" oversells what is essential to many decomposition algorithms.
- The work appears to be highly influenced by Hartebrodt & Röttger (2023); advances beyond prior art remain unclear, and the contribution is seemingly minor. 
- The multi-matrix extension in FedGCUR is a small adjustment over existing work (Gidisu & Hochstenbach, 2022).
- Contextualization, related work, and experimental results of federated matrix factorization and on component-wise 'pivoted' / matched Federated NMF is absent.
- The problem solved is mathematically equivalent to federated CUR decomposition with a specific pivot selection. 
- The paper provides theoretical reconstruction error bounds (Theorem 2) for the FedGCUR approximation largely follows the centralized proof strategy.
- Claims of privacy preservation remain hard to verify since the paper lacks a formal privacy definition and contains informal privacy statements.
- The paper heavily relies on a secure aggregation protocol SECAGG without providing sufficient details to fully understand or reproduce this key component. 
- Communication efficiency claim is somewhat overstated, since FedCPQR aggregates O(d^2) scalars per iteration. For high-dimensional datasets, this is practically massive and highly substantial. 
- Experiments do not contain scalability results. Algorithmic scalability with number of clients is unclear as experiments with varying number of clients are missing.
- The authors do not provide ablations isolating the benefit of pivoting over non-pivoted FedQR. 
- The training pipeline for FedAvg (MLP architecture, optimizer, hyperparameters) are underspecified and the reason and location of the experiments are confusing.  
- Details regarding the non-i.i.d. Dirichlet partitioning are minimal, which makes reproducibility harder.
- The core contribution is federating existing CPQR/CUR techniques via existing secure aggregation of scalars. The conceptual benefitover straightforward integration is modest.
- The labels in Figure 1 are difficult to read.
- FedCPQR is stated to "use all data" in experiments while some baselines use half (Sec. 4.2). The reason behind it is not quite clear.

### Questions
- Clarify which algorithmic/theoretical components are new vs. direct federated extensions of existing work (Hartebrodt & Röttger 2023, Gidisu & Hochstenbach 2022). 
- Formally specify all components.
- Clarify communication efficiency claims
- Clarify contributions clearly.
- Provide ablations isolating the benefit of pivoting over non-pivoted FedQR.
- Contextualize your work regarding prior art in federated matrix decomposition methods.

### Soundness
2

### Presentation
2

### Contribution
2
