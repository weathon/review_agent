# Complexity-Separated Schemes for Addressing Structured Heterogeneity in Federated Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 0, 6, 4

## Abstract
Federated learning faces challenges due to heterogeneity in local training sets. Existing methods typically treat this as a monolithic challenge, leading to communication overhead. In this work, we suggest examining the structure of data heterogeneity in more detail. We identify two forms of this phenomenon: mode-based, where clients differ in the presence of common versus unique data modes; and coordinate-based, where groups of model parameters vary in statistical similarity. We develop algorithms that decouple communication complexity along these structural dimensions and consequently achieve reduced synchronization frequency without deterioration in convergence. Our analysis establishes the optimality of the proposed schemes. Extensive experiments on image and multimodal classification tasks demonstrate improvements in communication efficiency over state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a well-motivated approach to federated optimization by exploiting structured heterogeneity, with rigorous theoretical foundations and empirical validation. The algorithms demonstrate clear communication efficiency gains in non-convex settings, as shown in theoretical bounds (Theorems 1–3, Corollaries 1–2) and experiments (Figures 1–2, Table 1). However, the experimental section lacks detailed ablation studies and reproducibility information, such as hyperparameter tuning and statistical significance tests. Additionally, comparisons with closely related methods (e.g., ProxyProx) could be more explicit in both theory and experiments.

### Strengths
The paper is the first to systematically address structured heterogeneity in non-convex federated learning, separating complexity for distribution-based and coordinate-based cases.

The analysis includes upper bounds, matching lower bounds, and detailed proofs leveraging zero-chain functions and Hessian similarity assumptions.

### Weaknesses
Only two datasets (CIFAR-10, Avito) and two neural architectures (ResNet-18, BERT) are used, raising questions about generalizability.

Differences from ProxyProx and Accelerated ExtraGradient are not thoroughly analyzed in terms of convergence speed or robustness. 

Theoretical results depend on Hessian similarity, but no empirical validation of these assumptions is provided.

The impact of algorithmic components (e.g., probability pp, reference points) is not systematically ablated.

Experimental results lack error bars or repeated runs, making it difficult to assess robustness. 

Scenarios where HASCA or C-HASCA underperform are not discussed, such as extreme heterogeneity or noisy clients. No direct evidence found in the manuscript.

The subproblem in Line 3 of Algorithms 1–2 requires exact minimization, but no analysis of its computational cost is provided.

### Questions
Please respond to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper's goal is to decouple two forms of data heterogeneity in federated learning and design communication-efficient optimization algorithms. The algorithm is similar to FedProx, but applied to composite functions. The analysis and experimental results are quite rudimentary and limited.

### Strengths
The goal of the paper is to consider two different forms of data hetergeneity: mode-based and coordinate-based.

### Weaknesses
I have several major concerns about this paper:

* The paper spends several pages on motivating and setting up the problem, first talking about distributed learning with IID data splits, then federated learning with data heterogeneity. For an established research topic like federated learning this extended introduction is not necessary. The space would have been better used to present the paper's own algorithm, analysis, and results.

* The notation section talks about 3 measures of communication complexity: 1) communication rounds, which is used in synchronous methods, 2) asynchronous aggregation, and 3) compressed communication. Each of these paradigms pose separate challenges for algorithm design and analyses that have been addressed by a long line of previous works. I don't believe how the authors can unify all of these in their analysis.

* The communication complexity bounds are rudimentary order-wise bounds. Also the order of the rounds in the bounds seems incorrect to me. Without strong convexity assumptions the rate is O(1/\epsilon^2) not O(1/\epsilon). 

* Experimental results compare with a limited set of methods. Why haven't the authors considered more standard baselines like FedAvg, FedProx, SCAFFOLD etc? The omission of FedProx, which is closely related to this paper, is particularly glaring. The paper does not even include a citation to FedProx.

### Questions
Please clarify what exact problem setting you are working with in this paper. The problem formulation, proposed algorithm and contributions are not clear at all based on my reading of the paper.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies efficient federated learning under heterogeneous data distributions. The authors decompose data heterogeneity into two structural types — mode-based and coordinate-based — and design theoretically optimal algorithms that decouple communication complexity along these two structural dimensions.

### Strengths
1.	The paper provides a novel theoretical perspective by decomposing data heterogeneity into distinct structural forms.

2.	It offers detailed and rigorous theoretical analysis with formal convergence and optimality results.

### Weaknesses
1.	The overall organization and presentation are highly theoretical, making the main ideas and algorithmic insights difficult to follow.

2.	The evaluation is limited in scope, using relatively simple models and datasets that may not fully demonstrate the practical value of the proposed approach.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of data heterogeneity in federated learning by proposing an innovative structured solution. The authors identify two heterogeneity patterns: heterogeneity based on data distribution and heterogeneity based on model parameters, and design corresponding HASCA and C-HASCA algorithms. These algorithms achieve effective separation of communication complexity by employing differentiated communication frequencies for different heterogeneous components.

### Strengths
1. This paper delves into the inherent structural differences in heterogeneity, namely uneven pattern distribution and varying parameter block characteristics. This deconstruction provides a novel theoretical foundation for designing more refined and efficient algorithms.

2. The core mechanism of HASCA and C-HASCA is a general framework. This idea can be easily transferred to other distributed optimization scenarios beyond federated learning.

### Weaknesses
1. The theoretical complexity analysis in this paper primarily focuses on the number of communication rounds or communication attempts. However, in practical federated learning systems, communication bottlenecks often manifest as bandwidth consumption and latency. The paper lacks a quantitative assessment of actual communication time and the total amount of data transmitted.

2. The theoretical analysis is based on the strong assumption that "the server can solve the subproblems precisely" (Algorithm 1, Line 3), but the experimental section mentions that "usually only approximate solutions are needed" (Section 7). The impact of this approximate solution on convergence is not fully discussed in the theoretical section, creating a gap between theory and practice. Furthermore, engineering techniques such as momentum and learning rate scheduling introduced in the algorithm (Section 7.1) are not included in the theoretical framework.

3. The degree of heterogeneity in the experiments (e.g., κ=0.8) is artificially set. How does the algorithm perform in more extreme cases (e.g., some client data is completely unrelated to server data, i.e., δg is maximum)? The robustness analysis in this paper (Appendix A.3) mainly focuses on the proportion of categories covered by the server data, and the analysis of the systematic impact of extreme changes in δ values ​​is insufficient.

### Questions
See weeknesses

### Soundness
3

### Presentation
3

### Contribution
2
