# Powerful Independence Testing on Heterogeneous Federated Clients with Theoretical Guarantees

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
We propose a novel federated independence testing framework that addresses both theoretical and practical challenges arising from client heterogeneity. We begin by revisiting existing federated independence testing methods and showing why they often fail to provide valid guarantees or maintain statistical power under data distributional shift across clients. Building on this analysis, we introduce a copula-based marginal alignment technique together with a stacking-based aggregation strategy that amplifies intra-client dependence while mitigating inter-client variation, yielding a theoretically sound and powerful global test. For practicality, we further accelerate the aggregation step and incorporate a privacy-preserving mechanism. On the theoretical side, we establish both the correctness of our method and the validity of the test. Extensive experiments on both synthetic and real-world datasets demonstrate the superiority of our solution over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors investigate the problem of federated independence testing across heterogeneous local distributions by introducing a copula-based marginal alignment and stacking-aggregation method with privacy-preserving secure aggregation. They provide rigorous theoretical guarantees for Type I error control and statistical power, and validate their approach through both simulated and real-world data experiments.

### Strengths
This paper explores an interesting and practical topic — federated independence testing. The authors present a novel method supported by rigorous theoretical guarantees, and importantly, they incorporate privacy provisions that earlier approaches overlooked.

### Weaknesses
1. The consistent dependence assumption considered in this paper is quite restrictive, and may not hold in many real-world federated settings where dependence structures differ across domains.

2. The experiments appear relatively simple: the authors focus mainly on small-scale settings (for example, only three participating clients) and use quite basic data-generation processes.

3. The novelty of the proposed method seems incremental: the approach appears to be a direct combination of existing techniques, such as copula transforms, random projection, homomorphic encryption and so on.

### Questions
1. Assumption 1 is very strong and restrictive. Could the authors provide more concrete application scenarios that satisfy this assumption? In those cases, what are X and Y (in each silo), and why would the correlation / dependence structure be consistent across silos? Also, it is recommended to cite more literature that supports such settings.
2. Assumption 2 posits that only the marginal distributions are heterogeneous across clients. Could the authors consider the case where the joint distribution ($P_{X,Y}$) or the conditional distribution ($P_{Y|X}$) is heterogeneous (i.e., concept shift), which is also common in federated learning? Does the proposed method apply (or can it be extended) to such scenarios?
3. On lines 170-171: why is $f_k := H_k \phi(x^{k})$ defined this way, but then $C_{xy} = \frac1n \sum_k f_k^{T} f_k$? Could the authors clarify this derivation?
4. At present the proposed method appears to consist of a relatively straightforward combination of existing methods. Could the authors explicitly introduce the specific technical challenges their method addresses?
5. Regarding the experimental section: other than comparing with FUIT, the baselines seem to be variants of the proposed method. Would it be possible to include more state-of-the-art alternatives for comparison? Besides, the scale of experiments is small: only 3 clients, small sample sizes, low dimensionality of variables. Could the authors expand the empirical evaluation to larger number of clients, higher dimensional feature spaces, and more realistic federated settings?

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
This paper tackles the challenge of testing statistical independence in federated learning systems where data distributions vary significantly across clients. It introduces a framework called FedIT CS that aligns client data distributions using copula transforms and aggregates dependency signals through a stacking based mechanism. The method provides theoretical guarantees for controlling Type I error while maintaining detection power, even under client heterogeneity.
Extensive experiments demonstrate its superiority over existing approaches in both synthetic and real world datasets.

### Strengths
S1) This paper points out and demonstrates that FUIT's aggregation strategy is equivalent to the naive concatenation of samples in the feature space, and theoretically explains why such a simplistic approach fails in heterogeneous settings.
S2）The proposed method strikes a balance between privacy and efficiency. FedIT-CS-ML significantly reduces computational complexity while maintaining performance.
S3）This theorem provides a rigorous guarantee that the proposed federated independence test strictly controls the Type I error rate at or below the nominal significance level α.

### Weaknesses
W1) The core components employed in this paper, including Copula, RFF+CCA, permutation testing and homomorphic encryption, all represent existing methodologies. The paper's principal contribution resides in their systematic integration and domain-specific adaptation rather than pursuing  theoretical breakthroughs.
W2) The paper relies solely on the Sachs dataset for real world validation. This biological network benchmark contains a limited number of variables with well defined relationships and high data quality, which fails to represent the more chaotic and complex scenarios commonly encountered in practical federated learning environments. Consequently, the method's success on this curated dataset cannot convincingly demonstrate its generalizability. Additional experiments on more diverse real world datasets would be necessary to strengthen the persuasiveness of the proposed approach.
W3) The method works on the assumption that all clients must show the same kind of relationship all together—either all independent or all dependent. But in real situations, if some clients show independence while others show dependence, the goal of this method becomes unclear, and the way it combines results gets confusing. As a result, the method's promises might not hold. For example, Hospital A mainly treats patients with serious diabetes. There, insulin levels and blood sugar values are strongly connected. On the other hand, a health check-up center with mostly healthy people may show no clear connection that the method can find between these two measures. This is because a healthy body keeps them in balance naturally.

### Questions
Please refer to W1-W3.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of federated independence testing (FedIT) under client heterogeneity, where data distributions (marginal distributions, dependence strength/functional forms) vary across clients, but the global dependence status (independent/dependent) is consistent. It first analyzes limitations of existing methods (e.g., FUIT), which adopt naive feature-space concatenation and lack theoretical guarantees or effective handling of dependence dilution. The paper then proposes the FedIT-CS framework, combining copula-based marginal alignment (to eliminate marginal distribution discrepancies while preserving dependence structures) and stacking-based aggregation (to amplify intra-client dependence signals and select optimal client subsets). The framework includes three variants (FedIT-CS-M, FedIT-CS-ML, FedIT-CS-S) and incorporates homomorphic encryption (HE) for privacy preservation. Theoretically, it proves the soundness of the aggregated statistic (Theorem 4) and Type I error control (Theorem 5). Experimentally, it validates the method on synthetic datasets (covariance, frequency, functional heterogeneity) and the real-world Sachs dataset, showing that FedIT-CS variants outperform FUIT in balancing Type I/II error rates.

### Strengths
1.	Systematic problem analysis: The paper clearly identifies two core challenges of FedIT under heterogeneity—"naive aggregation pitfalls" (spurious dependence/independence) and "dependence dilution" (opposing correlations canceling out)—and links these to the limitations of existing methods, providing a clear motivation for the proposed framework.
2.	Theoretical rigor: It establishes formal guarantees for Type I error control and the soundness of the aggregated statistic, filling the gap of theoretical inadequacy in prior FedIT work (e.g., FUIT).
3.	Practical design: The framework balances privacy, efficiency, and performance: HE ensures privacy without accuracy loss; FedIT-CS-ML achieves linear-time complexity (O(KB)) for large-scale clients; and the use of second-order moments reduces communication costs compared to FUIT’s covariance matrix transmission (O(Kh²)).
4.	Comprehensive validation: Experiments cover both synthetic (three heterogeneous scenarios) and real-world datasets, with detailed analysis of Type I/II error rates and scalability, ensuring the method’s robustness across different settings.

### Weaknesses
1.	Scalability of FedIT-CS-M: FedIT-CS-M’s exponential complexity (O(2^K B)) makes it infeasible for K > 10 (as shown in Appendix I.3, FedIT-CS-MB—its permutation-based variant—fails to scale beyond 8 clients). The paper does not propose heuristic optimizations (e.g., greedy subset selection) to mitigate this, restricting its use in large federated systems.
2.	Limited real-world validation: The only real-world dataset used is Sachs (a signal network dataset), which has a fixed number of clients (7) and variables (11). No experiments on other domains (e.g., healthcare, finance) or larger client counts (e.g., K=50) are provided, making it hard to assess the framework’s generalizability to diverse real-world FedIT scenarios.
3.	Data splitting trade-off unaddressed: The paper uses a simple 8:2 data split to separate aggregation strategy training and testing, which reduces statistical power (evidenced by lower performance of FedIT-CS-M/FedIT-CS-ML vs. their "-F" variants with extra training data). No alternative strategies (e.g., nested cross-validation, split-free methods like Schrab et al. 2022) are explored, limiting the framework’s adaptability to small-dataset scenarios.
4.	Homomorphic encryption details lacking: Appendix D outlines the HE procedure but does not report key metrics like encryption/decryption time, communication latency, or memory usage. This makes it difficult for practitioners to evaluate the framework’s practicality in low-latency federated environments.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
To tackle independence testing in federated learning, the authors have proposed a copula-based marginal alignment technique combined with a stacking-based aggregation strategy.

### Strengths
The paper is overall well written. Type I error bound and soundness of aggregated statistics in Section 5 are the main strengths of this paper.

### Weaknesses
My major concern is the lack of comparison with closely related work and baselines. In particular, the independence testing in federated setting can be solved via density ratio estimation and matching: 

[1] M. Yamada and M. Sugiyama. Dependence minimizing regression with model selection for non-linear causal inference under non-Gaussian noise. AAAI 2010.

[2] M. Sugiyama and T. Suzuki. Least-squares independence test. IEICE TRANSACTIONS on Information and Systems, 94(6), pp.1333-1336, 2011.


[3] M. Sugiyama, T. Suzuki,  and T. Kanamori. Density-ratio matching under the bregman divergence: a unified framework of density-ratio estimation. Annals of the Institute of Statistical Mathematics, 64(5), pp.1009-1044, 2012.

[4] A. Ramezani-Kebrya, F. Liu, T. Pethick, G. Chrysos, and V. Cevher. Federated learning under covariate shifts with generalization guarantees. TMLR 2023.

[5] Z. Wu, C. Choi, X. Cao, V. Cevher, and A. Ramezani-Kebrya. Addressing label shift in distributed learning via entropy regularization. ICLR 2025.

For protecting privacy, the authors have used Homomorphic Encryption (HE). However, I do not see any novelty compared to typical aggregations with HE.

### Questions
Please address the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
