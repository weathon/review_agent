# Variance Matters: Improving Domain Adaptation via Stratified Sampling

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 2

## Abstract
Domain shift remains a key challenge in deploying machine learning models to the real world. Unsupervised domain adaptation (UDA) aims to address this by minimising domain discrepancy during training, but the discrepancy estimates suffer from high variance in stochastic settings, which can stifle the theoretical benefits of the method. This paper proposes Variance-Reduced Domain Adaptation via Stratified Sampling (VaRDASS), the first specialised stochastic variance reduction technique for UDA. We consider two specific discrepancy measures – correlation alignment and the maximum mean discrepancy (MMD) – and derive ad hoc stratification objectives for these terms. We then present expected and worst-case error bounds, and prove that our proposed objective for the MMD is theoretically optimal (i.e., minimises the variance) under certain assumptions. Finally, a practical k-means style optimisation algorithm is introduced and analysed. Experiments on three domain shift datasets demonstrate improved discrepancy estimation accuracy and target domain performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Discrepancy minimization-based domain adaptation methods suffer from empirically estimating the domain discrepancy due to the variance of the estimated discrepancy caused by the mini-batch stochasticity. This work addresses this problem by constructing mini-batches based on stratified sampling to reduce the variance of the discrepancy. This work proposes to improve two domain adaptation methods, MMD and CORAL, based on the proposed approach, and demonstrates their rationality both theoretically and experimentally.

### Strengths
1. Theoretical claims are solid. I reviewed the theoretical descriptions and proofs thoroughly but found no obvious errors. The proposed methods are supported by these theoretical insights and are considered reasonable.
2. The effectiveness of the proposed methods has been experimentally verified. This work proposes improved versions of MMD and CORAL, experimentally demonstrating their superiority over the respective baseline methods.

### Weaknesses
1. The contribution of this work to the research are of domain adaptation is not so significant. The proposed theoretical framework has only been applied to MMD and CORAL, and its effectiveness has not been demonstrated for the numerous other domain adaptation methods.
2. Comparison with previous works is not sufficient. The comparison results with existing domain adaptation methods are not given, making it unclear how well the proposed method performs relative to current domain adaptation works.
3. Discussion about the related works is not given. The paper lacks a comprehensive discussion of related research, leaving the positioning and contributions of the proposed method unclear.
4. The paper organization is not well and it is difficult to read. Some mathematical notations (e.g., $f(S)_{(i)}$, $g(S)_{(i)}$) are not properly defined, making it difficult to understand.

### Questions
Can the theoretical findings of this work be useful beyond MMD and CORAL?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
VaRDASS is the first specialised stochastic variance reduction technique designed for Unsupervised Domain Adaptation (UDA). The authors designs stratified sampling to reduce the variance of the estimates and derives the mathematical relationship between stratified sampling and variance minimization, theoretically deriving the optimal stratification objectives for both the MMD and CORAL losses. The authors proposes a dynamically-weighted kernel k-means clustering algorithm, where the dynamic weighting achieves more balanced strata, and employs a randomized greedy algorithm to reduce computational complexity. Experiments demonstrate that VaRDASS both reduces the estimation error of the domain discrepancy for a given minibatch size and improves target domain accuracy on three UDA benchmark datasets.

### Strengths
1、Originality: This work presents the first stochastic variance reduction method specifically designed for Unsupervised Domain Adaptation (UDA), introducing a novel perspective by investigating the stability of domain alignment through variance control.

2、Quality: The mathematical proofs are rigorous, effectively integrating theory with experimental validation. Ablation studies are provided to verify the contribution of each component of the proposed algorithm.

3、Clarity: The paper is well-structured, with a logically coherent flow from the problem motivation through to the theoretical derivation, algorithmic design, and experimental validation.

4、Significance: The method enhances model stability and target domain performance.

### Weaknesses
1、The theoretical explanations are sometimes overly dense and could benefit from more intuitive elucidation. Furthermore, the formatting of some formulas and the definition of certain symbols are inadequate; for instance, the formatting of Equation 8 is non-standard, and the range of the symbol "Z" in the last paragraph of Page 5 is ambiguous.

2、In Section 2.5 of the paper, it is mentioned that the stratified structure is reconstructed every T=100 steps, but no analysis is provided on how variations in T affect performance, variance, and computational time.

3、The experimental comparisons are limited to only three sampling methods, lacking validation of the method's generalization capability to other, more recent domain adaptation algorithms.

4、The paper lacks a quantitative analysis of the total time cost (wall-clock time) associated with the proposed approach.

### Questions
Please refer to Weakness

### Soundness
3

### Presentation
2

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
The paper addresses the problem of high variance in domain discrepancy estimation within unsupervised domain adaptation (UDA). Specifically, it focuses on two commonly used discrepancy measures: MMD and correlation alignment. The authors propose the first specialised stochastic variance reduction technique for UDA, called Variance-Reduced Domain Adaptation via Stratified Sampling (VaRDASS). In particular, they prove that the proposed objective for MMD is theoretically optimal under certain assumptions. Experiments on three UDA benchmark datasets demonstrate that VaRDASS effectively improves target-domain accuracy compared with baseline sampling strategies.

### Strengths
(i) This paper focuses on the high variance in domain discrepancy estimation and introduces the first specialized stochastic variance reduction technique for UDA, filling a technical gap in the field.

(ii) The proposed method is based on strong theoretical foundations. In particular, the theoretical framework for the MMD provides solid justification of its optimality.

(iii) Experiments on three UDA benchmarks demonstrate that the proposed method outperforms other sampling variants in terms of the target-domain accuracy.

### Weaknesses
(i) The significance and generality of the method are a concern. Since the paper specifically focuses on the high variance of domain discrepancy estimation in UDA, and discrepancy minimization is only one line of UDA methods, its impact on advancing the state-of-the-art in UDA—including performance, efficiency, and generality—is unclear.

(ii) The baselines for variance reduction in the experiments are limited, which restricts the experimental justification of the method. The literature discussion on variance reduction in Lines 42–63 is comprehensive. However, in the experimental results shown in Table 1, only naive variants of the proposed method are considered for comparison, while all relevant and applicable variance reduction methods mentioned previously are ignored.

(iii) The empirical justification and analysis of the method are limited. Although the method aims to reduce variance, no empirical evidence is provided to demonstrate whether it achieves this goal, and only accuracy metrics are reported. According to Table 4 in the Appendix, the proposed method appears to increase both the mean and variance of domain discrepancy compared with other sampling methods, which is confusing.

(iv) As noted in the limitations section, the stability and scalability of the proposed method to large-scale datasets are a concern.

### Questions
The main concern is the weak empirical justification of the method due to the lack of key baselines and analyses.

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
4

### Summary
The paper addresses the instability in the domain alignment loss due to high variance when estimating domain discrepancy losses using small minibatches. These minibatches are usually created using uniform random sampling. The paper proposes a technique of stratified sampling (organizing the dataset into minbatches) such that the variance in the domain alignment measure is minimized. 
This is a NP hard problem. They propose a greedy approach. Show theoretical justification in the reduction of Variance due to stratified sampling for MMD and CORAL.

### Strengths
1. Novel approach to effectively align domains. 
2. Theoretical justification to demonstrate stratified sampling reduces variance in L_MMD and L_CORAL measures.

### Weaknesses
1. Theoretical guarantees for reduction in variance have not translated to empirical gains in better domain alignment. There is not much experimental evidence this works. There is no comparison with baselines (other DA methods)
2. The proposed model has not been evaluated with standard Domain Adaptation datasets in Computer Vision like DomainNet, OfficeHome, VisDA or Office-31. 
3. Variance in L_DA can be addressed with exponential Moving Average approaches like we do with momentum. The paper could have compared that as a baseline. 
4. What is the intuition behind the weighting D_ij(n_j + 1). What does it physically mean to weight the distance by the current cluster size before making an assignment. This has not been discussed. 

Minor: 
1. Notation is confusing. n_s is used for Source dataset size. It is also used for minibatch size in algorithm 2.  Likewise n_1,...,n_j are used to indicate cluster sizes. A more discernable notation would help. 
2. There is no discussion on how mini-batching with different batch sizes was implemented or if and how the cluster sizes were balanced. 
3. Was clustering re-implemented at every epoch or after every gradient update and what was its effect on the compute overhead.

### Questions
None.

### Soundness
2

### Presentation
2

### Contribution
3
