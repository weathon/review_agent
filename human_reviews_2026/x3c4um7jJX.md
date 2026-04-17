# Accurate Estimation of Mutual Information in High Dimensional Data

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Mutual information (MI) is a fundamental measure of statistical dependence between two variables, yet accurate estimation from finite data remains notoriously difficult. No estimator is universally reliable, and common approaches fail in the high-dimensional, undersampled regimes typical of modern experiments. Recent machine learning–based estimators show promise, but their accuracy depends sensitively on dataset size, structure, and hyperparameters, with no accepted tests to detect failures.
We close these gaps through a systematic evaluation of classical and neural MI estimators across standard benchmarks and new synthetic datasets tailored to challenging high-dimensional, undersampled regimes. We contribute: (i) a practical protocol for reliable MI estimation with explicit checks for statistical consistency; (ii) confidence intervals (error bars around estimates) that existing neural MI estimator do not provide; and (iii) a new class of probabilistic critics designed for high-dimensional, high-information settings. We demonstrate the effectiveness of our protocol with computational experiments, showing that it consistently matches or surpasses existing methods while uniquely quantifying its own reliability.
We show that reliable MI estimation is sometimes achievable even in severely undersampled, high-dimensional datasets, provided they admit accurate low-dimensional representations. This broadens the scope of applicability of neural MI estimators and clarifies when such estimators can be trusted.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper provides an evaluation of discriminative mutual information estimators and claims to provide three main contributions regarding reliable MI estimation checking the statistical consistency, confidence intervals, and a new class of critics for high-dimensional settings. Moreover, I would add that there is a fourth minor contribution related to the analysis of the finite-data regime. The results in high dimensions look promising, but the only analyzed scenario consists in low values of true MI.

### Strengths
- The paper targets a fundamental problem in MI estimation
- Good results with high dimensionality and few samples available
- Investigation of finite-data regime, which has been neglected in many previous works

### Weaknesses
- The paper treats only few discriminative MI estimators, thus excluding more novel discriminative MI estimators and generative MI estimators. The paper mainly focuses on MINE, SMILE, and InfoNCE. Thus it is not considering a broad set of discriminative MI estimators. For instance, the paper is not including in the main part the estimators based on the variational representation of the f-divergence, such as NWJ (used in some experiments) and $f$-DIME [1]. If the authors want to keep the focus on discriminative MI estimators, I think that this should be explicitly stated, and they should analyze also more novel estimators (e.g., [1]). If instead they want the paper to be considering any neural MI estimator, they should include generative estimators, such as [2].
- Related to the above point, in line 137 the authors claim that neural network-based MI estimators typically rely on a critic that approximates the log-density ratio, but this mostly holds for the DV-based estimators. Many other discriminative and generative estimators target the estimate of the density-ratio. Some estimators directly estimate the different densities.
- The authors analyze the joint and separable critics, that have been proposed in 2018. However, a different critic has been proposed in NeurIPS 2024, with the advantage of a lighter computational complexity [1]. Since one of the paper novelties is the novel critic, I think the authors should include in the analysis also this recent architecture. 
- The authors claim to work with "high-information", but the experiments in high dimension focus on MI<10 and in particular on MI=4
- Minor readability issues: MI defined two times, first sentence in abstract and introduction is exactly the same. Figure 7 covers part of the text.

[1] Letizia, N. A., Novello, N., & Tonello, A. M. (2024). Mutual Information Estimation via $ f $-Divergence and Data Derangements. Advances in Neural Information Processing Systems, 37, 105114-105150.

[2] Franzese, G., Bounoua, M., & Michiardi, P. (2023). MINDE: Mutual information neural diffusion estimation. ICLR 2024.

### Questions
Questions:
- In line 108 the authors write that MINE suffers from high variance. However, if MINE is defined as in line 106, it is also biased. Did the authors mean something else?
- For the proposed Concatenated Quadratic Critic, the paragraph the authors write starting in line 152 holds only for Gaussian distributions? So how does $I_{CCA}$ perform for non-Gaussian scenarios (even when the initial distributions are not Gaussian)?
- It appears that in the finite sample setting the mi estimate can increase above the true value, am I right? Did the authors investigate this phenomenon? Why does this happen?
- The stopping heuristic in line 306 stops where the test value peaks. But what is the guarantee that this does not overestimates the MI? Is there any theoretical justification for this approach?
- In line 388 the authors write that no estimator is unbiased for all distributions. Why and when is NWJ biased for instance?
- Why should the linear fitting proposed work in general (in Sec. 4.3)? What is the rationale behind this? What theoretical guarantee leads to this? I see that empirically this works in the experiments that you reported, but the experiments only consider a true MI of 4.
- In the guidelines in Appendix A.4 the authors say that the user should choose the MI estimator based on the true range of MI. How can I choose the MI estimator based on what’s best for the considered case, if I am in an unknown scenario for which I have no idea about the true MI?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a practical protocol for accurate mutual information (MI) estimation in high-dimensional data. The protocol includes early-stopping heuristics, internal bias checks, and a subsampling-extrapolation workflow. Additionally, the authors introduced probabilistic critics (VSIB variants) designed for high-information regimes and provided confidence intervals. The authors demonstrate that reliable MI estimation is possible in high dimensions when data has low-dimensional latent structure, requiring sufficient critic expressiveness and adequate sample size relative to the latent dimensionality, as validated on the noisy MNIST dataset. Critically, the authors claim that their approach consistently avoids overestimating the ground truth, which is crucial for preventing false positives in scientific applications.

### Strengths
- **Practical methods for challenging regimes**: The paper introduces multiple techniques (early-stopping, VSIB regularization, subsampling-extrapolation) that enable MI estimation in severely undersampled, high-dimensional settings where traditional methods fail—for instance, 784-D MNIST with ~10⁴ samples versus the hundreds of thousands required by traditional approaches.

- **Clear regime categorization and comparative analysis of estimators**: The authors categorized MI estimation into three distinct regimes—(1) low-dimensional with infinite data, (2) high-dimensional with infinite data, and (3) high-dimensional with finite data—and systematically compared classical methods (CCA, KSG) against neural estimators (InfoNCE, SMILE, and their VSIB variants), demonstrating that InfoNCE and SMILE consistently outperform alternatives across these regimes.

- **Detailed step-by-step protocol for practical implementation**: The paper provides a comprehensive workflow thatl addresses the practical challenge that existing neural MI estimators lack clear guidelines for hyperparameter selection and reliability assessment.

### Weaknesses
- **Insufficient Discussion of Recent Work in Related Work Section**: The Related Work section (Section 2) provides a solid foundation covering traditional methods and early neural estimators. I suggest expanding the discussion to include several recent approaches (2020-2024) for high-dimensional MI estimation. While some of these works appear in the References and Appendix comparisons (e.g., [B] in A.7.3), discussing them in Section 2 would help readers better understand the current landscape and the paper’s contributions.
    - Normalizing flows for MI estimation [A]
    - Latent space reduction [B]
    - Data derangement techniques [C]
    - Loss regularization and moving averages [D, E]
Adding a brief paragraph in Section 2 that surveys these methods and explains how the proposed approach relates to them would provide valuable context and clarify the novelty of this work.
    
- **Limited Comparison with Regularization Methods**: The main paper lacks sufficient experimental comparison and analysis with methods that employ regularization techniques, such as self-regularizing approaches (e.g., NWJ), loss regularization methods [D, E], or gradient regularization through data sampling strategies [C]. Given that one of the paper’s key contributions is addressing training instability through the VSIB wrapper and early-stopping, it would be valuable to include discussion of these alternative stabilization strategies in the main text. Could the authors comment on the limitations of these regularization-based methods and explain why the proposed VSIB approach is preferable or complementary to them? This would help readers better understand the positioning and advantages of the proposed method.

- **Questionable representativeness of validation datasets**: The paper validates its methods primarily on synthetic data and MNIST. Although MNIST is 784-dimensional, it is highly structured and relatively simple. It is unclear whether these benchmarks represent the complexity of real-world high-dimensional problems in the target application domains. Compared to [D], which validated their regularization approach on more complex real-world datasets like CIFAR-10 and CIFAR-100, the experimental settings in this paper may not provide sufficient evidence for the practical applicability of the proposed methods in challenging real-world scenarios.

[A] Butakov, Ivan, et al. Mutual Information Estimation via Normalizing Flows, NeurIPS 2024

[B] Gowri, Gould, et al. Approximating mutual information of high-dimensional variables using learned representations, NeurIPS 2024

[C] Letizia, Nunzio Alexandro, et al. Mutual Information Estimation via $ f $-Divergence and Data Derangements. NeurIPS 2024

[D] Choi, Kwanghee, and Siyeong Lee. Combating the instability of mutual information-based losses via regularization. UAI 2022

[E] Choi, Kwanghee, and Siyeong Lee. Regularized mutual information neural estimation. Arxiv 2020

### Questions
* VSIB regularization and loss/gradient regularization methods do not appear to be mutually exclusive. Can REMINE be applied on top of the VSIB wrapper? If such a combination is feasible, wouldn’t they be complementary since VSIB addresses instability through probabilistic embeddings while REMINE regularizes the loss itself?”

* I remain open to revising my assessment should the authors clarify any misunderstandings or address the concerns raised in this review.

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
This paper presents a comprehensive study of mutual information (MI) estimation in high-dimensional data settings, focusing in particular on the practical challenges faced by classical and neural network-based estimators. The authors propose a principled protocol incorporating new probabilistic critic architectures, a stopping heuristic to prevent overfitting, and explicit checks for estimator reliability through statistical consistency and confidence intervals. Extensive experiments on synthetic and real datasets are provided to benchmark the approach, alongside detailed guidelines for MI estimation workflows. Throughout, the paper positions its contributions in the context of high-dimensional, undersampled regimes that commonly defeat standard approaches.

### Strengths
**1. Practical Protocol for High-Dimensional MI Challenges**: This work delivers a ready-to-use estimation protocol with confidence intervals, overfitting checks (e.g., max-test early stopping), and expressivity diagnostics. It tackles undersampled high-dim regimes where prior methods fail, offering a complete toolkit unseen before. Ideal for neuroscience and vision, it curbs errors in causal inference. 

**2. Comprehensive Benchmarking Across Scenarios**: The analysis probes N, k_Z, MI strength, and expressivity via synthetic (Gaussian/nonlinear) and real (Noisy MNIST, K=784) data. Phase diagrams (Fig. 5) illuminate failure boundaries (e.g., N ≳ K_Z^2 / I), spanning infinite/finite limits.

### Weaknesses
**1. Limited Novelty in Methodology (major)**: The protocol builds on existing estimators (e.g., InfoNCE, SMILE, VSIB variants) with added analyses and tweaks, but lacks groundbreaking innovations. It feels more like a refined integration than a fundamental advance, potentially diluting its standout contribution in a crowded neural MI landscape.

**2. Incomplete Benchmarking Against existing methods (major)**: methods like MINE and works in [1] and [2] are suitable for estimating high-dimensional MI. Leaving claims of outperformance underexplored.

**3. Empirical Scope and Heuristic Reliance**: While synthetic/Noisy MNIST tests are thorough, real-world validation seems to be conducted only on one dataset (Noisy MNIST, K=784). Broader, high-complexity experiments would strengthen the authors' point.


[1] Chen, Yanzhi, et al. "Neural approximate sufficient statistics for implicit models." arXiv preprint arXiv:2010.10079 (2020).

[2] Gowri, Gokul, et al. "Approximating mutual information of high-dimensional variables using learned representations." Advances in Neural Information Processing Systems 37 (2024): 132843-132875.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a “practical protocol” to improve neural MI estimation using:
+ An optimal early-stopping rule.
+  Stratified sampling
+ A probabilistic critic (VSIB) wrapper for DV-style estimators. 

Main results are on synthetic teacher models; one real dataset is noisy-MNIST.

### Strengths
+ The paper shows explicitly that the latent dimension and not the dimension of the data governs sample complexity.
+ The appendix contains insightful, but known, derivation relating mutual information to CCA.

### Weaknesses
+ The low dimensional structure angle is unoriginal and under-cited. The whole pipeline hinges on low-dimensional latent structure; the paper even talks in those terms but doesn’t situate itself in the manifold-hypothesis [3] and ignores that it is the corner stone on which neural estimation of mutual information is built [1]. 
+ Empirical evaluation is too weak for the claim. One real dataset (noisy-MNIST) and a thin slice of synthetic tasks. 
+ The bias/variance trade offs of separable/joint critics are already well studied in [2]. 
+The paper also suffer from awkward pacing, the  protocol itself is relegated to page 8.
+ The tone of the paper is sometimes grandiose while referring to known or marginal results.

[1] Belghazi, Mohamed Ishmael, et al. "Mutual information neural estimation." International conference on machine learning. PMLR, 2018.
[2] Poole, Ben, et al. "On variational bounds of mutual information." International conference on machine learning. PMLR, 2019.
[3] Bengio, Yoshua, Aaron Courville, and Pascal Vincent. "Representation learning: A review and new perspectives." IEEE transactions on pattern analysis and machine intelligence 35.8 (2013): 1798-1828.

### Questions
How does the concatenatic quadratic critic relate to discriminators in LQG setting [4]?
[4] Feizi, S., Farnia, F., Ginart, T., & Tse, D. (2017). Understanding gans: the lqg setting. arXiv preprint arXiv:1710.10793.

### Soundness
1

### Presentation
3

### Contribution
1
