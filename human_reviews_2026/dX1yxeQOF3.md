# Double Momentum and Error Feedback for Clipping with Fast Rates and Differential Privacy

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Achieving both strong Differential Privacy (DP) and efficient optimization is critical for Federated Learning (FL), where client data must remain confidential without compromising model performance. However, existing methods typically sacrifice one for the other: they either provide robust DP guarantees at the cost of assuming bounded gradients/data heterogeneity, or they achieve strong optimization rates without any privacy protection. In this paper, we bridge this gap by introducing Clip21-SGD2M, a novel method that integrates gradient clipping, heavy-ball momentum, and error feedback to deliver state-of-the-art optimization and strong privacy guarantees. Specifically, we establish optimal convergence rates for non-convex smooth distributed problems, even in the challenging setting of heterogeneous client data, without requiring restrictive boundedness assumptions. Additionally, we demonstrate that Clip21-SGD2M achieves competitive (local-)DP guarantees, comparable to the best-known results. Numerical experiments on non-convex logistic regression and neural network training confirm the superior optimization performance of our approach across a wide range of DP noise levels, underscoring its practical value in real-world FL applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Clip21-SGD2M as a federated learning (FL) algorithm that simultaneously delivers strong differential privacy (DP) guarantees and optimal optimization rates. The method integrates heavy-ball momentum with gradient-clipping updates, corrected via error feedback (EF) to mitigate clipping bias. Although the algorithmic novelty is limited, the paper offers several theoretical contributions. Without assuming bounded gradients or bounded client heterogeneity, it proves: (i) an optimal $O(1/T)$ rate for the squared gradient norm under non-convex loss functions with full gradients; (ii) an optimal $O(1/\sqrt{nT})$ rate under stochastic (sub-Gaussian) gradients; and (iii) under local DP and certain conditions, a privacy–utility trade-off that matches the best known non-convex utility bound. The effectiveness of proposed method is investigated through several numerical experiments.

### Strengths
**S1: Clarifying the limitations of related works via counterexamples (Sec. 2)**

Using example and theorem, the paper shows that existing methods (Clip-SGD, Clip21-SGD) may fail to converge.




**S2: Sufficient theoretical contributions (Sec. 3)**

S2a: Weak assumptions. The analysis does not assume bounded gradients or bounded data heterogeneity, restricting assumptions to $L$-smoothness, lower boundedness, and sub-Gaussianity regarding stochastic gradients.

S2b: Full-gradient case. Achieves the optimal rate $O(1/T)$ for the squared gradient norm.

S2c: Stochastic-gradient case. Achieves the optimal rate $O(1/\sqrt{nT})$.

S2d: With local DP. Presents a privacy–utility trade-off for locally client updates wity DP, and shows consistency with the best-known non-convex utility bound when the model dimension $d$ is sufficiently large.

### Weaknesses
**W1: Concerns about the tightness of the convergence rates (Sec. 3)**

In FL, data heterogeneity is empirically known to significantly affect convergence rates. While the paper gains generality by not assuming bounded data heterogeneity, the reported rates do not depend on the data heterogeneity parameter, which raises the possibility that the rates may NOT be tight. Discussion appears limited regarding whether (and how) data heterogeneity parameters could or should appear in the bounds.


**W2: Limited experimental coverage (Sec. 4)**

While I understand the paper’s primary contribution is theoretical, there are several critical experimental gaps:


**W2a (Lack of heterogeneity evaluation).**
The paper does not report performance changes under controlled data heterogeneity, which is standard in FL (e.g., partitioning data to clients using a Dirichlet concentration parameter to artificially control heterogeneity).


**W2b (Insufficient SOTA comparisons).**
Comparisons to state-of-the-art methods seem incomplete. In the non-DP setting, could you compare with EF21-SGDM to isolate the pure effect of clipping? In the DP setting, beyond the three baselines already included, could you add DP-SCAFFOLD, PORTER, and DIFF2?

**W2c (Limited model cale).**
It appears there are no evaluations with larger models. In the DP experiments, could you test beyond CNN-MNIST, for example, CIFAR-10/100 classification with ResNet-20/VGG16?

**W2d (Result consistency).**
In several settings, the proposed method does NOT appear consistently superior.

### Questions
Q1: In the DP experiments with multiple $\epsilon$ values, how did you compute the Gaussian noise size?

Q2: Could you report the learning-rate tuning results? (e.g., a plot with learning rates on the x-axis and test accuracy on the y-axis, to make clear that the chosen learning rate is not at endpoints.)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a novel algorithm Clip21-SGD2M for federated learning with differential privacy. The algorithm combines 1) gradient clipping, 2) error feedback, and 3) double momentum (momentum on client and server). It achieves optimal convergence rates in the challenging but realistic setting of arbitrary data heterogeneity and stochastic gradients without any bounded-gradient assumption as in prior work. Experiments demonstrate a) that momentum is indeed essential to stabilize Clip21-SGD, and b) the proposed algorithm has empirically competitive (and in some cases superior) privacy/utility tradeoff to baselines that do not employ all three techniques.

### Strengths
This is a high-quality, technically dense paper that identifies a clear failure mode in existing methods, proposes a novel and well-motivated algorithm to fix it, and provides good theoretical and empirical validation. Theorem 2.2 proves that a recent and related algorithm, Clip21-SGD fails in the stochastic setting. The proposed algorithm combines three existing ideas to achieve optimal convergence rates in the challenging but realistic setting of arbitrary data heterogeneity and stochastic gradients without any bounded-gradient assumption as in prior work (Theorems 3.1/3.2). The paper also proves that the full algorithm achieves a competitive utility-privacy trade-off that matches state-of-the-art bounds (Theorem 3.3, Corollary 3.4). The experiments are very clear and demonstrate a) that the proposed algorithm is robust to the clipping threshold, while the baselines are not, and b) that controlling for the privacy budget $\epsilon$, the proposed algorithm has competitive-to-superior performance. The work is very clearly written and highly significant for the FL and private optimization communities.

### Weaknesses
The primary weakness of the paper is that the entire theoretical analysis assumes a full-participation model, where all clients participate at every round. This weakness is transparently acknowledged by the authors. The experiment in Fig 5 partially alleviates this by providing at least some preliminary empirical support for the effectiveness of the algorithm under subsampling.

Also, the experiments with DP were run only on MNIST -- even rerunning these experiments on CIFAR would strengthen the conclusions.

### Questions
Could you give the equivalent of Figure 4 using CIFAR? Or better yet, a truly federated dataset (and a different architecture) for example an LSTM trained on the Stack overflow dataset? This would strengthen the empirical claims of the paper, although I would not insist that it is necessary for a conference publication.

Typo: extra parens in algs 1/2

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript addresses the challenge of designing algorithms that provide both strong Differential Privacy (DP) guarantees and efficient optimization rates for Federated Learning (FL). The authors propose a framework, called, Clip21-SGD2M, to improve optimization performance while maintaining strong privacy guarantees. Clip21-SGD2M integrates gradient clipping, heavy-ball momentum, and error feedback to as improvement. The authors prove that the proposed method achieves optimal convergence rates for non-convex smooth distributed problems with heterogeneous client data, without requiring restrictive boundedness assumptions. They also provide a privacy analysis of Clip21-SGD2M, demonstrating competitive local DP guarantees.

### Strengths
The manuscript provides extensive theoretical analysis for smooth non-convex distributed objectives under arbitrary data heterogeneity, and Clip21-SGD2M achieves the optimal O(1/T) in the full-batch regime while maintaining a competitive local DP guarantee.

### Weaknesses
1. There are several prior works that has explored the application of error feedback 21 (EF21) in the DP setup. For example, I noted that the structure proposed in this paper is almost identical to that shown in "Smoothed Normalization for Efficient Distributed Private Optimization" while the latter paper seems to include some further improvement through normalization. Given the different assumptions on gradient (this paper assumes stronger sub-Gaussian), I cannot easily compare the results and the contribution of this paper is unclear. 

2. Clip21-SGD2M requires storing all historical updates for each client’s data, which introduces significant memory overhead. Moreover, the authors do not employ client or data sub-sampling in Clip21-SGD2M, which limits its scalability. 

3. The paper did not present a proper and consistent comparison with sota DP-SGD works. The experiment results are organized in a confusing way. The majority of the experiments ignore the noise but only focus on Clip21-SGD2M. For the experiments with DP guarantees, the results seem much worse than the best-known results even for simple dataset like MNIST. For example, from "DIFFERENTIALLY PRIVATE LEARNING NEEDS BETTER FEATURES (OR MUCH MORE DATA)", even for $\epsilon=1$, one can achieve accuracy over 98%. Also, the results for DP-Clip21-SGD2M on CIFAR10 is missing. 

3. A relatively minor issue here is that the theoretical analysis of Clip21-SGD2M is conducted under the assumptions of local smoothness and sub-Gaussian stochastic gradients. The latter is a stronger assumption than bounded variance, although the authors note that it is common to use sub-Gaussian local gradients for the type of high-probability analysis while only expectation-based results are presented in the paper.

### Questions
1. The experiments use privacy budgets ε∈{3,5.2,9,15.6,27} with δ=10^(-3), which covers moderate to weak privacy levels. Could the authors justify this choice and, if feasible, include results for smaller privacy budgets (e.g., ε≤1.5, δ>10^(-5)) to better illustrate performance under more practical privacy constraints? Also, please include concrete comparison with sota DP-SGD results on both MNIST and CIFAR10. 

2. The manuscript claims robustness under arbitrary data heterogeneity, but the experimental setup lacks details. Could the authors specify the exact data partitioning schemes (e.g., type and degree of non-IIDness) used in each experiment? 

3. Could the author detail and explain the key contribution of this paper compared to the prior work using EF21 in DP-SGD?

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
3

### Summary
The paper proposes a new DP-FedAvg style algorithm that additionally integrates error feedback into design. The proposed algorithm proven to converge without bounded gradient assumption on heterogeneous data.

### Strengths
1. Comprehensive analysis are provided. The algorithm mixes a few mechanism including gradient clipping, error feedback, and momentum under arbitrary client heterogeneity, and convergence is proven without boundedness assumptions which is somehow novel.
2. Experiments are provided to show validity and superiority of the proposed algorithm.

### Weaknesses
1. The main weakness is, as acknowledged in the paper, the algorithm can not benefit from privacy amplification by subsampling. This makes the privacy-utility trade-off way worse.
2. Gradient distribution is assumed to be sub-gaussian, which is still a relatively restrictive assumption.
2. Experiments are in limited scale and limited variants.

### Questions
Can authors comment more on privacy-utility trade-offs of the proposed algorithm given it cannot benefit from privacy amplification by subsampling?

### Soundness
3

### Presentation
3

### Contribution
2
