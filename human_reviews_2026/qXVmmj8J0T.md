# Smooth Calibration Error: Uniform Convergence and Functional Gradient Analysis

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
Calibration is a critical requirement for reliable probabilistic prediction, especially in high-risk applications. However, the theoretical understanding of which learning algorithms can simultaneously achieve high accuracy and good calibration remains limited, and many existing studies provide empirical validation or a theoretical guarantee in restrictive settings. To address this issue, in this work, we focus on the smooth calibration error (CE) and provide a uniform convergence bound, showing that the smooth CE is bounded by the sum of the smooth CE over the training dataset and a generalization gap. We further prove that the functional gradient of the loss function can effectively control the training smooth CE. Based on this framework, we analyze three representative algorithms: gradient boosting trees, kernel boosting, and two-layer neural networks. For each, we derive conditions under which both classification and calibration performances are simultaneously guaranteed. Our results offer new theoretical insights and practical guidance for designing reliable probabilistic models with provable calibration guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper adds on to the accumulating evidence on the connection between loss minimisation and calibration. They adopt smooth calibration error and derive the uniform convergence result to relate the training set calibration error to test time error, and then show that training set smooth calibration error can be studied in connection to functional gradient. While the insight is simple, in hindsight, this helps answering some questions as to when one can get better predictors (accuracy wise) while also being calibrated, and the paper develops results for boosting (gradient and kernel) and two-layer neural networks. There is an extensive amount of work and details in the appendix as well (which I didn't carefully check) but also the experimental demonstration.

### Strengths
1.This is a strong paper, that helped answer some of my own questions on the connections between loss minimisation and calibration. My favourite part is the connection to functional gradient. 

Question wise: is it fair to conclude that loss minimisation will yield calibration together with good performance if the training resembles that of the functional gradient descent or some approximation of it? In that sense, an earlier paper by (Błasiok et al., 2023; when does minimising proper losses yield calibration) argued for the emergent calibration behaviour in modern large neural networks, and how do the current functional gradient analysis add on to that?


Overall, I think this is a good paper, with not much concerns, and I think this should be accepted.

### Weaknesses
1. The paper misses to mention [1]. While the focus of this paper is re-calibration, I think they also study the generalisation of calibration error, and might be worth to look at. 
2. While the focus of the paper is theoretical, could the study give some concrete practical guidance to design predictors to achieve both calibration and better accurate predictors. 



[1] Masahiro Fujisawa, Futoshi Futami. PAC-Bayes Analysis for Recalibration in Classification. ICML 2025

### Questions
see above.

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
This paper presents a two part framework for smooth calibration error or smCE (for the binary case). It first provides uniform convergence bounds that control the population level smCE by training smCE plus a generalization gap that depends on the hypothesis class. The arguments involve standard chaining number, Dudley entrop-integral type techniques (standard Kolmogorov-Tikhomirov fare) for showing a Rademacher bound. Second, it presents a functional gradient view that provides an upper bound on the training smCE by norms (of functional gradients) for proper losses. This can result in concrete guarantees for gradient boosting machines, kernel boosting, and two layer NNs (NTK regime), Under clearly articulated margin assumptions, appropriate values for T, w, and n, we get $\epsilon$-smCE and $\epsilon$ misclassification simultaneously. This teases out a clear optimization v capacity trade off. 

To expand in more detail: The paper focuses on a proper loss setup, in particular on squared loss and cross entropy. smCE is defined as a supermum over 1-lipschitz post-processing of the prediction. This is connected to the post-processing gap under squared loss (Biasiok et al). For cross entropy, the dual smCE is defined on the logit with 1/4-lipschitz post-processing and the relation showing the gap is explained. Then a covering number argument is used to bound sup_f |smCE(f,Ste) − smCE(f,Str)|, avoiding explicit composition class Lip \odot f. Theorem 1 provides a tail bound with an integral over log covering numbers of F (only F, not of Lip \odot f), plus an O(1\(sqrt(n))) term reflecting the Lipschitz part. Theorem 2 provides an alternative Rademacher bound. Again, the key point is that the bound depends on F and not on Lip \odot F. The main observation is that training smCE equals a dual pairing between h(f(X)) and negative functional gradients. This means that small functional gradients imply small training smCE. After this three cases are explicated (GBT, kernel boosting, 2 layer NNs).

### Strengths
I like the conceptual clarity of the paper, although I can't say I am extremely familiar with the theoretical literature here. It links calibration to functional gradients, turning calibration control into optimization-style quantities.

The uniform convergence bounds only depend on  F (plus a small fixed term).

The results for the three cases are interesting and non-trivial and make the framework concrete, with explicit T, w, n, choices and accuracy-calibration trade-off.

### Weaknesses
The margin assumptions (classification-style) are strong for a calibration question. Many realistic miscalibration settings lack separability. (pages.4–8)

I found the kernel dependence opaque. The population bounds hide kernel spectrum dependence. 

Another weakness has more to do with NTK than the framework itself.: Finite-width and feature-learning behaviors would have been interesting to discuss. 

The dual smCE <> post-processing gap constants are not discussed for tightness. The $C/\sqrt(n)$ term's origin and size is not specified enough (unless I missed it).

### Questions
In the dual post-processing gap relation (p.2), can the (2,4) constants be improved? Would tighter constants significantly change the optimization/complexity trade-offs later?

The first $C/\sqrt(n)$  term is attributed to the Lip class. Is it possible to expose the exact constant (and L dependence) and show whether it is improvable by structure in h (e.g. by monotonicity)? 

If $\gamma$ is small/zero (heavy overlap, label noise), do functional-gradient controls still yield non-trivial smCE bounds (perhaps with slower rates) via early stopping or other regularization?

Is it possible to restate Theorem 4’s population term explicitly in terms of eigen-decay (like effective dimension $N(\gamma)$. It might be useful to help pick kernels.

What breaks in the analysis if we move to softmax?

For the Dudley chaining analysis, and some of the other standard arguments, please provide in-line citations.

### Soundness
3

### Presentation
3

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
1.	The authors derive a uniform convergence bound for smooth CE.
2.	They prove that such training smooth CE can be bounded by the norm of the functional gradient of the loss evaluated on training data
3.	They apply the theoretical framework to three representative algorithms for the binary classification closely tied to functional gradients.

### Strengths
The paper provides a solid theoretical guarantee for the calibration performance and the classification error for three standard binary classification algorithms.

### Weaknesses
1. The technical novelty of deriving the results in Section 4 is not clear. In Section 4, the authors present the connection between functional gradients and the smooth CE and derive the convergence rates for three example models. These results heavily rely on the existing conclusion and analysis, but it should be clarified which contents are novel and important contributions, and why they are. 

2. Lack of comparison for the uniform convergence results with the previous ones.

### Questions
1. Can you compare with existing works that analyze the test accuracy or smooth CE to validate your results in Section 4?

2. What guidance or improvement do the theoretical results (for example, Corollaries 3,4) bring to the algorithms in practice? Or could you design better algorithms to achieve both calibration and accuracy?

3. Neural networks are known as non-calibrated, so calibration methods, such as temperature scaling and vector scaling, are proposed. Can the theoretical framework be applied to these post-hoc techniques, and what results can be derived?

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
This paper presents a theoretical understanding of learning algorithms that can produce accurate and well calibrated results. The paper mainly focuses on Smooth Calibratin Error(smCE) and provides a uniform convergence bound to that. Also, this paper analyses classifiation and calibration performance of three representative algorithms gradient boosting trees, kernel boosting and two-layer neural networks.

### Strengths
1. The paper attempts to provie a theoretical understanding between the relation of calinbration and classification and provided bounds of smCE. 
2. The paper analyses the conditions an algorithm needs for achieving both classification and calibration gurrantees. This is extremely crucial as calibration is very important in safety critical tasks.

### Weaknesses
1. There are issues in the proof of theorem 1. In line 999, the author has refered to Lemma 7.6 of  Blasiok et al. (2023) and justified the existence of an optimal soultion to the convex problem in equation 8. But the assocaiation with lemma 7.6 is not obvious. In the lemma 7.6 of Blasiok et al. (2023), $\Pi(u,v,y)$ is a distribution function, hence $\Pi(u,v,y) \in [0,1]$. But, in line 997 the author has stated that $\omega_i \in [-1,1]  $.

2. The inequality in line 1343 is not clear. Means how the inequality is being derived from line 1339. 

3.Same problem with the inequality in line 1357. 

Overall, the theoretical section is excessively long; however, the three issues mentioned above are particularly critical, as they undermine the validity of Theorem 1. Consequently, the soundness of Corollary 1 and Theorem 2 also comes into question.

### Questions
1. Explain the relation with lemma 7.6.

2. Explain the inequality of both 2 and 3 in the weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1
