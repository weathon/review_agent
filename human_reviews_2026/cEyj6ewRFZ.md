# Nonparametric Teaching for Sequential Property Learners

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
We sincerely thank the reviewers for their time and feedback.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ReNT, a nonparametric teaching-based paradigm that accelerates the training of RNNs on sequence-structured tasks. The central idea is to analyze how sequence order shapes parameter-space gradients in RNNs and to recast the resulting evolution in function space. Across sequence-level regression/classification tasks, ReNT reduces wall-clock training time by ~33–46% while maintaining comparable generalization.

### Strengths
1. The paper gives a clear parameter-space derivation of order-aware gradients for RNNs (Eq. 10–12), showing how the power of recurrent weights encodes temporal dependencies and that the gradient shape is independent of sequence length (Eq. 12). This supports the claim that order impacts RNN updates in a principled way.

2. Results show consistent wall-clock savings with comparable test MAE/ACC on six benchmarks.

3. The work bridges nonparametric teaching with order-aware sequential learners and formalizes an RNTK→canonical-kernel connection for RNNs. 

4. The paper is clearly written and easy to follow.

### Weaknesses
1. The study focuses almost exclusively on vanilla RNNs. Modern sequence learners—state-space models (SSMs) and Transformer-based architectures—are not discussed or evaluated, leaving the practical impact and portability of the method to mainstream models unclear.

2. Comparisons are primarily “with vs. without ReNT.” Strong, SOTA data selection/curriculum baselines are missing. The evaluated setups emphasize simple RNNs and relatively small/clean datasets; the scale performance of large and noisy datasets is unknown. The paper provides insufficient architectural detail for the RNNs, which hinders reproducibility and fair comparison.

3. The paper lacks systematic ablations on key hyperparameter choices such as learning rate and epochs. Figure 4 is difficult to read (small fonts/markers); differences among methods are not clearly visible. Moreover, Figure 4 appears to show faster convergence in epochs for the baseline (better performance with fewer epochs). Please clarify whether compute/epoch budgets are matched, and how this affects the interpretation. This raises questions about generalization and stability on larger models/datasets.

### Questions
1. Could you report results on at least one SSM and one Transformer variant to demonstrate portability? If compute is tight, small-scale ablations (same data, reduced depth/width) would still be informative.

2. Please provide full RNN model specifications: number of layers, hidden sizes, activation functions, normalization, dropout, initialization, sequence lengths/windows, etc, for reproducibility.

3. Could you include more results of SOTA baselines, large models, and datasets to further investigate the effectiveness and scaling performance of the method?

4. Could you include metrics such as FLOPs and peak Memory usage for a more comprehensive efficiency evaluation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method to improve the training efficiency of RNNs by adaptively selecting the mini-batch at each gradient step.

### Strengths
N/A

### Weaknesses
The proposed method (Algorithm 1) is very simple, and the approach appears to be a minor extension of Zhang et al. (2023a;b; 2024; 2025) with modest adaptations to sequential data. The substantive contribution remains unclear, and the high computational cost of selecting teaching sequences may significantly limit both efficiency and applicability. The heavy terminology in the main text ("non-parametric teaching", "RNTK", "sequential property learner", etc.) is irrelevant and distracting.

I recommend that the authors rewrite the paper: move Algorithm 1 into the main text, add more discussion on the algorithm itself, and reduce the heavy terminology to improve clarity. The derivations in Sections 4.1 and 4.2 are lengthy with little insight (Proposition 4 seems fairly standard) and could be shortened.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ReNT, a nonparametric training paradigm where the training data is sequenced according to the largest $m$ samples which maximizes $\\|f_{\theta}-f^*\\|$. The proposed algorithm led to a reduce in time for training RNNs for about $-40\\%$ while keeping the test performance.

### Strengths
1. The ReNT algorithm managed to reduce the training time of RNN significantly while maintaining the genralization performance.
2. The paper writing is good and comprehensive.

### Weaknesses
1. The contribution compared with [1] seems limited. Theorem 3 and Proposition 4 is exactly the same as in [1], and the key idea of the ReNT algorithm, which is amplifying the steepest gradients by selecting the largest $m$ data to maximize $\\|f_{\theta}-f^*\\|$, does not depend on the specific neural network architecture but rather the loss function of $\mathcal{L}=\frac{1}{2}(f_{\theta} - f ^ *)^2$ . The implementation of ReNT algorithm therfore should not be much different from [1] besides changing the baseline neural network from an MLP to an RNN. The new part is perhaps sec 4.1 that examines the impact of sequence order on parameter-based gradient descent. But this is merely computing the gradient of RNN, which should not be a contribution of its own.




[1] Zhang, C., Luo, S. T. S., Li, J. C. L., Wu, Y. C., & Wong, N. (2024). Nonparametric teaching of implicit neural representations. arXiv preprint arXiv:2405.10531.

### Questions
See weakness.

### Soundness
4

### Presentation
4

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
The paper interprets gradient descent on parameters through a non-parametric lens, highlighting its similarity to functional gradient descent. This perspective goes beyond the standard Recurrent Neural Tangent Kernel (RNTK) analysis by being order-aware, explicitly taking the sequential nature of the data into account. Furthermore, by leveraging this equivalence, the authors propose a novel algorithm for learning RNNs. This algorithm strategically selects a subset of the data to ensure faster convergence. Empirical results demonstrate the algorithm's performance gains.

### Strengths
- The paper goes beyond the  recurrent neural tangent kernel and shows that gradient descent  on parameter descent aligns with the evolution in functional gradient descent with the order-aware canonical kernel. This looks an interesting tool to study the dynamics of parameter-based gradient descent.  

- Using this equivalence the paper develops an efficient learning algorithm of RNNs which is empirically validated which seems convincing.

### Weaknesses
The paper appears not to be mathematically rigorous, see the questions below for detailed description.

### Questions
i) The algorithm the paper suggests require access to f_*(S), what happens when $f_*$ cannot be computed ? 

ii)  For the sufficient loss reduction lemma, what is the descent algorithm used here - is it parameter descent or functional descent ? 

iii) In theorem 3, is the parameter convergence assumed ? In particular in the proof how $\lim_{t\to\infty} \left[ \frac{\partial L(f_{\theta^t}(S), y}{\partial f_{\theta^t}(S)} \right]  = 0$?

### Soundness
1

### Presentation
3

### Contribution
2
