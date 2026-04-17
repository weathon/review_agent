# Breaking Independence: Learning Correlated Views for Variational Incomplete Multi-View Clustering

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Incomplete multi-view clustering (IMVC) aims to uncover shared cluster structures from data with partially observed views. Although recent imputation-free methods based on variational inference demonstrate robustness to missing views, they commonly rely on a conditional independence assumption across views, which fails to capture the inherently structured and potentially correlated nature of multi-view data. In this paper, we propose a variational framework that explicitly breaks this assumption by introducing a learnable cross-view correlation structure. Specifically, we explicitly model and learn correlations between views by utilizing the covariance structure of posterior estimation errors. To facilitate robust and efficient learning, the correlation matrix is parameterized through a normalized Cholesky decomposition, ensuring positive definiteness and enabling the entire model to be trained jointly through a unified variational objective. Extensive experiments on multiple IMVC benchmarks demonstrate that our method consistently outperforms state-of-the-art approaches across a wide range of missing-view settings. These results highlight the effectiveness of adaptive correlation modeling in variational incomplete multi-view clustering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the independence assumption in multi-view learning, arguing that in real-world scenarios, different views are often statistically correlated rather than independent. The authors propose a new framework that explicitly models inter-view correlations, achieving better results on various benchmarks.

### Strengths
1.The proposed framework introduces a clear mathematical mechanism to capture inter-view dependencies, instead of assuming factorized likelihoods.

2.The derivations are formal, and empirical results confirm that dependency modeling improves both accuracy and robustness.

### Weaknesses
1.Experiments only consider moderate missing ratios (30–70%), not extreme cases like 90%. When co-occurrence among views becomes sparse, correlation estimation becomes unreliable.

2.Although the paper claims to “jointly learn correlation structures,” the learned parameters correspond to a global covariance, capturing only linear, global dependencies.

3.The text claims experiments are conducted on four datasets, while tables clearly list five. This inconsistency undermines experimental clarity and should be corrected.

### Questions
1.Can your method handle extreme missing-view ratios (e.g., 90%)? Could you provide performance curves versus missing ratios?

2.The learned correlation structure is always shared globally. Can it happen that view dependencies behave differently for different samples? Is a single correlation matrix no longer a good fit?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the limitations of existing imputation-free variational methods for incomplete multi-view clustering (IMVC), which typically rely on the assumption of conditional independence across views. 

The authors propose ACOVA (Adaptive Correlation-aware Variational Aggregation), a novel variational framework that explicitly models and learns inter-view correlations by leveraging the covariance of estimation errors between view-specific posteriors. The correlation matrix is parameterized through a normalized Cholesky decomposition, ensuring positive definiteness and enabling joint end-to-end optimization with model parameters.

Comprehensive experiments on several standard IMVC benchmarks (Scene15, Caltech5V, Handwritten, Fashion-MV, NoisyMNIST) show that ACOVA consistently outperforms previous state-of-the-art methods under various missing-view settings. Ablation and visualization analyses further confirm the benefit of modeling adaptive correlations for robust and discriminative latent representation learning.

### Strengths
- The paper identifies a fundamental issue in variational IMVC — the conditional independence assumption among view posteriors. Proposing to “break independence” by modeling cross-view correlation of estimation errors is both theoretically motivated and empirically useful.
- Built upon a linear–Gaussian variational framework, the proposed ACOVA explicitly models structured posterior covariance by decomposing the estimation error covariance as Σ=DRD.  The normalized Cholesky parameterization of R captures the correlations among view-specific variances and enables joint optimization, providing a principled generalization from DVIMC (independent) → CoDE (fixed scalar correlation) → ACOVA (adaptive correlation).
- Experiments on five IMVC benchmarks demonstrate consistent gains across missing-view rates (10–70%), confirming the method’s robustness. Qualitative visualizations — including t-SNE plots and learned correlation matrices — clearly illustrate that the model captures meaningful inter-view dependencies.
- Overall, the paper — together with its appendices — forms a coherent and complete study, covering theoretical justification, empirical validation, and in-depth analysis.

### Weaknesses
Motivation
1. The work is focused narrowly on incomplete multi-view clustering (IMVC), a relatively specific subproblem. It would strengthen the contribution to discuss whether the proposed adaptive correlation learning principle generalizes to broader multi-modal or self-supervised representation learning tasks.
2. The paper defines the “error of estimation” as the bridge to cross-view correlation modeling, but this definition appears somewhat _indirect_. The authors should provide a higher-level motivation early in the introduction — e.g., why modeling estimation errors is the right abstraction for inter-view dependence, rather than just citing Winkler (1981) and Mancisidor et al. (2025).

Method

3. The adaptive correlation learning is elegant, but the derivation (Eq. 9–11) and the optimization of R (Eq. 16) could use _clearer intuition_. It’s not entirely clear how learning R avoids degeneracy when views are highly incomplete or uncorrelated.
4. The paper provides only a Frobenius norm bound for R, which is purely structural and does not guarantee _stability_ or _identifiability_ of learned correlations, especially under high missing ratios.
5. In Eq. 6, since both the diagonal scaling $D$ and the correlation matrix $R$ are learned jointly, and $R$ itself is derived from a normalized Cholesky factor $L$, how do you ensure parameter identifiability? In other words, could different $L$ (or scaling of $D$ ) lead to equivalent $\Sigma=D R D$ and thus yield degenerate solutions?

Experiments

6. The authors mention averaging over five runs, but this should be stated clearly in the main text rather than only in the appendix. 
7. I appreciate that you provide a complexity analysis in Appendix A.4 and a time-cost comparison in Appendix C.3. However, the theoretical complexity $𝑂(𝑁𝐷𝑉^3)$ scales cubically with the number of views $𝑉$, primarily due to per-sample matrix inversion. Have you analyzed how the actual runtime or memory cost changes with increasing $𝑉$?
8. Although the proposed framework ensures $R \succ 0$ via the normalized Cholesky parameterization, matrix inversion of $R$ (and thus $\Sigma=D R D$ ) is required per sample and per latent dimension. This could introduce numerical instability when $R$ becomes ill-conditioned during training.
9. All datasets used are standard, small to medium-scale benchmarks (Scene15, Caltech5V, Handwritten, Fashion-MV, NoisyMNIST). There are no experiments on high-dimensional or large multi-view datasets to demonstrate scalability or generalization capability.

If the authors can adequately address the above concerns, I would be inclined to raise my score.

### Questions
Please refer to weaknesses section, thanks.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper targets the problem of Incomplete Multi-View Clustering (IMVC). It identifies a key limitation in recent variational IMVC approaches: their reliance on aggregators that assume conditional independence between views, which is a potentially over-restrictive assumption in real-world multi-view scenarios. To alleviate this limitation, the paper proposes ACOVA, a variational framework that relaxes this assumption by explicitly modeling cross-view correlations to achieve a more robust representation aggregation method. The core mechanism involves learning a correlation matrix, which is parameterized to ensure positive definiteness and trained end-to-end within a unified variational objective. Experimental results on several datasets demonstrate that ACOVA achieves superior performance, particularly surpassing methods that rely on an independence assumption.

### Strengths
The paper identifies a practical limitation in prior variational IMVC work: the conditional independence assumption inherent in the aggregation method, especially in the Product-of-Experts (PoE) approach, which might be over-restrictive in many multi-view scenarios, thereby limiting such IMVC method performance. The proposed solution of parameterizing a learnable correlation matrix via Cholesky decomposition (to ensure positive definiteness) is an elegant and technically sound approach to address this limitation in an unsupervised manner. Additionally, the experiments reveal the effectiveness of explicitly modeling and incorporating inter-view correlations.

### Weaknesses
1. Several key notations in the methodology (Sec 4.1) are ambiguous or insufficiently defined, which significantly hinders the comprehension of the proposed method. For example, the definition of $\mu$ as a $\mathbb{R}^{VD\times1}$ column vector is confusingly represented as $[\mu^1, \mu^2, ..., \mu^D]^{\mathrm{T}}$. Furthermore, the precise structure of the design matrix $\mathbb{1}$ in Eq.5 is not clearly specified; it appears to be a block matrix where the $d$-th $V$-row block contains ones in the $d$-th column and zeros elsewhere, but this requires explicit definition. Additionally, terms like $\mathbf{A}_\mathbf{M}$, $\mathbf{M}$ in Eq.9-11 are used in the main text without sufficient introduction.
2. While the current experiments demonstrate the method's effectiveness to some extent, the chosen multi-view datasets (e.g., Handwritten, Fashion) appear relatively simple, with potentially obvious inter-view correlations. To further validate the robustness and generalizability of the proposed correlation modeling, the method should be evaluated on more complex, real-world multi-view datasets where inter-view relationships may be more subtle or heterogeneous.

### Questions
1.Modeling view correlation seems fundamentally beneficial. Theoretically, should this approach also be expected to outperform independence-assuming methods in the complete multi-view setting?

2.The current datasets seem to have relatively simple view structures. How is the method expected to perform on more complex heterogeneous views (e.g., involving different modalities) where the inter-view correlations might be more intricate?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents an IMVC model by introducing a variational framework that explicitly models and learns cross-view correlations, thereby addressing a significant limitation of existing imputation-free methods based on variational inference.  Overall, with more extensive experimental validation, sensitivity analysis, discussion of limitations, and visualizations, this paper has the potential to make a substantial impact in the field.

### Strengths
1. The introduction of a learnable cross-view correlation structure is a significant contribution to the field of IMVC. 
2. The authors designed an adaptive cross-view correlation learning mechanism to jointly learn the correlation matrix and model parameters, leading to improved performance

### Weaknesses
1. It is suggested to perform more comparison experiments with state-of-the-art methods on a wider range of datasets and evaluation metrics. This would help to solidify the claims about the robustness and efficiency of the proposed approach.
2. The advancement of this paper and the derivation of the intermediate process are expected to receive more mathematical support, such as Eq. (1)-(3).

### Questions
1. What is the complexity of the proposed method? How does it compare with other methods, especially on the large-scale datasets?
2. How does it perform on the data with complex noise? How to ensure the accuracy of the learned cross-view correlation structure?
3. How many hyperparameters are there in the proposed model, and how can they be set in a new IMVC task?
4. How does the computational complexity scale with the number of views or the size of the dataset? Are there any scenarios where the proposed method might not perform well?

### Soundness
2

### Presentation
3

### Contribution
2
