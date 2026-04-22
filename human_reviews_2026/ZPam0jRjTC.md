# Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
We address the problem of causal effect estimation in the presence of hidden confounders using nonparametric instrumental variable (IV) regression. An established approach is to use estimators based on learned \emph{spectral features}, that is, features spanning the top singular subspaces of the operator linking treatments to instruments. While powerful, such features are agnostic to the outcome variable. Consequently, the method can fail when the true causal function is poorly represented by these dominant singular functions.

To mitigate, we introduce **Augmented Spectral Feature Learning**, a framework that makes the feature learning process **outcome-aware**. Our method learns features by minimizing a novel contrastive loss derived from an **augmented** operator that incorporates information from the outcome. By learning these task-specific features, our approach remains effective even under spectral misalignment. We provide a theoretical analysis of this framework and validate our approach on challenging benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles causal effect estimation under hidden confounding using nonparametric instrumental variable regression. Existing spectral feature learning methods rely on features from the top singular subspaces of the treatment–instrument operator, which are outcome-agnostic and may fail when the true causal function is misaligned with these subspaces. To overcome this, the authors propose Augmented Spectral Feature Learning, which introduces outcome-awareness by constructing an augmented operator that integrates information from the outcome variable. A novel contrastive loss enables learning task-specific spectral features robust to spectral misalignment. The paper provides theoretical guarantees and empirical results demonstrating improved accuracy and robustness on challenging synthetic and semi-synthetic benchmarks.

### Strengths
1. The paper identifies a fundamental limitation of spectral NPIV methods (outcome agnosticism leading to failure under spectral misalignment) and proposes a principled solution.
2. The proposed method is supported by rigorous theoretical analysis.
3. Experimental results demonstrate that the proposed approach.

### Weaknesses
1. The method enhances the operator using a rank-one augmentation, which simplifies theory but limits its ability to capture complex, multidimensional dependencies between $Y$ and the instrument–treatment relationship.

2. All experiments are conducted on synthetic or semi-synthetic datasets (e.g., dSprites), leaving the method’s practical effectiveness in noisy, unstructured real-world environments untested.

3. The paper evaluates the proposed method against only a few baselines (e.g., KIV, DFIV), missing more recent state-of-the-art approaches for nonparametric IV regression. For an ICLR submission, it would be beneficial to include comparisons with modern representation learning–based IV methods, to better contextualize the contribution and demonstrate broader relevance.

### Questions
1. The paper assumes sub-Gaussian distributions (Assumption 4), which might be restrictive in practice. Could the authors clarify how sensitive their theoretical results are to this assumption, and whether the framework could extend to heavier-tailed (e.g., sub-exponential) settings?
2. The paper is quite dense and may be difficult to follow for readers unfamiliar with spectral feature learning.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of nonparametric instrumental variable (NPIV) regression, a central challenge in causal inference when unobserved confounders bias treatment-outcome relationships. Existing spectral methods, such as SpecIV (Sun et al., 2025), learn low-rank representations of the conditional expectation operator T=E[h(X)∣Z], but are outcome-agnostic, leading to poor estimation when the structural function h_0 is misaligned with the top singular subspace of T. 

To overcome this limitation, the authors propose Augmented Spectral Feature Learning (ASFL), which introduces an outcome-aware contrastive loss that modifies the feature learning process. This is done via an augmented operator T_δ incorporating outcome information through a regularization parameter δ. The method learns features that are both predictable from instruments and predictive of outcomes.

### Strengths
A generalization bound for the two-stage least squares (2SLS) estimator using learned features.

An analysis showing robustness of ASFL under spectral misalignment.

Empirical validation on synthetic and dSprites-based IV benchmarks demonstrate the performance of the proposed method。

The theoretical novelty and conceptual framing are impactful, even if the experiments are limited.

### Weaknesses
The tuning parameter δ remains heuristic, with no principled selection strategy, which limits the method’s reproducibility and practical applicability.

While theoretically elegant, real-world problems often require capturing multiple outcome-relevant directions. Extending the framework to multi-dimensional or adaptive augmentations would enhance its generality.

The experimental evaluation is also limited, with comparisons restricted to DFIV and KIV and no validation on real-world datasets.

### Questions
Would higher-rank or learned outcome embedding improve robustness?

How does the computational cost of ASFL compare to SpelIV or DFIV?

How would the proposed method perform in real econometric or policy evaluation datasets (e.g., education or healthcare)?  

The theoretical contribution is elegant and addresses an important limitation of SpecIV. However, the empirical validation is limited to synthetic setups and toy datasets. Without evidence of real-world performance or a principled tuning strategy for δ, the paper’s practical impact remains unclear.

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
2

### Summary
This paper proposes a novel framework for outcome aware spectral feature learning in nonparametric instrumental variable regression. It solves a fundamental limitation in existing methods by making the feature learning process sensitive to the outcome variable through augmenting the spectral decomposition with outcome information. This approach maintains robust causal effect estimation even in challenging regimes of spectral misalignment where standard methods fail.

### Strengths
The paper demonstrates high originality by identifying and tackling a fundamental, previously overlooked flaw in a state-of-the-art method. It formally characterizes the problem of "spectral misalignment," where outcome-agnostic feature learning fails, and introduces a principled solution through the novel concept of an augmented operator and a corresponding contrastive loss.

The work is supported by exceptional methodological rigor, combining a compelling theoretical analysis with comprehensive empirical validation. It provides non-asymptotic generalization bounds for the proposed estimator and thoroughly benchmarks the method against strong baselines across synthetic and semi-synthetic datasets, convincingly demonstrating its superiority in challenging regimes.

### Weaknesses
While the empirical results are compelling within the constructed synthetic and semi-synthetic frameworks, the practical significance of these benchmarks for real-world causal inference problems remains less clear.  The experiments, including the new dSprites benchmark, operate in controlled environments where the core assumptions of the model, such as the validity of the instrumental variable and the specific form of the structural causal model, are guaranteed by design.  In practice, these assumptions are untestable and often subject to intense debate.  A more convincing demonstration of impact would require application to a real-world dataset with a well-documented and long-standing causal puzzle, showing that the method can generate a plausible and interpretable estimate where traditional approaches have struggled. Furthermore, the paper's reproducibility is currently hampered by the absence of publicly available code.  

I also note that my own expertise in the theoretical foundations of spectral methods for operator learning is limited, and I am therefore unable to provide a substantive assessment of the technical soundness or novelty of the theoretical contributions in Sections 3 and 4.

### Questions
Please see my concerns in weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new method to enhance SSL on graph-structured data by explicitly incorporating outcome information into the feature alignment process. Traditional spectral SSL approaches often rely solely on the graph structure, which may overlook how node features relate to prediction outcomes. This work introduces an outcome-aware spectral feature alignment framework that adjusts the feature space according to the outcome distribution, leading to more discriminative embeddings and improved classification accuracy. The authors derive a principled optimization objective grounded in spectral theory and demonstrate that their approach can be efficiently implemented through eigendecomposition of an adjusted Laplacian matrix. Experimental results across several benchmark datasets show consistent performance gains over classical spectral and graph neural network baselines, highlighting the method’s robustness and scalability.

### Strengths
- The paper introduces a novel “outcome-aware” perspective in spectral feature alignment, an area typically dominated by purely structure-based graph methods. It bridges the gap between spectral methods and outcome-driven modeling, creating a new formulation that incorporates outcome distributions directly into spectral regularization.
- The paper presents a solid theoretical foundation, with clear derivations connecting the proposed objective to classical spectral theory and graph Laplacian properties.
- The proposed framework is generalizable: it can be adapted to various graph learning settings, including GNN pretraining and kernel-based SSL, broadening its applicability.

### Weaknesses
- The paper offers a rigorous spectral derivation, but lacks a clear theoretical link between outcome-aware alignment and generalization performance. The intuition that aligning features with outcomes leads to better predictive embeddings is compelling but not mathematically formalized.
- The experimental evaluation mainly benchmarks against classical spectral and GNN methods (e.g., GCN, LapRLS), but omits comparison with more recent outcome- or label-sensitive graph models.
- Although the results show performance gains, the paper provides little qualitative insight into how outcome-awareness modifies the embedding geometry.

### Questions
- Does the proposed OSFA method preserve key spectral properties (e.g., orthogonality of eigenvectors, positive semi-definiteness of the Laplacian) after outcome integration?
- Since OSFA explicitly depends on outcome information, how does it perform when labels are noisy or partially incorrect?

### Soundness
3

### Presentation
3

### Contribution
4
