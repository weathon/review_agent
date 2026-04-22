# DyGB: Dynamic Gradient Boosting Decision Trees with In-Place Updates for Efficient Data Addition and Deletion

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Gradient Boosting Decision Tree (GBDT) is one of the most popular machine learning algorithm in various applications. However, in the traditional settings, all data should be simultaneously accessed in the training procedure: it does not allow to add or delete any data instances after training. In this paper, we propose DyGB (Dynamic GBDT), a novel framework that enables efficient support for both incremental and decremental learning within GBDT. To reduce the learning cost, we present a collection of optimizations for DyGB, so that it can add or delete a small fraction of data on the fly. We theoretically show the relationship between the hyper-parameters of the proposed optimizations, which enables trading off accuracy and cost on incremental and decremental learning. Empirical results on backdoor and membership inference attacks demonstrate that DyGB can effectively add and remove data from a well-trained model through incremental and decremental learning. Furthermore, experiments on public datasets validate the effectiveness and efficiency of the proposed DyGB framework and optimizations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DyGB (Dynamic GBDT), a framework for efficient, in-place incremental (data addition) and decremental (data deletion) learning in GBDTs. Instead of retraining, DyGB traverses existing trees and checks if node splits are still optimal using statistics from the data delta ($D'$). If a split is suboptimal, it retrains only the subtree below that node. The framework's speed relies on several optimizations: (1) using stored statistics to avoid accessing the original dataset, (2) lazy derivative updates, (3) split candidate sampling ($\alpha$), and (4) a robustness tolerance ($\sigma$) to ignore minor, low-impact split changes. Extensive experiments show this approach is significantly faster than retraining while maintaining comparable accuracy, which is validated using backdoor attack and removal simulations.

### Strengths
S1. The paper addresses a critical and highly practical limitation of GBDTs. The static, batch-only nature of GBDT training is a major bottleneck in real-world systems that require models to adapt to new data or forget old data (e.g., for privacy compliance or addressing data drift).

S2. The evaluation of the incremental and decremental quality is comprehensive. This includes functional validation via attack simulations (backdoor and membership inference), direct accuracy comparisons against retrained models, and functional similarity comparisons (Appendix K). Figure 3 is particularly impressive, showing that the incremental and decremental learning framework tracks the accuracy of a fully retrained model even when adding or removing large ratios of the dataset.

S3. The proposed framework is unified and efficient. It is the first to support both incremental and decremental learning within a single, in-place mechanism for GBDTs.

### Weaknesses
W1. The paper's method for feature discretization (Algorithm 4), a critical component for any GBDT implementation, is relegated to the appendix (Appendix C). This method is presented without sufficient experimental validation or theoretical proof of its superiority over other common techniques. A robust dynamic GBDT framework heavily depends on the stability of its feature histograms. The paper fails to compare its binning strategy against alternatives, such as naive sample-based discretization or the "balanced robust histogram" method used in related work like DeltaBoost. Such a comparison would be necessary to demonstrate the proposed method's balance and robustness to histogram shifts caused by data addition and deletion.

W2. The paper presents the observation in Section 3.4 that "For adding or deleting a single data point, the best split does not change in most cases" as a novel finding motivating its optimizations. However, this is a known phenomenon that was theoretically proven in prior work (e.g., Theorem 3.1 in the DeltaBoost paper). The paper should explicitly acknowledge this and clarify what, if any, new insights its empirical findings add beyond what is already established.

W3. The impact of the "Adaptive Lazy Update for Derivatives" (Sec 3.2) is not fully clear in the main text. The paper states that derivatives are updated "only when retraining occurs," which implies that the decision to check a split (gain computation) might be made using stale derivatives. This is a key approximation. While Appendix Q touches on the resulting error, the main paper would be stronger if it discussed the impact of this specific approximation on the decision-making of the algorithm (i.e., does using stale derivatives cause the model to miss a necessary retrain, or retrain an unnecessary one?).

**Minor Comments**

C1. The text and values within Figure 1 are very small and low-resolution, which makes it difficult to follow the concrete example presented in Section 2.3.

C2. The font size used in many of the tables (e.g., Table 2, 15, 16) is very small, making them difficult to read.

### Questions
Q1. Does the incremental learning or decremental learning require access to the original full training set? This should be clarified in the problem statement.

Q2. The experimental comparisons raise several questions about fairness and reliability.
* First, why does DyGB consistently outperform highly optimized libraries like XGBoost in both full training time (Table 3) and accuracy (Figure 3)? Since DyGB is a prototype built on Robust LogitBoost, what is the nature of this significant improvement? Is it due to the algorithm itself, the C++ implementation, or the specific hyper-parameter choices used for all models?
* Second, there appear to be discrepancies with prior work. For instance, the decremental learning time reported for DeltaBoost on the Covtype dataset seems significantly slower (e.g., ~20x) than the times reported in the original DeltaBoost paper. Can the authors explain this difference in experimental outcomes?
* Finally, the efficiency evaluations (e.g., Table 2) do not report variance or standard deviations. Given the potential for high variance in dynamic updates, this makes it difficult to assess the reliability and consistency of the speedup claims. Were multiple runs conducted, and if so, what was the variance?

Q3. What is the ratio of the changed nodes in each dataset? The appendix (e.g., Figure 11) presents the absolute number of retrained nodes, but not the ratio relative to the total number of non-terminal nodes in the ensemble. This ratio is the most important factor in evaluating whether the claimed efficiency improvement is reasonable. Can the authors provide these ratios for the experiments, as this metric directly connects the amount of avoided work (nodes not retrained) to the efficiency gains of incremental/decremental learning when compared to a full retraining?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DyGB, a new framework for dynamic gradient boosting decision trees (GBDT) that supports both incremental (adding new data) and decremental (removing data) learning in-place, without retraining the entire model. The authors present theoretical analyses on trade-offs between accuracy and computational cost, introduce several optimizations (e.g., adaptive lazy updates, split candidate sampling, robustness tolerance), and conduct extensive experiments comparing DyGB against strong baselines such as XGBoost, LightGBM, CatBoost, ThunderGBM, DeltaBoost, and MUinGBDT. Results show substantial speedups (up to 1200× in some settings) while maintaining accuracy close to retrained models.

### Strengths
1.	Timely and important topic. Dynamic or “unlearning” capabilities are becoming crucial for privacy, continual learning, and compliance (e.g., GDPR). Extending GBDT to support both incremental and decremental updates efficiently is a valuable direction.
2.	Comprehensive experiments. The paper evaluates DyGB across 10 datasets, with comparisons to several strong baselines and both incremental and decremental scenarios. The backdoor and membership inference attack studies are a creative way to validate unlearning ability.
3.	Algorithmic innovations. The paper introduces meaningful technical contributions, such as adaptive lazy updates and split robustness tolerance, that help reduce computational costs without major accuracy loss.
4.	Clarity of motivation. The introduction effectively argues why dynamic GBDT is relevant and distinct from existing online boosting or incremental tree methods.

### Weaknesses
1.	Limited novelty in core mechanism. While DyGB integrates several optimization strategies, the overall structure (updating splits, retraining subtrees, and incremental statistics updates) follows existing frameworks like MUinGBDT (Lin et al., 2023). The paper’s novelty seems incremental rather than groundbreaking.
2.	Theoretical analysis is shallow. The “robustness” definitions and proofs are mostly intuitive and not deeply rigorous. Theoretical guarantees (e.g., on convergence or bounds on model drift) are missing.
3.	Experimental reporting could be clearer. The results focus heavily on runtime speedups but offer limited insight into accuracy degradation, especially under frequent dynamic updates. A figure or table quantifying trade-offs between time and accuracy loss would strengthen the claims.
4.	Presentation issues. The paper is dense and sometimes difficult to follow, particularly in the algorithmic sections (Algorithms 2–3). Some pseudo-code lacks clarity in notation (e.g., subscripts and primes are inconsistently defined).
5.	Missing ablation details. Although Appendix S is mentioned for ablation studies, the main paper does not summarize which components (e.g., adaptive updates, split sampling) contribute most to efficiency.

### Questions
1.	How does DyGB handle concept drift or non-stationary data distributions over time?
2.	Can DyGB support weighted unlearning or partial forgetting?
3.	What is the impact of hyperparameters α and σ on model stability—are there guidelines for tuning them automatically?

### Soundness
3

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
The paper proposes DyGB, a framework that performs in-place updates to trained GBDT models to support both incremental (adding data) and decremental (removing data) learning while keeping the model size (number of trees/parameters) unchanged. The key idea is to avoid touching the original training set by caching per-split statistics from training and then recomputing gains using only the delta dataset D’; affected nodes are selectively retrained and only those samples’ derivatives/residuals are updated for subsequent trees.

### Strengths
- Clear problem and practical impact. The paper targets in-place updates that support both data addition and deletion—addressing real needs such as privacy-compliant unlearning and continual learning—while keeping the ensemble size fixed.

- Simple, effective mechanism. By caching per-split gradient/Hessian statistics and applying localized residual/derivative updates, the method enables efficient in-place updates without rescanning the full training set.

- Strong empirical evidence. Extensive experiments across multiple public datasets consistently demonstrate efficiency of the proposed approach.

- Reproducibility. The authors provide an anonymized implementation and scripts, facilitating independent verification and reuse.

### Weaknesses
- Introduction needs tightening. The current introduction spends too many paragraphs on broad background and related work (which are covered again later).

- Clarify the “histogram” connection. Caching per-split gradient/Hessian statistics is effectively histogram binning à la LightGBM. I think the author should foreground this explicitly (use “histogram-based” terminology).

- Approximation from lazy derivatives. Using outdated derivatives until a subtree retrains introduces approximation error that can grow with larger D’.

- Accuracy parity missing. The experiments emphasize speed and memory but omit a head-to-head accuracy comparison against from-scratch retraining in XGBoost/LightGBM. 

- Clarify “robust” (Defs. 1 & 2). Why are Definitions 1 and 2 termed robust?

### Questions
- Full (global) derivative updates are costly; relying on lazy local derivatives incurs approximation error. Is there a systematic strategy to balance accuracy and cost?

- Is there a parallel version (multi-node) for dyGB?

### Soundness
3

### Presentation
2

### Contribution
3
