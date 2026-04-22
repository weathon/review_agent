# Quantile-Free Uncertainty Quantification in Graph Neural Networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Uncertainty quantification (UQ) in graph neural networks (GNNs) is crucial in high-stakes domains but remains a significant challenge. In graph settings, message passing often relies on strong assumptions such as exchangeability, which are rarely satisfied in practice. Moreover, achieving reliable UQ typically requires costly resampling or post-hoc calibration. To address these issues, we introduce Quantile-free Prediction Interval GNN (QpiGNN), a framework that builds on quantile regression (QR) to enable GNN-based UQ by directly optimizing coverage and interval width without requiring quantile inputs or post-processing. QpiGNN employs a dual-head architecture that decouples prediction and uncertainty, and is trained with label-only supervision through a quantile-free joint loss. This design allows efficient training and yields robust prediction intervals, with theoretical guarantees of asymptotic coverage and near-optimal width under mild assumptions. Experiments on 19 synthetic and real-world benchmarks show QpiGNN achieves average 22% higher coverage and 50% narrower intervals than baselines, while ensuring efficiency and robustness to noise and structural shifts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces QpiGNN, a novel framework for uncertainty quantification in GNNs. Unlike Bayesian or post-hoc conformal methods that rely on exchangeability or resampling, QpiGNN uses a dual-head GNN to decouple prediction and uncertainty estimation. They also provide theoretical coverage guarantees (asymptotic and finite-sample) under mild dependence assumptions.

### Strengths
### Clear motivation
* UQ in GNNs is still underexplored and practically critical. The paper correctly identifies the limitations of current quantile-based and conformal approaches in graph settings

### Novelty
* Quantile-free uncertainty learning is a clean formulation. The dual-head architecture effectively decouples prediction and uncertainty, addressing oversmoothing problems in GNNs

### Theoretical analysis
* Asymptotic coverage and finite-sample concentration results are clearly stated, with mild assumptions that are reasonable in practice

### Experiments
* This paper provides extensive experiments to support their claims (clear comparison against strong baselines)

### Weaknesses
### Graph dependence
* The coverage guarantees rely on weak dependence assumptions, but the treatment remains at a high level. There’s no rigorous dependence model or formal graph-specific concentration analysis beyond citing WLLN

### Comparison with conformal prediction theory
* While CP methods are used as baselines, the paper does not deeply analyze why QpiGNN succeeds under violations of exchangeability. A theoretical comparison would strengthen the claims

### Positioning
* While the quantile-free formulation is original in GNNs, it builds closely on RQR. The conceptual gap between RQR and QpiGNN could be articulated more strongly

### Questions
Please see the above weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an architecture to address uncertainty quantification for node regression problems: Instead of using a convolutional layer to output the interval mean and width, it instead uses MLP-based heads to mitigate overly smooth intervals. It provides theoretical arguments for the validity of quantile regression in GNNs and shows the performance over an extensive benchmark of synthetic and real-world datasets.

### Strengths
- The paper is well-written and easy to follow.
- It covers the relevant related work and does a good job at showcasing the differences to OpiGNN.
- It covers a range of synthetic and real datasets.
- It provides insightful ablations regarding various architectural choices.

### Weaknesses
### Major
- The main weakness is the originality / novelty of the approach: It differs from related work like RQR in two ways: i) It uses MLP-based heads for predicting the mean and interval width instead of convolution-based, and ii) uses a different loss for training. The contribution of i) is very incremental to me (the authors even acknowledge that dual-head architectures have been used in Bayesian regression). For ii), the work does not conclusively show the merits of changing the objective from Eq (3) or (4) to Eq (5). In particular, I did not find ablations that showcase the performance of OpiGNN trained with (4) (or RQR trained with (5)) to justify the new objective.
- The authors also claim that their novel loss Eq (5) disentangles coverage and interval width, but the same statement can be made about the existing loss of Eq (3) (the latter term, weighted by $\lambda / 2$ penalizes interval width, while the first penalizes miscoverage). It is unclear to me from both a theoretical and empirical (see point above) perspective where the merits of the new loss in Eq (5) lie.
- The authors report a comparison to the baselines only at a coverage of $\alpha=0.9$. On real data, both SQR and RQR only slightly miss the coverage requirement, but at significantly smaller intervals. At the same time, OpiGNN achieves coverage at a probability that is significantly above the target, potentially at the cost of too conservative intervals. 
   - While I acknowledge that having a lower coverage than required is the worse failure mode, neither of the models seems to be well calibrated in terms of realizing the minimal interval width at the desired coverage requirement. 
   - It would be interesting to see if the baselines can realize different trade-offs at different penalties for the interval width (e.g., a slightly lower penalty may just push them into the regime of meeting the 90% coverage consistently but still providing narrower intervals).
   - A more useful depiction of these results would be a Pareto plot similar to Figure 6. There, it would be more apparent that neither the baselines nor OpiGNN realizes the optimal intervals for $\alpha=0.9$.
- The theoretical arguments of the paper are not very involved and show rather obvious claims. E.g., Prop. 1 basically just re-states the weak law of large numbers. To do so, it makes strong assumptions that are not obviously justified to me (e.g., that means and intervals converge to the targets in probability).
   - Importantly, neither of the claims is specific to OpiGNN and can be transferred to the baselines SQR and RQR as well. The theoretical findings, therefore, do not support the architectural contribution of the paper.

### Minor:
- Some minor claims are not fully supported by experiments:
   - L. 141: SQR is supposedly unstable and miscalibrated: The results show that, with the exception of three (heterophilic?) datasets, it just slightly underachieves the desired coverage, but provides smaller intervals instead.
   - L. 161: Why should (4) be a penalty that explicitly targets miscalibration in GNNs? This seems to be applicable to non-GNN architectures. In particular, I do not see how this formulation directly targets the graph domain.
- OpiGNN seems to be insensitive to the desired coverage $\alpha$ (see Figure 3, and the close to 100% coverage in Table 1). How do baselines and OpiGNN perform at different coverage levels (a similar ablation to Figure 3 that includes baselines at suitable penalty strenghts would be insightful here).
- Remark 1: The authors argue that each neighbour contributes at most O(1 / deg(v)). However, this does not imply that each neighbour contributes $\mathcal{O}(1 / N)$, as $1 / deg(v)  > 1 / N$, and therefore, if something is smaller than the inverse of the degree, it does not imply that it is smaller than $1/N$. Furthermore, I do not see why the influence over different layers should behave additively (even if the provided $\mathcal{O}(1 / N)$ bound was correct).
- All ablations are conducted on synthetic data. I believe that ablations on real data is more insightful to highlight the advantages and limitations of the approach in practice.

### Questions
- In L. 273, you claim that L1 regularization is advantageous over L2 regularization because of its robustness against outliers. Do you have any experimental evidence that this is relevant for the datasets studied here?
- In Figure 6, where do RQR and SQR lie here? Where does OpiGNN trained with (3) lie here? These kinds of plots would be interesting to have as a supplement for Table 1.
- In Appendix B.5, how does the robustness toward single-node perturbations (which also holds for any of the other GNN-baselines) justify the concentration arguments as claimed? Can you elaborate on this?
- Figure 4 shows that there is no smooth trade-off between interval width and coverage, at least in terms of the interval width penalty. Instead, there seems to be two regimes, one with a fixed coverage close to 100%, and a second regime with rapidly collapsing coverage. Why is that? Is there a way to make OpiGNN realize a broader range of desired coverage / interval-width trade-offs?

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
This paper proposes QpiGNN, a novel framework for Uncertainty Quantification in Graph Neural Networks for node-level regression. It addresses the instability of traditional Quantile Regression in GNNs by introducing a quantile-free joint loss that directly optimizes for both coverage and width through a dual-head architecture that predicts the center and half-width. The method provides theoretical asymptotic coverage guarantees and empirically demonstrates superior performance in achieving calibrated and compact prediction intervals across numerous benchmarks.

### Strengths
The dual-head, quantile-free approach is an elegant solution that inherently satisfies the non-crossing constraint of prediction intervals, a major practical and theoretical challenge in standard QR.

Interpreting the joint loss as a Lagrangian relaxation provides a strong, principled foundation for the training objective, formally justifying the trade-off between coverage and interval width.

### Weaknesses
The theoretical coverage guarantee relies on the assumption that the local dependencies introduced by GNN message passing decay rapidly enough to allow for the use of concentration inequalities. Given the complex, often high-homophily nature of real-world graphs, this assumption of weak dependence on graph structure is often violated, potentially making the finite-sample guarantees less robust than claimed.

The key architectural decision is the dual-head decoupling of $\hat{y}$ and $\hat{d}$. While intuitively helpful for mitigating oversmoothing, the paper lacks a clear ablation study to formally prove its necessity. Specifically, an experiment comparing the proposed dual-head architecture against a single-head architecture that outputs both $\hat{y}$ and $\hat{d}$ should be included to validate this crucial design choice.

### Questions
See weaknesses

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
3

### Summary
The paper proposes Quantile-free Prediction Interval Graph Neural Networks (QpiGNNs), a framework for node-level uncertainty quantification in GNNs from a quantile prediction view. QpiGNNS has a dual-head architecture, one for prediction and one for interval width, which is trained with a quantile-free joint loss that directly optimizes both empirical coverage and interval sharpness using only label supervision. Theoretical analysis shows that the proposed loss is non-convex but guarantees asymptotic coverage under weak dependency assumptions across graph nodes. Empirical results demonstrate that QpiGNN achieves higher coverage accuracy and narrower prediction intervals than existing quantile or conformal GNN baselines across various datasets.

### Strengths
1. The paper is well presented and easy to follow, with clear problem formulation and motivation for quantile-free interval learning.
2. The authors conduct comprehensive analysis of the proposed model, including convergence under non-convex optimization, hyperparameter sensitivity, and generalization under structural graph shifts.
3. The method provides a theoretically grounded approach that avoids quantile supervision or resampling, while showing strong empirical robustness under noise and distributional changes.

### Weaknesses
1. While the task is interesting from the perspective of quantile prediction, the paper positions itself as an uncertainty quantification work. Therefore, comparisons with other established uncertainty quantification methods for regression—such as evidential regression or Bayesian neural networks—would provide a stronger baseline context.
2. The motivation for designing this framework specifically for graphs is underdeveloped. There is no graph-specific adaptation in the architecture or loss beyond using a GNN backbone, and the theoretical analysis assumes weak dependence among neighboring nodes without addressing how real-world graph correlations may affect coverage guarantees.
3. The paper’s practical application and generalization capability remain somewhat unclear. While results show improved coverage, it is uncertain how well the approach scales to large or dynamically evolving graphs, or how it performs in non-regression graph tasks such as classification or link prediction.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
