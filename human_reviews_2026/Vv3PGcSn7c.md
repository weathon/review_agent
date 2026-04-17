# Fair Classification by Direct Intervention on Operating Characteristics

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
We develop new classifiers under group fairness in the attribute-aware setting for binary classification with multiple group fairness constraints (e.g., demographic parity (DP), equalized odds (EO), and predictive parity (PP)). 
We propose a novel approach based on directly intervening on the operating characteristics of a pre-trained base classifier, by: 
(i) identifying optimal operating characteristics using the base classifier's group-wise ROC convex hulls; 
(ii) post-processing the base classifier to match those targets.
As practical post-processors,
we consider randomizing a mixture of group-wise thresholding rules subject to minimizing the expected number of interventions. 
We further extend our approach to handle multiple protected attributes and multiple linear fractional constraints.
On standard datasets (COMPAS and ACSIncome), 
our method simultaneously 
satisfies approximate DP, EO, and PP with few interventions and a nearly optimal drop in accuracy; and compare favorably to previous methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ROCF, a post-processing framework that works directly in rate space, leveraging each group’s ROC convex hull to find realizable regions that satisfy multiple fairness constraints (including linear-fractional ones such as PPV parity). The authors propose (i) a centroid-based linearization of linear-fractional constraints that yields a linear program for any fixed centroid, and (ii) a feasibility guard that minimally relaxes constraints when the desired fairness region is infeasible. Empirical evaluation on COMPAS and ACSIncome shows that ROCF achieves approximate DP, EO, PP, and FOR with minimal intervention rate and only small accuracy drops. The method unifies a variety of fairness notions within one efficient LP-based framework.

### Strengths
Originality: Nice and elegant reframing of fairness post-processing as optimization directly over operating characteristics; the centroid-based linearization for fractional constraints is novel and mathematically neat.

Quality: Sound algorithmic development and proofs; empirical evidence across fairness notions and with multiple random seeds; good engineering of a “feasibility guard” to guarantee graceful degradation.

Clarity: Clearly written and well-modularized. Rate-space figures and Algorithm listings support understanding.

Significance: Offers a practical, easily implementable method to jointly achieve fairness notions that were previously handled separately. Useful for any group dealing with fairness auditing or deployment.

### Weaknesses
Limited scope and datasets: Evaluation restricted to two small tabular datasets (COMPAS, ACSIncome). 

Binary-only setting: The framework and proofs assume binary classification with continuous scores; no discussion of multi-class, where scores live in the probabilistic simplex. The assumption that scores are continuous (“ties broken deterministically”) could fail in applications.

Baselines: Comparisons are mostly to post-processing methods (Hardt et al. 2016, Hsu et al. 2022); in-processing reduction approaches (Agarwal et al. 2018) could be included for a broader view.

Feasibility guard uniformity: The relaxation of fairness constraints is uniform; no discussion of prioritizing one constraint (e.g., EO) over another (e.g., PP). This limits interpretability of fairness–utility trade-offs.

Intervention optimality: The AntiDiagonal and LabelFlipping constructions are heuristically minimal but not formally proven globally optimal; empirical stability to hull noise could be further demonstrated.

### Questions
Centroid grid sensitivity: How fine must the centroid grid be to approximate the true feasible region? Any quantitative link between grid resolution and sub-optimality?

Denominator constraints: How are $\epsilon_k$ parameters for denominator stability selected, and do they materially alter fairness feasibility for small groups?

Constraint prioritization: Could the feasibility guard relax only selected constraints (e.g., PP but not EO)? A weighted relaxation could make the framework more flexible.

Calibration drift: How robust is ROCF if the base model’s score calibration shifts between training and deployment?

Multi-attribute scalability: How does ROCF scale when there are many protected attributes or intersecting subgroups?

Dataset expansion: Adding a credit-risk or medical dataset would make results more convincing and highlight ROCF’s linear-fractional handling.

AntiDiagonal vs. LabelFlipping: Under what conditions does one yield strictly fewer interventions? An ablation or theorem would clarify.

Oracle definition: Clarify what makes the Oracle baseline “infeasible in practice” (e.g., requiring group labels at test time).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes post-processing algorithms for achieving (multiple) group fairness (criteria simultaneously), covering criteria that can be expressed in linear or linear-fractional forms: this covers SP, EO, TPR/FPR parity, and predictive rate parity.  The focus is on binary classification and the attribute-aware setting (sensitive attribute can be used explicitly as input to the classifier/post-processor).

Given the ROC (i.e., feasible TPR-FPR tradeoffs) of the Bayes optimal score function:

1. With a reformulation trick, the authors show that, if the "centroid" of the group statistics of the fair classifier is known, then the fairness criteria can be expressed as linear constraints in terms of the group-conditional TPR and FPR.
2. If the performance of the classifier is a function of the TPR and FPR (e.g., accuracy), then the optimal fair classifier is fully characterized by its TPR-FPR tradeoff.

With these two facts,

1. The authors proposed a bi-level optimization procedure where the outer layer sweeps over the possible centroids, and the inner layer solves the fair classification with the linear constraints as a linear program. Finally, it returns the one with the centroids that leads to the best performance.
2. To return a classifier achieving a specific TPR-FPR tradeoff, two methods are proposed: one is based on randomly interpolating a thresholded classifier (which lies on the boundary of the ROC) and a input-independent randomized classifier (which lies on the diagonal), and another which is based on combining thresholding and label flipping.

### Strengths
- To the reviewer's knowledge, this is the first paper proposing a post-processing based method for achieving fair classification on linear (SP, EO, TPR/FPR) and linear-fractional (predictive rate equality) fairness constraints simultaneously.
- Experiment results show good empirical performance.
- The reviewer finds the reformulation trick for "linearizing" the linear-fractional constraints interesting, and could be also be applied to the more general attribute-blind and multi-class setting (for representing the optimal fair classifier).

### Weaknesses
- The fact that a bi-level optimization procedure involving a grid search for finding the optimal "centroid" is required is a bit dissatisfying, and suggests that the proposed algorithm may be hard to scale to more general settings.
- The lack of a sample complexity analysis (probably about the same as fitting a linear model with $O(K)$ parameters), and an error propagation analysis: if the upstream score function is non-Bayes-optimal, how much would it hurt performance and fairness?

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a method for jointly optimizing multiple fairness constraints under approximation thresholds. The method leverages the ROC to satisfy linear fractional fairness constraints, and provides extensive analytical results on the framework.

The authors empirically evaluate on standard fairness datasets, and demonstrate near-oracle performance over multiple fairness measures, with a principled and flexible delta-approximate fairness.

### Strengths
1. The model is well-motivated and the authors situate their work well in the introduction. The application of the ROC is intuitive to a novice reader at a conceptual level.

2. The method evaluates quite well. The method consistently achieves near-oracle performance over multiple fairness measures. The authors evaluate over standard fairness-related datasets which is fine

3. The method allows an intuitive approximate fairness threshold delta, which allows adaptation to the application. Furthermore, the adaptive feasibility search allows this parameter to be over-constrained while still providing a solution in a principled manner .

### Weaknesses
1. Overall, the paper is challenging for a general AI researcher reader, and likely challenging for a fair AI researcher. I don't have a great recommendation for improving this, as the analytical results are necessarily technical. 

2. I have low-confidence in verifying the proofs in the paper, particularly Thm 4.1 in A.4.1. Therefore I defer to other reviewers on correctness and evaluate as if true. 
- I believe the proof requires a Q_k construction, which is deferred to A.5.1. This should be moved up in the proof context

3. The method is sensitive to the cardinality of the region search grid. Finding the appropriate scale on this approximation seems to be challenging. The authors address this in terms of runtime, but does an overly-granular grid produces issues of over-sparsity?

### Questions
1. Does this method suffer from sparse values in the grid search? i.e. is the only trade-off runtime, or does the result fail on overly-granular grids?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ROCF (Fair Classification via Operating Characteristic Feasibility Regions), a post-processing framework for achieving approximate group fairness under multiple constraints such as demographic parity (DP), equalized odds (EO), and predictive parity (PP).
Instead of retraining models, the method intervenes directly on the operating characteristics (TPR/FPR) of a pre-trained classifier. It identifies feasible fairness-adjusted points using each group’s convex hull and post-processes predictions to reach those points. The approach extends to multiple protected attributes and supports linear-fractional fairness metrics.
Experiments on COMPAS and ACSIncome show that ROCF achieves approximate fairness across several criteria with minimal accuracy loss and few label interventions, outperforming prior post-processing baselines (Celis et al., 2019; Hsu et al., 2022).

### Strengths
Novel geometric perspective: Optimizing fairness directly in ROC space via convex hulls is elegant and intuitive.
Handles multiple fairness criteria: Supports several linear and fractional constraints simultaneously, a limitation of most existing post-processing methods.
Practical deployment path: Works on top of pre-trained models, making it applicable when retraining is infeasible.

### Weaknesses
Limited dataset diversity: Evaluation on only two benchmark datasets limits generalization; larger and more complex domains should be tested.
Scalability concerns: The grid search may be expensive when handling many groups or fairness constraints.
Sensitivity to data size: Since ROCF relies on convex hull estimation, small or noisy samples may affect feasibility, but this is not quantified.
Post-processing limitations: As with all post-processing methods, bias in the learned representation remains unaddressed.

### Questions
Complexity and scalability: How does the computational cost of the grid search scale with the number of fairness constraints and protected groups?
Generalization beyond binary tasks: Could the ROC-based formulation extend to multi-class or regression settings, or is it inherently tied to binary classification?

### Soundness
3

### Presentation
3

### Contribution
3
