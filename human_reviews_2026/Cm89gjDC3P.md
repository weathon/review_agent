# FracAug: Fractional Augmentation boost Graph-level Anomaly Detection under Limited Supervision

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Graph-level anomaly detection (GAD) is critical in diverse domains such as drug discovery, yet high labeling costs and dataset imbalance hamper the performance of Graph Neural Networks (GNNs). To address these issues, we propose FracAug, an innovative plug-in augmentation framework that enhances GNNs by generating semantically consistent graph variants and pseudo-labeling with mutual verification. Unlike previous heuristic methods, FracAug learns semantics within given graphs and synthesizes fractional variants, guided by a novel weighted distance-aware margin loss. This captures multi-scale topology to generate diverse, semantic-preserving graphs unaffected by data imbalance. Then, FracAug utilizes predictions from both original and augmented graphs to pseudo-label unlabeled data, iteratively expanding the training set. As a model-agnostic module compatible with various GNNs, FracAug demonstrates remarkable universality and efficacy: experiments across 14 GNNs on 12 real-world datasets show consistent gains, boosting average AUROC, AUPRC, and F1-score by up to 5.72\%, 7.23\%, and 4.18\%, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FracAug, a plug-in augmentation framework for graph-level anomaly detection (GAD) under limited supervision. FracAug has three parts: (i) a Fractional Graph Generator (FGG) that forms augmented graphs via fractional powers of a transformed adjacency (spectral) operator; (ii) a Weighted Distance-aware Margin Loss (WDML) to handle class imbalance with reweight and keep synthetic samples near their originals; and (iii) a Mutual Verification Pseudo-labeler (MVP) that accepts pseudo-labels only when predictions on original and augmented graphs agree. Experiments cover 12 datasets and multiple GNN backbones.

### Strengths
- The mutual-verification pseudo-labeler is a sensible way to reduce pseudo-label noise in low-label regimes; the formalization of agreement criteria is clear. 

- This paper uses simple thorem to help readers better understand the motivation of the method, which is good.

### Weaknesses
- Scalability and resources. The graph generation relies on eigenvalue decomposition of the adjacency matrix, which is computationally expensive. Please report how time and memory grow with graph size, and include wall-clock time and peak memory for representative datasets.

- Spectral choices. During graph generation, this work retains a small set of the smallest eigenvalues and drop the middle of the spectrum. Why are the smallest ones kept, given that they often align with noise, and why is the middle removed? Please provide either a clear rationale or an ablation that compares different keep/drop bands.

- Organization of experiments. Important analyses (such as ablations and hyperparameter studies) are in the appendix, while three long tables are put in the main text. Please shorten the main tables and move core analyses into the main text so readers do not need to hunt for critical evidence in the Appendix.

- Variance and significance. Tables 1–3 omit standard deviations and the number of runs; the phrase 'multiple runs' is not informative. In a limited-supervision setting—especially when mean gaps are small in the result—significance testing is necessary to establish that the observed gains are statistically meaningful rather than noise. Please report mean and standard deviation with the number of runs, and include significance tests against the baselines. Also clarify what “+FA” denotes in Table 3.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focus on the issues of class imbalance and lack of lables in graph-level anomaly detection. To handle the problems, the paper  proposes to augment graphs using fractional graphs that preserve semantics, a dynamic marginal loss, and a verification strategy to pseudo-label graphs. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. Lable scarcity and class imbalance are crucial problems in graph anomaly detection.
2. The proposed method is model-agonstic, improving performance of SOTA detection methods.
3. The motivation of each technique design is clear.

### Weaknesses
1. The novelty of each component is limited, only a few improvments on existing frameworks.
2. The effectivenss of each component is doubtable, in Table 8, w/o WDML w/o MVP sometimes fall behind GIN.
3. The experiment setting (1% labeled training, 98% unlabeled for testing) brings advantage to pseudo-labeling methods, I think other unsupervised GAD method should be compared [1,2].

[1] GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection
[2] Towards self-interpretable graph-level anomaly detection

### Questions
1. The exp of Table 3 is not convincing, I'd like to see the results of other augmentations + WDML+MVP, to demonstrate the superiority of FGG.
2. What is performance of other pseudo-labeling methods like ConsisGAD?

### Soundness
3

### Presentation
2

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
FracAug proposes an augmentation framework that enhances GNNs by generating semantically consistent graph variants through fractional powers of adjacency matrices and pseudo-labeling with mutual verification to address limited supervision and data imbalance in graph-level anomaly detection. The method consists of three components: (1) Fractional Graph Generator guided by weighted distance-aware margin loss, (2) a margin loss to handle class imbalance, and (3) Mutual Verification Pseudo-Labeler that uses predictions from both original and augmented graphs to iteratively expand the training set. Experiments across 14 GNNs on 12 real-world datasets show consistent gains, boosting average AUROC, AUPRC, and F1-score by up to 5.72%, 7.23%, and 4.18%, respectively.

### Strengths
1. The paper addresses an important and practical problem of graph-level anomaly detection under limited supervision and severe class imbalance, which is common in real-world applications like drug discovery.
2. The proposed fractional augmentation method provides theoretical guarantees (Theorems 1-2) showing that fractional powers of adjacency matrices can preserve semantic spaces while enabling continuous interpolation of graph structures.
3. Extensive experiments demonstrate consistent improvements across 14 different GNN architectures on 12 real-world datasets, showing the method's broad applicability and model-agnostic nature as a plug-in framework.

### Weaknesses
1. Important experiments such as ablation studies should be included in the main paper rather than appendix, as they are critical for validating the core contributions.
2. The variant design in ablation study is problematic. "w/o largest" and "w/o smallest" are not real ablations of the important FGG component but rather create incomplete FGG methods, making worse performance unsurprising. The real ablation should completely remove FGG and replace it with standard graph augmentation methods like edge or node dropout. Similarly, the ablation "w/o MVP" uses only single predictions from original graphs, which obviously performs worse. It should be replaced with widely-used methods like ensemble predictions with dropout enabled to provide fair comparison.
3. The ablation study only involves one backbone (GIN), which is insufficient to support the model-agnostic claim. At least two or three diverse backbones should be tested, especially given that Tables 1-2 show dramatically different improvement magnitudes across architectures.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the challenges of graph-level anomaly detection (GAD) under extremely limited supervision and severely imbalanced class distributions. Existing graph-level augmentation techniques (e.g., MAA, GMixup, GLA) are often heuristic or rely on linear interpolation, making semantic preservation difficult—especially under scarce labels or extreme imbalance. The authors propose FracAug, a model-agnostic augmentation plugin designed to improve GAD performance without modifying the underlying GNN architectures.

### Strengths
1. Clear problem formulation.
    
    The paper accurately identifies two fundamental challenges in GAD: limited supervision and class imbalance. The proposed combination of augmentation and pseudo-labeling is well aligned with these difficulties.
    
2. Methodological contributions.
    - The use of fractional powers of the adjacency matrix to generate multi-scale structural variations is theoretically grounded and contrasts with discrete perturbation–based augmentations (e.g., MAA, GMixup). The paper argues that such fractional operators preserve semantic continuity in the spectral domain.
    - The proposed Weighted Distance-aware Margin Loss (WDML) integrates instance-specific alignment distance and class reweighting, making it more suitable for heterogeneous GAD samples than fixed-margin approaches such as LMCL or LDAM.
    - The introduction of instance-dependent dynamic margins further distinguishes WDML from fixed-boundary losses and offers better adaptability.
3. High model generality.
    
    FracAug functions as a plug-in module that can be attached to various GNN backbones without architectural modifications, demonstrating strong extensibility.
    
4. Comprehensive experimentation.
    - Evaluation across 12 datasets and 14 GNN architectures is extensive.
    - Ablation studies, hyperparameter analysis, and complexity analysis strengthen the empirical claims.

### Weaknesses
1. Gap between “semantic preservation” and “label preservation.”
    
    Theorem 1 shows that (A^{\alpha} x) can be approximated within the original “semantic space,” but this does not directly imply label invariance. For boundary cases or datasets with strong heterophily/feature heterogeneity, even small spectral perturbations may alter anomaly separability.
    
2. Potential issues in the evaluation protocol.
    
    The MVP procedure appears to incorporate samples from the validation/test sets back into the training loop, essentially creating a semi-supervised setting. However, final metrics are still reported on these same validation/test splits, raising concerns about possible data leakage.
    
3. Possible bias from baseline tuning and loss replacement.
    
    All baselines are replaced with reweighted losses, while FracAug is tuned via grid search to maximize validation performance. It is unclear whether baselines receive equally extensive hyperparameter search or adhere to their original recommended settings. This raises fairness concerns.
    
4. Marginal gains and lack of statistical significance.
    
    Many reported improvements are around 1% or less. In heavily imbalanced tasks, such small margins require confidence intervals and statistical significance tests, especially for AUPRC. Multi-seed averaging are recommended.

### Questions
1. Evaluation protocol.
    
    Does the pseudo-labeling process incorporate samples from the test set back into training and then evaluate on the same test set?
    
    If so, could the authors additionally report results under a strictly held-out test split or cross-split evaluation to eliminate potential leakage?
    
2. Computational complexity.
    
    Please provide explicit time/memory complexity formulas for FGG per iteration, as well as empirical scaling curves with respect to (($k_l, k_s, H_l, H_s$)).
    
3. Empirical justification of semantic preservation.
    
    Beyond Theorem 1, could the authors report:
    
    - label stability rates under continuous sampling of ($\alpha$),
    - and comparisons of feature-level or spectral energy distributions?
        
        These would more directly support the claim that augmentation does not alter label semantics.
        
4. Fairness of baseline tuning.
    
    Can the authors provide a comparison using baselines with their default losses and recommended settings, to rule out potential bias introduced by replacing the original loss functions?
    
5. Failure or degradation cases.
    
    Since several combinations in Tables 1–3 show negligible improvements or even degradation, can the authors provide typical failure-case analyses (including structural or spectral characteristics) to clarify the applicability range of FracAug?

### Soundness
1

### Presentation
3

### Contribution
3
