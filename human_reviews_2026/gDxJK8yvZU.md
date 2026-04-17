# Minimax-Optimal Aggregation for Density Ratio Estimation

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 8

## Abstract
Density ratio estimation (DRE) is fundamental in machine learning and statistics, with applications in domain adaptation and two-sample testing. However, DRE methods are highly sensitive to hyperparameter selection, with suboptimal choices often resulting in poor convergence rates and empirical performance. To address this issue, we propose a novel model aggregation algorithm for DRE that trains multiple models with different hyperparameter settings and aggregates them. Our aggregation provably achieves minimax-optimal error convergence without requiring prior knowledge of the smoothness of the unknown density ratio. Our method surpasses cross-validation-based model selection and model averaging baselines for DRE on standard benchmarks for DRE and large-scale domain adaptation tasks, setting a new state of the art on image and text data.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides an aggregation method for solving density ratio estimation (DRE). Through this, it tackles both the issue of method selection as well as hyperparameter tuning by instead training multiple models with multiple hyperparameter setups and aggregating them instead of just selecting one. The authors provide minimax-optimal error bounds without smoothness assumptions for RKHS and convex settings. Thorough empirical validation of the method against cross-validation baselines is done on models beyond just RKHS settings including neural networks.

### Strengths
* Proposes a novel and effective strategy for dealing with model and hyperparameter aggregation for density ratio estimation.
* The proposed closed form solution beats every candidate model instead of just selecting a single one.
* Theoretical minimax guarantees for RKHS and convex settings.
* Empirical results also show significant improvements over existing selection baselines in different data settings and using models like neural networks.

### Weaknesses
* It is unclear if the theory covers all the empirical settings like the case where models are neural networks, though this is a tough ask.
* The method requires empirical computation of a large Gram matrix of size ~ number of models X hyperparameter settings. This could be numerically unstable for various real world applications.

### Questions
As per the weakness section,

1. Does the theory extend to all your empirical evaluations?
2. Could the authors comment briefly on the numerical stability of computing the Gram matrix in real applications, and also comment on the costs of computing Hessians in Alg 1?
3. Are there guarantees for the aggregated DRE to always be non-negative?

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
This paper proposes a method for training multiple models under different hyperparameter configurations and optimally aggregating their outputs. The proposed approach demonstrates the ability to achieve minimax-optimal convergence rates without requiring any prior knowledge about the smoothness of the density ratios. Experimental results also demons	trate that the proposed method outperforms cross-validation, Bayesian model averaging, and Super Learner.

### Strengths
* The method achieves minimax-optimal convergence rates under reasonable assumptions.
* The proposed algorithm is theoretically guaranteed and practically computable.  The proposed algorithm is a simple least-squares method that is intuitively easy to understand.
* Through various density ratio estimation methods, they demonstrate the superiority of the proposed approach over the baseline methods.

### Weaknesses
* The following descriptions of the experiments are insufficient.
  * The tuning method for the hyperparameter, lambda
  * The details of BMA:  It is better to write which algorithm in Fragoso et al. (2018) is used.

* Since cross-validation requires only inference from a single selected density ratio estimator during inference, it has the computational advantage of requiring less processing time than the proposed method. Therefore, I consider the superiority of the proposed method to be limited to only certain applications.

* Using $f$ both for DRE models in equation (1) and for DA models in equation (13).   The use of the symbol may confuse readers.

### Questions
* How did you choose the regularization parameter value, lambda?  
* Could you discuss how the lambda selection affected the computational complexity in Fig. 2? In my understanding, the trained models and empirical Hessians can be shared across different lambda values, so there is little impact on overall training time. When you tried 10 different lambda values, $\{10^{−6},10^{−5},...,10^{4}\}$ in the experiments, how much did the whole training time increase compared with the single lambda value?

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
The paper find suitable weights that is used to aggregate results from multiple models trained with different hyperparameter to predict density ratio estimation(DRE). The weights are obtained by minimizing the upper bound on the Bregman divergence.

### Strengths
1. There is a strong mathemtical proof
2. There a many benchmarks across various task and domains

### Weaknesses
1. The paper could provide analysis on the number of models trained and results obtained through this method?
2. The paper could also show analysis of the distribution of the hyper-parameters used.

### Questions
1. Does the variance of the models output affect the bounds?
2. How would noise within the data affect the results analytical?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper propose a new algorithm for Density Ratio Estimation (DRE) that aggregates multiple models with different hyper parameter settings. The empirical experiments focus on domain adaptation, as well as comparisons with cross validation for several different loss objectives. The authors prove that the aggregation achieves minimax-optimal error convergence under very mild conditions on the density ratio.

### Strengths
Overall I think this is a good paper. It's written well and proposes a novel method for DRE estimation that is more computationally efficient than simple cross validation, and improves performance over several other baselines, on several relevant tasks. 
The method is supported by a provably minimax-optimal error convergence. 
Several ablations are done to show the robustness of the method.

### Weaknesses
Slightly more detail about the dataset for the top part of Table 1 would have been useful. What does "c3,d1.70" stand for instance, and how many samples were used in the dataset "c3,d1.70". 

Minor:
I think the left part of figure two can be improved. If my understanding is correct, the number of models needed is reduced by a factor of the number of k-folds. By plotting it out instead of just stating the mechanical relationship, people may get confused and look for some patterns in the figure.

### Questions
Can the method also be applied when instead of having different hyperparameters in the same objective, we have multiple models from different objectives (KuLSIF, Exp, etc.)?

### Soundness
3

### Presentation
3

### Contribution
3
