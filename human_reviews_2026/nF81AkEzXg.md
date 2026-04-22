# CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
Accurate estimation of uncertainty in deep learning is critical for deploying models in high-stakes domains such as medical diagnosis and autonomous decision-making, where overconfident predictions can lead to harmful outcomes. In practice, understanding the reason behind a model’s uncertainty and the type of uncertainty it represents can support risk-aware decisions, enhance user trust, and guide additional data collection. However, many existing methods only address a single type of uncertainty or require modifications and retraining of the base model, making them difficult to adopt in real-world systems. We introduce CUPID (Comprehensive Uncertainty Plug-in estImation moDel), a general-purpose module that jointly estimates aleatoric and epistemic uncertainty without modifying or retraining the base model. CUPID can be flexibly inserted into any layer of a pretrained network. It models aleatoric uncertainty through a learned Bayesian identity mapping and captures epistemic uncertainty by analyzing the model’s internal responses to structured perturbations. We evaluate CUPID across a range of tasks, including classification, regression, and out-of-distribution detection. The results show that it consistently delivers competitive performance while offering layer-wise insights into the origins of uncertainty. By making uncertainty estimation modular, interpretable, and model-agnostic, CUPID supports more transparent and trustworthy AI.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Comprehensive Uncertainty Plug-in estImation moDel (CUPID), an uncertainty estimation method that quantifies both epistemic uncertainty (EU) and aleatoric uncertainty (AU) without requiring retraining of the base model. The method introduces a CUPID module, which operates on top of intermediate network features to estimate uncertainty. This module learns to capture AU through a log-likelihood objective and EU through a loss which encourages maximal feature differences and minimal reconstruction differences. Experiments show performance of CUPID on downstream tasks like misclassification detection, out-of-distribution (OOD) detection, and regression.

### Strengths
1. Novelty of the proposed method. The proposed objectives and CUPID module are novel ideas that seem quite robust and well-performant for downstream applications. The method also has the benefit of not requiring retraining of the base model, reducing computational overhead.
2. Clarity. The presentation of the paper is well organized and discussion are clear / easy to follow.
3. Experimental results. The experiments cover a good number of baselines and applications, giving a good benchmark of how the method is expected to perform in real-world scenarios.

### Weaknesses
1. Computational analysis. While the method doesn't require retraining of the base model, some metrics (e.g., runtime, memory consumption) of CUPID should be provided to give an idea of the expected computational overhead.
2. For the misclassification experiment in Section 4.1, it would be good to find some empirical evidence that GLV2 is AU dominant and HAM1000 is EU dominant (i.e., not just by looking at other baseline results, but by examining at the data itself).

### Questions
1. I'm a little unclear on the intuition behind the EU loss function in Section 3.3. Why do we want the intermediate features to be maximally different, but the outputs to be minimally different? 
2. What are the recommended strategies for choosing $\lambda_1$ (e.g., is there a fixed value that tends to work well or does this need to be carefully tuned)?

### Soundness
4

### Presentation
4

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
The CUPID framework proposes a model-agnostic, plug-in method for estimating and decomposing aleatoric and epistemic uncertainties within deep neural networks. Instead of retraining or modifying the base model, CUPID attaches an auxiliary uncertainty branch to intermediate feature layers, allowing uncertainty estimation through a single forward pass. The aleatoric component is predicted via a regression branch that models data-dependent variance, while the epistemic component is derived from the feature reconstruction error of a decoder, both regularized under a Bayesian identity mapping to prevent mutual interference. This layer-wise design enables localized uncertainty interpretation and can be flexibly applied to pretrained networks across different tasks. Experimental results on classification, regression, and OOD detection demonstrate that CUPID achieves competitive or superior calibration and interpretability compared to existing post-hoc and evidential approaches, highlighting its practicality as a lightweight uncertainty quantification module.

### Strengths
The proposed CUPID framework offers a simple yet flexible plug-in design that can be applied to pretrained models without retraining. This practical modularity makes the method attractive for real-world applications where uncertainty estimation must be added post hoc. The joint decomposition of aleatoric and epistemic uncertainties provides a unified view of model and data uncertainty, and the layer-wise formulation enables interpretable analysis of which network components contribute most to uncertainty. These design choices improve usability and diagnostic insight with minimal computational overhead.

### Weaknesses
**Methods**

The proposed methods for estimating aleatoric and epistemic uncertainty are conceptually straightforward and largely aligned with prior works. The aleatoric branch essentially performs feature-based variance regression, similar to heteroscedastic uncertainty modeling in Kendall & Gal (2017) and Evidential Deep Learning (Sensoy et al., 2018). The epistemic uncertainty is derived from feature reconstruction errors, an idea previously explored in autoencoder-based OOD detection (An & Cho, 2015) and variational approaches. Therefore, the underlying mechanisms are not fundamentally novel. The contribution of CUPID mainly lies in its plug-in, layer-wise formulation and the joint decomposition enabled by the Bayesian identity mapping, which improves modularity and interpretability but does not constitute a substantial algorithmic innovation.

References
An, J. & Cho, S. (2015). Variational Autoencoder based Anomaly Detection using Reconstruction Probability. — Workshop on Machine Learning for Signal Processing (MLSP), IEEE.


**Experiments**

Although Section 3.4 describes a unified objective for jointly training the aleatoric and epistemic branches, the experiments appear to use separately trained versions (CUPID-A and CUPID-E) rather than a fully joint model. The paper would benefit from clarifying whether the joint training was actually performed, and if so, providing explicit results comparing the jointly trained version with the individual branches. Otherwise, the claimed joint decomposition may remain only conceptual rather than empirically validated.

### Questions
Q1.
In Tables 1, 2, and 3, I could not find the results of the unified model that jointly trains the aleatoric and epistemic branches. Moreover, the performance trends between the aleatoric and epistemic versions appear quite contradictory, suggesting that joint training might degrade performance. This inconsistency limits the practical benefit of the proposed method, as it becomes cumbersome to decide which type of uncertainty is more suitable for a given task.

Q2.
Using feature reconstruction as a proxy for epistemic uncertainty is not new. What distinguishes CUPID’s reconstruction-based approach from prior autoencoder-based OOD detection or evidential methods? Furthermore, is there any theoretical justification for interpreting reconstruction error specifically as epistemic uncertainty rather than as a general measure of representation distance?

Q3.
The aleatoric branch directly regresses variance from intermediate feature activations. How is this branch trained in classification settings where ground-truth noise levels are unavailable? Are the predicted variances calibrated, normalized, or otherwise constrained to ensure meaningful uncertainty estimates?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a joint estimation of aleatoric and epistemic uncertainty throuhgh a plug-and-play  auxililary module insertable at any layer of a pretrained deep model. The aleatoric uncertainty is estimated through an uncertainty branch and the epistemic uncertainty is estimated through the difference between original and perturbed outputs. The aleatoric and epistemic uncertainty loss is applied together to train the auxiliary module. The usefulness of the quantified uncertainty has been demonstrated in medical image misclassification detection, OOD detection, and image resolutions.

### Strengths
**1. Well-written:** The paper is well-written and well-motivated with good literature review. 

**2. Experimental results:** The paper is enriched with numerous experimental evaluations of their methods in different tasks and demonstrated their success with AUC, AURC, spearman and Pearson coefficients. They also evaluated the calibration error. Their proposed method perform better in almost all cases.

### Weaknesses
**1.Confusing interpretation of epistemic uncertainty:** The epistemic uncertainty has been considered as the discrepancy between the original and the perturbed prediction. However, why this discrepancy captures the model's lack of knowledge is not clear from the paper. 

**2. Limited novelty in aleatoric uncertainty:** The estimation of aleatoric uncertainty seems similar with BayesCap [2] and therefore that part has limited novelty.  


[1] Depeweg, S., Hernandez-Lobato, J.M., Doshi-Velez, F. and Udluft, S., 2018, July. Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. In International conference on machine learning (pp. 1184-1193). PMLR

[2] Upadhyay, U., Karthik, S., Chen, Y., Mancini, M. and Akata, Z., 2022, October. Bayescap: Bayesian identity cap for calibrated uncertainty in frozen neural networks. In European Conference on Computer Vision (pp. 299-317). Cham: Springer Nature Switzerland.

### Questions
The epistemic uncertainty part of the loss function seems similar with the identity mapping of the BayesCap function. Is it possible to draw a direct comparison with this paper's loss with BayesCap?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a plug-in module that estimates both aleatoric and epistemic uncertainty without retraining or changing the base network. The proposed CUPID is attached to intermediate layers of an existing network and includes two branches: an uncertainty branch that learns heteroscedastic variance for aleatoric uncertainty and a reconstruction branch that generates feature perturbations to quantify epistemic uncertainty. Experiments cover misclassification detection, out-of-distribution detection, and image super-resolution regression

### Strengths
+ CUPID generally outperforms or matches baselines like MC Dropout, Rate-in, PostNet, BNN, DEC, and BayesCap, with its two branches complementing each other across tasks.
+ A theoretical section provides a first-order Taylor expansion showing that epistemic uncertainty scales with the product of network sensitivity and feature deviation.

### Weaknesses
- The AU branch resembles standard heteroscedastic regression, and EU relies on output-preserving perturbations, which is close in spirit to prior work that estimates distributional shift via reconstruction error [RUE (Wang et al., 2023)]. It is unclear what the most significant novelty of this paper is.
- CUPID adds a learned module plus an extra forward of the perturbed path, but there is no computational cost analysis. 
- The authors claim CUPID produces reliable uncertainty estimates, however, they do not check if high predicted uncertainty actually corresponds to higher error or misclassification rate. They only reports a calibration measure UCE in the regression (super-resolution) tasks, but not in classification or OOD detection.

### Questions
Can the authors explain why CUPID Aleatoric performs poorly on the PAPILA OOD detection experiment?

### Soundness
2

### Presentation
3

### Contribution
2
