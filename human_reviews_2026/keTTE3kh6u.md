# Not All Samples Are Equal: Quantifying Instance-level Difficulty in Targeted Data Poisoning

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Targeted data poisoning attacks pose an increasingly serious threat due to their ease of deployment and high success rates. These attacks aim to manipulate the prediction for a single test sample in classification models. Unlike indiscriminate attacks that aim to decrease overall test performance, targeted attacks present a unique threat to individual test instances. This threat model raises a fundamental question: what factors make certain test samples more susceptible to successful poisoning than others? We investigate how attack difficulty varies across different test instances and identify key characteristics that influence vulnerability. This paper introduces three predictive criteria for targeted data poisoning difficulty: ergodic prediction accuracy  (analyzed through clean training dynamics), poison distance, and poison budget. Our experimental results demonstrate that these metrics effectively predict the varying difficulty of real-world targeted poisoning attacks across diverse scenarios, offering practitioners valuable insights for vulnerability assessment and understanding data poisoning attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies instance-level difficulty for targeted data poisoning. Instead of reporting averaged attack success rates (ASR), it asks why some test samples are easier to poison than others and proposes three predictive metrics computed without running an attack: (i) Ergodic Prediction Accuracy (EPA) from clean-training dynamics; (ii) a poisoning distance δ, the minimal one-step parameter move along the sample’s gradient that flips the target sample to a chosen poison class; and (iii) a lower bound on poison budget τ derived from a reachability condition (Lambert-W expression based on Lu et al., 2023). Across CIFAR-10 (ResNet-18) and a TinyImageNet ablation (VGG-16) with GM/FC/BP attacks, the authors show: EPA separates easy vs. hard targets; δ and τ provide finer ranking conditioned on poison class; and results hold under different budgets and training regimes (from-scratch vs. linear-head transfer). The work argues these metrics can guide defenders to pre-screen vulnerable instances.

### Strengths
- The paper asks a concrete, useful question at the right granularity and sticks to it.

- Experiments are careful and show stable trends across setups rather than one-off wins.

- The ideas are easy to plug into existing workflows and yield actionable guidance for practitioners.

### Weaknesses
- Scalability is questionable since the training-dynamics signal is computationally expensive at realistic scales.
- The metrics are proxies rather than outcomes of an actual poisoning process, and robustness across optimizers or norms is underexplored.
- Evidence is concentrated on standard image classification; generalization to larger models, other modalities, or backdoor threats remains unclear.
- The paper does not report calibrated predictive utility (for example AUC or operational thresholds), which limits direct deployment guidance.

### Questions
See weakness as above.

### Soundness
3

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
In this paper, the authors studied data poisoning attacks for image classification. In particular, what are the main factors that affect the success rate of a targeted poisoning attack. They have identified three factors: 1) classification difficulty for the clean model, 2) poisoning distance and 3) poison budget, as the main factors. They provide some insights about why these three factors may affect the attack successful rate and also how to compute those factors with access to only the clean model, training data and the targeted test sample. They have demonstrated the correlation between these factors and the ASR on attacking models trained on CIFAR-10 and Tiny-ImageNet.

### Strengths
The authors studied an interesting problem and the structure of the paper is clear and easy to follow.

### Weaknesses
I listed a few weaknesses on the paper as follows:
1. It is not very clear how the authors measure and determine if the attack is success or not. There are at least two criteria to evaluate the success of a poisoning attack at test time: 1) if the test sample $x_t$ is successfully mis-classified as the target class $y_p$; 2) if the overall model performance on other test samples are almost the same as a clean model. I do not see any mentioned of the 2nd criteria in the discussion.
2. Are there any co-relation among these three metrics? Does one affect the other two? As computing them require different information and compute resources (one may be more costly than the other), does one need to compute all of them to understand attack difficulty?
3. Is there any other factor that may affect the attack difficulty? Like the hyperparameters setting of the attack algorithm, the model architecture? More ablation study should be conducted to evaluate these factors.

### Questions
1. In figure 2, it seems like in general the ASRs are higher for high EPA groups than low EPA groups per poison class, however, the variance of the high EPA groups are much larger than low EPA groups. It makes it hard to tell if EPA is a good metric to determine attach difficulty? There is no explanation on why the variance for high EPA groups is almost twice larger than low EPA groups.
2. The 3rd metric is already proposed by Lu et al. (2023)? So this is not a new metric?
3. Fig 3 a) looks different than Figure 2, what have changed in the experimental setting?
4. In the paragraph discuss Fig 3, it seems like per class accuracy is also a very important indicator for attack difficulty? How important is per class accuracy comparing to EPA?

### Soundness
2

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
3

### Summary
This paper investigates the heterogeneity in vulnerability among individual test instances under targeted data poisoning attacks. Unlike prior works that evaluate attack success rates in aggregate, the authors propose three predictive metrics: Ergodic Prediction Accuracy (EPA), Poisoning Distance, and Poison Budget Lower Bound, to quantify instance-level poisoning difficulty. These metrics are derived solely from clean training dynamics and model parameters, enabling defenders to assess vulnerability without performing actual attacks. The authors demonstrate strong correlations between these metrics and empirical attack success rates. The paper contributes a conceptual and empirical framework for understanding and predicting targeted poisoning susceptibility.

### Strengths
1. The work introduces a fresh viewpoint by shifting the analysis from aggregate attack success rates to per-sample poisoning difficulty, a dimension rarely explored in the literature on adversarial and poisoning attacks.
2. The three complementary metrics (EPA, $\delta$, $\tau$) cover distinct yet interrelated aspects of vulnerability: training dynamics, model-space perturbations, and resource constraints, forming a coherent conceptual framework.
3. The proposed metrics can be computed without executing any attack, making the approach practical for real-world defensive analysis where retraining and large-scale attack simulations are infeasible.

### Weaknesses
1. The theoretical foundation for metric relationships is limited. While there exits some empirical correlations, the paper lacks formal theoretical grounding explaining why EPA, $\delta$, and $\tau$ should predict poisoning difficulty beyond intuitive hypotheses. The metrics’ interdependence and causal relationships remain underexplored.
2. Experiments are primarily limited to CIFAR-10 and TinyImageNet with ResNet-18 and VGG-16. The findings may not generalize to large-scale or non-vision models such as multimodal systems, where training dynamics differ significantly.
3. The sensitivity and calibration of metrics is unclear. The quantitative scales of $\delta$ and $\tau$ lack interpretability, and there is no discussion on thresholds or normalization, making it difficult for practitioners to operationalize these measures in practice.
4. The analysis of failure cases and anomalies is incomplete. The authors acknowledge anomalies where $\delta$ or $\tau$ predictions deviate from observed attack success, but provide limited analysis of these inconsistencies, leaving open questions about robustness under different data distributions.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
