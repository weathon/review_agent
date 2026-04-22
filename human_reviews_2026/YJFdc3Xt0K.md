# Train, Mutate, or Reward? A Unified View of Supervised Ensembling for Time Series Anomaly Detection.

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Time series anomaly detection (TSAD) is a long-standing and extensively studied problem with applications across a large panel of domains. Despite the maturity of the field, recent benchmark studies have revealed that no single detection method consistently outperforms others across diverse datasets. While model selection approaches (i.e., choosing the best detector for a given scenario) have shown promising results, their effectiveness remains inherently limited by the performance ceiling of existing individual detectors.
To address this limitation, supervised ensembling offers a promising path to surpass individual detectors by learning to combine their strengths. In this work, we unify and formalize the problem of supervised ensemble-based anomaly detection in time series, and introduce three principled strategies for learning such ensembles: (1) classical Machine Learning, (2) Reinforcement Learning, and (3) Genetic Programming. We perform a rigorous comparative evaluation across these strategies using identical model components, inputs, and experimental conditions to ensure fairness. Our findings not only highlight the strengths and trade-offs of each approach, but also illuminate promising directions, paving the road for future research on this topic.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies supervised ensemble learning for time-series anomaly detection (TSAD), proposing a unified framework that formulates ensembling as a supervised problem. Three learning paradigms are examined, which are classical machine learning (ML), reinforcement learning (RL), and genetic programming (GP).  Each is trained to learn weights for combining anomaly detectors. The paper aims to evaluate these strategies under a unified experimental protocol on the TSB-UAD benchmark, showing that supervised ensembles can outperform both individual detectors and unsupervised ensemble methods.

### Strengths
1. The paper provides a well-structured overview of TSAD and positions supervised ensembling as a complementary paradigm to model selection and unsupervised ensembles.

2. The effort to evaluate three distinct strategies (,i.e., ML, RL, GP) within a controlled experimental pipeline is valuable for benchmarking purposes. 

3. The paper is well-written and easy to follow. The research question itself is interesting.

### Weaknesses
1. The framework assumes access to labeled anomaly data for supervised training, which contradicts the fundamental challenge of anomaly detection (scarcity of labeled anomalies). The paper does not justify how this setup could be applied or adapted in realistic scenarios where labels are limited or unavailable.

2. The core idea, i.e., learning adaptive weights for multiple detectors, is conceptually close to mixture-of-experts (MoE) and existing adaptive weighting schemes. RL and GP are treated as alternative optimization strategies, not introducing fundamentally new principles for ensembling.

3. Only MLPs and small CNNs are used as the main models. More modern time-series architectures (Transformers, LSTMs, or xLSTMs) should be included to evaluate robustness across model.

4. No ablation studies on model depth, hierarchy, or the number of detectors are provided to verify stability of the observations.

5. The analysis remains largely empirical. The authors report that one method outperforms another, but do not explain why RL succeeds or how the learned weight distribution differs from ML or GP.

6. Qualitative results (e.g., visualization of weight distributions, failure cases, or t-SNE plots of feature spaces) would strengthen the interpretability of findings.

7. The benchmark lacks comparison to recent state-of-the-art TSAD models (e.g., TranAD, OmniAnomaly, Anomaly Transformer,  etc.). Without these, it is unclear whether supervised ensembling truly advances the field.

8. Figures (e.g., Fig. 4) lack explanations of error bars and variance computation.

9. Quantitative results are mostly presented in plots; tabulated performance scores would improve clarity.

### Questions
1. How do the authors justify the assumption of having sufficient labeled anomaly data for supervised training, given that real-world TSAD scenarios are typically label-scarce?

2. Could the proposed framework be extended or adapted to semi-supervised or weakly-supervised settings?

3. The proposed supervised ensemble approach appears conceptually close to mixture-of-experts (MoE) or adaptive weighting frameworks. Could the authors elaborate on what fundamentally differentiates their method from MoE-based approaches?

4. Why should RL or GP be viewed as conceptually distinct ensemble learning paradigms rather than merely alternative optimization schemes?

5. Why did the authors restrict the core models to simple MLPs and CNNs?

6. Have the authors explored more modern architectures such as Transformers, LSTMs, or xLSTMs? If not, could the authors discuss expected differences or challenges in incorporating these architectures?

7. Could the authors provide ablation results on model depth, architectural hierarchy, and the number of detectors to assess the stability of the conclusions?

8. How sensitive are the main findings (e.g., RL outperforming ML/GP) to these hyperparameters?

9. What are the theoretical or intuitive reasons that explain why RL-based ensembling performs better than ML or GP in certain settings?

10. How do the learned weight distributions differ across the three strategies (ML, RL, GP)?

11. Could the authors provide qualitative analyses such as t-SNE visualizations, detector weight distributions, or case studies of failure modes to better understand model behavior?

12. Why were recent state-of-the-art TSAD methods (e.g., TranAD, OmniAnomaly, Anomaly Transformer, THOC, etc.) excluded from the benchmark? Would the proposed supervised ensemble still outperform these modern baselines?

13. In Figure 4, what do the error bars represent, and how were they computed (e.g., across runs, datasets, or random seeds)? Could the authors provide the variance or confidence intervals explicitly?

14. Could the authors include tabulated performance scores (e.g., mean ± std) for all methods across datasets to improve transparency and enable easier cross-paper comparison?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
For time-series anomaly detection, the authors propose supervised ensemble learning.   The ensemble of detectors are linearly weighted and combined.  Three strategies were considered as the underlying detectors: gradient based machine learning (ML) method, reinforcement learning (RL) and genetic programming (GP).  For the gradient-based method, they use MSE as the loss function.  For the reinforcement learning method, they use Proximal Policy Optimization (PPO) algorithm.  For genetic algorithms, each hypothesis/individual in the population is a neural network and the fitness function is accuracy.

The baseline methods are 12 individual detections from the TSB-UAD benchmark, unsupervised ensemble (average of the output of  individual detectors), the best method in (Sylligardos et al., 2023).  They have two scenarios: in-distribution (all 16 datasets from TSB-UAD) in training and test sets) and out-of-distribution.(15 datasets for training, one for testing).  For in-distribution, they found the supervised ensemble with RL has a better performance.  For out-of-distribution, ensemble with RL and GP have a better performance.  When compared the Oracle (the best model for a time series), Supervised ensembles using RL with features can outperform the oracle on some time series.   For in-distribution, they find fewer detectors better, while for out-of-distribution, more detectors better.

### Strengths
1.  Exploring ensemble methods with 3 strategies (ML, RL, and GP) is interesting.

2. The empirical results indicate that supervised ensemble using RL with statistical features generally outperform the compared methods.

3.  The paper is generally well organized and written.

### Weaknesses
1.  Most of the proposed methods are existing so the novelty level is not high.

2.  Other ensemble methods were not included for comparison.

### Questions
Sec 3.1: what are the statistical features?

Sec 4.4: Any insights on why more activated detectors are better in out-of-distributions, while fewer activated detectors are better in in-distribution?

Comments:

Figure 5, x axis: Oralee -> Oracle

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
4

### Summary
This paper addresses the challenge that no single anomaly detector consistently outperforms others across different datasets and anomaly types in time-series anomaly detection (TSAD). The authors explore three supervised ensembling strategies (classical supervised learning, reinforcement learning, and genetic algorithms) for learning detector weighting schemes. They propose a unified pipeline and evaluate these methods on the TSB-UAD benchmark, comparing them against individual detectors, unsupervised averaging ensembles, and model selection approaches. Through extensive experiments spanning 16 datasets and 12 detectors, the authors demonstrate that reinforcement learning with feature-based inputs achieves the best in-distribution performance, outperforming individual detectors and even the model-selection oracle on several datasets. Moreover, models trained on raw inputs exhibit strong generalization in the leave-one-dataset-out (out-of-distribution) setting.

### Strengths
* The paper provides a systematic and insightful comparison between model selection approaches, which are inherently limited by the best individual detector’s performance against the supervised ensembling methods that learn to combine multiple detectors.

* The experimental design ensures fairness: all strategies use the same architectures (MLP for feature inputs, CNN for raw data), consistent data splits (approximately 60/40 train–test), and uniform evaluation metrics (VUS-PR or its proxy). This setup isolates the impact of the learning strategy itself.

* Per–time–series analyses clearly demonstrate that supervised ensembles leverage the complementary strengths of individual detectors, rather than merely selecting one, thereby reinforcing the authors’ motivation.

* The scalability study, which covers training time, inference time, and the trade-off between the number of detectors and overall performance, is both practical and informative.

### Weaknesses
* The paper omits comparisons to widely used ensemble methods such as stacking, blending, or weighted voting. Without these, it is unclear whether the observed improvements stem from the concept of supervised ensembling itself or from the specific learning strategies explored (ML, RL, GP).

* The paper would benefit from situating itself more clearly within existing literature. Are there prior studies that applied supervised ensembling or stacking to time-series anomaly detection? If so, how does this work differ methodologically or empirically?

* The RL approach clips detector weights to  [0,1], while ML and GP use unconstrained (possibly negative) weights.  An ablation varying clipping or allowing signed RL weights would clarify the effect of this constraint.

* ML and GP models occasionally learn negative weights, yet the paper does not analyze their meaning or impact. Are such weights beneficial, or do they indicate overfitting or instability? Reporting constrained vs. unconstrained variants would strengthen the interpretation.

* The paper does not present random seeds, standard deviations, or confidence intervals for results. Given the stochastic nature of RL and GP, this omission makes it difficult to assess the robustness or reproducibility of performance gains.

* The computation of catch22 features is not described, nor is the handling of variable-length time series. Clarifying whether the series are truncated, padded, or resampled would improve methodological transparency.

### Questions
Please address the questions mentioned in the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
