# FedReLa: Imbalanced Federated Learning via Re-Labeling

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Federated learning has emerged as the foremost approach for decentralized model training with privacy preserving. The global class imbalance and cross-client data heterogeneity naturally coexist, and the mismatch between local and global imbalances exacerbates the performance degradation of the aggregated model. The agnosticism of global minority classes poses significant challenges for data-level methods, especially under extreme conditions with severe class deficiencies across clients. In this paper, we propose FedReLa, a novel data-level approach that tackles the coexistence of data heterogeneity and class imbalance in federated learning. By re-labeling samples with a feature-dependent label re-allocator, FedReLa corrects the biased decision boundaries without requiring knowledge of the global
class distribution. This modular, model-agnostic approach can be integrated with algorithmic methods to offer consistent improvements without any extra communication burden. Through extensive experiments, our method significantly improves the accuracy of minority classes and the overall accuracy on step-wise-imbalanced and long-tailed datasets, outperforming the previous state of the art.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets a realistic FL setting where global class imbalance coexists with cross-client heterogeneity. It proposes a data-level “one-shot label reallocation” approach (FedReLa). Without relying on global priors or extra communication, the method uses the global model’s local posteriors, within-class z-score normalization, and a tanh thresholding step to identify “suspicious majority-class samples” near minority-class regions and probabilistically relabels them as minority. The goal is to collectively “push back” biased decision boundaries at both local and aggregated levels to mitigate imbalance. The paper claims plug-and-play integration, very low overhead, and composability with algorithm-level methods.

### Strengths
- The paper focuses on a more realistic and challenging combination in FL: global imbalance + heterogeneity + local-global mismatch. This is a relevant and important problem.
- The solution is data-level, requires no global priors or auxiliary data, and can be easily integrated into existing FL pipelines—practically attractive.
- The one-shot local forward pass is low-cost, introduces no extra trainable parameters or communication, and is thus deployment-friendly.

### Weaknesses
- The paper models the aggregated global posterior as a weighted sum of local posteriors, whereas FedAvg averages parameters, and neural-network posteriors are not linearly additive. Deriving the aggregated decision boundary via “posterior averaging” is not rigorous. The authors should justify or correct the assumption that the global posterior equals a weighted sum of local posteriors. If this assumption does not hold, do Lemma 3 and the aggregated-boundary analysis still stand? Please provide empirical evidence quantifying the discrepancy between predictions from parameter-averaged models and those from weighted averaging of local predictions.
-  The theoretical analysis appears restricted to binary classification. Can it be extended to the multi-class case? The theory adopts a Bayesian decision-boundary perspective: strategic label reallocation is equivalent to adjusting effective prior ratios, which pulls back biased boundaries at local/global levels and improves minority/tail recognition—purportedly without global priors or extra communication. Related ideas appear in [1][2]; what are the precise differences and advantages of this work compared to [1][2]? Please add a focused discussion. Also, FedETF seems to pursue a similar effect; why is it meaningful to use FedReLa jointly with FedETF rather than redundantly?
-  Appendix A’s Example 1 is vague. Please clarify what, exactly, the example is intended to illustrate. Also, in FL, one may use uniform averaging rather than weighted averaging—how would that change the conclusions?
-  Despite the theoretical discussion, the proposed method seems essentially heuristic. Please clarify the precise relationship between the method and the theory—what parts of the method are directly justified by the theory, and what parts are heuristic design choices?
- Why use independent Bernoulli sampling rather than a single Categorical draw? If multiple ones occur and you then take argmax, does this introduce bias?
- Why prefer label reallocation over approaches such as SMOTE-like interpolation to synthesize new samples? What advantages does FedReLa offer relative to such feature-space/data-space augmentation methods under FL constraints?
- The paper does not appear to provide code. Could the authors release reproducible code?

[1] Federated Learning with Label Distribution Skew via Logits Calibration

[2] Aligning model outputs for class imbalanced non iid federated learning

### Questions
Refer to Weaknesses.

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
This paper introduces FedReLa, a data-level approach for improving federated learning in both global and local class imbalance and data heterogeneity. The key idea is to employ probabilistic, feature-dependent re-labeling of samples using posterior probabilities estimated from a shared global model, without requiring global class prior knowledge or extra communication. Extensive experiments on Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets, under various imbalance and heterogeneity settings, show notable improvements in minority class and overall accuracy compared to strong baselines and state-of-the-art methods.

### Strengths
1. This paper offers a careful, mathematically grounded analysis of how re-labeling affects Bayesian decision boundaries in both local and global models. 
2. FedReLa is modular and model-agnostic.

### Weaknesses
1. While the design uses the global model to estimate posterior probabilities for label reallocation, the paper does not thoroughly analyze how inaccuracies in these posteriors—especially early in training or under extreme class absence scenarios—might lead to over-flipping or even degrade minority class representation. 
2. While the method is designed to work without minority samples (by only flipping labels into minority categories), it is not sufficiently explained what happens if a class is entirely absent from the whole federation or present in vanishingly small quantities. Is the method robust to “missing labels” at the global level?
3. The risk of introducing noisy labels, especially if the global model is overfitting or miscalibrated. The paper should provide concrete safeguards or error analysis here.
4. Some presentation choices reduce readability. e.g., the indexing of notations and the mixture of probabilistic and empirical normalization steps.

### Questions
See above.

### Soundness
2

### Presentation
2

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
This paper presents a data based federated learning (FL) approach to tackle the challenges of class imbalance and class heterogeneity. The authors introduce asymmetric, feature-based label noise into local data. The main contributions are: 
- It does not require access to the global class distribution 
- It utilizes existing global models to inject label noise without extra communication or local computations  
- It does not depend on any specific model architecture.

### Strengths
The strengths have been outlined in the summary.

### Weaknesses
- The method heavily relies on posterior estimates from the global model to inject noise. This is susceptible to model bias, especially in severely imbalanced scenarios.

- The novelty of the method is incremental. Essentially adding noise FL might be new, but it builds on prior concepts from label-noise learning. A better explanation on novelty should be included. 

- There needs to be a better understanding of the z-score calculation. In particular computationally. How much does it cost to calculate it. 

- The experiments are carried out on a small number of clients. It is needed to run experiments on larger number of clients to empirically validate the method.

### Questions
The strengths have been outlined in the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FedReLA, aiming to address heterogeneous and long-tailed data distributions by re-labeling majority-class samples as minority ones, thereby expanding the decision boundaries of minority classes. Experimental results demonstrate improvements over existing approaches.

### Strengths
- The overall presentation is clear and well-organized.
- The paper is easy to follow.

### Weaknesses
- The main weakness of this paper lies in the lack of evaluation on large-scale datasets, such as ImageNet-LT and Places-LT. Such direct label-space enlargement may face challenges when applied to large-scale scenarios.
- The paper only provides two Fed-LT comparisons, FedETF and FedLoGe, which are insufficient to demonstrate the superiority of the proposed method. More recent approaches should be included.
- It is interesting that the proposed method improves the performance of the majority classes when applied to Fed-LT approaches (as shown in Tab. 2). Intuitively, re-labeling the majority-class samples as minority ones should compromise majority-class performance. Could you provide more explanations (ideally with empirical analysis) for this phenomenon?

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
