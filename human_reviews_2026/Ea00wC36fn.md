# Bi-level Heterogeneous Learning for Time Series Foundation Models: A Federated Learning Approach

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Heterogeneity in time series data is more pronounced than in vision or language, as temporal dynamics vary substantially across domains and tasks. Existing efforts on training time series foundation models (TSFMs) from scratch are often trained with mixed-batch strategies that merge large-scale datasets, which can cause gradient conflicts and degrade representation quality. To address this, we propose a fine-grained learning method that distills invariant knowledge from heterogeneous series while reducing cross-domain interference. We characterize heterogeneity at two levels: inter-domain and intra-domain. To tackle this bi-level heterogeneity, we design a federated learning method that mitigates intra-domain conflicts by enforcing domain-invariant and semantically consistent representations through local regularization, and addresses inter-domain discrepancies by enhancing cross-domain collaboration via domain-aware aggregation. Experiments across diverse benchmarks show that TSFMs trained with our method consistently outperform both centralized and federated TSFM baselines in point and probabilistic forecasting, while also achieving competitive zero-shot performance at scale, offering a flexible pathway for training TSFMs from scratch in heterogeneous environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a federated learning framework (FedTRL) to address bi-level heterogeneity, including both inter-domain and intra-domain heterogeneity. FedTRL primarily includes a fine-grained joint optimization–aggregation strategy, which integrates domain-adversarial regularization and domain-aware aggregation.

### Strengths
S1: The paper introduces federated learning into the development of time series foundation models.

S2: Extensive experiments are conducted on multiple benchmarks, including TSLib, GIFT-eval, and the FEV leaderboard.

S3: The experimental descriptions are relatively detailed.

### Weaknesses
W1 (Motivation): The authors claim that mixing heterogeneous data during pretraining can obscure domain-specific structures, thereby limiting model generalization. However, this point lacks in-depth explanation and empirical support. On the contrary, why couldn’t combining data from different domains help the model learn cross-domain shared knowledge and enhance generalization? The authors should provide a more detailed discussion and analysis here.

W2 (FedTRL Training): FedTRL’s pretraining on large-scale data requires training separate encoders, decoders, and prediction heads for each domain, and then aggregating encoder parameters across domains to update them. This design may raise several issues: 1. The decoder/prediction head might become incompatible with the updated encoder. 2. The framework must store all models for each domain, resulting in high storage overhead. 3. If downstream data come from a domain not included in the pretraining domains, it is unclear how the model would handle such cases.

W3 (Pretraining Data): The pretraining dataset exhibits severe domain imbalance (i.e., different domains have significantly different data proportions). Such imbalance may substantially affect parameter interactions between the clients and the server in FedTRL. If so, how does the method solve this problem?

### Questions
Q1: In Equation (3), how is ( y_{i}^{dom} ) obtained? Does each patch correspond to a label ( y )?

Q2: The prototype in the paper is simply derived by averaging features, without considering time series characteristics such as seasonality or trends. Would it be better to design prototypes that explicitly encode time series characteristics to better align local and global representations?

Q3: Why design a dual-head architecture instead of using a single probabilistic head that can perform both probabilistic and deterministic predictions simultaneously?

Q4: In Table 1, the federated training results are worse than training on individual datasets (e.g., FedTRL’s results are far below those of PatchTST in its original paper).

Q5: In Table 3, the paper lacks comparisons with recent time series foundation models, such as VisionTS (ICML 2025), Sundial (ICML 2025), and LightGTS (ICML 2025).

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
5

### Summary
This paper proposes a time series forecasting modeling method based on federated learning.

### Strengths
1. This paper optimizes the design from two perspectives: cross-domain and intra-domain.

### Weaknesses
1. Besides data privacy concerns, what are the differences between federated learning and pre-training and fine-tuning? Why is this approach being considered for the time series field?

2. In order to train a good foundation model, it is important to extract general capabilities, but for each client, specific patterns are beneficial to its own downstream tasks. Using adversarial learning in local optimization will sacrifice specific knowledge.

3. In the experiments corresponding to table 1, methods such as PatchTST are not designed for joint training, so the corresponding experiments are meaningless and cannot explain any problems.

4. In the experimental setting corresponding to Table 2, when there are multiple test datasets for testing, will the local model parameters after fine-tuning the first test dataset be fed back to the base model for model update?

5. A follow-up question: During pre-training, is it better to include more training datasets? Can you provide relevant experiments to make a qualitative judgment?

6. How can we achieve fine-grained data segmentation within a domain?

### Questions
See weaknesses.

### Soundness
2

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
This paper proposes FedTRL, a federated framework tackling bi-level heterogeneity in time-series foundation models. It addresses inter-domain differences and intra-domain conflicts through local adversarial regularization and prototype alignment for semantic consistency, and domain-aware aggregation for cross-domain collaboration. Experiments show that FedTRL outperforms centralized and traditional federated baselines in both point and probabilistic forecasting, achieving stronger zero-shot generalization.

### Strengths
The paper clearly defines and formalizes inter- and intra-domain heterogeneity in time-series foundation model training, providing an insightful perspective.

The overall design of FedTRL — combining local optimization and global aggregation — is logically sound and systematically organized.

The experiments cover multiple datasets and both point/probabilistic forecasting tasks, supporting the claims effectively.

The model demonstrates stable performance on unseen domains, showing effective cross-domain transfer.

### Weaknesses
Although FedTRL uses adversarial regularization and prototype alignment, these mainly enforce coarse semantic consistency and may fail to capture continuous or nonlinear sub-domain drift.

The framework jointly updates local adversarial modules and prototypes while performing domain-aware aggregation each round, leading to heavy overhead in large-scale federations.

Despite claiming “domain awareness,” the paper lacks visualization or empirical analysis showing how representations differ across domains.

Aggregation weights are heuristically defined without rigorous justification

### Questions
See weakness.

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
4

### Summary
The paper proposes **FedTRL**, a federated learning framework designed to train **time series foundation models (TSFMs)** under **bi-level heterogeneity** — both inter-domain and intra-domain differences across clients.
It introduces a **dual-level optimization** strategy combining adversarial local regularization and domain-aware global aggregation to achieve domain-invariant and temporally coherent representations.
Extensive experiments across in-domain, full-shot, and zero-shot forecasting tasks show that FedTRL achieves **state-of-the-art performance**, outperforming both centralized and existing federated baselines.

### Strengths
## **Strengths**

* The proposed method is technically sophisticated and conceptually “fancy.” The architectural design and figures are very clear and visually appealing.
* The paper is well written and easy to follow.
* Experimental results are strong, with the proposed method achieving solid performance across multiple benchmarks.

### Weaknesses
## **1. Motivation & Novelty**

The **motivation is not clearly justified**. I am not fully convinced that *heterogeneity* itself should be viewed as a negative factor. On the contrary, **sufficient heterogeneity and diversity in data are often the key enablers for the success of Time Series Foundation Models (TSFMs)**. The paper criticizes heterogeneity but provides **no empirical evidence** or **quantitative analysis** showing that heterogeneity indeed harms federated training.

For example, in the discussion of *inter-domain heterogeneity*, the authors claim that it leads to “overfitting to domain-specific signals” and the failure to “capture globally consistent dynamics.” However, no supporting evidence, ablation, or theoretical justification is provided.

Moreover, the paper frequently refers to *“domain-invariant”* and *“temporally coherent patterns”* without giving a clear definition. These notions remain vague. What exactly constitutes a “global dynamic” in time series? In time-series data, the notion of a *domain* is often **weakly constrained**—for example, even within a single domain such as weather, the underlying distributions can vary substantially. This is very different from computer vision, where the concept of domain is more clearly defined. Therefore, **learning domain-invariant representations for time series seems conceptually questionable**, and this undermines the central motivation of the work.

This is my **primary concern**—without a clearer and empirically grounded motivation, it is difficult to see the necessity of the proposed framework.

---

## **2. Methodological Concerns**

**2.1** Given the above discussion, I suspect that the actual benefit of FedTRL might not stem from addressing heterogeneity per se, but rather from **implicitly improving data diversity and balance during training**. In other words, the observed performance gain may come from a more diversified and higher-quality training process, rather than from the federated or “heterogeneity-resolving” design itself.

**2.2** From the title and framing, the paper claims to propose a *federated learning framework for TSFMs*. However, the proposed FedTRL looks more like a **representation learning model and its training algorithm**, rather than a general-purpose training framework for foundation models. This conceptual mismatch between the claimed goal (framework for TSFMs) and the actual technical contribution (a particular model design) feels somewhat inconsistent.

---

## **3. Experimental Evaluation**

**3.1** The comparison with baselines could be more appropriate. Since the paper emphasizes a *federated* setup, it would be more convincing to compare FedTRL with **federated time-series forecasting baselines**, rather than primarily with representation learning methods. Alternatively, the authors could demonstrate FedTRL as a **general plug-in or enhancement framework** that can consistently improve various federated baselines.

**3.2** Regarding the **GIFT-Eval** evaluation, the authors should clearly state **which specific datasets** from GIFT-Eval were used. It is also important to clarify **whether any of these datasets overlap with those included in the Time-MoE-300B pretraining corpus**, as this could raise concerns about potential data leakage or unfair comparisons.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
2
