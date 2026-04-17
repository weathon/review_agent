# Online Domain Indexing

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Domain adaptation (DA) in real-world applications often unfolds in an online fashion, where data arrives sequentially with limited domain access and imbalanced sampling across domains. For example, in personalized ads prediction, users from different demographic groups (e.g., countries or age cohorts) correspond to distinct domains with highly skewed data availability, and user interests evolve over time. Recent work has explored domain indices to capture latent inter-domain relationships and improve adaptation (Wang et al., 2020, Xu et al., 2023). However, existing methods such as Variational Domain Index (VDI) (Xu et al., 2023) assume full domain observability and balanced mini-batches, limiting their applicability to real-world scenarios with online domain shift and data imbalance. To address these challenges, we propose Online Domain Indexing (ODI), the first continual domain indexing and adaptation framework designed for partial domain access and inter-domain sample imbalance. Starting from a base model pretrained on historical source and target domains, ODI incrementally updates domain indices over time using a smoothed reweighting kernel and a replay buffer to ensure stable adaptation. Experiments on both synthetic and real-world datasets demonstrate that ODI consistently outperforms state-of-the-art baselines in long-term accuracy under dynamic and resource-constrained conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The method tackles the problem of existing DA methods, which assumes that all data can be accessed with balanced distribution. However, in most applications, data are accessed online and in order, where only partial domains’ data are seen and accessible. They proposed the online-domain indexing framework, with the online-imbalanced DA problem, where only a few domains can be accessed with imbalanced data number. The proposed contributions incorporate the temporal prior on domain indices, domain-index-aware-reweighting, and the domain-aware replay buffer.

### Strengths
-	The new problem seems sound and from the real-world application scenarios.
-	The optimization and the lower bound seems correct.
-	A toy-example dataset is generated to prove the method and theory.

### Weaknesses
-	This method seems to integrate the partially observable (continual DA) with the VDI. The complexity of the setup seems unrealistic. 
-	The method wise, it seems that the authors missed another line of works, focusing on the ``index-less’’ continuous domain adaptation [r1]. This paper seems to consider the continually coming data but without knowing the domain index. It seems a good solution to the proposed setup and also the simplified setup. 
-	The evaluation is only on the small-scale dataset; The comparison methods are very limited. 
-	Reweighting techniques have been utilized and applied in DA for a long time. The overall novelty seems limited by integrating several techniques together.

[r1] Delving into the Continuous Domain Adaptation, MM, 2022

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a realistic online multi-domain adaptation setting where each training round receives only a subset (k ≪ C) of domains with domain imbalance. It extends VDI by (1) introducing a time-varying prior to encourage temporal continuity, and (2) designing Domain-Index-Aware Reweighting to mitigate data imbalance. The method is evaluated on Growing Circle, CompCars, and TPT-48. On all of these datasets, the proposed method beats all baselines.

### Strengths
* This is a realistic problem for online products: partial access + imbalance + drift.
* The design is simple and easy to extend VDI.
* The gains in both in-round and next-round evaluation are consistent.

### Weaknesses
* Major: Motivation is online indexing for ads/recs, but no CTR/CVR datasets are used. Circles/cars/temperature don’t guarantee benefits for CTR/CVR datasets.

* No ablation studies about the importance of each component: temporal prior vs. DIAR vs. replay.

* The paper mentions dynamic and resource-constrained conditions in the abstract but doesn’t discuss this in the results.

* Most shifts are smooth; it should be tested on harsher cases such as abrupt shifts, tiny k, and extreme imbalance.

* TPT-48 tables label MSE as “Accuracy.”

### Questions
Please check Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new setting, online domain adaptation learning with imbalanced labels. They proposed a new framework with (1) temporal priors over domain indices, and (2) domain-index-aware reweighting to tackle the challenge. Abundant experiments are conducted to validate the method's effectiveness. Overall, this paper is interesting and technically sound.

### Strengths
1. Easy to read, easy to follow, well organized.
2. The new problem formulation is well defined and close to the real application.
3. The framework design is technically sound.

### Weaknesses
1. Please add more details about the experiment implementation. More specifically, how do you simulate "online scenarios with distributional shift, partial observability, and data sparsity"?
2. How about the method's performance in the traditional DA setting? I think it is fair to compare ODI and VDI in the previous DA setting.
3. Ablation study is missed. Please analyze the contribution of (1)  temporal priors, (2) domain-index-aware reweighting, and (3) replay buffer?

### Questions
Please answer the question in Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
