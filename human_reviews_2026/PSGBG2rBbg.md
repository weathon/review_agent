# UFL: Uncertainty-Driven Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Federated Learning (FL), a privacy-preserving distributed machine learning, encounters numerous challenges in practical applications, notably Data Heterogeneity (DH). Current methods primarily address DH, relying on coarse dataset statistics, server aggregation, or local model uncertainty. This paper reveals that FL exhibits a distinct sample-level uncertainty distribution during training, characterized by a pronounced long-tail effect. We further show that this long-tail effect is not solely attributable to DH, but is also an inherent characteristic of the FL framework itself. To this end, we propose Uncertainty-driven Federated Learning (UFL), a framework designed to address the uncertainty challenge at the sample level. UFL employs Monte Carlo (MC) dropout to estimate sample uncertainty and adaptively re-weights the loss function accordingly. Moreover, we design U-Agg, a robust aggregation method using clients' accumulated high-uncertainty sample uncertainty to adjust aggregating weights and improve convergence with theoretical guarantees. Unlike existing approaches that alleviate DH at coarser levels, UFL introduces a sample-centric perspective that directly addresses the uncertainty challenge from its fundamental source, offering an orthogonal yet complementary dimension to traditional techniques. Extensive experiments demonstrate that UFL outperforms SOTA FL methods by mitigating the long-tail effect of sample uncertainty, offering a novel and complementary perspective on sample-level uncertainty to enhance FL efficacy over DH solutions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces uncertainty aware FL, a framework designed to tackle the heterogeneity challenge in FL. The work is motivated from the observation that FL exhibits a long tail of sample uncertainty which means the model is less confident in the predictions unlike the centralized setting. The work addresses this by modifying both the local training and the aggregation mechanism. During local training the monte carlo dropout is used to estimate the samples level uncertainty and the samples are reweighted according to this uncertainty for the next round of training. For aggregation, the server down weights the parameter weights of the clients where the uncertainty is higher.

### Strengths
1. The paper is well written and easy to read.
2. The integration of the sample level uncertainty in local training and server aggregation is a nice idea is elegant.
3. Due to its generality, the idea can be utilized with many existing FL algorithms.

### Weaknesses
1. All the experimental results show very minor improvements over existing baselines, therefore the actual impact of the idea is not clear, especially given the fact that applying MC dropout during local training incurs significantly more costs for clients.\
2. The authors bring up sample level uncertainty to be intrinsic to FL, but it is only shown empirically, discussion around how and why this occurs would enhance the motivation of this work and may bring up new insights to develop a stronger algorithm.

### Questions
1. In different FL setups, how is the uncertainty distributed across clients? And how does that distribution affect the performance of the model?
2. The experiments section mention that each round involves only 1 epoch local training, does having more epochs worsen the performance? 
3. How does the performance compare in IID FL setting?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript is motivated by the so-called uncertainty, which is exactly the confidence values on ground truth classes. It claims that the "uncertainty" stems from the framework of FL itself, which is actually client drift during local training. The non-IID settings is actually class-wise uniform. Motivated by "sample-wise long-tail uncertainty", this paper propose to weight the aggregation ratio of global models.

### Strengths
- Simple method and easy to read.
- Clear convergence bound of a non-convex classical problem.

### Weaknesses
- The bad performance is the cause of uncertainty, not the other way around. Notice that the one-hot label already filtered other information than the ground truth. The metric is highly related to ground truth labels, where a better performance leads smaller uncertainty, i.e., the confidence values (probability logits) on ground truth classes. The main claim is invalid, because any models of bad performance could lead to the so-called "uncertainty". Under supervised testing data, a good model is only good in one way, while a bad model can be bad in a thousand strange and bizarre ways. This is why it seems like logarithm long-tail of predicted probabilities on ground truth classes.
- The theoretical analysis is a classic analytical framework on convergence rate of non-convex problem. The only uncertainty-aware term is B_{G}^{2}+V \hat{U}_{k} in Assumption 4.4, which is an upper bound of client-wise gradient variance. Not discussion about whether this assumptions holds at all. The related results is the 4-th term of the component B_{G}^{2}+V\hat{U}_{k}, the unchanged upper bound of classic clien-wise gradient variance!
- Thus, the no novel and dived-into empirical and theoretical insights are provided by this paper.

### Questions
Could your provide a correlation between performance (e.g., accuracy) and uncertainty on varied settings besides class-partition CIFAR-10?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes UFL, a federated learning framework designed to address sample-level uncertainty. Specifically, UFL applies MC dropout to estimate sample-wise uncertainty, assigning higher weights to high-uncertainty samples during local client training. Furthermore, it utilizes the accumulated uncertainty across clients to adjust the global aggregation weights, down-weighting high-uncertainty local models and emphasizing low-uncertainty ones. Experiments demonstrate improvements over existing methods.

### Strengths
- The overall presentation is clear and well-organized.
- The paper is easy to follow and provides open-source code.
- The paper presents theoretical analyses to demonstrate that convergence is affected by client uncertainty.
- The paper provides comprehensive visualizations that effectively illustrate the proposed uncertainty phenomenon in federated learning.

### Weaknesses
This paper is a promising work but requires major revision before publication:
1. The uncertainty phenomenon is not well-explained. For instance:
- It’s unclear why the paper uses the uncertainty distributions at 75% epochs instead of others. IMHO, the distributions at 100% epochs show little to no long-tail behavior.
- It’s also unclear how the hard/easy samples are computed and identified in Figure 1 (c-e). 
- The experimental settings of Figure 1 are missing, e.g., the number of clients and the degree of heterogeneity.
These missing details make it difficult to fully understand the uncertainty phenomenon.
2. The paper lacks a clear interpretation of how this long-tail uncertainty affects the federated learning system. A more detailed explanation (ideally with empirical analysis) would be much better.
3. The experimental evaluation is not comprehensive enough:
- The compared baselines are relatively old, and more recent approaches should be included.
- The ablation study only focuses on parameter sensitivity in UFL, missing the key ablations on the proposed components (i.e., the re-weighting and aggregation components).

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors handle data heterogeneity (DH) in federated settings and approximates the sample-level uncertainty via Monte Carlo Dropout (Gal & Ghahramani, 2016). The authors have shown some convergence results.

### Strengths
Addressing sample-level uncertainty is a major and important problem.

### Weaknesses
- Robust aggregation schemes against adversarial updates/nodes has nothing with handling distribution shifts and data heterogeneity.These are two distinct problems with different objectives and solutions. 

- Assumption 4.4 does not capture any meaningful distribution shift. The upper-bound can be very large, which makes the bound impractical. Convergence results in Theorem 4.5 do not provide any insight into handling data heterogeneity under a meaningful and practical setting.

- Baselines in Figure 2 do not handle data heterogeneity in terms of distribution shifts. Some important baselines are missed: 

[4] A. Ramezani-Kebrya, F. Liu, T. Pethick, G. Chrysos, and V. Cevher. Federated learning under covariate shifts with generalization guarantees. TMLR 2023.

[5] Z. Wu, C. Choi, X. Cao, V. Cevher, and A. Ramezani-Kebrya. Addressing label shift in distributed learning via entropy regularization. ICLR 2025.

### Questions
Please address the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2
