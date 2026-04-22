# Adaptive Quality-Diversity Trade-offs for Large-Scale Batch Recommendation

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
A core research question in recommender systems is to propose batches of highly relevant and diverse items, that is, items personalized to the user's preferences, but which also might get the user out of their comfort zone. This diversity might induce properties of serendipidity and novelty which might increase user engagement or revenue. However, many real-life problems arise in that case: e.g., avoiding to recommend distinct but too similar items to reduce the churn risk, and computational cost for large item libraries, up to millions of items. First, we consider the case when the user feedback model is perfectly observed and known in advance, and introduce an efficient algorithm called B-DivRec combining determinantal point processes and a fuzzy denuding procedure to adjust the degree of item diversity. This helps enforcing a quality-diversity tradeoff throughout the user history. Second, we propose an approach to adaptively tailor the quality-diversity tradeoff to the user, so that diversity in recommendations can be enhanced if it leads to positive feedback, and vice-versa. Finally, we illustrate the performance and versatility of B-DivRec in the two settings on synthetic and real-life data sets on movie recommendation and drug repurposing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
B-DivRec aims to maximize both individual- and aggregate-level diversity of an existing model by post-processing while maintaining accuracy.
In this process, users provide feedbacks for their recommendation lists.
B-DivRec modifies DPP and introduces hyperparameters to control trade-off between accuracy and diversity.

### Strengths
* B-DivRec targets both individual- and aggregate-level diversity in sequential recommendation.
* Authors provide well-packaged experimental code for reproducibility.

### Weaknesses
* The paper is hard to follow.
For instance, the problem definition is scattered across the Notation and Metric sections, and the proposed method is mixed with existing works, making it difficult to clearly distinguish the contributions.
* Problem definition is unclear.
I assume that the problem is to maximize both individual and aggregate diversity while maintaining accuracy by post-processing an existing model for a series of users where they provide feedbacks for each item in the recommendation.
Yet, some details are still unclear such as how users provide their feedback, and how does the feedback model predicting those exist.
* The paper reviews only DPP and MMR for previous works and compares the proposed method with only DPP and its variants.
However, both aggregately and individually diversified recommendations are deeply studied topics so that authors should compare their work with other existing methods.
* The novelty of the proposed method appears insufficient.
As I understand it, the main idea of B-DivRec is:
(1) introducing a trade-off hyperparameter $\lambda$ to balance the contributions of the rating matrix and the similarity matrix in the DPP kernel, and
(2) filtering similar items using a threshold hyperparameter $\alpha$.
Compared to the conventional DPP framework, this approach seems incremental, as it mainly involves adding a few hyperparameters and performing hyperparameter tuning.
Furthermore, the paper does not clearly explain what specific challenges this idea aims to address, nor why introducing these hyperparameters is an effective way to tackle them.
* Proposed metric seems highly sensitive to the threshold $\tau$, which may lead to unfair experimentation.
* Experiments are performed mainly on synthetic datasets rather than real datasets.
* Backbone recommendation models are not clearly specified.
* Performance improvement of the proposed method is marginal.

### Questions
* Please refer to weakneses above.
* How does the recommender system receive the user feedback for the recommendation results during the experiments?
Real-world datasets such as MovieLens contain user feedback for only interacted items which are very little, so most recommended items would not be interacted with the user.
Moreover, if duplicate recommendations of items seen in the training dataset are not allowed as usual, conducting such experiments would have been more challenging.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a unified DPP-based framework that cleanly separates quality and diversity with an explicit weight $\lambda$, subsuming prior DPP variants. Building on this, B-DivRec introduces a “fuzzy denuding” step that filters items too close to a user’s history in feature space, enabling scalable global diversity with linear-in-N computation via Nyström and fast MAP/ $\alpha$ -DPP routines. An adaptive $\lambda$ procedure (AdaHedge-style) tunes the quality-diversity balance per user online. Experiments on synthetic datasets up to 15M items, MovieLens, and a drug repurposing benchmark show competitive or superior trade-offs compared to conditional DPP and MMR; notably, B-DivRec is strong on PREDICT while MMR dominates on MovieLens, and adaptive $\lambda$ improves relevance in some settings.

### Strengths
- The paper presents a unified DPP-based formulation with an explicit trade-off parameter $\lambda$, providing a clear theoretical foundation that encompasses several existing diversity-aware recommendation methods such as conditional DPP and MMR.  
- The proposed B-DivRec approach combines a denuding operation in feature space with Nystrom approximation, achieving linear scalability and enabling large-scale batch recommendation with explicit control over both global and local diversity.  
- The adaptive update of $\lambda$ per user is conceptually appealing, allowing personalized control of the quality–diversity balance and offering a promising direction for user-adaptive recommendation systems.

### Weaknesses
* Despite its theoretical elegance, the paper does not demonstrate consistent empirical superiority of the proposed method. On the MovieLens benchmark, MMR achieves higher relevance scores than B-DivRec; the explanation (history-vector collinearity) is qualitative and lacks deeper quantitative analysis.
* The overall effectiveness of B-DivRec appears dataset-dependent, strong on PREDICT but weaker on MovieLens, raising questions about robustness and generality across domains with different diversity characteristics.
* The experimental evaluation includes only classical baselines and omits stronger modern re-ranking or diversification methods (e.g., xQuAD, deep DPPs, intent-aware models), so the practical advantage remains unclear.
* The adaptive  $\lambda$ update lacks formal guarantees (e.g., convergence or regret bounds) and is tested only under clean, noise-free feedback, limiting confidence in real-world, noisy environments.
λ update lacks formal guarantees (e.g., convergence or regret bounds) and is tested only under clean, noise-free feedback, limiting confidence in real-world, noisy environments.

### Questions
* Could you provide quantitative analysis for why MMR outperforms B-DivRec on MovieLens (e.g., effects of popularity bias, embedding collinearity, or limited intrinsic diversity), and how B-DivRec might be adapted to mitigate these factors?
* Would B-DivRec improve on MovieLens with richer embeddings (e.g., hybrid content–collaborative features) or with per-cluster tuning of $\alpha$ and $\lambda$?
* Can the adaptive $\lambda$ be evaluated under noisy/implicit feedback (clicks, exposure bias) to assess robustness and practicality?

### Soundness
3

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
This paper investigates a scalable approach that considers the relevance-diversity trade-off in recommender systems.
Although the proposed framework is simple, the authors provide a detailed discussion of the implementation (especially in the appendices) to account for practical scalability.
While it lacks theoretical contributions such as regret bounds, the paper's rich discussion on practical aspects makes it valuable to the recommender systems community.
On the other hand, the numerical experiments on execution time are limited to small-scale settings (e.g., $B=3$), and the method does not appear to be faster than existing methods (Tables 10-14).
Therefore, the superiority of the proposed method is not sufficiently demonstrated.

### Strengths
1. This paper addresses a highly complex yet practical problem: incorporating diversity in a setting where the item set/batch is recommended sequentially to a user.
2. This paper provides comprehensive discussions covering both theory and implementation.

### Weaknesses
1. The experimental results lack persuasiveness. Specifically, the paucity of comparisons regarding execution time with existing methods undermines the paper's main claim of scalability.
2. Although the paper uses theoretical notation, its theoretical contribution is limited. For example, it mentions regret but does not discuss an algorithmic regret bound or similar rigorous analysis.
3. The assumption of noiseless feedback (Assumption 3.4) appears highly unrealistic. In the context of recommender systems, there are few practical scenarios where the expected reward can be directly observed.
4. In my opinion, addressing the relevance-diversity trade-off is a means to an end, and the authors should have prioritized a proper problem formulation. Framing the problem using a model like rotting bandits [a] might have enabled a more robust discussion of concepts such as regret.


[a] Rotting bandits, N Levine, K Crammer, S Mannor - Advances in neural information processing systems, 2017

### Questions
1. What were the results of the numerical experiments on execution time when varying the batch size?
2. How is the proposed method intended to be executed in scenarios where the expected reward cannot be observed, such as with binary feedback (e.g., clicks)?

### Soundness
2

### Presentation
2

### Contribution
2
