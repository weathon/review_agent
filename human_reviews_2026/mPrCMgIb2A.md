# Guaranteed Top-Adaptive-K in Recommendation

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Recommender systems (RS) are crucial in offering personalized suggestions tailored to user preferences. While conventionally, Top-\(K\) recommendation approach is widely adopted, its reliance on fixed recommendation sizes overlooks the diverse needs of users, leading to some relevant items not being recommended or vice versa. While recent work has made progress, they determine \(K\) by searching over all possible recommendation sizes for each user during inference. In real-world scenarios, with large datasets and numerous users with diverse and extensive preferences, this process becomes computationally impractical. Moreover, there is no theoretical guarantee of improved performance with the personalized K. In this paper, we propose a novel framework, **K-Adapt**, which determines dynamic K-prediction set size for each user efficiently and effectively. Specifically, it reformulates adaptive Top-\(K\) recommendation as a utility-based risk control problem, where a calibrated threshold based on user utility metrics determines the prediction sets. A lightweight greedy optimization algorithm efficiently learns this threshold to generate dynamic recommendations. Theoretical analysis is provided by establishing upper bounds on expected risk as well as near-optimality and stability of the learned threshold. Extensive experiments on multiple datasets demonstrate that the K-Adapt framework outperforms baseline methods in both performance and time efficiency, offering a guaranteed solution to fixed Top-\(K\) challenges.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes K-Adapt, a model-agnostic adaptive framework for recommender systems that automatically chooses recommendation size per user via a calibrated score threshold $\lambda$. It formulates the problem as utility-based risk control, extends conformal prediction to ranking metrics, and uses a greedy calibration algorithm. Theoretical analysis provides theoretical guarantees, near-optimality, and perturbation-stability of the proposed method. Experiments on three datasets and five backbones show improvements over selected dynamic-K baselines.

### Strengths
- This paper studies the adaptive recommendation size, which is an important and practical problem in RS.
- Theoretical analysis is provided for the proposed method.
- The experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
- The motivation for the proposed adaptive threshold $\lambda$ is not convincing due to the trivial solutions.
- Some critical assumptions in the theoretical analysis are not reasonable.
- The write-up and mathematical notations can be improved for better clarity.

### Questions
**Concerns about Motivation.** The proposed adaptive threshold $\lambda$ for restricting recommendations is not convincingly motivated:  
- The definition of the optimal $\lambda^*$ assumes that $R(\lambda)$ is non-decreasing (equivalently $U(\lambda)$ is non-increasing), as also stated in Theorem 2. This holds for some metrics (e.g., Recall, NDCG) but fails for others, such as Precision, original F1-score (not the one defined in Appendix A.2), etc. This limits the applicability of the proposed method in real-world RS, and the authors should provide further discussion on this point.
- Even when the assumption holds, minimizing $\lambda$ trivially minimizes $R(\lambda)$. For instance, $\lambda \to -\infty$ yields maximum Recall and NDCG by recommending all items. However, this trivial solution is meaningless in practice. The proposed greedy algorithm with $K_{\max}$ avoids the "all items" case but not the equally trivial "always recommend $K_{\max}$ items", since $\lambda$ only decreases. The authors should clarify how their method avoids such trivial solutions.

**Concerns about Theoretical Results.** Some critical assumptions in the theoretical results are not convincing, which needs to be addressed:
- As stated above, the monotonicity assumption on $R(\lambda)$ may not hold, which invalidates Theorem 2 and subsequent results.
- Theorems 2 and 3 also assume that there exists a positive constant $c$ such that for any $\lambda \ge \lambda^\star$, $R(\lambda) - R(\lambda^\star) \geq c(\lambda - \lambda^\star)$. This assumption is obviously not true in general for almost all metrics due to their discrete nature and stepwise changes.

**Discussion on Top-$K$ Recommendation Optimization.** Recent literature has extensively studied top-$K$ recommendation optimization, including LambdaLoss@$K$ [R1], SONG@$K$ [R2], LLPAUC [R3], and SL@$K$ [R4]. These methods optimize fixed top-$K$ metrics with a fixed $K$, achieving strong empirical performance and theoretical guarantees. Since the proposed K-Adapt method is model-agnostic and can be applied only at inference time, it would be interesting to see whether it can further improve the performance of these top-$K$ recommendation losses.

**Minor Concerns:**

- Theorem 1: What is the meaning of $\lambda^*$ in the proof? Should it be $\hat\lambda$? In addition, it seems that the deviation $\delta(\epsilon)$ is in fact not dependent on $\epsilon$.

**References:**

- [R1] On Optimizing Top-K Metrics for Neural Ranking Models. SIGIR '22.
- [R2] Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence. ICML '22.
- [R3] Lower-Left Partial AUC: An Effective and Efficient Optimization Metric for Recommendation. WWW '24.
- [R4] Breaking the Top-K Barrier: Advancing Top-K Ranking Metrics Optimization in Recommender Systems. KDD '25.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is about controlling the score threshold for recommendations.\
To do this, they propose K-adapt (GUARANTEED ADAPTIVE-K IN RECOMMENDATIONS).\
Specifically, based on the calibration dataset, they set the threshold for ensuring the pre-defined performance.\
Then, they use the threshold for the inference to control the number of recommendations for each user.

### Strengths
1. The paper is well formulated and easy to follow.
- The paper has a good structure and good citation format.
- Figures and Tables are well organized and presented.

2. Fast inference time.
- They use the pre-selected threshold for the inference time, which results in faster inference than existing methods.

3. Experiment on real-world datasets demonstrates the superiority of the proposed method.

### Weaknesses
1. The method should be described in a more detailed way.
- Algorithm 1 should be included in the main manuscript, not in Appendix.

2. Technical contribution is marginal.
- From my understanding, the method selects the threshold to ensure a certain performance in the calibration set.\
Then, it uses the threshold for the inference.
- Is the global threshold enough for all users?

3. Not guaranteed performance?
- In the manuscript and title, they noted that their method "guarantees" the performance.
- In Table 1, however, $\alpha=0.05$ and the performance is not 0.95.
- Also, in Figure 4, recall is almost zero when $\alpha=0.4$.

### Questions
Please refer to Weaknesses.\
Also, what is the purpose of Eq.5? I think both cases are the same.

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
2

### Summary
The paper introduces K-Adapt, a theoretical framework that dynamically determines the number of recommendations (K) for each user, rather than relying on a fixed K. Traditional Top-K recommender systems use the same list length for all users, ignoring individual differences and potentially reducing user satisfaction. In contrast, K-Adapt learns a calibrated threshold that defines each user’s personalized recommendation size with formal risk guarantees. Experiments on the MovieLens, Last.fm, and AmazonOffice datasets demonstrate the effectiveness and robustness of the proposed approach.

### Strengths
- The paper is well-written and easy to follow, with a clear structure and logical flow.

- The proposed framework is built on conformal prediction theory, providing formal statistical guarantees for adaptive-K recommendations, which is a strong and theoretically sound foundation.

### Weaknesses
- The experiments mainly focus on offline metrics, with no exploration of latency, real user feedback, or real-world deployment aspects.

- The framework’s performance potentially depends on the calibration data. If the data doesn’t represent real-world conditions well, or if things change over time, the guarantees might not hold up. It would be beneficial to do more analysis in terms of this apsect.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
