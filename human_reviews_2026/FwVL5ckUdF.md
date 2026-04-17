# A General Theoretical Paradigm to Understand Two Tower Recommendation Models

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Production-grade recommender systems rely heavily on a large-scale corpus used by online media services, including Netflix, Pinterest, and Amazon. These systems enrich recommendations by learning users' and items' embeddings projected in a low-dimensional space with two tower models (two deep neural networks), which facilitate their embedding constructs to predict users' feedback associated with items. Despite its popularity for recommendations, its theoretical behaviors remain comprehensively unexplored. We study the asymptotic behaviors of the two tower model applied in two-stage recommenders that entail a strong convergence to the optimal recommender system. We establish certain theoretical properties and statistical assurance of the two tower recommender. In addition to asymptotic behaviors, we demonstrate that recommendation with two tower architecture attains faster convergence by relying on the intrinsic dimensions of the input features. Finally, we show numerically that the two tower recommender enables encapsulating the impacts of items' and users' attributes on ratings, resulting in better performance compared to existing methods conducted using synthetic and real-world data experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a theoretical study of the two-tower recommender system model. The authors analyze the asymptotic behaviors of the model, focusing on its convergence to the optimal recommender system. They establish theoretical properties, including the relationship between the convergence rate and the inherent dimensionality of the input features. The study demonstrates that the two-tower model converges faster as the smoothness of the true model increases or the intrinsic dimensionality decreases.

### Strengths
1. The paper provides a theoretical analysis perspective of the two-tower recommender model, establishing key properties and convergence behaviors.
2. The paper highlights the two-tower model's ability to address challenges like cold-start problems, offering insights into its effectiveness in real-world recommendation scenarios with both synthetic and real-world data.
3. The paper introduces a novel approach to quantifying the convergence speed and robustness of two-tower models, contributing new statistical guarantees to the understanding of their performance.

### Weaknesses
1. Does the convergence speed of the two-tower model depend on the specific architecture of the model itself?
2. Convergence speed is influenced by various factors such as data, model parameters, and training time. Why is the inherent dimensionality of user and item features the primary focus in this study?
3. How does the convergence speed of the two-tower model compare to other deep learning models or LLM-based models in terms of convergence rate? What factors can demonstrate that the two-tower model converges faster?

### Questions
The citation format seems to mix content with references, which affects readability.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a theoretical framework to analyze the asymptotic behavior of two-tower recommendation models commonly used in large-scale recommender systems. The authors prove that under certain smoothness and boundedness assumptions, the two-tower model converges to the optimal recommender system and achieve explicit convergence rates in terms of approximation and estimation errors. The convergence rate depends on the Hölder smoothness of user/item embedding functions and the intrinsic dimensions of input features. Additionally, the paper complements theoretical results with synthetic and real-world experiments to demonstrate the performance and convergence trends of two-tower models.

### Strengths
1. This paper provides a new perspective to analyze the approximation and convergence properties of two-tower models.

2. The paper derives explicit convergence rates that depend on the smoothness of the target function and the intrinsic dimensionality of user and item feature spaces. 

3. The inclusion of empirical validation on synthetic and real-world datasets helps connect theoretical results to practical scenarios such as cold-start recommendation. The results show consistency with derived asymptotic behaviors.

### Weaknesses
1. Limited perceived novelty for core ICLR audience: Two-tower recommendation models, while impactful in practice, belong to a niche within recommender systems and information retrieval. The main contribution is theoretical interpretation rather than a new learning paradigm. As a result, its impact may be underestimated by ICLR community than data mining venues.

2. Mathematical presentation issues: It's interesting that some simple concepts (e.g., l2-norm in line 110) are formally defined while more complex objects like $\epsilon$-balls, $ |\cdot |\_0 $ in line 111 and 123 are used without definition or intuition; notation is occasionally introduced after first usage (e.g., $k_{ui}$ appears in Eq.(2) but is defined in Eq.(3)), which shows structural inconsistencies; Several typos exist: e.g. $B_1$ in line 114, improper capitalization at line 149; overly complex notation with poor layout: lines 180–190 use highly dense notation and problematic formatting to describe what is essentially a standard gradient descent update. This creates unnecessary cognitive burden and reduces readability—even for technically strong readers.

3. Misinterpretation of ranking metrics: The paper claims to provide theoretical guarantees for “ranking objectives” in 4.2.1, yet the analysis and experiments are primarily based on Top-k accuracy. However, Top-k metrics are classification-style metrics (whether the ground-truth item appears in the top k), rather than a true ranking metric. In recommender systems, ranking objectives more commonly refer to metrics that account for ordered relevance, such as NDCG, MRR, or MAP. Therefore, the paper’s claim of analyzing “ranking guarantees” is somewhat misleading, as it does not address position-sensitive ranking quality. This mismatch between theoretical objective and commonly accepted ranking metrics undermines the applicability of the theoretical results.

4. Experiments insufficiently tied to core theory: The experiments illustrate the practical usefulness of the two-tower model but do not rigorously validate the core theoretical claims. In particular, no experiments quantify the effect of intrinsic dimension or smoothness on convergence, and key ablation/sensitivity analyses are missing. Hence, while functional, the experimental section lacks depth for an ICLR-level theoretical paper.

### Questions
Please refer to Weakness. The reviewer strongly recommends that the authors substantially improve the presentation and clarity of the theoretical exposition — including notation consistency, definition of symbols, and mathematical layout — otherwise the theoretical contributions may not be fairly assessed by the community.

### Soundness
2

### Presentation
1

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
This paper focuses on the theoretical foundation of the two-tower model. It proves that the two-tower model’s approximation error and convergence are related to the intrinsic dimension and smoothness, suggesting its superiority in handling complex functions. Additionally, the theorem on ranking suggests that a pairwise loss can lead to good performance on a Top-K ranking task.

### Strengths
1. The paper provides solid theoretical proofs that are applied specifically to the two-tower model, which has been lacking this kind of formal backing .

2. It proves the model's advantage comes from its performance scaling with the data's low intrinsic dimension, not its high nominal dimension. This explains why it works so well on sparse, high-dimensional data.

### Weaknesses
1. The theoretical guarantees offer few actionable guidelines for real-world tasks. The key factors identified intrinsic dimension and smoothness are inherent properties of the *data*, not model hyperparameters that an engineer can easily measure or change.
2. The experiments demonstrate that the two-tower model is superior to baselines like SVD++ and KNN. This are more likely to confirm a known result rather than to provide novel insights.

### Questions
1. The synthetic experiments do not test different nominal dimensions. What is the effect of this?

2. What is the relationship shown with the rating matrix sizes in the synthetic experiments? With different intrinsic dimensions, the results show different trends. What is the cause?

3. What is the relationship between the theory and the cold-start problem? My understanding is that the cold-start capability is mainly based on the idea of covariates, not the theorems.
4. What is the meaning of the shading in the experiment tables? It seems some better results from other models are not correctly shaded.

### Soundness
2

### Presentation
2

### Contribution
2
