# Gradient-Based Diversity Optimization with Differentiable Top-$k$ Objective

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Predicting relevance is a pervasive problem across digital platforms, covering social media, entertainment, and commerce. However, when optimized solely for relevance and engagement, many machine-learning models amplify data biases and produce homogeneous outputs, reinforcing filter bubbles and content uniformity. To address this issue, we introduce a pairwise top-k diversity objective with a differentiable smooth-ranking approximation, providing a model-agnostic way to incorporate diversity optimization directly into standard gradient-based learning. Building on this objective, we cast relevance and diversity as a joint optimization problem, we analyze the resulting gradient trade-offs, and propose two complementary strategies: direct optimization, which modifies the learning objective, and indirect optimization, which reweights training data. Both strategies can be applied either when training models from scratch or when fine-tuning existing relevance-optimized models. We use recommendation as a natural evaluation setting where scalability and diversity are critical, and show through extensive experiments that our methods consistently improve diversity with negligible accuracy loss. Notably, fine-tuning with our objective is especially efficient, requiring only a few gradient steps to encode diversity at scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on relevance-diversity optimization in recommender systems. Specifically, it proposes a differentiable top-$k$ diversity reward objective (DDRO) that can be integrated into RS training in a model-agnostic and loss-agnostic manner. To achieve this, the authors introduce the softrank technique to approximate the non-differentiable sorting operation, and derive a soft relaxation of the top-$k$ diversity reward (DRO), which is defined as the average pairwise distance dissimilarity among the top-$k$ recommended items. Furthermore, they borrow the idea of MGDA for multi-objective optimization to balance relevance and diversity during training. The proposed method is evaluated in both training-from-scratch and fine-tuning settings on various datasets and two recommendation backbones, exhibiting improvements in both relevance and diversity.

### Strengths
- This paper introduces a differentiable top-$k$ diversity objective by leveraging the softrank technique, facilitating end-to-end optimization.
- This paper performs relevance-diversity optimization in a joint manner with theoretical guarantees.
- The proposed method is evaluated with comprehensive experiments on multiple datasets and recommendation backbones, demonstrating its effectiveness in improving both relevance and diversity.

### Weaknesses
- The evaluation metrics used in the main results are not widely adopted in the previous work, making it difficult to assess the recommendation quality.
- The experiments only consider the MSE loss, which may limit the generalizability of the proposed method.
- The choice of top-$k$ approximation could be further justified by comparing with other differentiable top-$k$ operators.

### Questions
- **Evaluation Metrics.** In the main results (Tables 2, 3, and 5, Figures 1 and 3), the reported revelance metric is defined as the proportion of recommended items whose likelihood is greater than a certain threshold. However, this metric is not commonly used in the recommender systems literature --- it merely measures the sharpness of the predicted probability distribution, which may not directly reflect the recommendation quality. Although Figure 4 presents Recall, MSE, HR, and Precision results, the baseline methods are not compared. Thus, I suggest the authors include more standard recommendation metrics (e.g., Recall@$k$ and NDCG@$k$) in the main results for fair comparison.
- **Recommendation Losses.** It seems that the proposed DDRO can be integrated with any recommendation loss. Nonetheless, the experiments only consider the MSE loss, which has been shown to be suboptimal for recommendation tasks. It would be more convincing if the authors could evaluate DDRO with other widely used recommendation losses, such as AUC-oriented losses (e.g., BPR [R1]), ranking-oriented losses (e.g., PSL [R2]), and top-$k$-oriented losses (e.g., LambdaLoss@$k$ [R3] and SL@$k$ [R4]), to see whether the improvements still hold on both accuracy and diversity compared to these SOTA baselines.
- **Top-$k$ Approximation.** The authors employ the top-$k$ soft indicator based on the softrank technique to handle the non-differentiability. Recently, there are several works on differentiable top-$k$ operators that provide an alternative quantile-based approximation (e.g., SL@$k$ [R4] and SONG@$k$ [R5]). It would be beneficial to compare these approaches with the approximation used in this paper, both theoretically and empirically.
- **Minor Comments.** The proof of Lemma 3 seems to be generated by LLMs (although no errors were found). The proof is in fact quite straightforward, so I suggest the authors simplify it to make the presentation more concise.

**References:**
- [R1] BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI '09.
- [R2] PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation. NIPS '24.
- [R3] On Optimizing Top-K Metrics for Neural Ranking Models. SIGIR '22.
- [R4] Breaking the Top-K Barrier: Advancing Top-K Ranking Metrics Optimization in Recommender Systems. KDD '25.
- [R5] Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence. ICML '22.

### Soundness
2

### Presentation
2

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
This paper addresses the issue that existing recommendation models often optimize solely for relevance and user engagement, which may amplify data bias and reduce the diversity of recommended results. To tackle this, the authors propose a general pairwise Top-K diversity optimization framework that formulates relevance and diversity as a joint optimization problem. The approach achieves diversity enhancement by modifying the learning objective function and introducing a reweighting mechanism.

### Strengths
1.The authors propose Direct Diversity-Guided Tuning (DDT), which augments the original loss function by incorporating a relevance–diversity joint term.

2.The authors further introduce Meta-Diversity Reweighting (MDR), a mechanism that retains the conventional relevance-based training pipeline while adopting the joint loss as a meta-objective to reweight training samples dynamically.

3.Rich experimental results

### Weaknesses
1.Readability needs improvement. Although the paper includes a series of lemmas and proofs to validate the proposed method, the workflow of DDT remains abstract and difficult to follow. It is recommended to include a model architecture or workflow diagram for DDT and MDR to make the methodology more comprehensible.

2.The rationale of meta-learning-based reweighting requires further clarification. In Section 4.2, the authors employ meta-learning to adjust sample weights, yet it is not clear how this process ensures the preservation of relevance. Specifically, the computation of the weight w_{u,i} is not described in detail (it seems to be obtained via an MLP). If this is the case, additional explanation is needed to justify why using an MLP-based reweighting does not compromise the relevance term in Equation (8).

3.Implementation details of DDT are insufficient. The loss formulation of DDT is not presented in Section 4.1 but rather briefly mentioned in Section 3, which is not reader-friendly. Furthermore, Algorithm 1 only states “compute DDRO with (5)” without further elaboration. I hope the author can describe the practical details of the DDT method in section 4.1.

If the author can resolve my doubts, I am willing to raise my score.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a model-agnostic, differentiable top-k diversity objective for ranking. It relaxes the non-differentiable top-k set via soft ranking/permutahedron projection and couples it with a relevance loss, optimizing a joint objective either Direct Diversity Tuning and Meta-Diversity Reweighting. Experiments on five datasets (MovieLens, Netflix, Yahoo-R2, Coat, KuaiRec) with MF/NCF models show large diversity gains with minimal relevance drops.

### Strengths
1. The paper proposes a differentiable top-k diversity via soft rankings.
2. The MGDA-based adaptive parameter has a transparent feasible region and a closed-form. It guarantees convergence to Pareto-stationary points under standard conditions.
3. Two complementary routes (DDT vs. MDR) cover both objective-level and data-level interventions.

### Weaknesses
1. The DDRO approximation error depends on $\epsilon$ and k. For larger k, the method needs a smaller $\epsilon$ to be accurate, which may affect stability/cost. The paper notes this, but practical guidance/ablation across datasets is limited.

2. While MDR reduces bias via reweighting, classic recommendation confounders (position bias, missing-not-at-random feedback) aren’t explicitly modeled.

### Questions
1. How sensitive are results to the choice of the item similarity matrix (e.g., Jaccard on genres vs. learned embedding cosine vs. taxonomy distance)?

2. Any plans for counterfactual or bandit-style evaluation to account for feedback loops and exposure bias?

3. Does optimizing DDRO measurably increase long-tail exposure?

### Soundness
3

### Presentation
3

### Contribution
3
