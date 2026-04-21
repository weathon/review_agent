# Preference Diffusion for Recommendation

- Avg Score: 5.75
- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Recommender systems aim to predict personalized item rankings by modeling user preference distributions derived from historical behavior data. While diffusion models (DMs) have recently gained attention for their ability to model complex distributions, current DM-based recommenders typically rely on traditional objectives such as mean squared error (MSE) or standard recommendation objectives. These approaches are either suboptimal for personalized ranking tasks or fail to exploit the full generative potential of DMs. To address these limitations, we propose \textbf{PreferDiff}, an optimization objective tailored for DM-based recommenders. PreferDiff reformulates the traditional Bayesian Personalized Ranking (BPR) objective into a log-likelihood generative framework, enabling it to effectively capture user preferences by integrating multiple negative samples. To handle the intractability, we employ variational inference, minimizing the variational upper bound. Furthermore, we replace MSE with cosine error to improve alignment with recommendation tasks, and we balance generative learning and preference modeling to enhance the training stability of DMs. PreferDiff devises three appealing properties. First, it is the first personalized ranking loss designed specifically for DM-based recommenders. Second, it improves ranking performance and accelerates convergence by effectively addressing hard negatives. Third, we establish its theoretical connection to Direct Preference Optimization (DPO), demonstrating its potential to align user preferences within a generative modeling framework. Extensive experiments across six benchmarks validate PreferDiff's superior recommendation performance. 
Our codes are available at \url{https://github.com/lswhim/PreferDiff}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel recommendation approach, PreferDiff, that optimizes diffusion models specifically for personalized ranking. By reformulating BPR as a log-likelihood objective and incorporating multiple negative samples, PreferDiff enhances the model’s ability to capture nuanced user preferences and better distinguish between positive and negative interactions. The authors demonstrate PreferDiff’s effectiveness through extensive evaluations, showing improved performance in sequential recommendation tasks, faster convergence, and superior handling of complex user preference distributions.

### Strengths
1. The paper introduces a unique objective tailored for DM-based recommenders by transforming BPR into a log-likelihood ranking format. This innovation effectively uses the generative capacity of DMs in recommendation tasks. 

2. The model incorporates a balance between generative modeling and preference learning, which enhances stability and performance.

3. The model's gradient-weighted approach to negative samples allows it to focus on challenging cases where negative samples are mistakenly ranked high. This focus on "hard negatives" is particularly valuable in recommendation.

### Weaknesses
1. PreferDiff can be very slow in both the training and inference stages. It would be useful if the authors could show the efficiency-effectiveness comparison between PreferDiff and other baselines in a 2-D figure.

2. As shown in Appendix D, the maximum length of interaction history is small (i.e., 10 or 20) following prior works, however, as we know that users may generally have a much longer interaction history. Is this a general limitation of DM-based recommenders that they are too expensive to train on larger sequences? What would PreferDiff perform if training with a longer sequence?

### Questions
Could you share more details on why high embedding dimensions are necessary for PreferDiff’s performance? Have you experimented with any regularization techniques or embedding pruning methods to mitigate this dependence?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces PreferDiff, a model designed to capture user preferences using diffusion model-based recommenders. PreferDiff primarily focuses on enhancing loss functions. The core contributions are twofold: First, the authors incorporate negative samples into the model training process through the use of ranking loss. Second, they transform the loss function in DM-based methods from rating estimation of samples to distribution estimation, formulated as the L_{BPR-Diff} loss, which is approximated by an upper bound. Experiments are conducted across multiple dataset settings to validate the approach.

### Strengths
1. The paper is well-written and easy to follow. It is well-structured and provides a detailed description of the experimental setup, which will aid the community in reimplementation.
2. The authors conducted experiments under various settings to validate the effectiveness of their method.

### Weaknesses
1. The evaluation difference between the full-ranking approach and the leave-one-out method is not clearly described. There is no mention of a full-ranking approach in [1]. The 'leave-one-out' evaluation setting is the most mainstream approach. The authors need to provide a justification for using the full-ranking manner as the primary evaluation setting in the main text. Additionally, do these two evaluation methods affect the ranking of different approaches?
2. Performance of other recommenders in Tab.1 is confusing. For TIGER, the performance in Tab.1 is much lower than the performance reported by the original paper. The authors should explain the reasons behind these performance gap. One reason maybe contribute to the performance gap is the ID quantifer. Authors take the PQcode of VQREC[2] as the ID quantifier instead of the RQVAE used in TIGER[3]。However, the authors should clearly specify whether these performance metrics are reproduced results or those reported in the original paper. For LLM2BERT4Rec, the performance on Beauty dataset is also inconsistent with the performance in original paper.
3. The explanation in Section 3.2 is somewhat unconvincing. L_{BPR-Diff} is essentially a ranking loss in metric learning.
4. For generative models, timestamps is crucial. The authors should conduct an ablation study on timestamps to evaluate their impact on the generation of positive and negative samples. Additionally, given that generative models are typically time-consuming, it would be beneficial to assess how PreferDiff performs in terms of inference speed compared to other sequence models. Specifically, can it achieve a favorable trade-off between hit rate/NDCG and speed?

[1] Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion.
[2] Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders
[3] Recommender Systems with Generative Retrieval

### Questions
see in Weaknesses. 
I am open to revising my score based on further clarifications or additional information provided by the authors during the rebuttal process.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents PreferDiff, a novel optimization objective designed specifically for diffusion model (DM)-based recommenders in sequential recommendation tasks. PreferDiff aims to address the limitations of traditional objectives by incorporating a log-likelihood ranking approach, allowing the model to better capture user preferences through the integration of multiple negative samples. Extensive experiments demonstrate that PreferDiff outperforms baseline models in both performance and convergence speed, validating its effectiveness in modeling complex preference distributions.

### Strengths
- Innovative Objective Design: PreferDiff introduces a log-likelihood ranking objective tailored to diffusion models, marking a significant advancement in personalized ranking for DM-based recommenders.
- Comprehensive Negative Sampling: The incorporation of multiple negative samples enhances user preference understanding, leading to better separation between positive and negative items and improved recommendation performance.
-  Effective Performance and Convergence: PreferDiff demonstrates superior recommendation accuracy and faster convergence compared to existing models, showcasing both theoretical and practical value.

### Weaknesses
1.Limited Originality: The formulation of PreferDiff shows considerable overlap with Direct Preference Optimization (DPO), as several of its mathematical expressions and objective functions appear directly inspired or derived from DPO's original framework​. This raises concerns about the novelty of PreferDiff's contribution to preference learning within diffusion models, as the paper does not introduce substantial modifications or unique approaches that deviate meaningfully from DPO's foundational equations.


2.Dependency on High Embedding: PreferDiff’s performance is highly dependent on large embedding sizes, which may limit its scalability and increase computational costs.

### Questions
1. How does PreferDiff handle real-time recommendation scenarios where embedding dimensions need to be minimized due to latency constraints?
2. Could the authors provide more insights into the optimal range for the hyperparameter \lamda, especially in varied recommendation domains?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents PreferDiff, an improved diffusion based recommendation model. The key contribution is the incorporation of an adapted  BPR objective and in turn jointly optimize the traditional MSE objective together with this new ranking loss. Variational method is employed to make the optimization tractable. The authors also show that the proposed approach is connected to DPO in formulation.
Experiments on Amazon review data set show that the proposed methods outperform a number of different baselines.

### Strengths
+ The intuition of incorporating ranking loss into the DM recommender makes sense. The development of the proposed method looks reasonable and (to my knowledge) correct.
+ The paper is clearly presented and easy to follow.
+ Experiments and ablation studies are mostly solid.

### Weaknesses
- There is essentially only 1 data set being used (amazon review), no matter how many categories you include, this data set may not be representative enough which may raise concerns regarding the generalizability of your findings 
- Some of the questions remain unanswered (or observations without explanation) , e.g,: 1) what caused PreferDiff to be faster than DreamRec? 2) why diffusion models are more sensitive to d_model? 
- Novelty seems to be minimum, the overall approach makes sense but is also straightforward. The connection to DPO is rather weak and the claim of this as a theoretical contribution (1 of the 3 contributions) is not very sound.

### Questions
1. Consider use a few other data sets, especially data sets with diverse background (e.g, Yahoo!, Criteo)
2. At least some efforts should be made to explain unexpected observations, e.g, e.g,: 1) what caused PreferDiff to be faster than DreamRec? 2) why diffusion models are more sensitive to d_model? 
3. The connection to DPO is rather weak and the claim of this as a theoretical contribution (1 of the 3 contributions) is not very sound.
4. Eq(12) added the MSE loss back to the formula, the authors claimed that this is to mitigate the learning stability issues, it would be interesting to the readers if the authors could report that instability observations directly. It would also be worthy looking into this instability issue to root-cause it. Since PreferDiff converges faster, would this indicate that ranking loss itself might be more stable than the MSE loss and could it be possible that there are other ways to mitigate the instability issue without taking the hybrid path?

### Soundness
3

### Presentation
3

### Contribution
2
