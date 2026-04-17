# CoNRec: Context-Discerning Negative Recommendation with LLMs

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Understanding what users like is relatively straightforward; understanding what users dislike, however, remains a challenging and underexplored problem. Research into users' negative preferences has gained increasing importance in modern recommendation systems. Numerous platforms have introduced explicit negative feedback mechanisms and leverage such signals to refine their recommendation models. Beyond traditional business metrics, user experience-driven metrics, such as negative feedback rates, have become critical indicators for evaluating system performance. However, most existing approaches primarily use negative feedback as an auxiliary signal to enhance positive recommendations, paying little attention to directly modeling negative interests, which can be highly valuable in offline applications. Moreover, due to the inherent sparsity of negative feedback data, models often suffer from context understanding biases induced by positive feedback dominance. To address these challenges, we propose the first large language model (LLM) framework for negative feedback modeling with special designed context-discerning modules. We use hierarchical semantic ID Representation to replaces text-based item descriptions and introduce an item-level alignment task that enhances the LLM’s understanding of the semantic context behind negative feedback. Furthermore, we design a Progressive Group Relative Policy Optimization (GRPO) training paradigm that enables the model to dynamically balance the positive and negative behavioral context utilization. Besides, our investigation further reveals a fundamental misalignment between the conventional next-negative-item prediction objective and users’ true negative preferences, which is heavily influenced by the system’s recommendation order. To mitigate this, we propose a novel reward function and evaluation metric grounded in multi-day future negative feedback and their collaborative signals. Extensive experiments on a real-world industry-scale dataset from Taobao demonstrate that our method achieves state-of-the-art performance. Our work offers meaningful insights not only for the emerging field of negative feedback modeling but also for the broader recommendation community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoNRec, an offline candidate filtering framework for modeling user negative preferences in recommender systems. The method represents items using discrete hierarchical semantic codes learned through an RQ-VAE, enabling a large language model to generate semantic categories of items the user is likely to dislike in the future. It further applies a contrastive item-level alignment mechanism to distinguish “liked vs. disliked” items, and introduces a progressive GRPO reinforcement training strategy in which aggregated negative feedback over the next 7 days serves as a more stable supervision signal, while predictions similar to future positive feedback are penalized. The generated semantic codes are decoded back into embeddings and used to filter candidate items by similarity. Experiments on real-world Taobao datasets show improvements, particularly for long-tail users and items.

### Strengths
1.	The motivation is clearly grounded in real industrial needs, addressing limitations of rule-based filtering and single-instance negative feedback.
2.	The method shows stronger performance for long-tail users and long-tail items, indicating enhanced robustness under sparse feedback conditions.
3.	The use of a 7-day aggregated negative feedback signal is data-driven and empirically justified, improving the stability of negative preference supervision.

### Weaknesses
1.	The core reliance on RQ-VAE semantic coding lacks validation regarding stability, interpretability, and semantic consistency. Critical configuration and robustness analyses are missing.
2.	The contrastive alignment module may not ensure true separation of “liked vs. disliked” semantics and may instead capture co-purchase or exposure patterns. No embedding visualization or case analysis supports its claimed effect.
3.	The progressive GRPO strategy is heuristic, lacking theoretical grounding and sensitivity studies; performance gains may stem from tuning rather than the designed mechanism.
4.	The filtering mechanism may harm recommendation diversity and suppress potential interests, yet no analyses of filtering rate, recall drop, diversity, or coverage are provided.
5.	Implementation and efficiency details are insufficient. Model size, training cost, inference latency, and deployment overhead are not reported, making scalability and reproducibility uncertain.

### Questions
1.	Can you provide evidence that the RQ-VAE semantic tokens remain stable under small perturbations, and demonstrate semantic clustering or temporal consistency? Also, what is the measured false filtering rate?
2.	Can you show embedding visualizations or fine-grained category case studies to confirm that the contrastive alignment captures true negative preference semantics rather than co-purchase or exposure bias?
3.	How were the phase transitions and reward weights in the progressive GRPO training strategy determined? Have you conducted sensitivity analysis on window sizes and stage scheduling?

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
Modeling users’ negative preferences has become increasingly important in modern recommender systems. This paper introduces **CoNRec**, the first large language model (LLM)-based framework designed to understand users’ negative preferences. To address the performance degradation caused by additional contextual information, CoNRec incorporates progressive GRPO training with a curriculum learning strategy, which gradually enhances the historical information included in the prompts. Moreover, CoNRec extends the ground-truth definition from the next negative feedback to the next items within seven days, and further expands the set of negative items by incorporating highly co-interacted items related to users’ actual negative feedback. These strategies improve the alignment between negative feedback and users’ negative interests. In addition, CoNRec adapts reward shaping to more precisely capture users’ negative preferences. Empirical results demonstrate the superiority of CoNRec, and ablation studies confirm the effectiveness of both progressive GRPO and the customized reward-shaping mechanism.

### Strengths
- **S1: Intuitive Method.** The designs of progressive GRPO, ground-truth extension, and reward shaping are intuitive and well-motivated. These components enhance the consistency between negative items and users’ negative interests, offering valuable insights for future research on negative preference modeling.
- **S2: Comprehensive Analysis**. The ablation and analysis studies thoroughly validate the effectiveness of each component and investigate the impact of different reward formulations on CoNRec. The forgetting rate analysis further shows reduced noise and improved stability in the target task. 
- **S3: High Generality.** CoNRec demonstrates strong generality across backbone models of different architectures and sizes, as evidenced by the results in Table 5.

### Weaknesses
- **W1: Lack of Datasets.** The paper evaluates CoNRec on only one dataset, which is insufficient to convincingly demonstrate the model’s effectiveness. Incorporating additional datasets would strengthen the validity of the conclusions.
- **W2: Reproducibility.** The paper does not provide source code, which affects the reproducibility and transparency of the results. It is recommended that the authors release the corresponding code to enhance credibility and facilitate future research.

### Questions
- **Q1: User Negative Interest Acquisition.** The process of obtaining users’ negative interests is not clearly described. Providing more details on how these negative interests are identified or constructed would make the experimental setting clearer and more reproducible.

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
This paper introduces large language model framework for negative feedback modeling (CoNRec), which leverages large language models to model user dispreferences. It first utilizes hierarchical semantic ID Representation and item-level alignment task that enhances the LLM’s understanding of the semantic context behind the item descriptions of negative feedback. Then, it designs a Progressive GRPO training paradigm that enables the model to dynamically balance the positive and negative behavioral context utilization. Experiments on a real-world industry-scale dataset demonstrates its effectiveness.

### Strengths
1. The motivation of CoNRec is interesting and the preliminary motivation study is clear in the misalignment between user negative interest and next feedback item, as well as the performance drop with extra context.
2. The proposed method is easy to follow and the corresponding figures are easy to understand.
3. This paper defines a new task scenario as Negative Recommendation and utilize several evaluation metrics tailored for this scenatio, such as FHR@20, LUF@20 and LIF@20. 
4.  The author clearly admites the drawbacks for users lacking negative feedback.

### Weaknesses
1. The technical contribution is limited and the proposed method is not very novel, the utilized RQ-VAE, LoRA and GRPO methods are widely utilized in many recommendation studies.
2. CoNRec leverages the explicit negative user feedback, whereas most of the baselines mainly rely on users’ historical interactions. This may lead to an unfair comparison. The authors should implement several baselines that also model negative feedback, or incorporate negative feedback into existing baselines to ensure a fair comparison.
3. Lack of the source code and the corresponding dataset.
4. The proposed CoNRec is just verified in Real-world Taobao Dataset, its robustness and universality to other data distributions have not been sufficiently explored or validated.
5. Among the baselines used for comparison, only one is from 2024, while all the others are before 2023. The lack of comparison with the latest 2025 studies is unacceptable for the rapidly evolving LLM4Rec field.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
