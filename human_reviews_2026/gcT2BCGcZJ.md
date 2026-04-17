# Steering Diffusion Models Towards Credible Content Recommendation

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
In recent years, diffusion models (DMs) have achieved remarkable success in recommender systems (RSs), owing to their strong capacity to model the complex distributions of item content and user behaviors. Despite their effectiveness, existing methods pose the danger of generating uncredible content recommendations (e.g., fake news, misinformation) that may significantly harm social well-being, as they primarily emphasize recommendation accuracy while neglecting the credibility of the recommended content. To address this issue, in this paper, we propose Disco, a novel method to steer diffusion models towards credible content recommendation. Specifically, we design a novel disentangled diffusion model to mitigate the harmful influence of uncredible content on the generation process while preserving high recommendation accuracy. This is achieved by reformulating the diffusion objective to encourage generation conditioned on preference-related signals while discouraging generation conditioned on uncredible content-related signals. In addition, to further improve the recommendation credibility, we design a progressively enhanced credible subspace projection that suppresses uncredible content by projecting diffusion targets into the null space of uncredible content. Extensive experiments on real-world datasets demonstrate the effectiveness of Disco in terms of both accurate and credible content recommendations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Disco, a diffusion model-based recommender system designed to mitigate uncredible content recommendations such as fake news and misinformation. The authors identify two factors causing existing diffusion models to generate uncredible recommendations: uncredible conditions from users' historical interactions with uncredible items, and uncredible diffusion targets when the target item itself is uncredible. Disco addresses these through three main contributions: a disentangled diffusion model that separates preference signals from uncredible content signals by jointly encouraging generation conditioned on preference-related embeddings while discouraging generation conditioned on uncredible content embeddings; a credible subspace projection module using SVD-based null-space projection to remove uncredible features from diffusion targets; and a progressive enhancement strategy to handle limited labeled data by iteratively detecting and incorporating potential uncredible items. Experiments on PolitiFact, GossipCop, and MHMisinfo datasets with only 20% labeled uncredible items demonstrate improvements in both recommendation accuracy and credibility metrics.

### Strengths
1. **Addresses a Critical Real-World Problem**. The paper tackles the important societal concern of recommender systems amplifying uncredible content, with well-motivated examples including COVID-19 misinformation spread (lines 59-62). The problem formulation is realistic, acknowledging that only partial credibility labels are available in practice (lines 135-138), which distinguishes this work from prior methods like Rec4Mit, HDInt, and PRISM that assume complete label availability. This practical constraint makes the research more applicable to real-world deployment scenarios.

2. **Strong Theoretical Foundations for the design**. The disentangled diffusion model uses the diffusion objective itself as a disentangler without auxiliary networks (lines 211-214), reducing computational overhead. The reformulation in Equation 4 that minimizes variational bound conditioned on preference while maximizing it conditioned on uncredible content is theoretically well-motivated. The credible subspace projection using SVD null-space decomposition (Equations 7-8) is grounded in prior null-space projection work, and Appendix C provides both empirical evidence (Table 6) and theoretical proofs demonstrating how uncredible conditions and targets enhance uncredible generation, strengthening the paper's technical contributions.

### Weaknesses
1. **Limited Experimental Scope and Insufficient Baseline Analysis**. The evaluation is restricted to three datasets, all in news/video domains (lines 337-343), limiting generalizability to other content types like e-commerce or music. More critically, the paper fixes the labeled data ratio at 20% (line 360) without ablation studies on different ratios (e.g., 10%, 15%, 30%, 50%), making it unclear how sensitive Disco is to label availability.

2. **Unclear Evaluation Protocol**. While the paper states "complete labels are provided during testing" (lines 137-138), it doesn't clarify whether uncredible items are included in the candidate pool during evaluation or how this impacts metrics like CR@K. The data augmentation strategy in Appendix B.1 (lines 772-776) transforms each user sequence into multiple sub-sequences, but doesn't specify how train/test splits prevent leakage when subsequences from the same user appear in both sets.

3. **Inadequate Justification and Analysis of results**. Table 2's ablation reveals that replacing cosine loss with MSE ("w/o CE") causes severe performance degradation (HR@5 drops from 0.2664 to 0.1034 on PolitiFact, line 439), attributed to training instability (lines 227-229), yet the paper provides no learning curves, gradient analysis, or convergence studies to characterize this instability. The weight parameter w in Equation 11 varies across datasets (w=0.5 for PolitiFact, w=1.5 for GossipCop, w=1 for MHMisinfo in Figure 4), suggesting dataset-dependent tuning requirements that are not adequately discussed.

### Questions
1. What characteristics of each dataset (e.g., proportion of uncredible items, interaction sparsity, content diversity) determine the optimal w? 
2. Can you provide guidelines or a principled approach for setting w on new datasets without exhaustive grid search? 
3. Given the data augmentation strategy that creates multiple sub-sequences from each user's interaction history (lines 772-776), how do you ensure no data leakage occurs when sub-sequences from the same user might span the train/test boundary?

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
This work primarily addresses the critical problem of untrustworthy content generation in recommendation systems. To mitigate this issue, the authors propose Disco, a framework designed to guide diffusion models toward generating credible content recommendations. The core idea is to reformulate the diffusion objective by incorporating preference-related signals that encourage credible generation, while simultaneously suppressing signals associated with untrustworthy or low-credibility content. Extensive experiments conducted on three benchmark datasets demonstrate the effectiveness of Disco.

### Strengths
1. The critical problem of untrustworthy content generation in recommendation systems is highly relevant to both the broader societal context and the ICLR research community, given its implications for information integrity, and user trust.
2. The authors focus on addressing the limitations of diffusion models in content recommendation and propose an effective framework for credible content generation, which presents an interesting to the field.
3. The authors provide sufficient experimental validation of the proposed Disco framework, including comparative evaluations, ablation studies, and hyperparameter analyses.
4. The authors provide sufficient mathematical proofs to support the theoretical soundness of the proposed Disco framework.

### Weaknesses
1.  While this work aims to advance credible content recommendation, the experimental results primarily focus on traditional accuracy-based metrics, without providing dedicated evaluations or metrics to assess credibility. It remains unclear how the Disco framework identifies uncredible content or ensures the generation of credible recommendations. The authors should further elaborate on (1) how credibility is defined and operationalized within their framework, (2) whether the model explicitly detects or filters uncredible content, and (3) which metrics or benchmarks are used to quantify credibility in the evaluation. 
2. Fake news is often defined based on human judgment, which inherently involves subjective and context-dependent factors. However, the Disco framework relies primarily on ID-based embeddings to identify uncredible content, which may lack the semantic richness and contextual understanding necessary to capture the nuanced nature of credibility. Consequently, the motivation for improving content credibility by adjusting the diffusion process solely based on ID-level signals is potentially limited and requires further theoretical justification and empirical validation.
3. The proposed improvement to the diffusion model appears to be incremental, primarily involving the addition of a disentanglement strategy. To better position the novelty of their work, the authors should provide a more detailed comparison with existing diffusion-based recommendation methods such as DreamRec and DiffuRec. Specifically, it is important to clarify how Disco differs in terms of model architecture, and objective formulation.
4. The motivation behind the use of null-space projection for filtering uncredible information is insufficiently explained and appears debatable. While the method is intended to suppress untrustworthy signals, the theoretical justification for why null-space projection effectively filters such content is not clearly articulated. Moreover, the authors do not provide any dedicated experiments to validate the effectiveness of this mechanism in isolating or removing uncredible information. 
5. The three datasets used in the experiments are relatively small and may not be sufficient to robustly validate the effectiveness and generalizability of the proposed model. To strengthen the empirical evaluation, the authors are encouraged to consider larger and more diverse datasets that better reflect real-world recommendation scenarios.

### Questions
1.  Authors should explain how the Disco framework identifies uncredible content or ensures the generation of credible recommendations. The authors should further elaborate on (1) how credibility is defined and operationalized within their framework, (2) whether the model explicitly detects or filters uncredible content, and (3) which metrics or benchmarks are used to quantify credibility in the evaluation. 
2. The authors should provide empirical evidence to support the claim that ID-based embeddings are capable of identifying uncredible content. Without such validation, it remains unclear whether these embeddings capture meaningful credibility-related signals. Additionally, a visualization demonstrating the consistency between the learned embeddings and the ground-truth credibility labels would help substantiate this claim. 
3. The proposed improvement to the diffusion model appears to be incremental, primarily involving the addition of a disentanglement strategy. The authors should provide a more detailed comparison with existing diffusion-based recommendation methods such as DreamRec and DiffuRec. 
4. The motivation behind the use of null-space projection for filtering uncredible information is insufficiently explained and appears debatable. 
5. The authors should consider larger and more diverse datasets to validate the model's effectiveness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Disco, a novel framework aimed at steering diffusion models (DMs) towards credible content recommendations. While DMs have proven effective in recommendation systems, they risk generating uncredible content such as fake news or misinformation. Disco addresses this issue by introducing a disentangled diffusion model that separates credible user preferences from uncredible content. This separation ensures that content generation is guided by preferences while discouraging the inclusion of uncredible signals. Additionally, Disco incorporates a progressively enhanced credible subspace projection, which suppresses uncredible content by projecting the diffusion targets into a null space that excludes uncredible features. The effectiveness of Disco is demonstrated through experiments on real-world datasets, showing that it delivers both accurate and credible recommendations.

### Strengths
1. Innovative Approach to Credibility: Disco offers a pioneering solution to mitigate uncredible content generation in DMs, which is crucial for real-world recommendation systems, particularly in sensitive areas like news recommendations.

2. Disentanglement of Credible and Uncredible Content: The disentangled diffusion model efficiently separates preference-related content from uncredible content, preserving the user’s genuine preferences while filtering out harmful signals.

### Weaknesses
1. **Paper Writing**: The formatting in Section 2.2 is quite poor, which makes it difficult to follow the explanation. Furthermore, the terms preference-aware embedding and uncredible content-aware embedding are introduced, but their definitions and distinctions aren't clear until Section 3.1. This lack of clarity is problematic, especially in the context of recommendation tasks or datasets, as readers are left unsure about what these embeddings represent and how they differ from each other until much later in the paper. Providing clearer definitions and explanations earlier in the paper would greatly improve the readability and understanding of these concepts.

2. **Novelty**: The Credible Subspace Projection seems to be inspired by AlphaFuse and other heuristic methods. The formulation in Eq (4), where the two conditions are directly subtracted, also appears heuristic. It's worth questioning why this approach of direct subtraction was chosen—why not explore other possible forms?

3. **Experiments**: All the experiments are conducted on datasets related to Fake News or misinformation videos. Has the author tried applying the method to other domains, such as Beauty or Sports? Expanding the dataset to include more diverse domains would help verify the method's effectiveness and generalizability.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Disco, a diffusion model-based recommendation model designed to mitigate uncredible content while maintaining recommendation accuracy. It identifies two key factors that cause existing diffusion models to generate uncredible recommendations: (1) uncredible conditions when users have interacted with uncredible items, and (2) uncredible diffusion targets when the target item itself is uncredible. Disco addresses these through a disentangled diffusion model that separates preference-related and uncredible content signals, combined with a progressively enhanced credible subspace projection that projects diffusion targets into the null space of uncredible content features.

### Strengths
1. Credible content recommendation is a new but important problem. The paper addresses a significant real-world issue - recommender systems amplifying misinformation and fake news.
2. The paper considers a realistic scenario where only 20% of uncredible items are labeled during training, which can be more representative than full label availability.

### Weaknesses
1. The strong performance relies heavily on comparing against baselines (Traditional, Contrastive, and generic DM-based methods) that were not originally designed to address credible content recommendation under conditions of limited labels.
2. Achieving strong performance requires a large embedding dimension (e.g., 3072 for DM-based methods) and a large number of diffusion steps. How to deal with the tradeoff between efficiency and performance is a challenge.
3. The overall framework is intricate, integrating a disentangled diffusion model, a separate credible subspace projection, and a preference contrast term. The paper does not well explore the intrinsic connections or necessity of coupling all these components, potentially make the design of the model a bit too complex.

### Questions
1. If Disco uses a low-dimensional embedding (e.g., 64 or 128) or fewer time steps, how does it perform comparable to the non-DM baselines?
2. How does the method perform when the proportion of labeled uncredible items varies (not just 20%)?

### Soundness
3

### Presentation
2

### Contribution
2
