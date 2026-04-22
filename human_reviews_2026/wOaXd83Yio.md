# Personalized Visual Representation Alignment for Generative Multimodal Recommendation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
With the development of Vision-Language Models (VLMs) for multimodal understanding, recommender systems have increasingly leveraged them to process heterogeneous sources of user-interacted items for recommendation. By fine-tuning VLMs on user interaction data, prior works have adapted these models to capture user preferences, enabling personalized multimodal recommendation. Despite these advances, however, we identify two key limitations: 1) the visual features directly extracted by vision encoders (e.g., CLIP) are insufficient for capturing personalized user preferences, as such encoders are generally trained for generic visual perception rather than capturing user-specific preferences; and 2) existing VLM-based methods often underutilize visual features of user-interacted items in later LLM layers, relying instead on textual descriptions for recommendation—an unexpected bias that diminishes the contribution of visual features. To address these two limitations, we propose PerVRA, a VLM-based recommendation model consisting of a Personalized Visual Representation Learning (PVRL) module and a Personalized Multimodal Alignment (PMA) module. Specifically, we employ dual contrastive learning, where each module is equipped with its own contrastive objective: The PVRL module learns personalized visual features from user interaction history, while the PMA module enhances the contribution of visual features to the VLMs by explicitly aligning them with text features. Extensive experiments on real-world Amazon and H\&M Fashion datasets demonstrate that PerVRA consistently outperforms strong VLM-based methods over diverse personalized tasks. Moreover, our ablation studies show that addressing these two limitations is critical for building effective VLM-based recommender systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the PerVRA framework for VLM-based recommendation systems, employing dual contrastive learning to learn personalized visual representations and align them with textual features. The paper identifies key limitations of existing methods, proposes reasonable solutions, and validates effectiveness across multiple datasets.

### Strengths
S1. Figure 1 provides intuitive visual evidence that text-only model outperforms multimodal variant.

S2. The paper clearly identifies two key issues in VLM-based recommendation systems, the motivation is  convincing.

S3. The PVRL and PMA modules have clear division of labor, optimizing visual space and multimodal alignment respectively.

### Weaknesses
W1. The contrastive learning in recommender systems is not novel, and the core dual contrastive learning idea lacks originality.

W2.  No analysis of λ1 and λ2 impact, Table 4 only tests complete removal.

W3. Design of Equation 9 lacks theoretical basis, and it may lead to negative loss.

### Questions
Q1. Suggest clarifying why this form is used instead of standard InfoNCE in Eq.4.

Q2. Why not contrast lt with lt+ in Eq.7? What are the advantages of this design?

Q3. The  "PerVRA" and "PerVLA" are used mixed in multiple places, e.g., Equation 8.

Q4. On H&M dataset in Table3, PerVRA's text-only evaluation (HR@5=0.079) is even slightly higher than multimodal evaluation (HR@5=0.078). This contradicts core motivation that "visual information is important"

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
5

### Summary
This paper introduces PerVRA, a personalized visual representation alignment framework addressing two key issues in VLM-based recommendation: (1) frozen vision encoders lack user-specific preference modeling; (2) VLMs underutilize visual signals and over-rely on text. PerVRA consists of PVRL and PMA modules with dual contrastive learning, treating user history as both positives and hard negatives. The training objective combines task loss and contrastive losses without adding inference cost. Experiments on multiple datasets show strong gains over SOTA baselines and robustness to missing modalities.

### Strengths
- Proposes a novel dual-role use of user history (as both positive and negative) for contrastive learning at both visual (PVRL) and textual (PMA) levels, moving beyond standard image-text matching or InfoNCE.  
- Outperforms UniMP in sequential recommendation across Amazon/H&M/Netflix/Book, and shows improvements across search, selection, preference prediction, and explanation tasks; robust under missing modality settings.  
- Objective functions and pipeline are clearly illustrated, with t-SNE plots supporting the method’s effect on latent structure.  
- Demonstrates practical value for making vision "count" in unified VLM frameworks for personalized recommendation, without adding inference cost.

### Weaknesses
1. Results are mostly single-run without variance/confidence intervals or t-tests, making it hard to judge reproducibility, especially for low-score datasets. Recommend reporting mean ± std over 3–5 seeds.  
2. Hyperparameters like λ₁, λ₂, τ, and MLP size are fixed but not analyzed. The effect of sampling ratio and strategy (e.g., history as hard negatives) is also unclear.  
3. Equation (9) introduces a repetition penalty term, but its intuition and difference from standard diversity constraints are insufficiently explained.  
4. The mechanism for selecting historical items as negatives is not fully detailed (e.g., sampling ratio, subsampling, temporal decay).  
5. Although inference cost is equal, training conditions may differ (e.g., batch size, resolution, cleaning), possibly leading to unfair baseline comparisons.  
7. The method is validated on CLIP ViT-L + 3B LLM only. It’s unclear how it generalizes across backbones, model sizes, or alignment layers.

### Questions
1. What is the sampling strategy and N for using history items as positives/negatives in UCL/RCL? Is there any N-sweep or heatmap to show trade-offs?  
2. How does the repetition penalty differ from typical diversity or de-duplication rules in generation? Any ablation before/after applying it?  
3. In missing modality experiments, does randomly dropping half of the images during training reflect real-world distributions? Are text-only test sets matched in candidate composition?  
4. PMA aligns visual to the final LLM layer. Has the method tried aligning at cross-attention or intermediate layers (multi-layer distillation)?  
5. With 4×A6000 and 10 epochs, how does training cost compare to UniMP? If training cost is equalized, are the gains still significant?  
6. Amazon Beauty shows strong gains, but what about visually weak or long-tail domains (e.g., books/comments)? Any cross-domain fine-tuning or zero-shot generalization?

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
3

### Summary
The paper points out that there are two key problems in current VLM recommendation tasks: 1) the features extracted by visual encoders are universal perception-oriented and lack personalization;2) visual features are underestimated in the LLM layer, and recommendation systems rely too much on textual information. To this end, PerVRA has introduced two modules: PVRL (Personalized Visual Representation Learning) and PMA (Personalized Multimodal Alignment), which respectively enhance the personalized expression of visual features through double contrast learning goals and establish more effective alignment between vision and text.

### Strengths
+ The motivation of VLM's bias in recommendations is interesting.
+ The evaluation is extensive, and the experimental results look promising.

### Weaknesses
+ Existing VLM-based methods often underestimate visual features of user-interacting items in later LLM layers, which is only an empirical inference and lacks rigorous proof.
+ The reason behind the design choice of the method is not clearly explained.
+ There are some presentation errors in the paper, such as line 329.
+ PerVRA and PerVLA alternately appear, which makes it confusing.
+ Lack of comparison with other methods that focus on personalized visual modeling limits the method's innovative evaluation of visual personalization

### Questions
+ Why Text-only is better than Text+image is contrary to the conclusion in UniMP.
+ Why can dual contrastive learning objectives balance the contributions of visual and textual features and avoid bias?
+ Multiple contrastive losses and modules are introduced during the training phase, and the actual training costs are not detailed.

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
3

### Summary
This paper aims to address two limitations in existing VLM-based recommendation tasks: (1) visual features are insufficient for capturing personalized user preferences, and (2) visual features are underutilized by current VLM recommendation models. To tackle these issues, the authors propose PerVRA, which consists of a Personalized Visual Representation Learning (PVRL) module and a Personalized Multimodal Alignment (PMA) module, both built upon contrastive learning. The experiments are conducted using the UniMP model and the Amazon review dataset.

### Strengths
1. The task is clearly defined, and the limitations are explicitly described with empirical evidence.
2. The paper is easy to read and understand.
3. PerVRA shows significant improvements over baseline approaches.

### Weaknesses
1. The main concern is the generalization of this method. Since PerVRA has been specifically designed for UniMP, it is unclear whether it can be applied to or remain effective for other VLM-based recommendation models. For example, if the vision encoder and text encoder are already highly aligned in a VLM, would PerVRA still provide improvements?
2. Figure 1(a) appears to be inconsistent with Table  3. In Figure 1(a), the text-only setting outperforms the multimodal setting, while in Table 3, the text-only setting performs worse than multimodal.
3. There are no hyperparameter sensitivity experiments. As such, it is unclear how changes to $\lambda_1$ and $\lambda_2$ would affect the results.
4. Several typos exist. For instance, in the OpenReview keywords, “Multimodal RecommeXx” should be corrected. In Section 4.1, Line 329, “Book-Crossing () datasets” appears incomplete.

### Questions
If a VLM has a strong text encoder, and the visual encoder that is highly aligned with the text encoder, the problem described by the authors, such as “if a user prefers kitchen-related items, objects like knives and frying pans should be embedded closer together rather than treated as distinct classes”, may not occur, since knives and frying pans would already be close in the semantic space. In such a scenario, would the problem that PerVRA aims to address still exist?

### Soundness
2

### Presentation
3

### Contribution
2
