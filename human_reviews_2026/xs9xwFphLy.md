# Revisiting [CLS] and Patch Token Interaction in  Vision Transformers

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Vision Transformers have emerged as powerful, scalable and versatile representation learners. To capture both global and local features, a learnable [CLS] class token is typically prepended to the input sequence of patch tokens. Despite their distinct nature, both token types are processed identically throughout the model.
In this work, we investigate the friction between global and local feature learning under different pre-training strategies by analyzing the interactions between class and patch tokens.
Our analysis reveals that standard normalization layers introduce an implicit differentiation between these token types. Building on this insight, we propose specialized processing paths that selectively disentangle the computational flow of class and patch tokens, particularly within normalization layers and early query-key-value projections.
This targeted specialization leads to significantly improved patch representation quality for dense prediction tasks. Our experiments demonstrate segmentation performance gains of over 2 mIoU points on standard benchmarks, while maintaining strong classification accuracy. The proposed modifications introduce only an 8\% increase in parameters, with no additional computational overhead.
Through comprehensive ablations, we provide insights into which architectural components benefit most from specialization and how our approach generalizes across model scales and learning frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the interaction dynamics between the [CLS] token and patch tokens in Vision Transformers (ViTs), particularly under different pre-training strategies. The authors identify that standard ViT architectures treat both token types identically, despite their distinct roles—[CLS] for global representation and patches for local features. Through empirical analysis, they reveal that normalization layers implicitly differentiate these tokens. Building on this insight, they propose architectural modifications that explicitly specialize the processing paths for [CLS] and patch tokens, especially in normalization and QKV projection layers. Their approach improves dense prediction tasks (e.g., segmentation and depth estimation) by up to 2 mIoU points, with minimal parameter overhead and no increase in computational cost.

### Strengths
* The proposed specialization strategy is simple yet impactful, improving dense prediction performance with only ~8% increase in parameters and no additional FLOPs.

* The authors conduct extensive evaluations across model scales (ViT-B, L, H), training paradigms (DINOv2, DeiT-III), and tasks (segmentation, depth estimation, classification).

### Weaknesses
* Lack of theoretical justification. The paper is empirically strong but could benefit from deeper theoretical grounding to explain why specialization improves representation quality.

* To highlight the effectiveness of the proposed layer specialization, while PCA visualizations are provided (Figure 1), more quantitative metrics on feature quality (e.g., clustering scores) could strengthen the argument.

### Questions
* Why the proposed specialization strategy hurts the performance of image classification (Table 6(c) and Figure 9(a)) but benefits dense prediction tasks? Especially, the [CLS] token, which your method focus on, seems to be more important for image classifiction rather than dense prediction tasks.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an architecture specialization method to treat the CLS token and patch tokens distinctively in the ViT models, resulting in a disentangled distribution of two types of tokens. The sufficient and reasonable experiments support the methods and hyper-parameter configurations. The visualizations also coincide with the effectiveness in dense prediction tasks.

### Strengths
1. The method section, along with comprehensive experiments, is impressive. The analyses are comprehensive with both qualitative and quantitative results. The statements are validated by the experiments and clearly support the proposed Specialization for the CLS token and patch tokens.

2. The proposed method is compatible with existing methods (register tokens and attention bias), further improving the overall results of ViT across multiple benchmarks, as shown in Table 2. And it can also stabilize downstream tasks training, as shown in Figure 9. The above experiments stand for the practical application of the proposed method in the general vision area.

3. Table 1 demonstrates that the performance gain does not simply come from the increasing parameters, but rather an intricate design through complete analyses.

### Weaknesses
1. The motivation is unclear at the beginning, such as in the Abstract, Introduction, and Methods. The motivation starts with rethinking the model architecture design, while the specific target is to improve the downstream performance. I would recommend starting with the tasks and the subsequent modifications that could leverage the disentangled distribution to improve the performance.

2. The analysis and explanation of "Dimension separation" from L206 is not clear enough. For example, how to better understand Figure 4?  Is the other axis the "token index"? More explanation and a clear figure are required. 

3. What is the training strategy for ViT-H? This detail is missing.

4. How to explain the visualization impact? For example, Figure 1 demonstrates the two distinctive results by register token and attn bias. However, when applying the proposed specialization method, the visualization becomes similar in general. Can it be explained that the specialization is more fundamental and effective in changing the feature distribution? In addition, from Figures 14-16, the specialization method produces more ambiguous and blurred visual details. Why could these features contribute to better dense prediction tasks?

### Questions
The questions have been listed in the weakness part. I would consider further raising the score if the authors could address my concerns.

### Soundness
4

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
4

### Summary
The paper analyzes how Vision Transformers (ViTs) process the global class token [CLS] and local patch tokens and argues that treating them identically causes friction for dense prediction tasks. Building on observations about token dynamics and normalization, the authors propose layer specialization: in selected blocks they use separate parameters for [CLS] vs. patch tokens (e.g., distinct LayerNorms and Q/K/V projections) while keeping attention shared, so information can still flow. On ImageNet pretraining (supervised DeiT-III and self-supervised DINOv2), they report essentially unchanged ImageNet classification accuracy but around +2 mIoU improvements on semantic segmentation and better depth RMSE, with only a small (~8%) parameter increase. The approach relates to earlier ViT foundations (ViT, DeiT-III, DINO/DINOv2) and to fixes for high-norm/outlier tokens (e.g., Registers) while differing from CaiT (which adds dedicated class-attention layers) by specializing within standard blocks rather than inserting new class-attention stages.

Summary of the review: The paper presents a clear and well-motivated analysis of why identical processing of global and local tokens can limit dense prediction, proposing a simple yet effective solution via token-type-specific normalization and projections. The method is elegant, lightweight, and yields consistent improvements (~+2 mIoU) across models and pretraining setups. However, the novelty is somewhat incremental given overlap with CaiT’s class-attention, and broader validation (e.g., detection, in-context tasks) plus efficiency reporting would strengthen the case. Overall, it’s a technically solid and well-executed contribution with moderate originality, justifying a rating of 6 for clear insight and strong empirical support.

### Strengths
1) Clear problem framing: Why identical processing of global/local tokens can hinder dense tasks is well-motivated and consistent with literature on token dynamics and normalization. 

2) Simple, effective change: Token-type-specific LayerNorm/QKV yields consistent gains without changing attention/FLOPs; minimal engineering overhead.

3) Robust evaluation: Results across pretraining regimes (DINOv2, DeiT-III) and model sizes; improvement magnitude (~+2 mIoU) is meaningful for segmentation.

### Weaknesses
1. Novelty overlap: Since CaiT[1] already introduces class-attention (explicitly specializing how [CLS] aggregates information), authors should include a head-to-head comparison or discussion clarifying differences/advantages (e.g., same backbone size/recipe; dense-task transfer). 

2. Limited Task coverage: Add a few more tasks, like the hummingbird evaluation for in-context vision understanding, object detection/instance segmentation (e.g., COCO) to show benefits generalize beyond conventional segmentation/depth.

3. Training efficiency: Report training memory/time impact of the ~8% parameter increase (batch size, wall-clock), not just inference parity.

[1] Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., & Jégou, H. (2021). Going deeper with Image Transformers. arXiv [Cs.CV].  http://arxiv.org/abs/2103.17239

### Questions
1. CaiT vs. this work: On equal pretraining and decoder setups, how does your method compare to CaiT backbones for dense tasks? Any synergy if both are combined? 

2. Registers vs. specialization: With identical training, which helps dense tasks more on its own—Registers or your specialization? Any redundancy when combined? 

3. Detection results: Do COCO detection/instance-seg metrics show similar benefits? 

4. In-Context Visual Understanding: Does the model gain any benefits on the recent proposed Hummingbird evaluation [1] ( implemented openly by [2])?

5. Stability & hyperparams: Does specialization require different learning-rate/weight-decay schedules to avoid the minor classification drops you mention in fully-specialized variants?

[1] Balažević, I., Steiner, D., Parthasarathy, N., Arandjelović, R., & Hénaff, O. J. (2023). Towards In-context Scene Understanding. arXiv [Cs.CV]. http://arxiv.org/abs/2306.01667
[2] https://github.com/vpariza/open-hummingbird-eval

### Soundness
4

### Presentation
4

### Contribution
3
