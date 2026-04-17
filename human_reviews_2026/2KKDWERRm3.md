# DETR-ViP: Detection Transformer with Robust Discriminative Visual Prompts

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Visual prompted object detection enables interactive and flexible definition of target categories, thereby facilitating open-vocabulary detection. Since visual prompts are derived directly from image features, they often outperform text prompts in recognizing rare categories. Nevertheless, research on visual prompted detection has been largely overlooked, and it is typically treated as a byproduct of training text prompted detectors, which hinders its development. To fully unlock the potential of visual-prompted detection, we investigate the reasons why its performance is suboptimal and reveal that the underlying issue lies in the absence of global discriminability in visual prompts. Motivated by these observations, we propose DETR-ViP, a robust object detection framework that yields class-distinguishable visual prompts. On top of basic image-text contrastive learning, DETR-ViP incorporates global prompt integration and visual-textual prompt relation distillation to learn more discriminative prompt representations. In addition, DETR-ViP employs a selective fusion strategy that ensures stable and robust detection. Extensive experiments on COCO, LVIS, ODinW, and Roboflow100 demonstrate that DETR-ViP achieves substantially higher performance in visual prompt detection compared to other state-of-the-art counterparts. A series of ablation studies and analyses further validate the effectiveness of the proposed improvements and shed light on the underlying reasons for the enhanced detection capability of visual prompts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a openset detection baseline model called DETR-ViP, The model aims to address a long-standing problem in open-vocabulary object detection: how to use visual prompts more effectively.

### Strengths
[1The paper is easy to follow.
[2] The ablation experiments are thorough and demonstrate the effectiveness of each module.

### Weaknesses
[1] The comparison experiments do not include enough existing methods for a comprehensive evaluation like dinov.
[2]  The method mainly relies on prompt ensembling and supervised contrastive learning, which are not particularly novel, and the related work section lacks corresponding discussions.

### Questions
[1] Can the Q-selection mechanism be effective in other transformer-based visual prompting architectures, and is the module plug-and-play?
[2] How does the method handle ambiguous targets, such as conflicts between category names and action names?
[3] Table 2 introduces a new metric, but it should clearly explain how its variation reflects the model’s performance.
[4] The zero-shot experiment only covers 10 common categories, which is insufficient to demonstrate the method’s advantage in discriminative ability.
[5] What is the difference between the distillation method in the paper and the contrastive learning used in CLIP?

### Soundness
4

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
This paper suggests the visual prompts suffer from a lack of sufficient semantic discriminability. The reseaerch question motivates the authors to propose the detection model, DETR-ViP, by enhancing the baseline model, VIS-GDINO, with more robust and discriminative visual prompts. Extensive experiments are conducted.

### Strengths
The paper enhances Grounding-DINO with text and visual prompts. Furthermore, three strategies are utilized to improve the visual prompts.

The experiments show that the proposed method outperforms existing detection models in the task of zero-shot generic detection.

### Weaknesses
In the introduction, the visualizaiton for semantic discriminability in visual prompts looks similar to VIS-GDINO, so the motivation is similar? From Figure 5 in T-Rex2, I do not find the same phenomenon in tSNE visualization. Would be nice if the authors evaluate some publicly used models to clarify the motivation.

Some parts of is a bit unclear to me. For instance, in preliminary, the $\mathcal{L}_{dn}$ in Equation 4 is not defined. Also, I couldn't find detailed description for the proposed VIS-DINO. Since VIS-DINO is also a new proposed model, the details of it can be explained?

 In the second paragraph of Introduction, the authors state "Nevertheless, visual prompts still underperform text prompts overall, which limits their practical applicability." Is there any reference or evidence for this phenomenon? 

It seems the proposed method can also be used for other detection models, so would you please clarify the reason behind choosing Grounding-DINO as backbone? From Table 1 and 2, it seems Grounding-DINO can not perform better than YOLOv11 on LVIS.

### Questions
Please see weaknesses above.

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
4

### Summary
This paper proposes DETR-ViP, a visual-prompt-based open-vocabulary detector built upon Grounding-DINO and T-Rex2. Recent studies suggest that visual prompts can outperform text prompts for rare categories, though they still underperform on common and frequent categories. Motivated by this trade-off, the method focuses on the visual-prompt regime and introduces three components to improve semantic structure and robustness: (1) Global Prompt Integration to aggregate prompts across a batch, (2) Visual–Text Relation Distillation to transfer semantic relations from language to visual prompts, and (3) Selective Fusion to suppress irrelevant prompts during feature fusion. Empirical results show competitive performance across COCO, LVIS, ODinW, and RoboFlow100 in zero-shot detection, with consistent improvements over baselines across rare, common, and frequent category groups.

### Strengths
- **Relevant and timely problem**

    The paper addresses an emerging direction in open-vocabulary detection, focusing on strengthening visual prompts. This is particularly relevant given recent findings that visual prompts can better support rare categories compared to textual prompts, even though they still trail on common and frequent ones. Framing the work around this trade-off makes the study well-motivated and impactful.

- **Novelty and contributions**

    The paper proposes clear and meaningful contributions toward improving the effectiveness of visual prompts for open-vocabulary detection. The introduced mechanisms are sensible, well-motivated, and grounded in practical challenges of visual-prompt learning. While the ideas are not radically complex, they are thoughtfully designed, address concrete limitations of existing approaches, and together offer an interesting and coherent advancement in visual-prompt-based detection.

- **Good empirical results**

    The evaluation covers a broad set of benchmarks (COCO, LVIS, ODinW, RoboFlow100), demonstrating consistent improvements across datasets. Importantly, gains are observed across rare, common, and frequent category subsets, strengthening the motivation for exploring visual prompts in open-vocabulary detection and supporting the practical value of the proposed design.

### Weaknesses
- **Method description lacks clarity in several key areas**

    Several core components are not described with sufficient precision in the main text, which makes the method harder to fully understand without forcing the reader to infer implementation behavior. In particular: the mechanics of global prompt integration remain poorly described, or the fact that selective fusion is used during training and/or inference only. Clarifying these points would greatly improve readability and transparency.
- **Fairness of comparison to T-Rex2**

    Given the architectural similarities to T-Rex2, a cleaner evaluation would require matching training conditions. However, the training data differs substantially (T-Rex2 leverages SA-1B and additional datasets, whereas DETR-ViP does not), which complicates attribution of performance gains to the proposed design alone. This point is further relevant because the T-Rex2 paper itself notes that including the SA-1B dataset “lightly weakens its generic capability.” As a result, it remains unclear to what extent differences in training data rather than architectural changes contribute to the performance gap.

- **Lack of sensitivity analysis for key hyperparameters**

    Several important hyperparameters are introduced throughout the method, including those for fusion $\lambda$, distillation strength ($λ_{distill}$), temperature parameters (τₜ, τᵥ), and focal loss weighting (α, β, γ). However, the paper does not provide sensitivity studies or robustness analysis for these choices. Since these components play a central role in shaping the behavior of the model, it would be valuable to understand how performance varies with different settings and whether the method remains stable across reasonable ranges. Including such experiments would help substantiate the robustness of the approach and clarify the extent to which reported gains depend on careful hyperparameter tuning.

### Questions
1. The “Global Prompt Integration” section remains very vague and the code snippet is not at the standard of top-tied venue. For example: “aggregates prompts from all samples and integrates them into a unified classifier”. What exactly are “all samples”? What does “unified classifier” refer to? My understanding is that DETR-ViP pools visual prompts from all images $\underline{\text{in the batch}}$, groups them by category, and averages them to form one prototype per class present in that batch. These batch-shared prototypes then serve as classifier weights, increasing negative diversity and stabilizing training. If this understanding is correct, I recommend describing the mechanism in this more explicit way in the paper.

2. The paper uses the $L_{dn}$ loss that appears to be the denoising loss (as in DINO). It should be explicitly stated for completeness. 

3. The classification loss description does not specify whether a sigmoid is applied to similarity scores before focal loss, as done in T-Rex2. Also, are embeddings L2-normalized prior to similarity computation? Could the authors clarify the exact pipeline?

4. If visual prompts are already aligned with their corresponding text embeddings through the visual-text alignment loss as in T-Rex2, why do they not inherit from the textual semantic structure?  Can the authors comment on why is a relation-distillation loss needed in addition? Clarifying the conceptual motivation for combining both would help and strengthen the paper beyond good empirical results.

5. From my understanding, the selective fusion gating mechanism is applied during both training and inference to handle variable prompt availability and suppress irrelevant prompts. Is this correct? If so, please make it explicit in the method section.

6. Since the selective fusion strategy is motivated by interactive scenarios with variable numbers of prompts, could the authors include interactive detection and few-shot counting evaluations, similar to Tables 2–3 in T-Rex2?

7. In Table 1, T-Rex2 is trained on Object365, OpenImages, HierText, CrowdHuman and SA-1B, while DETR-ViP uses Object365 + GoldG. T-Rex2 notes SA-1B improves interactive ability but can weaken generic capability. Could the authors provide a comparison under the same training data regime? This would strengthen the paper and isolate architectural contributions.

8. I understand that the core of this paper is to improve the visual-prompt capabilities for open-vocabulary detection. But why remove text-prompt capability? Is there a practical issue in conserving this textual-prompt capability while improving the visual prompt capability? Could the authors explain why this capability was not retained?

9. Grounding DINO already includes cross-modality (image-text) feature fusion in its encoder. The ablation includes a line “+Encoder Fusion”. Is this not already present in VIS-GDINO? Please clarify what exactly is disabled/enabled.

10. In Eq. (10), the output appears to be prompt-space features (V = Wᵥ Pᵀ), but the text refers to fused image features. Could the authors adjust the description in lines 265–266 for consistency?

11. Could the authors provide sensitivity results for λ (fusion gate), α/β/γ (focal loss), τₜ/τᵥ (temperatures), and λ_distill?

12. Am I correct that the main difference from T-Rex2 in negative sampling is the global prompt integration mechanism, whereas T-Rex2 samples visual prompts within each image to avoid cross-image label inconsistencies? A clearer contrast in the " global prompt integration" section would help highlight the contribution.

### Soundness
3

### Presentation
2

### Contribution
3
