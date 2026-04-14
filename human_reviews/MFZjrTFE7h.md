# D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
We introduce D-FINE, a powerful real-time object detector that achieves outstanding localization precision by redefining the bounding box regression task in DETR models. D-FINE comprises two key components: Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). FDR transforms the regression process from predicting fixed coordinates to iteratively refining probability distributions, providing a fine-grained intermediate representation that significantly enhances localization accuracy. GO-LSD is a bidirectional optimization strategy that transfers localization knowledge from refined distributions to shallower layers through self-distillation, while also simplifying the residual prediction tasks for deeper layers. Additionally, D-FINE incorporates lightweight optimizations in computationally intensive modules and operations, achieving a better balance between speed and accuracy. Specifically, D-FINE-L / X achieves 54.0% / 55.8% AP on the COCO dataset at 124 / 78 FPS on an NVIDIA T4 GPU. When pretrained on Objects365, D-FINE-L / X attains 57.1% / 59.3% AP, surpassing all existing real-time detectors. Furthermore, our method significantly enhances the performance of a wide range of DETR models by up to 5.3% AP with negligible extra parameters and training costs. Our code and models: https://github.com/Peterande/D-FINE.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
D-FINE is a real-time object detection model that refines bounding box regression in Detection Transformers using Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). These techniques enhance localization accuracy and efficiency by iteratively refining bounding boxes and transferring refined knowledge across layers. D-FINE outperforms other real-time detectors on the COCO dataset, achieving a better balance of speed and precision with minimal added costs.

### Strengths
1.	D-FINE surpasses all existing real-time detectors with negligible extra parameters and training costs. Source code is provided. This method appears to be highly solid and reproducible.
2.	Table 2 illuminates that the proposed FDR and GO-LSD are efficient for a series of DETR models, demonstrating the robustness of the module.
3.	The proposed method is both innovative and systematic.

### Weaknesses
1.	In Table 3, FDR and GO-LSD modules have a 1% mAP improvement in total for D-FINE, which is noticeably less than the improvement shown in Table 2.
2.	The language in the article needs refinement, as some phrases may lead to ambiguity. For example, line 206-208, “Initially, the first …” and line 209, “…one for each edge”.
3.	Variable names in the Method need to be consistent. For example, line 213, {W, H} and line 139 {w,h}. The lowercase ‘w’ is preferable, as the uppercase ‘W’ repeats the notation used for the weighting function below.

### Questions
1.	In Table 3, FDR and GO-LSD modules have a 1% mAP improvement in total for D-FINE. However, according to table 2, these two modules improve accuracy by 2% at least. Can the author explain this phenomenon?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces D-FINE, a real-time object detection model that enhances DETR by redefining bounding box regression with Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). The FDR transforms bounding box predictions into refined probability distributions, while GO-LSD uses refined localization knowledge to improve earlier layers' predictions. Experiments on the COCO dataset show that D-FINE achieves good performance and high speed, positioning it as a strong competitor to existing real-time detectors.

### Strengths
1. The paper presents D-FINE, an enhancement to the DETR framework that tackles bounding box regression with a novel fine-grained distribution refinement and self-distillation mechanism. 

2. The experimental results on the COCO dataset demonstrate competitive performance, showing that D-FINE outperforms many existing real-time object detectors, with a favorable trade-off between speed and accuracy. Additionally, the paper’s ablation studies and hyperparameter tuning offer insightful explanations of the model’s design choices.

3. The experiments and comparisons with various DETR-based models are thorough. The inclusion of both small and large models, alongside real-world visualizations, strengthens the validity of the claims about D-FINE's performance across diverse conditions.

### Weaknesses
The experiments focus primarily on the COCO dataset. Evaluating D-FINE across a broader range of datasets could strengthen its generalizability claims and confirm its robustness across different object detection contexts, such as crowded scene CrowdHuman and long tail scene LVIS.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes D-FINE, which consists of two methods, Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD) to improve the performance of real-time object detectors. Based on the probability distribution representation of bbox, the paper uses multiple D-FINE head to predict the prob. distributions of bboxes on $L$ layers in one decoder. FDR uses a hand-crafted weight function to weight the prob. distribution of bbox and then iteratively refine the bboxes. GO-LSD is inspired from localization distillation, which distills the localization knowledge from the prob. distribution of bboxes of the last layer to the ones of the shallow layers. The paper also designs two losses to work with the proposed FDR and GO-LSD. Experiments on COCO benchmark show the effectiveness of the proposed method.

### Strengths
1. The proposed method is technically sound, which is supported by sufficient experiments.

2. The paper shows the roadmap from the baseline model to the proposed D-FINE framework, making the technical contribution clear and transparent.

3. The proposed D-FINE achieves state-of-the-art performance in real-time object detection.

### Weaknesses
1. It is confusing to me that in Fig. 2, it seems that D-FINE with FDR is applied to the 1-st decoder layer of the object detector. Then, within 1 decoder layer, multiple D-FINE head is used to generate the prob. distribution of bboxes. However, on lines 207-208, you mention "the first decoder layer predicts preliminary bounding boxes and preliminary probability distributions through a traditional bounding box regression head and **a D-FINE head**". In Sec. 4.2,  GO-LSD utilizes the final layer’s refined distribution predictions to distill localization knowledge into the earlier layers. One can see in Fig. 3 that the self-distillation is conducted between different decoders. Thus, is that the meaning of the word "layer" different in FDR and GO-LSD? What is the exact meaning of "layer" in the context of FDR versus GO-LSD? 

2. In Fig. 3, the matched prediction and unmatched prediction are colored by green squares and yellow squares. What does the gray squares stand for? I suggest that the authors could explicitly state what the gray squares represent in the caption or legend of Figure 3.

### Questions
1. How many D-FINE heads are used in the 1-st decoder, e.g., the value of $L$? Is FDR applied within each decoder layer or across decoder layers? 

2. If FDR is applied differently across different decoders, whether the number of D-FINE heads is consistent across all decoder layers or if it varies? How and why it differs? I suggest that the authors add this information to provide a clearer picture of the model's architecture and operation would enhance the reproducibility of the proposed method.

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
This paper introduces a real-time object detector based on DETR through two main components: fine-grained distribution refinement, which refines probability distributions iteratively for improved accuracy, and global optimal localization self-distillation, which optimizes localization through a bidirectional self-distillation approach.

### Strengths
The paper is easy to understand and well-presented. A set of experiments are conducted on the COCO and Objects365 datasets.

### Weaknesses
1. I do not fully agree with the claim that using fixed coordinates results in poor performance due to inadequate modelling of localization uncertainty. While the statement shows potential limitations of fixed-coordinate regression, many SOTA object detectors (Faster R-CNN and RetinaNet), represent significant advances in both speed and detection. Therefore, to say that all methods using fixed coordinates were fundamentally limited overlooks these advancements. The authors may clarify their claim by acknowledging the success of fixed-coordinate methods while explaining how the proposed approach can address these limitations

2. It has been stated, " As early predictions improve, the subsequent layers can focus on refining smaller residuals.", what do you mean by  "refining smaller residuals"?

3. In page 5, Eq. (3).  The normalizing updated logits can result in very small gradients when logits are large, making it harder for the model to learn effectively in deeper layers. This can reduce the model’s precision in refining bounding box edges. Moreover, softmax forces the output to sum to one, which can limit its ability to model complex relationships between bins and reduce its performance for handling localization uncertainty. Have you tried any other normalization methods such as Entmax normalization which well fit into DERT methods? and why softmax has been chosen over other options?

4. In Figure 4 what those sub-plots (right, left, ...) mean and represent? There is no discussion about these plots. It is not clear why these plots are shown and what they are representing. Moreover, it is unclear, what the pick value means and what different lines (red, green) are supposed to be for the best performance. The authors need to provide a more detailed caption or explanation for this figure, specifically describing what each subplot represents and the meaning of the different colored lines.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
