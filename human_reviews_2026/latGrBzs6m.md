# MeasureNet: Polyline Detection Based Measurement for Celiac Disease Identification and Grading

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Celiac disease is an autoimmune disorder triggered by the consumption of gluten. It causes damage to the villi, the finger-like projections in the small intestine that are responsible for nutrient absorption. Additionally, the crypts, which form the base of the villi, are also affected, impairing the regenerative process. The deterioration in villi length, computed as the villi-to-crypt length ratio, is one indicator of the severity of the disease. However, manual measurement of villi-to-crypt length can be both time-consuming and susceptible to inter-observer variability.

Automated systems may perform measurement as a post-hoc process over segmentation outputs, but developing a villi/crypt segmentation model requires costly expert annotation, which is made worse by complex villi structures and unclear crypt boundaries. In response, our pathologists provide inexpensive annotation of (approximate) bisectors of villis/crypts for length measurement, and of auxiliary tissues that represent villi and crypt boundaries. To use such annotation, we propose MeasureNet, a polyline detection framework that integrates polyline localization with object-driven losses tailored for measurement tasks. Additionally, we leverage a segmentation model that provides auxiliary guidance on crypt locations when crypts are partially visible. To prevent over-reliance on (noisy) segmentation masks, we employ a mask feature mixup technique. We validate MeasureNet on our novel dataset CeDeM, and also a publicly available DeepBacs dataset. Compared to best SoTA baseline, MeasureNet obtains 9 and 11 percent points improvement in binary and multi-class grading of celiac disease.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces MeasureNet, a novel polyline detection framework designed to measure villi and crypt lengths directly from biopsy images without relying on full segmentation.

### Strengths
This paper proposes an interesting task that has research significance in Diagnosing celiac disease (CeD).

### Weaknesses
There are lots of unclear details regarding to the training process and problems in presentations

### Questions
1.The task definition is unclear. The work seems to target a structured detection problem rather than conventional object detection. It would help to add a short subsection in Section 3 that clearly defines the task, input/output format, and evaluation target.
2.The training process is not well explained. Are the strong and weak segmenters trained jointly or separately? During detector training, are auxiliary masks provided as input? The overall pipeline should be clarified.
3.In Figure 3, the “Segmentation” block should be labeled as SegFormer for consistency.
4.The paper does not explain how different loss terms are balanced. Please describe or justify the weighting strategy.
5.The dataset is highly imbalanced. Did the authors consider any resampling or weighting strategy to address this issue?
6.It would be useful to include a comparison of training cost or efficiency with baseline models.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents MeasureNet, a polyline detection framework that integrates polyline localization with object-driven losses for measurement tasks. It further employs auxiliary segmentation guidance and a mask feature mixup to handle partially visible crypts and reduce noise dependence.

### Strengths
The paper is clearly written, with a precise formulation of the problem and convincing demonstrations.

### Weaknesses
1. The motivation for using both a strong and weak segmentor is not clearly explained.
2. The introduction of two segmentation models inevitably increases computational overhead — the paper lacks analysis of the training and inference efficiency.
3. The method presents limited technical novelty. The combination of the strong/weak segmentors and the mixup strategy appears conceptually similar to a form of data augmentation.
4. The proposed framework is built upon DINO-DETR, a strong pre-trained model, whereas the competing baselines in Tables 1 and 2 do not seem to use such a pretrained backbone, raising concerns about fairness in comparison. It remains unclear whether the observed performance gains mainly stem from DINO-DETR itself.

### Questions
Will CeDeM be made publicly available?

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
3

### Summary
This work proposes a new method for polyline regression using MeasureNET. They showed that using segmentation information from the villus and crypt regions helps the detector to identify and measure the lengths of the polylines. By analyzing the ratio between villus and crypt, they could perform celiac disease grading with results that outperform the state-of-the-art, even when compared with classification models.

### Strengths
- The authors collected a new dataset of histological images with segmentation annotation of the villi/crypts for the study of celiac disease. The dataset will be made publicly available upon possible acceptance.

- A cascade of segmentation and object detection frameworks outperforming state-of-the-art polyline detection is proposed.

-  The earth mover's distance was applied to complement the most common losses used for object detection, showing that this term improves polyline detection performance.

- Ablation studies were performed to assess the value of each contribution to the work, showing that the introduced methods, in fact, improve performance.

### Weaknesses
- Presentation can be improved regarding text syntax and placement of figures and tables. Tables and figures are often within the text, which compromises the paper's readability.
- The paper's main strength lies in combining segmentation information to improve detection. Nevertheless, this might have limited application.
- Some of the results can be better discussed to solidify the contributions.

### Questions
- In Section 3, the authors said: "measurement losses to compare lengths of the predicted polylines with the gold". What does gold mean? Gold-standard? This term is repeated along the manuscript.

- The authors mentioned that the weak segmentation model is trained with 50% of capacity. Can the authors give more information on that? It would also be valuable to present the metrics for the weak and strong segmentation models.

- In Table 1, it is shown that MeasureNET outperforms the state-of-the-art. Nevertheless, the proposed method uses the segmentation information to leverage its detection capabilities, which the other methods do not benefit from. Can the features outputted by the feature merger (fUIM) be incorporated into lane detection models such as LETR? If so, a small evaluation on the performance of LETR would be
valuable to assess if reducing the exposure bias also improves other models. 

- When comparing with state-of-the-art lane detectors, are the other detection models fine-tuned using the earth mover&#39;s distance in the loss function? Especially for DeepBacs, where no segmentation is needed, it would be valuable to see the impact of this proposed term on the other models.

- In the related works, it is said that the primary tool for diagnosing CeD is the IEL counting, and a model (DeGPR) performs classification based on this counting. Why are the results for DeGPR not present in the experiments section? Since measuring the villi/crypt is just an alternative approach, a comparison between a model based on the standard procedure would be valuable for the work.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a specific problem in medical image analysis, villus and crypt detection. The key objective is not to achieve full segmentation but rather, guided by pathologists’ insight that precise segmentation of villi and crypts is not essential in clinical practice. The proposed architecture, MeasureNet, is constructed from several established components, incorporating existing network architectures or elements from such network architectures (e.g. SegFormer, DINO, inverted residual block from  MobileNetV2). In addition, the paper introduces a novel dataset, CeDeM, along with corresponding annotations. The performance of the proposed method is evaluated both on this dataset and on a second public dataset addressing a different task.

### Strengths
The insight to simplify the task to polyline detection makes this otherwise complex medical image analysis problem far more tractable for practical applications. The paper presents an interesting and well-motivated solution to the reduced task.

The use of mixup between strong and weak segmentation masks to mitigate exposure bias during model training appears to be a novel and promising approach. 

Furthermore, the introduction of the CeDeM dataset, along with its corresponding annotations, represents a valuable contribution to the research community.

### Weaknesses
The methodological contribution of this work is relatively incremental. The proposed MeasureNet architecture is assembled from several established components, combined with various adaptations of the loss function.

While the selected components appear reasonable, the rationale behind their specific choice is not always clearly articulated.

A considerable number of additional terms are introduced into the loss functions. Although these terms aim to capture different aspects of the task, their relative weighting becomes increasingly problematic. For instance, the total segmentation loss is defined as Lseg = Ldice +Lce +LDTW, where the last term operates on a significantly different numerical scale. Without appropriate normalization, simply assigning all weighting factors to one may render some terms ineffective. A similar issue arises in the final training loss for Dθ. The authors state: “We simply add all terms, and do not introduce any hyperparameters in the final loss, to save on hyperparameter tuning”. However, this simplification may inadvertently diminish the impact of certain loss components.

The comparison with other methods, referred to as baselines, is also somewhat misleading. The compared approaches are “general-purpose” methods that have not been tailored to this specific task. Consequently, the reported superior performance of the proposed model is not entirely unexpected.

### Questions
The Earth Mover’s Distance (EMD) exhibits issues with smoothness and differentiability. Should the authors consider using a differentiable alternative, such as the Sinkhorn distance (entropic-regularized EMD)?

How critical are the many loss terms used in training? Which terms contribute most to performance, and does tuning their weights substantially affect results?

What is the LWNet reported in Table 3? This model is not described in the paper.

Table 4 suggests that mask augmentation and mixup drive most of the performance gains; the contribution of the three L terms is unclear. Are these loss components truly essential?

### Soundness
3

### Presentation
2

### Contribution
2
