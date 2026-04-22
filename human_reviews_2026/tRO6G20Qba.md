# Dual Distillation for Few-Shot Anomaly Detection

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Anomaly detection is a critical task in computer vision with profound implications for medical imaging, where identifying pathologies early can directly impact patient outcomes. While recent unsupervised anomaly detection approaches show promise, they require substantial normal training data and struggle to generalize across anatomical contexts. We introduce D$^2$4FAD, a novel dual distillation framework for few-shot anomaly detection that identifies anomalies in previously unseen tasks using only a small number of normal reference images. Our approach leverages a pre-trained encoder as a teacher network to extract multi-scale features from both support and query images, while a student decoder learns to distill knowledge from the teacher on query images and self-distill on support images. We further propose a learn-to-weight mechanism that dynamically assesses the reference value of each support image conditioned on the query, optimizing anomaly detection performance. To evaluate our method, we curate a comprehensive benchmark dataset comprising 13,084 images across four organs, four imaging modalities, and five disease categories. Extensive experiments demonstrate that D$^2$4FAD significantly outperforms existing approaches, establishing a new state-of-the-art in few-shot medical anomaly detection. Code is available at https://github.com/ttttqz/D24FAD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on image-level anomaly detection across anatomical contexts. It proposes a dual distill framework for anomaly detection in few-shot setting, where only up to 8 normal images as references. Specifically,  a spatial feature reconstruction loss is implemented for query images, and a spatially discrepancy loss between the query image and references images are implemented. The proposed framework is highly closed to the realistic medical setting, the proposed framework achieves satisfactory results.

### Strengths
Strengths:
- This paper is well-written and easy to follow. 
- The proposed dual distillation framework is effective and easy to implement. 
- The experiments are extensive.

### Weaknesses
Weaknesses:
 -  The introduction part from Line 52- Line 54 can be better motivated.
   - 1] actually leverages a large amount of unlabeled abnormal data for anomaly detection. I don’t quite understand why the authors cite this paper here, since they are arguing that, in some cases, normal data can be rarer than abnormal data.
   -  The authors should have supported reference papers for the statement/claim from Line 54-55.
- Figure 1 is not very informative. For example, the authors could include notations such as  X to represent the feature space of the teacher encoder and Z for the space of the student decoder. In addition, an illustration showing how the proposed loss terms are computed would make the figure more informative and easier to understand.
 - The title and the abstract are a bit mis-leading. After reading the paper, it seems to me that the inference is few-shot but not the training. My understanding is that only $K$ images for training if the setting is $K$-shot. However, the Line 291-309 suggest the training requiring the whole dataset. Please clarify this if I am wrong.

Minor weaknesses: 
- Line 63 should be student decoder.

### Questions
See weaknesses.

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
3

### Summary
This paper introduces D²FAD, a novel framework for few-shot anomaly detection (FAD), with a specific focus on medical imaging. The core problem addressed is the scarcity of normal reference images in many clinical scenarios, which limits the applicability of traditional unsupervised anomaly detection methods. D²FAD employs a dual distillation strategy: 1) Teacher-Student Distillation, where a student decoder learns to reconstruct multi-scale features from a pre-trained (and frozen) teacher encoder on query images, and 2) Student Self-Distillation, where the student decoder learns to produce consistent representations between a query image and a small support set of normal images from the same domain. Additionally, the authors propose a "learn-to-weight" mechanism to dynamically assign importance to different support images based on their relevance to the query. The authors also contribute a new comprehensive benchmark dataset for medical FAD, curated from several public sources. Experiments show that D²FAD significantly outperforms a wide range of unsupervised and few-shot anomaly detection methods across this new benchmark.

### Strengths
Clear Motivation and Problem Formulation: The paper does an excellent job of motivating the need for few-shot anomaly detection in clinical practice, grounding the research in a real-world problem. The formalization of the FAD task is clear and precise.

Elegant and Effective Method: The D²FAD framework is simple yet powerful. The dual distillation concept is intuitive and well-justified. By using a frozen pre-trained encoder as the teacher, the method is parameter-efficient and avoids the need for complex architectural designs.

Thorough and Convincing Experiments: The empirical evaluation is a major strength. The authors have not only proposed a new method but have also built a strong benchmark to validate it. The comparisons against a wide array of baselines are comprehensive, and the ablation studies provide strong evidence for the efficacy of each component of their proposed method. The consistently high performance across multiple datasets and few-shot settings (2, 4, 8-shot) is very impressive.

High Practical Relevance: The method is well-aligned with clinical reality, where often only a few trusted normal cases are available for reference. The high AUROC scores, combined with the method's efficiency (as shown in Figure 3), suggest strong potential for practical deployment.

### Weaknesses
Limited Technical Depth in "Learn-to-Weight": While the "learn-to-weight" mechanism (Eq. 4) is a good idea, its presentation is somewhat brief. It is essentially a scaled dot-product attention between the query and support features. The paper could benefit from a deeper analysis or discussion of this component. For example, are there other ways to instantiate this weighting? How does this mechanism behave in practice (e.g., does it learn to ignore outlier-like support images)?

Sensitivity to the Pre-trained Teacher: The entire framework's performance is heavily dependent on the quality of the frozen teacher encoder (pre-trained on ImageNet). While the ablation in Table 4 explores different backbones, it raises a question: how does the choice of pre-training dataset (not just architecture) affect performance? Would a teacher pre-trained on a large medical dataset (if available) perform even better? A brief discussion on the limitations imposed by the teacher's domain (natural images vs. medical) would strengthen the paper.

"Simplicity" as a Double-Edged Sword: While the architectural simplicity is a strength, some might argue that the individual components (knowledge distillation, self-distillation, attention) are themselves not new. The novelty lies in their specific combination for the FAD task. This is a minor point, as the combination is non-trivial and highly effective, but it is worth noting that the paper's contribution is more about a novel framework than a fundamentally new algorithm.

### Questions
Regarding the "Learn-to-Weight" Mechanism: This is an interesting component. Could you provide some qualitative analysis to help understand what it learns? For instance, if you provide a "bad" or less relevant normal image in the support set, does the model learn to assign it a lower weight w_i? Visualizing these weights for a few examples could provide valuable insight.

Impact of Teacher Model: Your results in Table 4 show that WideResNet-50 is the best teacher. This is consistent with many distillation-based anomaly detection papers. However, you also note that very large models like Swin Transformers perform suboptimally. Do you have an intuition for why this "representational gap" occurs? Is it because the student decoder is too simple to distill from such a complex teacher, or is there another reason?


Dataset Details: Thank you for curating this excellent benchmark. Could you clarify if the train/test splits for the leave-one-out protocol are strictly separated by dataset source (e.g., train on HIS, LAG, APTOS, RSNA; test on Brain Tumor) or by anatomy/modality? For instance, if you train on all datasets except Brain Tumor (MRI), would you also exclude other MRI datasets from the training set to test for modality generalization?

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
3

### Summary
This paper proposes $D^24FAD$, a dual-distillation framework for few-shot anomaly detection in medical imaging. The method employs a pre-trained encoder on ImageNet as a teacher and a learnable decoder as a student. The student distills knowledge from the teacher on query images and performs self-distillation on a few normal support images, creating a compact representation of “normal” anatomy for each task. A learn-to-weight module dynamically adjusts the contribution of each support sample conditioned on the query. The approach is trained only on normal data and evaluated in a few-shot setting (2-, 4-, and 8-shot) across five medical datasets spanning different modalities and organs. Experiments show consistently good AUROC results and competitive inference efficiency.

### Strengths
1. The paper introduces a clear and well-motivated dual-distillation framework ($D^24FAD$) for few-shot anomaly detection, combining a teacher–student distillation mechanism with an additional student self-distillation path and a learn-to-weight module that adaptively re-weights support images conditioned on the query. While the core components (knowledge distillation, few-shot learning) are known, their integration into a unified few-shot medical anomaly detection framework is novel and conceptually elegant. The problem formulation, detecting unseen anomalies from only a few normal references, addresses a challenging and underexplored setting in medical imaging.

2. The work is technically solid and empirically well supported. The authors evaluate their approach on five heterogeneous medical datasets and include ablation studies that clearly demonstrate the contribution of each component. The training procedure is reproducible and relies on standard architectures and optimization methods. The model achieves consistently strong performance, often surpassing baselines by a substantial margin, while maintaining efficient inference and low memory consumption. It should be noted, however, that several competing methods (e.g., PatchCore, RD4AD) were originally designed for pixel-level anomaly localization. In this paper they are evaluated only at the image level, which might underestimate their actual capability. Consequently, while the reported results indicate strong performance, the comparative advantage should be interpreted with some caution.

3. The paper is clearly organized and readable. The motivation, methodology, and evaluation are logically connected.

4. The paper targets an important real-world problem, automatic anomaly detection under limited supervision in medical imaging, which has substantial clinical implications.

### Weaknesses
The main limitation of the paper lies in the formulation of the task. Although the work is presented as addressing few-shot anomaly detection, the evaluation is restricted to image-level AUROC, effectively turning the problem into a binary classification task (normal versus abnormal). While the model internally produces anomaly maps, no quantitative localization results are provided (e.g., Dice, IoU, or AUPRO). This simplification reduces the methodological complexity of the problem and limits the demonstrated clinical applicability of the approach, where spatial localization of pathological regions is often essential.

A second concern relates to the evaluation of competing methods. Several baselines included in the comparison, such as PatchCore and RD4AD, were originally designed for pixel-level anomaly localization. Converting these approaches to image-level evaluation may underestimate their true capability and potentially exaggerate the reported relative gains of the proposed model. A clearer justification or separate localization-based comparison would strengthen the empirical claims.

In terms of experimental details, several important aspects remain unspecified. The procedure used to select the weighting coefficient λ (Table 3) is not described; it is unclear whether this was tuned using a validation set or chosen post-hoc on test performance, which risks overfitting on test data. Similarly, the results reported in Figure 3 for inference time and memory usage are not linked to a particular dataset, making the comparison less transparent.

The framework’s architectural flexibility also appears limited. Because the distillation losses are computed layer-wise between teacher and student features of matching dimensions, both networks must share a compatible structure. The paper does not mention whether any projection or adaptation modules were used to support heterogeneous architectures.

Finally, the analysis remains confined to a teacher pretrained on ImageNet, which introduces a potential domain mismatch for medical images. The effect of employing a medical-domain pretrained backbone (e.g. BioMedCLIP) is not investigated. Such an analysis would provide valuable insight into the robustness and domain generalization of the proposed framework.

### Questions
1.Task formulation and localization:
Can you provide quantitative results for anomaly localization (e.g., Dice, IoU) to demonstrate the spatial detection ability of your method? If such annotations are not available, could you at least show quantitative proxy metrics or additional qualitative evidence supporting the localization quality shown in Appendix B?

2. Evaluation of segmentation-based baselines:
Several baselines such as PatchCore and RD4AD were originally designed for pixel-level anomaly localization. How were these methods adapted to the image-level AUROC evaluation used in this paper? Could you clarify whether their localization outputs were averaged or thresholded, and how this might affect the reported results?

3. Weight coefficient λ selection:
How was the optimal value of λ = 0.1 (Table 3) determined? Was there a validation set or a held-out meta-task used for tuning, or was the selection based on post-hoc test performance? A clear description of this procedure would help assess the fairness of comparisons.

4. Inference time and memory usage (Figure 3):
For which dataset(s) were the inference time and memory measurements obtained? Are the reported values averaged across datasets or computed on a single representative dataset? 

5. Architectural flexibility:
Does the proposed framework require the teacher and student networks to share an identical architecture and layer resolution? If not, are there any projection or adaptation modules that enable distillation between heterogeneous backbones?

6. Pretraining domain and robustness:
Have you considered evaluating the method with a medical-domain pretrained teacher network (e.g.,  BioMedCLIP)? Such experiments would clarify whether the reliance on ImageNet pretraining affects the generalization ability across medical modalities.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a dual-distillation framework for few-shot anomaly detection in medical images. The method freezes an ImageNet-pretrained encoder as a “teacher” and trains a lightweight decoder as a “student,” combining (1) feature distillation on query images and (2) self-distillation on support images. A 5-dataset benchmark (13,084 images) is introduced, and the method is shown to outperform prior work in image-level AUROC under 2/4/8-shot settings.

### Strengths
- Clear motivation and task definition for few-shot anomaly detection in medical settings.
- Architecture is simple, fast, and avoids large generative models.
- Strong image-level AUROC across multiple datasets and shot settings.

### Weaknesses
- The work repeatedly emphasizes “dual distillation” as a key contribution, but the process does not fully match established definitions of distillation in the literature. Since the teacher network is frozen, and the student is not learning logits or semantic knowledge but merely reconstructing features, the term distillation may be overstated. This weakens the conceptual positioning of the contribution: the method is an anomaly-detection reconstruction framework rather than a genuine knowledge-transfer framework. A more precise formulation would strengthen the paper’s technical clarity.
- The fairness of the baseline comparison is still unclear. Although it is likely that MediCLIP, MVFA, and INP-Former are evaluated using their official pretrained checkpoints followed by few-shot inference (rather than re-trained from scratch), this introduces a different concern: the pretraining data used for these models may not be aligned with the data available to the proposed method. Since these baselines were originally trained on large external datasets (e.g., CLIP pretraining or full normal medical datasets), it is important to clarify whether (1) their pretraining data overlaps with the test datasets, and (2) whether the proposed method has access to comparable pretraining resources. Without such clarification, the performance difference may reflect differences in dataset exposure rather than architectural effectiveness.
- Only image-level detection is evaluated, no localization metrics. Pixel-level anomaly localization (e.g., heatmaps, PRO, Dice, pixel-AUROC) is essential for clinical use. Several baselines do support localization (e.g., MVFA, AnomalyGPT), but the paper does not report or discuss localization performance, making the clinical impact claim incomplete.
- Inconsistent reporting of “Ours” across tables. The AUROC results for the proposed method in Table 1 are lower than those in the final row of Table 2, even though both appear to evaluate the same benchmark. This suggests the two tables use different configurations (e.g., stronger backbone, more loss terms, or ablation-optimized settings), but this is not stated anywhere. It becomes unclear which version represents the “main result,” and whether the strongest performance was omitted from the primary comparison.

### Questions
- Clarify the evaluation setting for MediCLIP, MVFA, and INP-Former
- Does the method support pixel-level localization? If yes, why are localization metrics not reported?
- Why does Table 1 report lower performance than the last row of Table 2? 
- Have you experimented with stronger pretrained teachers (e.g., CLIP-ViT) instead of ImageNet CNNs?

### Soundness
3

### Presentation
3

### Contribution
3
