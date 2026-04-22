# OBI CHARiot: Full-page OBI Rubbing Segmentation with Dual Data Flywheels

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Oracle Bone Inscriptions (OBI), one of the earliest mature writing systems globally, serve as a crucial carrier of human civilization. 
However, directly segmenting OBI characters from full-page rubbings remains unexplored, primarily due to the scarcity of high-quality annotated data.
To address this challenge, we propose a two-stage training framework, named OBI CHARiot, which establishes a cycle of mutual improvement between model performance and data quality. In the first stage, a data flywheel mechanism is employed to iteratively train a SAM2 while automatically aligning  existing low-quality annotations, rather than directly utilizing the model's initial predictions. In the second stage, we employ an iterative strategy on a large collection of unlabeled rubbings, integrating automatic annotation with continuous model refinement.
For reliable evaluation, we invite domain experts to annotate 2,226 rubbings, resulting a test set OBIMDTest.
Experimental results demonstrate that OBI CHARiot offers advantages in both model performance and data quality. Specifically, training SAM2 with our framework yields a remarkable 14.99% gain in mask AP$_{50}$ over the baseline trained on raw data. Similarly, various off-the-shelf instance segmentation models exhibit improved performance when trained on data processed by OBI CHARiot. Moreover, the characters segmented by our framework yield a 22.75% improvement in top-1 accuracy on the downstream deciphering task. These findings confirm the significant potential of OBI CHARiot to advance OBI research. To support future studies, our model and the processed data will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This article proposes OBI CHARiot, a segmentation framework based on SAM for the oracle bone inscriptions on full-page rubbings, which follows a two-stage data collection-training strategy. A high-quality test set termed OBIMDTest containing 2,226 rubbings is built. Experiments show that models trained via OBI CHARiot outperform those trained with other frameworks. Meanwhile, OBI CHARiot achieves significant improvement in top-1 accuracy for the downstream OBI deciphering task.

### Strengths
1. Reasonable motivations from character-level denoising to full page-level segmentation task.
2. An semiautomatic framework to iteratively train and refine the model.
3. The experimental results have shown a decent improvement.

### Weaknesses
1. The technical contributions are relatively limited mostly related to engineering implementation issues. The core of OBI CHARIOT focuses on building the pipeline using SAM. The replaceability of this part is worth discussing to highlight the generality of this framework. Meanwhile, the contribution of the dataset part mainly focused on the modification of OBIMD, and the new contributions were relatively limited.
2. Lack of some details, such as the prompt information for SAM model at different stages of *OBI CHARIOT.*
3. Some illustrations could be more clear, such as the pixel union operation between the outputs of SAM and raw binary facsimiles (add in Fig .2). 
4. I suggest that the illustration in Fig. 2 should connect OBI CHARiot-1 and OBI CHARiot-2. Currently, it is not clear enough, considering the numerous steps mentioned.
5. Missing detailed information about the personnel with expert annotations.
6. The so-called decipherment in the experiment is actually a classification task. This point must not be confused.

### Questions
1. Some writings and typos should be fixed, such as “bulit” in line 79, the necessary space after "i.e.” in line 183, capitalization issue of “Training’s” in line 361, “*prmopt*” in line 356, and grammar issue of “it add” in line 364. Please carefully review the entire text.
2. A direct pixel union operation between the outputs of SAM and raw binary facsimiles in Step-1 may introduce unrelated area to char-level masks? What I understand is to use raw binary facsimiles to fill in the missing pixels in the output of SAM. Any intermediate steps are omitted in the description? 
3. Is there any reason for setting the iou threshold to 0.6?
4. How is the detection performance based on the YOLOv12 model? Will it introduce noise interference to the pipeline?
5. In Fig. 3, what is the relationship between the right image of "Expert-annotated Facsimiles in OBIMDTest" and the binaryized character-level image on the left? It seems that the correspondence between the low-quality part of the OBIMD raw and the binary image is inconsistent.
6. Lack of implementation details for training SAM2 （Sec. 4.1）
7. Lack of comparison with other segmentation models (Tab. 1), except the SAM series.
8. Has there been any comparison regarding the ground-truth? At present, it seems that the results of OBI CHARiot in Fig. 4 are somewhat different from those of other models and the raw data in terms of their outer contour.
9. At present, the qualitative results are a bit limited. It is suggested to add more case visualizations.
10. Lacking quantitative indicators such as PSNR, SSIM and other image generation metrics, the outputs of the denoising model and the OBICHARiot model were compared to provide numerical supplements as shown in Figure 5.
11. Only one denoising model Segformer has been compared*.* It is suggested to include the denoising models released in the past two years (e.g. CharFormer, ) as well as the models specifically designed for Oracle denoising.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the task of stroke-level segmentation in oracle bone script. The authors observe that previous segmentation models typically take single-character images as input, thereby failing to effectively leverage background information that might improve segmentation. They instead propose feeding page-level images into the segmentation model.

However, prior work lacked precise, page-level annotations, only coarse annotations were available. To address this, the authors design a data annotation pipeline similar to the one in Segment Anything Model (SAM): starting from coarse annotations, they train a model, use the trained model to generate slightly more refined labels for the training data, and then iteratively refine both the model and its pseudo-labels to achieve increasingly accurate page-level annotations. The iteratively optimized annotations, according to the authors, yield models that outperform those trained on the original coarse labels. In addition, the authors manually refine the coarse annotations of the test set to create high-quality ground truth.

### Strengths
1. This paper proposes replacing single-character inputs with page-level inputs, enabling the model to utilize background cues, which improves segmentation performance.
2. This paper adapts a SAM-style automatic annotation pipeline to the oracle bone script segmentation task, using iterative pseudo-supervised training to improve annotation accuracy; annotations refined in this way yield slightly better model performance compared to the original coarse annotations.
3. This paper manually curated precise annotations for the test set and plan to release them publicly.

### Weaknesses
1. The idea of using page-level images to leverage background information is straightforward. The proposed automatic annotation pipeline is essentially the same as the pseudo-supervised iterative training described in the SAM paper, with only minor engineering adjustments for oracle bone script segmentation. The work is largely an engineering application with minimal conceptual novelty.

2. The paper claims that the optimized SAM-based pipeline produces better annotations than the original SAM pipeline, but does not clearly explain the specific implementation differences between the two. A clear side-by-side comparison (in either figures or text) is needed to highlight how they differ.

3. In lines 361–366, the analysis attributes the performance gap between the original SAM pipeline and the authors’ optimized version to the latter’s ability to use existing coarse bounding boxes to filter out inaccurate predictions. However, the original SAM pipeline did not use this information simply because its target images were unlabelled Internet data without any coarse boxes, not because this use of box information was a non-obvious idea. When coarse bounding boxes are available, filtering with them is straightforward and should not be considered a principal innovation.

4. According to Figure 4, the manual refinement of the test set annotations mainly involved removing distracting shell-edge background artifacts. Given that the dataset already has character bounding-box annotations, this could be achieved trivially by zeroing out all pixels outside the boxes. Thus, Figure 4 suggests that these background artifacts could be eliminated by a simple rule without the need for manual intervention. While manual refinement is of course more precise than automatic processing, the paper does not convincingly demonstrate that the improvement over simple automatic filtering is significant or necessary.

### Questions
In Figure 2, the IoU threshold for the alignment step in the automatic annotation pipeline is set to 0.6. How sensitive is the result to this choice? What impact would using different IoU thresholds have?

### Soundness
2

### Presentation
2

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
Aiming at the defects of traditional paradigms and dataset issues in the field of Oracle Bone Inscription (OBI) rubbing segmentation, this paper proposes a new full-page segmentation paradigm and a two-stage OBI CHARiot framework. It optimizes data quality and scale through a data flywheel mechanism, constructs a high-quality test set, and experimental verification shows that it has significant advantages in segmentation performance and downstream tasks. Additionally, the publicly available resources facilitate research in this field. Overall, the design is reasonable, with strong innovation and practicality.

### Strengths
1. It breaks through the traditional "character-level denoising" paradigm for Oracle Bone Inscription (OBI) rubbings and proposes a new "full-page rubbing segmentation" paradigm to synergistically preserve character structures and full-page contextual information, filling the gap in this field.  
2. Additionally, it constructs the expert-annotated OBIMDTest dataset, subdivides evaluation scenarios, and innovates the evaluation system.
The research quality is solid: the method features strong implementability and targeted handling of rubbing noise, while the experimental design is reliable with multi-dimensional comparative verification.
3. The expression is highly clear: the technical details are explicit and supplemented with figures and tables, and the information transparency is high.
4. Academically, it provides a transferable framework for processing low-quality and sparsely labeled data in specific domains, filling relevant research gaps, and offering benchmark references. In terms of application, it aligns with the practical needs of OBI decipherment, the generated outcomes can assist experts in their work, and the publicly released resources promote the improvement of digital research efficiency and technical standardization in this field.

### Weaknesses
1. The paper mentions achieving iterative alignment through SAM-rub (for processing rubbings) and SAM-fac (for processing facsimiles), yet it fails to clarify the feature interaction mechanism between the two models (e.g., whether they share feature layers, how to resolve alignment biases caused by modal differences) and does not explain how edge distortion is avoided during the mask resolution enhancement process
2. The parameters of the data flywheel lack optimization verification: the impact of iteration count and IoU threshold (currently set to 0.6) on performance has not been explored, making it impossible to determine the optimal configuration and convergence conditions. It is recommended to supplement comparative experiments under different parameters and provide the basis for parameter selection 
3. In the second stage, YOLOv12 is used to generate character prompt boxes for SAM, but the paper does not explain the impact of YOLOv12's detection accuracy on SAM's segmentation results (e.g., how missed detection/false detection boxes are handled) nor clarify whether a prompt box correction mechanism is designed. This may cast doubt on the reliability of automatic annotation for unlabeled data
4. The coverage of comparative models is insufficient: models suitable for low-data/small-sample segmentation scenarios and lightweight Transformer-based models are not included, and only ResNet50 is used as the classifier in downstream tasks. The demonstration of compatibility and universality is inadequate. It is recommended to supplement comparative experiments with such models to verify the framework's advantages
5. Details on noise handling are lacking: the paper does not elaborate on how to distinguish between character noise and non-character interference (e.g., boundary lines) nor verify noise resistance. It is recommended to supplement technical details of noise handling and experimental results on high-noise subsets
6. The zero-shot verification scenario is single: only HuaDong rubbings are used for evaluation, without covering rubbings from different periods or produced via different processes, making it impossible to demonstrate broad generalization ability. It is recommended to supplement zero-shot test sets from multiple sources and analyze the causes of performance degradation

### Questions
1. For the dual SAMs (SAM-rub and SAM-fac), the feature interaction mechanism and the specific structure of the deconvolution layer are not clarified, nor is it explained how edge distortion is avoided during resolution enhancement. Could you supplement the schematic diagram of model interaction and the edge fidelity solution?
2. In the second stage, the detection accuracy (miss rate, false detection rate) of YOLOv12 and the impact of detection errors on SAM segmentation are not mentioned. Could you provide the mAP index of YOLOv12 and the prompt box correction strategy (if any)? 
3. For the termination of the data flywheel iteration, only "almost no additional aligned data" is mentioned, with no quantitative standard. Could you specify the termination index (e.g., the proportion of new additions in N consecutive rounds < X%) and the basis for its selection? 
4. Whether a multi-task loss function is used for full-page facsimile and character mask generation, and how the loss weights are allocated, are not explained. Could you supplement the design of the loss function and the impact of weights on the performance of the two tasks?

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
4

### Summary
This paper proposes OBI CHARiot, a two-stage framework for full-page Oracle Bone Inscription (OBI) rubbing segmentation. It introduces a new paradigm that captures both character structure and contextual information. The first stage aligns misaligned rubbing–facsimile pairs through an iterative data flywheel process, while the second expands the dataset using automated annotation of unlabelled rubbings. A new expert-annotated test set, OBIMDTest, is also built. Experiments show large gains in segmentation accuracy and downstream OBI deciphering, demonstrating the framework’s effectiveness in improving both model performance and data quality.

### Strengths
1. The paper accurately identifies the core problem with the existing OBI dataset (OBIMD)—namely, the pixel-level misalignment between facsimiles and rubbings—and the limitation of the existing paradigm (character-level denoising) in losing context.

2. The OBI CHARiot framework is a clever, multi-stage solution. The Stage 1 data flywheel for correcting misalignment and the Stage 2 data flywheel for expanding the dataset is a "fix-then-enhance" strategy that is highly logical and effective.

3. The authors invested the effort to create an expert-annotated test set (OBIMDTest). This greatly enhances the credibility of the experimental results, as the evaluation no longer depends on the low-quality labels of the original OBIMD dataset.

### Weaknesses
1. In Stage 2, the authors use a YOLOv12 model to generate bounding boxes as prompts for SAM. The justification is that OBI characters are too sparse for SAM's default grid-based prompting to work. This is a strong claim, but the paper does not seem to provide experimental data (e.g., the recall of SAM's "segment everything" mode) to fully support this choice. This adds an extra model dependency.

2. The paper fails to specify the exact number of iterations for the data flywheel mechanism, and the selection of hyperparameters such as IoU appears rather arbitrary. The authors should provide a more rational explanation and conduct experiments to validate the choices of these hyperparameters.

### Questions
1. The paper mentions using SAM2 (Ravi et al., 2024) in Section 4.1 (Compared Training Frameworks), but primarily cites SAM (Kirillov et al., 2023) in the methodology (Section 3). Was the OBI CHARiot framework built on SAM or SAM2? Please clarify.

2. The use of YOLOv12 in Stage 2 is justified by SAM's grid-prompting failing due to pixel sparsity. Could you provide a brief ablation or data to support this? For instance, what is the recall of SAM's "segment everything" mode on the OBIMDTest set?

3. How many iterations did the data flywheels in Stage 1 and Stage 2 each require to converge? Furthermore, the 0.6 IoU threshold for alignment in Stage 1 is a hyperparameter; how sensitive is the model's performance to this threshold?

### Soundness
2

### Presentation
2

### Contribution
3
