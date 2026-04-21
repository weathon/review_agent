# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

- Avg Score: 6.00
- Decision: Reject
- Scores: 5, 8, 5

## Abstract
In this paper, we develop an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. 
The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. 
To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. 
While previous works mainly evaluate open-set object detection on novel categories, we propose to also perform evaluations on referring expression comprehension for objects specified with attributes. 
Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. 
Grounding DINO achieves a $52.5$ AP on the COCO detection zero-shot transfer benchmark, i.e.,  without any training data from COCO. It sets a new record on the ODinW zero-shot benchmark with a mean $26.1$ AP.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose an open-vocabulary object detection network namely Grounding DINO. Based on the original DINO object detector, the authors introduce 1) text encoder and cross-modal feature enhancer to incorporate language-specific feature into image features; and 2) language-guided query selection with cross-modal decoder to detect and recognize objects with language guidance. The proposed Grounding DINO obtains promising results on various zero-shot object detection tasks and referring object detection tasks.

### Strengths
1. This overall writing is polished and easy to understand. 

2. The proposed Grounding DINO shows good generalization ability on various tasks, including zero-shot general object detection and referring object detection.

### Weaknesses
1. The key components of proposed method (e.g., feature enhancer and cross-modality decoder) are not new. The authors should thoroughly discuss the relation and difference of between feature enhancer in GLIP [1] and the counterpart in this work, and conduct the same analysis for cross-modality (with X-Decoder [2]).

2. Though utilizing better detector and better pretrained models, this work does not lead to better performance than previous work, e.g., DetCLIP-V2 [3] (CVPR 23), which obtains much better performance on LVIS benchmark and ODinW benchmark than Grounding DINO. 

[1] Grounded Language-Image Pre-training, CVPR 2022
[2] Generalized Decoding for Pixel, Image and Language, CVPR 2023
[3] DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment, CVPR 2023

### Questions
See weakness section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This study proposes a strong open-set object detector called Grounding DINO. When compared with other open-set detectors like GLIP, Grounding DINO introduces the DINO object detector as its overall architecture. It adopts the Swin transformer as the backbone and the encoder-decoder pipeline to replace the DyHead structure as used in GLIP. More fusions between the textual and visual features in the pipeline are also introduced to enable the model to perform better. Moreover, the Referring Expression Comprehension (REC) tasks are also introduced in its pertaining.

Grounding DINO is evaluated on different tasks, including MS-COCO, LVIS, ODinW, and RefCOCO/+/g. It achieves 52.5 AP zero-shot detection on COCO and also obtains state-of-the-art performance on ODinW with 26.1 AP. Albeit strong in common tasks, Grounding DINO performs worse than GLIP on rare classes on OdinW, and the authors also admit its limitation on its zero-shot ability on REC data without fine-tuning.

### Strengths
This is a thorough study composed of different shining components. 
- First of all, the integration of the DINO object detector pushes forward the performance on different detection benchmarks thanks to the strong feature extraction ability of the new architecture. 
- The generalization of different feature fusion methods is clear and inspiring. The proposed feature enhancer calls for more attention to more important features according to the text input. The language-guided query selection module resembles the top-K operation in Efficient DETR/DINO but is endowed with new physical meanings in this pipeline, as only queries that are closely related to the input text are kept. 
- The introduction of REC tasks in both pertaining and evaluation opens a new gate for open-set object detectors. The definition of traditional open-sent object detection naturally extends to the fields of describing open-set objects using more versatile expressions. 
- The performance reported in this study is competitive. It achieves state-of-the-art performance on the zero-shot ability on MS-COCO and ODinW datasets. 
- The reviewer appreciates the presentation regarding the limitations of this study.

### Weaknesses
- Novelty issues. Despite the effectiveness of the proposed architectures, the composed modules are not innovative enough as their own. For example, the major component that makes a difference should be the DINO architecture that has already set a record on various detection tasks. The query selection is also similar to that of the top-K operators in Efficient DETR/DINO. Although Grounding DINO extends its tasks to referring expression comprehension, it is only comparable to its precedent studies like GLIP and does not introduce specific modules to address the shortcomings on REC. Nevertheless, I would also admit that these novelty issues are not critical as each of them is not trivial in the task.  
- I have some concerns regarding its significance and effectiveness in true open-set scenarios. 
Despite the effectiveness of the zero-shot performance on MS-COCO and ODinW datasets, the performance seems to rely on its pertaining data heavily. As also revealed by the author, Grounding DINO transfers better on common classes, classes that are more likely included in the pertaining data in some format, yet performs worse on rare classes in LVIS. The zero-shot performance on the RefCOCO dataset is significantly lower than that after including RefCOCO in its pertaining data. 
- A closer look into Table 4 would also reveal that the performance on ODinW is actually on par with GLIPv2 if given the same pertaining data. This would also raise a concern about its true ability in true open-set scenarios. 
- Some minor writing issues. e.g., "hard-crafted" -> "hand-crafted" (page 2), "Even though" -> "Even though the performance plateaus with larger input size" (page 7).

### Questions
Overall, this is a nice work and an interesting study. I encourage the authors to respond to the above weaknesses.  Besides, I also have some additional questions regarding the paper and possibly some future work. 
- It is assumed that the query selection module might be the major reason for the low performance on rare classes in LVIS. Have the authors increased the top-K values to validate the assumption?
- As for the limited ability of the REC dataset, is it possible to increase the model size of the BERT text encoder to obtain a better representation of complex text input to alleviate this?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Grounding DINO, an open-set object detector that combines DINO with grounded pre-training. This allows it to detect objects based on human inputs like category names or expressions. 
The innovation lies in using language to enhance closed-set detectors for better open-set detection. The detection process is conceptually divided into three phases, including a feature enhancer, language-guided query selection, and a cross-modality decoder for merging vision and language. 
This work also evaluates referring expression comprehension for attribute-specified objects. Grounding DINO excels in multiple benchmarks, including COCO and ODinW, setting new records like a 52.5 AP on COCO's zero-shot transfer benchmark without using its training data.

### Strengths
1. The paper shows that text and vision fusion and multiple stages helps the model achieve better performance compared to later fusion.
2. The paper shows benefits on multiple settings, including closed-set detection, open-set detection, and referring object detection, to comprehensively evaluate open-set detection performance
3. Grounding DINO outperforms competitors by a large margin and establishes a new state of the art on the ODinW (zero-shot benchmark with a 26.1 mean AP

### Weaknesses
1. More than 900 queries would be interesting to see if the model generalizes well to rare classes as well
2. The ablations in table 6 and 7 do not show strong contribution of individual modeling choices except tight fusion. The authors dont explain this

### Questions
Please refer to weakness 1 and 2

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
