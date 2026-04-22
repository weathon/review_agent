# OD$^3$: Optimization-free Dataset Distillation for Object Detection

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Training large neural networks on large-scale datasets requires substantial computational resources, particularly for dense prediction tasks such as object detection. Although dataset distillation (DD) has been proposed to alleviate these demands by synthesizing compact datasets from larger ones, most existing work focuses solely on image classification, leaving the more complex detection setting largely unexplored. In this paper, we introduce OD$^3$, a novel optimization-free data distillation framework specifically designed for object detection. Our approach involves two stages: first, a candidate selection process in which object instances are iteratively placed in synthesized images based on their suitable locations, and second, a candidate screening process using a pre-trained observer model to remove low-confidence objects. We perform our data synthesis framework on MS COCO and PASCAL VOC, two popular detection datasets, with compression ratios ranging from 0.25% to 5%. Compared to the prior solely existing dataset distillation method on detection and conventional core set selection methods, OD$^3$ delivers superior accuracy, establishes new state-of-the-art results, surpassing prior best method by more than 14% on COCO mAP$_{50}$ at a compression ratio of 1.0%. Code is available at https://github.com/VILA-Lab/OD3.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
OD3 is an optimization-free dataset distillation framework for object detection that synthesizes compact datasets by iteratively placing and screening object instances using a pre-trained observer model. It achieves state-of-the-art performance, surpassing prior methods by over 14% in mAP50 on COCO at 1% compression ratio.

### Strengths
1.OD3 achieves strong performance with clear speed advantages—dataset synthesis on COCO takes ~4.7 hours on a single GPU, significantly faster than optimization-based alternatives.

2.The SA-DCE component provides a practical way to retain contextual cues for small objects, offering a useful insight for dataset distillation in detection.

### Weaknesses
1. The synthesis relies on instance copy-pasting, leading to a domain gap between synthetic and real images that may limit generalization.

2. The method depends on a pre-trained observer model; if this model is biased (e.g., performs poorly on certain categories), it may filter out valid instances and introduce distillation bias.

3. Performance at higher compression ratios (e.g., 5%，10%，20%，100%) is not explored, leaving the scalability and upper performance bound unclear.

4. Backgrounds are randomly sampled from the original dataset, which can create semantically implausible scenes (e.g., indoor objects placed on outdoor backgrounds).

### Questions
In Weaknesses

### Soundness
3

### Presentation
3

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
This paper proposes OD³, an optimization-free framework for dataset distillation in object detection. The method synthesizes a compact dataset by strategically selecting and placing object instances from the original dataset onto blank canvases and then screening them using a pre-trained observer model.

### Strengths
The proposed "optimization-free" paradigm, which employs a rule-based placement and screening strategy, enables extremely fast dataset synthesis and significantly reduces the computational overhead of the distillation process itself.

### Weaknesses
1. The screening module directly removes objects deemed low-confidence by the observer model. However, these low-confidence objects could precisely be the "hard examples"—such as rare categories, heavily occluded, or unusually shaped objects—that are crucial for training a robust detector.
﻿
2. The final optimization objective does not account for inter-class balance and intra-class diversity, both of which are critical for the object detection task. The current metrics for Information Density (Φ) and Diversity (N) are too coarse and may lead to a distilled dataset that is biased towards dominant or easy-to-detect classes.

﻿3.  The core of the method involves placing objects randomly on a canvas with only an overlap ratio as a constraint. This ignores the spatial layout relationships, relative scales, and co-occurrence contexts of objects in the real world. Consequently, the synthesized images may lack semantic realism, which could limit the detector's ability to learn high-level contextual reasoning.

### Questions
see Weaknesses.

### Soundness
3

### Presentation
4

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
This paper introduces $\text{OD}^3$ (Optimization-free Dataset Distillation for Detection), a novel framework explicitly designed to address the unique challenge of synthesizing small, high-fidelity datasets for object detection, a task more complex than image classification due to its spatial and semantic demands. $\text{OD}^3$ leverages a training-free, two-stage process—candidate selection followed by screening using a pre-trained "observer" model—to strategically reconstruct diverse, labeled training images without complex optimization. Evaluations on MS COCO and PASCAL VOC demonstrate that $\text{OD}^3$ effectively achieves significant dataset size reduction (up to $99.75\%$) while successfully maintaining the accuracy of object detection models.

### Strengths
1. This framework employs a clever two-stage design that combines iterative candidate selection with pre-trained observer model filtering. This approach avoids the complex and time-consuming optimization processes typical of traditional DD methods, making the dataset distillation process more efficient, straightforward, and easier to deploy.

2. $\text{OD}^3$ achieves extremely high compression ratios (up to 99.75%) on MS COCO and PASCAL VOC while maintaining the accuracy of trained models, providing an efficient solution for saving computational and data resources when training object detectors.

### Weaknesses
1. The second contribution in the introduction extends the concept of dataset distillation from the relatively well-explored domain of image classification to the more challenging task of object detection. This has already been achieved by DCOD.
2. Both DCOD and OD3 experience substantial performance degradation compared to their uncompressed counterparts. At compression rates of 5% or even 10%, it remains unclear how much additional compression would be necessary to approach the performance of the uncompressed models.

### Questions
1. Tables 1 and 2 should include upper bounds to illustrate the potential for improvement in existing methods within the field of DD for object detection.
2. Table 4 indicates that the best results are obtained when the observer and target detectors share the same architecture. Employing higher-performance detectors does not enhance the distillation effect. Therefore, it would be insightful to compare the performance of student models that also adopt DETR or other transformer-based architectures—that is, models of similar design.

### Soundness
3

### Presentation
3

### Contribution
3
