# Zero-shot Quantization for Object Detection

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Zero-shot quantization (ZSQ) has achieved remarkable success in classification tasks by leveraging synthetic data for network quantization without accessing the original training data. However, when applied to object detection networks, current ZSQ methods fail due to the inherent complexity of the task, which encompasses both localization and classification challenges. On the one hand, the precise location and size of objects within the samples for object detection remain unknown and elusive in zero-shot scenarios, precluding artificial reconstruction without ground-truth information. On the other hand, object detection datasets typically exhibit category imbalance, and random category sampling methods designed for classification tasks cannot capture this information.
To tackle these challenges, we propose a novel ZSQ framework specifically tailored for object detection. The proposed framework comprises two key steps: First, we employ a novel bounding box and category sampling strategy in the calibration set generation process to infer the original training data from a pre-trained detection network and reconstruct the location, size and category distribution of objects within the data without any prior knowledge. Second, we incorporate feature-level alignment into the Quantization Aware Training (QAT) process, further amplifying its efficacy through the integration of feature-level distillation.
Extensive experiments conducted on the MS-COCO and Pascal VOC datasets demonstrate the efficiency and state-of-the-art performance of our method in low-bit-width quantization. For instance, when quantizing YOLOv5-m to 5-bit, we achieve a 4.2\% improvement in the mAP metric, utilizing only about 1/60 of the calibration data required by commonly used LSQ trained with full trainset.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work only uses the pre-trained object detection network to reconstruct the training samples without accessing any original data or labels, and realizes zero-shot quantization of the object detection network.

### Strengths
1. The problem raised by the paper is clearly expressed and the motivation is clearly stated.
2. This paper demonstrates the effectiveness of the proposed method through the extensive experiments.

### Weaknesses
Although the paper has demonstrated the effectiveness of the method in experiments, the explanation of the proposed method is difficult to follow.
1. In Section 3.2 and ‘Relabel’ paragraph, The supervision mentioned in formula 5 is done by GT. In my understanding, the role of GT is replaced by the generation of the "teacher" model. In the initial stage, all information such as image and position are initialized as Gaussian noise. How does the teacher model obtain information to complete this optimization? I still feel unclear about the establishment process of this optimization.
2. After this sentence, the author only mentioned that this is the 'teacher' model. Is it the model itself that is being quantized or is it a model with other requirements? The author did not elaborate.
3. It is recommended that the author add a detailed explanation of f_l after formula 7. What specific layers does the author distill?
4. In appendix C, the author only compared the amount of data and the cost of the training process. What is the time consumption of generating samples in the first stage?

### Questions
In page 1, the paper mentions that ‘Classification networks require only a randomly sampled category id label for data synthesizing’, which means that classification task also samples the gt labels. Why can not detection task methods take gt as label for location and size generation?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel zero-shot quantization (ZSQ) framework designed specifically for object detection tasks. Unlike existing ZSQ approaches that are mainly suited for classification, this framework tackles the unique challenges of object detection, including localization and class imbalance. The framework involves two main components: (1) a bounding box and category sampling method for generating a synthetic calibration set, which approximates the spatial and categorical distribution of the objects without any real data, and (2) a quantization-aware training (QAT) process that integrates feature-level distillation for effective knowledge transfer. The proposed method achieves impressive results, outperforming baselines like LSQ on MS-COCO and Pascal VOC with only 1/60 of the data size typically required.

### Strengths
1. The bounding box and category sampling strategy provide a data-efficient way to approximate real-world object distributions, making it suitable for privacy-sensitive applications.
2. Integrating feature-level distillation into the QAT stage improves knowledge transfer and performance in low-bit quantization settings, significantly narrowing the performance gap between full-precision and quantized networks.
3. The framework demonstrates superior performance on both MS-COCO and Pascal VOC datasets, even in ultra-low-bit scenarios, thus highlighting its robustness and practical utility.
4. By requiring only a fraction of the typical calibration data size, the method enhances scalability for deployment on resource-constrained devices.

### Weaknesses
1. The framework combines synthetic data generation, adaptive sampling, and feature-level distillation within quantization-aware training (QAT), creating a complex pipeline. This complexity may hinder practical adoption and implementation in real-world settings.
2. The framework relies heavily on the quality of synthetic data for calibration, which may introduce variability in performance, especially in challenging real-world environments. Ensuring consistency in synthetic data quality is critical for stable results.
3. While the paper demonstrates strong results with YOLOv5 and Mask R-CNN, it lacks testing across a wider range of object detection architectures. Broader experimentation would strengthen confidence in the framework's generalizability.
4. The method's reliance on synthetic category distribution for handling class imbalance may not fully capture the complexities of real-world distributions. This could impact performance in scenarios with highly skewed class distributions.
5. Although the paper suggests that a small calibration set suffices, a more granular analysis of the calibration set size across different bit-widths could provide insights into optimal data requirements and help streamline data efficiency.
6. The method's effectiveness largely depends on feature-level and prediction-matching distillation. Over-reliance on distillation may limit its flexibility for future developments in quantization techniques that don’t use distillation.

### Questions
Please refer to the Weaknesses box.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a method for solving the problem of Network Quantization for Object Detection in the zero-shot setting. The contribution of this paper is an adaptive sampling method (Relabel), which is used in the Synthetic data generation phase of the zero-shot setting. The main idea of Relabel is to use the prediction from the pretrained full-precision detection model to refine the bounding box/ class label during the optimization of Synthetic data generation. With the adaptive sampling strategy, the model performance can be improved while using less synthetic data than the baselines.

### Strengths
- Except for the points pointed out in the weakness section, this paper has a clear structure, and running the experiments/ablation study to validate the effectiveness of the proposed methods.

- The idea of Relabel is good, and can effectively improve the generation quality of the Zero-shot Quantization method.

- The paper claims this strategy can reduce the number of samples, thus reducing the training time, which is essential for the Quantization problem.

### Weaknesses
**Clarity**: There are some claims that need to clarify for the experiment/qualitative/quantitative analysis:
-  In L78 and L79, the paper claims category imbalance in the current detection dataset, can you provide some examples illustrating that the previous method is struggling with it? Also, can you provide visualization/analysis your method is capable of handling the imbalance problems? (For example, using some metric to show the class distribution of the generated samples of the previous method and your method.)

- In L80 and L81, the paper claims "fine-tuning strategy for zero-shot quantized detection network with synthetic calibration data has not been studied". Can the author clarify more about this claim, since [1] also fine-tuning from synthetic data? What is the difference between your fine-tuning approach compared to [1] and how you can improve from [1]?

**Experiment**:
- In the main paper, the paper should include the qualitative results of the synthetic data, compared with the baseline and [1], and also include the qualitative results from the quantized network and the full-precision network.
- Table 5, why the False positive sample from [1] is not included?
- What is the purpose of the weak baseline Gaussian in Table 5?


[1] Akshay Chawla, Hongxu Yin, Pavlo Molchanov, and Jose Alvarez. Data-free knowledge distillation for object detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3289–3298, 2021.

### Questions
All concerns and questions are listed in the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
