# Contrast-Aware Calibration for Fine-Tuned CLIP: Leveraging Image-Text Alignment

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Vision-language models (VLMs), such as CLIP, have demonstrated exceptional generalization capabilities and can quickly adapt to downstream tasks through prompt tuning. Unfortunately, in classification tasks involving non-training classes, fine-tuned VLMs often overfit to train classes, resulting in a misalignment between confidence scores and actual accuracy on unseen classes, which significantly undermines their reliability in real-world deployments. Existing confidence calibration methods typically require training parameters or analyzing features from the training dataset, restricting their ability to generalize unseen classes without corresponding train data. Moreover, VLM-specific calibration methods rely solely on text features from train classes as calibration indicators, which inherently limits their ability to calibrate train classes and other evaluation settings, like cross-dataset and domain-generalization settings. To address these challenges, we propose a multimodal calibration method $\textbf{Contrast-Aware Calibration (CAC)}$. Building on the original CLIP's zero-shot adaptability and the conclusion from empirical analysis that poor intra-class and inter-class discriminative ability on unseen classes is the root cause, we calculate calibration weights based on the contrastive difference between the original and fine-tuned CLIP. This method is not only effective for calibrating unseen classes but also overcomes the limitations of previous VLM calibration methods that struggle to calibrate train classes and other settings. In multiple setting experiments with 5 fine-tuning methods, CAC achieves strong calibration in all settings without sacrificing accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Contrast-Aware Calibration (CAC) for confidence calibration in fine-tuned CLIP.  The authors first point out that fine-tuned CLIP suffers from miscalibration and attribute miscalibration on unseen classes to weakened inter-class and intra-class discrimination, which is quantified by a “contrast” metric derived from logits. Motivated by this phenomenon, the authors propose CAC to rescale fine-tuned logits using a weight computed from the L1 distance between original CLIP and fine-tuned logits. Extensive experiments on 11 datasets show that CAC reduces calibration error without changing accuracy and with negligible overhead.

### Strengths
1.	The motivation is clear. The author presents a clear relation from contrast to calibration.
2.	The experimental results are promising. CAC effectively calibrates train classes, which outperform other calibration methods in Table 3. Moreover, A key strength is that CAC addresses a major limitation of previous DAC, which only applies to unseen classes.

### Weaknesses
1.	The connection between motivation and method seems weak. The paper invests significant effort to define a "Contrast Metric" in Section 3.1 and 3.2. However, the proposed method does not use this metric.
2.	The design of hyperparameters is heuristic, especially the piecewise thresholds. As shown in Table 5, CAC could show worse calibration than baseline without any hyperparameter. Such complexity weakens the contribution of CAC. 
3.	The ablation for CAC is not enough. Is CAC sensitive to the metric for distance measurement in Equation 3?

### Questions
1.	Does the calibrated CLIP show a better degree of contrast?
2.	Can CAC be combined with training-based methods for better CLIP calibration [1-2]?

[1] Towards calibrated robust fine-tuning of vision-language models. NeurlPS, 2024.

[2] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models. ICML, 2025.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a contrastive perception calibration method to address the issue of poor confidence calibration on unseen categories in fine-tuned CLIP models. The method leverages the discrepancy between the outputs of the original and fine-tuned CLIP models to design a training-free post-hoc calibration strategy, which improves calibration performance across various settings, including open-vocabulary recognition, cross-dataset evaluation, and domain generalization, without compromising model accuracy.

### Strengths
1. The overall presentation is clear and easy to follow;

2. The proposed plug-and-play component can be seamlessly integrated into vision-language models;

3. The method achieves promising results across multiple benchmarks and evaluation settings;

### Weaknesses
1. The use of an exponential function for weight adjustment is largely based on empirical observation and engineering heuristics, lacking deeper exploration of the function space or theoretical motivation. In fact, there exist numerous weight adjustment and parameter update mechanisms, especially within optimizers.

2. Exploration of intra-class and inter-class relationships has been studied in many prior works. While the authors implement this from a confidence-based perspective, the strategy is, in essence, similar to uncertainty-aware or uncertainty-estimation approaches.

3. The authors emphasize that CAC is insensitive to hyperparameters and adopt default values (k=15, α=1.10). However, the ablation studies in the appendix (e.g., Tables 10 and 11) indicate that performance does fluctuate with different values of k and α. For instance, when k=5 or 25, the ECE results deteriorate noticeably. Therefore, the claim of “hyperparameter insensitivity” may need to be stated more cautiously.

4. This work primarily focuses on CLIP and its prompt-learning variants. However, the development of vision-language models has now entered the era of multimodal large language models, such as LLaVA. The paper could broaden its perspective in the conclusion and future work by discussing whether and how the CAC approach, which leverages the alignment capabilities of a foundation model, might be applicable to these more complex, generative model architectures.

### Questions
Please refer to the Weaknesses part.

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
4

### Summary
This paper targets the over-confidence problem of fine-tuned CLIP on unseen classes in open-vocabulary scenarios. It proposes a training-free, plug-and-play multimodal post-processing method called Contrast-Aware Calibration (CAC). The core idea is to treat the logit-level contrastive difference between the original CLIP and the fine-tuned CLIP as a miscalibration score, transform it into sample-specific scaling weights via an exponential function, and re-scale the temperature of the fine-tuned model accordingly. In this way, CAC simultaneously calibrates both training and unseen classes.

### Strengths
1. Originality: For the first time, the paper uses the “contrastive difference” itself as a direct miscalibration measure and establishes its negative correlation with ECE; it achieves sample-level calibration without any training data or validation set.
2. Quality: The derivation is rigorous, and ablation studies are comprehensive.
3. Clarity: The paper is well-structured; figures and tables intuitively verify the effectiveness of the proposed method.
4. Significance: It removes the limitation that existing VLM calibration methods can only handle either training classes or in-distribution unseen classes, providing a unified and trustworthy confidence output for open-vocabulary deployment.

### Weaknesses
1. The method seems to assume that the original CLIP is well-calibrated. If the pre-trained model itself has systematic bias, CAC may amplify this bias.
2. Experiments are conducted only on ViT-B/16, ViT-B/32, and RN50; larger backbones such as ViT-L/14 or other VLMs are not covered.
3. End-to-end evaluations in concrete application scenarios are missing; only classification ECE is reported.
4. Although the default parameters work well, the involved hyper-parameters still appear to be manually tuned, and no automatic selection strategy is provided.

### Questions
1. If the original CLIP itself is under-calibrated on certain domains, will CAC pull the fine-tuned model toward even worse confidence?
2. Performance varies greatly across different CLIP sizes; could you provide comparative results on larger CLIP variants?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a calibration method called Contrast-Aware Calibration (CAC) for CLIP models. Specifically, CAC calculates the confidence difference between the original and fine-tuned CLIP models, which is then scaled and transformed using an exponential function. The resulting CAC score is employed to reweight the logits for improved calibration. Experimental results demonstrate that CAC enhances the performance of various fine-tuned CLIP models.

### Strengths
1. The paper presents extensive experimental results across different prompt-learning methods and datasets, as detailed in Appendix E.

2. Experimental results demonstrate the effectiveness of proposed method in multiple datasets.

### Weaknesses
1.  The claim is not well supported with evidence. For instance, the authors claim that over-confidence and under-confidence arise from high or low inter-class and intra-class similarity (Lines 187-197). However, Figure 1(a) does not show the inter-class or intra-class similarity across different datasets. It only illustrates the phenomenon of over-confidence and under-confidence on two specific datasets. Furthermore, the authors claim that contrast and ECE exhibit a negative correlation for unseen classes (Line 202). It’s unclear how this conclusion follows from Figure 1(c-d). Did the authors conduct experiments exclusively on unseen classes for these figures? This point needs clarification.

2.  In Eq (3), z represents the confidence difference between the original and fine-tuned VLMs. However, the relationship between this equation and the earlier claim that over-confidence and under-confidence arise from high or low inter-class/intra-class similarity is not well explained. A more detailed explanation of this connection would help clarify the authors' argument.

3.  In Lines 242-243, the authors state that "the logits of … are equivalent to the contrast metric." However, logits are vectors in the N-way classification problem, whereas the contrast metric in Eq (1) is a constant. The illustration is not entirely accurate and needs refinement for consistency.

4.  The performance of the proposed CAC and advanced CAC methods is highly sensitive to the values of hyper-parameters. As shown in Tables 11-12 (Appendix D), small variations in hyper-parameters can significantly impact performance. Additionally, CAC (and advanced CAC) uses four different hyper-parameters (k, $\alpha$, $\lambda_1$, $\lambda_2$), which limits the method's general applicability.

### Questions
See my questions in weakness

### Soundness
2

### Presentation
1

### Contribution
2
