# Strengthen Out-of-Distribution Detection with Uncertainty-Driven Adaptively Rectified Backpropagation

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Out-of-distribution (OOD) detection aims to ensure AI system reliability by detecting inputs outside the training distribution. Recent work shows that overfitting during later stages of training can hurt OOD detection. To overcome overfitting, several methods attempt to distill the model after training or prune the model during training from a model-centric perspective. In contrast, this paper proposes a data-centric end-to-end solution called Uncertainty-driven Adaptively Rectified Backpropagation (UARB), which follows the principle that once the model has mastered an instance, training on it should stop to prevent overfitting. UARB considers an instance mastered if the zero-order and second-order differences of its uncertainty value remain within a small range around zero, offering a more consistent measure of an instance’s learning status. Additionally, since different classes exhibit varying optimization progress, using a fixed threshold to determine when to exclude an instance from backpropagation is theoretically unsound. UARB develops an adaptive threshold by incorporating class-informed statistics to determine when to exclude an instance. Extensive experiments demonstrate that UARB can enhance OOD detection performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This study aims to mitigate overfitting and the consequent decline in OOD detection performance by dynamically excluding samples that the model has already mastered during training. To this end, the authors propose a novel uncertainty-based adaptive data selection strategy UARB, which leverages both zeroth- and second-order uncertainty measures and introduces class-level thresholds as data selection criteria. Comprehensive experiments conducted on both small- and large-scale datasets include evaluations in both far-OOD and near-OOD scenarios, as well as compatibility tests with a wide range of existing methods, consistently demonstrating promising performance.

### Strengths
**Originality:**
Unlike prior methods such as UM, which mitigate overfitting to improve OOD detection performance by performing gradient ascent on the loss over an entire batch of samples, UARB addresses overfitting at a finer, more granular level. Additionally, UARB overcomes the limitation of UM’s reliance on pretrained models, enabling direct end-to-end training.

**Quality:**
The paper is generally a good paper with a clear central idea.

**Clarity:**
The organization of the paper is good and it is easy to follow the topic and the proposed algorithms.

**Performance:** 
Extensive and comprehensive experiments validate the effectiveness of UARB, demonstrating promising improvements in OOD detection performance from a data-centric perspective.

### Weaknesses
1. In the introduction, it would be helpful to clarify why overfitting to the training data leads to a decline in OOD detection performance. The current manuscript may be somewhat difficult to follow for readers who are not familiar with the related work.

2. Although the overall presentation is clear, several minor issues need to be addressed. For example, the purpose of the green dashed line in Figure 1 is somewhat unclear; an extra parenthesis appears in line 354; since OOD detection has already been abbreviated earlier, it may not be necessary to repeat the full term to avoid redundancy; and the captions of Tables 7 and 8 could be revised to more accurately reflect the experimental content.

3. It has not been analyzed whether the decline in OOD detection performance in the later stages of training is fully mitigated after applying the UARB training strategy.

### Questions
Could you further analyze whether the hyperparameter settings are sensitive to the dataset, or if they are related to the learning difficulty of the dataset?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the performance degradation in the late training stage due to overfitting on mastered in-distribution samples. The authors propose a data-centric end-to-end framework UARB that identifies mastered samples and then implements instance-level early stopping by excluding mastered samples from backpropagation. Experiments on CIFAR-10/100 and ImageNet show UARB improves OOD detection when combined with baselines like MSP, Energy, and KNN.

### Strengths
- Unlike model-centric methods that treat symptoms of overfitting, UARB dynamically filters training data, which aligning with the growing focus on data quality in OOD detection.  
- By integrating class-wise uncertainty variance, UARB balances training of easy and hard classes.

### Weaknesses
- Most experimental improvements are margin, as can be observed in the tables, especially on ImageNet dataset. Such gains are insufficient to justify the added complexity over simpler baselines.  
- The paper claims UARB mitigates overfitting but there is no direct visualization of overfitting, which is critical to validate its core mechanism.  
- Experiments only use ResNet-18, a lightweight but outdated architecture. Modern OOD detection increasingly relies on transformers (e.g., ViT, Swin Transformer) or pre-trained models. UARB’s effectiveness on these architectures is unproven, limiting its practical relevance.  
- UARB adds non-trivial computations, e.g., second-order difference across epochs, class-wise variance calculation, but provides no analysis of computational cost compared to baselines. For resource-constrained edge devices, this omission makes it impossible to assess practicality.  
- The paper relies entirely on experimental results. There is no theoretical guarantees for the proposed method.

### Questions
None

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
3

### Summary
Overfitting during the training phase can significantly degrade OOD detection performance. This paper addresses this issue by proposing Uncertainty-driven Adaptively Rectified Backpropagation (UARB), which mitigates overfitting by blocking the backpropagation of mastered instances. The authors first demonstrate that OOD detection performance declines in the later stages of training due to overfitting. To alleviate this, they propose UARB, which leverages uncertainty to estimate the degree of learning for each instance and selectively stops training for sufficiently learned samples. Furthermore, they highlight the issue of class imbalance during training and introduce an adaptive threshold mechanism, which employs both the uncertainty and its second-order difference to dynamically balance per-class thresholds when identifying mastered instances. Extensive experiments across various datasets and OOD detection scenarios demonstrate the effectiveness and generality of their method compared with UM.

### Strengths
1. UARB is a plug-and-play method with strong applicability to various existing OOD detection approaches.
2. The problem of overfitting is clearly demonstrated through experimental evidence (as shown in Figure 1).
3. The motivation behind using uncertainty, the second-order difference, and the adaptive threshold mechanism is clearly explained.
4. UARB clearly outperforms UM, the most relevant prior approach, across multiple benchmarks.

### Weaknesses
1. In Section 2.2, the authors mention that they empirically verify the increase in instances that meet the mastered criteria. However, no corresponding experimental results are presented in the paper.
2. The authors argue that using a fixed threshold to determine when to exclude an instance from backpropagation is theoretically unsound, yet no further explanation or theoretical analysis is provided. I encourage the authors to elaborate on this point with more detailed justification.
3. UARB requires tracking the uncertainty across three consecutive epochs for the entire training dataset, which introduces substantial computational and memory overhead — especially as the dataset size and model complexity increase.
4. Although this paper mainly focuses on OOD detection during the training phase, it omits discussion of more recent and powerful OOD detection methods, such as DICE [1] and LINe [2].
5. While variance normalization can provide certain benefits, it assumes that all classes are generally imbalanced. In early training stages or fine-grained datasets where class-wise variation is minimal, this assumption may instead lead to adverse effects.
6. I recommend that the authors revise the full manuscript to improve readability and completeness.
- In the Introduction, some expressions (e.g., mastered instance, data-level issue) are difficult to understand immediately.
- Minor typos:
 Line 269: scenatios → scenarios
 Line 411: Table → Equation
- In the main manuscript, Table 7 is not referenced or used.

[1] "Dice: Leveraging sparsification for out-of-distribution detection." ECCV 2022.\
[2] "LINe: Out-of-distribution detection by leveraging important neurons." CVPR 2023.

### Questions
1. Extending from the concern about computational cost, I am curious about the applicability of the proposed method to larger backbones, such as ResNet-50 or ViT.
2. UARB utilizes both uncertainty and the second-order difference. Could the authors clarify why the first-order difference was not considered in their formulation? (or provide some experimental results).
3. Since UARB uses uncertainty from three consecutive epochs to compute the second-order difference, I recommend conducting an ablation study on this hyperparameter to analyze its sensitivity and impact.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose Uncertainty‑driven Adaptively Rectified Backpropagation (UARB) to improve OOD detection performance of classification models. The motivation is that there is an issue that during training the model may over-fit to ID instances in later epochs, thereby degrading its ability to distinguish OOD. The authors monitor the uncertainty score (zero-order) and the second-order difference of uncertainty (roughly the change in change) over epochs. If both fall below thresholds, they declare the instance mastered. Once an instance is marked mastered, they exclude it from further back-propagation so as to prevent over-training on that sample and reduce over-fitting. They use class-informed statistics to derive adaptive thresholds for mastering each class. They report experiments showing improved OOD detection across benchmarks when using UARB compared with baseline training.

### Strengths
- The observation that OOD detection performance can decline during later training epochs is important and less emphasized in many OOD works.
- Intervening in the training loop is a novel angle for OOD detection.
- By recognizing that different classes converge at different rates, the authors design class-specific thresholds.

### Weaknesses
- The methodology relies heavily on an uncertainty score per instance and its second-order difference across epochs. But the paper is somewhat light on which uncertainty metric is used. How sensitive results are to this choice, and how robustly it correlates with mastery.
- Is the second order difference stable and sufficiently noise-free to reliably identify mastered instances? How does noise in uncertainty estimates affect the criteria?
- The details of how the class-specific thresholds are computed are relatively brief. How sensitive are results to the thresholds
- By excluding mastered samples from further training, the method effectively reduces the effective training set size and focuses training on harder instances. While this may reduce over-fitting, it could potentially under-fit some classes or lead to class imbalance issues. Is there analysis of whether ID classification accuracy or calibration suffers?
- It would be great to add the following ablation studies. 
1. the effect of using only zero-order vs. also second-order difference
2. fixed threshold vs adaptive threshold;
3. baseline training without UARB but with same number of epochs/training budget.

### Questions
- It would be great to add discussion on uncertainty metrics (entropy, MSP, and loss) and show how the criteria for mastery behaves under each.
- It would be great to include sensitivity analyses on δ₁, δ₂ and show robustness of performance
- It would be great to add experiments on more diverse datasets and OOD types including near-OOD/far-OOD
- It should report the impact on primary classification task metrics (ID accuracy, calibration, robustness)

### Soundness
3

### Presentation
2

### Contribution
3
