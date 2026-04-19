# On the Over-Memorization During Natural, Robust and Catastrophic Overfitting

- Decision: Accept (poster)
- Scores: 5, 6, 6, 8

## Abstract
Overfitting negatively impacts the generalization ability of deep neural networks (DNNs) in both natural and adversarial training. Existing methods struggle to consistently address different types of overfitting, typically designing strategies that focus separately on either natural or adversarial patterns. In this work, we adopt a unified perspective by solely focusing on natural patterns to explore different types of overfitting. Specifically, we examine the memorization effect in DNNs and reveal a shared behaviour termed over-memorization, which impairs their generalization capacity. This behaviour manifests as DNNs suddenly becoming high-confidence in predicting certain training patterns and retaining a persistent memory for them. Furthermore, when DNNs over-memorize an adversarial pattern, they tend to simultaneously exhibit high-confidence prediction for the corresponding natural pattern. These findings motivate us to holistically mitigate different types of overfitting by hindering the DNNs from over-memorization training patterns. To this end, we propose a general framework, $\textit{Distraction Over-Memorization}$ (DOM), which explicitly prevents over-memorization by either removing or augmenting the high-confidence natural patterns. Extensive experiments demonstrate the effectiveness of our proposed method in mitigating overfitting across various training paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzes three types of overfitting (natural, robust, and catastrophic) observed during the training process of deep neural networks and introduces methodologies to mitigate these phenomena. The authors are particularly motivated by the observation that, during periods of learning decay of standard training, the training loss for certain datasets sharply decreases. They designate these specific datasets as "transformed data" to differentiate them from the rest. When this transformed data is excluded from training, a reduction in the generalization gap is observed. This trend is similarly noted in settings where both robust and catastrophic overfitting are evident. Drawing from these observations, it is inferred that the transformed data might be excessively memorized, leading to overfitting. To counteract this, the authors propose the "distraction over memorization (DOM)" methodology, which emphasizes data augmentation specifically for the transformed data. Experimental results suggest that models trained using this approach exhibit a superior generalization gap compared to those trained with data augmentation applied across the entire dataset.

### Strengths
The paper demonstrates that natural overfitting can be mitigated by removing data characterized by a rapid decrease in training loss, termed "transformed data." Through this analysis, the authors highlight the occurrence of overfitting in standard settings due to such data and propose a method to distinguish data that has been excessively memorized. Furthermore, the properties of transformed data are not limited to natural overfitting; they exhibit similar trends in other types of overfitting, namely robust and catastrophic overfitting. The authors suggest a universal overfitting mitigation method by applying various data augmentation techniques to the transformed data. Experimental results are presented to validate the efficacy of this approach.

### Weaknesses
The motivation behind this paper, specifically the analysis of transformed data, has already been explored in a paper that introduced the MLCAT methodology [1]. The distinction is that the previous study limited its analysis to robust overfitting, whereas the current paper expands the analysis to three types of overfitting, demonstrating that these phenomena manifest commonly across all three. However, given that there isn't much difference in the learning algorithms or model structures between the standard, adversarial, and fast adversarial settings, one could easily anticipate that the characteristics of transformed data in the adversarial setting, as delineated in MLCAT [1], would manifest similarly in both the standard and fast adversarial settings. Therefore, the current analysis does not offer much novelty beyond the findings of the previous study. While the proposed methodology of applying data augmentation specifically to transformed data does have the advantage of being universally applicable to various types of overfitting, it only demonstrates an improved generalization gap in comparison to the baseline model. Given the inherent differences in training data for the standard, adversarial, and fast adversarial settings, one might question the necessity of a universally applicable overfitting mitigation method. To bolster this claim, the authors should compare the proposed method against methodologies in individual overfitting studies (natural, robust, catastrophic) and demonstrate that their approach offers competitive performance.

[1] Chaojian Yu, Bo Han, Li Shen, Jun Yu, Chen Gong, Mingming Gong, and Tongliang Liu. Understanding robust overfitting of adversarial training and beyond. In International Conference on Machine Learning, pp. 25595–25610. PMLR, 2022b.

### Questions
- When compared to the analysis performed in the previously cited study (MLCAT) mentioned under weaknesses, are there notable strengths in this paper that I might have missed, aside from the observation that similar phenomena manifest across standard, adversarial, and fast adversarial settings?
- In the "distraction over memorization" methodology, is there a specific reason for applying data augmentation iteratively rather than in a straightforward manner?
- Has the study investigated whether similar phenomena occur with learning rate scheduling methods that decrease at a more gradual pace, such as cosine, as opposed to the step learning decay?
- Are there any experimental results comparing the proposed approach to traditional methodologies under the same settings?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a general framework for explicitly preventing over-memorization by either removing or augmenting the high-confidence natural patterns. It is based on the observation that the model suddenly exhibits high confidence in predicting certain training patterns, which subsequently hinders the DNNs’ generalization capabilities.

### Strengths
**Strength:**

-   This paper is overall well-structured and easy to follow.
-   Extensive empirical evaluation with various training paradigms, baselines, datasets, and network architectures demonstrates its effectiveness. Results are reported with the standard deviation.
- Significant performance improvements are demonstrated.

### Weaknesses
**Weakness**

-   According to Figure 5, the proposed method may require careful hyper-parameter (i.e. loss threshold) selection, which could be a significant drawback.
-   The proposed method might result in repeated gradient computation and extensive extra computation. It is also interesting to include a detailed analysis of the introduced extra computation.
-   The terminology "pattern" might be confusing and could be further explained. Does it refer to specific samples in datasets?
-   Lack of results on large-scale datasets. It will be convincing to have some on Tiny-ImageNet or ImageNet
-   Lack of results on diverse network backbone architectures beyond ResNets.
-   As discussed in the related works, there are various techniques for mitigating the overfitting issues. Comparisons with other techniques like dropout, ensemble, smoothing, etc. can be helpful.

### Questions
Refer to the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper provides an empirical investigation into the generalization capabilities of deep neural networks (DNNs), focusing on understanding various facets of overfitting. The authors introduce the concept of over-memorization, a phenomenon where DNNs excessively retain specific training patterns, leading to diminished generalization. To mitigate this issue, the paper suggests techniques such as the removal of high-confidence natural patterns and the application of data augmentation. The effectiveness of these strategies is demonstrated through a series of experiments.

This paper makes a valuable contribution to the field by shedding light on the over-memorization behavior in DNNs and its implications for generalization. By addressing the highlighted areas for improvement, the authors have the potential to further enhance the significance and applicability of their work.

### Strengths
1. Clarity and Structure: The paper is commendable for its well-organized structure and clear exposition. The authors have provided a thorough background and review of related work, successfully setting the stage for their empirical analysis.

2. Robust Experimental Design: The experimental setup is meticulously designed, encompassing various types of overfitting and delving into the over-memorization behavior of DNNs. This comprehensive approach enhances the validity of the findings.

3. Novel Insight into Overfitting: The identification of over-memorization as a common thread linking different types of overfitting is an innovative contribution. This insight adds depth to our understanding of how overfitting impacts the generalization abilities of DNNs.

### Weaknesses
1. Limited Scope of Empirical Analysis: The paper's empirical analysis predominantly focuses on a specific network architecture and dataset. Expanding the analysis to include a wider array of cases or providing a theoretical framework to support the observed behaviors would bolster the generality and impact of the findings.

2. Partial Improvement on Overfitting Types: According to the results presented in Tables 2-4, the proposed strategies seem to predominantly ameliorate Class Overfitting (CO), with only marginal improvements on Natural Overfitting (NO) and Random Overfitting (RO). A more detailed exploration of why these discrepancies occur would provide valuable insights.

3. Need for Larger-Scale Evaluation: The experiments are confined to relatively simple datasets (CIFAR-10/100) and ResNet-based architectures. Extending the evaluation to encompass larger-scale datasets and alternative architectures, such as transformers, would enhance the representativeness of the results and the applicability of the findings.

### Questions
1. Expand Empirical Analysis: To strengthen the paper's contributions, the authors should consider conducting additional empirical analyses across diverse network architectures and datasets.

2. Deepen Analysis on Overfitting Types: A more nuanced exploration of the varying impacts on different types of overfitting would provide a richer understanding of the phenomena at play.

3. Consider Larger-Scale and Diverse Architectures: Incorporating experiments with larger datasets and a variety of neural network architectures would ensure that the findings are more widely applicable and representative of the broader deep learning landscape.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers a unified perspective on various overfitting, including NO (natural overfitting), RO (robust overfitting), and CO (catastrophic overfitting). On top of this, the authors discover the "over-memorization" phenomenon that the overfitted model tends to exhibit high confidence in predicting certain training patterns and retaining a persistent memory for them. Unlike previous methods, this paper proposes a general framework called DOM (Distraction Over-Memorization) to alleviate the unified over-fitting issue. Experiments show that the proposed method outperforms other baselines.

### Strengths
1. The discovery of the behavior "over-memorization" unifies different types of overfittings, which is of great help when analyzing the cause of overfitting.
2. The paper is generally well-written, and the motivation is stated clearly.
3. The proposed DOM framework seems promising.

### Weaknesses
1. In the DOM framework, the loss threshold is set with a fixed value. However, with different datasets and loss functions, the optimal threshold could be different. Therefore, the given threshold may not be general on other occasions. The authors should further conduct ablation studies about this and discuss how to overcome this issue.
2. The experiment settings are not precisely introduced in 3.1 and 3.2, making these conclusions challenging to reproduce. 
3. In section 3.2, the authors claim, “the AT-trained model never actually encounters natural patterns.” However, methods like TRADES do encounter natural patterns. What will happen in this case? Are the conclusions observed in this paper still applicable?
4. Why are there many 0.00 in Table 4? The authors need to give more explanation.

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
