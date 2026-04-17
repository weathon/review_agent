# DD-Ranking: Rethinking the Evaluation of Dataset Distillation

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
In recent years, dataset distillation has provided a reliable solution for data compression, where models trained on the resulting smaller synthetic datasets achieve performance comparable to those trained on the original datasets. To further improve the performance of synthetic datasets, various training pipelines and optimization objectives have been proposed, greatly advancing the field of dataset distillation. Recent decoupled dataset distillation methods introduce soft labels and stronger data augmentation during the post-evaluation phase and scale dataset distillation up to larger datasets (e.g., ImageNet-1K). However, this raises a question: Is accuracy still a reliable metric to fairly evaluate dataset distillation methods? Our empirical findings suggest that the performance improvements of these methods often stem from additional techniques rather than the inherent quality of the images themselves, with even randomly sampled images achieving superior results. Such misaligned evaluation settings severely hinder the development of DD. Therefore, we propose DD-Ranking, a unified evaluation framework, along with new general evaluation metrics to uncover the true performance improvements achieved by different methods. By refocusing on the actual information enhancement of distilled datasets, DD-Ranking provides a more comprehensive and fair evaluation standard for future research advancements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper first observes that existing dataset distillation approaches adopt inconsistent evaluation protocols, differing in their use of label types (e.g., fixed hard/soft labels or per-image soft labels), data augmentation strategies, etc. To enable a fair comparison across methods, the authors propose a unified benchmark along with two new evaluation metrics, LRS and ARS, designed to be robust against variations in the use of label types and augmentation techniques. Experimental results reveal that recent approaches relying heavily on soft labels (i.e., decoupled methods) are ineffective, often performing worse than a random selection baseline. In contrast, more conventional methods employing a single hard or soft label per image continue to demonstrate superior performance.

### Strengths
- The introduction of a unified benchmark for dataset distillation methods is both timely and significant for the community. Establishing a standardized evaluation protocol is crucial for ensuring fair comparisons and clear assessments across approaches. The paper convincingly shows that recent methods relying on per-epoch soft labels have been overrated.  
- The paper proposes novel evaluation metrics, LRS and ARS. These metrics provide a more robust and flexible framework for evaluating methods that adopt different recipes to train networks on distilled datasets.
- The paper provides extensive experimental results.

### Weaknesses
- The main finding that improvements in decoupled methods come largely from knowledge distillation rather than the distilled dataset itself is interesting, but this point has already been raised in [1]. It would be helpful to acknowledge and connect to that prior work.  
- Some experimental results appear to be missing. For example, lines 149–161 mention results on random noise with soft labels, but I could not find the corresponding figure.  
- While comparing against randomly selected samples is meaningful, I am less convinced by the comparisons using hard labels or without augmentation. Many methods are explicitly designed with particular label formats and augmentation strategies (e.g., DATM, EDF, IDC, FYI, etc.), so the evaluation feels less fair in those settings.
- The evaluation metrics introduce tunable parameters ($\lambda$ in Eq. (3) and $\gamma$ in Eq. (4)). This flexibility may unintentionally influence the results and deserves some discussion.
- A few methods evaluated in Sec. 4.2 are not included in Sec. 4.3, which leaves the comparison incomplete.

[1]: Qin, Tian, Zhiwei Deng, and David Alvarez-Melis. "A label is worth a thousand images in dataset distillation." Advances in Neural Information Processing Systems 37 (2024): 131946-131971.

### Questions
- How were the soft labels generated for random images when comparing with DATM or EDF? In the original methods, labels are jointly optimized with the images. Clarification on this process would be helpful.
- Why are learning rates tuned specifically for randomly selected images (lines 267–269) instead of applying the same learning rate used with synthetic images? Please explain the rationale behind this choice.

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
This paper introduces a new evaluation benchmark for dataset distillation (DD) in image classification, aiming to assess the effectiveness of distilled datasets compared to random selection. The benchmark focuses on two proposed metrics: label-robust score and augmentation-robust score.
As summarized in Table 1, existing DD methods differ significantly in their use of labels (hard vs. soft labels, and whether soft labels come from a fully-trained teacher or are jointly optimized during distillation) and augmentations (e.g., resize-crop, patch-shuffle, cutmix). These differences make direct comparison of DD methods difficult. The paper argues that prior evaluations, which each use their own label and augmentation setups, are unfair and inconsistent.
To address this, the authors propose:
* Label-robust score: compares the accuracy of distilled data versus random selection under the same label setting (e.g., both using hard labels or the same soft labels). 
* Augmentation-robust score: compares distilled data versus random selection under the same augmentation setting (e.g., same augmentation type or no augmentation). 

The proposed benchmark aims to standardize evaluation conditions and reveal the true contribution of the distilled images themselves, separate from the effects of labels or augmentations.

### Strengths
* The paper provides a meaningful attempt to standardize the evaluation of dataset distillation methods, enabling a more controlled comparison against random selection under matched label and augmentation setups. 
* The results highlight interesting findings: under hard-label usage, matching-based DD methods remain stronger than recent soft-label–based approaches, suggesting that much of the improvement in newer methods (e.g., SRe2L) may stem from knowledge distillation rather than from the intrinsic quality of the synthetic images.

### Weaknesses
* Limited applicability of the metric: Although the proposed metrics allow comparisons under matched label/augmentation setups, they do not measure the ultimate achievable performance of each DD method under its best hyperparameter and setup choices. Since DD performance also depends on factors like architecture, optimizer, and training configuration, comparing distilled datasets only under uniform conditions offers limited insight into each method’s full potential. 
* Ambiguous interpretability of the two measures: The two metrics—label-robust and augmentation-robust scores—merely quantify relative test accuracies rather than any intrinsic quality of the synthetic datasets. It is unclear how these two scores should be used jointly or whether they could be unified into a single, more interpretable evaluation measure. 
* Limited scope beyond image classification: The paper focuses solely on image classification. Modern distillation applications extend to vision-language and language model distillation, where data efficiency is more critical. It is unclear how the proposed robustness metrics could generalize to multimodal or text-based distillation tasks, limiting the broader applicability and fundamental impact of the proposed benchmark.

### Questions
1. On metric applicability:
    * How do the proposed label-robust and augmentation-robust scores reflect the best achievable performance of each DD method? 
    * Could the benchmark be extended to allow comparisons when each method is evaluated under its own optimal settings (e.g., best label/augmentation choices)? 
2. On metric design and coherence:
    * How should users interpret the two robustness scores jointly? 
    * Is there a principled way to combine the label-robust and augmentation-robust scores into a single unified measure that better reflects dataset quality? 
3. On generalization beyond image classification:
    * Can the proposed evaluation framework be adapted for multimodal or language model distillation tasks, where label and augmentation definitions are more complex? 
    * If not, how might the authors envision extending these metrics to broader domains?

### Soundness
2

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
This paper identifies significant unfairness in current dataset distillation evaluation practices, mainly caused by inconsistent training configurations. In particular, the use of soft labels often leads to performance gains that originate from knowledge distillation rather than the actual quality of the synthetic data, while improvements from data augmentation do not necessarily indicate better dataset informativeness. To address these issues, the authors introduce three new evaluation metrics, namely Hard Label Recovery (HLR), Improvement Over Random (IOR), and Label Robust Score (LRS), which aim to disentangle the effects of knowledge distillation and data augmentation from the intrinsic performance of distilled datasets.

### Strengths
1. The paper clearly demonstrates that performance improvements in existing dataset distillation methods often result from knowledge distillation or data augmentation rather than from the informativeness of the synthetic images.
2. The proposed evaluation metrics for comparing the performance of different models are clearly defined.
3. The authors conduct experiments with LRS and ARS across different model architectures, teacher models, and hyperparameter settings to verify the robustness of the proposed method.

### Weaknesses
1. The paper spends excessive space analyzing the limitations of existing methods. This part is repetitive and should be condensed into a shorter empirical motivation section.
2. Although LRS and ARS are intuitively motivated, their theoretical foundation is weak and lacks conceptual depth.
3. In Section 3, the definition of DD RANKING is unclear. It is not specified whether it refers to LRS, ARS, or a combination of both.
4. In line 240, the description of the normalization method for computing ARS is vague and should be explicitly defined.

### Questions
In line 210, does the random subset refer to real images from the original dataset or to synthetic noise samples?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that prior work on dataset distillation has been evaluated on an unfair playing field, and that simple accuracy comparisons do not guarantee fair or consistent assessment. To address this, the authors introduce DD-Ranking, which proposes a unified evaluation framework comprising four components to compare diverse DD methods under equitable criteria. A key strength is that DD-Ranking aims to deliver consistent evaluation irrespective of model architecture, the presence or absence of soft-label optimization, and the specific data-augmentation settings.

### Strengths
* The paper is well-written with a clear, thorough, and concise introduction that effectively summarizes key points
* The authors demonstrated through extensive experiments that the proposed evaluation metrics are meaningful and effective.

### Weaknesses
* Discussion of limitations is lacking
* Theoretical background would be needed

### Questions
## Discussion of limitations is lacking
* This paper evaluates performance only on datasets and models designed for classification tasks. I am curious how the authors envision establishing fair evaluation protocols in the context of multi-modal dataset distillation (MDD), where tasks and modalities may differ substantially.

## Theoretical background would be needed
* The paper aims to mitigate the unfairness introduced by data augmentation through the Augmentation-Robust Score (ARS). However, γ is fixed to 0.5, equally weighting the accuracy gaps obtained under augmentation and non-augmentation settings.
Conceptually, if the goal is to isolate and remove the influence of augmentation, measuring only the non-augmented accuracy gap (i.e., $acc_{syn-naug} − acc_{rdm-naug}$) might be sufficient and more principled. Why is the augmented gap ($acc_{syn-aug} − acc_{rdm-aug}$) necessary to include in the computation?
* Moreover, have the authors examined which of the two gaps (augmented or non-augmented) plays a more significant role in determining distillation quality?

### Soundness
3

### Presentation
3

### Contribution
2
