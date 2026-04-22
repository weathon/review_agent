# Gradient Rectification for Robust Calibration under Distribution Shift

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Deep neural networks often produce overconfident predictions, undermining their reliability in safety-critical visual applications. This miscalibration is further exacerbated under test distribution shift. Existing methods improve calibration via training-time regularization or post-hoc adjustment, but often rely on access to (or simulation of) target domains, limiting practicality. We propose Frequency-aware Gradient Rectification (FGR), a target-agnostic training framework for robust calibration. From a frequency perspective, FGR applies low-pass filtering to a subset of training images to diminish spurious high-frequency cues and bias learning toward domain-invariant structure. However, the associated information loss can degrade In-Distribution (ID) calibration. To resolve this trade-off, FGR treats ID calibration as a hard optimization constraint and rectifies parameter updates via geometric projection whenever they conflict with calibration. This projection-based update guarantees a first-order non-increase of the ID calibration objective without introducing additional weighting hyperparameters.
Experiments on CIFAR-10/100-C and WILDS show that FGR significantly improves calibration under diverse shifts while preserving ID performance, and it remains compatible with post-hoc temperature scaling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel training loss design aimed at improving calibration under dataset shift. The proposed method, Frequency-aware Gradient Rectification (FGR), is based on the assumption that dataset shift mainly occurs in the high-frequency components of images. It applies low-pass filtering to reduce feature representations that are likely to be spurious, and additionally proposes an orthogonal projection of the original loss onto the calibration loss to prevent the degradation of in-distribution (ID) confidence caused by filtering. Through experiments, the method demonstrates comparable or superior performance to existing methods on ID data in terms of ECE and CECE, while also showing improvements in multiple out-of-distribution (OOD) cases.

### Strengths
- Clarity of presentation: The main argument and logical flow of the paper are easy to follow and clearly articulated throughout.
- The proposed method does not require access to the target distribution: This approach overcomes a major limitation of conventional methods that rely on access to the target distribution.

### Weaknesses
- Limitations of frequency-based assumptions in distribution shift: The method developed in this paper assumes that invariant features are primarily embedded in low-frequency components, whereas spurious features arise from high-frequency regions. However, this assumption does not consistently reflect real-world conditions. For example, distribution shifts may result from variations in low-frequency aspects such as overall image tone (e.g., brightness or hue), or from frequency-independent structural changes, including shifts in the spatial relationship between foreground and background. Furthermore, in fine-grained classification tasks, critical invariant features may inherently reside in high-frequency components. These scenarios reveal a key limitation of the proposed method, which may underperform when such complex or non-frequency-aligned distribution shifts are present.


- Ambiguity regarding the novelty of gradient rectification: Gradient surgery [1] in multi-task learning is a well-known technique for resolving conflicts between gradients from different loss functions. The proposed gradient rectification is conceptually quite similar, and the paper does not clearly explain where its novelty lies or how it offers superior calibration under distribution shift.

- Limited evaluation metrics for calibration: Only ECE and CECE are used, both of which are known to be sensitive to confidence bias in the model. Evaluation using Adaptive Calibration Error (ACE) [2], which addresses this issue, would provide a more convincing assessment.

[1] Yu, Tianhe, et al. "Gradient surgery for multi-task learning." Advances in neural information processing systems 33 (2020): 5824-5836.

[2] Nixon, Jeremy, et al. "Measuring calibration in deep learning." CVPR workshops. Vol. 2. No. 7. 2019.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Frequency-aware Gradient Rectification (FGR) to improve model calibration under distribution shift without accessing target-domain data. For details, FGR combines low-pass filtering to suppress spurious high-frequency features with gradient rectification to maintain in-distribution calibration by resolving gradient conflicts. Experiments achieves superior robustness and calibration compared to existing methods.

### Strengths
(1) Exploring domain shift calibration is important.

(2) The paper is well written and easy to follow

### Weaknesses
(1) Why use DCT is not clear. As described in the introduction, "learning to recognize 'birds' based on special texture (e.g., green
leafy patterns) rather than shape . Motivated by this, we apply Discrete Cosine Transform (DCT)
filtering to isolate low-frequency image components, encouraging the model to rely on shape-related
information that is more consistent across distributions.". However, i do not think this is well motivated to use DCT. More analysis are needed.

(2) Why use gradient rectification is also not clear. 

(3)The paper lacks an explicit analysis or report of this computational overhead compared to standard training.


(4) The method’s robustness to semantic distribution shifts remains untested. For example. Office-Home dataset can be tested.

### Questions
my major concern is the relationship between the problem faced by the author (calibration) and the solution strategy is not clear.

### Soundness
2

### Presentation
3

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
The paper combines a frequency based filtering mechanism and a gradient rectification technique to improve the out-domain and in-domain calibration performance. Some randomly sampled images are low-pass filtered to suppress spurious correlations and then combined with original unfiltered images to compute the standard classification loss and corresponding (main) gradient. The mini-batch of original images are used to compute the soft-ECE loss and its (calib) gradient. If the main gradient conflicts with the calib gradient, then the former is projected to a hyperplane that is orthogonal to the calib gradient. This gradient projection acts as hard ID calibration constraint to sustain the ID calibration performance, which can get compromised if only filtered images are used to obtain supervisory signal for training the network. Results on different in-domain and out-domain scenarios claim to improve calibration performance.

### Strengths
- The challenge of improving out-domain calibration performance while also retaining the ID calibration performance is quite relevant due to its practical significance. 

- Putting ID calibration as a hard constraint via gradient projection to sustain ID calibration performance is interesting and requires no weighting parameters.

- The related work provides a good coverage of recent and relevant methods in model calibration.

- The experimental comparison is shown with different baselines and in different ID and OOD scenarios and the results claim to show that the proposed methods reduces miscalibration.

### Weaknesses
- To compute the main gradient, why both low-pass filtered and original images are put as hybrid? 

- Is it possible to use some other loss than Soft-ECE and expect similar or even better OOD and ID calibration performance?

- The weighted sum in Table 3 perform very closely to the FGR idea. Is there any explanation to that?

- It is not obvious how filtering the rectification improves OOD calibration performance?

- The ID calibration performance is not better than other methods in many cases (Table 2). Furthermore, without temperature scaling, it is not the best in any case. Given that the core hypothesis of the paper is sustaining/improving ID calibration performance and the gradient rectification is primarily proposed for this, how the Table 2 results justify this critical point?

- How the Fig. 5 visualizations are relevant to ID and OOD calibration performance?

- There are no real stats. or empirical analyses which validate the notable occurrence of gradient conflicts and connects it with the ID and OOD calibration improvements.

### Questions
- The core idea explicitly encourages learning of domain-invariant features, it should also help in improving out-domain accuracy notably, but this doesn't seem to be the case. Related to this, can the ID calibration hard constraint be counterproductive to improving out-of-domain accuracy in any case?

- It is not clear why the results are shown with temperature scaling in Table 1? given that the proposed method is primarily a train-time calibration contribution.

- How the method's performance is sensitive to different mini-batch size variations? Given that, the core idea is based on gradient rectification, it would be interesting to see the trend.

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
3

### Summary
The paper proposes Frequency-aware Gradient Rectification (FGR), a training-time framework to enhance calibration of deep neural networks under distribution shift, without requiring any target domain access. The approach combines low-pass discrete cosine transform (DCT) filtering of part of the training data, intended to encourage reliance on domain-invariant, low-frequency features, with a geometric gradient rectification step that projects parameter updates to prevent an increase in in-distribution (ID) calibration error. The authors support their method with theoretical analysis, comprehensive empirical studies on CIFAR-C variants, Tiny-ImageNet-C, and WILDS benchmarks, and ablation/visualization experiments.

### Strengths
1. The use of DCT-based block-wise low-pass filtering is a creative tool for encouraging robustness to shift, going beyond standard pixel-level or data augmentation strategies.
2. The proposed projection-based gradient rectification is well-motivated and avoids hyperparameter tuning.

### Weaknesses
1. The proposed method improves upon the compared method for metrics except accuracy. 
2. Poor performance on most metrics for the TinyImageNet dataset, including in the results in the appendix. This reflects poorly on the efficacy of the proposed approach.

### Questions
1. Can the authors justify why the proposed approach performs not as well on the TinyImageNet dataset?
2. Can the authors discuss how the accuracy of the proposed method can be improved through any trade-off.

### Soundness
2

### Presentation
2

### Contribution
2
