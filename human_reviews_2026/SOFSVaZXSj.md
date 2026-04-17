# Rethinking Model Calibration through Spectral Entropy Regularization in Medical Image Segmentation

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Deep neural networks for medical image segmentation often produce overconfident predictions, posing clinical risks due to miscalibrated uncertainty estimates. In this work, we rethink model calibration from a frequency-domain perspective and identify two critical factors causing miscalibration: spectral bias, where models overemphasize low-frequency components, and confidence saturation, which suppresses overall power spectral density in confidence maps. To address these challenges, we propose a novel frequency-aware calibration framework integrating spectral entropy regularization and power spectral smoothing. The spectral entropy term promotes a balanced frequency spectrum and enhances overall spectral power,  enabling better modeling of high-frequency boundary and low-frequency structural uncertainty. The smoothing module stabilizes frequency-wise statistics across training batches, reducing sample-specific fluctuations. Extensive experiments on six public medical imaging datasets and multiple segmentation architectures demonstrate that our approach consistently improves calibration metrics without sacrificing segmentation accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical problem of overconfidence in deep neural networks for medical image segmentation. The authors propose a novel approach that reframes model calibration from a frequency-domain perspective. They identify two primary causes of miscalibration: spectral bias, where models over-prioritize low-frequency components, and confidence saturation, where overconfidence leads to a general suppression of power spectral density (PSD).

To solve this, the paper introduces a frequency-aware calibration framework consisting of two main components:
1. Spectral Entropy Regularization: A novel loss term that encourages a more balanced and informative frequency spectrum in the model's confidence maps (derived from logits).
2. Power Spectral Smoothing: A temporal smoothing module that stabilizes the spectral statistics used to create a dynamic target for the regularization loss.

The authors conduct an extensive evaluation on six public medical imaging datasets and multiple segmentation architectures. Their results show that the proposed method significantly improves calibration metrics (ECE, SCE) while simultaneously maintaining or even improving segmentation accuracy (DSC, HD95).

### Strengths
1. The paper's primary strength is its highly original approach. Reframing the calibration problem through the lens of spectral bias and confidence saturation is a novel and insightful contribution. While frequency-domain methods exist, this appears to be the first work to build a coherent framework specifically for confidence calibration by regularizing the output's intrinsic spectral properties. The results are highly significant. The method achieves state-of-the-art calibration, showing massive and consistent improvements in ECE and SCE across all six datasets (Table 2).

2. The technical quality of the submission is exceptionally high. The link between overconfidence and a suppressed, low-frequency-dominated PSD is clearly motivated and visualized (Fig. 1) . The methodology is also well-justified, including the sound rationale for operating on logits rather than saturated softmax probabilities. This technical rigor extends to the evaluation, which is both comprehensive and robust. The authors validate their method on six diverse datasets, compare it against five strong baselines, and use a full suite of segmentation and calibration metrics.

3. Furthermore, the method's "plug-and-play" nature is convincingly demonstrated, showing consistent improvements across six different segmentation architectures, including U-Net, nnUNet, and SwinUNETR (Fig. 4) . The analysis is also thorough; the ablation study (Table 3) clearly isolates the contribution of each component, and the appendix provides further strong evidence (e.g., radar charts, rank heatmaps). Crucially, this work breaks the common accuracy-calibration trade-off, often improving segmentation performance while fixing calibration (Table 1)

### Weaknesses
1. The weaknesses are minor and do not undermine the contribution.

2. The paper does not quantify the computational overhead of the proposed method. 

3. Calculating 3D FFTs, aggregating power, and managing a historical buffer for each batch introduces a non-trivial training cost compared to the baseline, and this information would be valuable for other researchers. Additionally, the entire spectral analysis relies on a scalar confidence map derived from the maximum logit value. 

4. This simplification discards information from other classes, and it remains unclear how this approach would scale to problems with many classes. 

5. While the sensitivity analysis for $K$ (the number of frequency bands) is good, the paper could benefit from a more qualitative discussion of why $K=5$ provides an optimal balance, perhaps by linking the bands to specific anatomical structures.

### Questions
1. Could you please provide an analysis of the training-time overhead? For example, what is the percentage increase in training time per epoch when adding the proposed regularization framework compared to the standard cross-entropy baseline?

2. The temporal smoothing uses a simple sliding window of size $W$. Did you consider using an Exponential Moving Average (EMA) instead? An EMA might offer a more flexible trade-off between stability and adaptability, especially in non-stationary settings where the "target" spectral profile might evolve.

3. The method relies on the max-logit to create the confidence map. Have you experimented with other scalar representations of confidence, such as the predictive entropy of the softmax output? It would be interesting to see if its spectral properties provide a similar or complementary signal for regularization.

4. The motivation in Figure 1 defines a "well-calibrated" boundary pixel as having $\alpha=0.5$ (maximum uncertainty). This is intuitive for a binary foreground/background decision. How does this intuition translate to a multi-class boundary, for instance, the boundary between "edema" and "enhancing tumor" in the BraTS dataset (which has 4 classes)? Is the goal still maximum uncertainty, or is the ideal spectral signature more complex in this scenario?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a power spectral smoothing mechanism for calibrating segmentation models. Specifically, it proposes a spectral entropy regularisation to improves frequency balance in confidence maps, involving spectral decomposition, window smoothing, leading to better model calibration results.

### Strengths
Handle the calibration task from the spectral domain is novel;
The paper is easy to follow.

### Weaknesses
1) Motivation is not solid;
2) Too many sophisticated hyper-parameters to tune;
3) Missing important baselines (Temperature scaling, Platt scaling) and metrics (IoU).

### Questions
1) In Fig.1, why a lower confidence score at boundary indicates better calibrated outputs? Especially if this is for ground truth labels.
2) Calibration is important for non-segmentation tasks, but how is this important for segmentation? One will have to set a threshold for the segmentation results to be useful, so the accuracy/IoU naturally becomes the focus.
3) Is there any reason that in experiments Temperature scaling/Platt scaling and IoU is not report? Temperature scaling/Platt scaling may require validation set but since there are already so many hyper-parameters have been tuned this should be doable.

### Soundness
3

### Presentation
2

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
This paper investigates the problem of confidence miscalibration in medical image segmentation and provides a new perspective by analyzing it in the frequency domain. The authors identify two core causes—spectral bias (models’ preference for low-frequency components) and confidence saturation (loss of spectral power in confidence maps)—that contribute to overconfident predictions. To tackle these issues, the paper introduces a frequency-aware calibration framework combining spectral entropy regularization and power spectral smoothing. Experiments across six public datasets and multiple segmentation architectures demonstrate that the proposed method consistently improves calibration metrics while maintaining segmentation accuracy.

### Strengths
Novel Perspective: The work provides an original frequency-domain interpretation of model miscalibration, which is rarely explored in medical image analysis.

Theoretical Insight: The identification of spectral bias and confidence saturation as underlying causes is conceptually sound and empirically validated.

Methodological Innovation: The proposed spectral entropy regularization and power spectral smoothing are well-motivated, simple to implement, and complementary.

Comprehensive Evaluation: Experiments on multiple datasets and architectures show consistent improvements in calibration without accuracy degradation.

Clinical Relevance: By enhancing uncertainty reliability, the approach contributes directly to improving trustworthiness in medical AI systems.

### Weaknesses
While the proposed frequency-aware calibration framework is conceptually interesting, several aspects require further clarification and discussion. The frequency-domain operations in this work are quite similar to those in [1][2], which also explore the distinction between low- and high-frequency components; however, the paper does not explicitly analyze or compare with these prior studies. Moreover, key training details are missing—particularly the batch size used and how it affects frequency statistics. Performing repeated frequency-domain transformations across multiple batches could introduce considerable computational overhead. The penalty minimization is said to promote a more balanced spectral distribution by transferring energy from dominant low-frequency to underrepresented high-frequency regions, but the paper does not explain how this spectral balance is quantitatively measured, nor whether it might degrade segmentation accuracy. Finally, in class-imbalanced scenarios (e.g., when certain anatomical categories have far more samples), it remains unclear whether the proposed approach retains its effectiveness.
[1]Feng, Wei, et al. "Unsupervised domain adaptation for medical image segmentation by selective entropy constraints and adaptive semantic alignment." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 1. 2023.
[2]Chen, Guangyao, et al. "Amplitude-phase recombination: Rethinking robustness of convolutional neural networks in frequency domain." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
3
