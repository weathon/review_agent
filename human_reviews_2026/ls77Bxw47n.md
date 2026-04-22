# DAVIS: Out-of-Distribution Detection via Dominant Activations and Variance for Increased Separation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Detecting out-of-distribution (OOD) inputs is a critical safeguard for deploying machine learning models in the real world. However, most post-hoc detection methods operate on penultimate feature representations derived from global average pooling (GAP) – a lossy operation that discards valuable distributional statistics from activation maps prior to global average pooling. We contend that these overlooked statistics, particularly channel-wise variance and dominant (maximum) activations, are highly discriminative for OOD detection. We introduce DAVIS, a simple and broadly applicable post-hoc technique that enriches feature vectors by incorporating these crucial statistics, directly addressing the information loss from GAP. Extensive evaluations show DAVIS sets a new benchmark across diverse architectures, including ResNet, DenseNet, and EfficientNet. It achieves significant reductions in the false positive rate (FPR95), with improvements of 48.26% on CIFAR-10 using ResNet-18, 38.13% on CIFAR-100 using ResNet-34, and 26.83% on ImageNet-1k benchmarks using MobileNet-v2. Our analysis reveals the underlying mechanism for this improvement, providing a principled basis for moving beyond the mean in OOD detection. Our code is available here: https://github.com/epsilon-2007/DAVIS

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper hypothesizes that the penultimate feature is information lossy due to the global average pooling (GAP) operation, resulting in an inferior OOD performance.  Therefore,  they propose to modify the feature before the GAP via 1)  utilizing channel-wise variance and 2) dominant (maximum) activations  and Energy score is adopted as the final OOD score. However, the proposed method only works for the architectures that use a spatial aggregation operations as pointed in the section "Limitations".

### Strengths
Strengths: 

- The hypothesis that feature after GAP operation is information lossy sounds reasonable for OOD detection. 
- Comprehensive experiments are done for architectures with global average pooling operations.
- The Discussion section is well-written and comprehensive.

### Weaknesses
- I have to say that this paper miss a lot of highly relevant related works including but not limited to Mahalanobis [1], GradNorm[2], ViM [3], GEN [4]  and NN-guide [5]. I list some of important and relevant ones, but please read  Generalized Out-of-Distribution Detection: A Survey  for reference.  
- The proposed method is limited to the architectures with global average pooling, which constrains its applicability. Yet the authors have done experiments extensively on different architectures,  there are several strong baseline methods are missing to compare.  I highly recommend the authors follow the standard benchmark openOOD (https://zjysteven.github.io/OpenOOD/) for a fair and comprehensive experiments as other papers do!
-  Another concern is that the performance of the proposed two alternatives $\text{DAVIS}(m)$ and $\text{DAVIS}(\mu, \sigma)$ is unstable and it is really difficult to tell from the table 1 and 2. It seems that the best performance is highlighted and it looks like  the variant of $\text{DAVIS}(m)$ and $\text{DAVIS}(\mu, \sigma)$ equipped with DICE and DICE* achieve the SOTA performance. However, there is no one method is consistently better than or equivalently good as other baselines.  For readability, please report the averaged results across architectures,  $i.e.$, adding two columns at the end of each table.  

[1] A simple unified framework for detecting out-of-distribution samples and adversarial attacks, NeurIPS, 2018.
[2] On the importance of gradients for detecting distributional shifts in the wild, NeurIPS, 2021.
[3] Vim: Out-of-distribution with virtual-logit matching, CVPR, 2022.
[4] Gen: Pushing the limits of softmax-based out-of-distribution detection, CVPR, 2023.
[5] Nearest neighbor guidance for out-of-distribution detection, ICCV, 2023.

 
Minor weaknesses: 
- Figures 1 and 2 should be arranged in one row for better readability.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DAVIS, a post-hoc enhancement framework for OOD detection. The authors argue that GAP discards discriminative statistical information and propose incorporating channel-wise maximum activations and variances into the features to improve the separability between ID and OOD data. Experiments show that this method significantly reduces the FPR95 on standard benchmarks like CIFAR and ImageNet and works synergistically with several leading post-hoc methods

### Strengths
- The authors have conducted exhaustive evaluations across a wide range of architectures and datasets.
- The paper provides an open-source implementation and details its hyperparameter search strategy.

### Weaknesses
I believe the main issue with this paper is its novelty. The motivation and the proposed method primarily stem from the distribution of activation values, particularly the distribution before the GAP layer. However, this has actually been proposed and explored in both [r1] and [r2]. I believe the core of this paper almost completely overlaps with these two 2024 papers. Additionally, there are in fact many other similar studies centered around activations; I have only listed the two most relevant ones here.

[r1] Tang, Keke, et al. "Cores: Convolutional response-based score for out-of-distribution detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[r2] Wan, Weilin, et al. "Out-of-distribution detection using neural activation prior." arXiv preprint arXiv:2402.18162 (2024).

### Questions
please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work identifies a key limitation in standard post-hoc OOD detection: the information loss from relying solely on Global Average Pooling (GAP), which discards valuable statistical cues like channel-wise variance and maximum activations. The authors propose DAVIS, a simple and effective method that augments the standard feature vector with these discriminative statistics, leading to a more powerful representation for separating in-distribution from out-of-distribution data. Extensive experiments demonstrate that DAVIS establishes a new state-of-the-art across diverse model architectures, significantly reducing the FPR95, and provides a principled basis for moving beyond mean-based features.

### Strengths
1. The paper compellingly identifies a critical, previously overlooked flaw in standard post-hoc OOD detection: the significant information loss inherent in relying solely on Global Average Pooling (GAP), which discards valuable spatial distribution statistics like variance and maximum activations. The proposed DAVIS method directly and effectively addresses this core limitation by enriching features with these discriminative statistics.
2. DAVIS is presented as a remarkably simple, plug-and-play technique that seamlessly integrates with existing pipelines and diverse architectures (ResNet, DenseNet, MobileNet-V2, EfficientNet). Its strength lies in its synergistic ability to significantly boost the performance of standard baselines without requiring architectural changes or retraining.
3. The paper provides extensive and convincing experimental evidence, demonstrating that DAVIS establishes a new state-of-the-art across multiple challenging benchmarks (CIFAR-10, CIFAR-100, ImageNet-1k).  The analysis offers valuable insights into the mechanism behind the improvement.

### Weaknesses
1. Although the reported results demonstrate the effectiveness of the method, comparisons with some highly relevant approaches are missing, such as BATS [1] and LAPS [2].
2. The method is predicated on models that utilize Global Average Pooling (GAP). It may not be applicable to models that do not employ GAP.
3. According to Table 1, the results of DAVIS(m) alone are not as good as ASH, and it is only when combined with DICE that DAVIS(m) surpasses ASH. This suggests that the performance improvement of DAVIS(m) by itself is limited.

[1] Zhu, Yao, et al. "Boosting out-of-distribution detection with typical features." Advances in Neural Information Processing Systems 35 (2022): 20758-20769.
[2] He, Rundong, et al. "Exploring channel-aware typical features for out-of-distribution detection." Proceedings of the AAAI conference on artificial intelligence. Vol. 38. No. 11. 2024.

### Questions
The paper only reports the combinations of the proposed DAVIS with ASH and DICE. How about the combination effects of DAVIS with NCI and fdbd?

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
This paper proposes DAVIS, a simple post-hoc OOD detection method that augments penultimate features with channel-wise maximum activations and variance to recover information lost in global average pooling.
It works plug-and-play with existing methods (Energy, DICE, etc.).

### Strengths
- Clear motivation and solid intuition: The paper convincingly identifies information loss in GAP and proposes a simple yet effective fix by incorporating dominant activations and variance.

- Strong empirical results: Consistent and substantial improvements (up to 48% FPR95 reduction) across reported multiple architectures and datasets.

### Weaknesses
- Dependence on pretrained classifier head:
The method reuses the original classifier weights for modified features; though empirically justified, this may not always be theoretically optimal.
Some adaptive fine-tuning or learned re-weighting could further stabilize the approach.

- Lack of a unified variant combining both signals:
The paper defines two separate DAVIS variants — DAVIS(m) using channel-wise maxima and DAVIS(µ,σ) incorporating mean plus variance — but does not explore combining both dominant and variance cues jointly.
A unified formulation leveraging both peak intensity and distribution spread could potentially yield further improvement.

- Limited coverage of transformer-based models:
The authors acknowledge DAVIS currently applies only to architectures with spatial aggregation (CNNs).
Extending to ViTs or hybrid architectures would broaden applicability.
Regardless of outcome, such experiments could offer important insights into how attention models encode distributional statistics for OOD detection.

- Hyperparameter tuning validation datasets:
DAVIS tunes γ only on Gaussian-noise validation sets and separately for each dataset and architecture.
No tests on other perturbations or cross-dataset stability are shown, so the claimed robustness is likely distribution-specific rather than general.

- More evaluation setting:
The experiments in DAVIS use dataset splits, metrics, and model sources different from the standardized OpenOOD benchmark.
Evaluating DAVIS under a unified OpenOOD setting would strengthen reproducibility and clarify whether it still outperforms recent strong SOTAs such as AdaSCALE or CombOOD.

### Questions
- Could you comment on whether re-training or lightly fine-tuning the classifier head with DAVIS features would further improve separability?

- Would combining DAVIS with logit-based regularization (e.g., Outlier Exposure) yield additive gains, or do the effects saturate?

- Have you tested whether the choice of activation function (beyond SiLU/ReLU) consistently affects the relative importance of maximum vs. variance statistics?

- Could you clarify whether DAVIS(m) and DAVIS(μ,σ) are complementary or mutually exclusive in practice—e.g., has concatenation been tested?

### Soundness
2

### Presentation
2

### Contribution
2
