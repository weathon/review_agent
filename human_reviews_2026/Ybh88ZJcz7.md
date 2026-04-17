# Face-Feature Tuning: Post-hoc Calibration for Fair and Accurate Deepfake Detection

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Deepfake detectors often show large performance gaps across demographic groups, undermining trust in deployment. Existing fairness approaches typically require demographic labels, retraining the detector, or accept notable drops in overall accuracy. We introduce Face-Fairness (FF), a plug-and-play framework for mitigating bias in deepfake detection. Our primary contribution, Face-Feature Tuning (FFT), is the first demographic label-free fairness method demonstrated for deepfake detection: a lightweight calibrator that performs post-hoc logit remapping conditioned on frozen face embeddings, trained on a held-out validation split. We complement FFT with two thresholding variants: FF-Max, which maximizes worst-group accuracy when demographics are available on validation, and FF-Discover, which applies the same objective to embedding-discovered groups when demographics are unknown. Across in-domain, cross-protocol, and cross-dataset test settings, FF consistently reduces FPR/TPR gaps and improves minimum group accuracy while maintaining (often improving) overall accuracy. The approach is detector-agnostic, adds negligible runtime overhead, and requires no access to identity attributes—making fairness in deepfake detection practical for real-world systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to advance group fairness (i.e. mitigating disparate performance across demographic groups) in deepfake detection. The authors propose three post-processing methods to maximize worst-group accuracy and achieve fair and accurate preditions: FF-Max that learns group-specific decision thresholds (with supervised demographic labels); FF-Discover which clusters face embeddings and learns cluster-conditional thresholds (without demographic labels); FFT that learns a decision boundary via training a lightweight multi-layer perceptron (without demographic labels). Experiments are carried out on both in-the-wild and canonical deepfake detection test beds, comparing with standard and state-of-the-art methods.

### Strengths
- The experiments are extensive. The choices of metrics encompass both group fairenss (equal opportunity) and min-max fairness (worst group accuracy) definitions.
- The literature review is well-organized, precise and complete.
- The paper is well-written and clearly positions the contribution in literature. Also, the mathematical notation and tables are clean and minimal, effectively conveying the message.

### Weaknesses
Unfortunately, at the current state, the paper feels quite narrow in terms of impact, given its strong application-specific nature. Below I'm listing the main reasons for the recommendation, which require further attention:

**W1.**

The contribution feels limited, apart from the method and some (good) related works. A starting point could be to dedicate a section to testing why and how the method works, when it breaks, and if there are theoretical guarantees that could be derived. Maybe also some ablation study (eg. on the width of the two-layer in FFT) could be an addition. 

For instance, there are some claims in Section 3.2.1. that could be expanded and theoretically tested: the discussions about FF-Max and FF-Discover being special cases, FFT delivering "better-conditioned probabilities" etc. (LL265-269) could be shown to hold over some controllable setup.

------------------

**W2.**

The results reported in the Tables seem to tightly cluster together, leading to very little improvements which at the moment feel not decisive. Furthermore, it is unclear if the results are averages of multiple runs: in this case, given they are very close it would be helpful to report also standard deviation. Otherwise, I'd suggest to try multiple runs with different seeds and report means and standard deviations.

On the same topic, Figure 1 could be improved. First of all, text and markers could be made bigger and non-overlapping for improved readability (this applies also for Figure 2). Secondly, as it is currently plotted, it seems the contribution of FFT is drastic, while the reality is that it improves by just 1% to 3% with respect to the worst method under comparison. Also here I'd suggest to report some confidence intervals.

### Questions
Thanking in advance for their response, I'd kindly invite the authors to address the points raised in the Weaknesses section of this review.

In addition, I'd kindly ask the following question:

- I couldn't find the reason of choosing older architectures like MobileNet or Xception, instead of newer architectures (eg. Vision Transformers). Is this an arbitrary choice, or, are they linked to the experimental test beds?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a Face-Feature Tuning (FFT) method that aims to achieve fairness in facial analysis without relying on demographic labels, retraining, or compromising accuracy. The key idea is that face embeddings extracted from pre-trained models inherently encode visual patterns that correlate with detector failures in a continuous feature space. To address this, FFT trains a lightweight neural network that refines the decision boundaries based on these embeddings, effectively recalibrating predictions in regions where systematic biases appear while maintaining consistent and reliable performance elsewhere.

Several concerns arise regarding the depth and impact of this work:

- The proposed FFT framework appears to be more of an engineering tweak on existing face-feature representations rather than a fundamentally new approach. The paper offers limited exploration of underlying principles, theoretical insights, or broader implications for fairness in vision models.

- The reported baselines, whether pre-processing, in-processing, or post-processing methods, already exhibit near-saturated performance, suggesting that the chosen task may not present substantial difficulty. This diminishes the perceived necessity and overall impact of the proposed method.

- The experiments are restricted to two relatively basic backbones, MobileNetV3 and Xception, without evaluation on forensics-oriented architectures that have been developed in recent deepfake detection or fairness-related research. This narrow experimental design limits the generalizability and significance of the findings, making it unclear how well FFT would perform in more advanced or domain-specific settings.

- (?) The title in the paper does not match the title in the system.

Overall, while FFT is technically sound, the novelty, motivation, and empirical validation of the paper remain modest, and its contribution to advancing fairness research in face analysis appears limited.

### Strengths
The proposed FFT method is lightweight, computationally efficient, and does not require retraining or access to sensitive demographic information, making it appealing for real-world deployment.

### Weaknesses
See summary

### Questions
NA

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
The paper proposes a novel approach to deal with bias and fairness in the deepfake detection task exhibited by existing models to improve the performance gaps across demographic groups. This key contribution of this work is that existing models do not need to be retrained to remove their biasness. The authors propose 3 methods to tackle the biasness issue depending on the availability of demographic labels, and they show that their technique improves detection fairness and FPR/TPR across model backbones and datasets.

### Strengths
- The paper tackles a very important problem of existing deepfake detection methods, which is the problem of fairness against demographic groups caused due to underrepresentation in training datasets.
- The proposed method is statistically sound, and covers 3 different regimes of detection depending on availability of demographic labels. 
- The biggest strength of the work compared to existing ones is that it does not require any retraining and can be added to detectors as a post training step.
- The paper is structured well and explains the related work in depth - this is important for readers no familiar with the domain.

### Weaknesses
- Some explanations and notation are hard to understand, particularly section 3.3 and 2.2.2. Instead of writing a bunch of equations a bit more explanation would be useful.
- The plot in Figure 2 requires better captioning or more description in the text. What is ECE? What does the baseline diagonal represent?
- The introduction / motivation needs to show an analysis of existing datasets or model performances to verify the claims made by the authors regarding the demographic imbalance.

### Questions
- Although the method is sound and novel, the results show only marginal improvement compared to existing techniques, such as performance improvements of 0.01 or 0.03 (Table 2). The training free methods FF Max and FF Discover perform worse than other post processing methods across the board. What is the cause of this? The result section needs more discussion on limitations and degree of improvement.
- The result also lacks comparison against existing deepfake detectors. The authors only use Xception and MobileNetv3 as backbone models. What were these models trained on?  There are far better performing or specialized deepfake detection networks (https://arxiv.org/pdf/2506.03007v1). Is this method similarly applicable for these models? 
- A robustness study is necessary. How does a model augmented with FFT handle the typical image attacks such as image compression, noise addition or augmentations. 

I am happy to change my score given a proper discussion to address the problems.

### Soundness
3

### Presentation
3

### Contribution
4
