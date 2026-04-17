# Enabling Your Forensic Detector Know ​How Well​ It Performs on Distorted Samples

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Generative AI has substantially facilitated realistic image synthesizing, posing great challenges for reliable forensics. When image forensic detectors are deployed in the wild, the inputs usually undergone various distortions including compression, rescaling, and lossy transmission. Such distortions severely erode forensic traces and make a detector fail silently—returning an over-confident binary prediction while being incapable of making reliable decision, as the detector cannot explicitly perceive the degree of data distortion. This paper argues that reliable forensics must therefore move beyond "is the image real or fake?" to also ask "how trustworthy is the detector's decision on the image?" We formulate this requirement as Detector's Distortion-Aware Confidence (DAC): a sample-level confidence that a given detector could properly handle the input. Taking AI-generated image detection as an example, we empirically discover that detection accuracy drops almost monotonically with full-reference image quality scores as distortion becomes severer, while such references are in fact unavailable at test time. Guided by this observation, the Distortion-Aware Confidence Model (DACOM) is proposed as a useful assistant to the forensic detector. DACOM utilizes full-reference image quality assessment to provide oracle statistical information that labels the detectability of images for training, and integrates intermediate forensic features of the detector, no-reference image quality descriptors and distortion-type cues to estimate DAC. With the estimated confidence score, it is possible to conduct selective abstention and multi-detector routing to improve the overall accuracy of a detection system. Extensive experiments have demonstrated the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes DACOM, a neural regressor that predicts the probability that a given forensic detector will correctly classify a (possibly distorted) image. The key insight is that FR-IQA scores correlate monotonically with detector accuracy conditioned on distortion type. Training labels are obtained by bucketing FR-IQA scores per distortion type and mapping bin-wise balanced accuracies to [0,1]. At inference DACOM uses detector features, NR-IQA features and a distortion-type embedding to estimate sample-level confidence without references. The score enables “selective abstention’’ and “top-1 routing’’ among detectors and improves overall accuracy on several benchmark.

### Strengths
1. The paper introduces a novel framework that tackles reliability estimation, which is orthogonal and complementary to existing works.
2. It provides a principled, detector-conditioned confidence definition and a practical solution that avoids requiring reference images at test time.
3. Extensive experimental validation on diverse detectors and a broad spectrum of distortions is conducted. Ablations clearly show each component’s contribution.
4. The paper is well written and easy to follow.

### Weaknesses
1. Evaluations rely mainly on Balanced Accuracy and EER. In practice, real and fake samples are imbalanced. Would precision-recall–based measures (e.g., AUC-PR, F1) change the conclusions?
2. The inference pipeline runs QualiCLIP, ARNIQA and the detector for every input may be costly on edge devices. A detailed analysis of timing/FLOPs is missing. 
3. While abstention can improve safety, it also off-loads decisions to human operators and potentially offers adversaries a mechanism to trigger systematic abstention. The paper lacks discussion of these risks and possible mitigations.

### Questions
1. How would the results change if Balanced Accuracy is replaced with AUC-PR or F1 measures?
2. Please report the inference overhead of DACOM and compare it to baselines.
3. Please add more discussions about ethical aspects. 
4. Minor issue: Text in Fig. 2 and Fig.6 is tiny, please enlarge.

### Soundness
4

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
5

### Summary
This paper addresses the problem that forensic detectors for AI-generated images produce predictions without indicating reliability when test images undergo various distortions such as compression and noise. The authors propose DACOM (Distortion-Aware Confidence Model), which uses full-reference image quality assessment metrics to label training data with detectability scores during training, then learns to predict sample-level confidence using detector features, no-reference image quality descriptors, and distortion-type information at inference time.

### Strengths
1. The paper's motivation and perspective are well-founded and novel. Few existing works approach AI-generated image detection from the lens of image quality assessment to evaluate detector reliability under distortions. This angle provides a fresh and meaningful contribution to the forensics community.
2. The paper is well-organized with clear logical flow. The authors systematically progress from problem identification (Section 3 analysis) to method design (Section 4), making the work easy to follow. The empirical analysis establishing the correlation between FR-IQA scores and detection accuracy provides solid justification for the proposed approach.
3. The experimental evaluation is comprehensive and thorough. The authors provide extensive ablation studies (Section 5.5), test on multiple distortion types (both seen and unseen), evaluate across different datasets (Evaluation-dataset and Cross-dataset), and include detailed supplementary results in the appendix, all of which substantiate their claims with adequate evidence.

### Weaknesses
1. Concerns regarding the use of detector performance for labeling in Stage A. The authors use the detector's balanced accuracy on each bin to generate detectability labels (Equation 3-4). This raises several concerns: (a) If FR-IQA scores already exhibit monotonic correlation with detection performance (as shown in Section 3), why is the additional step of computing detector accuracy necessary? Could the FR-IQA scores themselves serve as supervision? (b) More critically, this design may limit generalizability—if training data is labeled using Detector A's performance, will DACOM trained on this data generalize well to Detector B? This detector-specific labeling could hinder practical deployment across different detection models. (c) The requirement to evaluate detector performance on large distorted datasets during Stage A significantly increases the computational cost of the training pipeline.
2. Limited discussion on robustness training and its interaction with the observed monotonicity. The authors address distortion robustness by applying light data augmentation (10% JPEG compression and blur) during detector training. However: (a) Only two distortion types are used for augmentation, which seems insufficient given the diversity of real-world distortions. (b) It remains unclear whether the monotonic relationship between FR-IQA and detection accuracy (Section 3) still holds when detectors are trained with more aggressive data augmentation strategies. If extensive augmentation flattens the performance curve across distortion levels, would DACOM's premise still be valid? This interaction between robustness training and the proposed method deserves further investigation.
3. Limited training data scale may restrict generalization. The model is trained on only 2,500 base images from a single dataset (ProGAN subset). While distortion augmentation expands this to 200K+ samples, the underlying content diversity remains limited. This could lead to: (a) Overfitting to the specific visual patterns in these 2,500 images. (b) Poor generalization to different content types, as evidenced by the noticeable performance drop in Cross-dataset evaluation (Table 4). Experimenting with larger and more diverse training sets would strengthen the claims about generalizability.
4. The proposed applications have practical limitations that merit further exploration. While the paper demonstrates two uses of DACOM—selective filtering and multi-detector routing—both have notable drawbacks: (a) Selective abstention necessarily reduces coverage, which may be unacceptable in applications like content moderation where all samples must be processed. (b) Multi-detector routing requires maintaining and running multiple detectors (6× computational cost in experiments), which may be prohibitively expensive for real-time systems. I suggest the authors explore alternative applications that leverage DACOM more seamlessly, such as incorporating confidence-aware calibration directly into a single detector's training or inference process, or using confidence scores to dynamically adjust decision thresholds rather than completely abstaining from prediction.

Overall assessment: Despite these concerns, I view this as a valuable contribution that introduces a novel perspective on detector reliability. If the authors can adequately address the above concerns—I would be inclined to raise my score.

### Questions
1. The formula y = 2×|BAcc - 0.5| assumes equal difficulty in improving accuracy across the entire range (e.g., 50%→75% vs. 75%→100%). However, achieving near-perfect accuracy is typically much harder than reaching moderate levels, suggesting a non-linear relationship.
Have you experimented with non-linear transformations (e.g., y = (2×|BAcc - 0.5|)^α with α > 1, or logarithmic scaling) that better reflect the diminishing returns at higher accuracy? Alternatively, why not use BAcc directly as labels without transformation? An ablation study comparing different label functions would clarify whether this linear design is optimal or just a convenient choice.

2. You train four DACOM variants with different FR-IQA metrics (Table 6) and all perform similarly. Does this mean the choice of FR-IQA is not critical? If so, why present four variants instead of selecting one? What guidance do you offer to practitioners on choosing the FR-IQA metric?

3. What is the parameter count and FLOPs of DACOM? Since it must run alongside the detector, efficiency matters. How does DACOM's overhead compare to the detector itself (e.g., DACOM adds X% latency)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Conventionally, fake image detectors assign a real/fake label. However, detecting these fake images in the wild involves dealing with post-processing operations modify the original signal and hence affect their detection. The authors try to develop an effective method through which one can also predict the confidence of the detector. This is especially challenging since, neural networks are not well-calibrated. In order to do this, the authors leverage image quality to measure the confidence. The change in image quality, depends on the reference image, which is conventionally not available during test time, in order to predict this, the authors train the neural network to do so. Experiments show improved calibration and the authors also show how this can be used in effectively detecting fake images.

### Strengths
1. The problem of uncertainty estimation in fake image detection is both interesting as well as practically relevant. It is also an understudied problem.
2. The focus of the study on distortions also makes the paper practically relevant.
3. Multi-Detector Routing and Confidence-Based filtering are good use cases for the method.

### Weaknesses
1. The experiment reported in section 3 would benefit from the inclusion of more details. For instance, what is the data used, what is the amount to which each post-processing operation is applied, etc. 
2. My main concern comes from the fact that the existing method seems to only account for single distortions. However, one can compose distortions (resize first, then blur later for instance). It is currently unclear to me as to how the method would work given these settings. 
3. The limitations should be discussed with further detail. The issues that the current method has with respect to data coming from different sources would be interesting and insightful to the community.
4. The plots have extremely small text and it can be hard to follow, it would be better if the text in the plots are much bigger than they currently are. Especially for the plots in the appendix and Fig 2.

Minor Weaknesses
1. Line 42-43: This statement is not correct. A lot of detectors use common post-processing operations as part of their training [1,2].

References,
1. Wang, S. Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A. (2020). CNN-generated images are surprisingly easy to spot... for now. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8695-8704).
2. Gragnaniello, D., Cozzolino, D., Marra, F., Poggi, G., & Verdoliva, L. (2021). Are GAN generated images easy to detect? A critical analysis of the state-of-the-art. arXiv preprint arXiv:2104.02617.

### Questions
1. Does the method currently account for multiple distortions, if not can it be made to account for this case?
2. For the training, why does equation 7 use MSE loss as opposed to the binary cross entropy loss?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose to train a predictor for calibration-type confidence in the sense of probability of the prediction being wrong. They do this specifically for GAN-generated image detection.

To do this, they use detector features, features from a predictor of distortion type, features from a predictor of no-reference image quality, and use this with an MLP on top to predict the confidence via regression.

They perform experiments on correlation between the predicted and the true calibration, they show the usefulness of the predictor for top-1 routing to GAN-generated image detectors, and for confidence based vote abstention, evaluated by a ranking measure.

### Strengths
- They perform experiments beyond just training the calibrator. 
    - It shows its usability for top-1 routing and for vote abstention / or low confidence flagging.
- if one would want to hide the fact that an image was generated by a deep learning model, then image distortions are a natural candidate for obfuscation, so the setting makes sense
- Clear idea 
- well readable paper

### Weaknesses
- it has not the greatest novelty, it is not what one would think has to be shown as an oral
- The dataset used for the main experiments, PROGAN. is from the pre-diffusion model era. 

It would be better to see the results for distortions also for diffusion datasets. They do this for the cross-evaluation in section 5.4 but it would be good to have done it also for section 5.3 and also for the confidence evaluation.

### Questions
none

### Soundness
3

### Presentation
4

### Contribution
3
