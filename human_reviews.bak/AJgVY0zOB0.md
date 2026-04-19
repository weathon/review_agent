# Weakly-supervised Camera Localization by Ground-to-satellite Image Registration

- Decision: Reject
- Scores: 6, 6, 3

## Abstract
The ground-to-satellite image matching/retrieval was initially proposed for city-scale ground camera localization. Recently, more and more attention has been paid to increasing the camera pose accuracy by ground-to-satellite image matching, once a coarse location and orientation has been obtained from the city-scale retrieval.  This paper addresses the same scenario. 
However, existing learning-based methods for solving this task require accurate GPS labels of ground images for network training. 
Unfortunately, obtaining such accurate GPS labels is not always possible, often requiring an expensive RTK setup and suffering from signal occlusion, multi-path signal disruptions, \etc. 
To address this issue, this paper proposes a weakly-supervised learning strategy for ground-to-satellite image registration. It does not require highly accurate ground truth (GT)
pose labels for ground images in the training dataset. Instead, a coarse location and orientation label, either derived from the city-scale retrieval or noisy sensors (GPS, compass, \etc), is sufficient. 
Specifically, we present a pseudo image pair creation strategy for cross-view rotation estimation network training, and a novel method that leverages deep metric learning for translation estimation between ground-and-satellite image pairs.
Experimental results show that our weakly-supervised learning strategy achieves the best performance on cross-area evaluation, compared to the recent state-of-the-art methods that require accurate pose labels for supervision, and shows comparable performance on same-area evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on camera localization with ground-to-satellite registration but uses a weakly supervised setting. It assumes that the pose label during training is not accurate. The proposed method first estimates the orientation with a self-supervision. Then the translation is estimated using feature similarity matching which is supervised by a soft-margin triplet loss. The proposed method is evaluated on KITTI and VIGOR datasets with quantitative comparisons. It outperforms most of the SOTA methods, except a recent work in ICCV 2023.

### Strengths
+ The writing is mostly clear. 
+ The motivation and problem formulation are interesting.
+ The proposed method is evaluated on two major datasets with different settings. Although it does not outperform some SOTA methods, the result is still impressive for a weakly supervised method.
+ The proposed method is more robust on initialization errors than previous methods.
+ The limitations and other discussions are also very interesting.
+ The visualizations and figures are well presented.

### Weaknesses
-	My main concern is about the setting of the proposed method. Given that the accurate pose label is not available for some places, it is still possible to get small-scale accurate labels at known locations. In this case, a semi-supervised setting could be more suitable than a weakly supervised formulation. The small-scale accurate label could also help improve the performance if properly designed.
-	Some suggestions to think about (Not a weakness): It might also be interesting to consider larger errors for the training images, for example, the satellite-view image has a certain probability of being an incorrect match of the query but is still close to the ground-truth locations. 
-	There is still room for improvement in the writing. The motivation part of using the proposed approach could be improved:” Leveraging a blackbox network to regress……”. Minor issues: RTK is not explained when it first appears.

### Questions
-	One advantage of weakly supervised learning is that more training data could be used. There might be some ways to demonstrate that more training data could be obtained with less accuracy requirement on the label.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a technique able to register satellite images with images taken from ground, e.g. by a camera mounted on a car. The results are promising on existing datasets, especially when compared to other existing methods. The idea setting this technique apart from others is the abandonment of rigorous exploitation of ground truth camera poses. The paper is well structured and reads easily. Estimating rotation and translation of ground cameras on large scale is a significant problem in many applications. The paper, altough not explicitely mentioned, target representation learning as it addresses the gap in feature representation between overhead satellite images and perspective
 images taken by the ground camera.

### Strengths
Registration of large baseline images is still a challenging problem, especially when it comes to the extreme. Registration of top view and ground images is especially difficult as image features for matching are rare, e.g. because of occlusion, out of plane rotation, etc. The idea in this work is to circumvent this matching problem by considering CNN features on much higher level with much larger receptive fields and semantic information.
Another idea of the technique is the possible avoidance of ground truth camera poses needed for training the model. The inference of rotation is thereby trained by a self-supervised learning approach, while translation is interpreted as correlation problem between satellite images and ground images. The training is supervised by a few labeled ground truth poses similar to metric learning problems.
The results in table 1 are promising, altough a gap between the solutions w vs. w/o knowledge of ground truth poses still exists. The comparison with other SOTA methods clearly show superiority of the technique (table 2).

### Weaknesses
The paper is well structured and readable, however I suggest to improve the introduction by motivating in much more clarity the matching problem and the idea of how you represent the problem of learning the high-level features and the registration via the correlation. The approach seems to be analog to popular representation learning approaches using contrastive learning. By focusing in you introduction and motivation on these problems and novelties, the paper would make clear its contributions to representation learning, the main topic of this conference.

At some places in the paper I found incorrect statements:
- page 2: "neural networks are inherently sensitive to rotations ..." in its generality this statement needs to be refused as e.g. equivariant networks address this problem (https://arxiv.org/pdf/2205.07362.pdf)
- page 2: "... synthesised overhead view image ..." the sentence gives the reader the impression that satellite images are always "overhead view" orthographic images which of course is not the case
- page 4: "magnitude of the rotation" ?
- page 4: "the equivariance property of convolution to translations ..." convolution is invariant to translation
- page 5: Eq 2 descibes the correlation coefficient but later on page 14 a convolution was used

### Questions
- page 4: Eq.1. the rotation is parametrised so why not writing this? R(a)
- page 4: what does |R-R*| mean? L1 of ... this formulation needs a more precise writing
- would it not be benificial to minimise R^(-1)*R → I ?, as R is SO(3)

The satellite images used in this work are 512x512pixel. In practice satellite images are orders of magnitude larger. To be useful in practice, the method needs to find regions of interest in satellite images in order to perform registration. How could you achieve this?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes to register an image captured on the ground to a satellite image in two stages. In the first stage, the ground image is projected to a bird's-eye view, and the 2D rotation and translation are estimated between the projected ground image and the satellite image. In the second stage, the 2D translation within the rotated ground image is re-estimated by dense feature patch matching. To facilitate the training, the authors synthesize bird's-eye view images with known rotation and translation as self-supervision from the satellite images. To demonstrate the effectiveness of the proposed methods, the authors conduct experiments on the KITTI and Vigor datasets and achieve better performance than some recent works on ground-satellite image registration.

### Strengths
- The method is clearly introduced and easy to implement.
- Self-supervision from the bird's-eye view synthesis is a reasonable design.
- The experiments are well conducted in terms of the evaluated metrics and ablation studies.

### Weaknesses
In general, the paper lacks sufficient insights and contributions:
- The bird's-eye view synthesis in Sec. 3 needs to be further investigated and justified. 
   - The synthesized bird's-eye view could be dramatically different from the projected bird's-eye view from the ground image due to parallax, occlusion, and appearance changes. It is very difficult to learn feature representations that are invariant to these changes. 
   - In order to convince the audience that the gap is small, the authors should a) introduce their data augmentation in detail, b) analyze the feature differences between the projected and actual bird's-eye view images.
- An important reference is missing: "OrienterNet: Visual Localization in 2D Public Maps with Neural Matching", CVPR '23. 
  - The proposed method in this paper is very similar to this missing reference except for the self-supervision. The OrienterNet has achieved better performance than the paper, and more importantly, it is better written with deeper insights. 
  - To highlight the difference, the authors should clearly describe why precise supervision is difficult, which does not seem to be the case for me since we can get precise reconstructions even from a mobile phone and then register a sequence of images to the satellite images.

### Questions
To decouple the contribution between model structure and supervision, can we do supervised training and see the upper-bound?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
