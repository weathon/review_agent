# Loc$^{2}$: Interpretable Cross-View Localization via Depth-Lifted Local Feature Matching

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
We propose an accurate and interpretable fine-grained cross-view localization method that estimates the 3 Degrees of Freedom (DoF) pose of a ground-level image by matching its local features with a reference aerial image. Unlike prior approaches that rely on global descriptors or bird’s-eye-view (BEV) transformations, our method directly learns ground–aerial image-plane correspondences using weak supervision from camera poses. The matched ground points are lifted into BEV space with monocular depth predictions, and scale-aware Procrustes alignment is then applied to estimate camera rotation, translation, and optionally the scale between relative depth and the aerial metric space. This formulation is lightweight, end-to-end trainable, and requires no pixel-level annotations. Experiments show state-of-the-art accuracy in challenging scenarios such as cross-area testing and unknown orientation. Furthermore, our method offers strong interpretability: correspondence quality directly reflects localization accuracy and enables outlier rejection via RANSAC, while overlaying the re-scaled ground layout on the aerial image provides an intuitive visual cue of localization performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a fine-grained cross-view localization approach that leverages off-the-shelf depth estimation and integrates local feature matching with scale estimation for robust pose estimation, featuring an analytic pose head that incorporates RANSAC for outlier filtering.

### Strengths
1. The analytic pose head, combined with RANSAC and overlay-based alignment, offers reliable diagnostics of correspondence quality.

2. The method leverages off-the-shelf depth estimation, which is already well-established and helps improve localization performance.

### Weaknesses
1. The method relies on off-the-shelf depth estimation, while depth provides rich geometric information, its inclusion appears to improve localization accuracy only marginally. Furthermore, were the depth models used here trained on datasets such as KITTI?

2. The results in Table 3 seem unintuitive: ignoring scale should significantly impact matching quality, yet the performance remains close to the full method. How do the authors explain this?

3. Running RANSAC on a large number of correspondences may be computationally expensive, and could its robustness degrade when the outlier ratio is high?

4. Since the approach depends heavily on local features and depth priors, it may overfit to VIGOR-style scenes and struggle to generalize to novel or unseen environments.

### Questions
1. Does the rotation matrix R account for pitch and roll components?

2. The results show that the method improves localization accuracy more than orientation. Do the authors have an explanation for this? You mention that orientation often aligns with road direction. Why does the local feature matching fail to capture this information effectively?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a cross-view localization method that first builds matches between ground and aerial images, and then analytically computes the camera pose from these matches. The idea is reasonable and demonstrates good generalization ability.

### Strengths
The paper is well-written and easy to follow. The idea is clear and makes sense to me.

### Weaknesses
1. **Purpose of “aerial points” (Line 696)**  
   - What are the “aerial points” used for? I did not find a clear explanation. Do you match ground images to **regular grids** sampled on the aerial images? If yes, please justify this design choice. My concern is that downsampling features to a grid may degrade matching accuracy.

2. **Resolution vs. Accuracy: feature maps and aerial-grid density**  
   - I suspect localization accuracy depends on (i) the **feature-map resolution**, (ii) the **grid resolution** on the aerial images, and (iii) the **original image resolution**. How do you balance these factors? A deeper analysis would help clarify the sensitivity to these design choices. Please provide ablations such as:  
     - Fix the aerial grid and vary the **backbone** to increase feature-map resolution.  
     - Fix the feature-map resolution and vary the **aerial-grid resolution**.  
     - Optionally, explore multi-scale/FPN settings to study trade-offs.  

3. **Temperature parameter in Equation (1)**  
   - Are the matching results sensitive to the temperature parameter \( \tau \)? Please include a sensitivity analysis (e.g., performance vs. \( \tau \)) and report the chosen value and tuning protocol.

4. **Aerial image scale (ground sampling distance, GSD)**  
   - Since you directly match ground to aerial features, the **GSD** (e.g., meters per pixel) will influence representational fidelity and, hence, matching accuracy. Have you evaluated aerial images captured at **higher altitudes** (coarser GSD)? Please report performance across different GSDs/heights to quantify robustness and discuss how scale changes affect the aerial feature maps and matching.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper estimates the 3-DoF pose of a ground-level image based on a reference aerial image. This task is achieved by conduct local feature matching, monocular depth predictions and scale-aware Procrustes alignment in an end-to-end framework. Experiments show state-of-the-art accuracy and strong interpretability.

### Strengths
1 The paper is well-written and easy to follow. The  proposed framework are presented clearly.

2 The attempt to formulate cross-view matching within a differentiable training pipeline is a commendable direction.

### Weaknesses
1 The authors claim the method's effectiveness, but the experimental results show that it does not achieve state-of-the-art (SOTA) performance on several key metrics.

2 The method appears to be a straightforward composition of existing modules. The core challenges of cross-view matching do not seem to be fundamentally addressed by a novel technical contribution. This relates to my questions Q2-Q4.

3 The generalization ability is a major concern. The method is evaluated on a limited data split. To better validate robustness, I strongly suggest an additional cross-dataset evaluation, for instance, ​training on KITTI and testing on VIGOR.

### Questions
1 In the introduction, the term "scale-aware Procrustes alignment" is used. For clarity, could the authors confirm if this refers specifically to a ​2D similarity transformation​? A more detailed explanation in the text would be helpful.

2 The method seems to assume that the ground-level query images are oriented orthogonal to gravity. Could the authors explicitly state this assumption? Its practical limitation should be discussed, as consumer-grade images (e.g., from mobile phones) with arbitrary orientations may not easily be transformed into a canonical BEV space.

3 The satellite imagery used is a Digital Orthophoto Map (DOM), which does not account for building facades, unlike a True Digital Orthophoto Map (TDOM). This creates a fundamental misalignment with BEV images, especially in urban areas with tall structures. ​How does the proposed method account for or mitigate this inherent domain discrepancy?​​

4 The accuracy of the pose (location and orientation) annotations in KITTI and VIGOR is critical, as they provide the supervision signal. ​Have the authors validated the accuracy of these annotations?​​ If there is inherent noise, how does the method ensure reliable supervision? 

5 Since the method establishes correspondences, an informative ablation would be to compare the performance of using a ​homography​ versus the proposed ​scale-aware Procrustes alignment.

6 The authors mention using RANSAC. ​For which specific geometric model is RANSAC employed?​​ Is it used to estimate a homography for outlier rejection before the Procrustes alignment?

### Soundness
2

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
5

### Summary
Loc² proposes an interpretable fine-grained cross-view localization method to estimate the 3-DoF pose of ground-level images by matching local features directly with aerial images. Unlike BEV transformation-based or global descriptor-based prior works, it learns image-plane correspondences under weak camera pose supervision, lifts ground points to BEV via monocular depth predictions, and estimates pose/scale using differentiable scale-aware Procrustes alignment. Key contributions include: direct cross-view local feature matching avoiding BEV distortion, strong interpretability via explicit correspondences and layout overlay, compatibility with both metric/relative depth, and state-of-the-art performance on KITTI and VIGOR.

### Strengths
- **Strong interpretability and practicality**: Explicit local correspondences enable RANSAC outlier rejection, and scaled ground layout overlay provides intuitive localization quality cues, addressing the "black box" issue of prior methods.
- **Robust and flexible depth adaptation**: Compatible with both metric and relative depth predictors, with minimal performance degradation when using lightweight relative depth models, enhancing real-world deployment potential.

### Weaknesses
The paper claims that 'However, these methods offer limited interpretability, as they cannot explicitly identify which objects in the ground view correspond to those in the aerial view.'  However, there is no experiments to directly compare the advantages and disadvantages of "direct image-plane matching (used in Loc²)" and "BEV transformation followed by matching". Specifically, the paper does not construct a BEV variant of Loc² (e.g., keeping the DINOv2 feature extractor, scale-aware Procrustes alignment, and loss function unchanged, but only converting the ground-level image to BEV as input for feature matching) for comparison. Without such a comparison, even though Loc² uses depth information, it cannot quantitatively prove: (1) how much performance loss BEV transformation actually causes in ground-aerial matching; (2) whether the direct matching method truly avoids the information loss of BEV transformation (e.g., whether the height information retained by direct matching with depth contributes to better correspondence accuracy).

### Questions
The paper argues that BEV transformation causes ray-directional distortions and height information loss, but does not conduct a direct comparative experiment between "direct image-plane matching" and "BEV transformation + matching". Have you ever constructed a BEV variant of Loc² (e.g., converting ground-level images to BEV first, then using the same DINOv2 feature extractor, matching head, and scale-aware Procrustes alignment as the original method) and compared its performance with the original direct matching？

### Soundness
3

### Presentation
3

### Contribution
3
