# EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Camera Settings

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
The feed-forward based 3D Gaussian Splatting method has demonstrated exceptional capability in real-time human novel view synthesis. However, existing approaches are restricted to dense viewpoint settings, where camera view angles are less than 60 degrees. This limitation constrains their flexibility in free-viewpoint rendering across a wide range of camera view angle discrepancies. To address this limitation, we propose a real-time pipeline named EVA-Gaussian for 3D human novel view synthesis across diverse multi-view camera settings. Specifically, we first introduce an Efficient cross-View Attention (EVA) module to accurately estimate the position of each 3D Gaussian from the source images. Then, we integrate the source images with the estimated Gaussian position map to predict the attributes and feature embeddings of the 3D Gaussians. Moreover, we employ a recurrent feature refiner to correct artifacts caused by geometric errors in position estimation and enhance visual fidelity. To further improve synthesis quality, we incorporate a powerful anchor loss function for both 3D Gaussian attributes and human face landmarks. Experimental results on the THuman2.0 and THumansit datasets showcase the superiority of our EVA-Gaussian approach in rendering quality across diverse camera settings. Project page: https://anonymousiclr2025.github.io/iclr2025/EVA-Gaussian.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel EVA-Gaussian that uses multi-view images to synthesize human novel views. It introduces multi-view transformer into the field of human NVS and makes improvements to improve efficiency. It also proposes two concise feature refinement and anchor loss to enhance detail performance. Experiments on two prevailing datasets can well validate the claimed contributions and state-of-the-art performance compared to existing methods.

### Strengths
- Performance is good both quantitatively and qualitatively. The results are promising.
- This work can work with significant changes in camera viewpoint angles.
- The paper is easy to follow.

### Weaknesses
- The cross-domain generalizability should be further discussed. The paper seems to only conduct in-domain tests. As a generalizable method, it is necessary to evaluate its cross-domain generalizability on different datasets or data captured in the real world. 
- Resource cost. As multi-view images are aggregated by a unified attention module, there may be a higher GPU memory cost than previous works. Need more reported results, explanations, and discussion about the consumption.
- Typos. In figure 2, "4.1", "4.2", "4.3" should be "4.2", "4.3", "4.4" to align with the sections.

### Questions
Refer to weaknesses.

### Soundness
3

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
3

### Summary
The paper pays attention on real-time human novel view synthesis with a new pipeline called EVA-Gaussian across diverse camera settings. based on 3D Gaussian representation, which is composed of a position estimation stage, a attributes estimation stage, and a feature refinement stage. To improve position estimation, the paper designs an Efficient cross-View Attention (EVA) module to enhance multi-view correspondence retrieval. Then a recurrent feature refiner is proposed to mitigate geometric artifacts caused by position estimation errors. Experiments on THuman2.0 and THumansit demonstrate the effectiveness of the proposed pipeline.

### Strengths
1 A new human novel view synthesis pipeline composed of a position estimation stage, a attributes estimation stage, and a feature refinement stage with 3D Gaussian.

2 An Efficient cross-View Attention module to enhance learning of 3D Gaussian.

3 A recurrent feature refiner that fuses RGB images and feature maps to mitigate geometric artifacts caused by position estimation errors.

### Weaknesses
1 The movitivation of EVA shoule be expressed more clearly. The EVA module uses cross-view attention to enhance 3D Gaussian position learning. However, the idea have been used in various feature matching methods to establish correspondences across different veiws, such as SuperGlue [1], LoFTR [2], DKM [3]. It is suggested to explain the module more clearly.

[1] SuperGlue: Learning feature matching with graph neural networks. In CVPR, 2020.

[2] LoFTR: Detector-Free Local Feature Matching with Transformers. In CVPR, 2021.

[3] DKM: Dense Kernelized Feature Matching for Geometry Estimation


2 In Table 1, the paper presents the running time of each method wat a resolution of 256×256. However, GPS-Gaussian declares that it can synthesize 2K-resolution novel views with 25 FPS. This situation is suggeseted to be explained more clearly.

### Questions
The questions and suggestions are listed in the part of Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes a multiview-based method for human novel view synthesis using a 3D Gaussian representation, which consists of three main steps.  
First, it estimates position maps with cross-view attention. Next, it combines the position maps and raw images to estimate Gaussian parameters along with feature embedding. Finally, a feature refiner is applied to correct artifacts by splatting features and refining the output image.

### Strengths
- The paper is well-written, well-structured, and clear.
- The proposed EVA module is effective in initializing the Gaussian position, outperforming one-step Gaussian parameter regression.
- Incorporating features with original 3D Gaussian and splatting to refine rendering results is interesting.

### Weaknesses
- There are concerns about time expansion regarding the project page provided in the abstract.
- Methods like GPS address the sparse-view human Gaussian splatting problem, which conflicts with lines 33–35 of the abstract.
- The claim of "diverse camera settings" is somewhat overstated, as the method cannot resolve single-view settings, which are addressed by other approaches, such as HumanSplat [1].
- The method cannot infer the back view when only front stereo views are provided, as demonstrated in their video.
- The ablation study implies that the effectiveness of anchor loss is limited.
- There are no examples of in-the-wild scenarios presented.
- The video exhibits issues with jumping and transparency.


[1] HumanSplat: Generalizable Single-Image Human Gaussian Splatting with Structure Priors

### Questions
- Can the method only interpolate between input views, or is it able to hallucinate invisible parts?
- What causes the jumping and transparency issues in the video?
- How does the shifted window embedding strategy work? Are the window positions for cross-attention consistent across different image sources?
- Does "any" in line 197 include views such as up and side-up perspectives?
- Why do regularizing opacities help ensure consistency? Could it contribute to transparency artifacts, as seen in the video results?
- GPS results show broken hands and feet; why does the proposed method avoid this issue? Since the method does not use human priors, would it also work for objects if the dataset were changed?
- What specific baseline is used for the ablation of the EVA module?
---
I may change my opinion depending on the authors' rebuttal and whether they can address my concerns.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a real-time human novel view synthesis framework that leverages 3D Gaussians as the core representation. The approach employs a cross-view attention module to estimate geometric properties for each 3D Gaussian and uses image features to predict their attributes. The novel view image is generated through a splatting technique refined in a recurrent manner.

### Strengths
The experimental results demonstrate improved performance over GPS-Gaussian, particularly in the facial region.

### Weaknesses
he proposed pipeline shows considerable similarity to GPS-Gaussian. While GPS-Gaussian uses cost volume to estimate depth from two selected views, this work relies on cross-view attention. Furthermore, embedding image features into depth maps is also similar in both approaches.

The novelty and specific contributions of the cross-view attention module are unclear. Cross-view transformers, as implemented here, have been extensively explored in prior work, such as [A] and [B]. A more thorough analysis and additional experiments are needed to compare the proposed EVA module with existing techniques.

The use of facial landmarks as a regularization method is intuitive but not unique. While many related works are focused on human head avatars, this approach is already well-established and should be appropriately credited and compared with existing methods.

Although the paper emphasizes that the proposed approach is real-time and adaptable to various camera settings, this adaptability seems largely due to the two-view correlation strategy, which should also apply to GPS-Gaussian. It is unclear what unique contributions in this work specifically enhance real-time performance and camera adaptability.

The experiment videos show noticeable artifacts, similar to those seen in GPS-Gaussian, including transparent Gaussians in novel views and abrupt transitions during smooth view changes.

[A] Zhou, Brady, and Philipp Krähenbühl. "Cross-view transformers for real-time map-view semantic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
[B] Pham, Trung X., Zhang Kang, and Chang D. Yoo. "Cross-view Masked Diffusion Transformers for Person Image Synthesis." arXiv preprint arXiv:2402.01516, 2024.

### Questions
Given the points raised in the weaknesses, could the authors clarify the main novelty and contribution of this paper?
What specific advantages does the proposed EVA module offer?

### Soundness
3

### Presentation
2

### Contribution
2
