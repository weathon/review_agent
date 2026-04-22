# EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Egocentric vision is essential for both human and machine visual understanding, particularly in capturing the detailed hand-object interactions needed for manipulation tasks. Translating third-person views into first-person views significantly benefits augmented reality (AR), virtual reality (VR) and robotics applications. However, current exocentric-to-egocentric translation methods are limited by their dependence on 2D cues, synchronized multi-view settings, and unrealistic assumptions such as the necessity of an initial egocentric frame and relative camera poses during inference.  To overcome these challenges, we introduce *EgoWorld*, a novel framework that reconstructs an egocentric view from rich exocentric observations, including point clouds, 3D hand poses, and textual descriptions. Our approach reconstructs a point cloud from estimated exocentric depth maps, reprojects it into the egocentric perspective, and then applies diffusion model to produce dense, semantically coherent egocentric images. Evaluated on four datasets (*i.e.,* H2O, TACO, Assembly101, and Ego-Exo4D), *EgoWorld* achieves state-of-the-art performance and demonstrates robust generalization to new objects, actions, scenes, and subjects. Moreover, *EgoWorld* exhibits robustness on in-the-wild examples, underscoring its practical applicability. 
Project page is available at https://redorangeyellowy.github.io/EgoWorld/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EgoWorld, a two-stage framework to translate a single exocentric image into a high-fidelity egocentric view. Stage one extracts geometric and semantic cues (a sparse reprojected point cloud, 3D egocentric hand pose, and text) from the exocentric image. Stage two uses a diffusion model to inpaint and reconstruct the final egocentric view, conditioned on these cues. The method shows great performance across four datasets.

### Strengths
- The paper addresses the challenging task of translating third-person to first-person views, a problem with clear applications in AR/VR and robotics.
- The method design makes a lot of sense to me. It logically separates 3D geometric reprojection (creating a sparse map) from generative inpainting (using diffusion with multimodal cues) to reconstruct the final image.
- EgoWorld achieves strong results on four different datasets, with thorough testing against unseen objects, actions, scenes, and subjects, including in-the-wild examples. The provided ablation study is helpful.

### Weaknesses
- I feel the method's success critically depends on the 3D egocentric hand pose estimator ($\phi_{ego}$), which predicts the *egocentric* hand pose from the *exocentric* image. This key component is only briefly described in the main paper. It's unclear what insights make this difficult cross-view prediction possible. More importantly, the paper should justify why it uses hand poses ($P_{exo}$ and $P_{ego}$) for the Umeyama alignment instead of simply predicting the egocentric *head pose*. Using the head pose seems like a much more direct way to find the camera's transformation matrix, and the benefit of the hand-based alignment is not explained.
- The proposed method is image-based, but the paper's motivation, applications, and datasets are all video-centric. It would be helpful if the authors discuss how this single-frame approach could be extended to a full video-to-video translation framework. A discussion on the expected challenges, such as maintaining temporal consistency across generated frames, would be a valuable addition.
- The name of the first stage, "Exocentric View Observation" is confusing. This stage *outputs* an "egocentric RGB map $S_{ego}$" and a "3D egocentric hand pose $P_{ego}$". A name like "Exocentric-to-Egocentric Cue Translation" would be more accurate and less confusing.

### Questions
See weaknesses for questions.

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
This paper proposes EgoWorld, a two-staged framework that generates an egocentric view from external camera views. EgoWorld extracts multimodal cues, such as textual descriptions, hand pose, and depth map from external views. Then, the egocentric view is estimated sparsely from the pointcloud generated from depth map and relative transform between egocentric and exocentric view. This is further refined using predicted hand pose and a textual description via a latent diffusion model to produce an egocentric image. Method is evaluated on H2O, TACO, Assembly101, and Ego-Exo4D and achieves state-of-the-art performance.

### Strengths
* Framework utilizes multi-modal information, such as textual description, from an exocentric view in a plausible way.
* The overall proposed method is well presented with clarity.

### Weaknesses
* Evaluation compares the proposed framework to novel view synthesis methods, many of which are outdated and not specialized for the egocentric setup. This is particularly problematic when there are methods that focus on an egocentric view.
* Egocentric view transform is estimated based on the egocentric hand pose predicted from the external view. This would be highly dependent on the learned camera setup and the device's visual from an external view. Using ground-truth data in some evaluation further weakens the claim that such a framework is advantageous.

### Questions
* Some experiments in the appendix should instead be in the main paper (e.g., results using prediction instead of GT for H2O).
* Previous exocentric-to-egocentric translation approaches can be evaluated with the same setup and framework, or partially ablated to use the same input (one camera, single frame) to demonstrate the advantage of the proposed framework.
* In particular, Exo2Ego's hand-object interaction layout requirement doesn't significantly differ from using hand pose in the proposed method. They can be easily compared by providing a hand-only layout estimated in the same way as the proposed method.
* The evaluation should show whether using hand pose for relative camera pose estimation is more advantageous than a wider variety of baselines (despite one naive comparison in the appendix). The claim of being camera parameter-free is not significant without it.

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
3

### Summary
The paper proposes a new framework, EgoWorld, for translating third-person views into first-person (ego) view images. Unlike prior methods that depend on multi-view inputs or known ego-camera poses, EgoWorld introduces a two-stage framework. It first extracts multimodal information—such as 3D hand pose, textual description, and depth—from a single third-person image using existing specialized models. These modalities are then integrated into a diffusion-based image reconstruction model to generate the first-person view. Experiments on four public datasets (H2O, TACO, Assembly101, and Ego-Exo4D) demonstrate that EgoWorld achieves state-of-the-art performance and shows promising generalization to unseen scenarios.

### Strengths
- The concept of using textual cues to enhance visual synthesis across viewpoints is intuitive and promising.

- The combination of sparse maps as partial observation with textual description for completion is interesting.

### Weaknesses
1. It would be better to highlight the difference compared with 4Diff or Exo2Ego. 4Diff adopts depth maps/point clouds while Exo2Ego adopts hand layout. In addition to depth maps and hand poses, the proposed method adopts an additional textual description for LDM as a condition, which like a combination.

2. The introduction of textual cues is the key of the proposed method and provides semantic alignment. However, the quality of the text and its contribution to the results is unclear. It would be better to include metrics like clip score to measure the alignment between text and generated images. Moreover, in Fig.C, only the incorrect textual description for hand-held objects is shown, which is insufficient. Most importantly, it would be interesting to see the balance between sparse maps and incorrect textual description like Fig.D. Also, those analysis are important and should in the main paper.

3. The comparison with baselines seems unfair. For example, CFLD is primarily designed for pose-guided person image synthesis in general human view translation, not specifically for hand-centric first-person generation. It would be better to incorporate more directly relevant comparisons (e.g., Exo2Ego or EgoExo-Gen). 

4. It would be better to provide qualitative results and more analysis for unseen objects or novel environments and to show its strong generalization capability. Based on L.81-84 ("Yet, it depends heavily...overfitting to the training dataset"), overfitting is easy to happen and can be alleviated by exocentric observations. However, Fig.4 shows that the images generated by the proposed method are close to the groundtruth even in unknown regions, which also raises the concerns about overfitting.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the task of exocentric-to-egocentric cross-view translation. The authors propose a two-stage training framework to tackle this problem: (1) extracting point clouds, 3D hand poses, and textual descriptions from the exocentric view, and (2) reconstructing the egocentric view based on these extracted cues. The key idea is to leverage multimodal information, including textual supervision, 3D geometric cues, and hand pose data to better guide the exo-to-ego synthesis process. The proposed approach is evaluated on four datasets: H2O, TACO, Assembly101, and Ego-Exo4D.

### Strengths
- The paper is well-written and easy to follow. 

- The idea of leveraging multimodal information—including rich textual supervision, 3D geometric cues, and hand pose data—is simple, intuitive, and effective. 

- The evaluation is comprehensive, covering four representative ego–exo datasets.

### Weaknesses
- **Qualitative Results**: Most of the qualitative examples are drawn from the H2O dataset except for Fig 4. It would be beneficial to include more examples from the other three datasets—TACO, Assembly101, and Ego-Exo4D—to showcase a broader range of scenarios beyond desktop activities. This would help readers gain a more comprehensive understanding of the proposed approach’s generalizability across diverse environments.

- **Ablation Study**: In Table 3, the presented ablation results are somewhat limited. Including a more complete ablation table in the supplementary material would provide a clearer and more comprehensive analysis of the contribution of each component within the framework.

- **Backbone Model**: To further improve reconstruction quality, it may be beneficial to adopt more recent video diffusion models, such as Wan or CogVideoX, as the backbone. The current choice, LDM (published in 2022), appears somewhat outdated compared to recent advancements in video generation.

- **Quantitative Evaluation**: The quantitative evaluation primarily focuses on image-based quality metrics. It would strengthen the analysis to include additional metrics assessing hand generation accuracy and object-level generalization, providing a more holistic evaluation of the model’s performance.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
