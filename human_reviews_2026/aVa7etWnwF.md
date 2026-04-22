# GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8, 6

## Abstract
Human perception for effective object tracking in 2D video streams arises from the implicit use of prior 3D knowledge and semantic reasoning. In contrast, most generic object tracking (GOT) methods primarily rely on 2D features of the target and its surroundings, while neglecting 3D geometric cues, making them susceptible to partial occlusion, distractors, and variations in geometry and appearance.
To address this limitation, we introduce GOT-Edit, an online cross-modality model editing approach that integrates geometry-aware cues into a generic object tracker from a 2D video stream. Our approach leverages features from a pre-trained Visual Geometry Grounded Transformer to infer geometric cues from only a few 2D images. To address the challenge of seamlessly combining geometry and semantics, GOT-Edit performs online model editing. By leveraging null-space constraints during model updates, it incorporates geometric information while preserving semantic discrimination, yielding consistently better performance across diverse scenarios. Extensive experiments on multiple GOT benchmarks demonstrate that GOT-Edit achieves superior robustness and accuracy, particularly under occlusion and clutter, establishing a new paradigm for combining 2D semantics with 3D geometric reasoning for generic object tracking. The project page is available at https://chenshihfang.github.io/GOT-EDIT.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the GOT-Edit method to integrate visual geometric knowledge into general object trackers through online model editing. First, the Visual Geometric Grounded Transformer (VGGT) trained on large-scale 3D annotated data is used to directly extract reliable geometric cues from 2D images. Secondly, to achieve seamless fusion of geometry and semantics, an online model editing strategy with null space constraints is designed. This adaptively fuses geometric information while retaining the semantic knowledge learned by the tracker, making its performance better than the simple fusion of geometric and semantic cues in a variety of scenarios. Finally, through extensive experiments on multi-class GOT benchmark datasets, GOT-Edit performs well in robustness and accuracy, especially in occluded and cluttered scenes. It has established a new paradigm for combining 2D semantics and 3D geometric reasoning for the field of general object tracking.

### Strengths
**1. Problem identification aligns with practical needs:** This paper accurately identifies the performance bottlenecks of current 2D general object tracking (GOT) under conditions of occlusion, background interference, and drastic changes in target geometry/appearance. It also proposes a targeted solution by integrating 3D geometric information, addressing the core pain point of "lack of scene depth understanding" in tracking tasks.

**2. Technical adaptability is reasonable:** Addressing the limitations of existing 3D-assisted tracking methods, which rely on additional 3D inputs such as RGB-D and point clouds, this paper employs VGGT to extract geometric features from 2D images, addressing the unavailability of 3D data in real-world scenarios. Furthermore, this paper improves AlphaEdit's offline null space constraint to online model editing, meeting the real-time requirements of tracking dynamic targets and backgrounds.

**3. Comprehensive experimental design:** Performance is verified on multiple benchmark datasets (such as AVisT, NfS, LaSOT, and GOT-10k), covering scenarios such as "in-distribution/out-of-distribution objects" and "poor visibility." Detailed ablation experiments verify the effectiveness of components such as semantic/geometric feature sources, null space constraints, and input regularization, resulting in clear results (see Tables 4 and 5).

### Weaknesses
1. The paper's proposal to "use 3D geometric features to compensate for the shortcomings of 2D semantics" is not a new idea. Existing work (e.g., Tan et al., 2025a/b; Chen et al., 2025b) has attempted to use 3D information for tracking, but has relied on additional 3D input. This paper merely changes the "source of 3D input" to "extracted from 2D images" (using VGGT). This is essentially an "adjustment of the input method" rather than an "original breakthrough in the tracking mechanism," and does not propose a completely new geometric-semantic fusion logic.

2. Model Editing Module: This is directly based on the nullspace constraint of AlphaEdit (Fang et al., 2025), merely changing "offline editing" to "online updating." It lacks theoretical innovation (such as nullspace calculation methods or constraint optimization), making it an "engineering tweak" rather than an "algorithmic originality."

3. Overall Framework: This is entirely based on the "track-by-detection" paradigm of ToMP (Mayer et al., 2022), adding only two components: "VGGT geometric feature extraction" and "nullspace fusion." No new tracking architecture (such as feature interaction mechanisms or localization head design) has been designed. This essentially replaces and supplements components of the existing framework.

4. The paper claims that "nullspace constraints protect semantic features from degradation," but fails to explain why geometric features inevitably fall into the nullspace of semantic features.

5. Some existing work (such as Wang et al., 2025, which uses VGGT for point tracking) has attempted to extract geometric information from 2D images. The paper does not clearly explain the fundamental difference between GOT-Edit and such work in terms of "tracking task adaptation," emphasizing only that "point tracking lacks semantics" without demonstrating the uniqueness of its semantic-geometric fusion.

6. The paper only notes that it is "inadequate for scenes with rapid motion and viewpoint changes" without further analyzing the underlying causes (e.g., estimation errors of geometric features in dynamic scenes? Improper adaptation of the dynamic weights of semantic-geometric fusion?), nor does it propose potential solutions, demonstrating a lack of understanding of the technical bottlenecks.

### Questions
1. Compared to existing work that "extracts geometric information from 2D images to assist tracking without requiring additional 3D input" (e.g., tracking methods based on monocular depth estimation), what is the original difference in the core mechanism (non-input method) of GOT-Edit? If the sole contribution is "the introduction of VGGT + null-space constraints," why does this combination constitute an innovative contribution?

2. Besides "converting AlphaEdit from offline to online," are there any other original improvements (e.g., optimizing the calculation of the null-space projection matrix, dynamic constraint strength adjustment)? If the sole contribution is "online," please justify the necessity of these adjustments for the tracking task—why are the dynamic update mechanisms of existing online tracking methods (e.g., ToMP) ineffective?

3. Why does geometric information fail in LaSOT's "fast motion" and "viewpoint changes" scenarios? Is it because the geometric features extracted by VGGT are insufficiently accurate, or is the fusion mechanism not well-suited to dynamic scenes? Please provide additional quantitative analysis and ablation experiments (e.g., to separately verify the effectiveness of geometric features in dynamic scenes).

### Soundness
2

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
4

### Summary
This paper proposed GOT, which, by introducing the geometric knowledge of VGGT, achieves highly competitive experimental results with smaller trainable parameters.

### Strengths
1.The motivation of this paper is quite clear, and the performance-related experiments have well demonstrated the contribution. It is reasonable that information from multiple views (2D + 3D) can be better represented.
2.Sufficient visualization clearly demonstrates the improvement of GOT compared to the baseline. The dataset involved in the experiment is sufficient to demonstrate the generalization ability of GOT.
3.The author's most significant contribution lies in demonstrating the existence of the missing geometric prior and the necessity of integration, and proposing an effective editing method to integrate multi-channel information.

### Weaknesses
1.The ablation experiment results in Table 5 seem unable to be compared horizontally with those in Table 1. There are also many models based on DiNOv2-L, and their performances vary. Therefore, the backbone part is still worth comparing.
2.I have concerns about parameter and inference efficiency. Specifically, Table 6 shows the proportion of reasoning time for each part. The VGGT introduced as an innovation point accounts for the vast majority, and no horizontal comparison is made. The author only presented the trainable parameters and did not show the complete cost of model deployment. In fact, the results of some methods in specific scenarios (e.g., Table 1, Full Overlap column, PiVOT-378 row) do not differ much from GOT. Stating these differences clearly helps to clarify contributions.

### Questions
1.Apart from the limitations in comparison with related work, what improvement directions does the author think there are for GOT itself (e.g., the utilization of backbone and the future work of GOT)?
2.Which of the author 's experiments completed the declaration of 'without degrading the dominant semantic features'(on line 100)?
3.The author believes that the most core contribution is the creation of the engineering model or the proposal of an editing method that can effectively integrate the information of the two types of models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GOT-Edit, the first framework that integrates geometry-grounded reasoning into generic object tracking without requiring explicit 3D inputs. By introducing constraints during online model editing to preserve learned semantic consistency, GOT-Edit prevents model degradation while incorporating geometric cues that are often overlooked by conventional 2D trackers. Through online model editing with a null-space constraint, the method adaptively fuses geometric information while retaining semantic knowledge, thereby achieving greater robustness under occlusion, clutter, and visual ambiguity. The proposed approach has been evaluated on multiple state-of-the-art (SOT) benchmarks and demonstrates competitive performance.

### Strengths
1. The introduction of a geometry-aware correspondence learning mechanism is interesting and effective in visual tracking.
2. The proposed approach has been evaluated on multiple state-of-the-art (SOT) benchmarks and demonstrates competitive performance.

### Weaknesses
- Compared with the baseline, VGGT introduces additional computational overhead. It is recommended that the authors include a speed and FLOPs comparison to better illustrate efficiency.
- The VGGT component continuously updates learned knowledge during training. How does the method address potential error accumulation in this iterative learning process?
- While obtaining geometric features directly from 2D data without using 3D inputs can reduce data collection costs, it raises concerns about whether accurate geometric representations can be captured in this manner.
- The experiments are conducted using only 3 frames, and the observed performance gains might primarily come from the multi-template setting rather than the proposed algorithmic innovation. Moreover, since the model aims to learn geometric knowledge from multiple frames, there is a concern that using only three frames may be insufficient to capture rich and reliable geometric representations.

### Questions
See Weaknesses.

### Soundness
3

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
5

### Summary
This paper proposes GOT-Edit, a geometry-aware framework for online model editing in 2D object tracking. The key innovation lies in dynamically updating the target model during tracking, guided by multi-geometry information. The framework integrates a geometry-aware representation module with an online editing mechanism that refines the target model incrementally based on geometric consistency.

### Strengths
1. Introducing 3D geometric cues into online model editing for object tracking is both innovative and practical, providing a new perspective for improving robustness against shape and viewpoint variations.
2. The paper is well-written, logically structured, and easy to follow.
3. Demonstrates improvements on several benchmarks and provides qualitative examples showing better model adaptability during long-term tracking.

### Weaknesses
1. Since online model editing is computationally non-trivial, reporting runtime comparisons with baselines would help clarify practical deployment feasibility.

### Questions
Please see the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GOT-EDIT, an innovative approach that integrates 3D geometric features into general object tracking tasks through online model editing. It primarily leverages the VGGT module, trained on large-scale 3D datasets, to extract 3D geometric information from a set of 2D images. By employing null-space constraints, it seamlessly combines 3D geometric information with 2D semantic information, enabling the model to adaptively fuse 3D geometric information while preserving the semantic knowledge learned by the tracker. Extensive experiments demonstrate GOT-EDIT's robustness and accuracy.

### Strengths
The authors propose an innovative method for integrating 2D semantic information with 3D geometric information, and conducts experimental validation using datasets from diverse scenarios.

### Weaknesses
It is recommended to incorporate additional visualization results across diverse scenarios to substantiate the method's generality. While online model editing has achieved performance gains, it may introduce increased algorithmic complexity and computational overhead, potentially impacting real-time performance—particularly with high-resolution video inputs. Furthermore, the section detailing the online model editing methodology lacks detailed formulaic steps for operations based on AlphaEdit, which may cause confusion.

### Questions
1. One innovation of this paper is that seamlessly integrating 3D geometric information with 2D semantic information can enhance model performance. However, the mechanism by which 3D geometric information influences 2D semantic information remains undeclared. It is recommended to add relevant content.
2. Why was the VGGT module chosen for extracting 3D geometric features? It is recommend to compare it with other commonly used modules for 3D geometric feature extraction and detailing its specific advantages.
3. The paper employs null-space constraints but does not discuss whether these constraints may cause overfitting and performance degradation while preserving semantic integrity. The authors are advised to address this issue.
4. The fonts in Equations 6 and 7 do not match the subsequent text, potentially causing confusion. A thorough review and correction of relevant content is recommended.
5. Figure 1 lacks explanations for the annotations, including the red solid and dashed boxes as well as the green box.

### Soundness
3

### Presentation
2

### Contribution
3
