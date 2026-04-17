# Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
In this work, we present Color3D, a highly adaptable framework for colorizing both static and dynamic 3D scenes from monochromatic inputs, delivering visually diverse and chromatically vibrant reconstructions with flexible user-guided control. In contrast to existing methods that focus solely on static scenarios and enforce multi-view consistency by averaging color variations which inevitably sacrifice both chromatic richness and controllability, our approach is able to preserve color diversity and steerability while ensuring cross-view and cross-time consistency. In particular, the core insight of our method is to colorize only a single key view and then fine-tune a personalized colorizer to propagate its color to novel views and time steps. Through personalization, the colorizer learns a scene-specific deterministic color mapping underlying the reference view, enabling it to consistently project corresponding colors to the content in novel views and video frames via its inherent inductive bias. Once trained, the personalized colorizer can be applied to infer consistent chrominance for all other images, enabling direct reconstruction of colorful 3D scenes with a dedicated Lab color space Gaussian splatting representation. The proposed framework ingeniously recasts complicated 3D colorization as a more tractable single image paradigm, allowing seamless integration of arbitrary image colorization models with enhanced flexibility and controllability. Extensive experiments across diverse static and dynamic 3D colorization benchmarks substantiate that our method can deliver more consistent and chromatically rich renderings with precise user control. Project Page: https://yecongwan.github.io/Color3D/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Color3D proposes a method for colorizing monochromatic images in 3D representations for static and dynamic scenes. Naive implementations, like colorizing 2D multi-view images and reconstructing the 3D scene lead to severe cross-view inconsistencies. Recent methods that distil the color information to a 3D representation sacrifice controllability and often have desaturated colors in the final output.

Key contributions of the proposed methodology are as follows:

-   The key idea is to colorize (automatic/reference-based/prompt-based)  a single "key" view which is the most informative one and then fine-tune a scene-specific colorization network on that view.
-   This scene-specific colorizer learns a deterministic color mapping for the scene and is then applied to all other views/frames, enforcing cross-view and cross-time color  consistency .
-   Finally, the colorized views (with known luminance and predicted chrominance) are fused into a Gaussian Splatting representation in Lab color space.

Experiments on standard benchmarks (LLFF, Mip-NeRF 360, DyNeRF) and "in-the-wild" legacy videos show that Color3D produces vivid and consistent colorizations.

### Strengths
- **[S1] Technical Novelty:** The per-scene colorization for a single view is a novel idea which achieves consistent colorization across views. The robust technical pipeline achieves this consistency through key design choices. Specifically, utilizing a pre-trained 2D colorization encoder (DDColor) preserves the generalization capability of the model. The use of the Lab color space leads to more stable results. Finally, warm-up training of the 3D representation on the Luma ($\text{L}$) channel first ensures the model establishes a strong geometric structure before introducing complex color information.

- **[S2] Practical Applications:** The authors show results on "in-the-wild" multi-view images and historical video (Fig.6), producing vivid and plausible colors while maintaining consistency.

- **[S3] Controllability**: The proposed method allows users to control the colorization using text descriptions or reference images. Earlier methods did not offer this type of user control.

- **[S4]** Thorough Experimentation for Key-View selection: The authors performed detailed experiments for "Key-View Selection" module. Results in Fig. 7 demonstrate that this module is critical for better colorization.

### Weaknesses
- **[W1] Limited Comparison to Recent Methods:** The authors mention "ChromaDistill" in Related Work, but do not perform a quantitative comparison. For complete experimentation, it is also necessary to quantitatively compare the results against a video colorization baseline.

- **[W2] Cross-View Consistency**: ChromaDistill utilized a  long-term and short-term view consistency method for measuring the geometric consistency. However, the authors do not show this metric. The manuscript will benefit from this metric.

- **[W3]** As the method utilizes training for each scene, it has an extra training overhead.


**Typos:**

-   L74: It should be "aim" instead of "aims".
-   L857: It should be "suffers" instead of "suffer".
-   L347: It should be "entire" instead of "entir".

### Questions
-   **[Q1]** Have the authors tried fine-tuning on _more_ than one view? Using two or three colorized views might improve coverage for large scenes. Is there a reason the approach is limited to one view?

-   **[Q2]** How does the proposed method handle motion for dynamic scenes? In case the object that was not in the key view appears in the scene, how is its colour determined? Does the generative augmentation simulate such cases? This is not clear in the current manuscript.

### Soundness
4

### Presentation
4

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
1. This paper focuses on the task of controllable and consistent 3D scene editing. It aims to address the limitations of previous methods that often lack precise controllability and multi-view color consistency.

2. The authors propose Color3D, a two-stage framework. In the first stage, a text-to-image diffusion model is used to modify the reference view’s color according to user input. In the second stage, Score Distillation Sampling (SDS) is applied to enforce 3D consistency. Unlike prior work, the method performs Gaussian Splatting optimization in the LAB color space, which helps maintain color consistency across different viewpoints.

3. Experiments on Color-Edit-3D dataset show that Color3D achieves state-of-the-art results. The ablation study highlights the importance of the Color Consistency Regularization (CCR) module, which provides a clear performance gain when included.

### Strengths
1. The method is technically sound and well-motivated.

2. The experiments are thorough and provide strong empirical validation.

### Weaknesses
1. The computational cost is unclear — how long does the optimization take during the second stage?

2. The main novelty seems to lie in optimizing in the LAB color space instead of RGB. While this is a reasonable choice for editing tasks, the contribution may be somewhat limited in scope.

3. It would be interesting to see how the approach performs under challenging lighting conditions (e.g., scenes with lamps or strong reflections).

### Questions
Please address the issues listed in the weaknesses section.

### Soundness
3

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
This paper introduces Color3D, a framework for colorizing both static and dynamic 3D scenes (represented by 3D/4D Gaussian Splatting) from grayscale inputs. Its core idea is to avoid the multi-view inconsistency of applying 2D colorizers independently by instead personalizing a single colorization model per scene. This is done by selecting and colorizing one key view, then fine-tuning a pre-trained colorizer (using adapters) on augmented versions of this single view to learn a scene-specific, deterministic color mapping. This personalized colorizer is then used to colorize all other views/frames consistently. A dedicated Lab color space Gaussian representation with a warm-up strategy is used to improve reconstruction fidelity.

### Strengths
1. The paper is well-written and easy to follow. The proposed method can be adapted to various colorization models, making it highly practical and valuable for real-world applications, especially in AR/VR scenarios.

2. The experimental results on LLFF and Mip-NeRF 360 and D-NeRF colorization setting have shown the Color3D's good performance, not only on statistic 3d scenes but also on dynamic 4d scenes. Also, the method demonstrably outperforms existing alternatives across multiple  metrics (FID, CLIP Score, Matching Error), showing superior consistency, color vividness, and alignment with user intent.

### Weaknesses
1. It seem like the requirement to fine-tune a personalized colorizer for every new scene adds a non-trivial computational cost (∼8 minutes per scene) compared to a generic, one-time-trained model, limiting its scalability for large-scale applications.
2. The entire color propagation relies on a single key view. If this view is unrepresentative or lacks critical scene elements, the colorizer's generalization may be hampered, potentially leading to incomplete or less vibrant colorization in occluded regions.

### Questions
The paper shows that the personalized colorizer is robust to viewpoint changes from the key view. How does it handle novel views containing objects or textures that are semantically similar but visually different (e.g., another type of chair)? Would it incorrectly transfer learned colors or struggle to color them plausibly?

### Soundness
2

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
The paper presents Color3D, a unified framework for colorizing both static and dynamic 3D scenes reconstructed from monochrome inputs. Rather than colorizing multiple views independently, which causes cross-view inconsistency, the method colorizes one key view using any off-the-shelf 2D colorization model, then fine-tunes a per-scene personalized colorizer to propagate the learned color mapping to all other views or time steps.E xperiments on LLFF, Mip-NeRF 360, and DyNeRF datasets demonstrate consistent and controllable colorization across viewpoints and time, with improvements in FID, CLIP score, and Matching Error compared to baselines.

### Strengths
**High controllability and flexibility.**
The system supports multiple control modalities: reference-based colorization, language-conditioned colorization, and automatic default color prediction.

**Computational practicality.**
Despite using fine-tuning, the reported per-scene personalization time (~8 minutes) is relatively efficient compared to retraining full colorization networks. The framework does not require full 3D model retraining, and adapters make it lightweight enough for scene-level deployment.

### Weaknesses
**Limited generalization beyond scene-specific fine-tuning.**
The reliance on per-scene personalized colorizer tuning implies that a new model must be trained for each scene. This restricts scalability for large datasets or interactive applications. The approach is elegant but computationally heavy when many scenes must be processed.

**Inductive bias assumption not rigorously examined.**
The claim that a single-view-trained colorizer generalizes to novel viewpoints via inductive bias remains empirical. The paper could benefit from deeper theoretical or diagnostic analysis to substantiate why the learned mapping remains consistent under large view changes.

**Limited evaluation.**
The datasets being evaluated on are LLFF (static), Mip-NeRF-360 (static), and DyNeRF(dynamic). Although promising results are shown, the scale of evaluation is still limited. Also, for dynamic scenes, there are more dimensions for evaluation like temporal color-consistency, which is missing in the current experiments.

### Questions
1. How stable is the personalized colorizer’s output under large camera baselines (e.g., >60° change)? If still good, what are the fundamental factors that may contribute to this effect?

2. How sensitive is performance to errors in key-view selection?

3. How could this appoarch be generalized to feed-forward style 3D generation models? This means per-scene optimizations are removed.

### Soundness
3

### Presentation
2

### Contribution
2
