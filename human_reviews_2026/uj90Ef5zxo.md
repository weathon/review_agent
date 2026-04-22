# Fixation-Driven Time-Aware 3D Human Motion Forecasting in Indoor Scenes

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Forecasting human motion in indoor scenes is crucial for collaborative robotics and embodied AI. While prior approaches have incorporated gaze implicitly or used it only to rank segmented objects, we argue that gaze, particularly fixations, offers a more intentional and spatially precise signal for predicting human intent. In this work, we introduce a fixation-driven, time-aware framework for 3D human motion forecasting that explicitly supervises a gaze network to distinguish fixations from saccades, and uses fixation-weighted vectors to not only rank candidate objects but to also localize precise interaction points, improving robustness to segmentation errors. Our contribution further includes a duration prediction module that generates variable-length motion sequences, adapting to the spatial and temporal demands of the task. We evaluate our approach on the GIMO and GTA-IM datasets to show more accurate predictions particularly in challenging scenes with small or merged objects, and varying interaction durations through variable-length motion generation. Our code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work addresses the challenging and interesting task of 3D human motion forecasting in indoor scenes by leveraging gaze information. The core idea is to use gaze, particularly fixations, as an intentional signal to guide the prediction of future human poses.

### Strengths
The proposed framework is comprehensive, involving several key steps: scene segmentation, gaze classification, object ranking, end-pose prediction, and variable-length motion in-betweening.

### Weaknesses
1.The manuscript appears to be a draft, as evidenced by the unmodified conference title template.
2.The paper claims superior end-pose prediction accuracy. While the use of gaze for precise trajectory prediction is convincing, it is not guaranteed to predict accurate end pose (see the comments).

### Questions
The task addressed in this work is inherently challenging and quite interesting. The authors utilize datasets containing gaze and human motion information to predict gaze first, and then leverage gaze to forecast future human motions. This requires handling multiple components: object segmentation, gaze prediction, and target object prediction. They begin by predicting the final end pose, and then generate intermediate poses through interpolation.

However, the work has the following issues:

1. The title of the paper has not been properly revised.

2. The paper appears to present an engineering-oriented pipeline that integrates several components: PointNet-based segmentation, Transformer-based gaze prediction, target object ranking, CVAE-based end pose prediction, and diffusion-based interpolation for intermediate poses. While it is a solid engineering effort, the innovation aspect is not clearly articulated.

3. A core issue is that while gaze is undoubtedly helpful for predicting human motion trajectories, it theoretically should not contribute to predicting the final end pose. This is because gaze primarily provides directional guidance. Compared methods improve end pose prediction accuracy by incorporating global environmental context, while this paper does not seem to consider this. Further explanation is needed regarding why the end pose accuracy outperforms existing methods.

4. Equation 1 appears to be an elegant formulation that unifies various tasks. However, independent models are used for implementation. Is this type of description formally appropriate in mathematical terms?

5. The GIMO dataset is relevant to this work. However, the use of GTA-IM, with manually defined fixation thresholds such as velocities below 5 degrees or displacements less than 0.3m, makes the approach seem somewhat fragile.

6. The captions for Figure 1 and Figure 2 are insufficiently explanatory. In particular, the meaning of the blue bars in Figure 2 is unclear. It is recommended to make figure captions self-explanatory to improve readability without requiring reviewers to search through the main text for clarifications.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a fixation-driven, time-aware framework for 3D human motion forecasting in indoor scenes. The core innovation is the explicit supervision of a gaze network to distinguish fixations (intentional gaze) from saccades (transient gaze shifts). 
The paper's key innovation are:
- Fixation-Weighted Vectors: The model utilizes fixation-weighted vectors to provide a more intentional and spatially precise signal. This is used to both rank candidate objects for interaction and to localize precise, fine-grained interaction points or surfaces.
- Robustness to Errors: By leveraging fixation points to localize interactions, the method enhances robustness against segmentation errors, particularly concerning small or poorly-segmented objects.
- Duration Prediction Module: The framework incorporates a duration prediction module that estimates the number of frames required to complete an action. This enables the generation of variable-length motion sequences via diffusion-based in-betweening, tailoring the forecast to the spatial distance and temporal demands of the interaction task.
The method achieves state-of-the-art performance on the GIMO and GTA-IM datasets.

### Strengths
1. The benchmark is well defined, using multiple datasets and models, achieving strong and consistent SotA quantitative results.
2. The problem formulation is well-motivated by explicitly distinguishing fixations (intentional gaze) from saccades (transient gaze shifts), grounding the work in eye-tracking literature and providing a more intentional signal for motion prediction.
3. Fixations are used not only for attention classification (ranking objects) but also for geometric localization to identify precise interaction points, enhancing model interpretability. (Ah si?)
4. The introduction of a duration prediction module is a significant practical contribution. It solves the fixed-length limitation common in prior diffusion models.

### Weaknesses
1. Individual components use standard architectures (Transformer, cVAE, diffusion); the contribution is primarily in their combination and the fixation modeling approach.
2. Training components separately rather than end-to-end may be suboptimal and lead to error accumulation.
3. GTA-IM uses head orientation as a proxy for gaze, which is acknowledged as limiting. The method's performance gains are smaller on this dataset, raising questions about generalization.

**Minor** 

4. The title is wrong.
5. At L180 I think there should be no space between 0.3 and the unit measure m.
6. I think it should be emphasized in the text which blocks are optimized and which ones are used as a black box model.

### Questions
1. Could the authors clarify which parts of the pipeline are took from literature and which ones are novel?
2. Could the authors clarify on the choice behind the threshold of 0.3m?
3. Which GPUs are used and which are the time/bandwidth requirements?

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
3

### Summary
The paper addresses human motion prediction in 3D scenes conditioned on gaze and scene geometry. It introduces a fixation-driven framework that explicitly distinguishes fixations from saccades, localizes interaction points on object surfaces, and generates variable-length motion sequences via diffusion-based in-betweening. The design aims to improve interpretability and robustness compared to prior gaze-conditioned motion models such as BiFU, SIF3D, and GAP3DS. Results on GIMO and GTA-IM show performance gains on long-horizon metrics and qualitative realism.

### Strengths
1. The paper clearly formulates a novel fixation-driven task pipeline linking gaze behavior and motion generation, providing an interpretable reasoning process from gaze to final motion.
2. Experiments on real and synthetic datasets are extensive, showing consistent improvements and reasonable qualitative motion visualizations that support the proposed design.
3. The modular design makes each subtask—fixation analysis, object ranking, end-pose estimation, and motion in-betweening—explicitly measurable, improving interpretability and diagnostic transparency.

### Weaknesses
1. Figures are informative but visually crowded, making it difficult to follow the logical flow and perceive the claimed advantages. The authors could improve visual hierarchy by separating perception, reasoning, and generation stages in the main framework figure and adding clearer annotations or zoomed-in views in qualitative examples to highlight where the proposed method performs better, especially on small-object interactions.
2. The five-stage modular structure introduces considerable complexity and requires intermediate supervision, reducing the feasibility for joint or real-time training.
3. The paper would benefit from a clearer problem statement and a more pedagogical presentation of the fixation-driven mechanism. The current writing style reads like a method checklist rather than a reasoning flow. It would benefit from adding short motivation sentences at the beginning of each subsection to clarify why a specific design choice is made.

### Questions
1. It seems to rely heavily on accurate eye-tracking data to classify fixations and compute intersections. Real-world gaze data often may contain jitter, latency, or occlusion noise, which could severely degrade the fixation classifier. The performance gap and generalization robustness are unclear.
2. Ablations are present but do not fully validate key claims: the paper shows a “w/o Time” variant and architecture swaps for the fixation classifier, but lacks sensitivity analysis for the 0.3 m fixation threshold and does not isolate the benefit of gaze–object intersection or diffusion vs. simpler temporal generators. Adding such ablations would strengthen the empirical support beyond what is currently reported.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel approach for 3D human motion forecasting in indoor environments. The core thesis is that fixations are a high-precision, intentional signal that is more valuable than general gaze or saccades. The authors argue that explicitly modeling fixations provides a more intentional and spatially precise signal than existing approaches that use gaze implicitly. The main technical contributions to achieve this are as follows: First, they explicitly train a network to classify fixations, generating a fixation-weighted vector as a signal of interaction intent. Second, this fixation vector is used to localize the precise interaction point, improving robustness to segmentation errors. Finally, a duration prediction module is introduced to estimate the required frames, allowing the model to generate variable-length motion sequences. The approach is evaluated on the GIMO and GTA-IM datasets and claims state-of-the-art performance, especially in challenging scenes requiring precise interaction and temporal variation.

### Strengths
### Originality
The work is highly original in its approach to using gaze. The explicit classification of fixations versus saccades for this task is a novel idea. The resulting fixation-weighted vector is a purer signal of intent than existing approaches that use gaze implicitly. The use of the resulting fixation vector to directly localize a 3D interaction point can improve spatial precision and robustness to point cloud segmentation errors. The combination of an $A^*$ planner, an explicit CNN-based duration predictor, and a diffusion model to achieve variable-length motion in-betweening is a novel architecture for this domain.

### Quality: 
The technical execution is sound. The system component is well-motivated, each addressing a specific limitation in the previous work. The qualitative examples highlight the model's ability to correctly forecast interactions with small, often-missed targets (e.g., a cup or a banana) that prior methods fail on. This demonstrates high practical quality and robustness.

### Significance: 
The model's robustness to segmentation error is a significant practical advantage for real-world robotics. The move from fixed-length to variable-length motion generation is a critical step toward more realistic and useful human motion forecasting.

### Weaknesses
1.  Supervision for Fixation Labels: The definition of "fixation" for the GIMO dataset is "gaze points within 0.3 m of the target". This is a dataset-dependent definition that requires ground-truth (GT) object-of-interest labels to create the fixation labels for training the classifier. This is a very strong form of supervision which may directly imply the position of target object. For the GTA-IM dataset, the fixation labels are generated by gaze velocity threshold not object position. And the improvement over baseline on GTA-IM dataset is not significant as GIMO (Table 1).

Actionable Feedback: The authors should discuss this limitation. How would this system be trained on GIMO dataset without GT object-of-interest labels? A comparison of performance would significantly strengthen the paper's claims of practical applicability.

2.  Assumption of a Single, Stable Target: The model aggregates all detected fixations within the observation window into a single average vector.

Actionable Feedback: This assumes the user has only one, stable intention during the observation window. What happens if the user's intent is ambiguous or shifts (e.g., they look at a cup, then a phone)? Averaging these two fixations would produce a meaningless vector pointing between them. The paper should discuss this assumption. The model could be made more robust by, for example, clustering the fixations and using only the most recent/dominant cluster, rather than a simple average.

3.  Scalability of Action-Specific cVAEs: The method uses "a separate cVAE per action type family (e.g., reach, sit, lie), selected based on the instance label". This seems to introduce a new dependency on a predefined, limited set of actions. This approach needs to have a correct instance label. (e.g., knowing a segment is a 'chair' to select the 'sit' cVAE) This undermines the segmentation-robustness claims, as it still relies on a correct label from the potentially faulty segmentation. 

Actionable Feedback: The authors should clarify how this selection is made at test time and discuss the scalability of this approach. How would it handle novel objects or more diverse actions beyond "reach, sit, lie"? A discussion on moving toward a single, unified, action-agnostic end-pose generator would be welcome.

### Questions
1. On the Supervision Strength of Fixation Labels

My primary concern relates to the definition of "fixation" used for the GIMO dataset. You define it as "gaze points within 0.3 m of the target". This appears to be a very strong form of supervision, as it requires ground-truth (GT) object-of-interest locations to be known during the training of the gaze classifier. This seems to risk leaking information about the final target location to the gaze module. In contrast, for the GTA-IM dataset, you use a more standard, velocity-based threshold, and the performance gains in Table 1 are less significant than on GIMO.

Question: Could you please provide an ablation study showing the performance on the GIMO dataset if you use only a velocity-based (or other non-GT-location-based) method for labeling fixations? This is crucial for understanding how much of the performance gain comes from the using fixations versus the strong supervision used to define them.

2. On the Assumption of a Single, Stable Target

The model aggregates all detected fixations within the observation window into a single average vector, $g_*$. This method implicitly assumes the user has only one, stable intention during the observation period.

Question: How does the model behave in cases of user ambiguity or intent-switching (e.g., the user looks at a cup, then at a phone)? In such a scenario, would the averaged $g_*$ vector not point to an empty space between the two objects, leading to a failure? Could you discuss this limitation and whether you explored more robust aggregation methods, such as clustering or using only the most recent fixation cluster?

3. On the Dependency of Action-Specific cVAEs

You state that the method uses "a separate cVAE per action type family (e.g., reach, sit, lie), selected based on the instance label". This seems to create a dependency on a correct semantic label from the segmentation module, which feels at odds with the paper's core strength of being robust to segmentation errors.

Question: Could you clarify this selection mechanism? How does the model select the correct cVAE if the object is mislabeled or merged with the background, as in your "banana on the floor" example? Furthermore, how does this approach scale to novel objects or actions outside these predefined families?

### Soundness
3

### Presentation
2

### Contribution
2
