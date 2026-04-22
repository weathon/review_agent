# DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
With the rapid progress of video generation, demand for customized video editing is surging, where subject swapping constitutes a key component yet remains under-explored. Prevailing swapping approaches either specialize in narrow domains—such as human-body animation or hand-object interaction—or rely on some indirect editing paradigm or ambiguous text prompts that compromise final fidelity. In this paper, we propose DreamSwapV, a mask-guided, subject-agnostic, end-to-end framework that swaps any subject in any video for customization with a user-specified mask and reference image. To inject fine-grained guidance, we introduce multiple conditions and a dedicated condition fusion module that integrates them efficiently. In addition, an adaptive mask strategy is designed to accommodate subjects of varying scales and attributes, further improving interactions between the swapped subject and its surrounding context. Through our elaborate two-phase dataset construction and training scheme, our DreamSwapV outperforms existing methods, as validated by comprehensive experiments on VBench indicators and our first introduced DreamSwapV-Benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an end-to-end framework based on mask guidance, which enables the replacement of any subject in a video using a specified mask and a reference object. Its key contributions are as follows:

- The proposal of an adaptive masking strategy to address practical issues.
- A dedicated conditional fusion module that integrates rich conditional information.
- The introduction of a new benchmark.

### Strengths
1. This paper has a clear logical flow, elaborating on its key contributions with a focus on two main aspects.
2. The adaptive masking strategy takes full account of various practical issues and achieves favorable results.
3. This paper demonstrates substantial research effort, solid argumentation, and has obtained excellent model performance.

### Weaknesses
1. There is still room for improvement and supplementation in the quantitative results of the experimental section of this paper.
2. The work of this paper mainly focuses on the targeted processing of data, with limited improvements to the model itself.

### Questions
1. In the Extra Shape Augmentation section, this paper mentions intentionally making the mask slightly larger than the target subject, which the authors consider a better approach. However, this approach is not supported by quantitative evidence in the subsequent experiments, and additional relevant experiments are expected to be supplemented.
2. This paper employs the newly proposed Benchmark for model evaluation in the experimental section. The evaluation results seem to be inconsistent with the quantitative results of Hunyuan Custom. Traditional quantitative comparison methods remain indispensable, and supplementary experiments in this regard are anticipated.

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
This paper presents DreamSwapV, which is a mask-guided and subject-agnostic framework for video subject swapping. It allows users to replace any subject in any video using a mask and a reference image. The method treats swapping as a video inpainting task, enabling seamless blending between the new subject and the original scene. A condition fusion module integrates mask, pose, and 3D-hand inputs, while an adaptive mask strategy handles subjects of different sizes. The model is trained in two phases for better generalization. Experiments on a new DreamSwapV-Benchmark show that it achieves higher visual quality and consistency than existing methods.

### Strengths
1. The proposal of the condition fusion module and adaptive mask strategy enables fine-grained control, resulting better subject-context interaction together with high-quality visual improvements.
2. The paper introduces a new benchmark DreamSwapV-Benchmark in customized image editing domain and the proposed method shows improvements over previous baselines with both quantitative metrics and user studies.
3. The model can be extended beyond subject swapping to related tasks like video inpainting, addition, and try-on, which may show good generalization potential.

### Weaknesses
1. The method relies on many detailed design choices and a two-phase training process across multiple datasets. While these improve performance, they make the system complicated and harder to reproduce.
2. Some baselines, like AnyV2V, are training-free or designed for broader editing rather than subject swapping, which makes the comparison less fair.

### Questions
1. Can you provide some failure case analysis for your method?
2. The two-phase training scheme and dataset mixture seem central to performance. Could the authors clarify how much improvement the second phase brings quantitatively, and whether similar performance could be achieved with a single large-scale fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents DreamSwapV, a mask-guided, subject-agnostic framework for end-to-end video subject swapping. The method reformulates the swapping process as a video inpainting task, enabling more seamless integration between the inserted subject and the background. The proposed system combines a condition fusion module that integrates multiple signals and an adaptive mask strategy that accommodates subjects of varying scales.

### Strengths
* The combination of a multi-condition fusion module and adaptive mask strategy is intuitive yet effective, addressing common artifacts, e.g., boundary leakage and poor subject-context blending.

* The proposed DreamSwapV-Benchmark is a valuable addition to the field, with well-defined metrics and an attempt at quantitative evaluation for a task lacking standard benchmarks.

* The paper is clearly written and easy to follow, with well-organized figures and methodological explanations.

### Weaknesses
* While the new benchmark is appreciated, the evaluation relies heavily on VBench-style metrics that may not capture identity preservation or semantic consistency robustly. A few qualitative examples are shown, but it remains unclear how the model generalizes to truly out-of-domain subjects or complex dynamic interactions.
* Some ablations, e.g., condition fusion variants, mask augmentation parameters, are only qualitatively discussed. Quantitative ablations would make the argument stronger.
* The method is heavily trained and evaluated on HumanVID-derived data, raising concerns about its generalization to more diverse subjects.

### Questions
My key questions are related to the weaknesses mentioned above; besides those, I still have a few minor questions.
* How does the model perform on truly cross-domain swapping (e.g., animal → human or object → character), and what failure modes are observed?
* How does the method behave under mask noise or misalignment, such as slightly inaccurate user-provided masks — does performance degrade sharply or remain stable?

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
5

### Summary
This paper proposed DreamSwapV, an mask-guided framework for video subject swapping that allows users to swap any subject in any video using a mask and reference image. It introduced a multi-condition fusion module and an adaptive mask strategy to handle subjects of varying scales and attributes. DreamSwapV leverages an elaborate two-phase training scheme on a carefully curated dataset and introduced the DreamSwapV-Benchmark, and achieving best performance on the benchmark.

### Strengths
S1) The method is well ablated and justified the design choices that yields finer details and more robust subject-context integration.

S2) Adaptive mask makes a good design by dynamically adjusting the grid size based on the subject's scale and augmenting mask boundaries with geometric shapes.

S3) Its great to see the discussion on handling long videos, as most related works can only edit videos on a certain length / training length.

### Weaknesses
W1) In Table 1, the automatic metrics "Video Quality & Video Consistency" show AnyV2V with high scores on most quantitative indicators (subject consistency, background consistency, motion smoothness), nearly on par with other leading methods. However, the user study metric (human-rated reference detail, subject interaction, and visual fidelity) shows AnyV2V achieving nearly zero. This shows the proposed metrics are insufficiently sensitive to qualitative breakdowns. It seems only the reference appearance and background preservation metrics are making sense here. The authors might consider replacing VBench metrics with alternatives.

W2) Since the method is guided on the masks, the quality of the mask directly affects the final swapping results. Although detection models like TrackingSAM can handle the mask, but any errors could happen in fast motions and propagate to the downstream swapping result. Any remedy to make sure the mask is correctly handled?

### Questions
Q1) Since phase 1 is mainly trained on human (HumanVID), and phase 2 is trained on small subjects (Subject200K etc), this raise an interesting question. How much gain did the model get after each stage? Comparing the model’s performance after each training phase would give more insights on how effective this method is.

### Soundness
2

### Presentation
2

### Contribution
2
