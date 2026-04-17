# DreamActor-M2: Unleashing Pre-trained Video Models for Universal Character Image Animation via In-Context Fine-tuning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Character image animation aims to generate high-fidelity videos from a reference image and a driving video, with broad applications in digital humans. Despite recent advances, current methods suffer from two key limitations: reliance on auxiliary pose encoders introduces  modality gaps that weaken alignment pre-trained generative priors, and dependence on explicit pose signals severely limits generalization beyond human-centric scenarios. We propose DreamActor-M2, a universal framework that redefines motion conditioning through an in-context LoRA fine-tuning paradigm.  By directly concatenating motion signals and reference images into a unified input, our approach preserves the backbone’s native modality and fully exploits pre-trained capabilities without architectural modifications, enabling plug-and-play motion control consistent with the principles of in-context learning.
Furthermore, we extend this formulation beyond pose-driven control to an end-to-end framework that conditions directly on raw video frames, trained by a synthesis-driven data generation pipeline.
Extensive experiments demonstrate that DreamActor-M2 achieves state-of-the-art performance with superior fidelity, controllability, and cross-domain generalization, marking a significant step toward more flexible and scalable motion-driven video generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the problem of character image animation and proposes DreamActor-M2. DreamActor-M2 concatenates the motion signal and the reference image as a unified input. This in-context learning paradigm addresses the limitation of reliance of pose encoder or explicit pose signal in previous works. The paper also designs a data generation pipeline to generate training videos. Experiments show that DreamActor-M2 achieves state-of-the-art performance.

### Strengths
1. The data generation pipeline generating pairs of videos with the same motion is reasonable and useful for this task. 
2. A benchmark containing human motion and non-human motion is collected.
3. Extensive evaluation is conducted.
4. DreamActor-M2 obtains state-of-the-art performance.

### Weaknesses
1. The evaluation is only conducted on the constructed benchmark. It is unclear whether the good performance can bee translated to other benchmarks.
2. No systematic human evaluation is conducted. The evaluation metrics (most rely on features extracted from models) used in the paper may now fully reveal the generation quality. So human evaluation is necessary.
3. No training or inference efficiency comparison between pose-based methods.
4. The problem can also be formulated as video editing. Some existing methods like [1] adapt temporal attention layers within video diffusion models to generate new videos. These methods are not discussed (advantanges and disadvantages) or compared in the paper.

[1] VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models. CVPR2024.

### Questions
1. Is there any human evaluation results?
2. What is the efficiency of the proposed method?
3. Waht is the advantanges and disadvantages of the proposed method compared to video editing methods adapting motion signal within diffusion models?

### Soundness
2

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
5

### Summary
This paper proposes a universal framework for character image animation. Performing subject motion transfer using in-context learning, without relying on separate pose encoders and explicit skeleton control, and could be applied to heterogeneous subjects, with applications in digital humans, entertainment, and virtual avatars. The experimental results demonstrated the superiority of this method.

### Strengths
* This paper proposes a multi-stage pipeline to complete the motion transfer through a context-learning manner. This framework does not rely on a separate pose encoder and can be applied for motion transfer on heterogeneous subjects (such as humans and dogs).
* Due to the removal of the explicit pose control, it does not rely on complete paired training data, and the proposed manner is scalable.

### Weaknesses
* In Line 84, the author claims the flaws of relying on pose estimation, but in the first two steps of the proposed method, pose estimation is still relied upon. This conflicts with the authors’ insight. As the authors say, how should the errors introduced by pose estimation in the first two steps be addressed, because this is the foundation for the in-context learning that follows.
* Regarding the MLLM and LLM employed, what are their specific functions, and is there corresponding experimental validation? In my view, these encoders may still exhibit instability, similar to the pose encoder; how can such instability be prevented or mitigated?
* Given that the MMDiT architecture computes self-attention jointly across temporal and spatial dimensions, how can we ensure that LoRA robustly disentangles motion from appearance and avoids confounding influences from appearance?
* The experimental section lacks substantive analysis and primarily relies on final metrics to demonstrate the effects of the proposed designs. However, these metrics are not fully reliable. I would prefer to see deeper examinations of the model’s internal behavior to substantiate that the design is genuinely effective; otherwise, I am inclined to attribute the observed gains to the foundation model and the data, suggesting limited technical innovation.

### Questions
See the Weaknesses, especially the first point, the third point, and the last point.

### Soundness
3

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
4

### Summary
This paper presents DreamActor-M2, a universal framework for character image animation, which generates high-fidelity videos from a reference image and a driving video. The key innovation is an in-context LoRA fine-tuning strategy: motion cues (from pose or raw video) and reference images are directly concatenated as input, preserving the backbone's pre-trained modality and forgoing explicit pose encoders. DreamActor-M2 supports both pose-driven and end-to-end (raw video-driven) control, with a two-stage pseudo-data synthesis pipeline enabling annotation-free dataset construction. The model aims to improve fidelity, motion transfer consistency, and ability to generalize to human and non-human characters. Extensive quantitative and qualitative experiments, including ablations, are used to assess the framework on a curated benchmark.

### Strengths
- The approach sidesteps the modality gap typical of auxiliary pose encoders, leveraging the backbone's native, pre-trained input space and in turn likely reducing the need for custom architecture augmentation.
- The method is extended to work directly with raw videos as motion drivers, not just pose signals. The two-stage pseudo-pair synthesis (data generation and filtering) is thoughtfully conceived, allowing scale-up without additional human annotation.
- The paper constructs a reasonably comprehensive evaluation benchmark with wide coverage (humans, non-humans, animated, animal, and multimodal), and reports both subjective and qualitative outcomes.

### Weaknesses
- The paper briefly describes the composite input $\mathbf{C}$, but leaves out critical details on how concatenated regions are encoded by the backbone and, more importantly, how the mask conditioning function $\mathbf{M}=\mathbf{M}_r\oplus \mathbf{M}_m$ is formally computed and applied. Figure 2 shows the workflow, but does not concretely illustrate the transformation of these masks or their effect during denoising. Ambiguity here risks spurious alignment.
- The approach’s theoretical underpinnings, especially regarding why in-context spatial concatenation aligns better with backbone pre-training, remain mostly intuitive. There are no explicit derivations or formal analysis on information preservation, context mixing, or potential pitfalls of spatial concatenation versus attention-based fusion, aside from empirical ablation. For instance, the paper should consider if joint input could result in detrimental "information bleeding" between reference and motion halves in $\mathbf{C}$, or when this method might fail relative to alternatives.

- Several newly surfaced, highly relevant prior works are not discussed[1,2]. Notably, the absence of engagement with DreamVideo, which directly addresses disentangled subject–motion video generation, as well as MotionFollower, which tackles score-guided motion diffusion, diminishes confidence that DreamActor-M2’s core advances are sufficiently differentiated or contextualized.

- Although the benchmark is admirably diverse, reliance on synthetic pairs and filtering may not capture true in-the-wild cross-identity challenges, or the full diversity of non-human articulations. Furthermore, the method’s generalization is mostly claimed on the provided benchmark, with little evidence or discussion for genuinely unseen domains or real-world robustness. No real statistics (distributional differences, label coverage, complexity) of the benchmark are provided.

- The “Limitations” section briefly mentions difficulty with multi-character tracking and assignment. However, it skirts over quality degradation observed in challenging ablation cases, fails to analyze scenarios where semantic fusion yields off-topic or ambiguous guidance, and does not address edge-cases where spatial in-context concatenation results in brittle or inconsistent outputs. Nor does it address when the pseudo-pair filtering pipeline may introduce bias or failure in underrepresented motion categories.

- The evaluation is almost entirely based on Video-Bench, with no reporting (even for reference) of FID/FVD or other standard, domain-relevant automated metrics. While the argument that these do not always correlate with human judgment is plausible, their complete absence makes quantitative comparison to the wider literature more tenuous—especially for subsequent researchers benchmarking against DreamActor-M2.

- The pipeline posits high-level LLM-powered compositional semantic guidance as a central contributor, yet the actual process by which motion and appearance text are fused, what type of prompts are generated, and how those captions are used during training vs. inference remain vague. Ablation suggests some boost, but no concrete examples of generated captions or analysis of semantic failures are given.

- Generating 600,000 pseudo-paired training units by two exhaustive model runs, followed by MLLM and Video-Bench-based filtering, is computationally burdensome. No cost or runtime analysis is given. For practical adoption, some assessment of whether similar performance could be achieved with weaker or smaller datasets would be invaluable.

- Several hyperparameters are missing (e.g., no details on skeleton augmentation rates, filtering thresholds except for average Video-Bench score >4, no clarity on precise cropping, normalization, or LLM/MLLM parameterization). The authors promise to release code, but critical low-level details are not currently presented.

[1] DreamVideo: Composing Your Dream Videos with Customized Subject and Motion

[2] MotionFollower: Editing Video Motion via Score-Guided Diffusion

### Questions
- Could the authors provide concrete, visual examples or schematic overlays showing how composite input $\mathbf{C}$ and mask $\mathbf{M}$ are constructed during training/inference? Does spatial concatenation ever result in artifacts at the boundary or mix reference and motion signals in a confusing manner? A demonstration on problematic or edge-case inputs would be valuable.
- Can the authors share actual motion/appearance text prompts generated by the MLLM/LLM, along with typical outputs in ambiguous or failure cases? Are there scenarios where the LLM merges semantically irrelevant information, causing deterioration in output quality?
- What is the statistical breakdown of subject types and motion categories in the evaluation benchmark? Are there classes/motions underrepresented or particularly challenging? Was any attempt made to test outside this distribution?
- What is the average computational overhead (in core-hours or $) of the pseudo-pair synthesis and filtering pipeline? Could similar results be achieved with a smaller synthetic “seed” or by semi-supervising on a smaller dataset?
- While human alignment is desirable, could the authors augment future papers with more conventional FID/FVD results for cross-paper comparability, even if those metrics are imperfect?

### Soundness
2

### Presentation
2

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
DreamActor-M2 is a universal framework for character image animation (generating high-fidelity videos from reference images and driving videos). The core goal is to address two major limitations of existing methods: the domain gap caused by reliance on auxiliary pose encoders and the lack of generalization due to dependence on explicit pose signals. The paper proposes a two-stage training paradigm, transitioning from Pose-based to End-to-End models. New evaluation metrics are introduced to align with human perception.

### Strengths
+ Eliminating the domain gap between different features by not using auxiliary pose encoders.
+ Richness of test samples: the generated examples in the paper show a wide variety of results. The model supports cross-modal animation, multi-person motion transfer, and non-human-driven videos.
+ The use of new evaluation metrics to align with human perception.

### Weaknesses
- The effectiveness on longer video sequences is not assessed. The paper does not evaluate the temporal consistency and fidelity of animations for longer sequences (over 5 seconds), making it unclear whether the model is suitable for use in longer animation scenes, such as movie clips.
- When comparing with other methods, there is no clarification of the number of parameters and backbones used by the comparison models. Additionally, the training scale is not clearly stated.
- The rationale behind the LoRA rank setting is not explored.

### Questions
- When using MLLM to evaluate the synthesized pairs, which multimodal large model was used? Since large models can sometimes generate hallucinations, how can we ensure the reliability and semantic coherence of the generated pairs? What is the accuracy of the filtering process?
- Does the model’s temporal consistency and visual fidelity degrade when generating longer video sequences?
- Is there a specific justification for setting the LoRA rank to 256?

### Soundness
3

### Presentation
3

### Contribution
2
