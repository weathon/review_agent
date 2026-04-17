# SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Existing research of 3D LLMs still struggles to achieve efficient and explainable reasoning, primarily due to the under-exploration of the mechanism of human-like scene-object grounded reasoning. This paper bridges the gap by presenting a novel framework. We first introduce a Chain-of-Thought reasoning framework in 3D scenes (SceneCOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To enable such a framework, we build the first large-scale 3D scene Chain-of-Thought reasoning dataset, SceneCOT, including more than 190k high-quality data instances. Extensive experiments across various complex 3D scene reasoning benchmarks demonstrate that our new framework achieves state-of-the-art with clear interpretability. To our knowledge, this is the first attempt to successfully implement the COT technique for achieving human-like step-by-step reasoning for 3D scene understanding, where we show great potential in extending it to a wider range of 3D scene understanding scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces grounded Chain-of-Thought reasoning into 3D large language models, decoupling complex reasoning tasks into simpler and more manageable problems, and constructing corresponding visual cues through multimodal expert modules. To achieve this, the authors focus on dataset curation. By training on the proposed dataset, the model attains state-of-the-art performance on several 3D understanding benchmarks.

### Strengths
1. The first to introduce CoT reasoning into 3D understanding.
2. Proposes a large-scale dataset to support the study.
3. The model achieves leading performance across multiple benchmarks.

### Weaknesses
1. **Limited evaluation:** The authors primarily evaluate on MSQA and Beacon3D, while several widely used benchmarks such as ScanQA and SQA-3D are not considered.
2. **Limited data sources:** The annotations mainly originate from Nr3D and MSQA. Incorporating larger and more diverse datasets and scenes, such as MMScan[3], could enhance generalization.
3. **Object-centric input:** The method mainly relies on object-centric input, which requires an additional segmentation model during inference and thus limits broader applicability. The ablation results show that performance is highly dependent on segmentation labels. Since object-centric input already provides a strong spatial prior for grounded reasoning, it remains unclear whether the proposed method can generalize to video-based 3D LLMs such as Video-3D LLM[1] or LLaVA-3D[2].

[1] https://arxiv.org/abs/2412.00493 (CVPR 25)
[2] https://arxiv.org/abs/2409.18125 (ICCV 25)
[3] https://arxiv.org/abs/2406.09401 (NIPS 24)

### Questions
1. Can grounded large-scale datasets like **3D-GRAND** be integrated into your pipeline?
2. What is the purpose of introducing probability during reasoning? Are there any ablation studies to justify its effect?
3. The **LEO** model adopts a similar object-centric input as the proposed method. Why does the performance on MSQA decrease after training on your dataset? The effectiveness of the proposed reasoning mechanism remains questionable.

### Soundness
3

### Presentation
2

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
This paper presents SceneCoT, a 3D scene understanding framework that employs step-by-step chain-of-thought (CoT) reasoning to enhance spatial reasoning performance on benchmarks such as MSQA and Beacon3D. To facilitate 3D CoT reasoning, the authors construct a large-scale dataset, SceneCoT-185K. Experimental results further demonstrate that SceneCoT effectively improves grounding–QA coherence.

### Strengths
1. SceneCoT demonstrates strong performance in spatial reasoning, particularly on counting and grounding questions. Moreover, its step-wise grounded reasoning provides a transparent and interpretable rationale.

2. The construction of the CoT steps is reasonable and aligns well with how humans approach spatial question answering.

3. The paper is well-written, clearly organized, and easy to follow.

### Weaknesses
1. SceneCOT is designed around specific reasoning tasks—namely Situated Reasoning and Object-Centric Reasoning—and limited question types such as counting, attribute, and spatial relationship queries. This task-specific design may restrict the model’s ability to generalize to unseen question types encountered in more complex 3D world.
2. In Table 1, the overall performance of LEO and MSR3D on MSQA drops noticeably after fine-tuning on the SceneCoT-185K dataset. Could the authors provide analysis on the possible reasons behind this degradation?
3. Since SceneCOT is designed based on LLaVA-1.5, it would be helpful to show the performance comparison of these two models.
4. It would be helpful to evaluate SceneCoT on out-of-domain datasets (e.g., SQA3D [1], Hypo3D [2], VSIBench [3]) to verify whether CoT fine-tuning improves spatial reasoning beyond the training domain.

[1] Ma, Xiaojian, et al. "Sqa3d: Situated question answering in 3d scenes."

[2] Mao, Ye, et al. "Hypo3D: Exploring Hypothetical Reasoning in 3D."

[3] Yang, Jihan, et al. "Thinking in space: How multimodal large language models see, remember, and recall spaces."

### Questions
I have concluded all my questions in the weakness sections.

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
This paper addresses the problem of poor grounding in 3D vision-language models, where models often generate plausible-sounding answers that are not factually connected to the 3D scene. The authors propose SCENECOT, a novel framework that introduces step-by-step, Chain-of-Thought (CoT) reasoning to 3D question answering. The method explicitly decomposes a complex 3D reasoning task into four manageable stages: task recognition, task-relevant region localization, entity/attribute grounding using expert modules, and final grounded reasoning. To train this framework, the authors also developed SCENECOT-185K, the first large-scale dataset containing 185,000 grounded CoT reasoning traces for 3D scenes. Experiments demonstrate that SCENECOT achieves competitive performance on the general MSQA benchmark and, most notably, significantly outperforms all baselines on the Beacon3D benchmark, which is specifically designed to measure grounding-QA coherence.

### Strengths
1. This work directly targets the critical and well-documented problem of poor grounding-QA coherence in 3D-VL models. Instead of just aiming for better QA accuracy, it focuses on ensuring the answers are correctly derived from the scene's visual context.
2. The framework's multi-stage design (task recognition, region localization, grounding, reasoning) is intuitive and inherently interpretable. This transparency makes it easier to diagnose failure cases, as shown in the qualitative examples.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The SCENECOT framework is a complex, multi-stage pipeline rather than a simple end-to-end model. Its performance is heavily reliant on a cascade of specialized, pre-trained modules (e.g., Mask3D for object proposals, PQ3D for grounding). This introduces multiple potential points of failure, and the overall performance is strongly coupled to the quality of these "expert" modules.
2. The dataset, while large, is constructed from existing benchmarks (MSQA, Nr3D) that are primarily based on the ScanNet dataset. This limits the diversity of scenes, objects, and tasks. The paper acknowledges that the framework does not yet extend to more complex, long-horizon embodied tasks.
3. The paper introduces a "grounded CoT" framework, but it lacks experiments on standard 3D visual grounding benchmarks (e.g., Nr3D, Sr3D, or ScanRefer). While Beacon3D measures coherence (grounding + QA), evaluating the grounding module's performance in isolation on these tasks seems essential to fully validate the "grounded" aspect of the CoT.

### Questions
1. The generation of the SCENECOT-185K dataset relies on rule-based methods and LLM (GPT-4O) generation. What was the extent of manual verification to ensure the quality and correctness of the intermediate reasoning "thoughts"?
2. In the 4-stage pipeline, how does the model handle error propagation? For instance, if the initial "Task Recognition" step fails, or the "Region Localization" identifies the wrong area, does this inevitably lead to a final failure, or are there mechanisms for recovery in the later stages?
3. The multi-step inference process, which involves calls to symbolic engines and expert grounding modules, seems computationally more intensive than an end-to-end model. What is the comparative inference latency of SCENECOT versus baselines like Chat-Scene or LLaVA-3D? Is the framework practical for real-time applications?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
1. The paper proposes SCENECOT, a grounded Chain-of-Thought (CoT) framework for interpretable 3D scene reasoning. It decomposes complex reasoning into four explicit steps—task recognition, region localization, entity grounding, and grounded reasoning—each supported by symbolic and multimodal expert modules.

2. It introduces SCENECOT-185K, a dataset of 185K stepwise reasoning traces covering Situated (MSQA) and Object-Centric (Beacon3D, GQA3D) tasks.

3. Experiments on MSQA and Beacon3D demonstrate improved grounding–QA coherence (34.7% vs. 19.5%) and validated gains from question-type recognition, region filtering, and grounding loss.

### Strengths
1. Novel application of CoT in 3D reasoning: Introduces an interpretable step-by-step framework that explicitly grounds each reasoning stage in scene elements. Visualization of reasoning chains (Fig. 6) makes the model’s decision process transparent and easier to diagnose.

2. Large-scale dataset: SCENECOT-185K is the first dataset pairing CoT traces with 3D scene data, supporting both situated and object-centric reasoning.

3. Clear empirical validation: Comprehensive experiments (MSQA + Beacon3D) and ablations demonstrate consistent improvement in 3D VQA task.

### Weaknesses
1. Lack of Human Verification. All QA pairs in the SCENECOT-185K dataset are generated by GPT-4o, which can introduce factual or logical errors. The paper does not mention any human validation or quality control process to ensure the correctness of the generated reasoning traces.

2. Limited Baseline Coverage. Although GPT-4o is included as a comparison model, the experiments omit stronger recent multimodal baselines such as Gemini 2.5 Pro or Claude Opus, which would provide a more comprehensive evaluation of reasoning capability.

### Questions
1. In the Grounded Reasoning, how is the "Object Image Tokens" generated? Are they derived from cropped RGB patches, projected 3D features, or another visual encoding process?

2. What exactly is the symbolic engine mentioned in the architecture?

### Soundness
4

### Presentation
3

### Contribution
3
