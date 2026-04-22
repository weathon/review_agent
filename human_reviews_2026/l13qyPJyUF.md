# Modality-Inconsistent Continual Learning of Multimodal Large Language Models

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2, 6

## Abstract
In this paper, we introduce Modality-Inconsistent Continual Learning (MICL), a new continual learning scenario for Multimodal Large Language Models (MLLMs) that involves tasks with inconsistent modalities (image, audio, or video) and varying task types (captioning or question-answering). Unlike existing vision-only or modality-incremental settings, MICL combines modality and task type shifts, both of which drive catastrophic forgetting. To address these challenges, we propose MoInCL, which employs a Pseudo Targets Generation Module to mitigate forgetting caused by task type shifts in previously seen modalities. It also incorporates Instruction-based Knowledge Distillation to preserve the model's ability to handle previously learned modalities when new ones are introduced. We benchmark MICL using a total of six tasks and conduct experiments to validate the effectiveness of our proposed MoInCL. The experimental results highlight the superiority of MoInCL, showing significant improvements over representative and state-of-the-art continual learning baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce a more realistic Continual Learning (CL) benchmark, where both modalities and task types vary across the model's lifecycle, which is closer to real-world use cases. They also introduce an effective framework to train an MLLM in a way that prevents forgetting both of tasks and of modalities. Their evaluation shows the framework achieves the best performance across all tasks compared to recent baselines.

### Strengths
- As noted already, the authors present a difficult and more realistic continual learning scenario: one where modalities and tasks change simultaneously.
- Their method makes sense as a replay-free method to mitigate forgetting of modalities and tasks: it leverages the previous version of the model to constrain the outputs of the new version of the model

I think the paper is well motivated, and the method explained in enough detail, making this a worthwhile contribution.

### Weaknesses
- Arbitrary, no-replay constraint. This constrains the baselines evaluated.
- Moreover, the authors generate new tasks on the fly and use the previous version of the model to generate responses in order to constrain the new version of the model. I am not exactly sure how this is preferable to simply storing some examples to replay in future stages, as you do have to maintain the previous version of the model to do so, and it requires additional inference over the generated samples.

As such, the lack of replay-based baselines is a glaring omission in the paper, and some of the most recent such baselines need to be incorporated into the paper to fully substantiate the claims in the paper. Otherwise, please explain why the no-replay constraint is realistic in the presence of the rest of your method, as it is not convincing in the paper at the moment. This is the main issue with the paper as far as I can discern.

### Questions
- The methodology, and specifically Section 3.3, I feel require a better description. The level of detail (for example, referring to sets of tasks and modalities, as they would be used in Python itself) actually hinders the ability of the reader to fully understand the method proposed. Consider switching to a higher-level description of the method, and leave readers to infer how it was implemented.

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
3

### Summary
The paper introduces a continual multi-modal learning framework composed of two modules: the Pseudo Target Generation Module (PTGM) for task-level retention and the Instruction-based Knowledge Distillation (IKD) for modality-level consistency. The approach aims to mitigate catastrophic forgetting without storing past data, evaluated mainly on captioning and VQA benchmarks.

### Strengths
The proposed dual-module design offers a conceptually unified way to address both task and modality forgetting within a continual multi-modal setting.

### Weaknesses
Limited task diversity and weak task gap
The evaluation focuses on captioning and VQA, which are linguistically similar and share overlapping objectives. The method’s effectiveness on more structured or numerically grounded tasks (e.g., object detection, regression, or reasoning with symbolic output) remains untested, making the generality questionable.

Dependence on prior task presets and uncertainty for pure-text scenarios
PTGM implicitly assumes access to prior task definitions or representations to generate pseudo-targets. It is unclear how this mechanism extends to purely textual continual tasks. This raises the question of whether PTGM truly captures task-general retention or merely adapts to multimodal patterns seen before. It would be helpful to evaluate whether fine-tuning affects the model’s pure-text capabilities.

Robustness under heterogeneous or numerical tasks
Extending from (1), PTGM’s stability on tasks requiring structured or numeric outputs is uncertain. The pseudo-target generation process may not yield consistent or interpretable supervision when the task output deviates substantially from natural language responses, leading to potential degradation or overfitting.

IKD under modality-missing inputs and hallucination risk
The IKD module aligns model outputs based on textual instructions while certain modalities are absent. Such modality-missing alignment can unintentionally encourage the model to hallucinate content from unseen modalities, especially if the textual prompts induce cross-modal imagination. . No quantitative analysis (e.g., factuality, grounding score) is provided to support the claim of improved retention.

### Questions
See the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Modality-Inconsistent Continual Learning (MICL) as a new setting in CL. MICL requires models in CL ettings to handle sequential tasks that vary both in modality and task type, creating modality-task inconsistencies that reflect real-world application. To address catastrophic forgetting in MICL, the MoInCL algorithm is developed. MoInCL integrates (i) the Pseudo Target Generation Module (PTGM) that leverages the generative capabilities of the underlying language model to create pseudo input–output pairs for previously learned task types and (ii) the Instruction-based Knowledge Distillation (IKD) mechanism which preserves the model’s modality- and task-aware reasoning by aligning the language model’s outputs across tasks using pure textual instructions. This integration reduce forgetting due to type shifts as well as modality shifts. MoInCL is evaluated on six tasks defined using three datasets . Comparative results show performance gains over existing baselines.  Ablative experiments are offered to demonstrate that both components of MoInCL contribute to the optimal performance.

### Strengths
1. The paper is well-organized and can be followed straightforwardly. 

2. PTGM and IKD provide complementary mechanisms for mitigating both task-type and modality forgetting, demonstrating a working solution for the MICL setting.

3. The approach avoids explicit data replay, making it privacy-preserving and computationally efficient compared to most CL methods that use memory buffers.

4. Experiments span six multimodal tasks across three modalities (image, audio, video) and two task types (captioning, QA)  and consider broad MLLM continual learning scenarios.

5. Ablative experiments confirm the independent contribution of each component (PTGM and IKD).

### Weaknesses
1. Despite thoughtful motivation, the way MICL is implemented in the paper is quite controlled and artificial. The task sequence is manually constructed and does not simulate natural data acquisition. The model is trained on disjoint academic datasets rather than on a dynamic data stream. Each ``incremental task'' is treated as an independent domain jump and there’s no gradual domain drift or overlapping modalities, which is typically the case in practice. Also, the method assumes no access to past data constraint which is rarely a hard constraint in applied settings. Hence, it might not be a practically usable method and is mostly a hypothetical study.

2. PTGM relies on the LLM’s ability to generate realistic pseudo input–output pairs. However, since the pseudo data are synthetically generated and not validated, they may contain factual errors, stylistic inconsistencies, or narrow diversity. Poor pseudo-data quality could introduce noise or bias, especially when earlier tasks have limited linguistic variety. The authors acknowledge this partially but the extent of this limitation isn’t studied and it hence it is unclear how well it can generalize on other tasks.

3. The experiments lack comparison with SOTA foundation models. LLaVA-like architecture is used but there’s no test on larger or more capable MLLMs, e.g., Qwen-VL or Kosmos-2. Hence, it is unclear whether this setting is indeed a challenge for larger models that have been trained on a more diverse training dataset. Moreover, it’s unclear whether MoInCL’s mechanisms scale effectively to large models or high-capacity encoders.

4. The baselines are mostly generic continual learning (CL) methods. With the exception of PathWeave, these baselines were not designed for multimodal or instruction-tuned large language models, and certainly not for modality-inconsistent continual learning.
Thus, they are methodologically mismatched and easy to beat. Outperforming vision-only CL baselines does not necessarily prove superiority for multimodal continual learning when advanced methods are missing.

5. The experiments focus mainly on final average performance and forgetting, but they lack per-step performance trajectories (how fast forgetting accumulates), task transfer analysis (positive or negative forward transfer). Without these metrics, the robustness and reliability of the results remain uncertain. The analysis of why QA accuracy underperforms in one task order is interesting but limited .

6. The code is not provided which makes judgment about reproducibility challenging.

### Questions
1. It looks like that the current setup treats modalities as disjoint and sequential, but in reality, they often co-occur, e.g., video contains both visual and audio information. How can continual learning handle multimodal fusion tasks, where multiple modalities are learned jointly and evolve asynchronously?

2. What is the computational load for executing MoInCL. Can you report both time and memory usage?

2. Can MoInCL scale to large foundation models such as Gemini or Qwen-VL and web-scale multimodal datasets with managable compute costs?

3. The experiments include only six tasks which is a relatively short continual learning horizon. How does performance degrade over large number of incremental tasks? What happens if pseudo-target drift or instruction-distillation error accumulate and eventually destabilize the method?

4. In real applications, data streams are unlabeled and mixed and the model must infer both the modality and the task type autonomously. Is it easy to extend the method such that the model can perform unsupervised or self-supervised task detection and adaptation?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Modality-Inconsistent Continual Learning (MICL), a continual learning scenario for multimodal large language models, addresses catastrophic forgetting driven by the sequential introduction of tasks that combine both inconsistent modalities and varying task types. To mitigate these issues, the authors propose MoInCL, a framework that employs two components: the Pseudo Targets Generation Module (PTGM) and Instruction-based Knowledge Distillation (IKD).

### Strengths
- The MICL scenario is an extension of continual learning for MLLMs, moving beyond the limited scope of vision-only or modality-incremental settings without task-type variations. 
- The method operates entirely in a memory-free setting, avoiding the computational and privacy overhead associated with memory-based replay, which is an advantage for scaling up to real-world MLLMs with massive, diverse datasets.

### Weaknesses
- The central claim of the paper rests on introducing a modality-inconsistent scenario. However, the benchmark is severely limited to only three modalities (Image, Audio, Video) and two simple task types (Captioning, QA). This minimal setup is not truly inconsistent for a modern MLLM and fails to validate the framework’s robustness against the complexity an MLLM will actually face, e.g., 3D, depth, or complex reasoning/planning tasks. 
- The proposed Pseudo Targets Generation strategy is the same as that in [A]. Specifically, both methods utilize the model's own generative capabilities on new inputs to create pseudo-targets for old task structures, then distill knowledge from these targets. However, there is no citation or discussion of this prior art. Given the strong conceptual similarity, the novelty of this core strategy is severely limited compared with [A]. 

[A] Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning，ACL 2024.

- The current PTGM is narrowly tailored to simple, unstructured text-generation tasks. These tasks benefit from the LLM's inherent generative capability to produce pseudo-labels. However, the MLLM landscape rapidly expands to include highly structured or complex reasoning tasks such as visual grounding requiring bounding box coordinates, visual-language navigation requiring action sequences, or advanced chain-of-thought reasoning requiring complex, verified intermediate steps. The paper fails to discuss how PTGM can reliably be extended to generate high-fidelity structured pseudo-labels without explicit external supervision. 
- IKD is designed to prevent forgetting by distilling the LLM's output using pure text instructions. Forgetting in a multimodal context occurs when the LLM forgets how to interpret the features passed from the modality-specific trainable projection layer. IKD's text-only approach ensures the LLM retains its general instruction-following and language generation capabilities, but there is insufficient evidence that it effectively preserves the cross-modal alignment knowledge necessary for the old modality to generate a correct output from its encoded features. It may only be preserving a generic language model, not a multimodal one.

- The authors state they use a LLaVA-like architecture but do not use LLaVA's pre-trained parameters for fairness. This decision is problematic: it decouples the model from the robust pre-training that makes MLLMs powerful, and it places the continual learning baselines in an artificially disadvantaged position. A stronger comparison would involve a highly-tuned, fully pre-trained MLLM (e.g., a real LLaVA/QwenVL) followed by the continual learning process.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new modality-inconsistent continual learning task that addresses the combined challenges of modality and task-type shifts, both of which contribute to catastrophic forgetting in continual learning. To tackle these challenges, the authors proposed MoInCL, a framework that integrates a Pseudo Target Generation Module to alleviate forgetting caused by task-type changes and an Instruction-based Knowledge Distillation strategy to preserve previously acquired knowledge. Extensive experiments and ablation studies are conducted to evaluate the efficacy of the proposed method and verify the contribution of each component.

### Strengths
+ The paper introduces a new continual learning task, modality-inconsistent continual learning (MoInCL), where the learning process must handle sequential tasks that vary in both modality and task types.
+ The proposed method is technically sound, addressing the challenges of cross-modality and cross-task-type learning effectively.
+ The proposed method is exemplar-free, utilizing a pseudo target generation module to produce replay samples for self-distillation. This design enhances practicality and scalability in real-world continual learning scenarios.
+ The quantitative results demonstrate the superiority of the proposed method over SOTA baselines across different multimodal continual learning tasks, with significant performance gains.
+ Extensive experiments and ablation studies are conducted to validate the effectiveness of the proposed method and contributions of each component.

### Weaknesses
- The proposed framework currently supports only a single non-text modality combined with text, limiting its generalizability to more complex multimodal settings (e.g., involving video, audio, and text simultaneously).
- The robustness to task order remains a concern. In experiments, the results show that different orders cause varying performance across different tasks. Experimental results indicate that performance varies under different task sequences, which could hinder the method’s applicability in real-world scenarios where task order is often unpredictable. For example, when a challenging task (e.g., image captioning) precedes other QA tasks, the model may still experience substantial forgetting.
- The pseudo target generation relies on large language model (LLM) outputs, which may suffer from hallucination issues. Inaccurate or low-quality pseudo targets could negatively affect the model’s ability to preserve previously learned knowledge.

### Questions
Please see the concerns in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
