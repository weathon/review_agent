# OccVLA: Vision-Language-Action Model with Implicit 3D Occupancy Supervision

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Multimodal large language models (MLLMs) have shown strong vision–language reasoning abilities but still lack robust 3D spatial understanding, which is critical for autonomous driving. This limitation stems from two key challenges: (1) the difficulty of constructing accessible yet effective 3D representations for open-world object modeling, and (2) the loss of fine-grained spatial details in VLMs due to the absence of large-scale 3D vision–language pretraining. To address these challenges, we propose OccVLA, a novel framework that integrates 3D occupancy representations into a unified multimodal reasoning process. Unlike prior approaches that rely on explicit 3D inputs, OccVLA treats dense 3D occupancy as both a predictive output and a supervisory signal, enabling the model to learn fine-grained spatial structures directly from 2D visual inputs. The occupancy prediction are regarded as implicit reasoning processes and can be skipped during inference without performance degradation, thereby adding no extra computational overhead. OccVLA achieves state-of-the-art results on the nuScenes benchmark for trajectory planning and demonstrates superior performance on 3D visual question-answering tasks, offering a scalable, interpretable, and fully vision-based solution for autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel framework called OccVLA, which integrates 3D occupancy supervision into a VLA model. The model is trained to predict latent occupancy representations, while the occupancy reasoning can be skipped during inference. Experimental results demonstrate that OccVLA achieves strong performance on the nuScenes planning and QA tasks.

### Strengths
1.	OccVLA presents an effective approach to integrating occupancy supervision and demonstrates promising results on the nuScenes planning and QA tasks.

2.	The method introduces no additional inference cost.

3.	The visualization in Figure 5 shows that OccVLA can generate meaningful occupancy representations.

### Weaknesses
1.	In OccVLA training, both occupancy supervision and CoT supervision are used jointly. It would be clearer to isolate the contribution of each component and provide detailed ablation studies. Additionally, it is unclear which model serves as the baseline that excludes only the OccVLA-nuScenes training.

2.	It remains unclear how occupancy prediction is removed during inference and how much the resulting misalignment between training and testing affects performance. Furthermore, is this skipping process a default setting used in all experiments or an optional acceleration setting?

3.	(Minor) It looks like the authors modify the line space significantly. The captions of Figure 1 and Figure 5 are nearly overlap with the following text.

4.	(Minor) There is a spelling error in Line 299, where the word “choozaizuose” is misspelled.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces OccVLA, a Vision-Language-Action model designed to integrate 3D spatial understanding critical for autonomous driving. It addresses the limitation of current Multimodal Large Language Models (MLLMs) by treating dense 3D occupancy as an implicit supervisory signal, allowing the model to learn fine-grained spatial structures directly from 2D input. Crucially, the 3D occupancy prediction is an intermediate reasoning step that can be skipped during inference, ensuring no added computational overhead or latency degradation. The comprehensive experiments demonstrate the model's effectiveness across various tasks, including VQA, grounded perception, and autonomous driving planning on standard benchmarks.

### Strengths
1. The concept of utilizing implicit 3D occupancy as a self-supervisory signal represents a significant contribution to spatial reasoning in autonomous driving.
2. The unified architecture is capable of handling diverse multimodal tasks (perception, reasoning, and action planning) within a single, consistent framework.

### Weaknesses
1. The core concept bears significant resemblance to the approach proposed in previous works ([*] Ross: Reconstructive Visual Instruction Tuning, ICLR2025), yet the paper fails to discuss or provide a comparative analysis against these related works.
2. The paper introduces the "Occupancy Transformer" as a core component, yet it fails to clearly articulate its novel architectural differences compared to a standard transformer block.
3. The description of the Latent Occupancy Prediction module is incomplete in several crucial aspects. Firstly, the motivation for choosing VQ-VAE over alternative modern generative models, such as diffusion-based variants, is insufficiently discussed. Secondly, many critical architectural hyperparameters are missing, most notably the specific number of queries used for the transformer-based prediction.
4. The evaluation of the planning capability is limited to open-loop metrics on the nuScenes dataset, lacking essential simulation or closed-loop experiments.
5. A significant concern remains that the overall complexity and integrated architecture—even without explicitly activating the 3D occupancy prediction—still introduces an inherent latency overhead compared to simpler, pure VLA baselines. A detailed latency profile comparing the final OccVLA model (with the module skipped) against relevant efficient baselines and against its own full model variant (with the occupancy module included) is essential for validation.
6. Line 299 contains a clear typo ("choozaizuose").

### Questions
None

### Soundness
3

### Presentation
2

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
The paper proposes OccVLA, a novel Occupancy Vision-Language-Action framework that integrates dense 3D occupancy prediction into the VLM backbone, enabling the model to learn fine-grained spatial understanding directly from 2D images. This approach treats occupancy prediction as an implicit reasoning process that can be skipped during inference for zero computational overhead, achieving state-of-the-art results on nuScenes for trajectory planning and 3D visual question-answering.

### Strengths
1. The paper is highly motivated. Using a dense 3D signal to help the VLM/MLLM improve the spatial understanding capability is a novel idea.
2. The paper is easy to follow.
3. The methods demonstrate SOTA performance on two benchmarks.

### Weaknesses
I have major concerns in the paper's incorrect statements and experiments. 
1) The authors make an invalid claim regarding existing VLM supervision. The statement in the introduction section that "supervision relies on 3D annotations described in text (e.g., coordinates or bounding boxes), which are inherently weak and sparse"  lacks supporting evidence. While bounding boxes are inherently sparser than dense occupancy, the term 'weak' is judgmental and should either be supported by a metric-based comparison or replaced with a more neutral term.
2) The argument against the scalability of 3D bounding box supervision is weakened by recent advances in auto-labeling. The critique that prior methods are constrained by a lack of scalability due to the need for "extensive manual labeling" is outdated. Given that the proposed OccVLA-nuScenes dataset relies on automated pipelines to generate occupancy, the authors should acknowledge that recent advancements have also enabled high-quality auto-labeling for 3D detection/bounding box supervision, thereby allowing both dense (occupancy) and sparse (box) supervision methods to scale.
3) The characterization of the EMMA baseline is inaccurate and potentially misleading. The paper claims that the state-of-the-art method EMMA relies on costly supervision (3D/BEV coordinates & 3D bounding box) that "limits its scalability". This is incorrect for the base model, which achieves strong results via self-supervision on motion planning trajectories alone. The authors should clarify the specific EMMA variant they are comparing against and justify why its data requirements fundamentally limit scalability more than their own approach, which requires dense, automatically generated 3D occupancy data.
4) The related work section and experimental evaluation lack important context. (1) The authors should discuss recent and relevant work such as S4-Driver[1], which also focuses on improving Large Language Models for driving by enhancing spatio-temporal visual representation and lifting 2D information to 3D. (2) The nuScenes planning benchmark is rapidly becoming outdated for challenging scenarios. To demonstrate the robust performance of OccVLA on the latest benchmarks and its capability in long-tail scenarios, the authors are strongly encouraged to provide results on the current standard, such as the Waymo Open Dataset End-to-End Driving (WOD-E2E) challenge[2].

[1]S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Modelwith Spatio-Temporal Visual Representation
[2] WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios

### Questions
**Verification of Foundational Claims**
The paper's narrative regarding the limitations of existing supervision requires justification, especially concerning scalability and the quality of ground truth.
- Scalability of Bounding Box Supervision: The claim that models requiring 3D Bounding Boxes are not scalable due to human effort is debatable, given recent advances in high-quality auto-labeling techniques for 3D detection. To validate the paper's core motivation, the authors must address the reliance on automated pipelines for both dense and sparse supervision. I request that the authors either:
1)  Provide quantitative evidence that the auto-labeling quality of the occupancy grid map is significantly superior to modern auto-labeled 3D detection bounding boxes, or that occupancy GT requires no auto-labeling at all.
2) Acknowledge the scalability argument and adjust the paper's storyline accordingly.
- Unsupported Claim on Supervision Quality.  The paper states that 3D annotations described in text are "inherently weak and sparse." While they are sparser than dense occupancy, the term "weak" lacks a clear technical definition or reference. I request that the authors either provide evidence to justify the use of "weak"

**Comprehensive Evaluation**
- The analysis of related work is incomplete. I believe a crucial comparison with S4-Driver is necessary.
- Outdated Benchmark : The nuScenes planning benchmark is becoming outdated. To demonstrate the method's real-world generalization to challenging long-tail scenarios, I strongly recommend evaluating performance on  Waymo Open Dataset End-to-End Driving (WOD-E2E) challenge. If the method cannot produce good occupancy grid maps or competitive planning results in WOD-E2E, this indicates a severe limitation in its generalization or scalability that must be clearly discussed as a weakness.

### Soundness
1

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
This paper proposes OccVLA, a framework that integrates 3D occupancy representations into vision-language models for autonomous driving. The key innovation is treating occupancy prediction as both an output and supervisory signal, which can be skipped during inference. The authors evaluate on nuScenes for trajectory planning and 3D visual question answering, achieving competitive results.

### Strengths
1.Using occupancy as implicit supervision rather than explicit input is creative and addresses computational efficiency concerns during inference.
2.The paper evaluates on multiple tasks (motion planning, VQA, occupancy prediction) and provides extensive comparisons with relevant baselines.
3.The automated data pipeline for generating meta-actions and CoT annotations could be valuable for the community.
4.Achieves state-of-the-art on trajectory planning (0.28m average L2 error) and superior performance on NuScenes-QA (59.5% accuracy).
5.The paper includes useful ablations on occupancy supervision and ego trajectory input.

### Weaknesses
1.The core architecture (cross-attention between occupancy queries and visual features) is relatively standard. The main contribution is using occupancy as supervision, which feels incremental.
2.The occupancy prediction results (~10% mIoU) are not comprehensively evaluated against specialized occupancy prediction methods
3.Several grammatical errors and awkward phrasings (e.g., "choozaizuose" in line 299)
4.The paper structure could be improved - the three-stage training is mentioned but not clearly motivated
5.All experiments are on nuScenes. How does the approach generalize to other datasets or driving scenarios?

### Questions
1.Can you provide actual inference time measurements comparing with/without occupancy prediction?
2.Can you ablate the value of λ_occ and provide more justification for your choice?

### Soundness
3

### Presentation
2

### Contribution
2
