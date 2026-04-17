# RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Advances in large vision-language models (VLMs) have stimulated growing interest in vision-language-action (VLA) systems for robot manipulation. However, existing manipulation datasets remain costly to curate, highly embodiment-specific, and insufficient in coverage and diversity, thereby hindering the generalization of VLA models. Recent approaches attempt to mitigate these limitations via a plan-then-execute paradigm, where high-level plans (e.g., subtasks, trace) are first generated and subsequently translated into low-level actions, but they critically rely on extra intermediate supervision, which is largely absent from existing datasets. To bridge this gap, we introduce the RoboInter Manipulation Suite, a unified resource including data, benchmarks, and models of intermediate representations for manipulation. It comprises RoboInter-Tool, a lightweight GUI that enables semi-automatic annotation of diverse representations, and RoboInter-Data, a large-scale dataset containing over 230k episodes across 571 diverse scenes, which provides dense per-frame annotations over more than 10 categories of intermediate representations, substantially exceeding prior work in scale and annotation quality. Building upon this foundation, RoboInter-VQA introduces 9 spatial and 20 temporal embodied VQA categories to systematically benchmark and enhance the embodied reasoning capabilities of VLMs. Meanwhile, RoboInter-VLA offers an integrated plan-then-execute framework, supporting modular and end-to-end VLA variants that bridge high-level planning with low-level execution via intermediate supervision. In total, RoboInter establishes a practical foundation for advancing robust and generalizable robotic learning via fine-grained and diverse intermediate representations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on the plan-then-execute paradigm for robotic manipulation and tries to fill gap of lacking fine-grained intermediate supervision with the proposed RoboInter Manipulation Suite. RoboInter Manipulation Suite contributes in three aspects: 1) RoboInter-Data, a human-veriffed dataset with over 200k episodes across 571 diverse scenes, 2) RoboInter-VQA, 8 spatial and 20 temporal embodied QA categories to benchmark and enhance the embodied capabilities of current large vision-language models, and 3) RoboInter-VLA, a ffexible plan-then-execute framework with modular and end-to-end variants that link planning to execution. Experiments indicate that RoboInter-Data significantly improves the model's reasoning capabilities for embodied tasks. In closed-loop evaluations, RoboInter-VLA is shown to effectively complete both in-distribution and out-of-distribution tasks when compared to Pi0 and OpenVLA.

### Strengths
1. The dataset makes a significant contribution to the plan-then-execute paradigm and can also be utilized in other research paradigms.

2. Comprehensive experiments validate the contributions of the dataset effectively.

3. The insightful analysis of the variants of RoboInter-VLA provides valuable inspiration for other researchers in this field.

### Weaknesses
1. There is a lack of evaluation of other plan-then-execute methods trained on the proposed dataset, which would provide a more comprehensive validation of the dataset's contributions.

2. There is insufficient analysis regarding which types of intermediate representations (subtask, primitive, target box, affordance) are most beneficial for robotic manipulation tasks. Are all the mentioned representations utilized as inputs to RoboInter-VLA during both open-loop and closed-loop evaluations?

3. An analysis of inference time for RoboInter-VLA is missing.

### Questions
1. The RoboInter-Modular variant demonstrates superior performance in open-loop evaluations. However, it is not included in the closed-loop evaluation. What are the reasons for this exclusion?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents RoboInter Manipulation Suite, a unified resource of data, benchmarking, and training vision-language-action (VLA) models. A key contribution of this paper is to propose a new dataset for improving embodied reasoning capabilities of VLA models. The dataset consists of fine-grained annotations of 200k video data, including language, key frame, target object, gripper coordinate annotations. The annotations were labeled in the RoboInter-Tool. Based on this data, the authors designed the visual question answering (VQA) tasks that require vision-language models to answer spatial or temporal questions. Finally, this paper also proposes a VLA model trained on the collected data. Experiments show that RoboInter-VLA improves reasoning capabilities on both third-party benchmarks (e.g., RoboVQA) and the proposed RoboInter-VQA tasks. RoboInter-VLA was also deployed to the real-world environment, and it showed improved performance, especially in out-of-distribution robot manipulation tasks.

### Strengths
- [S1] The paper proposes a new data, a benchmark, and a VLA model for embodied reasoning, which could be a great resource for the community. 
- [S2] Experiments validate the effectiveness of the collected data on both other benchmarks and the proposed benchmark.
- [S3] The paper is well-written and easy to understand.

### Weaknesses
- [W1] While a plan-then-execute framework improves reasoning and interpretability by generating intermediate representations, it inherently hurts the latency of robotic systems. The paper does not provide any analysis regarding the inference speed of RoboInter-VLA compared with other VLA models. This could improve the understanding of the trade-off between reasoning capabilities and real-time performance in robotic systems.

### Questions
- [Q1] Will the RoboInter-VQA dataset be made publicly available?

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
This paper introduces the RoboInter manipulation suite, which includes RoboInter-Tool, a lightweigh tGUI for semi-automatic
 per-frame annotation of embodied videos,and RoboInter-Data,a human-verified dataset with over 200k episodes across 571diverse scenes. Based on them, This paper also introduces a RoboInter-VQA benchmark and there 'plan-then-execute' framework (IC-E2E / EC-E2E / Modular). Experiments show the pre-trained Planner improves open- and closed-loop generalization.

### Strengths
1. The introduced RoboInter-Data is large-scale, including 200k episodes, 91M frames, 571 scenes.

2. This paper offers rich intermediate representation categories (sub-tasks, skills, affordances, boxes, traces, contact frames, etc.) during constructing RoboInter-VQA. These representations are critical for improving the abilities (like spatial intelligence) of embodied brain and explaination during bulding VLA systems.

3. This paper also introduces a full toolchain (RoboInter-Tool): an open-source GUI integrating SAM-2 and GPT-4o for semi-automatic segmentation, tracking and language annotation, reducing labeling cost.

4. Based on these rich intermediate representation, this paper also develop three flexible training framework (IC,EC,Module).

5. Extensive are conducted in multimodal benchmarks, RoboInter-VQA, and real-world tasks.

### Weaknesses
1. The paper does not introduce architectural or training-paradigm innovations for vision-language-action (VLA) models. Its contributions mainly lie in demonstrating that the proposed intermediate representations can be effective within existing VLA and plan-then-execute frameworks.

2. While the paper mentions prior works that employ explicit chain-of-thought (EC) representations (e.g., ECoT), it does not provide sufficient discussion or comparative analysis with these conceptually related methods. This omission makes it difficult to position RoboInter’s contributions relative to previous efforts.

3. In most real-world experiments, the IC variant performs better than the EC one, raising uncertainty about the practical utility of explicit intermediates. A more systematic analysis of when and how different forms of intermediate representation benefit VLA learning would strengthen the paper.

### Questions
Please see the weaknesses.

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
The paper presents the RoboInter Manipulation Suite, which operationalizes a plan-then-execute paradigm for robotic manipulation. It first introduces RoboInter-Tool for semi-automatic, per-frame annotation, and uses it to build RoboInter-Data—a large-scale, scene-diverse dataset that includes object and gripper bounding boxes, contact frames, trajectories, affordances, atomic skills, and sub-tasks. Based on these annotations, the authors design RoboInter-VQA, which turns intermediate representations into concrete task types to train and evaluate VLMs’ embodied perception and planning. At the model level, the paper proposes RoboInter-VLA with three variants—implicitly conditioned end-to-end (IC-E2E), explicitly conditioned end-to-end (EC-E2E), and a modular Planner→Executor setup—and bridges planning and control with a composable intermediate “chain-of-thought” (F-CoT) in text or visual form. Experiments show that planners trained on the proposed data and VQA substantially outperform base and embodied VLMs on Where2Place, RoboRefIt, RoboVQA, and the RefCOCO family; in both open-loop (OLS) and real-world closed-loop (Franka, ID/OOD) evaluations, explicit intermediate supervision and modular designs yield faster convergence and stronger generalization.

### Strengths
This paper makes important systematic contributions to the field of robotic manipulation. Originality: It represents the first large-scale systematic effort to address the data scarcity problem of intermediate representations in robotic manipulation, with RoboInter-Tool providing an innovative semi-automatic annotation solution. Quality: High-quality annotations for 200,000 videos are ensured through human verification and multi-round cross-validation, with comprehensive experimental design covering spatial/temporal understanding generation tasks as well as open-loop/closed-loop evaluation, achieving significant improvements across multiple benchmarks. Clarity: Experimental settings and evaluation metrics are clearly defined, and ablation studies help understand the contributions of different components. Significance: This work addresses a key bottleneck constraining the development of the "plan-then-execute" paradigm, provides valuable data resources and tools for the robotic learning community, and demonstrates good generalization capabilities in real-world experiments.

### Weaknesses
The accuracy verification of camera parameter estimation and 2D-3D projection is insufficient, lacking quantitative analysis of inter-annotator consistency and failure case studies. Meanwhile, current experiments are primarily based on the Franka robotic platform, lacking cross-platform validation and generalization capability evaluation in more complex scenarios, with limited scale and diversity in real-world experiments. The systematic ablation studies on F-CoT design and various intermediate representations are inadequate. The time costs and economic overhead of human-in-the-loop annotation have not been quantitatively analyzed and require more detailed evaluation.

### Questions
Technical Implementation Details**: Could you provide more detailed analysis of camera parameter estimation accuracy and 2D-3D projection errors? What is the quantitative inter-annotator agreement (e.g., Cohen's kappa) for different annotation types?
What are the specific time costs and economic overhead for human-in-the-loop annotation? 
Given that experiments primarily use Franka robots, how would you validate generalization to other robotic platforms ? Do you have plans for cross-embodiment evaluation?

### Soundness
4

### Presentation
3

### Contribution
3
