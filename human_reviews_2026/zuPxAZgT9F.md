# STVG-R1: Incentivizing Instance-Level Reasoning and Grounding in Videos via Reinforcement Learning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
In vision–language models (VLMs), misalignment between textual descriptions and visual coordinates often induces hallucinations. This issue becomes particularly severe in dense prediction tasks such as spatial–temporal video grounding (STVG). Prior approaches typically focus on enhancing visual–textual alignment or attaching auxiliary decoders. However, these strategies inevitably introduce additional trainable modules, leading to significant annotation costs and computational overhead. In this work, we propose a novel visual prompting paradigm that avoids the difficult problem of aligning coordinates across modalities. Specifically, we reformulate per-frame coordinate prediction as a compact instance-level identification problem by assigning each object a unique, temporally consistent ID. These IDs are embedded into the video as visual prompts, providing explicit and interpretable inputs to the VLMs. Furthermore, we introduce STVG-R1, the first reinforcement learning framework for STVG, which employs a task-driven reward to jointly optimize temporal accuracy, spatial consistency, and structural format regularization. Extensive experiments on six benchmarks demonstrate the effectiveness of our approach. STVG-R1 surpasses the baseline Qwen2.5-VL-7B by a remarkable margin of 20.9% on m_IoU on the HCSTVG-v2 benchmark, establishing a new state of the art (SOTA). Surprisingly, STVG-R1 also exhibits strong zero-shot generalization to multi-object referring video object segmentation task, achieving a SOTA 47.3% J&F on MeViS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new visual prompt paradigm without alignment coordinates, which transforms frame by frame coordinate prediction into instance level ID recognition. For the first time, reinforcement learning has been introduced into spatial-temporal video grounding, and task driven reward joint optimization of temporal, spatial, and format constraints has been designed. In multiple benchmark tests, refreshing SOTA resulted in a 20.9% improvement in mIoU for HCSTVG-v2.

### Strengths
1. The writing of the paper is clear, and the illustrations in the intro section effectively explain the core contribution points of this paper. In addition, the drawing of Figures 2 and 3 is also quite intuitive.

2. The problem studied in this article (cross modal alignment) is one of the core issues in the field of video grounding, and the alignment effect between the spatial-temporal dimensions directly determines the accuracy of grounding in these two dimensions.

3. The experiment used four commonly used indicators and demonstrated the accuracy of the model on the dataset used.

### Weaknesses
1. The last sentence of the second paragraph of the intro should be a description of the core idea, but this sentence is not clear. What does' a compact and interpretable formulation 'refer to.

2. Insufficient contribution in model design, only introducing pre-segmentation and reinforcement learning for video objects, lacking new method design for this problem.

3. The commonly used datasets in the field of Video Grounding include VidSTG and HC-STVG. This article only uses HC-STVG, and the effect on VidSTG is unknown, especially the results of ablation experiments.

### Questions
See [Weaknesses].

### Soundness
2

### Presentation
3

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
Brief Summary: The paper tackles the task of video spatio-temporal localization where given a video and a corresponding query, the model should output the bounding boxes + temporal timestamps. The authors propose visual prompting using a combination of yolov12 detector (for object detection) + sam2 (for visual tracking) and overlaying the information on the image itself and providing particular object ids which are also overlayed. Given the visual-prompted video, a Qwen2.5-VL model is trained via GRPO to predict the target id (which has associated bounding box) + time-range. Experiments on spatio-temporal datasets like HCSTVGv1, v2, and temporal grounding like charades-sta show that proposed STVG-R1 outperforms competitive baselines.

### Strengths
Pros:

1. The paper poses a nice application of combining spatio-temporal understanding with VLMs. Spatio-temporal understanding is an important sub-topic in video understanding and how to best leverage VLMs for this task is a well motivated problem. 

2. The proposed method is conceptually simple in re-using existing detection and tracking pipelines to utilize VLMs inherent understanding of vision without additional tokens or requiring VLM to do additional bounding box predictions. 

3. Authors provide visualization (in appendix A.2) and ablation on various prompt designs. In particular, Table 8 is very interesting, that pure SFT without GRPO training leads to worse results.

### Weaknesses
Cons:

1. It seems the absolute improvement over previous baselines is marginal? For instance on hcstvg-v1, performance matches with space-vllm and on v2, it is slightly improved over TA-STVG. On ST-Align, it is same as LLava-ST-7B.

2. The core novelty is slightly limited, the paper suggests doing visual-prompting + grpo training works. This is good to know, but unclear what are the main challenges here. 

3. One issue with the visual prompting (assuming the visualization at face value), it is unclear how the approach would tackle things with text (OCR). If you overlay a color bounding box, the text is completely lost. So the model cannot answer questions like "when did the person look at <some_text>". 

4. The model is essentially restricted to detection quality and classes of yolov12 and the quality of sam2. As such, there is no direct way to leverage VLMs internal association capability. Further dependence on separate models would lead to worse inference times requiring heavy video encoding/decoding. 

5. Table 9 in Appendix A.3 seems to suggest direct visual prompting is in fact worse? That seems like a major drawback? 

6. (Minor) It would be interesting to see results on more diverse videos, such as some ego-centric datasets (such as ego4d) or movie datasets (such as grounded-vidsitu [Ref1]).

7. The main comparison to baselines is somewhat unfair. The proposed model is able to leverage external models for tracking while baseline models need to do the prediction on their own? I could be missing something obvious here. 

8. (Minor) I am slightly confused on the r_s reward, why is it not simply 3d-iou? 

9. Authors should show additional downstream tasks which gain from such visual prompting.

---

[Ref1]: Khan, Zeeshan, C. V. Jawahar, and Makarand Tapaswi. "Grounded video situation recognition." Advances in Neural Information Processing Systems 35 (2022): 8199-8210.

---

Overall Rating: 4/10
The paper proposes grpo RL training with appropriate visual prompting for spatio-temporal video grounding. The scope however is somewhat narrow and not shown to be applicable for other video understanding tasks, and seems to slightly degrade temporal grounding. The proposed visual prompting itself might interfere with ocr reasoning, and comparison to baselines is not strictly fair.

### Questions
Q1. In general, it is advisable to do an initial cold-start before GRPO, but I don't see any references on that in the paper. Did the authors try it and didn't give good results?

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
5

### Summary
This work addresses the misalignment between textual descriptions and visual coordinates , which often induces hallucinations in Vision-Language Models (VLMs) for the Spatial-Temporal Video Grounding (STVG) task. The authors propose an "object-centric visual prompting paradigm" : via a pre-processing pipeline of detection , tracking , and ReID , each object is assigned a unique, temporally consistent ID. This reformulates per-frame coordinate prediction into a compact instance-level identification problem. Building on this, the paper introduces STVG-R1 , the first reinforcement learning framework for STVG , which employs a task-driven reward to jointly optimize temporal accuracy, spatial consistency, and structural format regularization . Experiments show the approach achieves SOTA and exhibits strong zero-shot generalization on the unseen multi-object referring video object segmentation task.

### Strengths
•	Novelty: Reformulating STVG from dense per-frame coordinate prediction into a "compact instance-level identification task", a novel idea that effectively avoids the difficult problem of VLMs handling coordinate prediction.

•	Novel RL Framework: Proposing STVG-R1, the first reinforcement learning framework for STVG, which employs a task-driven reward to optimize the VLM's reasoning.

•	State-of-the-art Results: Achieves new SOTA performance on multiple STVG benchmarks.

•	Strong Generalization: Exhibits SOTA zero-shot performance on the unseen multi-object referring video object segmentation task (MeViS), highlighting the method's robust generalization ability.

### Weaknesses
**Regarding the nature and robustness of the "visual prompting" pipeline:**

•	The pipeline is essentially a complex, training-free data pre-processing pipeline reliant on external SOTA models such as YOLO, SAM2, and ReID, rather than a novel model component.

•	The robustness of this pipeline is not discussed. For example, what happens when detection, tracking, or ReID fail? Many critical details of the pipeline, such as the arbitration logic between components, are missing, which hinders reproducibility.

•	The visual prompts themselves introduce 'visual pollution' and occlude critical information. Embedding ID characters into video frames is a lossy operation that can obscure key details of an object (e.g., facial expressions, specific markings), thereby hindering the model's understanding. Experiments also show the method is sensitive to hyperparameters such as font size, which further confirms the risk of introducing visual interference.

•	The paper does not quantify the additional computational and storage overhead introduced by this complex pipeline. How much (per-video) computational and memory overhead does running YOLO/SAM2/ReID add before using the VLM? This is crucial for assessing the method's practical usability.

**Lack of ablation studies:**

•	The paper provides no ablation study for the three components （r_t, r_s r_f） of the RL reward function R(o), making it impossible to determine the key factors driving the performance improvement.

•	Lack of ablation on the necessity of each component in the pre-processing pipeline.

**Contribution Positioning:**

•	The paper's RL algorithm (GRPO) is heavily borrowed from DeepSeek-R1, which is more of an "application-level innovation" rather than an "algorithmic innovation", and this should be clearly stated in the manuscript.

**Clarity and Justification of Key Mathematical Formulations:**

•	Regarding the spatial reward function r_s(o) in Equation (5): This reward function is designed as a sparse, binary (0 or 1) signal, which prevents the model from receiving "partially correct" feedback (e.g., for predicting a spatially adjacent but incorrect ID). The authors should justify why this sparse reward was chosen over a smoother, continuous reward that could reflect the spatial proximity of the predicted ID to the ground truth, and discuss the considerations for RL training stability and efficiency.

•	Regarding the majority voting rule in Equation (3): This formula determines the global target ID A via majority voting, which is a strong heuristic assumption. This assumption may fail for queries describing transient events (e.g., "a person who flashes by"), potentially leading to incorrect training labels. The paper should discuss the limitations of this design and its potential impact on performance.

•	Regarding the total reward function R(o) in Equation (6): The paper combines temporal and spatial rewards via simple addition, mathematically treating them as independent optimization objectives. However, the evaluation metric for spatio-temporal grounding (vIoU) is inherently coupled. The authors need to explain why a decoupled reward was chosen for training and whether there is a potential misalignment between this design and the final evaluation goal.

### Questions
See the questions above

### Soundness
3

### Presentation
3

### Contribution
3
