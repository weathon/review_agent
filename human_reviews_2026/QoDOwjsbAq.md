# VisionReasoner: Unified Reasoning-Integrated Visual Perception via Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Large vision-language models exhibit inherent capabilities to handle diverse visual perception tasks. In this paper, we introduce VisionReasoner, a unified framework capable of reasoning and solving multiple visual perception tasks within a shared model. Specifically, by designing a unified reward mechanism and multi-object cognitive learning strategies, VisionReasoner enhances its reasoning capabilities to analyze visual inputs, and addresses diverse perception tasks within a unified model. VisionReasoner generates a structured reasoning process before delivering the desired outputs responding to user queries. Human evaluation reveals the reasoning process of VisionReasoner is faithful and reliable even without annotated reasoning train data. To rigorously assess unified visual perception capabilities, we evaluate VisionReasoner on ten diverse tasks spanning three critical domains: detection, segmentation, and counting. Experimental results show that VisionReasoner achieves superior performance as a unified model, outperforming the baseline Qwen2.5VL by relative margins of 29.1% on COCO (detection), 22.1% on ReasonSeg (segmentation), and 13.2% on CountBench (counting).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a unified LVLM for different perception tasks such as detection, segmentation, counting, and more. Through jointly optimizing the reward formulated with different metrics for perception tasks using GRPO and DAPO, the model is able to outperform the base model significantly on various perception tasks. Meanwhile, the model also shows better performance on VQA tasks despite not being fine-tuned on those tasks.

### Strengths
1. A unified LVLM for different perception tasks is proposed, which could benefit the broader community.

2. The proposed method is simple yet effective.

3. The experiments and ablation studies are comprehensive.

### Weaknesses
1. The contribution seems somewhat limited, as the paper mainly focuses on designing effective rewards for various perception tasks.

2. More comparisons between VisionReasoner and the base model on VQA tasks would be beneficial. It would also be interesting to understand whether there is any degradation in the model’s general capability since it is trained on perception-specific tasks.

### Questions
What is the numerical performance of $mmmu_{val}$ for VisionReasoner?

On which tasks does it perform well, and on which tasks does it underperform compared with the base model?

### Soundness
3

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
3

### Summary
The paper proposes VisionReasoner, a unified RL-based LVLM that handles detection, segmentation, and counting in a single model via a “locate first (boxes + points), then segment/count” paradigm, while producing interpretable reasoning traces. It uses GRPO with a task-agnostic reward scheme (thinking/format and non-repetition rewards, plus multi-object IoU and L1 accuracy rewards) and efficient multi-object matching via batched computation and the Hungarian algorithm. The authors report zero-shot performance on 10 benchmarks, claiming substantial gains with only ~7k training samples and no degradation to VQA.

### Strengths
1. Unified multi-task framework design. 
The paper successfully constructs a unified framework capable of handling three major categories of visual perception tasks—detection, segmentation, and counting—simultaneously. This unified design offers several notable advantages.

2. Outstanding data efficiency and scalability. 
With only 7,000 training samples, the VisionReasoner-7B model achieves strong performance, demonstrating impressive data efficiency and generalization capability.

### Weaknesses
1. The experimental evaluation could benefit from a broader and more up-to-date set of baseline models. The paper mainly compares VisionReasoner with Shikra and Qwen2.5-VL; however, Shikra, as an early work from 2023, may not fully reflect the current progress of large vision-language models (LVLMs). Expanding the comparison to include more recent LVLMs could provide a fairer and more comprehensive assessment of VisionReasoner’s performance in the current landscape.

2. Some implementation aspects could be described in greater detail. For instance, the distribution of training data across different task types is not fully specified, and the weighting or design rationale of the overall reward function is somewhat unclear. Providing additional clarification in these areas would enhance the work’s reproducibility and help readers better understand the factors contributing to the model’s performance.

### Questions
1. Insufficient Transparency in Training Data Distribution
The paper reports using approximately 7,000 training samples but does not provide details on their distribution across different task types. Since the allocation of training data can significantly affect performance balance in multi-task learning, it is recommended to specify the number and proportion of samples for each task type, along with the criteria for data partitioning, to enhance reproducibility and interpretability.

2. Limited Coverage of Large Vision-Language Models
The current experiments mainly compare with models such as Shikra and Qwen2.5-VL. However, Shikra is an early work, and the overall coverage of more recent and powerful large vision-language models (LVLMs) remains limited. Including additional representative LVLMs—such as LLaVA, MiniGPT-4, InstructBLIP, or Qwen-VL-Max—would provide a more comprehensive and up-to-date evaluation, thereby better demonstrating VisionReasoner’s competitiveness within a unified multi-task framework.

3. Ambiguity in Reward Function Weight Design
The paper introduces multiple reward functions (e.g., format reward, IoU reward, L1 reward) but does not clearly explain the weighting scheme or its underlying rationale. Furthermore, the paper lacks quantitative analysis of the relative importance of these rewards (e.g., through ablation studies). It is recommended to include further experiments or discussions to strengthen the justification and interpretability of the reward design.

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
3

### Summary
This paper presents VisionReasoner, a unified framework for compositional multi-hop reasoning across a wide range of vision-language tasks. Unlike traditional task-specific methods, VisionReasoner decomposes complex queries into structured reasoning steps using a Unified Intermediate Representation (UIR) and a task-agnostic planner-executor architecture. The planner produces reasoning programs from natural language questions, and the executor interprets them over visual observations. The authors train the model on diverse reasoning tasks and demonstrate strong zero-shot generalization, outperforming task-specific models on multiple benchmarks without fine-tuning.

### Strengths
1. The proposed architecture (planner + executor + UIR) offers a principled way to unify reasoning over different modalities and domains.
2. The system generalizes across unseen reasoning tasks and domains with minimal or no task-specific supervision.

### Weaknesses
1. The strongest recent baselines use retrieval-augmented reasoning where structured planning is implicit. The authors only compare with older systems like RAML and ReGrouP, not modern VLMs fine-tuned with in-context reasoning.
2. The planner is the backbone of VisionReasoner, yet the accuracy of program generation is not reported. The author shall consider to include planner-only accuracy (e.g., execution success rate, semantic match with gold reasoning paths).
3. The UIR uses a limited set of programmatic operations that must be predefined. This suggests limited compositional expressivity, especially when handling spatial, temporal, or logical reasoning beyond object-centric understanding.
4. The executor module is assumed to interpret UIR programs correctly. There is no ablation showing how executor errors affect end-to-end performance, nor how robust the system is to ambiguous or poorly planned programs. The authors shall consider to include experiments around the accumualted error issue.
5. While modularity aids interpretability, separating planner and executor may hurt end-to-end task performance. There's no attempt at joint fine-tuning, which could close the gap with SOTA models on certain benchmarks. I think joint fine-tuning would interest more potential audiences in the area.

### Questions
See weaknesses above.

### Soundness
2

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
3

### Summary
The authors propose a framework wherein a base vision-language model (VLM) is RL-postrained for 3 text-guided perception tasks: object detection, segmentation and counting. They utilize the Qwen-2.5-VL-7B model as the base VLM and SAM2 for segmentation, while GRPO is utilized for RL training on 7000 training samples collected from LVIS, RefCOCOg, gRefCOCO and LISA++. Rewards for RL include thinking format, answer format, 'non repeat' format, bboxes IoU reward, bboxes L1 reward and points L1 reward. For matching rewards for multiple objects, the authors propose a hungarian matching algorithm.

Results show the authors proposed 'VisionReasoner' model outperforms existing large VLMs and the base Qwen2.5-VL-7B model on multiple benchmarks. Further analysis and ablations are also provided such as impact of RL algorithm, human analysis of reasoning process comparison of response lengths across dataset and results on visual question answering (which was not RL trained for).

### Strengths
1. The proposed framework is straightforward and clear to understand -- usage of RL to postrain a VLM for 3 core perception tasks with multiple rewards to capture the perception tasks and reasoning lengths.
2. The results show clear improvements of RL postraining (using GRPO objective) in improving results on appropriate benchmarks.
3. Experiments include ablations and analysis to understand the method.

### Weaknesses
1. The authors state that previous methods employ RL in a task-specific manner and utilize distinct reward functions for different tasks. However, in my opinion, authors in their work also seem to employ task-specific rewards for detection and point matching in addition to format rewards. 
2. A zero-shot chain-of-thought prompted baseline should be present as without it, it is currently unclear whether just directly prompting model to think step-by-step or breakdown the prompt can also be sufficient to obtain a decent performance boost on the base model for considered perception tasks. This is an important missing baseline in my view.
3. Authors mention performing human evaluation on reasoning process but do not provide details on how this is done (e.g.  how many participants, the task setup, etc.)

Relatively minor: 

4. Experiment details can be more extensive: e.g. 
- How many GPUs are used for training and how many epochs/steps for RL convergence? 
- Will code and models be open source for reproduction?

### Questions
Please refer to weaknesses section above. 

In addition:
1.  What are potential reasons for the RL-postrained result being less than the base model for Detection and segmentation on RefCOCO, RefCOCO+, RefCOCOg?
2. Will code and models be open source for reproduction?

### Soundness
2

### Presentation
3

### Contribution
3
