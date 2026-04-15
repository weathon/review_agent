# Musketeer: Joint Training/Inference for Multi-task Vision-Language Model with Task Explanation Prompts

- Decision: Reject
- Scores: 3, 8, 6

## Abstract
We present a sequence-to-sequence vision-language model whose parameters are jointly trained on all tasks and fully shared among multiple tasks, resulting in a single model which we named Musketeer. The integration of knowledge across heterogeneous tasks is enabled by a novel feature called Task Explanation Prompt (TEP). TEP reduces interference among tasks, allowing the model to focus on their shared structure. With a single model, Musketeer achieves results comparable to or better than strong baselines trained on single tasks, almost uniformly across multiple tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies how to more effectively train a unified (sequence-to-sequence) vision-language model across multiple tasks. In prior work, each training example comes with a simple description of the intended task, e.g., “Which region does the text V describe”. The main argument in this work is that such simple descriptions may not be enough and more detailed and exhaustive task descriptions are beneficial. 

For 7 pre-training tasks, the authors created detailed task descriptions consisting of data description, input/output format, and output description. Training with the detailed task descriptions shows improvement over training with simple and plain descriptions.

### Strengths
Training with more detailed task descriptions is a natural step to take for training instruction-following vision-language models. The authors verify that using such task descriptions indeed improves performance upon baselines such as OFA.

### Weaknesses
- Limited novelty and performance improvement

Replacing simple tasks descriptions with complex descriptions is a simple idea and has been successfully explored in language model literature [1]; thus the novelty is limited.

The paper shows that by using complex task descriptions, the model improves marginally upon the baseline on training tasks. It is not shown whether training on such task descriptions brings any new capacities (e.g., transfer to a new task with a new task description, or using task description to transfer to a new data domain, as is done in [1]); the core appeal of using complex task descriptions seems to be missing.

[1] Generalization via Declarative Instructions on 1600+ NLP Tasks. Wang et al. 2022



- Limited number of tasks and formats of task descriptions

The paper only studies 7 pre-training tasks, which makes the generalization of the conclusion questionable. For example, one big contributor in description is the inclusion of “data source”. Could it be because out of the 7 tasks, many data are from COCO so the model learns to utilize this information? What if the pre-training datasets all come from different image sources?

### Questions
- In Section 1.1, the paper states “For example, in visual grounding of some concept V , the prompt “Which region does the text V describe” requires the model to interpret “find” and represent the word “region” with sets of coordinates on the image plane, which do not have a meaningful (topologically consistent) representation in natural language.”  I am not quite sure what “interpret “find” and ‘represent the word “region”’ and “do not have a meaningful (topologically consistent) representation” means. Could the authors elaborate on this issue? The base prompts seem okay to me and are not ambiguous.

- How many "soft task vectors" does each task have for the learnable task vector baseline?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates how to jointly finetuning a vision-language pretrained model onto several downstream tasks to achieve both optimal performance as well as task generalization. A model called Musketeer is proposed which utilizes Task Explanation Prompt (TEP) to reduce the interference among tasks, which helps the model to optimize each single-task better during multi-task downstream finetuning. The TEP contains sufficient task meta information, including data description, input/output format, output description and instance prompt. On downstream tasks, the Musketeer model achieves comparable or better single-task results over single-task finetuned baselines. Compared with multi-task finetuned baselines, Musketeer obtains much better results.

### Strengths
1. The research question is clear and important. Currently most pretrained-then-finetuned VL models still cannot achieve task generalization and SOTA performance together.
2. Detailed TEP of each downstream VL task are given, increasing the reproducibility of this work.
3. The baseline used is competitive, demonstrating the effectiveness of Musketeer model.
4. Abundant ablation analysis is conducted.

### Weaknesses
Since the TEP contains abundant downstream meta task information, if more discussion and experiment on zero-shot new task generalization, it will be much better.

### Questions
Since the OFA model not only unifies the VL tasks but also text-only tasks. Can Musketeer also be applied on text-only tasks? Is there any experimental evidence?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new sequence-to-sequence vision-language model called Musketeer that can be trained jointly on multiple visual tasks using a shared set of parameters. The key idea is to use a novel Task Explanation Prompt (TEP) to reduce interference between tasks and allow the model to leverage shared structures. The TEP provides detailed natural language instructions about the dataset, input/output formats, output targets, etc. Experiments on 7 vision-language tasks like visual grounding, VQA, captioning etc show Musketeer matches or exceeds performance of task-specific models and other multi-task baselines. Without any task-specific tuning, Musketeer shows strong performance on all tasks using the descriptive power of TEPs to instantiate task-specific pathways at inference.

### Strengths
- The paper is well-written and clearly presented; 
- The paper proposes a novel TEP approach to reduce multi-task interference using natural language specifications. It provides a unified architecture without any task-specific tuning or heads.
- It shows strong empirical results demonstrating effectiveness for diverse vision-language tasks comparing to baselines; 
- Detailed experiments on the effects of each mixed dataset (vg, captain, ic, etc.) to the downstream have been provided across scales, which may benefit future researchers in the same area;

### Weaknesses
- TEP still relies on pretrained weights for initialization which can be expensive, the discussion regarding the additional cost might be good to provide; 
- The hyper-parameter setting as well as the Needs carefully designed TEPs for new tasks which may require some expertise.
- The study on how well TEPs could transfer to unseen tasks is unknown; 
- Some related works might also be good to include or discuss [1, 2, 3, 4]; 

[1] Dai, Wenliang et al. “InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning.” ArXiv abs/2305.06500 (2023); 
[2] Shen, Sheng, et al. "Multitask vision-language prompt tuning." WACV 2024.
[3] Asai, Akari, et al. "Attempt: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts." Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2022.
[4] Liu, Haokun, et al. "Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning." Advances in Neural Information Processing Systems 35 (2022): 1950-1965.

### Questions
- Could the author explain more on the varied performance on VQA in table 4, will using full VQAv2 training data mitigate the problems?
- Could the author provide additional training cost including the pretrainining cost for the proposed methods in Table 3 and 4 for a comprehensive evaluations;

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
