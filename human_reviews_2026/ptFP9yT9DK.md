# Enhanced Continual Learning of Vision-Language Models with Model Fusion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Vision-Language Models (VLMs) represent a significant breakthrough in artificial intelligence by integrating visual and textual modalities to achieve impressive zero-shot capabilities. 
However, VLMs are susceptible to catastrophic forgetting when sequentially fine-tuned on multiple downstream tasks. Existing continual learning methods for VLMs face various limitations, often relying on additional reference datasets, compromising zero-shot performance, or being restricted to parameter-efficient fine-tuning scenarios.
In this paper, we propose a novel Continual Decoupling-Unifying (ConDU) approach that pioneers the use of model fusion for continual learning in VLMs. 
Specifically, ConDU maintains a unified model along with task triggers and prototype sets, employing an iterative process of decoupling task experts for previous tasks and unifying them with the task expert for the newly learned task. 
Additionally, we introduce an inference strategy for zero-shot scenarios by aggregating predictions from multiple decoupled task experts.
Extensive experiments on the MTIL benchmark show that ConDU achieves up to a 2\% improvement in average performance across all seen tasks compared to state-of-the-art baselines, while also enhancing zero-shot capabilities relative to the original VLM. 
Our code is available at https://github.com/zhangzicong518/ConDU.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ConDU (Continual Decoupling–Unifying), a novel framework for continual learning (CL) of vision-language models (VLMs) such as CLIP. The core idea is to employ model fusion—traditionally used for combining independently fine-tuned models—to enable continual learning without catastrophic forgetting or reliance on replay data. Extensive experiments on MTIL, task-agnostic MTIL, and few-shot MTIL benchmarks show that ConDU outperforms existing methods by up to 2% in average accuracy while maintaining or improving zero-shot transfer performance.

### Strengths
1. Strong empirical gains across multiple benchmarks and settings.
2. Novel use of model fusion for continual learning of VLMs.
3. Fast inference and training efficiency due to training-free decoupling/unifying steps.

### Weaknesses
While the technical contribution is solid, the presentation quality should be improved to make the paper easier to follow. If these issues are addressed, I would be inclined to raise my evaluation.
Key suggestions:
1.	Figure clarity.
The text in figures (e.g., Fig. 1, 3, 4) is too small to read. Redrawing them with larger labels and more space would improve readability.
2.	Figure redundancy.
Fig. 1 is not clearly explained and highly overlaps with Fig. 3; consider merging the key information into Fig. 3 and removing Fig. 1 to streamline the narrative.
3.	Structure of the paper.
The introduction currently contains too much methodological detail. The core method description should be moved entirely into Section 4 (“ConDU”), leaving the introduction focused on motivation and contribution.
4.	Language and phrasing.
Several passages could benefit from linguistic polishing for clarity and conciseness. Improving transitions and consistent terminology would greatly enhance readability.

### Questions
1.	Scalability.
How does ConDU behave as the number of tasks increases significantly (e.g., >20)? Does the unifying process saturate, or does performance plateau as the unified delta model grows?
2.	Memory usage.
Since ConDU stores task triggers and prototype sets, how does the memory footprint scale with the number of tasks? Could the method benefit from pruning or compressing old triggers to maintain constant memory?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes ConDU, a continual learning framework for VLMs that introduces training-free decoupling–unifying model fusion. A single unified model is maintained; per-task “triggers” (sign mask + rescaler) reconstruct task-specific deltas, and a prototype-based semantic aggregator combines multiple reconstructed experts for zero-shot or unknown-task inference. ConDU supports both full fine-tuning and LoRA. On MTIL (plus task-agnostic and few-shot variants), it improves Average by up to ~2% over SOTA while preserving/boosting zero-shot transfer, with low fusion overhead (~1% of fine-tuning time).

### Strengths
**1. Model-fusion continual learning, broadly compatible.** ConDU formalizes a decouple–unify routine with task triggers, works with full FT and PEFT, and keeps only one deployable model plus small metadata—no replay or reference dataset. 

**2. Zero-shot/unknown-task inference.** Prototype-weighted aggregation of reconstructed experts enables robust prediction without task IDs, aligning with VLM zero-shot goals. 

**3. Consistent empirical gains and efficiency.** Across MTIL variants, ConDU raises Transfer/Average/Last and reports modest storage/time costs; decoupling/unifying are training-free and ~1% of tuning.

### Weaknesses
**1. Limited novelty.** The contribution largely integrates known model-merging ideas into a continual pipeline; the unify rule (signed extreme per parameter) and trigger design, while clever, are incremental versus established fusion schemes (Fisher, TIES, EMR). Stronger positioning against these would clarify originality. 

**2. Lacking sufficient ablation study.** Beyond an insensitivity check on the top-K selector, there is little analysis of design choices: unify rule vs. Fisher/TIES/EMR, trigger rescaling, mask binarization, prototype variants, or alternative gating (learned vs. cosine). Such ablations would substantiate the mechanism-level claims. 

**3. Benchmark scope.** Results focus on MTIL and its task-agnostic/few-shot variants (11 domains, classification); broader VLM tasks (detection, OCR, medical, video, captions) aren’t covered, limiting external validity. 

**4. Assumptions/theory.** Convergence analysis assumes identical parameter signs and i.i.d. deltas—conditions unlikely across heterogeneous tasks—so theoretical guarantees may not reflect practice. 

**5. Aggregation sensitivity.** Zero-shot prediction hinges on prototypes from the pre-trained VLM; domain shifts or text prompt bias could mis-weight experts, and calibration is not discussed. 

**6. Efficiency reporting.** While fusion is fast and memory-light, end-to-end latency under multi-expert aggregation and edge constraints is only briefly discussed; more granular profiles would aid adoption.

### Questions
1. How would performance change if the unify step used Fisher/TIES/EMR-style merging or soft masks/rescalers; which component (masking, rescale, signed-extrema) drives gains? 

2. Can learned routers or temperature-calibrated cosine improve task weighting, and how robust is aggregation to prompt wording or domain shift in the prototype bank? 

3. What happens when the theorem’s sign/i.i.d. assumptions are violated—e.g., conflicting signs across tasks—and can relaxed bounds be established empirically or analytically? 

4. What are end-to-end latency/throughput and memory footprints for K-expert aggregation on commodity GPUs/edge devices, and how does this compare to MoE or per-task checkpoints?

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
3

### Summary
The paper proposed a novel continue learning framework called ConDU, which introduces model fusion into the continue learning framework for VLM and is capable of both Full-finetuning and Parameter-efficient tuning. In the framework, the newly fine-tuned model will be merged with the old model through parameter delta based on trigger output.

### Strengths
- The trigger framework enables the model to deal with both old tasks and unknown tasks.
- The framework enables test-time scaling, which aggregates prediction results from multiple decoupled models.

### Weaknesses
- Though improvement on the benchmark is higher on average, compared to the former method, there are always certain subsets that perform worse than the former method. And the average gain is growing smaller as they continue learning and progress to the end.
- The similarity used for prototypes and task selection is computed entirely from the pre-trained VLM’s image/text features, rather than from the features of each specialized task model. This can cause a mismatch in “task selection” when a task’s decision boundary shifts substantially after fine-tuning—especially under severe domain shift. While the paper reports insensitivity to K, it does not compare “pre-trained vs. expert features.”
- The theorem assumes conditions such as “each delta has the same sign per dimension” and “i.i.d.,” which conflict with the directional conflicts commonly seen in real multi-domain/multi-task fine-tuning; therefore, its convergence claims have limited applicability to practical training.

### Questions
- When applying such a paradigm in the real world, as the number quickly increases, does such a paradigm still work well with the growing requirement for trigger saving?
- Considering the current VLM paradigm that involves LLM, would such a fusion strategy also work? The benchmark is kind of old and is hard to view as a real continuous learning setting.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel Continual Decoupling-Unifying approach to handle the fusion of delta model in VLM continual learning. Specifically, it contains a unified model along with task triggers and a set of prototypes. It derives the new unified model utilizing the old unified model with new task trigger and prototype. Extensive experimental results show the effectiveness of this method.

### Strengths
1. The proposed method is quite novel. The unified delta model it utilizes is a very novel design, and the decoupling and unifying operations are dual, which is quite interesting.

2. In continual learning, it's not necessary to store only one model. Combining a general model with multiple task-specific substructures is a good decoupling approach, which also alleviates the forgetting problem to some extent and ensures the model's ability to learn new content.

### Weaknesses
1. The proposed method doesn't seem to provide a metric for the error of the model after unification and decoupling relative to the original model. My understanding is that while the unification and decoupling in this algorithm are dual, it shouldn't guarantee a complete reconstruction of the model corresponding to task t. If I'm mistaken, please point it out (and offer my apologies). Otherwise, could you provide a rough experimental analysis to assess the error of this algorithm after unification and decoupling relative to the original model?

### Questions
The previous question was raised in Weaknesses. Regarding the performance of the experiments, the authors have compared it with some state-of-the-art (SOTA) methods and achieved a significant improvement. My main concern is the algorithm design, because this kind of hard model editing problem often leads to a huge shift in the model's parameters in its space during continual learning. Simply put, sometimes a poorly edited layer parameter can cause a huge drop in continual learning, which seriously impairs the learning of subsequent stages. Therefore, an editing method that can ensure that multiple stages do not collapse is very interesting.

### Soundness
3

### Presentation
3

### Contribution
3
