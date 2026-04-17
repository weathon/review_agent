# RegionReasoner: Region-Grounded Multi-Round Visual Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large vision-language models have achieved remarkable progress in visual reasoning, yet most existing systems rely on single-step or text-only reasoning, limiting their ability to iteratively refine understanding across multiple visual contexts.
To address this limitation, we introduce a new multi-round visual reasoning benchmark with training and test sets spanning both detection and segmentation tasks, enabling systematic evaluation under iterative reasoning scenarios.
We further propose RegionReasoner, a reinforcement learning framework that enforces grounded reasoning by requiring each reasoning trace to explicitly cite the corresponding reference bounding boxes, while maintaining semantic coherence via a global–local consistency reward.
This reward extracts key objects and nouns from both global scene captions and region-level captions, aligning them with the reasoning trace to ensure consistency across reasoning steps.
RegionReasoner is optimized with structured rewards combining grounding fidelity and global–local semantic alignment.
Experiments on detection and segmentation tasks show that RegionReasoner-7B, together with our newly introduced benchmark RegionDial-Bench, considerably improves multi-round reasoning accuracy, spatial grounding precision, and global–local consistency, establishing a strong baseline for this emerging research direction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RegionReasoner, an RL-trained vision–language policy that emits structured per-turn trajectories for multi-round, region-grounded visual reasoning. 
Two novel reward components are proposed: (1) an explicit reference-citation reward that forces <think> to verbatim-cite bbox coordinates and penalizes hallucinated citations, and (2) a global–local semantic consistency reward that aligns keywords across <scene>, <focus>, and <think>. 
The authors also present RegionDial-Bench, a multi-turn benchmark built from RefCOCO+/RefCOCOg, and show that RegionReasoner-7B improves multi-turn detection and segmentation metrics, especially at later turns.

### Strengths
1. Reward design — The reference-citation and global–local consistency rewards are intuitive, easily implementable, and well tied to the structured output format. They provide fine-grained shaping for intermediate (reasoning trace) tokens rather than only final outputs.
2. Results in Tables 1 and 2 show consistent improvements, and the claim that improvements compound at later turns (robustness to error accumulation) is supported by both quantitative and qualitative examples.

### Weaknesses
1. The authors state that test dialogues reformulate later queries to explicitly cite bounding boxes predicted in earlier rounds. If the test references are model-predicted boxes (rather than strictly ground truth), the evaluation can be sensitive to the upstream model used to generate them. This raises two issues:  inconsistent comparison if different baselines consume different predicted references, and potential leakage effects. Please clarify exactly how test references are created and ensure all methods receive the same predicted references (or show oracle vs. predicted-reference performance). 

2. The global–local consistency reward depends on a handcrafted lightweight keyword extractor + lemmatizer + noun filter. This may be brittle: paraphrases, synonyms, pronouns, coreference, or longer expressions are likely missed. More importantly, if baselines do not produce structured <scene>/<focus> text, how is the comparison fair? Forcing <think> to repeat the same noun form may advantage RL-trained models.

### Questions
1. How about the ablations with different reward weight hyperparameters $\alpha$, $\beta$?

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
This paper extends a previous work VisionReasoner by adapting to the multi-round setting. They present RegionReasoner, which is a reinforcement learning framework that uses SegLLM to bring multi-round interactions. To validate RegionReasoner, the authors also introduce a new benchmark RegionDial-Bench, which is designed to test the multi-round reasoning ability. The main tasks focus on detection and segmentation. In each round of reasoning, the model can refer to information such as box coordinates of previous rounds, and thus provide more grounded reasoning. In each round, RegionReasoner generates a structured text action includes scene, focus, think and answer, and the memory is updated accordingly. RegionReasoner forces the reasoning to cite evidence to reduce the hallucination by adding the reference citation reward. RegionReasoner-7B outperforms VisionReasoner-7B and other VLMs such as QwenVL-7B in multi-round detection and segmentation tasks on RegionDial-Bench.

### Strengths
(1) RegionReasoner extends a strong previous single-round model VisionReasoner and adapts to the challenging multi-round setting. Results on the proposed benchmark show the validity of RegionReasoner. 

(2) The benchmark itself can be used later form multi-round vision reasoning studies. The motivation of referring to object locations is direct and clear.

### Weaknesses
(1) The paper claims "RegionReasoner consistently outperforms strong Vision-Language Models and task-specific baselines on both referring segmentation and detection.". Previous benchmarks focus on single-round detection/segmentation, but in the main table 1 and table 2, the results are shown on the proposed multi-round benchmark. I think it would be reasonable to add the table to show some "task-specific baselines" for the previous single-round benchmarks. 

(2) Also, the proposed benchmark uses RefCOCO+ and RefCOCOg, but there are also other benchmarks such as MSCOCO and Visual Genome, which are diverse and have boxes and segmentations masks. Have the authors tried to use other datasets to construct the benchmark? Why RefCOCO+ and RefCOCOg are selected here?

### Questions
Some of the figures should be polished. For example, the text in Figure 2 is not clear enough when zooming in.

### Soundness
3

### Presentation
2

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
This paper presents a multi-round visual reasoning benchmark for detection and segmentation. They propose a grounded reasoning method, RegionReasoner, which incorporates reinforcement learning and a global-local consistency reward to enhance semantic coherence. On RegionDial-Bench, the proposed method achieves improvement compared to other VLMs, especially in the later turns.

### Strengths
- This paper presents an interesting reasoning task that integrates QA, referring expression in a multi-turn manner. 
- They propose new reward functions for the new task. They propose a global-local consistency reward to align keywords from the global and local context.

### Weaknesses
- The way they expand the referring expression to multiple turns is confusing and may not be natural. In Appendix B, they illustrate how to simply use a preposition + bbox coordinates in the later turns. A natural referring expression considers the composition between objects. However, in the qualitative examples, they have more complicated and natural questions, such as "Which slice of pizza is R1 about to eat"? "Who is the person next to R1"? They mention that those GPT-style questions used in the previous paper may hallucinate, but it is unclear how they convert the question to this.
- The task setting may not be challenging enough or fair. 1) If the latter turn is based on the ground-truth previous turn (as Appendix B), then the task is essentially a regular single-turn QA, which is not novel. 2) If the latter turn is based on the previous turn, then the reason other models can not achieve good performance is that they are not trained on these templates. If we feed the ground-truth in the question, RegionReasoner may not perform much better than previous methods. It would be nice to see the comparison with the provided ground-truth of the previous step object.
- It is unclear if the new training data affects the performance on standard REC and RES benchmarks.

### Questions
- How did you generate the questions, or did you use the templates in Table 5?
- In the later turns, do you provide the ground truth box of the previous object? 
- Could you compare your methods on standard REC and RES benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the problem of grounding visual referents in multi-turn dialogues for vision-language models (VLMs). They introduce RegionDial-Bench, a benchmark for evaluating multi-round question-answering where each response must be grounded in a specific object instance within the image, annotated via bounding boxes. Alongside the benchmark, they propose RegionReasoner-7B, a model trained using a GRPO-based reinforcement learning approach. The reward function incorporates three key objectives: correctness of the object grounding, global-local semantic consistency, and answer accuracy. Experimental results on RegionDial-Bench demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper introduces RegionDial-Bench, a new benchmark designed to study multi-round conversational reasoning in VLMs, with a specific focus on the groundedness of evidential objects in each dialogue turn.

2. The authors propose a GRPO-based training framework that rewards models for accurate object grounding, global-local semantic consistency, and answer correctness. Experimental results demonstrate the effectiveness of their resulting model, RegionReasoner, on the proposed benchmark.

### Weaknesses
1. The creation process of RegionDial-Benchmark, which constitutes a major contribution of this work, is not sufficiently detailed in the paper. The authors should include a clear description of the benchmark construction methodology， such as data sources, annotation protocols, and key statistics (e.g., number of dialogues, turns, and object categories)，to  facilitate wider adoption.

2. The evaluation of RegionReasoner is currently limited to the proposed RegionDial-Bench. To better assess the generalizability of the method, it is important to also report performance on established benchmarks such as V*. Without such results, it remains unclear whether the improvements are specific to the proposed benchmark or reflect broader applicability.

3. The multi-round conversation results in Table 1 are notably lower than those in the single-round setting, which appears strange. Furthermore, the result for RefCOCOg Multi-turn (R6) stands out as an outlier, being significantly higher than those of R5 and R7. These inconsistencies warrant further analysis and explanation.

4. As shown in Table 3, the model consistently performs better in single-round settings compared to multi-round scenarios across multiple metrics. This recurring pattern suggests a systematic challenge in handling multi-turn grounded dialogues, which should be explicitly discussed in the paper.

### Questions
1. How was RegionDial-Bench constructed? should detail the data sources, annotation protocols, and key statistics.

2. Does RegionReasoner generalize to other VQA benchmarks beyond RegionDial-Bench? Evaluation on established benchmarks (e.g., V*) is needed to verify its broader applicability.

3. Why does multi-round conversation performance consistently lag behind single-round? Furthermore, what explains the outlier for RefCOCOg Multi-turn (R6) in Table 1?

### Soundness
2

### Presentation
2

### Contribution
2
