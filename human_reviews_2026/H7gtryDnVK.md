# No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
Visual reasoning is challenging, requiring both precise object grounding and understanding complex spatial relationships. Existing methods fall into two camps: language-only chain-of-thought approaches, which demand large-scale (image, query, answer) supervision, and program-synthesis approaches which use pre-trained models and avoid training, but suffer from flawed logic and erroneous grounding. We propose an annotation-free training framework that improves both reasoning and grounding. Our framework uses AI-powered verifiers: an LLM verifier refines LLM reasoning via reinforcement learning, while a VLM verifier strengthens visual grounding through automated hard-negative mining, eliminating the need for ground truth labels. This design combines the strengths of modern AI systems: advanced language-only reasoning models for decomposing spatial queries into simpler subtasks, and strong vision specialist models improved via performant VLM critics. We evaluate our approach across diverse spatial reasoning tasks, and show that our method improves visual reasoning and surpasses open-source and proprietary models, while with our improved visual grounding model we further outperform recent text-only visual reasoning methods. Project webpage: https://glab-caltech.github.io/valor/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work applies multimodal-based verifiers in form of tool-use in order to improve the visual reasoning capabilities of a base model. These verifiers are used to finetune the systems program and visual grounding modules. The final approach is evaluated on VQA tasks and compared to monolithic as well as program synthesis based models.

### Strengths
The experimental results across a diverse set of datasets appear promising.

### Weaknesses
General Comments:
I find the paper's overall narrative and key contributions difficult to follow. Below, I outline specific areas that require clarification.

Conceptual Clarity:
	1.	Method characterization: The authors describe their approach as "label-free finetuning via pretrained model verifiers." Could the authors clarify how this fundamentally differs from knowledge distillation from the verifiers into the base model(s)? A clearer distinction would help readers understand the novelty of the approach.
	2.	Code execution mechanism: Section/Figure 2 shows images being passed to an executor, but the technical details of how Python code is actually executed over the image remain unclear. Specifically, what representation is used for this? The current presentation makes this process appear overly simplistic or unclear.
	3.	Vision specialist integration: The paper would benefit from a clearer explanation of how vision specialist models are integrated into the overall system architecture.

Presentation Issues:
Figure readability: Most figures contain text boxes that are too small to read, severely limiting their utility. If the information in these figures is essential for understanding the authors' arguments and results, they must be made readable. Currently, I cannot verify the claims or follow the clarifications the authors make in reference to these figures.

Experimental Design:
	1.	Evaluation objectives: For several experiments, the specific goal and relevance to the main claims are unclear. I recommend adding brief statements at the beginning of each evaluation section explicitly stating what research question is being addressed.
	2.	VALOR vs. VALOR (RL): The distinction between these two variants is not clearly explained in the text. Please provide a concise description of the key differences.
	3.	Verifier and base model specification: Critical information about which models serve as verifiers versus base models appears to be missing (or I may have overlooked it). This should be clearly indicated in the results tables alongside the VALOR specifications, as the choice of verifier likely has substantial impact on performance.

Critical Concern - Experimental Comparisons:
I find the experimental comparisons difficult to interpret due to potentially confounded variables. For example, when comparing VALOR with Qwen3-8B:
	•	Was Qwen3-8B also finetuned in any way?
	•	Was Qwen3-8B used as a verifier in VALOR's training?
	•	The approaches appear fundamentally different, with VALOR receiving substantially more information, possibly larger models as verifiers and additional training. Doesn't this alone explain the performance differences?
Without clarity on these points, it's difficult to assess whether the comparisons are fair or what specific conclusions can be drawn.

Recommendations:
	1.	Ablation study on verifiers: Given the central role of verifiers in the method, an ablation study examining how different verifiers affect performance would be valuable. This would help isolate the contribution of the VALOR framework from the contribution of powerful verifiers.
	2.	Clarify contributions: Each evaluation section would benefit from a concluding paragraph that: (a) summarizes the key findings, (b) relates them back to the paper's main claims, and (c) explains their significance for the overall contribution.
	3.	Sharpen claims: Throughout the paper, the goals and contributions could be stated more crisply and specifically, making it easier for readers to follow the logical thread from motivation through methods to conclusions.

Overall Assessment:
Due to the issues outlined above, I find it challenging to fully evaluate the validity of the authors' claims. Addressing these concerns would significantly strengthen the paper.

### Questions
It seems VALOR-RL performs worse than VALOR in nearly all evaluations. Whats the intuition behind this? Is this just an error in the training set generation?

VALOR vs proprietary models: I'm a little confused what the benefit is of comparing to different propietary models, couldn't VALOR be applied on top of these as well? 

Why were different dataset subsets used across the different evaluations?

What is the difference between evaluations in section 4.4 and 4.1?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the challenging problem of query-based visual reasoning, which requires both accurate object grounding and understanding of complex spatial relationships. The authors propose an annotation-free training framework that integrates two AI-powered verifiers, an LLM verifier that refines reasoning via reinforcement learning, and a VLM verifier that strengthens grounding through hard-negative mining. The approach is evaluated on a variety of spatial reasoning tasks, where it demonstrates superior reasoning and grounding performance compared to both open-source and proprietary baselines.

### Strengths
- The paper is well written and clearly structured. It effectively presents both the high-level motivation and the technical components of the method, with helpful examples that aid comprehension.   
- It tackles an important and timely problem in visual reasoning, particularly the issue of grounding, which remains a key bottleneck for reliable program-based reasoning systems.
- The annotation-free design is compelling and addresses an important challenge in scaling visual reasoning systems.

### Weaknesses
- My only concern lies in the VLM verifier quality. Since the proposed method uses the VLM to generate the training data for grounding, the VLM may itself produce imperfect outputs. Do you think the fine-tuned model can ever outperform the VLM that labeled the data? Did you compare VALOR against GPT-5-mini at some point, as in Table 4? It might be helpful to discuss this topic more explicitly in the paper. For example, should the long-term strategy be to continually use the strongest available VLMs for data generation, or are there other potential directions?

### Questions
1. When prompting the LLMs and VLMs, which generation strategy did you use, e.g. greedy generation? For proprietary models, did you use default parameters, or did you adjust reasoning settings (e.g., reasoning effort in GPT-5-mini)?
2. In Table 3, you compare results on only two benchmarks. Is there a specific reason for this selection?
3. In Figure 1, you mention "GPT-5-Thinking", presumably the model option from the OpenAI web app? It might be good to clarify which model these terms refer to (as the API only has gpt-5 and gpt-5-mini, etc.)
4. Line 318: "framing VQA as yes/no rather than open-ended", is this referring to example (c) or to something else? I didn't understand the yes/no framing.
5. I like the various examples given in the paper; however, Figure 4 requires quite some zooming. I think it would improve the paper if the number of examples were reduced or the way the small texts are presented were improved.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces VALOR, a verifier-guided, tool-using Visual Language Model (VLM) designed to improve spatial and reasoning-heavy visual understanding tasks without human-annotated labels. The method integrates a language-only model (Qwen3-8B) with structured reinforcement learning through Group Relative Policy Optimization (GRPO) and a multimodal verifier that evaluates generated reasoning programs along six dimensions: logic, spatial, attribute, syntax, adherence, and format. The system further enhances visual grounding by retraining GroundingDINO on verifier-filtered pseudo-labels, improving object detection and spatial relation accuracy. The authors demonstrate strong results across diverse benchmarks such as OMNI3D-BENCH, VSR, and ROBOSPATIAL, outperforming both proprietary and open-source VLMs. The paper provides clean ablations for the reasoning-RL and grounding-improvement stages, showing monotonic performance gains with increasing verifier data. Overall, VALOR presents a unified, label-free approach that aligns structured reasoning, verifier feedback, and improved grounding to achieve significant gains in compositional and spatial visual reasoning.

### Strengths
The following are the strenghts of the paper:

**Originality:** The paper presents an integration of verifier-guided reinforcement learning with explicit tool use for visual reasoning, eliminating the need for labeled supervision. The structured multi-head verifier and verifier-filtered pseudo-label pipeline seem to be novel and effective extensions of prior VLM paradigms.

**Technical Quality:** The paper is empirically strong. The experimental design cleanly isolates the effects of reasoning-level RL and grounding-level retraining, supported by consistent scaling studies. The improvements are large and reproducible across benchmarks.

**Clarity:** The paper is clearly written and logically structured. The pipeline is illustrated with detailed figures and algorithmic descriptions that make both the reasoning and grounding stages transparent. 

**Significance:** VALOR overcomes a key limitation of current VLMs, weakness in spatial and multi-step reasoning, without extra manual annotation. It generalizes across domains and offers a scalable, verifier-driven way to enhance foundation models, making it practical and influential.

### Weaknesses
The following are the weaknesses of the paper:

- The paper introduces a rich multi-head verifier reward but does not ablate the contribution of each component. Since logic, spatial, attribute, syntax, and adherence rewards are argued to fix distinct reasoning errors, omitting a head-wise ablation leaves uncertainty about which rewards are essential versus redundant.

- The three-stage verifier pipeline for generating pseudo-labels in grounding (coarse filter, per-crop verification, deduplication) is central to the claimed improvement, yet there is no quantitative breakdown of each stage’s impact on precision, recall, or downstream accuracy. This weakens the causal link between the pipeline design and the observed performance gains.

- The paper relies on Group Relative Policy Optimization (GRPO) but does not compare it against simpler alternatives such as PPO or supervised fine-tuning on top-rewarded samples. Without this comparison, it is unclear whether the advantage comes from the RL algorithm itself or from the verifier-generated signal.

### Questions
The following are the list of questions:

1. Verifier Reward: Can the authors provide an ablation isolating each reward head (logic, spatial, attribute, syntax, adherence) to show which components most affect reasoning performance?

2. Verifier Pipeline: Could the authors quantify how each stage of the pseudo-labeling pipeline (coarse filter, per-crop check, deduplication) impacts label precision and downstream accuracy?

3. RL Baseline: Can the authors compare GRPO with simpler baselines such as PPO or supervised fine-tuning on top-rewarded samples to confirm that GRPO itself drives the improvement?

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
The paper proposes a novel visual reasoning training paradigm, VALOR, which enables models to improve both reasoning and visual grounding capabilities without any manual annotations. By allowing the model to tackle visual reasoning tasks through tool invocation, VALOR introduces a two-stage training approach: It first refines the reasoning capability of the base LLM via RL with a Python interpreter and an LLM verifier, which strengthens its ability to plan and generate executable code. Then it fine-tunes the visual grounding module (serving as a tool) using a VLM verifier that generates pseudo-labels. Experiments demonstrate that VALOR surpasses both open-source and proprietary models across a broad range of spatial reasoning benchmarks.

### Strengths
- The paper is clearly written and well structured.

- It introduces an innovative training framework that enables the model to jointly improve both itself and the tools it invokes, entirely without human supervision.

- The ablation studies are convincing, showing that verifier-based RL enhances reasoning logic, while verifier-based pseudo-labeling improves visual grounding, with cumulative performance gains when both are combined.

### Weaknesses
- The visual grounding module in VALOR is further fine-tuned using verifier-generated pseudo-labels, while all baselines still rely on the frozen pre-trained detector (Table 1). This makes the comparison with baselines partially unfair, since the tools they are allowed to invoke are in fact not properly aligned.

- The paper lacks a quantitative analysis of verifier errors, which would help assess the reliability of verifier supervision.

### Questions
- The paper shows that both reasoning and grounding performance increase with the number of training samples (Table 5, Fig. 7).
Have the authors investigated how far this scaling trend continues? Does the performance saturate beyond the current dataset size, or does it continue improving at larger scales?

- The paper mentions using GPT-5-mini as the VLM verifier. Could the authors report its standalone performance on the same benchmarks to better understand the verifier's contribution?

- It would be interesting to explore whether the model could serve as its own verifier for self-training, rather than relying on an external verifier.

- Following Weakness 1, I am interested in understanding the isolated impact of post-training the visual grounding module. Compared to VALOR-RL, this stage appears to contribute the larger performance gain, so disentangling its effect would be insightful.

I would be happy to raise my score if the authors could address these questions and clarify the above points.

### Soundness
3

### Presentation
4

### Contribution
3
