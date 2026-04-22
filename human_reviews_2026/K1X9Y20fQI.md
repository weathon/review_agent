# PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
With the recent fast development of generative models, instruction-based image editing shows great potential in generating high-quality images. However, the quality of editing highly depends on a carefully designed instruction, placing the burden of task decomposition and sequencing entirely on the user. To achieve autonomous image editing, we present PhotoAgent, a system that rethinks the paradigm of autonomous image editing. PhotoAgent autonomously reasons about the necessary edits, generates a robust action plan, and executes adjustments through a closed-loop process, all without requiring detailed step-by-step prompt engineering from the user. Our approach integrates large language models for intentional reasoning and dynamic planning, alongside a vision-language model for precise localized editing. This combination allows PhotoAgent to interpret users' aesthetic intents, decompose them into executable sub-tasks, and refine the output iteratively based on visual feedback. Extensive experiments demonstrate that PhotoAgent significantly outperforms existing methods in both instruction faithfulness and visual quality across a diverse range of editing scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PhotoAgent, an autonomous system for closed-loop, reasoning-based image editing. The method reframes image enhancement as a sequential decision-making problem, integrating four modules. The authors also introduce UGC-Edit, a user-generated-content (UGC) photo preference dataset, built from LAION and RealQA with LLM-based filtering and human verification. Experiments compare PhotoAgent against single-step and agent-based baselines across multiple metrics. PhotoAgent achieves superior quantitative and qualitative results.

### Strengths
- Clear motivation and framework: The paper articulates the gap between user-driven and agentic editing paradigms, emphasizing the bottleneck of user-in-the-loop instruction decomposition. It convincingly positions PhotoAgent as a next step in computational photography by automating perception, reasoning, and actuation through a closed-loop design.
- Innovative integration of MCTS planning with vision-language reasoning: The combination of LLaVA-based perception, MCTS exploration, and learned reward evaluation represents a principled synthesis of planning and generation.

### Weaknesses
- Dataset diversity and sensitivity underexplored: Although the UGC-Edit dataset is claimed to contain “7k real photos,” it is largely filtered from LAION and RealQA, both web-based and English-dominant. There is no quantitative breakdown of geographic, cultural, or device diversity, nor any audit of sensitive attributes (e.g., faces, regions, or socioeconomic bias). Given that “visual aesthetics” vary culturally, the current dataset risks encoding narrow stylistic norms. No sensitivity analysis examines whether the reward model penalizes unconventional or non-Western aesthetics.
- Computational cost and efficiency trade-offs: The reported runtime (~580 s per image) suggests substantial overhead. Although justified by exploration depth, there is no efficiency–quality curve, and comparisons against lighter hierarchical planners or speculative rollouts are missing. For a claimed “autonomous system,” such runtime may hinder real-world usability.
- Evaluation metrics limited to aesthetic alignment: The study emphasizes perceptual quality (CLIP, Laion-Reward, BRISQUE) but lacks diversity in task scenarios. Real editing often includes semantic modification (object addition/removal, spatial relayout). The benchmarks limit the generality of the conclusions.
- Evaluator–planner coupling and circularity concerns: The learned reward model both trains on and evaluates UGC-style content. While ablations remove the evaluator to show degradation, there is no external validation on unseen evaluators or human-in-the-loop scoring beyond the correlation table. PhotoAgent may optimize toward its own learned aesthetic metric rather than general human preferences

### Questions
- Is the learned evaluator robust to out-of-distribution edits or artistic styles beyond photographic realism?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
1. PhotoAgent is introduced as an autonomous image editing system that reframes image enhancement as a sequential decision-making process, reducing reliance on detailed step-by-step prompt engineering from users.

2. The framework integrates large language models for intentional reasoning and dynamic planning with vision-language models for precise localized editing, forming a closed-loop process that iteratively refines outputs based on visual feedback.

3. Its workflow involves a Perception & Planner to propose candidate actions, MCTS search to explore and prune editing sequences, an Executor to apply the selected actions, and an Evaluator to score intermediate results and trigger re-planning when necessary, enabling robust multi-step image editing.

4. The authors further contribute the UGC-Edit preference dataset, a large-scale collection of multi-step editing tasks based on real user photos, along with a reward model trained on this dataset for evaluating visual quality.

5. Extensive experiments show that PhotoAgent outperforms existing methods in semantic faithfulness and visual quality across diverse editing scenarios.

### Strengths
1. The paper introduces a clear reframing of image editing as a sequential decision-making problem, reducing reliance on detailed step-by-step prompt engineering and enabling autonomous exploratory visual aesthetic planning.

2.The closed-loop integration of large language models for intentional reasoning and dynamic planning with vision-language models for precise localized editing is well-motivated and convincingly presented as a holistic pipeline.

3. The system demonstrates strong empirical performance: PhotoAgent consistently outperforms baselines across semantic alignment and non-reference aesthetic quality metrics such as CLIP Similarity, LPIPS, BRISQUE, Laion-Reward, and UGC Score.

4. The introduction of the UGC-Edit preference dataset and reward model is a valuable contribution, providing a benchmark and evaluation mechanism that better captures user-generated content preferences.

### Weaknesses
1. While the system design is interesting, the paper leans more towards an engineering integration of existing models rather than introducing fundamentally novel algorithmic insights. This may limit its perceived contribution.

2. The evaluation, although comprehensive in metrics, could be strengthened with more diverse baselines or user studies in real-world editing scenarios to better validate practical utility and robustness.

3. Some technical descriptions, such as the role of the “controller” (in the caption of Fig. 1) remain under-specified. Clarifying these components would improve the reproducibility and clarity of the work.

### Questions
1. Could the authors clarify how their approach differs from prior work that also leverages large vision models for aesthetic evaluation and editing? At present, the novelty relative to earlier systems is not fully clear.

2. The paper briefly mentions a “controller” (in the caption of Fig. 1) that uses MCTS, but this concept is only introduced once and not elaborated further. Could the authors provide more details on the controller’s role and implementation?

3. How well does the proposed framework generalize to broader editing tasks or domains beyond the photo scenarios tested? It would be helpful to understand potential limitations in applicability.

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
5

### Summary
This paper presents PhotoAgent, an autonomous image editing system that addresses the limitation of current instruction-based editing models requiring detailed, expert-level prompts. The system employs a closed-loop architecture consisting of four components: (1) a VLM-based perceiver that generates candidate editing actions, (2) an MCTS-based planner that explores multi-step editing trajectories, (3) an executor that applies edits using both classical and generative tools, and (4) an evaluator powered by a custom reward model. To support this work, the authors construct the UGC-Edit dataset containing 7K+ user-generated content images with editing intents, and train a Qwen2.5-VL-based reward model via Group Relative Policy Optimization (GRPO) to align with human aesthetic preferences. Experiments demonstrate that PhotoAgent outperforms single-step baselines and other agent-based approaches on semantic alignment and aesthetic quality metrics when given ambiguous editing instructions like "make this image better."

### Strengths
1、The UGC-Edit dataset addresses a gap in existing benchmarks by focusing on realistic user photo editing scenarios. The construction process with LLM-based filtering and human verification appears rigorous.

2、Training a specialized reward model via GRPO that achieves strong correlation with human judgments is a good contribution. The validation against baseline metrics demonstrates its superiority in capturing aesthetic preferences.

3、 The paper includes both quantitative metrics (CLIP, BRISQUE, LPIPS, Laion-Reward, UGC Score) and qualitative comparisons across diverse baselines, with thorough ablation studies validating key design choices.

### Weaknesses
1、 While the system integration is effective, the individual components (VLM perception, MCTS planning, reward model training via GRPO) are relatively standard techniques.

2、 The paper mentions computational expense as a limitation but provides no concrete analysis of runtime, cost per image, or number of MCTS simulations needed. Given the complexity of the approach (MCTS with generative model rollouts), scalability and practical deployment concerns are significant but unexplored.

3、The lack of quantitative comparisons with relevant work is notable. While the authors mention recent studies using agentic models for image editing, they appear to have omitted quantitative metric comparisons. Could the authors briefly compare their approach with existing works such as JarvisArt[1] and MonetGPT[2]?

[1] [NeurIPS' 2025] JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent
[2] SIGGRAPH 2025 MonetGPT: Solving Puzzles Enhances MLLMs' Image Retouching Skills

### Questions
see Weaknesses

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
This paper addresses the pain point that existing instruction-based image editing models heavily rely on users to design precise, multi-step instructions. It proposes an autonomous image editing agent framework called PhotoAgent. The system aims to simulate the decision-making process of a human expert, automatically performing complex image aesthetic enhancement tasks without requiring detailed step-by-step prompts from the user. At its core is a closed-loop "perceive-plan-execute-evaluate" process. The paper also contributes a new dataset called the UGC-Edit Preference Dataset and trains a new reward model (UGC Evaluator) based on it. Experiments show this model has a very high correlation with human preferences.

### Strengths
This paper redefines image editing as an "exploratory visual aesthetic planning" problem. By introducing forward-looking planning through the MCTS algorithm , the agent can evaluate the long-term impact of multi-step editing sequences. This avoids the local optima or "short-sighted" decisions that greedy methods (like ReAct (Closed-loop)) are prone to.

The paper emphasizes the central role of the evaluator in a closed-loop system. Instead of relying on existing general-purpose aesthetic models that perform poorly on UGC data , the authors built the UGC-Edit dataset and trained a specialized reward model. Table 2 demonstrates that this evaluator (PLCC 0.8300) aligns far better with human preferences than existing models (e.g., Laion-aesthetic, PLCC 0.5567).


The experimental design compares against comprehensive baselines, including non-agent SOTA models (GPT-4o, Flux.1) and agent-based models (HuggingGPT, ReAct) . Given a vague instruction, PhotoAgent achieves the best results across all metrics (Table 1), robustly proving the superiority of the proposed framework. Ablation studies also clearly validate the necessity of both MCTS and the new evaluator .

### Weaknesses
- The most significant weakness of this paper is its extremely high computational latency. Section 5.5 mentions that under the standard configuration, the average processing time per image is approximately 580 seconds. This high time cost makes the system currently unsuitable for any practical or interactive applications, rendering it more of a "proof-of-concept."

- To make MCTS feasible, the planner relies on "reduced-resolution processing" in a "fast-approximation environment". A potential issue is that an edit that looks "good" at low resolution (i.e., high simulated reward) may not be so when executed at full resolution (e.g., it might introduce artifacts not visible at low resolution). This "sim-to-real" gap could mislead the MCTS planner.

- The UGC-Edit dataset (7k images)  is a good contribution, but its scale is relatively limited. The evaluation test set is also small (89 images). It remains to be seen whether this reward model can generalize well to broader and more diverse UGC images (e.g., different cultures, lighting conditions, or photographic styles).

### Questions
1. **Regarding computational cost:** In the MCTS planning phase, is the main computational bottleneck the "simulation" stage in the low-resolution approximate environment, or the "evaluation" stage when calling the reward model? Are there any timing statistics for the different stages to identify the performance bottleneck?

2. **Sim-to-Real Gap:** MCTS simulates at low resolution , while the executor operates at full resolution . Is there a significant reward difference between these two? Does the system frequently encounter situations where MCTS simulation predicts a high reward for an action sequence, but after full-resolution execution, the Evaluator gives a low score, causing the action to be rolled back?

3. **Regarding the stopping criterion:** The system may make unnecessary modifications to already high-quality images. Does this imply that the evaluator struggles to assign a saturated score for "perfect" images, causing the agent to always want to "do something more"? Is there an explicit early termination mechanism in the system? For example, does the system automatically stop when the evaluator's score reaches a certain threshold or when it fails to improve the score for several consecutive iterations?

### Soundness
3

### Presentation
4

### Contribution
4
