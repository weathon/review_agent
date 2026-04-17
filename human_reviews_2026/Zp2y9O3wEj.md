# Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Recent advances in large multimodal models have leveraged image-based tools with reinforcement learning to tackle visual problems. However, existing open-source approaches often exhibit monotonous reasoning patterns and allow only a limited number of interaction turns, making them inadequate for difficult tasks that require trial-and-error exploration. In this work, we address this limitation by scaling up tool-based interactions and introduce Mini-o3, a system that executes deep, multi-turn reasoning—spanning tens of steps—and achieves state-of-the-art performance on challenging visual search tasks. Our recipe for reproducing OpenAI o3–style behaviors comprises three key components. First, we construct the Visual Probe Dataset, a collection of thousands of challenging visual search problems designed for exploratory reasoning. Second, we develop an iterative data collection pipeline to obtain cold-start trajectories that exhibit diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance. Third, we propose an over-turn masking strategy that prevents penalization of over-turn responses (those that hit the maximum number of turns) during reinforcement learning, thereby balancing training-time efficiency with test-time scalability. Despite training with an upper bound of only six interaction turns, our model generates trajectories that naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases. Extensive experiments demonstrate that Mini-o3 produces rich reasoning patterns and deep thinking paths, effectively solving challenging visual search problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Mini-03, a vision–language model designed for deep, multi-turn visual reasoning in complex visual search tasks. The authors propose a three-part training recipe: 

- VisualProbe Dataset – A challenging benchmark with high-resolution images, small targets, and distractors to necessitate iterative reasoning. 

- Cold-start Data Pipeline – An iterative synthesis process generating ~6,000 multi-turn trajectories from a few exemplars for supervised fine-tuning. 

- Over-turn Masking – A reinforcement learning modification to GRPO that prevents penalization of trajectories exceeding the turn limit, enabling test-time scaling. 

Built on Qwen2.5-VL-7B-Instruct, Mini-03 demonstrates emergent test-time turn scaling: trained with 6 turns, accuracy improves up to 32 turns at inference. It achieves state-of-the-art performance on VisualProbe-Hard (48.0% vs. 35.1% for DeepEyes) and competitive results on V*Bench and HR-Bench. Ablations confirm the necessity of cold-start SFT and over-turn masking.

### Strengths
- A reproducible recipe combining dataset, data generation pipeline, and RL. 

- Over-turn masking is simple yet effective, unlocking test-time scaling—a property with implications beyond vision tasks. 

- Promising results across benchmarks, systematic ablations, and insightful analysis of resolution vs. interaction depth trade-offs. 

- VisualProbe fills a gap for evaluating exploratory reasoning.

### Weaknesses
- The paper does not explain why over-turn masking enables scaling—mode collapse prevention or exploration diversity? 

- Relies on a teacher VLM for trajectory synthesis; sensitivity to exemplar quality is unexplored. 

- Reward computation details (model choice, prompt, bias analysis) are missing -- reproducibility concerns. 

- Focused on visual search; generalization to other reasoning domains (math, scientific figures) is untested – narrow scope. 

- Systematic study of error patterns or robustness to adversarial perturbations would strengthen the paper.

### Questions
- How sensitive is performance to the choice and diversity of cold-start exemplars? 

- Which LLM was used as the reward judge, and how was its reliability validated? 

- Can you provide theoretical or empirical analysis of why over-turn masking enables test-time scaling? 

- Does the approach generalize to other reasoning domains beyond visual search? 

- What are the failure modes—e.g., repetitive loops, grounding errors, or inability to backtrack?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose Mini-o3, a method capable of performing deep, multi-turn reasoning and achieving state-of-the-art performance on challenging visual search tasks.
They constructed a multi-turn visual reasoning trajectory dataset and introduced an over-turn masking strategy to balance training-time efficiency with test-time scalability.

### Strengths
The main innovation lies in the proposal of an over-turn strategy. In addition, it demonstrate that although only up to 6-round data  are used during training, the accuracy consistently improves as the upper bound on interaction turns increases from 4 to 32 during inference.

### Weaknesses
The setting of the 6-round budget has not been proven to be optimal.

### Questions
Have you tried testing the performance of different sizes of QWEN-VL-2.5?

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
3

### Summary
This paper presents **Mini-o3**, an open-source vision-language agent designed to perform **deep multi-turn reasoning** for visual search tasks. While prior open-source VLMs (e.g., DeepEyes, Chain-of-Focus) often exhibit shallow, repetitive reasoning with limited tool-use turns, Mini-o3 demonstrates reasoning trajectories spanning tens of steps. The authors propose a three-component recipe:

1. **VisualProbe dataset**—a new benchmark of high-resolution, hard visual-search problems requiring trial-and-error exploration;
2. **Iterative cold-start data synthesis**—an in-context prompting method for generating diverse reasoning trajectories for supervised fine-tuning;
3. **Over-turn masking**—a reinforcement-learning modification that avoids penalizing long, incomplete trajectories, enabling scalability beyond the training turn limit.

Trained with a 6-turn cap, Mini-o3 generalizes to much longer inference trajectories (up to 32 turns), achieving **state-of-the-art accuracy** on VisualProbe, V* Bench, HR-Bench, and MME-Realworld benchmarks. The model shows more diverse reasoning strategies and deeper “thinking-with-images” patterns than prior open-source agents.

### Strengths
* Addresses a **clear and under-explored gap**: enabling deep visual reasoning in open-source VLMs.
* **Elegant and reproducible training recipe** with quantitative and qualitative validation.
* **Comprehensive experiments** (four benchmarks + ablations).
* Demonstrates **turn-scaling property**—a rare capability among existing agents.
* Good ethical and reproducibility practice; code/data promised for release.

### Weaknesses
- [Reward Modeling] The paper employs an external LLM as a semantic judge for reinforcement learning but provides few details about its consistency, inter-run variance, or bias. It is unclear how reward noise affects policy stability or whether calibration against human-verified scores was attempted.
- [Ablation] While ablations exist, they focus on quantitative accuracy. The paper could include deeper analyses of emergent reasoning behaviors—e.g., trajectory diversity, failure patterns, or qualitative diagnostics showing how reasoning depth relates to success rates.

### Questions
Please see weaknesses above.

1. How sensitive is over-turn masking to the masking threshold or turn budget?
2. Do long-turn trajectories ever degenerate into loops or repetitive zooming?
3. How well would Mini-o3 scale with a larger base model?

Minor Grammar & Typo:

Line 014 - 015: “We address this limitation by scaling up tool-based interactions and introduce Mini-o3…” should be "introducing”

Line 018: “OpenAI o3–style behaviors” should be a regular hyphen

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Mini‑o3 is a multi‑turn visual search system that scales reasoning to tens of steps by combining a new Visual Probe dataset, an iterative cold‑start trajectory pipeline, and an over‑turn masking RL strategy that avoids penalizing responses hitting the turn limit. Trained with only six turns, it generalizes to longer test‑time trajectories where accuracy rises as the allowed turns increase, achieving state‑of‑the‑art results on visual search benchmarks.

### Strengths
- Mini-o3 achieves state-of-the-art performance on several visual search benchmarks, clearly outperforming previous open models. The improvement is from allowing the model to perform more reasoning turns. The over-turn masking method improves long-turn reasoning while keeping training efficient.
- The new Visual Probe dataset introduces difficult visual search tasks. Mini-o3 trained on Visual Probe has demonstrated some sort of trial-and-error reasoning.

### Weaknesses
- Missing details
    - The paper relies on an LLM judge for rewards, however the exact judging prompts/criteria are not fully specified in the main text.
    - The dataset difficulty splits (easy/medium/hard) are introduced and used in results but their criteria are only briefly mentioned. I would suggest the author to add more details on how the data split are decided.
    - Similarly, the cold‑start trajectory generation details (for example, exact prompts, hand-crafted in-context examples, acceptance filters) should be included in the main text or appendix.
- I wonder whether the authors have considered some other variants for over-turn masking
    - Mask only context-overflow instead of turn-overflow
    - Soft penalty instead of zero-mask for over-turn content
- Missing ablation of removing the DeepEyes portion or changing the ratio between VisualProbe and DeepEyes from the training to see how much the mixed-data balance matters.

### Questions
- Interesting that ChartQA performance dropped after the RL training compared to Qwen2.5-VL. I am curious about what the authors think might be the reason? Is it because of the data domain used for RL training?
- Curious how much improvements on general perception benchmarks, like BLINK, and CVBench. and also spatial reasoning benchmarks, like SAT.

### Soundness
2

### Presentation
2

### Contribution
2
