# GUI Knowledge Bench: Revealing the Knowledge Gap Behind VLM Failures in GUI Tasks

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Large vision–language models (VLMs) have advanced graphical user interface (GUI) task automation but still lag behind humans. We hypothesize this gap stems from missing core GUI knowledge, which existing training schemes (such as supervised fine-tuning and reinforcement learning) alone cannot fully address. By analyzing common failure patterns in GUI task execution, we distill GUI knowledge into three dimensions: (1) interface perception, knowledge about recognizing widgets and system states; (2) interaction prediction, knowledge about GUI interaction conventions; and (3) instruction understanding, knowledge about procedural knowledge of GUI operations. We further introduce GUI Knowledge Bench, a benchmark with multiple choice and yes/no questions across six platforms (Web, Android, MacOS, Windows, Linux, IOS) and 292 applications. Our evaluation shows that current VLMs identify widget functions but struggle with perceiving system states, predicting actions, and interpreting task goals. Experiments on real world GUI tasks further validate the close link between GUI knowledge and task success. By providing a structured framework for assessing GUI knowledge, our work supports the selection of VLMs with greater potential prior to downstream training and provides insights for building more capable GUI agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates why large vision–language models still struggle with automated GUI tasks despite recent advances. The authors hypothesize that missing core GUI knowledge in current VLMs is a key factor behind their performance gap compared to humans. To address this, the paper makes several notable contributions. First, it defines three dimensions of GUI knowledge based on common failure patterns: (1) interface perception, (2) interaction prediction, and (3) instruction understanding. Using these categories, the authors introduce GUI Knowledge Bench, a comprehensive benchmark composed of multiple-choice and yes/no questions derived from over 40,000 GUI screenshots and 400 execution traces across six platforms and 292 applications. This benchmark systematically evaluates what GUI-relevant knowledge is encoded in VLMs prior to any fine-tuning on task-specific data. The paper’s experiments show that while current VLMs can often recognize basic widget functions, they struggle with perceiving system state, predicting interaction outcomes, and verifying task completion. Importantly, the authors demonstrate a close link between performance on the knowledge benchmark and real GUI task success: models with more encoded GUI knowledge perform better on actual GUI automation tasks, and providing missing knowledge (e.g. in the form of operation plans) significantly improves task execution success. Overall, the paper’s contributions include a novel benchmark dataset for GUI knowledge, an analysis revealing specific knowledge gaps in state-of-the-art VLMs, and empirical evidence that addressing these knowledge gaps can improve GUI task automation.

### Strengths
1. The paper tackles a crucial and timely problem in multimodal AI - why VLM-driven GUI agents often fail in real scenarios. The authors clearly motivate that beyond reasoning and planning, knowledge of GUI specifics is missing in current models.
2. The introduction of a large-scale benchmark with 3483 knowledge-centric questions across 6 operating systems and 292 applications is a significant contribution.
3. The paper’s breakdown of GUI knowledge into three dimensions – interface perception, interaction prediction, and instruction understanding – is well-grounded in observed failure modes and prior literature.
4. The authors evaluate a wide range of state-of-the-art models, including both closed-source and open-source models. The benchmarking results are detailed per knowledge category and sub-task, allowing for nuanced comparisons. This extensive comparison lends credibility to the findings.
5. The results yield clear, actionable insights.
6. A significant strength is the additional experiment bridging the benchmark to real task execution.

### Weaknesses
1. While the paper states that the 3,483 questions were produced via automated generation plus manual annotation, it provides few details on this process. It’s not fully clear how questions and answers were generated or verified for correctness and difficulty.
2. Focus on base VLMs without fine-tuning. The benchmark specifically tests models “prior to downstream training” – i.e., base VLMs without task-specific fine-tuning. This isolates inherent knowledge but also means models are out-of-the-box, not specialized for UI understanding.
3. Evaluation of knowledge integration methods is limited. The paper’s solution implications mainly suggest selecting better base models or augmenting them with knowledge. While it demonstrates that adding operation plans helps one model, it stops short of exploring other ways to inject GUI knowledge (for instance, using the benchmark itself as additional training data or using retrieval during inference). A deeper discussion on concrete strategies (beyond the brief mention of retrieval augmentation) would have been useful to translate the findings into actionable guidance for building improved agents.

### Questions
1. How were the 3483 knowledge questions constructed and validated? The paper mentions a mix of automated generation and manual annotation, but clarification is needed on the process. For example, did you use templates to generate questions from execution trajectories, and what steps ensured that the questions have unambiguous correct answers and appropriate difficulty?
2. The OSWorld case study is great evidence of knowledge helping in one setting. Did you observe (or do you anticipate) a strong correlation between a model’s score on GUI Knowledge Bench and its success rate on various downstream GUI tasks (beyond OSWorld)? For example, if Model A scores 10% higher on the benchmark than Model B, is A consistently better when fine-tuned or evaluated on full task benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GUI Knowledge Bench, a diagnostic benchmark that evaluates vision-language models (VLMs) on GUI-specific knowledge across three dimensions: interface perception, interaction prediction, and instruction understanding. The benchmark spans 6 platforms and 292 applications with 3,483 questions, revealing that current VLMs struggle with system state reasoning, action prediction, and task completion verification despite reasonable performance on basic widget recognition.

### Strengths
1. Comprehensive Coverage: The benchmark's scope is impressive - covering 6 platforms, 292 applications, and over 40,000 screenshots. This diversity enables robust evaluation across different GUI environments.

2. Interesting Empirical Analysis: The evaluation reveals clear patterns - models perform well on widget function recognition but struggle with system states and interaction dynamics. The confusion matrix showing bias toward "click" actions is insightful.

### Weaknesses
1. The benchmark mixes three distinct capabilities: knowledge, perception, and grounding. For example, for some of the problems, it involves clicking on a coordinate which requires strong grounding ability. The mix of various abilities make it harder to understand what causes the deficiency in GUI models.

2. The paper assumes VLMs should possess extensive prior GUI knowledge, but as mentioned by the authors, GUI interfaces are constantly evolving, and have impossible broad coverage. Why should the model possesses knowledge, rather explore and learn by itself in the environment.

3. Poor figure quality and inconsistent fontsizes. Multiple figures (2-5) contain small text.

### Questions
1. Setting mimic real agent trial and error: Does using reflection style (multiple iterations, follow [1] settings) improve accuracy in GUI agent trials?

2. Does correlation exist between your GUI knowledge benchmark and other GUI agent benchmarks, and does possessing this knowledge lead to higher accuracy?

[1] Wang, Xingyao, et al. "Mint: Evaluating llms in multi-turn interaction with tools and language feedback."

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents GUI Knowledge Bench (GUIKB), a diagnostic benchmark designed to analyze knowledge gaps in Vision-Language Models (VLMs) applied to graphical user interface (GUI) understanding and control. The benchmark categorizes GUI-related reasoning into three dimensions Interface Perception, Interaction Prediction, and Instruction Understanding band builds a large-scale dataset spanning 6 operating systems, 292 real-world applications, and over 3,400 questions derived from around 40,000 screenshots and 400 GUI trajectories.

### Strengths
Timely and relevant problem focus It goes beyond measuring overall task success to dissecting the specific types of GUI knowledge involved.

Broad empirical coverage: The benchmark’s diversity across multiple OSes and hundreds of applications is impressive, offering a realistic assessment environment that surpasses previous GUI benchmarks in scope.

Practical community utility: The dataset could serve as a diagnostic suite to evaluate model grounding and reasoning for future GUI or computer-use agents.

### Weaknesses
The main conceptual novelty lies in organizing and scaling previous evaluation ideas. It extends earlier benchmarks (e.g., MMBench-GUI, SeeClick, Web-CogBench) in breadth rather than introducing new forms of reasoning or interaction.

The multiple-choice and yes/no question format, combined with visual hints, simplifies grounding and may allow models to exploit linguistic or positional biases instead of demonstrating genuine understanding. Free form questions (going beyond 25% random performance) could be richer.

The OSWorld improvement experiment is intriguing but small in scope and lacks variance or ablation analysis, making the link between benchmark performance and real agent success suggestive rather than proven. The paper seems to lack deeper analysis which can give insights on what current agents lack in terms of decision making.

### Questions
Have you tried a free-form response setting (without multiple-choice cues) to confirm that models truly possess GUI knowledge rather than exploiting format bias?

Could you share dataset composition metrics (OS/app balance, interface types, redundancy rate) to verify diversity and coverage?

I am not very confident accepting this paper, decision subject to rebuttal.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper posits that the performance gap between Vision-Language Models (VLMs) and humans in GUI automation stems from a core knowledge deficit, rather than failures in reasoning or planning alone. The authors introduce GUI Knowledge Bench, a diagnostic benchmark designed to evaluate this specific knowledge gap. The benchmark is structured around three dimensions derived from common agent failure modes: Interface Perception (recognizing widgets, layouts, and system states), Interaction Prediction (predicting action types and their effects), and Instruction Understanding (planning and verifying task completion).

### Strengths
1. The primary strength is the conceptual shift in evaluation. Instead of focusing on end-to-end task success, which conflates multiple agent capabilities (e.g., reasoning, planning, knowledge, grounding)
2. The benchmark is comprehensive, covering 6 platforms and 292 applications, which ensures the findings are generalizable.

### Weaknesses
1. Conflation of "Knowledge" and "Reasoning" in Benchmark: The paper's primary claim is to evaluate GUI "knowledge" (”Different from most existing benchmarks that primarily evaluate task success, which mainly focus on the grounding, reasoning, and planning”) as distinct from "reasoning" and "planning." However, several tasks within the benchmark, particularly in the "Ins Understanding" (e.g., Task Planning) and "Interaction Prediction" dimensions, appear to inherently require complex reasoning or logical capability of LMs.
2. There is a contradiction in the paper's terminology. The authors claim to test 'base VLMs,' but the models listed are clearly instruction-tuned, optimized, and/or RLHF'd (e.g., 'GPT-5-Chat'). 'UITARS-1.5-7B' is itself a fine-tuned GUI agent (sft/rled based on qwen2.5vl). This invalidates the premise of testing 'base' models.
3. The OSWorld validation in Sec 4.4 provides evidence that providing plans improves agent performance. This validates the 'Instruction Understanding' gap. However, the link for 'Interface Perception' and 'Interaction Prediction' is supported primarily by two examples described by words. A stronger validation would require a systematic correlation analysis.

### Questions
1. The benchmark's construction relies heavily on 'GPT-5' to generate question-answer pairs. This methodology is concerning as it may introduce artifacts or biases specific to the generator model, which could then be reflected in the evaluation of other VLMs. Does this benchmark test for fundamental GUI knowledge or for knowledge that 'GPT-5' happens to encode well?

### Soundness
1

### Presentation
2

### Contribution
2
