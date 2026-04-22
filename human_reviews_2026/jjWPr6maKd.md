# XGC-AVis: Towards Audio-Visual Content Understanding with a Multi-Agent Collaborative System

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In this paper, we propose **XGC-AVis**, a multi-agent framework that enhances the audio-video temporal alignment capabilities of multimodal large models (MLLMs) and improves the efficiency of retrieving key video segments through $4$ stages: perception, planning, execution, and reflection. We further introduce **XGC-AVQuiz**, the first benchmark aimed at comprehensively assessing MLLMs' understanding capabilities in both real-world and AI-generated scenarios. XGC-AVQuiz consists of $2,685$ question-answer pairs across $20$ tasks, with two key innovations: 1) **AIGC Scenario Expansion:** The benchmark includes $2,232$ videos, comprising $1,102$ professionally generated content (PGC), $753$ user-generated content (UGC), and $377$ AI-generated content (AIGC). These videos cover $10$ major domains and $53$ fine-grained categories. 2) **Quality Perception Dimension:** Beyond conventional tasks such as recognition, localization, and reasoning, we introduce a novel quality perception dimension. This requires MLLMs to integrate low-level sensory capabilities with high-level semantic understanding to assess audio-visual quality, synchronization, and coherence. Experimental results on XGC-AVQuiz demonstrate that current MLLMs struggle with quality perception and temporal alignment tasks. XGC-AVis improves these capabilities without requiring additional training, as validated on two benchmarks. The project page is available at: https://xgc-avis.github.io/XGC-AVis/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
XGC-AVis introduces a training-free multi-agent pipeline: (1) Perception, (2) Planning, (3) Execution and (4)Reflection, which enhance temporal alignment and quality perception in MLLMs. Inside XFC-AVis, two planners identify time segments relevant to a question, two executors reason over them, and a decider resolves conflicts, producing final answers. 

The paper also presents XGC-AVQuiz, a new benchmark with 2,232 videos and 2,685 QA pairs spanning PGC, UGC, and AIGC sources across four categories (recognition, localization, reasoning, and quality perception). Experiments show consistent gains over both open- and closed-source MLLMs. Ablation results confirm that the dual-planner and decider design is key to improvements.

### Strengths
- Writing is logical and easy to follow, motivation is clear
- The paper introduces a novel multi-agent pipeline (Perception–Planning–Execution–Reflection) that improves A/V reasoning without extra training.
- Evaluations cover multiple open- and closed-source MLLMs, two benchmarks, and detailed ablations.

### Weaknesses
- The first weakness is the multi-agent framework increases computational cost as slove one question, needs more llms/models envloves in. The author should quantifying latency or cost trade-offs.
- I appreciate the author propose a new benchmark, and do a lot testing on this benchmark. However, some tasks in XGC-AVQuiz overlap conceptually with existing A/V QA datasets (e.g., AVQA, Daily-Omni), can the author also report on those datasets, and compare with traditional models?
- For the model: while the multi-agent design is novel, it mainly combines existing reasoning and planning components rather than introducing a fundamentally new learning mechanism or model architecture.

### Questions
Please see weakness part.

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
This paper proposes XGC-AVis, a four-stage (perception, planning, execution, reflection) multi-agent framework designed to enhance audio-visual temporal alignment and key segment retrieval for multimodal large language models (MLLMs). It also introduces XGC-AVQuiz, a benchmark with 2,232 videos (PGC/UGC/AIGC) and 2,685 QA pairs covering 20 tasks, including a novel "quality perception" dimension. Experimental results claim XGC-AVis outperforms existing MLLMs on XGC-AVQuiz and Daily-Omni.

### Strengths
- The XGC-AVQuiz benchmark attempts to address the limitation of existing datasets by integrating PGC, UGC, and AIGC scenarios, and introduces a "quality perception" task to evaluate MLLMs’ ability to assess audio-visual synchronization and coherence—an underexplored area in current benchmarks.

- The four-stage multi-agent design of XGC-AVis provides a approach to tackle audio-visual temporal alignment, which is a known challenge for MLLMs.

### Weaknesses
- The paper is lack of novelty. It claims to enhance audio-video understanding of MLLMs via XGC-AVis, but it fails to address the fundamental issue of poor fine-grained capabilities in audio-video MLLMs. XGC-AVis only adopts a "component-stitching" approach (e.g., using Deepgram for subtitles and r1-aqa for audio descriptions, Section 3) to interweave unimodal outputs, without realizing feature-level or semantic-level deep cross-modal interaction—leaving the core problem of weak fine-grained cross-modal association unresolved
- The paper explicitly states that XGC-AVis "interweaves video frames, audio segments, subtitles, and audio descriptions to form coherent multimodal units" (Section 3, 1-26), but it completely ignores the feature space gap between audio and visual modalities—an issue that directly undermines the "coherence" of the resulting multimodal units.
- Directly concatenating or interleaving unaligned features forces the subsequent planners/executors  to handle inconsistent feature spaces, which can lead to two critical issues: The model may over-rely on one modality (e.g., visual features, which are more structured) and ignore audio features, weakening the intended "multimodal collaboration". The paper does not acknowledge or address these risks.
- XGC-AVis integrates multiple expert models/tools (Deepgram, r1-aqa, Aria, Qwen2.5-Omni, Gemini 2.0 Flash) to form a multi-expert ensemble system, while the comparison baselines in Table 2 are standard single MLLMs . This comparison is unfair, as XGC-AVis leverages "collective intelligence" and higher computational costs that standard MLLMs lack, yet the paper only emphasizes performance advantages without acknowledging such trade-offs.
- The paper uses Aria (Planner 1) and Qwen2.5-Omni (Planner 2), but Aria "can only process video and text" while Qwen2.5-Omni "processes video, audio, and text" (Appendix A.3). Why combine a unimodal (video-text) planner with a multimodal (video-audio-text) planner? Does this combination introduce redundancy, and if not, what unique value does each planner provide?
- Tables 2 only report "average accuracy" for broad tasks (e.g., A/V Localization) but not task-specific metrics for time alignment—such as Temporal Localization Accuracy (TLA): The percentage of cases where the model’s predicted time segment overlaps with the ground-truth by ≥50% IoU
- The "perception" stage uses both Deepgram (speech-to-subtitle) and r1-aqa (audio descriptor). However, the paper does not clarify whether their outputs are redundant—for example, if Deepgram’s subtitles already capture speech content, what unique information does r1-aqa’s audio description add (e.g., background noise, emotion)?
- Generally, audio and video in datasets are consistent (Section 4.1), so audio should only bring marginal improvements for most tasks. However, Table 2 shows a significant performance boost when adding audio (e.g., Qwen2.5-Omni’s A/V Recognition accuracy jumps ~16% from Vid. to Vid.+Aud.). This is unusual. Additionally, if video-only inputs already achieve high accuracy on supposed "audio-driven" tasks, it implies potential flaws in the evaluation’s task design, undermining its credibility.
- The paper does not clarify whether the questions in Table 2 differ between the "video-only" and "video+audio" settings. Without disclosing the distribution of modality-dependent questions, the observed performance boost from adding audio may stem from question design bias (forcing audio reliance) rather than true multimodal understanding, making the results uninterpretable.
- The authors do not evaluate whether questions can be answered with a single modality (e.g., audio-only). The experimental input settings (Section 5.1) only include video-only, video+audio, and video+audio+subtitle, without testing audio-only inputs. This omission means it is impossible to verify if multimodal integration is truly necessary—for example, some A/V reasoning tasks may be solvable via video alone (facial expressions) or audio alone (tone), rendering XGC-AVis’s multimodal design redundant.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **XGC-AVis**, a multi-agent framework designed to enhance multimodal large language models’ (MLLMs) ability to align audio and video temporally and to improve the efficiency of retrieving key video segments. The system operates through four stages—perception, planning, execution, and reflection—enabling coordinated agent interaction for more robust multimodal reasoning.

To evaluate such capabilities, the authors also propose **XGC-AVQuiz**, a new benchmark that comprehensively assesses MLLMs’ understanding **across real-world and AI-generated scenarios**. XGC-AVQuiz features both professionally generated (PGC), user-generated (UGC), and AI-generated (AIGC) videos spanning multiple domains, and it introduces a quality perception dimension that tests synchronization, coherence, and audio-visual quality awareness. Experimental results suggest that existing MLLMs struggle in these aspects, while XGC-AVis achieves measurable improvements without additional training.

### Strengths
**High-quality benchmark contribution.**
XGC-AVQuiz provides a more comprehensive, multi-level evaluation benchmark for multimodal reasoning, covering diverse data sources (PGC, UGC, AIGC) and introducing the novel dimension of quality perception. This significantly enriches the evaluation landscape for audio–visual MLLMs.

**Exploratory insights into multi-agent design.**
XGC-AVis presents valuable design insights for constructing multi-agent collaborative systems, offering a practical exploration into how workflow-level designs can help MLLMs overcome model-level limitations when dealing with complex multimodal understanding tasks.

### Weaknesses
- **Lack of empirical support for efficiency claims.**  
  The paper states that **XGC-AVis** *"improves the efficiency of retrieving key video segments”*, but presents no quantitative evidence or analysis to support this. Without metrics or user-study results, this claim remains unsubstantiated.

- **Insufficient citations and context in the introduction.**  
  The discussion of MLLMs and representative models cites mainly vision–language (e.g., Llava-OneVision, InternVL3) or pure language models , but fails to reference existing multimodal reasoning efforts that go beyond surface-level fusion [1–3]. This weakens the argumentative foundation of the introduction.

- **Unconvincing “data source bias” argument.**  
  The claim that existing datasets are dominated by user-generated content (UGC) and lack professionally generated content (PGC) is poorly justified. Given that platforms like YouTube already contain high-quality professional material, the UGC–PGC distinction should be better defined or empirically supported.

- **Loose connection between Related Work and the proposed system.**  
  The Related Work section focuses mainly on architectural summaries of MLLMs, while XGC-AVis is a workflow-level, multi-agent system. To strengthen positioning, the authors should include relevant **agent-level** or **system-level** works such as [4–5].

- **Ambiguous baseline description.**  
  It is unclear whether *VideoLLaMA2* or *VideoLLaMA2.1-AV* is used as the baseline. These two differ substantially in multimodal input handling, and the choice directly affects the validity of the comparison.

- **Limited comparative and generalization experiments.**  
  (1) Table 1 compares only VLM and Qwen2.5-Omni; adding *Qwen2.5-VL* would provide a fairer multimodal baseline.  
  (2) Reporting XGC-AVis’s performance on an external benchmark such as *WorldSense* would help demonstrate generalizability.

- **Missing ablations on key system components.**  
  While the data ablations are informative, the paper lacks **component-level** analysis. For example, how does the *Interleave* step in “Align multimodal data” affect performance? Additionally, replacing *Gemini 2.0 Flash* (executors/decider) with open-source alternatives would help isolate whether gains stem from the multi-agent design or the strength of the base models.

---
# Reference

[1] Aligned Better, Listen Better for Audio-Visual Large Language Models, ICLR 2025, https://arxiv.org/abs/2504.02061

[2] HumanOmni: A Large Vision-Speech Language Model for Human-Centric Video Understanding, https://arxiv.org/abs/2501.15111

[3] SALMONN family: A suite of advanced multi-modal LLMs,https://github.com/bytedance/SALMONN

[4] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions, https://arxiv.org/abs/2505.14668

[5] Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities

### Questions
The key concerns are outlined in the weaknesses section. Should the authors **provide convincing responses or improvements** addressing these points, **I would be open to increasing my evaluation score**.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The submitted manuscript proposes XGC-AVis a multi-agent framework designed to enhance temporal alignment and audio-visual reasoning capabilities in MLLMs. The authors also introduce XGC-AVQuiz, a comprehensive benchmark comprising 2,232 videos covering both real-world and AI-generated content and 2,685 question–answer pairs, aimed at evaluating recognition, localization, quality perception, and reasoning abilities in audio-visual tasks. Extensive experiments verify the effectiveness of XGC-AVis, demonstrating a certain degree of novelty.

### Strengths
1. The manuscript clearly defines the problem and provides a well-motivated solution.
2. The construction of the XGC-AVQuiz benchmark is valuable and contributes to the development of the audio-visual research community.
3. The proposed XGC-AVis framework achieves superior performance across multiple benchmarks, and the ablation studies are comprehensive.
4. The paper is clearly written and easy to follow.

### Weaknesses
1. The related work section omits several directly relevant recent studies on multi-agent, multimodal, and video question-answering architectures and benchmarks, such as *MAGNET*, *VideoMultiAgents*, and *OmAgent*. The authors are encouraged to discuss distinctions from these works.
2. Given that MLLMs are used to generate distractors, how do the authors ensure that the benchmark does not inadvertently favor architectures used for generation or self-reflection?
3. Compared to single-agent or monolithic MLLM designs, how does XGC-AVis perform in terms of computational cost, especially for longer video inputs?
4. Since each query involves multiple *planner–executor* agents, the computational cost could be significant; a discussion or analysis of scalability is suggested.
5. Some experimental details require further clarification, for example, the input strategy for candidate time segments and the handling of inconsistent answers.
6. It would be helpful to elaborate on how conflicting segmentation results from different planners are weighted or fused during decision integration.

### Questions
My main questions are reflected in the Weaknesses Section.

### Soundness
2

### Presentation
2

### Contribution
2
