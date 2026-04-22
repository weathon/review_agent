# Progressive Online Video Understanding with Evidence-Aligned Timing and Transparent Decisions

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Visual agents operating in the wild must respond to queries precisely when sufficient evidence first appears in a video stream, a critical capability that is overlooked by conventional video LLMs evaluated in offline settings. The shift to an online, streaming paradigm introduces significant challenges: a lack of decision transparency, the difficulty of aligning response timing with visual evidence, and the need to maintain a global, causally consistent understanding under tight computational budgets. To address these issues, we propose a novel framework that decouples reasoning control from memory integration. We introduce Thinking-QwenVL, an instantiation of this framework with two core components. First, the Active Thinking Decision Maker (ATDM) is a transparent reasoning controller that externalizes its decision process using observable progress ($\boldsymbol{\rho}$) and confidence ($\boldsymbol{c}$) metrics. This allows it to precisely time its response $t_r$ to match the first-sufficient-evidence timestamp $t^\star$ while streaming its reasoning to the user. Second, the Hierarchical Progressive Semantic Integration (HPSI) module acts as an efficient memory system. It employs a set of learnable, multi-level aggregation tokens that are propagated across clips to build a rich, global cognitive state without exceeding token budgets. Extensive experiments demonstrate the effectiveness of ATDM and HPSI, e.g., Thinking-QwenVL improves the accuracy of the previous state-of-the-art from 67.63\% to 71.60\% on the StreamingBench benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel framework for online video understanding in streaming scenarios, where visual agents must respond to queries precisely when sufficient evidence first emerges in a video stream. The proposed system, Thinking-QwenVL (built on Qwen2.5-VL-7B), decouples reasoning control from memory integration through two core modules: 1. Active Thinking Decision Maker (ATDM): A transparent reasoning controller that decomposes queries into sub-questions, tracks observable metrics like progress (ρ) and confidence (c), and self-triggers reflections for cross-clip causal updates. 2. Hierarchical Progressive Semantic Integration (HPSI): An efficient memory system using learnable multi-level aggregation tokens inserted at different decoder depths with structured sparse attention. Extensive experiments on online benchmarks (e.g., StreamingBench, OVOBench, OVBench, RTVBench) show significant improvements.

### Strengths
1. The paper effectively bridges the gap between offline and real-world streaming video understanding by emphasizing evidence-aligned timing and transparency.
2. HPSI's hierarchical aggregation reduces token overhead while preserving cross-clip relations and causal consistency, making it suitable for long videos under tight budgets.
3. Extensive evaluations across multiple benchmarks validate the approach, with clear improvements over state-of-the-art models.

### Weaknesses
1. Built on Qwen2.5-VL-7B, the results may not generalize to smaller or different architectures. What about the performance in smaller-sized models.
2. Focuses primarily on visual evidences potentially lacks diversity. Have you considered the multi-modal settings?

### Questions
What about the computational overhead of the multi-stage ATDM process?

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
3

### Summary
The paper introduces Thinking-QwenVL, a framework for streaming video understanding. It consists of two core components: 1) Active Thinking Decision Marker (ATDM) : a module that determines when to provide an answer based on task progress and confidence. 2) Hierarchical Progressive Semantic Integration (HPSI): an efficient memory system for streaming videos that uses multi-level learnable aggregation tokens to capture video content effectively. Thinking-QwenVL demonstrates strong performance across various online and offline video understanding benchmarks.

### Strengths
- The paper is well written and of high quality.

- The proposed method is both powerful and elegant. The hierarchical processing of visual signals allows progressive handling of long video features from coarse to fine levels. Moreover, decomposing questions into sub-questions and leveraging confidence scores enhances flexibility while preserving the base model’s capabilities.

- The experiments are extensive and comprehensive; including both online and offline benchmarks effectively demonstrates the framework’s robustness and versatility.

### Weaknesses
Despite its effectiveness, several concerns and missing details remain:

- In HPSI, it is unclear how the authors decided on the number of aggregation levels (i.e., three). Does this configuration balance efficiency and accuracy? From Table 4, the performance gain from levels 2–3 appears marginal, as the third row (only level 1) performs comparably to the full method. Further clarification would help justify the design choice.

- In ATDM, the use of sub-questions and confidence scores is crucial, yet the paper lacks sufficient detail to understand how the module operates. Prior work [1] shows that sub-questioning can increase complexity or even degrade performance if applied indiscriminately. Have authors faced such issues? Furthermore, how were the thresholds (e.g., 0.85 and 0.5) for confidence scores in Parts 4 and 5 determined? A clearer interpretation is needed.

- In Table 1, Thinking-QwenVL performs worse than its backbone model (Qwen2.5-VL), but this issue is not addressed, which contradicts the claim in L410–414.

- The recent SOTA method [2] is missing from the comparison, and Thinking-QwenVL shows inferior performance to [2].

*I would consider revising the rating if I misunderstood any part, and the authors clarify these issues in the rebuttal phase.*

**References**

[1] Confidence-guided Refinement Reasoning for Zero-shot Question Answering, arXiv 2025

[2] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant, arXiv 2025

### Questions
See the Weaknesses.

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
4

### Summary
This paper addresses the problem of online video understanding, focusing on two critical but often overlooked aspects: evidence-aligned response timing and decision transparency. The authors argue that conventional video LLMs, typically evaluated in offline settings, are ill-suited for real-world streaming scenarios where an agent must respond precisely when sufficient evidence becomes available. To tackle this, they propose a novel framework, Thinking-QwenVL, which decouples reasoning control from memory integration. The framework has two core components: The Active Thinking Decision Maker (ATDM): a reasoning controller that externalizes its decision process using explicit progress ($p$) and confidence ($c$) metrics, allowing it to align its response time ($t_r$) with the first-sufficient-evidence timestamp ($t^*$). The Hierarchical Progressive Semantic Integration (HPSI) module: an efficient memory system that uses multi-level aggregation tokens to maintain a global cognitive state under tight computational budgets. The paper demonstrates strong empirical results, achieving a new state-of-the-art on the StreamingBench benchmark.

### Strengths
- The paper clearly articulates and tackles the critical, practical challenges of response timing and decision transparency in streaming video analysis, which is a major step towards real-world applications.

- The ATDM module provides an effective framework for making the model's decision-making process transparent and quantifiable through progress ($p$) and confidence ($c$) scores. This is a significant strength for user trust and controllability.

- The HPSI module is a technically sound and well-executed solution for maintaining long-term, causally consistent context within a limited token budget, as demonstrated by its strong performance on long-video benchmarks.

- The paper achieves state-of-the-art results on several challenging online video benchmarks, most notably StreamingBench, providing strong evidence for the effectiveness of the proposed Thinking-QwenVL framework. The ablation studies are thorough and convincing.

### Weaknesses
- The primary weakness is the lack of analysis on the computational latency introduced by the ATDM module. The five-part reasoning process (generating instructions, decomposing the question, captioning, extracting answers, and reflecting) seems to require multiple LLM inference steps for each incoming video clip. This could create a significant processing bottleneck in a true real-time scenario, a concern that is not adequately addressed in the paper.

- The 5-part ATDM process, while transparent, appears highly structured and heavily engineered. This raises questions about its robustness and generalizability. How dependent is this structure on the specific base model (Qwen2.5-VL) and extensive prompt engineering? It's unclear if this complex chain-of-thought would transfer effectively to other video LLMs without significant re-tuning.

- The concept of "Active, Self-triggered Thinking" (Part 5 of ATDM) is an interesting idea but is not described in sufficient detail. The paper mentions it is triggered by low confidence or major semantic shifts, but the exact trigger conditions (e.g., thresholds, detection mechanisms) and the concrete steps of the "reflection" process are not clearly defined. This makes the mechanism less reproducible and its contribution harder to assess.

### Questions
- Could you provide an analysis of the wall-clock latency or computational overhead (e.g., number of forward passes per second of video) introduced by the ATDM module? How does this compare to simpler streaming models like Flash-VStream or Dispider, and how might it impact real-time performance?

- Can you elaborate on the development process for the 5-part ATDM prompt structure? How sensitive is the model's performance to the specific wording and structure of these prompts? Have you experimented with applying ATDM to other base models to test its generalizability?

- Could you provide a more detailed explanation of the "self-triggered reflection" mechanism? What are the specific criteria for triggering it (e.g., confidence thresholds, how are "major semantic shifts" detected)? What does the model do during reflection (e.g., does it re-process past clips, revise the question decomposition)?

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
4

### Summary
This paper introduces Thinking-QwenVL, a new framework for online, evidence-aligned video understanding, addressing the key challenge of determining when a model should respond based on visual evidence as it streams. The framework explicitly separates reasoning control from memory integration, leading to two key components: Active Thinking Decision Maker (ATDM) and Hierarchical Progressive Semantic Integration (HPSI). The authors evaluate Thinking-QwenVL across several online and offline benchmarks (StreamingBench, OVOBench, RTVBench, OVBench, MLVU, VideoMME) and report significant improvements, e.g., +3.97% over Dispider on StreamingBench and +4.9% on OVOBench.

### Strengths
1. The work tackles a largely under-explored but crucial problem — real-time, evidence-aligned video understanding — distinct from traditional offline long-video reasoning.
2. The idea of decoupling reasoning control from memory integration is conceptually strong and practically motivated. The multi-level aggregation mechanism (HPSI) is an elegant solution for progressive semantic integration, offering a new perspective beyond standard pooling or RAG-based methods.
3. The paper is clearly written and well-structured, with informative visual diagrams (e.g., Fig. 1–4) illustrating timing, aggregation, and decision flow. Each module (ATDM, HPSI) is described step-by-step, making a complex architecture accessible.

### Weaknesses
1. While empirically strong, the paper could benefit from more conceptual analysis of why ATDM’s quantitative progress and confidence signals yield better evidence alignment. For instance, is the model implicitly learning uncertainty calibration or temporal gating?
2. The paper claims transparency and interpretability, but does not include user studies or objective interpretability metrics to support these claims (e.g., human evaluation of decision clarity or correctness of rationales).
3. Although ATDM aims for real-time decision-making, the paper does not explicitly report latency, throughput, or computational overhead compared to simpler baselines. Such results would help assess deployment feasibility for real-world streaming agents.
4. The ablations treat ATDM and HPSI largely separately, but it would be informative to analyze their synergy — for example, whether ATDM decisions remain robust if memory integration is partially disabled or simplified.

### Questions
1. For the reference in Latex, please use \citep instead of \cite to fix the reference issue in the main paper so that "Xun et al. (2025)" could be "(Xun et al. 2025)"
2. Have you conducted human or expert evaluations on whether ATDM’s progress/confidence signals improve user trust or understanding compared to black-box baselines?
3. What is the per-frame or per-second latency of Thinking-QwenVL relative to Dispider and Flash-VStream? How does hierarchical aggregation affect token throughput?
4. How does the model behave when the input stream contains missing frames, abrupt scene transitions, or noisy temporal cues? Does ATDM maintain stability in such conditions?

### Soundness
3

### Presentation
3

### Contribution
3
