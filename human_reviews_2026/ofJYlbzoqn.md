# Chronological Thinking in Full-Duplex Spoken Dialogue Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Recent advances in spoken dialogue language models (SDLMs) reflect growing interest in shifting from turn-based to full-duplex systems, where the models continuously perceive user speech streams while generating responses. This simultaneous listening and speaking design enables real-time interaction and the agent can handle dynamic conversational behaviors like user barge-in. However, during the listening phase, existing systems keep the agent idle by repeatedly predicting the silence token, which departs from human behavior: we usually engage in lightweight thinking during conversation rather than remaining absent-minded. Inspired by this, we propose Chronological Thinking, a on-the-fly conversational thinking mechanism that aims to improve response quality in full-duplex SDLMs. Specifically, chronological thinking presents a paradigm shift from conventional LLM thinking approaches, such as Chain-of-Thought, purpose-built for streaming acoustic input. (1) Strictly causal: the agent reasons incrementally while listening, updating internal hypotheses only from past audio with no lookahead. (2) No additional latency: reasoning is amortized during the listening window; once the user stops speaking, the agent halts thinking and begins speaking without further delay. Experiments demonstrate the effectiveness of chronological thinking through both objective metrics and human evaluations show consistent improvements in response quality. Furthermore, chronological thinking robustly handles conversational dynamics and attains competitive performance on full‑duplex interaction metrics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a chronological thinking mechanism, incorporated within the full-duplex pipeline of spoken dialogue systems. The core motivation is to prevent the agent idle when listening to user's utterances, which is not aligned with cognitive human interactions. There are five nodes consisting of the chronological thinking, and they are implemented on top of SALM-duplex model. Experiments are based on multi-turn benchmarks (SpokenWOZ, MtBenchEval) and factual QA benchmarks (Llama Questions, Web Questions).

### Strengths
* The motivation is strong. Agent's idle state generation is a critical issue regarding performance and efficiency in duplex systems. This concurrent speech/reasoning processing is aligns with recent trends in this field.

* The overall writing flow is easy to read.

### Weaknesses
1. **Evaluation metric**: 

    1. It is widely acknowledged that the evaluation performance of LLMs is highly sensitive to the choice of input prompt. However, the manuscript does not provide sufficient details regarding the prompts or the specific evaluation features used in experiments.

    1. For multi-turn dialogues, controlling conversation flow is inherently challenging. The manuscript does not clarify whether the evaluation is conducted at the turn level or at the dialogue level. For example, is the model provided with the entire dialogue history and evaluated solely on its final response?

    1. Using syntactic or semantic similarity metrics such as BLEU and Sentence-BERT is not well-suited for task-oriented dialogue systems, as these metrics do not assess task completion or goal success. Metrics that measure task success rates or completion performance would be more appropriate for this evaluation scenario.

1. **Analysis on the proposed method**: Although the authors emphasize the motivation rooted in the ACT-R cognitive framework, the manuscript lacks further analysis or ablation studies to support the claim. For instance, the five node types described in Table 1 rely entirely on LLM-generated content, yet no analysis is provided to verify whether the generated content aligns with the authors' intended definitions or how each node type contributes to performance across different conditions.

1. **Actual performance benefit of the proposed thinking mechanism**: The performance improvements between the variants with and without the proposed thinking mechanism are marginal. In particular, in Table 4, the claim that chronological thinking outperforms SALM-Duplex appears overstated. The reported gains may instead stem from (1) predicting audio tokens only and (2) the addition of a Transformer decoder module, rather than from the thinking mechanism itself.

1. **# of parameters in Table 1**: Although the proposed architecture introduces an additional Transformer decoder relative to SALM-Duplex, Table 1 does not clearly report the total number of parameters. The manuscript should provide explicit details regarding the parameter count and the architectural additions.

1. **Concerns on the figure**: Figure 2 appears visually similar to figures used in the SALM-Duplex paper, which may raise concerns about originality. It is recommended that the figure highlight this paper's distinctive contribution - specifically, the chronological thinking mechanism.

1. **Subjective results**: Information on evaluator recruitment, sample size, and evaluation guidelines is insufficient. These details are necessary to ensure reproducibility and to support the credibility of the subjective evaluation results.

1. **Some editorial suggeestions**:
    1. Terms "SpokenWOZ" after the line 312 need to be described as "SpokenWOZ-G" to prevent confusion?
    1. MtBenchEval in line 356 should be properly cited.
    1. Typo: "Impatiet" in Table 5.

### Questions
See weaknesses.

### Soundness
1

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
5

### Summary
This manuscript explores the integration of deliberative reasoning capabilities into end-to-end spoken dialogue models, representing an interesting and nascent direction in the audio domain. The core technique involves a "Think-While-Listen" paradigm: specific output tokens emitted by the Language Model during the user's speaking time are substituted with corresponding textual reasoning tokens. This concurrent approach is claimed to introduce zero latency for the reasoning process. The experimental section includes basic conversational QA tests and latency measurements, reporting decent performance gains over weak, foundational baselines.

### Strengths
1. The work is one of the first attempts to systematically explore and implement a deliberative (or "thinking") mode within the architecture of end-to-end spoken dialogue models.

2. The proposed method, while structurally simple, is conceptually sound. Leveraging the user's silence or speaking time for concurrent processing is a valid and pragmatic approach to integrating complex reasoning without incurring additional latency.

### Weaknesses
1. The exploration of the reasoning mechanism is overly simplistic, relying merely on substituting special LLM output tokens with scratchpad text. Further architectural and procedural investigation is warranted. Potential avenues for future exploration include: Expanding beyond Think-While-Listen to Think-While-Speak (concurrent reasoning during both input and output phases). Introducing a dedicated reasoning output head, separate from the main dialogue response head, purely for inference and planning. Investigating whether fine-tuning the reasoning component using Reinforcement Learning (RL) post-pre-training could enhance the quality of the deliberation. Analyzing the reasoning capability's sensitivity to diverse training data distributions.
2. Many conversational scenarios do not necessitate complex reasoning. The training data generation phase should be critically optimized to focus on high-level logical inference dialogues (e.g., those found in benchmarks like BigBenchAudio in GPT-realtime). Furthermore, the examples provided in Appendix A1 appear heavily biased toward tool-use/Agent functionality (which often involves a search path). It is crucial to include more generalized conversational scenarios and provide a clearer manifestation of the ACT-R theory (or similar cognitive architectures) in these common dialogue contexts.
3. The current experimental results are limited by the selection of relatively weak baselines (e.g., GLM-4-Voice). The evaluation must be expanded to include comparisons against state-of-the-art competitive models such as Qwen3-Omni and Kimi-Audio to properly benchmark the proposed technique's efficacy.
4. The thinking paradigm is not inherently constrained to the spoken dialogue model architecture. Validation should be extended to a wider array of models and tasks, such as non-interruptive Spoken QA, and training should ideally be conducted on larger language models to confirm scalability.
5. The ablation study (Table 4) shows relatively marginal gains, suggesting that the current data and scenario design may not sufficiently highlight the benefits of the reasoning process. Further optimization of the data and evaluation scenarios is highly recommended, potentially utilizing broader benchmarks like VoiceBench.
6. The authors should include a more thorough discussion and comparison with concurrent related works [1] and [2].

[1]. Can SpeechLLMs Think while Listening?

[2]. STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models

### Questions
N/A

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
This paper proposes Chronological Thinking, a strictly causal, on-the-fly reasoning mechanism for full-duplex Spoken Dialogue Language Models (SDLMs) that allows them to "think while listening" instead of outputting repeated silence tokens. The system replaces silence tokens during the listening phase with structured reasoning nodes inspired by ACT-R cognitive architecture, enabling better real-time conversational reasoning without added latency. The contributions include (a) the CT-SDLM architecture integrating chronological thinking, (b) a synthetic multi-speaker dialogue dataset generation pipeline, (c) objective and subjective evaluations showing improved reasoning quality, and (d) comparisons against LLM and SDLM baselines such as SALM-Duplex and GT-LM.

### Strengths
1. Introduces a novel and practical causal reasoning mechanism tailored for full-duplex spoken dialogue, addressing a well-defined gap in how agents utilize the listening phase.  
2. Demonstrates improvements in reasoning-heavy dialogue tasks without cost to latency or conversational dynamics.  
3. Evaluation includes comparisons to baselines, across multiple metrics (GPT score, BLEU, Sentence-BERT, factual QA accuracy, turn-taking/barge-in), offering a thorough empirical picture.

### Weaknesses
1. Heavy reliance on synthetic datasets may limit conclusions about real-world conversational robustness and generalization.  
2 The proposed improvement in factual QA performance is negligible, indicating the method’s gains may be task-specific.  
3. Lack of ablation isolating the benefit of ACT-R-inspired node types versus more naive streaming reasoning approaches beyond brief mentions.  
4. Architecture depends on a specific multi-component pipeline (speech encoder, LLM backbone, transformer decoder) which may reduce reproducibility or applicability for researchers without comparable resources.

### Questions
- How does the chronological thinking mechanism perform on purely real, noisy speech data and spontaneous interruptions, beyond synthetic datasets?  
- Could you provide detailed ablation results comparing ACT-R node structuring versus unstructured incremental reasoning to confirm the design choice?

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
The paper proposes Chronological Thinking (CT) for full-duplex spoken dialogue LLMs: while the user is speaking, the model fills the silence/idle slots with a causal, interruptible 5-node "thinking" chain (Entity / Intent / Action / Knowledge / Logic), so that when speech input ends, the model can answer with little extra latency. This directly targets the "inner monologue but unused" issue in Moshi and the "asynchronous thinking" idea in SALMONN-omni. On synthetic duplex dialogue sets, CT improves text/speech quality over its own duplex baseline, while keeping turn-taking roughly in the same range as prior work.

### Strengths
- Clear target & mechanism. Reusing silence tokens for structured, causal planning is simple and fits current full-duplex pipelines. 

- Aligned with 2025 trend. Very close to SCoT, "Can Speech LLMs Think while Listening?", and SHANKS, all of which seek reasoning-while-listening in streaming setups. CT is a reasonable variant in this space. 

- Empirical uplift on its own data. Within their synthetic setup, "with CT" beats "no CT", so the idea is at least self-consistent.

### Weaknesses
- Novelty over Moshi/SALMONN-omni/SCoT is modest. All of these already run an inner or asynchronous text stream during duplex; CT mostly adds a fixed 5-slot structure and a replacement rule. The paper should say why this is better. 

- Comparisons miss the closest 2025 baselines (SCoT, SHANKS, "Can Speech LLMs Think while Listening?"), so it’s hard to attribute the gains to CT itself. 

- Latency claim is fragile. Many duplex systems are at ~200–400 ms E2E; any 200–300 ms overhead from longer thinking streams is user-visible, but the paper does not give distributional or hardware-normalized numbers. (Contrast Moshi’s 160–200 ms.) 

- Data realism. Most evidence is on TTS/synthetic duplex; no noisy/overlapped human speech to show the CT trigger is robust. This is exactly where recent SALMONN-omni reports strength.

### Questions
- Can you run CT on one public recipe (Moshi streaming eval, SALMONN-omni duplex tasks, or SCoT’s streaming CoT benchmark) to show cross-setup gains? 

- Do we really need all 5 node types? Please give ablations (2-3 nodes, or a single "thinking token").

- How is CT triggered online (VAD, ASR partial, fixed frame)? What happens if the user talks unusually fast/slow?

### Soundness
3

### Presentation
3

### Contribution
3
