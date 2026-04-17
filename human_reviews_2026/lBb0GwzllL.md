# VoiceAgentBench: Are Voice Assistants ready for agentic tasks?

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large-scale Speech Language Models (SpeechLMs) have enabled voice assistants capable of understanding natural spoken queries and performing complex tasks. However, existing speech benchmarks primarily focus on isolated capabilities such as transcription, or question-answering, and do not systematically evaluate agentic scenarios encompassing multilingual and cultural understanding, as well as adversarial robustness. To address this, we introduce VoiceAgentBench, a comprehensive benchmark designed to evaluate SpeechLMs in realistic spoken agentic settings. It comprises over 6,000 synthetic spoken queries, including dialogues grounded in Indian context, covering single-tool invocations, multi-tool workflows, multi-turn interactions, and safety evaluations. The benchmark supports English, Hindi, and 5 other Indian languages, reflecting real-world linguistic and cultural diversity. We simulate speaker variability using a novel sampling algorithm that selects audios for TTS  voice conversion based on its speaker embeddings, maximizing acoustic and speaker diversity. Our evaluation measures tool selection accuracy, structural consistency, and the correctness  of tool invocations, including adversarial robustness. Our experiments reveal significant gaps in contextual tool orchestration tasks, Indic generalization, and adversarial robustness, exposing critical limitations of current SpeechLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VoiceAgentBench, a new benchmark for SpeechLLM agents. The benchmark covers various tool-calling scenarios and safety evaluations. It also covers several languages beyond English. The evaluation reveals significant limitations of current SpeechLLMs as agents, providing some insights into this direction.

### Strengths
- The topic is quite timely in the era of AI agents.
- Well-scoped problem & broad coverage.
- Layered evaluation design. The split into Tool Selection, Tool-Call Structure, Parameter-Filling, and Refusal Rate gives interpretable failure modes, instead of a single pass/fail.

### Weaknesses
- Synthetic speech only: Whether the synthesized speech is of good quality is unknown. There is also no consideration of real-world noisy environments.
- Lacking details of LLM-judge, such as decoding strategies and evaluation qualtiy (compared with human annotation).
- Metric design choices may bias outcomes. Tool Selection is an exact-match on function name; Tool-Call Structure hinges on Pydantic schema validation. These can over-penalize semantically correct outputs with minor format variance, while PF then depends on an LLM judge. A fuller error taxonomy or tolerance bands (e.g., argument similarity) beyond the judge would strengthen conclusions.
- The coverage of the evaluated models is quite restricted. The paper should include more models for a more comprehensive overview.
- No confidence intervals or significance tests for the results.

### Questions
- Why was GPT-4o-mini chosen as the sole judge for PF/RR? Did you try alternative judges? Any evidence of judge–model bias?
- In ASR→LLM pipelines, did you test stronger Indic ASR? Can you quantify how much of the Indic gap remains after substituting better ASR models? This will provide better insights than directly using ground-truth transcriptions.
- Can you provide details in the one-shot settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces VoiceAgentBench, a novel and comprehensive benchmark designed to systematically evaluate the agentic capabilities of Speech Language Models (SpeechLMs) in realistic spoken interaction scenarios. Unlike existing benchmarks that focus on isolated tasks such as transcription or question answering, VoiceAgentBench assesses a wide range of agentic behaviors, including single and multi-tool invocation, multi-turn dialogue, and safety/adversarial robustness. The benchmark consists of over 5,500 synthetic spoken queries generated via TTS, covering English, Hindi, and five other Indic languages, with a significant portion contextualized in Indian cultural scenarios. The authors propose a diversity-based sampling method for TTS voice conversion to ensure coverage across accents and speaker characteristics, and they define a layered evaluation framework to measure tool selection, structural consistency, parameter filling, and refusal of unsafe requests. Overall, the work fills a critical gap in the evaluation of SpeechLMs for real-world voice assistant applications.

### Strengths
1. The proposed evaluation framework is practical and effectively measures agentic capabilities of SpeechLMs in realistic scenarios.

2. The experiments are thorough, covering multiple languages, diverse tasks, and adversarial robustness, with clear and convincing results.

3. The paper is well-structured and clearly written, making the methodology and findings easy to understand.

### Weaknesses
1. As a benchmark, the evaluation in this paper is not sufficiently comprehensive; it mainly focuses on agent scenarios in the Indian context, which limits the generality of the assessment.

2. The paper’s originality is somewhat lacking. In the benchmark design, the authors do not clearly explain the fundamental differences between speech agents benchmark and text agents benchmark, nor the necessity of a dedicated speech agent benchmark. The three speech scenarios described seem to be subsets of text agent scenarios. Regarding the claimed innovation of diversity-based TTS generation, it is also unclear whether this approach is necessary, or if using a rich set of audio resources with a zero-shot TTS strategy could achieve similar results.

3. The presentation of the paper could be improved. For the many audio-related cases, a demo page may be a better choice than displaying cases across multiple pages.

### Questions
Could the authors clarify the fundamental differences between the proposed speech agent benchmark and existing text agent benchmarks where queries are simply converted to speech using TTS? Specifically, what unique challenges or evaluation aspects are addressed by your benchmark that cannot be captured by applying TTS to text-based agent benchmarks?

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
5

### Summary
This work introduces Voiceagentbench, a new benchmark specifically designed to evaluate the agent-based (speech-to-text) functionalities of Speech LLMs. The benchmark comprises over 5,000 synthetic spoken queries, including dialogues grounded in Indian contexts, and is structured to cover single-tool invocations, multi-tool workflows, multi-turn interactions, and safety evaluations. The dataset construction follows a conventional pipeline: text prompts are sourced from existing text-based LLM agent benchmarks and then converted to speech via TTS synthesis. The authors evaluate several prominent models, including qwen2.5-omni, kimi-audio, and audioFalmingo3, alongside various cascaded speech-to-text LLM systems. The results indicate that, with the exception of kimi-audio, the other end-to-end models (qwen2.5-omni, audioFalmingo3) exhibit significant performance degradation on this benchmark.

### Strengths
1. Voiceagentbench represents a timely and pioneering effort, being one of the first benchmarks dedicated to evaluating the agent capabilities of speech-based LLMs.

2. Although the methodology for dataset creation is not highly innovative, the resulting dataset, pipeline, and evaluation framework provide a contribution that will be useful to the research community.

### Weaknesses
1. The dataset is heavily skewed towards Indian contexts, in both its textual content and synthesized audio. This poses a significant domain mismatch for mainstream models, which are predominantly trained on general English data. This specialization limits the benchmark's broader applicability and generalizability.
2.  Authors need to clarify the relationship between their work and existing research on voice agents. A more thorough discussion is required to situate this benchmark relative to recent advancements, such as[1][2][3][4]:
3. The benchmark's design appears to primarily target cascaded speech-to-text LLMs, rather than being optimized for evaluating true end-to-end speech agent models.
4. The exclusive use of synthetic data introduces a potential gap between the benchmark's evaluation scenarios and the complexities of real-world, spontaneous human-machine conversation.
5. Many of the conclusions drawn from the experiments offer limited insight. For instance, the poor performance of qwen2.5-omni (often scoring 0) is likely attributable to a simple lack of agent-specific training data, rather than a deeper flaw in the model's architecture. Such conclusions do not foster deeper investigation into the core challenges of speech agents.
6. The overall pipeline—dataset creation, model selection, and evaluation—is conventional, albeit representing a non-trivial amount of work. The benchmark overlooks several critical dimensions of agent performance that warrant further exploration, such as: The latency introduced by agent function calls. The potential degradation of the base model's core abilities when agent functionalities are integrated. The evaluation of complex, multi-agent collaborative scenarios.




[1]. Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents

[2]. Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage

[3]. GPT-RealTime

[4]. Step-Audio 2 Technical Report

### Questions
It would be highly beneficial for the authors to include a more comprehensive set of examples or case studies from the dataset. This would provide greater transparency and a clearer understanding of the benchmark's tasks and data quality.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VoiceAgentBench (VAB), the first benchmark designed to evaluate agentic speech-based systems---i.e., voice assistants that can reason, call tools, and handle multi-step workflows. The benchmark includes 5,500+ synthetic spoken queries in 7 languages (English + 6 Indic), covering:

- Single-tool and multi-tool usage  
- Sequential dependent workflows  
- Multi-turn interactions  
- Safety and adversarial robustness  

Six models (SpeechLMs/ASR to LLM pipelines) are evaluated. Results highlight significant weaknesses in multilingual performance, tool reasoning, and safety alignment.

### Strengths
1. Novel and timely: first benchmark for speech-based agentic reasoning and tool orchestration.  
2. Comprehensive task design: Includes sequential tool use, multi-tool planning, multi-turn dialogue, and adversarial safety tasks.  
3. Multilingual and culturally grounded: covers 7 languages, with 30% Indian-context queries, which has been rare in previous work.  
4. Acoustic diversity: uses Farthest Point Sampling (FPS) over TTS speaker embeddings to maximize voice variation.  
5. Granular evaluation metrics. Tool Selection (TS), Tool Call Structure (TCS), Parameter Filling (PF), and Refusal Rate (RR) allow fine-grained diagnostics.  
6. Extensive experiments and ablations, including error propagation from ASR, prompting strategies, and safety prompting.

### Weaknesses
1. All speech is synthetic (TTS), which limits realism and excludes natural speech variability, noise, and disfluencies.
2. Ground truth for sequential tool workflows is generated using GPT-4o-mini rather than human annotation, which may introduce hallucinations or inaccuracies.
3. Evaluation relies on GPT-4o-mini as a judge for parameter accuracy without human agreement studies or reproducibility checks.
4. Multi-turn dialogue is only evaluated in English and entirely missing for Indic languages.
5. Safety evaluation treats refusal as a binary outcome. And also does not seem to assess partial compliance, harmful outputs, or nuanced behavior.
6. No statistical significance testing, confidence intervals, or variance reporting; dataset construction and filtering steps lack transparency.

### Questions
- Has any portion of the GPT-generated ground truth been validated by humans?
- What is the agreement between human judges and GPT-4o-mini in evaluation?
- Why were multi-turn tasks excluded for Indic languages, and will they be added?
- Do you plan to include real or noisy speech in future versions of the benchmark?
- Can you provide per-language breakdowns of failure types?

### Soundness
3

### Presentation
3

### Contribution
3
