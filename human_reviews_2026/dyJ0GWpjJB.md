# LightMem: Lightweight and Efficient Memory-Augmented Generation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 4

## Abstract
Despite their remarkable capabilities, Large Language Model (LLM) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often incur substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson–Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognitive-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. 
Experiments on LongMemEval with GPT and Qwen backbones show that LightMem outperforms strong baselines in accuracy (up to 10.9% gains) while reducing token usage by up to 117×, API calls by up to 159×, and runtime by over 12×. Code will be released on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes LightMem, a lightweight memory system for LLM agents that aims to keep long-term dialogue context both accurate and cheap to use. LightMem is explicitly inspired by the Atkinson–Shiffrin model of human memory, and contains: (1) a sensory memory module that pre-compresses and filters incoming dialogue with LLMLingua-2 and groups turns by topic, (2) a topic-aware short-term memory (STM) buffers these topic segments and summarises them adaptively rather than at fixed window boundaries, and (3) a long-term memory (LTM) that is updated in two stages — fast “soft updates” during inference and a later offline “sleep-time” consolidation. 

The system is evaluated in an incremental multi-turn dialogue setting on LONGMEMEVAL-S, where the model only sees past turns as they arrive (simulating agents that operate over very long histories). On GPT-4o-mini and Qwen3-30B backbones, LightMem improves QA accuracy by up to ~10.9% over strong memory baselines such as A-Mem, and simultaneously cuts token usage by as much as 117×, reduces API calls by up to 177×, and lowers runtime by over 12×. 

Ablations varying compression ratio and STM buffer threshold show a tradeoff between efficiency and accuracy but still indicate large gains over prior work across both backbones.

### Strengths
1. Clear, biologically grounded design. Structuring memory into sensory → STM → LTM with delayed consolidation is conceptually clean and directly motivated by cognitive models of human memory and sleep-driven consolidation, which is underexplored in LLM memory pipelines. Decoupling long-term memory maintenance from online inference and executing updates in parallel queues is a nice systems contribution. It directly addresses latency and consistency problems in prior approaches that update memories synchronously during inference. 

2. Strong efficiency gains. The paper optimises for both higher QA accuracy as well as token cost, number of model calls, and wall-clock runtime. These are central bottlenecks for deployed LLM agents, and LightMem shows 1–2 orders of magnitude improvements on these metrics versus baselines like A-Mem, LangMem, MemoryOS, and Mem0. 

3. Topic-aware STM. The idea of grouping turns by semantic/topic boundaries (via attention peaks and embedding similarity) instead of treating each turn or fixed window as a separate memory unit feels like a meaningful refinement over naive chunking and helps avoid both over-fragmented and over-mixed summaries.

4. Clarity. The paper is clearly written and easy to follow, with clear analysis/ablations of each component.

### Weaknesses
1. Benchmark breadth. All main results are on LONGMEMEVAL-S (500 long multi-session dialogues, ~110k tokens each on average) and QA-style evaluation judged by GPT-4o-mini. It’s not obvious how well LightMem generalizes to other domains (e.g., tool-use agents, planning tasks, code assistants) or to evaluation regimes that don’t rely on an LLM judge aligned with the backbone family.

2. Stability of offline updates. After “OP-update” (offline parallel update), accuracy sometimes drops relative to the purely online/soft-update configuration (e.g., GPT backbone at r=0.7 goes from 68.64% to 67.07%), suggesting that consolidation can introduce regressions. The paper claims improved long-term consistency, but the quantitative story is mixed and could use deeper analysis of failure modes.

3. Safety / faithfulness of summaries. The system stores summarized memory units as ground truth for future retrieval. We don’t see discussion of hallucinated summaries or factual drift when compressing many sessions into a single topic-level entry, which is a known risk for LLM-generated memory.

### Questions
1. Your evaluation is limited to LONGMEMEVAL-S and a QA-style setup with LLM-based judging. Can you provide evidence that LightMem generalizes beyond that setting? For example, to tool-using agents, planning/assistive tasks, or code assistants? If not, what are the expected failure modes outside the current benchmark?

2. When do you “sleep” the model? The LTM “sleep-time update” is described as happening offline in parallel and supposedly without user-facing latency. In a real interactive assistant, what triggers this (e.g. wall clock time, buffer size, explicit idle periods)? And what happens if the agent is in continuous use and never becomes idle?

3. Summary faithfulness audits. Have you measured factual faithfulness of STM summaries vs. gold dialogue histories (e.g., hallucination rate, omission rate)? Accuracy on QA is an indirect proxy but doesn’t capture whether LightMem invents or loses user-specific facts in ways that could create safety/privacy issues downstream.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper presents LightMem, a lightweight memory system for large language models (LLMs) designed to improve long-context reasoning in multi-turn dialogues while minimizing inference and maintenance overhead. Inspired by the Atkinson–Shiffrin model of human memory, LightMem consists of three stages: (1) a pre-compression sensory memory that filters redundant tokens using LLMLingua-2, (2) a topic-aware short-term memory (STM) that segments and summarizes dialogue by semantic topics, and (3) a sleep-time long-term memory (LTM) update mechanism that asynchronously consolidates memory to reduce runtime latency.

### Strengths
1. The decomposition of memory into sensory / STM / LTM stages reflects cognitive plausibility, offering interpretability and modularity.
2. Across GPT and Qwen backbones, LightMem consistently outperforms prior methods in accuracy, latency, API usage, and token efficiency.
3. Employing LLMLingua-2 for token-level retention filtering is well-grounded and shown to maintain semantic fidelity even at 50–70% compression ratios.
4. Writing is clear, figures and tables are well-structured, and Appendix contains implementation details, metrics, and additional experiments.

### Weaknesses
1. All evaluation is performed on the QA-focused LONGMEMEVAL benchmark. It remains unclear how LightMem would perform in generation-based tasks, e.g., reasoning or question-answering.

### Questions
1. How well LightMem scales over long durations of deployment, such as weeks- or months-long open-ended interactions? Does the framework include mechanisms to manage memory lifecycle over time to avoid memory bloat and coherence degradation?

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
The paper describes LightMem, a memory module for LLMs, inspired by a model of the human memory. It has three stages: (1) filtering incoming tokens which are of low quality or redundant, (2) a short-term memory which groups together semantically related utterances, and (3) a long-term memory which is maintained during designated offline periods where memories are re-organized, removing redundancies and resolving inconsistencies. By moving this heavy lifting offline, LightMem ensures that the processes involved with maintaining the memory have little impact on the LLM’s latency. Experiments show significant improvement over baselines in terms of accuracy, number of tokens, number of API calls, and runtime.

### Strengths
By and large, the paper is well written and well motivated. The stages of creating and generating memory entries are inspired by the model of the human brain. The experimental setup is sufficient and results are convincing. The field of memory-augmented LLMs is crowded. While the paper does not present fundamentally different ideas to the ones in previous work, I still find it interesting and I think that it is a solid addition to this growing body of work.

### Weaknesses
I don’t have any major issues with the paper. The description in Section 3.3 is not clear to me. What happens after a queue for each entry is created? How is f_{update} implemented? What happens when the LTM reaches capacity?

### Questions
N/A

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
2

### Summary
This paper proposes LightMem, a memory system approach that augments LLMs to better retain and retrieve information over long conversations. LightMem, inspired by Atkinson-Shiffrin's model of human memory, consists 3 components: (1) a sensory memory that filters out noise and retain only the most informative parts, (2) a topic aware short term memory that consolidates information for more structed access, and (3) a long term memory that that decouples consolidation from online inference. The authors report significant gains on several metrics using LightMem on GPT and Qwen models.

### Strengths
* I think this is decent engineering effort to build memory augmented LLMs.

### Weaknesses
* I am not sure about the novelty of this work. The base idea seems to have been studied in prior works with different names/flavors.

### Questions
1. I am not really an expert in the memory-augmented LLMs area, but with cursory google search I found [He et al, 2025](https://arxiv.org/pdf/2405.06067) and [Sun and Zeng, 2025](https://arxiv.org/abs/2507.22925), both of them seem very relevant to this work. You should include a discussion how your approach differ from these previous works, possibly also include them as a baseline. 

2. The paper might be missing some important citations such as [Wu et al, 2025](https://arxiv.org/pdf/2504.15965) that has some discussion on memory systems in humans vs LLMs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Inspired by the Atkinson–Shiffrin human memory model, this paper proposes LightMem, a memory-augmented LLM framework composed of three stages: (1) precompression into sensory memory using a classifier, (2) grouping memory entries by topic with another model to form short-term memory, and (3) updating entries in long-term memory during “sleep time” using a summarization model The authors implement this hierarchical memory structure with GPT and Qwen backbones and report consistent efficiency and performance improvements on LongMemEval tasks compared with state-of-the-art memory-augmented LLMs.

### Strengths
The architecture is inspired by cognitive science, and the translation of the human memory model into an LLM framework is novel and well-motivated.

The model demonstrates promising efficiency compared to other memory-augmented systems, while achieving superior performance on the evaluated benchmark.

### Weaknesses
The paper claims significant efficiency gains, but this is not well supported. LightMem introduces at least three additional components (a compression model, an embedding or topic model, and a summarization model), making it unclear how overall runtime and token usage could be lower than baselines such as RAG. Were the costs of “sleep-time” updates included? The paper should provide standard efficiency metrics such as FLOPs or throughput, or a theoretical analysis explaining why and by how much LightMem calculates and improves efficiency.

The experiments are conducted only on one benchmark. Other long-context evaluation datasets, such as MemoryAgentBench, MemoryBank, LoCoMo, and PerLTQA, should be included to demonstrate generalization.

Several design aspects are underexplained. For instance, GPT-4o-mini is used as the evaluation judge even when the same GPT backbone is part of the model itself, which risks inductive bias. A human evaluation or consistency check between human and LLM judgments would prove the soundness of the reported results. Moreover, the paper does not explain how the hierarchical memory (sensory, short-term, and long-term) is actually used during inference. Are all kinds of memories input into the model at every turn? Are they prepended to the user query as additional context? If there is any retrieval, how does it work?

### Questions
What is the model choice for the summarization model f_{sum}? Is it a stronger model than the backbone?

The soft update mechanism allows preserving both old and new events, even if they appear contradictory. Does this also mean it retains genuinely conflicting facts (e.g., “the user will go to Tokyo at 2 PM” vs. “the user will go to Kyoto at 2pm”)? Will it mislead the model when making responses?

### Soundness
3

### Presentation
2

### Contribution
3
