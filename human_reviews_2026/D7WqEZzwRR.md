# ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Large language models (LLMs) deployed in user-facing applications require long-horizon consistency: the capacity to remember prior interactions, respect user preferences, and ground reasoning in past events. However, contemporary memory systems often adopt complex architectures such as knowledge graphs, multi-stage retrieval, and operating-system–style schedulers, which introduce engineering complexity and reproducibility challenges. We present ENGRAM, a lightweight state-of-the-art memory system that organizes conversation into three canonical memory types—episodic, semantic, and procedural—through a single router and retriever. Each user turn is converted into typed memory records with normalized schemas and embeddings and persisted in a database. At query time, the system retrieves top-k dense neighbors per type, merges results with simple set operations, and provides relevant evidence as context to the model. ENGRAM attains state-of-the-art results on the LoCoMo benchmark, a realistic multi-session conversational question-answering (QA) suite for long-horizon memory, and exceeds the full-context baseline by 15 absolute points on LongMemEval, an extended-horizon conversational benchmark, while using only ${\approx}\$1% of the tokens. Our results suggest that careful memory typing and straightforward dense retrieval enable effective long-term memory management in language models, challenging the trend toward architectural complexity in this domain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents ENGRAM, a lightweight memory layer for conversational agents that separates interactions into episodic, semantic, and procedural stores. A simple router writes normalized records to each store, and per-type dense retrieval selects a small set of relevant snippets for prompt injection. Experiments on multi-session and long-horizon benchmarks report higher LLM-as-judge scores than full-context and several memory baselines, with large reductions in tokens and latency. Ablations indicate that typed storage and retrieval contribute substantially to the reported gains.

### Strengths
1. Method is simple and straightforward. The approach uses a small set of components with clear roles, which makes the design easy to implement. The datastore types and per-type retrieval are intuitive and require minimal orchestration. The empirical performance and latency is also good. 
1. The writing is self-contained and easy to understand.

### Weaknesses
1. Task scope is narrow. The evaluation focuses on conversational memory for chat assistants, primarily via LoCoMo and LongMemEval. By contrast, more general memory architectures can be applied to and are actually tested on long-context and RAG tasks. [1] [2]
1. More experiments are required for Table 3 and 4. Table 3 would benefit from a standard RAG baseline and Table 4 needs an ablation baseline that remove exactly one store at a time.  
1. As the evaluation tasks are more fact-centric, it is unclear to what extent the procedural memory contributes to the performance. The paper would benefit from analyzing the retrieval composition across stores and further studies on workflow-oriented agent memory settings such as that used in [3]. 



[1] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. Gutiérrez et al., 2025. 

[2] M+: Extending memoryllm with scalable long-term memory. Wang et al., 2025.

[3] Agent Workflow Memory. Wang et al., 2024.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ENGRAM, a memory system that organizes conversation into three memory types: episodic, semantic, and procedural. During the memory creation phase, the system includes a memory router (LLM-based) which, based on the conversation between the user and the LLMs, determines which memory bucket applies to the incoming utterance: episodic (events and experiences with temporal context), semantic (facts, observations, and preferences), and procedural (how-to information and processes). Then, for each memory type, a dedicated extractor (LLM-based) is used to extract and canonicalize the raw utterance before storing it in the vector data store. During the memory retrieval phase, the system embeds the user query, retrieves from each memory type, aggregates the retrieved top-k chunks, and augments the user query with this information.

### Strengths
1. The writing of the paper is easy to follow. The author also provides detailed prompts for each module.
2. Benchmarks on the LoCoMo dataset show better retrieval performance.
3. The system is straightforward to implement and can potentially be extended to different memory types as well.

### Weaknesses
1. The memory system treats the input utterance independently of the existing conversation history. When storing this information in the vector database, each chunk does not have the context of the conversation. Often, during a conversation with LLMs, users refer to different components in the chat. The current proposed system seems unable to handle these references.
2. The chunks stored in the vector store seem to be independent over time. Recency dependency and the order of the utterance are not captured in the current design.
3. The current system supports only information addition. Users often want to change or modify their preferences and information throughout the conversation. The lack of deletion and modification of memory seems to me to be a critical issue of the proposed system.
4. On the experimental side, I would love to see more experiments on multi-agent applications.

### Questions
During the chunk aggregation step, how do you re-rank or order the chunks? Do you also use an LLM? Does the aggregation step require generation as well?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ENGRAM, a simple memory layer for conversational LLM agents that splits memories into three types—episodic, semantic, and procedural—routes each turn with a minimal router, stores normalized records with embeddings, and at query time retrieves per-type top-k neighbors, merges and deduplicates them, and assembles a fixed prompt for the answering model. The system uses a single retriever and a fixed evidence budget chosen from ablations.

### Strengths
The design is clear and small: three typed stores, one router, one dense retriever, and a fixed template. This reduces orchestration knobs and makes analyses easier. The formulation spells out record schemas and the retrieval/aggregation steps, including speaker-aware banks and a deterministic template, which helps reproducibility. 

Empirically, the method delivers strong semantic correctness with a strict token budget. Reporting both judge-based and lexical metrics, plus retrieval and end-to-end latency, gives a balanced view of quality and cost.

### Weaknesses
1. The main metric is LLM-as-a-judge with GPT-4o-mini; while they report mean ± sd, judge bias is a risk. A human-rated subset or cross-judge agreement study would raise confidence.

2.  As the main and only judge, a larger/better LLM model should be considered.

### Questions
1. How sensitive is performance to the choice of embedding model and its vector dimension? A sweep over encoders (and mixed-precision indexes) would help quantify portability.

2. Will the change of the judge LLM have an impact on the results?

3. What is the failure profile when gold evidence exists but dense retrieval misses it? Please add breakouts for “evidence found vs. not found” and retrieval-error cascades.

4. The paper reports median end-to-end latency, but it does not mention the variance or distribution of latency across sessions. Could the authors provide latency spread (e.g., mean ± sd or percentile breakdown) to clarify stability and tail behavior?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces ENGRAM, a lightweight memory system that organizes conversational history into three memory types: episodic, semantic and procedural. A single router and retriever handle all memory operations. Each user turn is stored as a typed record with structured JSON fields and dense embeddings. At query time, ENGRAM retrieves the top-k nearest records from each memory type, merges them with set operations and injects the results into the model prompt. Evaluation results show that ENGRAM achieves significantly better performance than full-context baseline while improves latency.

### Strengths
* The proposed method shows even better performance than full-context while achieving lower latency.
* The proposed method is simple.
* The ablation study shows that separating episodic, semantic and procedural memory reduces retrieval competition and improves reasoning diversity.

### Weaknesses
* While ENGRAM demonstrates impressive results on LoCoMo and LongMemEval, the evaluation scope remains relatively narrow and may overstate its general effectiveness. Both benchmarks are synthetic and constrained to conversational QA settings which do not full represent the complexity of long-horizon reasoning in interactive agents. The large performance gap compared to baselines might partly stem from dataset alignment with ENGRAM's design rather than true general improvements in long-term memory. To strengthen its claims, the paer should extend evaluation to agent benchmakers like AgentGym, DeepResearch and SWE-agent.
* Although ENGRAM's empirical results support the utility of separating memory into episodic, semantic and procedural types, this design choice is not theoretically grounded or analytically motivated. The paper does not provide a formal argument, ablation-driven rationale, or illustrative case studies demonstrating why these specific three types are necessary or sufficient for long-term memory. The approach might be just an enigeering heuristics.
* The routing is minimal and rule-based, not learned or adaptive. While this keeps the design simple, it limits the system's ability to handle complex or multi-faceted utterances that may involve overlapping types.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
