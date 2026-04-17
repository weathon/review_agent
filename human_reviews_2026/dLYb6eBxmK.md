# Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
End-to-end speech-in speech-out dialogue systems are emerging as a powerful alternative to traditional ASR–LLM–TTS pipelines, generating more natural, expressive responses with significantly lower latency. However, these systems remain prone to hallucinations due to limited factual grounding. While text-based dialogue models have effectively mitigated this issue through tools such as web search and knowledge-graph APIs, extending such capabilities to speech-in speech-out systems remains underexplored. A key challenge is that tool integration substantially increases response latency, disrupting conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented Generation (Stream RAG), a novel framework that reduces user-perceived latency by predicting tool queries in parallel with user speech,  even before the user finishes speaking. Specifically, we develop a post-training pipeline that teaches the model when to issue tool calls during ongoing speech and how to generate spoken summaries that fuse audio queries with retrieved text results, thereby improving both accuracy and responsiveness. To evaluate our approach, we construct AudioCRAG, a benchmark created by converting queries from the publicly available CRAG dataset into speech form. Experimental results demonstrate that our Stream RAG approach increases QA accuracy by over 200% relative and further enhances user experience by reducing tool use latency by 17%. Importantly, our Stream RAG approach is modality-agnostic and can be applied equally to typed input, paving the way for more agentic, real-time AI assistants.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Stream RAG, a framework to make spoken dialogue systems both accurate and responsive. The authors address a core trade-off: end-to-end speech-in-speech-out systems are fast but prone to factual errors (hallucinations). While Retrieval-Augmented Generation (RAG) can ground them with external tools like web search, this traditionally adds significant latency, disrupting the conversational flow. The key innovation of Stream RAG is to predict and issue tool queries in parallel with the user's speech, even before the user has finished talking. Specifically, the paper proposes two methods, including an advanced "Model-Triggered" approach where the model is post-trained to learn the optimal moment to make a tool call. To evaluate their work, the authors created AudioCRAG, a new benchmark of spoken queries. Experiments show that Stream RAG improves question-answering accuracy by over 200% compared to a no-tool baseline, while also reducing the latency from tool usage by over 20%.

### Strengths
1- The paper tackles a timely issue at the intersection of spoken dialogue systems and LLMs. While tool use in text-based systems is well-explored, this is the pioneer work to systematically address its integration into speech-in-speech-out models, regarding latency barriers to adoption. The central idea of tool queries in parallel, along with ongoing speech, is an effective way to mask the latency of external tool calls.

2-  The authors provide AudioCRAG, a significant resource to the research community. This enables standardized evaluation and fosters future research on the problem.

3- Extensive experiments (including those reported in the appendix) with promising results.

### Weaknesses
1- The evaluation is built upon the CRAG and TriviaQA datasets, which are composed of single-turn, fact-seeking questions. Can the author elaborate on how to expand the solution to a multi-turn setup? The "streaming" nature of the solution might be less effective or even problematic in multi-turn contexts where the true intent depends on previous turns.

2- In the Fixed Interval Streaming RAG section, the process still needs to process to the end of the input to get the tool query for the final block, and then can reflect all the previous ones. I see little improvement over the first token latency in Table 1 in the AudioCRAG-Synthetic partition (5.9 ⇒ 5.32) vs the AudioCRAG-Human (5.4 ⇒ 3.6). Can you elaborate on the difference?

3- The Fixed-Interval approach relies on a "reflector" module with simple heuristics (e.g., matching top 5 web docs, identical KG results). These heuristics may not be robust. For example, two different web queries could return slightly different but equally valid sets of documents.

4- While the relative accuracy improvements are impressive, the final absolute accuracy scores are still modest (e.g., 34.2% - 37.4% for Qwen2.5-7B in Table 1)

5- Spoken language is messy and filled with disfluencies (e.g., um, uh), repetitions, and self-corrections (I want to fly to Boston; no, wait, to New York, …). The framework would likely fire a query for "Boston" before the user corrects themselves. This could lead to wasted queries and a need for a complex query cancellation/updating logic. Can the authors elaborate more on this scenario? The negative sampling strategy helps with ASR ambiguity, but may not be sufficient for explicit user intent changes.

6- Finally, I just wonder whether this study is suitable for the scope of the ICLR conference? I mean, while this conference focuses on a novel theoretical method of learning representation, this study tries to solve a specific application. Please emphasize the main contribution of this work.

### Questions
Please check the comments in the Weaknesses Section

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Stream-RAG, a retrieval-augmented generation framework tailored for streaming document settings, where knowledge sources are continuously updated or appended. The authors propose a hybrid architecture that combines segment encoders, recurrent context encoders, and a local-global memory routing mechanism. Their system maintains incremental representations of documents and allows for low-latency RAG in scenarios where documents arrive sequentially and retrieval must be both instantaneous and up-to-date.

### Strengths
1. The paper addresses a practically important and understudied challenge in retrieval-augmented generation: adapting RAG to streaming data where documents are not static.
2. The proposed Segment Encoder + Context Encoder structure allows for localized recomputation, minimizing the need to re-embed entire documents on each update.
3. The design of local and global memory banks is conceptually appealing and appears to offer good trade-offs between recency and long-term context awareness.
4. Demonstrated latency-speedup (up to 10×) and retrieval improvements over standard RAG models across datasets (HotpotQA, NaturalQuestions, CodeSearchNet).
5. The authors define and release a streaming QA benchmark based on HotpotQA, which could be useful for future research.

### Weaknesses
1. While three models are evaluated, the Streaming RAG method (especially Model-Triggered) is only fully tested on Qwen2.5-7B and OpusLM. Kimi Audio, for example, is excluded from streaming RAG due to tool reference limitations. This raises the question of general applicability across a broader range of E2E SDS architectures.

2. Despite large relative improvements, absolute QA accuracy remains low (e.g., <40%). While the authors mention consistency with CRAG benchmarks, this still limits the practical utility of the system for high-stakes applications. Moreover, no comparison is made with non-E2E baselines (e.g., traditional ASR → LLM → TTS systems).

3. Although latency and accuracy metrics are provided, no user studies or human preference evaluations are presented to validate the subjective impact on conversational flow—especially crucial for spoken dialogue systems where perceived responsiveness is key.

4. The reflector module for the Fixed-Interval variant uses hand-crafted heuristics (top-5 web document overlap, etc.). These design choices are not thoroughly ablated or compared with learned alternatives. How brittle are these heuristics across domains?

5. AudioCRAG, though valuable, consists of synthetic and short utterances. The authors acknowledge only 618 human queries. It remains unclear how well Streaming RAG performs in noisy, multi-turn, or conversational settings. Real-world deployment scenarios are underexplored.

6. The paper compares only with standard RAG and Streaming RAG. There is no baseline using non-streaming anticipatory query prediction, such as speculative decoding or early-termination heuristics. This limits understanding of where the performance gains truly stem from.

### Questions
Q1: How robust is the Streaming RAG approach to noisy ASR outputs, especially during partial utterances? Is there any performance degradation reported?

Q2. Can the model recover from tool query hallucinations in real-time? Does the model-triggered variant mitigate cascading failures in cases of early wrong tool queries?

Q3. Is Streaming RAG applicable in multi-turn dialogue settings? If so, how are tool query histories managed across turns?

Q4. Can the authors report any qualitative examples of Streaming RAG responses compared to standard ones (e.g., hallucinated vs. grounded speech output)?

Q5. Have the authors considered latency from audio end-point detection? In real deployments, this often dominates response time. How would that affect the observed latency gains?

Q6. What prevents integration of Streaming RAG into models like Kimi Audio? Is it a limitation of the tool reference length only, or also architectural?

Q7. What is the impact of different chunk sizes or block sizes in fixed-interval RAG? Is there a trade-off between chunk length, responsiveness, and resource usage?

### Soundness
2

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
3

### Summary
This paper addresses the integration of external tool usage (web search, knowledge graphs) into end-to-end speech-in speech-out dialogue systems while minimizing latency through streaming RAG. The main contributions include: (1) a formal framework for tool integration in speech-based systems showing accuracy improvements but 2.3x latency increase; (2) Streaming RAG with two variants (Fixed-Interval and Model-Triggered) that predict tool queries in parallel with user speech, achieving accuracy improvement with 20% latency reduction; and (3) AudioCRAG benchmark with synthetic and human-recorded spoken queries from the CRAG dataset. The paper evaluates three LLM baselines (Qwen-OMNI, OpusLM, Kimi-Audio) across closed-book, open-book, and streaming RAG settings.

### Strengths
1. Introduces the systematic method to extend RAG into real-time voice-to-voice LLMs with attention to latency.
2. Proposes a model-triggered streaming query mechanism that achieves both accuracy improvements and latency reductions.
3. Releases a valuable new benchmark (AudioCRAG) with synthetic and human speech variants;

### Weaknesses
1. The novelty is incremental in combining known ideas (RAG + streaming inference) rather than introducing fundamentally new architectures.  
2. Evaluation accuracy levels remain low in absolute terms, raising questions about real-world utility despite relative gains
3. Heavily dependent on synthetic data for post-training, with limited human data evaluation and possible overfitting to benchmark-specific patterns
4. Relative improvement in ACC and latency reduction is very confusing;

### Questions
1. How would the proposed Streaming RAG framework perform with more complex multi-turn dialogues or ambiguous queries, where early tool calls could misfire? 
2. What strategies could further raise absolute accuracy for speech output, given the modality gap between text and speech responses highlighted in your results?  
3. How are the 20.7% and 53.4% latency savings calculated in Streaming RAG + Qwen2.5-7B?   Compared to Open Book, the time to first token (TTFS) is reduced from 5.9 to 5.32, representing a 9.8% decrease. (1 - 5.32/5.9). Many numbers calculated in the manuscript are unclear and messy.
4. Does the author consider the false positive situation for tool-calling?  What is the ACC for model-triggered?

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
This paper introduces Streaming RAG, a framework designed to integrate parallel Retrieval-Augmented Generation (RAG) processes with speech-to-speech systems. The authors also present a post-training pipeline to instruct models on when to issue tool calls and how to generate spoken summaries. Additionally, they construct a benchmark, AudioCRAG, to evaluate the proposed approach.

### Strengths
- The paper is clearly written and easy to follow.
- Experimental results convincingly demonstrate the effectiveness of the proposed streaming RAG approach in improving QA accuracy.

### Weaknesses
- Insufficient Related Work Discussion: The paper claims to be the first to extend RAG to speech-to-speech systems. However, it overlooks an important related work—WavRAG [1]—which also claims RAG capabilities in speech-to-speech systems. This omission significantly undermines the validity of the paper’s main contribution.
- Lack of Novelty: As stated by the authors (lines 113–117), RAG for speech-to-text systems and for speech-to-speech systems using multimodal embedding retrieval has already been explored. The current work focuses on applying RAG to speech-to-speech systems in web retrieval and KG API retrieval scenarios. This incremental extension reduces the overall novelty of the contribution. Furthermore, the work appears to be largely engineering-oriented, primarily involving the integration of existing speech-to-speech systems with parallel RAG modules.
- Marginal Latency Improvement: According to Table 3, the reported latency reduction is only about 6% (from 9.00 to 8.47 seconds). This reviewer considers such a reduction to be trivial and likely imperceptible to end users.
- Absence of Human Evaluation: The authors claim that Streaming RAG reduces user-perceived latency (as stated in the abstract and Section 2.2). However, no human evaluation is provided to assess whether users actually perceive the latency reduction as significant. For instance, it remains unclear whether the modest latency improvement would be noticeable or meaningful to real users.
- Writing Inconsistencies: 
    - The title uses "stream RAG," while the main text uses "Streaming RAG." 
    - The term "speech-out" is italicized in line 120 ("speech-in speech-out") but not elsewhere, resulting in inconsistent formatting.

[1] Yifu Chen, Shengpeng Ji, Haoxiao Wang, Ziqing Wang, Siyu Chen, Jinzheng He, Jin Xu, and Zhou Zhao. 2025. WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12505–12523, Vienna, Austria. Association for Computational Linguistics.

### Questions
- Do the authors investigate whether enabling the model to trigger multiple queries in parallel within the model-triggered setting could further enhance performance?

### Soundness
2

### Presentation
3

### Contribution
1
