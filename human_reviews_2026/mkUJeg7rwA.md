# Intra-Prompt Parallel Decoding for Common-Context Question Answering

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
In common-context question answering (CCQA) tasks, multiple questions share a common context to base their answers from. However, Large Language Models (LLMs) typically generate each answer using an independent prompt. While existing batching and caching techniques help improve parallelism and reduce repeated computations, the separation of questions across prompts limits the achievable speedup, as modern GPUs are underutilized due to a memory bottleneck during attention. We present Intra-Prompt Parallel Decoding (IPPD), a novel inference method that answers multiple common-context questions in parallel within a single prompt. IPPD directly addresses the bottleneck by efficiently sharing both memory and computation during the attention process, as the next token for every question is decoded in a single inference step. IPPD uses virtual position IDs and attention mask manipulation to generate the same output as standard prompting without requiring fine-tuning or any changes to the LLM architecture. Since all parallelism occurs within a prompt, IPPD is fully compatible with batched inference, even when each prompt features a different context. Our experiments show that IPPD delivers up to 7X the effective throughput as standard decoding without quality degradation, and outperforms state-of-art inference acceleration methods on real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on accelerating common-context question answering, where a shared context (such as a document) is followed by multiple independent questions. The paper introduces Intra-Prompt Parallel Decoding (IPPD).
IPPD combines multiple questions sharing a common context into a single, structured prompt. By setting position_ids and attention_mask, IPPD decodes all questions in parallel.
The authors modify existing benchmarks, such as NarrativeQA and SQuAD 2.0, to fit the above structure format, thereby accelerating their evaluation. The authors say that IPPD achieves up to a 7x throughput improvement over standard batched inference and outperforms Prefix-Caching+PagedAttention.

### Strengths
1. The underlying logic is sound: Users may ask an LLM multiple questions in a single API call.
2. The experimental results are robust: the experiments show that IPPD outperforms the baselines listed in the paper.

### Weaknesses
1. Limited Practical Applicability due to Input Formatting Assumptions: The paper does not specify how this structured input is created from natural user queries. For instance, when a user submits a prompt such as "Given this report, please summarize the key findings, list the involved parties, and tell me the final conclusion," there is currently no proposed mechanism to parse it into the three distinct questions required by IPPD. Without a defined parsing or pre-processing step, the method is confined to offline batch processing of already-structured datasets and is not directly usable in interactive or conversational systems where user input is unstructured. Although the authors claim to only focus on offline scenarios in the introduction, the scenarios are quite limited.
2. Insufficient Experimental Comparison and Novelty Concerns: Cascade Inference is a highly relevant baseline that also targets the shared-prefix, multi-question scenario. The paper discusses Cascade Inference in Appendix A.2.3, acknowledging that it has a similar objective of reducing memory access. However, the paper lacks a direct experimental comparison, and its introduction section provides only a vague description of cascade inference. This makes it difficult to assess the true novelty and relative advantages of the proposed IPPD approach.

### Questions
1. Suggest adding a subsection to the methodology or discussion that addresses the pre-processing of unstructured user queries.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes the intra-prompt parallel decoding (IPPD) for common-context question answering (CCQA). This method generates multiple answers in parallel by constructing a compound prompt that includes multiple prompts with shared context, by assigning the correct attention mask.

### Strengths
- The evaluation includes multiple QA datasets (NarrativeQA, SQuAD 2.0, RACE and LongHealth).

### Weaknesses
- The proposed method is well-known in the literature.

### Questions
The proposed IPPD method is common practice in the literature.
Please refer to the following papers that have already proposed similar methods:
1. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification
2. DeFT: Decoding with Flash Tree-Attention for Efficient Tree-structured LLM Inference
3. FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference
4. FlashForge: Ultra-Efficient Prefix-Aware Attention for LLM Decoding

The following are some more details about these related works:
The author discuss the differences between the proposed IPPD method and prefix caching in section 2, which is appreciated.
The author claim the major novelty of IPPD compared to prefix caching is that it can generate multiple answers of multiple questions in parallel, where prefix caching need to recompute the attention scores and only reuse the KV cache.
However, many more other works have studied the parallel decoding of requests with shared context, the CCQA task is just one of the many applications of this idea (a two depth tree with one shared root).
Besides, these works also explore more advanced techniques to further optimize the attention computation, which can serve as a stonger baseline to compare with in this work.

(Minor) I also find the inference experiment setting confusing. The author turn off the asynchronous decoding feature of vLLM since the baseline methods do not support it.
However, this feature is indeed esential to speed up the decoding when there are multiple requests with shared context and I can't see the reason why the baseline methods cannot support it.

### Soundness
1

### Presentation
2

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
Intra Prompt Parallel Decoding (IPPD) proposes packed matrix multiplication computation to speed up jagged prefill and decode requests.

### Strengths
I think IPPD is quite efficient when the questions and decoding parts are brief, because we do not need to implement a jagged tensor, which often results in lower utilization of TensorCores (matrix multiplication accelerators).

The method is very intuitive and simple.

### Weaknesses
I think this method is only effective when the question (later part prompt) and the answer are extremely short (<128).
However, such a scenario is extremely rare in the agentic AI era. Therefore the effective-ness of this method is pretty limited.

### Questions
### Questions
- What if the decoding length is long? e.g., reasoning models
- What if prefill is always large? e.g., tool calling

### Formattings
Can you update the figures to make them more intuitive about your method?

### Soundness
3

### Presentation
1

### Contribution
3
