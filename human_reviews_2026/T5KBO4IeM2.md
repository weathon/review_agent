# Parallel Prompting: Fast LLM Inference for Shared-Context, Short-to-Moderate Output

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
We introduce $\emph{Parallel Prompting}$, a method for high-throughput, quality-preserving decoding of multiple large language model (LLM) queries that share a common prefix. Such shared-context structure arises naturally in applications including document question answering, few-shot learning, multi-user chat, and evaluation pipelines. Prior approaches either degrade generation quality by merging queries into a single prompt that the model cannot reliably disentangle or impose rigid batching and preallocated memory that limit practical deployment. Parallel Prompting is a free lunch for batch prompting: it improves throughput and memory efficiency without requiring model retraining or sacrificing accuracy. The gains are most pronounced when prefix overlap is high and output lengths are short to moderate, with the relative advantage diminishing as unique suffixes grow longer.

Our method executes a single pass over the shared context and decodes all continuations in parallel through efficient matrix–matrix operations, while avoiding cross-query interference and supporting flexible batching across multiple sharing groups with dynamic, on-demand KV-cache management. This design enables high resource utilization during decoding without compromising output quality. Experiments on popular datasets with Llama 3-8B show up to a 4× reduction in end-to-end latency relative to competitive baselines, with no loss in accuracy, demonstrating that Parallel Prompting complements existing batching strategies and expands the practical throughput of LLM-based systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work focuses a practical question: how to efficiently decode multiple queries that share a common prefix. The Parallel Prompting finds that the inference throughput requires a careful balance between attention parallelism and batch size.

### Strengths
* This work try to improve the efficiency of generation that shares a common prefix in large language models (LLMs). For example, the current LLM usually has a system prompt.
* The Figure 1 presents that there is an optimal point between the parallel size and batch size. This is the strength of the motivation of this work:.
* This work significantly improves the generation thoughtoutput, as presented in Figure 1.
* This work sufficient discuss the effect of query number and query length in Figure 3
* In Table 1, this work almost reduce half time cost than the vllm.

### Weaknesses
* Is the Parallel Size P the number of sub batch that are process? In Algorithm 1, why logits ← outputs[:, −P :], as the P should be at the batch size dimension?
* In Figure 1 middle, why the increase of log(parallel/batch_size) will lead to the memory cost decrease?
* Does this work propose a method to estimate the optimal P for the processing? How to determine the Parallel Size P?
* The method performance may not be good with long context.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper introduces Parallel Prompting to efficiently decode multiple queries that share a common prefix in LLMs. This method introduces a method for efficiently generating answers to multiple questions in parallel by independently encoding prompts and leveraging shared context in large language models. Experimental results show that Parallel can improve end-to-end Llama3-8B latency by up to 4× against competitive baselines, without compromising output quality.

### Strengths
Experimental results show its effectiveness.

### Weaknesses
1. The writing is poor in Lines 110 - 151. This makes the paper very difficult to understand. 

2. The paper is very difficult to understand in terms of what was actually done and why the method is effective. Its algorithm is highly non-intuitive.

3. The second and third paragraphs of the introduction are written very poorly. The author hasn’t even figured out how to organize their own work.

4. This approach requires a longer context. Wouldn't that lead to increased computational costs?

5. Why does SeqBatch take less time on the QuAC dataset?

6. The placement of figures and tables in the experimental section of this paper needs to be rearranged.

7. In Line 110, Suppose we have a context C and N sentence queries -> Suppose we have a context C and n sentence queries7

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Parallel Prompting, a novel method for efficiently decoding multiple queries that share a common prefix in large language models. By leveraging parallel processing and matrix-matrix operations, the method significantly improves inference throughput while maintaining output quality. The authors provide theoretical grounding and extensive experiments demonstrating its effectiveness, particularly for short-to-moderate output lengths with high prefix overlap.

### Strengths
1. Novelty: The proposed method effectively addresses inference inefficiencies in shared-prefix scenarios.

2. Theoretical Foundation: A solid analysis based on Amdahl’s Law and hardware constraints explains the trade-off between parallel size and batch size.

3. Comprehensive Experiments: Systematic evaluations across multiple models and datasets show improvements in throughput, memory usage, and output quality.

### Weaknesses
1. Advantage Diminishes and Reverses with Longer Outputs: This is the most significant limitation. As shown in this paper, the generation time of the proposed method begins to exceed that of vLLM when the output length per query exceeds approximately 200 tokens. Therefore, this method is applicable to a quite limited range of tasks.

2. System Complexity and Scheduling Overhead: To achieve optimal performance, the method necessitates a sophisticated scheduler to dynamically balance parallel size (P) and batch size (B). This scheduler must make real-time decisions based on hardware specs, model size, prefix length, output length, etc. The computational and logical complexity of this scheduling itself, compared to the relatively "dumb" but simple scheduler in vLLM, represents an additional engineering and runtime cost that the paper does not evaluate.

### Questions
1. In extreme cases, such as when the prefix length reaches tens of thousands of tokens and the number of queries is also large, could the initial prefill stage and memory usage of the method become a new bottleneck? How would it perform compared to other methods?

2. Given the core weakness identified, the paper suggests a hybrid scheduling policy in production. Could you elaborate on the design principles of such a hybrid scheduler? For instance, would the decision to use Parallel Prompting or fall back to a dynamic batching method (like vLLM's default) be based on the estimated output length or the real-time observed generation length?

3. The paper identifies that maximizing throughput requires balancing the parallel size (P) and batch size (B), and finds the optimal (P, B) through experimental search. However, for production systems, the cost of this search itself can be high. Could the authors propose a low-cost method or a rule of thumb (e.g., building a predictive model based on key factors like model size, prefix length, available memory, etc.) to efficiently determine or dynamically adapt near-optimal P and B values, rather than relying on expensive grid search?

### Soundness
3

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
2

### Summary
This paper proposes ParallelPrompting for efficiently serving many document - many questions scenarios.

### Strengths
ParallelPrompt can speed up overall serving throughput

### Weaknesses
I have much confusion about the figures and the main texts. Please review my questions.

### Questions
### Questions
What is the fundamental difference between RelayAttention + Prefix cached batched decode?

Figure 1 Left Center. X Label should be formatted formally. What do you mean by Parallel Size / Batch Size exactly?

Figure 1 Left Center. The legend does not appear to be formally formatted. What is Length(Doc)? Context length? Length of shared tokens? Number of documents in request?

Figure 1 Right. What is 8 x 64, 8x 128, 8x 256? Does it mean (document count ==) batch size and parallel question sizes? X label looks confusing.

Figure 2. Did you use huggingface's static cache? Why is it OOM?

Figure 4. Why no SGlang?

Figures should be PDF or SVG exported

Table 1. What is lossless, or not (vLLM and SGlang must be lossless, right?)

Table 3. num_q -> change to formal naming

### Formattings
Algorithm 1: Why are local variables not plain text? Font formatting should be textt or text

Typo line 320: inference.CodeLlama -> inference. [ ] CodeLlama

### Soundness
2

### Presentation
1

### Contribution
2
