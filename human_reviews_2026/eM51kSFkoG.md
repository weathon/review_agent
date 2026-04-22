# Batch Speculative Decoding Done Right

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Speculative decoding speeds up LLM inference by using a small draft model to propose multiple tokens that a target model verifies in parallel. Extending this idea to batches is essential for production serving, but it introduces the ragged tensor problem: sequences in the same batch accept different numbers of draft tokens, breaking right-alignment and corrupting position IDs, attention masks, and KV-cache state. We show that several existing batch implementations violate output equivalence—the fundamental requirement that speculative decoding must produce identical token sequences to standard autoregressive generation. These violations occur precisely due to improper handling of the ragged tensor problem. In response, we (1) characterize the synchronization requirements that guarantee correctness, (2)  present a correctness-first batch speculative decoding \oursb that exposes realignment as consuming 40\% of overhead,  and (3) introduce \oursx, which maintains a sliding pool of sequences and dynamically forms same-length groups, to reduce the realignment overhead while preserving per-sequence speculative speedups. On SpecBench dataset, across Vicuna-7B/68M, Qwen3-8B/0.6B, and GLM-4-9B/0.6B pairs, our approach achieves up to 3× throughput improvement at batch size 8 compared to batch size 1, with efficient scaling through batch size 8, while maintaining 95\% output equivalence. Our method requires no custom kernels and integrates cleanly with existing inference stacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EQSpec and EXSpec, focusing on the correctness of batch speculative decoding. The paper claims that existing speculative decoding systems fail to preserve output correctness (aligned to the original base model) when batch size > 1, due to corrupted position IDs, attention masks, and KV cache state. To solve this, it proposes a group-and-padding batch scheduling algorithm to correctly synchronizing among the ragged tensors in a batch. Experimental results show that EXSpec can largely preserve the correctness (over 90%) while other systems can merely do so.

### Strengths
The topic is highly related to practical LLM usage. Batch speculative decoding is an important direction, yet there has been few existing works focusing on the correctness, mostly on speed performances.

The group-then-padding algorithm is practically useful. It can mitigate the length misalignment in an efficient way. Specifically, as stated in Section2, it does not involve modifying position IDs, avoiding re-implementing a whole new kernel, and also preserve the accepted tokens from being cropped.

The demonstration figures (Fig.3 and 4) are quite informative, making the main design easily understood.

### Weaknesses
Main concerns:

As the paper stated, the problem of current inference systems is about incorrect output, which is caused by, alleged, KV-cache and position-ID errors. I think the root causes should be more specified and quantified. Is it because current systems have not implemented batch SD supports, or the implementation is incorrect, or just float precision is not accurate enough? Specifically, vLLM can achieve high match accuracy on Vicuna, but lower on other models. If the cause is about missing or incorrect implementation, I think the results would be uniformly low. If the authors provide detailed implementation or code examples, this concern will be clarified.

The paper claims that the output will be corrupted at batch size > 1, but Table 1 shows that BSP and DSD also have significantly low accuracy when batch size=1. That is a misalignment between the claim and evidence. Furthermore, this result also indicates that the problem is not about batch size, but other factors, while the optimization method largely targets at batch size > 1.

Minor concerns:

The baseline of token throughput in Fig.5(a) is ‘no speculation’. ExSpec only achieves marginal acceleration compared to the original auto-regressive decoding, which is slow. As a speculative method, it should be compared to other speculative-decoding baselines for token throughput.

The claim that ‘speculative decoding needs to yield identical output’ can be more accurate: it is only true when temperature=0, while for temperature>0 the output of base model is a distribution and the output tokens are sampled from this distribution, so there is basically no ‘correct output’ but only a distribution.

The introduction and experiment analysis are hard to read. Improvement on writing would be beneficial.

### Questions
1. Could you provide details about the cause of corrupted output of existing systems , e.g. missing or incorrect implementation, to further clarify the cause of corrupted outputs?
2. Does the incorrectness also exist when batch size = 1? If so, how does the proposed method address this issue, given that the modification is only about batch scheduling?
3. How does the methods perform compared to speculation-based baselines in terms of inference speed?

### Soundness
2

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
This paper proposes a correctness-first batch speculative decoding EQSPEC and EXSPEC to accelerate batch speculative decoding. The method is validated on SpecBench dataset, across Vicuna-7B/68M, Qwen3-8B/0.6B, and GLM-4-9B/0.6B pairs.

### Strengths
- The paper provides a detailed analysis of the existing problems in batch speculative decoding and proposes direct solutions to the most critical issues. For example, the EQSPEC design introduces the *unpad–repad* strategy to ensure correctness, while EXSPEC employs a *dynamic scheduling mechanism* to improve efficiency.
- The paper quantitatively analyzes the actual cost composition and speedup factors in batch speculative decoding, offering an in-depth breakdown of various cost sources and their respective impacts on overall performance.
- The experiments comprehensively compare the proposed methods with multiple existing batch speculative decoding approaches, and further integrate them into system-level frameworks such as vLLM and SGLang. The paper also provides unique insights into the results of current methods and potential directions for future improvements.

### Weaknesses
- The models used for validation in this paper, such as Vicuna and GLM, are relatively outdated and small in scale. Since speculative decoding provides limited acceleration benefits for smaller models, the effectiveness and impact of the proposed methods may be somewhat diminished.
- The paper does not introduce substantial optimizations for KV cache management. Its realignment process is implemented by re-concatenating a rank-4 KV tensor, which imposes significant memory overhead. In contrast, modern systems such as vLLM and SGLang include specialized optimizations for KV cache handling that could potentially improve efficiency.
- The proposed methods are primarily designed for offline batch inference, where the distribution of sequence lengths is relatively uniform. However, in real-world speculative decoding scenarios, task lengths often vary widely. Such heterogeneity may cause a noticeable drop in EXSPEC’s grouping success rate and overall throughput performance.

### Questions
- How does the proposed method perform on larger-scale LLMs and SOTA LLMs? Testing on more powerful models would strengthen the paper’s practical relevance and applicability.
- Is it possible to incorporate more advanced scheduling strategies to further improve EXSPEC’s grouping success rate and overall throughput?
- Since speculative decoding is primarily adopted in online serving environments by major LLM providers, the authors could consider applying their methods in more realistic inference scenarios to better demonstrate their real-world effectiveness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the core challenge of scaling batch speculative decoding for production use: the disruption of output equivalence. The authors correctly identify that the varying number of accepted draft tokens across sequences in a batch leads to the "ragged tensor problem." This prevents existing batch implementations from guaranteeing the output matches standard autoregressive generation. The paper proposes a "correctness-first" framework. First, it rigorously identifies the precise synchronization invariants required to maintain output equivalence. It then presents two implementation strategies: EQSPEC taht strictly enforces these invariants but incurs a high overhead of up to 40% for realignment, and EXSPEC that cleverly avoids the realignment cost entirely by using cross-batch scheduling to dynamically group sequences of the same length. The experimental results show that the proposed method achieves up to 3x throughput improvement (at batch size 8) while successfully maintaining over 95% output equivalence.

### Strengths
1. **An important topic:** It addresses a critical correctness vs. performance trade-off in the LLM production environment.
2. **Innovation:** The clever use of the scheduling mechanism (cross-batch) to resolve a data structure problem (realignment overhead) is a prime example of system-level optimization.
3. **Significant improvement:** The 3x throughput improvement is achieved while maintaining a high correctness guarantee.

### Weaknesses
1. **Compatibility issues:** The compatibility with common modern inference techniques like continuous batching and paged attention remains future work.
2. **Lack of fully quantified metrics:** While EXSPEC avoids realignment, cross-batch scheduling itself might introduce new scheduling latency. The paper needs to further discuss and quantify EXSPEC's scheduling overhead under realistic high-concurrency workloads.

### Questions
see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4
