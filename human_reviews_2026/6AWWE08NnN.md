# CompLLM: Compression for Long Context Q&A

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Large Language Models (LLMs) face significant computational challenges when processing long contexts due to the quadratic complexity of self-attention. While soft context compression methods, which map input text to smaller latent representations, have shown promise, their real-world adoption is limited. Existing techniques typically compress the context as a single unit, which leads to quadratic compression complexity and an inability to reuse computations across queries with overlapping contexts.
In this work, we introduce CompLLM, a soft compression technique designed for practical deployment. Instead of processing the context holistically, CompLLM divides it into segments and compresses each one independently. This simple design choice yields three critical properties: efficiency, as the compression step scales linearly with the context length; scalability, enabling models trained on short sequences (e.g., 1k tokens) to generalize to contexts of 100k tokens; and reusability, allowing compressed segments to be cached and reused across different queries.
Our extensive experiments show that with a 2x compression rate, CompLLM speeds up Time To First Token (TTFT) by up to 4x, reduces the KV cache size by 50%, and effectively doubles an LLM's maximum context length. Furthermore, CompLLM achieves performance competitive with using the uncompressed context, and even surpasses it on very long sequences, demonstrating its effectiveness and practical utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduce compLLM, a soft compression method to segment and compress extremely long context for Q&A problems. Experiments show significant speedup and memory efficiency, and achieve the same level as results with uncompressed context.

### Strengths
1. The research topic is meaningful. The paper is well organized and easy to follow.
2. The formulation of CompLLM is simple but reasonable. Detailed description and explanation in the embedding space reveals the inner mechanism of the proposed method.
3. The author coundct extensive experiments to demonstrate the effectiveness of CompLLM.

### Weaknesses
1. The idea of compression is well-studied and tested in many previous works, e.g., [1][2][3], and seems not to be novel.
2. Only QA tasks are evaluated for CompLLM. A significant drawback in compression-based approach is that they may fail in fine-grained retrieval settings, such as Needle-in-haystack (NIAH). Besides, there are many other typical domains that require long-context understanding capabilities, such as code completion and in-context learning. More experimental results should be provided to prove the generalizability of the proposed method.

[1] Chevalier, Alexis, et al. "Adapting language models to compress contexts." arXiv preprint arXiv:2305.14788 (2023).

[2] Han, Wei, et al. "Two are better than one: Context window extension with multi-grained self-injection." arXiv preprint arXiv:2410.19318 (2024).

[3] Zhang, Peitian, et al. "Soaring from 4k to 400k: Extending llm’s context with activation beacon." arXiv preprint arXiv:2401.03462 2.3 (2024): 5.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes CompLLM, a soft context compression approach for long-context QA. It segments the input into short chunks and compresses each segment independently into “concept embeddings” (CEs). And then feeding them to an unmodified LLM. The claimed benefits are linear-time compression, scalability to very long contexts, and reusability of compressed segments. With a 2× compression rate, the authors report up to 4×speedups and 2× KV-cache reduction, while maintaining or improving accuracy on long contexts. Experiments cover open-ended and multiple-choice Q&A and a LOFT-style long-context setup.

### Strengths
1.	The paper contains a clear complexity analysis (distinguishing KV prefill vs. decoding), and reports TTFT and KV cache savings that match the theoretical scaling

2.	The method description (segmenting, S=20, C=2, LoRA + linear head) is easy to follow.

### Weaknesses
1.	Lack of novelty: The core difference from prior soft compression is to compress in segments rather than all long-context. This is only a design hyperparameter (segmenting strategy and segment length S) rather than a fundamentally new modeling principle. A deeper analysis or theory explaining why segment-wise independence preserves task-relevant information—and when it fails—would be needed to elevate the conceptual contribution.

2.	Lack of baselines: The experimental comparison is primarily w/ CompLLM vs. w/o compression. There is no broad comparison with existing compression baselines, which undermines the empirical claims. Strong, relevant baselines such as LLMLingua-2 are absent. Without these, it is unclear whether the reported gains are specific to CompLLM.

3.	“Segment for Long Context” is positioned as the main contribution, but there is no thorough study of how segment length S (and overlap, boundary effects, tokenizer granularity) impacts accuracy, latency, and reuse. An ablation over S, with detailed accuracy/latency/KV trade-offs and error analysis, is missing.

4.	The paper motivates reusability in RAG-style pipelines but does not provide results against standard RAG baselines (e.g., top-k retrieval + reranking + uncompressed prompting). It remains unclear whether CompLLM improves end-to-end RAG outcomes (accuracy, latency, cost) versus optimized retrieval that reduces N in the first place.

### Questions
1. Can you add head-to-head comparisons at matched compression rates with strong baselines (e.g., LLMLingua-2)?

2. How does performance vary with segment length (and optional overlap)?

3. Are the method is useful for RAG?

4. How do results scale to larger models and higher compression (C>2)? Is degradation graceful?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors introduce a technique for compressing transformer context by a factor of 2.  The initial context is first broken into chunks, and each chunk is then compressed separately, using gist tokens.   Compression only requires local attention within the chunk.  The outputs from the gist tokens are then fed in as "soft tokens" to a frozen pre-trained LLM.

The technique reduces prefill costs by (up to) a factor of 4, because attention during prefill is O(N^2) wrt. to sequence length, and the sequence length has been reduced by a factor of 2.  It also reduces inference costs by a factor of 2, since each newly generated token attends over half the sequence length. 

On various long-context tasks, compression often /increases/ accuracy.

### Strengths
To be honest, I was ready to reject this paper after reading the abstract.  Using gist tokens (i.e. EOS tokens) for in-context compression is a widely used technique -- see the "in-context autoencoder," and "getting to the gist of gisting" (cited by the authors).  Dividing the input into segments is also hardly new; sliding window attention is widely used, and other works (e.g. Melodi), not only do compression at the segment level, but use gist tokens to implement the compression.  

However, the authors have made a number of well-thought-out design decisions in this paper, and the total is more than the sum of its parts.  The main contributions as I see them are:

* Uses a frozen pre-trained LLM as-is, which is very important.
* The compression model is the *same* pre-trained LLM, fine-tuned with LORA so that they share weights.  
* Fine-tuning splits the prompt into context/question/answer, and the distillation loss measures changes in the *hidden states* of all layers for the answer, rather than just the final output.  
* The authors normalize hidden activations, and use a smooth-L1 loss.

These are all non-obvious details that are crucial to the success of the work, IMO.   

IMO, the reason for the *increase* in accuracy is probably because the main LLM is run twice -- once to compress, and once to process the context, and thus has twice the depth.  That's a clever way to reduce any negative impacts from compression.

### Weaknesses
The authors' claims of "4x reduction in prefill costs" are theoretical, and over-stated.   At small context lengths, prefill is actually 2x *more* expensive -- there's a 50% increase from the gist tokens, and another 50% at having to run both the compression and the frozen LLM.  This should be discussed.  The O(N) parts of an LLM are actually quite expensive -- the feed-forward-blocks are large, and in a model like gemini, many layers use local attention, so global attention dominates only at very long sequence lengths.  

Moreover, most uses cases for very long contexts involve many documents concatenated together, in which case block-sparse attention can be used -- a local variant which only attends within each document.  This needs to be discussed.  

Given that prefill is 2x more expensive at small context lengths, and potentially 4x less expensive at large lengths, I am most interested in the crossover point -- the point where compression actually starts providing an advantage.  Unfortunately, the authors' charts are scaled in such a way that you can't see this point.

The overall structure of CompLLM is actually quite similar to a number of earlier works on sparse attention patterns; what is essentially happening is two-level attention -- global attention attends only to gist tokens, which in turn attend locally to the original context.  That should be in the discussion and lit review.

### Questions
What is the crossover point in sequence length?

A 2x compression rate is not very high -- I would like to see some ablations of what happens with a 4x or 8x rate.   

The "getting to the gist of gisting" paper found benefits from evenly spacing the gist tokens throughout the sequence -- it would be nice to see that as an ablation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CompLLM, a framework for efficient long-context question answering by segment-wise soft compression. Long documents are divided into segments; each segment is independently compressed into a small set of concept embeddings (CEs) using a lightweight compressor model. The CEs, together with the user query, are fed to a frozen LLM to generate answers. This segmentation enables linear-time compression, parallel processing, and caching of compressed segments for reuse across queries. Empirical results show comparable or improved accuracy to uncompressed baselines, with substantial reductions in KV-cache size and latency. While the system demonstrates strong practical gains, the conceptual novelty beyond prior work such as AutoCompressor and LLMLingua is limited.

### Strengths
- Simple, scalable design that makes soft-compression linear-time and cacheable.
- Strong empirical performance on long-context QA benchmarks with clear latency and memory gains.
- Compatible with frozen LLMs, making the method easy to integrate into real systems.

### Weaknesses
- Conceptual novelty is limited — largely a segment-wise parallel variant of AutoCompressor.

- Baseline coverage is only partial: while the paper includes LLMLingua2 as a baseline, it omits comparisons to other recent or stronger soft-compression and pruning methods (e.g., AutoCompressor, LongLLMLingua, or token-selection approaches).

- Lacks deeper analysis or theory explaining why compression sometimes improves accuracy (e.g., “attention dilution” remains a hypothesis).

### Questions
How does CompLLM perform on tasks requiring fine-grained token fidelity (e.g., code, math, counting)? Including such cases would better delineate the method’s limitations.

### Soundness
2

### Presentation
3

### Contribution
2
