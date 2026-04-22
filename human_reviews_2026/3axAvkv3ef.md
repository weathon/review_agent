# ARC-Encoder: learning compressed text representations for large language models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Recent techniques such as retrieval-augmented generation or chain-of-thought reasoning have led to longer contexts and increased inference costs. Context compression techniques can reduce these costs, but the most effective approaches require fine-tuning the target model or even modifying its architecture. This can degrade its general abilities when not used for this specific purpose. Here we explore an alternative approach: an encoder that compresses the context into continuous representations which replace token embeddings in decoder LLMs. First, we perform a systematic study of training strategies and architecture choices for the encoder. Our findings led to the design of an Adaptable text Representations Compressor, named ARC-Encoder, which outputs $x$-times fewer continuous representations (typically $x \in $ {4,8}) than text tokens. We evaluate ARC-Encoder across a variety of LLM usage scenarios, ranging from in-context learning to context window extension, on both instruct and base decoders. Results show that ARC-Encoder achieves state-of-the-art performance on several benchmarks while improving computational efficiency at inference. Finally, we demonstrate that our models can be adapted to multiple decoders simultaneously, allowing a single encoder to generalize across different decoder LLMs. This makes ARC-Encoder a flexible and efficient solution for portable encoders that can support multiple LLMs, requiring only small model-specific projectors for adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed a new context compressor, ARC-Encoder to reduce the size of context representations. ARC-Encoder can generalize to many LLMs and show advanced performance on downstream tasks.

### Strengths
1. The paper is well-written and easy to follow. The motivation of this paper is strong and practical. 
2. The method formulation is clear and reasonable.
3. Authors conduct a comprehensive evaluation on many downstream tasks to demonstrate the effectiveness and generalizability of the proposed method.

### Weaknesses
1. Despite of the effectiveness, the idea of context compression has been provided in many previous works, some of which are not included in the referece, e.g., [1][2][3]. This impairs the novelty of the prompsed method.
2. A big issue that resides in context compression is inevitable information loss, especially when being tested on fine-grained retrieval task, such as Needle-in-Haystack (NIAH). The authors should provide more results on these types of tasks to demonstrate their model's  capability in handling fine-grained retrieval tasks.



[1] Chevalier, Alexis, et al. "Adapting language models to compress contexts." arXiv preprint arXiv:2305.14788 (2023).

[2] Han, Wei, et al. "Two are better than one: Context window extension with multi-grained self-injection." arXiv preprint arXiv:2410.19318 (2024).

[3] Zhang, Peitian, et al. "Soaring from 4k to 400k: Extending llm’s context with activation beacon." arXiv preprint arXiv:2401.03462 2.3 (2024): 5.

### Questions
See weakness

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
4

### Summary
This paper proposes ARC-Encoder, a method that compresses input text into compact continuous representations to reduce context length for LLMs. It achieves efficient inference without changing the model (decoder) architecture.

### Strengths
1. The proposed ARC-Encoder offers a solution for context compression, achieving this without altering the underlying LLM architecture.

2. The method demonstrates strong empirical results over different tasks.

### Weaknesses
1. The claim that ARC-Encoder “works seamlessly with multiple LLMs” is overstated, since in practice it still requires fine-tuning separate projectors for each target model, even if the number of parameters is small.

2. The encoder model is very large (e.g., ~3B parameters), which raises serious concerns about practical efficiency. The paper should provide detailed FLOPs and latency analyses to substantiate efficiency claims.

3. It is unclear how the text encoder’s embeddings are initialized, given that different LLMs have distinct embedding spaces. Without careful alignment, transferring across models could lead to catastrophic forgetting or representation collapse.

4. Training such a large encoder (e.g., 3B) to replace token embeddings in decoder LLMs is resource-intensive, involving multiple training stages (pretraining, finetuning, multi-decoder adaptation). The paper should clarify why this approach is preferable to soft compression methods that require LLM fine-tuning.

5. It would further strengthen the paper if the authors could expand the related work discussion to include a recent study that adopts vision encoder for token compression [ref1]. Incorporating this perspective would help situate ARC-Encoder more comprehensively within the evolving landscape of encoder-based compression approaches and clarify its unique contributions.

[ref1] Vision-centric Token Compression in Large Language Model (NeurIPS 2025)

### Questions
1. Are the fine-tuning datasets and downstream evaluation datasets strictly out-of-domain, ensuring a fair assessment of generalization?

2. In the long-context experiments, what is the maximum context length tested, and how does performance scale with length?

3. Does ARC-Encoder also provide benefits in short-context scenarios, or is its advantage limited to long-context settings?

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
4

### Summary
This paper proposes ARC-Encoder, which adopts an architecture consisting of an "encoder (based on Llama3.2 3B, with the output head and causal mask removed) + a 2-layer MLP projector". It performs average pooling on consecutive queries in the last self-attention layer (with a compression factor of 4/8) to generate continuous representations that replace token embeddings in the decoder, without modifying the decoder. The training process employs alternating pretraining tasks of "reconstruction-continuation" and task-specific fine-tuning, enabling a single encoder to adapt to multiple decoders (with dedicated MLP parameters accounting for < 1%). Experiments verify that ARC-Encoder achieves performance close to the open-book baseline on QA, translation, summarization, and long-context tasks. Additionally, the storage size of compressed Wikipedia representations is comparable to that of raw text. Its core contributions lie in high compatibility, multi-decoder adaptation, and long-context extension.

### Strengths
ARC-Encoder does not require decoder modification, enabling adaptation to existing LLMs. For multi-decoder adaptation, only a small amount of parameters are needed, resulting in low deployment costs. It covers both short- and long-context tasks, and memory analysis supports precomputation, indicating great potential for practical application.

### Weaknesses
It has weak innovation: its framework is highly similar to ICAE, and there are no breakthrough designs in multi-decoder adaptation or long-context strategies. Furthermore, it fails to explore performance at high compression factors (16×/32×) and generalization in professional domains, nor does it provide comparisons of inference latency in real-world scenarios.

### Questions
Have the authors compared the performance of ARC-Encoder with similar context compression architectures like ICAE under the same settings? What are the core technical differences between them?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes ARC-Encoder, a plug-and-play encoder that compresses textual context into continuous representations for frozen LLM decoders. The approach aims to reduce inference cost while maintaining downstream performance, without modifying the decoder architecture. The work is comprehensive and empirically strong, with experiments spanning multiple decoders, tasks, and compression factors.

### Strengths
1. This paper introduces a new formulation of context compression that does not alter the decoder. Unlike prior “memory token” or “gist token” methods, ARC-Encoder performs fixed-ratio pooling within the encoder’s attention layers and connects to decoders through a lightweight MLP. This architectural separation is elegant and conceptually clean.
2. The authors conduct a broad and fair evaluation across multiple domains. Results show consistent improvements over strong baselines, often matching or surpassing open-book settings despite heavy compression.
3. The analytical experiments are extensive, and ablations on pretraining tasks, pooling strategies, and encoder truncation are well thought out.

### Weaknesses
1. This paper does not provide a deeper theoretical discussion of why pooled query averaging in attention preserves semantic fidelity or why it outperforms token-level compression. A brief analytical or representational argument could strengthen the paper’s foundation.
2. How sensitive is performance to the dimensionality of the MLP bottleneck?

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
