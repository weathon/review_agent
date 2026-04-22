# Rectified Sparse Attention for Efficient Long-Sequence Generation

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 6

## Abstract
Efficient long-sequence generation is a critical challenge for Large Language Models. While recent sparse decoding methods improve efficiency, they suffer from KV cache misalignment, where approximation errors accumulate and degrade generation quality. In this work, we propose Rectified Sparse Attention (ReSA), a simple yet effective method that combines block-sparse attention with periodic dense rectification. By refreshing the KV cache at fixed intervals using a dense forward pass, ReSA bounds error accumulation and preserves alignment with the pretraining distribution. Experiments across math reasoning, language modeling, and retrieval tasks demonstrate that ReSA achieves near-lossless generation quality with significantly improved efficiency. Notably, ReSA delivers up to 3.77x
 end-to-end speedup under decoding at 256K sequence length, making it a practical solution for scalable long-context inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Rectified Sparse Attention (ReSA) addresses the key challenge of error accumulation in sparse decoding for long-sequence LLMs. While prior methods like Quest and RetroAttention reduce computation via sparse attention, they suffer from KV-cache misalignment—where approximation errors gradually degrade generation quality. ReSA proposes a two-phase hybrid framework combining group block-sparse attention for fast decoding and periodic dense rectification to correct accumulated errors.

### Strengths
1. Simple yet principled design:ReSA introduces a conceptually simple but effective hybrid pipeline
2. Strong empirical performance: Achieves near-lossless generation quality compared to dense attention across diverse tasks
3. Strong clarity and structure: The paper is well-written and organized, with step-by-step algorithm descriptions, pseudo-code and clear presentation of results, making the method easy to understand and reproduce.

### Weaknesses
1. Potential inefficiency in streaming or real-time decoding: Since rectification requires a dense re-encoding of the last f tokens, the method might be unsuitable for low-latency, token-by-token generation
2. Limited diversity of test models: Experiments are performed on a narrow set of LLMs, without validation on decoder architectures with different attention layouts
3. Rectification overhead underreported: The dense rectification step adds non-trivial cost. It is unclear how much latency each dense phase incurs in isolation, or how it scales with generation length, since this paper target on the misalignment along decoding steps

### Questions
1. Does ReSA generalize across model architectures like llama, mistral?
2. Can the authors provide a per-rectification latency breakdown and its scaling trend as generation length grows? 
3. What is the number of generated tokens for benchmark datasets?

### Soundness
4

### Presentation
4

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
To address the bottlenecks of high memory overhead in standard autoregressive decoding and degraded generation quality from accumulated KV cache errors in existing sparse decoding for LLM long-sequence generation, this paper proposes Rectified Sparse Attention (ReSA), a training-free method. ReSA combines Group Block-Sparse Attention (GBSA)—which partitions the KV cache into blocks, uses statistical descriptors to dynamically select relevant blocks, and shares sparse patterns within GQA groups for efficient decoding—with periodic dense rectification that refreshes the recent KV cache via dense attention every 32 steps to bound error accumulation.

### Strengths
- **Training-Free & Easy Deployment**: ReSA is a training-agnostic framework that requires no extra fine-tuning or reconstruction of LLMs, enabling direct integration into existing LLM inference workflows. It also supports modern LLM serving optimizations (e.g., continuous batching, chunked prefill) without modifying the underlying service architecture, lowering engineering implementation barriers.  
- **Balanced Efficiency & Generation Quality**: By adopting Group Block-Sparse Attention (GBSA) to reduce computational and memory costs, and periodic dense rectification to limit error accumulation from sparse decoding, ReSA achieves up to 3.77× end-to-end speedup at 256K sequence length. Meanwhile, its performance in tasks like math reasoning and language modeling is close to the dense attention baseline, breaking the traditional "efficiency vs. quality" trade-off.  
- **Strong Hardware Adaptability**: A custom GPU kernel tailored for Group-Query Attention (GQA) assigns each GQA group to an independent Streaming Multiprocessor (SM), minimizing inter-SM communication. Additionally, block partitioning and dynamic block selection optimize memory access patterns, making latency growth much slower than dense decoding as sequence length increases—well-adapted to the hardware needs of long-sequence inference.

### Weaknesses
- **Limited Experimental Scope**: Experiments only use mid-sized models (e.g., Qwen2.5, DeepSeek-R1-Qwen-Distill, max 7B parameters) and exclude ultra-large LLMs (70B+). Tasks are restricted to math reasoning, language modeling, and retrieval, with no validation in non-technical scenarios (e.g., dialogue generation, creative writing), failing to fully prove its versatility across diverse LLM applications.  
- **Dependence on Manual Hyperparameter Tuning**: Core parameters (block size = 16, rectification frequency = 32, sparsity ratio = 0.9) are set manually, with no adaptive adjustment mechanism. For different sequence lengths (e.g., extremely short sequences < 32 tokens) or task types, fixed parameters may cause redundant rectification overhead (short sequences) or insufficient error control (long sequences), increasing tuning costs in practice.  
- **Unverified Overhead in Extremely Long Sequences**: While rectification overhead accounts for only 14% at 256K sequences, performance at ultra-long sequences (e.g., millions of tokens) is untested. As sequence length scales to extremely large sizes, cumulative rectification overhead may rise, and the accuracy of block selection for sparse attention (e.g., due to over-dispersed context) may decline—issues not further analyzed.

### Questions
I would be happy to increase my rating if my views are given a thorough discussion.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Rectified Sparse Attention (ReSA) for long-sequence generation. It combines group block-sparse decoding with periodic dense “rectification” that re-encodes the last f tokens to refresh the KV cache and limit error accumulation. The authors claim near-lossless quality with up to 3.77× end-to-end speedups at 256K context length, and present results on math reasoning, language modeling perplexity, and RULER retrieval, plus a custom kernel and Nano-vLLM integration.

### Strengths
- Straightforward algorithmic interface to existing decoding stacks; the method is training-free and compatible with batching and chunked prefill.

- Clear articulation of the KV misalignment problem in sparse decoding.

### Weaknesses
- Novelty issues: Periodic dense recomputation to correct sparse approximation resembles verification steps in speculative and self-speculative decoding. The paper argues away accept/reject control but does not convincingly separate the idea from existing rectification/verification motifs beyond implementation details. The conceptual delta over “sparse for speed + periodic dense for correctness” is thin. 

- Efficiency uses Nano-vLLM and a shared KV cache across layers “to prevent memory overflow,” which is not a default serving configuration and can distort memory traffic profiles and scaling. 

- The fairness of speedup claims under this altered regime is unclear. 

- The paper cites SeerAttention and training-aware NSA/MoBA but does not compare against them under equalized quality or equalized latency settings.

- Accuracy and perplexity do not cover long-form coherence, factuality, and failure under distribution shift. The claim of “near-lossless generation quality” should be backed by longer outputs, multi-turn contexts, and error analyses, not only exam-style benchmarks.

- The Flash-Decoding integration remains high level. There is no open-sourced kernel, profiling against strong dense baselines like FlashAttention-3, or evaluation across GPUs. This weakens the practicality argument.

### Questions
- How does ReSA compare against SeerAttention (fine-tuned block keys) and NSA/MoBA under equal-latency and equal-quality protocols? Please include strong speculative/self-speculative baselines with optimized controllers.

- What happens when f is increased beyond 64–128 under strict latency budgets? Provide failure cases where rectification is too infrequent and quantify quality decay over 128K–1M tokens.

- Can this work be integrated with SGLang’s prefix cache? This is the key point.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ReSA, a mechanism combining block-sparse attention with periodic dense “refresh” passes to mitigate the error accumulation in the KV cache during sparse decoding.

ReSA maintains nearly lossless generation quality while achieving up to 2.4× speed-up compared with standard dense attention.

### Strengths
- the idea is very simple
- works across different datasets
- paper provides an efficient implementation (using `nano-vLLM`) for ReSA that gives wall-clock speed ups

### Weaknesses
- there are only compute savings, and not memory savings during inference. this is especially important since most GPUs are memory bound and not compute bound

### Questions
- what is the latency of baselines compared with ReSA ? authors need not provide latency for H2O and streamingLLM since they are significantly worse, but i'd be curious to see latency of BlockSparse.
- just curious, other than H2O and streamingLLM, are there no other competitve baselines out there except BlockSparse ?
- in figure 8, why does f=128 sometimes outperform f=64 or f=16, especially when sparsity ratio is 0.9? any thoughts/intuitions on this?

### Soundness
3

### Presentation
3

### Contribution
3
