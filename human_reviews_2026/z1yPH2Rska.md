# Diagonal Batching Unlocks Parallelism in Recurrent Memory Transformers for Long Contexts

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 4, 8

## Abstract
Long-context inference with Transformers is constrained by quadratic attention
and linear memory growth. Many linear-time alternatives require pretraining from
scratch, whereas Recurrent Memory Transformers (RMTs) convert pretrained 
models into segment-recurrent variants via finetuning without modifying the original
model architecture. However, their sequential memory updates underutilize GPUs.
We show that RMT-style architectures with layer-level memory (PRMTs) 
(e.g., ARMT) can be among the most latency-efficient linear approaches when scheduled
properly. We introduce Diagonal Batching, a compute-reordering scheme that
preserves exact recurrence while exposing inter-step parallelism by executing 
"diagonals" concurrently with grouped layers. On LLaMA (1B/3B/8B) up to 131,072
tokens on A100/H100, Diagonal Batching achieves up to 3.3× lower latency than
full-attention inference and 1.8× over a sequential ARMT baseline, with no 
custom CUDA kernels. With the right scheduling, PRMTs achieve linear scaling with
context length and stand out as competitive, scalable architectures among linear
recurrent models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents diagonal batching, a method for training and inference of multi-layered recurrent sequence architectures (recurrent memory transformers), where parallelization happens along the diagonal of subsequent segment/token and model layers. The operations across layers can be fused in a batched GEMM-kernels reducing the scheduling overhead. The authors show that their method leads to significant speed-ups for long sequence lengths.

### Strengths
- general method for sequence-recurrent architectures
- good speed ups (1.1-3.3) for the pre-fill part (latency) of inference

### Weaknesses
- limited application to pre-filling (latency) optimization of recurrent LLMs
- no practical RMT model (e.g. an existing RWKV/xLSTM/Mamba-based model) shown where this is applied (e.g. for reasoning tasks, where long sequences would be strongly beneficial)
- RMTs in general break translational invariance in text, so some tokens would be "different" from others depending on their positions (especially at the segment border)

### Questions
- Are there existing RMT models where this can have practical benefits?
- Could you show how this could be implemented on large recurrent models, like FalconMamba-7B [1] or xLSTM-7B [2] for practical use?
- Can this also be beneficial in model training on long sequences?
- Can you add a small section on why this method does not work for decoding/generation?
- you mention TC0 model complexity as a general downside of Transformers/State Space Models, but don't position your method along this axis with general state-tracking enabled architectures like xLSTM [3] or DeltaProduct [4], how does it relate?

[1] https://huggingface.co/tiiuae/falcon-mamba-7b

[2] https://huggingface.co/NX-AI/xLSTM-7b

[3] Beck et al. (2024): Extended Long Short-Term Memory

[4] Siems et al. (2025): DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Diagonal Batching, a scheduling method that improves GPU utilization in Parallel Recurrent Memory Transformers (PRMTs), a class of models that maintain per-layer recurrent memory, such as ARMT. The key idea is to reorder computation across layers and segments into “diagonals,” allowing concurrent execution of operations that were previously serialized, without breaking exact recurrence. The authors implement the method in the ARMT framework and evaluate it on LLaMA-based models ranging from 1B to 8B parameters, showing up to 3.3× speedup over standard full-attention inference and 1.8× over sequential ARMT, without requiring custom CUDA kernels. The paper argues that compute scheduling, rather than algorithmic complexity, is the main bottleneck in RMT-style architectures and that Diagonal Batching offers a practical path to efficient, exact linear-time inference for long-context models.

### Strengths
1. Practical relevance – The paper addresses a real bottleneck: GPU underutilization during long-context inference in memory-augmented transformers. The proposal is pragmatic, compatible with existing hardware, and doesn’t require custom CUDA, which makes it accessible.

2. Elegant scheduling insight – The diagonal reordering idea is conceptually simple yet powerful, exposing latent parallelism while preserving recurrence—an often tricky balance.

3. Strong empirical evidence – The results convincingly show latency reduction across multiple model sizes and segment lengths. The experiments are thorough, including FLOPS scaling, micro-batching comparison, and numerical drift analysis.

4. Clarity of technical exposition – The method is mathematically and algorithmically well explained, with clear diagrams (e.g., Figure 1 and 2) that make the scheduling logic intuitive.

5. Compatibility with existing optimizations – The method integrates with FlashAttention, grouped GEMMs, and standard PyTorch implementations, making it easy to adopt.

### Weaknesses
1. Incremental nature – The main novelty lies in scheduling, not modeling or theory. While the implementation is clever, the conceptual leap from standard pipelining or grouped execution is limited. The paper frames this as a major innovation, but it’s more of an engineering optimization than a new algorithmic idea.

2. Limited empirical diversity – All experiments are on LLaMA-based ARMTs. There is no exploration of how the method generalizes to other PRMT architectures (e.g., RWKV or Mamba) beyond brief mentions. Without such experiments, claims of broad applicability remain untested.

3. Reproducibility and accessibility – Although the authors claim to release code, there’s little clarity on integration with existing toolchains or benchmarks. The heavy reliance on ARMT-specific infrastructure might limit reproducibility for broader research use.

4. Evaluation bias – The comparisons are mostly latency-based. There’s no strong discussion of trade-offs in throughput, numerical stability, or memory overhead under multi-request workloads (a realistic serving scenario).

5. Overemphasis on speedups – Some reported gains (like 3.3×) are cherry-picked from small segment sizes or single-GPU setups. Scaling trends on multi-GPU or distributed systems are not shown, yet such setups are where scheduling optimizations often hit diminishing returns.

6. Writing style and tone – The paper reads more like an extended technical report than a standard ICLR submission. I would be more in favor of making the paper concise and 'to-the-point'.

### Questions
1. Have you tested Diagonal Batching on other PRMT variants like RWKV or Mamba to verify general applicability?

2. How does the method perform under concurrent multi-user loads, where GPU memory fragmentation and kernel scheduling might differ?

3. Could you provide memory overhead statistics—does grouped GEMM increase peak memory usage?

4. How significant are the numerical drifts beyond 64K tokens in real tasks?

5. Have you tried combining Diagonal Batching with quantized or sparsified models (e.g., AWQ or FlashDecoding)?

### Soundness
2

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
2

### Summary
[Disclaimer that I worked on this field in 2023 and 2024 and have not kept up to date with the latest trends in 2025, so please take this review with a grain of salt.]

The paper introduces a new scheduling scheme for recurrent transformers, which the authors call "Diagonal Batching". The framing is as follows: RMTs have linear inference but are sequential and hence not easily parallelizable. Parallel RMTs localize recurrence within layers and eliminate all inter-layer memory flow. The issue is that PRMTs underutilize GPUs for single, long input requests. The paper introduces diagonal batching as a way to utilizes the GPUs better and hence improve latency. This is done by rearranging the layers and segments into a diagonal structure, where parallel operations can occur at the same time but across different layers.

The authors show that this provides better latency, especially in long token settings, though the results are less pronounced in the short token settings. They show that results hold across scales. The authors also provide an error analysis which shows that the error accumulation of the method is minimal.

### Strengths
- good exposition of background information and framing of the contributions of the paper
- proposed method seems simple and can be applied quite generally across different models. If indeed true, the method provides a lot of latency gains basically for free.
- method section seems quite complete. The authors go through both the high-level motivations but also outline the implementation details.
- experiments section seems complete as well. The authors show comparisons with different scales, sequence lengths, and baselines.

### Weaknesses
- The method is actually slower for sequence length 4096 and 8192.
- The tables and figures mainly show latency. I would have wanted to see the effect also on other metrics like memory or GPU utilization.
- The paper reported error accumulation numbers, but I would have wanted to also see the actual effect on the produced tokens, or maybe just verify that scores on some common benchmarks like MMLU remain the same.

### Questions
- How would these results change with smaller/larger GPUs? Every few years, new GPUs with larger memory and faster processing come out, so I'm curious if these would affect the results.
- Why is the error accumulation only upto 32k sequences? It seems like the main benefits of the method are more pronounced for longer sequences, so an error accumulation at these scales would be good to see as well.

### Soundness
3

### Presentation
4

### Contribution
3
