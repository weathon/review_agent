# Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via MoEs

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Despite the significant research efforts, subquadratic attention methods still suffer from inferior performance in practice. 
We hypothesize that dynamic, learned content-based sparsity can lead to more efficient attention mechanisms.
We present Mixture of Sparse Attention (MoSA), a novel approach inspired by Mixture of Experts (MoE). MoSA dynamically selects tokens for each attention head, allowing arbitrary sparse attention patterns.
By selecting $k$ tokens from a sequence of length $T$, MoSA reduces the computational complexity of each attention head from $O(T^2)$ to $O(k^2+T)$. This enables using more heads within the same computational budget, allowing higher specialization. We show that among the tested sparse attention variants, MoSA is the only one that can outperform the dense baseline, sometimes with up to 27\% better perplexity for an identical compute budget. 
MoSA can also reduce the resource usage compared to dense self-attention. 
Despite using torch implementation without an optimized kernel, perplexity-matched MoSA models are simultaneously faster in wall-clock time, require less memory for training, and drastically reduce the size of the KV-cache compared to the dense transformer baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Mixture of Sparse Attention (MoSA), a novel sparse attention mechanism designed to reduce the quadratic computational cost of standard self-attention while maintaining or even improving model performance.

### Strengths
- **Inspired by Mixture of Experts (MoE)** with **expert-choice routing**, where each attention head acts as an expert and selects its own set of \(k\) tokens from the sequence.
- **Dynamic, content-based sparsity**: Each head learns which tokens to attend to via a trainable router.
- **Complexity reduction**: From $O(T^2)$ to $O(k^2 + T)$ per head.
- **Hybrid design**: Combines a few dense heads with many sparse MoSA heads for stability and performance.

### Weaknesses
* Short Sequence Struggle: The model was trained on long sequences (T=1024) but evaluated on downstream tasks with very short sequences (e.g., BLiMP examples are often <10 tokens). In these cases, the token selection mechanism is forced to operate out-of-distribution. Selecting 2 tokens from a 10-token sentence (20%) is fundamentally different from selecting 16 tokens from a 1024-token sequence (1.56%), leading to a significant performance drop on tasks like BLiMP.

* Expert Overspecialization: This is a known issue in Mixture-of-Experts (MoE) models. While the highly specialized MoSA heads excel at the language modeling pre-training objective (hence the low perplexity), they may fail to generalize to diverse downstream tasks that require different reasoning patterns. 

* Toy Model Experiments: The largest model tested has 516M parameters, which is considered a "toy model" by today's standards for LLM research. The field's focus has shifted to models of 7B parameters and larger. The paper does not demonstrate that MoSA's benefits (or its stability issues) hold at these realistic, larger scales. Performance and behavior can change dramatically with scale, so the conclusions are preliminary until validated on larger models.

* Anomalous Long-Sequence Results (Fig. 4): The results in Figure 4 are counter-intuitive and require deeper discussion. Perplexity is expected to increase (get worse) as sequence length grows because predicting the next token in a longer, more complex context is harder. However, the figure shows perplexity decreasing for all methods as the sequence length increases from 1024 to 8192.

* Narrow Downstream Benchmark Suite: The evaluation on downstream tasks is limited to only six benchmarks (LAMBADA, WinoGrande, BLiMP, HellaSwag, PIQA, AI2ARC). It lacks a broader range of challenging evaluations that are now standard, such as:
   - Reasoning Tasks: (e.g., GSM8K, MATH)
   - Knowledge-Intensive Tasks: (e.g., MMLU, TriviaQA)
   - Code Generation: (e.g., HumanEval)
   - Massive Multi-Task Benchmarks: (e.g., BIG-Bench Hard).

   This limited scope makes it difficult to fully assess the model's capabilities and the true impact of the MoSA architecture.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

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
This paper propose an architecture combing the concept of MoE with head selection of attention computation. MoSA performs experter-routing, (experts choose the topk tokens) while MoSAIC performs token-routing, (tokens choose the expert).

### Strengths
The idea of expert-routing for attention is ineresting, but unfortunately, it seem not suitable to fit for modern auto-regressive generation.

### Weaknesses
- The evaluation is unsound, as PPL is used for most part. It's widly known that PPL is not affected by attention a lot. You can do many crazy sparse attention algorithms with PPL in a reasonble range. 

- The concept of MoSA/MoSAIC is not seperated clearly. I believe some of the MoSAIC's concept like KV cache is missused in MoSA.

- The speedup evaluation setup is not clear.

### Questions
1. Why MoSA has KV cache? I think it's not auto-regressive.
2. How is KV cache being managed in MoSAIC? Do you only keep k-tokens on each head? If so, what is the eviction KV cache algorithm being used? 
3. Can you show results on DROP and GSM-8k in the benchmark? It would be better if you also include ruler. The benchmarks you used in the current evaluation can not reflect the attention ability well. 
4. How's the wall clock speedup baseline being measure? Please show the setting/framework using used and differentiate prefill/decode case. 
5. Can you also analyze the communication overhead of MoSA/MoSAIC with TP (tensor-parallel) where different head are placed in different GPUs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a "Learnable Sparse Attention" adopted from MoE designs and claims it performs better than "Fixed" and "routed" ones.

### Strengths
Clear and informative figures. The visualizations (e.g., Fig. 1) illustrate the workflow of dense versus MoSA attention. and the paper is well written and logically structured.

### Weaknesses
Poor baselines. The experimental comparison is weak. The paper only compares MoSA with fixed sparse attention and Routing Transformer–style baselines, omitting stronger and more recent sparse attention methods such as NSA, MoBA, DuoAttention, XAttention, SeerAttention, and MInference. Without these, the claim that “MoSA consistently outperforms dense attention” is not convincing, it only holds under a limited and outdated baseline set.

Lack of novelty. The claimed novelty, “a Mixture of Sparse Attention (MoSA) inspired by Mixture of Experts with expert-choice routing” (from the introduction), is questionable. Similar ideas have already appeared in NSA, MoBA, SeerAttention, which first use dynamic sparsity and content-based gating. The paper does not clearly differentiate MoSA from these prior works or explain what unique advantage it brings.

### Questions
Provide more comparison with SOTA works.

Explain what the difference between you and other previous "learnable sparse attention" works.

Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention, https://arxiv.org/abs/2502.11089

DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads, https://arxiv.org/abs/2410.10819

XAttention: Block Sparse Attention with Antidiagonal Scoring, https://arxiv.org/abs/2503.16428

SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs, https://arxiv.org/abs/2410.13276

MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention, https://arxiv.org/abs/2407.02490

### Soundness
2

### Presentation
3

### Contribution
2
