# Breaking Memory and Communication Barriers in Model-Parallel Fine-Tuning of Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Model parallelism (MP) has emerged as a promising paradigm for distributed large language model (LLM) training across multiple computing nodes.  Yet, almost all existing works about MP focus on first-order methods, which faces two persistent challenges: high communication costs from transmitting activations and gradients, and substantial memory overhead from caching them. Zeroth-order (ZO) methods, by avoiding gradient computation and storage, can naturally alleviate both memory and communication bottlenecks, but they have been largely unexplored in MP for LLM fine-tuning. In this work, we propose ***SparQ***, a ZO MP framework with **Sp**lit layer **a**llocation info**r**med by **Q**uantization-induced activation sparsity, designed to reduce memory and communication costs. *SparQ* builds on three key components: (1) leveraging the gradient-free nature of ZO optimization to eliminate gradient storage and transmission, significantly reducing memory and communication demands incurred by gradients; (2) applying quantization to induce activation sparsity that can be encoded with sparse representations; (3) strategically placing split layers at activation-sparse regions and using sparse representation to lower communication cost from activations almost without compromising model quality. Theoretically, *SparQ* achieves a sublinear convergence rate in non-convex settings, matching that of centralized ZO methods. Empirically, *SparQ* reduces GPU memory usage by over 3× and communication cost by $50\%$+ compared to state-of-the-art baselines, while maintaining comparable model performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tried zero order (ZO) methods on multi-nodes. Then do activation quantization to reduce communication cost.

### Strengths
Good quantitive analysis in experiments.

Nice and informative figure 2

### Weaknesses
1. The authors may not have a deep understanding of model parallelism frameworks themselves. For example, in Page1 L41-42, ZeRO belongs to DeepSpeed, and ZeRO itself is not a open-source "framework"

2. these kind of activation quantization has be well-studied previously, such as ZeRO++[1], LLM-FP4[2] just to name a few. With (FO) or without (ZO) backward does not change anything on the forward propagation of activations. Thus the major novelty of this paper is very similar to existed work. 

3. The reason for ZO not being widely used is due to its instability and may lead to worse model accuracy. The authors do not propose anything to fix this major issue of ZO. 

4. the experiments use vey old models (e.g., llama-1b, gpt2), which can not be representatives of latest LLM results.  

5. Indeed, split layer like pipeline parallelism, there is no any training logic difference compared with training whole model on single device. Thus making the problem itself (i.e. quote page 1 L52-53: "the potential of applying ZO optimization within a MP framework for fine-tuning LLMs remains unexplored.") not very fundamental and meaningful. 

6. some equations are over-decorated. For example, I don't see any one represent such simple Pipeline parallelism idea where one device's layer output serve as next device layer input, to such a complicated equation on L127 of page 3. 

[1] ZeRO++: Extremely Efficient Collective Communication for Large Model Training, ICLR 2024

[2] LLM-FP4: 4-Bit Floating-Point Quantized Transformers, EMNLP 2023

### Questions
N/A

### Soundness
2

### Presentation
2

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
The paper proposes a zeroth-order (ZO) model-parallel framework SparQ that drastically reduces the communication and memory overhead of LLM fine-tuning. SparQ uses gradient-free ZO optimization, paired with quantization-induced activation sparsity to minimize data transfer between partitions. By placing model split layers where activations become sparse after quantization, only a small number of nonzero activations are transmitted in a sparse format, yielding large efficiency gains, while maintaining comparable fine-tuning accuracy.

### Strengths
1.	The paper presents a clear and intuitive motivation, and the overall logic of the work is easy to follow.
2.	By demonstrating consistent results across models of different sizes, the paper provides strong empirical support for the proposed approach and makes the study more convincing overall.
3.	The authors support their system design with a non-convex convergence guarantee matching centralized ZO methods, reinforcing technical soundness beyond empirical evidence.

### Weaknesses
1.	SparQ’s effectiveness heavily depends on activation sparsity after quantization, which holds for ReLU/GELU-based transformers but becomes unstable in layers with normalization, attention, or mixed activations. This makes the “sparsity-guided split” design less generalizable across architectures and limits its scalability to more diverse LLMs.
2.	The paper primarily evaluates on GLUE and SuperGLUE benchmarks, which are relatively simple for LLM fine-tuning under both first- and zeroth-order methods. Including results on more challenging tasks would further strengthen the work. This is a suggestion rather than a weakness.
3.	The authors conduct memory profiling without using gradient checkpointing, which is a common technique to reduce activation memory during backpropagation. This likely overestimates the memory footprint of FO-SGD. Since the main advantage of SparQ lies in its lower memory usage compared to FO-SGD, this comparison becomes less compelling once gradient checkpointing is enabled.
4.	Although the authors claim that their framework can also be easily extended to scenarios with multiple computing nodes (M > 2). However, I haven't seen any experimental results when M>2, so the results only when M=2 are not convincing enough for me.

### Questions
1.	How would results look like when if running FO-SGD with gradient checkpointing?
2.	The 4-Bit Quantized Activations Comparison (Fig4) is very interesting, and I'm curious about how the author defines trainable and untrainable. If you define it by accuracy, I'd like to know what the accuracy is when it's untrainable compared to when it's trainable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes SparQ, a ZO MP framework with Split layer allocation informed by Quantization-induced activation sparsity. The paper proves that SparQ achieves a sublinear convergence rate in non-convex settings and empircally shows that it reduces GPU and communication overheads.

### Strengths
1. The paper proves that SparQ across multiple nodes achieves a sublinear convergence rate of for non-convex functions.
2. The evaluation provides test accuracies of their method accross models and datasets.

### Weaknesses
1. The experiments do not appear to include real-world distributed training across multiple machines, making it difficult to reflect the end-to-end training time advantages of SparQ in real-world scenarios. Meanwhile, memory cost and communication cost can be easily calculate without experiments in real-world hardware settings.

### Questions
What hardware environments did you use in your experiments?

What is the expected improvement in end-to-end convergence speed?

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
4

### Summary
This paper introduces SparQ, a framework that combines zeroth-order (ZO) optimization, quantization-induced activation sparsity, and sparsity-aware split-layer allocation to reduce memory and communication overhead during model-parallel (MP) fine-tuning of large language models (LLMs). The key observation is that post-quantization activations (after ReLU, GELU, SwiGLU) become highly sparse, enabling sparse transmission between partitions.

### Strengths
1. Clearly motivated by real bottlenecks in model-parallel fine-tuning.

2. Integrates ZO optimization and activation quantization in a coherent framework.

3. Provides a theoretical convergence analysis under compression.

4. Demonstrates substantial GPU memory and communication savings with minor performance loss.

5. Ablation studies on quantization levels and split positions provide useful insights.

### Weaknesses
1. The evaluation lacks direct comparisons against strong existing baselines such as [1] or [2]. Since these methods are designed for similar purposes—reducing memory and communication in fine-tuning—the absence of head-to-head experiments makes it difficult to quantify SparQ’s real advantage.

[1] Zeroth-Order Fine-Tuning of LLMs with Extreme Sparsity

[2] ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory

2. The largest model tested is 6.7B parameters (OPT-6.7B). In practice, such a size can often fit on a single GPU or two GPUs with ZeRO-2/ZeRO-3, and rarely requires model parallelism. If the proposed method is specifically targeting MP fine-tuning, stronger evidence on larger-scale (>13B or multi-node) models is expected. The current scale undermines the relevance of MP in this context.

3. Although communication and memory reductions are clearly reported, there are no measurements of actual training speed, wall-clock time, or GPU utilization. ZO optimization introduces multiple forward passes per step, which may offset the savings. Without runtime comparisons, the claimed “efficiency” remains uncertain.

4. The convergence proof assumes unbiased and bounded compression noise, yet sparse quantization is inherently biased. The paper does not measure or analyze the effect of this bias on convergence or final accuracy.

5. The evaluation does not cover multi-split (M>2) or multi-node settings, and no communication profiling under realistic hardware interconnects (e.g., NVLink, RDMA, Ethernet) is presented. Thus, the practical scalability and deployment feasibility are unclear.

6. The default configuration (4-bit quantization) is fixed throughout, but the trade-off between bitwidth, accuracy, and communication volume is not fully explored. It remains unclear whether the observed gains hold under different precision levels or tasks.

7. The cost of sparse activation encoding/decoding and synchronization is omitted. These may introduce non-negligible overhead that counteracts communication savings. More system-level profiling (kernel time, communication overlap) would strengthen the paper’s claims.

### Questions
1. Can you report end-to-end runtime comparisons (e.g., total fine-tuning time vs. MeZO, ZO-Adam, or first-order methods)?

2. Does increasing P significantly affect the communication–accuracy trade-off?

3. How large is the bias introduced by the sparse quantizer, and does it affect convergence?

4. Can SparQ scale to multi-node / multi-partition (M > 2) configurations?

5. How does SparQ perform on larger models (> 70B) that truly require MP fine-tuning?

6. What is the computational cost of sparse activation reconstruction during forward/backward passes?

### Soundness
3

### Presentation
3

### Contribution
2
