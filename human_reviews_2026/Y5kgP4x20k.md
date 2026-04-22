# SparseSkeleton: Prefill sparse attention by decomposition

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Multi-head attention (MHA) and grouped query head attention (GQA) consti-
tute essential architectural components of modern large language models (LLMs).
Even though attention computations remain relatively inexpensive for small-scale
inputs, the computational cost increases quadratically as the input size expands.
In long-context scenarios, including tasks such as book-level summarization or
code repos analysis, time-to-first-token (TTFT) performance can deteriorate sig-
nificantly. Although various studies have improved prefill stage performance by
exploiting sparsity structure, sparsity can still be further increased with structure
refinements.
In this work, we propose an approximate on-line decomposition of the attention
matrix which is able to dynamically identify additional sparsity. The attention
matrix is decomposed into three components: a slash component, a vertical com-
ponent, and a horizontal component. Each component requires only linear space,
thereby enabling more efficient processing compared to the full attention matrix.
The decomposition is computed from query and key tokens using a linear-time
algorithm. The statistical properties of the decomposition allow generation of the
mask by merely selecting elements that exceed a threshold. The threshold itself
can be chosen to limit the difference with regular dense attention or to respect a
certain time-budget.
We demonstrate that this technique can be directly applied – without requiring
retraining – to networks employing standard dense attention mechanisms (MHA,
GQA) and RoPE. We show that precision is maintained across the ∞Bench and
PG-19 benchmarks for LLAMA-3-8B-INSTRUCT-1048K. Furthermore, we ob-
serve substantial increases in sparsity and corresponding speedup compared to
previous methods. We halve the number of FLOP relative to State-of-the-Art on
one million tokens.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SparseSkeleton, a training-free, online prefill sparse-attention method that decomposes the attention matrix into three factors: slash, vertical, and horizontal, and then builds a block-sparse mask by thresholding or by meeting a time-budget target. It supports MHA/GQA with RoPE without retraining and integrates into vLLM with custom kernels for the prefill phase. Experiments on PG-19 and ∞Bench with Llama-3-8B-Instruct-1048K show comparable accuracy to dense and prior sparse-prefill baselines (including MInference) while further reducing prefill PFLOPs.

### Strengths
1. The design is training-free, with no model modification.
2. The slash/vertical/horizontal factorization is simple, statistically motivated, and maps cleanly to block-sparse execution with either an error-threshold or budget control.
3. Authors integrated the method into vLLM with custom kernels for prefill, demonstrating the engineering practicality.

### Weaknesses
1.  Although well-packaged and more unified, the proposed method appears to be a reformulation and modest generalization of MInference. The core ideas of combining diagonal and vertical structures to sparsify prefill attention are already central to prior work. The “horizontal” term is a reasonable extension, but its essentialness is not convincingly demonstrated.
2. The paper emphasizes PFLOPs and mask density, but lacks the TTFT evaluation, which is important for real acceleration.
3. On ∞Bench, the method underperforms MInference on some retrieval-heavy tasks (e.g., Retrieval.KV: 1.40 vs 10.20), indicating possible over-pruning of long-range signals. This raises concerns about the robustness of the sparsity pattern across diverse workloads.
4. The paper lacks strong ablation studies to justify the necessity of all three components. It remains unclear how much the horizontal term contributes relative to slash + vertical (as in MInference), or whether the added complexity is always worth the compute.
5. Only evaluated on a single model, single scale. The method is only tested on LLaMA-3-8B-Instruct. It remains uncertain how well this scales to larger or smaller models (e.g., 1.7B, 14B, 32B).

### Questions
1. What is the marginal gain of the horizontal term? Could you provide an ablation where A_h is removed or replaced with a constant, and quantify its effect on FLOPs and accuracy?
2. What is the actual runtime breakdown and the e2e latency? Could you quantify the proportion of total prefill time spent in decomposition vs attention computation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new sparse attention mechanism based on online decomposition. It decomposes the standard attention computation into three components: *slash*, *vertical*, and *horizontal*. The product of these components is used to construct a block-sparse mask during prefill. Experiments validate the FLOPs and performance of the proposed method.

### Strengths
1. The proposed method is clearly formulated and well-explained. Alternative designs and approximations are discussed systematically.
2. The method demonstrates improved average performance and reduced FLOPs compared to existing sparse attention baselines across evaluated benchmarks.

### Weaknesses
1. Limited experiments: The paper lacks sufficient empirical validation. Although FLOP reduction is reported, no end-to-end wall-clock speedup is provided, leaving readers uncertain about the practical efficiency gains. Moreover, ablation studies analyzing the contribution of each component are missing. The paper also does not analyze the resulting sparse patterns in detail. The experiment section is the main weakness of the paper and dominates the reviewer's final decision.

2. Presentation: The discussion of the method is somewhat flat, without emphasizing the key principles or design motivations early on. This structure may make it difficult for readers to grasp the main ideas before diving into implementation details.

### Questions
1. What is the end-to-end wall-clock speedup achieved by the proposed method compared to dense attention?

2. In which types of tasks or data domains does the proposed method particularly excel compared to other sparse attention methods?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SparseSkeleton, a training-free, online prefill sparsification scheme that factorizes the attention matrix into three interpretable components, slash (Toeplitz/diagonal), vertical (columnwise), and horizontal (rowwise), and forms a block-sparse prefill mask by thresholding the product of these factors. The method is designed for MHA/GQA with RoPE and is integrated into vLLM with custom CUDA/Triton kernels for prefill; decoding remains dense (FlashAttention). Experiments on Llama-3-8B-Instruct-1048K report similar PG-19 perplexity to dense and prior sparse-prefill baselines, reduced prefill PFLOPs, and ∞Bench results that are mixed across tasks.

### Strengths
The paper is clearly written and presents a well-motivated approach to sparse prefill attention through a simple and interpretable factorization into slash, vertical, and horizontal components. This decomposition is intuitive, mathematically grounded, and maps naturally to block-sparse execution. The method is training-free and integrates smoothly into vLLM with custom kernels, demonstrating good engineering practicality. Experimental results show reduced prefill PFLOPs with comparable perplexity to dense and prior sparse baselines, suggesting the approach can achieve meaningful compute savings without retraining. Overall, the paper's clarity, sound formulation, and practical integration are notable strengths.

### Weaknesses
While the method is well-formulated, its empirical validation is limited. The paper focuses heavily on theoretical FLOP reduction and mask sparsity but does not provide end-to-end runtime measurements such as TTFT or wall-clock latency. Without this evidence, it remains unclear whether the proposed method offers real-world acceleration beyond simulated efficiency gains. Additionally, the evaluation is restricted to a single model scale (Llama-3-8B-Instruct), leaving questions about generality and scalability unanswered.

The contribution over prior work, particularly MInference, is also somewhat incremental. The slash and vertical components largely replicate existing ideas, and while the horizontal factor is novel, its necessity and impact are not convincingly demonstrated. The absence of ablation studies isolating this term makes it difficult to assess how much it contributes to performance or sparsity improvements.

Finally, the paper's results on ∞Bench reveal inconsistent behavior, especially in retrieval-heavy tasks, where the method underperforms compared to prior approaches. This suggests potential over-pruning of long-range dependencies and raises concerns about robustness across task types. Together, these limitations weaken the overall empirical strength and make the contribution appear less substantial than it could be with more comprehensive experimentation and analysis.

### Questions
* What is the actual end-to-end speedup (TTFT or wall-clock prefill latency) compared to dense attention and existing sparse prefill baselines such as MInference?
* How much does the horizontal component contribute to accuracy and sparsity? Could you provide an ablation removing or simplifying this term?
* How does the runtime cost of the decomposition (computing slash, vertical, and horizontal factors) compare to the attention computation itself?
* Have you tested the method on different model scales or architectures to evaluate generality beyond Llama-3-8B?
* What factors explain the performance drop on retrieval-heavy tasks in ∞Bench, and could the mask generation be adjusted to preserve long-range attention in those cases?

### Soundness
2

### Presentation
3

### Contribution
2
