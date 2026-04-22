# Unlocking Full Efficiency of Token Filtering in Large Language Model Training

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Token filtering has been proposed to enhance the utility of large language models (LLMs) by eliminating inconsequential tokens during training. While using fewer tokens is expected to reduce computational workloads, existing methods have not yet achieved a real-world efficiency boost. This is primarily due to two factors: (1) existing work has inadequate sparsity for speedup, and (2) token filtering operates within a sparsity range that is non-standard in existing machine learning (ML) libraries and thus cannot be efficiently supported. This paper presents Centrifuge, a system that leverages algorithm and system co-design to unleash the full efficiency of token filtering in LLM training. At the algorithm level, Centrifuge filters activations of inconsequential tokens in the attention backward kernel to amplify the sparsity in backward computation. At the system level, Centrifuge proposes an automatic workflow that transforms sparse GEMM into dimension-reduced dense GEMM for optimized efficiency using standard ML libraries. Evaluations on models with various scales—from 1.1B to 40B—demonstrate that Centrifuge reduces backpropagation time by up to 49.9\% and end-to-end training time by up to 34.7\% when filtering 50\% of tokens. Utility assessments indicate that Centrifuge preserves the utility benefits of token filtering and significantly enhances model performance by up to 26.6\% compared to standard training. Centrifuge is designed for seamless integration into existing LLM training frameworks, enabling systems already utilizing token filtering to accelerate training with just one line of code.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents CENTRIFUGE, a system that combines algorithm and system-level optimizations to improve the efficiency of token filtering in LLM training. The authors identify two key limitations of existing token filtering approaches: (1) inadequate sparsity propagation beyond the output layer, and (2) mismatch between token filtering sparsity (30-50%) and what sparse GEMM libraries can efficiently handle (>95%). CENTRIFUGE addresses these through filtering activations in attention backward kernels and transforming sparse GEMM into dimension-reduced dense GEMM, achieving up to 49% backward time reduction.

### Strengths
1. The paper clearly identifies why existing token filtering methods fail to deliver efficiency gains despite their utility benefits, with strong empirical evidence (Figure 2, Section 3).

2. Testing across four models (1.1B-8B parameters) and multiple training scenarios (DP, TP, LoRA) demonstrates broad applicability. The utility evaluation on 9 tasks shows preserved benefits.

3. The one-line integration for existing systems and consideration of real-world distributed training scenarios (TP, PP, MoE) enhance practical value.

4. The paper is generally well-written with good use of figures to illustrate concepts.

### Weaknesses
1. Limited algorithmic novelty. The core contribution (filtering K, V activations while preserving Q) is relatively straightforward once the problem is identified. meanwhile, theoretical analysis of why this approach is optimal and exploration of alternative filtering strategies are missing. The connection to existing memory-efficient attention implementations could also be more rigorously analyzed

2. Insufficient Analysis of Strawman Approach. Section 3.1 claims the strawman approach "cannot work" but provides limited evidence, In particular, figure 4(a) shows non-convergence but lacks quantitative analysis of gradient corruption, ablation studies on different filtering strategies, or detailed explanation of the "interference" mechanism.

### Questions
1. How does this interact with other optimizations like FlashAttention-2 or PagedAttention? In particular, how exactly does this integrate with FlashAttention's tiling strategy?

2. What is the overhead of the offline preparation phase for new models?

3. Can you provide theoretical analysis of the gradient approximation quality?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Token filtering aims to improve training efficiency by focusing on only the most important tokens. However, current methods only introduce marginal training speedup because they do not fully utilize the token sparsity. This paper introduces CENTRIFUGE, a novel system designed to unlock the full efficiency of token filtering in large language model training. At the algorithm level, CENTRIFUGE proposes filtering activations in the attention backward kernel to amplify the sparsity. At the system level, CENTRIFUGE transforms sparse GEMM into dimension-reduced dense GEMM to maximize the computional efficiency. Experiments show that CENTRIFUGE maintains a similar training loss with the traditional approach of only filtering loss, while reducing training time by 31.7%.

### Strengths
1. The paper provides a clear diagnosis of why token filtering has not delivered superior training efficiency. The proposed method is intutive and effectively overcomes current limitations.
2.  Experiments on fine-tuning foundation models show that CENTRIFUGE preserves the benefits of token filtering and reduces end-to-end training time by 31.7%.

### Weaknesses
1. Experiments are only conducted on fine-tuning pre-trained foundation models for downstream tasks. However, since the main advantage of the proposed method lies in its training efficiency, it should be validated on computationally intensive pretraining tasks.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work tackles the problem of improving training efficiency for LLM via token filtering. Prior work showed token filtering can boost model utility/quality but fails to yield meaningful speedups in practice. The authors identify two key obstacles: 1. Insufficient sparsity propagation: existing token filtering (applied only at the loss layer) produces sparse gradients at the output, but upstream layers still process all tokens, negating potential speed gains. Naively filtering intermediate activations breaks standard attention backpropagation and harms gradient correctness. 2. Unsupported sparsity regime: Token filtering typically drops ~30-50% of tokens which leads to a "moderate" sparsity level at which standard sparse deep learning libraries (optimized for >=95% sparsity) become inefficient or even slower. To address these issues, the paper proposes an integrated solution called CENTRIFUGE, with contributions at both the algorithm and system levels:

* Algorithm level: introduces a modified attention backward pass that filters out activation for unimportant tokens during backprop, amplifying sparsity in the gradient computation. This new backward kernel avoids interfering with the attention softmax by processing each output’s gradient separately, which preserves the correct gradient flow and the model quality gains of token filtering. This design is also compatible with high-performance attention implementations (e.g. FlashAttention)

* System level: develops an automated workflow to convert sparse operations into smaller dense operations that can run efficiently on existing libraries. CENTRIFUGE exploits the structure of token sparsity (i.e. whole rows or columns dropped) to shrink the matrix multiplications, achieving the effect of sparse computation by adjusting tensor dimensions at runtime. The authors devise a robust method to dynamically alter the computation graph (despite PyTorch’s dynamic nature) by detecting stable graph patterns and updating tensor shapes before each backward pass. This allows the "moderate" sparsity to be leveraged for real speedups using standard GPU kernels instead of generic sparse GEMM routines which has high overheads.

Experiments on four LLMs (1.1B to 8B parameters) fine-tuned on a large math reasoning dataset show that this method achieves significant speedup with on-par or improved model performance. For example, filtering 50% tokens cut down backward pass time by up to 49% and end-to-end training time by up to 31.7%. Test performance is improved by up to 26.6% on individual tasks (+9.5% on average) comparing to no-filtering baseline.

### Strengths
1. The authors identify a crucial bottleneck (lack of real efficiency gains from token filtering) and provide a novel solution with algorithm and system co-design, effectively addressing both the ML aspect (ensuring gradients remain correct and useful) and the systems aspect (making use of sparsity in existing hardware/software). By further filtering attention activations in the backward pass in a safe way, they increase sparsity where it matters, and by transforming sparse ops to dense via graph manipulation, they avoid the usual pitfalls of sparse computation overhead. Together, these yield a unique solution that had not been explored before in literature on token filtering.
2. The experimental evidence is compelling and comprehensive. The achieved speedups are significant (nearly half the backprop time eliminated, and ~30% overall training time reduction at 50% token drop); and importantly, the approach does not harm model quality but in fact improves it. The experiment coverage is also comprehensive: multiple models (ranging 1B to 8B parameters) and different training regimes (single GPU, multi-GPU with DP and TP, and fine-tuning with LoRA) are all tested. This diversity builds confidence that the method is robust and general.
3. The work has high practical significance. Two facts of CENTRIFUGE lower the barrier to adoption: 1) it can be integrated with only one extra line of code for users already doing token filtering, and 2) it's compatible with common distributed training (DP+TP) and in fact further improve throughput by reducing gradient communication payload (Appendix D).

### Weaknesses
1. The experiments, although extensive for the provided settings, are limited to relatively small-scale dense LLMs and a specific domain (mathematical reasoning). It remains an open question how well CENTRIFUGE would scale to much larger LLMs (e.g. 70B+ params) or to training on more diverse, general-domain data. Larger LLM might introduce new bottlenecks or slightly different execution characteristics (communication load, optimizer overheads) that were not encountered at 3–8B scale. It might also be helpful to evaluate how the effect of CENTRIFUGE might change in MoE. 
2. Forward pass remains unoptimized: Centrifuge focuses on filtering tokens during backprop and does not skip any computation in the forward pass. This design is intentional to preserve model utility, but it means the maximum possible speedup is inherently limited. It would be valuable for the authors to clarify this limitation in the paper. Currently, the paper implies this in Section 2.1 and via citations, but a more explicit discussion of forward-pass cost might be helpful to set expectations for the reader.
3. The system-level solution, involving dynamic graph rewriting in PyTorch, is quite complex. While the authors do an admirable job automating it, one might worry about maintainability. If underlying libraries or model architectures change, these techniques might need adjustment. A discussion on how robust the implementation is to different model architectures (beyond the ones tested) and framework versions would be useful. This isn’t a flaw in the core idea per se, but acknowledging the complexity and how it’s managed would improve transparency.

### Questions
Echoing points listed in Weaknesses section:
1. How well do the authors expect CENTRIFUGE to scale to much larger models (e.g., 70B or 100B parameters) and training on massive corpora?
2. Have the authors considered evaluating CENTRIFUGE on different domains or tasks beyond math reasoning, such as coding?
3. Could some form of forward token filtering be incorporated in future to further boost efficiency, and what might be the impact on utility?

In addition, it would be also interesting to know:
4. In the experiments, a fixed fraction of tokens (e.g. 50%) is filtered each step. Have the authors considered making the filtering threshold adaptive, for instance, filtering more aggressively as training progresses or varying by batch?
5. Can CENTRIFUGE be combined with other training efficiency methods (e.g., MoE layers, gradient checkpointing (I think so), or data selection at the example level)? 
6. Considering the practical importance, do the authors plan to open-source CENTRIFUGE or integrate it into common libraries like DeepSpeed/FSDP, Megatron, HF/Lightning trainer?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CENTRIFUGE, a system that unlocks the full efficiency of token filtering in LLM training through algorithm-system co-design. At the algorithm level, it filters activations in the attention backward kernel to amplify sparsity while maintaining compatibility with memory-efficient attention implementations. At the system level, it transforms sparse GEMM into dimension-reduced dense GEMM through automatic graph updating. Experiments on four models (TinyLlama-1.1B to Llama3.1-8B) show up to 49% backward time reduction and 31.7% end-to-end speedup when filtering 50% tokens, while achieving 26.6% utility improvement over standard training.

### Strengths
1. The paper clearly identifies why existing token filtering methods fail to achieve efficiency gains (inadequate sparsity propagation and sparsity range mismatch with ML libraries), with solid empirical evidence.
2. The paper evaluates across multiple dimensions: different model sizes, various training scenarios, context lengths, and filtering ratios, demonstrating consistent gains and broad applicability.

### Weaknesses
1. All experiments use models ≤8B parameters. Given that efficiency gains are most critical for larger models (70B+), the absence of such experiments is a significant limitation. The 8B model uses TP=8, but modern large-scale training uses more complex parallelism (TP+PP+DP). How CENTRIFUGE scales to 70B+ models with TP=8, PP=4 setups remains unknown.
2. The automatic graph updating approach relies on "runtime stability" and special prime number markers, which may be fragile. The paper doesn't discuss failure cases, what happens when graph structure changes unexpectedly, or computational overhead of the offline preparation phase.

### Questions
1. Table 2: Why does the filter operator overhead vary significantly (2.98s for TinyLlama vs 14.35s for Llama3.1)? Is this linear in model size or are there other factors?
2. Why only evaluate Qwen2.5-1.5B but not Qwen2.5-7B, especially when Llama3.1-8B is included?

### Soundness
3

### Presentation
2

### Contribution
3
