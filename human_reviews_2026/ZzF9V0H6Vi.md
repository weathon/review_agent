# InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Long-sequence processing is a critical capability for modern large language models. However, the self-attention mechanism in the standard Transformer architecture faces severe computational and memory bottlenecks when processing long sequences. While trainable sparse attention methods offer a promising solution, existing approaches such as NSA introduce excessive extra parameters and disrupt the conventional pretrain-on-short, finetune-on-long workflow, resulting in slow convergence and difficulty in acceleration. To overcome these limitations, we introduce dense-sparse switchable attention framework, termed as InfLLM-V2. InfLLM-V2 is a trainable sparse attention that seamlessly adapts models from short to long sequences. Specifically, InfLLM-V2 reuses dense attention parameters through parameter-free architecture modification, maintaining consistency between short and long sequence processing. Additionally, InfLLM-V2 ensures computational efficiency across all sequence lengths, by using dense attention for short inputs and smoothly transitioning to sparse attention for long sequences. To achieve practical acceleration, we further introduce an efficient implementation of InfLLM-V2 that significantly reduces the computational overhead. Our experiments on long-context understanding and chain-of-thought reasoning demonstrate that InfLLM-V2 is 4$\times$ faster than dense attention while retaining 98.1\% and 99.7\% of the performance, respectively. Based on the InfLLM-V2 framework, we have trained and open-sourced MiniCPM4.1 (https://huggingface.co/openbmb/MiniCPM4.1-8B), a hybrid reasoning model, providing a reproducible implementation for the research community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduced Dense-Sparse Switchable Attention, a sparse attention that adapts model from short to long sequences. DSSA starts from natively trainable sparse attention (SSA), which further modifications, including: 1. Shared Key-Value Projection. Instead of using three sets of KV, DSSA use a single shared set. 2. Remove compressed attention and only used it to get sparse selected attention and sparse sliding attention. Then replace the MLP with pooling function. They did experiments on several benchmarks and demonstrated the efficiency while maintaining the performance.

### Strengths
1. The proposed architecture is simple to follow. Just a few modification to existing NSA.
2. The performance is demonstrated across different benchmarks.
3. The training process is quiet efficient. Figure 6 presented a clear comparision between NDA and DSSA.

### Weaknesses
1. Since DSSA is based on NSA, then the most important baseline should be NSA. Even NSA is 27b total parameters and 3b active parameters, which is larger than DSSA implementation in this paper, it is important to evaluate that on similar benchmarks. For examples, the main table, table 1 in NSA paper presented many benchmarks, which are also major benchmarks for LLM. But DSSA did not evaluate on those.

2. In NSA paper, NSA achieved better performance than Full Attention as in table 2 of the paper. However, in your paper the NSA is much worse than LongBench, which leads to concerns about implementation of NSA. 

3. Still, in Figure 6 of NSA paper, it is around 9x faster than full attention but in your case it is only 0.9x to 2.2x. The gap is too large to be ignored.

### Questions
1. In figure 1. portable is mentioned as an advantage. What is the meaning here other than reusable and aligned?
2. Since the author examed the speed on different gpu, then what are the software configureations like cuda version and pytorch version? Different version could lead to performance gap for other baselines.
3. What does number of visible tokens mean? Does it mean how many keys/values each query token can attend to?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Dense-Sparse Switchable Attention (DSSA), a novel trainable sparse attention framework designed to efficiently adapt Large Language Models (LLMs) from short-sequence pretraining to long-sequence finetuning. The authors identify critical limitations in prior trainable sparse attention methods like Natively Trainable Sparse Attention (NSA), namely the introduction of excessive parameters, architectural mismatches that destabilize training, and inefficiency on short sequences.

DSSA addresses these issues with three main contributions:

- Seamless Short-to-Long Adaptation: DSSA reuses the key-value (KV) projection parameters from the original dense attention model, eliminating the need for extra parameters. It fuses multiple sparse patterns into a unified sparse attention module, maintaining architectural consistency with standard attention. This leads to stable training and preserves knowledge from pretraining.
- Efficiency on All Sequence Lengths: The parameter-free design allows a model equipped with DSSA to seamlessly switch back to standard dense attention for short sequences without any overhead, while effectively using sparse attention for long sequences.
Hardware-Aware Efficient Implementation: The authors identify the block selection mechanism as a key bottleneck in sparse attention. They propose a novel, two-pass, FlashAttention-style CUDA kernel that significantly reduces I/O overhead by fusing computations in SRAM. An "LSE Approximation" technique further optimizes this kernel, reducing the computational cost of the two-pass approach.
- Experiments on long-context understanding and reasoning tasks show that DSSA achieves performance nearly on par with full attention (retaining over 98%) while being up to 4x faster on the attention component and providing significant end-to-end speedups.

### Strengths
1. Clear Problem Formulation: The paper does an excellent job of positioning its work. It clearly articulates the shortcomings of the existing state-of-the-art trainable sparse attention method (NSA) and directly proposes solutions for each identified problem (parameter overhead, training instability, and short-sequence inefficiency). The framing around the "pretrain-on-short, finetune-on-long" paradigm is highly relevant and practical.

2. Elegant and Simple Design: The core idea of reusing existing parameters and unifying sparse patterns is both simple and powerful. This design choice directly leads to several benefits: zero parameter overhead, stable finetuning (as evidenced by the loss curves in Figure 5), and the flexibility to switch between dense and sparse modes. This makes DSSA a much more practical and "drop-in" solution compared to more architecturally disruptive methods.

3. Strong Technical Contribution in Implementation: The work in Section 3.4 is a significant strength. Identifying the I/O bottleneck in the block selection step and engineering a custom CUDA kernel to solve it shows a deep understanding of hardware-aware optimization. The LSE Approximation is a clever trick to mitigate the cost of their two-pass solution. This focus on practical, low-level optimization is crucial for realizing the theoretical benefits of sparse attention and is a valuable contribution.

4. Comprehensive and Convincing Experiments: The empirical evaluation is thorough. The authors compare DSSA not only against full attention and NSA but also against popular training-free methods (InfLLM, MInference) and a context-extension method (YaRN).

### Weaknesses
While the paper is strong, there are a few areas that could be improved or clarified:
1.  Direct Comparison to NSA: The paper's primary baseline for trainable sparse attention is NSA. However, the authors note they used an open-source Triton implementation of NSA, as the official code was not available. This raises a potential concern about whether the performance drop observed for NSA is due to fundamental architectural flaws (as argued) or sub-optimal implementation/hyperparameters in the third-party version. The claims against NSA would be stronger if benchmarked against an official implementation or with more extensive hyperparameter tuning for the version used.
2.  Limited Exploration of Hyperparameters: The effectiveness of DSSA's sparse pattern depends on several hyperparameters, such as the number of initial, local, and top-k blocks (`N_init`, `N_local`, `|I_topk|`) and the block size (`B`). The paper fixes these values for the experiments but does not provide an ablation study on their impact. Understanding the sensitivity of the model's performance and efficiency to these choices would be valuable for practitioners.
3.  Generalizability to Other Architectures: The experiments are conducted on an 8B parameter model with a specific Grouped-Query Attention (GQA) configuration. While the principles of DSSA seem general, its effectiveness and the efficiency of its custom kernels might vary with different model sizes, head dimensions, or attention variants (e.g., Multi-Query Attention). Discussing the potential challenges or necessary adaptations for other architectures would enhance the paper's scope.
4.  Novelty of Component Ideas: The idea of merging local (sliding window) and global (top-k) attention patterns is not entirely new and exists in prior work like BigBird or Longformer. The paper's main novelty lies in how it frames this within a *trainable, zero-parameter* adaptation framework and the efficient implementation. The paper could be more precise in situating its contribution relative to these earlier sparse attention architectures.

### Questions
1.  Block Selection Robustness: The block selection mechanism relies on attention scores from a compressed, coarse-grained representation of the keys. Could you comment on the robustness of this selection? Are there scenarios or tasks where this coarse approximation might fail to identify crucial fine-grained information, leading to performance degradation?
2.  Training Stability Details: Figure 5 shows a compellingly smooth training curve for DSSA compared to NSA. Could you elaborate on why DSSA's fine-tuning is more stable? Is it solely due to the shared parameters, or does the unified single-output structure (as opposed to NSA's gated multi-output) also play a significant role?
3.  On the `DSSA-Dense` Performance: In several tables (e.g., Table 1 on RULER), the `DSSA-Dense` model (which is the finetuned model switched back to dense mode) outperforms the `FULLATTN` baseline that was also finetuned on long contexts. This is an interesting result. Do you have a hypothesis for why this "sparse-to-dense" training regimen leads to better performance than continuous dense training? Could the sparse fine-tuning be acting as a form of regularization?
4.  LSE Approximation Impact: The LSE Approximation is a key part of your efficiency gains, reducing the overhead of the two-pass approach from 2x to 1.25x. Table 1 shows that using the approximation does not hurt performance on RULER. Have you confirmed this holds true across all other tasks? Is there a theoretical or empirical bound on the error introduced by this approximation?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Dense Sparse Switchable Attention, abbreviated DSSA, a trainable sparse attention framework designed for smooth short to long sequence adaptation in transformer models. The core idea is to reuse the pretrained dense key value parameters for sparse attention, to merge selected block and sliding window patterns into a unified sparse mask, and to replace learned compression with a parameter free multi stage pooling scheme. The paper also proposes engineering optimizations including a fused head group summation and a log sum exp approximation to reduce GPU memory traffic and compute during block selection. Experimental results on long context understanding and long chain of thought reasoning show that DSSA keeps model performance close to dense attention while delivering substantial speedups. Reported highlights include up to 4 times faster end to end inference relative to dense attention with about 98 percent to 99.7 percent task performance retention and kernel level speedups of up to about 7 to 9 times versus FlashAttention in some settings.

### Strengths
The paper addresses an important and timely problem for large language models namely efficient and stable adaptation from short pretraining to long finetuning. The proposed design is conceptually appealing because it removes the architectural mismatch between dense and sparse attention by reusing existing dense KV projections and producing a single sparse attention output rather than multiple gated outputs. This design choice improves compatibility with the standard pretrain on short then finetune on long workflow and appears to reduce training instability. The technical contributions combine modeling and systems innovations in a way that is practical. On the modeling side the union of selected and sliding patterns into a single mask and the three stage coarse to fine pooling are novel and justified by the need to preserve granularity while remaining block sparse. On the systems side the fusion of group summation into an SRAM friendly loop and the LSE approximation are concrete optimizations that address real GPU IO bottlenecks and yield large empirical speedups. The empirical evaluation is comprehensive in scope and includes long input comprehension, long reasoning and general short sequence tasks, which supports the claim that DSSA preserves accuracy while improving latency. The paper is generally clear and well organized with algorithmic pseudocode, explicit hyperparameter choices, and a sensible set of baselines including dense attention and several sparse methods.

### Weaknesses
1. The paper promises to "unlock the full potential of sparse attention" but does not discuss the limits of DSSA in terms of model scale and when the FFN or other layers dominate end to end costs.
2. The training stability claim is interesting but under analyzed. The loss disruption observed for NSA is shown but the paper should quantify downstream performance variance, report random seed statistics, and show whether DSSA reduces catastrophic forgetting when switching modes.
3. The NSA comparison can be fragile because NSA is reimplemented and initialized by parameter replication. Authors can provide more details that ensure a fair comparison such as exact initialization, learning rate schedules, and etc.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
