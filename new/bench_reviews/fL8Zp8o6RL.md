## Summary
This paper proposes FTP (FFN Token Pruning), a training-free method for accelerating long-context LLM prefilling by selectively pruing tokens before the FFN module in each transformer layer. The method uses a cumulative attention-score threshold to determine which tokens are important enough to compute through the FFN, while pruned tokens pass unchanged via the residual connection. Evaluated on LongBench across multiple models (Llama3-8B, Qwen2-7B, Qwen1.5-32B, Qwen2-72B), FTP achieves 1.20x–1.45x TTFT speedups with modest accuracy degradation on Qwen-family models, but suffers substantial degradation on Llama3 for certain tasks.

## Strengths
- **Targets an under-explored bottleneck**: The paper correctly identifies and profiles the FFN as the dominant prefilling bottleneck (~60% walltime; Figure 3), shifting focus from the more commonly optimized attention/KV cache components. This is a useful reframing.
- **Training-free, plug-and-play design**: The method requires no fine-tuning or architectural changes. It leverages existing residual connections to "logically zero" FFN outputs for pruned tokens, which is conceptually clean. Algorithm 1 provides transparent, tensor-level pseudo-code.
- **Demonstrates scaling benefits to larger models**: Table 2 shows that FTP achieves higher speedups (up to 1.45x on Qwen1.5-32B-Chat) on deeper architectures with only modest accuracy impact, supporting the claim that larger models absorb higher pruning rates.
- **Ablation validates the attention heuristic**: Table 3 cleanly demonstrates that random pruning at the same token count causes catastrophic accuracy collapse, confirming that attention-magnitude-based selection is essential to the method's effectiveness.

## Weaknesses
### Fatal
// None rise to the level of invalidating the entire paper.

### Major
- **Unexplained model-dependent degradation undermines generality**: The method exhibits drastically different behavior across architectures. On Qwen2-7B-Instruct, accuracy drops are modest (1–3% across tasks), but on Llama3-8B-Instruct, the Code Completion score collapses from 55.17 to 35.91 (~19 points absolute, 35% relative drop; Table 1). The paper does not hypothesize why different models respond so differently to the same pruning strategy, nor does it analyze what characteristics of a model or task make FTP fragile. This unexplained inconsistency means the central claim — that FTP "reliably achieves a speedup with negligible decrease" — is model-specific rather than general.

- **GPU speedups lack implementation-level explanation**: The paper reports real wall-clock speedups (e.g., Table 3 shows 621ms → 498ms for Qwen2 Code Completion), which confirms the method does accelerate in practice. However, Algorithm 1 uses dynamic advanced indexing (`O[R, :]`) on a variable number of tokens per layer. On modern GPUs, gathering non-contiguous token embeddings, dispatching a GEMM for an irregular batch size, and scattering results back introduces memory bandwidth overhead and kernel launch latency. The paper provides no analysis of how this overhead is mitigated — no custom fused kernel, no kernel-level profiling, and no discussion of whether the speedup comes from compute savings or implementation-side optimizations. Without this, the speedup numbers are black-box measurements whose reproducibility and generalizability to inference stacks is unclear.

- **Tail-query scoring introduces unanalyzed positional bias**: Section 3.2.1 computes token importance from only the last N=50 queries. The paper cites SnapKV (Li et al., 2024) for the claim that "attention patterns from last queries are nearly consistent with all queries." However, this assumption is not validated in the paper's own experiments, and the severe positional bias it introduces — retaining prompt-end tokens while potentially pruning mid-context information — could explain Llama3's failure on tasks like Code Completion where mid-prompt context (function signatures, dependencies) is critical. The paper provides no layer-by-layer or position-by-position mask analysis to characterize this bias.

### Minor
- **Limited benchmark scope**: All experiments use only the LongBench benchmark. While LongBench covers six task categories, it does not include retrieval-augmented benchmarks (e.g., Needle-in-a-Haystack) or real-world RAG workloads where pruning mid-context tokens could be particularly harmful. Extending evaluation to such settings would strengthen confidence in the method's robustness.

### Trivial
- **Terminology inconsistency**: The paper uses "token pruning" to mean bypassing FFN computation (not actually removing tokens from the sequence), which could confuse readers expecting full token removal. A brief clarification of this terminology choice would help.

## Nice-to-Haves
- A layer-wise visualization of which token positions are pruned at different depths would directly expose the scorer's positional bias and aid debugging of failure cases.
- Reporting per-layer pruning rates as a function of η and task type would help practitioners tune the method for their use case.
- A Pareto frontier plot (like Figure 7 for all models, not just Qwen2) would more honestly represent the method's sensitivity to architecture.

## Removed Points
**These points are flagged to be removed. Treat them with caution.**

- *"The 'negligible decrease' claim in the abstract is directly contradicted by Llama3's 19-point Code Completion drop; this invalidates the central premise."* — The abstract specifically cites the Qwen2 result ("the Qwen2-7B-Instruct model with FTP achieves a speedup of 1.24x with only a 1.30% performance drop"), which is factually accurate per Table 1. The Llama3 degradation is a model-specific weakness (kept as Major), not an invalidation or misrepresentation.

- *"The reported speedups likely conflate FLOP reduction with actual latency; without Triton kernels, the TTFT speedup claim is unsupported."* — Table 3 provides actual measured TTFT in milliseconds on NVIDIA A100 GPUs, demonstrating real wall-clock speedup. The concern about how the speedup is achieved is valid (kept as Major), but claiming unsupported speedup is incorrect.

- *"Attention scoring from last N queries 'contradicts established empirical findings' about attention shifting across layers."* — The assumption follows from prior work (SnapKV) and while unverified in this paper, it is not contradicted within this paper's results. The concern about resulting positional bias is kept as Major, but the claim of contradiction is overstated.

- *"Hyperparameters likely validated on LongBench itself, raising benchmark overfitting concerns."* — η and F are architectural hyperparameters set to fixed values per model; this is standard practice for training-free methods. No evidence of overfitting is presented.

- *"PyramidInfer comparison is asymmetrical and inflates FTP's advantage."* — The paper is transparent about running both official PyramidInfer (which OOMs) and their own FlashAttention re-implementation, reporting both in Table 1 and explaining the discrepancy. This is good practice, not a weakness.

- *"Bypassing FFN is not 'information preservation'; the residual merely freezes representation."* — The paper's claim is that the residual "preserves a substantial amount of information," not that it is equivalent to full FFN processing. This is a semantic disagreement rather than a factual error in the paper.

- *"FLOP analysis assumes perfect linear scaling to wall-clock time."* — The FLOP analysis in Section 3.1 is presented as motivation; the actual results report measured TTFT. This is not a conflation.

- *"Missing NIAH benchmark experiments."* — This is scope creep; the paper's stated scope is LongBench evaluation.

## Novel Insights
The paper makes a useful observation: pruning tokens before the FFN (rather than via KV cache eviction) allows the attention mechanism to still process the full context in each layer, while only the non-linear FFN computation is skipped. This asymmetric pruning strategy — full attention, partial FFN — is distinct from prior token eviction methods and could inspire similar approaches targeting other compute-heavy submodules. However, the paper does not deeply explore why this asymmetry benefits some models but harms others, which limits the transferability of the insight.

## Suggestions
1. **Analyze the model-dependent degradation**: Add investigation into why Llama3 degrades significantly on Code Completion while Qwen2 does not. Consider reporting per-layer retention rates and position-wise mask visualizations to identify if the scoring heuristic's bias disproportionately harms certain models/tasks.
2. **Provide implementation-level efficiency analysis**: Even basic profiling (e.g., measuring the overhead of gather/scatter operations, reporting kernel launch times, or demonstrating the speedup is robust across different sequence lengths) would address reproducibility concerns.
3. **Clarify terminology**: Briefly note that "pruning" here means "skipping FFN" rather than removing tokens from attention processing.
4. **Discuss limitations of the tail-query scoring assumption**: Even acknowledging that the N=50-query approximation may not hold universally, and suggesting when it is likely sufficient or insufficient, would strengthen the paper.

## Score and Decision
I compare this paper against several calibration anchors:
- **FastGen (8.0)**: Stronger in every dimension — provides a released CUDA kernel and comprehensive analysis. FTP lacks kernel-level implementation analysis.
- **Radar (6.6)**: Stronger theoretical grounding and theoretical analysis that connects to experiments. FTP lacks theoretical analysis.
- **CipherPrune (6.25)**: Token pruning with better justification and cleaner methodology.
- **GemFilter (5.25) and HASA (5.0)**: Similar profile — straightforward training-free methods with decent empirical results but significant implementation/methodological gaps. Both were rejected at 5.
- **SqueezeAttention (5.5)**: Slightly better methodology with more thorough baseline comparisons.
- **KVTQ (4.4)**: Similar concerns about FLOP vs. wall-clock claims, but KVTQ lacks actual speedup measurements while FTP does report them. FTP is stronger.

This paper has genuine empirical contributions (real measured speedups on multiple models and tasks, a clean ablation study) that place it above KVTQ but below the accepted training-free methods like Radar and FastGen. Its major issues (unexplained model-dependent degradation, lack of kernel-level reasoning for speedups, unanalyzed positional bias) align it with the 5.0 cluster (HASA, GemFilter). The measured speedups and training-free simplicity give it a slight edge, but the Llama3 degradation is a real weakness that a careful reviewer would weigh.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>