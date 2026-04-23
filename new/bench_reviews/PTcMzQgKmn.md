Now I have enough information to write the comprehensive final review. Let me organize all findings.

## Summary

HiP Attention proposes a training-free framework for long-context LLM serving that integrates three components: (1) a hierarchical top-k key estimation algorithm exploiting "attention locality" to reduce attention complexity to O(T log T), (2) hardware-aware block-wise tiled optimization for TensorCore utilization, and (3) KV cache offloading that stores frequently accessed tokens on GPU while offloading the rest to host memory. Applied to Llama3.1-8B, HiP achieves 6.83× end-to-end decoding speedup at 128k context while maintaining 96% relative LongBench performance, and extends servable context from 16k to 64k on a commodity RTX 4090.

## Strengths

- **Complete system-level integration**: The paper combines algorithmic approximation (hierarchical pruning), hardware-aware optimization (block tiling for TensorCores), and memory management (KV offloading) into a single coherent framework — more useful than papers addressing only one aspect. The latency breakdown in Figure 8 clearly validates the system design.

- **Practical speedups are real and meaningful**: Table 5 shows 6.83× end-to-end decoding speedup at 128k with 96% relative LongBench quality, scaling to 14.30× with moderate quality tradeoff. Table 6 demonstrates extending context from 16k to 64k on RTX 4090 at 95.45 tok/s (Vector Map), versus 1.91 tok/s for FlashAttention with UVM — a 49.97× throughput improvement.

- **Training-free, drop-in replacement**: HiP is applied directly to Llama3.1-8B without weight modification (Section 5.1), and integration with vLLM and SGLang is described, lowering adoption barriers significantly.

- **Tunable speed-quality tradeoff**: Table 5 shows a clear knob (r_m) ranging from 6.83× speedup at 96% quality to 14.30× at 92.4%, giving practitioners explicit control.

- **Competitive with or superior to baselines across configurations**: On LongBench (Table 3), HiP^1/4 achieves 99% relative performance with 4.3× decode speedup; on passkey (Table 1), HiP achieves near-perfect accuracy up to 64k in multiple configurations.

## Weaknesses

### Fatal
None.

### Major

- **Abstract misrepresents the O(log T) GPU memory claim**: The abstract states HiP "stores only O(log T) tokens on the GPU while maintaining similar decoding throughput." In practice, the O(log T) HashMap implementation is unusably slow (10.15 tok/s at 64k on RTX 4090, Table 6), and the practical system uses the Vector Map with O(T) page table space complexity (95.45 tok/s). Section 3.3 does acknowledge the tradeoff: "we reduce the GPU memory footprint for KV tokens from O(T) to O(log T), but this comes with page table overhead that can range between O(T) and O(log T)" (line 175), and recommends Vector Maps for practical ranges. However, the abstract's framing "stores only O(log T) tokens on the GPU while maintaining similar decoding throughput" conflates the O(log T) structure with the throughput claim that only holds for the O(T) structure. The headline claim that readers take away is misleading. This matters because it sets incorrect expectations about the memory-performance tradeoff.

- **"Mathematically guaranteed" claim in abstract is substantially overstated**: The abstract states the algorithm is "mathematically guaranteed to have better performance than random attention pruning." What Theorem 1 actually proves (Section 4) is that for k=1 (finding the single highest-scoring key), under an unverified Gaussian assumption on score differences (δ_Δ ~ N(0, σ(Δ)²)), a single binary split is more likely than not to select the branch containing the top-1 key. This is a probabilistic statement under a specific distributional assumption, not a mathematical guarantee in the standard sense. The paper itself acknowledges the extension to the full algorithm is only "intuitive" (line 187): "By recursive application of HiP's key selection iterations, we can intuitively see that the probability..." It says nothing about (a) the full recursive algorithm with k > 1, (b) compounded error across log(T) iterations, or (c) the effect of block approximation. This matters because the abstract's framing inflates the theoretical contribution beyond what is formally established.

### Minor

- **Quality degradation on RULER in certain deployment configurations is underdiscussed**: On RULER at 128k context, HiP^1/2 in the sparse prefill + sparse decode configuration achieves only ~52% versus FlashAttention's 77% (Table 2). The abstract claims "maintaining high-quality generation with minimal degradation," which is accurate for LongBench (96–99% relative) and for dense-prefill configurations on RULER, but not for the fully sparse deployment setting on retrieval-heavy benchmarks. The paper presents all numbers but does not adequately discuss when and why quality degrades.

- **Attention locality analysis is based on a single layer and head**: Figure 6 analyzes only the 17th layer and 2nd head of Llama3.1-8B. It is well-established that different attention heads serve different functions — some retrieval-oriented heads are highly non-local. The paper provides no analysis of how locality varies across heads and layers, which is critical for understanding when and why HiP degrades on certain tasks.

- **Block approximation subsampling lacks quality ablation**: Section 3.2 introduces stride-based subsampling (b_sq, b_sk) within blocks for TensorCore efficiency, but no ablation is provided for how much quality this subsampling costs versus the hierarchical pruning itself. Since the competitive speedups depend on this approximation, its quality impact should be quantified.

- **HiP^1/2 and HiP^1/4 notation is not defined in the main text**: These symbols appear in Tables 2 and 3 but are not explained, creating a reproducibility gap. The l_d hyperparameter choice is also relegated to the appendix.

### Trivial
- The iteration count n_H := ⌈log₂ T⌉ (line 131) should more precisely be ⌈log₂(T/k)⌉, since starting from k chunks, each split doubles branches within each chunk. This does not affect the O(T log T) complexity analysis but is a minor imprecision in the algorithm description.

## Nice-to-Haves

- Per-head locality analysis across layers to explain which heads violate the locality assumption and how this correlates with quality degradation on specific tasks (especially RULER retrieval tasks).
- Ablation on block approximation subsampling (b_sq, b_sk) to quantify the quality cost of hardware-aware tiling.
- Per-task quality-speed tradeoff curves for different r_m values on LongBench subtasks, to reveal which task types are most sensitive to mask staleness.
- Comparison with more recent training-free dynamic sparse attention methods.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Missing comparisons with relevant contemporary baselines"** (Harsh Critic point 4): The critic claims "several more recent training-free sparse attention methods" are missing but does not name specific methods. The paper compares against StreamingLLM, H₂O, BigBird, HyperAttention, and AVD — a reasonable set of training-free methods. Without being able to confirm which specific methods should have been included, this criticism cannot be verified.

- **"16.5× speedup figure refers to attention computation only, not end-to-end"**: The abstract does specify "16.5× speedup attention computation in the decode stage" (line 77), distinguishing this from end-to-end speedup. While the distinction could be more prominent, the abstract does not conflate the two.

- **"The 2.7× prefix speedup cannot be verified"**: Without being able to extract exact numbers from Figure 7's rendered charts, this claim cannot be verified or refuted from the parsed paper.

- **"Passkey retrieval is too simple to generalize"**: While passkey is a simple benchmark, the paper also evaluates on RULER (more complex NIAH) and LongBench (realistic NLU tasks). The passkey results are one data point among several.

- **"Table structure mixes deployment configurations making it hard to parse"**: This is a formatting nitpick from the PDF parsing. The paper reasonably presents multiple configurations to show robustness.

- **"Speedup numbers not reported alongside quality in sparse decode configurations"** (Table 3): Table 3 does include speedup rows for both prefill and decode at 32k context. The RULER table (Table 2) also includes speedup rows at 128k context.

- **Generic "missing related works"**: Per the hard rules, I cannot verify existence of specific uncited methods.

## Novel Insights

The tension between the paper's O(log T) KV token storage claim and the O(T) page table overhead reveals an important design principle for KV offloading systems: the asymptotic space complexity of the *data* versus the *index* are separate concerns, and the index can dominate practical memory budgets. The paper's empirical finding that HashMap lookups are "not friendly to the GPU" (slower than naive UVM) is a concrete demonstration that GPU-unfriendly data structures can negate algorithmic complexity advantages — a lesson that extends beyond this specific system.

## Suggestions

- Revise the abstract to honestly reflect the practical memory-performance tradeoff: e.g., "KV cache offloading reduces GPU-resident KV tokens to O(log T), with a configurable page table overhead ranging from O(log T) to O(T) depending on the latency target."
- Tone down the "mathematically guaranteed" language to accurately reflect Theorem 1's scope: e.g., "provably outperforms random pruning for a single selection step under the attention locality assumption."
- Add a brief discussion of when quality degradation is most significant (sparse prefill + sparse decode on retrieval-heavy tasks) and recommend deployment configurations accordingly.

## Score and Decision

**Calibration comparison:**

- **High anchors (7+)**: Adaptive KV Cache Compression (8.0, oral) — clean claims, well-validated, no overclaiming. MagicPIG (7.2, spotlight) — strong theory, clean presentation. This paper is below these due to systematic overclaiming in the abstract and incomplete theoretical grounding.

- **Medium anchors (4-6)**: OmniKV (6.0, accept poster) — training-free KV offloading, less comprehensive system but cleaner claims. TidalDecode (5.75, accept poster) — position-persistent sparse attention. SqueezeAttention (5.5, accept poster) — KV cache sparsification. RecycledAttention (5.4, reject) — similar topic but missing LongBench evaluation. RetrievalAttention (5.5, reject) — similar topic but missing baselines and unclear method details. This paper has a more comprehensive system contribution and stronger empirical results than the medium-tier rejected papers, but more significant overclaiming than the medium-tier accepted papers.

- **Low anchors (<3)**: One-Pass Streaming Attention (2.33, reject) — misleading complexity claims with no experiments. This paper is far above this tier.

The paper's system contribution is substantial and the empirical results are convincing, but the abstract systematically overstates three core claims (O(log T) GPU memory, "mathematically guaranteed," "minimal degradation"). The paper does present the honest tradeoffs in the body (Section 3.3, Section 4, Table 6), which separates it from papers that fabricate or lack evidence. The overclaiming is a significant presentation issue, not a methodological one. I place this paper slightly above the medium-tier borderline — the comprehensiveness of the system and strength of the empirical results outweigh the framing issues, provided the abstract is corrected.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>