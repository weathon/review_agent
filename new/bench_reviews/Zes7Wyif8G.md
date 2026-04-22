Now I have enough calibration data. Let me write the final review.

## Summary

KLAY introduces a new data structure for representing arithmetic circuits—central to neurosymbolic AI—as layered index and scatter-reduce operations that map directly onto primitives in standard tensor libraries (PyTorch, Jax). This enables GPU parallelization without custom CUDA kernels. The paper reports speedups of up to four orders of magnitude over a naive per-node PyTorch baseline on the largest benchmark (Warcraft 12×12), and demonstrates orders-of-magnitude improvements over JUICE (the only other layerized circuit evaluator) on synthetic benchmarks.

## Strengths

- **Elegant core insight: circuit evaluation as gather + scatter-reduce** (Section 4, Algorithm 1, Figure 4). By reducing each layer transition to indexing (`N_{l-1}[S_l]`) followed by scatter-reduce, KLAY gets kernel fusion, JIT compilation, and hardware portability for free. This is a cleaner and more portable engineering approach than writing custom CUDA kernels, and the paper convincingly demonstrates its practical benefit.

- **Strong empirical speedups on large circuits** (Table 1: KLAY(cuda) achieves 1.59ms vs 18,931ms for naive CUDA on Warcraft; Figure 6 shows KLAY outperforming all baselines by over one order of magnitude on large synthetic circuits, including JUICE's custom GPU kernels). The speedups on the largest benchmarks are indisputable and meaningful for the target application.

- **Multi-rooted circuit support with deduplication** (Section 3.2, Theorem 1). The Merkle-hash-based linear-time deduplication addresses a real practical need for batched inference with different background knowledge. Table 2 shows this enabling training where DeepProbLog times out.

- **Hardware-agnostic design outperforms custom kernels** (Figure 6: KLAY on CUDA outperforms JUICE's hand-written CUDA kernels). This validates the core claim that expressing computation via library primitives is not just portable but performant, likely due to kernel fusion in modern tensor compilers.

- **Exact computation while outperforming approximate baselines** (Table 2: KLAY is faster than Scallop while remaining exact, e.g., 3-digit MNIST addition: 8.85s vs 5345.78s).

## Weaknesses

### Fatal
None.

### Major

- **No end-to-end learning accuracy results.** The paper measures only runtime, never whether neurosymbolic models trained with KLAY actually converge to correct solutions or achieve competitive accuracy. Table 2 shows KLAY enabling MNIST-3-digit training where DeepProbLog times out, but no accuracy is reported—leaving open the possibility that numerical issues (product underflow in the real semiring, scatter precision) degrade learning outcomes. The paper's stated goal is "paving the way towards scaling neurosymbolic AI," which implies improved learning outcomes, not just faster circuit evaluation. Without demonstrating that the speedup translates to better or even viable learning, the paper establishes an engineering speedup but not a scientific advance in neurosymbolic AI. This gap is partially mitigated by the fact that KLAY computes the same arithmetic circuit (same mathematical function), so the gradient should be identical—meaning accuracy should match the naive baseline given the same optimizer/hyperparameters. However, numerical precision differences between the scatter-reduce path and per-node evaluation (especially for the real semiring with many products) could matter in practice, and the paper does not verify this.

- **The headline "multiple orders of magnitude" speedup is misleadingly framed against a strawman GPU baseline.** In Table 1, the 4-order-of-magnitude claim (Warcraft: 18,931ms → 1.59ms) comes from comparing KLAY(cuda) against Naive(cuda), which is not a serious GPU implementation—it is per-node PyTorch autograd that the paper itself shows performs worse on GPU than CPU (42.70ms vs 17.35ms for Sudoku). The fairest comparison is KLAY(cpu) vs Naive(cpu), which still shows strong speedups (~2 orders of magnitude for Warcraft: 18.55ms vs 11,485ms) but far less dramatic results for the smaller benchmarks (Sudoku: 0.45 vs 17.35, ~1.6 orders). The abstract and introduction state "multiple orders of magnitude" as a general claim, which is accurate for the largest benchmark but not representative across the benchmark suite. The paper should calibrate its claims accordingly.

### Minor

- **Dismissing Liu et al. (2024) based on a density argument without empirical validation.** The paper argues that probabilistic circuits are "usually far more dense than arithmetic circuits" and therefore block-sparse techniques from Liu et al. are inapplicable. This may well be true, but it is not demonstrated empirically. A small experiment testing Liu et al.'s approach on neurosymbolic circuits—even showing it performs poorly—would substantially strengthen the claim that KLAY's scatter-reduce approach is genuinely superior on highly sparse circuits.

- **D4-compiled circuits may inflate node count, but the scope of this problem is unclear.** The paper notes (line 209) that node deduplication does not offset unary-node overhead for D4-compiled circuits, but defers details to Appendix B. Since D4 is a widely-used knowledge compiler, quantifying how much inflation occurs and for what proportion of practical neurosymbolic circuits D4 (vs. SDD) is the appropriate compiler would help readers assess the generality of KLAY's benefits.

- **GPU advantage is limited to large circuits, not stated explicitly.** Table 1 shows KLAY(cuda) ≈ KLAY(cpu) for small circuits (Sudoku: 0.46 vs 0.45ms), meaning GPU parallelization only helps for large circuits. This is a natural and expected finding, but the paper's framing of GPU acceleration as a general benefit should be qualified.

- **The "refute" framing is overly strong.** The paper states (line 189) that it "refute[s]" claims by Shah et al. and Vasimuddin et al. that GPUs cannot efficiently evaluate arithmetic circuits. Those works made empirical observations about their implementations and the hardware of their time; showing that a different algorithmic representation works on modern GPUs is evidence that the problem was algorithmic, not fundamental. This is a valuable contribution but does not constitute a logical refutation—rather, it updates prior conclusions.

### Trivial
None.

## Nice-to-Haves

- **End-to-end learning accuracy on at least one non-trivial task** (e.g., MNIST-addition across all digit counts) comparing KLAY-trained models against baselines. This would convert the paper from "KLAY is fast" to "KLAY advances neurosymbolic AI."

- **Scalability limit analysis**: at what circuit size does KLAY run out of GPU memory? What is the memory-to-circuit-size relationship? This is relevant to the "paving the way towards scaling" claim.

- **Per-layer compute/memory profiling**: a visualization of node/edge count per layer would reveal whether parallelism is evenly distributed or concentrated, affecting GPU utilization.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Jax cannot backpropagate through scatter-multiplication undermines hardware-agnostic claim** (Harsh Critic, Section 4): The paper explicitly notes this limitation and excludes Jax from the real-semiring comparison (line 205-206: "Jax does not support backpropagation on scatter multiplication, [so] these are excluded from the respective comparisons"). This is a documented limitation, not a hidden flaw. The log semiring (which uses logsumexp+sum, not product-scatter) works with Jax. Moved to minor/nice-to-have tier rather than a major criticism.

- **Cumulative runtime plot (Figure 6) is unusual/potentially misleading** (Harsh Critic, Section 6.1): Cumulative plots of sorted runtimes are a valid and common visualization pattern for benchmark suites with variable difficulty. The paper also reports per-instance timing averaged over 10 runs (line 225). This is a presentation preference, not a methodological flaw.

- **JUICE timeouts may reflect configuration issues rather than fundamental slowness** (Harsh Critic, Section 6.1): This is speculative—the paper reports a 30-minute timeout, which is a clear and objective measurement. Without evidence of misconfiguration, this is not a valid criticism.

- **Scallop vs KLAY runtime comparison is a "category error"** (Harsh Critic, Table 2): The paper explicitly notes "KLAY remains exact unlike Scallop" (line 259), making the reader aware of the difference. Comparing runtimes while noting the exact/approximate distinction is informative, not a category error—it shows that even an exact method can be faster than an approximate one.

- **Hash collision concerns in Theorem 1** (Harsh Critic, Section 3.2): The critic's own analysis shows collision probability of ~5×10⁻⁸ for 10⁸-node circuits, which they acknowledge is negligible. The concern about linearity of XOR is addressed by the `mix_hash` dispersal function explicitly mentioned in Equation 3—this is precisely what prevents structural collisions. The theorem is correct in expectation as stated.

- **Layer depth vs parallelism tradeoff** (Harsh Critic, Missing Parts): 84 layers for HMLC/Warcraft is not excessive by deep learning standards (modern networks have hundreds of layers). This is a valid direction for future analysis but not a weakness in the current paper.

## Novel Insights

The paper's key insight—that arithmetic circuit evaluation, long considered too sparse and irregular for GPUs, can be reduced to two widely-available tensor primitives (indexing and scatter-reduce)—is genuinely valuable because it reframes a domain-specific problem (sparse circuit traversal) into a generic hardware-friendly idiom. This is a different kind of contribution than a new algorithm or model; it is an *interfacial* contribution, connecting two communities (knowledge compilation and GPU systems) via a shared set of primitives. However, the paper stops short of demonstrating that this interface actually *enables* new neurosymbolic capabilities rather than merely speeding up existing ones.

## Suggestions

- Add end-to-end learning accuracy results for at least MNIST-addition (all digit counts) and one other benchmark. This is the single most impactful revision possible.
- Recalibrate the abstract/introduction framing: replace "multiple orders of magnitude" with a more nuanced statement (e.g., "up to four orders of magnitude on the largest benchmarks") and soften "refute" to "update" or "challenge."

## Evaluation

**Originality**: The core insight (circuit evaluation as gather + scatter-reduce) is elegant and, while each primitive is individually well-known, their combination to solve this specific problem is novel. The Merkle-hash deduplication for multi-rooted circuits is a nice supporting contribution. Overall: above average originality for a systems paper.

**Importance**: Accelerating the symbolic component of neurosymbolic AI is a real and important bottleneck. However, without end-to-end learning results, the impact remains potential rather than demonstrated.

**Claims support**: Runtime claims are well-supported by experiments. The broader claim that this "paves the way towards scaling neurosymbolic AI" is not fully substantiated without accuracy results.

**Soundness of experiments**: The speedup experiments are sound. The main issue is the absent accuracy evaluation and the somewhat inflated framing relative to the naive GPU baseline.

**Clarity**: The paper is well-written and well-structured. The running example (Figures 1–5) effectively communicates the core ideas.

**Value to community**: High for the neurosymbolic community—KLAY provides a practical, usable library that addresses a known bottleneck. The value would be significantly higher with end-to-end learning validation.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LogicMP (NeSy, parallel tensor ops for logic constraints) | BLGQ3oqldb | 6.0 | KLAY has stronger speedups and a more fundamental representation insight, but similarly lacks convincing end-to-end accuracy demonstration for the claimed neuro-symbolic advance |
| Bilevel NeSy optimization (100× speedup with accuracy on 8 datasets) | uJPWeZffgl | 5.25 | This paper had both speedup AND accuracy results across 8 datasets but still rejected due to inconsistent accuracy gains; KLAY has no accuracy at all |
| FlashSampling (speed/memory gains but poor quality evaluation) | V4Xs283LHH | 2.5 | Low anchor: speedup-only paper with no quality evaluation. KLAY is stronger because speedup is more dramatic and there's a clear mathematical argument that accuracy should be preserved |
| LightSeq (sequence parallelism, 1.24–2.01× speedup, no accuracy) | kC5i5X9xrn | 5.0 | Medium: systems paper with speedup only, rejected due to limited evaluation scope. KLAY has larger speedups and a clearer algorithmic insight |
| UKAN (GPU-accelerated KAN library) | wj4Az2454x | 5.33 | Medium: GPU acceleration with speedup benchmarks, rejected for limited scope. KLAY has more significant speedups |
| FlashAttention-2 (GPU attention kernel, better parallelism) | mZn2Xyh9Ec | 7.25 | High anchor: clean systems contribution with significant speedups. KLAY is structurally similar (reformulating computation for GPU efficiency) but less mature in evaluation |
| RaLMSpec (speculation-based RALM acceleration, 1.75–2.39× speedup) | vkzPuZJ80a | 4.25 | Borderline: modest speedup, no accuracy improvement, rejected. KLAY has much larger speedups but similarly lacks accuracy evaluation |

KLAY sits above the "speedup-only systems papers" in the 4–5 range because: (1) the speedups are truly dramatic (orders of magnitude, not 2×), (2) there is a genuine algorithmic insight (not just engineering), (3) there is mathematical reason to believe accuracy is preserved (same circuit, same gradient). But it sits below the high-scoring systems papers (7+) because it lacks end-to-end learning validation and overclaims relative to the baseline comparison. LogicMP at 6.0 is the closest comparison—a neurosymbolic paper with parallel tensor operations, accepted as poster despite limited accuracy evaluation. KLAY is arguably stronger on the systems side (larger speedups) but weaker on the learning validation side.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>