Now I have sufficient calibration information. Let me compile the final review.

Key calibration anchors:
- TTT (similar topic area): Scores 5, 6, 8, average ~6.25, but Reject — had issues with evaluation scope, missing efficiency details, incomplete baselines
- RetNet: Scores 3-6, Reject — overclaiming, limited ablation, evaluation scope
- FlashRNN: Scores 6-8, Accept (Poster) — speedup + kernel optimization, limited task evaluation
- DEER (parallelizing non-linear models): Scores 6, 6, 6, Accept (Poster) — 10× speedup, limited tasks, fairness of speedup comparisons questioned
- Patch-Level Training: Scores 6-8, Accept (Spotlight) — similar concerns about hyperparameter sensitivity, scale

TNT has genuine novelty and addresses a real problem. The core ideas (hierarchical memory with resets, Q-K projection, two-stage training) are sound and the results are substantial. But there are real issues: (1) the abstract claims TTT evaluation but only shows Titans, (2) mathematical notation issues make reproducibility hard, (3) speedups may conflate architectural and implementation effects, (4) limited scale, (5) overclaiming vs. Transformers.

I'd place this in the 5-6 range: a methods paper with genuine ideas but with enough execution and evaluation gaps that it doesn't fully deliver on its promises. This is comparable to TTT itself (which was rejected despite some impressive results) and somewhat below DEER (which had cleaner, more clearly validated speedup claims but smaller scope).</think>## Summary

TNT introduces a two-stage training paradigm for deep memory modules (e.g., Titans) that decouples training efficiency from inference performance. Stage 1 uses a hierarchical memory architecture—global modules processing large chunks for long-range context, and periodically-reset local modules for fine-grained parallel processing—to achieve massive context parallelism. Stage 2 fine-tunes local modules at smaller chunk sizes for inference accuracy. TNT achieves up to 17.37× training speedup over the most accurate Titans baseline while simultaneously improving model perplexity at 150M parameters.

## Strengths

- **Addresses a real and important bottleneck**: Deep memory modules like Titans and TTT are expressive but suffer from poor hardware utilization due to small chunk sizes and sequential dependencies. TNT identifies this problem clearly and proposes a principled solution. The three challenges identified (efficiency, compression-retrieval mismatch, chunksize sensitivity) are well-motivated and map to concrete design choices.

- **Effective parallelization strategy for non-linear recurrences**: The periodic reset of local memory states to a shared W_init is a clever mechanism that enables true context parallelism—breaking sequential dependencies that have been a long-standing challenge for non-linear RNNs. This is a genuine architectural contribution.

- **Solid empirical speedups**: Table 1 demonstrates substantial time-to-quality improvements (up to 17.37× over Titans C=8). Even with matched chunk configuration (TNT {8} vs. Titans C=8), there is a 7.68× speedup, which cannot be explained by chunk size alone and suggests genuine architectural benefits from the hierarchical design.

- **Well-structured ablations**: Table 3 shows monotonic improvement with increasing local modules (23.53 → 20.15 PPL), dramatic degradation without global memory (25.60 PPL), and a meaningful contribution from Q-K projection (21.04 → 22.01 without it). These validate the three proposed design components independently.

- **Q-K projection is a sensible contribution**: The observation that memory compression (key → value) and retrieval (query → output) operate in different domains, combined with a practical projection mechanism using a running sum of key outer products, is well-motivated and empirically validated.

## Weaknesses

### Major:

- **Abstract claims evaluation on TTT but experiments only cover Titans**: The paper's abstract states "Evaluated on Titans and TTT models," but the experimental section (Section 5) only instantiates TNT on the Titans architecture. Table 2 includes TTT as a *baseline*, but there is no TNT-enhanced TTT model. This directly undermines the generality claim. A model-agnostic training paradigm should demonstrate application to at least two distinct architectures to justify the claim.

- **Mathematical formulation is incomplete/opaque**: The core equations (Eqs. 5-7) have issues. Eq. 5 for global memory update appears garbled—missing the left-hand side definition of V_{(k+1)C_G}. Eq. 7 for retrieval has a dimensional inconsistency: the argument to f(W_t, ·) contains a sum of outer products (∑ k_τ k_τ^T / ||k_τ||), which is a matrix, not a vector, making it unclear how this functions as an input to the memory network f. While Appendix C promises details on efficient implementation, the main text lacks a coherent, self-contained specification of the TNT algorithm. This significantly hinders reproducibility.

- **Speedup attribution is not cleanly isolated from confounds**: The reported speedups conflate multiple factors—the hierarchical design, periodic resets enabling context parallelism, multi-resolution local modules, larger effective chunk sizes, and JAX implementation differences. While TNT {8} vs. Titans C=8 controls for chunk size, other differences (number of parameters due to extra modules, parallelization strategy) remain. Additionally, the comparison against FlashAttention-equipped Transformers mixes custom-kernel code with vanilla JAX implementations. The paper claims to address "FLOPs utilization below 5-10%" but reports no FLOPs or hardware utilization metrics—only wall-clock times. For a paper centered on hardware efficiency, this is a significant omission.

- **Limited evaluation scale and scope**: All experiments are at 150M parameters / 10B tokens with a fixed 16K context for quality evaluation. The paper's central motivation is enabling long-context modeling, yet no long-context benchmarks (e.g., NIAH, LongBench) are included—only standard LM perplexity and short-context commonsense reasoning tasks. Whether TNT's hierarchical design with periodic resets actually preserves fine-grained information over long contexts is not empirically validated.

### Minor:

- **Stage 2 fine-tuning gains are marginal and potentially attributable to additional compute**: The best Stage 2 result improves PPL from 23.13 to 23.09—a difference within noise for this scale. The paper states Stage 2 uses 5% additional compute but does not compare against simply continuing Stage 1 training for the same compute budget. Without this control, the claim that Stage 2 "resolves chunk-size mismatch" is not well-supported.

- **Global memory remains sequential**: The global module with C_G=2048 still processes chunks sequentially, creating a potential bottleneck at very long contexts. This is acknowledged indirectly (large chunks are "hardware-friendly") but not analyzed. As sequence lengths grow well beyond 32K, this sequential path could become limiting, which matters for the paper's scalability claims.

- **Common-sense reasoning comparison with Gated Transformer is overinterpreted**: TNT achieves 41.0% vs. Gated Transformer's 39.7% on 4 small benchmarks (PIQA, HellaSwag, ARC-e, CSQA) with no error bars. The authors themselves note "perplexity is a more stable metric," yet still foreground the reasoning gap. On perplexity—the more reliable metric—Gated Transformer clearly leads (22.39 vs. 23.09).

### Trivial:

- Overclaiming language: "removes a critical scalability barrier" and "establishing a practical foundation" for closing the gap with Transformers are premature given that TNT still underperforms Gated Transformer + FlashAttention in both speed and perplexity.

## Nice-to-Haves

- Evaluation on long-context benchmarks (NIAH, LongBench) to validate that the reset-based local memory does not lose critical information across boundaries.
- Experiments at 1B+ scale to validate scalability claims.
- Ablations varying S_L and C_G, which are key hyperparameters with no sensitivity analysis.
- Inference-time latency/throughput metrics, since Stage 2 is designed to optimize inference.
- A control experiment: Stage 1 continued for the same compute as Stage 2 to isolate the chunk-adaptation effect from additional training.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh Critic's claim that the paper does not provide a reproducible method at all (Point 1)**: While the mathematical notation in Eq. 7 is indeed problematic, the general structure of the method (hierarchical global+local memory, periodic resets, Q-K projection, two-stage training) is clearly described in the text and figures. The lack of clean equations is a real problem, but it does not make the method "effectively opaque" — the ideas are communicable.

- **Spark reviewer's claim of "no comparison with concurrent work (Zhang et al., 2025)"**: The paper does discuss Zhang et al. (2025) in the introduction, noting it "combines large chunks with local attention to enhance parallelism. However, this circumvents the inefficiency rather than solving it, complicates the analysis by mixing memory and attention, and neglects the need for small chunks during inference." While an empirical comparison would strengthen the paper, the absence of one is not a "factual mismatch."

- **Human finder's claim that "the claimed common-sense reasoning advantage... could easily be noise"**: While the small gap without error bars is a fair concern, Table 2 shows the advantage is consistent across 4 benchmarks (PIQA, HellaSwag, ARC-e, CSQA). It's plausible but uncertain—the authors' own caution about perplexity being more reliable mitigates the overclaim.

- **Harsh Critic's claim that the "causal link between resets and parallelism is not rigorously justified" and demand for a "reset-only baseline"**: The design makes clear theoretical sense—resets eliminate the sequential W dependency across shards, enabling context parallelism. The "w/o global memory" ablation shows PPL drops to 25.60, confirming that the scheme requires both components. A separate timing ablation would be nice, but the wall-clock speedups in Fig. 4 and Table 1 directly validate the parallelization claim.

- **Harsh Critic's demand that the paper prove convergence to the same loss floor**: This is a standard request but not a fundamental flaw in a methods paper showing time-to-quality improvements. The paper's evaluation framework (reach target loss) is standard in scaling studies.

- **Formatting/notation nitpicks about Appendix placement and equation rendering**: These are PDF extraction artifacts or minor style issues, not substantive weaknesses.

## Novel Insights

The most interesting insight is the tension inherent to the TNT design: by solving one problem (training parallelism via resets), the paper necessarily introduces another (information loss across shard boundaries). The global memory is intended to compensate, but operates at C_G=2048 granularity—orders of magnitude coarser than the local modules. Whether this hierarchy faithfully preserves fine-grained information across boundaries, especially for tasks requiring precise token-level recall (e.g., needle-in-a-haystack), remains an open question that the paper does not address. The Q-K projection mechanism, while empirically effective, also introduces a d×d running state whose memory costs are not analyzed; at large model widths, this is non-negligible.

## Suggestions

1. **Add experiments on TTT** to match the abstract's generality claim, or revise the abstract to say "applicable to any deep memory module, demonstrated on Titans."
2. **Fix Equation 7** to show the complete Q-K projection operation (P_t q_t or equivalent) and ensure dimensional consistency. Include or reference Appendix C explicitly in the equation.
3. **Report FLOPs or hardware utilization metrics alongside wall-clock times** to isolate algorithmic efficiency from implementation factors.
4. **Add a "same compute" control for Stage 2**: Continue Stage 1 training for 5% more steps and show the PPL, to demonstrate that Stage 2's gains come from chunk-size adaptation rather than additional optimization.
5. **Evaluate on at least one long-context benchmark** to validate the fundamental motivation of the work.

## Score and Decision

**Calibration comparison:**

- TTT paper (similar topic, similar evaluation issues): Scores 5, 5, 6, 8, average ~6, **Rejected** — had incomplete efficiency analysis, limited baselines, but genuine novelty and competitive results.
- RetNet (overclaimed relative to evidence): Scores 3-6, **Rejected** — similar pattern of overclaiming as a "Transformer successor."
- FlashRNN (kernel-level RNN speedup): Scores 6-8, **Accept (Poster)** — had fair speed comparisons, honest about scope, limited task evaluation.
- DEER (parallelizing non-linear sequential models): Scores 6, 6, 6, **Accept (Poster)** — 10× speedup, limited tasks, fairness concerns about comparisons, but genuinely novel and clearly articulated method.

TNT is most comparable to DEER: both address parallelizing non-linear RNNs, both show substantial speedups, both have limited evaluation scope. TNT also shares characteristics with TTT (deep memory module, claims of generality but single-architecture evaluation). TNT's strengths are genuine—17× speedup with improved perplexity is notable, and the architectural ideas are sound. However, TNT has more significant weaknesses than DEER: the abstract claims generality not demonstrated, the core equation is malformed, speedup attribution is confounded, and evaluation lacks the long-context benchmarks central to its motivation.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>