Now let me carefully verify the key claims from the reviews against the paper itself before synthesizing.

## Summary

ACT-IN-LLM proposes an adaptive vision token compression method that operates *within* the LLM's transformer layers, retaining all query tokens while compressing only key and value tokens in self-attention based on text-guided attention scores from the previous layer. This contrasts with Pre-LLM or Early-LLM compression approaches that discard tokens before or in early LLM layers. The paper provides theoretical analysis claiming that K/V compression provides a better low-rank approximation of full attention than query compression, and demonstrates improvements of ~6.3% over existing compression methods on high-resolution benchmarks while reducing vision tokens by ~60%.

## Strengths

- **Well-motivated approach with clear empirical evidence**: Figure 2 effectively demonstrates that early compression causes significant performance degradation (up to ~15% on high-resolution tasks), and that attention importance of vision tokens shifts across layers. This provides a solid rationale for compressing within rather than before the LLM.

- **Principled distinction between K/V and Q compression**: The insight of retaining all queries while compressing only K/V tokens is theoretically motivated (Theorem 3) and practically effective. The unified formulation (Eq. 7-9) cleanly categorizes different compression strategies and provides a useful abstraction.

- **Strong empirical results against compression baselines**: Table 2 shows ACT-IN-LLM achieving +5.5% average improvement over the best prior compression method (FastV at 39.9% → 45.35% on high-resolution benchmarks) under the same backbone and training setup, with competing general benchmark performance. The "w/o train" results also show the method works well without retraining.

- **Comprehensive ablations**: The ablation studies (Table 4) systematically explore compression ratios, compression methods (attention vs. pooling), and layer positions, providing useful design insights for the community.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed "no information loss" and "error correction" narrative**: The paper repeatedly claims that retaining all query tokens "ensures no vital information is lost" (Abstract, Sec. 3.2, Sec. 6) and provides "an inherent error correction mechanism." However, when a vision token is not selected into K/V at some layer, it can no longer influence other tokens' representations at that layer through the attention mechanism—the interaction is irrevocably lost for that step. While retaining queries preserves the token's *own representation* for future layers, this is not "error correction" in the conventional sense; the token simply persists in the query stream. The actual mechanism provides *less* information loss than Pre-LLM approaches (which remove tokens entirely), but claiming "no vital information is lost" is incorrect and structurally misleading given that hard top-k selection is still applied at each compressed layer. This matters because it forms the central comparative advantage claimed over baselines.

- **FlashAttention incompatibility and practical deployment overhead not addressed**: The ACM requires accessing explicit attention weights (A_{i-1}) from the previous layer to guide compression (Eq. 4), which is fundamentally incompatible with hardware-optimized implementations like FlashAttention that do not materialize full attention matrices. The paper reports single-forward pass times but does not quantify the overhead of ACM (attention weight extraction, top-k selection, sampling) relative to the savings, nor does it discuss how to reconcile this with practical inference frameworks. This is a significant practical concern for real-world deployment, similar to issues raised for PyramidDrop and ZipVL (which were rejected in part due to this limitation).

- **Theoretical disconnect from the concrete algorithm**: Theorems 2 and 3 establish existence of good low-rank K/V compression matrices under Assumption 1, but do not show that the specific *top-k selection based on last-row attention* used in the actual ACM satisfies the required properties. Theorem 1 asserts existence of a rank-Θ(log N) approximation without specifying distributional assumptions on A (these are presumably in the appendix). The critical Assumption 1 ("vision tokens receive much less attention than text tokens") is validated by a coarse aggregate bar chart (Fig. 5b) but is used to guarantee fine-grained structural properties. Furthermore, the comparison in Theorem 3 against idealized baselines (C^Q, I, I) for FlexAttention does not match FlexAttention's actual architecture. The net effect is that the theory provides intuition for why K/V compression might be preferable, but does not substantiate the specific design choices of ACM or prove its advantage over realistic baselines.

- **Ablation reveals that simple averaging nearly matches the sophisticated attention-guided method**: Table 4b shows that AvgPool-1D achieves 75.06% on general benchmarks (vs. 75.04% for attention-weight) and 45.08% on high-resolution (vs. 45.35%). The 0.27% gap on high-resolution benchmarks suggests the text-guided attention mechanism contributes marginal gains over a parameter-free average-pooling baseline. While the paper's conceptual contribution of *where* to compress (inside LLM layers on K/V) remains important, the specific "adaptive" mechanism (ACM) with its attention-weight guidance is undercut by this result, which the paper does not adequately discuss.

### Minor

- **Mask formulation inconsistency**: In Eq. 6, the sampled causal mask is described as M̄_i = M_i[s, s], which would yield a (M+L)×(M+L) matrix, but the text claims M̄_i ∈ R^{(N+L)×(M+L)} to match the dimensions of Q times K̄^⊤. Since Q retains full length (N+L) while K/V are compressed, the mask requires careful handling of both retained and compressed token positions, which is not clearly specified. While likely implementable, the notation inconsistency may hinder reproducibility.

- **Evaluation coverage on general multimodal benchmarks is limited**: The general benchmark suite (SEED, POPE, MME) is narrow compared to what is standard in the MLLM literature (missing MMBench, MMVet, MathVista, etc.). The high-resolution benchmarks are well-covered, but the method's impact on broader multimodal reasoning capabilities is less certain.

- **Scaling experiments lack non-compressed baselines**: Figure 7 shows ACT-IN-LLM across different LLM sizes and SFT data scales, but without non-compressed (full token) counterparts at the same sizes. This makes it impossible to assess how much performance is lost to compression at each scale, weakening the "plug-and-play" claim.

### Trivial
- Title contains a grammatical error: "Adaptively Compression" should be "Adaptive Compression" or "Adaptively Compressing."

## Nice-to-Haves

- Comparison with recent KV-cache compression methods adapted to the MLLM setting (e.g., PyramidKV, H2O) would strengthen the paper's claim of novelty over the closest family of methods.
- Per-task significance analysis or confidence intervals across multiple training runs to verify that the 5.5% average improvement is robust rather than driven by one or two benchmarks.
- Analysis of how compression affects attention distributions in later layers, to verify that the text-guided selection signal remains reliable after multiple rounds of K/V compression.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Baseline fairness concerns about re-implementation**: The critic raised concerns about re-implementing complex baselines like C-Abstracter and FlexAttention within the same backbone. However, the paper explicitly states they keep all settings constant and only vary the compression method—it is standard and actually fairer to compare under the same backbone rather than using each method's original setup, which would confound architecture differences. Removed because controlled comparisons under the same backbone are a strength, not a weakness.

- **"Reported averages hide large spread and lack variance"**: While reporting standard deviations would be nice, this is not standard practice in the MLLM community for benchmark evaluations. The improvement margins (5.5% average, with individual task improvements like ChartQA 35.0→46.1 and DocVQA 38.6→45.2) are large enough that statistical significance is likely. Moved to nice-to-have.

- **Pre-LLM gap analysis is "shallow and overgeneralized"**: The critic argues Fig. 2 only uses one backbone and simple dropping heuristics. However, Fig. 2 is a motivational analysis demonstrating that attention importance shifts across layers—a fact the paper validates. The full comparison against learned Pre-LLM methods is in Table 2. The motivational analysis is sufficient for its purpose.

- **"High-resolution vs general trade-offs underexplored"**: The paper clearly separates high-resolution and general benchmarks in Table 2, showing improvements on both. The high-resolution focus is the stated scope. This is scope creep.

- **Plug-and-play claim overstated**: The critic argued the scaling experiments change multiple variables. However, Table 2 shows "w/o train" results demonstrating the method works without retraining, and the scaling experiments show consistent improvement across architectures. The claim is partially supported.

- **Missing ablation on compressing Q tokens**: The spark reviewer suggested testing a variant that also compresses Q tokens. While this would be informative, the theoretical analysis (Theorem 3) already argues that Q compression is suboptimal, and the paper's core claim is that retaining Q is beneficial. This would strengthen but is not required.

- **Training time overhead not analyzed**: The spark reviewer noted the abstract claims ~20% training time reduction but only single-forward-pass time is reported. This is a valid concern but is a nice-to-have rather than a core flaw given that most efficiency claims in this area focus on inference.

## Novel Insights

The ablation in Table 4b revealing that simple AvgPool-1D nearly matches the attention-weight guided ACM (0.27% gap on high-resolution benchmarks) is a genuinely important finding: it suggests that the primary benefit of ACT-IN-LLM comes from the *architectural decision* of compressing K/V inside the LLM rather than before it, rather than from the specific text-guided selection mechanism. This insight has implications for future work—efficient K/V pooling within LLM layers may be sufficient for most practical purposes, and research effort might be better directed toward optimizing layer placement and compression ratios rather than sophisticated token importance estimators.

## Suggestions

- Temper the "no information loss" and "error correction" claims to accurately reflect that the method retains query representations (not full interaction information), distinguishing it more precisely from Pre-LLM methods as "partial information preservation" rather than "no information loss."
- Add a brief discussion of AvgPool-1D's near-competitive performance and what it implies about the design—this is actually a valuable result for practitioners.
- Discuss FlashAttention compatibility and quantify ACM overhead (attention weight extraction, top-k selection) or describe how to implement ACM in FlashAttention-compatible frameworks.

## Score and Decision

**Calibration papers:**
- PyramidDrop (3,3,3,3) — Very similar idea (hierarchical in-LLM token dropping), but weaker theoretical grounding and less empirical rigor. ACT-IN-LLM is clearly better than this.
- LLaVA-Mini (8,6,6,6) — More fundamental contribution (extreme compression to 1 token with modality pre-fusion), but different scope. ACT-IN-LLM is less transformative.
- Dynamic-LLaVA (6,6,6,6) — Similar scope (dynamic token sparsification for efficient MLLMs), accepted as poster. ACT-IN-LLM has more theoretical analysis but weaker practical deployment analysis.
- ZipVL (5,3,3,5) — Similar approach (KV cache + token sparsification), rejected. ACT-IN-LLM is stronger empirically but shares some overhead/novelty concerns.
- eRAM-V (3,6,5,6) — Similar scope (layer-wise visual token reduction), rejected due to analysis-method disconnect and limited evaluation.

ACT-IN-LLM sits between ZipVL/eRAM-V (rejected, ~4-5 range) and Dynamic-LLaVA (accepted poster, 6). It has meaningful empirical contributions and a clean K/V compression insight, but is weakened by overclaimed "no information loss," theoretical assumptions disconnected from the algorithm, FlashAttention incompatibility, and the nearly-competitive AvgPool-1D baseline undercutting the ACM narrative. These are not fatal but are substantive enough to prevent a confident accept.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>