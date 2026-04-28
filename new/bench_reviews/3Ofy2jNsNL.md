Now I have sufficient calibration. Let me write the final review.

## Summary
This paper proposes ACT-IN-LLM, a method that adaptively compresses vision Key/Value tokens within LLM layers rather than before input, preserving Query tokens to prevent irreversible information loss in high-resolution multimodal tasks. The method achieves strong empirical results on high-resolution benchmarks (5.5% improvement over prior compression methods) with demonstrated scaling across model sizes.

## Strengths
- **Strong empirical performance on high-resolution benchmarks**: Table 2 shows ACT-IN-LLM achieves 45.4% average on high-resolution tasks, outperforming FastV (39.9%) by 5.5% and significantly closing the gap to the uncompressed Full baseline (48.0%), while maintaining comparable efficiency (~83% of Full's inference time).
- **Well-motivated design with empirical support**: Figure 2 provides compelling evidence that early token dropping harms high-resolution performance (~15% gap when dropping at layer 5), and attention visualizations show tokens ignored in early layers become important later, justifying the layer-wise compression approach.
- **Demonstrated scalability**: Figure 7 shows consistent performance gains as LLM backbone scales from 0.5B to 7B and SFT data increases from 0.5M to 1.2M, indicating the method generalizes across model sizes and data regimes.
- **Architecture simplicity**: The Adaptive Compression Module (ACM) introduces no learnable parameters and can be integrated into existing MLLM architectures with minimal overhead.

## Weaknesses

### Fatal
None

### Major
- **Confounded SOTA comparison in Table 3**: The paper compares ACT-IN-LLM (using InternLM2-7B per Section 5.2) against baselines like LLaVA-Next and Mini-Gemini-HD that typically use Vicuna/LLaMA backbones. Section 5.2 demonstrates that backbone choice significantly impacts performance (e.g., scaling from Vicuna to Qwen/Intern shows substantial gains). Without a controlled comparison where both ACT-IN-LLM and baselines use the same backbone, the claim of achieving "SOTA performance among MLLMs with ≤1K tokens" is not fully supported. The paper frames this as a data/token efficiency comparison ("87.2% of InternVL2's performance with 32.8% of tokens and 24% of data"), which is valid, but the SOTA framing overstates what the experiments demonstrate.

### Minor
- **Theorem 3 has unresolved dimension mismatch**: The theorem compares `||Com(I, C^K, C^V) - Full||` vs `||Com(C^Q, C^K, C^V) - Full||`, but ACM produces output of dimension `(N+L)×D` (Query uncompressed) while Pre-LLM produces `(M+L)×D` (all compressed). Comparing norms of matrices with different row dimensions requires specifying how the mismatch is resolved (padding, projection, or restriction to common subspace). The theoretical claim needs reformulation or removal.
- **Training procedure for discrete selection not explained**: Section 3.2 uses a non-differentiable `Top` selection operation but does not clarify how gradients flow during training. Since ACM has no learnable parameters and selection is based on attention weights from the previous layer, gradients likely flow through the LLM weights normally, but this should be explicitly stated to avoid confusion about whether straight-through estimators or other techniques are used.

### Trivial
- **Efficiency framing could be clearer**: The abstract claims "~20% reduction" which is accurate vs. Full baseline, but Table 2 shows ACT-IN-LLM (515ms) is slower than FastV (499ms). The paper doesn't claim to be faster than all compression methods, but the positioning could more explicitly acknowledge the accuracy-efficiency trade-off shown in Figure 6.

## Nice-to-Haves
- Add a controlled experiment in Table 3 where ACT-IN-LLM and at least one baseline use the same LLM backbone to isolate the method's contribution from backbone improvements.
- Include visualization of which tokens are retained/dropped across layers for complex images to validate the "text-guided" selection claim.
- Clarify the theoretical formulation in Theorem 3 by either comparing projections onto a common subspace or reframing as an empirical observation.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic's "Unfair Base Model Comparison" as invalidating SOTA claim**: While the baseline comparison issue is valid (see Major weakness), the paper does frame Table 3 primarily as a data/token efficiency comparison rather than pure method superiority. The efficiency claim ("87.2% of InternVL2 with 32.8% tokens") is supported even if the SOTA framing is overstated. This is a presentation issue, not a fundamental invalidation.

- **Harsh Critic's "Misleading Efficiency Claims"**: The paper claims "~20% reduction" vs. Full (accurate: 621→515ms) and "competitive efficiency," not "best efficiency." Figure 6 shows the trade-off curve honestly. This criticism misreads the paper's claims.

- **Harsh Critic's gradient flow concern as weakening "Adaptive" claim**: Since ACM has no learnable parameters and selection is a heuristic based on attention (not learned), the "Adaptive" refers to layer-wise adaptation based on context, not learned adaptation. This should be clarified but doesn't undermine the method.

- **Strength Finder's "Theoretical and empirical justification for compressing K/V instead of Q"**: The theoretical proof (Theorem 3) has the dimension mismatch issue noted above, so this strength is partially undermined. The empirical support (Figure 5a) remains valid.

- **Generic strengths about "important problem" or "interesting question"**: Removed per guidelines.

## Novel Insights
The paper's core insight—that early token pruning causes irreversible information loss because tokens ignored in early layers may become critical later—is well-supported by Figure 2's attention visualizations and performance curves. This observation challenges the assumption in Pre-LLM and Early-LLM methods that token importance can be reliably determined before or in early layers. The proposed solution (retaining all Query tokens while compressing only K/V) is a practical middle ground between full attention and aggressive pruning. However, this insight, while valuable, is incremental rather than transformative—the method is a sensible architectural modification rather than a fundamentally new paradigm.

## Suggestions
1. **Revise Table 3 framing**: Explicitly position this as a data/token efficiency comparison rather than SOTA method comparison. Add a footnote or sentence acknowledging that backbone differences may contribute to performance gaps.
2. **Fix or remove Theorem 3**: Either reformulate to compare outputs in a common space (e.g., project both to `(M+L)×D` by restricting ACM output to compressed token indices) or remove the theoretical claim and rely on empirical results.
3. **Add training clarification**: In Section 3.2, add one sentence explaining that since ACM has no learnable parameters and selection indices are computed from attention weights (not learned), gradients flow through the LLM weights normally without requiring straight-through estimators.
4. **Add backbone-controlled comparison**: Even a small experiment comparing ACT-IN-LLM vs. FastV both using Vicuna-7B in the Table 3 setting would strengthen the method comparison.

## Score and Decision

**Calibration anchors consulted:**
- **High-scoring (≥6)**: CoTAM (6.0) - strong empirical analysis of compression distortion; LLaVA-FA (6.0) - elegant mathematical formulation with solid experiments; VideoChat-Flash (6.5) - comprehensive contribution with dataset, method, and benchmark; InfoTok (7.33) - adaptive video tokenization with information-theoretic grounding.
- **Medium (~5)**: Task-Related Token Compression (5.5) - innovative paradigm with comprehensive experiments but some clarity issues; GvPdSWZT31 (5.0) - multimodal coverage maximization with split reviewer scores.
- **Low (≤4)**: LaCo (3.5) - rejected for limited backbone testing and insufficient baseline comparisons; STF/MBTF (4.5) - rejected for missing related work and narrow evaluation; T4pK6ByRit (4.0) - withdrawn for efficiency claims not fully supported.

**Positioning**: This paper is stronger than LaCo (3.5) and STF/MBTF (4.5) due to better scaling experiments, clearer motivation (Figure 2), and stronger high-resolution results. It is comparable to Task-Related Token Compression (5.5) in empirical strength but has the theoretical proof issue that Task-Related avoids. The SOTA comparison issue is similar to weaknesses in rejected papers but less severe since the paper does frame it as efficiency comparison.

The core method is sound with strong empirical support, but the SOTA framing is overstated and the theoretical proof needs revision. This places it in the **borderline accept** range, similar to Task-Related Token Compression (5.5) but slightly lower due to the theoretical issue.

**Final score: 5.0** (borderline accept/poster)

The paper makes a solid contribution with strong empirical results, but the SOTA claims need tempering and the theoretical proof needs fixing. These are addressable in revision but prevent a higher score.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>