## Summary
This paper proposes RoPE++, an extension of Rotary Position Embeddings (RoPE) that reincorporates the imaginary component of the complex-valued attention computation, which is standardly discarded. The method introduces two configurations: RoPE++EH (equal attention heads, halved KV cache) and RoPE++EC (equal cache size, doubled attention heads). Theoretical analysis suggests the imaginary component better captures long-range dependencies, and empirical results across 376M, 776M, and 1.5B parameter models show consistent improvements on short- and long-context benchmarks over standard RoPE and other position encoding baselines.

## Strengths
- **Novel and Well-Motivated Core Idea**: The paper identifies a previously overlooked aspect of RoPE—the discarded imaginary component—and provides a principled, theoretically grounded argument for its utility via analysis of characteristic curves (sine vs. cosine integrals). The insight that this corresponds to a simple rotation of queries is elegant.
- **Comprehensive Empirical Evaluation**: The authors conduct extensive pre-training experiments at three model scales (up to 1.5B parameters) and evaluate on a wide suite of standard short- and long-context benchmarks. Results consistently show RoPE++ variants outperform RoPE and other PE methods, with gains more pronounced in long-context settings.
- **Practical Efficiency Benefits**: The RoPE++EH configuration offers a tangible efficiency gain, achieving comparable or superior performance to vanilla RoPE while halving the KV cache size and QKV parameters, leading to measurable reductions in memory cost and improvements in decoding throughput, as validated in Figure 4 and Table 11.

## Weaknesses
- **Limited Mechanistic Analysis of "Why It Works"**: The theoretical analysis focuses on expected behavior, and the empirical support (noise perturbation experiment, example attention patterns) is a good start. However, a deeper, quantitative dissection of how imaginary attention heads function *in practice* within trained models is missing. For instance, a statistical analysis of average attention distances per head type across layers and tasks would solidify the claim about long-context capture.
- **Model Scale for Definitive Scaling Claims**: While experiments at 376M-1.5B are valuable and the authors acknowledge resource limits, the current LLM research landscape often expects validation at larger scales (e.g., 7B+) to make robust, generalizable claims about architectural improvements. The positive trends are promising but not fully conclusive for state-of-the-art model sizes.

## Nice-to-Haves
- Include a more detailed quantitative analysis of attention distance distributions for real vs. imaginary heads to statistically validate the claimed functional difference.
- Provide a wall-clock time comparison during long-context inference for RoPE++EC vs. RoPE to better characterize the compute vs. memory trade-off.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: Statistical significance not reported** - Single-run evaluation is standard for large-scale LLM benchmarks; demanding confidence intervals imposes an arbitrary rigor requirement not typical in the field.
- **Weakness: Lack of realistic long-context task evaluation** - The paper uses standard synthetic long-context benchmarks (RULER, BABILong). Requesting additional task types (e.g., summarization) is scope creep for an architectural contribution focused on position encoding.
- **Weakness: Theoretical assumptions about i.i.d. queries/keys are unrealistic** - The analysis using expectations over random vectors is a standard and accepted theoretical tool for analyzing RoPE's properties; its purpose is to provide intuition, not to model a trained transformer exactly.
- **Weakness: Parameter sharing necessity is not fully justified** - The paper clearly states (Section 3.3) that allocating separate parameters would collapse to standard RoPE because rotating the query by π/2 in imaginary attention yields real attention, making independent heads redundant under the shared rotation framework. This is a reasonable architectural constraint.
- **Weakness: Extrapolation claims are overstated** - The paper does not overstate; it explicitly notes in Section 3.4 and shows in Figure 6 that RoPE++ does not extend the stable context window but slows the perplexity rise afterward, and discusses this as a limitation in Appendix D.
- **Weakness: Inconsistent 1.5B results are not discussed** - The paper does highlight the best results in Table 6 and discusses scaling in Appendix C.1. The performance is not "mixed" in a way that undermines the core claim; RoPE++ variants achieve the best average scores on key metrics.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct a deeper mechanistic analysis, such as computing the average attention distance per head type (real vs. imaginary) across all layers and heads on held-out long sequences to quantitatively validate the claim about long-range focus.
- If computationally feasible, include a pre-training experiment at a larger scale (e.g., 7B parameters) to strengthen the scaling claim and impact, even if limited to a smaller number of tokens.