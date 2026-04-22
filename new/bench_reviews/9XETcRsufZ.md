Now I have all the information I need. Let me write the final review.

## Summary

This paper identifies and characterizes a fundamental asymmetry in Mixture-of-Experts (MoE) architectures: scaling the number of experts disproportionately improves memorization performance over reasoning performance. The authors prove theoretical separation results for single-layer transformers (width is critical for reasoning tasks but not for memorization), validate these predictions on synthetic graph and phone-book tasks, and confirm the pattern in pre-trained LLMs evaluated on knowledge vs. reasoning benchmarks.

## Strengths

- **Identifies a genuine and important asymmetry in MoE architectures.** The finding that MoE parameters help memorization more than reasoning has direct implications for architecture design and deployment decisions. The paper establishes this across three complementary levels: theory (Section 3), synthetic experiments (Section 4), and pre-trained LLMs (Section 5).

- **Formal theoretical separation with clean proofs.** Corollary 3.4 shows that at matched total parameters, an MoE with sufficiently many experts (K ≥ Ω((log N)²)) provably cannot solve a graph reasoning task that a dense model solves. Theorems 3.5–3.6 show MoEs memorize with Õ(√nm) active parameters versus Ω̃(n) for dense, providing a concrete provable separation for memorization. The communication-complexity argument (Theorem 3.2) gives mechanistic insight into why width matters for reasoning.

- **Synthetic experiments tightly validate the theoretical predictions.** Figure 4 is one of the strongest parts of the paper. On phone-book memorization (Figure 4a), all model families overlap when plotted against total parameters, confirming total parameters govern memorization. On shortest path (Figure 4b), MoE performance correlates with active parameters rather than total, confirming width—not expert count—governs reasoning. This tight theory–experiment correspondence is unusual and valuable.

- **Perplexity-controlled comparison (Figure 6) provides an important additional lens.** Figure 6 shows that at fixed validation perplexity, MoEs outperform dense on world knowledge (6a) but only match them on reasoning (6b-c). This reveals an implicit architectural bias in what MoEs learn for a given training objective, ruling out the explanation that MoEs simply haven't trained enough.

- **Striking practical efficiency result for memorization.** Section 4.2 reports that an MoE with only 42M active parameters outperforms a dense model with 10× as many parameters on phone-book memorization—a practically significant result for knowledge-intensive applications.

- **Systematic and progressive experimental design.** The paper builds its case from theory → synthetic → real, with the same qualitative pattern appearing at each level, making the conclusions robust.

## Weaknesses

### Fatal
None.

### Major

- **The theoretical results apply only to depth-1 transformers, creating a theory–practice gap.** Theorems 3.2, 3.3 and Corollary 3.4 are all proved for single-layer transformers, but all experimental models use 12–20 layers (Sections 4.1, 5.1). The paper states this (Section 3.2: "single-layer transformer"), but does not discuss whether depth can substitute for width on graph reasoning problems, or provide any argument that the single-layer impossibility extends to multi-layer settings. While the empirical results in deeper models do show the predicted pattern, the theoretical contribution is formally disconnected from the practical setting. A clear discussion of this limitation and whether multi-layer models could overcome the width bottleneck would substantially strengthen the paper. (Sections 3.2, 4.1, 5.1)

- **The pre-trained experiments at equal total parameters conflate architectural effects with active-parameter effects, and the key isolating experiment is absent.** The headline comparisons (Figure 1) give MoEs fewer active parameters than the matched dense model. This makes MoEs look worse at reasoning for two reasons: (a) experts contribute less to reasoning (the paper's thesis), and (b) the MoE simply has less per-token compute. While the synthetic experiments (Figure 4b) do isolate effect (a) by comparing MoEs of varying expert counts at fixed width, the pre-trained experiments do not include an analogous ablation varying the number of experts at fixed width and active parameters on real benchmarks. Without this, the central practical claim that "increasing experts doesn't help reasoning" is directly shown only for synthetic tasks. (Sections 5.1, 5.2, Figure 1 vs. Figure 4b)

### Minor

- **The non-standard intermediate dimension choice (d instead of 4d) deserves more discussion.** The paper sets the FFN intermediate dimension to d instead of the standard 4d (used in Mistral, Llama, etc.), following the OLMoE codebase. The paper is transparent about this choice (Section 4.1: "we set the intermediate dimension in the FFN block to be equal to d (and not 4d)"), but does not discuss its implications. With d instead of 4d, the MoE/FFN block is proportionally much smaller relative to attention, which reduces the impact of the MoE architectural choice. Running at least one configuration with 4d would test robustness to this design choice. (Sections 4.1, 5.1)

- **The abstract states reasoning "saturates" with more experts, but the data suggests diminishing returns, not zero returns.** In Figure 4b, the 42M-active MoE shows improvement from ~67% to ~71% as experts increase. "Saturates" implies a hard ceiling, while the data shows a strong but potentially non-zero slope. The paper's core message still holds—MoE scaling is far less effective for reasoning than memorization—but the language in the abstract is slightly stronger than the evidence warrants. (Abstract, Figure 4b)

- **The generalization gap analysis (Figure 5) is confounded by the equal-total-parameters comparison.** The claim that MoEs' larger train-test gap reflects memorization is confounded by MoEs having fewer active parameters at equal total parameters—which could independently cause worse generalization. The paper appropriately uses cautious language ("suggestive that MoEs are more prone to overfit"), but does not discuss this confound. (Section 5.2, Figure 5)

- **The memorization theory's active-vs-total parameter framing could be clearer.** The theory shows a separation in active parameters (Õ(√nm) vs Ω̃(n)) but both MoE and dense have linear dependence on n in total parameter count. The empirical finding (Figure 4a) is that memorization scales with total parameters. The paper should more explicitly state that the theoretical contribution is about compute efficiency (same memorization with fewer FLOPs per token), not parameter efficiency per se. (Section 3.3)

### Trivial
None.

## Nice-to-Haves

- An experiment varying the number of experts at fixed width on the pre-trained benchmarks (e.g., train MoEs with d=256, E∈{8,16,32,64,128,256} on the same 65B tokens) would directly test the core claim in the practical setting and dramatically strengthen the paper.

- A 4d intermediate dimension ablation for at least one model size to test robustness of findings to the standard architecture choice.

- An explicit discussion of whether depth can substitute for width on graph reasoning, or an acknowledgment that the depth-1 limitation is a clear gap between theory and practice.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Criticisms about the Corollary 3.4 parameter counting only considering FFN parameters.** For depth-1 models, the attention parameters are O(m²) for both dense and MoE (the attention mechanism is identical), so matching FFN parameters is equivalent to matching total parameters modulo shared terms. The logic of Corollary 3.4 survives for total parameters when K is large.

- **Criticisms about missing related works or references.** Per rules, I cannot verify the existence of external references and should not flag their absence.

- **Criticisms about reproducibility of hyperparameters or training details.** The paper provides sufficient architectural and optimization details for the claimed contributions; further implementation details are standard reproducibility concerns not specific to the claims.

- **Criticisms about Figure 6 implications being under-discussed.** The paper actually does discuss this finding (Section 5.2, last paragraph), stating "Which strategy the model prefers is determined by the implicit bias of the architecture" and noting MoEs "prioritize" memorization. The discussion, while brief, adequately contextualizes the result.

- **Criticisms demanding the paper discuss practical deployment tradeoffs (memory vs. compute).** This is outside the paper's stated scope of analyzing the memorization–reasoning asymmetry in MoE; the Discussion already acknowledges that "increasing dimension may be unavoidable for reasoning tasks."

## Novel Insights

The perplexity-matched comparison (Figure 6) reveals something the paper could emphasize more: MoEs are not inherently worse at reasoning *per unit of training objective achievement*—they're worse *per total parameter*. This distinction matters because it suggests the MoE architectural bias operates during training (MoEs prioritize memorization when minimizing perplexity), not at inference. If one's bottleneck is inference compute rather than model memory, MoEs may not actually be worse at reasoning at all—they simply choose to allocate their capacity differently during training.

## Suggestions

- Run one additional experiment: train MoEs with d=256 and E∈{8,16,32,64,128,256} on the same 65B-token pre-training data, evaluating on all three benchmark categories. This directly tests the claim that adding experts doesn't improve reasoning in the practical setting and would close the most significant evidential gap.

- Add a brief discussion (2-3 sentences) in Section 3.2 acknowledging that the depth-1 limitation is a gap between the theoretical model and the experimental setting, and either give intuition for why depth doesn't circumvent the width bottleneck or explicitly flag this as a limitation.

- In the abstract, replace "saturate" with "diminish rapidly" or "exhibit diminishing returns" to more precisely match the empirical evidence.

## Evaluation on Key Axes

**Originality:** High. The memorization–reasoning asymmetry in MoE is an important and underappreciated insight, and the formal separation results are novel.

**Importance of research question:** High. Understanding what MoEs gain and lose versus dense models is critical for the design of frontier models, most of which now use MoE.

**Claims support:** Moderate-to-good. The theoretical claims are well-supported within their scope (depth-1); the empirical claims are well-supported in synthetic settings and partially supported in pre-trained settings, with the key isolating experiment (varying experts at fixed width on real benchmarks) missing.

**Soundness of experiments:** Good for synthetic; moderate for pre-trained (the equal-total-parameter comparison conflates two effects, and the non-standard architecture choice is not ablated).

**Clarity of writing:** Good. The paper is well-structured with a clear theory → synthetic → real progression.

**Value to research community:** High. The results have direct implications for MoE architecture design and deployment decisions, and the theoretical framing provides mechanistic insight.

## Calibration

**High-scoring anchors (>7):**
- *Memorization Capacity of Multi-Head Attention* (avg 7.5, Accept spotlight): Single-layer theory of memorization with synthetic validation — comparable scope to this paper's theory section, but more precise and better-scoped claims.
- *Understanding Factual Recall via Associative Memories* (avg 7.33, Accept spotlight): Shallow transformer theory on factual recall with proofs and synthetic experiments — similar depth limitation but more focused theoretical contribution. The paper under review has stronger and broader empirical support.
- *Chain-of-thought expressivity* (avg 8.67, Accept oral): Very focused theory (parity) but rigorous; this paper under review is broader in empirical scope but less tight theoretically.

**Medium-scoring anchors (4–6):**
- *MoE meets Instruction Tuning* (avg 6.75, Accept poster): Empirical MoE vs dense comparison — less theoretical depth but cleaner practical message. The paper under review is more novel and has theory.
- *Overclaimed empirical scaling* (avg 4.75–5.75, Reject): Strong experiments but overclaimed scope — the paper under review has better grounding and a genuine theoretical contribution, placing it clearly above these.
- *MoE sparsity-stability* (avg 3.0, Reject): Questionable theoretical assumptions (Lipschitz continuity for discontinuous top-K) — the paper under review has sound theory within its stated scope.

**Low-scoring anchors (<3):**
- *Inherent limitations of GPT/LLM* (avg 2.0, Reject): Attempted to prove fundamental reasoning limitations but the proof was unconvincing — the paper under review is far more rigorous and empirically grounded.

This paper is above the medium "overclaimed empirical" anchors (4.75–5.75) due to genuine theoretical contributions and clean synthetic validation. It is below the top-scoring theory papers (7.5–8.67) because the theory is explicitly limited to depth-1 models and the key pre-trained ablation is missing. The paper is comparable to *Understanding Factual Recall* (7.33) in having a depth-1 theory validated on synthetic tasks, but with broader empirical coverage.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>