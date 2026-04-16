Now I have a solid understanding of the paper and the calibration landscape. Let me synthesize the final review.

## Summary

ACT-IN-LLM proposes adaptively compressing vision tokens within LLM decoder layers for high-resolution Multimodal LLMs. Instead of reducing tokens before the LLM (Pre-LLM) or in early layers (Early-LLM), the method retains all query tokens across layers while compressing key/value tokens in the self-attention mechanism based on text-guided attention from the previous layer, with hierarchical compression ratios across early, middle, and latter stages. The paper provides theoretical analysis claiming K/V compression offers a better low-rank approximation than alternative compression strategies, and demonstrates empirical improvements over prior compression methods on general and high-resolution benchmarks.

## Strengths

- **Well-motivated problem with empirical grounding:** The paper provides compelling evidence (Figure 2) that early compression of vision tokens leads to disproportionate performance drops on high-resolution tasks (~15% gap), and that attention importance for vision tokens shifts across layers, motivating within-LLM adaptive compression.

- **Principled and clean K/V-only compression design:** The core architectural insight—keeping all query tokens while compressing only K/V—is elegant. It directly addresses the irreversibility problem of Pre-LLM methods while still achieving meaningful complexity reduction from O((N+L)²D) to O((N+L)(M+L)D) per attention layer.

- **Strong empirical results within the controlled setting:** Table 2 shows a clear +5.5% average improvement over the best prior compression method (FastV) on high-resolution benchmarks, and competitive performance with full-token models. The ablation studies (Table 4) are thorough and informative, covering compression ratios, compression methods, and layer positions.

- **Unified formulation is a valuable conceptual contribution:** The Com(C^Q, C^K, C^V) framework cleanly categorizes existing compression strategies and enables principled comparison. The complexity analysis (Table 1) is clear and useful.

- **Scalability demonstrated across model sizes (0.5B–7B):** The scaling experiments (Figure 7) show consistent improvements with increasing model and data scale, suggesting the approach generalizes beyond the primary Vicuna-7B setting.

## Weaknesses

### Major:

- **Theoretical claims are disconnected from the actual algorithm.** Theorems 1–3 are existence proofs about *some* compression matrices C^K, C^V satisfying approximation bounds. The actual ACM uses deterministic top-k selection on the last token's attention row (Eq. 4–6), and no argument is provided that these specific C^K, C^V satisfy the conditions of Theorems 2–3. The probability bounds in Eq. 11 (derived from random sampling arguments) have no clear connection to ACM's deterministic selection. Sec. 4 thus provides an *abstract* argument that K/V compression can in principle be better than full compression—not that the *specific* ACM achieves this. This matters because the "theoretically demonstrates" claim in the abstract and introduction is one of the paper's main selling points.

- **The "no information loss" / "error correction" narrative is overstated.** The paper claims ACT-IN-LLM "ensuring no vital information is lost" and provides an "inherent error correction mechanism" (Abstract, Sec. 1, Fig. 1). In reality, at each compressed layer, only a subset of K/V tokens participate in attention—tokens whose K/V are dropped in a given layer cannot propagate their content through that layer's attention. While keeping Q tokens means the *token embeddings persist*, the information those tokens provide to the collective attention computation is lost for that layer. The claim should be scaled to "preserves full query representations across layers" rather than "ensures no vital information is lost."

- **Efficiency analysis is incomplete for real-world deployment.** The ACM requires computing explicit attention weights from the previous layer to guide compression (which is incompatible with FlashAttention's kernel-based approach), and uses a modified sparse causal mask (Fig. 4b). The paper reports only single-forward-pass wall-clock times (Table 2, Fig. 6) on a V100 without detailing ACM overhead, FFN costs (which still operate on full N+L tokens), or memory breakdowns. The "~20% time reduction" and "~60% token reduction" claims need more rigorous accounting, including whether the method works with FlashAttention during inference—the standard for deploying transformers efficiently.

### Minor:

- **Limited novelty over closely related prior work.** The core idea of hierarchical, attention-guided layer-wise token reduction is very similar to PyramidDrop, and K/V-selective compression has been explored in RazorAttention. The main distinction—compressing K/V while keeping all Q tokens—is incremental relative to dropping tokens entirely. The paper does not explicitly differentiate from these closely related approaches in its Related Work or method comparison.

- **Ablation reveals near-competitive simpler baseline.** Table 4b shows AvgPool-1D achieves slightly *better* general performance (75.06 vs. 75.04) and only marginally worse HR performance (45.08 vs. 45.35), suggesting the full text-guided ACM mechanism may not be essential. This point is not critically discussed.

- **Guidance mechanism uses only the last token's attention row.** Eq. 4 uses A_{i-1}[N+L, :] (the last token) to guide compression, termed "text-guided." In many multimodal setups, the last token may be a punctuation or special token rather than the most semantically meaningful. There is no ablation comparing this against alternative guidance signals (e.g., mean over text tokens, or a dedicated [CLS]-style token).

- **No evaluation on multi-image or long-document scenarios.** All evaluations use single-image settings with ~1K vision tokens. The method's claimed advantage should scale with longer visual contexts, but this is unstudied.

### Trivial

- Eq. 3 as printed has a formatting issue where the compressed attention definition is garbled.

## Nice-to-Haves

- Evaluate on multi-image or video benchmarks where vision token counts are much larger—the regime where compression matters most.
- Provide FLOPs and memory breakdowns per component (MSA-K/V, MSA-Q, FFN, hidden state storage) to substantiate efficiency claims beyond single-forward-pass wall-clock time.
- Ablate the guidance signal: compare last-token attention vs. mean-of-text-tokens vs. entropy-based selection, to validate this important design choice.
- Discuss FlashAttention compatibility and potential modifications needed for production deployment.
- Test plug-and-play application on already-trained strong MLLMs (e.g., InternVL2) with and without retraining.

## Removed Points

- **FlashAttention incompatibility:** While valid as a deployment concern worth raising, the claim that ACM is inherently incompatible with FlashAttention is not verified in the paper and may be possible to work around. Flagged as a minor efficiency concern rather than a fatal flaw.
- **Overclaim about the 6.3% improvement headline number being imprecise:** The 5.5% number in Table 2 for the primary controlled setting is well-documented. The 6.3% comes from other settings. This is not a fairness issue.
- **Baseline configuration concerns (whether baselines used their own recommended settings):** The paper states "we maintain all other settings constant, varying only the method of vision token compression." This is a reasonable controlled comparison. Removing this as a fairness concern per the rule that baselines should not be criticized for not using different, asymmetric designs.
- **Missing related works (PyramidDrop, RazorAttention, FastV comparisons beyond what's in the paper):** Per instructions, I do not flag missing related works since I cannot independently verify their existence or relevance.
- **No variance/confidence intervals:** Single-run evaluation is the norm in this field; requesting confidence intervals is a nice-to-have, not a core flaw.
- **Scaling experiments don't isolate ACT-IN-LLM vs. full tokens at each scale:** While this would strengthen the paper, the 7B experiments with Table 2 already demonstrate this for the primary setting. Requesting it at every scale is scope creep.

## Novel Insights

The most insightful observation is the tension revealed by the ablation in Table 4b: a simple AvgPool-1D compression of K/V tokens achieves near-identical performance to the text-guided ACM. This suggests that the *primary* benefit of ACT-IN-LLM may not come from "text-guided intelligent selection" but rather from the architectural decision to compress K/V while preserving Q tokens and hidden states. The theoretical framework (keeping Q = identity in Com(I, C^K, C^V)) predicts this—what matters is *which component* you compress, not necessarily *how intelligently* you select tokens. This reframes the contribution: the main advance is identifying Q-preserving K/V compression as the right in-LLM compression target, and the text-guided selection is a secondary refinement. This distinction is underappreciated in the paper's narrative.

## Suggestions

- Revise the theoretical section to honestly acknowledge it establishes existence bounds rather than proving *the specific ACM algorithm* satisfies these bounds. Alternatively, add analysis showing that top-k on the last-row attention produces a C^K, C^V that satisfies Theorem 2's conditions.
- Tone down "ensuring no vital information is lost" to "preserving full query representations" or "mitigating irreversible information loss"—the current wording is inaccurate.
- Add a discussion of AvgPool-1D's near-competitive performance and what it implies about the necessity of text-guided selection vs. the architectural choice of compressing K/V specifically.

## Score and Decision

**Calibration anchors:**
- PyramidDrop (similar topic, weak novelty, scores 3/3/3/3) — very similar core idea but less developed
- SparseVLM (similar topic, training-free token pruning, scores 3/5/6/6) — rejected, with notable methodological issues
- Dynamic-LLaVA (similar topic, accepted poster, scores 6/6/6/6) — comparable empirical contributions but cleaner execution
- Inference Optimal VLMs (accept poster, scores 8/5/5/6/5) — stronger theoretical and empirical contributions

ACT-IN-LLM has a solid empirical contribution and a useful architectural insight (K/V-only compression within LLM layers). However, the overclaimed theory (existence proofs for abstract matrices, not the actual algorithm), the overstated "no information loss" narrative, and the near-competitive simpler baseline (AvgPool-1D) weaken the novelty and significance. The paper is clearly above the rejected papers in the calibration set (PyramidDrop at 3, SparseVLM at 3–6, LVP at 5–6) but below Dynamic-LLaVA (6) and well below the strong accept paper (Inference Optimal VLMs at 5–8). The disconnect between theory and implementation is the primary differentiating weakness.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>