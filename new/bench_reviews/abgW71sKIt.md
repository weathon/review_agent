Now I have a thorough understanding of the paper and relevant calibration anchors. Let me write the final review.

## Summary

This paper investigates why naive output alignment fails in 1-bit post-training quantization (PTQ) of LLMs, identifying three failure modes: (i) layer-wise output matching doesn't reduce block-level loss, (ii) quantization errors accumulate across layers, and (iii) indiscriminate output matching disrupts attention mechanisms. Based on these findings, the authors propose a selective layer-wise output alignment strategy (restricting output alignment to the last FC layer per block), reformulate the objective from activation-conditioned error to output error, and introduce an Attention Matrix Preservation (AMP) mechanism. Experiments on OPT and LLaMA models show improvements over existing 1-bit PTQ methods on most benchmarks.

## Strengths

- **The analysis in §3.2** demonstrating that layer-wise output alignment can *increase* block-level loss (Figure 1) is a genuine and useful insight. It provides concrete evidence for why ARB-X underperforms on some layers, grounding the paper's motivation in a real empirical observation.

- **The output error objective reformulation** (replacing ∥X̂W − X̂Ŵ∥ with ∥XW − X̂Ŵ∥) is a principled modification that directly addresses error accumulation. The closed-form derivations (Eqs. 5–8) for the new objective are correct, and Table 4 confirms this objective contributes ~0.7 PPL improvement on C4 for OPT-6.7B and ~0.72 PPL for LLaMA-2-7B.

- **The AMP mechanism has a large empirical effect on LLaMA-family models.** Table 3 shows AMP reduces LLaMA-2-7B C4 PPL from 29.12→19.25 and WikiText2 PPL from 26.24→15.42 — a nearly 10-point improvement that is essential to the method's viability on these architectures.

- **Overall results are strong on most settings.** The method outperforms ARB-RC, ARB-X, BiLLM, and PB-LLM on OPT-1.3B through OPT-30B, LLaMA-2-7B (C4/WikiText2), LLaMA-2-13B, and LLaMA-3-8B on both generative and discriminative tasks.

## Weaknesses

### Fatal

None.

### Major

- **Catastrophic failure on LLaMA-2-7B PTB (3166 PPL vs 763 for ARB-RC, 681 for ARB-X) directly contradicts the "consistently outperforms" framing.** The paper dismisses this by stating "the large perplexity indicates that the metric cannot provide a meaningful evaluation" (line 219). However, a ~4× gap relative to other methods at the same absolute PPL level is meaningful — ARB-RC and ARB-X produce substantially less distorted distributions even in this high-PPL regime. No analysis is provided for why the method fails here but succeeds elsewhere, which is essential for understanding generalizability. This is not necessarily fatal because the method genuinely outperforms on 11 out of 12 reported PPL configurations, but the "consistently outperforms" claim should be qualified and the failure analyzed.

- **The selective layer-wise design (output alignment restricted to the last FC layer) is a central proposed contribution but lacks ablation support.** Section 4.2 asserts this choice because the last FC layer "has the most direct impact on the block loss," but no empirical or theoretical validation is provided. There is no ablation comparing: output alignment on all layers, on attention layers only, on the last FC layer only, on random subsets, etc. Without this, it is impossible to evaluate whether the last FC layer is actually the optimal choice, or whether the improvement comes entirely from the modified objective and AMP.

- **The AMP mechanism's stated purpose ("Attention Matrix Preservation") is not well-supported by its actual implementation.** AMP computes masks as sign(∇θ L_AMP) and uses them to gate updates from the *main* output error objective. The gradient sign of L_AMP indicates whether a parameter increase would improve attention preservation, but the gated update α_r* comes from a completely different objective. There is no principled reason why accepting a main-objective update when the AMP gradient is positive actually preserves attention structure. Empirically, AMP is essential for LLaMA (Table 3), but this effect could easily arise from regularization rather than attention preservation per se. Presenting AMP with mathematical formalism that doesn't support its claimed function is misleading.

### Minor

- **No variance/std is reported** across calibration sets or seeds. Some improvements over ARB-RC are marginal (e.g., LLaMA-2-13B: C4 13.80 vs 14.77 is meaningful, but LLaMA-3-8B WikiText2: 27.20 vs 27.42 is only 0.22 PPL). At larger scales, it would be useful to know which improvements are robust.

- **The hypothesized explanation for LLaMA's AMP sensitivity** (RMSNorm vs LayerNorm) is interesting but untested. A brief experiment or deeper analysis would strengthen this claim.

### Trivial

- In Table 2, weight bits show 1.06 for LLaMA vs 1.11 for OPT (Table 1), presumably reflecting different proportions of non-binarized salient weights, but this difference is not explained.

## Nice-to-Haves

- Ablation over which layers receive output alignment (all, last FC only, attention only, etc.) would substantiate a key design choice.
- Root-cause analysis for the LLaMA-2-7B PTB failure, isolating whether it arises from the output error objective, the selective layer scheme, or AMP.
- Evaluation on at least one more model family (e.g., Mistral or Phi) to strengthen generalizability claims.
- A more honest presentation of AMP — either as a heuristic regularizer (which the evidence supports) or with a proper theoretical justification for how gradient-sign gating of a separate objective preserves attention structure.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No standard deviations reported"** as a major concern — this is standard in the PTQ/quantization community where single-run evaluation with fixed calibration sets is the norm; moved to Minor rather than Major.
- **"Only OPT and LLaMA evaluated, need more architectures"** — this is a reasonable scope for a paper targeting 1-bit PTQ for these specific model families; moved to Nice-to-Have.
- **"Zero-shot QA improvements are within noise (0.54% best)"** — the paper reports these as supplementary; PPL improvements are the primary metric and are clearly meaningful at smaller scales.
- **Strength claim that AMP's effect "validating all three proposed interventions"** — AMP's empirical success doesn't validate the attention preservation mechanism since the mechanism itself is not well-justified (see Major weakness above).

## Novel Insights

The most genuinely novel insight is the §3.2 demonstration that *layer-wise output alignment can increase block-level loss* compared to weight alignment — this is counterintuitive and challenges the common assumption that output-level objectives are always preferable. Combined with the observation that error accumulation causes the ARB-X optimization target to diverge from the true full-precision target (Figure 2, upper-right), this provides a clear mechanistic explanation for why naive output alignment fails. The finding that RMSNorm-architectures (LLaMA) are particularly vulnerable to attention disruption during quantization is also novel but insufficiently analyzed.

## Suggestions

- **Qualify the "consistently outperforms" claim** to acknowledge the LLaMA-2-7B PTB regression, and add at least a brief analysis of what drives it.
- **Add ablations for the selective layer-wise design** — even testing just "all layers" vs "last FC only" would be informative and easy to run.
- **Reframe AMP's theoretical justification** honestly as a heuristic that empirically preserves attention structure, rather than presenting gradient-sign gating as a principled attention preservation mechanism.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| PB-LLM (BifeBRhikU) | 6.75 | Comparable topic (1-bit LLM PTQ); this paper has stronger analysis but weaker ablations and a failure case PB-LLM doesn't have |
| CBQ (eW4yh6HKz4) | 7.60 | Spotlight PTQ method; this paper is clearly below CBQ's rigor |
| TesseraQ (GTe9PDhm8v) | 5.00 | Rejected ultra-low-bit PTQ; this paper is stronger than TesseraQ (better analysis, more complete method), but shares similar ablation weaknesses |
| AdaQTransform (JElN0LJMKB) | 5.25 | Accepted poster PTQ method; this paper has comparable insight quality but more significant weaknesses (PTB failure, no design ablation) |
| 1-Bit FQT (oWy06SBgt4) | 5.50 | Rejected 1-bit quantization; this paper has much stronger empirical results on LLMs |
| MOESR (QRvBXm7WK3) | 3.50 | Rejected with overclaimed contributions; this paper is well above this level |

This paper sits between TesseraQ/AdaQTransform (5.0–5.25) and PB-LLM (6.75). The genuine contributions (§3.2 analysis, output error objective, AMP effect) are real, but the combination of a catastrophic PTB failure dismissed without analysis, a central design choice without ablation support, and an overclaimed theoretical basis for AMP pushes it below PB-LLM. The paper is somewhat above the rejected 5.0-level PTQ papers because it has genuine analytical contributions and strong results on most benchmarks, but the major weaknesses are significant enough to prevent a confident acceptance. Score: **5.5**.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>