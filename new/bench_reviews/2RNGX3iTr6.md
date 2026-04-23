Now I have all the information I need. Let me synthesize the final review.

## Summary

Tabby proposes a post-training architectural modification for transformer-based LLMs that applies Mixture-of-Experts (MoE) layers to dedicate expert parameters to each tabular data column, either in the MLP layers (MMLP) or the LM head (MH). The paper evaluates across 6 datasets, 3 training paradigms (Plain, GReaT, GTT), and 4 Tabby variants, finding that Plain-trained Tabby MH achieves the strongest results among LLM methods, particularly on regression datasets (House, Rainfall).

## Strengths

- **Identifies a real architectural gap and proposes an intuitive solution.** The observation that LLM architectures lack column-specific inductive biases for tabular data is well-motivated, and applying MoE with per-column experts is a clean, interpretable design choice (Section 1, 3.1).

- **Plain MH achieves strong practical results, especially on regression datasets.** On House, Plain MH achieves 0.75 R² (vs. 0.81 original), outperforming all other methods including Tab-DDPM (0.59). On Rainfall, Plain MH (0.58) produces valid samples in all runs where many prior LLM methods fail entirely (Table 2, Section 4.1).

- **Thorough experimental grid.** The paper evaluates across 6 datasets, 3 training paradigms, 4 Tabby variants, and multiple baselines (CTGAN, TVAE, Tab-DDPM), providing a useful empirical landscape (Tables 2, 3).

- **Per-column loss decomposition yields useful diagnostics.** Figure 4 reveals that Occupancy dominates early loss while Median Income plateaus — information that could guide preprocessing or data collection decisions (Section 4.3).

- **Honest discussion of baseline findings.** Section 4.4 candidly notes that the simple Plain NT baseline achieves near-optimal MLE on several standard datasets and recommends the community identify harder benchmarks — showing intellectual honesty about the evaluation landscape.

- **Figure 2 provides a meaningful qualitative comparison.** The scatter plots clearly show that Tabby generates continuous-valued distributions matching real data, while Tab-DDPM is limited to integer-valued targets.

## Weaknesses

### Fatal

None.

### Major

- **Parameter count confound undermines the core mechanistic claim.** Table 3 reveals that Tabby MH on Distilled-GPT2 has 270M parameters versus 80M for the NT baseline — a 3.4× increase. The paper's central claim is that "column-specific MoE routing improves tabular synthesis" (Section 1, contributions), but no parameter-matched non-MoE control is provided. A standard Distilled-GPT2 with a widened (non-MoE) LM head at ~270M parameters would be the natural control; without it, the improvement could simply come from having more parameters rather than from the MoE routing mechanism. This problem pervades Tables 2 and 3. The paper even contains a factual error in the Figure 3 caption: it claims "a similar parameter count (270M) to NT DGPT2 (80M)," which is clearly incorrect — 270M is 3.375× larger than 80M.

- **The generation-time expert routing mechanism is entirely unspecified.** Section 3.3 describes training-time routing (column *i* → expert *i*), but the paper never explains how expert selection works during autoregressive generation. For Plain training with fixed column order, deterministic routing by counting `<EOC>` tokens is possible, but this is never stated. For GReaT-trained models with shuffled column order (included in Table 2), the routing mechanism is genuinely unclear. This is a reproducibility-blocking gap and a conceptual one: if routing is simply deterministic/positional, the "expert specialization" framing is misleading, as there is no learned routing.

- **Key claims are overstated relative to the evidence.** (1) The abstract claims "up to 7% improvement compared to previous tabular dataset synthesis methods," but this 7% figure traces to comparing Plain MH against Plain NT (the paper's own baseline) on the House dataset — not against previous methods like CTGAN, TVAE, or Tab-DDPM. (2) The contributions state "higher-quality synthetic data for 4 out of 6 datasets," but among ALL methods (including Tab-DDPM), Tabby MH is the clear MLE leader on only 2 of 6 datasets (House and Rainfall). The "4 out of 6" holds only among LLM methods, a scope not stated in the contributions. (3) The table caption claims "reaching upper-bound performance on 4/6 datasets," but only 3 datasets (Diabetes, Travel, Adult) have Tabby MH at or above the Original row. (4) The conclusion says "parity in two out of three," conflicting with other claims of 3/6 or 4/6.

### Minor

- **MMLP consistently hurts performance, and this is unanalyzed.** Across Table 2, MMLP degrades results in nearly every configuration compared to NT. MMLP-MH (combining both modifications) also performs poorly. This pattern is more consistent with "a wider output layer helps but column-specific routing in intermediate layers hurts" than with the paper's framing of MoE as uniformly beneficial. The paper should analyze why this occurs — it is one of the most informative findings.

- **Tabby provides negligible benefit for larger models.** On Llama 3, MH improves MLE from 0.560 to only 0.562 while worsening discrimination from 24.2 to 25.3 (Table 3). This suggests the modification does not scale meaningfully to larger architectures, limiting the generality of the approach.

- **Claim 3 (per-column loss tracking) is not an architectural contribution.** Per-column loss can be trivially computed for any autoregressive model by masking the loss per column segment — it does not require the Tabby architecture. This is a convenient side-effect of the training procedure, not a meaningful architectural contribution.

### Trivial

- **Figure 3 caption contains a factual error.** It states Tabby MH DGPT-2 has "a similar parameter count (270M) to NT DGPT2 (80M)." 270M is not similar to 80M — it is a 3.375× increase.

## Nice-to-Haves

- A parameter-matched non-MoE baseline (e.g., a Distilled-GPT2 with a wider LM head at ~270M parameters) would definitively test whether the improvement comes from MoE routing or from parameter scaling alone.
- Analysis of expert utilization/activation patterns during generation to verify that different experts actually specialize in different columns.
- Qualitative comparison of generated rows (not just aggregate metrics) to illustrate what Tabby captures and misses.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's Claim 2 "only 28% of the way" calculation is mathematically wrong.** The reviewer states Tabby MH DGPT-2 is "only 28% of the way from NT DGPT-2's 0.474 to NT Llama's 0.560." The actual calculation is (0.525−0.474)/(0.560−0.474) = 0.051/0.086 = 59.3%, which is more than "splitting the difference." The paper's Claim 2 language is approximately correct on this point — though the Figure 3 caption's "similar parameter count" statement is still wrong.

- **Harsh Critic's criticism of the "Plain training" contribution.** Calling standard fine-tuning a "contribution" is indeed overblown, but this is a trivial issue, not a substantive weakness. The paper's own results show Plain NT already performs well, which somewhat undermines Tabby's own motivation — but this is more a self-critical finding than a flaw.

- **Strength Finder's claim "Tabby achieves state-of-the-art MLE on 4/6 datasets."** This is misleading — Tabby is SOTA only among LLM methods, not over all methods including Tab-DDPM. Moved to Removed Points because it conflicts with the verified Major weakness about overclaimed results.

- **Strength Finder's claim "MoE architecture modification provides disproportionate gains for smaller models — architecture change, not just raw parameter count."** This directly conflicts with the verified Major weakness about the parameter confound. The strength finder asserts what the paper fails to prove.

- **Strength Finder's claim about "Low discrimination scores indicate high-fidelity."** Tab-DDPM achieves better discrimination scores on 4/6 datasets (e.g., Adult: 0.9 vs 9.8; Travel: 1.4 vs 3.0), so this strength is misleading.

## Novel Insights

The most revealing pattern in the data is the divergence between MLE and discrimination: Tabby MH tends to win on MLE (especially for regression tasks), while Tab-DDPM tends to win on discrimination. This tension is underexplored in the paper but suggests that Tabby may be better at preserving downstream predictive utility while Tab-DDPM better avoids detectable distributional artifacts. Understanding this tradeoff — and whether the MLE metric alone is sufficient for evaluating tabular synthesis quality — would be a more valuable contribution than the current MoE framing.

## Suggestions

- **Add a parameter-matched non-MoE control.** Train a standard Distilled-GPT2 with a widened LM head (no MoE structure) at ~270M total parameters on the same datasets. If it matches Tabby MH, the "column-specific expert" framing collapses; if it underperforms, the MoE routing claim is validated. This single experiment would dramatically strengthen (or invalidate) the paper.

- **Explicitly describe the generation-time routing.** At minimum, state whether routing is deterministic (counting `<EOC>` tokens) or involves a learned gating function, and explain how GReaT-shuffled column orders are handled during sampling.

- **Temper claims to match the evidence.** Replace "4 out of 6 datasets" with "4 out of 6 datasets among LLM methods" in the contributions. Correct the "up to 7%" abstract claim to accurately reflect what is being compared. Fix the Figure 3 caption.

## Evaluation

**Originality:** The idea of applying per-column MoE to LLMs for tabular synthesis is straightforward and intuitive, but the novelty is limited — MoE has been extensively studied, and the application to tabular columns is a relatively direct mapping. The MMLP variant (which doesn't work) and the Plain training baseline (which is standard fine-tuning) add little novelty.

**Importance of research question:** Important — tabular data synthesis is practically valuable, and architectural improvements for LLMs in this domain are needed. However, the paper's own results show that simple baselines already perform well on standard benchmarks, somewhat diminishing the urgency.

**Claims support:** The core mechanistic claim (MoE routing improves synthesis) is poorly supported due to the parameter confound. The practical claim (Tabby produces better data) is better supported but limited to LLM-method comparisons and regression datasets.

**Soundness of experiments:** Reasonable breadth but a critical missing control (parameter-matched baseline). The missing generation procedure is a reproducibility concern.

**Clarity:** Generally clear writing, but the method section has a significant gap (generation procedure), and claims are inconsistent across abstract, contributions, and conclusion.

**Value to community:** The paper provides a useful empirical comparison of tabular synthesis methods and highlights the surprising strength of Plain training. The MoE approach, if validated with proper controls, could be practically useful.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison to Tabby |
|-------|------|-----------|-------------------|
| PTaRL | G32oY4Vnm8.md | 8.0 | Much stronger: clear theoretical motivation, well-validated claims, consistent improvements. Tabby is well below this. |
| DynMoE | T26f9z2rEe.md | 7.0 | Stronger: novel auto-tuning MoE mechanism, complete method description. Tabby's method is less complete. |
| ProgSyn | KTL534o7Ot.md | 5.33 | Similar overclaiming pattern, but ProgSyn has a clearer method. Tabby has a more serious parameter confound. |
| TabDAR | kkGIbmpCHU.md | 4.75 | Close topic (autoregressive tabular synthesis), similar incomplete description issues. Tabby has a worse parameter confound but better empirical coverage. |
| TAEGAN | pBqOH2g6K1.md | 4.5 | Overclaimed tabular synthesis results. Similar quality tier. |
| Hyper-UT | tI3eqOV6Yt.md | 5.0 | Same parameter confound pattern — "improvements from architecture or just more parameters?" Tabby has more experiments but also overclaims more. |
| MoDE | VAqRZIuW8m.md | 3.5 | MoE with parameter scaling and unclear inference. Tabby is somewhat better — more experiments, clearer motivation. |
| LCN | wYVP4g8Low.md | 3.0 | Severe parameter confound with unfair comparisons. Tabby is better than this. |

Tabby sits in the lower-medium range. It is better than the clearly broken papers (LCN at 3.0, MoDE at 3.5) but worse than papers with clearer methods and more honest claims (ProgSyn at 5.33, TabDAR at 4.75). The parameter confound is a serious issue that directly undermines the paper's central claim, and it is combined with an incomplete method description and overclaimed results. These are not fixable in a rebuttal — they require new experiments.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>